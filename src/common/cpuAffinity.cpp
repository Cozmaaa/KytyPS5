#include "common/cpuAffinity.h"

#include "common/abSwitch.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/ownCores.h"
#include "common/profiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#elif !defined(__APPLE__)
#include <fstream>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace Common {

void TraceCurrentCpu(const char* name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_CPU_TRACE");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled) {
		return;
	}
	struct State {
		uint32_t hist[256] {};
		uint64_t calls   = 0;
		uint64_t last_ms = 0;
	};
	thread_local State state;
	PROCESSOR_NUMBER processor {};
	GetCurrentProcessorNumberEx(&processor);
	state.hist[(processor.Group * 64u + processor.Number) & 255u]++;
	if ((++state.calls & 255u) != 0) {
		return;
	}
	const auto now = GetTickCount64();
	if (state.last_ms == 0) {
		state.last_ms = now;
		return;
	}
	if (now - state.last_ms < 10000) {
		return;
	}
	state.last_ms = now;
	uint64_t total = 0;
	for (const auto count: state.hist) {
		total += count;
	}
	std::string line;
	for (int shown = 0; shown < 6; shown++) {
		uint32_t best = 0;
		for (uint32_t cpu = 1; cpu < 256; cpu++) {
			if (state.hist[cpu] > state.hist[best]) {
				best = cpu;
			}
		}
		if (state.hist[best] == 0 || total == 0) {
			break;
		}
		line += " " + std::to_string(best) + ":" +
		        std::to_string(static_cast<int>(100.0 * state.hist[best] / static_cast<double>(total))) + "%";
		state.hist[best] = 0;
	}
	std::fill(std::begin(state.hist), std::end(state.hist), 0u);
	LOGF("CpuTrace %s:%s\n", name, line.c_str());
#else
	(void)name;
#endif
}

// --cpu-placement: -1 not given, 0 false, 1 true.
static std::atomic<int> g_cli_x3d {-1};

void SetCpuPlacementEnabled(bool enabled) {
	g_cli_x3d.store(enabled ? 1 : 0);
}

#if !defined(__APPLE__)

enum class Mode {
	Off,    // KYTY_PCORE_AFFINITY=0: nothing at all.
	Hybrid, // Default: only the hybrid-CPU rule (critical and high threads on performance cores).
	X3d,    // KYTY_X3D=1 / --cpu-placement true: the whole placement described in the header.
};

static Mode GetMode() {
	static const Mode mode = [] {
		const char* x3d    = std::getenv("KYTY_X3D");
		const char* pcores = std::getenv("KYTY_PCORE_AFFINITY");
		const int   cli    = g_cli_x3d.load();
		const bool  on     = cli >= 0 ? cli == 1 : (x3d != nullptr && x3d[0] == '1');
		if (on) {
			return Mode::X3d;
		}
		return pcores != nullptr && pcores[0] == '0' ? Mode::Off : Mode::Hybrid;
	}();
	return mode;
}

// Whether PlaceCurrentThread acts: X3D mode, or the hybrid rule on a hybrid CPU.
static std::atomic<bool> g_active {false};

static const char* ClassName(ThreadClass cls) {
	switch (cls) {
		case ThreadClass::Critical: return "critical";
		case ThreadClass::High: return "high";
		case ThreadClass::Background: return "background";
		case ThreadClass::Low: return "low";
	}
	return "?";
}

// Logical processor numbers as a compact list ("0-5,8,12-15") for the log.
template <typename T>
static std::string FormatList(const std::vector<T>& values) {
	std::string result;
	for (size_t i = 0; i < values.size();) {
		size_t j = i;
		while (j + 1 < values.size() && values[j + 1] == values[j] + 1) {
			j++;
		}
		if (!result.empty()) {
			result += ',';
		}
		result += std::to_string(values[i]);
		if (j > i) {
			result += '-' + std::to_string(values[j]);
		}
		i = j + 1;
	}
	return result.empty() ? "none" : result;
}

// One logical processor. `core` and `llc` are keys unique across processor groups; `l3` is the
// size of the last-level cache it shares (0: unknown).
struct LogicalProcessor {
	uint32_t os_id      = 0; // Windows: CPU set id. Linux: CPU number.
	uint32_t number     = 0; // For the log: group * 64 + index (Windows), CPU number (Linux).
	uint32_t core       = 0;
	uint32_t llc        = 0;
	uint32_t efficiency = 0; // Higher is faster (Windows efficiency class; Linux max MHz).
	uint64_t l3         = 0;
};

struct Topology {
	std::vector<LogicalProcessor> all;
	std::vector<uint32_t>         fast;       // os ids: V-Cache CCD / performance cores.
	std::vector<uint32_t>         slow;       // os ids: the rest (may be empty).
	std::vector<uint32_t>         normal;     // Process default: guest and untagged threads.
	std::vector<uint32_t>         high;       // Fast cores minus the reserved ones.
	std::vector<uint32_t>         background; // Slow cores, or `normal` when there are none.
	std::vector<std::vector<uint32_t>> reserved; // Per reserved core, its logical processors.
	std::vector<uint32_t>              reserved_core_keys;
	std::atomic<uint32_t>              next_reserved {0};
	bool                               vcache = false;
	bool                               hybrid = false;
};

static Topology& GetTopology() {
	static Topology topology;
	return topology;
}

static std::vector<uint32_t> Numbers(const Topology& t, const std::vector<uint32_t>& ids) {
	std::vector<uint32_t> numbers;
	for (const auto& lp: t.all) {
		if (std::find(ids.begin(), ids.end(), lp.os_id) != ids.end()) {
			numbers.push_back(lp.number);
		}
	}
	std::sort(numbers.begin(), numbers.end());
	return numbers;
}

// Splits the logical processors into the placement sets, from `all`. `reserve`: whether critical
// threads may get cores of their own.
static void Plan(Topology& t, bool reserve) {
	uint32_t highest = 0, lowest = UINT32_MAX;
	uint64_t max_l3 = 0, min_l3 = UINT64_MAX;
	std::set<uint32_t> llcs;
	for (const auto& lp: t.all) {
		highest = std::max(highest, lp.efficiency);
		lowest  = std::min(lowest, lp.efficiency);
		max_l3  = std::max(max_l3, lp.l3);
		min_l3  = std::min(min_l3, lp.l3);
		llcs.insert(lp.llc);
	}
	t.hybrid = lowest != highest;
	// Two or more L3 domains of different sizes: a stacked-cache CCD next to a plain one.
	t.vcache = llcs.size() > 1 && max_l3 > min_l3 && min_l3 != 0;

	// Without a stacked cache, several L3 domains (a 9950X's two CCDs) still matter: the critical
	// and high threads hand data to each other, so they share the first domain's L3.
	const uint32_t first_llc = *llcs.begin();
	std::map<uint32_t, std::vector<uint32_t>> fast_cores; // core key -> os ids
	for (const auto& lp: t.all) {
		const bool fast = lp.efficiency == highest &&
		                  (t.vcache ? lp.l3 == max_l3 : (t.hybrid || lp.llc == first_llc));
		(fast ? t.fast : t.slow).push_back(lp.os_id);
		if (fast) {
			fast_cores[lp.core].push_back(lp.os_id);
		}
	}

	// Physical cores first: each critical thread gets a whole core (both SMT siblings), taken from
	// the top of the fast set (core 0 takes most interrupts), while at least 4 fast cores are left
	// to everything else.
	uint32_t want = reserve ? 2u : 0u;
	if (const char* value = std::getenv("KYTY_X3D_RESERVE"); reserve && value != nullptr) {
		want = static_cast<uint32_t>(std::max(0, std::atoi(value)));
	}
	const auto cores = static_cast<uint32_t>(fast_cores.size());
	const auto count = cores > 4 ? std::min(want, cores - 4) : 0u;
	std::set<uint32_t> reserved_ids;
	for (auto it = fast_cores.rbegin(); it != fast_cores.rend() && t.reserved.size() < count; ++it) {
		t.reserved.push_back(it->second);
		t.reserved_core_keys.push_back(it->first);
		reserved_ids.insert(it->second.begin(), it->second.end());
	}
	const auto without_reserved = [&](const std::vector<uint32_t>& ids) {
		std::vector<uint32_t> result;
		for (const auto id: ids) {
			if (reserved_ids.count(id) == 0) {
				result.push_back(id);
			}
		}
		return result;
	};
	std::vector<uint32_t> all_ids;
	for (const auto& lp: t.all) {
		all_ids.push_back(lp.os_id);
	}
	// Guest threads follow the stacked cache too; on a hybrid CPU they keep the efficiency cores,
	// as before this placement existed.
	t.normal     = without_reserved(t.vcache ? t.fast : all_ids);
	t.high       = without_reserved(t.fast);
	t.background = t.slow.empty() ? t.normal : t.slow;
}

static void LogTopology(const Topology& t) {
	std::set<uint32_t> cores, llcs;
	uint64_t           max_l3 = 0;
	for (const auto& lp: t.all) {
		cores.insert(lp.core);
		llcs.insert(lp.llc);
		max_l3 = std::max(max_l3, lp.l3);
	}
	std::string reserved;
	for (const auto& core: t.reserved) {
		reserved += (reserved.empty() ? "" : " ") + std::string("[") + FormatList(Numbers(t, core)) + "]";
	}
	LOGF("CPU placement: %zu logical processors, %zu cores, %zu L3 domain(s), largest L3 %" PRIu64
	     " MiB; V-Cache CCD: %s; hybrid: %s\n",
	     t.all.size(), cores.size(), llcs.size(), max_l3 >> 20u, t.vcache ? "yes" : "no",
	     t.hybrid ? "yes" : "no");
	LOGF("CPU placement: fast %s, slow %s, reserved for critical threads %s\n",
	     FormatList(Numbers(t, t.fast)).c_str(), FormatList(Numbers(t, t.slow)).c_str(),
	     reserved.empty() ? "none" : reserved.c_str());
	LOGF("CPU placement: guest and untagged threads %s, background %s\n",
	     FormatList(Numbers(t, t.normal)).c_str(), FormatList(Numbers(t, t.background)).c_str());
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

static bool ReadTopology(Topology& t) {
	ULONG length = 0;
	GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	if (length == 0) {
		return false;
	}
	std::vector<uint8_t> buffer(length);
	if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
	                                length, &length, GetCurrentProcess(), 0)) {
		return false;
	}
	// L3 size per logical processor (group, index).
	std::map<uint32_t, uint64_t> l3_of;
	DWORD                        cache_length = 0;
	GetLogicalProcessorInformationEx(RelationCache, nullptr, &cache_length);
	if (cache_length != 0) {
		std::vector<uint8_t> caches(cache_length);
		if (GetLogicalProcessorInformationEx(
		        RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(caches.data()),
		        &cache_length)) {
			for (DWORD offset = 0; offset < cache_length;) {
				const auto* info =
				    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(caches.data() + offset);
				if (info->Size == 0) {
					break;
				}
				const auto& cache = info->Cache;
				if (cache.Level == 3) {
					const auto& mask = cache.GroupMask;
					for (uint32_t bit = 0; bit < 64; bit++) {
						if ((mask.Mask >> bit) & 1u) {
							l3_of[static_cast<uint32_t>(mask.Group) * 64u + bit] = cache.CacheSize;
						}
					}
				}
				offset += info->Size;
			}
		}
	}
	for (ULONG offset = 0; offset < length;) {
		const auto* info = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data() + offset);
		if (info->Size == 0) {
			break;
		}
		if (info->Type == CpuSetInformation) {
			const auto&      set = info->CpuSet;
			LogicalProcessor lp;
			lp.os_id      = set.Id;
			lp.number     = static_cast<uint32_t>(set.Group) * 64u + set.LogicalProcessorIndex;
			lp.core       = (static_cast<uint32_t>(set.Group) << 8u) | set.CoreIndex;
			lp.llc        = (static_cast<uint32_t>(set.Group) << 8u) | set.LastLevelCacheIndex;
			lp.efficiency = set.EfficiencyClass;
			const auto l3 = l3_of.find(lp.number);
			lp.l3         = l3 != l3_of.end() ? l3->second : 0;
			t.all.push_back(lp);
		}
		offset += info->Size;
	}
	return !t.all.empty();
}

static void DisablePowerThrottling(HANDLE thread) {
	THREAD_POWER_THROTTLING_STATE state {};
	state.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
	state.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
	state.StateMask   = 0;
	SetThreadInformation(thread, ThreadPowerThrottling, &state, sizeof(state));
}

void InitCpuPlacement() {
	static std::once_flag once;
	std::call_once(once, [] {
		const auto mode = GetMode();
		if (mode == Mode::Off) {
			LOGF("CPU placement: off (KYTY_PCORE_AFFINITY=0)\n");
			return;
		}
		auto& t = GetTopology();
		if (!ReadTopology(t)) {
			LOGF("CPU placement: no CPU set information, nothing done\n");
			return;
		}
		if (mode == Mode::Hybrid) {
			Plan(t, false);
			g_active = t.hybrid;
			LOGF("CPU placement: X3D mode off (KYTY_X3D=1 or --cpu-placement true turns it on)%s\n",
			     t.hybrid ? "; critical and high threads stay on the performance cores" : "");
			return;
		}
		Plan(t, true);
		g_active = true;
		LogTopology(t);
		// Windows may run the process at reduced speed (EcoQoS) when its window is not in front.
		PROCESS_POWER_THROTTLING_STATE power {};
		power.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
		power.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
		power.StateMask   = 0;
		SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &power, sizeof(power));
		if (t.normal.size() != t.all.size()) {
			const bool ok =
			    SetProcessDefaultCpuSets(GetCurrentProcess(), reinterpret_cast<const ULONG*>(t.normal.data()),
			                             static_cast<ULONG>(t.normal.size())) != 0;
			if (!ok) {
				LOGF("CPU placement: SetProcessDefaultCpuSets failed (%u)\n",
				     static_cast<unsigned>(GetLastError()));
			}
		}
	});
}

// The critical and high threads, which ApplyOwnCores may move, with the CPU sets PlaceCurrentThread
// gave them (empty: none, the process default applies).
struct PlacedThread {
	std::string           name;
	HANDLE                handle = nullptr;
	std::vector<uint32_t> ids;
};
static std::mutex                g_placed_mutex;
static std::vector<PlacedThread> g_placed;
static std::atomic<uint32_t>     g_placed_count {0};

static void RememberCurrentThread(const char* name, const std::vector<uint32_t>& ids) {
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
	                    THREAD_SET_LIMITED_INFORMATION, FALSE, 0) == 0) {
		return;
	}
	std::lock_guard lock(g_placed_mutex);
	g_placed.push_back({name, handle, ids});
	g_placed_count.fetch_add(1, std::memory_order_release);
}

void PlaceCurrentThread(ThreadClass cls, const char* name) {
	Profiler::SetThreadName(name);
	InitCpuPlacement();
	auto& t = GetTopology();
	if (t.all.empty()) {
		return;
	}
	if (GetMode() == Mode::Hybrid) {
		// The rule from before X3D mode: the threads the GPU waits on avoid the efficiency cores.
		if (cls != ThreadClass::Critical && cls != ThreadClass::High) {
			return;
		}
		if (g_active) {
			const bool ok = SetThreadSelectedCpuSets(GetCurrentThread(), reinterpret_cast<const ULONG*>(t.fast.data()),
			                                         static_cast<ULONG>(t.fast.size())) != 0;
			LOGF("CPU placement: %s on performance cores %s%s\n", name, FormatList(Numbers(t, t.fast)).c_str(),
			     ok ? "" : " - failed");
		}
		// After its own sets: ApplyOwnCores applies the own cores once it sees the thread.
		RememberCurrentThread(name, g_active ? t.fast : std::vector<uint32_t> {});
		return;
	}
	const std::vector<uint32_t>* ids = nullptr;
	std::string                  where;
	int                          priority = THREAD_PRIORITY_NORMAL;
	switch (cls) {
		case ThreadClass::Critical: {
			priority         = THREAD_PRIORITY_HIGHEST;
			const auto index = t.next_reserved.fetch_add(1);
			if (index < t.reserved.size()) {
				ids   = &t.reserved[index];
				where = "own core";
			} else {
				ids   = &t.high;
				where = "fast cores";
			}
			break;
		}
		case ThreadClass::High:
			priority = THREAD_PRIORITY_ABOVE_NORMAL;
			ids      = &t.high;
			where    = "fast cores";
			break;
		case ThreadClass::Background:
			ids   = &t.background;
			where = t.slow.empty() ? "shared cores" : "slow cores";
			break;
		case ThreadClass::Low:
			priority = THREAD_PRIORITY_BELOW_NORMAL;
			ids      = &t.background;
			where    = t.slow.empty() ? "shared cores" : "slow cores";
			break;
	}
	bool ok = true;
	if (ids != nullptr && !ids->empty() && ids->size() != t.all.size()) {
		ok = SetThreadSelectedCpuSets(GetCurrentThread(), reinterpret_cast<const ULONG*>(ids->data()),
		                              static_cast<ULONG>(ids->size())) != 0;
	}
	if (priority != THREAD_PRIORITY_NORMAL) {
		ok = SetThreadPriority(GetCurrentThread(), priority) != 0 && ok;
	}
	if (cls == ThreadClass::Critical || cls == ThreadClass::High) {
		DisablePowerThrottling(GetCurrentThread());
	}
	LOGF("CPU placement: %s (%s) on %s %s%s\n", name, ClassName(cls), where.c_str(),
	     ids != nullptr ? FormatList(Numbers(t, *ids)).c_str() : "-", ok ? "" : " - failed");
}

// ufc57: in its busiest seconds every GPU-thread zone ran 10-40 % slower per call, trivial ones
// too, while the game's threads were busier; as if the GPU thread shared its core with them.
// ufc58, A/B on a 9800X3D with the GPU thread alone on logical processors 2-3: +3.6 % +-1.8 % fps.
// ufc59, the Vulkan recording thread alone on 4-5 as well: +3.8 % +-2.5 %.
static constexpr uint32_t kAutoCore = UINT32_MAX;

// The default own cores, on CPUs with 8 or more cores: the fast core `index` (core 0 takes most
// interrupts, so the GPU thread gets 1, the recording thread 2). None on smaller CPUs, whose cores
// the guest's threads need. Returns its lowest logical processor number, or 0.
static uint32_t AutoCore(const Topology& t, size_t index) {
	std::set<uint32_t>           cores;
	std::map<uint32_t, uint32_t> fast_cores; // core key -> its lowest logical processor number
	for (const auto& lp: t.all) {
		cores.insert(lp.core);
		if (std::find(t.fast.begin(), t.fast.end(), lp.os_id) != t.fast.end()) {
			const auto it = fast_cores.emplace(lp.core, lp.number).first;
			it->second    = std::min(it->second, lp.number);
		}
	}
	if (cores.size() < 8 || fast_cores.size() <= index) {
		return 0;
	}
	return std::next(fast_cores.begin(), static_cast<ptrdiff_t>(index))->second;
}

static const LogicalProcessor* FindProcessor(const Topology& t, uint32_t number) {
	const auto it = std::find_if(t.all.begin(), t.all.end(), [&](const LogicalProcessor& lp) {
		return lp.number == number;
	});
	return it != t.all.end() ? &*it : nullptr;
}

void ApplyOwnCores() {
	static const auto parse = [](const char* value) {
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	};
	static const auto parse_auto = [](const char* value) {
		return value != nullptr ? parse(value) : kAutoCore;
	};
	static const AbSwitch<uint32_t> gpu("KYTY_GPU_OWN_CORE", parse_auto);
	static const AbSwitch<uint32_t> recording("KYTY_RECORDING_OWN_CORE", parse_auto);
	static const AbSwitch<uint32_t> scanner("KYTY_SCANNER_OWN_CORE", parse);
	static constexpr const char*    kThreads[] = {"GPU thread", "Vulkan recording thread",
	                                              "draw preparation scanner"};
	static constexpr const char*    kVariables[] = {"KYTY_GPU_OWN_CORE", "KYTY_RECORDING_OWN_CORE",
	                                                "KYTY_SCANNER_OWN_CORE"};
	// GPU thread only. Applied again when a critical or high thread starts, which takes the
	// performance cores (hybrid rule) or the process default and may land on an own core.
	static std::array<uint32_t, 3> applied {0, 0, 0};
	static uint32_t                applied_threads = 0;
	const std::array<uint32_t, 3>  wanted {gpu.Get(), recording.Get(), scanner.Get()};
	const auto                     threads = g_placed_count.load(std::memory_order_acquire);
	if (wanted == applied && threads == applied_threads) {
		return;
	}
	const bool changed = wanted != applied;
	applied            = wanted;
	applied_threads    = threads;
	auto& t            = GetTopology();
	if (GetMode() != Mode::Hybrid || t.all.empty()) {
		if (changed && std::any_of(wanted.begin(), wanted.end(),
		                           [](uint32_t n) { return n != 0 && n != kAutoCore; })) {
			LOGF("CPU placement: own cores ignored (%s)\n",
			     t.all.empty() ? "no CPU set information" : "X3D mode places the threads");
		}
		return;
	}
	std::vector<uint32_t> core_ids[3];
	std::set<uint32_t>    reserved;
	for (size_t role = 0; role < 3; role++) {
		const auto  number = wanted[role] == kAutoCore ? AutoCore(t, role + 1) : wanted[role];
		const auto* chosen = FindProcessor(t, number);
		if (number == 0 || chosen == nullptr) {
			if (number != 0 && changed) {
				LOGF("CPU placement: %s=%u: no such logical processor\n", kVariables[role], number);
			}
			continue;
		}
		for (const auto& lp: t.all) {
			if (lp.core == chosen->core) {
				core_ids[role].push_back(lp.os_id);
				reserved.insert(lp.os_id);
			}
		}
	}
	std::vector<uint32_t> others;
	for (const auto& lp: t.all) {
		if (reserved.count(lp.os_id) == 0) {
			others.push_back(lp.os_id);
		}
	}
	// An empty set clears: no own core leaves no process default (every processor) and each thread
	// with PlaceCurrentThread's sets.
	const auto data = [](const std::vector<uint32_t>& ids) {
		return ids.empty() ? nullptr : reinterpret_cast<const ULONG*>(ids.data());
	};
	const auto size = [](const std::vector<uint32_t>& ids) {
		return static_cast<ULONG>(ids.size());
	};
	const auto& process_ids = reserved.empty() ? std::vector<uint32_t> {} : others;
	bool ok =
	    SetProcessDefaultCpuSets(GetCurrentProcess(), data(process_ids), size(process_ids)) != 0;
	{
		std::lock_guard lock(g_placed_mutex);
		for (const auto& thread: g_placed) {
			const auto role = std::find(std::begin(kThreads), std::end(kThreads), thread.name) -
			                  std::begin(kThreads);
			std::vector<uint32_t> ids;
			if (role < 3 && !core_ids[role].empty()) {
				ids = core_ids[role];
			} else {
				std::copy_if(thread.ids.begin(), thread.ids.end(), std::back_inserter(ids),
				             [&](uint32_t id) { return reserved.count(id) == 0; });
			}
			ok = SetThreadSelectedCpuSets(thread.handle, data(ids), size(ids)) != 0 && ok;
		}
	}
	if (!changed && ok) {
		return;
	}
	std::string line;
	for (size_t role = 0; role < 3; role++) {
		if (!core_ids[role].empty()) {
			line += (line.empty() ? "" : ", ") + std::string(kThreads[role]) + " alone on " +
			        FormatList(Numbers(t, core_ids[role]));
		}
	}
	line = line.empty() ? "no own cores"
	                    : line + "; the others on " + FormatList(Numbers(t, others));
	const char* failed = ok ? "" : " - failed";
	LOGF("CPU placement: %s%s\n", line.c_str(), failed);
	Profiler::Message(ok ? Profiler::MessageInfo : Profiler::MessageFailure, "CPU placement: %s%s",
	                  line.c_str(), failed);
}

#else

static bool ReadNumber(const std::string& path, uint64_t& value) {
	std::ifstream file(path);
	return static_cast<bool>(file >> value);
}

// "96K", "32768K", "96M" -> bytes.
static uint64_t ReadSize(const std::string& path) {
	std::ifstream file(path);
	std::string   text;
	if (!(file >> text) || text.empty()) {
		return 0;
	}
	uint64_t value = std::strtoull(text.c_str(), nullptr, 10);
	switch (text.back()) {
		case 'K': value <<= 10u; break;
		case 'M': value <<= 20u; break;
		case 'G': value <<= 30u; break;
		default: break;
	}
	return value;
}

static bool ReadTopology(Topology& t) {
	const long count = sysconf(_SC_NPROCESSORS_CONF);
	for (long cpu = 0; cpu < count; cpu++) {
		const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
		uint64_t          core = 0, package = 0, khz = 0;
		if (!ReadNumber(base + "/topology/core_id", core)) {
			continue; // Offline.
		}
		ReadNumber(base + "/topology/physical_package_id", package);
		ReadNumber(base + "/cpufreq/cpuinfo_max_freq", khz);
		LogicalProcessor lp;
		lp.os_id  = static_cast<uint32_t>(cpu);
		lp.number = static_cast<uint32_t>(cpu);
		lp.core   = static_cast<uint32_t>((package << 16u) | core);
		lp.l3     = ReadSize(base + "/cache/index3/size");
		uint64_t llc = 0;
		if (!ReadNumber(base + "/cache/index3/id", llc)) {
			llc = package;
		}
		lp.llc = static_cast<uint32_t>(llc);
		// Hybrid: efficiency cores top out well below the performance cores (Raptor Lake 4.1
		// against 5.2 GHz), while one CPU's preferred cores differ by a few percent at most;
		// round to 15% steps so only the former count as different classes.
		lp.efficiency = static_cast<uint32_t>(khz / 1000u);
		t.all.push_back(lp);
	}
	uint32_t highest = 0;
	for (const auto& lp: t.all) {
		highest = std::max(highest, lp.efficiency);
	}
	for (auto& lp: t.all) {
		lp.efficiency = lp.efficiency * 100u >= highest * 85u ? 1u : 0u;
	}
	return !t.all.empty();
}

void InitCpuPlacement() {
	static std::once_flag once;
	std::call_once(once, [] {
		const auto mode = GetMode();
		if (mode == Mode::Off) {
			LOGF("CPU placement: off (KYTY_PCORE_AFFINITY=0)\n");
			return;
		}
		auto& t = GetTopology();
		if (!ReadTopology(t)) {
			return;
		}
		// Linux has no process-wide default that spares reserved cores without also binding every
		// thread created later, so nothing is reserved.
		Plan(t, false);
		g_active = mode == Mode::X3d || t.hybrid;
		if (mode == Mode::X3d) {
			LogTopology(t);
		}
	});
}

void PlaceCurrentThread(ThreadClass cls, const char* name) {
	Profiler::SetThreadName(name);
	InitCpuPlacement();
	auto& t = GetTopology();
	if (!g_active || t.all.empty()) {
		return;
	}
	const bool hot = cls == ThreadClass::Critical || cls == ThreadClass::High;
	if (GetMode() == Mode::Hybrid && !hot) {
		return; // The hybrid rule only keeps the hot threads off the efficiency cores.
	}
	// Affinity is inherited by threads created later, so only whole sets are used: the fast cores
	// for the critical and high threads, the slow ones for background work.
	const auto& ids = hot ? t.fast : t.background;
	if (ids.empty() || ids.size() == t.all.size()) {
		return;
	}
	cpu_set_t set;
	CPU_ZERO(&set);
	for (const auto id: ids) {
		CPU_SET(static_cast<int>(id), &set);
	}
	const bool ok = pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
	LOGF("CPU placement: %s (%s) on %s%s\n", name, ClassName(cls), FormatList(Numbers(t, ids)).c_str(),
	     ok ? "" : " - failed");
}

void ApplyOwnCores() {}

#endif

#else

void InitCpuPlacement() {}
void PlaceCurrentThread(ThreadClass /*cls*/, const char* name) {
	Profiler::SetThreadName(name);
}
void ApplyOwnCores() {}

#endif

} // namespace Common
