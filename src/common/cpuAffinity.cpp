#include "common/cpuAffinity.h"

#include "common/common.h"
#include "common/logging/log.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
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

static std::atomic<bool> g_cli_enabled {true};

void SetCpuPlacementEnabled(bool enabled) {
	g_cli_enabled.store(enabled);
}

#if !defined(__APPLE__)

static bool Enabled() {
	static const bool enabled = [] {
		const char* x3d    = std::getenv("KYTY_X3D");
		const char* legacy = std::getenv("KYTY_PCORE_AFFINITY");
		return g_cli_enabled.load() && (x3d == nullptr || x3d[0] != '0') &&
		       (legacy == nullptr || legacy[0] != '0');
	}();
	return enabled;
}

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
		if (!Enabled()) {
			LOGF("CPU placement: off (KYTY_X3D=0 or --cpu-placement false)\n");
			return;
		}
		auto& t = GetTopology();
		if (!ReadTopology(t)) {
			LOGF("CPU placement: no CPU set information, nothing done\n");
			return;
		}
		Plan(t, true);
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

void PlaceCurrentThread(ThreadClass cls, const char* name) {
	InitCpuPlacement();
	auto& t = GetTopology();
	if (!Enabled() || t.all.empty()) {
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
		if (!Enabled()) {
			LOGF("CPU placement: off (KYTY_X3D=0 or --cpu-placement false)\n");
			return;
		}
		auto& t = GetTopology();
		if (!ReadTopology(t)) {
			return;
		}
		// Linux has no process-wide default that spares reserved cores without also binding every
		// thread created later, so nothing is reserved.
		Plan(t, false);
		LogTopology(t);
	});
}

void PlaceCurrentThread(ThreadClass cls, const char* name) {
	InitCpuPlacement();
	auto& t = GetTopology();
	if (!Enabled() || t.all.empty()) {
		return;
	}
	// Affinity is inherited by threads created later, so only whole sets are used: the fast cores
	// for the critical and high threads, the slow ones for background work.
	const auto& ids =
	    (cls == ThreadClass::Critical || cls == ThreadClass::High) ? t.fast : t.background;
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

#endif

#else

void InitCpuPlacement() {}
void PlaceCurrentThread(ThreadClass /*cls*/, const char* /*name*/) {}

#endif

} // namespace Common
