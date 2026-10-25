#include "common/profiler.h"

#include "common/emulatorConfig.h"

#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <tracy/Tracy.hpp>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#include <intrin.h>
#include <tlhelp32.h>
#endif

namespace Profiler {

namespace {

enum class CounterKind : uint8_t { Events, Bytes, TimeUs, TimeNs, Gauge };

struct CounterInfo {
	const char* name;
	CounterKind kind;
	uint32_t    color;
};

// Plot names are compared by pointer: each must stay this one string.
constexpr CounterInfo kCounterInfo[] = {
    {"Draws", CounterKind::Events, 0x64b5f6},
    {"Dispatches", CounterKind::Events, 0x4dd0e1},
    {"Skipped: shader gave up", CounterKind::Events, 0xff5252},
    {"Skipped: skip list", CounterKind::Events, 0x9e9e9e},
    {"Skipped: resources unreadable", CounterKind::Events, 0xff8a80},
    {"Skipped: pipeline compiling", CounterKind::Events, 0xffab40},
    {"Shaders compiled", CounterKind::Events, 0x81c784},
    {"Shaders gave up", CounterKind::Events, 0xff5252},
    {"Shader compile ms", CounterKind::TimeUs, 0xa5d6a7},
    {"Graphics pipelines built", CounterKind::Events, 0x66bb6a},
    {"Compute pipelines built", CounterKind::Events, 0x43a047},
    {"Pipeline build ms", CounterKind::TimeUs, 0xc8e6c9},
    {"Pipelines pending", CounterKind::Gauge, 0xffd54f},
    {"SRT refreshes", CounterKind::Events, 0xce93d8},
    {"SRT replayed", CounterKind::Events, 0xba68c8},
    {"SRT abandoned", CounterKind::Events, 0xff6e40},
    {"SRT recorded", CounterKind::Events, 0xf48fb1},
    {"GPU waits", CounterKind::Events, 0xe57373},
    {"GPU wait ms", CounterKind::TimeUs, 0xef5350},
    {"Readbacks", CounterKind::Events, 0xffb74d},
    {"Readback bytes", CounterKind::Bytes, 0xffa726},
    {"Command buffers submitted", CounterKind::Events, 0x90a4ae},
    {"Recording drains", CounterKind::Events, 0xff7043},
    {"Upload bytes", CounterKind::Bytes, 0x4fc3f7},
    {"Images created", CounterKind::Events, 0xaed581},
    {"Images destroyed", CounterKind::Events, 0xdce775},
    {"Bindless keys resolved", CounterKind::Events, 0x7986cb},
    {"CPU write faults", CounterKind::Events, 0xf06292},
    {"CPU read faults", CounterKind::Events, 0xba68c8},
    {"CPU fault ms", CounterKind::TimeNs, 0xf48fb1},
};
static_assert(std::size(kCounterInfo) == static_cast<size_t>(Counter::Total));

void ConfigurePlots() {
	for (const auto& info: kCounterInfo) {
		const auto format = info.kind == CounterKind::Bytes ? tracy::PlotFormatType::Memory
		                                                    : tracy::PlotFormatType::Number;
		const bool time = info.kind == CounterKind::TimeUs || info.kind == CounterKind::TimeNs;
		tracy::Profiler::ConfigurePlot(info.name, format, true, !time, info.color);
	}
}

void PlotCounters() {
	for (size_t index = 0; index < std::size(kCounterInfo); index++) {
		const auto& info  = kCounterInfo[index];
		auto&       value = g_counter_slots[index].value;
		switch (info.kind) {
			case CounterKind::Gauge:
				tracy::Profiler::PlotData(info.name, value.load(std::memory_order_relaxed));
				break;
			case CounterKind::TimeUs:
				tracy::Profiler::PlotData(
				    info.name, static_cast<double>(value.exchange(0, std::memory_order_relaxed)) / 1000.0);
				break;
			case CounterKind::TimeNs:
				tracy::Profiler::PlotData(
				    info.name, static_cast<double>(value.exchange(0, std::memory_order_relaxed)) / 1e6);
				break;
			default:
				tracy::Profiler::PlotData(info.name, value.exchange(0, std::memory_order_relaxed));
				break;
		}
	}
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// Once a second while Tracy is connected: how much of a core each of the process's threads used
// over that second, from its cycle count (QueryThreadCycleTime counts TSC ticks), as a message
// "CPU 1000 ms: process 412 % ... | GPU thread 97 % | ..." (the busiest threads, by name) and the
// plots "Process CPU %" and "Busiest thread %". It shows what Tracy's zones cannot: whether the
// guest's own threads, which have none, are what holds a frame back. KYTY_PROFILE_CPU=0: off.
class ThreadCpuMonitor {
public:
	ThreadCpuMonitor(): m_thread([this](std::stop_token stop) { Run(stop); }) {}

	ThreadCpuMonitor(const ThreadCpuMonitor&)            = delete;
	ThreadCpuMonitor& operator=(const ThreadCpuMonitor&) = delete;

	~ThreadCpuMonitor() {
		m_thread.request_stop();
		// An emergency shutdown on the monitor's own thread (a crash there) cannot join it.
		if (m_thread.get_id() == std::this_thread::get_id()) {
			m_thread.detach();
		}
	}

private:
	struct Thread {
		HANDLE   handle = nullptr;
		uint64_t cycles = 0;
		uint64_t busy   = 0; // cycles in the last second
		bool     alive  = false;
	};

	static constexpr const char* ProcessPlot = "Process CPU %";
	static constexpr const char* BusiestPlot = "Busiest thread %";
	static constexpr size_t      MaxListed   = 20;
	// The thread list is read every RefreshSeconds (a Toolhelp snapshot lists every thread of the
	// system: read every second, it took ~6 % of a core and the frame rate dipped for ~50 ms each
	// time, UFC 5 fight, 2026-10-25); in between, only the known threads' cycle counts are read.
	static constexpr uint32_t RefreshSeconds = 10;

	void Run(const std::stop_token& stop) {
		tracy::SetThreadName("Profiler CPU monitor");
		bool     primed      = false;
		uint32_t since_list  = 0;
		uint64_t last_tsc    = 0;
		uint64_t last_times  = 0;
		auto     last_wall   = std::chrono::steady_clock::now();
		for (;;) {
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait_for(lock, stop, std::chrono::seconds(1), [] { return false; });
			}
			if (stop.stop_requested()) {
				break;
			}
			if (!Connected()) {
				Forget();
				primed = false;
				continue;
			}
			const bool list = !primed || ++since_list >= RefreshSeconds;
			if (list) {
				since_list = 0;
			}
			const auto     wall  = std::chrono::steady_clock::now();
			const uint64_t tsc   = __rdtsc();
			const uint64_t times = ProcessTimes();
			Sample(list);
			if (primed) {
				const double wall_ns =
				    static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(wall - last_wall).count());
				Report(tsc - last_tsc, wall_ns, static_cast<double>(times - last_times) * 100.0);
			}
			primed     = true;
			last_tsc   = tsc;
			last_times = times;
			last_wall  = wall;
			// Threads that ended: their last second is reported, now let their ids go.
			for (auto it = m_threads.begin(); it != m_threads.end();) {
				if (!it->second.alive) {
					if (it->second.handle != nullptr) {
						CloseHandle(it->second.handle);
					}
					it = m_threads.erase(it);
				} else {
					++it;
				}
			}
		}
		Forget();
	}

	// Kernel and user time of the whole process, in 100 ns units.
	static uint64_t ProcessTimes() {
		FILETIME creation {};
		FILETIME exit {};
		FILETIME kernel {};
		FILETIME user {};
		if (GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user) == 0) {
			return 0;
		}
		const auto value = [](const FILETIME& time) {
			return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
		};
		return value(kernel) + value(user);
	}

	// list: the thread list is read again (threads started since are found, ended ones dropped);
	// otherwise a known thread is alive until its exit code says it ended.
	void Sample(bool list) {
		const HANDLE snapshot = list ? CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) : INVALID_HANDLE_VALUE;
		for (auto& [id, thread]: m_threads) {
			DWORD code   = 0;
			thread.alive = snapshot == INVALID_HANDLE_VALUE &&
			               (thread.handle == nullptr ||
			                (GetExitCodeThread(thread.handle, &code) != 0 && code == STILL_ACTIVE));
		}
		const DWORD pid = GetCurrentProcessId();
		if (snapshot != INVALID_HANDLE_VALUE) {
			THREADENTRY32 entry {};
			entry.dwSize = sizeof(entry);
			for (BOOL ok = Thread32First(snapshot, &entry); ok != 0; ok = Thread32Next(snapshot, &entry)) {
				if (entry.th32OwnerProcessID == pid) {
					auto& thread = m_threads[entry.th32ThreadID];
					if (thread.handle == nullptr) {
						// Held until the thread is gone, so its id cannot be reused meanwhile.
						thread.handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
					}
					thread.alive = true;
				}
				entry.dwSize = sizeof(entry);
			}
			CloseHandle(snapshot);
		}
		// A thread first seen now started within the last second: all its cycles count.
		for (auto& [id, thread]: m_threads) {
			ULONG64 cycles = thread.cycles;
			if (thread.handle != nullptr) {
				QueryThreadCycleTime(thread.handle, &cycles);
			}
			thread.busy   = cycles >= thread.cycles ? cycles - thread.cycles : 0;
			thread.cycles = cycles;
		}
	}

	static std::string Name(HANDLE handle, DWORD id) {
		using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
		static const auto get_description = reinterpret_cast<GetDescription>(
		    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription")));
		std::string name;
		PWSTR       wide = nullptr;
		if (handle != nullptr && get_description != nullptr && SUCCEEDED(get_description(handle, &wide)) &&
		    wide != nullptr) {
			const int size = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
			if (size > 1) {
				name.resize(static_cast<size_t>(size));
				WideCharToMultiByte(CP_UTF8, 0, wide, -1, name.data(), size, nullptr, nullptr);
				name.resize(static_cast<size_t>(size - 1));
			}
			LocalFree(wide);
		}
		if (name.empty()) {
			name = "thread " + std::to_string(id);
		}
		return name;
	}

	void Report(uint64_t tsc_ticks, double wall_ns, double times_ns) {
		if (tsc_ticks == 0 || wall_ns <= 0) {
			return;
		}
		std::vector<std::pair<uint64_t, DWORD>> order;
		order.reserve(m_threads.size());
		uint64_t total = 0;
		for (const auto& [id, thread]: m_threads) {
			total += thread.busy;
			order.emplace_back(thread.busy, id);
		}
		std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		const auto percent = [&](uint64_t cycles) {
			return static_cast<double>(cycles) * 100.0 / static_cast<double>(tsc_ticks);
		};
		// "times": GetProcessTimes over the same second, a check on the cycle-based total.
		char text[2048];
		int  size = std::snprintf(text, sizeof(text),
		                          "CPU %.0f ms: process %.0f %% (times %.0f %%), %zu threads, TSC %.2f GHz",
		                          wall_ns / 1e6, percent(total), times_ns * 100.0 / wall_ns, order.size(),
		                          static_cast<double>(tsc_ticks) / wall_ns);
		for (size_t i = 0; i < order.size() && i < MaxListed && size > 0 && static_cast<size_t>(size) < sizeof(text);
		     i++) {
			const double share = percent(order[i].first);
			if (share < 1.0) {
				break;
			}
			const auto name = Name(m_threads[order[i].second].handle, order[i].second);
			size += std::snprintf(text + size, sizeof(text) - static_cast<size_t>(size), " | %s %.0f %%",
			                      name.c_str(), share);
		}
		if (size > 0) {
			tracy::Profiler::LogString(tracy::MessageSourceType::User, tracy::MessageSeverity::Info, 0x80cbc4, 0,
			                           std::min(static_cast<size_t>(size), sizeof(text) - 1), text);
		}
		tracy::Profiler::PlotData(ProcessPlot, percent(total));
		tracy::Profiler::PlotData(BusiestPlot, order.empty() ? 0.0 : percent(order.front().first));
	}

	void Forget() {
		for (auto& [id, thread]: m_threads) {
			if (thread.handle != nullptr) {
				CloseHandle(thread.handle);
			}
		}
		m_threads.clear();
	}

	std::unordered_map<DWORD, Thread> m_threads;
	std::mutex                        m_mutex;
	std::condition_variable_any       m_wake;
	std::jthread                      m_thread; // last: it starts once the members above exist
};

std::atomic<ThreadCpuMonitor*> g_cpu_monitor = nullptr;

void StartCpuMonitor() {
	if (const char* cpu = std::getenv("KYTY_PROFILE_CPU"); cpu != nullptr && cpu[0] == '0') {
		return;
	}
	if (g_cpu_monitor.load() == nullptr) {
		delete g_cpu_monitor.exchange(new ThreadCpuMonitor);
	}
}

void StopCpuMonitor() {
	delete g_cpu_monitor.exchange(nullptr);
}

#else

void StartCpuMonitor() {}
void StopCpuMonitor() {}

#endif

} // namespace

void SetThreadName(const char* name) {
	if (tracy::ProfilerAvailable() && name != nullptr) {
		tracy::SetThreadName(name);
	}
}

void MarkFrame() {
	if (tracy::ProfilerAvailable()) {
		FrameMark;
		if (g_counters) {
			PlotCounters();
		}
	}
}

bool Connected() {
	return tracy::ProfilerAvailable() && tracy::GetProfiler().IsConnected();
}

void Message(uint32_t color, const char* format, ...) {
	if (!Connected()) {
		return;
	}
	char    text[512];
	va_list args;
	va_start(args, format);
	const int size = std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	if (size <= 0) {
		return;
	}
	tracy::Profiler::LogString(tracy::MessageSourceType::User, tracy::MessageSeverity::Info, color,
	                           0, std::min(static_cast<size_t>(size), sizeof(text) - 1), text);
}

void Phases::SetText(const char* format, ...) {
	if (!Connected()) {
		m_text_size = 0;
		return;
	}
	va_list args;
	va_start(args, format);
	const int size = std::vsnprintf(m_text, sizeof(m_text), format, args);
	va_end(args);
	m_text_size = size <= 0 ? 0 : std::min(static_cast<size_t>(size), sizeof(m_text) - 1);
	if (m_zone && m_text_size != 0) {
		m_zone->Text(m_text, m_text_size);
	}
}

void Phases::Begin(const tracy::SourceLocationData* location) {
	m_zone.reset();
	if (!tracy::ProfilerAvailable()) {
		return;
	}
	m_zone.emplace(location, TRACY_CALLSTACK, true);
	if (m_text_size != 0) {
		m_zone->Text(m_text, m_text_size);
	}
}

namespace {
thread_local CommandZone* t_command_zone = nullptr;
} // namespace

CommandZone::CommandZone(const tracy::SourceLocationData* location) {
	if (Connected()) {
		m_zone.emplace(location, TRACY_CALLSTACK, true);
		m_previous     = t_command_zone;
		t_command_zone = this;
	}
}

CommandZone::~CommandZone() {
	if (m_zone) {
		t_command_zone = m_previous;
	}
}

void CommandZone::Annotate(const char* format, ...) {
	auto* zone = t_command_zone;
	if (zone == nullptr || !zone->m_zone) {
		return;
	}
	char    text[160];
	va_list args;
	va_start(args, format);
	const int size = std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	if (size > 0) {
		zone->m_zone->Text(text, std::min(static_cast<size_t>(size), sizeof(text) - 1));
	}
}

void Initialize() {
	if (const char* zones = std::getenv("KYTY_PROFILE_ZONES"); zones != nullptr && zones[0] == '0') {
		g_zones = false;
	}
	if (const char* fine = std::getenv("KYTY_PROFILE_FINE"); fine != nullptr && fine[0] == '1') {
		g_fine_zones = true;
	}
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		TracySetProgramName("KytyPS5");
		ConfigurePlots();
		g_counters = true;
		StartCpuMonitor();
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
	}
}

void Shutdown() {
	StopCpuMonitor();
	g_counters = false;
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler
