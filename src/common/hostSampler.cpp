#include "common/hostSampler.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h first
#include <dbghelp.h>
#include <psapi.h>
#include <timeapi.h>
#endif

namespace Common {

#ifdef _WIN32
namespace {

constexpr size_t MaxFrames = 32;
uint64_t         g_exe_base = 0; // the executable's range (Report)
uint64_t         g_exe_end  = 0;

struct Sample {
	std::array<uint64_t, MaxFrames> frames {};
	uint32_t                        count = 0;
};

// Every loaded module's function table (.pdata), read once at the start and searched without
// RtlLookupFunctionEntry's locks (the suspended thread may hold them, inside a fault's dispatch).
struct FunctionTable {
	struct Module {
		uint64_t                base  = 0;
		uint64_t                end   = 0;
		const RUNTIME_FUNCTION* table = nullptr;
		size_t                  count = 0;
	};
	std::vector<Module> modules; // sorted by base
	uint64_t            base = 0; // the executable's
	uint64_t            end  = 0;

	static bool Read(uint64_t module_base, Module& out) {
		const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module_base);
		if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
			return false;
		}
		const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(module_base + dos->e_lfanew);
		if (nt->Signature != IMAGE_NT_SIGNATURE ||
		    nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
			return false;
		}
		out.base        = module_base;
		out.end         = module_base + nt->OptionalHeader.SizeOfImage;
		const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
		if (dir.VirtualAddress == 0 || dir.Size == 0) {
			return false;
		}
		out.table = reinterpret_cast<const RUNTIME_FUNCTION*>(module_base + dir.VirtualAddress);
		out.count = dir.Size / sizeof(RUNTIME_FUNCTION);
		return true;
	}

	bool Load() {
		HMODULE handles[1024];
		DWORD   needed = 0;
		if (EnumProcessModules(GetCurrentProcess(), handles, sizeof(handles), &needed) == FALSE) {
			return false;
		}
		for (DWORD i = 0; i < std::min<DWORD>(needed / sizeof(HMODULE), 1024); i++) {
			Module module;
			if (Read(reinterpret_cast<uint64_t>(handles[i]), module)) {
				modules.push_back(module);
			}
		}
		std::sort(modules.begin(), modules.end(),
		          [](const Module& a, const Module& b) { return a.base < b.base; });
		const auto exe = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
		for (const auto& module: modules) {
			if (module.base == exe) {
				base = module.base;
				end  = module.end;
			}
		}
		return base != 0;
	}

	[[nodiscard]] const Module* ModuleOf(uint64_t address) const {
		auto it = std::upper_bound(modules.begin(), modules.end(), address,
		                           [](uint64_t a, const Module& m) { return a < m.base; });
		if (it == modules.begin()) {
			return nullptr;
		}
		--it;
		return address < it->end ? &*it : nullptr;
	}

	[[nodiscard]] PRUNTIME_FUNCTION Find(uint64_t address, uint64_t& module_base) const {
		const auto* module = ModuleOf(address);
		if (module == nullptr) {
			return nullptr;
		}
		module_base    = module->base;
		const auto rva = static_cast<uint32_t>(address - module->base);
		size_t     lo = 0, hi = module->count;
		while (lo < hi) {
			const auto mid = (lo + hi) / 2;
			if (rva < module->table[mid].BeginAddress) {
				hi = mid;
			} else if (rva >= module->table[mid].EndAddress) {
				lo = mid + 1;
			} else {
				return const_cast<PRUNTIME_FUNCTION>(&module->table[mid]);
			}
		}
		return nullptr;
	}
};

void Unwind(const FunctionTable& functions, CONTEXT context, Sample& sample) {
	for (sample.count = 0; sample.count < MaxFrames;) {
		const auto rip = context.Rip;
		if (rip == 0) {
			break;
		}
		sample.frames[sample.count++] = rip;
		uint64_t module_base = 0;
		auto*    entry       = functions.Find(rip, module_base);
		if (entry == nullptr) {
			if (functions.ModuleOf(rip) != nullptr && sample.count == 1) {
				// A leaf function (no unwind data): its return address is on top of the stack.
				context.Rip = *reinterpret_cast<const uint64_t*>(context.Rsp);
				context.Rsp += sizeof(uint64_t);
				continue;
			}
			// No module (generated code) or no unwind data mid-stack: the sample ends here.
			break;
		}
		void*   handler_data = nullptr;
		DWORD64 frame        = 0;
		RtlVirtualUnwind(UNW_FLAG_NHANDLER, module_base, rip, entry, &context, &handler_data, &frame,
		                 nullptr);
		if (context.Rip == rip) {
			break;
		}
	}
}

std::string Symbolize(HANDLE process, uint64_t address) {
	alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512] {};
	auto*                     symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
	symbol->SizeOfStruct             = sizeof(SYMBOL_INFO);
	symbol->MaxNameLen               = 512;
	DWORD64 displacement             = 0;
	if (SymFromAddr(process, address, &displacement, symbol) != FALSE) {
		return symbol->Name;
	}
	char module[MAX_PATH] {};
	HMODULE handle = nullptr;
	if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(address), &handle) != FALSE &&
	    GetModuleFileNameA(handle, module, MAX_PATH) != 0) {
		std::string name = module;
		if (const auto slash = name.find_last_of("\\/"); slash != std::string::npos) {
			name = name.substr(slash + 1);
		}
		return "[" + name + "]";
	}
	return fmt::format("0x{:016x}", address);
}

void Report(const char* name, const std::vector<Sample>& samples) {
	const auto process = GetCurrentProcess();
	char       exe_path[MAX_PATH] {};
	GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
	std::string search = exe_path;
	if (const auto slash = search.find_last_of("\\/"); slash != std::string::npos) {
		search.resize(slash);
	}
	SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
	(void)SymInitialize(process, search.c_str(), TRUE);
	std::unordered_map<uint64_t, std::string> names;
	const auto name_of = [&](uint64_t address) -> const std::string& {
		auto it = names.find(address);
		if (it == names.end()) {
			it = names.emplace(address, Symbolize(process, address)).first;
		}
		return it->second;
	};
	std::unordered_map<std::string, uint64_t> self, inclusive, callers, leaves;
	for (const auto& sample: samples) {
		if (sample.count == 0) {
			continue;
		}
		self[name_of(sample.frames[0])]++;
		// Raw leaf addresses, 16-byte buckets (llvm-symbolizer --inlining names the inlined code).
		leaves[fmt::format("0x{:x} {}", sample.frames[0] & ~uint64_t {0xf}, name_of(sample.frames[0]))]++;
		std::unordered_set<std::string> seen;
		for (uint32_t i = 0; i < sample.count; i++) {
			const auto& function = name_of(sample.frames[i]);
			if (seen.insert(function).second) {
				inclusive[function]++;
			}
		}
		// System and driver leaves by the first executable frame above them.
		if (sample.count >= 2 && (sample.frames[0] < g_exe_base || sample.frames[0] >= g_exe_end)) {
			for (uint32_t i = 1; i < sample.count; i++) {
				if (sample.frames[i] >= g_exe_base && sample.frames[i] < g_exe_end) {
					callers[name_of(sample.frames[0]) + " <- " + name_of(sample.frames[i])]++;
					break;
				}
			}
		}
	}
	const auto total = static_cast<double>(samples.size());
	const auto top   = [&](const std::unordered_map<std::string, uint64_t>& counts, size_t limit) {
        std::vector<std::pair<std::string, uint64_t>> sorted(counts.begin(), counts.end());
        std::sort(sorted.begin(), sorted.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string out;
        for (size_t i = 0; i < std::min(limit, sorted.size()); i++) {
			out += fmt::format("  {:5.1f}%  {}\n", 100.0 * static_cast<double>(sorted[i].second) / total,
			                   sorted[i].first);
		}
        return out;
	};
	const auto report = fmt::format(
	    "HostSampler {}: {} samples\n--- self ---\n{}--- inclusive ---\n{}--- system/driver leaves by caller ---\n{}"
	    "--- leaf addresses ---\n{}--- end ---\n",
	    name, samples.size(), top(self, 60), top(inclusive, 90), top(callers, 25), top(leaves, 80));
	LOGF("%s", report.c_str());
	// Also whole in its own file (the log interleaves other threads' lines).
	if (FILE* file = std::fopen("_host_sampler.txt", "wb"); file != nullptr) {
		std::fwrite(report.data(), 1, report.size(), file);
		std::fclose(file);
	}
}

void Run(HANDLE thread, std::string name, double delay, double seconds) {
	FunctionTable functions;
	if (!functions.Load()) {
		LOGF("HostSampler: no function table\n");
		return;
	}
	g_exe_base = functions.base;
	g_exe_end  = functions.end;
	timeBeginPeriod(1);
	std::this_thread::sleep_for(std::chrono::duration<double>(delay));
	LOGF("HostSampler %s: sampling for %.0f s\n", name.c_str(), seconds);
	std::vector<Sample> samples;
	samples.reserve(static_cast<size_t>(seconds * 1200.0));
	const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
	while (std::chrono::steady_clock::now() < end) {
		if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
			CONTEXT context {};
			context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
			if (GetThreadContext(thread, &context) != FALSE) {
				Sample sample;
				Unwind(functions, context, sample);
				samples.push_back(sample);
			}
			ResumeThread(thread);
		}
		Sleep(1);
	}
	timeEndPeriod(1);
	Report(name.c_str(), samples);
}

} // namespace
#endif

void SampleCurrentThread(const char* name) {
#ifdef _WIN32
	const char* value = std::getenv("KYTY_HOST_SAMPLER");
	if (value == nullptr) {
		return;
	}
	char*        end     = nullptr;
	const double delay   = std::strtod(value, &end);
	const double seconds = end != nullptr && *end == ':' ? std::strtod(end + 1, nullptr) : 20.0;
	HANDLE       thread  = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
	                                  FALSE, GetCurrentThreadId());
	if (thread == nullptr) {
		return;
	}
	std::thread(Run, thread, std::string(name), delay, seconds).detach();
#else
	(void)name;
#endif
}

} // namespace Common
