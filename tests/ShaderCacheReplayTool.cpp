// Offline replay of the shader disk cache (shaderDiskCache.h): every permutation entry's key holds
// what its translation read (the guest code, the stage input, the options, the resource
// specialization and the push-data start), so the tool translates and compiles each one again as
// ProgramCache::Get does, without the game.
//
// Uses: proving that a recompiler change leaves the SPIR-V of recorded shaders byte-identical (run
// the tool built before and after the change and compare the hash columns), and counting how the
// recorded shaders fare in a translation variant (table mode: --table).
//
// usage: shader_cache_replay_tool <cache folder> <log file> <out tsv> [--version <hex>]
//                                 [--first <n>] [--end <n>] [--table] [--dump-dir <folder>]
// The entries are the folder's .bin files in name order whose session has the given recompiler
// source version (default: the version of the newest file). The session's device switches and
// KYTY_* switch values are applied before the first translation. Prints "START <n> <file>" before
// each entry, so a driver can resume after a fatal error (exit code 321) or a crash at <n> + 1.
// One tab-separated line per permutation entry goes to <out tsv>, appended:
//   n, file, stage, hash, normal SPIR-V words, its XXH64 (0: no SPIR-V), valid (spirv-val),
//   the cached SPIR-V equal (1/0; with a * when the cache came from another source version),
//   then with --table the table mode columns: the refusal (TableRefusalName; "eligible" when it
//   compiled), the census's verdict on the source's plan with the permutation's specialization
//   and whether the two agree (a ShaderData refusal is the compile's alone), the SPIR-V's words,
//   XXH64 and spirv-val result, and the plan: slots, slots the shader reads, slots the host
//   evaluates, buffers, images, samplers, block dwords and whether it reads flat/global memory.
// --dump-dir writes each entry's modules there as <n>_normal.spv and <n>_table.spv.

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "spirv-tools/libspirv.hpp"
#include "ShaderCacheEntries.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <vector>
#include <xxhash.h>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace Libs::Graphics {
namespace {

namespace Disk = ShaderDiskCache;
namespace IR   = ShaderRecompiler::IR;

#ifdef _WIN32
// A crash (not a fatal error, which exits with 321) would lose the buffered output: print the
// fault on one CRASH line before the process dies.
LONG WINAPI ReportCrash(EXCEPTION_POINTERS* info) {
	const auto code = info->ExceptionRecord->ExceptionCode;
	if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_STACK_OVERFLOW &&
	    code != EXCEPTION_ILLEGAL_INSTRUCTION && code != EXCEPTION_INT_DIVIDE_BY_ZERO &&
	    code != EXCEPTION_PRIV_INSTRUCTION && code != EXCEPTION_ARRAY_BOUNDS_EXCEEDED) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	std::printf("CRASH\t0x%08lx\n", static_cast<unsigned long>(code));
	std::fflush(stdout);
	return EXCEPTION_CONTINUE_SEARCH;
}
#endif

using namespace ShaderCacheEntries;

struct Compiled {
	std::vector<uint32_t> spirv;
	bool                  unsupported = false;
	IR::TableRefusal      refusal     = IR::TableRefusal::Unknown;
	IR::TableRefusal      census      = IR::TableRefusal::Unknown;
	IR::TablePlan         plan;
	uint32_t              block_dwords = 0;
};

Compiled Compile(const Entry& entry, bool table) {
	const std::vector<uint32_t> user_data(entry.user_data_size, 0u);
	auto                        options    = Options(entry, user_data);
	auto                        translated = ShaderRecompiler::TranslateProgram(entry.code, options);
	if (translated.unsupported) {
		return {.unsupported = true};
	}
	Compiled compiled;
	if (table) {
		// The census's verdict, as the renderer reaches it (BindingLayout.h).
		const auto plan  = IR::ExtractResourcePlan(translated.program);
		compiled.census  = IR::TableEligibility(plan, &entry.specialization).refusal;
	}
	options.table_mode = table;
	auto result        = ShaderRecompiler::CompileProgram(std::move(translated), options,
	                                                      entry.specialization, entry.push_data_cursor);
	compiled.spirv     = std::move(result.spirv);
	compiled.refusal   = result.table_refusal;
	compiled.plan      = std::move(result.program.table_plan);
	compiled.block_dwords = compiled.plan.BlockDwords();
	return compiled;
}

uint64_t SpirvHash(const std::vector<uint32_t>& spirv) {
	return spirv.empty() ? uint64_t {0} : XXH64(spirv.data(), spirv.size() * sizeof(uint32_t), 0);
}

void Dump(const std::filesystem::path& folder, size_t n, const char* kind,
          const std::vector<uint32_t>& spirv) {
	if (folder.empty() || spirv.empty()) {
		return;
	}
	std::FILE* file = std::fopen((folder / fmt::format("{}_{}.spv", n, kind)).string().c_str(), "wb");
	if (file != nullptr) {
		std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), file);
		std::fclose(file);
	}
}

void Replay(size_t n, const std::filesystem::path& path, const Entry& entry,
            std::span<const uint8_t> payload, bool same_version, bool table,
            const std::filesystem::path& dump, std::FILE* out) {
	const auto normal = Compile(entry, false);
	Dump(dump, n, "normal", normal.spirv);
	const bool valid  = !normal.spirv.empty() && ValidSpirv(normal.spirv);
	// (From another source version the cached SPIR-V may differ legitimately.)
	Disk::PermutationRecord record;
	const bool cached = Disk::DecodePermutationRecord(payload, record) && record.spirv == normal.spirv;
	std::string line = fmt::format("{}\t{}\t{}\t{:016x}\t{}\t{:016x}\t{}\t{}{}", n,
	                               path.filename().string(), StageName(entry.stage), entry.hash,
	                               normal.spirv.size(), SpirvHash(normal.spirv), valid ? 1 : 0,
	                               cached ? 1 : 0, same_version ? "" : "*");
	if (table && !normal.unsupported) {
		const auto compiled = Compile(entry, true);
		Dump(dump, n, "table", compiled.spirv);
		const bool agree    = compiled.refusal == compiled.census ||
		                   (compiled.refusal == IR::TableRefusal::ShaderData &&
		                    compiled.census == IR::TableRefusal::Eligible);
		const bool table_valid = !compiled.spirv.empty() && ValidSpirv(compiled.spirv);
		const auto count = [](const std::vector<uint8_t>& set) {
			return std::ranges::count(set, uint8_t {1});
		};
		const auto& plan = compiled.plan;
		line += fmt::format("\t{}\t{}\t{}\t{}\t{:016x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
		                    IR::TableRefusalName(compiled.refusal), IR::TableRefusalName(compiled.census),
		                    agree ? 1 : 0, compiled.spirv.size(), SpirvHash(compiled.spirv),
		                    table_valid ? 1 : 0, plan.slots.size(), count(plan.gpu), count(plan.cpu),
		                    plan.buffers.size(), plan.images.size(), plan.samplers.size(),
		                    compiled.block_dwords, plan.global_memory ? 1 : 0);
	}
	std::fprintf(out, "%s\n", line.c_str());
	std::fflush(out);
}

} // namespace
} // namespace Libs::Graphics

int main(int argc, char* argv[]) {
	using namespace Libs::Graphics;
	if (argc < 4) {
		std::fprintf(stderr, "usage: shader_cache_replay_tool <cache folder> <log file> <out tsv> "
		                     "[--version <hex>] [--first <n>] [--end <n>] [--table]\n");
		return 2;
	}
	const std::filesystem::path folder = argv[1];
	std::optional<uint64_t>     version;
	size_t                      first = 0;
	size_t                      end   = SIZE_MAX;
	bool                        table = false;
	std::filesystem::path       dump;
	for (int i = 4; i < argc; i++) {
		const std::string arg = argv[i];
		if (arg == "--table") {
			table = true;
		} else if (arg == "--dump-dir" && i + 1 < argc) {
			dump = argv[++i];
		} else if (arg == "--version" && i + 1 < argc) {
			version = std::strtoull(argv[++i], nullptr, 16);
		} else if (arg == "--first" && i + 1 < argc) {
			first = std::strtoull(argv[++i], nullptr, 10);
		} else if (arg == "--end" && i + 1 < argc) {
			end = std::strtoull(argv[++i], nullptr, 10);
		} else {
			std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
			return 2;
		}
	}

	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction   = Config::LogDirection::File;
	config.printf_output_file = argv[2];
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();
#ifdef _WIN32
	AddVectoredExceptionHandler(1, ReportCrash);
#endif

	std::vector<std::filesystem::path> files;
	std::filesystem::file_time_type    newest {};
	std::filesystem::path              newest_file;
	for (const auto& item: std::filesystem::directory_iterator(folder)) {
		if (item.path().extension() == ".bin") {
			files.push_back(item.path());
			if (item.last_write_time() > newest) {
				newest      = item.last_write_time();
				newest_file = item.path();
			}
		}
	}
	std::ranges::sort(files);
	std::vector<uint8_t> key;
	std::vector<uint8_t> payload;
	if (!version) {
		Session session;
		uint32_t format = 0;
		if (!ShaderDiskCache::ReadEntryFile(newest_file, key, payload, &format)) {
			std::fprintf(stderr, "cannot read %s\n", newest_file.string().c_str());
			return 1;
		}
		ShaderDiskCache::ByteReader in(key);
		if (!ReadSession(in, key, session)) {
			return 1;
		}
		version = session.source_version;
	}
	std::FILE* out = std::fopen(argv[3], "a");
	if (out == nullptr) {
		return 1;
	}
	std::printf("VERSION\t%016" PRIx64 "\tTOOL\t%016" PRIx64 "\tFILES\t%zu\n", *version,
	            ShaderDiskCache::SourceVersion(), files.size());
	bool applied = false;
	for (size_t n = first; n < std::min(end, files.size()); n++) {
		uint32_t format = 0;
		if (!ShaderDiskCache::ReadEntryFile(files[n], key, payload, &format)) {
			continue;
		}
		ShaderDiskCache::ByteReader in(key);
		Session                     session;
		Entry                       entry;
		if (!ReadSession(in, key, session) || session.source_version != *version ||
		    !ReadPermutation(in, entry)) {
			continue;
		}
		if (!applied) {
			ApplySession(session);
			applied = true;
		}
		std::printf("START\t%zu\t%s\n", n, files[n].filename().string().c_str());
		std::fflush(stdout);
		Replay(n, files[n], entry, payload, session.source_version == ShaderDiskCache::SourceVersion(),
		       table, dump, out);
	}
	std::fclose(out);
	std::printf("DONE\t%zu\n", files.size());
	return 0;
}
