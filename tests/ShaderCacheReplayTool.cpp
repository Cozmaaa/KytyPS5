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
//                                 [--first <n>] [--end <n>] [--table]
// The entries are the folder's .bin files in name order whose session has the given recompiler
// source version (default: the version of the newest file). The session's device switches and
// KYTY_* switch values are applied before the first translation. Prints "START <n> <file>" before
// each entry, so a driver can resume after a fatal error (exit code 321) or a crash at <n> + 1.
// One tab-separated line per permutation entry goes to <out tsv>, appended:
//   n, file, stage, hash, normal SPIR-V words, its XXH64 (0: no SPIR-V), valid (spirv-val),
//   the cached SPIR-V equal (1/0; with a * when the cache came from another source version),
//   then with --table the table mode columns (see TableColumns).

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

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

struct Session {
	uint32_t format         = 0;
	uint64_t source_version = 0;
	bool     coherent_load_acquire = false;
	uint32_t clock_shift    = 0;
	bool     min_lod        = false;
	// KYTY_* switches the recompiler read, with their values (nullopt: unset).
	std::vector<std::pair<std::string, std::optional<std::string>>> switches;
};

bool GetBytes(Disk::ByteReader& in, size_t count, std::string& text) {
	text.resize(count);
	for (auto& c: text) {
		uint8_t byte = 0;
		if (!in.Get(byte)) {
			return false;
		}
		c = static_cast<char>(byte);
	}
	return true;
}

// The session key (ShaderDiskCache::SessionKey), up to the source key's 'S'.
bool ReadSession(Disk::ByteReader& in, std::span<const uint8_t> key, Session& session) {
	uint32_t magic = 0, vendor = 0, device = 0;
	bool     bindless = false, float_atomics = false;
	if (!in.Get(magic) || !in.Get(session.format) || !in.Get(session.source_version) ||
	    !in.Get(vendor) || !in.Get(device) || !in.Get(bindless) || !in.Get(float_atomics) ||
	    !in.Get(session.coherent_load_acquire) || !in.Get(session.clock_shift) ||
	    !in.Get(session.min_lod)) {
		return false;
	}
	// The switch list: (length, name, present, [length, value]) while a name follows.
	size_t offset = 4 + 4 + 8 + 4 + 4 + 1 + 1 + 1 + 4 + 1;
	for (;;) {
		if (key.size() - offset < 8 + 5) {
			return false;
		}
		uint64_t length = 0;
		std::memcpy(&length, key.data() + offset, sizeof(length));
		if (length < 5 || length > 128 || key.size() - offset - 8 < length ||
		    std::memcmp(key.data() + offset + 8, "KYTY_", 5) != 0) {
			break;
		}
		std::string name;
		bool        present = false;
		if (!in.Get(length) || !GetBytes(in, length, name) || !in.Get(present)) {
			return false;
		}
		offset += 8 + length + 1;
		std::optional<std::string> value;
		if (present) {
			uint64_t    size = 0;
			std::string text;
			if (!in.Get(size) || size > 4096 || !GetBytes(in, size, text)) {
				return false;
			}
			offset += 8 + size;
			value = std::move(text);
		}
		session.switches.emplace_back(std::move(name), std::move(value));
	}
	return true;
}

struct Entry {
	ShaderType                     stage          = ShaderType::Unknown;
	uint64_t                       hash           = 0;
	uint32_t                       wave_size      = 64;
	uint32_t                       user_data_base = 0;
	uint32_t                       user_data_size = 0;
	bool                           bindless       = false;
	bool                           non_fatal      = false;
	std::vector<uint32_t>          code;
	std::vector<uint32_t>          back_code;
	ShaderVertexInputInfo          vertex;
	ShaderPixelInputInfo           pixel;
	ShaderComputeInputInfo         compute;
	IR::ResourceSpecialization     specialization;
	uint32_t                       push_data_cursor = 0;
};

template <typename T>
bool GetAll(Disk::ByteReader& in, T& first) {
	return in.Get(first);
}
template <typename T, typename... Rest>
bool GetAll(Disk::ByteReader& in, T& first, Rest&... rest) {
	return in.Get(first) && GetAll(in, rest...);
}

// The stage input, as ShaderDiskCache::AppendInputKey wrote it.
bool ReadInput(Disk::ByteReader& in, Entry& entry) {
	uint8_t tag = 0;
	if (!in.Get(tag)) {
		return false;
	}
	if (tag == 'V') {
		auto&    info  = entry.vertex;
		uint32_t stage = 0;
		int32_t  resources = 0, attrib = 0, buffer = 0;
		if (!GetAll(in, stage, resources, attrib, buffer, info.wave_size, info.scratch_size_dwords,
		            info.pa_cl_vs_out_cntl, info.clip_space.enabled)) {
			return false;
		}
		info.logical_stage    = static_cast<ShaderType>(stage);
		info.resources_num    = resources;
		info.fetch_attrib_reg = attrib;
		info.fetch_buffer_reg = buffer;
		if (info.clip_space.enabled) {
			for (int i = 0; i < 2; i++) {
				if (!GetAll(in, info.clip_space.scale[i], info.clip_space.offset[i],
				            info.clip_space.half_extent[i])) {
					return false;
				}
			}
		}
		auto& mesh = info.mesh;
		auto& tess = info.tess;
		if (!GetAll(in, mesh.threads_num[0], mesh.threads_num[1], mesh.threads_num[2],
		            mesh.lds_size_dwords, mesh.scratch_size_dwords, mesh.host_subgroup_size,
		            mesh.wave_size, mesh.input_primitive, mesh.primitives_per_group,
		            mesh.vertices_per_group, mesh.max_vertices, mesh.max_primitives,
		            mesh.provoking_vertex, mesh.passes, mesh.fast_launch, tess.input_control_points,
		            tess.output_control_points, tess.ls_stride, tess.hs_stride, tess.domain,
		            tess.partitioning, tess.output_topology, info.fetch_external,
		            info.fetch_embedded)) {
			return false;
		}
		const int count = std::clamp(info.resources_num, 0, ShaderVertexInputInfo::RES_MAX);
		for (int i = 0; i < count; i++) {
			int32_t registers = 0, attr = 0;
			if (!GetAll(in, info.resources[i].fields[3], registers, attr)) {
				return false;
			}
			info.resources_dst[i].registers_num = registers;
			info.resources_dst[i].attr_id       = attr;
		}
		return true;
	}
	if (tag == 'P') {
		auto& info = entry.pixel;
		if (!in.Get(info.input_num)) {
			return false;
		}
		const auto inputs = std::min<uint32_t>(info.input_num, std::size(info.interpolator_settings));
		for (uint32_t i = 0; i < inputs; i++) {
			if (!in.Get(info.interpolator_settings[i])) {
				return false;
			}
		}
		if (!GetAll(in, info.wave_size, info.ps_system_input_base, info.custom_interpolation_mask,
		            info.ps_perspective_center_vgpr, info.ps_perspective_centroid_vgpr,
		            info.target_uint_mask, info.target_sint_mask, info.scratch_size_dwords)) {
			return false;
		}
		for (auto& mode: info.target_output_mode) {
			if (!in.Get(mode)) {
				return false;
			}
		}
		for (auto& mapping: info.target_export_mapping) {
			if (!in.Get(mapping.packed)) {
				return false;
			}
		}
		uint8_t blend = 0;
		if (!GetAll(in, info.ps_pos_x, info.ps_pos_y, info.ps_pos_z, info.ps_pos_w,
		            info.ps_front_face, info.ps_ancillary, info.ps_no_perspective,
		            info.ps_pixel_kill_enable, info.ps_depth_export_enable,
		            info.ps_sample_mask_export_enable, info.ps_sample_shading,
		            info.dual_source_blending, info.ps_early_z, info.ps_execute_on_noop, blend)) {
			return false;
		}
		info.alpha_blend_source = static_cast<ShaderAlphaBlendSource>(blend);
		return true;
	}
	if (tag == 'C') {
		auto&   info = entry.compute;
		int32_t thread_ids = 0, workgroup = 0;
		if (!GetAll(in, info.threads_num[0], info.threads_num[1], info.threads_num[2],
		            info.lds_size_dwords, info.scratch_size_dwords, info.host_subgroup_size,
		            info.wave_size, info.float_mode, info.group_id[0], info.group_id[1],
		            info.group_id[2], info.dispatch_thread_dimensions, info.lds_storage,
		            info.dispatch_dimensions_indirect, thread_ids, workgroup, info.tg_size_en)) {
			return false;
		}
		info.thread_ids_num     = thread_ids;
		info.workgroup_register = workgroup;
		return true;
	}
	return false;
}

// A permutation entry's key past the session (PipelineCache's BuildSourceKey and
// BuildPermutationKey). False for a source entry or a key this tool cannot read.
bool ReadPermutation(Disk::ByteReader& in, Entry& entry) {
	uint8_t  tag   = 0;
	uint32_t stage = 0;
	if (!in.Get(tag) || tag != 'S' ||
	    !GetAll(in, stage, entry.hash, entry.wave_size, entry.user_data_base, entry.user_data_size,
	            entry.bindless, entry.non_fatal) ||
	    !in.GetWords(entry.code) || !in.GetWords(entry.back_code) || !ReadInput(in, entry)) {
		return false;
	}
	entry.stage = static_cast<ShaderType>(stage);
	if (in.AtEnd() || !in.Get(tag) || tag != 'P' ||
	    !Disk::ReadSpecialization(in, entry.specialization) || !in.Get(entry.push_data_cursor)) {
		return false;
	}
	return in.AtEnd();
}

void ApplySession(const Session& session) {
	ShaderRecompiler::Spirv::SetCoherentLoadAcquire(session.coherent_load_acquire);
	ShaderRecompiler::Spirv::SetDeviceClockShift(session.clock_shift);
	ShaderRecompiler::Spirv::SetHostImageFeatures({.min_lod = session.min_lod});
	for (const auto& [name, value]: session.switches) {
		if (value) {
#ifdef _WIN32
			_putenv_s(name.c_str(), value->c_str());
#else
			setenv(name.c_str(), value->c_str(), 1);
#endif
		}
	}
}

bool ValidSpirv(const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&,
	                            const char* message) { LOGF("spirv-val: %s\n", message); });
	return tools.Validate(spirv);
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vs";
		case ShaderType::Pixel: return "ps";
		case ShaderType::Compute: return "cs";
		case ShaderType::Mesh: return "ms";
		case ShaderType::Local: return "ls";
		case ShaderType::TessellationControl: return "hs";
		case ShaderType::TessellationEvaluation: return "ds";
		default: return "?";
	}
}

ShaderRecompiler::CompileOptions Options(const Entry& entry, std::span<const uint32_t> user_data) {
	ShaderRecompiler::CompileOptions options;
	options.stage           = entry.stage;
	options.shader_hash     = entry.hash;
	options.wave_size       = entry.wave_size;
	options.user_data_base  = entry.user_data_base;
	options.user_data       = user_data;
	options.back_code       = entry.back_code;
	options.non_fatal       = entry.non_fatal;
	options.bindless_images = entry.bindless;
	options.dump_ir         = false;
	options.dump_label      = "ShaderCacheReplay";
	switch (entry.stage) {
		case ShaderType::Pixel: options.input_info.pixel = &entry.pixel; break;
		case ShaderType::Compute: options.input_info.compute = &entry.compute; break;
		default: options.input_info.vertex = &entry.vertex; break;
	}
	return options;
}

struct Compiled {
	std::vector<uint32_t> spirv;
	bool                  unsupported = false;
};

Compiled Compile(const Entry& entry) {
	const std::vector<uint32_t> user_data(entry.user_data_size, 0u);
	const auto                  options    = Options(entry, user_data);
	auto                        translated = ShaderRecompiler::TranslateProgram(entry.code, options);
	if (translated.unsupported) {
		return {.unsupported = true};
	}
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, entry.specialization,
	                                               entry.push_data_cursor);
	return {.spirv = std::move(result.spirv)};
}

void Replay(size_t n, const std::filesystem::path& path, const Entry& entry,
            std::span<const uint8_t> payload, bool same_version, std::FILE* out) {
	const auto normal = Compile(entry);
	const auto hash   = normal.spirv.empty()
	                        ? uint64_t {0}
	                        : XXH64(normal.spirv.data(), normal.spirv.size() * sizeof(uint32_t), 0);
	const bool valid  = !normal.spirv.empty() && ValidSpirv(normal.spirv);
	// (From another source version the cached SPIR-V may differ legitimately.)
	Disk::PermutationRecord record;
	const bool cached = Disk::DecodePermutationRecord(payload, record) && record.spirv == normal.spirv;
	std::fprintf(out, "%zu\t%s\t%s\t%016" PRIx64 "\t%zu\t%016" PRIx64 "\t%d\t%d%s\n", n,
	             path.filename().string().c_str(), StageName(entry.stage), entry.hash,
	             normal.spirv.size(), hash, valid ? 1 : 0, cached ? 1 : 0, same_version ? "" : "*");
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
	for (int i = 4; i < argc; i++) {
		const std::string arg = argv[i];
		if (arg == "--version" && i + 1 < argc) {
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
		if (!ShaderDiskCache::ReadEntryFile(newest_file, key, payload)) {
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
		if (!ShaderDiskCache::ReadEntryFile(files[n], key, payload)) {
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
		       out);
	}
	std::fclose(out);
	std::printf("DONE\t%zu\n", files.size());
	return 0;
}
