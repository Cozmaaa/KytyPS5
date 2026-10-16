// Table mode in the shader recompiler (IR::TablePlan, CompileOptions::table_mode;
// table-mode-port-plan.md step 1).
//
// - TablePlanRoundTrip: a table program's compiled info survives the shader disk cache.
// - UfcPairs: with KYTY_UFC_SHADER_CACHE naming EA SPORTS UFC 5's shader disk cache folder
//   (_ShaderCache/PPSA03541; game data stays out of the repository, so the check is skipped
//   without it), vertex and pixel shaders of pairs the fights draw most, and shaders table mode
//   refuses, translated from their recorded permutations (ShaderCacheEntries.h): the normal
//   translation is untouched, a table translation of each pair is valid SPIR-V with the block
//   layout the plan describes and the pair's shader data in push constants, and the census agrees
//   with the translation.

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/shaderDiskCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "ShaderCacheEntries.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fmt/format.h>
#include <map>
#include <string>
#include <vector>

namespace Libs::Graphics {
namespace {

namespace IR = ShaderRecompiler::IR;
using namespace ShaderCacheEntries;

int g_failures = 0;

void Check(bool condition, const std::string& test, const std::string& what) {
	if (!condition) {
		std::printf("FAIL %s: %s\n", test.c_str(), what.c_str());
		g_failures++;
	}
}

void TablePlanRoundTrip() {
	using Kind = IR::TablePlan::Operand::Kind;
	IR::CompiledShaderInfo program;
	program.stage                       = ShaderType::Pixel;
	program.table_mode                  = true;
	program.bindings.table_block_dwords = 2;
	program.bindings.push_data_start_dword = 4;
	auto& plan = program.table_plan;
	plan.slots.push_back({.low = {Kind::UserData, 2}, .high = {Kind::UserData, 3}, .offset = {}, .immediate = 16});
	plan.slots.push_back({.low       = {Kind::Slot, 0},
	                      .high      = {Kind::Immediate, 0x12},
	                      .offset    = {Kind::Immediate, 8},
	                      .immediate = 4,
	                      .records   = {Kind::Slot, 0},
	                      .bounded   = true});
	plan.buffers.push_back({IR::TablePlan::Operand {Kind::ShaderBaseLow, 0x40},
	                        IR::TablePlan::Operand {Kind::ShaderBaseHigh, 0x40},
	                        IR::TablePlan::Operand {Kind::Immediate, 64},
	                        IR::TablePlan::Operand {Kind::Slot, 1}});
	auto& image = plan.images.emplace_back();
	image.count = 8;
	for (uint32_t i = 0; i < 8; i++) {
		image.words[i] = {Kind::Slot, i % 2};
	}
	auto& sampler = plan.samplers.emplace_back();
	sampler.count = 4;
	plan.gpu           = {0, 1};
	plan.cpu           = {1, 1};
	plan.global_memory = true;
	const std::vector<uint32_t> spirv {0x07230203u, 1u, 2u, 3u};
	std::vector<uint8_t>        payload;
	ShaderDiskCache::EncodePermutationRecord(spirv, program, payload);
	ShaderDiskCache::PermutationRecord record;
	const char* test = "TablePlanRoundTrip";
	Check(ShaderDiskCache::DecodePermutationRecord(payload, record), test, "decode");
	Check(record.program.table_mode && record.program.table_plan == plan &&
	          record.program.bindings == program.bindings,
	      test, "the table plan or the bindings changed through the cache");
	Check(ShaderDiskCache::PermutationRecordRoundTrips(payload), test, "round trip");
	Check(plan.BlockDwords() == 2u + IR::TablePlan::BufferDwords, test, "block dwords");
	std::printf("[host]  TablePlanRoundTrip %s\n", g_failures == 0 ? "ok" : "FAILED");
}

struct Loaded {
	Session session;
	Entry   entry;
};

// The decodable permutation entries of these shaders, at most `limit` each.
std::map<uint64_t, std::vector<Loaded>> LoadEntries(const std::filesystem::path& folder,
                                                    const std::vector<uint64_t>& hashes,
                                                    size_t                       limit) {
	std::map<uint64_t, std::vector<Loaded>> result;
	std::vector<uint8_t>                    key;
	std::vector<uint8_t>                    payload;
	std::vector<std::filesystem::path>      files;
	for (const auto& item: std::filesystem::directory_iterator(folder)) {
		if (item.path().extension() == ".bin") {
			files.push_back(item.path());
		}
	}
	std::ranges::sort(files);
	for (const auto& path: files) {
		uint32_t format = 0;
		if (!ShaderDiskCache::ReadEntryFile(path, key, payload, &format)) {
			continue;
		}
		ShaderDiskCache::ByteReader in(key);
		Loaded                      loaded;
		if (!ReadSession(in, key, loaded.session) || !ReadPermutation(in, loaded.entry) ||
		    std::ranges::find(hashes, loaded.entry.hash) == hashes.end()) {
			continue;
		}
		auto& entries = result[loaded.entry.hash];
		if (entries.size() < limit) {
			entries.push_back(std::move(loaded));
		}
	}
	return result;
}

struct Translation {
	ShaderRecompiler::CompileResult result;
	IR::TableRefusal                census = IR::TableRefusal::Unknown;
};

Translation Translate(const Entry& entry, bool table, uint32_t push_data_cursor) {
	const std::vector<uint32_t> user_data(entry.user_data_size, 0u);
	auto                        options    = Options(entry, user_data);
	auto                        translated = ShaderRecompiler::TranslateProgram(entry.code, options);
	Translation                 translation;
	if (translated.unsupported) {
		return translation;
	}
	const auto plan     = IR::ExtractResourcePlan(translated.program);
	translation.census  = IR::TableEligibility(plan, &entry.specialization).refusal;
	options.table_mode  = table;
	translation.result  = ShaderRecompiler::CompileProgram(std::move(translated), options,
	                                                      entry.specialization, push_data_cursor);
	return translation;
}

std::string Disassemble(const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          text;
	(void)tools.Disassemble(spirv, &text,
	                        static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                            static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER));
	return text;
}

// A table program's layout and plan agree with each other.
void CheckTableProgram(const std::string& test, const ShaderRecompiler::CompileResult& compiled) {
	using Kind           = IR::TablePlan::Operand::Kind;
	const auto& program  = compiled.program;
	const auto& plan     = program.table_plan;
	const auto& bindings = program.bindings;
	Check(compiled.table_refusal == IR::TableRefusal::Eligible && !compiled.spirv.empty(), test,
	      std::string("refused: ") + IR::TableRefusalName(compiled.table_refusal));
	if (compiled.spirv.empty()) {
		return;
	}
	Check(ValidSpirv(compiled.spirv), test, "table SPIR-V fails spirv-val");
	Check(program.table_mode && bindings.table_block_dwords == 2u &&
	          bindings.memory_offset_count == 0u && !bindings.write_tracking,
	      test, "table layout");
	Check(IR::FindBinding(bindings, IR::DescriptorBindingKind::Buffers) == nullptr &&
	          IR::FindBinding(bindings, IR::DescriptorBindingKind::FlattenedSrt) == nullptr &&
	          IR::FindBinding(bindings, IR::DescriptorBindingKind::ShaderData) == nullptr,
	      test, "a buffer, flattened SRT or shader data binding in table mode");
	Check(bindings.UsesPushData(), test, "the block's address is not in the push constants");
	Check(plan.slots.size() == program.srt_reads.size() && plan.gpu.size() == plan.slots.size() &&
	          plan.cpu.size() == plan.slots.size() &&
	          plan.buffers.size() == program.info.buffers.size() &&
	          plan.images.size() == program.info.images.size() &&
	          plan.samplers.size() == program.info.samplers.size(),
	      test, "plan sizes");
	const auto evaluated = [&](const IR::TablePlan::Operand& operand, size_t below) {
		return operand.kind != Kind::Slot ||
		       (operand.value < below && operand.value < plan.cpu.size() && plan.cpu[operand.value] != 0);
	};
	for (size_t slot = 0; slot < plan.slots.size(); slot++) {
		Check(plan.gpu[slot] == 0 || plan.cpu[slot] != 0, test, "a slot the shader reads is not evaluated");
		if (plan.cpu[slot] == 0) {
			continue;
		}
		const auto& read = plan.slots[slot];
		for (const auto* operand: {&read.low, &read.high, &read.offset, &read.records}) {
			Check(evaluated(*operand, slot), test, "a slot's address reads an unevaluated or later slot");
		}
	}
	for (const auto& words: plan.buffers) {
		for (const auto& word: words) {
			Check(evaluated(word, plan.slots.size()), test, "a V# word reads an unevaluated slot");
		}
	}
	for (const auto* descriptors: {&plan.images, &plan.samplers}) {
		for (const auto& descriptor: *descriptors) {
			for (uint32_t i = 0; i < descriptor.count; i++) {
				Check(evaluated(descriptor.words[i], plan.slots.size()), test,
				      "a T# or S# word reads an unevaluated slot");
			}
		}
	}
	const auto text = Disassemble(compiled.spirv);
	Check(text.find("PhysicalStorageBuffer64") != std::string::npos &&
	          text.find("%flattened_srt") == std::string::npos &&
	          text.find("%buffers") == std::string::npos,
	      test, "table SPIR-V still binds buffers or the flattened SRT");
}

void CheckNormalProgram(const std::string& test, const ShaderRecompiler::CompileResult& compiled) {
	Check(!compiled.spirv.empty() && ValidSpirv(compiled.spirv), test, "normal SPIR-V invalid");
	Check(!compiled.program.table_mode && compiled.program.bindings.table_block_dwords == 0u &&
	          compiled.table_refusal == IR::TableRefusal::Unknown,
	      test, "table mode leaked into a normal translation");
}

void UfcPairs() {
	const char* folder = std::getenv("KYTY_UFC_SHADER_CACHE");
	if (folder == nullptr || folder[0] == '\0') {
		std::printf("[skip]  UfcPairs: set KYTY_UFC_SHADER_CACHE to UFC 5's shader disk cache folder\n");
		return;
	}
	// The pairs the fight's draw census (KYTY_LOG_DRAWS) counted most draws of, and shaders the
	// census refuses: a vertex shader whose V# words come from no slot, compute shaders on GDS.
	struct Pair {
		uint64_t vs;
		uint64_t ps;
	};
	const std::vector<Pair> pairs {{0x7c71909103369b83ull, 0xfc9365dadec45410ull},
	                               {0xfd14480606e101baull, 0xf2e494be492a1a19ull},
	                               {0x726bc3a1f7476372ull, 0x69dadabfa1cfd7dbull},
	                               {0xbb54d9fbebf2c3efull, 0xa186a7c9b904e354ull},
	                               {0xfcb4a1163dd615bbull, 0x0e28dfb7b7c0b292ull}};
	const std::vector<std::pair<uint64_t, IR::TableRefusal>> refused {
	    {0x79c5ff11588e2f17ull, IR::TableRefusal::DescriptorWords},
	    {0x03537e794a0a905cull, IR::TableRefusal::Gds}};
	std::vector<uint64_t> hashes;
	for (const auto& pair: pairs) {
		hashes.push_back(pair.vs);
		hashes.push_back(pair.ps);
	}
	for (const auto& [hash, refusal]: refused) {
		hashes.push_back(hash);
	}
	const auto entries = LoadEntries(folder, hashes, 3);
	bool       applied = false;
	for (const auto& [hash, loaded]: entries) {
		if (!applied && !loaded.empty()) {
			ApplySession(loaded.front().session);
			applied = true;
		}
	}
	const auto failures_before = g_failures;
	size_t     combinations    = 0;
	size_t     refusals        = 0;
	for (const auto& pair: pairs) {
		const auto vs = entries.find(pair.vs);
		const auto ps = entries.find(pair.ps);
		const auto name = fmt::format("UfcPairs {:016x}/{:016x}", pair.vs, pair.ps);
		Check(vs != entries.end() && ps != entries.end(), name, "no recorded permutation of the pair");
		if (vs == entries.end() || ps == entries.end()) {
			continue;
		}
		for (const auto& vs_entry: vs->second) {
			for (const auto& ps_entry: ps->second) {
				const auto normal_vs = Translate(vs_entry.entry, false, vs_entry.entry.push_data_cursor);
				const auto normal_ps = Translate(ps_entry.entry, false, ps_entry.entry.push_data_cursor);
				CheckNormalProgram(name + " vs", normal_vs.result);
				CheckNormalProgram(name + " ps", normal_ps.result);
				// The pair in table mode: the pixel shader's shader data follow the vertex shader's
				// in the push constants, as the pipeline lays them out.
				const auto table_vs = Translate(vs_entry.entry, true, vs_entry.entry.push_data_cursor);
				CheckTableProgram(name + " vs", table_vs.result);
				Check(table_vs.census == IR::TableRefusal::Eligible, name + " vs", "census refuses");
				uint32_t cursor = vs_entry.entry.push_data_cursor;
				table_vs.result.program.bindings.AdvancePushData(cursor);
				const auto table_ps = Translate(ps_entry.entry, true, cursor);
				CheckTableProgram(name + " ps", table_ps.result);
				Check(table_ps.census == IR::TableRefusal::Eligible, name + " ps", "census refuses");
				// A normal translation after table ones is unchanged.
				const auto again = Translate(vs_entry.entry, false, vs_entry.entry.push_data_cursor);
				Check(again.result.spirv == normal_vs.result.spirv, name, "normal SPIR-V changed");
				combinations++;
			}
		}
	}
	for (const auto& [hash, refusal]: refused) {
		const auto found = entries.find(hash);
		const auto name  = fmt::format("UfcPairs refused {:016x}", hash);
		Check(found != entries.end(), name, "no recorded permutation");
		if (found == entries.end()) {
			continue;
		}
		for (const auto& loaded: found->second) {
			const auto table = Translate(loaded.entry, true, loaded.entry.push_data_cursor);
			Check(table.result.table_refusal == refusal && table.result.spirv.empty(), name,
			      std::string("refusal: ") + IR::TableRefusalName(table.result.table_refusal));
			Check(table.census == refusal, name, "census disagrees");
			CheckNormalProgram(name, Translate(loaded.entry, false, loaded.entry.push_data_cursor).result);
			refusals++;
		}
	}
	Check(combinations >= pairs.size() && refusals >= refused.size(), "UfcPairs",
	      "too few recorded permutations");
	std::printf("[cache] UfcPairs %s (%zu shaders, %zu pair permutations, %zu refusals)\n",
	            g_failures == failures_before ? "ok" : "FAILED", entries.size(), combinations, refusals);
}

} // namespace
} // namespace Libs::Graphics

int main() {
	using namespace Libs::Graphics;
	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction = Config::LogDirection::Silent;
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();

	TablePlanRoundTrip();
	UfcPairs();
	if (g_failures != 0) {
		std::printf("ShaderTableModeTests: %d failures\n", g_failures);
		return 1;
	}
	std::printf("ShaderTableModeTests: ok\n");
	return 0;
}
