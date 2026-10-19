#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

[[noreturn]] void BindingFail(const char* message) {
	EXIT("shader binding layout failed: %s", message);
	std::abort();
}

void CollectShaderData(const Program& program, BindingLayout& layout) {
	std::array<bool, NumScalarRegs> registers {};
	bool uses_dispatch_threads = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (!inst.HasUses()) {
				continue;
			}
			uses_dispatch_threads |= inst.GetOpcode() == ValueOpcode::GetDispatchThreadExtent;
			if (inst.GetOpcode() != ValueOpcode::GetUserData) {
				continue;
			}
			if (inst.Arg(0).GetType() != Type::ScalarReg) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			const auto index = RegIndex(inst.Arg(0).ScalarRegister());
			if (index >= NumScalarRegs) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			registers[index] = true;
		}
	}
	for (uint32_t index = 0; index < registers.size(); index++) {
		if (registers[index]) {
			layout.user_data_registers.push_back(index);
		}
	}
	layout.memory_offset_dword = static_cast<uint32_t>(layout.user_data_registers.size());
	if (uses_dispatch_threads) {
		layout.dispatch_thread_dword = layout.memory_offset_dword;
		layout.memory_offset_dword += 3u;
	}
}

void AddBinding(BindingLayout& layout, DescriptorBindingKind kind,
                std::vector<uint32_t> resources = {}) {
	layout.descriptors.push_back({kind, std::move(resources)});
}

} // namespace

SharedMemoryResources CollectMemoryResources(const Program& program, std::vector<uint32_t>& buffers) {
	std::array<bool, ShaderInfo::MaxBuffers> live_buffers {};
	SharedMemoryResources shared;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (BufferAccessOf(op) == BufferAccess::None &&
			    SharedAccessOf(op) == SharedAccess::None &&
			    AddressOpcodeInfoOf(op).access == AddressAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				BindingFail("typed shader contains invalid memory metadata");
			}
			const auto& memory = program.memory_info[index];
			if (memory.planning_only) {
				continue;
			}
			shared.lds |= memory.kind == ResourceKind::FlatLocal;
			if (SharedAccessOf(op) != SharedAccess::None) {
				if (memory.kind != ResourceKind::Lds && memory.kind != ResourceKind::Gds) {
					BindingFail("typed shader contains invalid shared-memory metadata");
				}
				shared.gds |= memory.kind == ResourceKind::Gds;
				shared.lds |= memory.kind == ResourceKind::Lds;
			} else if (memory.kind == ResourceKind::Buffer || memory.kind == ResourceKind::ScalarBuffer) {
				EXIT_IF(memory.resource >= program.info.buffers.size());
				live_buffers.at(memory.resource) = true;
				for (const auto child: program.info.buffers[memory.resource].indirect_resources) {
					live_buffers.at(child) = true;
				}
			}
		}
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (live_buffers.at(i)) {
			buffers.push_back(i);
		}
	}
	return shared;
}

bool UsesFlattenedSrt(const Program& program) {
	const auto uses_mapping = [](const auto& resource) {
		return resource.indirect_search_iterations != 0u;
	};
	return std::ranges::any_of(program.blocks, [](const Block* block) {
		return std::ranges::any_of(*block, [](const Inst& inst) {
			return inst.GetOpcode() == ValueOpcode::ReadConst;
		});
	}) || std::ranges::any_of(program.info.buffers, uses_mapping) ||
	       std::ranges::any_of(program.info.images, [&](const ImageResource& image) {
		       return uses_mapping(image) || image.bindless;
	       }) ||
	       std::ranges::any_of(program.info.samplers, &SamplerResource::bindless);
}

void AllocateBindings(Program& program, uint32_t push_data_start_dword, bool lds_storage) {
	if (!program.shader_info_complete || program.binding_layout_complete) {
		EXIT("shader binding layout failed: %s", !program.shader_info_complete
		                                             ? "shader info is not ready"
		                                             : "binding layout already allocated");
	}
	BindingLayout next;
	std::vector<uint32_t> buffers;
	const auto shared = CollectMemoryResources(program, buffers);
	CollectShaderData(program, next);
	// Table mode reads its buffers' ranges and its slots from its block (TablePlan): no memory
	// offsets or lengths in the shader data, no buffer or flattened SRT bindings.
	const bool table = program.table_mode;
	next.memory_offset_count       = table ? 0u : static_cast<uint32_t>(buffers.size());
	next.write_tracking            = !table && std::ranges::any_of(buffers, [&](uint32_t resource) {
		           return resource < program.info.buffers.size() &&
		                  program.info.buffers[resource].write_tracked;
	           });
	next.table_block_dwords    = table ? 2u : 0u;
	next.push_data_start_dword =
	    PushData::StartFor(push_data_start_dword, next.ShaderDataDwords());

	if (!buffers.empty() && !table) {
		// Draw binding accesses this first group directly when memory_offset_count is nonzero.
		AddBinding(next, DescriptorBindingKind::Buffers, std::move(buffers));
	}

	std::array<std::vector<uint32_t>, ImageBindingCount> image_groups;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			EXIT("shader binding layout failed: image %u has an invalid binding class", i);
		}
		const auto group = ImageBindingIndex(*kind);
		if (group >= image_groups.size()) {
			EXIT("shader binding layout failed: image %u has an unmapped binding class", i);
		}
		auto&      resources = image_groups[group];
		const auto dynamic   = program.info.images[i].mip_mode == ImageMipMode::Dynamic;
		const auto count     = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			EXIT("shader binding layout failed: image %u has invalid specialized mip count %u", i,
			     program.info.images[i].mip_count);
		}
		resources.insert(resources.end(), count, i);
	}
	for (uint32_t i = 0; i < image_groups.size(); i++) {
		if (!image_groups[i].empty()) {
			AddBinding(next, static_cast<DescriptorBindingKind>(FirstImageBinding + i),
			           std::move(image_groups[i]));
		}
	}

	if (!program.info.samplers.empty()) {
		std::vector<uint32_t> resources(program.info.samplers.size());
		for (uint32_t i = 0; i < resources.size(); i++) {
			resources[i] = i;
		}
		AddBinding(next, DescriptorBindingKind::Samplers, std::move(resources));
	}
	if (shared.gds) {
		AddBinding(next, DescriptorBindingKind::Gds);
	}
	if (shared.lds && lds_storage) {
		AddBinding(next, DescriptorBindingKind::SharedMemory);
	}
	if (program.info.uses_dma) {
		AddBinding(next, DescriptorBindingKind::BdaPagetable);
		AddBinding(next, DescriptorBindingKind::FaultBuffer);
	}
	if (UsesFlattenedSrt(program) && !table) {
		AddBinding(next, DescriptorBindingKind::FlattenedSrt);
	}

	if (next.ShaderDataDwords() != 0 && !next.UsesPushData()) {
		AddBinding(next, DescriptorBindingKind::ShaderData);
	}

	program.bindings                = std::move(next);
	program.binding_layout_complete = true;
}

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind) {
	for (const auto& binding: layout.descriptors) {
		if (binding.kind == kind) {
			return &binding;
		}
	}
	return nullptr;
}

namespace {

using TableKind = TablePlan::Operand::Kind;

// A value table mode evaluates at the program's entry: an immediate, user data or an SRT slot
// below `slots`, or a phi of one of them alone (the fork's TableOperand / TablePhiOperand).
std::optional<TablePlan::Operand> TableOperand(Value value, size_t slots);

std::optional<TablePlan::Operand> TablePhiOperand(const Inst& phi, size_t slots) {
	std::optional<TablePlan::Operand> common;
	std::vector<const Inst*>          pending {&phi}, seen;
	while (!pending.empty()) {
		const auto* current = pending.back();
		pending.pop_back();
		if (std::ranges::find(seen, current) != seen.end()) {
			continue;
		}
		seen.push_back(current);
		if (seen.size() > 256) {
			return std::nullopt;
		}
		for (size_t i = 0; i < current->NumArgs(); ++i) {
			const auto incoming = current->Arg(i).Resolve();
			if (const auto* source = incoming.TryInstruction();
			    source != nullptr && source->GetOpcode() == ValueOpcode::Phi) {
				pending.push_back(source);
				continue;
			}
			const auto operand = TableOperand(incoming, slots);
			if (!operand || (common && *common != *operand)) {
				return std::nullopt;
			}
			common = operand;
		}
	}
	return common;
}

std::optional<TablePlan::Operand> TableOperand(Value value, size_t slots) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::U32) {
			return std::nullopt;
		}
		return TablePlan::Operand {TableKind::Immediate, value.U32()};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return std::nullopt;
	}
	if (inst->GetOpcode() == ValueOpcode::Phi) {
		return TablePhiOperand(*inst, slots);
	}
	if (inst->GetOpcode() == ValueOpcode::GetUserData && inst->Arg(0).GetType() == Type::ScalarReg) {
		return TablePlan::Operand {TableKind::UserData,
		                           static_cast<uint32_t>(RegIndex(inst->Arg(0).ScalarRegister()))};
	}
	if (inst->GetOpcode() != ValueOpcode::ReadConst) {
		return std::nullopt;
	}
	const auto slot = inst->Arg(1).Resolve();
	if (!slot.IsImmediate() || slot.GetType() != Type::U32 || slot.U32() >= slots) {
		return std::nullopt;
	}
	return TablePlan::Operand {TableKind::Slot, slot.U32()};
}

// A value of the stage's code address and immediates alone: a V# of constants the code embeds.
bool OfShaderBaseOnly(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) {
		return false;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::GetShaderBase: return true;
		case ValueOpcode::Identity:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
			for (size_t i = 0; i < inst->NumArgs(); ++i) {
				if (!OfShaderBaseOnly(inst->Arg(i), depth + 1)) {
					return false;
				}
			}
			return true;
		default: return false;
	}
}

// An OfShaderBaseOnly value with the code at `base` (the SRT walker's arithmetic).
std::optional<uint64_t> EvaluateAtShaderBase(Value value, uint64_t base, uint32_t depth = 0) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: return uint64_t {value.U1()};
			case Type::U32: return uint64_t {value.U32()};
			case Type::U64: return value.U64();
			default: return std::nullopt;
		}
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) {
		return std::nullopt;
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::GetShaderBase) {
		return base;
	}
	if (op == ValueOpcode::CompositeExtractU64 || op == ValueOpcode::CompositeExtractU32x2) {
		const auto index = inst->Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
			return std::nullopt;
		}
		const auto component = index.U32();
		if (op == ValueOpcode::CompositeExtractU64) {
			const auto packed = EvaluateAtShaderBase(inst->Arg(0), base, depth + 1);
			if (!packed) {
				return std::nullopt;
			}
			return uint64_t {static_cast<uint32_t>(*packed >> (32u * component))};
		}
		const auto* source = inst->Arg(0).ResolveInstruction();
		if (source != nullptr && source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
			return EvaluateAtShaderBase(source->Arg(component), base, depth + 1);
		}
		if (source != nullptr && source->GetOpcode() == ValueOpcode::IAddCarry32) {
			const auto lhs = EvaluateAtShaderBase(source->Arg(0), base, depth + 1);
			const auto rhs = EvaluateAtShaderBase(source->Arg(1), base, depth + 1);
			if (!lhs || !rhs) {
				return std::nullopt;
			}
			const auto sum = uint64_t {static_cast<uint32_t>(*lhs)} + static_cast<uint32_t>(*rhs);
			return uint64_t {static_cast<uint32_t>(component == 0u ? sum : sum >> 32u)};
		}
		return std::nullopt;
	}
	std::array<uint64_t, 3> v {};
	if (inst->NumArgs() > v.size()) {
		return std::nullopt;
	}
	for (size_t i = 0; i < inst->NumArgs(); ++i) {
		const auto arg = EvaluateAtShaderBase(inst->Arg(i), base, depth + 1);
		if (!arg) {
			return std::nullopt;
		}
		v[i] = *arg;
	}
	const auto a = v[0];
	const auto b = v[1];
	const auto u32 = [](uint64_t x) { return uint64_t {static_cast<uint32_t>(x)}; };
	switch (op) {
		case ValueOpcode::Identity: return a;
		case ValueOpcode::CompositeConstructU64: return u32(a) | u32(b) << 32u;
		case ValueOpcode::IAdd32: return u32(a + b);
		case ValueOpcode::IAdd64: return a + b;
		case ValueOpcode::ISub32: return u32(a - b);
		case ValueOpcode::ISub64: return a - b;
		case ValueOpcode::BitwiseAnd32: return u32(a & b);
		case ValueOpcode::BitwiseAnd64: return a & b;
		case ValueOpcode::BitwiseOr32: return u32(a | b);
		case ValueOpcode::BitwiseXor32: return u32(a ^ b);
		case ValueOpcode::ShiftLeftLogical32: return u32(u32(a) << (b & 31u));
		case ValueOpcode::ShiftRightLogical32: return u32(a) >> (b & 31u);
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32: return a != 0u ? b : v[2];
		case ValueOpcode::IEqual32: return uint64_t {u32(a) == u32(b)};
		case ValueOpcode::INotEqual32: return uint64_t {u32(a) != u32(b)};
		case ValueOpcode::ULessThan32: return uint64_t {u32(a) < u32(b)};
		case ValueOpcode::LogicalOr: return uint64_t {a != 0u || b != 0u};
		case ValueOpcode::LogicalAnd: return uint64_t {a != 0u && b != 0u};
		default: return std::nullopt;
	}
}

// A descriptor's base address dwords (`low`, `high`) as the stage's code address plus an offset
// below 2^31: proven by evaluating them at code addresses whose low dword carries into the high
// one too (the fork's ShaderBaseOffset).
std::optional<uint32_t> ShaderBaseOffset(Value low, Value high) {
	if (!OfShaderBaseOnly(low) || !OfShaderBaseOnly(high)) {
		return std::nullopt;
	}
	std::optional<uint32_t> offset;
	for (const uint64_t base: {0x2'0000'0000ull, 0x2'ffff'ff00ull, 0x9'0271'be00ull,
	                           0x7fff'ffff'ff00ull, 0x3'8000'0000ull}) {
		const auto low_word  = EvaluateAtShaderBase(low, base);
		const auto high_word = EvaluateAtShaderBase(high, base);
		if (!low_word || !high_word) {
			return std::nullopt;
		}
		const auto address = uint64_t {static_cast<uint32_t>(*low_word)} |
		                     uint64_t {static_cast<uint32_t>(*high_word)} << 32u;
		if (address < base || address - base >= 0x8000'0000ull ||
		    (offset && *offset != address - base)) {
			return std::nullopt;
		}
		offset = static_cast<uint32_t>(address - base);
	}
	return offset;
}

} // namespace

const char* TableRefusalName(TableRefusal refusal) {
	switch (refusal) {
		case TableRefusal::Unknown: return "unknown";
		case TableRefusal::Eligible: return "eligible";
		case TableRefusal::IncompletePlan: return "incomplete plan";
		case TableRefusal::Gds: return "gds";
		case TableRefusal::IndirectBuffer: return "indirect buffer";
		case TableRefusal::SlotNotScalarLoad: return "slot not a scalar load";
		case TableRefusal::SlotAddress: return "slot address";
		case TableRefusal::BufferAliased: return "aliased/swizzled/add_tid buffer";
		case TableRefusal::DescriptorWords: return "descriptor words";
		case TableRefusal::IndirectDescriptor: return "indirect/bindless descriptor";
		case TableRefusal::DynamicMip: return "dynamic mip";
		case TableRefusal::WrittenImage: return "written/atomic image";
		case TableRefusal::ImageDescriptor: return "image descriptor size";
		case TableRefusal::Stage: return "stage";
		case TableRefusal::WriteTracked: return "write-tracked buffer";
		case TableRefusal::ShaderData: return "shader data not in push constants";
		default: return "?";
	}
}

bool TableCensusEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TABLE_CENSUS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// The checks of the fork's EnterTableMode (BindingLayout.cpp there) and, with `out`, the plan they
// leave. One function, so that the census and the translation never disagree.
TableEligibilityResult BuildTablePlan(const ResourcePlan& program,
                                      const ResourceSpecialization* specialization, TablePlan* out) {
	TableEligibilityResult result;
	const auto refuse = [&](TableRefusal refusal) {
		result.refusal = refusal;
		return result;
	};
	TablePlan plan;
	result.slots = static_cast<uint16_t>(std::min<size_t>(program.srt_reads.size(), UINT16_MAX));
	if (program.stage != ShaderType::Vertex && program.stage != ShaderType::Pixel &&
	    program.stage != ShaderType::Compute) {
		return refuse(TableRefusal::Stage);
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete) {
		return refuse(TableRefusal::IncompletePlan);
	}
	for (const auto& memory: program.memory_info) {
		switch (memory.kind) {
			case ResourceKind::Gds: return refuse(TableRefusal::Gds);
			case ResourceKind::IndirectBuffer: return refuse(TableRefusal::IndirectBuffer);
			case ResourceKind::Flat:
			case ResourceKind::FlatLocal:
			case ResourceKind::Global: result.global = true; break;
			default: break;
		}
	}
	plan.global_memory = result.global;
	// The flattened SRT's slots in order (a slot's address only depends on lower slots and user
	// data).
	const auto slots = program.srt_reads.size();
	for (size_t slot = 0; slot < slots; ++slot) {
		const auto* load = program.srt_reads[slot].value.Resolve().TryInstruction();
		if (load == nullptr || load->Flags<MemoryFlags>().index >= program.memory_info.size()) {
			return refuse(TableRefusal::SlotNotScalarLoad);
		}
		const auto& memory = program.memory_info[load->Flags<MemoryFlags>().index];
		// A read through a V# (s_buffer_load_dword).
		if (load->GetOpcode() == ValueOpcode::ReadConstBuffer && load->NumArgs() == 2) {
			const auto* handle = load->Arg(0).Resolve().TryInstruction();
			const auto  offset = load->Arg(1).Resolve();
			if (memory.kind != ResourceKind::ScalarBuffer || handle == nullptr ||
			    handle->GetOpcode() != ValueOpcode::GetBufferResource || handle->NumArgs() != 4 ||
			    static_cast<int32_t>(memory.offset) < 0 || !offset.IsImmediate() ||
			    offset.GetType() != Type::U32) {
				return refuse(TableRefusal::SlotNotScalarLoad);
			}
			const auto low     = TableOperand(handle->Arg(0), slot);
			const auto high    = TableOperand(handle->Arg(1), slot);
			const auto records = TableOperand(handle->Arg(2), slot);
			if (!low || !high || !records) {
				return refuse(TableRefusal::SlotAddress);
			}
			plan.slots.push_back({.low       = *low,
			                      .high      = *high,
			                      .offset    = {TableKind::Immediate, offset.U32()},
			                      .immediate = static_cast<int32_t>(memory.offset),
			                      .records   = *records,
			                      .bounded   = true});
			continue;
		}
		if (load->GetOpcode() != ValueOpcode::LoadAddressU32 || load->NumArgs() != 4) {
			return refuse(TableRefusal::SlotNotScalarLoad);
		}
		const auto  active = load->Arg(3).Resolve();
		const auto* handle = load->Arg(0).Resolve().TryInstruction();
		if (memory.kind != ResourceKind::ScalarAddress || memory.address_is_full ||
		    !active.IsImmediate() || !active.U1() || handle == nullptr ||
		    handle->GetOpcode() != ValueOpcode::GetAddressResource || handle->NumArgs() != 2) {
			return refuse(TableRefusal::SlotNotScalarLoad);
		}
		const auto low    = TableOperand(handle->Arg(0), slot);
		const auto high   = TableOperand(handle->Arg(1), slot);
		const auto offset = TableOperand(load->Arg(1), slot);
		if (!low || !high || !offset) {
			return refuse(TableRefusal::SlotAddress);
		}
		plan.slots.push_back({.low       = *low,
		                      .high      = *high,
		                      .offset    = *offset,
		                      .immediate = static_cast<int32_t>(memory.offset)});
	}
	const auto words_of = [&](uint32_t source_index, uint32_t count, auto& words) {
		if (source_index >= program.descriptor_sources.size()) {
			return TableRefusal::DescriptorWords;
		}
		const auto& source = program.descriptor_sources[source_index];
		if (source.indirect_descriptor || source.bindless_sampler) {
			return TableRefusal::IndirectDescriptor;
		}
		if (source.dword_count != count) {
			return TableRefusal::DescriptorWords;
		}
		// A V# of constants the code embeds (s_getpc_b64 plus an offset), in a compute or pixel
		// shader: its base address words are the code address plus that offset.
		std::optional<uint32_t> embedded;
		if (count == 4 &&
		    (program.stage == ShaderType::Compute || program.stage == ShaderType::Pixel) &&
		    (!TableOperand(source.dwords[0], slots) || !TableOperand(source.dwords[1], slots))) {
			embedded = ShaderBaseOffset(source.dwords[0], source.dwords[1]);
		}
		for (uint32_t i = 0; i < count; ++i) {
			if (embedded && i < 2) {
				words[i] = {i == 0 ? TableKind::ShaderBaseLow : TableKind::ShaderBaseHigh, *embedded};
				continue;
			}
			const auto word = TableOperand(source.dwords[i], slots);
			if (!word) {
				return TableRefusal::DescriptorWords;
			}
			words[i] = *word;
		}
		return TableRefusal::Eligible;
	};
	// A permutation's resources are the plan's with its specialization applied
	// (ApplyResourceSpecialization), which can add indirect buffers and images and make images or
	// samplers bindless.
	if (specialization != nullptr && specialization->buffers.size() > program.info.buffers.size()) {
		return refuse(TableRefusal::IndirectBuffer);
	}
	if (specialization != nullptr && specialization->images.size() > program.info.images.size()) {
		return refuse(TableRefusal::IndirectDescriptor);
	}
	for (size_t i = 0; i < program.info.buffers.size(); ++i) {
		const auto& buffer = program.info.buffers[i];
		const auto* spec   = specialization != nullptr && i < specialization->buffers.size()
		                         ? &specialization->buffers[i]
		                         : nullptr;
		const auto  stride = spec != nullptr ? spec->packed_stride : buffer.packed_stride;
		if ((spec != nullptr ? spec->indirect_root : buffer.indirect_root) !=
		    BufferResource::NoIndirectBuffer) {
			return refuse(TableRefusal::IndirectBuffer);
		}
		if (buffer.image_alias != BufferResource::NoImageAlias ||
		    (stride & ((1u << 14u) | (1u << 20u))) != 0) {
			return refuse(TableRefusal::BufferAliased);
		}
		// Written at addresses the host cannot bound: the write bitmap's bases are per binding.
		if (spec != nullptr ? spec->write_tracked : buffer.write_tracked) {
			return refuse(TableRefusal::WriteTracked);
		}
		if (const auto words = words_of(buffer.source, 4, plan.buffers.emplace_back());
		    words != TableRefusal::Eligible) {
			return refuse(words);
		}
	}
	for (size_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		const auto* spec  = specialization != nullptr && i < specialization->images.size()
		                        ? &specialization->images[i]
		                        : nullptr;
		if ((spec != nullptr ? spec->indirect_root : image.indirect_root) !=
		        ImageResource::NoIndirectImage ||
		    (spec != nullptr ? spec->bindless : image.bindless) ||
		    (spec != nullptr ? spec->indirect_search_iterations : image.indirect_search_iterations) !=
		        0) {
			return refuse(TableRefusal::IndirectDescriptor);
		}
		if (image.mip_mode != ImageMipMode::None) {
			return refuse(TableRefusal::DynamicMip);
		}
		if (image.atomic ||
		    ((image.written || image.resource_class == ImageResourceClass::Storage) &&
		     program.stage != ShaderType::Pixel && program.stage != ShaderType::Compute)) {
			return refuse(TableRefusal::WrittenImage);
		}
		if (image.source >= program.descriptor_sources.size()) {
			return refuse(TableRefusal::DescriptorWords);
		}
		auto& descriptor = plan.images.emplace_back();
		descriptor.count = program.descriptor_sources[image.source].dword_count;
		if (descriptor.count != 4 && descriptor.count != 8) {
			return refuse(TableRefusal::ImageDescriptor);
		}
		if (const auto words = words_of(image.source, descriptor.count, descriptor.words);
		    words != TableRefusal::Eligible) {
			return refuse(words);
		}
	}
	for (size_t i = 0; i < program.info.samplers.size(); ++i) {
		const auto& sampler  = program.info.samplers[i];
		const bool  bindless = specialization != nullptr && i < specialization->samplers.size()
		                           ? specialization->samplers[i].bindless
		                           : sampler.bindless;
		if (bindless) {
			return refuse(TableRefusal::IndirectDescriptor);
		}
		auto& descriptor = plan.samplers.emplace_back();
		descriptor.count = 4;
		if (const auto words = words_of(sampler.source, 4, descriptor.words);
		    words != TableRefusal::Eligible) {
			return refuse(words);
		}
	}
	result.refusal = TableRefusal::Eligible;
	if (out != nullptr) {
		*out = std::move(plan);
	}
	return result;
}

TableEligibilityResult TableEligibility(const ResourcePlan& plan,
                                        const ResourceSpecialization* specialization) {
	return BuildTablePlan(plan, specialization, nullptr);
}

TableRefusal EnterTableMode(Program& program) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete ||
	        program.binding_layout_complete);
	TablePlan  plan;
	const auto result = BuildTablePlan(program, nullptr, &plan);
	if (result.refusal != TableRefusal::Eligible) {
		return result.refusal;
	}
	program.table_plan = std::move(plan);
	program.table_mode = true;
	return TableRefusal::Eligible;
}

void MarkTableSlots(Program& program) {
	if (!program.table_mode) {
		return;
	}
	auto& plan = program.table_plan;
	plan.gpu.assign(plan.slots.size(), 0);
	plan.cpu.assign(plan.slots.size(), 0);
	const auto mark = [](std::vector<uint8_t>& set, const TablePlan::Operand& operand) {
		if (operand.kind == TableKind::Slot && operand.value < set.size()) {
			set[operand.value] = 1;
		}
	};
	const auto slot_of = [](Value value) -> std::optional<uint32_t> {
		const auto* inst = value.Resolve().TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::ReadConst) {
			return std::nullopt;
		}
		return inst->Arg(1).Resolve().U32();
	};
	const auto planning = [&](const Inst& inst) {
		const auto index = inst.Flags<MemoryFlags>().index;
		return (inst.GetOpcode() == ValueOpcode::LoadAddressU32 ||
		        inst.GetOpcode() == ValueOpcode::ReadConstBuffer) &&
		       index < program.memory_info.size() && program.memory_info[index].planning_only;
	};
	// The slots the shader reads (from its block): any read with a use other than the address of
	// planning-only loads (which only tell the host's plan where the slots are).
	for (const auto& info: program.block_info) {
		for (const auto value: {info.condition, info.indirect_target}) {
			if (const auto slot = value.IsEmpty() ? std::nullopt : slot_of(value);
			    slot && *slot < plan.gpu.size()) {
				plan.gpu[*slot] = 1;
			}
		}
	}
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			const auto slot = slot_of(Value(&inst));
			if (!slot || *slot >= plan.gpu.size()) {
				continue;
			}
			for (const auto& use: inst.Uses()) {
				const auto& user = *use.user;
				if (user.GetOpcode() != ValueOpcode::GetAddressResource ||
				    !std::ranges::all_of(user.Uses(), [&](const Use& address) {
					    return planning(*address.user);
				    })) {
					plan.gpu[*slot] = 1;
				}
			}
		}
	}
	// The host evaluates those, the descriptor words, and the slots their addresses read.
	for (const auto& words: plan.buffers) {
		for (const auto& word: words) {
			mark(plan.cpu, word);
		}
	}
	for (const auto* descriptors: {&plan.images, &plan.samplers}) {
		for (const auto& descriptor: *descriptors) {
			for (uint32_t i = 0; i < descriptor.count; ++i) {
				mark(plan.cpu, descriptor.words[i]);
			}
		}
	}
	for (size_t slot = 0; slot < plan.slots.size(); ++slot) {
		plan.cpu[slot] |= plan.gpu[slot];
	}
	for (size_t slot = plan.slots.size(); slot-- > 0;) {
		if (plan.cpu[slot] == 0) {
			continue;
		}
		const auto& read = plan.slots[slot];
		for (const auto* operand: {&read.low, &read.high, &read.offset, &read.records}) {
			mark(plan.cpu, *operand);
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
