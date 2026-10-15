#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

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
	next.memory_offset_count       = static_cast<uint32_t>(buffers.size());
	next.write_tracking            = std::ranges::any_of(buffers, [&](uint32_t resource) {
		           return resource < program.info.buffers.size() &&
		                  program.info.buffers[resource].write_tracked;
	           });
	next.push_data_start_dword =
	    PushData::StartFor(push_data_start_dword, next.ShaderDataDwords());

	if (!buffers.empty()) {
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
	if (UsesFlattenedSrt(program)) {
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

// A value table mode evaluates at the program's entry: an immediate, user data or an SRT slot
// below `slots`, or a phi of one of them alone (the fork's TableOperand / TablePhiOperand).
// Returns an encoding of the operand (kind << 32 | value) for comparing phi inputs.
std::optional<uint64_t> TableOperand(Value value, size_t slots);

std::optional<uint64_t> TablePhiOperand(const Inst& phi, size_t slots) {
	std::optional<uint64_t>  common;
	std::vector<const Inst*> pending {&phi}, seen;
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

std::optional<uint64_t> TableOperand(Value value, size_t slots) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::U32) {
			return std::nullopt;
		}
		return uint64_t {value.U32()};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return std::nullopt;
	}
	if (inst->GetOpcode() == ValueOpcode::Phi) {
		return TablePhiOperand(*inst, slots);
	}
	if (inst->GetOpcode() == ValueOpcode::GetUserData && inst->Arg(0).GetType() == Type::ScalarReg) {
		return uint64_t {1} << 32u | RegIndex(inst->Arg(0).ScalarRegister());
	}
	if (inst->GetOpcode() != ValueOpcode::ReadConst) {
		return std::nullopt;
	}
	const auto slot = inst->Arg(1).Resolve();
	if (!slot.IsImmediate() || slot.GetType() != Type::U32 || slot.U32() >= slots) {
		return std::nullopt;
	}
	return uint64_t {2} << 32u | slot.U32();
}

// A value of the stage's code address and immediates alone: a V# of constants the code embeds
// (the fork proves the offset by evaluation; the census only checks the shape).
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

// The checks of the fork's EnterTableMode (BindingLayout.cpp there), without building the plan.
TableEligibilityResult TableEligibility(const ResourcePlan& program) {
	TableEligibilityResult result;
	const auto refuse = [&](TableRefusal refusal) {
		result.refusal = refusal;
		return result;
	};
	result.slots = static_cast<uint16_t>(std::min<size_t>(program.srt_reads.size(), UINT16_MAX));
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
			if (!TableOperand(handle->Arg(0), slot) || !TableOperand(handle->Arg(1), slot) ||
			    !TableOperand(handle->Arg(2), slot)) {
				return refuse(TableRefusal::SlotAddress);
			}
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
		if (!TableOperand(handle->Arg(0), slot) || !TableOperand(handle->Arg(1), slot) ||
		    !TableOperand(load->Arg(1), slot)) {
			return refuse(TableRefusal::SlotAddress);
		}
	}
	const auto words_ok = [&](uint32_t source_index, uint32_t count) {
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
		const bool embedded =
		    count == 4 &&
		    (program.stage == ShaderType::Compute || program.stage == ShaderType::Pixel) &&
		    (!TableOperand(source.dwords[0], slots) || !TableOperand(source.dwords[1], slots)) &&
		    OfShaderBaseOnly(source.dwords[0]) && OfShaderBaseOnly(source.dwords[1]);
		for (uint32_t i = embedded ? 2u : 0u; i < count; ++i) {
			if (!TableOperand(source.dwords[i], slots)) {
				return TableRefusal::DescriptorWords;
			}
		}
		return TableRefusal::Eligible;
	};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.indirect_root != BufferResource::NoIndirectBuffer) {
			return refuse(TableRefusal::IndirectBuffer);
		}
		if (buffer.image_alias != BufferResource::NoImageAlias ||
		    (buffer.packed_stride & ((1u << 14u) | (1u << 20u))) != 0) {
			return refuse(TableRefusal::BufferAliased);
		}
		if (const auto words = words_ok(buffer.source, 4); words != TableRefusal::Eligible) {
			return refuse(words);
		}
	}
	for (const auto& image: program.info.images) {
		if (image.indirect_root != ImageResource::NoIndirectImage || image.bindless ||
		    image.indirect_search_iterations != 0) {
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
		const auto count = program.descriptor_sources[image.source].dword_count;
		if (count != 4 && count != 8) {
			return refuse(TableRefusal::ImageDescriptor);
		}
		if (const auto words = words_ok(image.source, count); words != TableRefusal::Eligible) {
			return refuse(words);
		}
	}
	for (const auto& sampler: program.info.samplers) {
		if (sampler.bindless) {
			return refuse(TableRefusal::IndirectDescriptor);
		}
		if (const auto words = words_ok(sampler.source, 4); words != TableRefusal::Eligible) {
			return refuse(words);
		}
	}
	result.refusal = TableRefusal::Eligible;
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
