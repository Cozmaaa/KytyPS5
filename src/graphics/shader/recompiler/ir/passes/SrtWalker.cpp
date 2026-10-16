#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

thread_local SrtWalkCounters g_counters;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type)
	    : m_program(program), m_type(type) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type == RuntimeValueType::Integer &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type == RuntimeValueType::Integer && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type == RuntimeValueType::Integer && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(
				    Validate(inst->Arg(0)) &&
				    ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};

} // namespace

// One instruction of a plan in SrtCode. Operands are SrtCode operand codes.
struct SrtNode {
	enum class Op : uint8_t {
		Invalid, // Evaluation fails.
		UserData,
		ShaderBase,
		Copy, // Invariant phis, bit casts, condition references and extracts of constructs.
		ReadFirstLane,
		ExtractU64,
		ExtractCarry,
		ReadConst,
		RawRead,
		Select,
		LogicalAnd,
		LogicalOr,
		Arithmetic, // The operation is `opcode`, with its operands in order.
	};

	// Flags
	static constexpr uint8_t RawReadBuffer     = 1u << 0u; // Buffer descriptor with a record range.
	static constexpr uint8_t RawReadVector     = 1u << 1u; // Guarded LoadBufferU32.
	static constexpr uint8_t SelectFollowsMask = 1u << 0u; // Takes the true value under its mask.
	static constexpr uint8_t FlushDenorms      = 1u << 0u; // FP comparison flushes input denorms.

	Op          op     = Op::Invalid;
	uint8_t     flags  = 0;
	ValueOpcode opcode = {};
	// ReadConst slot, extract component, user data index or raw read immediate offset.
	uint32_t                aux  = 0;
	std::array<uint32_t, 6> args = {UINT32_MAX, UINT32_MAX, UINT32_MAX,
	                                UINT32_MAX, UINT32_MAX, UINT32_MAX};
};
static_assert(sizeof(SrtNode) == 32);

// Compact form of a plan's values for the walks repeated on every draw. Instructions become
// nodes in one array with operands already resolved through identities and invariant phis, so a
// walk indexes nearby memory instead of chasing IR pointers.
struct SrtCode {
	// An operand is a node index, or a constant index with ConstantBit set.
	static constexpr uint32_t ConstantBit = 1u << 31u;
	static constexpr uint32_t NoOperand   = UINT32_MAX;

	struct Constant {
		uint64_t value = 0;
		bool     valid = false; // Immediates of other types do not evaluate.
	};

	std::vector<SrtNode>  nodes;
	std::vector<Constant> constants;
	std::vector<uint32_t> srt_reads;   // Operand of each SRT read.
	std::vector<uint32_t> descriptors; // Eight operands per descriptor source.
	std::vector<uint32_t> conditions;  // Operand of each control-flow condition, if any.
	// Equal immediates share one operand, so comparing operands compares values.
	std::map<std::pair<Type, uint64_t>, uint32_t> constant_index;
};

namespace {

class SrtCodeBuilder {
public:
	SrtCodeBuilder(const ResourcePlan& program, SrtCode& code): m_program(program), m_code(code) {}

	uint32_t Operand(Value value) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		return inst != nullptr ? Node(*inst) : Constant(value);
	}

private:
	using Op = SrtNode::Op;

	uint32_t Constant(Value value) {
		uint64_t bits   = 0;
		bool     valid  = true;
		bool     shared = true;
		switch (value.GetType()) {
			case Type::U1: bits = value.U1(); break;
			case Type::U8: bits = value.U8(); break;
			case Type::U16: bits = value.U16(); break;
			case Type::U32: bits = value.U32(); break;
			case Type::U64: bits = value.U64(); break;
			case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); break;
			case Type::F16:
				bits  = value.F16Bits();
				valid = false;
				break;
			case Type::ScalarReg:
				bits  = RegIndex(value.ScalarRegister());
				valid = false;
				break;
			case Type::VectorReg:
				bits  = RegIndex(value.VectorRegister());
				valid = false;
				break;
			case Type::Void: valid = false; break;
			default:
				// Value equality never matches immediates of other types.
				valid  = false;
				shared = false;
				break;
		}
		const auto key = std::pair {value.GetType(), bits};
		if (shared) {
			if (const auto it = m_code.constant_index.find(key);
			    it != m_code.constant_index.end()) {
				return it->second;
			}
		}
		const auto operand = SrtCode::ConstantBit | static_cast<uint32_t>(m_code.constants.size());
		m_code.constants.push_back({.value = bits, .valid = valid});
		if (shared) {
			m_code.constant_index.emplace(key, operand);
		}
		return operand;
	}

	uint32_t Node(const Inst& inst) {
		auto       next  = static_cast<uint32_t>(m_code.nodes.size());
		const auto index = inst.EvaluationIndex(next);
		if (index < m_code.nodes.size()) {
			return index;
		}
		EXIT_IF(index != m_code.nodes.size());
		// The index is assigned before the operands are lowered, so cycles through phis refer
		// back to it.
		m_code.nodes.emplace_back();
		const auto node     = Lower(inst);
		m_code.nodes[index] = node;
		return index;
	}

	// Mirrors the operands each operation evaluates. Malformed instructions become Invalid
	// nodes, since operands not taken by a lazy operation are lowered as well.
	SrtNode Lower(const Inst& inst) {
		SrtNode    node {.opcode = inst.GetOpcode()};
		const auto count = inst.NumArgs();
		const auto take  = [&](size_t arity, Op op) {
			if (count < arity) {
				return;
			}
			for (size_t index = 0; index < arity; ++index) {
				node.args[index] = Operand(inst.Arg(index));
			}
			node.op = op;
		};
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData:
				if (count != 0 && inst.Arg(0).GetType() == Type::ScalarReg) {
					const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
					if (reg >= m_program.user_data_base) {
						node.op  = Op::UserData;
						node.aux = reg - m_program.user_data_base;
					}
				}
				break;
			case ValueOpcode::GetShaderBase: node.op = Op::ShaderBase; break;
			case ValueOpcode::Phi:
				if (const auto value =
				        ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
				    !value.IsEmpty()) {
					node.op      = Op::Copy;
					node.args[0] = Operand(value);
				}
				break;
			case ValueOpcode::ReadFirstLane:
				if (count >= 2) {
					node.op      = Op::ReadFirstLane;
					node.args[0] = Operand(inst.Arg(0));
					// An empty EXEC value gives the nested walk no active mask.
					node.args[1] =
					    inst.Arg(1).Resolve().IsEmpty() ? SrtCode::NoOperand : Operand(inst.Arg(1));
				}
				break;
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::ConditionRef: take(1, Op::Copy); break;
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2: LowerExtract(inst, node); break;
			case ValueOpcode::ReadConst:
				if (count >= 2) {
					const auto slot = inst.Arg(1).Resolve();
					if (slot.IsImmediate() && slot.GetType() == Type::U32 &&
					    slot.U32() < m_program.srt_reads.size()) {
						node.op      = Op::ReadConst;
						node.aux     = slot.U32();
						node.args[0] = Operand(m_program.srt_reads[slot.U32()].value);
					}
				}
				break;
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer:
			case ValueOpcode::LoadBufferU32: LowerRawRead(inst, node); break;
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectF32:
				take(3, Op::Select);
				if (count == 3) {
					node.flags = SrtNode::SelectFollowsMask;
				}
				break;
			case ValueOpcode::LogicalAnd: take(2, Op::LogicalAnd); break;
			case ValueOpcode::LogicalOr: take(2, Op::LogicalOr); break;
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPRecipIFlag32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::LogicalNot: take(1, Op::Arithmetic); break;
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
				take(2, Op::Arithmetic);
				if (inst.Flags<FPCompareFlags>().flush_input_denorms) {
					node.flags = SrtNode::FlushDenorms;
				}
				break;
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::IMul64:
			case ValueOpcode::UMulHi:
			case ValueOpcode::UMin32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::LogicalXor: take(2, Op::Arithmetic); break;
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract: take(3, Op::Arithmetic); break;
			case ValueOpcode::BitFieldInsert: take(4, Op::Arithmetic); break;
			default: break;
		}
		return node;
	}

	void LowerExtract(const Inst& inst, SrtNode& node) {
		if (inst.NumArgs() < 2) {
			return;
		}
		const auto index = inst.Arg(1).Resolve();
		if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
			return;
		}
		const auto component = index.U32();
		if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
			node.op      = Op::ExtractU64;
			node.aux     = component;
			node.args[0] = Operand(inst.Arg(0));
			return;
		}
		const auto* source = inst.Arg(0).Resolve().TryInstruction();
		if (source == nullptr) {
			return;
		}
		if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2 &&
		    component < source->NumArgs()) {
			node.op      = Op::Copy;
			node.args[0] = Operand(source->Arg(component));
		} else if (source->GetOpcode() == ValueOpcode::IAddCarry32 && source->NumArgs() >= 2) {
			node.op      = Op::ExtractCarry;
			node.aux     = component;
			node.args[0] = Operand(source->Arg(0));
			node.args[1] = Operand(source->Arg(1));
		}
	}

	// Operands: guard, address low, address high, offset, then the record count and word 3 of
	// buffer descriptors.
	void LowerRawRead(const Inst& inst, SrtNode& node) {
		if (inst.NumArgs() < 2 || !IsRawRead(m_program, inst)) {
			return;
		}
		const bool  vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
		const bool  buffer = vector || inst.GetOpcode() == ValueOpcode::ReadConstBuffer;
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->NumArgs() < 2u || (buffer && handle->NumArgs() != 4u)) {
			return;
		}
		node.op    = Op::RawRead;
		node.flags = static_cast<uint8_t>((vector ? SrtNode::RawReadVector : 0u) |
		                                  (buffer ? SrtNode::RawReadBuffer : 0u));
		node.aux =
		    static_cast<uint32_t>(m_program.memory_info[inst.Flags<MemoryFlags>().index].offset);
		if (vector) {
			node.args[0] = Operand(inst.Arg(4));
		}
		node.args[1] = Operand(handle->Arg(0));
		node.args[2] = Operand(handle->Arg(1));
		node.args[3] = Operand(inst.Arg(1));
		if (buffer) {
			node.args[4] = Operand(handle->Arg(2));
			node.args[5] = Operand(handle->Arg(3));
		}
	}

	const ResourcePlan& m_program;
	SrtCode&            m_code;
};

SrtCode& CodeFor(const ResourcePlan& program) {
	if (program.srt_code == nullptr) {
		auto           code = std::make_shared<SrtCode>();
		SrtCodeBuilder builder(program, *code);
		code->srt_reads.reserve(program.srt_reads.size());
		for (const auto& read: program.srt_reads) {
			code->srt_reads.push_back(builder.Operand(read.value));
		}
		code->descriptors.reserve(program.descriptor_sources.size() * 8u);
		for (const auto& source: program.descriptor_sources) {
			for (const auto& dword: source.dwords) {
				code->descriptors.push_back(builder.Operand(dword));
			}
		}
		code->conditions.reserve(program.control_flow.size());
		for (const auto& block: program.control_flow) {
			code->conditions.push_back(
			    block.condition.IsEmpty() ? SrtCode::NoOperand : builder.Operand(block.condition));
		}
		program.srt_code = std::move(code);
	}
	return *program.srt_code;
}

uint32_t ActiveMaskOperand(const ResourcePlan& program, Value active_mask) {
	return active_mask.Resolve().IsEmpty()
	           ? SrtCode::NoOperand
	           : SrtCodeBuilder(program, CodeFor(program)).Operand(active_mask);
}

} // namespace

const SrtWalkCounters& SrtCounters() {
	return g_counters;
}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     Value active_mask)
    : SrtWalker(program, runtime, clean_flat_slots, clean_evaluator,
                ActiveMaskOperand(program, active_mask), MaskOperand {}) {}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
                     uint32_t active_mask, MaskOperand)
    : m_program(program), m_code(CodeFor(program)), m_runtime(runtime),
      m_clean_flat_slots(clean_flat_slots), m_clean_evaluator(clean_evaluator),
      m_active_mask(active_mask), m_context(AcquireContext(program)) {
	if (m_context.values.size() < m_code.nodes.size()) {
		m_context.values.resize(m_code.nodes.size());
	}
}

SrtWalker::~SrtWalker() {
	--m_program.evaluation_depth;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateOperand(SrtCodeBuilder(m_program, m_code).Operand(value), wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

bool SrtWalker::EvaluateU32(uint32_t operand, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateOperand(operand, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext(const ResourcePlan& program) {
	if (program.evaluation_depth == program.evaluation_contexts.size()) {
		program.evaluation_contexts.emplace_back();
	}
	auto& context = program.evaluation_contexts[program.evaluation_depth++];
	context.generation += 2;
	return context;
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateOperand(uint32_t operand, uint64_t& result) {
	// Immediates evaluate before the active mask is checked.
	if ((operand & SrtCode::ConstantBit) != 0u) {
		const auto index = operand & ~SrtCode::ConstantBit;
		if (index >= m_code.constants.size() || !m_code.constants[index].valid) {
			return false;
		}
		result = m_code.constants[index].value;
		return true;
	}
	if (operand == m_active_mask) {
		result = 1u;
		return true;
	}
	return EvaluateNode(operand, result);
}

bool SrtWalker::EvaluateNode(uint32_t index, uint64_t& result) {
	if (m_active_mask != SrtCode::NoOperand) {
		const auto& node = m_code.nodes[index];
		if (node.op == SrtNode::Op::Select && (node.flags & SrtNode::SelectFollowsMask) != 0u &&
		    node.args[0] == m_active_mask) {
			return EvaluateOperand(node.args[1], result);
		}
	}
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_code.nodes.size());
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// The low generation bit marks an instruction that is still being evaluated.
	if (m_context.values[index].generation == (m_context.generation | 1u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	++g_counters.evaluations;
	uint64_t   out       = 0;
	const bool evaluated = EvaluateOperation(index, out);
	auto&      memo      = m_context.values[index];
	if (!evaluated) {
		memo.generation = 0;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result          = out;
	return true;
}

bool SrtWalker::EvaluateOperation(uint32_t index, uint64_t& result) {
	using Op        = SrtNode::Op;
	const auto node = m_code.nodes[index];
	switch (node.op) {
		case Op::Invalid: return false;
		case Op::UserData:
			if (node.aux >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[node.aux];
			return true;
		case Op::ShaderBase: result = m_runtime.shader_base; return true;
		case Op::Copy: return EvaluateOperand(node.args[0], result);
		case Op::ReadFirstLane: {
			if (m_active_mask != SrtCode::NoOperand && node.args[1] == m_active_mask) {
				return EvaluateOperand(node.args[0], result);
			}
			++g_counters.nested_walks;
			const auto clean_runtime = CleanRuntime(m_runtime);
			SrtWalker  clean_active(m_program, clean_runtime, {}, nullptr, node.args[1],
			                        MaskOperand {});
			SrtWalker  active(m_program, m_runtime, m_clean_flat_slots, &clean_active, node.args[1],
			                  MaskOperand {});
			return active.EvaluateOperand(node.args[0], result);
		}
		case Op::ExtractU64: {
			uint64_t packed = 0;
			if (!EvaluateOperand(node.args[0], packed)) {
				return false;
			}
			result = static_cast<uint32_t>(packed >> (node.aux * 32u));
			return true;
		}
		case Op::ExtractCarry: {
			uint64_t lhs = 0;
			uint64_t rhs = 0;
			if (!EvaluateOperand(node.args[0], lhs) || !EvaluateOperand(node.args[1], rhs)) {
				return false;
			}
			const auto sum =
			    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
			result =
			    node.aux == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		case Op::ReadConst:
			if (node.aux < m_clean_flat_slots.size() && m_clean_flat_slots[node.aux] != 0u &&
			    m_clean_evaluator != nullptr) {
				return m_clean_evaluator->EvaluateOperand(node.args[0], result);
			}
			return EvaluateOperand(node.args[0], result);
		case Op::RawRead:
			if ((node.flags & SrtNode::RawReadVector) != 0u && m_clean_evaluator != nullptr &&
			    m_clean_evaluator->m_active_mask == m_active_mask) {
				return m_clean_evaluator->EvaluateOperand(index, result);
			}
			return EvaluateRawRead(node, result);
		case Op::Select: {
			auto&    predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			uint64_t condition = 0;
			if (!predicate.EvaluateOperand(node.args[0], condition)) {
				return false;
			}
			return EvaluateOperand(node.args[condition != 0u ? 1u : 2u], result);
		}
		case Op::LogicalAnd: {
			uint64_t   a    = 0;
			uint64_t   b    = 0;
			const bool left = EvaluateOperand(node.args[0], a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (!EvaluateOperand(node.args[1], b) || (b != 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case Op::LogicalOr: {
			uint64_t   a    = 0;
			uint64_t   b    = 0;
			const bool left = EvaluateOperand(node.args[0], a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!EvaluateOperand(node.args[1], b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case Op::Arithmetic: return EvaluateArithmetic(node, result);
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const SrtNode& node, uint64_t& result) {
	const bool vector = (node.flags & SrtNode::RawReadVector) != 0u;
	if (vector) {
		const auto guard = node.args[0];
		if ((guard & SrtCode::ConstantBit) != 0u || guard != m_active_mask) {
			uint64_t enabled = 0;
			if (!EvaluateOperand(guard, enabled)) return false;
			if (enabled == 0u) {
				result = 0u;
				return true;
			}
		}
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!EvaluateOperand(node.args[1], low) || !EvaluateOperand(node.args[2], high) ||
	    !EvaluateOperand(node.args[3], offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(node.aux));
	uint64_t   address   = 0;
	if ((node.flags & SrtNode::RawReadBuffer) != 0u) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (!EvaluateOperand(node.args[4], records) || !EvaluateOperand(node.args[5], word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset = (static_cast<uint64_t>(immediate) & ~uint64_t {3}) +
		                         (static_cast<uint32_t>(offset) & ~3u);
		const auto stride      = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			return false;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative =
		    (immediate & ~int64_t {3}) + static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	++g_counters.memory_reads;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	if (reader != nullptr) {
		if (!reader(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		if (vector) return false;
		std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	}
	result = word;
	return true;
}

bool SrtWalker::EvaluateArithmetic(const SrtNode& node, uint64_t& result) {
	// Operands evaluate in order and stop at the first failure.
	std::array<uint64_t, 4> operands {};
	for (size_t index = 0; index < operands.size() && node.args[index] != SrtCode::NoOperand;
	     ++index) {
		if (!EvaluateOperand(node.args[index], operands[index])) {
			return false;
		}
	}
	const auto [a, b, c, d] = operands;
	switch (node.opcode) {
		case ValueOpcode::CompositeConstructU64:
			result =
			    static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::IAdd32: result = static_cast<uint32_t>(a + b); return true;
		case ValueOpcode::IAdd64: result = a + b; return true;
		case ValueOpcode::ISub32: result = static_cast<uint32_t>(a - b); return true;
		case ValueOpcode::ISub64: result = a - b; return true;
		case ValueOpcode::IMul32: result = static_cast<uint32_t>(a * b); return true;
		case ValueOpcode::IMul64: result = a * b; return true;
		case ValueOpcode::UMulHi:
			result =
			    (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
			return true;
		case ValueOpcode::UMin32:
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::ConvertF32U32:
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case ValueOpcode::ConvertU32F32: {
			const auto value = Float32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
				return false;
			}
			result = static_cast<uint32_t>(value);
			return true;
		}
		case ValueOpcode::FPMul32:
			result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
			return true;
		case ValueOpcode::FPTrunc32:
			result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
			return true;
		case ValueOpcode::FPRecipIFlag32: {
			const auto exponent = (a >> 23u) & 0xffu;
			// Normal positive powers of two have exact normal reciprocals in every FP mode.
			if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
			result = (254u - exponent) << 23u;
			return true;
		}
		case ValueOpcode::FPIsNan32: result = std::isnan(Float32(a)); return true;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32: {
			const auto operand = [&](uint64_t bits) {
				if ((node.flags & SrtNode::FlushDenorms) != 0u &&
				    (bits & 0x7fffffffu) < 0x00800000u) {
					bits &= 0x80000000u;
				}
				return Float32(bits);
			};
			result = node.opcode == ValueOpcode::FPOrdLessThanEqual32 ? operand(a) <= operand(b)
			                                                          : operand(a) >= operand(b);
			return true;
		}
		case ValueOpcode::BitwiseAnd32: result = static_cast<uint32_t>(a & b); return true;
		case ValueOpcode::BitwiseAnd64: result = a & b; return true;
		case ValueOpcode::BitwiseOr32: result = static_cast<uint32_t>(a | b); return true;
		case ValueOpcode::BitwiseXor32: result = static_cast<uint32_t>(a ^ b); return true;
		case ValueOpcode::BitwiseNot32: result = ~static_cast<uint32_t>(a); return true;
		case ValueOpcode::ShiftLeftLogical32:
			result = static_cast<uint32_t>(a) << (b & 31u);
			return true;
		case ValueOpcode::ShiftLeftLogical64: result = a << (b & 63u); return true;
		case ValueOpcode::ShiftRightLogical32:
			result = static_cast<uint32_t>(a) >> (b & 31u);
			return true;
		case ValueOpcode::ShiftRightLogical64: result = a >> (b & 63u); return true;
		case ValueOpcode::ShiftRightArithmetic32:
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >>
			                               (b & 31u));
			return true;
		case ValueOpcode::ShiftRightArithmetic64:
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case ValueOpcode::BitFieldUExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result          = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case ValueOpcode::BitFieldSExtract: {
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
				bits |= ~mask;
			}
			result = bits;
			return true;
		}
		case ValueOpcode::BitFieldInsert: {
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result =
			    (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::IEqual32:
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::INotEqual32:
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::ULessThan32:
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::ULessThanEqual32:
			result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::UGreaterThan32:
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			return true;
		case ValueOpcode::SGreaterThanEqual32:
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		case ValueOpcode::LogicalXor: result = (a != 0u) != (b != 0u); return true;
		case ValueOpcode::LogicalNot: result = a == 0u; return true;
		default: return false;
	}
}

bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result                 = {};
	result.dword_count     = descriptor.dword_count;
	if (descriptor.dword_count > descriptor.dwords.size()) {
		return false;
	}
	const auto* operands = &m_code.descriptors[static_cast<size_t>(source) * 8u];
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!EvaluateU32(operands[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	if (!m_program.srt_plan_complete) return false;
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_program.srt_reads.size()) return false;
		const auto& read  = m_program.srt_reads[slot];
		const bool  clean = read.flat_offset < m_clean_flat_slots.size() &&
		                    m_clean_flat_slots[read.flat_offset] != 0u;
		if (clean &&
		    (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		return read.flat_offset < flat.size() &&
		       evaluator.EvaluateU32(m_code.srt_reads[slot], flat[read.flat_offset]);
	};
	auto& active = m_program.active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow) {
		for (const auto source: block.sources)
			active.at(source) = 0u;
	}
	auto& visited = m_program.visited_blocks;
	auto& pending = m_program.pending_blocks;
	visited.assign(m_program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) continue;
		visited[index]    = 1u;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources)
			active[source] = 1u;
		for (const auto slot: block.srt_reads) {
			if (!refresh(slot)) return false;
		}
		uint32_t   condition         = 0;
		auto&      predicate         = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
		const auto condition_operand = m_code.conditions[index];
		if (condition_operand != SrtCode::NoOperand &&
		    m_runtime.read_specialization_memory != nullptr &&
		    predicate.EvaluateU32(condition_operand, condition)) {
			pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
