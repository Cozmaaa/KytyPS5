#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;
struct SrtCode;
struct SrtNode;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
};

enum class RuntimeValueType { Any, Integer };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);
// Work done by walkers on this thread, for profiling.
struct SrtWalkCounters {
	uint64_t evaluations  = 0; // Instructions evaluated.
	uint64_t memory_reads = 0; // Guest words read.
	uint64_t nested_walks = 0; // EXEC-specific walks started by ReadFirstLane.
};
[[nodiscard]] const SrtWalkCounters& SrtCounters();

// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	// Tags the constructor that takes the active mask as an SrtCode operand.
	struct MaskOperand {};

	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots, SrtWalker* clean_evaluator,
	          uint32_t active_mask, MaskOperand);

	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	static float                            Float32(uint64_t bits);
	bool                                    EvaluateU32(uint32_t operand, uint32_t& result);
	bool                                    EvaluateOperand(uint32_t operand, uint64_t& result);
	bool                                    EvaluateNode(uint32_t index, uint64_t& result);
	bool                                    EvaluateOperation(uint32_t index, uint64_t& result);
	bool                                    EvaluateRawRead(const SrtNode& node, uint64_t& result);
	bool EvaluateArithmetic(const SrtNode& node, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtCode&                         m_code;
	SrtRuntime                       m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                       m_clean_evaluator = nullptr;
	uint32_t                         m_active_mask; // SrtCode operand, NoOperand without a mask.
	ResourcePlan::EvaluationContext& m_context;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
