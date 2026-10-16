#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ResourceSpecialization;

void AllocateBindings(Program& program, uint32_t push_data_start_dword = 0,
                      bool lds_storage = false);

struct SharedMemoryResources {
	bool lds = false;
	bool gds = false;
};

SharedMemoryResources CollectMemoryResources(const Program& program, std::vector<uint32_t>& buffers);
bool UsesFlattenedSrt(const Program& program);

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind);

// Table mode (TablePlan; the Demon's Souls fork's, github.com/chenxiao07/KytyPS5): whether a
// program can be compiled in table mode, and why not. The census (KYTY_TABLE_CENSUS=1,
// table-mode-port-plan.md step 0) checks registered permutations with the same function the
// translation builds its plan with (BuildTablePlan), so the two never disagree.
enum class TableRefusal : uint8_t {
	Unknown = 0, // not checked
	Eligible,
	IncompletePlan,     // no complete SRT plan or resource tracking
	Gds,                // uses GDS
	IndirectBuffer,     // a buffer chosen at run time (indirect buffer table)
	SlotNotScalarLoad,  // an SRT slot that is no scalar address or buffer load
	SlotAddress,        // an SRT slot whose address is not user data, a lower slot or an immediate
	BufferAliased,      // an image-aliased, swizzled or ADD_TID buffer
	DescriptorWords,    // a descriptor word that is not user data, a slot or an immediate
	IndirectDescriptor, // an indirect or bindless image or sampler
	DynamicMip,         // an image of dynamic mip levels
	WrittenImage,       // a vertex shader that writes an image, or an atomic image
	ImageDescriptor,    // an image descriptor of neither 4 nor 8 dwords
	Stage,              // not a vertex, pixel or compute program
	WriteTracked,       // a buffer written at addresses the host cannot bound (write bitmap)
	ShaderData,         // the shader data do not fit the push constants (CompileProgram only)
	Count
};
const char* TableRefusalName(TableRefusal refusal);
struct TableEligibilityResult {
	TableRefusal refusal = TableRefusal::Unknown;
	uint16_t     slots   = 0;
	bool         global  = false; // flat or global memory reads (table mode readies their BDA ranges)
};
// The checks and, with `out`, the plan (slot and descriptor operands; MarkTableSlots marks the
// slot sets later). A plan of a source with a permutation's `specialization`, or a program with
// its specialization applied (null), decide alike.
TableEligibilityResult BuildTablePlan(const ResourcePlan& program,
                                      const ResourceSpecialization* specialization, TablePlan* out);
TableEligibilityResult TableEligibility(const ResourcePlan&           plan,
                                        const ResourceSpecialization* specialization = nullptr);
bool                   TableCensusEnabled();

// After ApplyResourceSpecialization, while the plan's values are still there: the program's plan
// (Program::table_plan) and table mode, or why not (the program is unchanged then).
TableRefusal EnterTableMode(Program& program);
// After dead code elimination: the slots the shader reads (TablePlan::gpu) and those the host
// evaluates (TablePlan::cpu).
void MarkTableSlots(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_ */
