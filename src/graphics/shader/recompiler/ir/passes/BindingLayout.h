#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

void AllocateBindings(Program& program, uint32_t push_data_start_dword = 0,
                      bool lds_storage = false);

struct SharedMemoryResources {
	bool lds = false;
	bool gds = false;
};

SharedMemoryResources CollectMemoryResources(const Program& program, std::vector<uint32_t>& buffers);
bool UsesFlattenedSrt(const Program& program);

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind);

// Table mode census (KYTY_TABLE_CENSUS=1): whether a program could be compiled in the Demon's
// Souls fork's table mode (github.com/chenxiao07/KytyPS5, IR::EnterTableMode), checked on the
// resource plan with the same rules; nothing is compiled. table-mode-port-plan.md, step 0.
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
	Count
};
const char* TableRefusalName(TableRefusal refusal);
struct TableEligibilityResult {
	TableRefusal refusal = TableRefusal::Unknown;
	uint16_t     slots   = 0;
	bool         global  = false; // flat or global memory reads (table mode readies their BDA ranges)
};
TableEligibilityResult TableEligibility(const ResourcePlan& plan);
bool                   TableCensusEnabled();

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_ */
