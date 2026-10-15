#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORCLEARCHECK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORCLEARCHECK_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <vulkan/vulkan.hpp>

namespace Libs::Graphics {

class CommandRecorder;

struct GraphicContext;
class CommandScheduler;

// TextureCache::MaterializeColorClear asks, on every color-target bind, whether a fill stamped one
// fast-clear code over a slice of the target's color metadata. The metadata is GPU-written (UFC 5
// stamps it with a compute shader), so answering on the CPU submitted and drained the GPU: ~4 times
// a frame, 5-13 ms in fights. This records the check on the GPU instead; the clears it decides
// run under conditional rendering (VK_EXT_conditional_rendering). KYTY_GPU_CLEAR_CHECK=1 turns it
// on.
class ColorClearCheck {
public:
	static constexpr uint32_t MaxCodes = 8;

	ColorClearCheck(GraphicContext& graphics, CommandScheduler& scheduler);
	~ColorClearCheck();
	KYTY_CLASS_NO_COPY(ColorClearCheck);

	struct Result {
		vk::Buffer     predicates;
		vk::DeviceSize code_offset[MaxCodes] = {}; // conditional-rendering predicate per code
		vk::DeviceSize any_offset            = 0;
	};
	// Records the check of the slice of `dwords` dwords at `metadata`: predicate k is non-zero
	// when every byte is codes[k]; with `expand`, a matched slice is then rewritten to 0xff.
	// Binds a compute pipeline and push state; call it outside a render pass.
	[[nodiscard]] Result Record(const CommandRecorder& command, vk::DeviceAddress metadata,
	                            uint32_t dwords, const std::array<uint8_t, MaxCodes>& codes,
	                            uint32_t code_count, bool expand);

private:
	// Ring of predicate entries. An entry is rewritten only after 4096 later checks, each
	// ordered behind the conditional clears before it.
	static constexpr uint32_t Entries     = 4096;
	static constexpr uint32_t EntryDwords = 16; // codes at 0-7, any at 8

	GraphicContext&         m_graphics;
	Buffer                  m_predicates;
	vk::DescriptorSetLayout m_set_layout      = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
	uint32_t                m_next            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORCLEARCHECK_H_
