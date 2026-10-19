#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTIMPORT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTIMPORT_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics {

struct GraphicContext;

// Experimental (KYTY_HOST_IMPORT=1): read-only shader buffers read the guest's memory directly.
//
// About 11 of the ~13 buffer bindings of a UFC 5 draw are bytes the game wrote on the CPU for that
// draw (constants); the buffer cache copies each into its GPU buffer first (tracker walk, staging
// copy, ~2 us a draw on the GPU thread). With VK_EXT_external_memory_host the guest backing store
// (host memory) is imported as Vulkan buffers in chunks, and such a binding points at the guest's
// own bytes instead: no sync, no upload. The GPU then reads them over PCIe at execution time, so
// so ranges are capped (KYTY_HOST_IMPORT_MAX_KB, default 1024), and only for bytes no GPU
// work wrote (FindGpuCleanBacking). Reading at execution time is what the console does; the
// emulator's early label writes may let the game rewrite the bytes first (see
// KYTY_LABELS_AT_COMPLETION).
class HostImport {
public:
	// The process-wide importer for this device, or null when off or unsupported. Never freed:
	// its buffers alias guest memory until the process ends.
	[[nodiscard]] static HostImport* Get(GraphicContext& graphics);
	// The largest binding served from imported memory.
	[[nodiscard]] static uint64_t MaxBytes();

	// A buffer over the host bytes [pointer, pointer + size) and the offset of `pointer` in it.
	[[nodiscard]] bool Find(const uint8_t* pointer, uint64_t size, vk::Buffer& buffer,
	                        uint64_t& offset);

private:
	explicit HostImport(GraphicContext& graphics);

	struct Chunk {
		vk::Buffer       buffer = nullptr;
		vk::DeviceMemory memory = nullptr;
	};
	[[nodiscard]] bool Import(uint64_t base, uint64_t size, Chunk& out);

	GraphicContext&                         m_graphics;
	uint64_t                                m_alignment = 4096;
	// Per chunk size: chunk base (host address) -> imported chunk, and bases that failed.
	std::unordered_map<uint64_t, Chunk>     m_large;
	std::unordered_map<uint64_t, Chunk>     m_small;
	std::unordered_set<uint64_t>            m_failed_large;
	std::unordered_set<uint64_t>            m_failed_small;
	uint64_t                                m_hits     = 0;
	uint64_t                                m_imports  = 0;
	uint64_t                                m_failures = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOSTIMPORT_H_
