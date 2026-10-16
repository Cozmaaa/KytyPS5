#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPROFILER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPROFILER_H_

#include "common/profiler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Tracy GPU zones: timestamps written into the render scheduler's command buffers. They are
// recorded only while a Tracy client is connected, so the calls are cheap otherwise.
namespace GpuProfiler {

// Creates the Tracy GPU context. Does nothing when Tracy is disabled or after the first call.
void Initialize(GraphicContext& graphics);
// Call after the GPU finished every command buffer with zones.
void Shutdown();
// Sends finished timestamps to Tracy and records query resets, so call it outside rendering.
void Collect(vk::CommandBuffer buffer);

class Zone {
public:
	Zone() = default;
	~Zone() { End(); }
	Zone(const Zone&)            = delete;
	Zone& operator=(const Zone&) = delete;

	void Begin(vk::CommandBuffer buffer, const tracy::SourceLocationData* location);
	// Writes the closing timestamp into the zone's command buffer; it reports the opening thread,
	// so a zone can end on another thread.
	void End();

private:
	vk::CommandBuffer m_buffer;
	uint64_t          m_connection = 0;
	uint32_t          m_thread     = 0;
	bool              m_active     = false;
};

} // namespace GpuProfiler

} // namespace Libs::Graphics

// A GPU zone around the commands recorded into `buffer` until the end of the scope. Nothing may
// submit or replace `buffer` inside the scope.
#define KYTY_GPU_ZONE(buffer, name)                                                                \
	static constexpr tracy::SourceLocationData KYTY_PROFILER_CONCAT(                               \
	    kyty_gpu_location_, __LINE__) {name, TracyFunction, TracyFile,                             \
	                                   static_cast<uint32_t>(__LINE__), 0};                        \
	Libs::Graphics::GpuProfiler::Zone KYTY_PROFILER_CONCAT(kyty_gpu_zone_, __LINE__);              \
	KYTY_PROFILER_CONCAT(kyty_gpu_zone_, __LINE__)                                                 \
	    .Begin(buffer, &KYTY_PROFILER_CONCAT(kyty_gpu_location_, __LINE__))

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPROFILER_H_ */
