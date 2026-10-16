#include "graphics/host_gpu/renderer/gpuProfiler.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"

// Kyty loads Vulkan dynamically, so Tracy must load its functions too. Only this file includes
// TracyVulkan.hpp, which keeps the macro consistent.
#define TRACY_VK_USE_SYMBOL_TABLE
#include <cstring>
#include <new>
#include <tracy/TracyVulkan.hpp>

namespace Libs::Graphics::GpuProfiler {

namespace {

tracy::VkCtx* g_context = nullptr;

} // namespace

void Initialize(GraphicContext& graphics) {
	if (g_context != nullptr || !tracy::ProfilerAvailable()) {
		return;
	}
	// Tracy aligns GPU and CPU clocks by submitting a few commands of its own.
	vk::CommandPoolCreateInfo pool_info {};
	// Tracy records the buffer several times.
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                             vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	pool_info.queueFamilyIndex = graphics.queue_family;
	vk::CommandPool pool;
	EXIT_NOT_IMPLEMENTED(graphics.device.createCommandPool(&pool_info, nullptr, &pool) !=
	                     vk::Result::eSuccess);
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::CommandBuffer buffer;
	EXIT_NOT_IMPLEMENTED(graphics.device.allocateCommandBuffers(&allocate, &buffer) !=
	                     vk::Result::eSuccess);
	{
		Common::LockGuard lock(graphics.queue_mutex);
		auto* memory = static_cast<tracy::VkCtx*>(tracy::tracy_malloc(sizeof(tracy::VkCtx)));
		// Without calibration, the symbol-table constructor leaves the calibrated-timestamps
		// function pointer unset; zeroed memory keeps Tracy on the device time domain.
		std::memset(static_cast<void*>(memory), 0, sizeof(tracy::VkCtx));
		g_context = new (memory)
		    tracy::VkCtx(static_cast<VkInstance>(graphics.instance),
		                 static_cast<VkPhysicalDevice>(graphics.physical_device),
		                 static_cast<VkDevice>(graphics.device),
		                 static_cast<VkQueue>(graphics.queue), static_cast<VkCommandBuffer>(buffer),
		                 VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
		                 VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr, false);
	}
	graphics.device.destroyCommandPool(pool, nullptr);
	constexpr char name[] = "Kyty render queue";
	g_context->Name(name, static_cast<uint16_t>(sizeof(name) - 1));
}

void Shutdown() {
	if (g_context != nullptr) {
		tracy::DestroyVkContext(g_context);
		g_context = nullptr;
	}
}

void Collect(vk::CommandBuffer buffer) {
	if (g_context != nullptr) {
		g_context->Collect(static_cast<VkCommandBuffer>(buffer));
	}
}

void Zone::Begin(vk::CommandBuffer buffer, const tracy::SourceLocationData* location) {
	EXIT_IF(m_active);
	if (g_context == nullptr || !tracy::GetProfiler().IsConnected()) {
		return;
	}
	m_active     = true;
	m_buffer     = buffer;
	m_connection = tracy::GetProfiler().ConnectionId();
	m_thread     = tracy::GetThreadHandle();

	// Mirrors tracy::VkCtxScope, which cannot end on another thread.
	const auto query = g_context->NextQueryId();
	buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
	                      vk::QueryPool(g_context->GetQueryPool()), query);
	auto* item = tracy::Profiler::QueueSerial();
	tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginSerial);
	tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
	tracy::MemWrite(&item->gpuZoneBegin.srcloc, reinterpret_cast<uint64_t>(location));
	tracy::MemWrite(&item->gpuZoneBegin.thread, m_thread);
	tracy::MemWrite(&item->gpuZoneBegin.queryId, static_cast<uint16_t>(query));
	tracy::MemWrite(&item->gpuZoneBegin.context, static_cast<uint8_t>(g_context->GetId()));
	tracy::Profiler::QueueSerialFinish();
}

void Zone::End() {
	if (!m_active) {
		return;
	}
	m_active = false;
	if (g_context == nullptr) {
		return;
	}
	const auto query = g_context->NextQueryId();
	m_buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
	                        vk::QueryPool(g_context->GetQueryPool()), query);
	if (tracy::GetProfiler().ConnectionId() != m_connection) {
		return;
	}
	auto* item = tracy::Profiler::QueueSerial();
	tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
	tracy::MemWrite(&item->gpuZoneEnd.cpuTime, tracy::Profiler::GetTime());
	tracy::MemWrite(&item->gpuZoneEnd.thread, m_thread);
	tracy::MemWrite(&item->gpuZoneEnd.queryId, static_cast<uint16_t>(query));
	tracy::MemWrite(&item->gpuZoneEnd.context, static_cast<uint8_t>(g_context->GetId()));
	tracy::Profiler::QueueSerialFinish();
}

} // namespace Libs::Graphics::GpuProfiler
