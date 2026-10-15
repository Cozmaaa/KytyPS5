#include "graphics/host_gpu/renderer/colorClearCheck.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/color_clear_check_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

namespace Libs::Graphics {

namespace {

struct PushConstants {
	uint32_t metadata_low;
	uint32_t metadata_high;
	uint32_t dwords;
	uint32_t codes_low;
	uint32_t codes_high;
	uint32_t code_count;
	uint32_t output_dword;
	uint32_t expand;
};

} // namespace

ColorClearCheck::ColorClearCheck(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_predicates(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                   vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eConditionalRenderingEXT,
                   Entries * EntryDwords * sizeof(uint32_t)) {
	SetVulkanObjectNameF(graphics.device, m_predicates.Handle(), "Color Clear Check Predicates");

	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                               &m_set_layout),
	                     "create color-clear-check descriptor layout");

	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0,
                                            sizeof(PushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create color-clear-check pipeline layout");

	const auto                        module = CompileSPV(COLOR_CLEAR_CHECK_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create color-clear-check pipeline");
}

ColorClearCheck::~ColorClearCheck() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

ColorClearCheck::Result ColorClearCheck::Record(const CommandRecorder& command,
                                                vk::DeviceAddress metadata, uint32_t dwords,
                                                const std::array<uint8_t, MaxCodes>& codes,
                                                uint32_t code_count, bool expand) {
	EXIT_IF(code_count > MaxCodes || dwords == 0);
	const auto entry  = m_next;
	m_next            = (m_next + 1u) % Entries;
	const auto offset = vk::DeviceSize {entry} * EntryDwords * sizeof(uint32_t);

	// The metadata comes from earlier shader, transfer or attachment writes; the entry's previous
	// predicate reads must finish before it is rewritten.
	vk::MemoryBarrier ready {};
	ready.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite |
	                      vk::AccessFlagBits::eColorAttachmentWrite;
	ready.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &ready, 0, nullptr, 0,
	                        nullptr);

	uint32_t codes_low = 0, codes_high = 0;
	for (uint32_t k = 0; k < code_count; k++) {
		(k < 4 ? codes_low : codes_high) |= static_cast<uint32_t>(codes[k]) << (8u * (k & 3u));
	}
	const vk::DescriptorBufferInfo info {m_predicates.Handle(), 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	const PushConstants push {static_cast<uint32_t>(metadata),
	                          static_cast<uint32_t>(metadata >> 32u),
	                          dwords,
	                          codes_low,
	                          codes_high,
	                          code_count,
	                          entry * EntryDwords,
	                          expand ? 1u : 0u};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, 1, &write);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch(1, 1, 1);

	// Predicates to the conditional clears; the rewritten metadata to every later reader.
	vk::MemoryBarrier done {};
	done.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	done.dstAccessMask = vk::AccessFlagBits::eConditionalRenderingReadEXT |
	                     vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                     vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 1, &done, 0, nullptr, 0,
	                        nullptr);

	Result result {};
	result.predicates = m_predicates.Handle();
	for (uint32_t k = 0; k < MaxCodes; k++) {
		result.code_offset[k] = offset + k * sizeof(uint32_t);
	}
	result.any_offset = offset + 8u * sizeof(uint32_t);
	return result;
}

} // namespace Libs::Graphics
