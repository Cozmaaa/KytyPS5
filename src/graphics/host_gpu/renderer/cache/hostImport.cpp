#include "graphics/host_gpu/renderer/cache/hostImport.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <cinttypes>
#include <cstdlib>

namespace Libs::Graphics {

namespace {

constexpr uint64_t LargeChunk = uint64_t {2} << 20u;
constexpr uint64_t SmallChunk = uint64_t {64} << 10u;

} // namespace

HostImport* HostImport::Get(GraphicContext& graphics) {
	static HostImport* const importer = [&]() -> HostImport* {
		const char* value = std::getenv("KYTY_HOST_IMPORT");
		if (value == nullptr || value[0] != '1') {
			return nullptr;
		}
		if (!graphics.external_memory_host_enabled) {
			LOGF("HostImport: off (no VK_EXT_external_memory_host)\n");
			return nullptr;
		}
		return new HostImport(graphics);
	}();
	return importer;
}

uint64_t HostImport::MaxBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_HOST_IMPORT_MAX_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {1024}) * 1024u;
	}();
	return bytes;
}

HostImport::HostImport(GraphicContext& graphics): m_graphics(graphics) {
	vk::PhysicalDeviceExternalMemoryHostPropertiesEXT host {};
	vk::PhysicalDeviceProperties2                     properties {};
	properties.pNext = &host;
	graphics.physical_device.getProperties2(&properties);
	m_alignment = std::max<uint64_t>(host.minImportedHostPointerAlignment, 4096u);
	LOGF("HostImport: on, alignment %" PRIu64 ", bindings up to %" PRIu64 " KiB\n", m_alignment,
	     MaxBytes() >> 10u);
}

bool HostImport::Import(uint64_t base, uint64_t size, Chunk& out) {
	const auto handle_type = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
	auto*      pointer     = reinterpret_cast<void*>(base);
	vk::MemoryHostPointerPropertiesEXT pointer_properties {};
	if (m_graphics.device.getMemoryHostPointerPropertiesEXT(handle_type, pointer,
	                                                        &pointer_properties) !=
	        vk::Result::eSuccess ||
	    pointer_properties.memoryTypeBits == 0) {
		return false;
	}
	vk::ExternalMemoryBufferCreateInfo external {};
	external.handleTypes = handle_type;
	vk::BufferCreateInfo buffer_info {};
	buffer_info.pNext       = &external;
	buffer_info.size        = size;
	buffer_info.usage       = vk::BufferUsageFlagBits::eStorageBuffer |
	                    vk::BufferUsageFlagBits::eUniformBuffer |
	                    vk::BufferUsageFlagBits::eTransferSrc;
	buffer_info.sharingMode = vk::SharingMode::eExclusive;
	vk::Buffer buffer;
	if (m_graphics.device.createBuffer(&buffer_info, nullptr, &buffer) != vk::Result::eSuccess) {
		return false;
	}
	vk::MemoryRequirements requirements {};
	m_graphics.device.getBufferMemoryRequirements(buffer, &requirements);
	const auto types = requirements.memoryTypeBits & pointer_properties.memoryTypeBits;
	uint32_t   type  = UINT32_MAX;
	const auto& memory_properties = m_graphics.physical_device_memory_properties;
	for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
		if ((types >> i) & 1u) {
			type = i;
			if (memory_properties.memoryTypes[i].propertyFlags &
			    vk::MemoryPropertyFlagBits::eHostCoherent) {
				break;
			}
		}
	}
	if (type == UINT32_MAX) {
		m_graphics.device.destroyBuffer(buffer, nullptr);
		return false;
	}
	vk::ImportMemoryHostPointerInfoEXT import_info {};
	import_info.handleType   = handle_type;
	import_info.pHostPointer = pointer;
	vk::MemoryAllocateInfo allocate {};
	allocate.pNext           = &import_info;
	allocate.allocationSize  = size;
	allocate.memoryTypeIndex = type;
	vk::DeviceMemory memory;
	if (m_graphics.device.allocateMemory(&allocate, nullptr, &memory) != vk::Result::eSuccess) {
		m_graphics.device.destroyBuffer(buffer, nullptr);
		return false;
	}
	m_graphics.device.bindBufferMemory(buffer, memory, 0);
	out = {buffer, memory};
	return true;
}

bool HostImport::Find(const uint8_t* pointer, uint64_t size, vk::Buffer& buffer,
                      uint64_t& offset) {
	const auto address = reinterpret_cast<uint64_t>(pointer);
	// A 2 MiB chunk, the 4 MiB span from the same base for a range that crosses its end, or a
	// 64 KiB chunk when the larger imports fail (the base address encodes the span in its low bits).
	struct Span {
		uint64_t alignment;
		uint64_t size;
	};
	for (const auto span: {Span {LargeChunk, LargeChunk}, Span {LargeChunk, 2u * LargeChunk},
	                       Span {SmallChunk, SmallChunk}}) {
		const auto base = address & ~(span.alignment - 1u);
		if (address + size > base + span.size || span.size % m_alignment != 0) {
			continue;
		}
		const auto key    = base | (span.size == LargeChunk ? 0u : span.size == SmallChunk ? 1u : 2u);
		const auto chunk_size = span.size;
		auto& chunks = span.alignment == LargeChunk ? m_large : m_small;
		auto& failed = span.alignment == LargeChunk ? m_failed_large : m_failed_small;
		auto  it     = chunks.find(key);
		if (it == chunks.end()) {
			if (failed.count(key) != 0) {
				continue;
			}
			Chunk chunk;
			if (!Import(base, chunk_size, chunk)) {
				failed.insert(key);
				if (++m_failures <= 16) {
					LOGF("HostImport: could not import 0x%016" PRIx64 " (+0x%" PRIx64 ")\n", base,
					     chunk_size);
				}
				continue;
			}
			m_imports++;
			it = chunks.emplace(key, chunk).first;
		}
		buffer = it->second.buffer;
		offset = address - base;
		if (++m_hits % 262144 == 1) {
			LOGF("HostImport: %" PRIu64 " bindings served, %" PRIu64 " chunks imported, %" PRIu64
			     " failed\n",
			     m_hits, m_imports, m_failures);
		}
		return true;
	}
	return false;
}

} // namespace Libs::Graphics
