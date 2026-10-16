#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_bindless_table(graphics, m_command_scheduler, m_page_manager) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_texture_cache.on_bindless_unregister = [this](ImageId id) {
		m_bindless_table.OnImageUnregistered(id);
	};
	m_texture_cache.on_image_release = [this](const Image& image) {
		if (!m_bindless_table.Enabled()) {
			return false;
		}
		std::vector<vk::ImageView> views;
		views.reserve(image.views.size());
		for (const auto& cached: image.views) {
			views.push_back(cached.view);
		}
		return m_bindless_table.ReleaseViews(views);
	};
	// vkQueueSubmit, and the queue lock that present also holds, runs on a queue thread instead
	// of the GPU thread. Frames reach the presenter only after their flip tick completed, and the
	// presenter's own scheduler stays synchronous. KYTY_ASYNC_SUBMIT=0 submits on the GPU thread.
	// KYTY_RECORD_THREAD=0: no recording thread (CommandScheduler::EnableRecordingThread), which
	// submits too.
	const char* record_thread = std::getenv("KYTY_RECORD_THREAD");
	const char* async_submit  = std::getenv("KYTY_ASYNC_SUBMIT");
	if (record_thread == nullptr || std::strcmp(record_thread, "0") != 0) {
		m_command_scheduler.EnableRecordingThread();
	} else if (async_submit == nullptr || std::strcmp(async_submit, "0") != 0) {
		m_command_scheduler.EnableAsyncSubmit();
	}
	LOGF("Recording thread: %s, async queue submit: %s\n",
	     m_command_scheduler.Threaded() ? "on" : "off",
	     m_command_scheduler.AsyncSubmit() ? "on" : "off");
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void WarmUpLocalMemory(GraphicContext& graphics, CommandScheduler& scheduler);

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	WarmUpLocalMemory(m_graphics, m_command_scheduler);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_bindless_table.OnCpuWrite(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
                                      uint64_t size) const noexcept {
	return IsMapped(vaddr, size) && m_page_manager.IsReadWatched(fault_vaddr) &&
	       m_buffer_cache.IsCleanForConcurrentRead(vaddr, size);
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_bindless_table.OnCpuWrite(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// The kernel unmaps every free range it is about to (re)map, most of which were never GPU
	// mapped. Buffers, images and GPU-dirty pages are only tracked inside mapped ranges (faults
	// and invalidations outside them are ignored), so such an unmap has nothing to invalidate:
	// skip the GPU-thread round trip and the full drain it forces. KYTY_UNMAP_SKIP_UNMAPPED=0
	// sends every unmap to the GPU thread again.
	static const bool skip_unmapped = [] {
		const auto* value = std::getenv("KYTY_UNMAP_SKIP_UNMAPPED");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	if (skip_unmapped && GuestRange {vaddr, size}.Valid()) {
		std::shared_lock lock(m_mapped_ranges_mutex);
		if (!m_mapped_ranges.Intersects(vaddr, size)) {
			return;
		}
	}
	const auto unmap = [this, vaddr, size] {
		if (m_command_scheduler.Active()) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_bindless_table.UnwatchRange(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	// Every cached buffer is synchronized so a global-memory shader sees the CPU's writes.
	// That walk touched every buffer per dispatch (a quarter of the GPU thread in the
	// world); it only has to repeat once something became CPU-dirty or a buffer appeared.
	// The walk runs at most once per guest submission: what the game wrote before submitting is
	// visible to all its commands, as on the console, and the GPU thread runs behind the game
	// anyway. Per draw, the walk and the re-protection it triggers took ~38 % of the GPU thread
	// on the title menu (21 -> 32 fps).
	const auto epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
	// KYTY_BDA_SYNC_EVERY=1: walk whenever the epoch moved, not at most once per submission.
	static const bool sync_every = [] {
		const char* value = std::getenv("KYTY_BDA_SYNC_EVERY");
		return value != nullptr && value[0] == '1';
	}();
	// KYTY_BDA_SYNC_PER_FRAME=1 (diagnostics, an upper bound: not safe in general): walk at most once
	// per guest frame (frame-done submission) instead of once per submission.
	static const bool per_frame = [] {
		const char* value = std::getenv("KYTY_BDA_SYNC_PER_FRAME");
		return value != nullptr && value[0] == '1';
	}();
	// A walk syncs every CPU write made before it starts, so a submission queued before the last
	// walk started is covered by it: the GPU thread runs a frame or so behind the game, and walking
	// again for each of its submissions re-uploaded and re-protected the pages the game had written
	// since (~5 walks a frame in UFC 5 fights, ~1.4 ms each). Writes made by the GPU thread itself
	// (command processor copies, new buffers) are no submission's, so they still start a walk.
	// So does a WAIT_REG_MEM: the game writes data after queuing the command buffer that reads it
	// and releases the GPU with a label (without that rule, black heads and plain shorts in UFC 5
	// fights). Opt-in (KYTY_BDA_WALK_ENQUEUED=1): UFC 5 waits before nearly every submission, so the
	// walks stay as frequent (2026-10-15: 36,864 walks either way, 28.2 vs 28.5 fps standing).
	static const bool walk_enqueued = [] {
		const char* value = std::getenv("KYTY_BDA_WALK_ENQUEUED");
		return value != nullptr && value[0] == '1';
	}();
	if (epoch != m_bda_synced_epoch && !sync_every && walk_enqueued && !per_frame) {
		const bool newer    = g_processing_enqueued_seq > m_bda_walk_enqueued ||
		                   g_gpu_wait_seq != m_bda_walk_wait;
		const bool gpu_made = g_gpu_thread_dirty_seq != m_bda_walk_gpu_dirty;
		if (!newer && !gpu_made) {
			m_fault_process_pending = true;
			return;
		}
		// Only the GPU thread wrote since the walk: its own ranges, not every dirty buffer.
		if (!newer && !g_gpu_thread_dirty_overflow) {
			for (const auto [address, bytes]: g_gpu_thread_dirty_ranges) {
				m_buffer_cache.SynchronizeCpuDirtyBuffersInRange(address, bytes);
			}
			g_gpu_thread_dirty_ranges.clear();
			m_bda_walk_gpu_dirty     = g_gpu_thread_dirty_seq;
			m_fault_process_pending  = true;
			m_bda_walk_gpu_only     += 1;
			return;
		}
		// Why walks run (diagnostics, in the walk log below).
		m_bda_walk_newer += newer ? 1u : 0u;
		m_bda_walk_gpu_made += gpu_made ? 1u : 0u;
	} else if (epoch != m_bda_synced_epoch && !sync_every) {
		const auto submission = per_frame && m_gpu != nullptr
		                            ? static_cast<uint64_t>(m_gpu->GetFrameNum())
		                            : g_guest_submission_seq.load(std::memory_order_relaxed);
		if (submission == m_bda_synced_submission) {
			m_fault_process_pending = true;
			return;
		}
		m_bda_synced_submission = submission;
	}
	if (epoch != m_bda_synced_epoch) {
		// The guest writes somewhere nearly all the time, so the epoch moves between most
		// dispatches; walk only the regions holding CPU-dirty pages, not every buffer.
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_bda_walk_enqueued  = g_guest_enqueued_seq.load(std::memory_order_acquire);
		m_bda_walk_gpu_dirty = g_gpu_thread_dirty_seq;
		m_bda_walk_wait      = g_gpu_wait_seq;
		g_gpu_thread_dirty_ranges.clear();
		g_gpu_thread_dirty_overflow = false;
		const auto start_time = std::chrono::steady_clock::now();
		const auto uploads    = m_buffer_cache.SyncUploads();
		m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
			m_buffer_cache.SynchronizeCpuDirtyBuffersInRange(start, end - start);
		});
		m_bda_synced_epoch = epoch;
		// How often the walk runs, against submissions and frames, what it costs and uploads.
		static uint64_t walks = 0, walk_ns = 0, walk_uploads = 0, walk_syncs = 0;
		walk_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                     std::chrono::steady_clock::now() - start_time)
		                                     .count());
		walk_uploads += m_buffer_cache.SyncUploads().second - uploads.second;
		walk_syncs += m_buffer_cache.SyncUploads().first - uploads.first;
		if (++walks % 2048 == 0 && m_gpu != nullptr) {
			LOGF("BDA sync walks: %" PRIu64 " (submission %" PRIu64 ", frame %d): %.1f us, %.1f buffer syncs, "
			     "%.1f uploads a walk\n",
			     walks, g_guest_submission_seq.load(std::memory_order_relaxed), m_gpu->GetFrameNum(),
			     static_cast<double>(walk_ns) / 2048.0 / 1000.0, static_cast<double>(walk_syncs) / 2048.0,
			     static_cast<double>(walk_uploads) / 2048.0);
			LOGF("BDA sync walk causes: newer submission %" PRIu64 ", GPU-thread writes %" PRIu64
			     ", GPU-thread ranges only %" PRIu64
			     " (enqueued %" PRIu64 ", processing %" PRIu64 ")\n",
			     m_bda_walk_newer, m_bda_walk_gpu_made, m_bda_walk_gpu_only,
			     g_guest_enqueued_seq.load(std::memory_order_relaxed),
			     g_processing_enqueued_seq);
			m_bda_walk_newer = m_bda_walk_gpu_made = m_bda_walk_gpu_only = 0;
			walk_ns = walk_uploads = walk_syncs = 0;
		}
	}
	m_fault_process_pending = true;
}

// The walk above synchronizes every CPU-dirty buffer, and its re-protection is most of the write
// faults the game's threads take: UFC 5 standing fight, 2026-10-15, ~8,400 GPU-thread protection
// downgrades a second over ~1.5 M pages, ~152,000 single-page write faults a second. A draw whose
// BDA loads have a read plan needs only the buffers in its ranges. Off by default
// (KYTY_BDA_READ_PLAN=1 turns it on): with it, UFC 5 fights lose detail (plain black shorts, black
// heads in the crowd), so some data a draw needs is outside the ranges or the walk was syncing it
// for another path; and it halved the protection changes without a measurable gain (2026-10-15:
// 28.4 vs 27.7 fps standing, 11.3 vs 11.2 us a draw).
bool RenderContext::PrepareBdaReadRanges(std::span<const GuestRange> ranges) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BDA_READ_PLAN");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled || ranges.empty() || ranges.size() > 128 ||
	    std::ranges::any_of(ranges, [](const GuestRange& range) { return !range.Valid(); })) {
		return false;
	}
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	for (const auto& range: ranges) {
		m_buffer_cache.SynchronizeCpuDirtyBuffersInRange(range.address, range.size);
	}
	m_fault_process_pending = true;
	return true;
}

// Diagnostics: KYTY_DUMP_GUEST=<frame>:<address>+<size>[,<address>+<size>...] (frame decimal,
// the rest hex) writes guest memory, with the GPU's writes downloaded first, to
// dump_<frame>_<address>.bin once that many frames have been presented. A frame written +<n>
// counts from the first use of the KYTY_WATCH_SHADER shader.
static void DumpGuestMemory(GraphicContext& graphics, BufferCache& cache) {
	struct Request {
		uint64_t                                   frame    = 0;
		bool                                       relative = false;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	static const Request request = [] {
		Request     result;
		const char* value = std::getenv("KYTY_DUMP_GUEST");
		if (value == nullptr) {
			return result;
		}
		char* end       = nullptr;
		result.relative = *value == '+';
		result.frame    = std::strtoull(result.relative ? value + 1 : value, &end, 10);
		value        = end != nullptr && *end == ':' ? end + 1 : nullptr;
		while (value != nullptr && *value != '\0') {
			const auto address = std::strtoull(value, &end, 16);
			uint64_t   size    = 4;
			if (end != nullptr && *end == '+') {
				size = std::strtoull(end + 1, &end, 16);
			}
			result.ranges.emplace_back(address, size);
			value = end != nullptr && *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	static bool done = false;
	if (done || request.ranges.empty()) {
		return;
	}
	const auto first = WatchedShaderFirstFrame();
	if (request.relative && first == UINT64_MAX) {
		return;
	}
	const auto frame = request.relative ? first + request.frame : request.frame;
	if (graphics.presented_frames.load(std::memory_order_relaxed) < frame) {
		return;
	}
	done = true;
	for (const auto& [address, size]: request.ranges) {
		cache.DownloadRangeForDiagnostics(address, size);
		std::vector<uint8_t> bytes(size);
		const bool read = LibKernel::Memory::TryReadBacking(address, bytes.data(), size);
		const auto name = "dump_" + std::to_string(frame) + "_" +
		                  std::to_string(address) + ".bin";
		if (read) {
			if (auto* file = std::fopen(name.c_str(), "wb"); file != nullptr) {
				std::fwrite(bytes.data(), 1, bytes.size(), file);
				std::fclose(file);
			}
		}
		LOGF("DumpGuest: 0x%016" PRIx64 "+0x%" PRIx64 " -> %s (%s)\n", address, size, name.c_str(),
		     read ? "written" : "not readable");
	}
}

void RenderContext::RunGarbageCollector() {
	DumpGuestMemory(GetGraphics(), m_buffer_cache);
	LogWatchedRereads(m_buffer_cache, GetGraphics().presented_frames.load(std::memory_order_relaxed));
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
