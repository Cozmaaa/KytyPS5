#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <utility>
#include <vector>
#include <mutex>
#include <thread>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		// A waiter spins on a plain load, with pauses, rather than on test_and_set: an atomic write
		// a turn took the line from the owner. It yields after a while: the owner may be in a
		// protection change (VirtualProtect), which waits for the guest address space's lock
		// while a guest map or unmap holds it, or preempted. The GPU thread and every game
		// thread's write fault take these locks (~22,000 faults a second in a UFC 5 fight).
		for (uint32_t spins = 0; m_lock.test_and_set(std::memory_order_acquire);) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			while (m_lock.test(std::memory_order_relaxed)) {
				if (spins < 256) {
					spins++;
					Pause();
				} else {
					std::this_thread::yield();
				}
			}
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static void Pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
		_mm_pause();
#else
		std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
	}

	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// Bumped whenever pages become CPU-dirty anywhere (a write fault, an untrack, an
// invalidation). A consumer that must see every CPU write, such as the pre-dispatch sync of
// all buffers for global-memory shaders, can skip its walk while this has not moved.
inline std::atomic<uint64_t> g_cpu_dirty_epoch {1};

// Bumped by the GPU thread when it starts a guest submission (its first slice).
inline std::atomic<uint64_t> g_guest_submission_seq {1};

// Bumped by the guest thread that submits, as it queues a submission (GuestGpu::Enqueue), which
// stamps the submission with it. Every CPU write a submission may see was made before its stamp.
inline std::atomic<uint64_t> g_guest_enqueued_seq {0};
// The stamp of the submission the GPU thread runs now (GPU thread only).
inline uint64_t g_processing_enqueued_seq = 0;
// Bumped by the GPU thread at each WAIT_REG_MEM: the game writes data after queuing the command
// buffer that reads it and lets the GPU past such a wait with a label, so its writes before the
// label become visible there, not at the submission (UFC 5's skinned fighters and crowd).
inline uint64_t g_gpu_wait_seq = 0;
// Set on the GPU thread. CPU-dirty pages and buffer registrations it makes itself (command
// processor copies and fills, new buffers) bump g_gpu_thread_dirty_seq: no submission stamp
// covers them.
inline thread_local bool g_tracker_on_gpu_thread = false;
inline uint64_t          g_gpu_thread_dirty_seq  = 0;
// The ranges behind g_gpu_thread_dirty_seq since RenderContext::PrepareBda last took them (GPU
// thread only); past GpuThreadDirtyCapacity the list is dropped and the next preparation walks.
inline constexpr size_t                            GpuThreadDirtyCapacity = 4096;
inline std::vector<std::pair<uint64_t, uint64_t>> g_gpu_thread_dirty_ranges;
inline bool                                        g_gpu_thread_dirty_overflow = false;
inline void NoteGpuThreadDirty(uint64_t vaddr, uint64_t size) {
	++g_gpu_thread_dirty_seq;
	if (g_gpu_thread_dirty_overflow) {
		return;
	}
	if (g_gpu_thread_dirty_ranges.size() >= GpuThreadDirtyCapacity) {
		g_gpu_thread_dirty_overflow = true;
		g_gpu_thread_dirty_ranges.clear();
		return;
	}
	g_gpu_thread_dirty_ranges.emplace_back(vaddr, size);
}

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }

	// Lock-free "any page in this region may be dirty" summaries. A binding that spans a
	// whole heap covers hundreds of regions; without these every draw locks and walks each
	// one. False is exact (set only after a walk saw no bits); true is conservative.
	template <DirtySource source>
	[[nodiscard]] bool MaybeModified() const noexcept {
		return GetSummary<source>().load(std::memory_order_acquire);
	}
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
			StoreSummary<source>(true);
			if constexpr (source == DirtySource::Cpu) {
				g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
				if (g_tracker_on_gpu_thread) {
					NoteGpuThreadDirty(vaddr, size);
				}
			}
		} else {
			bits.UnsetRange(start, end);
			if (bits.None()) {
				StoreSummary<source>(false);
			}
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	// A write fault's fault-ahead (MemoryTracker::MarkCpuDirtyAhead): the run of pages from vaddr's
	// on, at most max_pages and within this region, that are neither CPU- nor GPU-dirty becomes
	// CPU-dirty, as if each had faulted. Returns the pages marked. The caller holds the lock.
	size_t MarkCpuDirtyAhead(uint64_t vaddr, size_t max_pages) {
		const auto start = GetPageRange(vaddr, TRACKER_PAGE_SIZE).first;
		const auto limit = std::min(TRACKER_REGION_PAGES, start + max_pages);
		auto       end   = start;
		while (end < limit && !m_cpu_dirty.Get(end) && !m_gpu_dirty.Get(end)) {
			end++;
		}
		if (end != start) {
			ChangeState<DirtySource::Cpu, true>(m_cpu_addr + start * TRACKER_PAGE_SIZE,
			                                    (end - start) * TRACKER_PAGE_SIZE);
		}
		return end - start;
	}

	// protect = false (CPU, clear): the cleared pages stay writable until ProtectCleanCpuPages, so
	// several clears share one protection pass.
	template <DirtySource source, bool clear, bool protect = true, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(protect || (clear && source == DirtySource::Cpu));
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if (bits.None()) {
				StoreSummary<source>(false);
			}
			if constexpr (protect && source == DirtySource::Cpu) {
				UpdateProtection<true, false>();
			} else if constexpr (protect) {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// Write-protects the pages ForEachModifiedRange<Cpu, true, false> made clean. The caller holds
	// the lock.
	void ProtectCleanCpuPages() { UpdateProtection<true, false>(); }

	// The tracker mirrors the CPU summary in one bit per region (MemoryTracker::IsRangeCpuCleanHint).
	// Set before the region is published.
	void SetCpuSummaryBit(std::atomic<uint64_t>* word, uint64_t bit) noexcept {
		m_cpu_summary_word = word;
		m_cpu_summary_bit  = bit;
	}

	TrackingSpinLock lock;

private:
	// Every summary change goes through here; the CPU one is mirrored in the tracker's bitmap at
	// the same point, before any page protection changes.
	template <DirtySource source>
	void StoreSummary(bool value) {
		GetSummary<source>().store(value, std::memory_order_release);
		if constexpr (source == DirtySource::Cpu) {
			if (m_cpu_summary_word != nullptr) {
				if (value) {
					m_cpu_summary_word->fetch_or(m_cpu_summary_bit, std::memory_order_acq_rel);
				} else {
					m_cpu_summary_word->fetch_and(~m_cpu_summary_bit, std::memory_order_acq_rel);
				}
			}
		}
	}

	template <bool track, bool is_read>
	void UpdateProtection() {
		const auto protection = is_read ? ~m_gpu_dirty : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	std::atomic<bool>& GetSummary() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	template <DirtySource source>
	const std::atomic<bool>& GetSummary() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	// The constructor fills m_cpu_dirty, so the CPU summary starts true.
	std::atomic<bool> m_cpu_maybe_dirty {true};
	std::atomic<bool> m_gpu_maybe_dirty {false};
	std::atomic<uint64_t>* m_cpu_summary_word = nullptr;
	uint64_t               m_cpu_summary_bit  = 0;
	RegionBits   m_writable;
	RegionBits   m_readable;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
