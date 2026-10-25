#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_DRAWPREPSCANNER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_DRAWPREPSCANNER_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;
class CommandProcessor;

namespace HW {
class Context;
} // namespace HW

// The draw-prep scanner (host_gpu/renderer/drawPrep.h; KYTY_DRAW_PREP=0 turns it off): reads each graphics
// submission when it is queued, ahead of the GPU thread, follows its register writes on a command
// processor of its own (which never executes anything), and prepares the descriptor walks of its
// draws for the GPU thread to take.
class DrawPrepScanner final {
public:
	explicit DrawPrepScanner(RenderContext& renderer);
	~DrawPrepScanner();
	DrawPrepScanner(const DrawPrepScanner&)            = delete;
	DrawPrepScanner& operator=(const DrawPrepScanner&) = delete;

	// Under the GPU queue lock, in queue order: a submission's command stream (graphics queue 0,
	// compute queues 1..), and the register reset the GPU thread does at a suspend point.
	void EnqueueSubmission(uint32_t queue, uint64_t seq, std::span<const uint32_t> commands);
	void EnqueueReset(uint32_t queue);

	// GPU thread, at every draw and dispatch packet of a scanned submission (after
	// DrawPrep::SetCurrent, before the packet runs): when the scanner waits at this packet for a
	// register copy it can trust again (Resync), it gets this processor's registers.
	static void OfferRegisters(uint32_t queue, uint64_t seq, uint32_t ordinal,
	                           const uint32_t* packet, CommandProcessor& cp) {
		if (s_resync_want.load(std::memory_order_acquire) == ResyncTag(seq, ordinal)) [[unlikely]] {
			DeliverRegisters(queue, seq, ordinal, packet, cp);
		}
	}

private:
	[[nodiscard]] static uint64_t ResyncTag(uint64_t seq, uint32_t ordinal) {
		return (seq << 24u) | (ordinal & 0xffffffu); // nonzero: submissions count from 1
	}
	static void DeliverRegisters(uint32_t queue, uint64_t seq, uint32_t ordinal,
	                             const uint32_t* packet, CommandProcessor& cp);
	inline static std::atomic<uint64_t> s_resync_want {0};

	struct Item {
		uint32_t                  queue = 0;
		uint64_t                  seq   = 0; // 0: a register reset
		std::span<const uint32_t> commands;
	};
	// The register copy of one GPU queue (each queue has its own command processor).
	struct QueueState {
		std::unique_ptr<CommandProcessor> cp;
		std::unique_ptr<HW::Context>      saved_ctx; // the context-state push
		bool                              tainted = false;
		bool                              pushed  = false;
		bool                              no_resync = false; // a resync timed out: until the reset
	};
	QueueState& State(uint32_t queue);

	void Run();
	void Scan(const Item& item);
	void ScanDraw(uint64_t seq, uint32_t ordinal, const uint32_t* packet);
	void ScanDispatch(uint64_t seq, uint32_t ordinal, const uint32_t* packet, uint32_t opcode);
	bool ApplyRegisterIndirect(uint32_t header, const uint32_t* body);
	void ApplyContextState(uint32_t operation);
	// A tainted register copy at a draw or dispatch packet: false while it stays tainted; true when
	// it holds the GPU thread's registers before this packet (which the GPU thread runs now).
	bool Resync(uint64_t seq, uint32_t ordinal, const uint32_t* packet);

	RenderContext&                         m_renderer;
	std::array<std::unique_ptr<QueueState>, 64> m_queues;
	QueueState*                            m_state = nullptr; // of the item being scanned
	uint32_t                               m_queue = 0;

	std::mutex              m_mutex;
	std::condition_variable m_work;
	std::deque<Item>        m_items;
	bool                    m_stop = false;
	std::thread             m_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_DRAWPREPSCANNER_H_
