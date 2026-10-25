#include "graphics/guest_gpu/drawPrepScanner.h"

#include "common/cpuAffinity.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/drawPrep.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

namespace {

bool IsGraphicsDraw(uint32_t opcode) {
	return DrawPrep::CountsAsDrawOrDispatch(opcode) && opcode != Pm4::IT_DISPATCH_DIRECT &&
	       opcode != Pm4::IT_DISPATCH_INDIRECT;
}

// An indirect table that tainted a queue's register copy. A taint lasts until the queue's next
// suspend point (UFC 5: one taint skipped 45-300 draws, 2026-10-24), so which tables fail, and why,
// is in the log and the capture's messages: the first 48, then every 1024th.
void ReportIndirectTaint(uint32_t queue, uint32_t opcode, uint64_t address, uint32_t count,
                         const char* reason) {
	static std::atomic<uint32_t> reported {0};
	const auto                   n = reported.fetch_add(1, std::memory_order_relaxed);
	if (n >= 48 && n % 1024 != 0) {
		return;
	}
	const char* kind = opcode == Pm4::IT_SET_SH_REG_INDIRECT        ? "SH"
	                   : opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT ? "context"
	                                                                : "uconfig";
	LOGF("DrawPrep taint #%u: queue %u, %s pairs at 0x%016" PRIx64 " (%u): %s\n", n, queue, kind,
	     address, count, reason);
	Profiler::Message(Profiler::MessageWarning,
	                  "DrawPrep taint #%u: queue %u, %s pairs at 0x%016" PRIx64 " (%u): %s", n, queue,
	                  kind, address, count, reason);
}

// KYTY_DRAW_PREP_RESYNC=0: a taint lasts until the queue's next reset, as before.
bool ResyncEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_PREP_RESYNC");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// The packet a scanner waits at (Resync), and the registers the GPU thread hands it there.
struct ResyncChannel {
	std::mutex              mutex;
	std::condition_variable delivered;
	uint32_t                queue   = 0;
	uint64_t                seq     = 0;
	uint32_t                ordinal = 0;
	const uint32_t*         packet  = nullptr;
	bool                    ready   = false;
	HW::Context             ctx;
	HW::Context             saved_ctx;
	HW::Shader              sh;
	HW::UserConfig          ucfg;
	bool                    pushed = false;
	HW::UserSgprType        marker = HW::UserSgprType::Unknown;
};

ResyncChannel& GetResyncChannel() {
	static auto* channel = new ResyncChannel;
	return *channel;
}

} // namespace

void DrawPrepScanner::DeliverRegisters(uint32_t queue, uint64_t seq, uint32_t ordinal,
                                       const uint32_t* packet, CommandProcessor& cp) {
	KYTY_PROFILER_BLOCK("DrawPrep::DeliverRegisters");
	auto&           channel = GetResyncChannel();
	std::lock_guard lock(channel.mutex);
	// The scanner clears the want under this lock when it stops waiting.
	if (s_resync_want.load(std::memory_order_relaxed) != ResyncTag(seq, ordinal) || channel.ready ||
	    channel.queue != queue || channel.seq != seq || channel.ordinal != ordinal ||
	    channel.packet != packet) {
		return;
	}
	channel.ctx       = cp.GetCtx();
	channel.saved_ctx = cp.GetSavedCtx();
	channel.sh        = cp.GetShCtx();
	channel.ucfg      = cp.GetUcfg();
	channel.pushed    = cp.IsContextStatePushed();
	channel.marker    = cp.GetUserDataMarker();
	channel.ready     = true;
	channel.delivered.notify_one();
}

// A taint used to last until the queue's next suspend point: one failed indirect table cost the
// 45-300 draws after it their preparation (UFC 5: 2-8 % of all draws, ~0.5 ms of the GPU thread a
// frame, 2026-10-24). Instead the scanner waits at the next draw for the GPU thread's registers
// before it, and goes on from there with a copy it can trust. Its lead is gone then, but it walks a
// draw in a third of the GPU thread's time and is ahead again within a few draws.
bool DrawPrepScanner::Resync(uint64_t seq, uint32_t ordinal, const uint32_t* packet) {
	if (!ResyncEnabled() || m_state->no_resync) {
		return false;
	}
	const DrawPrep::Key key {.queue = m_queue, .seq = seq, .ordinal = ordinal, .packet = packet};
	if (!DrawPrep::GpuProgress(m_queue).Before(key)) {
		// At or past this packet the GPU thread offers nothing; tried again at the next one.
		DrawPrep::Add(DrawPrep::Counter::ResyncPassed);
		return false;
	}
	auto&            channel = GetResyncChannel();
	std::unique_lock lock(channel.mutex);
	channel.queue   = m_queue;
	channel.seq     = seq;
	channel.ordinal = ordinal;
	channel.packet  = packet;
	channel.ready   = false;
	s_resync_want.store(ResyncTag(seq, ordinal), std::memory_order_release);
	// The scanner can lead by thousands of packets (DrawPrep's MaxPending), so the wait may be long.
	// It ends early when the GPU thread goes past without offering (it read the want before it was
	// set, or its packets diverged from the scanner's).
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
	bool       passed   = false;
	while (!channel.ready) {
		channel.delivered.wait_for(lock, std::chrono::milliseconds(1));
		if (channel.ready) {
			break;
		}
		if (key.Before(DrawPrep::GpuProgress(m_queue))) {
			passed = true;
			break;
		}
		if (std::chrono::steady_clock::now() >= deadline) {
			break;
		}
	}
	s_resync_want.store(0, std::memory_order_relaxed);
	if (!channel.ready) {
		if (passed) {
			DrawPrep::Add(DrawPrep::Counter::ResyncPassed);
		} else {
			m_state->no_resync = true;
			DrawPrep::Add(DrawPrep::Counter::ResyncTimeout);
		}
		return false;
	}
	channel.ready = false;
	auto& cp      = *m_state->cp;
	cp.GetCtx()   = channel.ctx;
	cp.GetShCtx() = channel.sh;
	cp.GetUcfg()  = channel.ucfg;
	cp.SetUserDataMarker(channel.marker);
	*m_state->saved_ctx = channel.saved_ctx;
	m_state->pushed     = channel.pushed;
	m_state->tainted    = false;
	DrawPrep::Add(DrawPrep::Counter::Resynced);
	return true;
}

DrawPrepScanner::DrawPrepScanner(RenderContext& renderer): m_renderer(renderer) {
	m_thread = std::thread([this] { Run(); });
}

DrawPrepScanner::QueueState& DrawPrepScanner::State(uint32_t queue) {
	auto& state = m_queues[queue % m_queues.size()];
	if (state == nullptr) {
		state            = std::make_unique<QueueState>();
		state->cp        = std::make_unique<CommandProcessor>(m_renderer, queue == 0 ? 0 : 0x1f + static_cast<int>(queue));
		state->saved_ctx = std::make_unique<HW::Context>();
		// As GuestGpu: the graphics processor starts reset, a compute one as constructed.
		if (queue == 0) {
			state->cp->Reset();
		}
	}
	return *state;
}

DrawPrepScanner::~DrawPrepScanner() {
	{
		std::lock_guard lock(m_mutex);
		m_stop = true;
	}
	m_work.notify_one();
	if (m_thread.joinable()) {
		m_thread.join();
	}
}

void DrawPrepScanner::EnqueueSubmission(uint32_t queue, uint64_t seq,
                                        std::span<const uint32_t> commands) {
	{
		std::lock_guard lock(m_mutex);
		m_items.push_back({queue, seq, commands});
	}
	m_work.notify_one();
}

void DrawPrepScanner::EnqueueReset(uint32_t queue) {
	{
		std::lock_guard lock(m_mutex);
		m_items.push_back({queue, 0, {}});
	}
	m_work.notify_one();
}

void DrawPrepScanner::Run() {
	KYTY_PROFILER_THREAD("Thread_DrawPrep");
	Common::PlaceCurrentThread(Common::ThreadClass::High, "draw preparation scanner");
	// The clean-backing reads (TryReadGpuCleanBacking) answer "maybe GPU-written" on any other
	// thread; a read delegate checks the buffer cache's GPU-dirty ranges under their shared lock,
	// which the GPU thread holds exclusively whenever it changes them.
	LibKernel::Memory::SetGpuReadDelegate(true);
	for (;;) {
		Item item;
		{
			std::unique_lock lock(m_mutex);
			m_work.wait(lock, [this] { return m_stop || !m_items.empty(); });
			if (m_stop) {
				return;
			}
			item = m_items.front();
			Common::TraceCurrentCpu("draw preparation scanner");
			m_items.pop_front();
		}
		m_queue = item.queue;
		m_state = &State(item.queue);
		if (item.seq == 0) {
			// The GPU thread resets its command processor at a suspend point.
			m_state->cp->Reset();
			m_state->pushed    = false;
			m_state->tainted   = false;
			m_state->no_resync = false;
		} else {
			Scan(item);
		}
		DrawPrep::LogCountersIfDue();
	}
}

// As CommandProcessor::ApplyContextStateOperation, without its exits: a sequence the scanner
// cannot follow only makes its register copy unreliable.
void DrawPrepScanner::ApplyContextState(uint32_t operation) {
	auto&      ctx        = m_state->cp->GetCtx();
	const bool was_tainted = m_state->tainted;
	struct Count {
		const bool& was;
		const bool& now;
		~Count() {
			if (!was && now) {
				DrawPrep::Add(DrawPrep::Counter::TaintContext);
			}
		}
	} count {was_tainted, m_state->tainted};
	switch (static_cast<ContextStateOperation>(operation)) {
		case ContextStateOperation::Clear: ctx.Reset(); break;
		case ContextStateOperation::Push:
			m_state->tainted |= m_state->pushed;
			*m_state->saved_ctx = ctx;
			m_state->pushed     = true;
			break;
		case ContextStateOperation::Pop:
			m_state->tainted |= !m_state->pushed;
			ctx      = *m_state->saved_ctx;
			m_state->pushed = false;
			break;
		case ContextStateOperation::PushClear:
			m_state->tainted |= m_state->pushed;
			*m_state->saved_ctx = ctx;
			m_state->pushed     = true;
			ctx.Reset();
			break;
		default: m_state->tainted = true; break;
	}
}

// SET_*_REG_INDIRECT: the register pairs are read with the clean backing read (no fault, no wait)
// and handed to the command processor's own handler from a local copy. False when the pairs
// cannot be read that way (the GPU wrote them): the register copy is then unreliable.
bool DrawPrepScanner::ApplyRegisterIndirect(uint32_t header, const uint32_t* body) {
	const auto opcode = (header >> 8u) & 0xffu;
	if (KYTY_PM4_LEN(header) != 5u) {
		ReportIndirectTaint(m_queue, opcode, 0, 0, "packet length");
		return false;
	}
	const uint64_t address =
	    (static_cast<uint64_t>(body[0]) & 0xfffffffcu) | (static_cast<uint64_t>(body[1]) << 32u);
	const uint32_t count = body[3] & 0x3fffu;
	if (count == 0) {
		return true;
	}
	if (address == 0) {
		ReportIndirectTaint(m_queue, opcode, address, count, "null address");
		return false;
	}
	thread_local std::vector<uint32_t> pairs;
	pairs.resize(static_cast<size_t>(count) * 2u);
	if (!LibKernel::Memory::TryReadGpuCleanBacking(address, pairs.data(),
	                                               pairs.size() * sizeof(uint32_t))) {
		ReportIndirectTaint(m_queue, opcode, address, count,
		                    LibKernel::Memory::TryReadBacking(address, pairs.data(),
		                                                      pairs.size() * sizeof(uint32_t))
		                        ? "may be GPU-written"
		                        : "no backing");
		return false;
	}
	// The scanner reads ahead of the GPU thread, so the table may not be written yet: offsets the
	// GPU thread's handlers reject (they exit) taint the scan instead (UFC 5: an SH pair 0x401d2c00,
	// 2026-10-17). The GPU thread reads the table again when it executes the packet.
	const uint32_t limit = opcode == Pm4::IT_SET_SH_REG_INDIRECT       ? Pm4::SH_NUM
	                       : opcode == Pm4::IT_SET_UCONFIG_REG_INDIRECT ? Pm4::UC_NUM
	                                                                    : UINT32_MAX;
	for (uint32_t i = 0; i < count && limit != UINT32_MAX; i++) {
		const uint32_t raw = pairs[static_cast<size_t>(i) * 2u];
		if (raw != 0xffffffffu && (raw & ~0x70000000u) >= limit) {
			char reason[48];
			snprintf(reason, sizeof(reason), "pair %u offset 0x%08x", i, raw);
			ReportIndirectTaint(m_queue, opcode, address, count, reason);
			return false;
		}
	}
	const auto                    local = reinterpret_cast<uint64_t>(pairs.data());
	const std::array<uint32_t, 4> copy {static_cast<uint32_t>(local), static_cast<uint32_t>(local >> 32u),
	                                    body[2], body[3]};
	(void)g_cp_op_func[opcode](*m_state->cp, header, copy.data(), 5u, 5u);
	return true;
}

void DrawPrepScanner::ScanDraw(uint64_t seq, uint32_t ordinal, const uint32_t* packet) {
	if (m_state->tainted) {
		// Resynced or not, this packet is the GPU thread's to walk: it is running it.
		(void)Resync(seq, ordinal, packet);
		DrawPrep::Add(DrawPrep::Counter::Tainted);
		return;
	}
	const DrawPrep::Key key {.queue = m_queue, .seq = seq, .ordinal = ordinal, .packet = packet};
	if (!DrawPrep::GpuProgress(m_queue).Before(key)) {
		DrawPrep::Add(DrawPrep::Counter::SkipBehind);
		return;
	}
	const auto& ctx  = m_state->cp->GetCtx();
	const auto& sh   = m_state->cp->GetShCtx();
	const auto& ucfg = m_state->cp->GetUcfg();
	if (!DrawPrepHasVertexShader(sh)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return;
	}
	std::array<Prospero::ColorComponentMapping, 8> mapping {};
	const bool ps_active = DrawPrepPixelState(ctx, sh, mapping);
	DrawPrep::PacketResult result;
	DrawPrep::Recycle(result);
	result.key = key;
	if (m_renderer.GetPipelineCache().DrawPrepGraphics(sh.GetVs(), sh.GetPs(),
	                                                   ctx.GetShaderRegisters(), ctx, ucfg, mapping,
	                                                   ps_active, result)) {
		DrawPrep::Publish(std::move(result));
	}
}

// A dispatch: the compute inputs RenderExecutor::DispatchDirect / DispatchIndirect build.
void DrawPrepScanner::ScanDispatch(uint64_t seq, uint32_t ordinal, const uint32_t* packet,
                                   uint32_t opcode) {
	if (m_state->tainted) {
		// Resynced or not, this packet is the GPU thread's to walk: it is running it.
		(void)Resync(seq, ordinal, packet);
		DrawPrep::Add(DrawPrep::Counter::Tainted);
		return;
	}
	const DrawPrep::Key key {.queue = m_queue, .seq = seq, .ordinal = ordinal, .packet = packet};
	if (!DrawPrep::GpuProgress(m_queue).Before(key)) {
		DrawPrep::Add(DrawPrep::Counter::SkipBehind);
		return;
	}
	const auto& cs = m_state->cp->GetShCtx().GetCs();
	if (cs.cs_regs.data_addr == 0 || PipelineCache::IsSkipListed(cs.cs_regs.data_addr)) {
		DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
		return;
	}
	ShaderComputeInputInfo input_info {};
	if (opcode == Pm4::IT_DISPATCH_DIRECT) {
		if (KYTY_PM4_LEN(packet[0]) != 5u) {
			DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
			return;
		}
		const uint32_t groups[3] = {packet[1], packet[2], packet[3]};
		const uint32_t mode      = packet[4];
		if (groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
			DrawPrep::Add(DrawPrep::Counter::SkipUnsupported);
			return;
		}
		const bool use_thread_dimensions =
		    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		input_info.dispatch_thread_dimensions = use_thread_dimensions;
		const uint32_t group_sizes[3] = {cs.cs_regs.num_thread_x, cs.cs_regs.num_thread_y,
		                                 cs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			input_info.workgroup_counts[axis] = groups[axis];
			if (use_thread_dimensions) {
				const auto size = std::max(group_sizes[axis], 1u);
				input_info.workgroup_counts[axis] = groups[axis] / size + (groups[axis] % size != 0u);
			}
		}
	} else {
		const uint32_t mode = packet[0] == 0xc0021600u ? packet[3] : packet[2];
		const bool     use_thread_dimensions =
		    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
		input_info.dispatch_thread_dimensions   = use_thread_dimensions;
		input_info.dispatch_dimensions_indirect = use_thread_dimensions;
	}
	DrawPrep::PacketResult result;
	DrawPrep::Recycle(result);
	result.key = key;
	if (m_renderer.GetPipelineCache().DrawPrepCompute(cs, m_state->cp->GetCtx().GetShaderRegisters(),
	                                                  input_info, result)) {
		DrawPrep::Publish(std::move(result));
	}
}

// The packet walk of CommandProcessor::ProcessPm4, for registers only: register writes go to the
// scanner's command processor, draws are prepared, everything else is stepped over.
void DrawPrepScanner::Scan(const Item& item) {
	struct Cursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset = 0;
	};
	std::vector<Cursor> stack {{item.commands}};
	uint32_t            ordinal = 0;
	while (!stack.empty()) {
		auto& cursor = stack.back();
		if (cursor.offset >= cursor.commands.size()) {
			stack.pop_back();
			continue;
		}
		const auto* const packet    = cursor.commands.data() + cursor.offset;
		const auto        total     = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining = total - cursor.offset;
		const auto        header    = packet[0];
		if (header == 0x80000000u) {
			cursor.offset++;
			continue;
		}
		if (remaining < 2) {
			return;
		}
		if ((header >> 30u) == 0u) {
			const auto length = ((header >> 16u) & 0x3fffu) + 2u;
			if (length > remaining) {
				return;
			}
			cursor.offset += length;
			continue;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		const auto length = KYTY_PM4_LEN(header);
		if (length > remaining) {
			return;
		}
		if (DrawPrep::CountsAsDrawOrDispatch(opcode)) {
			const auto this_ordinal = ordinal++;
			DrawPrep::Add(DrawPrep::Counter::Scanned);
			if (IsGraphicsDraw(opcode)) {
				ScanDraw(item.seq, this_ordinal, packet);
			} else {
				ScanDispatch(item.seq, this_ordinal, packet, opcode);
			}
			cursor.offset += length;
			continue;
		}
		const auto command = header & ~1u;
		switch (opcode) {
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX: {
				// The GPU thread's handlers exit on registers they do not know; read ahead of it,
				// such a packet may be memory the game has not written (or has reused), so it
				// taints the scan instead, as for indirect tables (ApplyRegisterIndirect).
				if (!CpRegisterPacketAccepted(command, packet + 1)) {
					m_state->tainted = true;
					DrawPrep::Add(DrawPrep::Counter::TaintRegister);
					cursor.offset += length;
					continue;
				}
				const auto consumed = g_cp_op_func[opcode](*m_state->cp, command, packet + 1, remaining, total) + 1u;
				cursor.offset += consumed;
				continue;
			}
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
				if (!ApplyRegisterIndirect(command, packet + 1)) {
					m_state->tainted = true;
					DrawPrep::Add(DrawPrep::Counter::TaintIndirect);
				}
				break;
			case Pm4::IT_CLEAR_STATE: m_state->cp->GetCtx().Reset(); break;
			case Pm4::IT_INDIRECT_BUFFER: {
				if (length != 4u) {
					// A conditional branch on guest memory: the scanner cannot follow it.
					return;
				}
				const auto* sub = reinterpret_cast<const uint32_t*>(
				    packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
				const uint32_t control = packet[3];
				const uint32_t count   = control & 0xfffffu;
				cursor.offset += length;
				if (count != 0 && sub != nullptr) {
					if ((control & (1u << 20u)) != 0) {
						cursor = {std::span(sub, count)};
					} else {
						stack.push_back({std::span(sub, count)});
					}
				}
				continue;
			}
			case Pm4::IT_NOP: {
				const auto r = KYTY_PM4_R(header);
				if (r == Pm4::R_ZERO) {
					if ((packet[1] & 0xffff0000u) == 0x68750000u) {
						const auto id = packet[1] & 0xfffu;
						if (id == 0x4u || id == 0xdu) {
							(void)g_cp_op_func[Pm4::IT_NOP](*m_state->cp, command, packet + 1, remaining,
							                                total);
						}
					}
				} else if (r == Pm4::R_CONTEXT_STATE) {
					ApplyContextState(packet[1]);
				} else if (r == Pm4::R_DISPATCH_RESET) {
					m_state->cp->Reset();
					m_state->pushed = false;
				}
				break;
			}
			default: break;
		}
		cursor.offset += length;
	}
}

} // namespace Libs::Graphics
