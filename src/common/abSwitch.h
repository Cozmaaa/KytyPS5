#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_ABSWITCH_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_ABSWITCH_H_

#include <atomic>
#include <cstdint>

namespace Common {

// Diagnostics: an A/B test inside one run. KYTY_AB_B lists the B arm's environment, as
// NAME=VALUE;NAME=VALUE or NAME=VALUE,NAME=VALUE (an empty VALUE is an unset variable); the A arm is the process's own
// environment. Once started (AbStart, the diagnostic input's start message, or
// KYTY_AB_START=<seconds after the first frame>), the run alternates A B B A A B ... in blocks
// of KYTY_AB_FRAMES presented frames (default 300), drops the first KYTY_AB_WARMUP frames of each
// block (default 30) and logs, per block and for the arms so far, fps and draws per frame:
// "AB block" and "AB summary" lines. Alternating within one fight compares the arms on the same
// scenes, and one route gives both. Only switches read through AbSwitch follow the arm, and only
// those safe to change between two frames should be.
namespace AbDetail {
extern std::atomic<uint32_t> g_arm;
} // namespace AbDetail

// 0 for A, 1 for B.
inline uint32_t AbArm() {
	return AbDetail::g_arm.load(std::memory_order_relaxed);
}

// KYTY_AB_B is set.
bool AbEnabled();

// The variable's value in the current arm (nullptr when unset).
const char* AbEnv(const char* name, uint32_t arm);

void AbStart();
void AbOnFrame();                  // one presented frame
void AbRecordDraw(uint64_t ns);    // one draw and its time on the GPU thread (KYTY_TABLE_CENSUS)

template <typename T>
class AbSwitch {
public:
	AbSwitch(const char* name, T (*parse)(const char* value))
	    : m_values {parse(AbEnv(name, 0)), parse(AbEnv(name, 1))} {}
	[[nodiscard]] T Get() const { return m_values[AbArm()]; }
	[[nodiscard]] T Get(uint32_t arm) const { return m_values[arm]; }

private:
	T m_values[2];
};

} // namespace Common

#endif /* EMULATOR_INCLUDE_EMULATOR_COMMON_ABSWITCH_H_ */
