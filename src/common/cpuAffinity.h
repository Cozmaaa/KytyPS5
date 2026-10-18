#ifndef EMULATOR_SRC_COMMON_CPUAFFINITY_H_
#define EMULATOR_SRC_COMMON_CPUAFFINITY_H_

namespace Common {

// Where the emulator's own threads run, and at what priority: X3D mode, off by default.
//
// Measured on a 9800X3D (single CCD) with UFC 5, CPU vs CPU fight at 1440p, 2026-10-14: X3D off
// 29.2 fps (6 runs), X3D without reserved cores 27.2 (3), X3D with 2 reserved cores 25.9 (3).
// Taking cores away from the guest's many threads cost more than the dedicated cores gained.
// Turn it on with KYTY_X3D=1 or --cpu-placement true; dual-CCD X3D processors are unmeasured.
// Off, only the older hybrid rule stays: on a hybrid CPU the critical and high threads keep to
// the performance cores (KYTY_PCORE_AFFINITY=0 turns that off too).
//
// The GPU thread is the emulator's bottleneck, and the threads feeding it come next. The
// processor's layout decides the placement:
// - Ryzen X3D with two CCDs (7900X3D, 7950X3D, 9900X3D, 9950X3D): only one CCD has the stacked
//   L3. The emulator, guest threads included, is kept on that CCD; background work (pipeline
//   compilation, shader translation, file streaming, disk cache writes) goes to the other one.
// - Hybrid CPUs (Intel 12th gen and later): the critical and high threads stay on the
//   performance cores (on an efficiency core the GPU thread ran ~1.7x slower per operation,
//   i7-13700F, 2026-10-06), background work prefers the efficiency cores.
// - Every CPU with enough cores, single-CCD X3D (7800X3D, 9800X3D) included: each critical
//   thread gets a physical core to itself, both SMT siblings, which no other emulator or guest
//   thread is scheduled on (KYTY_X3D_RESERVE cores, default 2; 0 turns this off).
//
// Windows uses CPU sets (a strong scheduler preference that ends with the process; nothing
// system-wide changes) and thread priorities. Linux pins the critical and high threads to the
// fast cores only. macOS: nothing. "CPU placement:" log lines say what was done.
enum class ThreadClass {
	Critical,   // GPU thread, Vulkan recording thread: a dedicated core, highest priority.
	High,       // Threads the critical ones wait on: fast cores, above-normal priority.
	Background, // Throughput work: the other CCD or efficiency cores when there are any.
	Low,        // Background placement, below-normal priority.
};

// --cpu-placement true|false (the launcher's X3D mode checkbox): overrides KYTY_X3D. Called while
// parsing the command line, before any thread is placed.
void SetCpuPlacementEnabled(bool enabled);

// Applies the placement for `cls` to the calling thread. `name` is for the log line.
void PlaceCurrentThread(ThreadClass cls, const char* name);

// Reads the topology, reserves the critical cores and sets the process's default placement
// (which guest threads and every untagged thread follow). Idempotent; PlaceCurrentThread calls it
// too. Logs the topology once.
void InitCpuPlacement();

// Diagnostics (KYTY_CPU_TRACE=1): the calling thread notes the logical processor it runs on; every
// 10 s each traced thread logs where it ran ("CpuTrace <name>: cpu:share ..."). Call it from a hot
// loop of the thread (it costs one processor-number read).
void TraceCurrentCpu(const char* name);

} // namespace Common

#endif // EMULATOR_SRC_COMMON_CPUAFFINITY_H_
