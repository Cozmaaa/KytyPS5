#ifndef KYTY_COMMON_OWNCORES_H_
#define KYTY_COMMON_OWNCORES_H_

// Not in cpuAffinity.h: the shader disk cache key covers that header (the recompiler includes it),
// so a declaration added there would invalidate every cached shader.

namespace Common {

// Physical cores of their own (both SMT siblings) for the GPU thread and the threads it waits on;
// the process's other threads keep off them. AbSwitches (in-run A/B), each a logical processor
// number n, meaning n's core, or 0 for none:
//   KYTY_GPU_OWN_CORE        the GPU thread. Unset: the second fast core on CPUs with 8 or more
//                            cores, none on smaller ones.
//   KYTY_RECORDING_OWN_CORE  the Vulkan recording thread. Unset: the third fast core on CPUs with
//                            8 or more cores, none on smaller ones.
//   KYTY_SCANNER_OWN_CORE    the draw preparation scanner. Unset: none.
// Windows only; ignored in X3D mode, which places the threads itself. Called by the GPU thread
// between submissions: two loads and a compare unless an arm's value changed or a thread started.
void ApplyOwnCores();

} // namespace Common

#endif /* KYTY_COMMON_OWNCORES_H_ */
