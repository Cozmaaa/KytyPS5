#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_HOSTSAMPLER_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_HOSTSAMPLER_H_

namespace Common {

// Diagnostics: a sampling profiler of one host thread (KYTY_HOST_SAMPLER=<delay s>:<seconds>).
// The calling thread registers itself; a sampler thread then waits <delay> seconds, suspends it
// about 1,000 times a second for <seconds>, records its call stack (unwound with the
// executable's own function table: no loader locks, which the suspended thread may hold) and
// logs the functions by self and inclusive share of the samples. Unlike Tracy zones it adds no
// cost to the sampled code and misses nothing that has no zone. Windows only.
void SampleCurrentThread(const char* name);

} // namespace Common

#endif /* EMULATOR_INCLUDE_EMULATOR_COMMON_HOSTSAMPLER_H_ */
