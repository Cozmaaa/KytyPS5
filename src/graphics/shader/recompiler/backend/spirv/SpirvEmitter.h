#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info);

// Optional image features of the device, set once by the device layer.
struct HostImageFeatures {
	// shaderResourceMinLod: the MinLod image operand, used for IMAGE_SAMPLE*_CL.
	bool min_lod = false;
};

void              SetHostImageFeatures(const HostImageFeatures& features);
HostImageFeatures GetHostImageFeatures();

// S_MEMREALTIME counts at 100 MHz. The emitter reads the device clock (OpReadClockKHR, device
// scope) and shifts it right by this many bits to approach that rate. The device layer sets it
// from the device's clock rate: 3 for a 1 GHz clock (NVIDIA, the default), 0 for a 100 MHz one
// (AMD).
void     SetDeviceClockShift(uint32_t shift);
uint32_t GetDeviceClockShift();

// A coherent (GLC) buffer load is followed by a device-scope acquire barrier, so that a load
// polling for another workgroup's write sees it. Without one, AMD's drivers keep serving the
// cached value (BufferWorkgroupPublication, ISSUES.md #16); NVIDIA reloads anyway. The device
// layer enables it on every vendor but NVIDIA; KYTY_COHERENT_LOAD_ACQUIRE=0/1 overrides that.
void SetCoherentLoadAcquire(bool enabled);
bool GetCoherentLoadAcquire();

// The device can flush 32-bit denormals itself (shaderDenormFlushToZeroFloat32, settable apart from
// the 16- and 64-bit behavior). The device layer sets it.
// KYTY_NATIVE_FTZ (default 2; 0: the explicit flushes, as before), for shaders whose guest MODE
// flushes FP32 denormals:
// 1: on such a device, declare DenormFlushToZero for 32-bit floats and leave out the explicit
//    flushes in front of floating-point operations (legacy MAD/MAC, RCP, RSQ, SQRT, EXP, LOG):
//    ~4 ALU instructions for each operand and result;
// 2: leave them out on any device (also declaring the mode where supported). Inexact on devices
//    that keep denormals: values below 2^-126 then survive those operations, as they already do
//    plain V_MUL/V_ADD here. RTX 50 drivers report no 32-bit DenormFlushToZero. UFC 5 standing
//    fights, 2 interleaved runs each: 0: 30.8 / 30.4 fps, GPU frame 20.2 / 21.3 ms; 2: 35.9 / 35.4
//    fps, 15.9 / 15.3 ms (with per-command GPU timing, which makes the game GPU-bound); screenshots
//    unchanged (3d47b991). Without timing, 2 warm runs each: 0: 35.3 / 35.6, 2: 38.2 / 36.9 (+6 %).
void     SetDenormFlushF32Supported(bool supported);
bool     GetDenormFlushF32Supported();
uint32_t NativeDenormFlushF32Mode();

// Why a mesh program cannot run in passes (ShaderMeshInputInfo::passes), or nullptr.
[[nodiscard]] const char* MeshPassesUnsupported(const IR::Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
