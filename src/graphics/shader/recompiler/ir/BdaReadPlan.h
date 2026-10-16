#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BDAREADPLAN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BDAREADPLAN_H_

#include "graphics/host_gpu/regionDefinitions.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Program;
struct ResourceSnapshot;

// What a program's flat and global (BDA) loads can read, from the IR: each load's base address words
// (an immediate, a user data register or a flattened SRT slot) and a conservative byte interval of its
// offset. An offset with no bound covers every U32 value; an unknown base, a store or an atomic through
// an address leaves the plan incomplete, and the renderer then synchronizes every CPU-dirty buffer as
// before. Ported from the Demon's Souls fork (github.com/chenxiao07/KytyPS5, fba4c3d7).
struct BdaReadWord {
	enum class Kind : uint8_t { Immediate, UserData, Flat };
	Kind     kind  = Kind::Immediate;
	uint32_t value = 0;

	bool operator==(const BdaReadWord&) const = default;
};

struct BdaReadSpan {
	BdaReadWord low, high;
	int64_t     begin       = 0; // byte offsets from the base, exclusive end
	int64_t     end         = 0;
	bool        scalar_base = false;

	bool operator==(const BdaReadSpan&) const = default;
};

// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct BdaReadPlan {
	static constexpr uint32_t Capacity = 64;
	std::vector<BdaReadSpan>  spans;
	bool                      complete = false;

	bool operator==(const BdaReadPlan&) const = default;
};

[[nodiscard]] BdaReadPlan BuildBdaReadPlan(const Program& program);
// The guest ranges a draw's or dispatch's loads can read, merged and dword aligned. False (output
// untouched) when the plan is incomplete or a word or range does not evaluate.
[[nodiscard]] bool EvaluateBdaReadPlan(const BdaReadPlan& plan, const ResourceSnapshot& snapshot,
                                       std::vector<GuestRange>& output);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BDAREADPLAN_H_ */
