#pragma once

// std
#include <cstdint>
#include <string>
#include <cstddef>

// Debug for validation layers
#ifdef NDEBUG
#define HTN_RELEASE 1
#define HTN_DEBUG 0
#else
#define HTN_DEBUG 1
#define HTN_RELEASE 0
#endif

namespace HTN {
	using size = size_t;

	using u8 = uint8_t;
	using u16 = uint16_t;
	using u32 = uint32_t;
	using u64 = uint64_t;

	using i8 = int8_t;
	using i16 = int16_t;
	using i32 = int32_t;
	using i64 = int64_t;

	using f32 = float;
	using f64 = double;

} // namespace HTN
