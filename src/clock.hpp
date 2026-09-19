#pragma once

#include "defines.hpp"
#include <chrono>

namespace HTN {
	class Clock {
	public:
		void resetTime() {
			start = std::chrono::steady_clock::now();
		}
		u32 elapsedMs() {
			auto now = std::chrono::steady_clock::now();
			return (u32)std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
		}
	private:
		std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	};
} // namespace HTN
