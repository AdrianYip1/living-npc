#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "device.hpp"

namespace HTN {
	class Pipeline {
	public:
		Pipeline(Device& _device);
		~Pipeline();
		Pipeline(const Pipeline&) = delete;
		Pipeline& operator=(const Pipeline&) = delete;

	private:
		Device& device;
	};
} // namespace HTN
