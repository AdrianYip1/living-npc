#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "device.hpp"
#include "pipeline.hpp"
#include "drawing.hpp"
#include "Platform/window.hpp"

namespace HTN {
	class Renderer {
	public:
		Renderer(Window& _window);
		~Renderer();
		Renderer(const Renderer&) = delete;
		Renderer& operator=(const Renderer&) = delete;

		void drawFrame();
		void wait();

	private:
		Window& window;
		Device device;
		Pipeline pipeline;
		Drawing drawing;

		u32 currentFrame = 0;

		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;

		void createSyncObjects();
	};
} // namespace HTN
