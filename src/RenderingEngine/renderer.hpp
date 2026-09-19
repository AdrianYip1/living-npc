#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "device.hpp"
#include "pipeline.hpp"
#include "drawing.hpp"
#include "descriptor.hpp"
#include "Model/model.hpp"
#include "Model/uniform.hpp"
#include "Platform/window.hpp"
#include "Core/camera.hpp"

namespace HTN {
	class Renderer {
	public:
		Renderer(Window& _window, Camera& _camera);
		~Renderer();
		Renderer(const Renderer&) = delete;
		Renderer& operator=(const Renderer&) = delete;

		void drawFrame();
		void wait();

	private:
		Window& window;
		Camera& camera;
		Device device;
		Pipeline pipeline;
		Drawing drawing;
		Uniform uniform;
		Model model;

		VkDescriptorPool descriptorPool;
		VkDescriptorPool imguiPool;
		std::vector<VkDescriptorSet> descriptorSets;

		LightUBO light;
		std::vector<f32> faceWeights = std::vector<f32>(MAX_WEIGHTS, 0.0f);

		u32 currentFrame = 0;

		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;

		void createSyncObjects();
		void createDescriptors();
		void initImGui();
	};
} // namespace HTN
