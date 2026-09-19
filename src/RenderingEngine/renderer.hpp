#pragma once
#include <vulkan/vulkan.h>

#include "../defines.hpp"
#include "../clock.hpp"
#include "device.hpp"
#include "pipeline.hpp"
#include "drawing.hpp"
#include "descriptor.hpp"
#include "Model/model.hpp"
#include "Model/uniform.hpp"
#include "Model/texture.hpp"
#include "Model/modelLoading.hpp"
#include "Platform/window.hpp"
#include "Core/camera.hpp"
#include "../ProceduralAnimation/faceAnimator.hpp"

#include <memory>
#include <map>

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
		Skeleton skeleton;

		std::unique_ptr<faceAnim> animator;

		VkDescriptorPool descriptorPool;
		VkDescriptorPool imguiPool;
		std::map<std::string, std::unique_ptr<Texture>> textures;
		std::map<std::string, std::vector<VkDescriptorSet>> materialSets;

		LightUBO light;
		std::vector<f32> faceWeights = std::vector<f32>(MAX_WEIGHTS, 0.0f);
		std::vector<enginemath::Mat4> inverseBindMatrices;
		Clock animClock;

		u32 currentFrame = 0;

		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;

		void createSyncObjects();
		void createTextures();
		void createDescriptors();
		void initImGui();
	};
} // namespace HTN
