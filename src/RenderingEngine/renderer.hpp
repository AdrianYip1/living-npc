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
#include "../npcConfig.hpp"
#include "Utility/raycast.hpp"

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

		const CollisionMesh& getCollision() const { return collisionMesh; }
		const GroundGrid& getGroundGrid() const { return groundGrid; }

		faceAnim& getFace(u32 slot) { return *faces[slot]; }
		u32 faceCount() const { return static_cast<u32>(faces.size()); }
		bool anyBusy() const;
		void setNPCTransform(u32 slot, const enginemath::Mat4& t) { npcTransforms[slot] = t; }
		void setNPCAnimState(u32 slot, AnimState state) { npcAnimStates[slot] = state; }
		const WorldBounds& getWorldBounds() const { return WORLD_BOUNDS; }

	private:
		Window& window;
		Camera& camera;
		Device device;
		Pipeline pipeline;
		Pipeline scenePipeline;
		Pipeline skyboxPipeline;
		Drawing drawing;
		Uniform uniform;
		Model model;
		Model sceneModel;
		Skeleton skeleton;

		std::vector<std::unique_ptr<faceAnim>> faces;

		VkDescriptorPool descriptorPool;
		VkDescriptorPool imguiPool;
		VkDescriptorPool skyboxPool = VK_NULL_HANDLE;
		std::map<std::string, std::unique_ptr<Texture>> textures;
		std::map<std::string, std::vector<VkDescriptorSet>> materialSets;
		std::vector<VkDescriptorSet> skyboxSets;

		std::vector<VkImage> cubemapImages;
		std::vector<VkDeviceMemory> cubemapMemories;
		std::vector<VkImageView> cubemapViews;
		std::vector<VkSampler> cubemapSamplers;

		CollisionMesh collisionMesh;
		GroundGrid groundGrid;
		LightUBO light;
		std::vector<std::vector<f32>> faceWeights;
		std::vector<enginemath::Mat4> npcTransforms;
		std::vector<enginemath::Mat4> inverseBindMatrices;
		std::vector<AnimState> npcAnimStates;
		std::vector<f32> npcAnimTimes;
		Clock animClock;

		u32 currentFrame = 0;

		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;

		bool hasScene = false;

		void createSyncObjects();
		void createTextures();
		void createCubemap();
		void createDescriptors();
		void initImGui();
		void loadScene();

		std::vector<std::array<const char*, 6>> cubemapTexturePaths{
			{"models/skybox/kloppenheim_06_px.png",
			 "models/skybox/kloppenheim_06_nx.png",
			 "models/skybox/kloppenheim_06_py.png",
			 "models/skybox/kloppenheim_06_ny.png",
			 "models/skybox/kloppenheim_06_pz.png",
			 "models/skybox/kloppenheim_06_nz.png"},
			{"models/skybox/belfast_sunset_px.png",
			 "models/skybox/belfast_sunset_nx.png",
			 "models/skybox/belfast_sunset_py.png",
			 "models/skybox/belfast_sunset_ny.png",
			 "models/skybox/belfast_sunset_pz.png",
			 "models/skybox/belfast_sunset_nz.png"},
			{"models/skybox/industrial_sunset_px.png",
			 "models/skybox/industrial_sunset_nx.png",
			 "models/skybox/industrial_sunset_py.png",
			 "models/skybox/industrial_sunset_ny.png",
			 "models/skybox/industrial_sunset_pz.png",
			 "models/skybox/industrial_sunset_nz.png"}
		};
	};
} // namespace HTN
