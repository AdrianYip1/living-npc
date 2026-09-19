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
	struct ShadowPass {
		VkRenderPass renderPass = VK_NULL_HANDLE;
		VkFramebuffer framebuffer = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		u32 mapSize = 0;
	};

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
		void setNPCAnimState(u32 slot, AnimState state);
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
		std::vector<AnimState> npcPrevAnimStates;
		std::vector<f32> npcAnimTimes;
		std::vector<f32> npcBlendTimers;
		static constexpr f32 BLEND_DURATION = 0.3f;
		Clock animClock;

		u32 currentFrame = 0;

		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;

		bool hasScene = false;

		static constexpr u32 SHADOW_MAP_SIZE = 2048;
		VkImage shadowMapImage = VK_NULL_HANDLE;
		VkDeviceMemory shadowMapMemory = VK_NULL_HANDLE;
		VkImageView shadowMapView = VK_NULL_HANDLE;
		VkSampler shadowMapSampler = VK_NULL_HANDLE;
		ShadowPass shadow;

		void createSyncObjects();
		void createTextures();
		void createCubemap();
		void createDescriptors();
		void createShadowResources();
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
