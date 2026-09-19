#pragma once
#include <vulkan/vulkan.h>
#include "enginemath/vec2.hpp"
#include "enginemath/vec3.hpp"

#include "../../defines.hpp"
#include "../device.hpp"
#include "../Helpers/buffers.hpp"

#include <vector>
#include <array>

namespace HTN {
	struct Vertex {
		enginemath::Vec3 position;
		enginemath::Vec3 color;
		enginemath::Vec3 normal;
		enginemath::Vec2 texCoord;

		static VkVertexInputBindingDescription getBindingDescription() {
			VkVertexInputBindingDescription bindingDescription{};
			bindingDescription.binding = 0;
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
			bindingDescription.stride = sizeof(Vertex);
			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
			std::array<VkVertexInputAttributeDescription, 4> attr{};

			attr[0].binding = 0;
			attr[0].location = 0;
			attr[0].format = VK_FORMAT_R32G32B32_SFLOAT;
			attr[0].offset = offsetof(Vertex, position);

			attr[1].binding = 0;
			attr[1].location = 1;
			attr[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			attr[1].offset = offsetof(Vertex, color);

			attr[2].binding = 0;
			attr[2].location = 2;
			attr[2].format = VK_FORMAT_R32G32B32_SFLOAT;
			attr[2].offset = offsetof(Vertex, normal);

			attr[3].binding = 0;
			attr[3].location = 3;
			attr[3].format = VK_FORMAT_R32G32_SFLOAT;
			attr[3].offset = offsetof(Vertex, texCoord);

			return attr;
		}
	};

	class Model {
	public:
		Model() = default;
		~Model();
		Model(const Model&) = delete;
		Model& operator=(const Model&) = delete;

		static bool createModel(Device& device, const std::vector<Vertex>& vertices,
								const std::vector<u32>& indices, Model* model);

		void bind(VkCommandBuffer commandBuffer);
		void draw(VkCommandBuffer commandBuffer);

		u32 getIndexCount() const { return indexCount; }

	private:
		Device* device = nullptr;
		u32 indexCount = 0;

		VkBuffer vertexBuffer = VK_NULL_HANDLE;
		VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
		VkBuffer indexBuffer = VK_NULL_HANDLE;
		VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

		void createVertexBuffer(const std::vector<Vertex>& vertices);
		void createIndexBuffer(const std::vector<u32>& indices);
	};
} // namespace HTN
