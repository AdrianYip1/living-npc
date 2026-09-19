#include "model.hpp"

HTN::Model::~Model() {
	if (!device) return;
	vkDestroyBuffer(device->getDevice(), indexBuffer, nullptr);
	vkFreeMemory(device->getDevice(), indexBufferMemory, nullptr);
	vkDestroyBuffer(device->getDevice(), vertexBuffer, nullptr);
	vkFreeMemory(device->getDevice(), vertexBufferMemory, nullptr);
}

bool HTN::Model::createModel(Device& _device, const std::vector<Vertex>& vertices,
							 const std::vector<u32>& indices, Model* _model) {
	if (!_model) return false;

	_model->device = &_device;
	_model->indexCount = static_cast<u32>(indices.size());
	_model->createVertexBuffer(vertices);
	_model->createIndexBuffer(indices);

	return true;
}

void HTN::Model::bind(VkCommandBuffer commandBuffer) {
	VkBuffer vertexBuffers[] = { vertexBuffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
}

void HTN::Model::draw(VkCommandBuffer commandBuffer) {
	vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
}

void HTN::Model::createVertexBuffer(const std::vector<Vertex>& vertices) {
	VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	Buffer::createBuffer(*device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
						 stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(device->getDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, vertices.data(), bufferSize);
	vkUnmapMemory(device->getDevice(), stagingBufferMemory);

	Buffer::createBuffer(*device, bufferSize,
						 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
						 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer,
						 vertexBufferMemory);

	Buffer::copyBuffer(*device, stagingBuffer, vertexBuffer, bufferSize);

	vkDestroyBuffer(device->getDevice(), stagingBuffer, nullptr);
	vkFreeMemory(device->getDevice(), stagingBufferMemory, nullptr);
}

void HTN::Model::createIndexBuffer(const std::vector<u32>& indices) {
	VkDeviceSize bufferSize = sizeof(u32) * indices.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	Buffer::createBuffer(*device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
						 stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(device->getDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, indices.data(), bufferSize);
	vkUnmapMemory(device->getDevice(), stagingBufferMemory);

	Buffer::createBuffer(*device, bufferSize,
						 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
						 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer,
						 indexBufferMemory);

	Buffer::copyBuffer(*device, stagingBuffer, indexBuffer, bufferSize);

	vkDestroyBuffer(device->getDevice(), stagingBuffer, nullptr);
	vkFreeMemory(device->getDevice(), stagingBufferMemory, nullptr);
}
