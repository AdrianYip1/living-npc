#include "model.hpp"

HTN::Model::~Model() {
	if (!device) return;
	vkDestroyBuffer(device->getDevice(), indexBuffer, nullptr);
	vkFreeMemory(device->getDevice(), indexBufferMemory, nullptr);
	vkDestroyBuffer(device->getDevice(), vertexBuffer, nullptr);
	vkFreeMemory(device->getDevice(), vertexBufferMemory, nullptr);
	if (deltasBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(device->getDevice(), deltasBuffer, nullptr);
		vkFreeMemory(device->getDevice(), deltasBufferMemory, nullptr);
	}
}

bool HTN::Model::createModel(Device& _device, fModel _model, Model* _fmodel) {
	if (!_fmodel) return false;

	_fmodel->device = &_device;
	_fmodel->model = std::move(_model);

	_fmodel->createVertexBuffer();
	_fmodel->createIndexBuffer();
	if (!_fmodel->model.deltas.empty()) _fmodel->createDeltasBuffer();

	return true;
}

void HTN::Model::bind(VkCommandBuffer commandBuffer) {
	VkBuffer vertexBuffers[] = { vertexBuffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
}

void HTN::Model::draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout,
	VkDescriptorSet descriptorSet, u32 weightBase, u32 jointBase) {
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
		0, 1, &descriptorSet, 0, nullptr);

	for (const submesh& s : model.primitives) {
		MorphPush push{};
		push.model = enginemath::Mat4::identity();
		push.morphStartIndex = s.morphStartIndex;
		push.targetCount = s.targetCount;
		push.vertexOffset = s.vertexOffset;
		push.vertexCount = s.vertexCount;
		push.weightsStartIndex = s.weightsStartIndex;
		push.weightBase = weightBase;
		push.isSkinned = s.isSkinned;
		push.jointBase = jointBase;
		vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
		vkCmdDrawIndexed(commandBuffer, s.indexCount, 1, s.indexStart, 0, 0);
	}
}

void HTN::Model::createVertexBuffer() {
	VkDeviceSize bufferSize = sizeof(Vertex) * model.vertices.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	Buffer::createBuffer(*device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
						 stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(device->getDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, model.vertices.data(), bufferSize);
	vkUnmapMemory(device->getDevice(), stagingBufferMemory);

	Buffer::createBuffer(*device, bufferSize,
						 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
						 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer,
						 vertexBufferMemory);

	Buffer::copyBuffer(*device, stagingBuffer, vertexBuffer, bufferSize);

	vkDestroyBuffer(device->getDevice(), stagingBuffer, nullptr);
	vkFreeMemory(device->getDevice(), stagingBufferMemory, nullptr);
}

void HTN::Model::createIndexBuffer() {
	VkDeviceSize bufferSize = sizeof(u32) * model.indices.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	Buffer::createBuffer(*device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
						 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
						 stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(device->getDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, model.indices.data(), bufferSize);
	vkUnmapMemory(device->getDevice(), stagingBufferMemory);

	Buffer::createBuffer(*device, bufferSize,
						 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
						 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer,
						 indexBufferMemory);

	Buffer::copyBuffer(*device, stagingBuffer, indexBuffer, bufferSize);

	vkDestroyBuffer(device->getDevice(), stagingBuffer, nullptr);
	vkFreeMemory(device->getDevice(), stagingBufferMemory, nullptr);
}

void HTN::Model::createDeltasBuffer() {
	VkDeviceSize bufferSize = sizeof(enginemath::Vec4) * model.deltas.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	Buffer::createBuffer(*device, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		stagingBuffer, stagingBufferMemory);

	void* data;
	vkMapMemory(device->getDevice(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, model.deltas.data(), bufferSize);
	vkUnmapMemory(device->getDevice(), stagingBufferMemory);

	Buffer::createBuffer(*device, bufferSize,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, deltasBuffer,
		deltasBufferMemory);

	Buffer::copyBuffer(*device, stagingBuffer, deltasBuffer, bufferSize);

	vkDestroyBuffer(device->getDevice(), stagingBuffer, nullptr);
	vkFreeMemory(device->getDevice(), stagingBufferMemory, nullptr);
}
