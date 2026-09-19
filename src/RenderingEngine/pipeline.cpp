#include "pipeline.hpp"
#include "Model/model.hpp"

HTN::Pipeline::Pipeline(Device& _device, const std::string& _vertPath, const std::string& _fragPath) :
	device(_device), vertPath(_vertPath), fragPath(_fragPath) {
	createRenderPass();
	createGraphicsPipeline();
}

HTN::Pipeline::~Pipeline() {
	vkDestroyDescriptorSetLayout(device.getDevice(), uboSetLayout, nullptr);
	vkDestroyPipeline(device.getDevice(), graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(device.getDevice(), pipelineLayout, nullptr);
	vkDestroyRenderPass(device.getDevice(), renderpass, nullptr);
}

void HTN::Pipeline::createRenderPass() {
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = device.getSwapchainFormat();
	colorAttachment.samples = device.getMSAASampleCount();
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depthAttachment{};
	depthAttachment.format = device.findDepthFormat();
	depthAttachment.samples = device.getMSAASampleCount();
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription colorResolveAttachment{};
	colorResolveAttachment.format = device.getSwapchainFormat();
	colorResolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorResolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorResolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorResolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorResolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorResolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorResolveAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentRef{};
	depthAttachmentRef.attachment = 1;
	depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorResolveRef{};
	colorResolveRef.attachment = 2;
	colorResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;
	subpass.pDepthStencilAttachment = &depthAttachmentRef;
	subpass.pResolveAttachments = &colorResolveRef;

	std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorResolveAttachment };

	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
							  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
							  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
							   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo renderpassCreateInfo{};
	renderpassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderpassCreateInfo.attachmentCount = static_cast<u32>(attachments.size());
	renderpassCreateInfo.pAttachments = attachments.data();
	renderpassCreateInfo.subpassCount = 1;
	renderpassCreateInfo.pSubpasses = &subpass;
	renderpassCreateInfo.dependencyCount = 1;
	renderpassCreateInfo.pDependencies = &dependency;

	if (vkCreateRenderPass(device.getDevice(), &renderpassCreateInfo, nullptr, &renderpass) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create render pass");
	}
}

void HTN::Pipeline::createGraphicsPipeline() {
	auto vertShader = readFile(vertPath);
	auto fragShader = readFile(fragPath);

	VkShaderModule vertModule = createShaderModule(vertShader);
	VkShaderModule fragModule = createShaderModule(fragShader);

	std::array<VkPipelineShaderStageCreateInfo, 2> shaderCreateInfo{};
	shaderCreateInfo[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderCreateInfo[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderCreateInfo[0].module = vertModule;
	shaderCreateInfo[0].pName = "main";

	shaderCreateInfo[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderCreateInfo[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderCreateInfo[1].module = fragModule;
	shaderCreateInfo[1].pName = "main";

	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescriptions();

	VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{};
	vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
	vertexInputCreateInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputCreateInfo.vertexAttributeDescriptionCount = static_cast<u32>(attributeDescriptions.size());
	vertexInputCreateInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo{};
	inputAssemblyCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssemblyCreateInfo.primitiveRestartEnable = VK_FALSE;

	std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamicCreateInfo{};
	dynamicCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicCreateInfo.dynamicStateCount = static_cast<u32>(dynamicStates.size());
	dynamicCreateInfo.pDynamicStates = dynamicStates.data();

	VkPipelineViewportStateCreateInfo viewportCreateInfo{};
	viewportCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportCreateInfo.scissorCount = 1;
	viewportCreateInfo.viewportCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizationCreateInfo{};
	rasterizationCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationCreateInfo.depthClampEnable = VK_FALSE;
	rasterizationCreateInfo.rasterizerDiscardEnable = VK_FALSE;
	rasterizationCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationCreateInfo.lineWidth = 1.0f;
	rasterizationCreateInfo.cullMode = VK_CULL_MODE_NONE;
	rasterizationCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationCreateInfo.depthBiasEnable = VK_FALSE;

	VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo{};
	depthStencilCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilCreateInfo.depthTestEnable = VK_TRUE;
	depthStencilCreateInfo.depthWriteEnable = VK_TRUE;
	depthStencilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencilCreateInfo.depthBoundsTestEnable = VK_FALSE;
	depthStencilCreateInfo.stencilTestEnable = VK_FALSE;

	VkPipelineMultisampleStateCreateInfo multisamplingCreateInfo{};
	multisamplingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisamplingCreateInfo.rasterizationSamples = device.getMSAASampleCount();
	multisamplingCreateInfo.sampleShadingEnable = VK_FALSE;

	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
										  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo colorBlendingCreateInfo{};
	colorBlendingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendingCreateInfo.logicOpEnable = VK_FALSE;
	colorBlendingCreateInfo.attachmentCount = 1;
	colorBlendingCreateInfo.pAttachments = &colorBlendAttachment;

	Descriptor::createDescriptorSetLayout(device,
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
		{VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT},
		{0, 1},
		uboSetLayout);

	VkPipelineLayoutCreateInfo layoutCreateInfo{};
	layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutCreateInfo.setLayoutCount = 1;
	layoutCreateInfo.pSetLayouts = &uboSetLayout;
	layoutCreateInfo.pushConstantRangeCount = 0;

	if (vkCreatePipelineLayout(device.getDevice(), &layoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create pipeline layout");
	}

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderCreateInfo.data();
	pipelineInfo.pVertexInputState = &vertexInputCreateInfo;
	pipelineInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
	pipelineInfo.pViewportState = &viewportCreateInfo;
	pipelineInfo.pRasterizationState = &rasterizationCreateInfo;
	pipelineInfo.pMultisampleState = &multisamplingCreateInfo;
	pipelineInfo.pDepthStencilState = &depthStencilCreateInfo;
	pipelineInfo.pColorBlendState = &colorBlendingCreateInfo;
	pipelineInfo.pDynamicState = &dynamicCreateInfo;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = renderpass;
	pipelineInfo.subpass = 0;

	if (vkCreateGraphicsPipelines(device.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create graphics pipeline");
	}

	vkDestroyShaderModule(device.getDevice(), vertModule, nullptr);
	vkDestroyShaderModule(device.getDevice(), fragModule, nullptr);
}

std::vector<char> HTN::Pipeline::readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("ERROR: Failed to open shader file");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();

	return buffer;
}

VkShaderModule HTN::Pipeline::createShaderModule(const std::vector<char>& shaderCode) {
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = shaderCode.size();
	createInfo.pCode = reinterpret_cast<const u32*>(shaderCode.data());

	VkShaderModule module;
	if (vkCreateShaderModule(device.getDevice(), &createInfo, nullptr, &module) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: Failed to create shader module");
	}

	return module;
}
