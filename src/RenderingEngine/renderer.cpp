#include "renderer.hpp"

HTN::Renderer::Renderer(Window& _window) :
	window(_window),
	device(_window),
	pipeline(device, "src/shaders/shader.vert.spv", "src/shaders/shader.frag.spv"),
	drawing(device, pipeline) {
}

HTN::Renderer::~Renderer() {
}

void HTN::Renderer::drawFrame() {
}

void HTN::Renderer::wait() {
}

void HTN::Renderer::createSyncObjects() {
}
