#include "window.hpp"
#include <GLFW/glfw3.h>

#include <stdexcept>

HTN::Window::Window(u32 w, u32 h, const char* title) : W(w), H(h), TITLE(title) {
	initWindow();
}

HTN::Window::~Window() {
	glfwDestroyWindow(window);
	glfwTerminate();
}

GLFWwindow* HTN::Window::getWindow() {
	return window;
}

void HTN::Window::initWindow() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(W, H, TITLE, nullptr, nullptr);

	if (!window) {
		throw std::runtime_error("ERROR: Failed to create window");
	}

	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

bool HTN::Window::checkClose() {
	return glfwWindowShouldClose(window);
}

void HTN::Window::pollWindowEvents() {
	glfwPollEvents();
}

void HTN::Window::setFramebufferResized(bool b) {
	framebufferResized = b;
}

void HTN::Window::framebufferResizeCallback(GLFWwindow* window, int w, int h) {
	auto app = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}
