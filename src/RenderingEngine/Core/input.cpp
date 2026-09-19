#include "input.hpp"
#include <GLFW/glfw3.h>

HTN::Input::Input(Window& _window) : window(_window) {
}

bool HTN::Input::keyPressed(int key) {
	return glfwGetKey(window.getWindow(), key) == GLFW_PRESS;
}

bool HTN::Input::mousePressed(int button) {
	return glfwGetMouseButton(window.getWindow(), button) == GLFW_PRESS;
}
