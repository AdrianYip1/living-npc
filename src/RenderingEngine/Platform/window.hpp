#pragma once
#define GLFW_INCLUDE_VULKAN

#include "../../defines.hpp"

struct GLFWwindow;

namespace HTN {
	class Window {
	public:
		Window(u32 w, u32 h, const char* title);
		~Window();
		Window(const Window& other) = delete;
		Window& operator=(const Window& other) = delete;

		GLFWwindow* getWindow();
		void initWindow();
		bool checkClose();
		void pollWindowEvents();

		bool getFramebufferResized() { return framebufferResized; }
		void setFramebufferResized(bool b);

	private:
		GLFWwindow* window = nullptr;
		u32 W;
		u32 H;
		const char* TITLE;
		bool framebufferResized = false;

		static void framebufferResizeCallback(GLFWwindow* window, int w, int h);
	};
} // namespace HTN
