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

	private:
		GLFWwindow* window = nullptr;
		u32 W;
		u32 H;
		const char* TITLE;
	};
} // namespace HTN
