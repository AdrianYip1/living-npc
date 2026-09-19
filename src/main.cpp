#include "defines.hpp"
#include "RenderingEngine/Platform/window.hpp"
#include "RenderingEngine/renderer.hpp"

#include <iostream>

int main() {
	try {
		HTN::Window window(1600, 900, "Living Npc: Face Model");
		HTN::Renderer renderer(window);

		while (!window.checkClose()) {
			window.pollWindowEvents();
			renderer.drawFrame();
		}

		renderer.wait();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	return 0;
}
