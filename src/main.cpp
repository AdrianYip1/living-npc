#include "defines.hpp"
#include "RenderingEngine/Platform/window.hpp"
#include "RenderingEngine/renderer.hpp"

int main() {
	HTN::Window window(1600, 900, "Living Npc: Face Model");
	HTN::Renderer renderer(window);

	while (!window.checkClose()) {
		window.pollWindowEvents();
		renderer.drawFrame();
	}

	renderer.wait();

	return 0;
}
