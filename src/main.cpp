#include "defines.hpp"
#include "RenderingEngine/Platform/window.hpp"

int main() {
	HTN::Window window(1600, 900, "Living Npc: Face Model");

	while (!window.checkClose()) {
		window.pollWindowEvents();
	}

	return 0;
}
