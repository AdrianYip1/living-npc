#include "defines.hpp"
#include "clock.hpp"
#include "RenderingEngine/Platform/window.hpp"
#include "RenderingEngine/Core/camera.hpp"
#include "RenderingEngine/Core/input.hpp"
#include "RenderingEngine/Core/cameraControls.hpp"
#include "RenderingEngine/renderer.hpp"

#include <iostream>

int main() {
	try {
		HTN::Window window(1600, 900, "Living Npc: Face Model");
		HTN::Camera camera(enginemath::Vec3(0.0f, 1.56f, 0.5f),
						   1600.0f / 900.0f);
		HTN::Renderer renderer(window, camera);
		HTN::Clock clock;
		HTN::Input input(window);
		HTN::CameraControls controls(clock, window, input, camera);

		while (!window.checkClose()) {
			controls.accumulateMovement();
			controls.accumulateRotation();
			controls.updatePos();

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
