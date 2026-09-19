#include "cameraControls.hpp"

HTN::CameraControls::CameraControls(Clock& _clock, Window& _window, Input& _input, Camera& _camera) :
	clock(_clock), window(_window), input(_input), camera(_camera) {
	glfwSetInputMode(window.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	if (glfwRawMouseMotionSupported()) {
		glfwSetInputMode(window.getWindow(), GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
	}
	glfwGetCursorPos(window.getWindow(), &prevX, &prevY);
	prevTime = (f32)clock.elapsedMs();
}

void HTN::CameraControls::accumulateMovement() {
	movementTotal = enginemath::Vec3(0.0f);

	if (input.keyPressed(GLFW_KEY_W)) {
		movementTotal += camera.getFlatForward();
	}
	if (input.keyPressed(GLFW_KEY_A)) {
		movementTotal -= camera.getFlatRight();
	}
	if (input.keyPressed(GLFW_KEY_S)) {
		movementTotal -= camera.getFlatForward();
	}
	if (input.keyPressed(GLFW_KEY_D)) {
		movementTotal += camera.getFlatRight();
	}

	if (!movementTotal.basicallyZero()) movementTotal.normalize();
}

bool HTN::CameraControls::canLookAround() {
	return input.mousePressed(GLFW_MOUSE_BUTTON_RIGHT);
}

void HTN::CameraControls::accumulateRotation() {
	glfwGetCursorPos(window.getWindow(), &xPos, &yPos);

	if (canLookAround()) {
		f32 dX = (f32)(xPos - prevX);
		f32 dY = (f32)(yPos - prevY);

		smoothX = smoothX + (dX - smoothX) * smoothFactor;
		smoothY = smoothY + (dY - smoothY) * smoothFactor;

		camera.lookAround(smoothX, smoothY);
	}
	else {
		smoothX = 0.0f;
		smoothY = 0.0f;
	}

	prevX = xPos;
	prevY = yPos;
}

void HTN::CameraControls::updatePos() {
	dt = (f32)clock.elapsedMs() - prevTime;
	prevTime = (f32)clock.elapsedMs();

	if (movementTotal != enginemath::Vec3(0.0f)) {
		camera.move(cameraSpeed * movementTotal * dt);
	}
}
