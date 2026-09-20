#pragma once

#include "../Platform/window.hpp"
#include "camera.hpp"
#include "input.hpp"
#include "enginemath/mat4.hpp"
#include "enginemath/vec3.hpp"
#include "../../clock.hpp"

namespace HTN {
	class CameraControls {
	public:
		CameraControls(Clock& _clock, Window& _window, Input& _input, Camera& _camera);
		~CameraControls() = default;
		CameraControls(const CameraControls&) = delete;
		CameraControls& operator=(const CameraControls&) = delete;

		void accumulateMovement();
		void accumulateRotation();
		bool canLookAround();
		void updatePos();

	private:
		Clock& clock;
		Window& window;
		Input& input;
		Camera& camera;

		f32 cameraSpeed = 0.005f;
		f32 smoothX = 0.0f, smoothY = 0.0f;
		f64 xPos = 0.0f, yPos = 0.0f;
		f32 smoothFactor = 0.3f;

		f64 prevX;
		f64 prevY;
		f32 prevTime;
		f32 dt;
		enginemath::Vec3 movementTotal;
	};
} // namespace HTN
