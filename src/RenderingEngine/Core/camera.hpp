#pragma once
#include <GLFW/glfw3.h>
#include <enginemath/mat4.hpp>
#include <enginemath/vec3.hpp>
#include <enginemath/mathutils.hpp>

#include "../../defines.hpp"

#include <cmath>
#include <algorithm>

namespace HTN {
	class Camera {
	public:
		Camera(enginemath::Vec3 _pos, f32 _aspect, f32 _yaw = enginemath::toRad(-90.0f),
			   f32 _pitch = 0.0f, f32 _fov = enginemath::toRad(90.0f),
			   f32 _near = 0.1f, f32 _far = 10000.0f);
		~Camera() = default;
		Camera(const Camera&) = delete;
		Camera& operator=(const Camera&) = delete;

		enginemath::Vec3 getUp();
		enginemath::Vec3 getForward();
		enginemath::Vec3 getFlatForward();
		enginemath::Vec3 getRight();
		enginemath::Vec3 getFlatRight();

		enginemath::Mat4 getProj();
		enginemath::Mat4 getView();

		enginemath::Vec3 getPos() { return pos; }

		void setPos(enginemath::Vec3 _pos);
		void lookAround(f32 dY, f32 dP);
		void move(enginemath::Vec3 dP);

		void setAspect(f32 newAspect);

	private:
		enginemath::Vec3 pos;
		f32 aspect;
		f32 yaw;
		f32 pitch;
		f32 fov;
		f32 near;
		f32 far;

		enginemath::Vec3 up = enginemath::Vec3(0.0f, 1.0f, 0.0f);
		f32 rotateSpeed = 0.002f;
	};
} // namespace HTN
