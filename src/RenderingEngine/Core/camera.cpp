#include "camera.hpp"

HTN::Camera::Camera(enginemath::Vec3 _pos, f32 _aspect, f32 _yaw,
					f32 _pitch, f32 _fov, f32 _near, f32 _far) :
	pos(_pos), aspect(_aspect), yaw(_yaw), pitch(_pitch), fov(_fov), near(_near), far(_far) {
}

enginemath::Vec3 HTN::Camera::getUp() {
	return up;
}

enginemath::Vec3 HTN::Camera::getForward() {
	return enginemath::Vec3(
		std::cos(yaw),
		std::sin(pitch),
		std::sin(yaw) * std::cos(pitch)
	);
}

enginemath::Vec3 HTN::Camera::getFlatForward() {
	return enginemath::Vec3(std::cos(yaw), 0.0f, std::sin(yaw)).normalized();
}

enginemath::Vec3 HTN::Camera::getRight() {
	return getForward().cross(getUp()).normalized();
}

enginemath::Vec3 HTN::Camera::getFlatRight() {
	return getFlatForward().cross(getUp()).normalized();
}

enginemath::Mat4 HTN::Camera::getProj() {
	enginemath::Mat4 proj = enginemath::Mat4::projectionM(fov, aspect, near, far);
	proj.m[1][1] *= -1;
	return proj;
}

enginemath::Mat4 HTN::Camera::getView() {
	return enginemath::Mat4::lookAtM(pos, pos + getForward(), getUp());
}

void HTN::Camera::setPos(enginemath::Vec3 _pos) {
	pos = _pos;
}

void HTN::Camera::lookAround(f32 dY, f32 dP) {
	f32 dPitch = dP * rotateSpeed;
	pitch = std::clamp(pitch - dPitch, enginemath::toRad(-89.0f), enginemath::toRad(89.0f));
	yaw += rotateSpeed * dY;

	if (yaw >= 2 * enginemath::PI) yaw -= 2 * enginemath::PI;
	if (yaw <= -2 * enginemath::PI) yaw += 2 * enginemath::PI;
}

void HTN::Camera::move(enginemath::Vec3 dP) {
	pos += dP;
}

void HTN::Camera::setAspect(f32 newAspect) {
	aspect = newAspect;
}
