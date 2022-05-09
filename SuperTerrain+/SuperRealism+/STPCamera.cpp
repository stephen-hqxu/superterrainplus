#include <SuperRealism+/Utility/Camera/STPCamera.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//Camera Calculation
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>

using glm::dvec2;
using glm::dvec3;
using glm::dmat4;

using glm::normalize;
using glm::radians;
using glm::cross;

using SuperTerrainPlus::STPMatrix4x4d;
using namespace SuperTerrainPlus::STPRealism;

STPCamera::STPCamera(const STPEnvironment::STPCameraSetting& props, STPProjectionCategory proj_type) : 
	Camera(props), View(glm::identity<dmat4>()), ViewOutdated(true), ProjectionType(proj_type) {
	if (!this->Camera.validate()) {
		throw STPException::STPInvalidEnvironment("Camera setting not validated");
	}

	//calculate the initial values for the camera based on the default settings.
	this->updateViewSpace();
}

void STPCamera::updateViewSpace() {
	//calculate front based on the orientation
	const double cos_pitch = glm::cos(this->Camera.Pitch);
	this->Front = normalize(dvec3(
		glm::cos(this->Camera.Yaw) * cos_pitch,
		glm::sin(this->Camera.Pitch),
		glm::sin(this->Camera.Yaw) * cos_pitch
	));
	//update right and up
	//normalise the right vector because their length gets closer to 0 the more you look up or down which results in slower movement.
	this->Right = normalize(cross(this->Front, this->Camera.WorldUp));
	this->Up = normalize(cross(this->Right, this->Front));
}

inline auto STPCamera::findListener(STPStatusChangeCallback* listener) const {
	return std::find(this->CallbackRegistry.cbegin(), this->CallbackRegistry.cend(), listener);
}

void STPCamera::registerListener(STPStatusChangeCallback* listener) const {
	//make sure the same listener is not registered twice.
	if (this->findListener(listener) != this->CallbackRegistry.cend()) {
		//the same instance is found
		throw STPException::STPMemoryError("The same listener instance has been registered with this camera previously");
	}

	//ok, add
	this->CallbackRegistry.emplace_back(listener);
}

void STPCamera::removeListener(STPStatusChangeCallback* listener) const {
	//try to find this instance
	const auto it = this->findListener(listener);
	if (it == this->CallbackRegistry.cend()) {
		//not found
		throw STPException::STPMemoryError("This listener is not previously registered");
	}

	//found, remove it
	this->CallbackRegistry.erase(it);
}

const STPMatrix4x4d& STPCamera::view() const {
	if (this->ViewOutdated) {
		//update view matrix
		alignas(STPMatrix4x4d) const dmat4 view_data = glm::lookAt(
			this->Camera.Position,
			this->Camera.Position + this->Front,
			this->Up
		);
		this->View = STPMatrix4x4d(view_data);
		this->ViewOutdated = false;
	}

	return this->View;
}

const SuperTerrainPlus::STPEnvironment::STPCameraSetting& STPCamera::cameraStatus() const {
	return this->Camera;
}

void STPCamera::move(const STPMoveDirection direction, double delta) {
	//scale the movement speed with delta, delta usually is the frametime
	const double velocity = this->Camera.MovementSpeed * delta;

	switch (direction) {
	case STPMoveDirection::Forward: this->Camera.Position += this->Front * velocity;
		break;
	case STPMoveDirection::Backward: this->Camera.Position -= this->Front * velocity;
		break;
	case STPMoveDirection::Left: this->Camera.Position -= this->Right * velocity;
		break;
	case STPMoveDirection::Right: this->Camera.Position += this->Right * velocity;
		break;
	case STPMoveDirection::Up: this->Camera.Position += this->Camera.WorldUp * velocity;
		break;
	case STPMoveDirection::Down: this->Camera.Position -= this->Camera.WorldUp * velocity;
		break;
	default:
		//impossible
		break;
	}

	//trigger update the view matrix
	this->ViewOutdated = true;
	//trigger callback, if applicable
	for (auto callback : this->CallbackRegistry) {
		callback->onMove(*this);
	}
}

void STPCamera::rotate(const dvec2& offset) {
	static constexpr double YAW_LIM = radians(360.0), PITCH_LIM = radians(89.0);
	static constexpr auto modulof = [](double val, double bound) constexpr -> double {
		//a floating point modulo function
		if (val >= bound || val <= -bound) {
			return val - bound;
		}
		return val;
	};

	//using sensitivity to scale the offset
	const dvec2 rotateAmount = offset * this->Camera.RotationSensitivity;
	this->Camera.Yaw += rotateAmount.x;
	//modulo the angle
	this->Camera.Yaw = modulof(this->Camera.Yaw, YAW_LIM);
	//same for the pitch
	this->Camera.Pitch += rotateAmount.y;
	//does not allow pitch to go over vertical (so flip the camera over)
	this->Camera.Pitch = glm::clamp(this->Camera.Pitch, -PITCH_LIM, PITCH_LIM);

	//update camera front, right and up
	this->updateViewSpace();

	this->ViewOutdated = true;
	for (auto callback : this->CallbackRegistry) {
		callback->onRotate(*this);
	}
}