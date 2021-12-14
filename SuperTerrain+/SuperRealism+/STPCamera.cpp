#include <SuperRealism+/Utility/Camera/STPCamera.h>

//Camera Calculation
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

using glm::vec2;
using glm::vec3;
using glm::mat4;

using glm::normalize;
using glm::radians;
using glm::cross;

using namespace SuperTerrainPlus::STPRealism;

STPCamera::STPCameraProperty::STPCameraProperty() : 
	Yaw(radians(-90.0f)), Pitch(0.0f),
	MovementSpeed(2.5f), RotationSensitivity(0.1f),
	Position(vec3(0.0f)), WorldUp(0.0f, 1.0f, 0.0f) {

}

STPCamera::STPCamera(const STPCameraProperty& props) : Camera(props), LastRotateOffset(vec2(0.0f)), View(mat4(0.0f)), ViewOutdated(true) {
	//calculate the initial values for the camera based on the default settings.
	this->updateViewSpace();
}

void STPCamera::updateViewSpace() {
	//calculate front based on the orientation
	this->Front = normalize(vec3(
		glm::cos(this->Camera.Yaw) * glm::cos(this->Camera.Pitch),
		glm::sin(this->Camera.Pitch),
		glm::sin(this->Camera.Yaw) * glm::cos(this->Camera.Pitch)
	));
	//update right and up
	//normalise the right vector because their length gets closer to 0 the more you look up or down which results in slower movement.
	this->Right = normalize(cross(this->Front, this->Camera.WorldUp));
	this->Up = normalize(cross(this->Right, this->Front));
}

STPCamera::STPMatrixResult STPCamera::view() const {
	bool same = true;
	if (this->ViewOutdated) {
		//update view matrix
		this->View = glm::lookAt(
			this->Camera.Position,
			this->Camera.Position + this->Front,
			this->Up
		);
		this->ViewOutdated = false;
		same = false;
	}

	return STPMatrixResult(&this->View, same);
}

const STPCamera::STPCameraProperty& STPCamera::cameraStatus() const {
	return this->Camera;
}

void STPCamera::move(const STPMoveDirection direction, float delta) {
	//scale the movement speed with delta, delta usually is the frametime
	const float velocity = this->Camera.MovementSpeed * delta;

	switch (direction) {
	case STPMoveDirection::Forward: this->Camera.Position += this->Front * velocity;
		break;
	case STPMoveDirection::Backward: this->Camera.Position -= this->Front * velocity;
		break;
	case STPMoveDirection::Left: this->Camera.Position -= this->Right * velocity;
		break;
	case STPMoveDirection::Right: this->Camera.Position += this->Right * velocity;
		break;
	case STPMoveDirection::Up: this->Camera.Position += this->Up * velocity;
		break;
	case STPMoveDirection::Down: this->Camera.Position -= this->Up * velocity;
		break;
	default:
		//impossible
		break;
	}

	//update the view matrix
	this->ViewOutdated = true;
}

void STPCamera::rotate(vec2 offset) {
	static constexpr double PI = glm::pi<double>(), PI_BY_2 = PI * 0.5;
	static auto modulof = [](float val, float bound) constexpr -> float {
		//a float modulo function
		if (val > bound) {
			return val - bound;
		}
		if (val < -bound) {
			return val + bound;
		}
		return val;
	};

	//we reverse Y since Y goes from bottom to top (from negative axis to positive)
	//using sensitivity to scale the offset
	const vec2 rotateAmount = vec2(offset.x - this->LastRotateOffset.x, this->LastRotateOffset.y - offset.y) * this->Camera.RotationSensitivity;
	this->Camera.Yaw += rotateAmount.x;
	//modulo the angle
	this->Camera.Yaw = modulof(this->Camera.Yaw, PI);
	//same for the pitch
	this->Camera.Pitch += rotateAmount.y;
	this->Camera.Pitch = modulof(this->Camera.Pitch, PI_BY_2);

	//update camera front, right and up
	this->updateViewSpace();

	//update last position
	this->LastRotateOffset = offset;

	this->ViewOutdated = true;
}