#include <SuperRealism+/Utility/Camera/STPCamera.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//Camera Calculation
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

using glm::vec2;
using glm::vec3;
using glm::mat4;

using glm::normalize;
using glm::radians;
using glm::cross;

using namespace SuperTerrainPlus::STPRealism;

STPCamera::STPCamera(const STPEnvironment::STPCameraSetting& props) : 
	Camera(props), View(mat4(0.0f)), ViewOutdated(true) {
	if (!this->Camera.validate()) {
		throw STPException::STPInvalidEnvironment("Camera setting not validated");
	}

	//calculate the initial values for the camera based on the default settings.
	this->updateViewSpace();
}

void STPCamera::updateViewSpace() {
	//calculate front based on the orientation
	const float cos_pitch = glm::cos(this->Camera.Pitch);
	this->Front = normalize(vec3(
		glm::cos(this->Camera.Yaw) * cos_pitch,
		glm::sin(this->Camera.Pitch),
		glm::sin(this->Camera.Yaw) * cos_pitch
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

const SuperTerrainPlus::STPEnvironment::STPCameraSetting& STPCamera::cameraStatus() const {
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
	case STPMoveDirection::Up: this->Camera.Position += this->Camera.WorldUp * velocity;
		break;
	case STPMoveDirection::Down: this->Camera.Position -= this->Camera.WorldUp * velocity;
		break;
	default:
		//impossible
		break;
	}

	//update the view matrix
	this->ViewOutdated = true;
}

void STPCamera::rotate(const vec2& offset) {
	static constexpr float YAW_LIM = radians(360.0f), PITCH_LIM = radians(89.0f);
	static auto modulof = [](float val, float bound) constexpr -> float {
		//a float modulo function
		if (val >= bound || val <= -bound) {
			return val - bound;
		}
		return val;
	};

	//using sensitivity to scale the offset
	const vec2 rotateAmount = offset * this->Camera.RotationSensitivity;
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
}