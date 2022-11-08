#include <SuperRealism+/Utility/STPCamera.h>

#include <glad/glad.h>

//Error
#include <SuperTerrain+/Exception/STPMemoryError.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//Camera Calculation
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cstring>

//GLM
#include <glm/mat3x4.hpp>
#include <glm/mat4x4.hpp>

using glm::vec2;
using glm::vec3;
using glm::mat4;
using glm::mat3x4;
using glm::dvec2;
using glm::dvec3;
using glm::dmat4;

using glm::normalize;
using glm::radians;
using glm::cross;

using SuperTerrainPlus::STPMatrix4x4d;
using namespace SuperTerrainPlus::STPRealism;

struct STPCamera::STPPackedCameraBuffer {
public:

	vec3 Pos;
	float _padPos;

	mat4 V;
	mat3x4 VNorm;

	mat4 P, InvP, PVr, PV, InvPV;

	vec2 LDFac;
	float Far;

};

bool STPCamera::STPSubscriberBenefit::any() const noexcept {
	return this->Moved || this->Rotated || this->Zoomed || this->AspectChanged;
}

void STPCamera::STPSubscriberBenefit::unset() noexcept {
	this->Moved = false;
	this->Rotated = false;
	this->Zoomed = false;
	this->AspectChanged = false;
}

STPCamera::STPCamera(const STPEnvironment::STPCameraSetting& props) : 
	Camera(props), View(glm::identity<dmat4>()), PerspectiveProjection(glm::identity<dmat4>()),
	PositionOutdated(true), ViewOutdated(true), ProjectionOutdated(true),
	MappedBuffer(nullptr) {
	/* ----------------------- compile-time check ------------------- */
	static_assert(offsetof(STPPackedCameraBuffer, Pos) == 0
		&& offsetof(STPPackedCameraBuffer, V) == 16
		&& offsetof(STPPackedCameraBuffer, VNorm) == 80

		&& offsetof(STPPackedCameraBuffer, P) == 128
		&& offsetof(STPPackedCameraBuffer, InvP) == 192

		&& offsetof(STPPackedCameraBuffer, PVr) == 256
		&& offsetof(STPPackedCameraBuffer, PV) == 320
		&& offsetof(STPPackedCameraBuffer, InvPV) == 384

		&& offsetof(STPPackedCameraBuffer, LDFac) == 448
		&& offsetof(STPPackedCameraBuffer, Far) == 456,
	"The alignment of camera buffer does not obey std430 packing rule");

	/* -------------------------- validation -------------------------- */
	this->Camera.validate();
	//calculate the initial values for the camera based on the default settings.
	this->updateViewSpace();

	/* ------------------------ camera buffer ------------------------- */
	//set up buffer for camera transformation matrix
	this->CameraInformation.bufferStorage(sizeof(STPPackedCameraBuffer), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	this->MappedBuffer =
		reinterpret_cast<STPPackedCameraBuffer*>(this->CameraInformation.mapBufferRange(0, sizeof(STPPackedCameraBuffer),
			GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
	if (!this->MappedBuffer) {
		throw STPException::STPGLError("Unable to map the camera buffer");
	}
	//buffer has been setup, clear the buffer before use
	std::memset(this->MappedBuffer, 0x00u, sizeof(STPPackedCameraBuffer));

	//setup initial immutable values
	const double Cnear = this->Camera.Near,
		Cfar = this->Camera.Far;
	this->MappedBuffer->LDFac = static_cast<vec2>(dvec2(Cfar * Cnear, Cfar - Cnear));
	this->MappedBuffer->Far = static_cast<float>(Cfar);
	//update values
	this->CameraInformation.flushMappedBufferRange(0, sizeof(STPPackedCameraBuffer));
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

template<class Func>
inline void STPCamera::triggerSubscriberEvent(Func&& func) const {
	std::for_each(this->CameraSubscriber.cbegin(), this->CameraSubscriber.cend(), std::forward<Func>(func));
}

inline auto STPCamera::findSubcriber(STPSubscriberBenefit* benefit) const {
	return std::find(this->CameraSubscriber.cbegin(), this->CameraSubscriber.cend(), benefit);
}

void STPCamera::subscribe(STPSubscriberBenefit& benefit) {
	//make sure the same listener is not registered twice.
	if (this->findSubcriber(&benefit) != this->CameraSubscriber.cend()) {
		//the same instance is found
		throw STPException::STPMemoryError("The same listener instance has been registered with this camera previously");
	}

	//ok, add
	this->CameraSubscriber.emplace_back(&benefit);
}

void STPCamera::unsubscribe(STPSubscriberBenefit& benefit) {
	//try to find this instance
	const auto it = this->findSubcriber(&benefit);
	if (it == this->CameraSubscriber.cend()) {
		//not found
		throw STPException::STPMemoryError("This listener is not previously registered");
	}

	//found, remove it
	this->CameraSubscriber.erase(it);
}

void STPCamera::bindCameraBuffer(STPOpenGL::STPenum target, STPOpenGL::STPuint index) const {
	this->CameraInformation.bindBase(target, index);
}

void STPCamera::flush() {
	//recompute transform matrices
	if (this->ViewOutdated) {
		//update view matrix
		alignas(STPMatrix4x4d) const dmat4 view_data = glm::lookAt(
			this->Camera.Position,
			this->Camera.Position + this->Front,
			this->Up
		);
		this->View = STPMatrix4x4d(view_data);
	}
	if (this->ProjectionOutdated) {
		//update the projection matrix
		this->PerspectiveProjection = this->projection(this->Camera.Near, this->Camera.Far);
	}

	//update buffer whenever necessary
	if (this->PositionOutdated || this->ViewOutdated) {
		//position has changed
		if (this->PositionOutdated) {
			this->MappedBuffer->Pos = this->Camera.Position;
			this->CameraInformation.flushMappedBufferRange(offsetof(STPPackedCameraBuffer, Pos), sizeof(vec3));
		}

		//if position changes, view must also change
		//view matrix has changed
		this->MappedBuffer->V = static_cast<mat4>(this->View);
		this->MappedBuffer->VNorm = static_cast<mat3x4>(static_cast<mat4>(this->View.asMatrix3x3d().inverse().transpose()));
		this->CameraInformation.flushMappedBufferRange(offsetof(STPPackedCameraBuffer, V), sizeof(mat4) + sizeof(mat3x4));
	}
	if (this->ProjectionOutdated) {
		//projection matrix has changed
		this->MappedBuffer->P = static_cast<mat4>(this->PerspectiveProjection);
		this->MappedBuffer->InvP = static_cast<mat4>(this->PerspectiveProjection.inverse());
		this->CameraInformation.flushMappedBufferRange(offsetof(STPPackedCameraBuffer, P), sizeof(mat4) * 2);
	}
	//update compound matrices
	if (this->ViewOutdated || this->ProjectionOutdated) {
		const STPMatrix4x4d proj_view = this->PerspectiveProjection * this->View;

		//update the precomputed values
		this->MappedBuffer->PVr = static_cast<mat4>(this->PerspectiveProjection * (this->View.asMatrix3x3d()));
		this->MappedBuffer->PV = static_cast<mat4>(proj_view);
		this->MappedBuffer->InvPV = static_cast<mat4>(proj_view.inverse());
		this->CameraInformation.flushMappedBufferRange(offsetof(STPPackedCameraBuffer, PV), sizeof(mat4) * 2);
	}

	//reset states
	this->PositionOutdated = false;
	this->ViewOutdated = false;
	this->ProjectionOutdated = false;
}

const STPMatrix4x4d& STPCamera::view() const noexcept {
	return this->View;
}

const STPMatrix4x4d& STPCamera::projection() const noexcept {
	return this->PerspectiveProjection;
}

STPMatrix4x4d STPCamera::projection(double near, double far) const {
	//remember we are using reversed depth, so flip near and far plane
	alignas(STPMatrix4x4d) const dmat4 projection_data = glm::perspectiveRH_ZO(
		this->Camera.FoV,
		this->Camera.Aspect,
		far,
		near
	);
	return STPMatrix4x4d(projection_data);
}

const SuperTerrainPlus::STPEnvironment::STPCameraSetting& STPCamera::cameraStatus() const noexcept {
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

	//position change implies view change
	this->PositionOutdated = true;
	this->ViewOutdated = true;
	this->triggerSubscriberEvent([](auto* sub) { sub->Moved = true; });
}

void STPCamera::rotate(const dvec2& offset) {
	static constexpr double YAW_LIM = radians(360.0), PITCH_LIM = radians(89.0);

	//using sensitivity to scale the offset
	const dvec2 rotateAmount = offset * this->Camera.RotationSensitivity;
	this->Camera.Yaw += rotateAmount.x;
	//modulo the angle (floor modulo)
	this->Camera.Yaw = glm::mod(this->Camera.Yaw, YAW_LIM);
	//same for the pitch
	this->Camera.Pitch += rotateAmount.y;
	//does not allow pitch to go over vertical (so flip the camera over)
	this->Camera.Pitch = glm::clamp(this->Camera.Pitch, -PITCH_LIM, PITCH_LIM);

	//update camera front, right and up
	this->updateViewSpace();

	this->ViewOutdated = true;
	this->triggerSubscriberEvent([](auto* sub) { sub->Rotated = true; });
}

void STPCamera::zoom(double delta) {
	//change the view angle
	this->Camera.FoV += delta * this->Camera.ZoomSensitivity;
	//limit the zoom angle
	this->Camera.FoV = glm::clamp(
		this->Camera.FoV,
		this->Camera.ZoomLimit.x,
		this->Camera.ZoomLimit.y
	);

	this->ProjectionOutdated = true;
	this->triggerSubscriberEvent([](auto* sub) { sub->Zoomed = true; });
}

void STPCamera::setAspect(double aspect) {
	this->Camera.Aspect = aspect;

	this->ProjectionOutdated = true;
	this->triggerSubscriberEvent([](auto* sub) { sub->AspectChanged = true; });
}