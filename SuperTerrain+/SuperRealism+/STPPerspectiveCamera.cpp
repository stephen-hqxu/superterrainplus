#include <SuperRealism+/Utility/Camera/STPPerspectiveCamera.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//GLM
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

using glm::radians;

using glm::vec2;
using glm::mat4;

using namespace SuperTerrainPlus::STPRealism;

STPPerspectiveCamera::STPPerspectiveCamera(const STPEnvironment::STPPerspectiveCameraSetting& projection_props, 
	const STPEnvironment::STPCameraSetting& camera_pros) :
	STPCamera(camera_pros), Frustum(projection_props), PerspectiveProjection(glm::identity<mat4>()), ProjectionOutdated(true) {
	if (!this->Frustum.validate()) {
		throw STPException::STPInvalidEnvironment("Perspective projection setting not validated");
	}
}

inline void STPPerspectiveCamera::setOutdated() {
	//update the projection matrix
	this->ProjectionOutdated = true;
	//trigger
	for (auto callback : this->CallbackRegistry) {
		callback->onReshape(*this);
	}
}

const SuperTerrainPlus::STPEnvironment::STPPerspectiveCameraSetting& STPPerspectiveCamera::perspectiveStatus() const {
	return this->Frustum;
}

inline const mat4& STPPerspectiveCamera::perspective() const {
	if (this->ProjectionOutdated) {
		//update the projection matrix
		this->PerspectiveProjection = glm::perspective(
			this->Frustum.ViewAngle,
			this->Frustum.Aspect,
			this->Camera.Near,
			this->Camera.Far
		);
		this->ProjectionOutdated = false;

	}
	//return the projection
	return this->PerspectiveProjection;
}

const mat4& STPPerspectiveCamera::projection() const {
	return this->perspective();
}

mat4 STPPerspectiveCamera::projection(float near, float far) const {
	return glm::perspective(
		this->Frustum.ViewAngle,
		this->Frustum.Aspect,
		near,
		far
	);
}

void STPPerspectiveCamera::zoom(float delta) {
	//change the view angle
	this->Frustum.ViewAngle += delta * this->Frustum.ZoomSensitivity;
	//limit the zoom angle
	this->Frustum.ViewAngle = glm::clamp(
		this->Frustum.ViewAngle,
		this->Frustum.ZoomLimit.x,
		this->Frustum.ZoomLimit.y
	);

	this->setOutdated();
}

void STPPerspectiveCamera::rescale(float aspect) {
	this->Frustum.Aspect = aspect;

	this->setOutdated();
}

void STPPerspectiveCamera::reshape(vec2 shape) {
	this->Camera.Near = shape.x;
	this->Camera.Far = shape.y;

	this->setOutdated();
}