#include <SuperRealism+/Utility/Camera/STPPerspectiveCamera.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//GLM
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>

using glm::radians;

using glm::dvec2;
using glm::dmat4;

using SuperTerrainPlus::STPMatrix4x4d;
using namespace SuperTerrainPlus::STPRealism;

STPPerspectiveCamera::STPPerspectiveCamera(const STPEnvironment::STPPerspectiveCameraSetting& projection_props, 
	const STPEnvironment::STPCameraSetting& camera_pros) :
	STPCamera(camera_pros, STPProjectionCategory::Perspective), PerspectiveProjection(glm::identity<dmat4>()), Frustum(projection_props), ProjectionOutdated(true) {
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

const STPMatrix4x4d& STPPerspectiveCamera::perspective() const {
	if (this->ProjectionOutdated) {
		//update the projection matrix
		alignas(STPMatrix4x4d) const dmat4 projection_data = glm::perspective(
			this->Frustum.ViewAngle,
			this->Frustum.Aspect,
			this->Camera.Near,
			this->Camera.Far
		);
		this->PerspectiveProjection = STPMatrix4x4d(projection_data);
		this->ProjectionOutdated = false;
	}
	//return the projection
	return this->PerspectiveProjection;
}

const STPMatrix4x4d& STPPerspectiveCamera::projection() const {
	return this->perspective();
}

STPMatrix4x4d STPPerspectiveCamera::projection(double near, double far) const {
	alignas(STPMatrix4x4d) const dmat4 projection_data = glm::perspective(
		this->Frustum.ViewAngle,
		this->Frustum.Aspect,
		near,
		far
	);
	return STPMatrix4x4d(projection_data);
}

void STPPerspectiveCamera::zoom(double delta) {
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

void STPPerspectiveCamera::rescale(double aspect) {
	this->Frustum.Aspect = aspect;

	this->setOutdated();
}

void STPPerspectiveCamera::reshape(dvec2 shape) {
	this->Camera.Near = shape.x;
	this->Camera.Far = shape.y;

	this->setOutdated();
}