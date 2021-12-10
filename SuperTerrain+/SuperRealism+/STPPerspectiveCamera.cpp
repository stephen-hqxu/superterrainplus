#include <SuperRealism+/Utility/Camera/STPPerspectiveCamera.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>

using glm::radians;

using glm::vec2;
using glm::mat4;

using namespace SuperTerrainPlus::STPRealism;

STPPerspectiveCamera::STPProjectionProperty::STPProjectionProperty() : 
	ViewAngle(radians(45.0f)), ZoomSensitivity(1.0f), 
	ZoomLimit(radians(1.0f), radians(90.0f)),
	Aspect(1.0f), Near(0.1f), Far(1.0f) {

}

STPPerspectiveCamera::STPPerspectiveCamera(const STPProjectionProperty& projection_props, const STPCamera::STPCameraProperty& camera_pros) : 
	STPCamera(camera_pros), PerspectiveFrustum(projection_props), PerspectiveProjection(mat4(0.0f)), ProjectionOutdated(true) {

}

const STPPerspectiveCamera::STPProjectionProperty& STPPerspectiveCamera::perspectiveStatus() const {
	return this->PerspectiveFrustum;
}

const mat4& STPPerspectiveCamera::perspective() const {
	if (this->ProjectionOutdated) {
		//update the projection matrix
		this->PerspectiveProjection = glm::perspective(
			this->PerspectiveFrustum.ViewAngle,
			this->PerspectiveFrustum.Aspect,
			this->PerspectiveFrustum.Near,
			this->PerspectiveFrustum.Far
		);
		this->ProjectionOutdated = false;
	}
	//return the projection
	return this->PerspectiveProjection;
}

void STPPerspectiveCamera::zoom(float delta) {
	//change the view angle
	this->PerspectiveFrustum.ViewAngle += delta * this->PerspectiveFrustum.ZoomSensitivity;
	//limit the zoom angle
	this->PerspectiveFrustum.ViewAngle = glm::clamp(
		this->PerspectiveFrustum.ViewAngle,
		this->PerspectiveFrustum.ZoomLimit.x,
		this->PerspectiveFrustum.ZoomLimit.y
	);

	//update the projection matrix
	this->ProjectionOutdated = true;
}

void STPPerspectiveCamera::rescale(float aspect) {
	this->PerspectiveFrustum.Aspect = aspect;

	this->ProjectionOutdated = true;
}

void STPPerspectiveCamera::reshape(vec2 shape) {
	this->PerspectiveFrustum.Near = shape.x;
	this->PerspectiveFrustum.Far = shape.y;

	this->ProjectionOutdated = true;
}