#include <SuperRealism+/Utility/Camera/STPPerspectiveCamera.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>

using glm::radians;

using glm::vec2;
using glm::mat4;

using namespace SuperTerrainPlus::STPRealism;

STPPerspectiveCamera::STPPerspectiveCamera(const STPEnvironment::STPPerspectiveCameraSetting& projection_props, 
	const STPEnvironment::STPCameraSetting& camera_pros) :
	STPCamera(camera_pros), Frustum(projection_props), PerspectiveProjection(mat4(0.0f)), ProjectionOutdated(true) {
	if (!this->Frustum.validate()) {
		throw STPException::STPInvalidEnvironment("Project setting not validated");
	}
}

const SuperTerrainPlus::STPEnvironment::STPPerspectiveCameraSetting& STPPerspectiveCamera::perspectiveStatus() const {
	return this->Frustum;
}

inline STPPerspectiveCamera::STPMatrixResult STPPerspectiveCamera::perspective() const {
	bool same = true;
	if (this->ProjectionOutdated) {
		//update the projection matrix
		this->PerspectiveProjection = glm::perspective(
			this->Frustum.ViewAngle,
			this->Frustum.Aspect,
			this->Frustum.Near,
			this->Frustum.Far
		);
		this->ProjectionOutdated = false;
		same = false;
	}
	//return the projection
	return STPMatrixResult(&this->PerspectiveProjection, same);
}

STPPerspectiveCamera::STPMatrixResult STPPerspectiveCamera::projection() const {
	return this->perspective();
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

	//update the projection matrix
	this->ProjectionOutdated = true;
}

void STPPerspectiveCamera::rescale(float aspect) {
	this->Frustum.Aspect = aspect;

	this->ProjectionOutdated = true;
}

void STPPerspectiveCamera::reshape(vec2 shape) {
	this->Frustum.Near = shape.x;
	this->Frustum.Far = shape.y;

	this->ProjectionOutdated = true;
}