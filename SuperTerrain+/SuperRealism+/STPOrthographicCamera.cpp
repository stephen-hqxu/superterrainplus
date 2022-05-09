#include <SuperRealism+/Utility/Camera/STPOrthographicCamera.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

using glm::dvec2;
using glm::dvec4;
using glm::dmat4;

using SuperTerrainPlus::STPMatrix4x4d;
using namespace SuperTerrainPlus::STPRealism;

STPOrthographicCamera::STPOrthographicCamera
	(const STPEnvironment::STPOrthographicCameraSetting& projection_props, const STPEnvironment::STPCameraSetting& camera_props) : 
	STPCamera(camera_props, STPProjectionCategory::Orthographic), OrthographicProjection(glm::identity<dmat4>()), Frustum(projection_props), ProjectionOutdated(true) {
	if (!this->Frustum.validate()) {
		throw STPException::STPInvalidEnvironment("Orthographic projection setting not validated");
	}
}

const STPMatrix4x4d& STPOrthographicCamera::ortho() const {
	if (this->ProjectionOutdated) {
		alignas(STPMatrix4x4d) const dmat4 projection_data = glm::ortho(
			this->Frustum.Left, 
			this->Frustum.Right, 
			this->Frustum.Bottom, 
			this->Frustum.Top,
			this->Camera.Near,
			this->Camera.Far
		);
		this->OrthographicProjection = STPMatrix4x4d(projection_data);
		this->ProjectionOutdated = false;
	}
	//return the matrix
	return this->OrthographicProjection;
}

const STPMatrix4x4d& STPOrthographicCamera::projection() const {
	return this->ortho();
}

STPMatrix4x4d STPOrthographicCamera::projection(double near, double far) const {
	alignas(STPMatrix4x4d) const dmat4 projection_data = glm::ortho(
		this->Frustum.Left,
		this->Frustum.Right,
		this->Frustum.Bottom,
		this->Frustum.Top,
		near,
		far
	);
	return STPMatrix4x4d(projection_data);
}

void STPOrthographicCamera::reshape(dvec4 side, dvec2 depth) {
	this->Frustum.Left = side.x;
	this->Frustum.Right = side.y;
	this->Frustum.Bottom = side.z;
	this->Frustum.Top = side.w;

	this->Camera.Near = depth.x;
	this->Camera.Far = depth.y;

	this->ProjectionOutdated = true;
	//trigger update
	for (auto callback : this->CallbackRegistry) {
		callback->onReshape(*this);
	}
}