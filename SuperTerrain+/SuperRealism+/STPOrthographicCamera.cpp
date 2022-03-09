#include <SuperRealism+/Utility/Camera/STPOrthographicCamera.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

using glm::dvec2;
using glm::dvec4;
using glm::dmat4;

using namespace SuperTerrainPlus::STPRealism;

STPOrthographicCamera::STPOrthographicCamera
	(const STPEnvironment::STPOrthographicCameraSetting& projection_props, const STPEnvironment::STPCameraSetting& camera_props) : 
	STPCamera(camera_props, STPProjectionCategory::Orthographic), Frustum(projection_props), OrthographicProjection(glm::identity<dmat4>()), ProjectionOutdated(true) {
	if (!this->Frustum.validate()) {
		throw STPException::STPInvalidEnvironment("Orthographic projection setting not validated");
	}
}

inline const dmat4& STPOrthographicCamera::ortho() const {
	if (this->ProjectionOutdated) {
		this->OrthographicProjection = glm::ortho(
			this->Frustum.Left, 
			this->Frustum.Right, 
			this->Frustum.Bottom, 
			this->Frustum.Top,
			this->Camera.Near,
			this->Camera.Far
		);
		this->ProjectionOutdated = false;

	}
	//return the matrix
	return this->OrthographicProjection;
}

const dmat4& STPOrthographicCamera::projection() const {
	return this->ortho();
}

dmat4 STPOrthographicCamera::projection(double near, double far) const {
	return glm::ortho(
		this->Frustum.Left,
		this->Frustum.Right,
		this->Frustum.Bottom,
		this->Frustum.Top,
		near,
		far
	);
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