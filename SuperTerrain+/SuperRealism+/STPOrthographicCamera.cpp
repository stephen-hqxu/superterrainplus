#include <SuperRealism+/Utility/Camera/STPOrthographicCamera.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

using glm::vec2;
using glm::vec4;
using glm::mat4;

using namespace SuperTerrainPlus::STPRealism;

STPOrthographicCamera::STPOrthographicCamera
	(const STPEnvironment::STPOrthographicCameraSetting& projection_props, const STPEnvironment::STPCameraSetting& camera_props) : 
	STPCamera(camera_props), Frustum(projection_props), OrthographicProjection(glm::identity<mat4>()), ProjectionOutdated(true) {
	if (!this->Frustum.validate()) {
		throw STPException::STPInvalidEnvironment("Orthographic projection setting not validated");
	}
}

inline const mat4& STPOrthographicCamera::ortho() const {
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

const mat4& STPOrthographicCamera::projection() const {
	return this->ortho();
}

mat4 STPOrthographicCamera::projection(float near, float far) const {
	return glm::ortho(
		this->Frustum.Left,
		this->Frustum.Right,
		this->Frustum.Bottom,
		this->Frustum.Top,
		near,
		far
	);
}

void STPOrthographicCamera::reshape(vec4 side, vec2 depth) {
	this->Frustum.Left = side.x;
	this->Frustum.Right = side.y;
	this->Frustum.Bottom = side.z;
	this->Frustum.Top = side.w;

	this->Camera.Near = depth.x;
	this->Camera.Far = depth.y;

	this->ProjectionOutdated = true;
	//trigger update
	if (this->Callback) {
		this->Callback->onReshape(*this);
	}
}