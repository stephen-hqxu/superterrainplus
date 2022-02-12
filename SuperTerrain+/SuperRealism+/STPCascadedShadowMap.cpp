#include <SuperRealism+/Scene/Light/STPCascadedShadowMap.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

//System
#include <numeric>
#include <limits>
#include <functional>
#include <algorithm>

using glm::vec3;
using glm::mat4;
using glm::vec4;

using glm::radians;
using glm::make_vec4;

using std::numeric_limits;

using namespace SuperTerrainPlus::STPRealism;

STPCascadedShadowMap::STPCascadedShadowMap(const STPLightFrustum& light_frustum) : LightDirection(vec3(0.0f)),
	LightSpaceOutdated(true), LightFrustum(light_frustum) {
	const auto& [res, div, band_radius, focus_camera, distance_mul] = this->LightFrustum;

	if (res == 0u) {
		throw STPException::STPBadNumericRange("The shadow map resolution should be a positive integer");
	}
	if (distance_mul < 1.0f) {
		throw STPException::STPBadNumericRange("A less-than-one shadow distance is not able to cover the view frustum");
	}
	if (div.size() == 0ull) {
		throw STPException::STPBadNumericRange("There is no shadow level being defined");
	}
	if (band_radius < 0.0f) {
		throw STPException::STPBadNumericRange("Shadow cascade band radius must be non-negative");
	}
	//register a camera callback
	focus_camera->registerListener(this);
}

STPCascadedShadowMap::~STPCascadedShadowMap() {
	this->LightFrustum.Focus->removeListener(this);
}

mat4 STPCascadedShadowMap::calcLightSpace(float near, float far, const mat4& view) const {
	//min and max of float
	static constexpr float minF = numeric_limits<float>::min(),
		maxF = numeric_limits<float>::max();
	//unit frusum corner, as a lookup table to avoid recomputation.
	static constexpr STPFrustumCornerFloat4 unitCorner = []() constexpr -> STPFrustumCornerFloat4 {
		STPFrustumCornerFloat4 frustumCorner = { };

		unsigned int index = 0u;
		for (unsigned int x = 0u; x < 2u; x++) {
			for (unsigned int y = 0u; y < 2u; y++) {
				for (unsigned int z = 0u; z < 2u; z++) {
					frustumCorner[index++] = {
						2.0f * x - 1.0f,
						2.0f * y - 1.0f,
						2.0f * z - 1.0f,
						1.0f
					};
				}
			}
		}

		return frustumCorner;
	}();

	//calculate the projection matrix for this view frustum
	const mat4 projection = this->LightFrustum.Focus->projection(near, far);
	STPFrustumCornerVec4 corner;
	std::transform(unitCorner.cbegin(), unitCorner.cend(), corner.begin(), [inv = glm::inverse(projection * view)](const auto& v) {
		//convert from normalised device coordinate to clip space
		const vec4 pt = inv * make_vec4(v.data());
		return pt / pt.w;
	});

	//each corner is normalised, so sum them up to get the coordinate of centre
	const vec3 centre = std::reduce(corner.cbegin(), corner.cend(), vec3(0.0f), std::plus<vec3>()) / (1.0f * corner.size());
	const mat4 lightView = glm::lookAt(centre + this->LightDirection, centre, vec3(0.0f, 1.0f, 0.0f));

	//align the light frusum tightly around the current view frustum
	//we need to find the eight corners of the light frusum.
	vec3 minExt = vec3(maxF),
		maxExt = vec3(minF);
	for (const auto& v : corner) {
		//convert the camera view frustum from world space to light view space
		const vec3 trf = static_cast<vec3>(lightView * v);

		minExt = glm::min(minExt, trf);
		maxExt = glm::max(maxExt, trf);
	}

	const float zDistance = this->LightFrustum.ShadowDistanceMultiplier;
	//Tune the depth (maximum shadow distance)
	float& minZ = minExt.z, 
		&maxZ = maxExt.z;
	if (minZ < 0.0f) {
		minZ *= zDistance;
	}
	else {
		minZ /= zDistance;
	}

	if (maxZ < 0.0f) {
		maxZ /= zDistance;
	}
	else {
		maxZ *= zDistance;
	}

	//finally, we are dealing with a directional light so shadow is parallel
	const mat4 lightProjection = glm::ortho(minExt.x, maxExt.x, minExt.y, maxExt.y, minZ, maxZ);

	return lightProjection * lightView;
}

void STPCascadedShadowMap::calcAllLightSpace(mat4* light_space) const {
	const STPCamera& viewer = *this->LightFrustum.Focus;
	const auto& shadow_level = this->LightFrustum.Division;
	//this offset pushes the far plane away and near plane in
	const float level_offset = this->LightFrustum.CascadeBandRadius;

	//The camera class has smart cache to the view matrix.
	const mat4& camView = viewer.view();
	const STPEnvironment::STPCameraSetting& camSetting = viewer.cameraStatus();
	const float near = camSetting.Near, far = camSetting.Far;

	const size_t lightSpaceCount = this->lightSpaceDimension();
	//calculate the light view matrix for each subfrusta
	for (unsigned int i = 0u; i < lightSpaceCount; i++) {
		//current light space in the mapped buffer
		mat4& current_light = light_space[i];

		if (i == 0u) {
			//the first frustum
			current_light = this->calcLightSpace(near, shadow_level[i] + level_offset, camView);
		}
		else if (i < shadow_level.size()) {
			//the middle
			current_light = this->calcLightSpace(shadow_level[i - 1u] - level_offset, shadow_level[i] + level_offset, camView);
		}
		else {
			//the last one
			current_light = this->calcLightSpace(shadow_level[i - 1u] - level_offset, far, camView);
		}
	}
}

void STPCascadedShadowMap::onMove(const STPCamera&) {
	//view matix changes
	this->LightSpaceOutdated = true;
}

void STPCascadedShadowMap::onRotate(const STPCamera&) {
	//view matrix also changes
	this->LightSpaceOutdated = true;
}

void STPCascadedShadowMap::onReshape(const STPCamera&) {
	//projection matrix changes
	this->LightSpaceOutdated = true;
}

const STPCascadedShadowMap::STPCascadePlane& STPCascadedShadowMap::getDivision() const {
	return this->LightFrustum.Division;
}

void STPCascadedShadowMap::setDirection(const vec3& dir) {
	this->LightDirection = dir;
	this->LightSpaceOutdated = true;
}

const vec3& STPCascadedShadowMap::getDirection() const {
	return this->LightDirection;
}

bool STPCascadedShadowMap::updateLightSpace(mat4* light_space) const {
	if (this->LightSpaceOutdated) {
		//need to also update light space matrix if shadow has been turned on for this light
		this->calcAllLightSpace(light_space);

		this->LightSpaceOutdated = false;
		return true;
	}
	return false;
}

inline size_t STPCascadedShadowMap::lightSpaceDimension() const {
	return this->LightFrustum.Division.size() + 1ull;
}

unsigned int STPCascadedShadowMap::shadowMapResolution() const {
	return this->LightFrustum.Resolution;
}

void STPCascadedShadowMap::forceLightSpaceUpdate() {
	this->LightSpaceOutdated = true;
}