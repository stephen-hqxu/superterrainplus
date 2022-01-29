#include <SuperRealism+/Scene/STPCascadedShadowMap.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

//System
#include <numeric>
#include <limits>
#include <functional>

using glm::vec3;
using glm::mat4;
using glm::vec4;

using glm::radians;

using std::numeric_limits;

using namespace SuperTerrainPlus::STPRealism;

STPCascadedShadowMap::STPCascadedShadowMap(const STPLightFrustum& light_frustum) : LightDirection(vec3(0.0f)),
	LightSpaceOutdated(true), LightFrustum(light_frustum) {
	if (this->LightFrustum.ShadowDistanceMultiplier < 1.0f) {
		throw STPException::STPBadNumericRange("A less-than-one shadow distance is not able to cover the view frustum");
	}
	if (this->LightFrustum.Division.size() == 0ull) {
		throw STPException::STPBadNumericRange("There is no shadow level being defined");
	}
	
	//register a camera callback
	this->LightFrustum.Focus->registerListener(this);
}

STPCascadedShadowMap::~STPCascadedShadowMap() {
	this->LightFrustum.Focus->removeListener(this);
}

mat4 STPCascadedShadowMap::calcLightSpace(float near, float far, const mat4& view) const {
	//min and max of float
	static constexpr float minF = numeric_limits<float>::min(),
		maxF = numeric_limits<float>::max();
	static auto calcCorner = [](const mat4& projection, const mat4& view) -> STPFrustumCorner {
		//convert from clip space back to world space
		const mat4 inv = glm::inverse(projection * view);

		STPFrustumCorner frustumCorner;
		unsigned int index = 0u;
		for (unsigned int x = 0u; x < 2u; x++) {
			for (unsigned int y = 0u; y < 2u; y++) {
				for (unsigned int z = 0u; z < 2u; z++) {
					//convert from normalised device coordinate to clip space
					const vec4 pt = inv * vec4(2.0f * vec3(x, y, z) - 1.0f, 1.0f);
					frustumCorner[index++] = pt / pt.w;
				}
			}
		}

		return frustumCorner;
	};

	//calculate the projection matrix for this view frustum
	const mat4 projection = this->LightFrustum.Focus->projection(near, far);
	const STPFrustumCorner corner = calcCorner(projection, view);

	//each corner is normalised, so sum them up to get the coordinate of centre
	const vec3 centre = std::reduce(corner.cbegin(), corner.cend(), vec3(0.0f), std::plus<vec3>()) / (1.0f * corner.size());
	const mat4 lightView = glm::lookAt(centre + this->LightDirection, centre, vec3(0.0f, 1.0f, 0.0f));

	//align the light frusum tightly around the current view frustum
	//we need to find the eight corners of the light frusum.
	float minX = maxF,
		maxX = minF,
		minY = maxF,
		maxY = minF,
		minZ = maxF,
		maxZ = minF;
	for (const auto& v : corner) {
		//convert the camera view frustum from world space to light view space
		const vec4 trf = lightView * v;

		minX = glm::min(minX, trf.x);
		maxX = glm::max(maxX, trf.x);
		minY = glm::min(minY, trf.y);
		maxY = glm::max(maxY, trf.y);
		minZ = glm::min(minZ, trf.z);
		maxZ = glm::max(maxZ, trf.z);
	}

	const float zDistance = this->LightFrustum.ShadowDistanceMultiplier;
	//Tune the depth (maximum shadow distance)
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
	const mat4 lightProjection = glm::ortho(minX, maxX, minY, maxY, minZ, maxZ);

	return lightProjection * lightView;
}

void STPCascadedShadowMap::calcAllLightSpace(mat4* light_space) const {
	const STPCamera& viewer = *this->LightFrustum.Focus;
	const auto& shadow_level = this->LightFrustum.Division;

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
			current_light = this->calcLightSpace(near, shadow_level[i], camView);
		}
		else if (i < shadow_level.size()) {
			//the middle
			current_light = this->calcLightSpace(shadow_level[i - 1u], shadow_level[i], camView);
		}
		else {
			//the last one
			current_light = this->calcLightSpace(shadow_level[i - 1u], far, camView);
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

void STPCascadedShadowMap::forceLightSpaceUpdate() {
	this->LightSpaceOutdated = true;
}