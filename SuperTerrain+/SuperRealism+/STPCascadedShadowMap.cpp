#include <SuperRealism+/Renderer/STPCascadedShadowMap.h>

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

using glm::uvec2;
using glm::uvec3;
using glm::vec3;
using glm::mat4;
using glm::vec4;

using glm::radians;

using std::numeric_limits;

using namespace SuperTerrainPlus::STPRealism;

STPCascadedShadowMap::STPCascadedShadowMap(const STPLightFrustum& light_frustum) : ShadowMap(GL_TEXTURE_2D_ARRAY),
	LightBuffer{ }, LightDirection(vec3(0.0f)), LightFrustum(light_frustum) {
	const uvec2 resolution = this->LightFrustum.Resolution;

	if (resolution.x == 0u || resolution.y == 0u) {
		throw STPException::STPBadNumericRange("Both components of the shadow map resolution should be a positive integer");
	}
	if (this->LightFrustum.ShadowDistanceMultiplier < 1.0f) {
		throw STPException::STPBadNumericRange("A less-than-one shadow distance is not able to cover the view frustum");
	}
	if (this->LightFrustum.Level.size() == 0ull) {
		throw STPException::STPBadNumericRange("There is no shadow level being defined");
	}

	/* --------------------------------------- depth texture setup ------------------------------------------------ */
	this->ShadowMap.textureStorage<STPTexture::STPDimension::THREE>(1, GL_DEPTH_COMPONENT24, 
		uvec3(resolution, static_cast<unsigned int>(this->cascadeCount())));

	this->ShadowMap.filter(GL_LINEAR, GL_LINEAR);
	this->ShadowMap.wrap(GL_CLAMP_TO_BORDER);
	this->ShadowMap.borderColor(vec4(1.0f));
	//setup compare function so we can use shadow sampler in the shader
	this->ShadowMap.compareFunction(GL_LESS);
	this->ShadowMap.compareMode(GL_COMPARE_REF_TO_TEXTURE);

	this->ShadowMapHandle.emplace(this->ShadowMap);

	/* -------------------------------------- depth texture framebuffer ------------------------------------------- */
	//attach the new depth texture to the framebuffer
	this->ShadowContainer.attach(GL_DEPTH_ATTACHMENT, this->ShadowMap, 0);
	//we are rendering shadow and colors are not needed.
	this->ShadowContainer.drawBuffer(GL_NONE);
	this->ShadowContainer.readBuffer(GL_NONE);

	if (this->ShadowContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw STPException::STPGLError("Framebuffer for capturing shadow map fails to setup");
	}
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
	const mat4 projection = this->LightFrustum.Camera->projection(near, far);
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
	const STPCamera& viewer = *this->LightFrustum.Camera;
	const STPCascadeLevel& shadow_level = this->LightFrustum.Level;

	//The camera class has smart cache to the view matrix.
	const mat4& camView = viewer.view();
	const STPEnvironment::STPCameraSetting& camSetting = viewer.cameraStatus();
	const float near = camSetting.Near, far = camSetting.Far;

	const size_t lightSpaceCount = this->cascadeCount();
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

void STPCascadedShadowMap::setLightBuffer(const STPBufferAllocation& allocation) {
	this->LightBuffer = allocation;
}

void STPCascadedShadowMap::setDirection(const vec3& dir) {
	this->LightDirection = dir;

	const auto [buffer, start, light_space] = this->LightBuffer;
	//need to also update light space matrix if shadow has been turned on for this light
	this->calcAllLightSpace(light_space);

	//after update, the data needs to be flushed.
	buffer->flushMappedBufferRange(start, sizeof(mat4) * this->cascadeCount());
	
}

void STPCascadedShadowMap::captureLightSpace() {
	this->ShadowContainer.bind(GL_FRAMEBUFFER);
}

void STPCascadedShadowMap::clearLightSpace() {
	this->ShadowContainer.clearDepth(1.0f);
}

inline size_t STPCascadedShadowMap::cascadeCount() const {
	return this->LightFrustum.Level.size() + 1ull;
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPCascadedShadowMap::handle() const {
	return **this->ShadowMapHandle;
}