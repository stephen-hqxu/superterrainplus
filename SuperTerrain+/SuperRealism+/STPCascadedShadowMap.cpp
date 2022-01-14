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
using glm::vec2;
using glm::uvec3;
using glm::vec3;
using glm::mat4;
using glm::vec4;

using glm::radians;

using std::numeric_limits;

using namespace SuperTerrainPlus::STPRealism;

STPCascadedShadowMap::STPCascadedShadowMap(const STPLightFrustum& light_frustum) :
	ShadowMap(GL_TEXTURE_2D_ARRAY), Viewer(*light_frustum.Camera), ShadowLevel(light_frustum.Level), ShadowDistance(light_frustum.ShadowDistanceMultiplier), 
	LightDirection(vec3(0.0f)) {
	const vec2& resolution = light_frustum.Resolution;
	if (resolution.x == 0u || resolution.y == 0u) {
		throw STPException::STPBadNumericRange("Both components of the shadow map resolution should be a positive integer");
	}
	if (this->ShadowDistance < 1.0f) {
		throw STPException::STPBadNumericRange("A less-than-one shadow distance is not able to cover the view frustum");
	}

	/* --------------------------- setup depth texture -----------------------------*/
	this->ShadowMap.textureStorage<STPTexture::STPDimension::THREE>(1, GL_DEPTH_COMPONENT24, uvec3(resolution, 1u));

	this->ShadowMap.filter(GL_NEAREST, GL_NEAREST);
	this->ShadowMap.wrap(GL_CLAMP_TO_BORDER);
	this->ShadowMap.borderColor(vec4(1.0f));
	//setup compare function so we can use shadow sampler in the shader
	this->ShadowMap.compareFunction(GL_LESS);
	this->ShadowMap.compareMode(GL_COMPARE_REF_TO_TEXTURE);
	/* ------------------------ setup capture framebuffer --------------------------*/
	this->ShadowContainer.attach(GL_DEPTH_ATTACHMENT, this->ShadowMap, 0);
	//we are rendering shadow and colors are not needed.
	this->ShadowContainer.drawBuffer(GL_NONE);
	this->ShadowContainer.readBuffer(GL_NONE);

	if (this->ShadowContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw STPException::STPGLError("Framebuffer for capturing shadow map fails to setup");
	}
}

mat4 STPCascadedShadowMap::calcLightSpace(float near, float far, const mat4& view) const {
	using glm::inverse;

	//min and max of float
	static constexpr float minF = numeric_limits<float>::min(),
		maxF = numeric_limits<float>::max();
	static auto calcCorner = [](const mat4& projection, const mat4& view) -> STPFrustumCorner {
		//convert from clip space back to world space
		const mat4 inv = inverse(projection * view);

		STPFrustumCorner frustumCorner;
		unsigned int index = 0u;
		for (unsigned int x = 0u; x < 2u; x++) {
			for (unsigned int y = 0u; y < 2u; y++) {
				for (unsigned int z = 0u; z < 2u; z++) {
					//convert from normalised device coordinate to clip space
					const vec4 pt = inv * vec4(2.0f * x - 1.0f, 2.0f * y - 1.0f, 2.0f * z - 1.0f, 1.0f);
					frustumCorner[index++] = pt / pt.w;
				}
			}
		}

		return frustumCorner;
	};

	//calculate the projection matrix for this view frustum
	const mat4 projection = this->Viewer.projection(near, far);
	const STPFrustumCorner corner = calcCorner(projection, view);

	//each corner is normalised, so sum them up to get the coordinate of centre
	const vec3 centre = std::reduce(corner.cbegin(), corner.cend(), vec3(0.0f), std::plus<vec3>()) / (1.0f * corner.size());
	const mat4 lightView = glm::lookAt(centre + this->LightDirection, centre, vec3(0.0f, 1.0f, 0.0f));

	//align the light frusum tightly around the current view frustum
	//we need to find the eight corners of the light frusum.
	float minX = minF,
		maxX = maxF,
		minY = minF,
		maxY = maxF,
		minZ = minF,
		maxZ = maxF;
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

	//Tune the depth (maximum shadow distance)
	if (minZ < 0.0f) {
		minZ *= this->ShadowDistance;
	}
	else {
		minZ /= this->ShadowDistance;
	}

	if (maxZ < 0.0f) {
		maxZ /= this->ShadowDistance;
	}
	else {
		maxZ *= this->ShadowDistance;
	}

	//finally, we are dealing with a directional light so shadow is parallel
	const mat4 lightProjection = glm::ortho(minX, maxX, minY, maxY, minZ, maxZ);

	return lightProjection * lightView;
}

STPCascadedShadowMap::STPLightSpaceMatrix STPCascadedShadowMap::calcAllLightSpace() const {
	STPLightSpaceMatrix lightSpace;
	lightSpace.reserve(this->ShadowLevel.size());

	//The camera class has smart cache to the view matrix.
	const mat4& camView = *this->Viewer.view().first;
	const STPEnvironment::STPCameraSetting& camSetting = this->Viewer.cameraStatus();
	const float near = camSetting.Near, far = camSetting.Far;

	//calculate the light view matrix for each subfrusta
	for (unsigned int i = 0u; i < this->ShadowLevel.size(); i++) {
		if (i == 0u) {
			//the first frustum
			lightSpace.emplace_back(this->calcLightSpace(near, this->ShadowLevel[i], camView));
		}
		else if (i < this->ShadowLevel.size()) {
			//the middle
			lightSpace.emplace_back(this->calcLightSpace(this->ShadowLevel[i - 1u], this->ShadowLevel[i], camView));
		}
		else {
			//the last one
			lightSpace.emplace_back(this->calcLightSpace(this->ShadowLevel[i - 1u], far, camView));
		}
	}

	return lightSpace;
}

void STPCascadedShadowMap::setDirection(const vec3& dir) {
	this->LightDirection = dir;
}

void STPCascadedShadowMap::capture() {
	this->ShadowContainer.bind(GL_FRAMEBUFFER);
}