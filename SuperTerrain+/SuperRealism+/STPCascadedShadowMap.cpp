#include <SuperRealism+/Renderer/STPCascadedShadowMap.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

//System
#include <numeric>
#include <limits>
#include <functional>
#include <memory>

using glm::uvec2;
using glm::vec2;
using glm::uvec3;
using glm::vec3;
using glm::mat4;
using glm::vec4;

using glm::radians;

using std::unique_ptr;
using std::make_unique;
using std::numeric_limits;

using namespace SuperTerrainPlus::STPRealism;

/**
 * @brief Only contains part of the data, dynamic elements are not included.
 * Padded for std430.
*/
struct STPPackedLightBuffer {
public:

	float Far;
	float _padFar;
	vec2 Bias;
};

STPCascadedShadowMap::STPShadowOption::STPShadowOption(const STPCascadedShadowMap& shadow) : Instance(&shadow), 
	BindlessHandle(**this->Instance->ShadowMapHandle) {

}

void STPCascadedShadowMap::STPShadowOption::operator()(STPShaderManager::STPShaderSource::STPMacroValueDictionary& dictionary) const {
	dictionary("CSM_LIGHT_SPACE_COUNT", this->Instance->cascadeCount());
}

STPCascadedShadowMap::STPCascadedShadowMap(const STPLightFrustum& light_frustum) : ShadowMap(GL_TEXTURE_2D_ARRAY),
	Viewer(*light_frustum.Camera), ShadowLevel(light_frustum.Level), ShadowDistance(light_frustum.ShadowDistanceMultiplier), 
	LightDirection(vec3(0.0f)), Resolution(light_frustum.Resolution) {
	if (this->Resolution.x == 0u || this->Resolution.y == 0u) {
		throw STPException::STPBadNumericRange("Both components of the shadow map resolution should be a positive integer");
	}
	if (this->ShadowDistance < 1.0f) {
		throw STPException::STPBadNumericRange("A less-than-one shadow distance is not able to cover the view frustum");
	}
	if (this->ShadowLevel.size() == 0ull) {
		throw STPException::STPBadNumericRange("There is no shadow level being defined");
	}

	/* --------------------------------------- depth texture setup ------------------------------------------------ */
	this->ShadowMap.textureStorage<STPTexture::STPDimension::THREE>(1, GL_DEPTH_COMPONENT24, 
		uvec3(this->Resolution, static_cast<unsigned int>(this->cascadeCount())));

	this->ShadowMap.filter(GL_NEAREST, GL_NEAREST);
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

	/* --------------------------------------- light space buffer setup ------------------------------------------- */
	//The offset from the beginning of the light buffer to reach the light space matrix
	constexpr static size_t lightMatrixOffset = sizeof(STPPackedLightBuffer);
	const size_t lightMatrixSize = sizeof(mat4) * this->cascadeCount(),
		shadowPlaneSize = sizeof(float) * this->ShadowLevel.size(),
		lightBufferSize = lightMatrixOffset + lightMatrixSize + shadowPlaneSize;
	//create a temporary storage for initial buffer setup
	unique_ptr<unsigned char[]> initialLightBuffer = make_unique<unsigned char[]>(lightBufferSize);

	unsigned char* binlightBuffer = initialLightBuffer.get();
	memset(binlightBuffer, 0x00, lightBufferSize);
	//settings that have fixed size
	STPPackedLightBuffer* shadow_setting = reinterpret_cast<STPPackedLightBuffer*>(binlightBuffer);
	shadow_setting->Far = this->Viewer.cameraStatus().Far;
	shadow_setting->Bias = vec2(light_frustum.BiasMultiplier, light_frustum.MinBias);
	//variable sized settings, skip the light matrix because this will be updated at runtime
	binlightBuffer += lightMatrixOffset + lightMatrixSize;
	memcpy(binlightBuffer, this->ShadowLevel.data(), shadowPlaneSize);

	this->LightBuffer.bufferStorageSubData(initialLightBuffer.get(), lightBufferSize, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
	this->LightBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1u);

	//store the pointer to memory that requires frequent update
	this->BufferLightMatrix = reinterpret_cast<mat4*>(this->LightBuffer.mapBufferRange(lightMatrixOffset, lightMatrixSize,
		GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT));
	if (!this->BufferLightMatrix) {
		throw STPException::STPMemoryError("Unable to map the memory for updating light matrix");
	}
}

STPCascadedShadowMap::~STPCascadedShadowMap() {
	this->LightBuffer.unmapBuffer();
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
	const mat4 projection = this->Viewer.projection(near, far);
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

void STPCascadedShadowMap::calcAllLightSpace() const {
	//The camera class has smart cache to the view matrix.
	const mat4& camView = this->Viewer.view();
	const STPEnvironment::STPCameraSetting& camSetting = this->Viewer.cameraStatus();
	const float near = camSetting.Near, far = camSetting.Far;

	const size_t lightSpaceCount = this->cascadeCount();
	//calculate the light view matrix for each subfrusta
	for (unsigned int i = 0u; i < lightSpaceCount; i++) {
		//current light space in the mapped buffer
		mat4& lightSpace = this->BufferLightMatrix[i];

		if (i == 0u) {
			//the first frustum
			lightSpace = this->calcLightSpace(near, this->ShadowLevel[i], camView);
		}
		else if (i < this->ShadowLevel.size()) {
			//the middle
			lightSpace = this->calcLightSpace(this->ShadowLevel[i - 1u], this->ShadowLevel[i], camView);
		}
		else {
			//the last one
			lightSpace = this->calcLightSpace(this->ShadowLevel[i - 1u], far, camView);
		}
	}
}

void STPCascadedShadowMap::setDirection(const vec3& dir) {
	this->LightDirection = dir;

	//need to also update light space matrix
	this->calcAllLightSpace();
}

void STPCascadedShadowMap::captureLightSpace() {
	this->ShadowContainer.bind(GL_FRAMEBUFFER);
}

void STPCascadedShadowMap::clearLightSpace() {
	this->ShadowContainer.clearDepth(1.0f);
}

size_t STPCascadedShadowMap::cascadeCount() const {
	return this->ShadowLevel.size() + 1ull;
}

STPCascadedShadowMap::STPShadowOption STPCascadedShadowMap::option() const {
	return STPShadowOption(*this);
}