#include <SuperRealism+/STPScenePipeline.h>

//Error
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <numeric>

using glm::vec2;
using glm::vec3;
using glm::ivec4;
using glm::vec4;
using glm::mat4;

using glm::value_ptr;

using std::unique_ptr;
using std::make_unique;
using std::vector;
using std::to_string;

using namespace SuperTerrainPlus::STPRealism;

/**
 * @brief Packed struct for mapped camera buffer following OpenGL std430 alignment rule.
*/
struct STPPackedCameraBuffer {
public:

	vec3 Pos;
	float _padPos;
	mat4 V;
	mat4 P;
};

/**
 * @brief Contains part of the data for shadow mapping buffer, dynamic elements are not included.
 * Padded for std430.
*/
struct STPPackedLightBuffer {
public:

	GLuint64 Ptr;
	float Far;
	float _padFar;
	vec2 Bias;
	vec2 _padBias;
};

STPScenePipeline::STPShadowPipeline::STPShadowPipeline(const STPLightRegistry& light_shadow) : Light(light_shadow), 
	TotalLightSpace(std::reduce(this->Light.cbegin(), this->Light.cend(), 0ull, [](auto init, const auto& val) {
		return init + val->cascadeCount();
	})) {
	constexpr static size_t lightFrustumInfoSize = sizeof(STPPackedLightBuffer);
	if (this->Light.size() > 1ull) {
		throw STPException::STPUnsupportedFunctionality("The shadow pipeline currently only supports one light that can cast shadow, "
			"this will be supported in the future release");
	}
	this->ShadowData.reserve(1u);
	this->ShadowData["CSM_LIGHT_SPACE_COUNT"] = static_cast<unsigned int>(this->TotalLightSpace);

	/* --------------------------------------- light space buffer setup ------------------------------------------- */
	//calculate the total amount of memory needed for the buffer
	const size_t lightShadowInfoSize = sizeof(STPPackedLightBuffer) * this->Light.size(),
		//The offset from the beginning of the light buffer to reach the light space matrix
		lightMatrixSize = sizeof(mat4) * this->TotalLightSpace,
		//num shadow plane equals number of light matrix minux 1
		shadowPlaneSize = sizeof(float) * (this->TotalLightSpace - this->Light.size()),
		lightBufferSize = lightShadowInfoSize + shadowPlaneSize + lightMatrixSize;
	//create a temporary storage for initial buffer setup
	unique_ptr<unsigned char[]> initialLightBuffer = make_unique<unsigned char[]>(lightBufferSize);
	//clear
	unsigned char* const binlightBuffer = initialLightBuffer.get();
	memset(binlightBuffer, 0x00, lightBufferSize);

	//locate each memory region
	unsigned char* current_block = binlightBuffer;
	STPPackedLightBuffer* const lightShadowInfo_block = reinterpret_cast<STPPackedLightBuffer*>(current_block);
	//skip the light matrix because this will be updated at runtime 
	current_block += lightShadowInfoSize + lightMatrixSize;
	float* shadowPlane_block = reinterpret_cast<float*>(current_block);
	//try to copy all information
	for (int i = 0; i < this->Light.size(); i++) {
		STPCascadedShadowMap& current_light = *this->Light[i];
		const auto& [resolution, shadow_level, viewer, distance_mul, max_bias, min_bias] = current_light.LightFrustum;

		STPPackedLightBuffer& shadow_setting = lightShadowInfo_block[i];
		//settings that have fixed size
		shadow_setting.Ptr = current_light.handle();
		shadow_setting.Far = viewer->cameraStatus().Far;
		shadow_setting.Bias = vec2(max_bias, min_bias);
		//variable sized settings
		const size_t shadowLevel_size = shadow_level.size();
		memcpy(shadowPlane_block, shadow_level.data(), shadowLevel_size * sizeof(float));
		shadowPlane_block += shadowLevel_size;
	}

	//allocate shared memory
	this->ShadowBuffer.bufferStorageSubData(binlightBuffer, lightBufferSize, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	this->ShadowBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1u);

	//store the pointer to memory that requires frequent update
	mat4* light_matrix = reinterpret_cast<mat4*>(this->ShadowBuffer.mapBufferRange(lightShadowInfoSize, lightMatrixSize,
		GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT));
	if (!light_matrix) {
		throw STPException::STPMemoryError("Unable to map the memory for updating light matrix");
	}
	//assign this allocated memory to the light so it can use it to update light information
	for (int i = 0; i < this->Light.size(); i++) {
		STPCascadedShadowMap& current_light = *this->Light[i];
		const size_t lightMatrixCount = current_light.cascadeCount();

		STPCascadedShadowMap::STPBufferAllocation memory_block = {
			&this->ShadowBuffer,
			sizeof(mat4) * i * lightMatrixCount,//offset relative to the mapped range, in byte
			light_matrix
		};
		current_light.setLightBuffer(memory_block);

		light_matrix += lightMatrixCount;
	}
}

STPScenePipeline::STPShadowPipeline::~STPShadowPipeline() {
	this->ShadowBuffer.unmapBuffer();
}

const STPShadowInformation& STPScenePipeline::STPShadowPipeline::shadowInformation() const {
	return this->ShadowData;
}

STPScenePipeline::STPScenePipeline(const STPCamera& camera, const STPLightRegistry& light_shadow) : 
	ShadowManager(light_shadow), SceneCamera(camera), updatePosition(true), updateView(true), updateProjection(true) {
	//set up buffer for camera transformation matrix
	this->CameraBuffer.bufferStorage(sizeof(STPPackedCameraBuffer), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	this->CameraBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0u);
	this->MappedCameraBuffer = 
		this->CameraBuffer.mapBufferRange(0, sizeof(STPPackedCameraBuffer),
			GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
	if (!this->MappedCameraBuffer) {
		throw STPException::STPGLError("Unable to map camera buffer to shader storage buffer");
	}
	//buffer has been setup, clear the buffer before use
	memset(this->MappedCameraBuffer, 0x00u, sizeof(STPPackedCameraBuffer));
	
	//set up initial GL context states
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDepthMask(GL_TRUE);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	//tessellation settings
	//barycentric coordinate system
	glPatchParameteri(GL_PATCH_VERTICES, 3);

	//register camera callback
	this->SceneCamera.registerListener(dynamic_cast<STPCamera::STPStatusChangeCallback*>(this));
}

STPScenePipeline::~STPScenePipeline() {
	//release the shader storage buffer
	this->CameraBuffer.unmapBuffer();
}

void STPScenePipeline::updateBuffer() const {
	STPPackedCameraBuffer* camBuf = reinterpret_cast<STPPackedCameraBuffer*>(this->MappedCameraBuffer);

	//only update buffer when necessary
	if (this->updatePosition || this->updateView) {
		//position has changed
		if (this->updatePosition) {
			camBuf->Pos = this->SceneCamera.cameraStatus().Position;
			this->CameraBuffer.flushMappedBufferRange(0, sizeof(vec3));

			this->updatePosition = false;
		}

		//view matrix has changed
		camBuf->V = this->SceneCamera.view();
		this->CameraBuffer.flushMappedBufferRange(sizeof(vec4), sizeof(mat4));

		this->updateView = false;
	}
	if (this->updateProjection) {
		constexpr static size_t offset_P = sizeof(vec4) + sizeof(mat4);
		//projection matrix has changed
		camBuf->P = this->SceneCamera.projection();
		this->CameraBuffer.flushMappedBufferRange(offset_P, sizeof(mat4));

		this->updateProjection = false;
	}
}

//by using separate flags instead of just flushing the buffer,
//we can avoid flushing frequently if camera is updated multiple times before next frame.

void STPScenePipeline::onMove(const STPCamera&) {
	this->updatePosition = true;
	this->updateView = true;
}

void STPScenePipeline::onRotate(const STPCamera&) {
	this->updateView = true;
}

void STPScenePipeline::onReshape(const STPCamera&) {
	this->updateProjection = true;
}

ivec4 STPScenePipeline::getViewport() const {
	ivec4 viewport;
	glGetIntegerv(GL_VIEWPORT, value_ptr(viewport));
	
	return viewport;
}

void STPScenePipeline::setClearColor(vec4 color) {
	glClearColor(color.r, color.g, color.b, color.a);
}