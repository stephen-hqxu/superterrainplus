#include <SuperRealism+/STPScenePipeline.h>

//Error
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <numeric>

using glm::vec2;
using glm::uvec3;
using glm::vec3;
using glm::ivec4;
using glm::vec4;
using glm::mat4;

using glm::value_ptr;

using std::unique_ptr;
using std::make_unique;
using std::make_pair;
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

STPShadowPipeline::STPShadowMapMemory::STPShadowMapMemory(uvec3 resolution, STPShadowMapFilter filter) : DepthTexture(GL_TEXTURE_2D_ARRAY) {
	if (resolution.x == 0u || resolution.y == 0u || resolution.z == 0u) {
		throw STPException::STPBadNumericRange("All components of the shadow map resolution should be a positive integer");
	}
	/* --------------------------------------- depth texture setup ------------------------------------------------ */
	this->DepthTexture.textureStorage<STPTexture::STPDimension::THREE>(1, GL_DEPTH_COMPONENT24, resolution);

	if (filter == STPShadowMapFilter::Nearest) {
		this->DepthTexture.filter(GL_NEAREST, GL_NEAREST);
	}
	else {
		//all other filter options implies linear filtering.
		this->DepthTexture.filter(GL_LINEAR, GL_LINEAR);
	}
	this->DepthTexture.wrap(GL_CLAMP_TO_BORDER);
	this->DepthTexture.borderColor(vec4(1.0f));
	//setup compare function so we can use shadow sampler in the shader
	this->DepthTexture.compareFunction(GL_LESS);
	this->DepthTexture.compareMode(GL_COMPARE_REF_TO_TEXTURE);

	this->DepthTextureHandle.emplace(this->DepthTexture);

	/* -------------------------------------- depth texture framebuffer ------------------------------------------- */
	//attach the new depth texture to the framebuffer
	this->DepthRecorder.attach(GL_DEPTH_ATTACHMENT, this->DepthTexture, 0);
	//we are rendering shadow and colors are not needed.
	this->DepthRecorder.drawBuffer(GL_NONE);
	this->DepthRecorder.readBuffer(GL_NONE);

	if (this->DepthRecorder.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw STPException::STPGLError("Framebuffer for capturing shadow map fails to setup");
	}
}

SuperTerrainPlus::STPOpenGL::STPuint64 STPShadowPipeline::STPShadowMapMemory::handle() const {
	return **this->DepthTextureHandle;
}

STPShadowPipeline::STPShadowPipeline(const STPLightRegistry& light_shadow, STPShadowMapFilter filter) {
	constexpr static size_t lightFrustumInfoSize = sizeof(STPPackedLightBuffer);
	if (light_shadow.size() > 1ull) {
		throw STPException::STPUnsupportedFunctionality("The shadow pipeline currently only supports one light that can cast shadow, "
			"this will be supported in the future release");
	}
	//The total number of light space among all registering lights
	const size_t totalLightSpace = std::reduce(light_shadow.cbegin(), light_shadow.cend(), 0ull, [](auto init, const auto& val) {
		return init + val->lightSpaceSize();
	});

	//shadow setting
	this->ShadowOption.emplace_back("CSM_LIGHT_SPACE_COUNT", static_cast<unsigned int>(totalLightSpace));
	this->ShadowOption.emplace_back("CSM_SHADOW_FILTER", static_cast<std::underlying_type_t<STPShadowMapFilter>>(filter));

	/* --------------------------------------- light space buffer setup ------------------------------------------- */
	//calculate the total amount of memory needed for the buffer
	const size_t lightShadowInfoSize = sizeof(STPPackedLightBuffer) * light_shadow.size(),
		//The offset from the beginning of the light buffer to reach the light space matrix
		lightMatrixSize = sizeof(mat4) * totalLightSpace,
		//num shadow plane equals number of light matrix minux 1
		shadowPlaneSize = sizeof(float) * (totalLightSpace - light_shadow.size()),
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

	this->DepthDatabase.reserve(light_shadow.size());
	//try to copy all information
	for (int i = 0; i < light_shadow.size(); i++) {
		STPDirectionalLight& current_light = *light_shadow[i];
		const auto& [resolution, shadow_level, viewer, distance_mul, max_bias, min_bias] = current_light.LightFrustum;

		//allocate memory for storing depth texture
		const STPShadowMapMemory& shadow_map = this->DepthDatabase.emplace_back(uvec3(resolution, current_light.lightSpaceSize()), filter);

		STPPackedLightBuffer& shadow_setting = lightShadowInfo_block[i];
		//settings that have fixed size
		shadow_setting.Ptr = shadow_map.handle();
		shadow_setting.Far = viewer->cameraStatus().Far;
		shadow_setting.Bias = vec2(max_bias, min_bias);
		//variable sized settings
		const size_t shadowLevel_size = shadow_level.size();
		memcpy(shadowPlane_block, shadow_level.data(), shadowLevel_size * sizeof(float));
		shadowPlane_block += shadowLevel_size;
	}

	//allocate shared memory
	this->LightDataBuffer.bufferStorageSubData(binlightBuffer, lightBufferSize, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
	this->LightDataBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1u);

	//store the pointer to memory that requires frequent update
	mat4* light_matrix = reinterpret_cast<mat4*>(this->LightDataBuffer.mapBufferRange(lightShadowInfoSize, lightMatrixSize,
		GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT));
	if (!light_matrix) {
		throw STPException::STPMemoryError("Unable to map the memory for updating light matrix");
	}

	this->LightAllocationDatabase.reserve(light_shadow.size());
	//assign this allocated memory to the light so it can use it to update light information
	for (int i = 0; i < light_shadow.size(); i++) {
		STPDirectionalLight& current_light = *light_shadow[i];
		const size_t lightMatrixCount = current_light.lightSpaceSize();

		this->LightAllocationDatabase.emplace_back(STPBufferLightAllocation{
			current_light,
			light_matrix,
			sizeof(mat4) * i * lightMatrixCount,//offset relative to the mapped range, in byte
			this->DepthDatabase[i].DepthRecorder
		});

		light_matrix += lightMatrixCount;
	}
}

STPShadowPipeline::~STPShadowPipeline() {
	this->LightDataBuffer.unmapBuffer();
}

const STPShadowInformation& STPShadowPipeline::shadowInformation() const {
	return this->ShadowOption;
}

STPScenePipeline::STPScenePipeline(const STPCamera& camera, STPShadowPipeline& shadow_pipeline) : 
	ShadowPipeline(shadow_pipeline), SceneCamera(camera), updatePosition(true), updateView(true), updateProjection(true) {
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
	this->SceneCamera.registerListener(this);
}

STPScenePipeline::~STPScenePipeline() {
	//release the shader storage buffer
	this->CameraBuffer.unmapBuffer();
	//remove camera callback
	this->SceneCamera.removeListener(this);
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