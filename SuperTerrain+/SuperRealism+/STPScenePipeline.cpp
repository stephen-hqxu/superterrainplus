#include <SuperRealism+/Scene/STPScenePipeline.h>
#include <SuperRealism+/STPRealismInfo.h>
//Error
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>

//Base Off-screen Rendering
#include <SuperRealism+/Renderer/STPScreen.h>
//GL Object
#include <SuperRealism+/Object/STPSampler.h>
#include <SuperRealism+/Object/STPBindlessTexture.h>
#include <SuperRealism+/Object/STPFrameBuffer.h>
#include <SuperRealism+/Object/STPProgramManager.h>

#include <SuperRealism+/Utility/STPLogStorage.hpp>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

//System
#include <optional>
#include <array>
#include <algorithm>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

using glm::uvec2;
using glm::uvec3;
using glm::vec3;
using glm::ivec4;
using glm::vec4;
using glm::mat4;

using glm::value_ptr;

using std::optional;
using std::vector;
using std::array;
using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

STPShadowInformation STPScenePipeline::STPSceneInitialiser::shadowInitialiser() const {
	return STPShadowInformation{ 
		{ "LIGHT_SPACE_COUNT", this->LightSpaceCount }
	};
}

STPScenePipeline::STPSharedTexture::STPSharedTexture() : 
	DepthStencil(GL_TEXTURE_2D) {

}

class STPScenePipeline::STPCameraInformationMemory : private STPCamera::STPStatusChangeCallback {
private:

	//some flags to indicate buffer update status
	bool updatePosition, updateView, updateProjection;

	//by using separate flags instead of just flushing the buffer,
	//we can avoid flushing frequently if camera is updated multiple times before next frame.

	void STPCameraInformationMemory::onMove(const STPCamera&) override {
		this->updatePosition = true;
		this->updateView = true;
	}

	void STPCameraInformationMemory::onRotate(const STPCamera&) override {
		this->updateView = true;
	}

	void STPCameraInformationMemory::onReshape(const STPCamera&) override {
		this->updateProjection = true;
	}

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

public:

	//The master camera for the rendering scene.
	const STPCamera& Camera;
	STPBuffer Buffer;
	void* MappedBuffer;

	/**
	 * @brief Init a new camera memory.
	 * @param camera The pointe to the camera to be managed.
	*/
	STPCameraInformationMemory(const STPCamera& camera) : updatePosition(true), updateView(true), updateProjection(true),
		Camera(camera), MappedBuffer(nullptr) {
		//set up buffer for camera transformation matrix
		this->Buffer.bufferStorage(sizeof(STPPackedCameraBuffer), GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
		this->Buffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0u);
		this->MappedBuffer =
			this->Buffer.mapBufferRange(0, sizeof(STPPackedCameraBuffer),
				GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
		if (!this->MappedBuffer) {
			throw STPException::STPGLError("Unable to map camera buffer to shader storage buffer");
		}
		//buffer has been setup, clear the buffer before use
		memset(this->MappedBuffer, 0x00u, sizeof(STPPackedCameraBuffer));
		this->Buffer.flushMappedBufferRange(0, sizeof(STPPackedCameraBuffer));

		//register camera callback
		this->Camera.registerListener(this);
	}

	STPCameraInformationMemory(const STPCameraInformationMemory&) = delete;

	STPCameraInformationMemory(STPCameraInformationMemory&&) = delete;

	STPCameraInformationMemory& operator=(const STPCameraInformationMemory&) = delete;

	STPCameraInformationMemory& operator=(STPCameraInformationMemory&&) = delete;
	
	~STPCameraInformationMemory() {
		//release the shader storage buffer
		STPBuffer::unbindBase(GL_SHADER_STORAGE_BUFFER, 0u);
		this->Buffer.unmapBuffer();
		//remove camera callback
		this->Camera.removeListener(this);
	}

	/**
	 * @brief Update data in scene pipeline buffer.
	*/
	void updateBuffer() {
		STPPackedCameraBuffer* camBuf = reinterpret_cast<STPPackedCameraBuffer*>(this->MappedBuffer);

		//only update buffer when necessary
		if (this->updatePosition || this->updateView) {
			//position has changed
			if (this->updatePosition) {
				camBuf->Pos = this->Camera.cameraStatus().Position;
				this->Buffer.flushMappedBufferRange(0, sizeof(vec3));

				this->updatePosition = false;
			}

			//view matrix has changed
			camBuf->V = this->Camera.view();
			this->Buffer.flushMappedBufferRange(sizeof(vec4), sizeof(mat4));

			this->updateView = false;
		}
		if (this->updateProjection) {
			constexpr static size_t offset_P = sizeof(vec4) + sizeof(mat4);
			//projection matrix has changed
			camBuf->P = this->Camera.projection();
			this->Buffer.flushMappedBufferRange(offset_P, sizeof(mat4));

			this->updateProjection = false;
		}
	}

};

class STPScenePipeline::STPShadowPipeline {
private:

	STPTexture LightDepthTexture;
	optional<STPBindlessTexture> LightDepthTextureHandle;
	STPFrameBuffer LightDepthContainer;

	STPBuffer LightSpaceBuffer;
	mat4* MappedLightSpaceBuffer;

	const uvec2 ShadowResolution;

	/**
	 * @brief Query the viewport information in the current context.
	 * @return The X, Y coordinate and the width, height of the viewport.
	*/
	static inline ivec4 STPShadowPipeline::getViewport() {
		ivec4 viewport;
		glGetIntegerv(GL_VIEWPORT, value_ptr(viewport));

		return viewport;
	}

public:

	/**
	 * @brief Init a new STPShadowPipeline.
	 * @param shadow_init The pointer to the scene shadow initialiser.
	*/
	STPShadowPipeline(const STPSceneShadowInitialiser& shadow_init) : LightDepthTexture(GL_TEXTURE_2D_ARRAY), 
		ShadowResolution(shadow_init.ShadowMapResolution) {
		if (this->ShadowResolution == uvec2(0u)) {
			throw STPException::STPBadNumericRange("All components of the shadow map resolution should be a positive integer");
		}
		if (shadow_init.LightSpaceCount == 0u) {
			throw STPException::STPBadNumericRange("The number of light space should be a positive integer");
		}

		/* --------------------------------------- depth texture setup ------------------------------------------------ */
		const uvec3 dimension = uvec3(this->ShadowResolution, shadow_init.LightSpaceCount);
		this->LightDepthTexture.textureStorage<STPTexture::STPDimension::THREE>(1, GL_DEPTH_COMPONENT24, dimension);

		if (shadow_init.ShadowFilter == STPShadowMapFilter::Nearest) {
			this->LightDepthTexture.filter(GL_NEAREST, GL_NEAREST);
		}
		else {
			//all other filter options implies linear filtering.
			this->LightDepthTexture.filter(GL_LINEAR, GL_LINEAR);
		}
		this->LightDepthTexture.wrap(GL_CLAMP_TO_BORDER);
		this->LightDepthTexture.borderColor(vec4(1.0f));
		//setup compare function so we can use shadow sampler in the shader
		this->LightDepthTexture.compareFunction(GL_LESS);
		this->LightDepthTexture.compareMode(GL_COMPARE_REF_TO_TEXTURE);

		this->LightDepthTextureHandle.emplace(this->LightDepthTexture);

		/* -------------------------------------- depth texture framebuffer ------------------------------------------- */
		//attach the new depth texture to the framebuffer
		this->LightDepthContainer.attach(GL_DEPTH_ATTACHMENT, this->LightDepthTexture, 0);
		//we are rendering shadow and colors are not needed.
		this->LightDepthContainer.drawBuffer(GL_NONE);
		this->LightDepthContainer.readBuffer(GL_NONE);

		if (this->LightDepthContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			throw STPException::STPGLError("Framebuffer for capturing shadow map fails to setup");
		}

		/* ----------------------------------------- shadow pipeline ------------------------------------------------- */
		const size_t lightSpaceSize = sizeof(mat4) * shadow_init.LightSpaceCount;
		//setup light space information buffer
		this->LightSpaceBuffer.bufferStorage(lightSpaceSize, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
		this->LightSpaceBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1u);
		this->MappedLightSpaceBuffer = 
			reinterpret_cast<mat4*>(this->LightSpaceBuffer.mapBufferRange(0, lightSpaceSize,
				GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
		if (!this->MappedLightSpaceBuffer) {
			throw STPException::STPGLError("Unable to map light space information buffer to shader storage buffer");
		}
		//clear the garbage data
		memset(this->MappedLightSpaceBuffer, 0x00, lightSpaceSize);
		this->LightSpaceBuffer.flushMappedBufferRange(0, lightSpaceSize);
	}

	STPShadowPipeline(const STPShadowPipeline&) = delete;

	STPShadowPipeline(STPShadowPipeline&&) = delete;

	STPShadowPipeline& operator=(const STPShadowPipeline&) = delete;

	STPShadowPipeline& operator=(STPShadowPipeline&&) = delete;

	~STPShadowPipeline() {
		STPBuffer::unbindBase(GL_SHADER_STORAGE_BUFFER, 1u);
		this->LightSpaceBuffer.unmapBuffer();
	}

	/**
	 * @brief Get the bindless handle to the shadow map.
	 * @return A bindless handle, this handle remains valid as long as the instance is valid.
	*/
	inline GLuint64 handle() const {
		return **this->LightDepthTextureHandle;
	}

	/**
	 * @brief Render the shadow-casting objects to shadow-casting light space.
	 * @param shadow_obejct The pointer to all shadow-casting objects.
	 * @param shadow_light The pointer to all shadow-casting light.
	*/
	void renderToShadow(const vector<STPSceneObject::STPOpaqueObject<true>*>& shadow_object, const vector<STPSceneLight::STPEnvironmentLight<true>*>& shadow_light) {
		//record the original viewport size
		const ivec4 ori_vp = STPShadowPipeline::getViewport();

		//clear shadow map
		this->LightDepthContainer.bind(GL_FRAMEBUFFER);
		glClear(GL_DEPTH_BUFFER_BIT);

		size_t current_light_space_start = 0ull;
		for (auto shadowable_light : shadow_light) {
			const STPLightShadow& shadow_instance = shadowable_light->getLightShadow();
			const size_t current_light_space_dim = shadow_instance.lightSpaceDimension(),
				buffer_start = current_light_space_start * sizeof(mat4),
				current_buffer_size = current_light_space_dim * sizeof(mat4);

			//check if the light needs to update
			if (shadow_instance.updateLightSpace(this->MappedLightSpaceBuffer)) {
				//light matrix has been updated, flush the buffer
				//TODO: do not update shadow map every frame.
				this->LightSpaceBuffer.flushMappedBufferRange(buffer_start, current_buffer_size);
			}

			//change the view port to fit the shadow map
			glViewport(0, 0, this->ShadowResolution.x, this->ShadowResolution.y);

			//for those opaque render components (those can cast shadow), render depth
			for (auto shadowable_object : shadow_object) {
				shadowable_object->renderDepth(static_cast<unsigned int>(current_light_space_start), static_cast<unsigned int>(current_light_space_dim));
			}

			//increment to the next light
			current_light_space_start += current_light_space_dim;
		}

		//rollback the previous viewport size
		glViewport(ori_vp.x, ori_vp.y, ori_vp.z, ori_vp.w);
		//stop drawing shadow
		STPFrameBuffer::unbind(GL_FRAMEBUFFER);
	}

};

class STPScenePipeline::STPGeometryBufferResolution : private STPScreen {
private:

	//The dependent scene pipeline.
	const STPScenePipeline& Pipeline;

	typedef std::array<STPBindlessTexture, 5ull> STPGeometryBufferHandle;
	typedef std::array<GLuint64, 5ull> STPGeometryBufferRawHandle;

	STPSampler GSampler, DepthSampler;
	//G-buffer components
	//The depth buffer can be used to reconstruct world position.
	//Not all buffers are present here, some of them are shared with the scene pipeline (and other rendering components to reduce memory usage)
	STPTexture GAlbedo, GNormal, GSpecular, GAmbient;
	optional<STPGeometryBufferHandle> GHandle;
	STPFrameBuffer GeometryContainer;

	//Bindless handle for all lights from the scene pipeline
	optional<STPBindlessTexture> SpectrumHandle;

	constexpr static auto DeferredShaderFilename =
		STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPDeferredShading", ".frag");

public:

	STPProgramManager LightingProcessor;

	/**
	 * @brief Init a new geometry buffer resolution instance.
	 * @param pipeline The pointer to the dependent scene pipeline.
	 * @param shadow_init The pointer to the scene shadow initialiser.
	 * @param log The log from compiling light pass shader.
	*/
	STPGeometryBufferResolution(const STPScenePipeline& pipeline, const STPSceneShadowInitialiser& shadow_init, 
		STPScenePipelineLog::STPGeometryBufferResolutionLog& log) : Pipeline(pipeline),
		GAlbedo(GL_TEXTURE_2D), GNormal(GL_TEXTURE_2D), GSpecular(GL_TEXTURE_2D), GAmbient(GL_TEXTURE_2D) {
		//setup geometry buffer shader
		STPShaderManager screen_shader(std::move(STPGeometryBufferResolution::compileScreenVertexShader(log.QuadShader))),
			g_shader(GL_FRAGMENT_SHADER);

		//do something to the fragment shader
		STPShaderManager::STPShaderSource source(*STPFile(DeferredShaderFilename.data()));
		STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

		Macro("LIGHT_SPACE_COUNT", shadow_init.LightSpaceCount)
			("LIGHT_SHADOW_FILTER", static_cast<std::underlying_type_t<STPShadowMapFilter>>(shadow_init.ShadowFilter));

		source.define(Macro);
		//compile shader
		log.LightingShader.Log[0] = g_shader(source);

		//attach to the G-buffer resolution program
		this->LightingProcessor
			.attach(screen_shader)
			.attach(g_shader);
		log.LightingShader.Log[1] = this->LightingProcessor.finalise();
		//error checking
		if (!this->LightingProcessor) {
			throw STPException::STPGLError("Geometry buffer resolution program fails to validate");
		}

		/* ------------------------------- setup G-buffer sampler ------------------------------------- */
		this->GSampler.filter(GL_NEAREST, GL_NEAREST);
		this->GSampler.wrap(GL_CLAMP_TO_BORDER);
		this->GSampler.borderColor(vec4(vec3(0.0f), 1.0f));

		this->DepthSampler.filter(GL_NEAREST, GL_NEAREST);
		this->DepthSampler.wrap(GL_CLAMP_TO_BORDER);
		this->DepthSampler.borderColor(vec4(1.0f));

		/* ------------------------------- initial framebuffer setup ---------------------------------- */
		this->GeometryContainer.drawBuffers({
			GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3
		});

		/* --------------------------------- initial buffer setup -------------------------------------- */
		const auto& env_obj_db = this->Pipeline.SceneComponent.EnvironmentObjectDatabase;
		const auto& sha_env_obj = this->Pipeline.SceneComponent.ShadowEnvironmentObject;
		if (sha_env_obj.size() != 1ull) {
			//Because I am too buzy to implement all features right now, add a fatal flag to remind myself in the future
			//if I attempt to do so.
			throw STPException::STPUnsupportedFunctionality("Currently the system must have one shadow-casting environment light source, "
				"please remind the maintainer to support this feature :)");
		}

		//create light spectrum handle
		this->SpectrumHandle.emplace(env_obj_db[0]->getLightSpectrum().spectrum());
		//send to shader
		this->LightingProcessor.uniform(glProgramUniformHandleui64ARB, "LightSpectrum", **this->SpectrumHandle);

		//shadow setting
		const auto& sha_env = sha_env_obj[0]->getEnvironmentLightShadow();
		const STPCascadedShadowMap::STPCascadePlane& frustum_plane = sha_env.getDivision();
		this->LightingProcessor
			.uniform(glProgramUniformHandleui64ARB, "Shadowmap", this->Pipeline.GeometryShadowPass->handle())
			.uniform(glProgramUniform1fv, "CascadePlaneDistance", static_cast<GLsizei>(frustum_plane.size()), frustum_plane.data())
			.uniform(glProgramUniform1f, "LightFrustumFar", this->Pipeline.CameraMemory->Camera.cameraStatus().Far)
			.uniform(glProgramUniform1f, "MaxBias", shadow_init.ShadowMapBias.x)
			.uniform(glProgramUniform1f, "MinBias", shadow_init.ShadowMapBias.y);
	}

	STPGeometryBufferResolution(const STPGeometryBufferResolution&) = delete;

	STPGeometryBufferResolution(STPGeometryBufferResolution&&) = delete;

	STPGeometryBufferResolution& operator=(const STPGeometryBufferResolution&) = delete;

	STPGeometryBufferResolution& operator=(STPGeometryBufferResolution&&) = delete;

	~STPGeometryBufferResolution() = default;

	/**
	 * @brief Set the resolution of all buffers.
	 * This will trigger a memory reallocation on all buffers which is extremely expensive.
	 * @param texture The pointer to the newly allocated shared texture.
	 * @param dimension The buffer resolution, which should be the size of the viewport.
	 * The resolution should have the last component as one.
	*/
	void setResolution(const STPSharedTexture& texture, const uvec3& dimension) {
		//create a set of new buffers
		STPTexture albedo(GL_TEXTURE_2D), normal(GL_TEXTURE_2D), specular(GL_TEXTURE_2D), ao(GL_TEXTURE_2D);
		const auto& [depth_stencil] = texture;
		//reallocation of memory
		albedo.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB8, dimension);
		normal.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB16_SNORM, dimension);
		specular.textureStorage<STPTexture::STPDimension::TWO>(1, GL_R8, dimension);
		ao.textureStorage<STPTexture::STPDimension::TWO>(1, GL_R8, dimension);
		//we don't need position buffer but instead of perform depth reconstruction
		//so make sure the depth buffer is solid enough to construct precise world position

		//reattach to framebuffer with multiple render targets
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT0, albedo, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT1, normal, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT2, specular, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT3, ao, 0);
		this->GeometryContainer.attach(GL_DEPTH_STENCIL_ATTACHMENT, depth_stencil, 0);

		//verify
		if (this->GeometryContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			throw STPException::STPGLError("Geometry buffer framebuffer fails to verify");
		}

		//reset bindless handle
		this->GHandle = std::move(STPGeometryBufferHandle{
			STPBindlessTexture(albedo, this->GSampler),
			STPBindlessTexture(normal, this->GSampler),
			STPBindlessTexture(specular, this->GSampler),
			STPBindlessTexture(ao, this->GSampler),
			STPBindlessTexture(depth_stencil, this->DepthSampler)
		});
		//upload new handles
		STPGeometryBufferRawHandle raw_handle;
		std::transform(this->GHandle->cbegin(), this->GHandle->cend(), raw_handle.begin(), [](const auto& handle) { return *handle; });
		this->LightingProcessor.uniform(glProgramUniformHandleui64vARB, "GBuffer", static_cast<GLsizei>(raw_handle.size()), raw_handle.data());

		using std::move;
		//store the new objects
		this->GAlbedo = move(albedo);
		this->GNormal = move(normal);
		this->GSpecular = move(specular);
		this->GAmbient = move(ao);
	}

	/**
	 * @brief Enable rendering to geometry buffer and captured under the current G-buffer resolution instance.
	 * To disable further rendering to this buffer, bind framebuffer to any other target.
	*/
	inline void capture() {
		this->GeometryContainer.bind(GL_FRAMEBUFFER);
	}

	/**
	 * @brief Perform resolution of geometry buffer and perform lighting calculation.
	*/
	inline void resolve() {
		//prepare for rendering the screen
		this->LightingProcessor.use();

		this->drawScreen();

		//clearup
		STPProgramManager::unuse();
	}

};

class STPScenePipeline::STPSceneRenderMemory {
private:

	STPTexture InputImage;
	STPFrameBuffer InputContainer;

public:

	STPSceneRenderMemory() : InputImage(GL_TEXTURE_2D) {

	}

	STPSceneRenderMemory(const STPSceneRenderMemory&) = delete;

	STPSceneRenderMemory(STPSceneRenderMemory&&) = delete;

	STPSceneRenderMemory& operator=(const STPSceneRenderMemory&) = delete;

	STPSceneRenderMemory& operator=(STPSceneRenderMemory&&) = delete;

	~STPSceneRenderMemory() = default;

	/**
	 * @brief Set the resolution of the post process framebuffer.
	 * @param texture The pointer to the shared texture memory.
	 * @param dimension The new dimension for the post process input buffer.
	 * Note that doing this will cause reallocation of all post process buffer and hence
	 * this should only be done whenever truely necessary.
	*/
	void setResolution(const STPSharedTexture& texture, const uvec3& dimension) {
		//(re)allocate memory for texture
		STPTexture imageTexture(GL_TEXTURE_2D);
		const auto& [depth_stencil] = texture;
		//using a floating-point format allows color to go beyond the standard range of [0.0, 1.0]
		imageTexture.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB16F, dimension);

		//attach to framebuffer
		this->InputContainer.attach(GL_COLOR_ATTACHMENT0, imageTexture, 0);
		this->InputContainer.attach(GL_DEPTH_STENCIL_ATTACHMENT, depth_stencil, 0);

		//verify
		if (this->InputContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			throw STPException::STPGLError("Post process framebuffer validation fails");
		}

		using std::move;
		//re-initialise the current objects
		this->InputImage = move(imageTexture);
	}

	/**
	 * @brief Activate the post process framebuffer and all rendered contents will be drawn onto the post process frame buffer.
	 * To stop capturing, bind to any other framebuffers.
	*/
	inline void capture() {
		this->InputContainer.bind(GL_FRAMEBUFFER);
	}

	/**
	 * @brief Get the pointer to the captured post process input image buffer.
	 * @return The pointer to the image buffer.
	*/
	inline const STPTexture& getImageBuffer() const {
		return this->InputImage;
	}

};

STPScenePipeline::STPScenePipeline(STPSceneInitialiser&& init, const STPCamera& camera, STPScenePipelineLog& log) :
	SceneComponent(std::move(init.InitialiserComponent)), 
	CameraMemory(make_unique<STPCameraInformationMemory>(camera)), 
	GeometryShadowPass(make_unique<STPShadowPipeline>(init)), 
	GeometryLightPass(make_unique<STPGeometryBufferResolution>(*this, init, log.GeometryBufferResolution)), 
	RenderMemory(make_unique<STPSceneRenderMemory>()) {
	//set up initial GL context states
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDepthMask(GL_TRUE);

	glDisable(GL_STENCIL_TEST);
	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

	glDisable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	glStencilMask(0xFF);
	glClearStencil(0x00);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	//tessellation settings
	//barycentric coordinate system
	glPatchParameteri(GL_PATCH_VERTICES, 3);
}

STPScenePipeline::~STPScenePipeline() = default;

void STPScenePipeline::setClearColor(vec4 color) {
	glClearColor(color.r, color.g, color.b, color.a);
	glClear(GL_COLOR_BUFFER_BIT);
}

void STPScenePipeline::setResolution(uvec2 resolution) {
	if (resolution == uvec2(0u)) {
		throw STPException::STPBadNumericRange("The rendering resolution must be both non-zero positive integers");
	}
	const uvec3 dimension = uvec3(resolution, 1u);

	//create a new scene shared buffer
	STPSharedTexture scene_texture;
	auto& [depth_stencil] = scene_texture;
	//allocation new memory, we need to allocate some floating-point pixels for (potentially) HDR rendering.
	depth_stencil.textureStorage<STPTexture::STPDimension::TWO>(1, GL_DEPTH32F_STENCIL8, dimension);

	//we pass the new buffer first before replacing the exisiting buffer in the scene pipeline
	//to make sure all children replace the new shared buffer and avoid UB
	//resize children rendering components
	STPFrameBuffer::unbind(GL_FRAMEBUFFER);
	this->GeometryLightPass->setResolution(scene_texture, dimension);
	this->RenderMemory->setResolution(scene_texture, dimension);

	using std::move;
	//store the new buffer
	this->SceneTexture = move(scene_texture);
}

void STPScenePipeline::updateLightStatus(const vec3& direction) {
	//we don't need to care if the light casts shadow
	const STPSceneLight::STPEnvironmentLight<false>& env = *this->SceneComponent.EnvironmentObjectDatabase[0];

	this->GeometryLightPass->LightingProcessor
		.uniform(glProgramUniform1f, "Lighting.SpectrumCoord", env.getLightSpectrum().coordinate())
		.uniform(glProgramUniform3fv, "LightDirection", 1, value_ptr(direction));
}

void STPScenePipeline::setLightProperty(const STPEnvironment::STPLightSetting::STPAmbientLightSetting& ambient, 
	STPEnvironment::STPLightSetting::STPDirectionalLightSetting& directional, float shiniess) {
	if (!ambient.validate() || !directional.validate()) {
		throw STPException::STPInvalidEnvironment("The light settings are not validated");
	}

	this->GeometryLightPass->LightingProcessor
		.uniform(glProgramUniform1f, "Lighting.Ka", ambient.AmbientStrength)
		.uniform(glProgramUniform1f, "Lighting.Kd", directional.DiffuseStrength)
		.uniform(glProgramUniform1f, "Lighting.Ks", directional.SpecularStrength)
		.uniform(glProgramUniform1f, "Lighting.Shin", shiniess);
}

void STPScenePipeline::traverse() {
	const auto& [object, object_shadow, env, env_shadow, post_process] = this->SceneComponent;

	//Stencil rule: the first bit denotes a fragment rendered during geometry pass in deferred rendering.

	//before rendering, update scene buffer
	this->CameraMemory->updateBuffer();
	//process rendering components.

	/* ------------------------------------------ shadow pass -------------------------------- */
	//back face culling is useful for double-sided objects
	//out scene main consists of single-sided objects and light sources may travel between front face and back face
	//it is the best to disable face culling to avoid having light bleeding
	glDisable(GL_CULL_FACE);
	this->GeometryShadowPass->renderToShadow(object_shadow, env_shadow);
	glEnable(GL_CULL_FACE);

	/* ----------------------------------------------------------------------------------------- */

	//deferred shading geometry pass
	//remember the depth and stencil buffer of geometry framebuffer and output framebuffer is shared.
	this->GeometryLightPass->capture();
	//enable clearing stencil buffer
	glStencilMask(0x01);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	/* ------------------------------------ opaque object rendering ----------------------------- */
	//initially the stencil is empty, we want to mark the geometries on the stencil buffer, no testing is needed.
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, 0x01, 0x01);
	for (const auto& rendering_object : object) {
		rendering_object->render();
	}

	//render the final scene to an internal buffer memory
	this->RenderMemory->capture();
	//no need to clear anything, just draw the quad over it.
	
	//from this step we start performing off-screen rendering using the buffer we got from previous steps.
	//off-screen rendering does not need depth test
	glDisable(GL_DEPTH_TEST);
	//preserve the original geometry depth to avoid drawing stuff over the geometries later.
	glDepthMask(GL_FALSE);
	//like the depth mask, we need to preserve geometry stencil
	glStencilMask(0x00);
	//face culling is useless for screen drawing
	glDisable(GL_CULL_FACE);
	/* ----------------------------------------- light resolve ------------------------------------ */
	//we want to start rendering light, at the same time capture everything onto the post process buffer
	//now we only want to shade pixels with geometry data, empty space should be culled
	glStencilFunc(GL_EQUAL, 0x01, 0x01);
	this->GeometryLightPass->resolve();

	/* ------------------------------------- environment rendering ----------------------------- */
	//stencil test happens before depth test, no need to worry about depth test
	//draw the environment on everything that is not geometry
	glStencilFunc(GL_NOTEQUAL, 0x01, 0x01);
	for (const auto& rendering_env : env) {
		rendering_env->renderEnvironment();
	}

	//time to draw onto the main framebuffer
	STPFrameBuffer::unbind(GL_FRAMEBUFFER);
	//depth and stencil are not written, color will be overwritten, no need to clear
	/* -------------------------------------- post processing -------------------------------- */
	glDisable(GL_STENCIL_TEST);
	//use the previously rendered buffer image for post processing
	post_process->process(this->RenderMemory->getImageBuffer());

	/* --------------------------------- reset states to defualt -------------------------------- */
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
}