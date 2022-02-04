#include <SuperRealism+/STPScenePipeline.h>
#include <SuperRealism+/STPRealismInfo.h>
//Error
#include <SuperTerrain+/Exception/STPGLError.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>
#include <SuperTerrain+/Exception/STPMemoryError.h>

//Base Off-screen Rendering
#include <SuperRealism+/Scene/Component/STPScreen.h>
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
#include <algorithm>
#include <array>
//Container
#include <unordered_map>

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
using std::string;
using std::to_string;
using std::vector;
using std::array;
using std::unordered_map;
using std::unique_ptr;
using std::make_unique;
using std::pair;
using std::make_pair;

using namespace SuperTerrainPlus::STPRealism;

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
		mat4 PV;
		mat4 InvPV;
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
		const bool PV_changed = this->updateView || this->updateProjection;

		//only update buffer when necessary
		if (this->updatePosition || this->updateView) {
			//position has changed
			if (this->updatePosition) {
				camBuf->Pos = this->Camera.cameraStatus().Position;
				this->Buffer.flushMappedBufferRange(0, sizeof(vec3));

				this->updatePosition = false;
			}

			//if position changes, view must also change
			//view matrix has changed
			camBuf->V = this->Camera.view();
			this->Buffer.flushMappedBufferRange(16, sizeof(mat4));

			this->updateView = false;
		}
		if (this->updateProjection) {
			//projection matrix has changed
			camBuf->P = this->Camera.projection();
			this->Buffer.flushMappedBufferRange(80, sizeof(mat4));

			this->updateProjection = false;
		}
		if (PV_changed) {
			const mat4 proj_view = this->Camera.projection() * this->Camera.view();
			//update the precomputed values
			camBuf->PV = proj_view;
			camBuf->InvPV = glm::inverse(proj_view);
			this->Buffer.flushMappedBufferRange(144, sizeof(mat4) * 2);
		}
	}

};

class STPScenePipeline::STPShadowPipeline {
private:

	STPSampler LightDepthSampler;
	vector<STPTexture> LightDepthTexture;
	vector<STPBindlessTexture> LightDepthTextureHandle;
	vector<STPFrameBuffer> LightDepthContainer;

	STPBuffer LightSpaceBuffer;
	/**
	 * @brief STPLightSpaceBuffer contains pointers to the mapped light space buffer.
	*/
	struct STPLightSpaceBuffer {
	public:

		unsigned int* LightLocation = nullptr;
		//Pointers for each shadow-casting components with requested length allocated
		vector<mat4*> LightSpacePV;

	} MappedLightSpaceBuffer;

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
	 * @param light_space_limit The maximum number of light space matrix the buffer can hold.
	*/
	STPShadowPipeline(const STPSceneShadowInitialiser& shadow_init, size_t light_space_limit) {
		//setup depth sampler
		if (shadow_init.ShadowFilter == STPShadowMapFilter::Nearest) {
			this->LightDepthSampler.filter(GL_NEAREST, GL_NEAREST);
		}
		else {
			//all other filter options implies linear filtering.
			this->LightDepthSampler.filter(GL_LINEAR, GL_LINEAR);
		}
		this->LightDepthSampler.wrap(GL_CLAMP_TO_BORDER);
		this->LightDepthSampler.borderColor(vec4(1.0f));
		//setup compare function so we can use shadow sampler in the shader
		this->LightDepthSampler.compareFunction(GL_LESS);
		this->LightDepthSampler.compareMode(GL_COMPARE_REF_TO_TEXTURE);

		/* ----------------------------------------- light space buffer ------------------------------------------------- */
		//the light space size takes paddings into account
		const size_t lightSpaceSize = 16ull + sizeof(mat4) * light_space_limit;
		//setup light space information buffer
		this->LightSpaceBuffer.bufferStorage(lightSpaceSize, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT);
		this->LightSpaceBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1u);
		unsigned char* const light_space_buffer =
			reinterpret_cast<unsigned char*>(this->LightSpaceBuffer.mapBufferRange(0, lightSpaceSize,
				GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
		if (!light_space_buffer) {
			throw STPException::STPGLError("Unable to map light space information buffer to shader storage buffer");
		}
		//clear the garbage data
		memset(light_space_buffer, 0x00, lightSpaceSize);
		this->LightSpaceBuffer.flushMappedBufferRange(0, lightSpaceSize);

		auto& [light_location, light_PV] = this->MappedLightSpaceBuffer;
		light_location = reinterpret_cast<unsigned int*>(light_space_buffer);
		//the matrix is aligned to 16 byte in the shader
		light_PV.emplace_back(reinterpret_cast<mat4*>(light_space_buffer + 16));
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
	 * @brief Manage a new light that casts shadow.
	 * @param shadow_light The pointer to the an array of shadow-casting light.
	 * @return The depth bindless texture handle for this new light.
	*/
	GLuint64 addLight(const STPSceneLight::STPEnvironmentLight<true>& shadow_light) {
		/* --------------------------------------- depth texture setup ------------------------------------------------ */
		const STPLightShadow& shadow_instance = shadow_light.getLightShadow();
		const uvec3 dimension = uvec3(shadow_instance.shadowMapResolution(), shadow_instance.lightSpaceDimension());
		//allocate new depth texture for this light
		STPTexture& depth_texture = this->LightDepthTexture.emplace_back(GL_TEXTURE_2D_ARRAY);
		depth_texture.textureStorage<STPTexture::STPDimension::THREE>(1, GL_DEPTH_COMPONENT24, dimension);

		//create handle
		const GLuint64 depth_handle = *this->LightDepthTextureHandle.emplace_back(depth_texture, this->LightDepthSampler);

		/* -------------------------------------- depth texture framebuffer ------------------------------------------- */
		STPFrameBuffer& depth_container = this->LightDepthContainer.emplace_back();
		//attach the new depth texture to the framebuffer
		depth_container.attach(GL_DEPTH_ATTACHMENT, depth_texture, 0);
		//we are rendering shadow and colors are not needed.
		depth_container.drawBuffer(GL_NONE);
		depth_container.readBuffer(GL_NONE);

		if (depth_container.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			throw STPException::STPGLError("Framebuffer for capturing shadow map fails to setup");
		}

		/* ----------------------------------------- light space buffer ---------------------------------------------- */
		auto& light_PV = this->MappedLightSpaceBuffer.LightSpacePV;
		//last PV pointer indicates the start of the current one
		//the vector is guaranteed to have at least one pointer in it pointing to the start of the matrix array
		const size_t last_PV_loc = light_PV.size() - 1ull;
		//write a new pointer to indicate the start of the next pointer
		light_PV.emplace_back(light_PV[last_PV_loc] + dimension.z);

		return depth_handle;
	}

	/**
	 * @brief Render the shadow-casting objects to shadow-casting light space.
	 * @param shadow_obejct The pointer to all shadow-casting objects.
	 * @param shadow_light The pointer to all shadow-casting light.
	*/
	void renderToShadow(const vector<STPSceneObject::STPOpaqueObject<true>*>& shadow_object, const vector<STPSceneLight::STPEnvironmentLight<true>*>& shadow_light) {
		//record the original viewport size
		const ivec4 ori_vp = STPShadowPipeline::getViewport();

		size_t current_light_space_start = 0ull;
		for (int i = 0; i < shadow_light.size(); i++) {
			const auto* const shadowable_light = shadow_light[i];
			const auto& [light_location, light_PV] = this->MappedLightSpaceBuffer;
			mat4* const current_light_PV = light_PV[i];
			const STPLightShadow& shadow_instance = shadowable_light->getLightShadow();

			//check if the light needs to update
			if (!shadow_instance.updateLightSpace(current_light_PV)) {
				//no update, skip this light
				continue;
			}
			const size_t current_light_space_dim = shadow_instance.lightSpaceDimension(),
				buffer_start = 16ull + current_light_space_start * sizeof(mat4),
				current_buffer_size = current_light_space_dim * sizeof(mat4);

			//before rendering shadow map for this light, update the light space location for the current light
			//this acts as a broadcast to all shadow-casting objects to let them know which light they are dealing with.
			*light_location = static_cast<unsigned int>(current_light_space_start);
			this->LightSpaceBuffer.flushMappedBufferRange(0, sizeof(unsigned int));
			//light matrix has been updated, need to update the shadow map
			this->LightSpaceBuffer.flushMappedBufferRange(buffer_start, current_buffer_size);

			//re-render shadow map
			this->LightDepthContainer[i].bind(GL_FRAMEBUFFER);
			glClear(GL_DEPTH_BUFFER_BIT);

			//change the view port to fit the shadow map
			const uvec2 shadow_res = shadow_instance.shadowMapResolution();
			glViewport(0, 0, shadow_res.x, shadow_res.y);

			//for those opaque render components (those can cast shadow), render depth
			for (auto shadowable_object : shadow_object) {
				shadowable_object->renderDepth(static_cast<unsigned int>(current_light_space_dim));
			}

			//increment to the next light
			current_light_space_start += current_light_space_dim;
		}

		//rollback the previous viewport size
		glViewport(ori_vp.x, ori_vp.y, ori_vp.z, ori_vp.w);
	}

};

class STPScenePipeline::STPGeometryBufferResolution : private STPScreen {
private:

	//A memory pool for creating dynamic uniform name.
	string UniformNameBuffer;

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
	vector<STPBindlessTexture> SpectrumHandle;

	constexpr static auto DeferredShaderFilename =
		STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPDeferredShading", ".frag");
	
	//A constant value to be assigned to a shadow data index to indicate a non-shadow-casting light
	constexpr static unsigned int UnusedShadow = std::numeric_limits<unsigned int>::max();

	/**
	 * @brief Create a uniform name that reuses string memory.
	 * @param pre The first segment of the string.
	 * @param index The middle segment.
	 * @param post The final segment.
	 * @return A pointer to the final string, this pointer is valid until next time this function is called.
	*/
	inline const char* createUniformName(const char* pre, const string& index, const char* post) {
		this->UniformNameBuffer.clear();
		//this is valid in C++17 and later version, each sub-expression will guarantee to be evaluated in order.
		this->UniformNameBuffer.append(pre).append(index).append(post);
		return this->UniformNameBuffer.c_str();
	}

public:

	/**
	 * @brief STPLightIndexLocator is a lookup table, given an instance of light, find the index in the scene graph.
	*/
	struct STPLightIndexLocator {
	public:

		unordered_map<const STPSceneLight::STPEnvironmentLight<false>*, size_t> Env;

	} IndexLocator;

	STPProgramManager LightingProcessor;

	/**
	 * @brief Init a new geometry buffer resolution instance.
	 * @param pipeline The pointer to the dependent scene pipeline.
	 * @param shadow_init The pointer to the scene shadow initialiser.
	 * @param memory_cap The pointer to the memory capacity that specifies the maximum amount of memory to be allocated in the shader.
	 * @param log The log from compiling light pass shader.
	*/
	STPGeometryBufferResolution(const STPScenePipeline& pipeline, const STPSceneShadowInitialiser& shadow_init,
		STPScenePipelineLog::STPGeometryBufferResolutionLog& log) : Pipeline(pipeline),
		GAlbedo(GL_TEXTURE_2D), GNormal(GL_TEXTURE_2D), GSpecular(GL_TEXTURE_2D), GAmbient(GL_TEXTURE_2D) {
		//setup geometry buffer shader
		STPShaderManager screen_shader(std::move(STPGeometryBufferResolution::compileScreenVertexShader(log.QuadShader))),
			g_shader(GL_FRAGMENT_SHADER);

		//do something to the fragment shader
		const char* const lighting_source_file = DeferredShaderFilename.data();
		STPShaderManager::STPShaderSource source(lighting_source_file, *STPFile(lighting_source_file));
		STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

		const STPSceneShaderCapacity& memory_cap = this->Pipeline.SceneMemoryLimit;
		Macro("ENVIRONMENT_LIGHT_CAPACITY", memory_cap.EnvironmentLight)
			("DIRECTIONAL_LIGHT_SHADOW_CAPACITY", memory_cap.DirectionalLightShadow)
			("LIGHT_FRUSTUM_DIVISOR_CAPACITY", memory_cap.LightFrustumDivisionPlane)

			("LIGHT_SHADOW_FILTER", static_cast<std::underlying_type_t<STPShadowMapFilter>>(shadow_init.ShadowFilter))
			("UNUSED_SHADOW", STPGeometryBufferResolution::UnusedShadow);

		source.define(Macro);
		//compile shader
		log.LightingShader.Log[0] = g_shader(source);

		//attach to the G-buffer resolution program
		this->LightingProcessor
			.attach(screen_shader)
			.attach(g_shader);
		log.LightingShader.Log[1] = this->LightingProcessor.finalise();

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
		//global shadow setting
		this->LightingProcessor
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
	 * @brief Add shadow information for a shadow casting light.
	 * @param light The pointer to the newly added light.
	 * @param shadow_light The pointer to the shadow light instance, or nullptr.
	 * @param shadow_handle The bindless handle to the depth texture for this newlyt added light.
	 * For non shadow-casting light, this parameter can be null.
	*/
	void addLight(const STPSceneLight::STPEnvironmentLight<false>& light, const STPSceneLight::STPEnvironmentLight<true>* shadow_light = nullptr,
		optional<GLuint64> shadow_handle = std::nullopt) {
		const STPSceneGraph& scene_graph = this->Pipeline.SceneComponent;
		//current memory usage in the shader, remember this is the current memory usage without this light being added.
		const STPSceneShaderCapacity& scene_mem_current = this->Pipeline.SceneMemoryCurrent;
		auto& [env] = this->IndexLocator;

		//add pointer-index lookup entry
		//we can safely assumes the new light is emplaced at the back of the light object array
		env.try_emplace(&light, scene_graph.EnvironmentObjectDatabase.size() - 1ull);

		//setup light uniform
		const size_t lightLoc = scene_mem_current.EnvironmentLight;
		const string lightLoc_str = to_string(lightLoc);
		//create light spectrum handle and send to the program
		this->LightingProcessor.uniform(glProgramUniformHandleui64ARB, this->createUniformName("EnvironmentLightList[", lightLoc_str, "].LightSpectrum"),
			*this->SpectrumHandle.emplace_back(light.getLightSpectrum().spectrum()))
			.uniform(glProgramUniform1ui, "EnvLightCount", static_cast<unsigned int>(lightLoc + 1ull));

		if (shadow_light) {
			const STPCascadedShadowMap& lightShadow = shadow_light->getEnvironmentLightShadow();

			const size_t lightSpaceCount = lightShadow.lightSpaceDimension();
			//prepare shadow data for directional light
			const size_t lightShadowLoc = scene_mem_current.DirectionalLightShadow,
				frustumDivStart = scene_mem_current.LightFrustumDivisionPlane;
			const string lightShadowLoc_str = to_string(lightShadowLoc);

			this->LightingProcessor.uniform(glProgramUniform1ui,
					this->createUniformName("EnvironmentLightList[", lightLoc_str, "].DirShadowIdx"), static_cast<unsigned int>(lightShadowLoc))

				.uniform(glProgramUniformHandleui64ARB,
					this->createUniformName("DirectionalShadowList[", lightShadowLoc_str, "].CascadedShadowMap"), shadow_handle.value())
				.uniform(glProgramUniform1ui,
					//the index start from the length of the current array, basically we are doing am emplace_back operation
					this->createUniformName("DirectionalShadowList[", lightShadowLoc_str, "].LightSpaceStart"), 
					static_cast<unsigned int>(scene_mem_current.LightSpaceMatrix))
				.uniform(glProgramUniform1ui,
					this->createUniformName("DirectionalShadowList[", lightShadowLoc_str, "].DivisorStart"), 
					static_cast<unsigned int>(frustumDivStart))
				.uniform(glProgramUniform1ui,
					this->createUniformName("DirectionalShadowList[", lightShadowLoc_str, "].LightSpaceDim"), static_cast<unsigned int>(lightSpaceCount))
				.uniform(glProgramUniform1fv, 
					this->createUniformName("LightFrustumDivisor[", to_string(frustumDivStart), "]"), 
					static_cast<unsigned int>(lightSpaceCount - 1ull), lightShadow.getDivision().data());
		}
		else {
			//not shadow-casting? indicate in the shader this light does not cast shadow
			this->LightingProcessor.uniform(glProgramUniform1ui, 
				this->createUniformName("EnvironmentLightList[", lightLoc_str, "].DirShadowIdx"), STPGeometryBufferResolution::UnusedShadow);
		}
	}


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

	/**
	 * @see STPScenePipeline::setLight
	 * For spectrum coordinate, the coordinate must be provided through the data.
	*/
	template<STPScenePipeline::STPLightPropertyType Prop, typename T>
	inline void setLight(const string& index, T&& data) {
		const auto forwardedData = std::forward<T>(data);

		if constexpr (Prop == STPLightPropertyType::AmbientStrength) {
			this->LightingProcessor.uniform(glProgramUniform1f, this->createUniformName("EnvironmentLightList[", index, "].Ka"), forwardedData);
		}
		if constexpr (Prop == STPLightPropertyType::DiffuseStrength) {
			this->LightingProcessor.uniform(glProgramUniform1f, this->createUniformName("EnvironmentLightList[", index, "].Kd"), forwardedData);
		}
		if constexpr (Prop == STPLightPropertyType::SpecularStrength) {
			this->LightingProcessor.uniform(glProgramUniform1f, this->createUniformName("EnvironmentLightList[", index, "].Ks"), forwardedData);
		}
		if constexpr (Prop == STPLightPropertyType::Direction) {
			this->LightingProcessor.uniform(glProgramUniform3fv, this->createUniformName("EnvironmentLightList[", index, "].Dir"), 1, value_ptr(forwardedData));
		}
		if constexpr (Prop == STPLightPropertyType::SpectrumCoordinate) {
			this->LightingProcessor.uniform(glProgramUniform1f, this->createUniformName("EnvironmentLightList[", index, "].SpectrumCoord"), forwardedData);
		}
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
		this->InputContainer.attach(GL_STENCIL_ATTACHMENT, depth_stencil, 0);

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

STPScenePipeline::STPScenePipeline(const STPCamera& camera, const STPSceneShaderCapacity& shader_cap, 
	const STPSceneShadowInitialiser& shadow_init, STPScenePipelineLog& log) :
	SceneMemoryCurrent{ }, SceneMemoryLimit(shader_cap),
	CameraMemory(make_unique<STPCameraInformationMemory>(camera)), 
	GeometryShadowPass(make_unique<STPShadowPipeline>(shadow_init, this->SceneMemoryLimit.LightSpaceMatrix)),
	GeometryLightPass(make_unique<STPGeometryBufferResolution>(*this, shadow_init, log.GeometryBufferResolution)),
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

void STPScenePipeline::canLightBeAdded(const STPSceneLight::STPEnvironmentLight<true>* light_shadow) const {
	const STPSceneShaderCapacity& current_usage = this->SceneMemoryCurrent, 
		limit_usage = this->SceneMemoryLimit;

	//one env light takes one space for environment light list and array shadow map, each
	if (current_usage.EnvironmentLight == limit_usage.EnvironmentLight) {
		throw STPException::STPMemoryError("The number of environment light has reached the limit");
	}
	if (current_usage.DirectionalLightShadow == limit_usage.DirectionalLightShadow) {
		throw STPException::STPMemoryError("The number of shadow-casting directional light has reached the limit");
	}

	if (light_shadow) {
		const size_t lightSpaceCount = light_shadow->getLightShadow().lightSpaceDimension(),
			divisionPlaneCount = lightSpaceCount - 1ull;

		//one env light takes "light space" number of light space matrix
		if (current_usage.LightSpaceMatrix + lightSpaceCount > limit_usage.LightSpaceMatrix) {
			throw STPException::STPMemoryError("Unable to add more light because there is not enough memory to hold more light space matrix");
		}
		//and "light space" - 1 number of division plane
		if (current_usage.LightFrustumDivisionPlane + divisionPlaneCount > limit_usage.LightFrustumDivisionPlane) {
			throw STPException::STPMemoryError("Insufficient memory to hold more light frustum division plane");
		}
	}
}

void STPScenePipeline::addLight(const STPSceneLight::STPEnvironmentLight<false>& light, const STPSceneLight::STPEnvironmentLight<true>* light_shadow) {
	STPSceneGraph& scene_graph = this->SceneComponent;

	if (light_shadow) {
		//this is a shadow-casting light
		//allocate shadow memory for this light
		const GLuint64 shadow_handle = this->GeometryShadowPass->addLight(*light_shadow);

		//check and see if we need to update the array that stores unique ligit space count
		auto& unique_light_space = scene_graph.UniqueLightSpaceSize;
		const size_t newLightSpaceCount = light_shadow->getLightShadow().lightSpaceDimension();

		const auto it = lower_bound(unique_light_space.begin(), unique_light_space.end(), newLightSpaceCount);
		if (it == unique_light_space.end() || *it != newLightSpaceCount) {
			//the new light space size is new in that array
			unique_light_space.insert(it, newLightSpaceCount);

			//also we need to add this new light configuration to all shadow-casting objects
			for (auto shadow_obj : scene_graph.ShadowOpaqueObject) {
				shadow_obj->addDepthConfiguration(newLightSpaceCount);
			}
		}

		//add light settings to the lighting shader
		this->GeometryLightPass->addLight(light, light_shadow, shadow_handle);
	}
	else {
		this->GeometryLightPass->addLight(light);
	}

	//update memory usage
	STPSceneShaderCapacity& mem_usage = this->SceneMemoryCurrent;

	mem_usage.EnvironmentLight++;
	if (light_shadow) {
		const size_t lightSpaceDim = light_shadow->getLightShadow().lightSpaceDimension();

		mem_usage.DirectionalLightShadow++;
		mem_usage.LightSpaceMatrix += lightSpaceDim;
		mem_usage.LightFrustumDivisionPlane += lightSpaceDim - 1ull;
	}
}

const STPScenePipeline::STPSceneShaderCapacity& STPScenePipeline::getMemoryUsage() const {
	return this->SceneMemoryCurrent;
}

const STPScenePipeline::STPSceneShaderCapacity& STPScenePipeline::getMemoryLimit() const {
	return this->SceneMemoryLimit;
}

size_t STPScenePipeline::locateLight(const STPSceneLight::STPEnvironmentLight<false>* light) const {
	return this->GeometryLightPass->IndexLocator.Env.at(light);
}

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

//TODO: later when we have different types of light, 
//we can introduce a smarter system that helps us to determine the light type based on the range of index.
//For example when index < 10000 it is an environment light.

template<STPScenePipeline::STPLightPropertyType Prop>
void STPScenePipeline::setLight(size_t index) {
	const STPSceneLight::STPEnvironmentLight<false>& env = *this->SceneComponent.EnvironmentObjectDatabase[index];
	const string index_str = to_string(index);

	//these data can be retrieved from the scene graph directly
	if constexpr (Prop == STPLightPropertyType::SpectrumCoordinate) {
		this->GeometryLightPass->setLight<Prop>(index_str, env.getLightSpectrum().coordinate());
	}
	if constexpr (Prop == STPLightPropertyType::Direction) {
		this->GeometryLightPass->setLight<Prop>(index_str, env.lightDirection());
	}
}

template<STPScenePipeline::STPLightPropertyType Prop>
void STPScenePipeline::setLight(size_t index, float data) {
	this->GeometryLightPass->setLight<Prop>(to_string(index), data);
}

void STPScenePipeline::traverse() {
	const auto& [object, object_shadow, unique_light_space_size, env, env_shadow, post_process] = this->SceneComponent;

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

//Explicit Instantiation
#define SET_LIGHT_NO_DATA(PROP) template STP_REALISM_API void STPScenePipeline::setLight<STPScenePipeline::STPLightPropertyType::PROP>(size_t)
#define SET_LIGHT_FLOAT(PROP) template STP_REALISM_API void STPScenePipeline::setLight<STPScenePipeline::STPLightPropertyType::PROP>(size_t, float)
SET_LIGHT_FLOAT(AmbientStrength);
SET_LIGHT_FLOAT(DiffuseStrength);
SET_LIGHT_FLOAT(SpecularStrength);
SET_LIGHT_NO_DATA(SpectrumCoordinate);
SET_LIGHT_NO_DATA(Direction);