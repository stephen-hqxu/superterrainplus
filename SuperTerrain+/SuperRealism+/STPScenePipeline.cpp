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
#include <SuperRealism+/Scene/Component/STPAlphaCulling.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
//Hash
#include <SuperTerrain+/Utility/STPHashCombine.h>

//System
#include <optional>
#include <algorithm>
#include <sstream>
//Container
#include <array>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat3x4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

using glm::uvec2;
using glm::vec2;
using glm::uvec3;
using glm::vec3;
using glm::dvec3;
using glm::ivec4;
using glm::vec4;
using glm::mat3;
using glm::dmat3;
using glm::mat3x4;
using glm::mat4;
using glm::dmat4;
using glm::value_ptr;

using std::ostringstream;
using std::optional;
using std::vector;
using std::array;
using std::make_unique;

using namespace SuperTerrainPlus::STPRealism;

STPScenePipeline::STPShadowMapFilterFunction::STPShadowMapFilterFunction(STPShadowMapFilter filter) : Filter(filter), 
	DepthBias(vec2(0.0f)), NormalBias(vec2(0.0f)), BiasFarMultiplier(1.0f), CascadeBlendArea(0.0f) {

}

bool STPScenePipeline::STPShadowMapFilterFunction::valid() const {
	return this->DepthBias.x > this->DepthBias.y
		&& this->NormalBias.x > this->NormalBias.y;
}

template<STPShadowMapFilter Fil>
STPScenePipeline::STPShadowMapFilterKernel<Fil>::STPShadowMapFilterKernel() : STPShadowMapFilterFunction(Fil) {

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

	void onMove(const STPCamera&) override {
		this->updatePosition = true;
		this->updateView = true;
	}

	void onRotate(const STPCamera&) override {
		this->updateView = true;
	}

	void onReshape(const STPCamera&) override {
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
		mat3x4 VNorm;

		mat4 P, InvP, PV, InvPV;

		vec3 LDFac;
		float Far;
		bool Ortho;

	};

	static_assert(
		offsetof(STPPackedCameraBuffer, Pos) == 0
		&& offsetof(STPPackedCameraBuffer, V) == 16
		&& offsetof(STPPackedCameraBuffer, VNorm) == 80

		&& offsetof(STPPackedCameraBuffer, P) == 128
		&& offsetof(STPPackedCameraBuffer, InvP) == 192

		&& offsetof(STPPackedCameraBuffer, PV) == 256
		&& offsetof(STPPackedCameraBuffer, InvPV) == 320

		&& offsetof(STPPackedCameraBuffer, LDFac) == 384
		&& offsetof(STPPackedCameraBuffer, Far) == 396
		&& offsetof(STPPackedCameraBuffer, Ortho) == 400,
	"The alignment of camera buffer does not obey std430 packing rule");

public:

	//The master camera for the rendering scene.
	const STPCamera& Camera;
	STPBuffer Buffer;
	STPPackedCameraBuffer* MappedBuffer;

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
			reinterpret_cast<STPPackedCameraBuffer*>(this->Buffer.mapBufferRange(0, sizeof(STPPackedCameraBuffer),
				GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
		if (!this->MappedBuffer) {
			throw STPException::STPGLError("Unable to map camera buffer to shader storage buffer");
		}
		//buffer has been setup, clear the buffer before use
		memset(this->MappedBuffer, 0x00u, sizeof(STPPackedCameraBuffer));

		//setup initial values
		const STPEnvironment::STPCameraSetting& camSet = this->Camera.cameraStatus();
		const double Cnear = camSet.Near, Cfar = camSet.Far;

		this->MappedBuffer->LDFac = static_cast<vec3>(dvec3(
			2.0 * Cfar * Cnear,
			Cfar + Cnear,
			Cfar - Cnear
		));
		this->MappedBuffer->Far = static_cast<float>(Cfar);
		this->MappedBuffer->Ortho = this->Camera.ProjectionType == STPCamera::STPProjectionCategory::Orthographic;
		//update values
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
		STPPackedCameraBuffer* const camBuf = this->MappedBuffer;
		const bool PV_changed = this->updateView || this->updateProjection;

		//camera matrix is cached to avoid repetitive calculation so it is cheap to call these functions multiple times
		//only update buffer when necessary
		if (this->updatePosition || this->updateView) {
			//position has changed
			if (this->updatePosition) {
				camBuf->Pos = this->Camera.cameraStatus().Position;
				this->Buffer.flushMappedBufferRange(offsetof(STPPackedCameraBuffer, Pos), sizeof(vec3));

				this->updatePosition = false;
			}

			//if position changes, view must also change
			const dmat4& view = this->Camera.view();

			//view matrix has changed
			camBuf->V = view;
			camBuf->VNorm = static_cast<mat3x4>(glm::transpose(glm::inverse(static_cast<dmat3>(view))));
			this->Buffer.flushMappedBufferRange(offsetof(STPPackedCameraBuffer, V), sizeof(mat4) + sizeof(mat3x4));

			this->updateView = false;
		}
		if (this->updateProjection) {
			//projection matrix has changed
			const dmat4& proj = this->Camera.projection();

			camBuf->P = proj;
			camBuf->InvP = glm::inverse(proj);
			this->Buffer.flushMappedBufferRange(offsetof(STPPackedCameraBuffer, P), sizeof(mat4) * 2);

			this->updateProjection = false;
		}

		//update compound matrices
		if (PV_changed) {
			const dmat4 proj_view = this->Camera.projection() * this->Camera.view();

			//update the precomputed values
			camBuf->PV = proj_view;
			camBuf->InvPV = glm::inverse(proj_view);
			this->Buffer.flushMappedBufferRange(offsetof(STPPackedCameraBuffer, PV), sizeof(mat4) * 2);
		}
	}

};

class STPScenePipeline::STPShadowPipeline {
public:

	optional<STPShaderManager> DepthPassShader;

private:

	//The light space buffer contains data to be shared with different geometry renderer.
	STPBuffer LightSpaceBuffer;
	
	/**
	 * @brief Packed light space buffer based on std430 alignment.
	*/
	struct STPPackLightSpaceBuffer {
	public:

		GLuint64EXT LiPV;

	};
	STPPackLightSpaceBuffer* MappedBuffer;

	/**
	 * @brief Query the viewport information in the current context.
	 * @return The X, Y coordinate and the width, height of the viewport.
	*/
	static inline ivec4 getViewport() {
		ivec4 viewport;
		glGetIntegerv(GL_VIEWPORT, value_ptr(viewport));

		return viewport;
	}

	//this shader is used to do some additional operations during depth rendering
	constexpr static auto ShadowDepthPassShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPShadowDepthPass", ".frag");

	//shadow map texture properties
	GLsizei ShadowLevel = 1;
	GLfloat ShadowAni = 1.0f;

public:

	const STPShadowMapFilter ShadowFilter;
	//Denotes if this is a VSM derived shadow map filter technique
	const bool isVSMDerived;

	/**
	 * @brief Init a new STPShadowPipeline.
	 * @param shadow_filter The pointer to the scene shadow filter.
	*/
	STPShadowPipeline(const STPShadowMapFilterFunction& shadow_filter) :
		ShadowFilter(shadow_filter.Filter), isVSMDerived(this->ShadowFilter >= STPShadowMapFilter::VSM) {
		/* ------------------------------------------- depth shader setup -------------------------------------------------- */
		if (this->isVSMDerived) {
			//create a new depth shader
			this->DepthPassShader.emplace(GL_FRAGMENT_SHADER);

			const char* const shader_source_file = STPShadowPipeline::ShadowDepthPassShaderFilename.data();
			STPShaderManager::STPShaderSource shader_source(shader_source_file, *STPFile(shader_source_file));
			STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

			//VSM uses moments instead of regular depth value to calculate shadows
			Macro("WRITE_MOMENT", 1);

			shader_source.define(Macro);
			//compile the shader
			(*this->DepthPassShader)(shader_source);

			//stores shadow map settings for this type of shadow filters
			const auto& vsm_filter = dynamic_cast<const STPShadowMapFilterKernel<STPShadowMapFilter::VSM>&>(shadow_filter);
			this->ShadowLevel = vsm_filter.mipmapLevel;
			this->ShadowAni = vsm_filter.AnisotropyFilter;
		}

		/* ----------------------------------------- light space buffer ------------------------------------------------- */
		const size_t lightSpaceSize = sizeof(STPPackLightSpaceBuffer);

		this->LightSpaceBuffer.bufferStorage(lightSpaceSize, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
		this->LightSpaceBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1u);
		//map this buffer
		this->MappedBuffer = reinterpret_cast<STPPackLightSpaceBuffer*>(
			this->LightSpaceBuffer.mapBufferRange(0, lightSpaceSize,
				GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
		if (!this->MappedBuffer) {
			throw STPException::STPGLError("Unable to map light space information buffer to shader storage buffer");
		}
		//clear the garbage data
		memset(this->MappedBuffer, 0x00, lightSpaceSize);
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
	 * @param light_shadow A pointer to the shadow instance of the light.
	*/
	inline void addLight(STPLightShadow& light_shadow) {
		light_shadow.setShadowMap(this->ShadowFilter, this->ShadowLevel, this->ShadowAni);
	}

	/**
	 * @brief Render the shadow-casting objects to shadow-casting light space.
	 * @param shadow_obejct The pointer to all shadow-casting objects.
	 * @param shadow_light The pointer to all shadow-casting light.
	 * It is a undefined behaviour if any of the light is not shadow casting.
	*/
	void renderToShadow(const vector<STPSceneObject::STPOpaqueObject<true>*>& shadow_object, const vector<STPSceneLight*>& shadow_light) {
		//record the original viewport size
		const ivec4 ori_vp = STPShadowPipeline::getViewport();

		size_t current_light_space_start = 0ull;
		for (int i = 0; i < shadow_light.size(); i++) {
			auto* const shadowable_light = shadow_light[i];
			STPLightShadow& shadow_instance = *shadowable_light->getLightShadow();

			//check if the light needs to update
			if (!shadow_instance.updateLightSpace()) {
				//no update, skip this light
				continue;
			}
			const size_t current_light_space_dim = shadow_instance.lightSpaceDimension();

			//before rendering shadow map for this light, update the light space location for the current light
			//this acts as a broadcast to all shadow-casting objects to let them know which light they are dealing with.
			this->MappedBuffer->LiPV = shadow_instance.lightSpaceMatrixAddress();

			shadow_instance.captureDepth();
			//clear old values
			if (this->isVSMDerived) {
				shadow_instance.clearShadowMapColor(vec4(1.0f));
			} else {
				glClear(GL_DEPTH_BUFFER_BIT);
			}

			//change the view port to fit the shadow map
			const unsigned int shadow_extent = shadow_instance.ShadowMapResolution;
			glViewport(0, 0, shadow_extent, shadow_extent);

			//for those opaque render components (those can cast shadow), render depth
			for (auto shadowable_object : shadow_object) {
				shadowable_object->renderDepth(static_cast<unsigned int>(current_light_space_dim));
			}

			if (this->ShadowLevel > 1) {
				//shadow map has mipmap, generate
				shadow_instance.generateShadowMipmap();
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

	//The dependent scene pipeline.
	const STPScenePipeline& Pipeline;

	typedef std::array<STPBindlessTexture, 5ull> STPGeometryBufferHandle;
	typedef std::array<GLuint64, 5ull> STPGeometryBufferRawHandle;

	STPSampler GSampler, DepthSampler;
	//G-buffer components
	//The depth buffer can be used to reconstruct world position.
	//Not all buffers are present here, some of them are shared with the scene pipeline (and other rendering components to reduce memory usage)
	STPTexture GAlbedo, GNormal, GRoughness, GAmbient;
	optional<STPGeometryBufferHandle> GHandle;
	STPFrameBuffer GeometryContainer;

	//This object updates stencil buffer to update geometries that are in the extinction zone
	STPAlphaCulling ExtinctionStencilCuller;

	constexpr static auto DeferredShaderFilename =
		STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPDeferredShading", ".frag");

public:

	STPFrameBuffer AmbientOcclusionContainer, ExtinctionCullingContainer;

	/**
	 * @brief Init a new geometry buffer resolution instance.
	 * @param pipeline The pointer to the dependent scene pipeline.
	 * @param shadow_filter The pointer to the scene shadow filter.
	 * @param memory_cap The pointer to the memory capacity that specifies the maximum amount of memory to be allocated in the shader.
	 * @param lighting_init The pointer to the lighting shader initialiser
	*/
	STPGeometryBufferResolution(const STPScenePipeline& pipeline, const STPShadowMapFilterFunction& shadow_filter, 
		const STPScreenInitialiser& lighting_init) : Pipeline(pipeline),
		GAlbedo(GL_TEXTURE_2D), GNormal(GL_TEXTURE_2D), GRoughness(GL_TEXTURE_2D), GAmbient(GL_TEXTURE_2D),
		//alpha culling, set to discard pixels that are not in the extinction zone
		//remember 0 means no extinction whereas 1 means fully invisible
		ExtinctionStencilCuller(STPAlphaCulling::STPCullComparator::LessEqual, 0.0f, lighting_init) {
		const bool cascadeLayerBlend = shadow_filter.CascadeBlendArea > 0.0f;

		//do something to the fragment shader
		const char* const lighting_source_file = DeferredShaderFilename.data();
		STPShaderManager::STPShaderSource deferred_source(lighting_source_file, *STPFile(lighting_source_file));
		STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

		const STPSceneShaderCapacity& memory_cap = this->Pipeline.SceneMemoryLimit;
		Macro("AMBIENT_LIGHT_CAPACITY", memory_cap.AmbientLight)
			("DIRECTIONAL_LIGHT_CAPACITY", memory_cap.DirectionalLight)

			("LIGHT_SHADOW_FILTER", static_cast<std::underlying_type_t<STPShadowMapFilter>>(shadow_filter.Filter))
			("SHADOW_CASCADE_BLEND", cascadeLayerBlend ? 1 : 0);

		deferred_source.define(Macro);

		//compile shader
		STPShaderManager deffered_shader(GL_FRAGMENT_SHADER);
		deffered_shader(deferred_source);
		this->initScreenRenderer(deffered_shader, lighting_init);

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
		this->OffScreenRenderer
			.uniform(glProgramUniform2fv, "Filter.Db", 1, value_ptr(shadow_filter.DepthBias))
			.uniform(glProgramUniform2fv, "Filter.Nb", 1, value_ptr(shadow_filter.NormalBias))
			.uniform(glProgramUniform1f, "Filter.FarBias", shadow_filter.BiasFarMultiplier);
		if (cascadeLayerBlend) {
			this->OffScreenRenderer.uniform(glProgramUniform1f, "Filter.Br", shadow_filter.CascadeBlendArea);
		}
		//send specialised filter kernel parameters based on type
		shadow_filter(this->OffScreenRenderer);

		//no colour will be written to the extinction buffer
		this->ExtinctionCullingContainer.readBuffer(GL_NONE);
		this->ExtinctionCullingContainer.drawBuffer(GL_NONE);
	}

	STPGeometryBufferResolution(const STPGeometryBufferResolution&) = delete;

	STPGeometryBufferResolution(STPGeometryBufferResolution&&) = delete;

	STPGeometryBufferResolution& operator=(const STPGeometryBufferResolution&) = delete;

	STPGeometryBufferResolution& operator=(STPGeometryBufferResolution&&) = delete;

	~STPGeometryBufferResolution() = default;

	
	/**
	 * @brief Add a light to the rendering pipeline and flush the light information to the shader.
	 * @param light The pointer to the newly added light.
	 * This function assumes the light has yet been added to the scene graph.
	*/
	void addLight(const STPSceneLight& light) {
		//current memory usage in the shader, remember this is the current memory usage without this light being added.
		const STPSceneShaderCapacity& scene_mem_current = this->Pipeline.SceneMemoryCurrent;

		ostringstream list_name;
		const char* count_name = nullptr;
		size_t current_count = 0ull;
		const GLuint64EXT light_data_addr = light.lightDataAddress();
		//put the light into the correct bin based on its type
		switch (light.Type) {
		case STPSceneLight::STPLightType::Ambient:
			current_count = scene_mem_current.AmbientLight;
			list_name << "AmbientLightList[" << current_count << ']';
			count_name = "AmbCount";
			break;
		case STPSceneLight::STPLightType::Directional:
			current_count = scene_mem_current.DirectionalLight;
			list_name << "DirectionalLightList[" << current_count << ']';
			count_name = "DirCount";
			break;
		default:
			break;
		}

		this->OffScreenRenderer.uniform(glProgramUniformui64NV, list_name.str().c_str(), light_data_addr)
			//because we can safely assume this light has yet added to the scene, which mean after addition of this light,
			//the memory usage will be incremented by 1.
			.uniform(glProgramUniform1ui, count_name, static_cast<unsigned int>(current_count) + 1u);
	}

	/**
	 * @brief Set the resolution of all buffers.
	 * This will trigger a memory reallocation on all buffers which is extremely expensive.
	 * @param texture The pointer to the newly allocated shared texture.
	 * The old texture memory stored in the scene pipeline may not yet been updated at the time this function is called,
	 * so don't use that.
	 * @param dimension The buffer resolution, which should be the size of the viewport.
	 * The resolution should have the last component as one.
	*/
	void setResolution(const STPSharedTexture& texture, const uvec3& dimension) {
		//create a set of new buffers
		STPTexture albedo(GL_TEXTURE_2D), normal(GL_TEXTURE_2D), roughness(GL_TEXTURE_2D), ao(GL_TEXTURE_2D);
		const auto& [depth_stencil] = texture;
		//reallocation of memory
		albedo.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB8, dimension);
		normal.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB16_SNORM, dimension);
		roughness.textureStorage<STPTexture::STPDimension::TWO>(1, GL_R8, dimension);
		ao.textureStorage<STPTexture::STPDimension::TWO>(1, GL_R8, dimension);
		//we don't need position buffer but instead of perform depth reconstruction
		//so make sure the depth buffer is solid enough to construct precise world position

		//reattach to framebuffer with multiple render targets
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT0, albedo, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT1, normal, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT2, roughness, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT3, ao, 0);
		this->GeometryContainer.attach(GL_DEPTH_STENCIL_ATTACHMENT, depth_stencil, 0);

		//a separate framebuffer with ambient occlusion attachment
		this->AmbientOcclusionContainer.attach(GL_COLOR_ATTACHMENT0, ao, 0);
		//update the stencil buffer used for extinction culling
		this->ExtinctionCullingContainer.attach(GL_STENCIL_ATTACHMENT, depth_stencil, 0);

		//verify
		if (this->GeometryContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE
			|| this->AmbientOcclusionContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE
			|| this->ExtinctionCullingContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			throw STPException::STPGLError("Geometry buffer framebuffer fails to verify");
		}

		//reset bindless handle
		this->GHandle = std::move(STPGeometryBufferHandle{
			STPBindlessTexture(albedo, this->GSampler),
			STPBindlessTexture(normal, this->GSampler),
			STPBindlessTexture(roughness, this->GSampler),
			STPBindlessTexture(ao, this->GSampler),
			STPBindlessTexture(depth_stencil, this->DepthSampler)
		});
		//upload new handles
		STPGeometryBufferRawHandle raw_handle;
		std::transform(this->GHandle->cbegin(), this->GHandle->cend(), raw_handle.begin(), [](const auto& handle) { return *handle; });
		this->OffScreenRenderer.uniform(glProgramUniformHandleui64vARB, "GBuffer", static_cast<GLsizei>(raw_handle.size()), raw_handle.data());

		using std::move;
		//store the new objects
		this->GAlbedo = move(albedo);
		this->GNormal = move(normal);
		this->GRoughness = move(roughness);
		this->GAmbient = move(ao);
	}

	/**
	 * @brief Get the normal geometry buffer.
	 * @return The pointer to the normal geometry buffer.
	*/
	inline const STPTexture& getNormal() const {
		return this->GNormal;
	}

	/**
	 * @brief Clear geometry buffers that are bound to colour attachments.
	 * Depth and stencil attachments are not cleared by this function.
	*/
	inline void clearColorAttachment() {
		constexpr static vec4 ClearNormal = vec4(0.0f, 1.0f, 0.0f, 1.0f),
			ClearOne = vec4(1.0f);

		//clear geometry buffer because we cannot just clear to the clear colour buffer
		this->GeometryContainer.clearColor(0, this->Pipeline.DefaultClearColor);//albedo
		this->GeometryContainer.clearColor(1, ClearNormal);//normal
		this->GeometryContainer.clearColor(2, ClearOne);//roughness
		this->GeometryContainer.clearColor(3, ClearOne);//ambient occlusion
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
		this->ScreenVertex->bind();
		this->OffScreenRenderer.use();

		this->drawScreen();

		//clearup
		STPProgramManager::unuse();
	}

	/**
	 * @brief Perform culling for geometries that are not in the extinction area.
	 * @param colour The colour texture where the scene was rendered onto.
	*/
	inline void cullNonExtinction(const STPTexture& color) {
		this->ExtinctionStencilCuller.cull(color);
	}

	/**
	 * @brief Set a float uniform.
	 * @param name The name of the uniform.
	 * @param val The float value to be set.
	*/
	inline void setFloat(const char* name, float val) {
		this->OffScreenRenderer.uniform(glProgramUniform1f, name, val);
	}

};

STPScenePipeline::STPScenePipeline(const STPCamera& camera, STPScenePipelineInitialiser& scene_init) :
	SceneMemoryCurrent{ }, SceneMemoryLimit(scene_init.ShaderCapacity),
	CameraMemory(make_unique<STPCameraInformationMemory>(camera)), 
	GeometryShadowPass(make_unique<STPShadowPipeline>(*scene_init.ShadowFilter)),
	GeometryLightPass(make_unique<STPGeometryBufferResolution>(*this, *scene_init.ShadowFilter, *scene_init.GeometryBufferInitialiser)), 
	DefaultClearColor(vec4(vec3(0.0f), 1.0f)) {
	if (!scene_init.ShadowFilter->valid()) {
		throw STPException::STPBadNumericRange("The shadow filter has invalid settings");
	}

	//Multi-sampling is unnecessary in deferred shading
	glDisable(GL_MULTISAMPLE);
	//set up initial GL context states
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDepthMask(GL_TRUE);

	glDisable(GL_STENCIL_TEST);
	glStencilMask(0xFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

	glDisable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	glDisable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);

	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearStencil(0x00);
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	//tessellation settings
	//barycentric coordinate system
	glPatchParameteri(GL_PATCH_VERTICES, 3);
}

STPScenePipeline::~STPScenePipeline() = default;

const STPShaderManager* STPScenePipeline::getDepthShader() const {
	//if depth shader is not applicable, return nullptr
	return this->GeometryShadowPass->DepthPassShader.has_value() ? &this->GeometryShadowPass->DepthPassShader.value() : nullptr;
}

void STPScenePipeline::addLight(STPSceneLight& light) {
	{
		//test if we still have enough memory to add a light.
		const STPSceneShaderCapacity& current_usage = this->SceneMemoryCurrent,
			limit_usage = this->SceneMemoryLimit;
		switch (light.Type) {
		case STPSceneLight::STPLightType::Ambient:
			if (current_usage.AmbientLight < limit_usage.AmbientLight) {
				break;
			}
		case STPSceneLight::STPLightType::Directional:
			if (current_usage.DirectionalLight < limit_usage.DirectionalLight) {
				break;
			}
			throw STPException::STPMemoryError("The number of this type of light has reached the limit");
		default:
			break;
		}
	}

	STPSceneGraph& scene_graph = this->SceneComponent;
	STPLightShadow* const light_shadow = light.getLightShadow();
	//remember this light shadow pointer might be a null if the light does not cast shadow.
	if (light_shadow) {
		//this is a shadow-casting light
		//allocate shadow memory for this light
		this->GeometryShadowPass->addLight(*light_shadow);

		//check and see if we need to update the array that stores unique light space count
		auto& unique_light_space = scene_graph.UniqueLightSpaceSize;
		const size_t newLightSpaceCount = light_shadow->lightSpaceDimension();

		const bool isNewSize = unique_light_space.emplace(newLightSpaceCount).second;
		if (isNewSize) {
			//the new light space size is new in that array
			//we need to add this new light configuration to all shadow-casting objects
			const STPShaderManager* const depth_shader = this->getDepthShader();
			for (auto shadow_obj : scene_graph.ShadowOpaqueObject) {
				shadow_obj->addDepthConfiguration(newLightSpaceCount, depth_shader);
			}
		}
	}
	//add light to the lighting shader
	this->GeometryLightPass->addLight(light);

	//update memory usage
	STPSceneShaderCapacity& mem_usage = this->SceneMemoryCurrent;
	switch (light.Type) {
	case STPSceneLight::STPLightType::Ambient: mem_usage.AmbientLight++;
		break;
	case STPSceneLight::STPLightType::Directional: mem_usage.DirectionalLight++;
		break;
	default:
		//impossible for exhaustive enum
		break;
	}
}

const STPScenePipeline::STPSceneShaderCapacity& STPScenePipeline::getMemoryUsage() const {
	return this->SceneMemoryCurrent;
}

const STPScenePipeline::STPSceneShaderCapacity& STPScenePipeline::getMemoryLimit() const {
	return this->SceneMemoryLimit;
}

void STPScenePipeline::setClearColor(vec4 color) {
	glClearColor(color.r, color.g, color.b, color.a);
	glClear(GL_COLOR_BUFFER_BIT);
	//update member variable
	this->DefaultClearColor = color;
}

bool STPScenePipeline::setRepresentativeFragmentTest(bool val) {
	if (!GLAD_GL_NV_representative_fragment_test) {
		//does not support
		return false;
	}

	if (val) {
		glEnable(GL_REPRESENTATIVE_FRAGMENT_TEST_NV);
	} else {
		glDisable(GL_REPRESENTATIVE_FRAGMENT_TEST_NV);
	}
	return glIsEnabled(GL_REPRESENTATIVE_FRAGMENT_TEST_NV) == GL_TRUE;
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

	//we pass the new buffer first before replacing the existing buffer in the scene pipeline
	//to make sure all children replace the new shared buffer and avoid UB
	//resize children rendering components
	STPFrameBuffer::unbind(GL_FRAMEBUFFER);
	this->GeometryLightPass->setResolution(scene_texture, dimension);
	
	//update scene component (if needed)
	STPSceneGraph& scene = this->SceneComponent;
	if (scene.PostProcessObject) {
		scene.PostProcessObject->setPostProcessBuffer(&depth_stencil, resolution);
	}
	if (scene.AmbientOcclusionObject) {
		scene.AmbientOcclusionObject->setScreenSpace(&depth_stencil, resolution);
	}

	using std::move;
	//store the new buffer
	this->SceneTexture = move(scene_texture);
}

void STPScenePipeline::setExtinctionArea(float factor) const {
	if (factor < 0.0f && factor > 1.0f) {
		throw STPException::STPBadNumericRange("The extinction factor is a multiplier to far viewing distance and hence it should be a normalised value");
	}

	this->GeometryLightPass->setFloat("ExtinctionBand", factor);
}

void STPScenePipeline::traverse() {
	const auto& [object, object_shadow, trans_obj, env_obj, unique_light_space_size, light_shadow, ao, post_process] = this->SceneComponent;
	//determine the state of these optional stages
	const bool has_effect_ao = ao,
		has_effect_post_process = post_process;
	if (!has_effect_post_process) {
		throw STPException::STPUnsupportedFunctionality("It is currently not allowed to render to default framebuffer without post processing, "
			"because there is no stencil information written.");
	}

	//Stencil rule: the first bit denotes a fragment rendered during geometry pass in deferred rendering.

	//before rendering, update scene buffer
	this->CameraMemory->updateBuffer();
	//process rendering components.

	/* ------------------------------------------ shadow pass -------------------------------- */
	//back face culling is useful for double-sided objects
	//out scene main consists of single-sided objects and light sources may travel between front face and back face
	//it is the best to disable face culling to avoid having light bleeding
	glDisable(GL_CULL_FACE);
	this->GeometryShadowPass->renderToShadow(object_shadow, light_shadow);
	glEnable(GL_CULL_FACE);

	/* ====================================== geometry rendering ================================== */
	//deferred shading geometry pass. Old geometry buffer (except depth and stencil) is cleared automatically
	this->GeometryLightPass->clearColorAttachment();
	//remember the depth and stencil buffer of geometry framebuffer and output framebuffer is shared.
	this->GeometryLightPass->capture();
	//enable stencil buffer for clearing
	glStencilMask(0x01);
	glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	/* ------------------------------------ opaque object rendering ----------------------------- */
	//initially the stencil is empty, we want to mark the geometries on the stencil buffer, no testing is needed.
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, 0x01, 0x01);
	for (const auto& rendering_object : object) {
		rendering_object->render();
	}

	/* ==================================== start final scene rendering =================================== */
	//from this step we start performing off-screen rendering using the buffer we got from previous steps.
	//off-screen rendering does not need depth test
	glDisable(GL_DEPTH_TEST);
	//preserve the original geometry depth to avoid drawing stuff over the geometries later.
	glDepthMask(GL_FALSE);
	//face culling is useless for screen drawing
	glDisable(GL_CULL_FACE);

	//like the depth mask, we need to preserve geometry stencil
	glStencilMask(0x00);
	//we want to start rendering light, at the same time capture everything onto the post process buffer
	//now we only want to shade pixels with geometry data, empty space should be culled
	glStencilFunc(GL_EQUAL, 0x01, 0x01);

	//there is a potential feedback loop inside as the framebuffer has the depth texture attached even though we have only bound to stencil attachment point,
	//while the shader is reading from the depth texture.
	//this function makes sure the GPU flushes all texture cache.
	glTextureBarrier();
	/* ------------------------------------ screen-space ambient occlusion ------------------------------- */
	if (has_effect_ao) {
		//ambient occlusion results will be blended with the AO from geometry buffer
		glEnable(GL_BLEND);
		//computed AO will be multiplicative blended to the geometry AO
		//ambient occlusion does not have alpha
		glBlendFunc(GL_DST_COLOR, GL_ZERO);

		//all buffers used in AO pass are cleared to 1.0 initially, so multiplicative blending has no effect in intermediate stages as 1.0 * x = x,
		//until writing to the output geometry AO.
		ao->occlude(this->SceneTexture.DepthStencil, this->GeometryLightPass->getNormal(), this->GeometryLightPass->AmbientOcclusionContainer);

		glDisable(GL_BLEND);
	}

	//render the final scene to an post process buffer memory
	post_process->capture();
	//clear old colour data to default clear colour
	glClear(GL_COLOR_BUFFER_BIT);
	/* ----------------------------------------- light resolve ------------------------------------ */
	this->GeometryLightPass->resolve();

	/* --------------------------------------- extinction culling ---------------------------------- */
	//update the stencil buffer to include objects in the extinction area such that it can be blended with the environment
	glStencilMask(0x01);
	//cull for all geometry data, and replace to the environment stencil
	glStencilFunc(GL_NOTEQUAL, 0x00, 0x01);
	//no synchronisation of the colour attachment is needed as the extinction culling is performed on another framebuffer
	
	this->GeometryLightPass->ExtinctionCullingContainer.bind(GL_FRAMEBUFFER);
	this->GeometryLightPass->cullNonExtinction(**post_process);

	//turn off stencil writing
	glStencilMask(0x00);
	//switch back to post process buffer
	post_process->capture();

	/* ------------------------------------- environment rendering ----------------------------- */
	//stencil test happens before depth test, no need to worry about depth test
	//draw the environment on everything that is not solid geometry
	glStencilFunc(GL_NOTEQUAL, 0x01, 0x01);

	//enable extinction blending
	glEnable(GL_BLEND);
	//alpha 1 means there is no object (default alpha), or object is fully extinct
	//alpha 0 means object is fully visible
	glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_ZERO, GL_ONE);
	for (const auto& rendering_env : env_obj) {
		rendering_env->render();
	}

	glDisable(GL_BLEND);

	/* -------------------------------------- post processing -------------------------------- */
	glDisable(GL_STENCIL_TEST);
	//time to draw onto the main framebuffer
	STPFrameBuffer::unbind(GL_FRAMEBUFFER);
	//depth and stencil are not written, colour will be all overwritten, no need to clear

	//use the previously rendered buffer image for post processing
	post_process->process();

	/* --------------------------------- reset states to default -------------------------------- */
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
}

#define SHADOW_FILTER_CLASS(FILT) template struct STP_REALISM_API STPScenePipeline::STPShadowMapFilterKernel<STPShadowMapFilter::FILT>
#define SHADOW_FILTER_NAME(FILT) STPScenePipeline::STPShadowMapFilterKernel<STPShadowMapFilter::FILT>
#define SHADOW_FILTER_DEF(FILT) void SHADOW_FILTER_NAME(FILT)::operator()(STPProgramManager& program) const

//Explicit Instantiation of some shadow filters
SHADOW_FILTER_CLASS(Nearest);
SHADOW_FILTER_CLASS(Bilinear);

//Explicit Specialisation of some even more complicated shadow filters
SHADOW_FILTER_NAME(PCF)::STPShadowMapFilterKernel() : STPShadowMapFilterFunction(STPShadowMapFilter::PCF), 
	KernelRadius(1u), KernelDistance(1.0f) {

}

SHADOW_FILTER_DEF(PCF) {
	if (this->KernelRadius == 0u || this->KernelDistance <= 0.0f) {
		throw STPException::STPBadNumericRange("Both kernel radius and distance should be positive");
	}

	program.uniform(glProgramUniform1ui, "Filter.Kr", this->KernelRadius)
		.uniform(glProgramUniform1f, "Filter.Ks", this->KernelDistance);
}

SHADOW_FILTER_NAME(VSM)::STPShadowMapFilterKernel() : STPShadowMapFilterFunction(STPShadowMapFilter::VSM), 
	minVariance(0.0f), mipmapLevel(1u), AnisotropyFilter(1.0f) {

}

SHADOW_FILTER_DEF(VSM) {
	program.uniform(glProgramUniform1f, "Filter.minVar", this->minVariance);
}