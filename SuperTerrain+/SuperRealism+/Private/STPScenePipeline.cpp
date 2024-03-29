#include <SuperRealism+/STPScenePipeline.h>
#include <SuperRealism+/STPRealismInfo.h>
//Error
#include <SuperTerrain+/Exception/STPNumericDomainError.h>
#include <SuperTerrain+/Exception/STPInsufficientMemory.h>
#include <SuperTerrain+/Exception/STPUnimplementedFeature.h>
#include <SuperTerrain+/Exception/STPValidationFailed.h>
#include <SuperTerrain+/Exception/STPInvalidEnum.h>

//Base Off-screen Rendering
#include <SuperRealism+/Scene/Component/STPScreen.h>
#include <SuperRealism+/Scene/Component/STPAlphaCulling.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

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
using glm::dvec2;
using glm::vec3;
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
using std::for_each;

using namespace SuperTerrainPlus::STPRealism;

//Nothing can be more black than this...
constexpr static vec4 ConstantBlackColour = vec4(vec3(0.0f), 1.0f);

STPScenePipeline::STPShadingEquation::STPShadingEquation(const STPShadingModel model) : Model(model) {

}

STPScenePipeline::STPShadowMapFilterFunction::STPShadowMapFilterFunction(const STPShadowMapFilter filter) :
	Filter(filter), 
	DepthBias(vec2(0.0f)),
	NormalBias(vec2(0.0f)),
	BiasFarMultiplier(1.0f),
	CascadeBlendArea(0.0f) {

}

bool STPScenePipeline::STPShadowMapFilterFunction::validate() const {
	return this->DepthBias.x > this->DepthBias.y
		&& this->NormalBias.x > this->NormalBias.y;
}

template<STPShadowMapFilter Fil>
STPScenePipeline::STPShadowMapFilterKernel<Fil>::STPShadowMapFilterKernel() : STPShadowMapFilterFunction(Fil) {

}

STPScenePipeline::STPSharedTexture::STPSharedTexture() : 
	DepthStencil(GL_TEXTURE_2D) {

}

class STPScenePipeline::STPShadowPipeline {
public:

	STPShaderManager::STPShader DepthPassShader;

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

	//this shader is used to do some additional operations during depth rendering
	constexpr static auto ShadowDepthPassShaderFilename =
		STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPShadowDepthPass", ".frag");

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
			const char* const shader_source_file = STPShadowPipeline::ShadowDepthPassShaderFilename.data();
			STPShaderManager::STPShaderSource shader_source(shader_source_file, STPFile::read(shader_source_file));
			STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

			//VSM uses moments instead of regular depth value to calculate shadows
			Macro("WRITE_MOMENT", 1);

			shader_source.define(Macro);
			//compile the shader
			this->DepthPassShader = STPShaderManager::make(GL_FRAGMENT_SHADER, shader_source);

			//stores shadow map settings for this type of shadow filters
			const auto& vsm_filter = dynamic_cast<const STPShadowMapFilterKernel<STPShadowMapFilter::VSM>&>(shadow_filter);
			this->ShadowLevel = vsm_filter.mipmapLevel;
			this->ShadowAni = vsm_filter.AnisotropyFilter;
		}

		/* ----------------------------------------- light space buffer ------------------------------------------------- */
		constexpr static size_t lightSpaceSize = sizeof(STPPackLightSpaceBuffer);

		this->LightSpaceBuffer.bufferStorage(lightSpaceSize, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
		this->LightSpaceBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1u);
		//map this buffer
		this->MappedBuffer = new (this->LightSpaceBuffer.mapBufferRange(0, lightSpaceSize,
			GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT)) STPPackLightSpaceBuffer;
		STP_ASSERTION_VALIDATION(this->MappedBuffer, "Unable to map light space information buffer to shader storage buffer");
		//clear the garbage data
		memset(this->MappedBuffer, 0x00, lightSpaceSize);
	}

	STPShadowPipeline(const STPShadowPipeline&) = delete;

	STPShadowPipeline(STPShadowPipeline&&) = delete;

	STPShadowPipeline& operator=(const STPShadowPipeline&) = delete;

	STPShadowPipeline& operator=(STPShadowPipeline&&) = delete;

	~STPShadowPipeline() {
		STPBuffer::unbindBase(GL_SHADER_STORAGE_BUFFER, 1u);
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
	 * @param ori_vp The current size of the viewport.
	*/
	void renderToShadow(const vector<STPSceneObject::STPOpaqueObject*>& shadow_object, const vector<STPSceneLight*>& shadow_light, const ivec4& ori_vp) {
		size_t current_light_space_start = 0u;
		for (size_t i = 0u; i < shadow_light.size(); i++) {
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
				shadow_instance.clearShadowMapColor();
			} else {
				glClear(GL_DEPTH_BUFFER_BIT);
			}

			//change the view port to fit the shadow map
			const unsigned int shadow_extent = shadow_instance.ShadowMapResolution;
			glViewport(0, 0, shadow_extent, shadow_extent);

			//for those opaque render components (those can cast shadow), render depth
			for (const auto shadowable_object : shadow_object) {
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

class STPScenePipeline::STPGeometryBufferResolution {
private:

	//The dependent scene pipeline.
	const STPScenePipeline& Pipeline;

	typedef std::array<STPBindlessTexture::STPHandle, 5u> STPGeometryBufferHandle;
	typedef std::array<GLuint64, 5u> STPGeometryBufferRawHandle;

	STPScreen DeferredQuad;

	STPSampler GSampler, DepthSampler;
	//G-buffer components
	//The depth buffer can be used to reconstruct world position.
	//Not all buffers are present here, some of them are shared with the scene pipeline (and other rendering components to reduce memory usage)
	STPTexture GAlbedo, GNormal, GRoughness, GAmbient;
	//Material G-Buffer is only used by some rendering components, the primary deferred shader does not use it. No bindless handle required
	optional<STPTexture> GMaterial;
	optional<STPGeometryBufferHandle> GHandle;
	STPFrameBuffer GeometryContainer;

	//This object updates stencil buffer to update geometries that are in the extinction zone
	STPAlphaCulling ExtinctionStencilCuller;

	//A 1-by-1 texture of environment clear colour; can be used to draw to clear pixels belong to environment object.
	//If there is any environment present, the clear colour is set to black.
	//If there is no environment object, it is set to a user-specified clear colour.
	STPTexture ClearEnvironmentTexture;
	STPFrameBuffer ExtinctionCullingContainer;
	//Temporarily stores all environment colours before blended with the scene
	STPScreen::STPSimpleScreenFrameBuffer ExtinctionEnvironmentCache;

	constexpr static auto DeferredShaderFilename =
		STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPDeferredShading", ".frag");

	/**
	 * @brief Draw a texture onto a screen.
	 * @param texture The texture target to be drawn.
	 * @param sampler The sampler to be used.
	 * @param vp The coordinate of the screen.
	*/
	static inline void drawTextureScreen(const GLuint texture, const GLuint sampler, const vec4& vp) {
		glDrawTextureNV(texture, sampler, vp.x, vp.y, vp.z, vp.w, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f);
	}

public:

	//Holding the ambient occlusion and stencil.
	STPFrameBuffer AmbientOcclusionContainer;

	/**
	 * @brief Init a new geometry buffer resolution instance.
	 * @param pipeline The pointer to the dependent scene pipeline.
	 * @param shading_equa The pointer to the scene shading equation definer.
	 * @param shadow_filter The pointer to the scene shadow filter.
	 * @param memory_cap The pointer to the memory capacity that specifies the maximum amount of memory to be allocated in the shader.
	 * @param lighting_init The pointer to the lighting shader initialiser
	*/
	STPGeometryBufferResolution(const STPScenePipeline& pipeline, const STPShadingEquation& shading_equa,
		const STPShadowMapFilterFunction& shadow_filter, const STPScreen::STPScreenInitialiser& lighting_init) :
		Pipeline(pipeline),
		GAlbedo(GL_TEXTURE_2D), GNormal(GL_TEXTURE_2D), GRoughness(GL_TEXTURE_2D), GAmbient(GL_TEXTURE_2D),
		//alpha culling, set to discard pixels that are not in the extinction zone
		//remember 0 means no extinction whereas 1 means fully invisible
		ExtinctionStencilCuller(STPAlphaCulling::STPCullComparator::LessEqual, 0.0f,
			STPAlphaCulling::STPCullConnector::Or, STPAlphaCulling::STPCullComparator::Greater, 1.0f, lighting_init),
		ClearEnvironmentTexture(GL_TEXTURE_2D) {
		STP_ASSERTION_NUMERIC_DOMAIN(shadow_filter.validate(), "The shadow map filter has invalid values");
		STP_ASSERTION_NUMERIC_DOMAIN(shading_equa.validate(), "The shading model has invalid parameters");

		const bool cascadeLayerBlend = shadow_filter.CascadeBlendArea > 0.0f;

		//do something to the fragment shader
		const char* const lighting_source_file = DeferredShaderFilename.data();
		STPShaderManager::STPShaderSource deferred_source(lighting_source_file, STPFile::read(lighting_source_file));
		STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

		const STPSceneShaderCapacity& memory_cap = this->Pipeline.SceneMemoryLimit;
		Macro("AMBIENT_LIGHT_CAPACITY", memory_cap.AmbientLight)
			("DIRECTIONAL_LIGHT_CAPACITY", memory_cap.DirectionalLight)

			("LIGHT_SHADOW_FILTER", static_cast<std::underlying_type_t<STPShadowMapFilter>>(shadow_filter.Filter))
			("SHADOW_CASCADE_BLEND", cascadeLayerBlend ? 1 : 0)
			
			("SHADING_MODEL", static_cast<std::underlying_type_t<STPShadingModel>>(shading_equa.Model));

		deferred_source.define(Macro);

		//compile shader
		const STPShaderManager::STPShader deffered_shader = STPShaderManager::make(GL_FRAGMENT_SHADER, deferred_source);
		this->DeferredQuad.initScreenRenderer(deffered_shader, lighting_init);

		/* ------------------------------- setup G-buffer sampler ------------------------------------- */
		const auto setGBufferSampler = [](STPSampler& sampler) -> void {
			sampler.filter(GL_NEAREST, GL_NEAREST);
			sampler.wrap(GL_CLAMP_TO_BORDER);
			sampler.borderColor(ConstantBlackColour);
		};
		setGBufferSampler(this->GSampler);
		setGBufferSampler(this->DepthSampler);

		/* ------------------------------- initial framebuffer setup ---------------------------------- */
		this->GeometryContainer.drawBuffers({
			GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
			(static_cast<STPOpenGL::STPenum>(this->Pipeline.hasMaterialLibrary ? GL_COLOR_ATTACHMENT4 : GL_NONE))
		});

		/* --------------------------------- initial buffer setup -------------------------------------- */
		//global shadow setting
		this->DeferredQuad.OffScreenRenderer
			.uniform(glProgramUniform2fv, "Filter.Db", 1, value_ptr(shadow_filter.DepthBias))
			.uniform(glProgramUniform2fv, "Filter.Nb", 1, value_ptr(shadow_filter.NormalBias))
			.uniform(glProgramUniform1f, "Filter.FarBias", shadow_filter.BiasFarMultiplier);
		if (cascadeLayerBlend) {
			this->DeferredQuad.OffScreenRenderer.uniform(glProgramUniform1f, "Filter.Br", shadow_filter.CascadeBlendArea);
		}
		//send specialised filter kernel parameters based on type
		shadow_filter(this->DeferredQuad.OffScreenRenderer);
		shading_equa(this->DeferredQuad.OffScreenRenderer);

		//no colour will be written to the extinction buffer
		this->ExtinctionCullingContainer.readBuffer(GL_NONE);
		this->ExtinctionCullingContainer.drawBuffer(GL_NONE);
		//pure colour texture
		this->ClearEnvironmentTexture.textureStorage2D(1, GL_RGBA8, STPGLVector::STPsizeiVec2(1));
		this->ClearEnvironmentTexture.clearTextureImage(0, GL_RGBA, GL_FLOAT, value_ptr(ConstantBlackColour));
		this->ClearEnvironmentTexture.filter(GL_NEAREST, GL_NEAREST);
		this->ClearEnvironmentTexture.wrap(GL_REPEAT);
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
		size_t current_count = 0u;
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

		this->DeferredQuad.OffScreenRenderer.uniform(glProgramUniformui64NV, list_name.str().c_str(), light_data_addr)
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
	*/
	void setResolution(const STPSharedTexture& texture, const uvec2& dimension) {
		//create a set of new buffers
		STPTexture albedo(GL_TEXTURE_2D), normal(GL_TEXTURE_2D), roughness(GL_TEXTURE_2D), ao(GL_TEXTURE_2D);
		optional<STPTexture> material;
		const auto& [depth_stencil] = texture;
		//reallocation of memory
		albedo.textureStorage2D(1, GL_RGB8, dimension);
		normal.textureStorage2D(1, GL_RGB16_SNORM, dimension);
		roughness.textureStorage2D(1, GL_R8, dimension);
		ao.textureStorage2D(1, GL_R8, dimension);
		//we don't need position buffer but instead of perform depth reconstruction
		//so make sure the depth buffer is solid enough to construct precise world position

		//reattach to framebuffer with multiple render targets
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT0, albedo, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT1, normal, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT2, roughness, 0);
		this->GeometryContainer.attach(GL_COLOR_ATTACHMENT3, ao, 0);
		this->GeometryContainer.attach(GL_DEPTH_STENCIL_ATTACHMENT, depth_stencil, 0);
		//setup optional rendering targets
		if (this->Pipeline.hasMaterialLibrary) {
			material.emplace(GL_TEXTURE_2D);
			material->textureStorage2D(1, GL_R8UI, dimension);
			this->GeometryContainer.attach(GL_COLOR_ATTACHMENT4, *material, 0);
		}

		//a separate framebuffer with ambient occlusion attachment
		this->AmbientOcclusionContainer.attach(GL_COLOR_ATTACHMENT0, ao, 0);
		this->AmbientOcclusionContainer.attach(GL_STENCIL_ATTACHMENT, depth_stencil, 0);
		//update the stencil buffer used for extinction culling
		this->ExtinctionCullingContainer.attach(GL_STENCIL_ATTACHMENT, depth_stencil, 0);
		//environment rendering cache which stores environment colours, we definitely want to use HDR
		this->ExtinctionEnvironmentCache.setScreenBuffer(const_cast<STPTexture*>(&depth_stencil), dimension, GL_RGB16F);

		//verify
		this->GeometryContainer.validate(GL_FRAMEBUFFER);
		this->AmbientOcclusionContainer.validate(GL_FRAMEBUFFER);
		this->ExtinctionCullingContainer.validate(GL_FRAMEBUFFER);

		//reset bindless handle
		this->GHandle = std::move(STPGeometryBufferHandle{
			STPBindlessTexture::make(albedo, this->GSampler),
			STPBindlessTexture::make(normal, this->GSampler),
			STPBindlessTexture::make(roughness, this->GSampler),
			STPBindlessTexture::make(ao, this->GSampler),
			STPBindlessTexture::make(depth_stencil, this->DepthSampler)
		});
		//upload new handles
		STPGeometryBufferRawHandle raw_handle;
		std::transform(this->GHandle->cbegin(), this->GHandle->cend(), raw_handle.begin(),
			[](const auto& handle) { return handle.get(); });
		this->DeferredQuad.OffScreenRenderer.uniform(
			glProgramUniformHandleui64vARB, "GBuffer", static_cast<GLsizei>(raw_handle.size()), raw_handle.data());

		using std::move;
		//store the new objects
		this->GAlbedo = move(albedo);
		this->GNormal = move(normal);
		this->GRoughness = move(roughness);
		this->GAmbient = move(ao);
		if (material.has_value()) {
			this->GMaterial = move(material);
		}
	}

	/**
	 * @brief Set the colour of clear environment texture.
	 * @param colour The clear colour set to.
	*/
	inline void setClearEnvironmentColor(const vec4& colour) {
		this->ClearEnvironmentTexture.clearTextureImage(0, GL_RGBA, GL_FLOAT, value_ptr(colour));
	}

	/**
	 * @brief Draw a screen filled with environment clear colour. Fragment tests apply.
	 * The colour to be cleared depends on the number of environment object presented in the scene pipeline.
	 * @param vp The location of the viewport.
	*/
	inline void drawClearEnvironmentScreen(const vec4& vp) const {
		STPGeometryBufferResolution::drawTextureScreen(*this->ClearEnvironmentTexture, 0u, vp);
	}

	/**
	 * @brief Draw a screen filled with the extinction environment cache. Fragment tests apply.
	 * @param vp The location of the viewport.
	*/
	inline void drawExtinctionCacheScreen(const vec4& vp) const {
		STPGeometryBufferResolution::drawTextureScreen(*this->ExtinctionEnvironmentCache.ScreenColor, *this->GSampler, vp);
	}

	/**
	 * @brief Get the normal geometry buffer.
	 * @return The pointer to the normal geometry buffer.
	*/
	inline const STPTexture& getNormal() const {
		return this->GNormal;
	}

	/**
	 * @brief Get the material geometry buffer.
	 * @return The pointer to the material geometry buffer.
	 * Note that it is a undefined behaviour if material is unused for the current rendering pipeline.
	*/
	inline const STPTexture& getMaterial() const {
		return *this->GMaterial;
	}

	/**
	 * @brief Enable rendering to geometry buffer and captured under the current G-buffer resolution instance.
	 * To disable further rendering to this buffer, bind framebuffer to any other target.
	*/
	inline void capture() {
		this->GeometryContainer.bind(GL_FRAMEBUFFER);
	}

	/**
	 * @brief Enable rendering onto the extinction environment cache.
	*/
	inline void captureExtinctionEnv() {
		this->ExtinctionEnvironmentCache.capture();
	}

	/**
	 * @brief Enable rendering using the extinction culling framebuffer.
	 * This framebuffer only has a stencil attachment, no colour is rendered.
	*/
	inline void captureExtinctionCulling() {
		this->ExtinctionCullingContainer.bind(GL_FRAMEBUFFER);
	}

	/**
	 * @brief Perform resolution of geometry buffer and perform lighting calculation.
	*/
	inline void resolve() {
		this->DeferredQuad.drawScreen();
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
	 * @param value The float value to be set.
	*/
	inline void setFloat(const char* const name, const float value) {
		this->DeferredQuad.OffScreenRenderer.uniform(glProgramUniform1f, name, value);
	}

};

STPScenePipeline::STPScenePipeline(const STPMaterialLibrary* const mat_lib, const STPScenePipelineInitialiser& scene_init) :
	SceneMemoryCurrent{ }, SceneMemoryLimit(scene_init.ShaderCapacity),
	hasMaterialLibrary(mat_lib),
	GeometryShadowPass(make_unique<STPShadowPipeline>(*scene_init.ShadowFilter)),
	GeometryLightPass(make_unique<STPGeometryBufferResolution>(
		*this, *scene_init.ShadingModel, *scene_init.ShadowFilter, *scene_init.GeometryBufferInitialiser)), 
	DefaultClearColor(ConstantBlackColour) {
	if (this->hasMaterialLibrary) {
		//setup material library
		(**mat_lib).bindBase(GL_SHADER_STORAGE_BUFFER, 2u);
	}

	//we want to use reversed depth buffer, which is more suitable for DirectX clip volume.
	glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);

	//Multi-sampling is unnecessary in deferred shading
	glDisable(GL_MULTISAMPLE);
	//set up initial GL context states
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_GREATER);
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
	this->setClearColor(ConstantBlackColour);
	glClearStencil(0x00);
	glClearDepth(0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	//tessellation settings
	//barycentric coordinate system
	glPatchParameteri(GL_PATCH_VERTICES, 3);
}

STPScenePipeline::~STPScenePipeline() {
	//unbind camera buffer
	this->setCamera(nullptr);
	//unbind material buffer
	STPBuffer::unbindBase(GL_SHADER_STORAGE_BUFFER, 2u);
}

const STPShaderManager::STPShader* STPScenePipeline::getDepthShader() const {
	//if depth shader is not applicable, return nullptr
	return this->GeometryShadowPass->DepthPassShader ? &this->GeometryShadowPass->DepthPassShader : nullptr;
}

void STPScenePipeline::setCamera(const STPCamera* const camera) const {
	if (!camera) {
		//unbind
		STPBuffer::unbindBase(GL_SHADER_STORAGE_BUFFER, 0u);
		return;
	}
	//bind
	camera->bindCameraBuffer(GL_SHADER_STORAGE_BUFFER, 0u);
}

const STPScenePipeline::STPSceneShaderCapacity& STPScenePipeline::getMemoryUsage() const {
	return this->SceneMemoryCurrent;
}

const STPScenePipeline::STPSceneShaderCapacity& STPScenePipeline::getMemoryLimit() const {
	return this->SceneMemoryLimit;
}

void STPScenePipeline::add(STPSceneObject::STPOpaqueObject& opaque) {
	this->SceneComponent.OpaqueObjectDatabase.emplace_back(&opaque);
}

void STPScenePipeline::add(STPSceneObject::STPTransparentObject& transparent) {
	this->SceneComponent.TransparentObjectDatabase.emplace_back(&transparent);
}

void STPScenePipeline::add(STPSceneObject::STPEnvironmentObject& environment) {
	STPScenePipeline::STPSceneGraph& scene_graph = this->SceneComponent;

	if (scene_graph.EnvironmentObjectDatabase.empty()) {
		//we are going to add our first environment object, set clear colour to black.
		this->GeometryLightPass->setClearEnvironmentColor(ConstantBlackColour);
	}
	scene_graph.EnvironmentObjectDatabase.emplace_back(&environment);
}

void STPScenePipeline::add(STPSceneLight& light) {
	{
		using LT = STPSceneLight::STPLightType;
		//test if we still have enough memory to add a light.
		const STPSceneShaderCapacity& current_usage = this->SceneMemoryCurrent,
			limit_usage = this->SceneMemoryLimit;
		
		switch (light.Type) {
		case LT::Ambient:
			STP_ASSERTION_MEMORY_SUFFICIENCY(current_usage.AmbientLight, 1u, limit_usage.AmbientLight, "number of ambient light");
			break;
		case LT::Directional:
			STP_ASSERTION_MEMORY_SUFFICIENCY(current_usage.DirectionalLight, 1u, limit_usage.DirectionalLight, "number of directional light");
			break;
		default:
			throw STP_INVALID_ENUM_CREATE(light.Type, STPSceneLight::STPLightType);
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
			const STPShaderManager::STPShader* const depth_shader = this->getDepthShader();
			for (auto shadow_obj : scene_graph.ShadowOpaqueObject) {
				shadow_obj->addDepthConfiguration(newLightSpaceCount, depth_shader);
			}
		}

		scene_graph.ShadowLight.emplace_back(&light);
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

void STPScenePipeline::add(STPAmbientOcclusion& ambient_occlusion) {
	this->SceneComponent.AmbientOcclusionObject = &ambient_occlusion;
}

void STPScenePipeline::add(STPBidirectionalScattering& bsdf) {
	STP_ASSERTION_VALIDATION(this->hasMaterialLibrary, "Bidirectional scattering effect requires material data, "
		"however material library is not available in this scene pipeline instance");
	this->SceneComponent.BSDFObject = &bsdf;
}

void STPScenePipeline::add(STPPostProcess& post_process) {
	this->SceneComponent.PostProcessObject = &post_process;
}

void STPScenePipeline::addShadow(STPSceneObject::STPOpaqueObject& opaque_shadow) {
	STPScenePipeline::STPSceneGraph& scene_graph = this->SceneComponent;

	scene_graph.ShadowOpaqueObject.emplace_back(&opaque_shadow);

	//now configure this shadow-casting object with each depth configuration
	for_each(scene_graph.UniqueLightSpaceSize.cbegin(), scene_graph.UniqueLightSpaceSize.cend(),
		[&opaque_shadow, depth_shader = this->getDepthShader()]
		(const auto depth_config) { opaque_shadow.addDepthConfiguration(depth_config, depth_shader); });
}

void STPScenePipeline::setClearColor(const vec4 color) {
	glClearColor(color.r, color.g, color.b, color.a);
	//update member variable
	this->DefaultClearColor = color;
	if (this->SceneComponent.EnvironmentObjectDatabase.empty()) {
		//if there is any environment object, we should maintain black clear colour
		this->GeometryLightPass->setClearEnvironmentColor(color);
	}
}

void STPScenePipeline::setResolution(const uvec2 resolution) {
	STP_ASSERTION_NUMERIC_DOMAIN(resolution.x > 0u && resolution.y > 0u, "The rendering resolution must be both non-zero positive integers");

	//create a new scene shared buffer
	STPSharedTexture scene_texture;
	auto& [depth_stencil] = scene_texture;
	//allocation new memory, we need to allocate some floating-point pixels for (potentially) HDR rendering.
	depth_stencil.textureStorage2D(1, GL_DEPTH32F_STENCIL8, resolution);

	//we pass the new buffer first before replacing the existing buffer in the scene pipeline
	//to make sure all children replace the new shared buffer and avoid UB
	//resize children rendering components
	STPFrameBuffer::unbind(GL_FRAMEBUFFER);
	this->GeometryLightPass->setResolution(scene_texture, resolution);
	
	//update scene component (if needed)
	STPSceneGraph& scene = this->SceneComponent;
	if (scene.PostProcessObject) {
		scene.PostProcessObject->setPostProcessBuffer(&depth_stencil, resolution);
	}
	if (scene.AmbientOcclusionObject) {
		scene.AmbientOcclusionObject->setScreenSpace(&depth_stencil, resolution);
	}
	if (scene.BSDFObject) {
		scene.BSDFObject->setCopyBuffer(resolution);
	}

	using std::move;
	//store the new buffer
	this->SceneTexture = move(scene_texture);
}

void STPScenePipeline::setExtinctionArea(const float factor) const {
	STP_ASSERTION_NUMERIC_DOMAIN(factor >= 0.0f && factor <= 1.0f,
		"The extinction factor is a multiplier to far viewing distance and hence it should be a normalised value");

	this->GeometryLightPass->setFloat("ExtinctionBand", factor);
}

template<class Env>
inline void STPScenePipeline::drawEnvironment(const Env& env, const vec4& vp) const {
	//clear the environment area
	this->GeometryLightPass->drawClearEnvironmentScreen(vp);
	if (env.empty()) {
		return;
	}

	glEnable(GL_BLEND);
	//we want to sum all environment colours up while multiplying each colour by a visibility factor
	//for alpha, we mean them
	glBlendFunc(GL_CONSTANT_COLOR, GL_ONE);
	for_each(env.cbegin(), env.cend(), [alpha_mean_factor = 1.0f / static_cast<float>(env.size())](const auto rendering_env) {
		const float env_vis = rendering_env->EnvironmentVisibility;
		if (!rendering_env->isEnvironmentVisible()) {
			//invisible, skip rendering
			return;
		}

		glBlendColor(env_vis, env_vis, env_vis, alpha_mean_factor);
		rendering_env->render();
	});

	glDisable(GL_BLEND);
}

template<class Ao, class Pp>
inline void STPScenePipeline::shadeObject(const Ao* const ao, const Pp* const post_process, const unsigned char mask) const {
	//from this step we start performing off-screen rendering using the buffer we got from previous steps.
	//off-screen rendering does not need depth test
	glDisable(GL_DEPTH_TEST);

	//we need to preserve the geometry stencil
	glStencilMask(0x00);
	//we want to start rendering light, at the same time capture everything onto the post process buffer
	//now we only want to shade pixels with geometry data, empty space should be culled
	glStencilFunc(GL_EQUAL, mask, mask);

	//there is a potential feedback loop inside as the framebuffer has the depth texture attached even though we have only bound to stencil attachment point,
	//while the shader is reading from the depth texture.
	//this function makes sure the GPU flushes all texture cache.
	glTextureBarrier();
	/* ------------------------------------ screen-space ambient occlusion ------------------------------- */
	if (ao) {
		//blending is enabled for the ambient occlusion
		//output AO = calculated AO * texture AO (from the G-Buffer)
		ao->occlude(this->SceneTexture.DepthStencil, this->GeometryLightPass->getNormal(), this->GeometryLightPass->AmbientOcclusionContainer, true);
	}

	//render the final scene to an post process buffer memory
	post_process->capture();
	//no need to clear colour, we will always draw over the whole screen.
	//even if there is no environment object, we will still draw a constant colour.
	/* ----------------------------------------- light resolve ------------------------------------ */
	this->GeometryLightPass->resolve();
}

void STPScenePipeline::traverse() {
	const auto& [object, object_shadow, trans_obj, env_obj, unique_light_space_size, light_shadow, ao, bsdf, post_process] = this->SceneComponent;
	if (!post_process) {
		throw STP_UNIMPLEMENTED_FEATURE_CREATE(
			"It is currently not allowed to render to default framebuffer without post processing, "
			"because there is no stencil information written.");
	}

	/*
	 * Stencil rule
	 * 0 denotes the environment.
	 * The 1st bit denotes an object.
	 * The 2nd bit denotes if this object is transparent.
	 * The 3rd bit denotes if this object is affected by aerial perspective.
	*/
	constexpr static unsigned char EnvironmentMask = 0u,
		ObjectMask = 1u << 0u,
		TransparentMask = 1u << 1u,
		ExtinctionMask = 1u << 2u;

	//query the viewport size
	ivec4 viewport_dim;
	glGetIntegerv(GL_VIEWPORT, value_ptr(viewport_dim));
	const vec4 screen_coord = static_cast<vec4>(viewport_dim);

	//process rendering components.
	/* ------------------------------------------ shadow pass -------------------------------- */
	//back face culling is useful for double-sided objects
	//out scene main consists of single-sided objects and light sources may travel between front face and back face
	//it is the best to disable face culling to avoid having light bleeding
	this->GeometryShadowPass->renderToShadow(object_shadow, light_shadow, viewport_dim);
	glEnable(GL_CULL_FACE);

	/* ====================================== geometry rendering ================================== */
	//deferred shading geometry pass.
	//we don't need to waste time on clearing the G-Buffer except depth and stencil.
	//because we will never read pixels not shaded in the current frame as indicated by the stencil buffer,
	//nor any shading invocation in any shader reads a pixel belongs to other shading invocations (for example read neighbour pixels)
	//remember the depth and stencil buffer of geometry framebuffer and output framebuffer is shared.
	this->GeometryLightPass->capture();
	//enable stencil buffer for clearing
	glStencilMask(0xFF);
	glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	/* ------------------------------------ opaque object rendering ----------------------------- */
	//initially the stencil is empty, we want to mark the geometries on the stencil buffer, no testing is needed.
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, ObjectMask, 0xFF);
	for_each(object.cbegin(), object.cend(), [](const auto rendering_object) { rendering_object->render(); });

	/* ======================================== shading opaque objects =================================== */
	//face culling is useless for screen drawing
	glDisable(GL_CULL_FACE);
	this->shadeObject(ao, post_process, ObjectMask);

	/* -------------------------------------- environment rendering -------------------------------- */
	//draw the environment on everything that is empty
	glStencilFunc(GL_EQUAL, EnvironmentMask, 0xFF);
	this->drawEnvironment(env_obj, screen_coord);

	/* ===================================== transparent object rendering ================================= */
	if (trans_obj.empty()) {
		//don't waste time on the expensive state changes and draw calls later
		//I use go-to because it is ugly to enclosed this gigantic thing in a if statement
		goto skipTransparent;
	}
	//rendering of transparent objects behaves exactly the same as opaque objects
	if (bsdf) {
		//copy the current scene to be used later, because transparent object will overwrite existing pixels
		bsdf->copyScene(**post_process, this->SceneTexture.DepthStencil);
	}

	glEnable(GL_DEPTH_TEST);
	//all subsequent draws are for transparent objects, therefore update the bit on stencil for these objects
	glStencilMask(0xFF);
	glStencilFunc(GL_ALWAYS, TransparentMask | ObjectMask, 0xFF);
	//transparent objects do not cast shadow so there is no shadow pass
	//render them to G-Buffer directly
	this->GeometryLightPass->capture();
	for_each(trans_obj.cbegin(), trans_obj.cend(), [](const auto rendering_trans) { rendering_trans->render(); });

	/* ----------------------------------- shade transparent objects ----------------------------------- */
	//only shade newly rendered objects to the post process buffer.
	this->shadeObject(ao, post_process, TransparentMask);

	if (bsdf) {
		//render special effects for transparent objects
		//we want to blend the newly added special effects with the original colour such as lighting.
		glEnable(GL_BLEND);
		//preserve the alpha in the current buffer because it contains information about aerial perspective to be used later
		//source colour has already been pre-blended in the shader
		glBlendFuncSeparate(GL_ONE, GL_SRC_ALPHA, GL_ZERO, GL_ONE);
		bsdf->scatter(this->SceneTexture.DepthStencil, this->GeometryLightPass->getNormal(),
			this->GeometryLightPass->getMaterial());

		glDisable(GL_BLEND);
	}

	skipTransparent:
	/* =========================================== grand finale ========================================= */
	//before output, let's do some post-processing

	/* --------------------------------------- extinction culling ---------------------------------- */
	//update the stencil buffer to include objects in the extinction area such that it can be blended with the environment
	glStencilMask(ExtinctionMask);
	//cull all geometry data in the extinction zone to avoid re-computing aerial perspective for the environment pixel,
	//then set extinction mask to true
	glStencilFunc(GL_NOTEQUAL, ExtinctionMask, ObjectMask);
	//no synchronisation of the colour attachment is needed as the extinction culling is performed on another framebuffer
	
	this->GeometryLightPass->captureExtinctionCulling();
	this->GeometryLightPass->cullNonExtinction(**post_process);

	//turn off stencil writing
	glStencilMask(0x00);

	/* ------------------------------------- aerial perspective ----------------------------- */
	//render the environment again at extinction pixels onto a cache
	glStencilFunc(GL_EQUAL, ExtinctionMask, ExtinctionMask);
	//not an error: there is no need to clear the old data in the cache
	this->GeometryLightPass->captureExtinctionEnv();
	this->drawEnvironment(env_obj, screen_coord);

	//switch back to post process buffer
	post_process->capture();
	//enable extinction blending
	glEnable(GL_BLEND);
	//alpha 1 means there is no object (default alpha), or object is fully extinct
	//alpha 0 means object is fully visible
	glBlendFuncSeparate(GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_ZERO, GL_ONE);
	this->GeometryLightPass->drawExtinctionCacheScreen(screen_coord);

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
}

#define SHADOW_FILTER_NAME(FILT) STPScenePipeline::STPShadowMapFilterKernel<STPShadowMapFilter::FILT>
#define SHADOW_FILTER_CLASS(FILT) template struct STP_REALISM_API SHADOW_FILTER_NAME(FILT)
#define SHADOW_FILTER_DEF(FILT) void SHADOW_FILTER_NAME(FILT)::operator()(STPProgramManager& program) const
#define SHADOW_FILTER_VALIDATE(FILT) bool SHADOW_FILTER_NAME(FILT)::validate() const

//Explicit Instantiation of some shadow filters
SHADOW_FILTER_CLASS(Nearest);
SHADOW_FILTER_CLASS(Bilinear);

//Explicit Specialisation of some even more complicated shadow filters
SHADOW_FILTER_NAME(PCF)::STPShadowMapFilterKernel() :
	STPShadowMapFilterFunction(STPShadowMapFilter::PCF),
	KernelRadius(1u),
	KernelDistance(1.0f) {

}

SHADOW_FILTER_DEF(PCF) {
	program.uniform(glProgramUniform1ui, "Filter.Kr", this->KernelRadius)
		.uniform(glProgramUniform1f, "Filter.Ks", this->KernelDistance);
}

SHADOW_FILTER_VALIDATE(PCF) {
	return this->STPShadowMapFilterFunction::validate()
		&& this->KernelRadius > 0u
		&& this->KernelDistance > 0.0f;
}

SHADOW_FILTER_NAME(VSM)::STPShadowMapFilterKernel() :
	STPShadowMapFilterFunction(STPShadowMapFilter::VSM), 
	minVariance(0.0f),
	mipmapLevel(1u),
	AnisotropyFilter(1.0f) {

}

SHADOW_FILTER_DEF(VSM) {
	program.uniform(glProgramUniform1f, "Filter.minVar", this->minVariance);
}

SHADOW_FILTER_VALIDATE(VSM) {
	return this->STPShadowMapFilterFunction::validate();
}

#define SHADING_MODEL_NAME(MOD) STPScenePipeline::STPShadingModelDescription<STPScenePipeline::STPShadingModel::MOD>
#define SHADING_MODEL_DEF(MOD) void SHADING_MODEL_NAME(MOD)::operator()(STPProgramManager& program) const
#define SHADING_MODEL_VALIDATE(MOD) bool SHADING_MODEL_NAME(MOD)::validate() const

SHADING_MODEL_NAME(BlinnPhong)::STPShadingModelDescription() :
	STPShadingEquation(STPShadingModel::BlinnPhong), 
	RoughnessRange(vec2(0.0f, 1.0f)),
	ShininessRange(vec2(0.0f, 1.0f)) {

}

SHADING_MODEL_DEF(BlinnPhong) {
	program.uniform(glProgramUniform1f, "ShadingModel.minRough", this->RoughnessRange.x)
		.uniform(glProgramUniform1f, "ShadingModel.maxRough", this->RoughnessRange.y)
		.uniform(glProgramUniform1f, "ShadingModel.minShin", this->ShininessRange.x)
		.uniform(glProgramUniform1f, "ShadingModel.maxShin", this->ShininessRange.y);
}

SHADING_MODEL_VALIDATE(BlinnPhong) {
	return this->RoughnessRange.y > this->RoughnessRange.x
		&& this->ShininessRange.y > this->ShininessRange.x;
}