#include "STPMasterRenderer.h"

#include <SuperTerrain+/World/Diversity/STPBiomeDefine.h>

//SuperRealism+ Engine
#include <SuperRealism+/STPScenePipeline.h>
#include <SuperRealism+/Scene/STPMaterialLibrary.h>
#include <SuperRealism+/Scene/Component/STPHeightfieldTerrain.h>
#include <SuperRealism+/Scene/Component/STPSun.h>
#include <SuperRealism+/Scene/Component/STPStarfield.h>
#include <SuperRealism+/Scene/Component/STPAurora.h>
#include <SuperRealism+/Scene/Component/STPWater.h>
#include <SuperRealism+/Scene/Component/STPAmbientOcclusion.h>
#include <SuperRealism+/Scene/Component/STPBidirectionalScattering.h>
#include <SuperRealism+/Scene/Component/STPPostProcess.h>
#include <SuperRealism+/Scene/Light/STPAmbientLight.h>
#include <SuperRealism+/Scene/Light/STPDirectionalLight.h>

//SuperDemo+
#include "./Helpers/STPTerrainParaLoader.h"
#include "./World/STPWorldManager.h"
#include "./World/Layers/STPAllLayers.h"
#include "./World/Biomes/STPSplatmapGenerator.h"
#include "./World/Biomes/STPBiomefieldGenerator.h"

#include <glad/glad.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/ext/matrix_transform.hpp>

//Container
#include <array>
#include <optional>
#include <string>

#include <iostream>
#include <utility>

using std::array;
using std::optional;
using std::string;

using std::make_optional;
using std::make_pair;
using std::make_unique;

using std::cerr;
using std::endl;

using glm::dvec3;
using glm::mat3;
using glm::mat4;
using glm::u8vec3;
using glm::uvec2;
using glm::uvec3;
using glm::vec2;
using glm::vec3;
using glm::vec4;

using glm::identity;
using glm::normalize;
using glm::radians;
using glm::rotate;

//INI Data
namespace INI = SuperTerrainPlus::STPAlgorithm::STPINIData;
//Renderer
namespace Rdr = SuperTerrainPlus::STPRealism;
//Environment
namespace Env = SuperTerrainPlus::STPEnvironment;
using SuperTerrainPlus::STPDiversity::Seed;

using namespace STPDemo;

class STPMasterRenderer::STPRendererData {
private:

	const INI::STPINIStorageView &engineINI, &biomeINI;

	//Generation Pipeline
	optional<STPWorldManager> WorldManager;

	//Object
	optional<Rdr::STPSun> SunRenderer;
	optional<Rdr::STPStarfield> StarfieldRenderer;
	optional<Rdr::STPAurora> AuroraRenderer;
	optional<Rdr::STPHeightfieldTerrain> TerrainRenderer;
	optional<Rdr::STPWater> WaterRenderer;
	optional<Rdr::STPAmbientOcclusion> AOEffect;
	optional<Rdr::STPBidirectionalScattering> BSDFEffect;
	optional<Rdr::STPPostProcess> FinalProcess;
	//Light
	optional<Rdr::STPAmbientLight> Skylight;
	optional<Rdr::STPDirectionalLight> Sunlight;
	optional<Rdr::STPAmbientLight> Nightlight;
	//Material
	Rdr::STPMaterialLibrary SceneMaterial;
	//Rendering Pipeline
	optional<Rdr::STPScenePipeline> RenderPipeline;

	const dvec3& ViewPosition;

	//This time record the frametime from last frame that is not enough to round up to one tick
	double FrametimeRemainer = 0.0;

	//A simple seed mixing function
	Seed CurrentSeed;

	/**
	 * @brief Get the nest seed value.
	 * @return The next need value
	*/
	inline Seed getNextSeed() noexcept {
		this->CurrentSeed = std::hash<Seed>()(this->CurrentSeed);
		return this->CurrentSeed;
	}

	/**
	 * @brief Send colour data to a light spectrum
	 * @param spectrum The spectrum to send to.
	 * @param data An array of colour data.
	 * @param size The number of element in the array.
	*/
	inline void setSpectrumData(Rdr::STPLightSpectrum& spectrum, const u8vec3* const data, const size_t size) noexcept {
		spectrum.setData(static_cast<GLsizei>(size), GL_RGB, GL_UNSIGNED_BYTE, data);
	}

public:

	//same arguments as the master renderer
	STPRendererData(const INI::STPINIStorageView& engine, const INI::STPINIStorageView& biome, Rdr::STPCamera& camera) :
		engineINI(engine), biomeINI(biome), SceneMaterial(1u), ViewPosition(camera.cameraStatus().Position),
		CurrentSeed(this->biomeINI.at("simplex").at("seed").to<Seed>()) {
		//loading terrain parameters
		const Env::STPSimplexNoiseSetting simplex = STPTerrainParaLoader::getSimplexSetting(this->biomeINI.at("simplex"));
		const Env::STPChunkSetting chunk_setting = STPTerrainParaLoader::getChunkSetting(this->engineINI.at("Generators"));
		STPTerrainParaLoader::loadBiomeParameters(this->biomeINI);
		const Env::STPHeightfieldSetting heightfield_setting = STPTerrainParaLoader::getGeneratorSetting(this->engineINI.at("2DTerrainINF"));

		//setup world manager
		try {
			this->WorldManager.emplace(string(this->biomeINI.at("").at("texture_path_prefix").String), chunk_setting, simplex);

			this->WorldManager->attachBiomeFactory<STPDemo::STPLayerChainBuilder>(chunk_setting.MapSize, simplex.Seed);
			this->WorldManager->attachDiversityGenerator<STPDemo::STPBiomefieldGenerator>
				(this->WorldManager->SharedProgram, chunk_setting.MapSize, this->biomeINI.at("").at("interpolationRadius").to<unsigned int>());
			this->WorldManager->attachTextureFactory<STPDemo::STPSplatmapGenerator>
				(this->WorldManager->SharedProgram, this->WorldManager->getTextureDatabase(), chunk_setting,
					this->engineINI.at("Global").at("Anisotropy").to<float>());

			this->WorldManager->linkProgram(chunk_setting, heightfield_setting);
			if (!this->WorldManager) {
				//do not proceed if it fails
				std::terminate();
			}
		} catch (const SuperTerrainPlus::STPException::STPParserError::STPBasic& pe) {
			//catch parser error
			cerr << pe.what() << endl;
			std::terminate();
		}

		//setup vertex shader for off-screen rendering that can be shared
		//this buffer is a shared pointer wrapper and we don't need to manage its lifetime
		const auto OffScreenVertexBuffer = std::make_shared<const Rdr::STPScreen::STPScreenVertexBuffer>();
		Rdr::STPScreen::STPScreenInitialiser screen_renderer_init;
		screen_renderer_init.SharedVertexBuffer = OffScreenVertexBuffer;

		const auto SkyboxVertexBuffer = std::make_shared<const Rdr::STPSkybox::STPSkyboxVertexBuffer>();
		Rdr::STPSkybox::STPSkyboxInitialiser skybox_renderer_init;
		skybox_renderer_init.SharedVertexBuffer = SkyboxVertexBuffer;

		Rdr::STPMaterialLibrary::STPMaterialID waterMaterialID;
		//setup scene pipeline
		//-------------------------------------------------------------------------
		{
			Rdr::STPScenePipeline::STPScenePipelineInitialiser scene_init = {};

			//initialisation
			Rdr::STPScenePipeline::STPShadingModelDescription<Rdr::STPScenePipeline::STPShadingModel::BlinnPhong> scene_shadinig_model;
			scene_init.ShadingModel = &scene_shadinig_model;
			scene_shadinig_model.RoughnessRange = vec2(0.0f, 1.0f);
			scene_shadinig_model.ShininessRange = vec2(32.0f, 128.0f);

			Rdr::STPScenePipeline::STPShadowMapFilterKernel<Rdr::STPShadowMapFilter::PCF> scene_shadow_function;
			scene_init.ShadowFilter = &scene_shadow_function;
			scene_shadow_function.DepthBias = vec2(0.055f, 0.0055f);
			scene_shadow_function.NormalBias = vec2(15.5f, 5.5f);
			scene_shadow_function.BiasFarMultiplier = 0.45f;
			scene_shadow_function.CascadeBlendArea = 40.5f;
			scene_shadow_function.KernelRadius = 4u;
			scene_shadow_function.KernelDistance = 2.45f;

			Rdr::STPScenePipeline::STPSceneShaderCapacity& scene_cap = scene_init.ShaderCapacity;
			scene_cap.AmbientLight = 2u;
			scene_cap.DirectionalLight = 1u;

			//setup material library
			auto waterMaterial = Rdr::STPMaterialLibrary::DefaultMaterial;
			waterMaterial.Opacity = 0.01f;
			waterMaterial.Reflexivity = 2.5f;
			waterMaterial.RefractiveIndex = 1.333f;
			waterMaterialID = this->SceneMaterial.add(waterMaterial);

			//construct rendering pipeline
			scene_init.GeometryBufferInitialiser = &screen_renderer_init;

			this->RenderPipeline.emplace(&this->SceneMaterial, scene_init);
			this->RenderPipeline->setCamera(&camera);
		}
		//setup environment and light
		//-------------------------------------------
		{
			//sun shadow setting
			const Rdr::STPCascadedShadowMap::STPLightFrustum shadow_frustum = {
				{
					1.0 / 16.0,
					1.0 / 3.5,
					1.0 / 1.5
				},
				0.013,
				&camera,
				234.5
			};

			//sun
			const auto [sun_setting, atmo_setting] = STPTerrainParaLoader::getSkySetting(this->engineINI.at("Sky"));
			this->SunRenderer.emplace(sun_setting, 
				make_pair(
					normalize(vec3(1.0f, -0.1f, 0.0f)),
					normalize(vec3(0.0f, 1.0f, 0.0f))
				), skybox_renderer_init);
			this->RenderPipeline->add(*this->SunRenderer);
			//setup atmosphere
			const Env::STPAtmosphereSetting& atm_setting = atmo_setting;
			this->SunRenderer->setAtmoshpere(atm_setting);
			//-------------------------------------------
			//setup the spectrum
			mat4 raySpace = identity<mat4>();
			raySpace = rotate(raySpace, radians(2.7f), normalize(vec3(vec2(0.0f), 1.0f)));
			//generate a new spectrum
			auto [sky_spec, sun_spec] = this->SunRenderer->generateSunSpectrum(8192u, static_cast<mat3>(raySpace));

			using std::move;
			//setup light
			//daylight
			this->Skylight.emplace(move(sky_spec));
			this->Sunlight.emplace(make_optional<Rdr::STPCascadedShadowMap>(2048u, shadow_frustum), move(sun_spec));
			this->RenderPipeline->add(*this->Skylight);
			this->RenderPipeline->add(*this->Sunlight);

			//night-light
			Rdr::STPLightSpectrum nightlight_spec(3u, GL_SRGB8);
			const array<glm::u8vec3, 3u> nightlight_spec_data {{
				{ 0u, 0u, 0u },
				{ 29u, 56u, 97u },
				{ 218u, 223, 247u }
			}};
			this->setSpectrumData(nightlight_spec, nightlight_spec_data.data(), nightlight_spec_data.size());

			this->Nightlight.emplace(move(nightlight_spec));
			this->RenderPipeline->add(*this->Nightlight);
		}
		{
			//starfield
			const Env::STPStarfieldSetting starfield_setting =
				STPTerrainParaLoader::getStarfieldSetting(this->engineINI.at("Night"));

			Rdr::STPLightSpectrum starfield_spec(4u, GL_SRGB8);
			const array<glm::u8vec3, 4u> starfield_spec_data {{
				{ 129u, 194u, 235u },
				{ 232u, 169u, 146u },
				{ 101u, 184u, 155u },
				{ 225u, 208u, 242u }
			}};
			this->setSpectrumData(starfield_spec, starfield_spec_data.data(), starfield_spec_data.size());

			const Rdr::STPStarfield::STPStarfieldModel starfield_model = {
				&starfield_spec,
				true
			};

			this->StarfieldRenderer.emplace(starfield_model, skybox_renderer_init);
			this->StarfieldRenderer->setStarfield(starfield_setting, static_cast<unsigned int>(this->getNextSeed()));
			this->RenderPipeline->add(*this->StarfieldRenderer);
		}
		{
			//aurora
			const Env::STPAuroraSetting aurora_setting =
				STPTerrainParaLoader::getAuroraSetting(this->engineINI.at("Night"));

			using glm::u8vec3;
			//generate the colour spectrum for aurora
			Rdr::STPLightSpectrum aurora_spec(10u, GL_SRGB8);
			std::vector<u8vec3> aurora_colour;
			aurora_colour.reserve(10u);

			//main body colour
			constexpr static u8vec3 baseColA = u8vec3(25u, 79u, 60u), baseColB = u8vec3(91u, 255u, 190u);
			//transition colour
			constexpr static u8vec3 transColA = u8vec3(99u, 196u, 182u);
			//tail colour
			constexpr static u8vec3 tailColA = u8vec3(109u, 145u, 167u), tailColB = u8vec3(90u, 100u, 129u);
			aurora_colour.insert(aurora_colour.end(), 1u, baseColA);
			aurora_colour.insert(aurora_colour.end(), 4u, baseColB);
			aurora_colour.insert(aurora_colour.end(), 1u, transColA);
			aurora_colour.insert(aurora_colour.end(), 2u, tailColA);
			aurora_colour.insert(aurora_colour.end(), 2u, tailColB);

			this->setSpectrumData(aurora_spec, aurora_colour.data(), aurora_colour.size());

			this->AuroraRenderer.emplace(std::move(aurora_spec), skybox_renderer_init);
			this->AuroraRenderer->setAurora(aurora_setting);
			this->RenderPipeline->add(*this->AuroraRenderer);
		}

		//setup solid object
		//-------------------------------------------
		float TerrainAltitude = 0.0f;
		{
			//terrain
			const Env::STPMeshSetting mesh_setting =
				STPTerrainParaLoader::getRenderingSetting(this->engineINI.at("2DTerrainINF"));
			TerrainAltitude = mesh_setting.Altitude;

			const Rdr::STPHeightfieldTerrain::STPTerrainShaderOption terrain_opt = {
				this->ViewPosition,
				uvec3(128u, 128u, 6u),
				this->getNextSeed(),
				Rdr::STPHeightfieldTerrain::STPNormalBlendingAlgorithm::BasisTransform
			};
			Env::STPTessellationSetting DepthTessSetting = mesh_setting.TessSetting;
			DepthTessSetting.MaxTessLevel *= 0.5f;

			this->TerrainRenderer.emplace(this->WorldManager->getPipeline(), terrain_opt);
			this->RenderPipeline->add(*this->TerrainRenderer);
			this->RenderPipeline->addShadow(*this->TerrainRenderer);
			//initial setup
			this->TerrainRenderer->setMesh(mesh_setting);
			this->TerrainRenderer->setDepthMeshQuality(DepthTessSetting);
		}
		{
			//water
			const Env::STPWaterSetting water_setting =
				STPTerrainParaLoader::getWaterSetting(this->engineINI.at("Water"), TerrainAltitude);

			//define water level for watery biome
			Rdr::STPWater::STPBiomeWaterLevel water_level;
			water_level[0u] = 0.5f;

			this->WaterRenderer.emplace(*this->TerrainRenderer, water_level);
			this->RenderPipeline->add(*this->WaterRenderer);
			//setup
			this->WaterRenderer->setWater(water_setting);
			this->WaterRenderer->setWaterMaterial(waterMaterialID);
		}
		//setup effects
		//-------------------------------------------
		{
			const INI::STPINISectionView& ao_section = engine.at("AmbientOcclusion");
			//blur
			Rdr::STPGaussianFilter::STPFilterKernel<Rdr::STPGaussianFilter::STPFilterVariant::BilateralFilter> blur_kernel;
			blur_kernel.Variance = ao_section.at("variance").to<double>();
			blur_kernel.SampleDistance = ao_section.at("kernel_distance").to<double>();
			blur_kernel.Radius = ao_section.at("kernel_radius").to<unsigned int>();
			blur_kernel.Sharpness = ao_section.at("sharpness").to<float>();

			//ambient occlusion
			const Env::STPOcclusionKernelSetting ao_setting = STPTerrainParaLoader::getAOSetting(ao_section);
			Rdr::STPAmbientOcclusion::STPOcclusionKernel<Rdr::STPAmbientOcclusion::STPOcclusionAlgorithm::HBAO> ao_kernel(ao_setting);
			//For SSAO
			//ao_kernel.KernelSize = ao_section.at("kernel_size").to<unsigned int>();
			//For HBAO
			ao_kernel.DirectionStep = ao_section.at("direction_step").to<unsigned int>();
			ao_kernel.RayStep = ao_section.at("ray_step").to<unsigned int>();

			this->AOEffect.emplace(ao_kernel, Rdr::STPGaussianFilter(blur_kernel, screen_renderer_init), screen_renderer_init);
			this->RenderPipeline->add(*this->AOEffect);
		}
		{
			//BSDF
			const Env::STPBidirectionalScatteringSetting bsdf_setting =
				STPTerrainParaLoader::getBSDFSetting(this->engineINI.at("Water"));

			this->BSDFEffect.emplace(screen_renderer_init);
			this->BSDFEffect->setScattering(bsdf_setting);

			this->RenderPipeline->add(*this->BSDFEffect);
		}
		{
			//post process
			Rdr::STPPostProcess::STPToneMappingDefinition<Rdr::STPPostProcess::STPToneMappingFunction::Lottes> postprocess_def;

			this->FinalProcess.emplace(postprocess_def, screen_renderer_init);
			this->RenderPipeline->add(*this->FinalProcess);
		}

		//light property setup
		Env::STPLightSetting::STPAmbientLightSetting light_ambient = { };
		light_ambient.AmbientStrength = 0.5f;
		Env::STPLightSetting::STPDirectionalLightSetting light_directional = { };
		light_directional.DiffuseStrength = 1.6f;
		light_directional.SpecularStrength = 6.5f;
		this->Skylight->setAmbient(light_ambient);
		this->Sunlight->setDirectional(light_directional);
		//trigger an initial rendering to the shadow map to avoid reading garbage data
		//if the rendering loop starts at night when sunlight has zero intensity
		this->Sunlight->setLightDirection(this->SunRenderer->sunDirection());
		light_ambient.AmbientStrength = 0.15f;
		this->Nightlight->setAmbient(light_ambient);

		//scene pipeline setup
		this->RenderPipeline->setClearColor(vec4(vec3(44.0f, 110.0f, 209.0f) / 255.0f, 1.0f));
		this->RenderPipeline->setExtinctionArea(engine.at("Sky").at("extinction_band").to<float>());
	}

	STPRendererData(const STPRendererData&) = delete;

	STPRendererData(STPRendererData&&) = delete;

	STPRendererData& operator=(const STPRendererData&) = delete;

	STPRendererData& operator=(STPRendererData&&) = delete;

	~STPRendererData() = default;

	/**
	 * @brief Main rendering functions, called every frame.
	 * @param abs_second The current frame time in second.
	 * @param delta_second The time elapsed since the last frame.
	*/
	void render(const double abs_second, const double delta_second) {
		//Update light after that many second, to avoid doing expensive update every frame.
		constexpr static double LightUpdateFrequency = 0.1;
		//update timer
		this->FrametimeRemainer += delta_second;
		const double timeGain = glm::floor(this->FrametimeRemainer / LightUpdateFrequency);

		//prepare terrain texture first (async), because this is a slow operation
		this->TerrainRenderer->setViewPosition(this->ViewPosition);

		if (timeGain > 0.0) {
			const double update_delta = timeGain * LightUpdateFrequency;
			this->FrametimeRemainer -= update_delta;

			const float sun_specUV = this->SunRenderer->spectrumCoordinate(),
				//from experiments scattering is not visible when spectrum coordinate is below this value under the current setting
				sun_visibility = glm::smoothstep(-0.03f, 0.0f, sun_specUV);

			//change the sun position
			this->SunRenderer->EnvironmentVisibility = sun_visibility;
			this->SunRenderer->updateAnimationTimer(abs_second);
			const vec3 sunDir = this->SunRenderer->sunDirection();
			const float nightLum = 1.0f - glm::smoothstep(-0.03f, 0.03f, sunDir.y);

			//update light status
			this->Skylight->setSpectrumCoordinate(sun_specUV);
			//setting light direction triggers an update to the shadow map
			//so do not update shadow map if intensity of the sun is zero, e.g., at night
			this->Sunlight->STPSceneLight::getLightShadow()->ShadowMapUpdateMask = sun_visibility > 0.0f;
			this->Sunlight->setSpectrumCoordinate(sun_specUV);
			this->Sunlight->setLightDirection(sunDir);
			//update night status
			this->Nightlight->setSpectrumCoordinate(nightLum);
			this->StarfieldRenderer->EnvironmentVisibility = nightLum;
			this->AuroraRenderer->EnvironmentVisibility = nightLum;
		}

		//update animation timer
		this->WaterRenderer->updateAnimationTimer(abs_second);
		//update night animation, if they are visible
		if (this->StarfieldRenderer->isEnvironmentVisible()) {
			this->StarfieldRenderer->updateAnimationTimer(abs_second);
		}
		if (this->AuroraRenderer->isEnvironmentVisible()) {
			this->AuroraRenderer->updateAnimationTimer(abs_second);
		}

		//render, all async operations are sync automatically
		this->RenderPipeline->traverse();
	}

	/**
	 * @brief Get the scene pipeline.
	 * @return The reference to the scene pipeline.
	*/
	inline Rdr::STPScenePipeline& scenePipeline() noexcept {
		return *this->RenderPipeline;
	}

	/**
	 * @brief Get the post process.
	 * @return The reference to the post process.
	*/
	inline Rdr::STPPostProcess& postProcess() noexcept {
		return *this->FinalProcess;
	}

};

STPMasterRenderer::STPMasterRenderer(
	const INI::STPINIStorageView& engine, const INI::STPINIStorageView& biome, Rdr::STPCamera& camera) :
	Data(make_unique<STPRendererData>(engine, biome, camera)) {
	
}

STPMasterRenderer::~STPMasterRenderer() = default;

void STPMasterRenderer::render(const double abs, const double delta) {
	this->Data->render(abs, delta);
}

void STPMasterRenderer::resize(const uvec2& res) {
	this->Data->scenePipeline().setResolution(res);
}

void STPMasterRenderer::setGamma(const float gamma) {
	this->Data->postProcess().setEffect<SuperTerrainPlus::STPRealism::STPPostProcess::STPPostEffect::Gamma>(gamma);
}