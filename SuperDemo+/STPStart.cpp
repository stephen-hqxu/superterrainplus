//SuperTerrain+ Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPInvalidSyntax.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>

//A valid GL header needs to be included before some of the rendering engine headers as required
#include <glad/glad.h>
//SuperRealism+ Engine
#include <SuperRealism+/STPRendererInitialiser.h>
#include <SuperRealism+/Utility/Camera/STPPerspectiveCamera.h>
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
//Renderer Log
#include <SuperRealism+/Utility/STPLogHandler.hpp>
//GL helper
#include <SuperRealism+/Utility/STPDebugCallback.h>
//INI Utility
#include <SuperAlgorithm+/Parser/INI/STPINIReader.h>

//SuperDemo+
#include "./Helpers/STPTerrainParaLoader.h"
#include "./World/STPWorldManager.h"
#include "./World/Layers/STPAllLayers.h"
#include "./World/Biomes/STPSplatmapGenerator.h"
#include "./World/Biomes/STPBiomefieldGenerator.h"
//Image Loader
#include "./Helpers/STPTextureStorage.h"

//External
#include <GLFW/glfw3.h>

//System
#include <iostream>
#include <optional>

//GLM
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

using std::optional;
using std::string;
using std::make_pair;
using std::make_unique;
using std::make_optional;

using std::cout;
using std::endl;
using std::cerr;

using glm::uvec2;
using glm::vec2;
using glm::dvec2;
using glm::ivec3;
using glm::uvec3;
using glm::vec3;
using glm::dvec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::rotate;
using glm::radians;
using glm::identity;
using glm::normalize;

using SuperTerrainPlus::STPAlgorithm::STPINIStorageView;
using SuperTerrainPlus::STPAlgorithm::STPINISectionView;

namespace STPStart {

	/**
	 * @brief Rendering the entire terrain scene for demo.
	*/
	class STPMasterRenderer {
	private:

		const STPINIStorageView& engineINI, &biomeINI;

		//Generation Pipeline
		optional<STPDemo::STPWorldManager> WorldManager;

		//Object
		optional<SuperTerrainPlus::STPRealism::STPSun> SunRenderer;
		optional<SuperTerrainPlus::STPRealism::STPStarfield> StarfieldRenderer;
		optional<SuperTerrainPlus::STPRealism::STPAurora> AuroraRenderer;
		optional<SuperTerrainPlus::STPRealism::STPHeightfieldTerrain<true>> TerrainRenderer;
		optional<SuperTerrainPlus::STPRealism::STPWater> WaterRenderer;
		optional<SuperTerrainPlus::STPRealism::STPAmbientOcclusion> AOEffect;
		optional<SuperTerrainPlus::STPRealism::STPBidirectionalScattering> BSDFEffect;
		optional<SuperTerrainPlus::STPRealism::STPPostProcess> FinalProcess;
		//Light
		optional<SuperTerrainPlus::STPRealism::STPAmbientLight> Skylight;
		optional<SuperTerrainPlus::STPRealism::STPDirectionalLight> Sunlight;
		optional<SuperTerrainPlus::STPRealism::STPAmbientLight> Nightlight;
		//Material
		SuperTerrainPlus::STPRealism::STPMaterialLibrary SceneMaterial;
		//Rendering Pipeline
		optional<SuperTerrainPlus::STPRealism::STPScenePipeline> RenderPipeline;

		const dvec3& ViewPosition;

		//This time record the frametime from last frame that is not enough to round up to one tick
		double FrametimeRemainer = 0.0;

		//A simple seed mixing function
		unsigned long long CurrentSeed;

		/**
		 * @brief Get the nest seed value.
		 * @return The next need value
		*/
		unsigned long long getNextSeed() {
			this->CurrentSeed = std::hash<unsigned long long>()(this->CurrentSeed);
			return this->CurrentSeed;
		}

	public:

		/**
		 * @brief Init STPMasterRenderer.
		 * @param engine The pointer to engine INI settings.
		 * @param biome The pointer to biome INI settings.
		 * @param camera The pointer to the perspective camera for the scene.
		*/
		STPMasterRenderer(const STPINIStorageView& engine, const STPINIStorageView& biome, SuperTerrainPlus::STPRealism::STPPerspectiveCamera& camera) :
			engineINI(engine), biomeINI(biome), 
			SceneMaterial(1u), ViewPosition(camera.cameraStatus().Position), 
			CurrentSeed(this->biomeINI.at("simplex").at("seed").to<unsigned long long>()) {
			using namespace SuperTerrainPlus;
			using namespace STPDemo;

			//loading terrain parameters
			STPEnvironment::STPConfiguration config;
			const STPEnvironment::STPSimplexNoiseSetting simplex = STPTerrainParaLoader::getSimplexSetting(this->biomeINI.at("simplex"));
			{
				config.ChunkSetting = STPTerrainParaLoader::getChunkSetting(this->engineINI.at("Generators"));
				STPTerrainParaLoader::loadBiomeParameters(this->biomeINI);

				const auto& chunk_setting = config.ChunkSetting;
				config.HeightfieldSetting = STPTerrainParaLoader::getGeneratorSetting(this->engineINI.at("2DTerrainINF"));

				if (!config.validate()) {
					throw STPException::STPInvalidEnvironment("Configurations are not validated");
				}
			}

			//setup world manager
			try {
				this->WorldManager.emplace(string(this->biomeINI.at("").at("texture_path_prefix")), config, simplex);
				//the old setting has been moved to the world manager, need to refresh the pointer
				const auto& chunk_setting = this->WorldManager->getWorldSetting().ChunkSetting;

				this->WorldManager->attachBiomeFactory<STPDemo::STPLayerChainBuilder>(chunk_setting.MapSize, simplex.Seed);
				this->WorldManager->attachDiversityGenerator<STPDemo::STPBiomefieldGenerator>
					(this->WorldManager->SharedProgram, chunk_setting.MapSize, this->biomeINI.at("").at("interpolationRadius").to<unsigned int>());
				this->WorldManager->attachTextureFactory<STPDemo::STPSplatmapGenerator>
					(this->WorldManager->SharedProgram, this->WorldManager->getTextureDatabase(), chunk_setting);

				this->WorldManager->linkProgram(this->engineINI.at("Global").at("Anisotropy").to<float>());
				if (!this->WorldManager) {
					//do not proceed if it fails
					std::terminate();
				}
			} catch (const STPException::STPInvalidSyntax& se) {
				//catch parser error
				cerr << se.what() << endl;
				std::terminate();
			}

			//setup GL environment
			using namespace SuperTerrainPlus::STPRealism;
			//debug callback
			if (!STPDebugCallback::support()) {
				throw STPException::STPUnsupportedFunctionality("The current GL does not support debug callback");
			}
			glEnable(GL_DEBUG_OUTPUT);
			STPDebugCallback::registerAsyncCallback(cout);
			glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, NULL, GL_FALSE);
			glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, NULL, GL_FALSE);

			//setup vertex shader for off-screen rendering that can be shared
			const STPScreen::STPScreenVertexShader ScreenVertexShader;
			const STPSkybox::STPSkyboxVertexShader SkyboxVertexShader;

			//this buffer is a shared pointer wrapper and we don't need to manage its lifetime
			const auto OffScreenVertexBuffer = std::make_shared<const STPScreen::STPScreenVertexBuffer>();
			STPScreen::STPScreenInitialiser screen_renderer_init;
			screen_renderer_init.VertexShader = &ScreenVertexShader;
			screen_renderer_init.SharedVertexBuffer = OffScreenVertexBuffer;

			const auto SkyboxVertexBuffer = std::make_shared<const STPSkybox::STPSkyboxVertexBuffer>();
			STPSkybox::STPSkyboxInitialiser skybox_renderer_init;
			skybox_renderer_init.VertexShader = &SkyboxVertexShader;
			skybox_renderer_init.SharedVertexBuffer = SkyboxVertexBuffer;

			STPMaterialLibrary::STPMaterialID waterMaterialID;
			//setup scene pipeline
			//-------------------------------------------------------------------------
			{
				STPScenePipeline::STPScenePipelineInitialiser scene_init;

				//initialisation
				STPScenePipeline::STPShadingModelDescription<STPScenePipeline::STPShadingModel::BlinnPhong> scene_shadinig_model;
				scene_init.ShadingModel = &scene_shadinig_model;
				scene_shadinig_model.RoughnessRange = vec2(0.0f, 1.0f);
				scene_shadinig_model.ShininessRange = vec2(32.0f, 128.0f);

				STPScenePipeline::STPShadowMapFilterKernel<STPShadowMapFilter::PCF> scene_shadow_function;
				scene_init.ShadowFilter = &scene_shadow_function;
				scene_shadow_function.DepthBias = vec2(0.055f, 0.0055f);
				scene_shadow_function.NormalBias = vec2(15.5f, 5.5f);
				scene_shadow_function.BiasFarMultiplier = 0.45f;
				scene_shadow_function.CascadeBlendArea = 40.5f;
				scene_shadow_function.KernelRadius = 4u;
				scene_shadow_function.KernelDistance = 2.45f;

				STPScenePipeline::STPSceneShaderCapacity& scene_cap = scene_init.ShaderCapacity;
				scene_cap.AmbientLight = 2ull;
				scene_cap.DirectionalLight = 1ull;

				//setup material library
				auto waterMaterial = STPMaterialLibrary::DefaultMaterial;
				waterMaterial.Opacity = 1.0f;
				waterMaterial.Reflexivity = 2.5f;
				waterMaterial.RefractiveIndex = 1.333f;
				waterMaterialID = this->SceneMaterial.add(waterMaterial);

				//construct rendering pipeline
				scene_init.GeometryBufferInitialiser = &screen_renderer_init;

				this->RenderPipeline.emplace(camera, &this->SceneMaterial, scene_init);
			}
			//setup environment and light
			//-------------------------------------------
			{
				//sun shadow setting
				const double camFar = camera.cameraStatus().Far;
				const STPCascadedShadowMap::STPLightFrustum shadow_frustum = {
					{
						camFar / 16.0,
						camFar / 3.5,
						camFar / 1.5
					},
					32.5,
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
				const STPEnvironment::STPAtmosphereSetting& atm_setting = atmo_setting;
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
				this->Sunlight.emplace(make_optional<STPCascadedShadowMap>(2048u, shadow_frustum), move(sun_spec));
				this->RenderPipeline->add(*this->Skylight);
				this->RenderPipeline->add(*this->Sunlight);

				//night-light
				STPLightSpectrum nightlight_spec(3u, GL_RGB8);
				nightlight_spec.setData(STPLightSpectrum::STPColourArray<glm::u8vec3> {
					{  0u, 0u, 0u },
					{ 29u, 56u,	97u },
					{ 218u, 223, 247u }
				});

				this->Nightlight.emplace(move(nightlight_spec));
				this->RenderPipeline->add(*this->Nightlight);
			}
			{
				//starfield
				const STPEnvironment::STPStarfieldSetting starfield_setting =
					STPTerrainParaLoader::getStarfieldSetting(this->engineINI.at("Night"));

				STPLightSpectrum starfield_spec(4u, GL_RGB8);
				starfield_spec.setData(STPLightSpectrum::STPColourArray<glm::u8vec3> {
					{ 129u, 194u, 235u },
					{ 232u, 169u, 146u },
					{ 101u, 184u, 155u },
					{ 225u, 208u, 242u }
				});

				const STPStarfield::STPStarfieldModel starfield_model = {
					&starfield_spec,
					true
				};

				this->StarfieldRenderer.emplace(starfield_model, skybox_renderer_init);
				this->StarfieldRenderer->setStarfield(starfield_setting, static_cast<unsigned int>(this->getNextSeed()));
				this->RenderPipeline->add(*this->StarfieldRenderer);
			}
			{
				//aurora
				const STPEnvironment::STPAuroraSetting aurora_setting =
					STPTerrainParaLoader::getAuroraSetting(this->engineINI.at("Night"));

				using glm::u8vec3;
				//generate the colour spectrum for aurora
				STPLightSpectrum aurora_spec(10u, GL_SRGB8);
				STPLightSpectrum::STPColourArray<u8vec3> aurora_colour;
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

				aurora_spec.setData(aurora_colour);

				this->AuroraRenderer.emplace(std::move(aurora_spec), skybox_renderer_init);
				this->AuroraRenderer->setAurora(aurora_setting);
				this->RenderPipeline->add(*this->AuroraRenderer);
			}

			//setup solid object
			//-------------------------------------------
			float TerrainAltitude = 0.0f;
			{
				//terrain
				const STPEnvironment::STPMeshSetting mesh_setting =
					STPTerrainParaLoader::getRenderingSetting(this->engineINI.at("2DTerrainINF"));
				TerrainAltitude = mesh_setting.Altitude;

				const STPHeightfieldTerrain<true>::STPTerrainShaderOption terrain_opt = {
					this->ViewPosition,
					uvec3(128u, 128u, 6u),
					STPHeightfieldTerrain<true>::STPNormalBlendingAlgorithm::BasisTransform
				};
				STPEnvironment::STPTessellationSetting DepthTessSetting = mesh_setting.TessSetting;
				DepthTessSetting.MaxTessLevel *= 0.5f;

				this->TerrainRenderer.emplace(this->WorldManager->getPipeline(), terrain_opt);
				this->RenderPipeline->add(*this->TerrainRenderer, *this->TerrainRenderer);
				//initial setup
				this->TerrainRenderer->setMesh(mesh_setting);
				this->TerrainRenderer->setDepthMeshQuality(DepthTessSetting);
				this->TerrainRenderer->seedRandomBuffer(this->getNextSeed());
			}
			{
				//water
				const STPEnvironment::STPWaterSetting water_setting =
					STPTerrainParaLoader::getWaterSetting(this->engineINI.at("Water"), TerrainAltitude);

				//define water level for watery biome
				STPWater::STPBiomeWaterLevel water_level;
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
				const STPINISectionView& ao_section = engine.at("AmbientOcclusion");
				//blur
				STPGaussianFilter::STPFilterKernel<STPGaussianFilter::STPFilterVariant::BilateralFilter> blur_kernel;
				blur_kernel.Variance = ao_section.at("variance").to<double>();
				blur_kernel.SampleDistance = ao_section.at("kernel_distance").to<double>();
				blur_kernel.Radius = ao_section.at("kernel_radius").to<unsigned int>();
				blur_kernel.Sharpness = ao_section.at("sharpness").to<float>();

				//ambient occlusion
				const STPEnvironment::STPOcclusionKernelSetting ao_setting = STPTerrainParaLoader::getAOSetting(ao_section);
				STPAmbientOcclusion::STPOcclusionKernel<STPAmbientOcclusion::STPOcclusionAlgorithm::HBAO> ao_kernel(ao_setting);
				//For SSAO
				//ao_kernel.KernelSize = ao_section.at("kernel_size").to<unsigned int>();
				//For HBAO
				ao_kernel.DirectionStep = ao_section.at("direction_step").to<unsigned int>();
				ao_kernel.RayStep = ao_section.at("ray_step").to<unsigned int>();

				this->AOEffect.emplace(ao_kernel, STPGaussianFilter(blur_kernel, screen_renderer_init), screen_renderer_init);
				this->RenderPipeline->add(*this->AOEffect);
			}
			{
				//BSDF
				const STPEnvironment::STPBidirectionalScatteringSetting bsdf_setting =
					STPTerrainParaLoader::getBSDFSetting(this->engineINI.at("Water"));

				this->BSDFEffect.emplace(screen_renderer_init);
				this->BSDFEffect->setScattering(bsdf_setting);

				this->RenderPipeline->add(*this->BSDFEffect);
			}
			{
				//post process
				STPPostProcess::STPToneMappingDefinition<STPPostProcess::STPToneMappingFunction::Lottes> postprocess_def;

				this->FinalProcess.emplace(postprocess_def, screen_renderer_init);
				this->RenderPipeline->add(*this->FinalProcess);
			}

			//light property setup
			STPEnvironment::STPLightSetting::STPAmbientLightSetting light_ambient;
			light_ambient.AmbientStrength = 0.5f;
			STPEnvironment::STPLightSetting::STPDirectionalLightSetting light_directional;
			light_directional.DiffuseStrength = 1.6f;
			light_directional.SpecularStrength = 6.5f;
			this->Skylight->setAmbient(light_ambient);
			this->Sunlight->setDirectional(light_directional);
			light_ambient.AmbientStrength = 0.15f;
			this->Nightlight->setAmbient(light_ambient);

			//scene pipeline setup
			this->RenderPipeline->setClearColor(vec4(vec3(44.0f, 110.0f, 209.0f) / 255.0f, 1.0f));
			this->RenderPipeline->setExtinctionArea(engine.at("Sky").at("extinction_band").to<float>());
			if (this->RenderPipeline->setRepresentativeFragmentTest(true)) {
				cout << "GL_NV_representative_fragment_test is available for the current GL renderer and has been enabled." << endl;
			}
		}

		STPMasterRenderer(const STPMasterRenderer&) = delete;

		STPMasterRenderer(STPMasterRenderer&&) = delete;

		STPMasterRenderer& operator=(const STPMasterRenderer&) = delete;

		STPMasterRenderer& operator=(STPMasterRenderer&&) = delete;

		~STPMasterRenderer() = default;

		/**
		 * @brief Main rendering functions, called every frame.
		 * @param delta The time difference from the last frame.
		*/
		inline void render(double delta) {
			//Update light after that many second, to avoid doing expensive update every frame.
			constexpr static double LightUpdateFrequency = 0.1;
			//update timer
			this->FrametimeRemainer += delta;
			const double timeGain = glm::floor(this->FrametimeRemainer / LightUpdateFrequency);

			//prepare terrain texture first (async), because this is a slow operation
			this->TerrainRenderer->setViewPosition(this->ViewPosition);

			if (timeGain > 0.0) {
				const double update_delta = timeGain * LightUpdateFrequency;
				this->FrametimeRemainer -= update_delta;

				//change the sun position
				this->SunRenderer->advanceTime(update_delta);
				const vec3 sunDir = this->SunRenderer->sunDirection();
				const float nightLum = 1.0f - glm::smoothstep(-0.1f, 0.03f, sunDir.y);

				const float sun_specUV =  this->SunRenderer->spectrumCoordinate();
				//update light status.
				this->Skylight->setSpectrumCoordinate(sun_specUV);
				this->Sunlight->setSpectrumCoordinate(sun_specUV);
				this->Sunlight->setLightDirection(sunDir);
				//update night status.
				this->Nightlight->setSpectrumCoordinate(nightLum);
				this->StarfieldRenderer->EnvironmentVisibility = nightLum;
				this->AuroraRenderer->EnvironmentVisibility = nightLum;
			}

			//render, all async operations are sync automatically
			this->RenderPipeline->traverse();
		}

		/**
		 * @brief Resize the post processing framebuffer.
		 * @param res The resolution of the new framebuffer.
		*/
		inline void resize(const uvec2& res) {
			this->RenderPipeline->setResolution(res);
		}

		/**
		 * @brief Set the display gamma.
		 * @param gamma The gamma value for display.
		*/
		inline void setGamma(float gamma) {
			this->FinalProcess->setEffect<SuperTerrainPlus::STPRealism::STPPostProcess::STPPostEffect::Gamma>(gamma);
		}

	};

	class STPLogConsolePrinter : public SuperTerrainPlus::STPRealism::STPLogHandler::STPLogHandlerSolution {
	public:

		void handle(string&& log) override {
			if (!log.empty()) {
				cout << log << endl;
			}
		}

	};

	static optional<STPMasterRenderer> MasterEngine;
	static STPLogConsolePrinter RendererLogHandler;
	//Camera
	static optional<SuperTerrainPlus::STPRealism::STPPerspectiveCamera> MainCamera;

	/* ------------------------------ callback functions ----------------------------------- */
	constexpr static uvec2 InitialCanvasSize = uvec2(1600u, 900u);
	static GLFWwindow* GLCanvas = nullptr;
	static dvec2 LastRotation;

	static void frame_resized(GLFWwindow*, int width, int height) {
		if (width != 0 && height != 0) {
			//user has not minimised the window
			//updating the screen size variable
			MainCamera->rescale(1.0 * width / (1.0 * height));
			//update main renderer
			MasterEngine->resize(uvec2(width, height));
			//adjust viewport
			glViewport(0, 0, width, height);
		}
	}

	static void cursor_moved(GLFWwindow*, double X, double Y) {
		//we reverse Y since Y goes from bottom to top (from negative axis to positive)
		const dvec2 currentPos = dvec2(X, Y);
		const dvec2 offset = dvec2(currentPos.x - LastRotation.x, LastRotation.y - currentPos.y);
		MainCamera->rotate(offset);

		//update last rotation
		LastRotation = currentPos;
	}

	static void scrolled(GLFWwindow*, double, double Yoffset) {
		//we only need vertical scroll
		MainCamera->zoom(-Yoffset);
	}

#define STP_GET_KEY(KEY, FUNC) \
if (glfwGetKey(GLCanvas, KEY) == GLFW_PRESS) { \
	FUNC; \
}

	inline static void process_event(double delta) {
		using Dir = SuperTerrainPlus::STPRealism::STPCamera::STPMoveDirection;

		STP_GET_KEY(GLFW_KEY_W, MainCamera->move(Dir::Forward, delta))
		STP_GET_KEY(GLFW_KEY_S, MainCamera->move(Dir::Backward, delta))
		STP_GET_KEY(GLFW_KEY_A, MainCamera->move(Dir::Left, delta))
		STP_GET_KEY(GLFW_KEY_D, MainCamera->move(Dir::Right, delta))
		STP_GET_KEY(GLFW_KEY_SPACE, MainCamera->move(Dir::Up, delta))
		STP_GET_KEY(GLFW_KEY_C, MainCamera->move(Dir::Down, delta))

		STP_GET_KEY(GLFW_KEY_ESCAPE, glfwSetWindowShouldClose(GLCanvas, GLFW_TRUE))
	}
	
#undef STP_GET_KEY

	/* ------------------------------ framework setup ----------------------------------- */

	/**
	 * @brief Initialise GLFW engine.
	 * @return True if the GLFW window has been created.
	*/
	static bool initGLFW() {
		//Initialisation
		if (glfwInit() == GLFW_FALSE) {
			cerr << "Unable to Init GLFW." << endl;
			return false;
		}
		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);//we are running at OpenGL 4.6
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);//not necessary for forward compatibility
#ifdef _DEBUG
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
		glfwWindowHint(GLFW_CONTEXT_ROBUSTNESS, GLFW_LOSE_CONTEXT_ON_RESET);
#endif

		//rendering preferences
		glfwWindowHint(GLFW_RED_BITS, 8);
		glfwWindowHint(GLFW_GREEN_BITS, 8);
		glfwWindowHint(GLFW_BLUE_BITS, 8);
		glfwWindowHint(GLFW_ALPHA_BITS, 8);
		glfwWindowHint(GLFW_DEPTH_BITS, 24);
		glfwWindowHint(GLFW_STENCIL_BITS, 8);
		glfwWindowHint(GLFW_SAMPLES, 0);

		//creation of the rendering window
		GLCanvas = glfwCreateWindow(InitialCanvasSize.x, InitialCanvasSize.y, "SuperTerrain+ Demo", nullptr, nullptr);
		if (GLCanvas == nullptr) {
			cerr << "Unable to create GLFWwindow instance." << endl;
			return false;
		}
		//load icon
		STPDemo::STPTextureStorage iconImage("./Resource/landscape.png", 0);//all channels are required
		const ivec3 iconProps = iconImage.property();
		const GLFWimage icon = { iconProps.x, iconProps.y, const_cast<unsigned char*>(iconImage.texture()) };
		//icon data is copied by GLFW
		glfwSetWindowIcon(GLCanvas, 1, &icon);

		//setup window
		//hiding the cursor
		glfwSetInputMode(GLCanvas, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		//call back functions
		glfwSetFramebufferSizeCallback(GLCanvas, frame_resized);
		glfwSetCursorPosCallback(GLCanvas, cursor_moved);
		glfwSetScrollCallback(GLCanvas, scrolled);

		glfwGetCursorPos(GLCanvas, &LastRotation.x, &LastRotation.y);

		//finally return
		glfwMakeContextCurrent(GLCanvas);//enabling the current window as our master rendering thread
		return true;
	}

	/**
	 * @brief Create glad context, making the current running thread available to OpenGL
	 * @return True if the context is created successfully
	*/
	static bool initSTP() {
		const auto proc_addr = reinterpret_cast<GLADloadproc>(glfwGetProcAddress);
		//when we are using shared library build in GLAD, the GLAD context is shared to all libraries that are linked to it.
		if (!SuperTerrainPlus::STPEngineInitialiser::initGLexplicit(proc_addr)) {
			cerr << "Fail to initialise SuperTerrain+ engine." << endl;
			return false;
		}
		//CUDA context init on device 0 (only one GPU)
		SuperTerrainPlus::STPEngineInitialiser::init(0);
		//must init the main engine first because the renderer is depended on that.
		SuperTerrainPlus::STPRealism::STPRendererInitialiser::init();

		return SuperTerrainPlus::STPEngineInitialiser::hasInit();
	}

	/**
	 * @brief Terminate the engine and exit.
	*/
	static void clearup() {
		//make sure pipeline is deleted before the context is destroyed
		STPStart::MasterEngine.reset();

		if (GLCanvas != nullptr) {
			glfwMakeContextCurrent(nullptr);
			glfwDestroyWindow(GLCanvas);
		}
		glfwTerminate();
	}

}

int main() {
	using SuperTerrainPlus::STPAlgorithm::STPINIReader;
	using SuperTerrainPlus::STPFile;
	//read configuration
	const static STPFile engineData("./Engine.ini"), biomeData("./Biome.ini");
	//get INI
	static optional<const STPINIReader> engineINIReader, biomeINIReader;
	try {
		engineINIReader.emplace(*engineData);
		biomeINIReader.emplace(*biomeData);
	} catch (const SuperTerrainPlus::STPException::STPInvalidSyntax& se) {
		cerr << se.what() << endl;
		return -1;
	}
	const STPINIStorageView& engineINI(**engineINIReader), &biomeINI(**biomeINIReader);

	//engine setup
	//because GLFW callback uses camera, so we need to setup camera first
	if (!(STPStart::initGLFW() && STPStart::initSTP())) {
		//error
		STPStart::clearup();
		return -1;
	}
	//welcome
	cout << *SuperTerrainPlus::STPFile("./Resource/welcome.txt") << endl;
	cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
	cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << endl;
	cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << '\n' << endl;

	//setup camera
	{
		const STPINISectionView& engineMain = engineINI.at("");

		using namespace SuperTerrainPlus;
		STPEnvironment::STPCameraSetting cam;
		cam.Yaw = radians(90.0);
		cam.Pitch = 0.0;
		cam.MovementSpeed = engineMain.at("movementSpeed").to<double>();
		cam.RotationSensitivity = engineMain.at("mouseSensitivity").to<double>();
		cam.Position = dvec3(30.5, 600.0, -15.5);
		cam.WorldUp = dvec3(0.0, 1.0, 0.0);
		cam.Near = 1.0;
		cam.Far = 2500.0;
		
		STPEnvironment::STPPerspectiveCameraSetting proj;
		proj.ViewAngle = radians(60.0);
		proj.ZoomLimit = radians(dvec2(20.0, 140.0));
		proj.ZoomSensitivity = engineMain.at("zoomSensitivity").to<double>();
		proj.Aspect = 1.0 * STPStart::InitialCanvasSize.x / (1.0 * STPStart::InitialCanvasSize.y);

		STPStart::MainCamera.emplace(proj, cam);
	}

	//setup realism engine logging system
	SuperTerrainPlus::STPRealism::STPLogHandler::ActiveLogHandler = &STPStart::RendererLogHandler;
	//setup renderer
	try {
		STPStart::MasterEngine.emplace(engineINI, biomeINI, *STPStart::MainCamera);

		//allocate some memory
		STPStart::MasterEngine->resize(STPStart::InitialCanvasSize);
		STPStart::MasterEngine->setGamma(engineINI.at("Global").at("gamma").to<float>());
	} catch (const std::exception& e) {
		cerr << e.what() << endl;
		STPStart::clearup();
		return -1;
	}

	//rendering loop
	double currentTime, lastTime = 0.0, deltaTime, FPS = engineINI.at("").at("FPS").to<double>();
	cout << "Start..." << endl;
	while (!glfwWindowShouldClose(STPStart::GLCanvas)) {
		//frame time logic
		do {
			//busy-waiting fps limiter
			currentTime = glfwGetTime();
			deltaTime = currentTime - lastTime;
		} while (deltaTime < (1.0 / FPS));
		lastTime = currentTime;

		//draw
		STPStart::process_event(deltaTime);
		try {
			STPStart::MasterEngine->render(deltaTime);
		} catch (const std::exception& e) {
			cerr << e.what() << endl;
			STPStart::clearup();
			return -1;
		}

		//event update
		glfwPollEvents();
		//make sure the GPU has finished rendering the back buffer before swapping
		glFinish();
		//buffer swapping
		glfwSwapBuffers(STPStart::GLCanvas);
	}

	//termination
	cout << "Terminated, waiting for clear up..." << endl;

	STPStart::clearup();

	cout << "Done... Program now exit." << endl;
	return 0;
}