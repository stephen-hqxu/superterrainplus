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
#include <SuperRealism+/Scene/Component/STPHeightfieldTerrain.h>
#include <SuperRealism+/Scene/Component/STPSun.h>
#include <SuperRealism+/Scene/Component/STPAmbientOcclusion.h>
#include <SuperRealism+/Scene/Component/STPPostProcess.h>
//GL helper
#include <SuperRealism+/Utility/STPDebugCallback.h>

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
#include <SIMPLE/SIParser.h>

//System
#include <iostream>
#include <optional>

//GLM
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

using std::optional;
using std::string;
using std::make_pair;

using std::cout;
using std::endl;
using std::cerr;

using glm::uvec2;
using glm::vec2;
using glm::dvec2;
using glm::ivec3;
using glm::uvec3;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::rotate;
using glm::radians;
using glm::identity;
using glm::normalize;

namespace STPStart {

	/**
	 * @brief Rendering the entire terrain scene for demo.
	*/
	class STPMasterRenderer {
	private:

		const SIMPLE::SIStorage& engineINI, biomeINI;

		//Generation Pipeline
		optional<STPDemo::STPWorldManager> WorldManager;

		//Setting
		SuperTerrainPlus::STPEnvironment::STPSunSetting SunSetting;

		//Rendering Pipeline
		optional<SuperTerrainPlus::STPRealism::STPScenePipeline> RenderPipeline;
		SuperTerrainPlus::STPRealism::STPSun<true>* SunRenderer;
		SuperTerrainPlus::STPRealism::STPHeightfieldTerrain<true>* TerrainRenderer;
		SuperTerrainPlus::STPRealism::STPAmbientOcclusion* AOEffect;
		SuperTerrainPlus::STPRealism::STPPostProcess* FinalProcess;

		//A number that locates the renderer in the scene
		SuperTerrainPlus::STPRealism::STPScenePipeline::STPLightIdentifier SunIndex;

		const vec3& ViewPosition;

		//This time record the frametime from last frame that is not enough to round up to one tick
		double FrametimeRemainer = 0.0;

		/**
		 * @brief Try to print all logs provided to cout.
		 * @tparam L The log storage.
		 * @param log The pointer to the log.
		*/
		template<class L>
		static void printLog(const L& log) {
			for (unsigned int i = 0u; i < L::Count; i++) {
				const string& current_log = log.Log[i];
				if (!current_log.empty()) {
					cout << current_log << endl;
				}
			}
		}

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

		//The number of sample used next time the resize() function is called.
		unsigned int MultiSample = 1u;

		/**
		 * @brief Init STPMasterRenderer.
		 * @param engine The pointer to engine INI settings.
		 * @param biome The pointer to biome INI settings.
		 * @param camera The pointer to the perspective camera for the scene.
		*/
		STPMasterRenderer(const SIMPLE::SIStorage& engine, const SIMPLE::SIStorage& biome, SuperTerrainPlus::STPRealism::STPPerspectiveCamera& camera) :
			engineINI(engine), biomeINI(biome), ViewPosition(camera.cameraStatus().Position), 
			CurrentSeed(this->biomeINI("seed", "simplex").to<unsigned long long>()) {
			using namespace SuperTerrainPlus;
			using namespace STPDemo;

			//loading terrain parameters
			STPEnvironment::STPConfiguration config;
			STPEnvironment::STPSimplexNoiseSetting simplex = STPTerrainParaLoader::getSimplexSetting(this->biomeINI["simplex"]);
			{
				config.ChunkSetting = STPTerrainParaLoader::getChunkSetting(this->engineINI["Generators"]);
				STPTerrainParaLoader::loadBiomeParameters(this->biomeINI);

				const auto& chunk_setting = config.ChunkSetting;
				config.HeightfieldSetting = std::move(
					STPTerrainParaLoader::getGeneratorSetting(this->engineINI["2DTerrainINF"], chunk_setting.MapSize * chunk_setting.FreeSlipChunk));

				if (!config.validate()) {
					throw STPException::STPInvalidEnvironment("Configurations are not validated");
				}
			}
			//load renderer settings
			STPEnvironment::STPMeshSetting MeshSetting = STPTerrainParaLoader::getRenderingSetting(this->engineINI["2DTerrainINF"]);
			const auto sky_setting = STPTerrainParaLoader::getSkySetting(this->engineINI["Sky"]);
			this->SunSetting = sky_setting.first;

			//setup world manager
			try {
				this->WorldManager.emplace(this->biomeINI("texture_path_prefix")(), config, simplex);
				//the old setting has been moved to the world manager, need to refresh the pointer
				const auto& chunk_setting = this->WorldManager->getWorldSetting().ChunkSetting;

				this->WorldManager->attachBiomeFactory<STPDemo::STPLayerChainBuilder>(chunk_setting.MapSize, simplex.Seed);
				this->WorldManager->attachDiversityGenerator<STPDemo::STPBiomefieldGenerator>
					(this->WorldManager->SharedProgram, chunk_setting.MapSize, this->biomeINI("interpolationRadius").to<unsigned int>());
				this->WorldManager->attachTextureFactory<STPDemo::STPSplatmapGenerator>
					(this->WorldManager->SharedProgram, this->WorldManager->getTextureDatabase(), chunk_setting);

				this->WorldManager->linkProgram(this->engineINI("Anisotropy", "Global").to<float>());
				if (!this->WorldManager) {
					//do not proceed if it fails
					terminate();
				}
			}
			catch (const STPException::STPInvalidSyntax& se) {
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
			//MS is unnecessary in deferred shading
			glDisable(GL_MULTISAMPLE);
			glEnable(GL_DEBUG_OUTPUT);
			STPDebugCallback::registerAsyncCallback(cout);
			glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, NULL, GL_FALSE);
			glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, NULL, GL_FALSE);

			//setup vertex shader for off-screen rendering that can be shared
			STPScreen::STPScreenVertexShader::STPScreenVertexShaderLog screen_shader_log;
			const STPScreen::STPScreenVertexShader ScreenVertexShader(screen_shader_log);
			STPMasterRenderer::printLog(screen_shader_log);

			//this buffer is a shared pointer wrapper and we don't need to manage its lifetime
			const STPScreen::STPSharableScreenVertexBuffer OffScreenVertexBuffer = 
				std::make_shared<STPScreen::STPScreenVertexBuffer>();
			STPScreen::STPScreenLog screen_renderer_log;
			STPScreen::STPScreenInitialiser screen_renderer_init;
			screen_renderer_init.VertexShader = &ScreenVertexShader;
			screen_renderer_init.SharedVertexBuffer = &OffScreenVertexBuffer;
			screen_renderer_init.Log = &screen_renderer_log;

			//setup scene pipeline
			//-------------------------------------------------------------------------
			{
				STPScenePipeline::STPScenePipelineInitialiser scene_init;

				//initialisation
				STPScenePipeline::STPShadowMapFilterKernel<STPScenePipeline::STPShadowMapFilter::PCF> scene_shadow_function;
				scene_init.ShadowFilter = &scene_shadow_function;
				scene_shadow_function.DepthBias = vec2(0.055f, 0.0055f);
				scene_shadow_function.NormalBias = vec2(15.5f, 5.5f);
				scene_shadow_function.BiasFarMultiplier = 0.45f;
				scene_shadow_function.CascadeBlendArea = 40.5f;
				scene_shadow_function.KernelRadius = 4u;
				scene_shadow_function.KernelDistance = 2.45f;

				STPScenePipeline::STPSceneShaderCapacity& scene_cap = scene_init.ShaderCapacity;
				scene_cap.EnvironmentLight = 1ull;
				scene_cap.DirectionalLightShadow = 1ull;
				scene_cap.LightSpaceMatrix = 5ull;
				scene_cap.LightFrustumDivisionPlane = 5ull;

				//construct rendering pipeline
				scene_init.GeometryBufferInitialiser = &screen_renderer_init;

				this->RenderPipeline.emplace(camera, scene_init);

				STPMasterRenderer::printLog(screen_renderer_log);
				STPMasterRenderer::printLog(scene_init.DepthShader);
			}
			//setup light
			//-------------------------------------------
			{
				//sun shadow setting
				const float camFar = camera.cameraStatus().Far;
				const STPCascadedShadowMap::STPLightFrustum frustum = {
					2048u,
					{
						camFar / 16.0f,
						camFar / 3.5f,
						camFar / 1.5f
					},
					32.5f,
					&camera,
					234.5f
				};

				//sun
				STPSun<true>::STPSunLog sun_log;
				this->SunRenderer = this->RenderPipeline->add<STPSun<true>>(this->SunSetting, 8192u, frustum, sun_log);
				//print log
				STPMasterRenderer::printLog(sun_log.SpectrumGenerator);
				STPMasterRenderer::printLog(sun_log.SunRenderer);
				//setup atmoshpere
				const STPEnvironment::STPAtmosphereSetting& atm_setting = sky_setting.second;
				this->SunRenderer->setAtmoshpere(atm_setting);
				//-------------------------------------------
				//setup the spectrum
				mat4 raySpace = identity<mat4>();
				raySpace = rotate(raySpace, radians(2.7f), normalize(vec3(vec2(0.0f), 1.0f)));
				const STPSun<true>::STPSunSpectrum::STPSpectrumSpecification spectrum_spec = {
					&atm_setting,
					static_cast<mat3>(raySpace),
					make_pair(
						normalize(vec3(1.0f, -0.1f, 0.0f)),
						normalize(vec3(0.0f, 1.0f, 0.0f))
					)
				};
				//generate a new spectrum
				this->SunRenderer->SunSpectrum(spectrum_spec);
			}

			//setup solid object
			//-------------------------------------------
			{
				//terrain
				STPHeightfieldTerrain<true>::STPHeightfieldTerrainLog terrain_log;
				const STPHeightfieldTerrain<true>::STPTerrainShaderOption terrain_opt = {
					uvec3(128u, 128u, 6u),
					STPHeightfieldTerrain<true>::STPNormalBlendingAlgorithm::BasisTransform
				};
				STPEnvironment::STPMeshSetting::STPTessellationSetting DepthTessSetting = MeshSetting.TessSetting;
				DepthTessSetting.MaxTessLevel *= 0.5f;

				this->TerrainRenderer = this->RenderPipeline->add<STPHeightfieldTerrain<true>>(this->WorldManager->getPipeline(), terrain_log, terrain_opt);
				STPMasterRenderer::printLog(terrain_log);
				//initial setup
				this->TerrainRenderer->setMesh(MeshSetting);
				this->TerrainRenderer->setDepthMeshQuality(DepthTessSetting);
				this->TerrainRenderer->seedRandomBuffer(this->getNextSeed());

				//read logs from renering components after pipeline setup
				//some compilation happens after pipeline initialisation
				auto& terrain_depth_log_db = this->TerrainRenderer->TerrainDepthLogStorage;
				while (!terrain_depth_log_db.empty()) {
					const auto& terrain_depth_log = terrain_depth_log_db.front();
					STPMasterRenderer::printLog(terrain_depth_log);
					terrain_depth_log_db.pop();
				}
			}
			//-------------------------------------------
			{
				const SIMPLE::SISection& ao_section = engine["AmbientOcclusion"];
				//blur
				STPGaussianFilter blur_filter(
					ao_section("variance").to<double>(),
					ao_section("kernel_distance").to<double>(),
					ao_section("kernel_radius").to<unsigned int>(),
				screen_renderer_init);
				
				STPMasterRenderer::printLog(screen_renderer_log);

				//ambient occlusion
				const STPEnvironment::STPOcclusionKernelSetting ao_setting = STPTerrainParaLoader::getAOSetting(ao_section);

				this->AOEffect = this->RenderPipeline->add<STPAmbientOcclusion>(ao_setting, std::move(blur_filter), screen_renderer_init);

				STPMasterRenderer::printLog(screen_renderer_log);
			}
			{
				//post process
				STPPostProcess::STPToneMappingDefinition<STPPostProcess::STPToneMappingFunction::Lottes> postprocess_def;

				this->FinalProcess = this->RenderPipeline->add<STPPostProcess>(postprocess_def, screen_renderer_init);
				STPMasterRenderer::printLog(screen_renderer_log);
			}

			using PT = STPScenePipeline::STPLightPropertyType;
			//store this index so later we can update the light quicker
			this->SunIndex = this->RenderPipeline->locateLight(this->SunRenderer);

			STPEnvironment::STPLightSetting::STPAmbientLightSetting sun_ambient;
			sun_ambient.AmbientStrength = 0.45f;
			STPEnvironment::STPLightSetting::STPDirectionalLightSetting sun_directional;
			sun_directional.DiffuseStrength = 1.6f;
			sun_directional.SpecularStrength = 6.5f;

			this->RenderPipeline->setLight(this->SunIndex, sun_ambient);
			this->RenderPipeline->setLight(this->SunIndex, sun_directional);
			this->RenderPipeline->setClearColor(vec4(vec3(44.0f, 110.0f, 209.0f) / 255.0f, 1.0f));
			this->RenderPipeline->setExtinctionArea(0.785f);
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
			//Advance one tick after that many seconds have built up
			constexpr static double TickScale = 0.1;
			//update timer
			this->FrametimeRemainer += delta;
			//calculate how many tick has elapsed
			unsigned long long tickGain = static_cast<unsigned long long>(glm::floor(this->FrametimeRemainer / TickScale));
			//deduct used ticks and store the remainder
			this->FrametimeRemainer -= TickScale * tickGain;

			//prepare terrain texture first (async), because this is a slow operation
			this->TerrainRenderer->setViewPosition(this->ViewPosition);

			using PT = SuperTerrainPlus::STPRealism::STPScenePipeline::STPLightPropertyType;
			if (tickGain > 0ull) {
				//change the sun position
				this->SunRenderer->advanceTick(tickGain);
			}
			//updat terrain rendering settings.
			this->RenderPipeline->setLight<PT::SpectrumCoordinate>(this->SunIndex);
			this->RenderPipeline->setLight<PT::Direction>(this->SunIndex);

			//render, all async operations are sync automatically
			using namespace SuperTerrainPlus::STPRealism;
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
	static optional<STPMasterRenderer> MasterEngine;
	//Camera
	static optional<SuperTerrainPlus::STPRealism::STPPerspectiveCamera> MainCamera;
	//Configuration
	static SIMPLE::SIParser engineINILoader("./Engine.ini"), biomeINILoader("./Biome.ini");

	/* ------------------------------ callback functions ----------------------------------- */
	constexpr static uvec2 InitialCanvasSize = uvec2(1600u, 900u);
	static GLFWwindow* GLCanvas = nullptr;
	static dvec2 LastRotation;

	static void frame_resized(GLFWwindow*, int width, int height) {
		if (width != 0 && height != 0) {
			//user has not minimised the window
			//updating the screen size variable
			MainCamera->rescale(1.0f * width / (1.0f * height));
			//update main renderer
			MasterEngine->resize(uvec2(width, height));
			//adjust viewport
			glViewport(0, 0, width, height);
		}
	}

	static void cursor_moved(GLFWwindow*, double X, double Y) {
		//we reverse Y since Y goes from bottom to top (from negative axis to positive)
		const dvec2 currentPos = vec2(X, Y);
		const vec2 offset = vec2(currentPos.x - LastRotation.x, LastRotation.y - currentPos.y);
		MainCamera->rotate(offset);

		//update last rotation
		LastRotation = currentPos;
	}

	static void scrolled(GLFWwindow*, double, double Yoffset) {
		//we only need vertical scroll
		MainCamera->zoom(-static_cast<float>(Yoffset));
	}

#define STP_GET_KEY(KEY, FUNC) \
if (glfwGetKey(GLCanvas, KEY) == GLFW_PRESS) { \
	FUNC; \
}

	inline static void process_event(float delta) {
		using Dir = SuperTerrainPlus::STPRealism::STPCamera::STPMoveDirection;

		STP_GET_KEY(GLFW_KEY_W, MainCamera->move(Dir::Forward, delta))
		STP_GET_KEY(GLFW_KEY_S, MainCamera->move(Dir::Backward, delta))
		STP_GET_KEY(GLFW_KEY_A, MainCamera->move(Dir::Left, delta))
		STP_GET_KEY(GLFW_KEY_D, MainCamera->move(Dir::Right, delta))
		STP_GET_KEY(GLFW_KEY_SPACE, MainCamera->move(Dir::Up, delta))
		STP_GET_KEY(GLFW_KEY_C, MainCamera->move(Dir::Down, delta))

		STP_GET_KEY(GLFW_KEY_ESCAPE, glfwSetWindowShouldClose(GLCanvas, GLFW_TRUE))
	}

	/* ------------------------------ framework setup ----------------------------------- */

	/**
	 * @brief Initialise GLFW engine.
	 * @return True if the glfwwindow has been created.
	*/
	static bool initGLFW() {
		//Initialisation
		if (glfwInit() == GLFW_FALSE) {
			cerr << "Unable to Init GLFW." << endl;
			return false;
		}
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);//we are running at opengl 4.6
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);//not neccessary for forward compat
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
	 * @brief Create glad context, making the current running thread available to opengl
	 * @return True if the context is created successfully
	*/
	static bool initSTP() {
		const auto proc_addr = reinterpret_cast<GLADloadproc>(glfwGetProcAddress);
		//when we are using shared library build in GLAD, the GLAD context is shared to all libraries that are linked to it.
		if (!SuperTerrainPlus::STPEngineInitialiser::initGLexplicit(proc_addr)) {
			cerr << "Fail to initialise Super Terrain + engine." << endl;
			return false;
		}
		//cuda context init on device 0 (only one GPU)
		SuperTerrainPlus::STPEngineInitialiser::init(0);
		//must init the main engien first because the renderer is depended on that.
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
	//get INI
	const SIMPLE::SIStorage& engineINI = STPStart::engineINILoader.get(), 
		&biomeINI = STPStart::biomeINILoader.get();
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
		using namespace SuperTerrainPlus;
		STPEnvironment::STPCameraSetting cam = { };
		cam.Yaw = radians(90.0f);
		cam.Pitch = 0.0f;
		cam.MovementSpeed = engineINI("movementSpeed").to<float>();
		cam.RotationSensitivity = engineINI("mouseSensitivity").to<float>();
		cam.Position = vec3(0.0f, 600.0f, 0.0f);
		cam.WorldUp = vec3(0.0f, 1.0f, 0.0f);
		cam.Near = 1.0f;
		cam.Far = 2500.0f;
		cam.LogarithmicConstant = 1.0f;
		
		STPEnvironment::STPPerspectiveCameraSetting proj = { };
		proj.ViewAngle = radians(60.0f);
		proj.ZoomLimit = radians(vec2(20.0f, 100.0f));
		proj.ZoomSensitivity = engineINI("zoomSensitivity").to<float>();
		proj.Aspect = 1.0f * STPStart::InitialCanvasSize.x / (1.0f * STPStart::InitialCanvasSize.y);

		STPStart::MainCamera.emplace(proj, cam);
	}

	//setup renderer
	try {
		STPStart::MasterEngine.emplace(engineINI, biomeINI, *STPStart::MainCamera);

		//basic setup for the master renderer
		STPStart::MasterEngine->MultiSample = engineINI("MSAA", "Global").to<unsigned int>();
		//allocate some memory
		STPStart::MasterEngine->resize(STPStart::InitialCanvasSize);
		STPStart::MasterEngine->setGamma(engineINI("gamma", "Global").to<float>());
	}
	catch (const std::exception& e) {
		cerr << e.what() << endl;
		STPStart::clearup();
		return -1;
	}

	//rendering loop
	double currentTime, lastTime = 0.0f, deltaTime, FPS = engineINI("FPS").to<double>();
	cout << "Start..." << endl;
	while (!glfwWindowShouldClose(STPStart::GLCanvas)) {
		//frametime logic
		do {
			//busy-waiting fps limiter
			currentTime = glfwGetTime();
			deltaTime = currentTime - lastTime;
		} while (deltaTime < (1.0 / FPS));
		lastTime = currentTime;

		//draw
		STPStart::process_event(static_cast<float>(deltaTime));
		try {
			STPStart::MasterEngine->render(deltaTime);
		}
		catch (std::exception& e) {
			cerr << e.what() << endl;
			STPStart::clearup();
			return -1;
		}

		//event update
		glfwPollEvents();
		//make sure the GPU has finished rendering the backbuffer before swapping
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