//SuperTerrain+ Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPInvalidSyntax.h>
#include <SuperTerrain+/Exception/STPUnsupportedFunctionality.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>

//SuperRealism+ Engine
#include <SuperRealism+/Utility/Camera/STPPerspectiveCamera.h>
#include <SuperRealism+/STPScenePipeline.h>
#include <SuperRealism+/Renderer/STPHeightfieldTerrain.h>
#include <SuperRealism+/Renderer/STPSun.h>
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
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <SIMPLE/SIParser.h>

//System
#include <iostream>
#include <optional>

//GLM
#include <glm/trigonometric.hpp>
#include <glm/vec4.hpp>

using std::optional;
using std::string;

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
using glm::radians;

namespace STPStart {

	/**
	 * @brief Rendering the entire terrain scene for demo.
	*/
	class STPMasterRenderer {
	private:

		const SIMPLE::SIStorage& engineINI, biomeINI;

		//Generation Pipeline
		optional<STPDemo::STPWorldManager> WorldManager;

		//Rendering Pipeline
		optional<SuperTerrainPlus::STPRealism::STPHeightfieldTerrain> TerrainRenderer;
		optional<SuperTerrainPlus::STPRealism::STPSun> SunRenderer;
		optional<SuperTerrainPlus::STPRealism::STPScenePipeline> RenderPipeline;

		//Setting
		SuperTerrainPlus::STPEnvironment::STPSunSetting SunSetting;

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

	public:

		/**
		 * @brief Init STPMasterRenderer.
		 * @param engine The pointer to engine INI settings.
		 * @param biome The pointer to biome INI settings.
		 * @param camera The pointer to the perspective camera for the scene.
		*/
		STPMasterRenderer(const SIMPLE::SIStorage& engine, const SIMPLE::SIStorage& biome, SuperTerrainPlus::STPRealism::STPPerspectiveCamera& camera) :
			engineINI(engine), biomeINI(biome) {
			using namespace SuperTerrainPlus;
			using namespace STPDemo;

			//loading terrain parameters
			STPEnvironment::STPConfiguration config;
			config.ChunkSetting = STPTerrainParaLoader::getChunkSetting(this->engineINI["Generators"]);
			STPEnvironment::STPMeshSetting MeshSetting = STPTerrainParaLoader::getRenderingSetting(this->engineINI["2DTerrainINF"]);
			STPTerrainParaLoader::loadBiomeParameters(this->biomeINI);

			const auto& chunk_setting = config.ChunkSetting;
			config.HeightfieldSetting = std::move(
				STPTerrainParaLoader::getGeneratorSetting(this->engineINI["2DTerrainINF"], chunk_setting.MapSize * chunk_setting.FreeSlipChunk));
			STPEnvironment::STPSimplexNoiseSetting simplex = STPTerrainParaLoader::getSimplexSetting(this->biomeINI["simplex"]);

			if (!config.validate()) {
				throw STPException::STPInvalidEnvironment("Configurations are not validated");
			}

			//load renderer settings
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
			glEnable(GL_DEBUG_OUTPUT);
			STPDebugCallback::registerAsyncCallback(cout);
			glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, NULL, GL_FALSE);
			glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, NULL, GL_FALSE);

			//setup rendering components
			//-------------------------------------------
			//terrain
			STPHeightfieldTerrain::STPHeightfieldTerrainLog terrain_log;
			this->TerrainRenderer.emplace(this->WorldManager->getPipeline(), terrain_log, uvec3(128u, 128u, 6u));
			//print log
			STPMasterRenderer::printLog(terrain_log);
			this->TerrainRenderer->setMesh(MeshSetting);
			this->TerrainRenderer->seedRandomBuffer(this->biomeINI("seed", "simplex").to<unsigned long long>());
			//-------------------------------------------
			//sun
			STPSun::STPSunLog sun_log;
			this->SunRenderer.emplace(this->SunSetting, sun_log);
			STPMasterRenderer::printLog(sun_log);
			//setup atmoshpere
			this->SunRenderer->setAtmoshpere(sky_setting.second);

			//setup rendering pipeline
			STPScenePipeline::STPSceneWorkflow workflow = { };
			workflow.Terrain = &(*this->TerrainRenderer);
			workflow.Sun = &(*this->SunRenderer);
			this->RenderPipeline.emplace(camera, workflow);
			//basic setup
			this->RenderPipeline->setClearColor(vec4(vec3(44.0f, 110.0f, 209.0f) / 255.0f, 1.0f));
		}

		STPMasterRenderer(const STPMasterRenderer&) = delete;

		STPMasterRenderer(STPMasterRenderer&&) = delete;

		STPMasterRenderer& operator=(const STPMasterRenderer&) = delete;

		STPMasterRenderer& operator=(STPMasterRenderer&&) = delete;

		~STPMasterRenderer() = default;

		/**
		 * @brief Main rendering functions, called every frame.
		*/
		inline void render() {
			//change the sun position
			this->SunRenderer->deltaTick(1ull);

			this->RenderPipeline->traverse();
		}

	};
	static optional<STPMasterRenderer> MasterEngine;
	//Camera
	static optional<SuperTerrainPlus::STPRealism::STPPerspectiveCamera> MainCamera;
	//Configuration
	static SIMPLE::SIParser engineINILoader("./Engine.ini"), biomeINILoader("./Biome.ini");

	/* ------------------------------ callback functions ----------------------------------- */
	constexpr static uvec2 InitialCanvasSize = { 1600u, 900u };
	static GLFWwindow* GLCanvas = nullptr;
	static dvec2 LastRotation;

	static void frame_resized(GLFWwindow* window, int width, int height) {
		if (width != 0 && height != 0) {
			//user has not minimised the window
			//updating the screen size variable
			MainCamera->rescale(1.0f * width / (1.0f * height));
			//adjust viewport
			glViewport(0, 0, width, height);
		}
	}

	static void cursor_moved(GLFWwindow* window, double X, double Y) {
		//we reverse Y since Y goes from bottom to top (from negative axis to positive)
		const dvec2 currentPos = vec2(X, Y);
		const vec2 offset = vec2(currentPos.x - LastRotation.x, LastRotation.y - currentPos.y);
		MainCamera->rotate(offset);

		//update last rotation
		LastRotation = currentPos;
	}

	static void scrolled(GLFWwindow* window, double Xoffset, double Yoffset) {
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
	 * @brief Initialise GLFW engine
	 * @return True if the glfwwindow has been created
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

		//rendering preferences
		glfwWindowHint(GLFW_RED_BITS, 8);
		glfwWindowHint(GLFW_GREEN_BITS, 8);
		glfwWindowHint(GLFW_BLUE_BITS, 8);
		glfwWindowHint(GLFW_ALPHA_BITS, 8);
		glfwWindowHint(GLFW_DEPTH_BITS, 24);
		glfwWindowHint(GLFW_STENCIL_BITS, 8);
		glfwWindowHint(GLFW_SAMPLES, 8);

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

		return SuperTerrainPlus::STPEngineInitialiser::hasInit();
	}

	/**
	 * @brief Terminate the engine and exit.
	*/
	static void clearup() {
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
		
		STPEnvironment::STPPerspectiveCameraSetting proj = { };
		proj.ViewAngle = radians(60.0f);
		proj.ZoomLimit = radians(vec2(20.0f, 100.0f));
		proj.ZoomSensitivity = engineINI("zoomSensitivity").to<float>();
		proj.Aspect = 1.0f * STPStart::InitialCanvasSize.x / (1.0f * STPStart::InitialCanvasSize.y);
		proj.Near = 1.0f;
		proj.Far = 2200.0f;

		STPStart::MainCamera.emplace(proj, cam);
	}

	//setup renderer
	try {
		STPStart::MasterEngine.emplace(engineINI, biomeINI, *STPStart::MainCamera);
	}
	catch (const std::exception& e) {
		cerr << e.what() << endl;
		STPStart::clearup();
		return -1;
	}
	//welcome
	cout << *SuperTerrainPlus::STPFile("./Resource/welcome.txt") << endl;
	cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
	cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << endl;
	cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl << endl;

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
			STPStart::MasterEngine->render();
		}
		catch (std::exception& e) {
			cerr << e.what() << endl;
			STPStart::clearup();
			return -1;
		}

		//event update
		glfwPollEvents();
		//buffer swapping
		glfwSwapBuffers(STPStart::GLCanvas);
	}

	//termination
	cout << "Terminated, waiting for clear up..." << endl;

	//make sure pipeline is deleted before the context is destroyed
	STPStart::MasterEngine.reset();
	STPStart::clearup();

	cout << "Done... Program now exit." << endl;
	
	return 0;
}