//SuperTerrain+ Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPInvalidSyntax.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>

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

using std::optional;

using std::cout;
using std::endl;
using std::cerr;

using glm::uvec2;
using glm::ivec3;

namespace STPStart {

	static GLFWwindow* GLCanvas = nullptr;
	constexpr static uvec2 InitialCanvasSize = { 1600u, 900u };

	/**
	 * @brief Rendering the entire terrain scene for demo.
	*/
	class STPMasterRenderer {
	private:

		//Configuration
		SIMPLE::SIParser engineINILoader, biomeINILoader;

	public:

		const SIMPLE::SIStorage& engineINI, biomeINI;

	private:

		uvec2 CanvasSize;

		//Camera

		//Generation Pipeline
		optional<STPDemo::STPWorldManager> WorldManager;

		//Rendering Pipeline

	public:

		/**
		 * @brief Init STPMasterRenderer.
		*/
		STPMasterRenderer() : engineINILoader("./Engine.ini"), biomeINILoader("./Biome.ini"),
			engineINI(this->engineINILoader.get()), biomeINI(this->biomeINILoader.get()),
			CanvasSize(InitialCanvasSize) {
			using namespace SuperTerrainPlus;
			using namespace STPDemo;

			//loading terrain 2d inf parameters
			STPEnvironment::STPConfiguration config;
			config.ChunkSetting = STPTerrainParaLoader::getProcedural2DINFChunksParameter(this->engineINI["Generators"]);
			config.MeshSetting = STPTerrainParaLoader::getProcedural2DINFRenderingParameter(this->engineINI["2DTerrainINF"]);
			STPTerrainParaLoader::loadBiomeParameters(this->biomeINI);

			const auto& chunk_setting = config.ChunkSetting;
			config.HeightfieldSetting = std::move(
				STPTerrainParaLoader::getProcedural2DINFGeneratorParameter(this->engineINI["2DTerrainINF"], chunk_setting.MapSize * chunk_setting.FreeSlipChunk));
			STPEnvironment::STPSimplexNoiseSetting simplex = STPTerrainParaLoader::getSimplex2DNoiseParameter(this->biomeINI["simplex"]);

			if (!config.validate()) {
				throw STPException::STPInvalidEnvironment("Configurations are not validated");
			}
			const unsigned int unitplane_count =
				chunk_setting.ChunkSize.x * chunk_setting.ChunkSize.y * chunk_setting.RenderedChunk.x * chunk_setting.RenderedChunk.y;

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

		}

		STPMasterRenderer(const STPMasterRenderer&) = delete;

		STPMasterRenderer(STPMasterRenderer&&) = delete;

		STPMasterRenderer& operator=(const STPMasterRenderer&) = delete;

		STPMasterRenderer& operator=(STPMasterRenderer&&) = delete;

		~STPMasterRenderer() = default;

		/**
		 * @brief Main rendering functions, called every frame.
		 * @param frametime The time in sec spent on each frame.
		*/
		inline void render(double delta) {

		}

		/**
		 * @brief Framebuffer resize callback function.
		 * @param width The new width.
		 * @param height The new height.
		*/
		void reshape(int width, int height) {
			if (width != 0 && height != 0) {
				//user has not minimised the window
				//updating the screen size variable
				this->CanvasSize = uvec2(width, height);
				//adjust viewport
				glViewport(0, 0, width, height);
			}
		}

#define STP_GET_KEY(KEY, FUNC) \
if (glfwGetKey(GLCanvas, KEY) == GLFW_PRESS) { \
	FUNC; \
}

		/**
		 * @brief Check the input each frame.
		 * @param time Frame time.
		*/
		inline void processEvent(double delta) {
			STP_GET_KEY(GLFW_KEY_ESCAPE, glfwSetWindowShouldClose(GLCanvas, GLFW_TRUE))
		}

	};
	static optional<STPMasterRenderer> MasterEngine;

	/* ------------------------------ callback functions ----------------------------------- */
	static void frame_resized(GLFWwindow* window, int width, int height) {
		MasterEngine->reshape(width, height);
	}

	static void cursor_moved(GLFWwindow* window, double X, double Y) {

	}

	static void scrolled(GLFWwindow* window, double Xoffset, double Yoffset) {

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
		STPDemo::STPTextureStorage iconImage("./Resource/mountain.png", 0);//all channels are required
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
	//engine setup
	if (!(STPStart::initGLFW() && STPStart::initSTP())) {
		//error
		STPStart::clearup();
		return -1;
	}
	//welcome
	cout << *SuperTerrainPlus::STPFile("./Resource/welcome.txt") << endl;
	cout << glGetString(GL_VERSION) << endl;

	//setup renderer
	STPStart::MasterEngine.emplace();
	STPStart::MasterEngine->reshape(STPStart::InitialCanvasSize.x, STPStart::InitialCanvasSize.y);

	//rendering loop
	double currentTime, lastTime = 0.0f, deltaTime, FPS = STPStart::MasterEngine->engineINI("FPS").to<double>();
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
		STPStart::MasterEngine->processEvent(deltaTime);
		STPStart::MasterEngine->render(deltaTime);

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