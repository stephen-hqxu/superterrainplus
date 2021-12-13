//SuperTerrain+ Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//SuperDemo+
#include "./Helpers/STPTerrainParaLoader.h"
#include "./World/STPWorldManager.h"
//Image Loader
#include "./Helpers/STPTextureStorage.h"

//External
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <SIMPLE/SIParser.h>

//System
#include <iostream>

using std::cout;
using std::endl;
using std::cerr;

using glm::uvec2;
using glm::ivec3;

namespace STPStart {

	//Constants
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

		//Rendering Pipeline

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
			config.getChunkSetting() = STPTerrainParaLoader::getProcedural2DINFChunksParameter(this->engineINI["Generators"]);
			config.getMeshSetting() = STPTerrainParaLoader::getProcedural2DINFRenderingParameter(this->engineINI["2DTerrainINF"]);
			STPTerrainParaLoader::loadBiomeParameters(this->biomeINI);

			const auto& chunk_setting = config.getChunkSetting();
			config.getHeightfieldSetting() = std::move(
				STPTerrainParaLoader::getProcedural2DINFGeneratorParameter(this->engineINI["2DTerrainINF"], chunk_setting.MapSize * chunk_setting.FreeSlipChunk));
			STPEnvironment::STPSimplexNoiseSetting simplex = STPTerrainParaLoader::getSimplex2DNoiseParameter(this->engineINI["simplex"]);
			
			if (!config.validate()) {
				throw STPException::STPInvalidEnvironment("Configurations are not validated");
			}
			const unsigned int unitplane_count = 
				chunk_setting.ChunkSize.x * chunk_setting.ChunkSize.y * chunk_setting.RenderedChunk.x * chunk_setting.RenderedChunk.y;

		}

		~STPMasterRenderer() = default;

	public:

		STPMasterRenderer(const STPMasterRenderer&) = delete;

		STPMasterRenderer(STPMasterRenderer&&) = delete;

		STPMasterRenderer& operator=(const STPMasterRenderer&) = delete;

		STPMasterRenderer& operator=(STPMasterRenderer&&) = delete;

		/**
		 * @brief Get the global isntance of the master renderer.
		 * @return The pointer to the master renderer.
		*/
		static STPMasterRenderer& instance() {
			static STPMasterRenderer Instance;

			return Instance;
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

	};

	/* ------------------------------ callback functions ----------------------------------- */
	static void frame_resized(GLFWwindow* window, int width, int height) {
		STPMasterRenderer::instance().reshape(width, height);
	}

	static void cursor_moved(GLFWwindow* window, double X, double Y) {

	}

	static void scrolled(GLFWwindow* window, double Xoffset, double Yoffset) {

	}

	/* ------------------------------ framework setup ----------------------------------- */
	static GLFWwindow* GLCanvas = nullptr;

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
		GLCanvas = glfwCreateWindow(InitialCanvasSize.x, InitialCanvasSize.y, "Super Terrain + Wicked Demo", nullptr, nullptr);
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
	//setup renderer
	auto& Engine = STPStart::STPMasterRenderer::instance();
	
	return 0;
}