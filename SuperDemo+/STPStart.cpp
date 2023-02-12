//Core Engine
#include <SuperTerrain+/STPEngineInitialiser.h>
//Error
#include <SuperTerrain+/Exception/API/STPGLError.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>

//Rendering Engine
#include <SuperRealism+/STPRendererInitialiser.h>
#include <SuperRealism+/Utility/STPCamera.h>
#include <SuperRealism+/Utility/STPLogHandler.hpp>
#include <SuperRealism+/Utility/STPDebugCallback.h>

//INI Utility
#include <SuperAlgorithm+/Parser/STPINIParser.h>

//Demo Application
#include "STPMasterRenderer.h"
#include "./Helpers/STPCommandLineOption.h"
#include "./Helpers/STPTextureStorage.h"

//External
#include <glad/glad.h>
#include <GLFW/glfw3.h>

//System
#include <iostream>
#include <optional>
#include <tuple>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/trigonometric.hpp>

using std::make_tuple;
using std::optional;
using std::tuple;
using std::string;
using std::string_view;

using std::cerr;
using std::cout;
using std::endl;

using glm::dvec2;
using glm::dvec3;
using glm::ivec3;
using glm::radians;
using glm::uvec2;

namespace {

	void handleGLDebugCallback(const GLenum source, const GLenum type, const GLuint id, const GLenum severity,
		const GLsizei length, const GLchar* const message, const void* const userParam) {
		//let's print them
		SuperTerrainPlus::STPRealism::STPDebugCallback::print(source, type, id, severity, length, message, cout);

		//if error is severe, stop the application
		if (type == GL_DEBUG_TYPE_ERROR) {
			//we know the parameter is defined as non-constant, so we can safely cast away
			bool& shouldHalt = *reinterpret_cast<bool*>(const_cast<void*>(userParam));

			shouldHalt = true;
		}
	}

	class STPLogConsolePrinter : public SuperTerrainPlus::STPRealism::STPLogHandler::STPLogHandlerSolution {
	public:

		void handle(const string_view log) override {
			if (!log.empty()) {
				cout << log << endl;
			}
		}

	};

	optional<STPDemo::STPMasterRenderer> MasterEngine;
	STPLogConsolePrinter RendererLogHandler;
	//Camera
#pragma warning(push)
#pragma warning(disable: 4324)//padding due to alignment of AVX
	optional<SuperTerrainPlus::STPRealism::STPCamera> MainCamera;
#pragma warning(pop)

	/* ------------------------------ callback functions ----------------------------------- */
	GLFWwindow* GLCanvas = nullptr;
	dvec2 LastRotation;

	void reshape(const unsigned int width, const unsigned int height) {
		if (width != 0 && height != 0) {
			//user has not minimised the window
			//updating the screen size variable
			MainCamera->setAspect(1.0 * width / (1.0 * height));
			//update main renderer
			MasterEngine->resize(uvec2(width, height));
			//adjust viewport
			glViewport(0, 0, width, height);
		}
	}

	inline void resizeWindowCallback(GLFWwindow*, const int w, const int h) {
		reshape(static_cast<unsigned int>(w), static_cast<unsigned int>(h));
	}

	inline void moveCursorCallback(GLFWwindow*, const double X, const double Y) {
		//we reverse Y since Y goes from bottom to top (from negative axis to positive)
		const dvec2 currentPos = dvec2(X, Y);
		const dvec2 offset = dvec2(currentPos.x - LastRotation.x, LastRotation.y - currentPos.y);
		MainCamera->rotate(offset);

		//update last rotation
		LastRotation = currentPos;
	}

	inline void scrollWheelCallback(GLFWwindow*, double, const double Yoffset) {
		//we only need vertical scroll
		MainCamera->zoom(-Yoffset);
	}

#define STP_GET_KEY(KEY, FUNC) do { \
	if (glfwGetKey(GLCanvas, KEY) == GLFW_PRESS) { \
		FUNC; \
	} \
} while (false)

	void processEvent(double delta, const double speed_up) {
		using Dir = SuperTerrainPlus::STPRealism::STPCamera::STPMoveDirection;

		//modifier
		STP_GET_KEY(GLFW_KEY_LEFT_SHIFT, delta *= speed_up);

		//movement
		STP_GET_KEY(GLFW_KEY_W, MainCamera->move(Dir::Forward, delta));
		STP_GET_KEY(GLFW_KEY_S, MainCamera->move(Dir::Backward, delta));
		STP_GET_KEY(GLFW_KEY_A, MainCamera->move(Dir::Left, delta));
		STP_GET_KEY(GLFW_KEY_D, MainCamera->move(Dir::Right, delta));
		STP_GET_KEY(GLFW_KEY_SPACE, MainCamera->move(Dir::Up, delta));
		STP_GET_KEY(GLFW_KEY_C, MainCamera->move(Dir::Down, delta));

		//system control
		STP_GET_KEY(GLFW_KEY_ESCAPE, glfwSetWindowShouldClose(GLCanvas, GLFW_TRUE));
	}
	
#undef STP_GET_KEY

	/* ------------------------------ framework setup ----------------------------------- */

	/**
	 * @brief Get the active monitor.
	 * @param index The monitor index.
	 * @return The pointer to the monitor, or null if error occurs.
	*/
	GLFWmonitor* getGLFWMonitor(const unsigned int index) {
		int total;
		GLFWmonitor* const* const allMonitor = glfwGetMonitors(&total);
		if (index >= static_cast<unsigned int>(total)) {
			//make sure index is valid
			cerr << "Monitor index is not valid, the number of monitor found is " << total << endl;
			cerr << "Switch to windowed mode automatically" << endl;
			return nullptr;
		}

		//get the target monitor
		return allMonitor[index];
	}

	/**
	 * @brief Initialise GLFW engine.
	 * @param canvasDim Specifies the dimension of the window/canvas.
	 * This dimension might get modified to get the best display quality and performance.
	 * @param monitor Specifies the monitor to use for full screen display.
	 * Null to use windowed mode.
	 * @return True if the GLFW window has been created.
	*/
	bool createGLFWWindow(tuple<unsigned int, unsigned int>& canvasDim, GLFWmonitor* const monitor) {
		//get monitor setting, if any
		const auto getMonitorSetting = [&canvasDim, monitor]() noexcept -> auto {
			const GLFWvidmode* const monitorSetting = monitor ? glfwGetVideoMode(monitor) : nullptr;

			if (monitorSetting) {
				//also change the rendering resolution
				canvasDim = make_tuple(monitorSetting->width, monitorSetting->height);
				return make_tuple(
					monitorSetting->redBits, monitorSetting->greenBits, monitorSetting->blueBits, monitorSetting->refreshRate);
			}
			//otherwise use our default value
			return make_tuple(8, 8, 8, GLFW_DONT_CARE);
		};

		glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);//we are running at OpenGL 4.6
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);//not necessary for forward compatibility
		glfwWindowHint(GLFW_CONTEXT_ROBUSTNESS, GLFW_LOSE_CONTEXT_ON_RESET);
		glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_FLUSH);
#ifndef NDEBUG
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#else
		glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GLFW_TRUE);
#endif

		const auto [bit_r, bit_g, bit_b, refresh_rate] = getMonitorSetting();
		//rendering preferences
		glfwWindowHint(GLFW_RED_BITS, bit_r);
		glfwWindowHint(GLFW_GREEN_BITS, bit_g);
		glfwWindowHint(GLFW_BLUE_BITS, bit_b);
		glfwWindowHint(GLFW_ALPHA_BITS, 8);
		glfwWindowHint(GLFW_REFRESH_RATE, refresh_rate);
		//fragment tests are unused on default framebuffer
		glfwWindowHint(GLFW_DEPTH_BITS, 0);
		glfwWindowHint(GLFW_STENCIL_BITS, 0);
		glfwWindowHint(GLFW_SAMPLES, 0);

		//creation of the rendering window
		const auto [dimX, dimY] = canvasDim;
		GLCanvas = glfwCreateWindow(dimX, dimY, "SuperTerrain+ Demo", monitor, nullptr);
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
		glfwSetFramebufferSizeCallback(GLCanvas, resizeWindowCallback);
		glfwSetCursorPosCallback(GLCanvas, moveCursorCallback);
		glfwSetScrollCallback(GLCanvas, scrollWheelCallback);

		glfwGetCursorPos(GLCanvas, &LastRotation.x, &LastRotation.y);

		//finally return
		glfwMakeContextCurrent(GLCanvas);//enabling the current window as our master rendering thread
		return true;
	}

	/**
	 * @brief Create glad context, making the current running thread available to OpenGL
	*/
	void initSTP() {
		const auto proc_addr = reinterpret_cast<SuperTerrainPlus::STPEngineInitialiser::STPGLProc>(glfwGetProcAddress);
		//when we are using shared library build in GLAD, the GLAD context is shared to all libraries that are linked to it.
		//CUDA context init on device 0 (only one GPU)
		//if it fails, exception is generated
		SuperTerrainPlus::STPEngineInitialiser::initialise(0, proc_addr);
		//must init the main engine first because the renderer is depended on that.
		SuperTerrainPlus::STPRealism::STPRendererInitialiser::init();
	}

	/**
	 * @brief Terminate the engine and exit.
	*/
	void cleanUp() noexcept {
		//make sure pipeline is deleted before the context is destroyed
		MasterEngine.reset();

		if (GLCanvas) {
			glfwMakeContextCurrent(nullptr);
			glfwDestroyWindow(GLCanvas);
		}

		glfwTerminate();
	}

}

int main(const int argc, const char* argv[]) {
	namespace File = SuperTerrainPlus::STPFile;
	using namespace SuperTerrainPlus::STPAlgorithm::STPINIData;
	namespace INI = SuperTerrainPlus::STPAlgorithm::STPINIParser;

	/* ----------------- read command line option ---------------------- */
	STPDemo::STPCommandLineOption::STPResult commandLineOption;
	try {
		commandLineOption = STPDemo::STPCommandLineOption::read(argc, argv);
	} catch (const std::exception& e) {
		cerr << e.what() << endl;
		return -1;
	}

	/* --------------------- read configuration -------------------- */
	const string engineData = File::read("./Engine.ini"),
		biomeData = File::read("./Biome.ini");
	//get INI
	INI::STPINIReaderResult engineINIReader, biomeINIReader;
	try {
		engineINIReader = INI::read(engineData, "Engine INI Data");
		biomeINIReader = INI::read(biomeData, "Biome INI Data");
	} catch (const SuperTerrainPlus::STPException::STPParserError::STPBasic& pe) {
		cerr << pe.what() << endl;
		return -1;
	}
	const STPINIStorageView& engineINI(engineINIReader.Storage), &biomeINI(biomeINIReader.Storage);

	/* --------------------------- engine setup ------------------------------- */
	if (glfwInit() == GLFW_FALSE) {
		cerr << "Unable to initialise GLFW library" << endl;
		return -1;
	}
	//we may modify the resolution based on whether we are using windowed or full screen mode
	if (const auto& useFS = commandLineOption.UseFullScreen;
		!(createGLFWWindow(commandLineOption.WindowResolution, useFS ? getGLFWMonitor(*useFS) : nullptr))) {
		//error
		cleanUp();
		return -1;
	}

	try {
		initSTP();
	} catch (const std::exception& e) {
		cerr << "Unable to initialise SuperTerrain+ library" << endl;
		cerr << e.what() << endl;
		return -1;
	}

	//welcome
	cout << File::read("./Resource/welcome.txt") << endl;
	cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
	cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << endl;
	cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << '\n' << endl;

	const auto [canvasDimX, canvasDimY] = commandLineOption.WindowResolution;

	/* -------------------------------------------------------------------- */

	//capture any error from the GL callback
	bool shouldApplicationHalt = false;
	//setup debug callback
	{
		using namespace SuperTerrainPlus::STPRealism;
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(&handleGLDebugCallback, &shouldApplicationHalt);

		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
		glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
	}
	//setup camera
	{
		const STPINISectionView& engineMain = engineINI.at("");

		using namespace SuperTerrainPlus;
		STPEnvironment::STPCameraSetting cam = { };
		cam.Yaw = radians(90.0);
		cam.Pitch = 0.0;
		cam.FoV = radians(60.0);

		cam.MovementSpeed = engineMain.at("movementSpeed").to<double>();
		cam.RotationSensitivity = engineMain.at("mouseSensitivity").to<double>();
		cam.ZoomSensitivity = engineMain.at("zoomSensitivity").to<double>();

		cam.ZoomLimit = radians(dvec2(20.0, 140.0));
		cam.Position = dvec3(30.5, 600.0, -15.5);
		cam.WorldUp = dvec3(0.0, 1.0, 0.0);

		cam.Aspect = 1.0 * canvasDimX / (1.0 * canvasDimY);
		cam.Near = 1.0;
		cam.Far = 2500.0;

		MainCamera.emplace(cam);
	}

	//setup realism engine logging system
	SuperTerrainPlus::STPRealism::STPLogHandler::set(&RendererLogHandler);
	//setup renderer
	try {
		MasterEngine.emplace(engineINI, biomeINI, *MainCamera);

		//allocate some memory
		reshape(canvasDimX, canvasDimY);
		MasterEngine->setGamma(engineINI.at("Global").at("gamma").to<float>());
	} catch (const std::exception& e) {
		cerr << e.what() << endl;
		cleanUp();
		return -1;
	}

	//rendering loop
	const double FPS = commandLineOption.FrameRate.value_or(engineINI.at("").at("FPS").to<double>());
	double currentTime = 0.0, lastTime = 0.0, deltaTime = 0.0;
	cout << "Start..." << endl;
	while (!glfwWindowShouldClose(GLCanvas)) {
		//frame time logic
		do {
			//busy-waiting fps limiter
			currentTime = glfwGetTime();
			deltaTime = currentTime - lastTime;
		} while (deltaTime < (1.0 / FPS));
		lastTime = currentTime;

		//event update
		glfwPollEvents();
		processEvent(deltaTime, commandLineOption.SprintSpeedMultiplier);
		MainCamera->flush();
		//draw
		try {
			if (shouldApplicationHalt) {
				throw STP_GL_ERROR_CREATE("An erroneous GL debug message is encountered, check the output for more information...");
			}

			MasterEngine->render(currentTime, deltaTime);
		} catch (const std::exception& e) {
			cerr << e.what() << endl;
			cleanUp();
			return -1;
		}

		//make sure the GPU has finished rendering the back buffer before swapping
		glFinish();
		//buffer swapping
		glfwSwapBuffers(GLCanvas);
	}

	//termination
	cout << "Terminated, waiting for clear up..." << endl;

	cleanUp();

	cout << "Done... Program now exit." << endl;
	return 0;
}