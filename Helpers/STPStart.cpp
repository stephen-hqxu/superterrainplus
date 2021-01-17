#pragma once
#include "../Renderer/STPMasterRenderer.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques 
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	//Framework stuff
	GLFWwindow* frame = nullptr;
	SglToolkit::SgTCamera* Camera = nullptr;
	STPMasterRenderer* engine = nullptr;
	SIMPLE::SIParser* engineINILoader = nullptr;
	//Constants
	const SglToolkit::SgTCamera::SgTRange zoomLimit(20.0f, 100.0f);
	const int WINDOW_SIZE[2] = {1600, 900};

	//declaring callback functions
	void frame_resized(GLFWwindow*, int, int);
	void cursor_moved(GLFWwindow*, double, double);
	void scrolled(GLFWwindow*, double, double);

	/**
	 * @brief Initialise GLFW engine
	 * @param icon The icon of the window frame
	 * @param A pointer(dynamic) to the created glfwWindow context, nullptr if not successfully created
	 * @return True if the glfwwindow has been created
	*/
	const bool InitGLFW(GLFWwindow*& glfw_window, const GLFWimage* icon = nullptr) {
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
		glfw_window = glfwCreateWindow(WINDOW_SIZE[0], WINDOW_SIZE[1], "Super Terrain + Wicked Demo", nullptr, nullptr);
		if (glfw_window == nullptr) {
			cerr << "Unable to create GLFWwindow instance." << endl;
			return false;
		}
		glfwSetWindowIcon(glfw_window, 1, icon);
		//hiding the cursor
		glfwSetInputMode(glfw_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		//call back functions
		glfwSetFramebufferSizeCallback(glfw_window, frame_resized);
		glfwSetCursorPosCallback(glfw_window, cursor_moved);
		glfwSetScrollCallback(glfw_window, scrolled);

		//finally return
		glfwMakeContextCurrent(glfw_window);//enabling the current window as our master rendering thread
		return true;
	}

	/**
	 * @brief Create glad context, making the current running thread available to opengl
	 * @return True if the context is created successfully
	*/
	const bool InitGLAD() {
		if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
			cerr << "Fail to create GLAD context." << endl;
			return false;
		}
		return true;
	}

	//----------defining callback functions-----------
	void frame_resized(GLFWwindow* window, int width, int height) {
		engine->reshape(width, height);//the window frame got resized
	}

	void cursor_moved(GLFWwindow* window, double X, double Y) {
		Camera->mouseUpdate(static_cast<float>(X), static_cast<float>(Y), true);
	}

	void scrolled(GLFWwindow* window, double Xoffset, double Yoffset) {
		Camera->scrollUpdate(static_cast<float>(Yoffset), zoomLimit);
	}
}

using namespace SuperTerrainPlus;
/**
 * @brief Start the Super Terrain + engine for demo
 * @return Exit code
*/
int main() {
	//loading icon
	GLFWimage* icon = new GLFWimage();
	unsigned char* image = stbi_load("./Resource/opengl.png", &icon->width, &icon->height, nullptr, 0);
	icon->pixels = image;
	//loading engine INI
	engineINILoader = new SIMPLE::SIParser("./Engine.ini");
	//Init
	if (!InitGLFW(frame, icon)) {
		glfwTerminate();
		return -1;
	}
	if (!InitGLAD()) {
		glfwTerminate();
		return -1;
	}
	//Init camera and the master renderer
	Camera = dynamic_cast<SglToolkit::SgTCamera*>(new SglToolkit::SgTSpectatorCamera(90.0f, 0.0f, 
		std::stof((engineINILoader->get())("movementSpeed")), std::stof((engineINILoader->get())("mouseSensitivity")), 
		60.0f, SglToolkit::SgTvec3(0.0f, 20.0f, 0.0f)));//facing towards positive-x
	engine = new STPMasterRenderer(Camera, WINDOW_SIZE, engineINILoader->get());

	//welcome
	cout << "Super Terrain Plus: The ultimate terrain engine demo program" << endl;
	cout << glGetString(GL_VERSION) << endl;
	engine->init();
	engine->reshape(WINDOW_SIZE[0], WINDOW_SIZE[1]);

	//rendering loop
	double currentTime, lastTime = 0.0f, deltaTime, FPS = std::stod((engineINILoader->get())("FPS"));
	cout << "Start..." << endl;
	while (!glfwWindowShouldClose(frame)) {
		//frametime logic
		do {//fps limiter
			currentTime = glfwGetTime();
			deltaTime = currentTime - lastTime;
		} while (deltaTime < (1.0 / FPS));
		lastTime = currentTime;

		engine->input(frame, static_cast<float>(deltaTime));
		engine->draw(deltaTime);
		//event update
		glfwPollEvents();
		//buffer swapping
		glfwSwapBuffers(frame);

	}
	cout << "Terminated, waiting for clear up..." << endl;

	//clear up
	stbi_image_free(image);
	delete icon;
	delete engine;
	delete Camera;
	delete engineINILoader;
	
	cout << "Done... Program now exit." << endl;
	glfwTerminate();
	return 0;
}