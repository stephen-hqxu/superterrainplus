#pragma once
//System
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <future>
#include <string>
//Math library
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::identity;
using glm::perspective;
using glm::radians;
using glm::value_ptr;
//OpenGL engine
#include "glad/glad.h"
#include "GLFW/glfw3.h"
//Image loader by stb_image
#ifndef STBI_INCLUDE_STB_IMAGE_H
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif
//INI loader
#include "SIMPLE1.0/SIParser.h"
//OpenGL utilities
#include "SglToolkit1.0/SgTCamera/SgTSpectatorCamera.h"
#include "SglToolkit1.0/SgTShaderProc.h"
#include "SglToolkit1.0/SgTUtils.h"
//Multithreadding engine
#include "../Helpers/STPThreadPool.h"

//Invididual rendering engine
#include "../Helpers/STPIndirectCommands.h"
#include "../Helpers/STPTextureStorage.h"
#include "../Helpers/STPTerrainParaLoader.h"
#include "STPSkyRenderer.h"
#include "STPOffscreenRenderer.h"
#include "STPRendererCommander.h"
#include "../World/STPProcedural2DINF.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief Rendering the entire terrain scene for demo. There are multiple terrain rendering models for choices.
	*/
	class STPMasterRenderer {
	private:
		
		//camera matrix
		mat4 Projection, View;
		GLuint PVmatrix;//ssbo, layout: view, view_notrans(for skybox), projection
		char* PVblock_mapped = nullptr;//The persistent PVblock mapping

		SglToolkit::SgTCamera*& Camera;
		SIMPLE::SIStorage& engineSettings;
		int SCR_SIZE[2] = {0,0};//array of 2 int, width and height

		//genereal renderers
		STPSkyRenderer* sky = nullptr;
		STPOffscreenRenderer* screen = nullptr;
		STPRendererCommander* command = nullptr;
		//terrain renderers
		STPProcedural2DINF* terrain2d_inf = nullptr;
		//thread pool
		STPThreadPool* command_pool = nullptr;

		/**
		 * @brief Set up the shader storage buffer for pv matrix
		*/
		void setPVmatrix() {
			glCreateBuffers(1, &this->PVmatrix);
			//allocation
			glNamedBufferStorage(this->PVmatrix, 3 * sizeof(mat4), nullptr, GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

			//assign pvmatrix to ssbo block index 0
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->PVmatrix);
			//each char uses 1 byte, that makes pointer arithmetric easier
			PVblock_mapped = reinterpret_cast<char*>(glMapNamedBufferRange(this->PVmatrix, 0, sizeof(mat4) * 3, 
				GL_MAP_WRITE_BIT | GL_MAP_COHERENT_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
		}

		/**
		 * @brief Write the new value to shader storage buffer
		*/
		void updatePVmatrix() {
			//low-level buffer mapping
			if (PVblock_mapped == nullptr) {
				throw std::exception("SSBOMappingException");
			}
			//calculate the view matrix without translation
			const mat4 View_notrans = mat4(mat3(this->View));

			//copy buffer in multithreads
			static std::future<void*> bufferUploaders[3];
			//increasing char* will shift 1 byte so we can manipulate the pure address
			
			bufferUploaders[0] = this->command_pool->enqueue_future(memcpy, PVblock_mapped, &this->View, sizeof(mat4));
			bufferUploaders[1] = this->command_pool->enqueue_future(memcpy, PVblock_mapped + sizeof(mat4), &View_notrans, sizeof(mat4));
			bufferUploaders[2] = this->command_pool->enqueue_future(memcpy, PVblock_mapped + 2 * sizeof(mat4), &this->Projection, sizeof(mat4));
			//wait
			for (int i = 0; i < 3; i++) {
				bufferUploaders[i].get();
			}
			//we are using coherent bit, modifications are good to go
		}

		/**
		 * @brief Clear up the engine, called after the rendering loop
		*/
		void destroy() {
			glUnmapNamedBuffer(this->PVmatrix);
			glDeleteBuffers(1, &this->PVmatrix);
			//delete commander
			delete this->command;
			//delete renderers
			delete this->sky;
			delete this->terrain2d_inf;
			//delete thread pool
			delete this->command_pool;
		}

	public:

		/**
		 * @brief Init the master renderer
		 * @param camera The camera for the renderer
		 * @param windowSize The size of the window frame
		 * @param ini The ini setting file for the engine
		*/
		STPMasterRenderer(SglToolkit::SgTCamera*& camera, const int windowSize[2], SIMPLE::SIStorage& ini) : Camera(camera), engineSettings(ini) {
			this->SCR_SIZE[0] = windowSize[0];
			this->SCR_SIZE[1] = windowSize[1];
		}

		~STPMasterRenderer() {
			this->destroy();
		}

		/**
		 * @brief Init the terrain engine, called before each loop
		*/
		void init() {
			glClearColor(121.0f / 255.0f, 151.0f / 255.0f, 52.0f / 255.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

			//glEnable(GL_DEPTH_TEST);
			glDepthMask(GL_TRUE);
			glDepthFunc(GL_LESS);
			glDisable(GL_STENCIL_TEST);

			glEnable(GL_MULTISAMPLE);
			glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
			//glEnable(GL_CULL_FACE);
			glCullFace(GL_BACK);
			glDisable(GL_BLEND);

			//debug output
			static auto debug_callback = [](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) -> void {
				SglToolkit::SgTUtils::debugOutput(source, type, id, severity, length, message, userParam);
			};
			glEnable(GL_DEBUG_OUTPUT);
			glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
			glDebugMessageCallback(debug_callback, nullptr);
			glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, NULL, GL_FALSE);
			glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 0, NULL, GL_FALSE);
			
			glPatchParameteri(GL_PATCH_VERTICES, 3);//barycentric coordinate system

			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

			//starting thread pool
			this->command_pool = new STPThreadPool(3u);
			//loading terrain 2d inf parameters
			STPSettings::STPConfigurations config = STPSettings::STPConfigurations();
			std::future chunksPara_loader = this->command_pool->enqueue_future(STPTerrainParaLoader::getProcedural2DINFChunksParameters, std::ref(this->engineSettings["Generators"]));
			std::future renderPara_loader = this->command_pool->enqueue_future(STPTerrainParaLoader::getProcedural2DINFRenderingParameters, std::ref(this->engineSettings["2DTerrainINF"]));
			config.getChunkSettings() = chunksPara_loader.get();
			config.getMeshSettings() = renderPara_loader.get();
			
			const auto& chunk_settings = config.getChunkSettings();
			std::future noisePara_loader = this->command_pool->enqueue_future(STPTerrainParaLoader::getSimplex2DNoiseParameters, std::ref(this->engineSettings["Generators"]), chunk_settings.MapSize);
			//not quite sure why heightfield_settings isn't got copied to the config, it just share the pointer
			config.getHeightfieldSettings() = STPTerrainParaLoader::getProcedural2DINFGeneratorParameters(this->engineSettings["2DTerrainINF"], chunk_settings.MapSize * chunk_settings.FreeSlipChunk);
			config.getSimplexNoiseSettings() = noisePara_loader.get();

			assert(config.validate());
			const unsigned int unitplane_count = chunk_settings.ChunkSize.x * chunk_settings.ChunkSize.y * chunk_settings.RenderedChunk.x * chunk_settings.RenderedChunk.y;
			
			//rendering commands generation
			this->command = new STPRendererCommander(this->command_pool, unitplane_count);
			//setting up renderers
			this->sky = new STPSkyRenderer(this->engineSettings["SkyboxDay"], this->engineSettings["SkyboxNight"], this->engineSettings["Global"], this->command->Command_SkyRenderer, this->command_pool);
			//setting terrain 2d inf
			STPCompute::STPHeightfieldGenerator::InitGenerator(&config.getHeightfieldSettings());
			this->terrain2d_inf = new STPProcedural2DINF(&config, reinterpret_cast<void*>(this->command->Command_Procedural2DINF));
			this->terrain2d_inf->getChunkProvider().setHeightfieldErosionIteration(std::stoul(this->engineSettings("iteration", "2DTerrainINF")));
			this->terrain2d_inf->loadChunksAsync(this->Camera->getPosition());

			//setting up ssbo
			this->setPVmatrix();
		}

		/**
		 * @brief Check the input each frame
		 * @param frame The current glfw window
		 * @param time Frame time	
		*/
		void input(GLFWwindow*& frame, float time) {
			if (glfwGetKey(frame, GLFW_KEY_W) == GLFW_PRESS) {
				this->Camera->keyUpdate(SglToolkit::SgTSpectatorCamera::FORWARD, time);
			}
			if (glfwGetKey(frame, GLFW_KEY_S) == GLFW_PRESS) {
				this->Camera->keyUpdate(SglToolkit::SgTSpectatorCamera::BACKWARD, time);
			}
			if (glfwGetKey(frame, GLFW_KEY_A) == GLFW_PRESS) {
				this->Camera->keyUpdate(SglToolkit::SgTSpectatorCamera::LEFT, time);
			}
			if (glfwGetKey(frame, GLFW_KEY_D) == GLFW_PRESS) {
				this->Camera->keyUpdate(SglToolkit::SgTSpectatorCamera::RIGHT, time);
			}
			if (glfwGetKey(frame, GLFW_KEY_SPACE) == GLFW_PRESS) {
				this->Camera->keyUpdate(SglToolkit::SgTSpectatorCamera::UP, time);
			}
			if (glfwGetKey(frame, GLFW_KEY_C) == GLFW_PRESS) {
				this->Camera->keyUpdate(SglToolkit::SgTSpectatorCamera::DOWN, time);
			}
			if (glfwGetKey(frame, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
				glfwSetWindowShouldClose(frame, GLFW_TRUE);
			}
		}

		/**
		 * @brief Main rendering functions, called every frame
		 * @param frametime The time in sec spent on each frame
		*/
		void draw(const double& frametime) {
			//start loading terrain 2d inf async
			this->terrain2d_inf->loadChunksAsync(this->Camera->getPosition());

			//clear the default framebuffer
			glClearColor(121.0f / 255.0f, 151.0f / 255.0f, 52.0f / 255.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			//update the view matrix and projection matrix
			this->View = this->Camera->getViewMat();
			this->Projection = perspective(radians(this->Camera->getZoomDeg()), (1.0f * this->SCR_SIZE[0]) / (this->SCR_SIZE[1] * 1.0f), 1.0f, 1500.0f);
			this->updatePVmatrix();

			//rendering sky
			glDisable(GL_CULL_FACE);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);
			this->sky->render(frametime);

			//terrain renderer, choose whichever terrain renderer you like
			glDepthFunc(GL_LESS);
			glEnable(GL_CULL_FACE);
			this->terrain2d_inf->renderVisibleChunks(this->View, this->Projection, this->Camera->getPosition());
		}

		/**
		 * @brief Framebuffer resize callback function
		 * @param width The new width
		 * @param height The new height
		*/
		void reshape(const int& width, const int& height) {
			if (width != 0 && height != 0) {//user has not minimised the window
				//updating the screen size variable
				this->SCR_SIZE[0] = width;
				this->SCR_SIZE[1] = height;
				glViewport(0, 0, width, height);
			}
		}

	};
}

