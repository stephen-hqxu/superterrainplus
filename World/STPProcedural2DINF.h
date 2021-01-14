#pragma once
#ifndef _STP_PROCEDURAL_2D_INF_H_
#define _STP_PROCEDURAL_2D_INF_H_

//System
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
//OpenGL
#include "glad/glad.h"
//My Own Library
#include "SglToolkit1.0/SgTShaderProc.h"
#include "SglToolkit1.0/SgTUtils.h"
//GLM
using glm::mat4;
using glm::identity;
//Processing data with chunk manager so we can use the map directly
#include "Chunk/STPChunkManager.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPProcedural2DINF will create, render and update chunks with generated heightfield when rendering a 2D procedural infinite terrain.
	 * The infinite terrain is represented by procedurally generated heightmap and dynamically rendered chunks base on camera location.
	 * Each chunk is consist of a number of unit planes laying in horizontal orientaion.
	 * The chunk is defined by the its size and its location at top left corner (e.g., a 16x16 chunk (0,0) contains 16x16 unit planes and the first unit plane starts at (0,0)).
	 * The engine will auto update required chunks to be rendered based on player's location.
	 * STPProcedural2DINF also auto-manages the rendering of the chunks and LoD of the rendered chunks.
	 * Overall, this is an amazing class for rendering endless procedural terrain.
	*/
	class STPProcedural2DINF final : public STPChunkManager {
	private:

		//Calculated chunk data
		//Calculate the coordinate of the top-left corner of each chunk
		//It will be used as the key pair to search the map from our hash table
		vec2* BaseChunkPosition = nullptr;

		//drawing command
		const void* const command;
		//Chunk storage
		GLuint plane_vbo, plane_vao, plane_ebo, plane_basechunkposition, plane_indirect;
		SglToolkit::SgTShaderProc Terrain2d_shader;
		GLuint Terrain2d_pipeline;

		//terrain rendering settings
		const STPSettings::STPMeshSettings RenderingSettings;

		/**
		 * @brief Compile the 2D terrain shader
		 * @return True if successfully compiled
		*/
		const bool compile2DTerrainShader();

		/**
		 * @brief Calculate the base chunk position (the coordinate of top-left corner) for each chunk
		*/
		void calcBaseChunkPosition();

		/**
		 * @brief Load the unit plane
		*/
		void loadPlane();

		/**
		 * @brief Get uniform location of the 2D infinite terrain renderer
		 * @param name Name of the uniform
		 * @return The uniform location
		*/
		GLint getLoc(const GLchar* const);

		/**
		 * @brief Clearup the program, deleting buffers and exit
		*/
		void clearup();

	public:

		/**
		 * @brief Init the chunk manager
		 * @param settings Stored all parameters for the heightmap calculation launch, settings are copied later so no need to keep the object alive
		 * @param procedural2dinf_cmd The indirect rendering command for prodecural 2d inf terrain renderer
		 * @param shared_threadpool The thread pool for multithreading computing and rendering. It can share the pool with the main program or separate from.
		*/
		STPProcedural2DINF(STPSettings::STPConfigurations*, void* const, STPThreadPool* const);

		~STPProcedural2DINF();

		/**
		 * @brief Get the 2d infinite terrain program
		 * @return The program reference
		*/
		GLuint getTerrain2DINFProgram();

		/**
		 * @brief Render the procedural infinite 2D terrain
		 * @param view - The camera view matrix
		 * @param projection - The camera projection matrix
		 * @param position - The camera position
		*/
		void renderVisibleChunks(const mat4&, const mat4&, const vec3&);

	};
}
#endif//_STP_PROCEDURAL_2D_INF_H_