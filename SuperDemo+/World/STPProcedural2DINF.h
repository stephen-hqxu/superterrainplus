#pragma once
#ifndef _STP_PROCEDURAL_2D_INF_H_
#define _STP_PROCEDURAL_2D_INF_H_

//System
#include <iostream>
//OpenGL
#include <glad/glad.h>
//My Own Library
#include <SglToolkit/SgTShaderProc.h>
#include <SglToolkit/SgTUtil.h>
//Processing data with chunk manager so we can use the map directly
#include <SuperTerrain+/World/Chunk/STPChunkManager.h>

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
namespace STPDemo {

	/**
	 * @brief STPProcedural2DINF will create, render and update chunks with generated heightfield when rendering a 2D procedural infinite terrain.
	 * The infinite terrain is represented by procedurally generated heightmap and dynamically rendered chunks base on camera location.
	 * Each chunk is consist of a number of unit planes laying in horizontal orientaion.
	 * The chunk is defined by the its size and its location at top left corner (e.g., a 16x16 chunk (0,0) contains 16x16 unit planes and the first unit plane starts at (0,0)).
	 * The engine will auto update required chunks to be rendered based on player's location.
	 * STPProcedural2DINF also auto-manages the rendering of the chunks and LoD of the rendered chunks.
	 * Overall, this is an amazing class for rendering endless procedural terrain.
	*/
	class STPProcedural2DINF {
	private:

		//drawing command
		const void* const command;
		//Chunk storage
		GLuint plane_vbo, plane_vao, plane_ebo, plane_indirect;
		SglToolkit::SgTShaderProc Terrain2d_shader;
		GLuint Terrain2d_pipeline;

		//terrain rendering settings
		const SuperTerrainPlus::STPEnvironment::STPMeshSetting& MeshSetting;

		//chunk manager for this renderer
		SuperTerrainPlus::STPChunkManager& ChunkManager;

		/**
		 * @brief Compile the 2D terrain shader
		 * @return True if successfully compiled
		*/
		const bool compile2DTerrainShader();

		/**
		 * @brief Calculate the base chunk position (the coordinate of top-left corner) for the most top-left corner chunk
		 * @return The base chunk position
		*/
		glm::vec2 calcBaseChunkPosition();

		/**
		 * @brief Load the unit plane
		*/
		void loadPlane();

		/**
		 * @brief Get uniform location of the 2D infinite terrain renderer
		 * @param name Name of the uniform
		 * @return The uniform location
		*/
		GLint getLoc(const GLchar* const) const;

		/**
		 * @brief Clearup the program, deleting buffers and exit
		*/
		void clearup();

	public:

		/**
		 * @brief Init the chunk manager
		 * @param mesh_settings Stored all parameters for the heightmap calculation launch, settings are linked to this renderer so lifetime must be guaranteed.
		 * @param manager Chunk manager to be linked with this renderer
		 * @param procedural2dinf_cmd The indirect rendering command for prodecural 2d inf terrain renderer
		*/
		STPProcedural2DINF(const SuperTerrainPlus::STPEnvironment::STPMeshSetting&, SuperTerrainPlus::STPChunkManager&, void*);

		STPProcedural2DINF(const STPProcedural2DINF&) = delete;

		STPProcedural2DINF(STPProcedural2DINF&&) = delete;

		~STPProcedural2DINF();

		STPProcedural2DINF& operator=(const STPProcedural2DINF&) = delete;

		STPProcedural2DINF& operator=(STPProcedural2DINF&&) = delete;

		/**
		 * @brief Get the 2d infinite terrain program
		 * @return The program reference
		*/
		GLuint getTerrain2DINFProgram() const;

		/**
		 * @brief Render the procedural infinite 2D terrain
		 * @param view - The camera view matrix
		 * @param projection - The camera projection matrix
		 * @param position - The camera position
		*/
		void renderVisibleChunks(const glm::mat4&, const glm::mat4&, const glm::vec3&) const;

	};
}
#endif//_STP_PROCEDURAL_2D_INF_H_