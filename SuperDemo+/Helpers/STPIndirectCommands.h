#pragma once

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief glDrawElementsIndirect
	*/
	typedef struct {
		GLuint Count;
		GLuint InstanceCount;
		GLuint FirstIndex;
		GLuint BaseVertex;
		GLuint BaseInstance;
	} DrawElementsIndirectCommand;

	/**
	 * @brief glDrawArraysIndirect
	*/
	typedef struct {
		GLuint Count;
		GLuint InstanceCount;
		GLuint First;
		GLuint BaseInstance;
	} DrawArraysIndirectCommand;
}