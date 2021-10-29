#pragma once

namespace STPDemo {

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