#pragma once

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
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