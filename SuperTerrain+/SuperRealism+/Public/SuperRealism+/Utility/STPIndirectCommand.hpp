#pragma once
#ifndef _STP_INDIRECT_COMMAND_HPP_
#define _STP_INDIRECT_COMMAND_HPP_

//GL
#include <SuperTerrain+/STPOpenGL.h>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPIndirectCommand is a handy collection of GL indirect drawing commands.
	 * Indirect rendering is the process of issuing a drawing command to OpenGL, 
	 * except that most of the parameters to that command come from GPU storage provided by a Buffer Object.
	*/
	namespace STPIndirectCommand {

		/**
		 * @brief Provides indirect draw command for glDrawArrays related functions.
		*/
		struct STPDrawArray {
		public:

			STPOpenGL::STPuint Count, InstanceCount, First, BaseInstance;

		};

		/**
		 * @brief Provides indirect draw command for glDrawElements related functions.
		*/
		struct STPDrawElement {
		public:

			STPOpenGL::STPuint Count, InstancedCount, FirstIndex;
			STPOpenGL::STPint BaseVertex;
			STPOpenGL::STPuint BaseInstance;

		};

		/**
		 * @brief Provides indirect command for glDispatchCompute function.
		*/
		struct STPDispatchCompute {
		public:

			STPOpenGL::STPuint GroupCountX, GroupCountY, GroupCountZ;

		};

	}

}
#endif//_STP_INDIRECT_COMMAND_HPP_