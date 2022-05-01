#pragma once
#ifndef _STP_IMAGE_PARAMETER_HPP_
#define _STP_IMAGE_PARAMETER_HPP_

//GL Compatibility
#include <SuperTerrain+/STPOpenGL.h>

//GLM
#include <glm/vec4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPImageParameter is a base class for image-related GL objects.
	*/
	class STPImageParameter {
	public:

		/**
		 * @brief Init a STPImageParameter instance.
		*/
		STPImageParameter() = default;

		virtual ~STPImageParameter() = default;

		/**
		 * @brief Set the texture filtering.
		 * @param min The filter mode for minification.
		 * @param mag The filter mode for magnification.
		*/
		virtual void filter(STPOpenGL::STPenum, STPOpenGL::STPenum) = 0;

		/**
		 * @brief Set the texture wrap mode.
		 * @param s The texture warp mode for X direction.
		 * @param t The texture warp mode for Y direction.
		 * @param r The texture warp mode for Z direction.
		*/
		virtual void wrap(STPOpenGL::STPenum, STPOpenGL::STPenum, STPOpenGL::STPenum) = 0;

		/**
		 * @brief Set the same texture wrap mode for all directions.
		 * @param str The texture warp mode for XYZ direction.
		*/
		virtual void wrap(STPOpenGL::STPenum) = 0;

		/**
		 * @brief Set the border colour when texture is wrapped using border mode.
		 * @param colour The border colour.
		*/
		virtual void borderColor(glm::vec4) = 0;
		//Border colour using integer format.
		virtual void borderColor(glm::ivec4) = 0;
		//Border colour using unsigned integer format.
		virtual void borderColor(glm::uvec4) = 0;

		/**
		 * @brief Set the anisotropy filtering mode for the texture.
		 * @param ani The filter level.
		*/
		virtual void anisotropy(STPOpenGL::STPfloat) = 0;

		/**
		 * @brief Specifies the comparison operator used when GL_TEXTURE_COMPARE_MODE is set to GL_COMPARE_REF_TO_TEXTURE.
		 * @param function The texture compare function.
		*/
		virtual void compareFunction(STPOpenGL::STPint) = 0;

		/**
		 * @brief Specifies the texture comparison mode for currently bound depth textures. That is, a texture whose internal format is GL_DEPTH_COMPONENT_*.
		 * @param mode The texture comparison mode.
		*/
		virtual void compareMode(STPOpenGL::STPint) = 0;

	};

}
#endif//_STP_IMAGE_PARAMETER_HPP_