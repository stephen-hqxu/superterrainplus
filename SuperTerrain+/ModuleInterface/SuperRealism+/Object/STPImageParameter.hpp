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
	protected:

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
		 * @brief Set the border color when texture is wrapped using border mode.
		 * @param color The border color.
		*/
		virtual void borderColor(glm::vec4) = 0;

		/**
		 * @brief Set the anisotropy filtering mode for the texture.
		 * @param ani The filter level.
		*/
		virtual void anisotropy(STPOpenGL::STPfloat) = 0;

	};

}
#endif//_STP_IMAGE_PARAMETER_HPP_