#pragma once
#ifndef _STP_SERIALISATIONEXCEPTION_HPP_
#define _STP_SERIALISATIONEXCEPTION_HPP_

//System
#include <string>
#include <exception>

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief STPSerialisationException will be thrown when serialisation fails to operate.
	*/
	class STPSerialisationException : public std::exception {
	private:

		//error message
		const std::string error;

	public:

		/**
		 * @brief Create a new exception case
		 * @param reason The reason for this exception
		*/
		STPSerialisationException(std::string reason) noexcept : error(reason) {

		}

		~STPSerialisationException() override {

		}

		const char* what() const noexcept override {
			return this->error.c_str();
		}

	};
}
#endif //_STP_SERIALISATIONEXCEPTION_HPP_