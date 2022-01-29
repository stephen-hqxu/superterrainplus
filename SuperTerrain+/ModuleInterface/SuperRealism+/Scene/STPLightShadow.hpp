#pragma once
#ifndef _STP_LIGHT_SHADOW_HPP_
#define _STP_LIGHT_SHADOW_HPP_

//GLM
#include <glm/mat4x4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPLightShadow is a base class for all different types of shadow.
	*/
	class STPLightShadow {
	public:

		STPLightShadow() = default;

		virtual ~STPLightShadow() = default;

		/**
		 * @brief Check and update the light space matrix that converts an object from world space to light clip space.
		 * @param light_space A pointer to a, or an array or light space matrix depends on implementation,
		 * where the updated light space will be written.
		 * Note that this pointer should be allocated with sufficient size to hold all light space matrix/matrices.
		 * This function should not writen anything to the memory, if the internal status says the light space is still up-to-date,
		 * hence it is recommended to call this function using the same pointer.
		 * @return A status flag to indicate if any value has been written into the memory, true if memory has been writtn, false otherwise.
		 * After this function call the internal status will be reset as "light space matrix is now up-to-date".
		*/
		virtual bool updateLightSpace(glm::mat4*) const = 0;

		/**
		 * @brief Get the size of light space for this shadow mapping technique.
		 * @return The number of element in the light space matrix.
		*/
		virtual size_t lightSpaceDimension() const = 0;

		/**
		 * @brief Trigger a force update to the light space information.
		*/
		virtual void forceLightSpaceUpdate() = 0;

	};

}
#endif//_STP_LIGHT_SHADOW_HPP_