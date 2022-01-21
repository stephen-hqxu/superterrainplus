#pragma once
#ifndef _STP_DIRECTIONAL_LIGHT_H_
#define _STP_DIRECTIONAL_LIGHT_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Utility
#include "../Utility/Camera/STPCamera.h"

//System
#include <array>
#include <vector>

//GLM
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPDirectionalLight is a type of light source that does not have defined position but shoot parallel light.
	 * As it does not have position, shadow mapping is handled using cascaded shadow mapping technique.
	 * This algorithm divides view frustum into N subfrusta and 
	 * fits the ortho matrix for each frustum, for each frustum render a shader map as if seen from the directional light.
	 * Finally render the scene with shadow according to fragment depth value from corrected shadow map.
	*/
	class STP_REALISM_API STPDirectionalLight : private STPCamera::STPStatusChangeCallback {
	public:

		/**
		 * @brief STPLightFrustum defines the properties of the light frustum.
		*/
		struct STPLightFrustum {
		public:

			//The X and Y resolution of the shadow map. 
			//Higher resolution gives less jagged shadow but significantly increase memory usageand render time.
			glm::uvec2 Resolution;
			//An array of float that determines the planes position of each camera frustum cut, 
			//starting from the viewing camera near plane as the first array element.
			std::vector<float> Division;
			//The pointer to the camera where the light frustum will be constructed based on this camera.
			//This camera must remain valid until the current instance is destroyed.
			const STPCamera* Focus;
			//Specifies the depth multiplier of the light frustum.
			//A value of 1.0 specifies a minimum light frustum bounded around the camera view frustum.
			float ShadowDistanceMultiplier;
			//Specify the max and min bias based on the angle of light and fragment position.
			float MaxBias, MinBias;

		};

	private:

		//Eight vertices define the corners of a view frustum.
		typedef std::array<glm::vec4, 8ull> STPFrustumCorner;

		//CSM handles directional light rather than positional.
		glm::vec3 LightDirection;

		//A flag to indicate if light space matrices need to be recalculated
		bool LightSpaceOutdated;

		/**
		 * @brief Calculate the light space matrix for a particular view frustum.
		 * @param near The near plane.
		 * @param far The far plane.
		 * @param view The view matrix of the camera.
		 * @return The light space view matrix.
		*/
		glm::mat4 calcLightSpace(float, float, const glm::mat4&) const;

		/**
		 * @brief Calculate the light space view matrices for all divisions of view frustum and store them into mapped light buffer.
		 * @param light_space An array of light space matrix where the calculation will be stored, memory should be sufficient to 
		 * hold all the light cascade.
		*/
		void calcAllLightSpace(glm::mat4*) const;

		void onMove(const STPCamera&) override;

		void onRotate(const STPCamera&) override;

		void onReshape(const STPCamera&) override;

	public:

		const STPLightFrustum LightFrustum;

		/**
		 * @brief Initialise a directional light instance.
		 * @param light_frustum The property of the shadow map light frustum.
		*/
		STPDirectionalLight(const STPLightFrustum&);

		STPDirectionalLight(const STPDirectionalLight&) = delete;

		STPDirectionalLight(STPDirectionalLight&&) = delete;

		STPDirectionalLight& operator=(const STPDirectionalLight&) = delete;

		STPDirectionalLight& operator=(STPDirectionalLight&&) = delete;

		virtual ~STPDirectionalLight();

		/**
		 * @brief Update the direction of light.
		 * @param dir The new light direction.
		*/
		void setDirection(const glm::vec3&);

		/**
		 * @brief Get the current direction of light.
		 * @return dir The current light direction.
		*/
		const glm::vec3& getDirection() const;

		bool updateLightSpace(glm::mat4*);

		/**
		 * @brief Get the number of light space matrix, or the number of frustum division plane plus 1.
		 * @return The number of light space matrix.
		*/
		size_t lightSpaceSize() const;

	};

}
#endif//_STP_DIRECTIONAL_LIGHT_H_