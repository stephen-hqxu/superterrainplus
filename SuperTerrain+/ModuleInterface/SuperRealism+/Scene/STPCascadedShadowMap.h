#pragma once
#ifndef _STP_CASCADED_SHADOW_MAP_H_
#define _STP_CASCADED_SHADOW_MAP_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Utility
#include "../Utility/Camera/STPCamera.h"
//Base Shadow
#include "STPLightShadow.hpp"

//System
#include <array>
#include <vector>

//GLM
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPCascadedShadowMap is a type of shadow mapping technique for directional light source.
	 * This algorithm divides view frustum into N subfrusta and 
	 * fits the ortho matrix for each frustum, for each frustum render a shader map as if seen from the directional light.
	 * Finally render the scene with shadow according to fragment depth value from corrected shadow map.
	*/
	class STP_REALISM_API STPCascadedShadowMap : public STPLightShadow, private STPCamera::STPStatusChangeCallback {
	public:

		//An array of float that determines the planes position of each camera frustum cut, 
		//starting from the viewing camera near plane as the first array element.
		typedef std::vector<float> STPCascadePlane;

		/**
		 * @brief STPLightFrustum defines the properties of the light frustum.
		*/
		struct STPLightFrustum {
		public:
			
			//Specifies each level of shadow plane.
			STPCascadePlane Division;
			//The pointer to the camera where the light frustum will be constructed based on this camera.
			//This camera must remain valid until the current instance is destroyed.
			const STPCamera* Focus;
			//Specifies the depth multiplier of the light frustum.
			//A value of 1.0 specifies a minimum light frustum bounded around the camera view frustum.
			float ShadowDistanceMultiplier;

		};

	private:

		//Eight vertices define the corners of a view frustum.
		typedef std::array<glm::vec4, 8ull> STPFrustumCorner;

		//CSM handles directional light rather than positional.
		glm::vec3 LightDirection;

		const STPLightFrustum LightFrustum;

		//A flag to indicate if light space matrices need to be recalculated
		mutable bool LightSpaceOutdated;

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

		/**
		 * @brief Initialise a directional light instance.
		 * @param light_frustum The property of the shadow map light frustum.
		*/
		STPCascadedShadowMap(const STPLightFrustum&);

		STPCascadedShadowMap(const STPCascadedShadowMap&) = delete;

		STPCascadedShadowMap(STPCascadedShadowMap&&) = delete;

		STPCascadedShadowMap& operator=(const STPCascadedShadowMap&) = delete;

		STPCascadedShadowMap& operator=(STPCascadedShadowMap&&) = delete;

		virtual ~STPCascadedShadowMap();

		/**
		 * @brief Get an array of float that specifies how the camera is divided into different levels of cascade.
		 * @return An array of float with division information.
		*/
		const STPCascadePlane& getDivision() const;

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

		bool updateLightSpace(glm::mat4*) const override;

		size_t lightSpaceDimension() const override;

		void forceLightSpaceUpdate() override;

	};

}
#endif//_STP_CASCADED_SHADOW_MAP_H_