#pragma once
#ifndef _STP_CASCADED_SHADOW_MAP_H_
#define _STP_CASCADED_SHADOW_MAP_H_

#include <SuperRealism+/STPRealismDefine.h>
//Rendering Utility
#include "../../Utility/STPCamera.h"
//Base Shadow
#include "STPLightShadow.h"

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
	 * fits the orthographic matrix for each frustum, for each frustum render a shader map as if seen from the directional light.
	 * Finally render the scene with shadow according to fragment depth value from corrected shadow map.
	*/
	class STP_REALISM_API STPCascadedShadowMap : public STPLightShadow {
	public:

		//An array of float that determines the planes position of each camera frustum cut, 
		//starting from the viewing camera near plane as the first array element.
		typedef std::vector<double> STPCascadePlane;

		/**
		 * @brief STPLightFrustum defines the properties of the light frustum.
		*/
		struct STPLightFrustum {
		public:
			
			//Specifies each level of shadow plane.
			//Each far distance should be relative to the far plane of the view frustum.
			STPCascadePlane Division;
			//Specifies a bias value to each sub-frusta.
			//This expands each sub-frusta near and far outward by this amount multiplies the far plane of view frustum,
			//to allow capturing some contents outside the current view.
			//This value is useful when performing cross-cascade blending, such as doing shadow filtering
			double CascadeBandRadius;
			//The pointer to the camera where the light frustum will be constructed based on this camera.
			//This camera must remain valid until the current instance is destroyed.
			STPCamera* Focus;
			//Specifies the depth multiplier of the light frustum.
			//A value of 1.0 specifies a minimum light frustum bounded around the camera view frustum.
			//Increase/Decrease to extend/shrink the shadow frustum range, so objects outside the viewing range can also cast shadow visible to the camera.
			//This value should be tuned based on the scale of the scene.
			double ShadowDistanceMultiplier;

		};

	private:

		//Eight vertices define the corners of a view frustum.
		template<class T>
		using STPFrustumCorner = std::array<T, 8u>;

		//CSM handles directional light rather than positional.
		glm::vec3 LightDirection;

		//This light frustum should use absolute view distance.
		const STPLightFrustum LightFrustum;
		STPCamera::STPSubscriberStatus FocusEventData;
		//Memory to where light space matrices should be stored
		glm::mat4* LightSpaceMatrix;

		/**
		 * @brief Calculate the light space matrix for a particular view frustum.
		 * @param near The near plane.
		 * @param far The far plane.
		 * @param view The view matrix of the camera.
		 * @return The light space view matrix.
		*/
		glm::mat4 calcLightSpace(double, double, const STPMatrix4x4d&) const;

		/**
		 * @brief Trigger a shadow map update due to automatic shadow map update mechanism.
		 * The update is affected by the mask.
		*/
		void requireShadowMapUpdate();

		/**
		 * @brief Calculate the light space view matrices for all divisions of view frustum and store them into mapped light buffer.
		 * @param light_space An array of light space matrix where the calculation will be stored, memory should be sufficient to 
		 * hold all the light cascade.
		*/
		void calcAllLightSpace(glm::mat4*) const;

		void updateShadowMapHandle(STPOpenGL::STPuint64) override;

	public:

		/**
		 * @brief Initialise a directional light instance.
		 * @param resolution Set the resolution of the shadow map for each cascade.
		 * The shadow map must be a square, such that this value specifies the extent length.
		 * @param light_frustum The property of the shadow map light frustum.
		*/
		STPCascadedShadowMap(unsigned int, const STPLightFrustum&);

		STPCascadedShadowMap(const STPCascadedShadowMap&) = delete;

		STPCascadedShadowMap(STPCascadedShadowMap&&) noexcept = default;

		STPCascadedShadowMap& operator=(const STPCascadedShadowMap&) = delete;

		STPCascadedShadowMap& operator=(STPCascadedShadowMap&&) noexcept = default;

		~STPCascadedShadowMap() = default;

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

		bool updateLightSpace() override;

		size_t lightSpaceDimension() const override;

		STPOpenGL::STPuint64 lightSpaceMatrixAddress() const override;

	};

}
#endif//_STP_CASCADED_SHADOW_MAP_H_