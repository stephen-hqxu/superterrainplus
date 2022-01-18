#pragma once
#ifndef _STP_CASCADED_SHADOW_MAP_H_
#define _STP_CASCADED_SHADOW_MAP_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../Object/STPTexture.h"
#include "../Object/STPBindlessTexture.h"
#include "../Object/STPFrameBuffer.h"
#include "../Object/STPBuffer.h"
#include "../Object/STPShaderManager.h"

#include "../Utility/Camera/STPCamera.h"

//System
#include <array>
#include <vector>
#include <optional>

//GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPCascadedShadowMap is a shadow mapping technique that divides view frustum into N subfrusta and 
	 * fits the ortho matrix for each frustum, for each frustum render a shader map as if seen from the directional light.
	 * Finally render the scene with shadow according to fragment depth value from corrected shadow map.
	*/
	class STP_REALISM_API STPCascadedShadowMap {
	public:

		//An array of float that determines the planes position of each frustum cut, starting from the camera near plane as the first array element.
		typedef std::vector<float> STPCascadeLevel;

		/**
		 * @brief STPLightFrustum defines the properties of the light frustum.
		*/
		struct STPLightFrustum {
		public:

			//The X and Y resolution of the shadow map. 
			//Higher resolution gives less jagged shadow but significantly increase memory usageand render time.
			glm::uvec2 Resolution;
			//Specify how the view frustum will be divided.
			STPCascadeLevel Level;
			//The pointer to the camera where the shadow mapping will be using to calculate view frustum.
			//This camera must remain valid until the current instance is destroyed.
			const STPCamera* Camera;
			//Specifies the depth multiplier of the light frustum.
			//A value of 1.0 specifies a minimum light frustum bounded around the camera view frustum.
			float ShadowDistanceMultiplier;
			//Specify the multiplier to bias based on the angle of light and fragment position, and the minimum bias.
			float BiasMultiplier, MinBias;

		};

		/**
		 * @brief STPShadowOption is a visitor to a shadow map instance that allows automatically load up shader settings.
		*/
		class STP_REALISM_API STPShadowOption {
		private:

			friend class STPCascadedShadowMap;

			//The shadow map instance it is depended on.
			const STPCascadedShadowMap* Instance;

			/**
			 * @brief Create a shadow option loader.
			 * @param shadow The pointer to the shadow map instance it is depended on.
			*/
			STPShadowOption(const STPCascadedShadowMap&);

		public:

			//The handle to the shadow map as a bindless texture.
			//This handle remains valid until the dependent shadowm manager instance is destroyed.
			const STPOpenGL::STPuint64 BindlessHandle;

			~STPShadowOption() = default;

			/**
			 * @brief Automatically load the settings into a target macro definer for shader compilation.
			 * @param dictionary The pointer to the dictionary to be loaded.
			*/
			void operator()(STPShaderManager::STPShaderSource::STPMacroValueDictionary&) const;

		};

	private:

		//Eight vertices define the corners of a view frustum.
		typedef std::array<glm::vec4, 8ull> STPFrustumCorner;

		//Capture depths of each cascade
		STPTexture ShadowMap;
		std::optional<STPBindlessTexture> ShadowMapHandle;
		STPFrameBuffer ShadowContainer;

		//Buffer for sharing shadow parameters for all programs
		STPBuffer LightBuffer;
		glm::mat4* BufferLightMatrix;

		const STPCamera& Viewer;
		const STPCascadeLevel ShadowLevel;
		const float ShadowDistance;
		//CSM handles directional light rather than positional.
		glm::vec3 LightDirection;

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
		*/
		void calcAllLightSpace() const;

	public:

		//The resolution of each shadow map within a subfrustum.
		const glm::uvec2 Resolution;

		/**
		 * @brief Initialise a cascaded shadow map.
		 * @param light_frustum The property of the shadow map light frustum.
		*/
		STPCascadedShadowMap(const STPLightFrustum&);

		STPCascadedShadowMap(const STPCascadedShadowMap&) = delete;

		STPCascadedShadowMap(STPCascadedShadowMap&&) = delete;

		STPCascadedShadowMap& operator=(const STPCascadedShadowMap&) = delete;

		STPCascadedShadowMap& operator=(STPCascadedShadowMap&&) = delete;

		virtual ~STPCascadedShadowMap();

		/**
		 * @brief Update the direction of light.
		 * @param dir The new light direction.
		*/
		void setDirection(const glm::vec3&);

		/**
		 * @brief Activate the shadow framebuffer and all rendered contents as depth values will be drawn onto the shadow frame buffer.
		 * To stop capturing, bind to any other framebuffers.
		*/
		void captureLightSpace();

		/**
		 * @brief Clear all depth information.
		*/
		void clearLightSpace();

		/**
		 * @brief Get the number of cascade defined. This is the same as the number of light matrix, or the number of cascade plane plus 1.
		 * @return The number of cascade.
		*/
		size_t cascadeCount() const;

		/**
		 * @brief Get the shadow options loader.
		 * @return The shadow map setup instance, which is bounded to the current shadow map instance.
		*/
		STPShadowOption option() const;

	};

}
#endif//_STP_CASCADED_SHADOW_MAP_H_