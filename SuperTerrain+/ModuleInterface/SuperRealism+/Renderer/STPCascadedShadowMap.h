#pragma once
#ifndef _STP_CASCADED_SHADOW_MAP_H_
#define _STP_CASCADED_SHADOW_MAP_H_

#include <SuperRealism+/STPRealismDefine.h>
//GL Object
#include "../Object/STPTexture.h"
#include "../Object/STPBindlessTexture.h"
#include "../Object/STPFrameBuffer.h"
#include "../Object/STPBuffer.h"

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
			//Specify the max and min bias based on the angle of light and fragment position.
			float MaxBias, MinBias;

		};

		/**
		 * @brief STPBufferAllocation indicates an allocated section of memory in a shared buffer.
		*/
		struct STPBufferAllocation {
		public:

			STPBuffer* SharedMemory;
			//Define the offset in byte in the shared memory to locate the light matrix
			size_t Start;

			//The pointer directly pointing (with start offset applied to the base pointer of shared memory region) to the allocated memory region.
			glm::mat4* LightMatrix;

		};

	private:

		//Eight vertices define the corners of a view frustum.
		typedef std::array<glm::vec4, 8ull> STPFrustumCorner;

		//Capture depths of each cascade
		STPTexture ShadowMap;
		std::optional<STPBindlessTexture> ShadowMapHandle;
		STPFrameBuffer ShadowContainer;

		//Allocated memory to the light buffer where light information will be sent.
		//This memory must be initialised before shadow class can be used, or undefined behaviour.
		STPBufferAllocation LightBuffer;

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
		 * @param light_space An array of light space matrix where the calculation will be stored, memory should be sufficient to 
		 * hold all the light cascade.
		*/
		void calcAllLightSpace(glm::mat4*) const;

	public:

		const STPLightFrustum LightFrustum;

		/**
		 * @brief Initialise a cascaded shadow map.
		 * @param light_frustum The property of the shadow map light frustum.
		*/
		STPCascadedShadowMap(const STPLightFrustum&);

		STPCascadedShadowMap(const STPCascadedShadowMap&) = delete;

		STPCascadedShadowMap(STPCascadedShadowMap&&) = delete;

		STPCascadedShadowMap& operator=(const STPCascadedShadowMap&) = delete;

		STPCascadedShadowMap& operator=(STPCascadedShadowMap&&) = delete;

		virtual ~STPCascadedShadowMap() = default;

		/**
		 * @brief Set the shadow map shared memory to a allocated region of memory.
		 * This memory will be used by the current instance to update any light information for shadow calculation.
		 * @param allocation The block of memory for updating shadow information.
		*/
		void setLightBuffer(const STPBufferAllocation&);

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
		 * @brief Get the bindless handle to the shadow map.
		 * @return A bindless handle, this handle remains valid as long as the instance is valid.
		*/
		STPOpenGL::STPuint64 handle() const;

	};

}
#endif//_STP_CASCADED_SHADOW_MAP_H_