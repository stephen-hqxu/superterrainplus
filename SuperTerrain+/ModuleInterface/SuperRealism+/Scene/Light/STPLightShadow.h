#pragma once
#ifndef _STP_LIGHT_SHADOW_H_
#define _STP_LIGHT_SHADOW_H_

#include <SuperRealism+/STPRealismDefine.h>
//Shadow
#include "STPShadowMapFilter.hpp"
//GL Object
#include "../../Object/STPBindlessBuffer.h"
#include "../../Object/STPBindlessTexture.h"
#include "../../Object/STPFrameBuffer.h"

#include <optional>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPLightShadow is a base class for all different types of shadow.
	*/
	class STP_REALISM_API STPLightShadow {
	public:

		/**
		 * @brief STPShadowMapFormat defines the texture target of the shadow map.
		*/
		enum class STPShadowMapFormat : unsigned char {
			//A scalar shadow map.
			Scalar = 0x00u,
			//An array shadow map.
			Array = 0x01u,
			//A cubemap shadow map.
			Cube = 0x02u
		};

	private:

		//Please note that all std::optional fields are required, they are just for ease of deferred initialisation.

		//A rendering texture that contains depth information about the scene.
		std::optional<STPTexture> ShadowMap;
		//A bindless handle to the shadow map.
		STPBindlessTexture ShadowMapHandle;

		STPFrameBuffer ShadowMapContainer;

	protected:
		
		//A buffer stores data of the shadow light.
		STPBuffer ShadowData;
		//An address pointing to the shadow data to be shared with shaders.
		STPBindlessBuffer ShadowDataAddress;

		/**
		 * @brief Trigger a update to the shadow map bindless texture handle.
		 * Usually the implementation should store the handle into the shadow map buffer.
		 * @param handle The new handle.
		*/
		virtual void updateShadowMapHandle(STPOpenGL::STPuint64) = 0;

		/**
		 * @brief Get the handle to the shadow map texture.
		 * @return The shadow map texture bindless handle.
		*/
		STPOpenGL::STPuint64 shadowMapHandle() const;

	public:

		//The shadow map format which determines the texture target of the shadow map. 
		//The format depends on which shadow map technique is used.
		const STPShadowMapFormat ShadowMapFormat;
		//The resolution of the shadow map.
		//The shadow map is required to be a square such that this represents the extent length.
		const unsigned int ShadowMapResolution;

		/**
		 * @brief Init a light shadow instance.
		 * @param resolution The resolution of the shadow map. Specifically this should specify the extent length of the shadow map.
		 * @param format Specifies the format of the targeting shadow map.
		*/
		STPLightShadow(unsigned int, STPShadowMapFormat);

		STPLightShadow(const STPLightShadow&) = delete;

		STPLightShadow(STPLightShadow&&) noexcept = default;

		STPLightShadow& operator=(const STPLightShadow&) = delete;

		STPLightShadow& operator=(STPLightShadow&&) = delete;

		virtual ~STPLightShadow() = default;

		/**
		 * @brief Check and update the light space matrix that converts an object from world space to light clip space.
		 * The updated light space matrix will be written to the internal memory.
		 * @return A status flag to indicate if any value has been written into the memory, true if memory has been written, false otherwise.
		 * After this function call the internal status will be reset as "light space matrix is now up-to-date".
		*/
		virtual bool updateLightSpace() = 0;

		/**
		 * @brief Get the size of light space for this shadow mapping technique.
		 * @return The number of element in the light space matrix.
		 * For scalar shadow map, this should be 1.
		 * For array shadow map, this should be the number of array element in the shadow map.
		 * For cubemap shadow map, this should be 6.
		*/
		virtual size_t lightSpaceDimension() const = 0;

		/**
		 * @brief Get the address of the shadow data buffer to the light space matrix.
		 * @return The address to the light space matrix in the shadow data buffer.
		*/
		virtual STPOpenGL::STPuint64 lightSpaceMatrixAddress() const = 0;

		/**
		 * @brief Trigger a force update to the light space information.
		*/
		virtual void forceLightSpaceUpdate() = 0;

		/**
		 * @brief Set the texture used to capture scene depth information.
		 * This function should usually be called by the rendering pipeline automatically.
		 * Note that the last two parameters are only used when the shadow map filter corresponds to a compatible shadow map type.
		 * @param shadow_filter The shadow map filter used.
		 * @param level The number of level to be allocated to the shadow map.
		 * @param anisotropy The anisotropy filtering level to be used.
		*/
		void setShadowMap(STPShadowMapFilter, STPOpenGL::STPint = 1, STPOpenGL::STPfloat = 1.0f);

		/**
		 * @brief Get the address to the buffer stores shadow data.
		 * @return The address to to shadow data buffer to be shared.
		*/
		STPOpenGL::STPuint64 shadowDataAddress() const;

		/**
		 * @brief Start rendering scene depth into the current light shadow memory.
		 * To stop rendering, bind the framebuffer target to any other binding point.
		*/
		void captureDepth() const;

		/**
		 * @brief Clear colour in the shadow map.
		 * This option is ignored if the shadow map has no colour channel.
		 * @param clear_color The colour to be used to clear the shadow map to.
		*/
		void clearShadowMapColor(const glm::vec4&);

		/**
		 * @brief Generate shadow mipmap.
		*/
		void generateShadowMipmap();

	};

}
#endif//_STP_LIGHT_SHADOW_H_