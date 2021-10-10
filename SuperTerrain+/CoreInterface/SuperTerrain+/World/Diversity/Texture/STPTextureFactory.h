#pragma once
#ifndef _STP_TEXTURE_FACTORY_H_
#define _STP_TEXTURE_FACTORY_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Memory
#include "../../../Utility/STPSmartDeviceMemory.h"
//Texture
#include "STPTextureSplatBuilder.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief STPDiversity is a series of biome generation algorithm that allows user to define their own implementations
	*/
	namespace STPDiversity {

		/**
		 * @brief STPTextureFactory wraps texture database and texture splat builder into a fine data structure that allows device and rendering shader to 
		 * read such texture configuration to generate a terrain splat texture
		*/
		class STP_API STPTextureFactory {
		private:

			//Array cache data defined in texture information
			STPSmartDeviceMemory::STPDeviceMemory<Sample[]> RegistryLookup;
			STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPSplatRegistry[]> Registry;
			//A linear array that contains splat rules for all samples
			STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPAltitudeNode[]> AltitudeCache;
			STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPGradientNode[]> GradientCache;
			//Locating texture data
			STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPTextureDataLocation[]> TextureLocationCache;
			STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPRegion[]> Region;

			/**
			 * @brief Format texture data in a texture database into texture region such that texture group and data can be referenced using index and texture type.
			 * Texture type and group mapping must be sorted against texture ID, which must be unique.
			 * @param type_mapping The pointer to texture type mapping view.
			 * @param group_mapping The pointer to texture group ID mapping view.
			*/
			void formatRegion(const STPTextureDatabase::STPTypeMappingView&, const STPTextureDatabase::STPGroupView&);

		public:

			/**
			 * @brief Setup texture factory, manufacture texture data provided and process it to the way that it can be used by texturing system.
			 * After this function returns, all internal states are initialised and no further change can be made.
			 * No reference is retained after the function returns.
			 * @param builder The pointer to the splat builder
			 * @param database The pointer to the texture database
			*/
			STPTextureFactory(const STPTextureSplatBuilder&, const STPTextureDatabase&);

			STPTextureFactory(const STPTextureFactory&) = delete;

			STPTextureFactory(STPTextureFactory&&) = delete;

			STPTextureFactory& operator=(const STPTextureFactory&) = delete;

			STPTextureFactory& operator=(STPTextureFactory&&) = delete;

			~STPTextureFactory() = default;

		};

	}
}
#endif//_STP_TEXTURE_FACTORY_H_