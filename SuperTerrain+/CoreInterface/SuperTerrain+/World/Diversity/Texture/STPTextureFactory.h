#pragma once
#ifndef _STP_TEXTURE_FACTORY_H_
#define _STP_TEXTURE_FACTORY_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Memory
#include "../../../Utility/STPSmartDeviceMemory.h"
//Texture
#include "STPTextureDatabase.h"

//Container
#include <unordered_map>

//GLAD
#include <glad/glad.h>

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

			//A collection of all texture data
			std::vector<GLuint> Texture;
			std::vector<STPTextureInformation::STPTextureDataLocation> TextureRegion;
			//texture region lookup table, if region is not used equivalent -1 will be filled
			std::vector<unsigned int> TextureRegionLookup;

			//Convert sample to index in spalt registry
			std::vector<Sample> SplatLookup;
			std::vector<STPTextureInformation::STPSplatRegistry> SplatRegistry;

			//Splat configurations
			std::vector<STPTextureInformation::STPAltitudeNode> AltitudeRegistry;
			std::vector<STPTextureInformation::STPGradientNode> GradientRegistry;

			//Convert an ID to index to the final array
			template<typename T>
			using STPIDConverter = std::unordered_map<T, unsigned int>;
			STPIDConverter<STPTextureInformation::STPTextureID> TextureIDConverter;
			STPIDConverter<STPTextureInformation::STPTextureGroupID> GroupIDConverter;

			/**
			 * @brief Convert texture ID to index to texture region.
			 * @tparam N The node type
			 * @param node The splat structure to be converted.
			 * @parm reg The splat registry output after the convertion
			*/
			template<typename N>
			void convertSplatID(const STPTextureDatabase::STPDatabaseView::STPNodeRecord<N>&, std::vector<N>&);

		public:

			/**
			 * @brief Setup texture factory, manufacture texture data provided and process it to the way that it can be used by texturing system.
			 * After this function returns, all internal states are initialised and no further change can be made.
			 * No reference is retained after the function returns.
			 * @param database_view The pointer to the texture database view which contains all texture information
			*/
			STPTextureFactory(const STPTextureDatabase::STPDatabaseView&);

			STPTextureFactory(const STPTextureFactory&) = delete;

			STPTextureFactory(STPTextureFactory&&) = delete;

			STPTextureFactory& operator=(const STPTextureFactory&) = delete;

			STPTextureFactory& operator=(STPTextureFactory&&) = delete;

			~STPTextureFactory();

		};

	}
}
#endif//_STP_TEXTURE_FACTORY_H_