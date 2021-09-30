#pragma once
#ifndef _STP_TEXTURE_FACTORY_H_
#define _STP_TEXTURE_FACTORY_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Texture
#include "STPTextureSplatBuilder.h"
#include "STPTextureDatabase.h"

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
		public:

			/**
			 * @brief STPTextureDataLocation defines the location of a specific texture data in an array of layered texture
			*/
			struct STPTextureDataLocation {
			public:

				//The index to the layered texture.
				unsigned int GroupIdx;
				//The index to the texture in a layer.
				unsigned int LayerIdx;

			};

			//A region provides data to locate texture types for a texture collection
			typedef STPTextureDataLocation* STPRegion[static_cast<std::underlying_type_t<STPTextureType>>(STPTextureType::TYPE_COUNT)];

			/**
			 * @brief A base node for different types of structural regions
			*/
			struct STPStructureNode {
			public:

				//The index to the texture region array for this active region
				unsigned int RegionIdx;

			};

			/**
			 * @brief A single node which contains a configuration for an active region using altitude rule.
			 * Instead of mapping upper bound to texture ID referencing the texture database, it maps to the index to a group of texture types, so called region.
			*/
			struct STPAltitudeNode : public STPStructureNode {
			public:

				//The upper bound of the altitude which makes the current texture region active for this sample
				float UpperBound;

			};

			/**
			 * @brief A single node which contains a configuration for an active region using gradient rule.
			 * Instead of mapping each gradient configuration to texture ID, it maps to the index to a group of texture types, so called region.
			*/
			struct STPGradientNode : public STPStructureNode {
			public:

				//region starts from gradient higher than this value
				float minGradient;
				//region ends with gradient lower than this value
				float maxGradient;
				//region starts from altitude higher than this value
				float LowerBound;
				//region ends with altitude lower than this value
				float UpperBound;

			};

			/**
			 * @brief STPSplatRegistry contains information to the texture splat settings for a particular sample
			*/
			struct STPSplatRegistry {
			public:

				typedef Sample* STPRegistryDictionary;

				//The pointer to the altitude setup for this sample. nullptr if there is no altitude configuration.
				const STPAltitudeNode* AltitudeTable;
				//The number of element in the altitude table. Zero if none.
				unsigned int AltitudeSize;

				//The pointer to the gradient setup for this sample. nullptr if there is no gradient configuration.
				const STPGradientNode* GradientTable;
				//The number of elemenet in the gradient table. Zero is none.
				unsigned int GradientSize;

			};

		private:


		public:

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