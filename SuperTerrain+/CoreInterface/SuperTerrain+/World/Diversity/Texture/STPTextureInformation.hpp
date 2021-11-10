#pragma once
#ifndef _STP_TEXTURE_INFORMATION_HPP_
#define _STP_TEXTURE_INFORMATION_HPP_

//Biome
#include "../STPBiomeDefine.h"

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPTextureInformation stores texture data and essential information for how to apply texture to the generated terrain
	*/
	namespace STPTextureInformation {

		//Each texture collection has a unique ID to uniquely identify a texture with different types in the database
		typedef unsigned int STPTextureID;
		//Each group has an ID to uniquely identify a texture group in the database
		typedef unsigned int STPTextureGroupID;

		/**
		 * @brief STPTextureDataLocation defines the location of a specific texture data in an array of layered texture
		*/
		struct STPTextureDataLocation {
		public:

			//The index to the layered texture.
			unsigned int GroupIndex;
			//The index to the texture in a layer.
			unsigned int LayerIndex;

		};

		/**
		 * @brief A base node for different types of structural regions
		*/
		struct STPStructureNode {
		public:

			//A reference that tells where to find a structure, depends on context the key may behave differently
			union {

				//The texture ID referencing a texture entry in a texture database instance
				STPTextureID DatabaseKey;

				//The index to the texture region registry for this active region
				unsigned int RegionIndex;

			} Reference;

		};

		/**
		 * @brief A single node which contains a configuration for an active region using altitude rule.
		*/
		struct STPAltitudeNode : public STPStructureNode {
		public:

			//The upper bound of the altitude which makes the current texture region active for this sample
			float UpperBound;

		};

		/**
		 * @brief A single node which contains a configuration for an active region using gradient rule.
		 * A region will only be considered as active when all conditions are satisfied
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

			//The index to the altitude setup for this sample. If there is no altitude configuration, denoted by size, value will be undefined.
			unsigned int AltitudeEntry,
				//The number of element in the altitude table. Zero if none.
				AltitudeSize = 0u;

			//The index to the gradient setup for this sample. If there is no gradient configuration, denoted by size, value will be undefined.
			unsigned int GradientEntry,
				//The number of elemenet in the gradient table. Zero is none.
				GradientSize = 0u;

		};

		/**
		 * @brief STPSplatRuleDatabase contains arrays of all splat rules and other important information for terrain splat texture generation on device.
		*/
		struct STPSplatRuleDatabase {
		public:

			//An array of sample, the index of a sample can be used to locate the sample in the splat registry.
			Sample* SplatRegistryDictionary;
			unsigned int DictionaryEntryCount;
			//An array that contains terrain splat configuration for each sample.
			STPTextureInformation::STPSplatRegistry* SplatRegistry;

			//An array containing all altitude splating rules.
			STPTextureInformation::STPAltitudeNode* AltitudeRegistry;
			//An array containing all gradient splating rules.
			STPTextureInformation::STPGradientNode* GradientRegistry;

		};

		/**
		 * @brief Contains information for generating splatmap in device kernel
		*/
		struct STPSplatGeneratorInformation {
		public:

			/**
			 * @brief Stores information about rendered local chunks
			*/
			struct STPLocalChunkInformation {
			public:

				//local chunk coordinate needs to be generated with splatmap.
				unsigned int LocalChunkCoordinateX, LocalChunkCoordinateY;
				//local chunk offset of each map in for the current world coordinate
				float ChunkMapOffsetX, ChunkMapOffsetY;

			};
			
			//An array of local chunk information to chunk requesting to generate splatmap, device memory
			STPLocalChunkInformation* RequestingLocalInfo;
			//The number of local chunk needs to be generated with splatmap
			unsigned int LocalCount;

		};

	};
}
#endif//_STP_TEXTURE_INFORMATION_HPP_