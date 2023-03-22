#pragma once
#ifndef _STP_TEXTURE_INFORMATION_HPP_
#define _STP_TEXTURE_INFORMATION_HPP_

//Biome
#include "../STPBiomeDefine.h"

#ifndef __CUDACC_RTC__
#include <SuperTerrain+/STPOpenGL.h>
#endif

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPTextureInformation stores texture data and essential information for how to apply texture to the generated terrain
	*/
	namespace STPTextureInformation {

		//Each texture collection has a unique ID to uniquely identify a texture with different types in the database
		typedef unsigned int STPTextureID;
		//To uniquely identify a texture map group.
		typedef unsigned int STPMapGroupID;
		//To uniquely identify a texture view group.
		typedef unsigned int STPViewGroupID;

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
				//The number of element in the gradient table. Zero is none.
				GradientSize = 0u;

		};

		/**
		 * @brief STPSplatRuleDatabase contains arrays of all splat rules and other important information for terrain splat texture generation on device.
		*/
		struct STPSplatRuleDatabase {
		public:

			//An array of sample, the index of a sample can be used to locate the sample in the splat registry.
			const Sample* SplatRegistryDictionary;
			unsigned int DictionaryEntryCount;
			//An array that contains terrain splat configuration for each sample.
			const STPTextureInformation::STPSplatRegistry* SplatRegistry;

			//An array containing all altitude splatting rules.
			const STPTextureInformation::STPAltitudeNode* AltitudeRegistry;
			//An array containing all gradient splatting rules.
			const STPTextureInformation::STPGradientNode* GradientRegistry;

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
			const STPLocalChunkInformation* RequestingLocalInfo;
			//The number of local chunk needs to be generated with splatmap
			unsigned int LocalCount;

		};

//those structures are used by the renderer only, and contains typedef's not recognised by NVRTC
#ifndef __CUDACC_RTC__
		/**
		 * @brief STPTextureView allows specifying view of a texture.
		*/
		struct STPTextureView {
		public:

			//Control the UV scaling factor of a texture.
			//The primary scale will be used when the viewer is very closed, 
			//the scale is reduced to secondary when the viewer is further, then tertiary.
			unsigned int PrimaryScale, SecondaryScale, TertiaryScale;

		};

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
		 * @brief STPSplatTextureDatabase contains arrays of all splat texture for terrain texture splatting in the renderer.
		*/
		struct STPSplatTextureDatabase {

			//An array of OpenGL handles to bindless texture buffer objects in the form of texture 2D array.
			const STPOpenGL::STPuint64* TextureHandle;
			size_t TextureHandleCount;
			//An array of structure of indices to locate a specific type of texture for a region in the texture buffer array.
			const STPTextureDataLocation* LocationRegistry;
			size_t LocationRegistryCount;
			//An array of indices to locate a region in the location registry.
			const unsigned int* LocationRegistryDictionary;
			size_t LocationRegistryDictionaryCount;

			//An array of texture views for each texture in the splat region.
			const STPTextureView* TextureViewRegistry;
			//The total number of splat region used on a splatmap.
			size_t SplatRegionCount;
		};
#endif//__CUDACC_RTC__

	}
}
#endif//_STP_TEXTURE_INFORMATION_HPP_