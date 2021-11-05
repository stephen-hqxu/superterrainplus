#pragma once
#ifndef _STP_TEXTURE_FACTORY_H_
#define _STP_TEXTURE_FACTORY_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Container
#include <unordered_map>
#include <vector>
//Memory
#include "../../../Utility/Memory/STPSmartDeviceMemory.h"
//Texture
#include "STPTextureDatabase.h"

#include "../../../Environment/STPChunkSetting.h"

//GL
#include <SuperTerrain+/STPOpenGL.h>
//CUDA
#include <cuda_runtime.h>
//GLM
#include <glm/vec2.hpp>

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPTextureFactory wraps texture database and texture splat builder into a fine data structure that allows device and rendering shader to
	 * read such texture configuration to generate a terrain splat texture
	*/
	class STP_API STPTextureFactory {
	public:

		//The dimension of terrain map in one chunk
		const glm::uvec2 MapDimension;
		//The total number of chunk being rendered
		const unsigned int RenderedChunkCount;

		/**
		 * @brief STPLocalChunkInformation stores information about rendered local chunks
		*/
		struct STPLocalChunkInformation {
		public:

			//local chunk ID needs to be generated with splatmap.
			unsigned int LocalChunkID;
			//local chunk offset in world coordinate
			glm::vec2 ChunkWorldOffset;

		};

		//An array of chunk requesting for splatmap generation
		typedef std::vector<STPLocalChunkInformation> STPRequestingChunkInfo;

	protected:

		/**
		 * @brief STPSplatDatabase contains arrays of splat rules for terrain splat texture generation on device.
		*/
		struct STPSplatDatabase {
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
		 * @brief STPSplatInformation contains information for generating splatmap in device kernel
		*/
		struct STPSplatInformation {
		public:

			//An array of local chunk information to chunk requesting to generate splatmap, device memory
			STPLocalChunkInformation* RequestingLocalInfo;
			unsigned int LocalCount;

		};

	private:

		//stores local chunk information
		STPSmartDeviceMemory::STPDeviceMemory<STPLocalChunkInformation[]> LocalChunkInfo;

		//A collection of all texture data
		std::vector<STPOpenGL::STPuint> Texture;
		std::vector<STPTextureInformation::STPTextureDataLocation> TextureRegion;
		//texture region lookup table, if region is not used equivalent -1 will be filled
		std::vector<unsigned int> TextureRegionLookup;

		//Convert sample to index in spalt registry
		STPSmartDeviceMemory::STPDeviceMemory<Sample[]> SplatLookup_d;
		size_t SplatLookupCount;
		STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPSplatRegistry[]> SplatRegistry_d;
		//Splat configurations
		STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPAltitudeNode[]> AltitudeRegistry_d;
		STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPGradientNode[]> GradientRegistry_d;

		//Lookup table to convert an ID to index to the final array
		template<typename T>
		using STPIDConverter = std::unordered_map<T, unsigned int>;

		//TODO: maybe we can use template lambda in C++20 and put this function in the constructor
		/**
		 * @brief Convert texture ID to index to texture region.
		 * @tparam N The node type
		 * @param node The splat structure to be converted.
		 * @param converter A lookup table that converts texture ID to index
		 * @return The splat registry output after the convertion
		*/
		template<typename N>
		static std::vector<N> convertSplatID(const STPTextureDatabase::STPDatabaseView::STPNodeRecord<N>&, const STPIDConverter<STPTextureInformation::STPTextureID>&);

		/**
		 * @brief Copy data in a vector to a smart device pointer
		 * @tparam T The type of the data
		 * @param data The source of data to be copied from
		 * @return The smart device pointer with the data copied over.
		*/
		template<typename T>
		static STPSmartDeviceMemory::STPDeviceMemory<T[]> copyToDevice(const std::vector<T>&);

		/**
		 * @brief Launch a device kernel to generate rule-based biome-dependent terrain splatmap 
		 * @param biomemap_tex The biomemap texture object.
		 * @param heightmap_tex The heightmap texture object.
		 * @param splatmap_surf The terrain splatmap surface object. 
		 * @param info The information about the launching kernel.
		 * In addition getSplatDatabase() can be called to retrieve all splat rules.
		 * Moreover information about chunks such as map dimension are available in the base class.
		 * @param stream The kernel stream generation work will be sent to.
		*/
		virtual void splat(cudaTextureObject_t, cudaTextureObject_t, cudaSurfaceObject_t, const STPSplatInformation&, cudaStream_t) const = 0;

	protected:

		/**
		 * @brief Get a data structure containing spalt data.
		 * All pointers within are managed by the calling texture factory, and can only be access from device.
		 * @return The splat database with all splat data.
		*/
		STPSplatDatabase getSplatDatabase() const;

	public:

		/**
		 * @brief Setup texture factory, manufacture texture data provided and process it to the way that it can be used by texturing system.
		 * After this function returns, all internal states are initialised and no further change can be made.
		 * No reference is retained after the function returns.
		 * @param database_view The pointer to the texture database view which contains all texture information
		 * @param chunk_setting The configuration about terrain chunk
		*/
		STPTextureFactory(const STPTextureDatabase::STPDatabaseView&, const STPEnvironment::STPChunkSetting&);

		STPTextureFactory(const STPTextureFactory&) = delete;

		STPTextureFactory(STPTextureFactory&&) = delete;

		STPTextureFactory& operator=(const STPTextureFactory&) = delete;

		STPTextureFactory& operator=(STPTextureFactory&&) = delete;

		virtual ~STPTextureFactory();

		/**
		 * @brief Generate a terrain texture splatmap based on all splat rules set.
		 * For input texture objects addressing mode must be non-repeating.
		 * Invalid texture and surface objects will result in undefined behaviour.
		 * @param biomemap_tex The biomemap texture object.
		 * @param heightmap_tex The heightmap texture object.
		 * @param splatmap_surf The terrain splatmap surface object.
		 * @param requesting_local An array of local chunk information that need to have splatmap computed/updated
		 * If the array size is 0 no operation is performed.
		 * If array size is greater than the intended rendered chunk, exception will be thrown.
		 * @param stream The CUDA stream generation work will be sent to
		*/
		void operator()(cudaTextureObject_t, cudaTextureObject_t, cudaSurfaceObject_t, const STPRequestingChunkInfo&, cudaStream_t) const;

	};

}
#endif//_STP_TEXTURE_FACTORY_H_