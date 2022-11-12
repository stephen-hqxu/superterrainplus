#pragma once
#ifndef _STP_TEXTURE_FACTORY_H_
#define _STP_TEXTURE_FACTORY_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Memory
#include "../../../Utility/Memory/STPSmartDeviceMemory.h"
#include "../../../Utility/Memory/STPSmartDeviceObject.h"
//Texture
#include "STPTextureDatabase.h"

#include "../../../Environment/STPChunkSetting.h"

//GL
#include <SuperTerrain+/STPOpenGL.h>
//CUDA
#include <cuda_runtime.h>
//GLM
#include <glm/vec2.hpp>

#include <limits>
#include <functional>
//Container
#include <unordered_map>
#include <vector>

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPTextureFactory wraps texture database and texture splat builder into a fine data structure that allows device and rendering shader to
	 * read such texture configuration to generate a terrain splat texture
	*/
	class STP_API STPTextureFactory {
	public:

		typedef std::underlying_type_t<STPTextureType> STPTextureType_t;

		//The dimension of terrain map in one chunk
		const glm::uvec2 MapDimension;
		//The total number of chunk being rendered
		const glm::uvec2 RenderedChunk;
		const unsigned int RenderedChunkCount;

		//An array of chunk requesting for splatmap generation
		typedef std::vector<STPTextureInformation::STPSplatGeneratorInformation::STPLocalChunkInformation> STPRequestingChunkInfo;

		//Unused type means texture type is registered but no this type of texture can be found in the region.
		static constexpr unsigned int UnusedType = std::numeric_limits<unsigned int>::max();
		//Unregistered type means this type is not enabled for the splatting system.
		static constexpr STPTextureType_t UnregisteredType = std::numeric_limits<STPTextureType_t>::max();

	private:

		//stores local chunk information
		STPSmartDeviceMemory::STPDeviceMemory<STPTextureInformation::STPSplatGeneratorInformation::STPLocalChunkInformation[]> LocalChunkInfo;

		//A collection of all texture data
		std::vector<STPSmartDeviceObject::STPGLTextureObject> Texture;
		std::vector<STPSmartDeviceObject::STPGLBindlessTextureHandle> TextureHandle;
		std::vector<STPOpenGL::STPuint64> RawTextureHandle;

		//texture view settings for each valid texture. View of a texture can be located using the region ID on the splatmap.
		//This data structure ensures all texture regions have one and only one view record.
		std::vector<STPTextureInformation::STPTextureView> TextureViewRecord;

		std::vector<STPTextureInformation::STPTextureDataLocation> TextureRegion;
		//texture region lookup table, if region is not used equivalent -1 will be filled
		std::vector<unsigned int> TextureRegionLookup;
		//texture type being used
		STPTextureDatabase::STPDatabaseView::STPTextureTypeRecord ValidType;

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
		 * @return The splat registry output after the conversion
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
		virtual void splat(cudaTextureObject_t, cudaTextureObject_t, cudaSurfaceObject_t, const STPTextureInformation::STPSplatGeneratorInformation&, cudaStream_t) const = 0;

	protected:

		/**
		 * @brief Get a data structure containing spalt data.
		 * All pointers within are managed by the calling texture factory, and can only be accessed from device.
		 * @return The splat database with all splat data.
		*/
		STPTextureInformation::STPSplatRuleDatabase getSplatDatabase() const noexcept;
	
	public:

		/**
		 * @brief Modifies the sampler state, i.e., texture parameter for a given texture.
		 * Making any change to the texture other than texture parameter is undefined behaviour.
		 * @param texture The name of the texture to modify.
		*/
		typedef std::function<void(STPOpenGL::STPuint)> STPSamplerStateModifier;

		/**
		 * @brief Setup texture factory, manufacture texture data provided and process it to the way that it can be used by texturing system.
		 * After this function returns, all internal states are initialised and no further change can be made.
		 * No reference is retained after the function returns.
		 * @param database_view The pointer to the texture database view which contains all texture information.
		 * @param chunk_setting The configuration about terrain chunk.
		 * @param modify_sampler The sampler state modifier to change the sampler state of a given texture.
		*/
		STPTextureFactory(const STPTextureDatabase::STPDatabaseView&, const STPEnvironment::STPChunkSetting&, const STPSamplerStateModifier&);

		STPTextureFactory(const STPTextureFactory&) = delete;

		STPTextureFactory(STPTextureFactory&&) = delete;

		STPTextureFactory& operator=(const STPTextureFactory&) = delete;

		STPTextureFactory& operator=(STPTextureFactory&&) = delete;

		virtual ~STPTextureFactory() = default;

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
		 * @param stream The CUDA stream generation work will be sent to.
		*/
		void operator()(cudaTextureObject_t, cudaTextureObject_t, cudaSurfaceObject_t, const STPRequestingChunkInfo&, cudaStream_t) const;

		/**
		 * @brief Get a data structure containing splat texture.
		 * All pointers within are managed by the calling texture factory, and can only be accessed from host.
		 * It can then be sent to renderer.
		 * @return The splat texture database with all splat texture.
		*/
		STPTextureInformation::STPSplatTextureDatabase getSplatTexture() const noexcept;

		/**
		 * @brief Convert texture type to index that can be used to locate in the texture registry.
		 * @param type The type to be converted.
		 * @return The index correspond to this type.
		 * If the texture type is not used by texture factory, unregistered texture type identifier is returned.
		*/
		STPTextureType_t convertType(STPTextureType) const;

		/**
		 * @brief Get the number of texture type being used.
		 * @return The number of used type.
		*/
		size_t usedType() const noexcept;

	};

}
#endif//_STP_TEXTURE_FACTORY_H_