#pragma once
#ifndef _STP_HEIGHTFIELD_GENERATOR_CUH_
#define _STP_HEIGHTFIELD_GENERATOR_CUH_

//System
#include <mutex>
#include <vector>
//CUDA
//CUDA lib are included in the "Engine" section
#include <curand_kernel.h>
//Engine
#include "STPRainDrop.cuh"
#include "STPDiversityGenerator.hpp"
#include "../Helpers/STPMemoryPool.hpp"
//Settings
#include "../Settings/STPHeightfieldSettings.hpp"
#include "../Settings/STPChunkSettings.hpp"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief GPGPU compute suites for Super Terrain + program, powered by CUDA
	*/
	namespace STPCompute {
		
		/**
		 * @brief Generate the terriain height map, the height map will be then used to
		 * represent the offset in y direction of the terrain. Heightfield is generated by diversity generator.
		*/
		class STPHeightfieldGenerator {
		public:

			//STPGeneratorOperation controls the operations to perform during heightfield generation
			typedef unsigned short STPGeneratorOperation;

			//TODO You can change your preferred RNG here!
			//Choosen generator for curand
			typedef curandStatePhilox4_32_10 curandRNG;

			//Generate a new heightmap and store the result in the provided memory space
			constexpr static STPGeneratorOperation HeightmapGeneration = 1u << 0u;
			//Erode the heightmap. If HeightmapGeneration flag is not enabled, an available heightmap needs to be provided for the operation
			constexpr static STPGeneratorOperation Erosion = 1u << 1u;
			//Generate normal map and integrate into heightfield. If HeightmapGeneration flag is not enabled, an available heightmap needs to be provided for the operation
			//RGB channel will then contain normalmap and A channel contains heightmap
			//Then format the heightfield map from FP32 to INT16.
			constexpr static STPGeneratorOperation RenderingBufferGeneration = 1u << 2u;

			/**
			 * @brief STPMapStorage stores heightfield data for the generator
			*/
			struct STPMapStorage {
			public:

				//- A Sample array (sample is implementation defined, usually it's uint16) where biomemap is located.
				//- If biome interpolation if enabled, the number of biomemap should be the same as that in Heightmap32F.
				//See documentation of Heightmap32F for more details
				//- Only one biomemap should be provided if heightmap generation is enabled
				std::vector<STPDiversity::Sample*> Biomemap;
				//- A float array that will be used to stored heightmap pixles, must be pre-allocated with at least width * height * sizeof(float), i.e., R32F format
				//- If generator is instructed to generate only a single heightmap, only one map is required
				//- If hydraulic erosion and/or normalmap generation is enabled, a list of maps of neighbour chunks are required for edge sync, heightmap generation will 
				//only affect the central chunk, for neighbour chunks it must be precomputed with heightmap to be able to perform free-slip hydraulic erosion,
				//If free-slip hydraulic erosion is disabled, no neighbour chunks are required.
				//- The map pointers should be arranged in row major matrix, with defined neighbour dimension.
				std::vector<float*> Heightmap32F;
				//The x vector specify the offset on x direction of the map and and z on y direction of the map.
				//The offset parameter will only be applied on the heightmap generation.
				float2 HeightmapOffset = make_float2(0.0f, 0.0f);
				//A INT16 array that will be used to stored the heightmap and normalmap after formation. The final format will become RGBA.
				//The number of pointer provided should be the same as the number of heightmap and normalmap.
				std::vector<unsigned short*> Heightfield16UI;

			};

		private:

			/**
			 * @brief STPEdgeArrangement specifies the edge arrangement type when performing edge copy operations
			*/
			enum class STPEdgeArrangement : unsigned char;

			/**
			 * @brief Memory allocation for pinned memory
			*/
			class STPHeightfieldHostAllocator {
			public:

				/**
				 * @brief Allocate page-locked memory on host
				 * @param count The number of byte to allocate
				 * @return The memory pointer
				*/
				__host__ void* allocate(size_t);

				/**
				 * @brief Free up the host pinned memory
				 * @param count The size to free
				 * @param The host pinned pointer to free
				*/
				__host__ void deallocate(size_t, void*);

			};

			/**
			 * @brief CUDA nonblocking stream allocator
			*/
			class STPHeightfieldNonblockingStreamAllocator {
			public:

				/**
				 * @brief Allocate nonblocking stream
				 * @param count Useless argument, it will only allocate one stream at a time
				 * @return The pointer to stream
				*/
				__host__ void* allocate(size_t);

				/**
				 * @brief Destroy the stream
				 * @param count Useless argument, it will only destroy one stream
				 * @param The stream to destroy
				*/
				__host__ void deallocate(size_t, void*);
			};

			/**
			 * @brief A custom deleter for device memory
			 * @tparam T Type of the variable
			*/
			template<typename T>
			struct STPDeviceDeleter {
			public:

				void operator()(T*) const;

			};
			//An alias of unique_ptr with cudaFree as deleter
			template<typename T>
			using unique_ptr_d = std::unique_ptr<T, STPDeviceDeleter<T>>;

			//multi-biome heightmap generator linked with external
			const STPDiversityGenerator& generateHeightmap;
			//heightfield generation parameters
			const STPSettings::STPHeightfieldSettings& Heightfield_Settings_h;
			unique_ptr_d<STPSettings::STPHeightfieldSettings> Heightfield_Settings_d;

			//curand random number generator for erosion, each generator will be dedicated for one thread, i.e., thread independency
			unique_ptr_d<curandRNG> RNG_Map;
			/**
			 * @brief Convert global index to local index, making data access outside the central chunk available
			 * As shown the difference between local and global index
			 *		 Local					 Global
			 * 0 1 2 3 | 0 1 2 3	0  1  2  3  | 4  5  6  7
			 * 4 5 6 7 | 4 5 6 7	8  9  10 11 | 12 13 14 15
			 * -----------------	-------------------------
			 * 0 1 2 3 | 0 1 2 3	16 17 18 19 | 20 21 22 23
			 * 4 5 6 7 | 4 5 6 7	24 25 26 27 | 28 29 30 21
			 *
			 * Given that chunk should be arranged in a linear array (just an example)
			 * Chunk 1 | Chunk 2
			 * -----------------
			 * Chunk 3 | Chunk 4
			*/
			//Btw don't use `unsigned int[]` here since we are dealing with device memory
			unique_ptr_d<unsigned int> GlobalLocalIndex;
			//A lookup table that, given a chunkID, determine the edge type of this chunk within the neighbour chunk logic
			std::unique_ptr<STPEdgeArrangement[]> EdgeArrangementTable;

			//The size of the map generated
			const uint2 MapSize;
			//Free slip range in the unit of chunk
			const uint2 FreeSlipChunk;

			//Temp cache on device for heightmap computation
			mutable std::mutex MapCachePinned_lock;
			mutable std::mutex StreamPool_lock;
			mutable cudaMemPool_t MapCacheDevice;
			mutable STPMemoryPool<void, STPHeightfieldHostAllocator> MapCachePinned;
			mutable STPMemoryPool<void, STPHeightfieldNonblockingStreamAllocator> StreamPool;

			/**
			 * @brief Set the number of raindrop to spawn for each hydraulic erosion run, each time the function is called some recalculation needs to be re-done.
			 * Determine the number of raindrop to summon, the higher the more accurate but slower
			*/
			__host__ void setErosionIterationCUDA();

			/**
			 * @brief Initialise the local global index lookup table
			*/
			__host__ void initLocalGlobalIndexCUDA();

			/**
			 * @brief Initialise edge arrangement lookup table
			*/
			__host__ void initEdgeArrangementTable();

			/**
			 * @brief Copy the border of the texture using neighbour logic. Used mainly for rendering buffer edge synchronisation
			 * @param device - Device destination memory
			 * @param souce - The source chunks
			 * @param element_count - The number of pixel in one chunk
			 * @param stream - Async CUDA stream
			*/
			__host__ void copyNeighbourEdgeOnly(unsigned short*, const std::vector<unsigned short*>&, size_t, cudaStream_t) const;

		public:

			/**
			 * @brief Init the heightfield generator
			 * @param chunk_settings All parameters for the chunk to be linked with this generator
			 * @param heightfield_settings All parameters for heightfield generation to be linked with this generator
			 * @param diversity_generator A generator responsible for generating a multi-biome heightmap
			 * @param hint_level_of_concurrency The average numebr of thread that will be used to issue commands to this class.
			 * It's used to assume the size of memory pool to allocate.
			*/
			__host__ STPHeightfieldGenerator(const STPSettings::STPChunkSettings&, const STPSettings::STPHeightfieldSettings&, 
				const STPDiversityGenerator&, unsigned int);

			__host__ ~STPHeightfieldGenerator();

			__host__ STPHeightfieldGenerator(const STPHeightfieldGenerator&) = delete;

			__host__ STPHeightfieldGenerator(STPHeightfieldGenerator&&) = delete;

			__host__ STPHeightfieldGenerator& operator=(const STPHeightfieldGenerator&) = delete;

			__host__ STPHeightfieldGenerator& operator=(STPHeightfieldGenerator&&) = delete;

			/**
			 * @brief Generate the terrain heightfield maps, each heightfield contains four maps, being heightmap and normalmap.
			 * All storage spaces must be preallocated with width * height * sizeof(float), with the exception of normalmap, which requires width * height * sizeof(float) * 4.
			 * The function will first generate our epic height map diversity generator, using the parameter provided during class init.
			 * The generated heightmap will be in range [0,1]
			 * Then performing hydraulic erosion algorithm to erode the rough terrain into a more natrual form.
			 * The number of iteration must be set via setErosionIterationCUDA() so pre-computation can be done before launching the program.
			 * Lastly it will generate the normal map for the height map, the normalmap is normalised within the range [0,1].
			 * All four maps are kept in floating point pixel format.
			 * @param args The generator data, see STPMapStorage documentation for more details
			 * @param operation Control what type of operation generator does
			 * @return True if all operations are successful without any errors
			*/
			__host__ void operator()(STPMapStorage&, STPGeneratorOperation) const;

		};

	}
}
#endif//_STP_HEIGHTFIELD_GENERATOR_CUH_