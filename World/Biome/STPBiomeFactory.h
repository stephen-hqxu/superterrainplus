#pragma once
#ifndef _STP_BIOME_FACTORY_H_
#define _STP_BIOME_FACTORY_H_

//GLM
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
//Biome
#include "STPBiome.h"
#include "STPLayer.h"

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {
	/**
	 * @brief STPBiome is a series of biome generation algorithm that allows user to define their own implementations
	*/
	namespace STPBiome {

		/**
		 * @brief STPBiomeFactory provides definition for biome layers and biome lookup table
		*/
		class STPBiomeFactory {
		public:

			//Specify the dimension of the generated biome map, in 3 dimension
			const glm::uvec3 BiomeDimension;

			/**
			 * @brief Init the biome factory
			 * @param dimension The dimension of the biome map.
			 * If the y component of the dimension is one, a 2D biome map will be generated
			*/
			STPBiomeFactory(glm::uvec3);

			/**
			 * @brief Init the biome factory
			 * @param dimension The dimension of the biome map, this will init a 2D biome map generator, with x and z component only
			*/
			STPBiomeFactory(glm::uvec2);

			~STPBiomeFactory();

			/**
			 * @brief Build the biome layer chain
			 * @return The pointer to the last layer in the layer chain.
			*/
			virtual STPLayer* manufacture() = 0;

			/**
			 * @brief Lookup the biome by its biome id
			 * @param id The id of this biome
			 * @return The biome with that id
			*/
			virtual const STPBiome& interpret(Sample) = 0;

			/**
			 * @brief Generate a biome map using the biome chain implementation
			 * @param chain The biome chain implementation, it should be the returned value from "manufacture" 
			 * @param offset The offset of the biome map, that is equavalent to the world coordinate.
			 * @return The biome id map, it needs to be freed maunally
			*/
			const Sample* generate(STPLayer* const, glm::ivec3);

			/**
			 * @brief Free up the storage of a biome map generated
			 * @param map The biome map to free
			*/
			void dump(Sample*);

		};

	}
}
#endif//_STP_BIOME_FACTORY_H_