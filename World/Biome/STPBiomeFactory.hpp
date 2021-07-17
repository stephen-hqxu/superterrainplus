#pragma once
#ifndef _STP_BIOME_FACTORY_HPP_
#define _STP_BIOME_FACTORY_HPP_

//System
#include <queue>
#include <mutex>
//GLM
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
//Biome
#include "STPLayerManager.h"

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
		 * @brief STPBiomeFactory provides a safe environment for multi-threaded biome map generation.
		 * @tpara A layer supplier, which provides the algorithm for layer chain generation.
		*/
		template<class S>
		class STPBiomeFactory {
		private:

			const S Supplier;

			typedef std::unique_ptr<STPLayerManager> STPLayerManager_t;
			//Basically it behaves like a memory pool.
			//Whenever operator() is called, we search for an empty production line, and use that to generate biome.
			//If no available production line can be found, ask more production line from the manufacturer.
			std::queue<STPLayerManager_t, std::list<STPLayerManager_t>> LayerProductionLine;
			mutable std::mutex ProductionLock;

			/**
			 * @brief Request a production line.
			 * If no available production line is presented, a new production line is asked from layer supplier, and returned.
			 * @return Requested production line.
			*/
			STPLayerManager_t requestProductionLine() {
				{
					std::unique_lock lock(this->ProductionLock);
					STPLayerManager_t line;

					if (this->LayerProductionLine.empty()) {
						//no more idling line? Create a new one
						line = STPLayerManager_t(this->Supplier());
						return line;
					}
					//otherwise simply pop from the idling queue
					line = move(this->LayerProductionLine.front());
					this->LayerProductionLine.pop();
					return line;
				}
			}

			/**
			 * @brief Return the production back to idling queue.
			 * Any attemp to use the production line after returned will result in undefined behaviour
			 * @param line Production line to be returned
			*/
			void returnProductionLine(STPLayerManager_t& line) {
				{
					std::unique_lock lock(this->ProductionLock);
					//simply put it back
					this->LayerProductionLine.emplace(move(line));
				}
			}

		public:

			//Specify the dimension of the generated biome map, in 3 dimension
			const glm::uvec3 BiomeDimension;

			/**
			 * @brief Init biome factory with internal cache memory pool that can be used for multi-threading, each thread will be automatically allocaed one cache
			 * @param dimension The dimension of the biome map
			 * If the y component of the dimension is one, a 2D biome map will be generated
			 * @param supplier The biome layer chain generator function
			*/
			template<typename... Arg>
			STPBiomeFactory(glm::uvec3 dimension, Arg&&... arg) : BiomeDimension(dimension), Supplier(std::forward<Arg>(arg)...) {

			}

			/**
			 * @brief Init biome factory with internal cache memory pool that can be used for multi-threading, each thread will be automatically allocaed one cache
			 * @param dimension The dimension of the biome map, this will init a 2D biome map generator, with x and z component only
			 * @param supplier The biome layer chain generator function
			*/
			template<typename... Arg>
			STPBiomeFactory(glm::uvec2 dimension, Arg&&... arg) : STPBiomeFactory(glm::uvec3(dimension.x, 1u, dimension.y), std::forward<Arg>(arg)...) {

			}

			//make sure the thread stopped before deleting factory
			//stop all waiting workers and waiting for current worker to finish.
			~STPBiomeFactory() = default;

			/**
			 * @brief Generate a biome map using the biome chain implementation
			 * @param biomemap The output where biome map will be stored, must be preallocated with enough space
			 * @param offset The offset of the biome map, that is equavalent to the world coordinate.
			*/
			void operator()(Sample* biomemap, glm::ivec3 offset) {
				//request a production line
				STPLayerManager_t producer = this->requestProductionLine();

				//loop through and generate the biome map
				//why not using CUDA and do it in parallel? Because the biome layers are cached, tested and parallel performance is a piece of shit
				if (this->BiomeDimension.y == 1u) {
					//it's a 2D biome
					//to avoid making useless computation
					for (unsigned int x = 0u; x < this->BiomeDimension.x; x++) {
						for (unsigned int z = 0u; z < this->BiomeDimension.z; z++) {
							//calculate the map index
							const unsigned int index = x + z * this->BiomeDimension.x;
							//get the biome at thie coordinate
							biomemap[index] = producer->start()->sample(static_cast<int>(x) + offset.x, 0, static_cast<int>(z) + offset.z);
						}
					}
					return;
				}
				//it's a 3D biome
				for (unsigned int x = 0u; x < this->BiomeDimension.x; x++) {
					for (unsigned int y = 0u; y < this->BiomeDimension.y; y++) {
						for (unsigned int z = 0u; z < this->BiomeDimension.z; z++) {
							//calculate the map index
							const unsigned int index = x + y * this->BiomeDimension.x + z * (this->BiomeDimension.x * this->BiomeDimension.y);
							//get the biome at thie coordinate
							biomemap[index] = producer->start()->sample(static_cast<int>(x) + offset.x, static_cast<int>(y) + offset.y, static_cast<int>(z) + offset.z);
						}
					}
				}

				//free the producer
				this->returnProductionLine(producer);
			}

		};

	}
}
#endif//_STP_BIOME_FACTORY_HPP_