#pragma once
#ifndef _STP_LAYER_MANAGER_H_
#define _STP_LAYER_MANAGER_H_

//Data Structure
#include <vector>
//Layer Node
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
		 * @brief STPLayerManager is a graph structured class that manages all STPLayers as nodes.
		 * It makes layer creation and destroy easier.
		*/
		class STPLayerManager {
		private:

			//An array pointers to every layer.
			//STPLayerManager owns the pointer to each layer so vertices can be deleted with ease
			std::vector<std::unique_ptr<STPLayer>> Vertex;

		public:

			/**
			 * @brief Init STPLayerManager
			*/
			STPLayerManager() = default;

			~STPLayerManager() = default;

			/**
			 * @brief Create a layer instance
			 * @tparam L A layer instance
			 * @tparam C Cache size for this layer, it should be in the power of 2
			 * @tparam Arg A list of arguments for the child layer class
			 * @param global_seed The global seed is the seed that used to generate the entire world, a.k.a., world seed.
			 * @param salt The salt is a random number that used to mix the global to generate local and layer seed, such that each layer should use
			 * different salt value
			 * @param args All other arguments for the created layer to be used in their constructor.
			 * @return A pointer new layer instance with the type of the specified child layer. The pointer needs to be freed with destroy() function
			*/
			template <class L, size_t C = 0ull, class... Arg>
			STPLayer* create(Seed, Seed, Arg&&...);

			/**
			 * @brief Get the pointer to layer where the layer structure start.
			 * It's the last layer being added to the graph.
			 * During biome generation, this is the first layer to be called, and ascendant layers will be called from this layer recursively.
			 * @return The pointer to the starting layer.
			*/
			STPLayer* start();

			/**
			 * @brief Get the number of layer, i.e., the number of vertices presented in thsi graph, managed by manager.
			 * @return The number of layer
			*/
			size_t getLayerCount() const;

		};

	}
}
#include "STPLayerManager.inl"
#endif//_STP_LAYER_MANAGER_H_