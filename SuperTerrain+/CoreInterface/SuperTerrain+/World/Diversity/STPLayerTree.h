#pragma once
#ifndef _STP_LAYER_TREE_H_
#define _STP_LAYER_TREE_H_

//Data Structure
#include <vector>
//Layer Node
#include "STPLayer.h"

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPLayerTree is a tree structured class that manages all STPLayers as nodes.
	 * It makes layer creation and destroy easier.
	*/
	class STPLayerTree {
	private:

		//An array pointers to every layer.
		//std::vector default deallocator deletes each layer in reverse direction,
		//so it is safe for latter layer taking a pointer to the previous layer, which is what our data structure does
		//STPLayerTree owns the pointer to each layer so vertices can be deleted with ease
		std::vector<std::unique_ptr<STPLayer>> Vertex;

	public:

		/**
		 * @brief Init STPLayerTree
		*/
		STPLayerTree() = default;

		STPLayerTree(const STPLayerTree&) = delete;

		STPLayerTree(STPLayerTree&&) noexcept = default;

		STPLayerTree& operator=(const STPLayerTree&) = delete;

		STPLayerTree& operator=(STPLayerTree&&) noexcept = default;

		~STPLayerTree() = default;

		/**
		 * @brief Construct a new layer instance and add to the layer tree structure and let the current layer tree manages this layer.
		 * @tparam L A layer instance
		 * @tparam Arg... A list of arguments for the child layer class
		 * @param cache_size The cache size for this layer.
		 * @param args... All other arguments for the created layer to be used in their constructor.
		 * @return A pointer new layer instance with the type of the specified child layer. The pointer is owned by the current tree and will be freed automatically.
		*/
		template <class L, class... Arg>
		STPLayer* insert(size_t, Arg&&...);

		/**
		 * @brief Get the pointer to layer where the layer structure start.
		 * It's the last layer being added to the graph.
		 * During biome generation, this is the first layer to be called, and ascendant layers will be called from this layer recursively.
		 * @return The pointer to the starting layer.
		*/
		STPLayer* start() const noexcept;

		/**
		 * @brief Get the number of layer, i.e., the number of vertices presented in this graph, add to the current tree.
		 * @return The number of layer
		*/
		size_t getLayerCount() const noexcept;

	};

}
#include "STPLayerTree.inl"
#endif//_STP_LAYER_TREE_H_