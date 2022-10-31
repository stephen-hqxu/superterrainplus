#pragma once
#ifndef _STP_LAYER_MANAGER_H_
#define _STP_LAYER_MANAGER_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Data Structure
#include <vector>
//Layer Node
#include "STPLayer.h"

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPLayerManager is a graph structured class that manages all STPLayers as nodes.
	 * It makes layer creation and destroy easier.
	*/
	class STP_API STPLayerManager {
	private:

		//An array pointers to every layer.
		//STPLayerManager owns the pointer to each layer so vertices can be deleted with ease
		std::vector<std::unique_ptr<STPLayer, void(*)(STPLayer*)>> Vertex;

		/**
		 * @brief Delete a internally stored layer
		 * Since STPLayerManager is friend of STPLayer and STPLayerRecycler is friend of STPLayerManager, such that STPLayerRecycler can access deleter of STPLayer.
		 * @param layer The pointer to the layer to be deleted
		*/
		static void recycleLayer(STPLayer*);

	public:

		/**
		 * @brief Init STPLayerManager
		*/
		STPLayerManager() = default;

		STPLayerManager(const STPLayerManager&) = delete;

		STPLayerManager(STPLayerManager&&) = default;

		STPLayerManager& operator=(const STPLayerManager&) = delete;

		STPLayerManager& operator=(STPLayerManager&&) = default;

		~STPLayerManager() = default;

		/**
		 * @brief Construct a new layer instance and add to the layer chain structure and let the current layer manager manage this layer.
		 * @tparam L A layer instance
		 * @tparam C Cache size for this layer, it should be in the power of 2
		 * @tparam Arg A list of arguments for the child layer class
		 * @param args All other arguments for the created layer to be used in their constructor.
		 * @return A pointer new layer instance with the type of the specified child layer. The pointer is owned by the current manager and will be freed automatically.
		*/
		template <class L, size_t C = 0u, class... Arg>
		STPLayer* insert(Arg&&...);

		/**
		 * @brief Get the pointer to layer where the layer structure start.
		 * It's the last layer being added to the graph.
		 * During biome generation, this is the first layer to be called, and ascendant layers will be called from this layer recursively.
		 * @return The pointer to the starting layer.
		*/
		STPLayer* start();

		/**
		 * @brief Get the number of layer, i.e., the number of vertices presented in this graph, managed by manager.
		 * @return The number of layer
		*/
		size_t getLayerCount() const;

	};

}
#include "STPLayerManager.inl"
#endif//_STP_LAYER_MANAGER_H_