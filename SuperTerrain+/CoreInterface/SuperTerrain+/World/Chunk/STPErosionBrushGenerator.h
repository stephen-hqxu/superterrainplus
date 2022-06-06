#pragma once
#ifndef _STP_EROSION_BRUSH_GENERATOR_H_
#define _STP_EROSION_BRUSH_GENERATOR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Brush Information
#include "STPErosionBrush.hpp"
//Memory
#include "../../Utility/Memory/STPSmartDeviceMemory.h"

namespace SuperTerrainPlus {

	/**
	 * @brief STPErosionBrushGenerator generates erosion brush table, manages the memory
	 * and export the brush to the rain drop setting for device to use.
	*/
	class STP_API STPErosionBrushGenerator {
	private:

		//After generation, erosion brushes are copied directly to the device memory.
		STPSmartDeviceMemory::STPDeviceMemory<int[]> ErosionBrushIndex_d;
		STPSmartDeviceMemory::STPDeviceMemory<float[]> ErosionBrushWeight_d;

		//The underlying pointer is shared with the GPU kernel.
		STPErosionBrush Brush;

	public:

		/**
		 * @brief Init STPErosionBrushGenerator and generates the erosion brush data.
		 * @param freeslip_rangeX The number of element on the free-slip heightmap in the free-slip chunk range in X direction,
		 * i.e., the X dimension of the free-slip map.
		 * @param erosion_radius The radius of erosion.
		*/
		STPErosionBrushGenerator(unsigned int, unsigned int);

		STPErosionBrushGenerator(const STPErosionBrushGenerator&) = delete;

		STPErosionBrushGenerator(STPErosionBrushGenerator&&) = delete;

		STPErosionBrushGenerator& operator=(const STPErosionBrushGenerator&) = delete;

		STPErosionBrushGenerator& operator=(STPErosionBrushGenerator&&) = delete;

		~STPErosionBrushGenerator() = default;

		/**
		 * @brief Read the underlying pointers to the erosion brush.
		 * @return The erosion brush in the device memory region.
		 * The memory within the brush is managed by the current instance of erosion brush generator.
		*/
		const STPErosionBrush& getBrush() const;

	};

}
#endif//_STP_EROSION_BRUSH_GENERATOR_H_