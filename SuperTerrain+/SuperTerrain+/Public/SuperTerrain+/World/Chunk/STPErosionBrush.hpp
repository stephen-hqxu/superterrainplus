#pragma once
#ifndef _STP_EROSION_BRUSH_HPP_
#define _STP_EROSION_BRUSH_HPP_

namespace SuperTerrainPlus {

	/**
	 * @brief STPErosionBrush stores data for the generated erosion brush.
	*/
	struct STPErosionBrush {
	public:

		//Generated by the erosion brush generator after setting the erosion brush radius
		//Precomputed erosion brush indices, must be made available so it can be used on device.
		const int* Index;
		//Precomputed erosion brush weights, must be made available so it can be used on device.
		const float* Weight;
		//The number of element in the erosion brush.
		//The number of index and weight is the same.
		unsigned int BrushSize;

	};
}
#endif//_STP_EROSION_BRUSH_HPP_