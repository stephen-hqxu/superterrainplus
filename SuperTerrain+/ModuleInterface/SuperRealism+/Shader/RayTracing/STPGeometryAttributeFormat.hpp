#pragma once
#ifndef _STP_GEOMETRY_ATTRIBUTE_FORMAT_HPP_
#define _STP_GEOMETRY_ATTRIBUTE_FORMAT_HPP_

#ifndef __CUDACC_RTC__
#include <vector_types.h>
#endif

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPGeometryAttributeFormat defines how geometry data attribute used by ray tracer should be formatted.
	*/
	namespace STPGeometryAttributeFormat {

		/**
		 * @brief The vertex format for the geometry.
		 * Currently only fixed vertex format is supported and they should be laid out in the same manner as declared and tightly packed.
		*/
		struct STPVertexFormat {
		public:

			//required to be in range of [0.0f, 1.0f]
			float2 UV;

		};

		/**
		 * @brief The index format for the geometry.
		*/
		using STPIndexFormat = uint3;

		static_assert(
			sizeof(STPVertexFormat) == sizeof(float) * 2 && sizeof(STPIndexFormat) == sizeof(unsigned int) * 3,
			"A strange platform encountered which adds spurious padding in geometry attribute format structures.");
	}

}
#endif//_STP_GEOMETRY_ATTRIBUTE_FORMAT_HPP_