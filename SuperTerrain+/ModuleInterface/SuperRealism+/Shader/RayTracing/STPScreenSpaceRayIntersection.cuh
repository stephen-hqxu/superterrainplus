#pragma once
#ifndef _STP_SCREEN_SPACE_RAY_INTERSECTION_CUH_
#define _STP_SCREEN_SPACE_RAY_INTERSECTION_CUH_

#ifndef __CUDACC_RTC__
//CUDA
#include <texture_types.h>
#include <surface_types.h>
#endif
//OptiX
#include <optix_types.h>

#include "STPGeometryAttributeFormat.hpp"
#include "STPVectorUtility.cuh"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPScreenSpaceRayIntersectionData provides data for a simple ray-primitive intersection test shader,
	 * starting a ray from a screen-space origin.
	*/
	struct STPScreenSpaceRayIntersectionData {
	public:

		//Stencil identifiers for rays.
		//All the rest of bits and values can be used as primitive identifier.
		//The resulting stencil value should be a logical or between mask and ID.
		static constexpr unsigned char RayVisibilityMask = 1u << 7u,
			EnvironmentRayID = RayVisibilityMask - 1u;

		//Used for depth reconstruction from NDC to world space.
		float4x4 InvProjectionView;

		//texture and surface objects used during SSRI
		struct STPTextureData {
		public:

			/* ---------------------------- Input/Output ------------------------ */
			//All screen space texture should have the same dimension as the rendering resolution.
			unsigned char* SSStencil;
			/* -------------------------------- Input ----------------------------*/
			cudaTextureObject_t SSRayDepth;
			//range converted from [0, 1] to [-1, 1] after texture fetch.
			cudaTextureObject_t SSRayDirection;
			/* ------------------------------- Output --------------------------- */
			//Please note that all outputs are undefined if the ray misses the geometry, as indicated by stencil buffer.
			cudaSurfaceObject_t GPosition;
			cudaSurfaceObject_t GTextureCoordinate;

		} SSTexture;

		OptixTraversableHandle Handle;

		//closest hit
		struct STPPrimitiveHitData {
		public:

			//For vertex, it is required to have the following attribute structure, and tightly packed.
			const STPGeometryAttributeFormat::STPVertexFormat* const* const* PrimitiveVertex;
			//Primitive geometry data.
			//Each object is assigned with an object ID, which can be used as an index to the instance array.
			//Each instance contains a number of primitives, each is located by their instance ID.
			//Index and vertex both use the same index.
			const STPGeometryAttributeFormat::STPIndexFormat* const* const* PrimitiveIndex;

		};

	};
}
#endif//_STP_SCREEN_SPACE_RAY_INTERSECTION_CUH_