#pragma once
#ifndef _STP_SCREEN_SPACE_RAY_INTERSECTION_CUH_
#define _STP_SCREEN_SPACE_RAY_INTERSECTION_CUH_

#ifndef __CUDACC_RTC__
//CUDA
#include <cuda_runtime.h>
#endif
//OptiX
#include <optix_types.h>

//Maths
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
			EnvironmentRayID = 0u;

		//Used for depth reconstruction from NDC to world space.
		float4x4 InvProjectionView;

		/* ---------------------------- Input/Output ------------------------ */
		//All screen space texture should have the same dimension as the rendering resolution.
		//A stencil buffer to indicate which pixel should have ray launched from.
		//This stencil buffer also outputs result flags for ray intersection test,
		//for example which object does the ray hits, or if ray missed everything.
		cudaSurfaceObject_t SSStencil;//R8UI
		/* -------------------------------- Input ----------------------------*/
		//Stores the screen space depth from the rendering scene,
		//the specific ray origin can be recovered by performing depth reconstruction.
		cudaTextureObject_t SSRayDepth;//R32F
		//Stores screen space ray direction, which specifies a 3-component unit vector towards which ray is launched.
		//The last component is for padding and unused.
		cudaTextureObject_t SSRayDirection;//RGBA16, range converted from [0, 1] to [-1, 1] after texture fetch.
		/* ------------------------------- Output --------------------------- */
		//Please note that all outputs are undefined if the ray misses the geometry, as indicated by stencil buffer.
		//Record the 3-component vector of position of intersection in world space.
		//The last component is for padding hence unused.
		cudaSurfaceObject_t GPosition;//RGBA32F
		//The 2-component vector of normalised texture coordinate of the intersecting geometry.
		cudaSurfaceObject_t GTextureCoordinate;//RG16

		OptixTraversableHandle Handle;

		//ray generation
		struct STPLaunchedRayData {
		public:

		};

		//closest hit
		struct STPPrimitiveHitData {
		public:

			//stride of vertex buffer
			constexpr static unsigned int AttributeStride = 5u;

			//Primitive geometry data.
			//Each primitive is located by their instance ID.
			//Index and vertex both use the same index.
			const uint3** PrimitiveIndex;
			//For vertex, it is required to have the following attribute structure, and tightly packed:
			//vec3: position, vec2 UV.
			const float** PrimitiveVertex;

		};

		//miss
		struct STPEnvironmentHitData {
		public:

		};

	};
}
#endif//_STP_SCREEN_SPACE_RAY_INTERSECTION_CUH_