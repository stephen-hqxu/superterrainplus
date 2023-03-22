#pragma once
#ifndef _STP_FRAGMENT_UTILITY_CUH_
#define _STP_FRAGMENT_UTILITY_CUH_

//Helpers
#include "../Common/STPCameraInformation.glsl"
//Maths
#include "STPVectorUtility.cuh"

namespace SuperTerrainPlus::STPRealism {
	/**
	 * @brief STPFragmentUtility is a simple helper for fragment-related algebra calculations.
	*/
	namespace STPFragmentUtility {
	
		/**
		 * @brief Calculate the fragment texture coordinate.
		 * @param frag_index The index of the fragment to be calculated.
		 * Must be within the range [0u, dim - 1].
		 * @param dim The dimension of the texture.
		 * @return The texture coordinate for the given fragment index, within range [0.0f, 1.0f].
		*/
		__device__ __forceinline__ static float2 calcTextureCoordinate(const uint2 frag_index, const uint2 dim) {
			return make_float2(frag_index) / make_float2(dim);
		}

		/**
		 * @brief Perform depth reconstruction which converts a coordinate system from normalised device coordinate to world space.
		 * This is mainly used on OpenGL coordinate system.
		 * @param inv_pv The inverse projection view matrix.
		 * @param frag_depth The fragment depth value to be converted.
		 * @param frag_uv The fragment texture coordinate where the pixel is located.
		 * @return The reconstructed world position of the fragment.
		 * @see STPCameraInformation.glsl
		*/
		__device__ __forceinline__ static float3 reconstructDepthToWorld(const float4x4& inv_pv, const float frag_depth, const float2 frag_uv) {
			//compatibility type conversion
			float3 (*vec3)(const float2&, const float) = &make_float3;
			float4 (*vec4)(const float3&, const float) = &make_float4;

			const float4 position_world = inv_pv * STP_DEPTH_BUFFER_TO_NDC(frag_uv, frag_depth);
			return make_float3(position_world) / position_world.w;
		}

//normalised barycentric coordinate sums to one
#define BARYCENTRIC_INTERP_DEF (1.0f - bary.x - bary.y) * v1 + bary.x * v2 + bary.y * v3

		/**
		 * @brief Perform barycentric coordinate interpolation for a triangle vertex data.
		 * @param bary The triangle barycentric coordinate.
		 * @param v1 Vertex one.
		 * @param v2 Vertex two.
		 * @param v3 Vertex three.
		 * @return The interpolation vertex data.
		*/
		__device__ __forceinline__ static float2 barycentricInterpolation(const float2 bary, const float2 v1, const float2 v2, const float2 v3) {
			return BARYCENTRIC_INTERP_DEF;
		}
		//@see barycentricInterpolation(float2...)
		__device__ __forceinline__ static float3 barycentricInterpolation(const float2 bary, const float3 v1, const float3 v2, const float3 v3) {
			return BARYCENTRIC_INTERP_DEF;
		}

#undef BARYCENTRIC_INTERP_DEF
	
	}
}
#endif//_STP_FRAGMENT_UTILITY_CUH_