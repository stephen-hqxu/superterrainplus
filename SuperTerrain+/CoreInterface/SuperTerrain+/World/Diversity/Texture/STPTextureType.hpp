#pragma once
#ifndef _STP_TEXTURE_TYPE_HPP_
#define _STP_TEXTURE_TYPE_HPP_

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
		 * @brief STPTextureType defines the type of texture the texture group holds.
		 * Each enum item represents a texture type that is ordered continuously and can be used as indices.
		*/
		enum class STPTextureType : unsigned char {
			//A texture that defines the base color of the mesh being textured
			Albedo = 0x00u,
			//A texture that defines the normal vector which is then used to calculate light reflection and refraction on the surface of the mesh
			Normal = 0x01u,
			//A texture that defines the perpendicular offset to the surface of the mesh at a pixel
			Displacement = 0x02u,
			//A texture that defines the amount of specular highlight at a pixel
			Specular = 0x03u,
			//A texture that defines how much light is scattered across the surface of the mesh
			Glossiness = 0x04u,
			//A texture that controls how much color from the albedo map contributes to the diffuse and brightness
			Metalness = 0x05u,
			//A texture that defines how a texture reacts to light during rendering
			AmbientOcclusion = 0x06u,
			//A texture that defines which part of the object emits light, as well as the light color of each pixel
			Emissive = 0x07u,

			//The total number of type listed in the enum
			TYPE_COUNT = 0x08u
		};

	}
}
#endif//_STP_TEXTURE_TYPE_HPP_