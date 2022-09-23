#pragma once
#ifndef _STP_TEXTURE_TYPE_HPP_
#define _STP_TEXTURE_TYPE_HPP_

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPTextureType defines the type of texture the texture group holds.
	 * Each enum item represents a texture type that is ordered continuously and can be used as indices.
	*/
	enum class STPTextureType : unsigned char {
		//A texture that defines the base colour of the mesh being textured
		Albedo = 0x00u,
		//A texture that defines the normal vector which is then used to calculate light reflection and refraction on the surface of the mesh
		Normal = 0x01u,
		//A texture that defines how spread out a specular highlight is from the projection centre.
		Roughness = 0x02u,
		//A texture that defines how a texture reacts to light during rendering
		AmbientOcclusion = 0x3u,

		//The total number of type listed in the enum
		TypeCount
	};

}
#endif//_STP_TEXTURE_TYPE_HPP_