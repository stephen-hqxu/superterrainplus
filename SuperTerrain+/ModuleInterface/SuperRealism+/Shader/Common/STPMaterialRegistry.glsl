#ifndef _STP_MATERIAL_REGISTRY_GLSL_
#define _STP_MATERIAL_REGISTRY_GLSL_

struct STPMaterialProperty {

	//A colour spectrum of intrinsic colour of the material, given the distance of ray travelled within the material.
	//This is a bindless handle to a sampler1D.
	uvec2 IntrinsicSpectrum;
	//Specifies light attenuation inside the material.
	//The opacity of the material gradually decreases from the start to the end point as defined.
	//They are specified in term of distance of ray travelled within the material.
	float AttenuationStart, AttenuationEnd;
	//Index of refraction.
	float RefractiveIndex;
	//The Fresnel reflection coefficient.
	float Reflectance;

};

//This buffer contains all currently registered material
layout(std430, binding = 2) readonly restrict buffer STPMaterialRegistry {
	//given a material index, find the material data
	layout(offset = 0) STPMaterialProperty Material[];
};
#endif//_STP_MATERIAL_REGISTRY_GLSL_