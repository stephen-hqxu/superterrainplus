#ifndef _STP_MATERIAL_REGISTRY_GLSL_
#define _STP_MATERIAL_REGISTRY_GLSL_

/**
 * @brief STPMaterialProperty defines data of a material.
*/
struct STPMaterialProperty {

	//Defines how reflective an object it.
	//This is used to control the behaviour of Fresnel effect and makes object reflects more than refracts.
	float Reflexivity;
	//Specifies how much light should be allowed to passed through the object.
	float Opacity;
	//Index of refraction.
	float RefractiveIndex;

};

#ifndef __cplusplus
//This buffer contains all currently registered material
layout(std430, binding = 2) readonly restrict buffer STPMaterialRegistry {
	//given a material index, find the material data
	layout(offset = 0) STPMaterialProperty Material[];
};
#endif

#endif//_STP_MATERIAL_REGISTRY_GLSL_