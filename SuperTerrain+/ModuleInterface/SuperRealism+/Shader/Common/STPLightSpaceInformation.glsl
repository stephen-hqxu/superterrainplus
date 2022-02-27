#ifndef _STP_LIGHT_SPACE_INFORMATION_GLSL_
#define _STP_LIGHT_SPACE_INFORMATION_GLSL_

//This buffer requires the ability to use pointer in the shader.
layout(std430, binding = 1) readonly restrict buffer STPLightSpaceInformation {
	//This pointer specifies a light matrix for the light that is currently being rendered to shadow map.
	//The number of matrix element to be accessed should be acknowledged by the program already.
	//Convert from world space to light clip space
	layout(offset = 0) readonly mat4* restrict ProjectionView;
} LightSpace;

#endif//_STP_LIGHT_SPACE_INFORMATION_GLSL_