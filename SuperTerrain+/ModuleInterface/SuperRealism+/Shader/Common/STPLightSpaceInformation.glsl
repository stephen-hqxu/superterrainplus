#ifndef _STP_LIGHT_SPACE_INFORMATION_GLSL_
#define _STP_LIGHT_SPACE_INFORMATION_GLSL_

layout(std430, binding = 1) readonly restrict buffer STPLightSpaceInformation {
	//this value is used to locate the matrix for ONE light and will be updated as per-light basis when doing shadow pass.
	//this is used as the index to the light space matrix.
	//the number of matrix element to be accessed should be acknowledged by the program already.
	layout(offset = 0) unsigned int CurrentLightSpaceStart;
	//convert from world space to light clip space
	layout(offset = 16) mat4 ProjectionView[];
} LightSpace;

#endif//_STP_LIGHT_SPACE_INFORMATION_GLSL_