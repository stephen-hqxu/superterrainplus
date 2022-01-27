#ifndef _STP_LIGHT_SPACE_INFORMATION_GLSL_
#define _STP_LIGHT_SPACE_INFORMATION_GLSL_

//The number of light space matrix.
//Define a valid number for a fixed compile time constant
#ifndef LIGHT_SPACE_COUNT
#define LIGHT_SPACE_COUNT The number of light space matrix is unspecified
#endif

layout(std430, binding = 1) readonly restrict buffer STPLightSpaceInformation {
	//convert from world space to light clip space
	layout(offset = 0) mat4 LightProjectionView[LIGHT_SPACE_COUNT];
};

#endif//_STP_LIGHT_SPACE_INFORMATION_GLSL_