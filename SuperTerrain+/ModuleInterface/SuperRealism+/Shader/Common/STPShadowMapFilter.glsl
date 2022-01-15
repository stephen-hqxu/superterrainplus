#ifndef _STP_SHADOW_MAP_FILTER_GLSL_
#define _STP_SHADOW_MAP_FILTER_GLSL_

//Select shadow map filtering method.
/* #define SHADOW_MAP_FILTER_METHOD */

//Define the name of the sampler used in the include shader code.
/* #define SHADOW_MAP_SAMPLER_NAME */

float filterShadow(vec2 texCoord, float bias) {
	//choose different filter algorithm
}

#endif//_STP_SHADOW_MAP_FILTER_GLSL_