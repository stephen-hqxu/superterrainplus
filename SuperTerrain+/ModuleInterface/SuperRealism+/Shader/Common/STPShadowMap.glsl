#ifndef _STP_SHADOW_MAP_GLSL_
#define _STP_SHADOW_MAP_GLSL_

layout(std430, binding = 1) readonly restrict buffer STPShadowMap {
	layout(bindless_sampler, offset = 0) sampler2DArrayShadow ShadowMap;
};

//Implementation for writting depth data to shadow map
#ifdef RENDER_TO_SHADOW_MAP

#endif//RENDER_TO_SHADOW_MAP

//Implementation for reading from shadow map and filter
#ifdef READ_FROM_SHADOW_MAP

#endif//READ_FROM_SHADOW_MAP

#endif//_STP_SHADOW_MAP_GLSL_