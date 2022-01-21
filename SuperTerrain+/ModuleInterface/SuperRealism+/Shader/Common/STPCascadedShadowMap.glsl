#ifndef _STP_CASCADED_SHADOW_MAP_GLSL_
#define _STP_CASCADED_SHADOW_MAP_GLSL_

//The number of light space matrix.
//Define a valid number for a fixed compile time constant
/* #define CSM_LIGHT_SPACE_COUNT */
#ifndef CSM_LIGHT_SPACE_COUNT
#define CSM_LIGHT_SPACE_COUNT "The number of light space is unspecified"
#endif

//Define to emit implementation for sampling the shadow map
/* #define CSM_SAMPLING_IMPLEMENTATION */

//Define to use different shadow map filters, when sampling implementation is used
/* #define CSM_SHADOW_FILTER */

struct STPLightSpaceInformation {
	//accroding to ARB specification, uvec2 can be used instead of uint64 to avoid adding extension dependencies to every shader.
	uvec2 ShadowMap;
	float FarPlane;
	//First component defines the bias multiplier and the second one defines the minimum bias
	vec2 ShadowBias;
#if CSM_SHADOW_FILTER == 2
	vec2 FilterKernel;
#endif
};

layout(std430, binding = 1) readonly restrict buffer STPCascadedShadowMap {
	//for now we only support a single light that can cast shadow, this will be mitigated in the future
	layout(offset = 0) STPLightSpaceInformation LightSpace;
	layout(offset = 32) mat4 ShadowLightSpace[CSM_LIGHT_SPACE_COUNT];
	//Those are separated by these planes as well as the near/far plane.
	//They define the middle planes inside the cascade, if we have N frusta, there will be N - 1 cascade planes.
	layout(offset = 32 + CSM_LIGHT_SPACE_COUNT * 64) float CascadePlaneDistance[CSM_LIGHT_SPACE_COUNT - 1];
};

//Implementation for sampling Cascaded Shadow Map
#ifdef CSM_SAMPLING_IMPLEMENTATION
#ifndef CSM_SHADOW_FILTER
#define CSM_SHADOW_FILTER 255
#endif

//input normal and light direction must be normalised
//This function returns the light intensity multiplier in the range [0.0, 1.0], with 0.0 means no light and 1.0 means full light.
float sampleShadow(vec3 fragworldPos, mat4 view, vec3 normal, vec3 lightDir) {
	//select cascade level from array shadow texture
	const vec4 fragviewPos = view * vec4(fragworldPos, 1.0f);
	const float depthValue = abs(fragviewPos.z);
	const int cascadeCount = CascadePlaneDistance.length;

	int layer = -1;
	for (int i = 0; i < cascadeCount; i++) {
		if (depthValue < CascadePlaneDistance[i]) {
			layer = i;
			break;
		}
	}
	//no layer can be determined
	if (layer == -1) {
		layer = cascadeCount;
	}

	//convert world position to light clip space
	//as we are dealing with directional light, w component is always 1.0
	const vec4 fraglightPos = ShadowLightSpace[layer] * vec4(fragworldPos, 1.0f);
	//perform perspective division and transform to [0, 1] range
	const vec3 projCoord = (fraglightPos.xyz / fraglightPos.w) * 0.5f + 0.5f;

	//get depth of current fragment from light's perspective
	const float currentDepth = projCoord.z;
	if (currentDepth > 1.0f) {
		//keep the light intensity at 0.0 when outside the far plane region of the light's frustum.
		return 1.0f;
	}

	//calcualte bias based on depth map resolution and slope
	float bias = max(LightSpace.ShadowBias.x * (1.0f - dot(normal, lightDir)), LightSpace.ShadowBias.y);
	bias /= ((layer == cascadeCount) ? LightSpace.FarPlane : CascadePlaneDistance[layer]) * 0.5f;

	const sampler2DArrayShadow shadow_map = sampler2DArrayShadow(LightSpace.ShadowMap);
	//get closest depth value from light's perspective
	//the `texture` function computes the shadow value and returns (1.0 - shadow).
#if CSM_SHADOW_FILTER == 2
	//Percentage-Closer Filtering

#else
	//no filter, nearest and linear filtering are done by hardware automatically
	return texture(shadow_map, vec4(projCoord.xy, layer, currentDepth - bias)).r;
#endif
}
#endif//CSM_SAMPLING_IMPLEMENTATION

#endif//_STP_CASCADED_SHADOW_MAP_GLSL_