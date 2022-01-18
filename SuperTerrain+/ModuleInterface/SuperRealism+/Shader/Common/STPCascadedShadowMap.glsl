#ifndef _STP_CASCADED_SHADOW_MAP_GLSL_
#define _STP_CASCADED_SHADOW_MAP_GLSL_

//The number of light space matrix.
//Define a valid number for a fixed compile time constant
/* #define CSM_LIGHT_SPACE_COUNT */

//Define to emit implementation for sampling the shadow map
/* #define CSM_SAMPLING_IMPLEMENTATION */

layout(std430, binding = 1) readonly restrict buffer STPCascadedShadowMap {
	layout(offset = 0) float FarPlane;
	//First component defines the bias multiplier and the second one defines the minimum bias
	layout(offset = 8) vec2 ShadowBias;
	layout(offset = 16) mat4 ShadowLightSpace[CSM_LIGHT_SPACE_COUNT];
	//Those are separated by these planes as well as the near/far plane.
	//They define the middle planes inside the cascade, if we have N frusta, there will be N - 1 cascade planes.
	layout(offset = 16 + CSM_LIGHT_SPACE_COUNT * 64) float CascadePlaneDistance[CSM_LIGHT_SPACE_COUNT - 1];
};

//Implementation for sampling Cascaded Shadow Map
#ifdef CSM_SAMPLING_IMPLEMENTATION
//input normal and light direction must be normalised
float sampleShadow(vec3 fragworldPos, mat4 view, vec3 normal, vec3 lightDir, sampler2DArrayShadow shadow_map) {
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
		//keep the shadow at 0.0 when outside the far plane region of the light's frustum.
		return 0.0f;
	}

	//calcualte bias based on depth map resolution and slope
	float bias = max(ShadowBias.x * (1.0f - dot(normal, lightDir)), ShadowBias.y);
	bias *= (layer == cascadeCount) ? 1.0f / (FarPlane * 0.5f) : 1.0f / (CascadePlaneDistance[layer] * 0.5f);

	//get closest depth value from light's perspective
	return texture(shadow_map, vec4(projCoord.xy, layer, currentDepth - bias)).r;
}
#endif//CSM_SAMPLING_IMPLEMENTATION

#endif//_STP_CASCADED_SHADOW_MAP_GLSL_