#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require
//For non-uniform access to bindless texture
#extension GL_NV_gpu_shader5 : require

layout(early_fragment_tests) in;

#define UINT_MAX 4294967295

//Input
in vec2 FragTexCoord;
//Output
layout(location = 0) out vec4 FragColor;

/* ------------------------------------------ Lighting ----------------------------------------------- */
//these macros will be defined before compilation
#define ENVIRONMENT_LIGHT_CAPACITY 1
#define DIRECTIONAL_LIGHT_SHADOW_CAPACITY 1
#define LIGHT_FRUSTUM_DIVISOR_CAPACITY 1

struct EnvironmentLight{
	float Ka, Kd, Ks;
	vec3 Dir;
	//only environment light behaves as both ambient and diffuse light source
	layout(bindless_sampler) sampler1DArray LightSpectrum;
	//spectrum lighting requires information to sample from the spectrum
	//this piece of information is user-defined, depending on how the spectrum is generated by the user
	float SpectrumCoord;
	//index to the directional light shadow list
	//if this directional light does not use shadow, it will be set to a special value
	unsigned int DirShadowIdx;
};
uniform EnvironmentLight EnvironmentLightList[ENVIRONMENT_LIGHT_CAPACITY];

//Record the actual size available in each light list
uniform unsigned int EnvLightCount = 0u;

/* -------------------------------- Shadow ----------------------------------- */
#include </Common/STPLightSpaceInformation.glsl>

#define LIGHT_SHADOW_FILTER 255
#define UNUSED_SHADOW 1

struct DirectionalShadowData{
	layout(bindless_sampler) sampler2DArrayShadow CascadedShadowMap;
	//The starting index to the light space matrix array and frustum divisor array
	unsigned int LightSpaceStart, DivisorStart;
	unsigned int LightSpaceDim;
	//divisor size is light space size minus 1
};
uniform DirectionalShadowData DirectionalShadowList[DIRECTIONAL_LIGHT_SHADOW_CAPACITY];
uniform float LightFrustumDivisor[LIGHT_FRUSTUM_DIVISOR_CAPACITY];

//global shadow setting
uniform float LightFrustumFar;
uniform float MaxBias, MinBias;

/* -------------------------------------------------------------------------- */
#include </Common/STPCameraInformation.glsl>

//Geometry Buffer
layout(bindless_sampler) uniform sampler2D GBuffer[5];
#define G_ALBEDO GBuffer[0]
#define G_NORMAL GBuffer[1]
#define G_SPECULAR GBuffer[2]
#define G_AO GBuffer[3]
#define G_DEPTH GBuffer[4]

//Recover fragment world position from depth buffer
vec3 depthReconstruction();
//Calculate light color for the current fragment position
vec3 calcCasterLight(vec3, vec3, float, float, EnvironmentLight);
//input normal must be normalised
//This function returns the light intensity multiplier in the range [0.0, 1.0], with 0.0 means no light and 1.0 means full light.
float sampleShadow(vec3, vec3, vec3, DirectionalShadowData);

void main(){
	const vec3 position_world = depthReconstruction(),
		normal_world = normalize(texture(G_NORMAL, FragTexCoord).rgb);
	//get material data from the G-buffer
	const vec3 Albedo = texture(G_ALBEDO, FragTexCoord).rgb;
	const float Specular = texture(G_SPECULAR, FragTexCoord).r,
		Ambient = texture(G_AO, FragTexCoord).r;
	vec3 LightColor = vec3(0.0f);

	//lighting pass
	for(unsigned int i = 0u; i < EnvLightCount; i++){
		LightColor += calcCasterLight(position_world, normal_world, Specular, Ambient, EnvironmentLightList[i]);
	}
	
	//because the light calculation only calculates light color, 
	//we need to burn the geometry color into the final color
	FragColor = vec4(Albedo * LightColor, 1.0f);
}

vec3 depthReconstruction(){
	//depth has range [0, 1]
	const float Depth = texture(G_DEPTH, FragTexCoord).r;
	//OpenGL requires NDC to be in range [-1, 1], so we need to convert the range
	//Note that texture coordinate is also a [0, 1] range.
	const vec4 position_clip = vec4(vec3(FragTexCoord, Depth) * 2.0f - 1.0f, 1.0f),
		position_world = Camera.InvProjectionView * position_clip;

	return position_world.xyz / position_world.w;
}

vec3 calcCasterLight(vec3 position_world, vec3 normal, float specular_strength, float ambient_strength, EnvironmentLight env_light){
	//get light color
	//spectrum lighting, sample light color from the spectrum
	//the spectrum should define an indirect and a direct light color
	const float spec_uv = env_light.SpectrumCoord;
	const vec3 indirect_color = texture(env_light.LightSpectrum, vec2(spec_uv, 0.0f)).rgb,
		direct_color = texture(env_light.LightSpectrum, vec2(spec_uv, 1.0f)).rgb;

	//ambient
	const float ambient = ambient_strength * env_light.Ka;
	//diffuse
	const vec3 lightDir = normalize(env_light.Dir);
	const float diffuse = env_light.Kd * max(dot(lightDir, normal), 0.0f);
	//specular
	const vec3 viewDir = normalize(Camera.Position - position_world),
		reflectDir = reflect(-lightDir, normal),
		halfwayDir = normalize(lightDir + viewDir);
	//TODO: read shineness from material G-buffer
	const float specular = specular_strength * env_light.Ks * pow(max(dot(normal, halfwayDir), 0.0f), 32.0f);
	
	const uint dirShadowIdx = env_light.DirShadowIdx;
	//if this light has shadow, calculate light intensity after shadow calculation
	//the returned value represents the light intensity multiplier
	const float light_intensity = (dirShadowIdx != UNUSED_SHADOW) ? sampleShadow(position_world, normal, lightDir, DirectionalShadowList[dirShadowIdx]) : 1.0f;
	return indirect_color * ambient + direct_color * (diffuse + specular) * light_intensity;
}

float sampleShadow(vec3 fragworldPos, vec3 normal, vec3 lightDir, DirectionalShadowData dir_shadow) {
	//select cascade level from array shadow texture
	const vec4 fragviewPos = Camera.View * vec4(fragworldPos, 1.0f);
	const float depthValue = abs(fragviewPos.z);
	const uint cascadeCount = dir_shadow.LightSpaceDim - 1u;

	uint layer = UINT_MAX;
	for (uint i = 0u; i < cascadeCount; i++) {
		if (depthValue < LightFrustumDivisor[dir_shadow.DivisorStart + i]) {
			layer = i;
			break;
		}
	}
	//no layer can be determined
	if (layer == UINT_MAX) {
		layer = cascadeCount;
	}

	//convert world position to light clip space
	//as we are dealing with directional light, w component is always 1.0
	const vec4 fraglightPos = LightSpace.ProjectionView[dir_shadow.LightSpaceStart + layer] * vec4(fragworldPos, 1.0f);
	//perform perspective division and transform to [0, 1] range
	const vec3 projCoord = (fraglightPos.xyz / fraglightPos.w) * 0.5f + 0.5f;

	//get depth of current fragment from light's perspective
	const float currentDepth = projCoord.z;
	if (currentDepth > 1.0f) {
		//keep the light intensity at 0.0 when outside the far plane region of the light's frustum.
		return 1.0f;
	}

	//calcualte bias based on depth map resolution and slope
	float bias = max(MaxBias * (1.0f - dot(normal, lightDir)), MinBias);
	bias /= ((layer == cascadeCount) ? LightFrustumFar : LightFrustumDivisor[dir_shadow.DivisorStart + layer]) * 0.5f;

	//get closest depth value from light's perspective
	//the `texture` function computes the shadow value and returns (1.0 - shadow).
#if LIGHT_SHADOW_FILTER == 2
	//Percentage-Closer Filtering

#else
	//no filter, nearest and linear filtering are done by hardware automatically
	return texture(dir_shadow.CascadedShadowMap, vec4(projCoord.xy, layer, currentDepth - bias)).r;
#endif
}