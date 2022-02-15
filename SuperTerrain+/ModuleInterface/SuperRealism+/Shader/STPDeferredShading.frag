#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require
//For non-uniform access to bindless texture
#extension GL_NV_gpu_shader5 : require

layout(early_fragment_tests) in;

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
#define SHADOW_CASCADE_BLEND 0
#define UNUSED_SHADOW 1

//choose different sampler type based on the filter used automcatically
#if LIGHT_SHADOW_FILTER == 16
#define SHADOW_MAP_FORMAT sampler2DArray
#else
#define SHADOW_MAP_FORMAT sampler2DArrayShadow
#endif

struct DirectionalShadowData{
	layout(bindless_sampler) SHADOW_MAP_FORMAT CascadedShadowMap;
	//The starting index to the light space matrix array and frustum divisor array
	unsigned int LightSpaceStart, DivisorStart;
	unsigned int LightSpaceDim;
	//divisor size is light space size minus 1
};
uniform DirectionalShadowData DirectionalShadowList[DIRECTIONAL_LIGHT_SHADOW_CAPACITY];
uniform float LightFrustumDivisor[LIGHT_FRUSTUM_DIVISOR_CAPACITY];

struct ShadowMapFilter{
#if LIGHT_SHADOW_FILTER == 2
	uint Kr;
	float Ks;
#elif LIGHT_SHADOW_FILTER == 16
	float minVar;
#endif
	//each vector represents the max and min bias respectively
	vec2 Db, Nb;
	float FarBias;

#if SHADOW_CASCADE_BLEND
	//options to blend between cascades for directional light shadow
	float Br;
#endif
};
//global shadow setting
uniform ShadowMapFilter Filter;

/* -------------------------------------------------------------------------- */
//enable depth reconstruction to world space
#define EMIT_DEPTH_RECON_WORLD_IMPL
#include </Common/STPCameraInformation.glsl>

//Geometry Buffer
layout(bindless_sampler) uniform sampler2D GBuffer[5];
#define G_ALBEDO GBuffer[0]
#define G_NORMAL GBuffer[1]
#define G_SPECULAR GBuffer[2]
#define G_AO GBuffer[3]
#define G_DEPTH GBuffer[4]

//General settings for shading
uniform float ExtinctionBand;

//Calculate light color for the current fragment position
vec3 calcCasterLight(vec3, vec3, float, float, EnvironmentLight);
//This function returns the light intensity multiplier in the range [0.0, 1.0], with 0.0 means no light and 1.0 means full light.
float sampleShadow(vec3, float, DirectionalShadowData);

void main(){
	const float fragment_depth = texture(G_DEPTH, FragTexCoord).r;
	//recover world position
	const vec3 position_world = fragDepthReconstruction(fragment_depth, FragTexCoord),
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
	
	//extinction calculation, making the transition to camera far clipping plane smooth instead of having a strong cut
	const float cameraDistance = distance(Camera.Position, position_world),
		extinctionStart = Camera.Far * ExtinctionBand,
		//linearly interpolate within the extinction band, 0 means no extiontion while 1 means the geometry is fully extincted
		extinctionFactor = (cameraDistance - extinctionStart) / (Camera.Far - extinctionStart);

	//because the light calculation only calculates light color, 
	//we need to burn the geometry color into the final color
	FragColor = vec4(Albedo * LightColor, clamp(extinctionFactor, 0.0f, 1.0f));
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
	//calcualte bias based on light angle
	const float slopeFactor = 1.0f - dot(normal, lightDir),
		depth_bias = max(Filter.Db.x * slopeFactor, Filter.Db.y),
		normal_bias = max(Filter.Nb.x * slopeFactor, Filter.Nb.y);

	//if this light has shadow, calculate light intensity after shadow calculation
	//the returned value represents the light intensity multiplier
	const float light_intensity = (dirShadowIdx != UNUSED_SHADOW) ? 
		sampleShadow(position_world + normal * normal_bias, depth_bias, DirectionalShadowList[dirShadowIdx]) : 1.0f;
	return indirect_color * ambient + direct_color * (diffuse + specular) * light_intensity;
}

#if LIGHT_SHADOW_FILTER == 2
#define USE_PCF_FILTER
#endif

float filterShadow(SHADOW_MAP_FORMAT shadow_map, vec2 projection_coord, float frag_depth, uint layer
#ifdef USE_PCF_FILTER
, float pcf_sampling_texel, float pcf_totalKernel_sq_inv
#endif
){
#ifdef USE_PCF_FILTER
	//Percentage-Closer Filtering
	const int radius = int(Filter.Kr);

	float intensity = 0.0f;
	for(int i = -radius; i <= radius; i++){
		for(int j = -radius; j <= radius; j++){
			intensity += texture(shadow_map, vec4(projection_coord + pcf_sampling_texel * vec2(i, j), layer, frag_depth)).r;
		}
	}

	return intensity * pcf_totalKernel_sq_inv;
#elif LIGHT_SHADOW_FILTER == 16
	//Variance Shadow Map
	const vec2 moment = texture(shadow_map, vec3(projection_coord, layer)).rg;
	//one tail inequality is only valid if current depth > moment.x
	if(frag_depth <= moment.x){
		return 1.0f;
	}

	const float mean = moment.x,
		variance = max(moment.y - mean * mean, Filter.minVar),
		//use Chebyshev's inequality to compute an upper bound on the probability that the currently 
		//shaded surface at this depth is occluded
		difference = (frag_depth - mean),
		max_probability = variance / (variance + difference * difference);

	//return the probability the surface is lit
	return max_probability;
#else
	//no filter, nearest and linear filtering are done by hardware automatically
	return texture(shadow_map, vec4(projection_coord, layer, frag_depth)).r;
#endif
}

vec3 determineShadowCoord(vec4 worldPos, uint lightSpaceIdx){
	//convert world position to light clip space
	const vec4 fraglightPos = LightSpace.ProjectionView[lightSpaceIdx] * worldPos;
	//perform perspective division and transform to [0, 1] range
	return (fraglightPos.xyz / fraglightPos.w) * 0.5f + 0.5f;
}

vec2 determineLayerFarBias(uint layer, uint cascadeCount, uint divIdx, float original_bias){
	const float layerFar = (layer == cascadeCount) ? Camera.Far : LightFrustumDivisor[divIdx];
	//scale the bias depends on how far the frustum plane is
	return vec2(layerFar, original_bias / (layerFar * Filter.FarBias));
}

float sampleShadow(vec3 world_position, float rawBias, DirectionalShadowData dir_shadow) {
	//as we are dealing with directional light, w component is always 1.0
	const vec4 fragworldPos = vec4(world_position, 1.0f);
	//The shadow map is always a square
	const float shadowTexel = 1.0f / float(textureSize(dir_shadow.CascadedShadowMap, 0).x);

	//select cascade level from array shadow texture
	const vec4 fragviewPos = Camera.View * fragworldPos;
	const float depthValue = abs(fragviewPos.z);
	const uint cascadeCount = dir_shadow.LightSpaceDim - 1u;
	
	//determine the correct shadow layer to use
	//use the last layer in case if no layer can be determined
	uint layer = cascadeCount;
	for (uint i = 0u; i < cascadeCount; i++) {
		if (depthValue < LightFrustumDivisor[dir_shadow.DivisorStart + i]) {
			layer = i;
			break;
		}
	}

	const vec3 projCoord = determineShadowCoord(fragworldPos, dir_shadow.LightSpaceStart + layer);
	const float currentDepth = projCoord.z;
	if (currentDepth > 1.0f) {
		//keep the light intensity at 0.0 when outside the far plane region of the light's frustum.
		return 1.0f;
	}

	//x is the layer far and y is the layer bias
	const vec2 layerFarBias = determineLayerFarBias(layer, cascadeCount, dir_shadow.DivisorStart + layer, rawBias);
	const float layerBias = layerFarBias.y;

	//get closest depth value from light's perspective
	//the `texture` function computes the shadow value and returns (1.0 - shadow).
#ifdef USE_PCF_FILTER
	const uint totalKernel = Filter.Kr * 2u + 1u;
	const float totalKernel_sq_inv = 1.0f / float(totalKernel * totalKernel),
		//by scaling the shadow texel, we can configure the step size within a filter kernel
		filter_texel = Filter.Ks * shadowTexel;
#endif

	const float light_intensity = filterShadow(dir_shadow.CascadedShadowMap, projCoord.xy, currentDepth - layerBias, layer
#ifdef USE_PCF_FILTER
		, filter_texel, totalKernel_sq_inv
#endif
	);

#if SHADOW_CASCADE_BLEND
	const float currentLayerFar = layerFarBias.x;
	//blend between cascade layers
	if(layer != cascadeCount && depthValue + Filter.Br > currentLayerFar){
		//not the last layer and the current texel is in the blending region.
		//no need to blend if we are at the last cascade.
		//Such that it is valid to have "the next layer"
		const uint nextLayer = layer + 1u;

		//now repeat all previous calculations for this new layer
		const vec3 blendProjCoord = determineShadowCoord(fragworldPos, dir_shadow.LightSpaceStart + nextLayer);
		const float blendDepth = blendProjCoord.z;
		if(blendDepth > 1.0f){
			//if the blending layer is outside, simply abort blending
			return light_intensity;
		}

		//bias calculation
		const vec2 blendLayerFarBias = determineLayerFarBias(nextLayer, cascadeCount, dir_shadow.DivisorStart + nextLayer, rawBias);
		const float blendBias = blendLayerFarBias.y;

		//apply filter as usual
		const float blend_light_intensity = filterShadow(dir_shadow.CascadedShadowMap, blendProjCoord.xy, blendDepth - blendBias, nextLayer
#ifdef USE_PCF_FILTER
			, filter_texel, totalKernel_sq_inv
#endif
		);

		//mix the values together smoothly
		//The current layer far plane cuts the blend region somewhere in between
		return mix(blend_light_intensity, light_intensity, (currentLayerFar - depthValue) / Filter.Br);
	}
#endif
	//return the value as usual either when blending is not enabled or not applicable
	return light_intensity;
}