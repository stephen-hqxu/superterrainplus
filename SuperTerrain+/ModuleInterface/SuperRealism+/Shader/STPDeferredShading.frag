#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require
//For non-uniform access to bindless texture
#extension GL_NV_gpu_shader5 : require
#extension GL_NV_shader_buffer_load : require

layout(early_fragment_tests) in;

//Input
in vec2 FragTexCoord;
//Output
layout(location = 0) out vec4 FragColor;

/* -------------------------------- Shadow ----------------------------------- */
#define LIGHT_SHADOW_FILTER 255
#define SHADOW_CASCADE_BLEND 0

//choose different sampler type based on the filter used automatically
#if LIGHT_SHADOW_FILTER == 16
#define SHADOW_MAP_FORMAT sampler2DArray
#else
#define SHADOW_MAP_FORMAT sampler2DArrayShadow
#endif

struct DirectionalShadow{
	layout(bindless_sampler) SHADOW_MAP_FORMAT CascadedShadowMap;
	unsigned int LightSpaceDim;
	//divisor size is light space size minus 1
	readonly mat4* restrict LightMatrix;
	readonly float* restrict Divisor;
};

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

/* ------------------------------------------ Lighting ------------------------------------------ */
#include </Common/STPNullPointer.glsl>

//these macros will be defined before compilation
#define AMBIENT_LIGHT_CAPACITY 1
#define DIRECTIONAL_LIGHT_CAPACITY 1
#define SHADING_MODEL 0

//For each light, there is a light spectrum for light colour information.
//spectrum lighting requires information to sample from the spectrum
//this piece of information is user-defined, depending on how the spectrum is generated by the user
//For shadow-casting light there is a pointer to shadow.
//If the light does not use shadow, it will be set to nullptr

struct AmbientLight{
	float Ka;
	float SpecCoord;
	layout(bindless_sampler) sampler1D AmbSpec;
	//ambient light never casts shadow
};
uniform AmbientLight* AmbientLightList[AMBIENT_LIGHT_CAPACITY];

struct DirectionalLight{
	vec3 Dir;
	float SpecCoord;
	float Kd, Ks;
	layout(bindless_sampler) sampler1D DirSpec;
	readonly DirectionalShadow* restrict DirShadow;
};
uniform DirectionalLight* DirectionalLightList[DIRECTIONAL_LIGHT_CAPACITY];

//Record the actual size available in each light list
uniform unsigned int AmbCount = 0u, 
	DirCount = 0u;

//Shading model selection
struct ShadingDescription{
#if SHADING_MODEL == 0
	//Blinn-Phong
	float minRough, maxRough;
	float minShin, maxShin;
#endif
};
uniform ShadingDescription ShadingModel;

/* ------------------------------------------------------------------------------------------------ */
//enable depth reconstruction to world space
#define EMIT_DEPTH_RECON_WORLD_IMPL
#include </Common/STPCameraInformation.glsl>

//Geometry Buffer
layout(bindless_sampler) uniform sampler2D GBuffer[5];
#define G_ALBEDO GBuffer[0]
#define G_NORMAL GBuffer[1]
#define G_ROUGHNESS GBuffer[2]
#define G_AO GBuffer[3]
#define G_DEPTH GBuffer[4]

//General settings for shading
uniform float ExtinctionBand;

//Calculate light colour for the current fragment position
vec3 calcAmbientLight(float, AmbientLight* restrict);
vec3 calcDirectionalLight(vec3, vec3, vec3, float, DirectionalLight* restrict);
//This function returns the light intensity multiplier in the range [0.0, 1.0], with 0.0 means no light and 1.0 means full light.
float sampleShadow(vec3, float, DirectionalShadow* restrict);

void main(){
	const float fragment_depth = textureLod(G_DEPTH, FragTexCoord, 0.0f).r;
	//recover world position
	const vec3 position_world = fragDepthReconstruction(fragment_depth, FragTexCoord),
		normal_world = normalize(textureLod(G_NORMAL, FragTexCoord, 0.0f).rgb);
	//get material data from the G-buffer
	const vec3 Albedo = textureLod(G_ALBEDO, FragTexCoord, 0.0f).rgb;
	const float Roughness = textureLod(G_ROUGHNESS, FragTexCoord, 0.0f).r,
		Ambient = textureLod(G_AO, FragTexCoord, 0.0f).r;

	const vec3 viewDirection = normalize(Camera.Position - position_world);
	vec3 LightColor = vec3(0.0f);
	//For non-PBR rendering equation, a simple linear function can be used.
	//For PBR rendering equation, well just, implement the equation.
#if SHADING_MODEL == 0
	const float Shininess = mix(ShadingModel.minShin, ShadingModel.maxShin,
		clamp((Roughness - ShadingModel.minRough) / (ShadingModel.maxRough - ShadingModel.minRough), 0.0f, 1.0f));
#endif

	//ambient light pass
	for(int i = 0; i < AmbCount; i++){
		LightColor += calcAmbientLight(Ambient, AmbientLightList[i]);
	}
	//directional light pass
	for(int i = 0; i < DirCount; i++){
		LightColor += calcDirectionalLight(position_world, viewDirection, normal_world, Shininess, DirectionalLightList[i]);
	}
	
	//extinction calculation, making the transition to camera far clipping plane smooth instead of having a strong cut
	const float cameraDistance = distance(Camera.Position, position_world),
		extinctionStart = Camera.Far * ExtinctionBand,
		//linearly interpolate within the extinction band, 0 means no extinction while 1 means the geometry is fully extinct
		extinctionFactor = smoothstep(extinctionStart, Camera.Far, cameraDistance);

	//because the light calculation only calculates light colour, 
	//we need to burn the geometry colour into the final colour
	FragColor = vec4(Albedo * LightColor, extinctionFactor);
}

vec3 calcAmbientLight(float ambient_strength, AmbientLight* restrict amb_light){
	const vec3 indirect_color = textureLod(amb_light->AmbSpec, amb_light->SpecCoord, 0.0f).rgb;
	const float ambient = ambient_strength * amb_light->Ka;

	return indirect_color * ambient;
}

vec3 calcDirectionalLight(vec3 position_world, vec3 view_direction, vec3 normal, float shininess, DirectionalLight* restrict dir_light){
	const vec3 direct_color = textureLod(dir_light->DirSpec, dir_light->SpecCoord, 0.0f).rgb;

	//diffuse
	const vec3 lightDir = normalize(dir_light->Dir);
	const float diffuse = dir_light->Kd * max(dot(lightDir, normal), 0.0f);
	//specular
	const vec3 halfwayDir = normalize(lightDir + view_direction);
	const float specular = dir_light->Ks * pow(max(dot(normal, halfwayDir), 0.0f), shininess);
	
	float light_intensity = 1.0f;
	if(!isNull(dir_light->DirShadow)){
		//this light casts shadow
		//calculate bias based on light angle
		const float slopeFactor = 1.0f - dot(normal, lightDir),
			depth_bias = max(Filter.Db.x * slopeFactor, Filter.Db.y),
			normal_bias = max(Filter.Nb.x * slopeFactor, Filter.Nb.y);

		//if this light has shadow, calculate light intensity after shadow calculation
		//the returned value represents the light intensity multiplier
		light_intensity = sampleShadow(position_world + normal * normal_bias, depth_bias, dir_light->DirShadow);
	}
	return direct_color * (diffuse + specular) * light_intensity;
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
			intensity += textureLod(shadow_map, vec4(projection_coord + pcf_sampling_texel * vec2(i, j), layer, frag_depth), 0.0f).r;
		}
	}

	return intensity * pcf_totalKernel_sq_inv;
#elif LIGHT_SHADOW_FILTER == 16
	//Variance Shadow Map
	const vec2 moment = texture(shadow_map, vec3(projection_coord, layer)).rg;
	//one tail inequality is only valid if current depth < moment.x, because of reversed depth
	if(frag_depth >= moment.x){
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
	return textureLod(shadow_map, vec4(projection_coord, layer, frag_depth), 0.0f).r;
#endif
}

vec3 determineShadowCoord(vec4 worldPos, mat4* restrict light_space){
	//convert world position to light clip space
	const vec4 fraglightPos = *light_space * worldPos;
	//perform perspective division and transform to [0, 1] range
	vec3 fragShadowCoord = (fraglightPos.xyz / fraglightPos.w);
	//depth is already in [0, 1] because the projection definition uses DirectX convention
	fragShadowCoord.xy = fragShadowCoord.xy * 0.5f + 0.5f;
	return fragShadowCoord;
}

vec2 determineLayerFarBias(uint layer, uint cascadeCount, float* restrict div, float original_bias){
	const float layerFar = (layer == cascadeCount) ? Camera.Far : div[layer];
	//scale the bias depends on how far the frustum plane is
	return vec2(layerFar, original_bias / (layerFar * Filter.FarBias));
}

float sampleShadow(vec3 world_position, float rawBias, DirectionalShadow* restrict dir_shadow) {
	//as we are dealing with directional light, w component is always 1.0
	const vec4 fragworldPos = vec4(world_position, 1.0f);
	//The shadow map is always a square
	const float shadowTexel = 1.0f / float(textureSize(dir_shadow->CascadedShadowMap, 0).x);

	//select cascade level from array shadow texture
	const vec4 fragviewPos = Camera.View * fragworldPos;
	const float depthValue = abs(fragviewPos.z);
	const uint cascadeCount = dir_shadow->LightSpaceDim - 1u;
	
	//determine the correct shadow layer to use
	//use the last layer in case if no layer can be determined
	uint layer = cascadeCount;
	for (uint i = 0u; i < cascadeCount; i++) {
		if (depthValue < dir_shadow->Divisor[i]) {
			layer = i;
			break;
		}
	}

	const vec3 projCoord = determineShadowCoord(fragworldPos, dir_shadow->LightMatrix + layer);
	const float currentDepth = projCoord.z;
	if (currentDepth < 0.0f) {
		//keep the light intensity at 0.0 when outside the far plane region of the light's frustum.
		return 1.0f;
	}

	//x is the layer far and y is the layer bias
	const vec2 layerFarBias = determineLayerFarBias(layer, cascadeCount, dir_shadow->Divisor, rawBias);
	const float layerBias = layerFarBias.y;

	//get closest depth value from light's perspective
	//the `texture` function computes the shadow value and returns (1.0 - shadow).
#ifdef USE_PCF_FILTER
	const uint totalKernel = Filter.Kr * 2u + 1u;
	const float totalKernel_sq_inv = 1.0f / float(totalKernel * totalKernel),
		//by scaling the shadow texel, we can configure the step size within a filter kernel
		filter_texel = Filter.Ks * shadowTexel;
#endif
	
	//the bias moves the fragment towards the light slightly, and in reversed depth buffer, closer is bigger, so use addition
	const float light_intensity = filterShadow(dir_shadow->CascadedShadowMap, projCoord.xy, currentDepth + layerBias, layer
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
		const vec3 blendProjCoord = determineShadowCoord(fragworldPos, dir_shadow->LightMatrix + nextLayer);
		const float blendDepth = blendProjCoord.z;
		if(blendDepth < 0.0f){
			//if the blending layer is outside, simply abort blending
			return light_intensity;
		}

		//bias calculation
		const vec2 blendLayerFarBias = determineLayerFarBias(nextLayer, cascadeCount, dir_shadow->Divisor, rawBias);
		const float blendBias = blendLayerFarBias.y;

		//apply filter as usual
		const float blend_light_intensity = filterShadow(dir_shadow->CascadedShadowMap, blendProjCoord.xy, blendDepth + blendBias, nextLayer
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