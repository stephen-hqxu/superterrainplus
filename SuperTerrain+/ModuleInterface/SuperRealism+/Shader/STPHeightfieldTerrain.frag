#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require
//this extension allows access array of bindless samplers non-uniformly
#extension GL_NV_gpu_shader5 : require
#extension GL_NV_shader_buffer_load : require

#define TWO_PI 6.283185307179586476925286766559

/* ------------------ Texture Splatting ---------------------------- */
//Macros are define by the main application, having an arbitrary number just to make the compiler happy.
#define GROUP_COUNT 1
#define REGISTRY_COUNT 1
#define SPLAT_REGION_COUNT 1

//texture type indexing
#define ALBEDO 1
#define NORMAL 1
#define ROUGHNESS 1
#define AO 1

//default texture values if a specific type is not used
#define DEFAULT_ALBEDO vec3(0.5f)
#define DEFAULT_NORMAL vec3(vec2(0.5f), 1.0f)
#define DEFAULT_ROUGHNESS 0.0f
#define DEFAULT_AO 1.0f

#define TYPE_STRIDE 6
#define UNREGISTERED_TYPE 0

struct TextureRegionSmoothSetting{
	unsigned int Kr;
	float Ks;
	unsigned int Ns;
};

struct TextureRegionScaleSetting{
	float Prim, Seco, Tert;
};

struct TerrainTextureData{
#if ALBEDO != UNREGISTERED_TYPE
	vec3 TerrainColor;
#endif
#if NORMAL != UNREGISTERED_TYPE
	vec3 TerrainNormal;
#endif
#if ROUGHNESS != UNREGISTERED_TYPE
	float TerrainRoughness;
#endif
#if AO != UNREGISTERED_TYPE
	float TerrainAmbientOcclusion;
#endif
};

//Rule-based texturing system
layout (bindless_sampler) uniform sampler2DArray RegionTexture[GROUP_COUNT];
//Each pointer is pointing to a location to texture region.
//The pointer might be null to indicate this region has no texture.
//For the location data, x: array index, y: layer index
uniform uvec2* RegionRegistry[REGISTRY_COUNT];
//each texture region will have one and only one scale setting
uniform uvec3 RegionScaleRegistry[SPLAT_REGION_COUNT];

uniform TextureRegionSmoothSetting SmoothSetting;
uniform TextureRegionScaleSetting ScaleSetting;
/* -------------------------- Terrain Lighting ------------------------- */
//Normalmap blending algorithm
#define NORMALMAP_BLENDING 255

/* --------------------------------------------------------------------- */

#include </Common/STPCameraInformation.glsl>
#include </Common/STPNullPointer.glsl>

//Input
in VertexTES{
	vec3 position_world;
	vec2 texCoord;
} fs_in;
//Output
#include </Common/STPGeometryBufferWriter.glsl>

layout (binding = 0) uniform sampler2D Heightmap;
layout (binding = 1) uniform usampler2D Splatmap;
layout (bindless_sampler) uniform sampler3D Noisemap;

//The number of visible chunk in x,z direction
uniform uvec2 VisibleChunk;
uniform vec2 ChunkHorizontalOffset;
//The strength of the z component on normalmap
uniform float NormalStrength;

const ivec2 ConvolutionKernelOffset[8] = {
	{ -1, -1 },
	{  0, -1 },
	{ +1, -1 },
	{ -1,  0 },
	{ +1,  0 },
	{ -1, +1 },
	{  0, +1 },
	{ +1, +1 },
};

//Get all types of available texture using texture region smoothing algorithm
TerrainTextureData getSmoothTexture(vec2);
/**
 * We have three normal systems for terrain, plane normal, terrain normal and terrain texture normal.
 * plane normal is simply (0,1,0), that's how our model is defined (model space normal)
 * terrain normal is the vector that perpendicular to the tessellated terrain
 * terrain texture normal is the normal comes along with the texture on the terrain later in the fragment shader,
 * each texture has its own dedicated normal map.
 * So we need to do the TBN transform twice: plane->terrain normal then terrain normal->terrain texture normal
*/
vec3 calcTerrainNormal();
//Blend a main normalmap with detail normalmap
vec3 blendNormal(vec3, vec3);

void main(){
	const vec3 MeshNormal = calcTerrainNormal();
	//this function make sure the UV is stable when the rendered chunk shifts
	//here we need to use the UV of the current pixel (not the argument one)
	//so when we are doing smoothing over a convolution kernel the texture colour remains the same for the same pixel.
	const vec2 worldUV = fs_in.texCoord * VisibleChunk + ChunkHorizontalOffset;

	//terrain texture splatting
	const TerrainTextureData TerrainTexture = getSmoothTexture(worldUV);

	//albedomap processing
#if ALBEDO != UNREGISTERED_TYPE
	const vec3 RenderingColor = TerrainTexture.TerrainColor;
#else
	//by default we visualise the terrain normal if there is no albedo texture
	const vec3 RenderingColor = (MeshNormal + 1.0f) * 2.0f;
#endif
	
	//normalmap processing
	//convert from tangent space to world space
	//since the original mesh is a upward plane, we only need to flip the normal
	const mat3 MeshTBN = mat3(
		vec3(1.0f, 0.0f, 0.0f),
		vec3(0.0f, 0.0f, 1.0f),
		vec3(0.0f, 1.0f, 0.0f)
	);
#if NORMAL != UNREGISTERED_TYPE
	//convert the basis of texture normal from [0,1] to [-1,1]
	const vec3 TextureNormal = normalize(TerrainTexture.TerrainNormal * 2.0f - 1.0f);
	
	//normalmap blending
	const vec3 RenderingNormal = normalize(MeshTBN * blendNormal(MeshNormal, TextureNormal));
#else
	const vec3 RenderingNormal = normalize(MeshTBN * MeshNormal);
#endif

	//roughnessmap processing
#if ROUGHNESS != UNREGISTERED_TYPE
	const float RenderingRoughness = TerrainTexture.TerrainRoughness;
#else
	const float RenderingRoughness = 0.0f;
#endif

	//ambient-occlusionmap processing
#if AO != UNREGISTERED_TYPE
	const float RenderingAO = TerrainTexture.TerrainAmbientOcclusion;
#else
	const float RenderingAO = 1.0f;
#endif

	//finally
	writeGeometryData(RenderingColor, RenderingNormal, RenderingRoughness, RenderingAO);
}

//dx_dy is the derivative used for sampling texture, the first two components store dx while the last two store dy.
vec3 getRegionTexture(vec2 texture_uv, vec3 replacement, unsigned int region, unsigned int type, vec4 dx_dy){
	const uint regionLoc = region * TYPE_STRIDE + type;
	//invalid region
	if(regionLoc >= REGISTRY_COUNT){
		return replacement;
	}

	uvec2* restrict const textureLoc = RegionRegistry[regionLoc];
	if(isNull(textureLoc)){
		//this region has no texture data
		return replacement;
	}

	const sampler2DArray selected_sampler = RegionTexture[textureLoc->x];
	//region is valid
	return textureGrad(selected_sampler, vec3(texture_uv, textureLoc->y), dx_dy.st, dx_dy.pq).rgb;
}

void sampleTerrainTexture(in out TerrainTextureData data, vec2 sampling_uv, unsigned int region, float weight, vec4 dx_dy){
#if ALBEDO != UNREGISTERED_TYPE
	data.TerrainColor += getRegionTexture(sampling_uv, DEFAULT_ALBEDO, region, ALBEDO, dx_dy) * weight;
#endif
#if NORMAL != UNREGISTERED_TYPE
	data.TerrainNormal += getRegionTexture(sampling_uv, DEFAULT_NORMAL, region, NORMAL, dx_dy) * weight;
#endif
#if ROUGHNESS != UNREGISTERED_TYPE
	data.TerrainRoughness += getRegionTexture(sampling_uv, vec3(DEFAULT_ROUGHNESS), region, ROUGHNESS, dx_dy).r * weight;
#endif
#if AO != UNREGISTERED_TYPE
	data.TerrainAmbientOcclusion += getRegionTexture(sampling_uv, vec3(DEFAULT_AO), region, AO, dx_dy).r * weight;
#endif
}

TerrainTextureData getSmoothTexture(vec2 world_uv){
	const uint regionBinSize = SPLAT_REGION_COUNT + 1u;
	//the bin is used to record the number of each region presented in the smoothing kernel.
	//The last bin acts as a dummy bin to handle any invalid region
	uint RegionBin[regionBinSize];
	//zero initialise
	for(uint i = 0u; i < regionBinSize; i++){
		RegionBin[i] = 0u;
	}

	const float Kr_inv = 1.0f / (1.0f * SmoothSetting.Kr),
		Kr_2_inv = 1.0f / (1.0f * SmoothSetting.Kr * SmoothSetting.Kr);
	//perform smoothing to integer texture
	for(int i = 0; i < SmoothSetting.Kr; i++){
		for(int j = 0; j < SmoothSetting.Kr; j++){
			//we first divide the kernel matrix into cells with equal spaces
			const vec2 domain = vec2(i, j) * Kr_inv,
				//then we jittered each sampling points from the cell centre
				//however we need to make sure samples are not jittered out of its cell
				stratified_domain = clamp(domain + Kr_inv * 
					textureLod(Noisemap, vec3(world_uv * SmoothSetting.Ns, (1.0f * i + j * SmoothSetting.Kr) * Kr_2_inv), 0).r, 0.0f, 1.0f);
			
			//then we map a squared domain into a disk domain.
			const float sq_domain_x = TWO_PI * stratified_domain.x;
			const vec2 disk_domain = sqrt(stratified_domain.y) * vec2(
				cos(sq_domain_x),
				sin(sq_domain_x)
			);
			
			//now apply the sampling points to the actual texture
			const vec2 uv_offset = SmoothSetting.Ks * disk_domain / vec2(textureSize(Heightmap, 0).xy),
				sampling_uv = fs_in.texCoord + uv_offset;
			const uint region = textureLod(Splatmap, sampling_uv, 0).r;

			//accumulate region count
			if(region < SPLAT_REGION_COUNT){
				//valid region
				RegionBin[region]++;
			}else{
				//invalid region to be recorded to the dummy bin at the end
				RegionBin[SPLAT_REGION_COUNT]++;
			}
		}
	}

	//prepare for the final output texture data
	TerrainTextureData TerrainTexture;
	//zero initialise
#if ALBEDO != UNREGISTERED_TYPE
	TerrainTexture.TerrainColor = vec3(0.0f);
#endif
#if NORMAL != UNREGISTERED_TYPE
	TerrainTexture.TerrainNormal = vec3(0.0f);
#endif
#if ROUGHNESS != UNREGISTERED_TYPE
	TerrainTexture.TerrainRoughness = 0.0f;
#endif
#if AO != UNREGISTERED_TYPE
	TerrainTexture.TerrainAmbientOcclusion = 0.0f;
#endif
	
	const float texelDistance = distance(Camera.Position, fs_in.position_world);
	//for each region, calculate their weights which is used as a blending factor to the final texture data
	for(uint region = 0u; region < SPLAT_REGION_COUNT; region++){
		const uint regionCount = RegionBin[region];
		if(regionCount == 0u){
			//skip region that does not contribute to the final colour
			continue;	
		}
		//normalise the region count as a weight
		const float regionWeight = regionCount * Kr_2_inv;

		//pre-compute derivatives for each scale levels because they branch
		vec4 UVScaleDxDy[3];
		for(int level = 0; level < 3; level++){
			const vec2 level_uv = world_uv * RegionScaleRegistry[region][level];
			
			UVScaleDxDy[level].xy = dFdx(level_uv);
			UVScaleDxDy[level].zw = dFdy(level_uv);
		}

		//default to use tertiary scale
		uvec2 scaleIdx = uvec2(2u, 0u);
		float blendFactor = 0.0f;
		//determine uv for the current region
		//the current implementation only allows a two different scale blended together
		if(texelDistance < ScaleSetting.Prim){
			//use the primary scale only
			scaleIdx[0] = 0u;
		}
		else if(texelDistance < ScaleSetting.Seco){
			//blend between primary and secondary
			blendFactor = smoothstep(ScaleSetting.Prim, ScaleSetting.Seco, texelDistance);
			scaleIdx[0] = 0u;
			scaleIdx[1] = 1u;
		}
		else if(texelDistance < ScaleSetting.Tert){
			//blend between secondary and tertiary
			blendFactor = smoothstep(ScaleSetting.Seco, ScaleSetting.Tert, texelDistance);
			scaleIdx[0] = 1u;
			scaleIdx[1] = 2u;
		}
		//otherwise use tertiary only

		//sample texture using the scaling information
		for(int scaleLayer = 0; scaleLayer < 2; scaleLayer++){
			const uint index = scaleIdx[scaleLayer];
			sampleTerrainTexture(TerrainTexture, world_uv * RegionScaleRegistry[region][index], region, regionWeight * (1.0f - blendFactor), UVScaleDxDy[index]);

			//check if there is another scale layer needs to be blended
			if(blendFactor == 0.0f){
				break;
			}
			//update the blending factor for the next scale layer to make sure the sum weight is 1.0
			blendFactor = 1.0f - blendFactor;
		}
	}

	//in case the dummy bin is non-zero, the invalid region is used to fetch the texture.
	//The region texture fetching function will handle this case and output the alternative value.
	//This can ensure the output is normalised (sum of all weights is one) and not purely black when the region is not valid.
	const uint invalidRegionCount = RegionBin[SPLAT_REGION_COUNT];
	if(invalidRegionCount != 0u){
		const float weight = invalidRegionCount * Kr_2_inv;

#if ALBEDO != UNREGISTERED_TYPE
		TerrainTexture.TerrainColor = DEFAULT_ALBEDO * weight;
#endif
#if NORMAL != UNREGISTERED_TYPE
		TerrainTexture.TerrainNormal = DEFAULT_NORMAL * weight;
#endif
#if ROUGHNESS != UNREGISTERED_TYPE
		TerrainTexture.TerrainRoughness = DEFAULT_ROUGHNESS * weight;
#endif
#if AO != UNREGISTERED_TYPE
		TerrainTexture.TerrainAmbientOcclusion = DEFAULT_AO * weight;
#endif
	}

	return TerrainTexture;
}

vec3 calcTerrainNormal(){
	//calculate terrain normal from the heightfield
	//the uv increment for each pixel on the heightfield
	const vec2 unit_uv = 1.0f / vec2(textureSize(Heightmap, 0).xy);

	float cell[ConvolutionKernelOffset.length()];
	//convolve a 3x3 kernel with Sobel operator
	for(int a = 0; a < cell.length(); a++){
		const vec2 uv_offset = unit_uv * ConvolutionKernelOffset[a];
		cell[a] = textureLod(Heightmap, fs_in.texCoord + uv_offset, 0).r;
	}

	//apply filter
	return normalize(vec3(
		cell[0] + 2 * cell[3] + cell[5] - (cell[2] + 2 * cell[4] + cell[7]), 
		cell[0] + 2 * cell[1] + cell[2] - (cell[5] + 2 * cell[6] + cell[7]),
		1.0f / NormalStrength
	));
}

//n1 is the main normalmap, n2 adds details
vec3 blendNormal(vec3 n1, vec3 n2){
	//there exists many different algorithms for normal blending, pick one yourself

#if NORMALMAP_BLENDING == 0
	//Linear
	return normalize(n1 + n2);
#elif NORMALMAP_BLENDING == 1
	//Whiteout  
	return normalize(vec3(n1.xy + n2.xy, n1.z * n2.z));
#elif NORMALMAP_BLENDING == 2
	//Partial Derivative
	return normalize(vec3(n1.xy * n2.z + n2.xy * n1.z, n1.z * n2.z));
#elif NORMALMAP_BLENDING == 3
	//Unreal Developer Network
	return normalize(vec3(n1.xy + n2.xy, n1.z));
#elif NORMALMAP_BLENDING == 4
	//Basis Transform
	const mat3 TBN = mat3(
		vec3(n1.z, n1.y, -n1.x),
		vec3(n1.x, n1.z, -n1.y),
		vec3(n1.x, n1.y, n1.z)
	);
	return normalize(TBN * n2);
#elif NORMALMAP_BLENDING == 5
	//Reoriented Normal Mapping
	n1.z += 1.0f;
	n2 *= vec3(vec2(-1.0f), 1.0f);
	return normalize(n1 * dot(n1, n2) / n1.z - n2);
#else
	//no blending
	return n1;
#endif
}