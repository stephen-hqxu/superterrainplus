#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require

#define TWO_PI 6.283185307179586476925286766559

/* ------------------ Texture Splatting ---------------------------- */
//Macros are define by the main application, having an arbitary number just to make the compiler happy.
#define REGION_COUNT 1
#define REGISTRY_COUNT 1
#define REGISTRY_DICTIONARY_COUNT 1

//texture type indexing
#define ALBEDO 1
#define NORMAL 1
#define BUMP 1
#define SPECULAR 1
#define AO 1
#define EMISSIVE 1

#define TYPE_STRIDE 6
#define UNUSED_TYPE 0
#define UNREGISTERED_TYPE 0

struct TextureRegionSmoothSetting{
	unsigned int Kr;
	float Ks;
	float Ns;
};

struct TerrainTextureData{
#if ALBEDO != UNREGISTERED_TYPE
	vec3 TerrainColor;
#endif
#if NORMAL != UNREGISTERED_TYPE
	vec3 TerrainNormal;
#endif
#if BUMP != UNREGISTERED_TYPE
	float TerrainBump;
#endif
#if SPECULAR != UNREGISTERED_TYPE
	float TerrainSpecular;
#endif
#if AO != UNREGISTERED_TYPE
	float TerrainAmbientOcclusion;
#endif
};

//Rule-based texturing system
layout (bindless_sampler) uniform sampler2DArray RegionTexture[REGION_COUNT];
//x: array index, y: layer index
uniform uvec2 RegionRegistry[REGISTRY_COUNT];
uniform uint RegistryDictionary[REGISTRY_DICTIONARY_COUNT];

uniform TextureRegionSmoothSetting SmoothSetting;
uniform unsigned int UVScaleFactor;
/* -------------------------- Terrain Lighting ------------------------- */
struct LightColor{
	vec3 InDir, Dir, Ref;
};

struct LightSetting{
	float Ka, Kd, Ks;
	float Shin;
	LightColor Col;
};

#include </Common/STPCameraInformation.glsl>

uniform vec3 LightDirection;
uniform LightSetting Lighting;

//Normalmap blending algorithm
#define NORMALMAP_BLENDING 255

/* --------------------------------------------------------------------- */

//Input
in VertexGS{
	vec3 position_world;
	vec4 position_clip;
	vec2 texCoord;
} fs_in;
//Output
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform usampler2D Biomemap;
layout (binding = 1) uniform sampler2D Heightfield;
layout (binding = 2) uniform usampler2D Splatmap;
layout (binding = 3) uniform sampler3D Noisemap;

uniform uvec2 RenderedChunk;
uniform vec2 ChunkOffset;
//The strength of the z component on normalmap
uniform float NormalStrength;
uniform uvec2 HeightfieldResolution;

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

//Get region index
uint getRegion(vec2);
vec3 getRegionTexture(vec2, vec3, unsigned int, unsigned int);
//Get all types of available texture using texture region smoothing algorithm
TerrainTextureData getSmoothTexture(vec2);
//Calculate the UV scaling based on the size of texture
vec2 getUVScale(ivec2);
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
//Calculate light color for the current fragment position
vec3 calcLight(vec3, vec3, float, float);

void main(){
	const vec3 MeshNormal = calcTerrainNormal();
	//this function make sure the UV is stable when the rendered chunk shifts
	//here we need to use the UV of the current pixel (not the argument one)
	//so when we are doing smoothing over a convolution kernel the texture color remains the same for the same pixel.
	const vec2 worldUV = fs_in.texCoord * RenderedChunk + ChunkOffset;

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
		vec3(0.0f, 0.0f, -1.0f),
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

	//bumpmap processing
#if BUMP != UNREGISTERED_TYPE
	const float RenderingBump = TerrainTexture.TerrainBump;
#else
	const float RenderingBump = 0.0f;
#endif

	//specularmap processing
#if SPECULAR != UNREGISTERED_TYPE
	const float RenderingSpecular = TerrainTexture.TerrainSpecular;
#else
	const float RenderingSpecular = 1.0f;
#endif

	//ambient-occlusionmap processing
#if AO != UNREGISTERED_TYPE
	const float RenderingAO = TerrainTexture.TerrainAmbientOcclusion;
#else
	const float RenderingAO = 1.0f;
#endif
	
	//lighting pass
	const vec3 TerrainLight = calcLight(RenderingColor, RenderingNormal, RenderingSpecular, RenderingAO);

	//finally
	FragColor = vec4(TerrainLight, 1.0f);
}

uint getRegion(vec2 splatmap_uv){
	return texture(Splatmap, splatmap_uv).r;
}

vec3 getRegionTexture(vec2 world_uv, vec3 replacement, unsigned int region, unsigned int type){
	if(type == UNREGISTERED_TYPE || region >= REGISTRY_DICTIONARY_COUNT){
		//type not used
		//no region is defined
		return replacement;
	}

	const uint regionLoc = RegistryDictionary[region * TYPE_STRIDE + type];
	if(regionLoc >= REGISTRY_COUNT){
		//handle the case when texture type is not used
		return replacement;
	}
	const uvec2 textureLoc = RegionRegistry[regionLoc];

	const sampler2DArray selected_sampler = RegionTexture[textureLoc.x];
	//region is valid
	return texture(selected_sampler, vec3(world_uv * getUVScale(textureSize(selected_sampler, 0).xy), textureLoc.y)).rgb;
}

TerrainTextureData getSmoothTexture(vec2 world_uv){
	TerrainTextureData TerrainTexture;

	const float Kr_inv = 1.0f / (1.0f * SmoothSetting.Kr),
		Kr_2_inv = 1.0f / (1.0f * SmoothSetting.Kr * SmoothSetting.Kr);
	//perform smoothing to integer texture
	for(int i = 0; i < SmoothSetting.Kr; i++){
		for(int j = 0; j < SmoothSetting.Kr; j++){
			//we first divide the kernel matric into cells with equal spaces
			const vec2 domain = vec2(i, j) * Kr_inv,
				//then we jittered each sampling points from the cell centre
				//however we need to make sure samples are not jittered out of its cell
				stratified_domain = clamp(domain + Kr_inv * 
					texture(Noisemap, vec3(world_uv * SmoothSetting.Ns, (1.0f * i + j * SmoothSetting.Kr) * Kr_2_inv)).r, 0.0f, 1.0f);
			
			//then we map a squared domain into a disk domain.
			const float sq_domain_x = TWO_PI * stratified_domain.x;
			const vec2 disk_domain = sqrt(stratified_domain.y) * vec2(
				cos(sq_domain_x),
				sin(sq_domain_x)
			);
			
			//now apply the sampling points to the actual texture
			const vec2 uv_offset = SmoothSetting.Ks * (disk_domain * 2.0f - 1.0f) / HeightfieldResolution,
				sampling_uv = fs_in.texCoord + uv_offset;
			const uint region = getRegion(sampling_uv);

#if ALBEDO != UNREGISTERED_TYPE
			TerrainTexture.TerrainColor += getRegionTexture(world_uv, vec3(0.5f), region, ALBEDO);
#endif
#if NORMAL != UNREGISTERED_TYPE
			TerrainTexture.TerrainNormal += getRegionTexture(world_uv, vec3(vec2(0.5f), 1.0f), region, NORMAL);
#endif
#if BUMP != UNREGISTERED_TYPE
			TerrainTexture.TerrainBump += getRegionTexture(world_uv, vec3(0.0f), region, BUMP).r;
#endif
#if SPECULAR != UNREGISTERED_TYPE
			TerrainTexture.TerrainSpecular += getRegionTexture(world_uv, vec3(1.0f), region, SPECULAR).r;
#endif
#if AO != UNREGISTERED_TYPE
			TerrainTexture.TerrainAmbientOcclusion += getRegionTexture(world_uv, vec3(1.0f), region, AO).r;
#endif
		}
	}

	//average the final color
#if ALBEDO != UNREGISTERED_TYPE
	TerrainTexture.TerrainColor *= Kr_2_inv;
#endif
#if NORMAL != UNREGISTERED_TYPE
	TerrainTexture.TerrainNormal *= Kr_2_inv;
#endif
#if BUMP != UNREGISTERED_TYPE
	TerrainTexture.TerrainBump *= Kr_2_inv;
#endif
#if SPECULAR != UNREGISTERED_TYPE
	TerrainTexture.TerrainSpecular *= Kr_2_inv;
#endif
#if AO != UNREGISTERED_TYPE
	TerrainTexture.TerrainAmbientOcclusion *= Kr_2_inv;
#endif
	
	return TerrainTexture;
}

vec2 getUVScale(ivec2 texDim){
	return float(UVScaleFactor) / vec2(texDim);
}

vec3 calcTerrainNormal(){
	//calculate terrain normal from the heightfield
	//the uv increment for each pixel on the heightfield
	const vec2 unit_uv = 1.0f / vec2(HeightfieldResolution);

	float cell[ConvolutionKernelOffset.length()];
	//convolve a 3x3 kernel with Sobel operator
	for(int a = 0; a < cell.length(); a++){
		const vec2 uv_offset = unit_uv * ConvolutionKernelOffset[a];
		cell[a] = texture(Heightfield, fs_in.texCoord + uv_offset).r;
	}

	//apply filter
	return normalize(vec3(
		-(cell[0] + 2 * cell[3] + cell[5] - (cell[2] + 2 * cell[4] + cell[7])), 
		-(cell[0] + 2 * cell[1] + cell[2] - (cell[5] + 2 * cell[6] + cell[7])),
		1.0f / NormalStrength
	));
}

//n1 is the main normalmap, n2 adds details
vec3 blendNormal(vec3 n1, vec3 n2){
	//there exixts many different algorithms for normal blending, pick one yourself

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
	mat3 TBN = mat3(
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

vec3 calcLight(vec3 material, vec3 normal, float specular_strength, float ambient_strength){
	//get light color
	const LightColor color = Lighting.Col;

	//ambient
	const vec3 ambient = ambient_strength * Lighting.Ka * material * color.InDir,
		//diffuse
		lightDir = normalize(LightDirection),
		diffuse = Lighting.Kd * max(dot(lightDir, normal), 0.0f) * material * color.Dir,
		//specular
		viewDir = normalize(CameraPosition - fs_in.position_world),
		reflectDir = reflect(-lightDir, normal),
		halfwayDir = normalize(lightDir + viewDir),
		specular = specular_strength * Lighting.Ks * pow(max(dot(normal, halfwayDir), 0.0f), Lighting.Shin) * color.Ref;

	return ambient + diffuse + specular;
}