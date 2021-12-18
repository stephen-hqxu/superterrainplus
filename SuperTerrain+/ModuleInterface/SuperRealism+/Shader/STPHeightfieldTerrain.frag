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

#define UNUSED_TYPE 1
#define UNREGISTERED_TYPE 1

struct TextureRegionSmoothSetting{
	unsigned int Kr;
	float Ks;
	float Ns;
};

//Rule-based texturing system
layout (bindless_sampler) uniform sampler2DArray RegionTexture[REGION_COUNT];
//x: array index, y: layer index
uniform uvec2 RegionRegistry[REGISTRY_COUNT];
uniform uint RegistryDictionary[REGISTRY_DICTIONARY_COUNT];

uniform TextureRegionSmoothSetting SmoothSetting;
uniform unsigned int UVScaleFactor;

/* ----------------------------------------------------------------- */

//Input
in VertexGS{
	vec4 position_world;
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

vec3 getRegionTexture(vec2, vec3);
//Calculate the UV scaling based on the size of texture
vec2 getUVScale(ivec2);
//this function make sure the UV is stable when the rendered chunk shifts
//here we need to use the UV of the current pixel (not the argument one)
//so when we are doing smoothing over a convolution kernel the texture color remains the same for the same pixel.
vec2 getWorldUV();
/**
 * We have three normal systems for terrain, plane normal, terrain normal and terrain texture normal.
 * plane normal is simply (0,1,0), that's how our model is defined (model space normal)
 * terrain normal is the vector that perpendicular to the tessellated terrain
 * terrain texture normal is the normal comes along with the texture on the terrain later in the fragment shader,
 * each texture has its own dedicated normal map.
 * So we need to do the TBN transform twice: plane->terrain normal then terrain normal->terrain texture normal
*/
vec3 calcTerrainNormal();

void main(){
	//for demo to test if everything works, we display the normal map for now
	const vec3 Normal = calcTerrainNormal();

	vec3 TerrainColor = vec3(0.0f);
	const float Kr_inv = 1.0f / (1.0f * SmoothSetting.Kr),
		Kr_2_inv = 1.0f / (1.0f * SmoothSetting.Kr * SmoothSetting.Kr), 
		cellWidth = 1.0f / SmoothSetting.Kr;
	//perform smoothing to integer texture
	for(int i = 0; i < SmoothSetting.Kr; i++){
		for(int j = 0; j < SmoothSetting.Kr; j++){
			//we first divide the kernel matric into cells with equal spaces
			const vec2 domain = vec2(i, j) * Kr_inv,
				//then we jittered each sampling points from the cell centre
				//however we need to make sure samples are not jittered out of its cell
				stratified_domain = clamp(domain + cellWidth * 
					texture(Noisemap, vec3(getWorldUV() * SmoothSetting.Ns, (1.0f * i + j * SmoothSetting.Kr) * Kr_2_inv)).r, 0.0f, 1.0f);
			
			//then we map a squared domain into a disk domain.
			const float sq_domain_x = TWO_PI * stratified_domain.x;
			const vec2 disk_domain = sqrt(stratified_domain.y) * vec2(
				cos(sq_domain_x),
				sin(sq_domain_x)
			);
			
			//now apply the sampling points to the actual texture
			const vec2 uv_offset = SmoothSetting.Ks * (disk_domain * 2.0f - 1.0f) / HeightfieldResolution;
			TerrainColor += getRegionTexture(fs_in.texCoord + uv_offset, Normal);
		}
	}

	FragColor = vec4(TerrainColor * Kr_2_inv, 1.0f);
}

vec3 getRegionTexture(vec2 splatmap_uv, vec3 replacement){
#if ALBEDO == UNREGISTERED_TYPE
	return replacement;
#else
	const uint region = texture(Splatmap, splatmap_uv).r;
	if(region >= REGISTRY_DICTIONARY_COUNT){
		//no region is defined
		return replacement;
	}

	//find albedo texture for this region, currently the stride is 1
	const uint regionLoc = RegistryDictionary[region + ALBEDO];
	const uvec2 textureLoc = RegionRegistry[regionLoc];
	if(textureLoc == UNUSED_TYPE){
		//handle the case when texture type is not used
		return replacement;
	}

	const vec2 world_uv = getWorldUV();
	const sampler2DArray selected_sampler = RegionTexture[textureLoc.x];
	const vec3 terrainColor = texture(selected_sampler, vec3(world_uv * getUVScale(textureSize(selected_sampler, 0).xy), textureLoc.y)).rgb;

	//region is valid, can be visualised, otherwise we display a replacement color
	return terrainColor;
#endif
}

vec2 getUVScale(ivec2 texDim){
	return float(UVScaleFactor) / vec2(texDim);
}

vec2 getWorldUV(){
	return fs_in.texCoord * RenderedChunk + ChunkOffset;;
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
	const vec3 TerrainNormal = normalize(vec3(
		cell[0] + 2 * cell[3] + cell[5] - (cell[2] + 2 * cell[4] + cell[7]), 
		cell[0] + 2 * cell[1] + cell[2] - (cell[5] + 2 * cell[6] + cell[7]),
		1.0f / NormalStrength
	));
	//transfer the range from [-1,1] to [0,1]
	return (clamp(TerrainNormal, -1.0f, 1.0f) + 1.0f) / 2.0f;
}