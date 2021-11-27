#version 460 core
#extension ARB_bindless_texture: require

/* ------------------ Texture Splatting ---------------------------- */
#define TEXTURE_COUNT 1
#define TEXTURE_LOCATION_COUNT 6
#define TEXTURE_LOLCATION_DICTIONARY_COUNT 18

struct TextureLocation{
	uint Group, Layer;
};

/* ----------------------------- ---------------------------- */

//Input
in VertexGS{
	vec4 position_world;
	vec4 position_clip;
	vec2 texCoord;
	vec3 normal;
} fs_in;
//Output
layout (location = 0) out vec4 FragColor;

layout (binding = 0) uniform usampler2D Biomemap;
layout (binding = 1) uniform sampler2D Heightfield;
layout (binding = 2) uniform usampler2D Splatmap;

//Rule-based texturing system
layout (bindless_sampler) uniform sampler2DArray RegionalTexture[TEXTURE_COUNT];
uniform TextureLocation RegionLocation[TEXTURE_LOCATION_COUNT];
uniform uint RegionLocationDictionary[TEXTURE_LOLCATION_DICTIONARY_COUNT];

//A temporary color array for visualising regions
const vec3 RegionColor[6] = {
	vec3(0.568f, 0.204f, 0.176f),
	vec3(0.298f, 0.78f, 0.384f),
	vec3(0.215f, 0.329f, 0.106f),
	vec3(0.859f, 0.753f, 0.267f),
	vec3(0.42f, 0.251f, 0.043f),
	vec3(0.235f)
};

vec3 getRegionTexture(vec2, vec3);

void main(){
	//for demo to test if everything works, we display the normal map for now
	const vec3 Normal = fs_in.normal.rgb;

	FragColor = vec4(getRegionTexture(fs_in.texCoord, Normal), 1.0f);
}

vec3 getRegionTexture(vec2 uv, vec3 replacement){
	//visualise region
	const uint Region = texture(Splatmap, uv).r;

	//region is valid, can be visualised, otherwise we display a replacement color
	return (Region < RegionColor.length()) ? RegionColor[Region] : replacement;
}