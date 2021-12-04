#version 460 core
#extension ARB_bindless_texture: require

/* ------------------ Texture Splatting ---------------------------- */
#define TEXTURE_COUNT 1
#define TEXTURE_LOCATION_COUNT 6
#define TEXTURE_LOCATION_DICTIONARY_COUNT 6

//texture type indexing
uniform uint ALBEDO;

//An indentifier to shows this texture type is not used
uniform uint NullType;

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
//x: array index, y: layer index
uniform uvec2 RegionLocation[TEXTURE_LOCATION_COUNT];
uniform uint RegionLocationDictionary[TEXTURE_LOCATION_DICTIONARY_COUNT];

uniform uvec2 rendered_chunk_num;
uniform vec2 chunk_offset;

vec3 getRegionTexture(vec2, vec3);

void main(){
	//for demo to test if everything works, we display the normal map for now
	const vec3 Normal = fs_in.normal.rgb;

	FragColor = vec4(getRegionTexture(fs_in.texCoord, Normal), 1.0f);
}

vec3 getRegionTexture(vec2 uv, vec3 replacement){
	const uint region = texture(Splatmap, uv).r;
	if(region >= TEXTURE_LOCATION_DICTIONARY_COUNT){
		//no region is defined
		return replacement;
	}

	//find albedo texture for this region, currently the stride is 1
	const uint regionLoc = RegionLocationDictionary[region + ALBEDO];
	const uvec2 textureLoc = RegionLocation[regionLoc];
	if(textureLoc == NullType){
		//handle the case when texture type is not used
		return replacement;
	}

	//this formula make sure the UV is stable when the rendered chunk shifts
	const vec2 world_uv = uv * rendered_chunk_num + chunk_offset;
	const vec3 terrainColor = texture(RegionalTexture[textureLoc.x], vec3(world_uv * 10.0f, textureLoc.y)).rgb;

	//region is valid, can be visualised, otherwise we display a replacement color
	return terrainColor;
}