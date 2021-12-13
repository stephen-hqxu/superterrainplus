#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require

/* ------------------ Texture Splatting ---------------------------- */
//Macros are define by the main application, having an arbitary number just to make the compiler happy.
#define REGION_COUNT 1
#define REGISTRY_COUNT 1
#define REGISTRY_DICTIONARY_COUNT 1

//texture type indexing
#define ALBEDO 1
#define UNUSED_TYPE 1
#define UNREGISTERED_TYPE 1

//Rule-based texturing system
layout (bindless_sampler) uniform sampler2DArray RegionTexture[REGION_COUNT];
//x: array index, y: layer index
uniform uvec2 RegionRegistry[REGISTRY_COUNT];
uniform uint RegistryDictionary[REGISTRY_DICTIONARY_COUNT];

/* ----------------------------------------------------------------- */

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

vec3 calcTerrainNormal();
//The functions below are used to transformed terrain normal->terrain texture normal, given vertex position, uv and normal
mat2x3 calcTangentBitangent(vec3[3], vec2[3]);
mat3 calcTerrainTBN(mat2x3, vec3);

void main(){
	//for demo to test if everything works, we display the normal map for now
	const vec3 Normal = calcTerrainNormal();

	FragColor = vec4(getRegionTexture(fs_in.texCoord, Normal), 1.0f);
}

vec3 getRegionTexture(vec2 uv, vec3 replacement){
#if ALBEDO == UNREGISTERED_TYPE
	return replacement;
#else
	const uint region = texture(Splatmap, uv).r;
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

	//this formula make sure the UV is stable when the rendered chunk shifts
	const vec2 world_uv = uv * RenderedChunk + ChunkOffset;
	const vec3 terrainColor = texture(RegionTexture[textureLoc.x], vec3(world_uv * 10.0f, textureLoc.y)).rgb;

	//region is valid, can be visualised, otherwise we display a replacement color
	return terrainColor;
#endif
}

/*
	We have three normal systems for terrain, plane normal, terrain normal and terrain texture normal.
	- plane normal is simply (0,1,0), that's how our model is defined (model space normal)
	- terrain normal is the vector that perpendicular to the tessellated terrain
	- terrain texture normal is the normal comes along with the texture on the terrain later in the fragment shader,
		each texture has its own dedicated normal map

	So we need to do the TBN transform twice: plane->terrain normal then terrain normal->terrain texture normal
*/
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

mat2x3 calcTangentBitangent(vec3[3] position, vec2[3] uv){
	//edge and delta uv
	const vec3 edge0 = position[1] - position[0], edge1 = position[2] - position[0];
	const vec2 deltauv0 = uv[1] - uv[0], deltauv1 = uv[2] - uv[0];
	//mat(column)x(row). calculate tangent and bitangent
	//since glsl matrix is column major we need to do a lot of transpose
	return transpose(inverse(transpose(mat2(deltauv0, deltauv1))) * transpose(mat2x3(edge0, edge1)));
}

mat3 calcTerrainTBN(mat2x3 tangent_bitangent, vec3 normal){
	//calculate TBN matrix for 3 vertices, tangent space to world space
	const vec3 normal_normalised = normalize(normal), tangent_normalised = normalize(tangent_bitangent[0]);
	return mat3(
		//re-orthgonalise the tangent
		normalize(tangent_normalised - dot(tangent_normalised, normal_normalised) * normal_normalised),
		normalize(tangent_bitangent[1]),
		normal_normalised
	);
}