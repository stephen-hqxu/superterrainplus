#version 460 core

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

//A temporary color array for visualising regions
const vec3 RegionColor[6] = {
	vec3(0.568f, 0.204f, 0.176f),
	vec3(0.298f, 0.78f, 0.384f),
	vec3(0.215f, 0.329f, 0.106f),
	vec3(0.859f, 0.753f, 0.267f),
	vec3(0.42f, 0.251f, 0.043f),
	vec3(0.235f)
};

void main(){
	//for demo to test if everything works, we display the normal map for now
	const vec3 Normal = fs_in.normal.rgb;

	//visualise region
	const uint Region = texture(Splatmap, fs_in.texCoord).r;
	//region is valid, can be visualised, otherwise we display the normal (just for fun)
	FragColor = (Region < RegionColor.length()) ? vec4(RegionColor[Region], 1.0f) : vec4(Normal, 1.0f) ;
}