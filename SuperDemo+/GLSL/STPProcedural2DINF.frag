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

void main(){
	//for demo to test if everything works, we display the normal map for now
	FragColor = vec4(fs_in.normal, 1.0f);
}