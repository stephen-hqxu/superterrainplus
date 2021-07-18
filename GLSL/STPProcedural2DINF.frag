#version 460 core
#extension GL_ARB_bindless_texture : require

//Regional Texture
struct STPTextureLayer{
	float UpperBound;
	unsigned int Index;
};
struct STPTextureGradient{
	float minGradient;
	float maxGradient;	
	float LowerBound;
	float UpperBound;
	unsigned int Index;
};
layout (std430, binding = 1) buffer STPLayers{
	layout (offset = 0) STPTextureLayer layer[];
};
layout (std430, binding = 2) buffer STPGradients{
	layout (offset = 0) STPTextureGradient gradient[];
};
layout (std430, binding = 3) buffer STPTextureRegion{
	//each texture in the samplerarray is one layer.
	//each bindless texture handle represents one type of texture (color, normal, displacement, etc...)
	sampler2DArray TextureRegion[];
};

//Input
in VertexGS{
	vec4 position_world;
	vec4 position_clip;
	vec2 texCoord;
} fs_in;
//Output
layout (location = 0) out vec4 FragColor;

//Biomemap
layout (binding = 0) uniform usampler2D Biomemap;
//Heightfield, RGB is normalmap, A is heightmap
layout (binding = 1) uniform sampler2D Heightfield;

void main(){
	//for demo to test if everything works, we display the normal map for now
	FragColor = vec4(texture(Heightfield, fs_in.texCoord).rgb, 1.0f);
}