#version 460 core
#extension GL_ARB_shading_language_include : require

#define SHADER_PREDEFINE_VERT
#include </Common/STPSeparableShaderPredefine.glsl>

//Input
layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TexCoord;

//Output
out VertexVS{
	vec2 texCoord;
} vs_out;

uniform mat4 MeshModel;//The model matrix will be used to offset and scale unit planes globally

void main(){
	gl_Position = MeshModel * vec4(Position, 1.0f);
	vs_out.texCoord = TexCoord;
}