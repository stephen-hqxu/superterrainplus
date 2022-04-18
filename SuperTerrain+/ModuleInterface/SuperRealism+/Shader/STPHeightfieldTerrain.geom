#version 460 core
#extension GL_ARB_shading_language_include : require
#extension GL_NV_shader_buffer_load : require

#define SHADER_PREDEFINE_GEOM
#include </Common/STPSeparableShaderPredefine.glsl>
#include </Common/STPLightSpaceInformation.glsl>

//geometry shader is only used for terrain shadow pass
//Define how many times the GS should run for this shadow pass
#define HEIGHTFIELD_SHADOW_PASS_INVOCATION -1

layout (triangles, invocations = HEIGHTFIELD_SHADOW_PASS_INVOCATION) in;
layout (triangle_strip, max_vertices = 3) out;

//Input, which is useless. Keep it here just for interface matching
in VertexTES{
	vec3 position_world;
	vec2 texCoord;
} gs_in[];

void main(){
	for(int i = 0; i < gl_in.length; i++){
		//output light information
		gl_Position = LightSpace.ProjectionView[gl_InvocationID] * gl_in[i].gl_Position;
		gl_Layer = gl_InvocationID;

		EmitVertex();
	}
	EndPrimitive();
}