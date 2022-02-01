#version 460 core
#extension GL_ARB_shading_language_include : require

/* -------------------------------- output setting -------------------------------- */
//Define to 0 to disable output to fragment shader, this is useful for shadow mapping.
#define HEIGHTFIELD_SHADOW_PASS 0
//Define how many times the GS should run for this shadow pass
#define HEIGHTFIELD_SHADOW_PASS_INVOCATION -1
/* -------------------------------------------------------------------------------- */

#if HEIGHTFIELD_SHADOW_PASS
layout (triangles, invocations = HEIGHTFIELD_SHADOW_PASS_INVOCATION) in;
#else
layout (triangles) in;
#endif
layout (triangle_strip, max_vertices = 3) out;

#if HEIGHTFIELD_SHADOW_PASS
#include </Common/STPLightSpaceInformation.glsl>
#else
#include </Common/STPCameraInformation.glsl>
#endif

//Input
in gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_in[];
in VertexTES{
	vec2 texCoord;
} gs_in[];
//Output
out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};
#if !HEIGHTFIELD_SHADOW_PASS
out VertexGS{
	vec3 position_world;
	vec2 texCoord;
} gs_out;
#endif

void main(){
	for(int i = 0; i < gl_in.length; i++){
#if HEIGHTFIELD_SHADOW_PASS
		//output light information
		gl_Position = LightSpace.ProjectionView[LightSpace.CurrentLightSpaceStart + gl_InvocationID] * gl_in[i].gl_Position;
		gl_Layer = gl_InvocationID;
#else
		//output the primitive information
		gl_Position = gl_in[i].gl_Position;
		gs_out.position_world = gl_Position.xyz;
		gl_Position = Camera.ProjectionView * gl_Position;
		gs_out.texCoord = gs_in[i].texCoord;
#endif
		EmitVertex();
	}
	EndPrimitive();
}