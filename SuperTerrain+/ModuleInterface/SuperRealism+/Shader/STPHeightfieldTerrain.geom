#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

#include </Common/STPCameraInformation.glsl>

//Input
in gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_in[];
in VertexTES{
	vec2 texCoord;
	vec3 normal;
	vec3 tangent;
	vec3 bitangent;
} gs_in[];
//Output
out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};
out VertexGS{
	vec4 position_world;
	vec4 position_clip;
	vec2 texCoord;
	vec3 normal;
} gs_out;

//Functions
void emitFace(int);

void main(){
	//no layer rendering by now, I will add that later
	emitFace(0);
}

void emitFace(int layer){
	for(int i = 0; i < gl_in.length; i++){
		//output the primitive information
		gl_Position = gl_in[i].gl_Position;
		gs_out.position_world = gl_Position;
		gl_Position = CameraProjection * CameraView * gl_Position;
		gs_out.position_clip = gl_Position;
		gs_out.texCoord = gs_in[i].texCoord;
		gs_out.normal = gs_in[i].normal;//this normal is the original flat plane normal not the terrain heightfield
		gl_Layer = layer;

		EmitVertex();
	}
	EndPrimitive();
}