#version 460 core
#extension ARB_bindless_texture: require

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

//PVM transformation
layout (std430, binding = 0) buffer PVmatrix{
	layout (offset = 0) mat4 View;
	layout (offset = 64) mat4 View_notrans;//removed translations of view matrix
	layout (offset = 128) mat4 Projection;//each float uses 4 bytes and mat4 has 16 of them
};

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

layout (binding = 1) uniform sampler2D Heightfield;

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
		gl_Position = Projection * View * gl_Position;
		gs_out.position_clip = gl_Position;
		gs_out.texCoord = gs_in[i].texCoord;
		gs_out.normal = gs_in[i].normal;//this normal is the original flat plane normal not the terrain heightfield
		gl_Layer = layer;

		EmitVertex();
	}
	EndPrimitive();
}