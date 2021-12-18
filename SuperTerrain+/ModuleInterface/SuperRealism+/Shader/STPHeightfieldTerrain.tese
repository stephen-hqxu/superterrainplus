#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require

//patches input
layout (triangles, fractional_odd_spacing, cw) in;

//Input
in gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_in[gl_MaxPatchVertices];
in VertexTCS{
	vec2 texCoord;
	vec3 normal;
} tes_in[];
//Output
out gl_PerVertex {
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
};
out VertexTES{
	vec2 texCoord;
	vec3 normal;
} tes_out;

//Uniforms
uniform float Altitude;

layout (binding = 1) uniform sampler2D Heightfield;

//Functions
vec2 toCartesian2D(vec2, vec2, vec2);
vec3 toCartesian3D(vec3, vec3, vec3);
vec4 toCartesian4D(vec4, vec4, vec4);

void main(){
	//interpolate barycentric to cartesian
	vec4 terrain_vertices = toCartesian4D(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);
	tes_out.texCoord = toCartesian2D(tes_in[0].texCoord, tes_in[1].texCoord, tes_in[2].texCoord);
	tes_out.normal = toCartesian3D(tes_in[0].normal, tes_in[1].normal, tes_in[2].normal);

	//displace the terrain, moving the vertices upward
	terrain_vertices.xyz += normalize(tes_out.normal) * texture(Heightfield, tes_out.texCoord).r * Altitude;
	gl_Position = terrain_vertices;
}


vec2 toCartesian2D(vec2 v1, vec2 v2, vec2 v3){
	return vec2(gl_TessCoord.x) * v1 + vec2(gl_TessCoord.y) * v2 + vec2(gl_TessCoord.z) * v3;
}

vec3 toCartesian3D(vec3 v1, vec3 v2, vec3 v3){
	return vec3(gl_TessCoord.x) * v1 + vec3(gl_TessCoord.y) * v2 + vec3(gl_TessCoord.z) * v3;
}

vec4 toCartesian4D(vec4 v1, vec4 v2, vec4 v3){
	return vec4(gl_TessCoord.x) * v1 + vec4(gl_TessCoord.y) * v2 + vec4(gl_TessCoord.z) * v3;
}