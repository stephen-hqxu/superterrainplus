#version 460 core
#extension GL_ARB_shading_language_include : require

#define SHADER_PREDEFINE_TESE
#include </Common/STPSeparableShaderPredefine.glsl>
#include </Common/STPCameraInformation.glsl>

//patches input
layout (triangles, fractional_odd_spacing, ccw) in;

//Define to true to disable `out` to the next shader stage.
//This is useful for terrain shadow rendering.
#define HEIGHTFIELD_TESE_NO_OUTPUT 0
#define STP_WATER 0

//Input
in VertexTCS{
	vec2 texCoord;
} tes_in[];

//Output
#if !HEIGHTFIELD_TESE_NO_OUTPUT
out VertexTES{
	vec3 position_world;
	vec2 texCoord;
} tes_out;
#endif

#if STP_WATER
#include </Common/STPAnimatedWave.glsl>

uniform WaveFunction WaterWave;
uniform unsigned int WaveGeometryIteration;
uniform float WaveTime;
#else
layout (binding = 0) uniform sampler2D Heightfield;

uniform float Altitude;
#endif//STP_WATER

//Functions
vec2 toCartesian2D(vec2, vec2, vec2);
vec4 toCartesian4D(vec4, vec4, vec4);

void main(){
	//interpolate Barycentric to Cartesian
	gl_Position = toCartesian4D(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);
	const vec2 uvCoord = toCartesian2D(tes_in[0].texCoord, tes_in[1].texCoord, tes_in[2].texCoord);

#if STP_WATER
	//procedural water animation
	//For wave position we can either use xz world coordinate or texture coordinate.
	//I think texture coordinate is better because the range of it is fixed regardless of the size of the chunk.
	gl_Position.y += waveHeight(uvCoord, WaterWave, WaveGeometryIteration, WaveTime);
#else
	//our plane is always pointing upwards
	//displace the terrain, moving the vertices upward
	gl_Position.y += textureLod(Heightfield, uvCoord, 0).r * Altitude;
#endif
	
	//space conversion
#if !HEIGHTFIELD_TESE_NO_OUTPUT
	tes_out.position_world = gl_Position.xyz;
	tes_out.texCoord = uvCoord;
#endif
	gl_Position = Camera.ProjectionView * gl_Position;
}

vec2 toCartesian2D(vec2 v1, vec2 v2, vec2 v3){
	return vec2(gl_TessCoord.x) * v1 + vec2(gl_TessCoord.y) * v2 + vec2(gl_TessCoord.z) * v3;
}

vec4 toCartesian4D(vec4 v1, vec4 v2, vec4 v3){
	return vec4(gl_TessCoord.x) * v1 + vec4(gl_TessCoord.y) * v2 + vec4(gl_TessCoord.z) * v3;
}