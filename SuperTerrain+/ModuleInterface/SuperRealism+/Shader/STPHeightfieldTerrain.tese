#version 460 core
#extension GL_ARB_shading_language_include : require
#extension GL_NV_gpu_shader5 : require

#define SHADER_PREDEFINE_TESE
#include </Common/STPSeparableShaderPredefine.glsl>
#include </Common/STPCameraInformation.glsl>

//patches input
layout(triangles, fractional_odd_spacing, ccw) in;

#define STP_WATER 0

//Input
#if STP_WATER
//The dummy shared variable from the control shader, declared here just for interface matching
patch in uint8_t WaterCulled[3];
#else
in VertexTCS{
	vec2 texCoord;
} tes_in[];
#endif

//Output
out VertexTES{
	vec3 position_world;
#if !STP_WATER
	vec2 texCoord;
#endif
} tes_out;

#if STP_WATER
#include </Common/STPAnimatedWave.glsl>

uniform float WaveHeight;
uniform WaveFunction WaterWave;
uniform uint WaveGeometryIteration;
uniform float WaveTime;
#else
layout(binding = 0) uniform sampler2D Heightfield;

uniform float Altitude;
uniform uint TerrainRenderPass;
#endif//STP_WATER

//Functions
vec2 toCartesian2D(const vec2, const vec2, const vec2);
vec4 toCartesian4D(const vec4, const vec4, const vec4);

void main(){
	//interpolate Barycentric to Cartesian
	gl_Position = toCartesian4D(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);

#if STP_WATER
	//procedural water animation
	//For wave position we can either use xz world coordinate or texture coordinate.
	//I think world position is better because the UV is normalised to [0, 1] and might be compressed or stretched if the chunk size changes.
	gl_Position.y += getWaveHeight(gl_Position.xz, WaterWave, WaveGeometryIteration, WaveTime) * WaveHeight;
#else
	tes_out.texCoord = toCartesian2D(tes_in[0].texCoord, tes_in[1].texCoord, tes_in[2].texCoord);
	//our plane is always pointing upwards
	//displace the terrain, moving the vertices upward
	gl_Position.y += textureLod(Heightfield, tes_out.texCoord, 0.0f).r * Altitude;
#endif
	
	tes_out.position_world = gl_Position.xyz;
#if !STP_WATER
	if(TerrainRenderPass == 0u)
#endif
	{
		//For regular rendering there is no geometry shader so we apply the final transformation here.
		//For depth rendering, we need geometry for instanced rendering and transformation will be applied later.
		gl_Position = Camera.ProjectionView * gl_Position;
	}
}

vec2 toCartesian2D(const vec2 v1, const vec2 v2, const vec2 v3){
	return vec2(gl_TessCoord.x) * v1 + vec2(gl_TessCoord.y) * v2 + vec2(gl_TessCoord.z) * v3;
}

vec4 toCartesian4D(const vec4 v1, const vec4 v2, const vec4 v3){
	return vec4(gl_TessCoord.x) * v1 + vec4(gl_TessCoord.y) * v2 + vec4(gl_TessCoord.z) * v3;
}