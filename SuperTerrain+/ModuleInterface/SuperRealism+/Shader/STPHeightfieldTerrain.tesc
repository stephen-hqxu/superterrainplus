#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require

//patches output
layout (vertices = 3) out;

#include </Common/STPCameraInformation.glsl>

struct TessellationSetting{
	float MaxLod;
	float MinLod;
	float FurthestDistance;
	float NearestDistance;
	float ShiftFactor;
};

//Input
in gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_in[gl_MaxPatchVertices];
in VertexVS{
	vec2 texCoord;
	vec3 normal;
} tcs_in[];

//Output
out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
	float gl_ClipDistance[];
} gl_out[];
out VertexTCS{
	vec2 texCoord;
	vec3 normal;
} tcs_out[];

//Uniforms
uniform float Altitude;
uniform TessellationSetting TessSetting;

//Heightfield, R channel denotes the terrain height factor
layout (binding = 1) uniform sampler2D Heightfield;

//Functions
float[3] calcPatchDistance(vec3);
float getTessLevel(TessellationSetting, float, float);
float distanceFunction(float, float, float, float);

void main(){
	//copy pasting the input to output
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
	tcs_out[gl_InvocationID].texCoord = tcs_in[gl_InvocationID].texCoord;
	tcs_out[gl_InvocationID].normal = tcs_in[gl_InvocationID].normal;
	
	if(gl_InvocationID == 0){
		//tessllation settings are shared across all local invocations, so only need to set it once
		float[3] camera_terrain_distance = calcPatchDistance(CameraPosition);

		gl_TessLevelOuter[0] = getTessLevel(TessSetting, camera_terrain_distance[1], camera_terrain_distance[2]);
		gl_TessLevelOuter[1] = getTessLevel(TessSetting, camera_terrain_distance[2], camera_terrain_distance[0]);
		gl_TessLevelOuter[2] = getTessLevel(TessSetting, camera_terrain_distance[0], camera_terrain_distance[1]);
		gl_TessLevelInner[0] = (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) / (3.0f * 4.0f);
	}
}

float[3] calcPatchDistance(vec3 origin){
	float[3] patch_distance;
	//calculate distance from origin to each vertex
	for(int i = 0; i < 3; i++){
		//calculate the vertex position on the actual terrain
		vec3 terrainVertexPos = gl_out[i].gl_Position.xyz + normalize(tcs_out[i].normal) * Altitude * texture(Heightfield, tcs_out[i].texCoord).r;
		//calculate distance
		patch_distance[i] = distance(origin, terrainVertexPos);
	}
	//return
	return patch_distance;
}

float getTessLevel(TessellationSetting levelControl, float distance1, float distance2){
	//calculate the distance from camera to the center of the edge
	float distance_to_edge = (distance1 + distance2) * 0.5f;
	//clamp the distance between nearest to furthest
	distance_to_edge = clamp(distance_to_edge, levelControl.NearestDistance, levelControl.FurthestDistance);
	
	//calculate the tess level base on distance 
	return mix(levelControl.MaxLod, levelControl.MinLod, 
	distanceFunction(levelControl.NearestDistance, levelControl.FurthestDistance, distance_to_edge, levelControl.ShiftFactor));
}

//X must be clamped, the result is undefined if X is out of range
//return value will be [0,1]
float distanceFunction(float minX, float maxX, float X, float power){
	//this is just a function desinged by my own, copyright!
	//linearly clamp X between minX and maxX
	float gradient = 1.0f / (maxX - minX);
	float clampedX = gradient * X - gradient * minX;//clampedX will always be [0,1]
	//a modified hermite interpolation
	//the higher the power, the more the curve shift towards maxX
	//when power is 2.0, f(0.5)=0.5
	return clamp(-pow(pow(clampedX, power) - 1.0f, 2.0f) + 1, 0.0f, 1.0f);
}