#version 460 core
#extension GL_ARB_bindless_texture : require

//patches output
layout (vertices = 3) out;

struct TessLevel{
	float MAX_TESS_LEVEL;
	float MIN_TESS_LEVEL;
	float FURTHEST_TESS_DISTANCE;
	float NEAREST_TESS_DISTANCE;
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
	vec3 tangent;
	vec3 bitangent;
	flat unsigned int chunkID;//chunkID defines the index of heightmap in "Heightmap" sampler array
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
	vec3 tangent;
	vec3 bitangent;
	flat unsigned int chunkID;
} tcs_out[];

//Uniforms
uniform vec3 cameraPos;
uniform float altitude;
uniform TessLevel tessParameters;
//control how far the mesh starts to decrease its LoD, (0, inf), in classic hermite interpolation, this factor will be 8.0f
//2.0 is the default value, mesh will half its original LoD at 50% of tessllation distance
uniform float shiftFactor;

//Heightfield, RGB is normalmap, A is heightmap
layout (binding = 0) uniform sampler2DArray Heightfield;

//Functions
float[3] calcPatchDistance(vec3);
float getTessLevel(TessLevel, float, float);
float distanceFunction(float, float, float, float);

void main(){
	//copy pasting the input to output
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
	tcs_out[gl_InvocationID].texCoord = tcs_in[gl_InvocationID].texCoord;
	tcs_out[gl_InvocationID].normal = tcs_in[gl_InvocationID].normal;
	tcs_out[gl_InvocationID].tangent = tcs_in[gl_InvocationID].tangent;
	tcs_out[gl_InvocationID].bitangent = tcs_in[gl_InvocationID].bitangent;
	tcs_out[gl_InvocationID].chunkID = tcs_in[gl_InvocationID].chunkID;
	
	if(gl_InvocationID == 0){//tessllation settings are shared across all local invocations, so only need to set it once
		float[3] camera_terrain_distance = calcPatchDistance(cameraPos);

		gl_TessLevelOuter[0] = getTessLevel(tessParameters, camera_terrain_distance[1], camera_terrain_distance[2]);
		gl_TessLevelOuter[1] = getTessLevel(tessParameters, camera_terrain_distance[2], camera_terrain_distance[0]);
		gl_TessLevelOuter[2] = getTessLevel(tessParameters, camera_terrain_distance[0], camera_terrain_distance[1]);
		gl_TessLevelInner[0] = (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) / (3.0f * 4.0f);
	}
}

float[3] calcPatchDistance(vec3 origin){
	float[3] patch_distance;
	//calculate distance from origin to each vertex
	for(int i = 0; i < 3; i++){
		//calculate the vertex position on the actual terrain
		vec3 terrainVertexPos = gl_out[i].gl_Position.xyz + normalize(tcs_out[i].normal) * altitude * texture(Heightfield, vec3(tcs_out[i].texCoord, uintBitsToFloat(tcs_out[i].chunkID))).a;
		//calculate distance
		patch_distance[i] = distance(origin, terrainVertexPos);
	}
	//return
	return patch_distance;
}

float getTessLevel(TessLevel levelControl, float distance1, float distance2){
	//calculate the distance from camera to the center of the edge
	float distance_to_edge = (distance1 + distance2) * 0.5f;
	//clamp the distance between nearest to furthest
	distance_to_edge = clamp(distance_to_edge, levelControl.NEAREST_TESS_DISTANCE, levelControl.FURTHEST_TESS_DISTANCE);
	
	//calculate the tess level base on distance 
	return mix(levelControl.MAX_TESS_LEVEL, levelControl.MIN_TESS_LEVEL, 
	distanceFunction(levelControl.NEAREST_TESS_DISTANCE, levelControl.FURTHEST_TESS_DISTANCE, distance_to_edge, shiftFactor));
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