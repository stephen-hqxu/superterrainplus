#version 460 core
#extension GL_ARB_shading_language_include : require

//patches output
layout (vertices = 3) out;

#include </Common/STPCameraInformation.glsl>

struct TessellationSetting{
	float MaxLod;
	float MinLod;
	float MaxDis;
	float MinDis;
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
uniform TessellationSetting Tess;

float calcLoD(float, float);

void main(){
	//tessllation settings are shared across all local invocations, so only need to set it once
	if(gl_InvocationID == 0){
		float vertexDistance[3];
		//first calculate the distance from camera to each vertex in a patch
		for(int i = 0; i < 3; i++){
			//override the altitude of view position and vertex position
			//to make sure they are at the same height and will not be affected by displacement of vertices later.
			const vec2 vertexPos = gl_in[i].gl_Position.xz,
				viewPos = CameraPosition.xz;

			//perform linear interpolation to the distance
			vertexDistance[i] = clamp((distance(vertexPos, viewPos) - Tess.MinDis) / (Tess.MaxDis - Tess.MinDis), 0.0f, 1.0f);
		}

		gl_TessLevelOuter[0] = calcLoD(vertexDistance[1], vertexDistance[2]);
		gl_TessLevelOuter[1] = calcLoD(vertexDistance[2], vertexDistance[0]);
		gl_TessLevelOuter[2] = calcLoD(vertexDistance[0], vertexDistance[1]);
		gl_TessLevelInner[0] = (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) / 3.0f;
	}
	
	//copy pasting the input to output
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
	tcs_out[gl_InvocationID].texCoord = tcs_in[gl_InvocationID].texCoord;
	tcs_out[gl_InvocationID].normal = tcs_in[gl_InvocationID].normal;
}

float calcLoD(float v1, float v2){
	return mix(Tess.MaxLod, Tess.MinLod, (v1 + v2) * 0.5f);
}