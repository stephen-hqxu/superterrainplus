#version 460 core
#extension GL_ARB_shading_language_include : require
#extension GL_ARB_bindless_texture : require

#define SHADER_PREDEFINE_TESC
#include </Common/STPSeparableShaderPredefine.glsl>
#include </Common/STPCameraInformation.glsl>

#define TWO_PI 6.283185307179586476925286766559

//patches output
layout (vertices = 3) out;

//Water plane tessellation shader shares a majority part with the terrain shader.
//Select to use the water shader routine.
#define STP_WATER 0

struct TessellationSetting{
	float MaxLod;
	float MinLod;
	float MaxDis;
	float MinDis;
};

//Input
in VertexVS{
	vec2 texCoord;
} tcs_in[];

//Output
#if !STP_WATER
out VertexTCS{
	vec2 texCoord;
} tcs_out[];
#endif

#if STP_WATER
//The texture should remain the same as the terrain on which the water is rendered.
layout (binding = 0) uniform usampler2D Biomemap;
//Water level acts as a lookup table, given biomeID as index, obtain the height of the water.
//This lookup table should use nearest filtering and clamp to border.
layout (bindless_sampler) uniform sampler1D WaterLevel;

uniform TessellationSetting WaterTess;
uniform float MinLevel;

//Water plane culling test
uniform unsigned int SampleCount;
uniform float SampleRadiusMul;
uniform float Altitude;
#else
//There are two terrain tessellation settings, one for regular rendering, the other low quality one is for depth rendering.
uniform TessellationSetting Tess[2];
//0: regular rendering; 1: depth pass
uniform unsigned int TerrainRenderPass;
#endif//STP_WATER

void tessellatePlane(TessellationSetting);
//Calculate the level-of-detail for the mesh
float calcLoD(TessellationSetting, float, float);
#if STP_WATER
//get the water level at the current texture coordinate
float waterLevelAt(vec2);
#endif

void main(){
	if(gl_InvocationID == 0){
#if STP_WATER
		//perform water plane culling based on biomemap
		//The culling algorithm essentially allows water rendered at water biome only, to reduce computational cost.
		//We can simply test if all vertices on the patch are in water biome.
		//To mitigate water leakage due to biome edge smoothing on the heightmap, cull water only when the patch is very far away from the biome.
		//We only need to test the outside of the patch, for biome smaller than the patch we can safely ignore (thus cull) it.

		//we use the centroid instead of circumcentre of the triangle because centroid is always inside the shape
		//first find the centroid of the triangle
		vec2 triCentriod = vec2(0.0f);
		for(int i = 0; i < 3; i++){
			triCentriod += tcs_in[i].texCoord;
		}
		triCentriod /= 3.0f;

		bool cullWater = true;
		//draw a circle at the centroid, sample at the circumference
		const float angleInc = TWO_PI / float(SampleCount);
		float currentAngle = 0.0f;
		for(uint i = 0u; i < SampleCount; i++){
			const vec2 samplePos = SampleRadiusMul * vec2(cos(currentAngle), sin(currentAngle)) + triCentriod;
			//grab the water level and test for validity of water level
			if(waterLevelAt(samplePos) >= MinLevel){
				//if we found any sample goes above min level, water plane should not be culled
				cullWater = false;
				break;
			}

			currentAngle += angleInc;
		}

		if(cullWater){
			//set any active outer level to a negative can cull the entire patch
			gl_TessLevelOuter[0] = -1.0f;
			return;
		}
		tessellatePlane(WaterTess);
#else
		//determine which tessellation setting to use for the terrain shader
		tessellatePlane(Tess[TerrainRenderPass]);
#endif
	}
	
	//copy pasting the input to output
#if STP_WATER
	//We can displace the water plane to the water level before tessellation because the displacement is constant per biome,
	//so we can save some memory bandwidth.
	vec4 position_world = gl_in[gl_InvocationID].gl_Position;
	position_world.y += waterLevelAt(tcs_in[gl_InvocationID].texCoord) * Altitude;

	gl_out[gl_InvocationID].gl_Position = position_world;
#else
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
	tcs_out[gl_InvocationID].texCoord = tcs_in[gl_InvocationID].texCoord;
#endif
}

void tessellatePlane(TessellationSetting tess){
	float vertexDistance[3];
	//first calculate the distance from camera to each vertex in a patch
	for(int i = 0; i < 3; i++){
		//override the altitude of view position and vertex position
		//to make sure they are at the same height and will not be affected by displacement of vertices later.
		const vec2 vertexPos = gl_in[i].gl_Position.xz,
			viewPos = Camera.Position.xz;

		//perform linear interpolation to the distance
		vertexDistance[i] = clamp((distance(vertexPos, viewPos) - tess.MinDis) / (tess.MaxDis - tess.MinDis), 0.0f, 1.0f);
	}

	gl_TessLevelOuter[0] = calcLoD(tess, vertexDistance[1], vertexDistance[2]);
	gl_TessLevelOuter[1] = calcLoD(tess, vertexDistance[2], vertexDistance[0]);
	gl_TessLevelOuter[2] = calcLoD(tess, vertexDistance[0], vertexDistance[1]);
	gl_TessLevelInner[0] = (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) / 3.0f;
}

float calcLoD(TessellationSetting tess, float v1, float v2){
	return mix(tess.MaxLod, tess.MinLod, (v1 + v2) * 0.5f);
}

#if STP_WATER
float waterLevelAt(vec2 uv){
	const uint biome = textureLod(Biomemap, uv, 0).r;
	return textureLod(WaterLevel, 1.0f * biome / (1.0f * textureSize(WaterLevel, 0)), 0).r;
}
#endif