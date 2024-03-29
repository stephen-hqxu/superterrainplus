#version 460 core
#extension GL_ARB_shading_language_include : require
#extension GL_ARB_bindless_texture : require
#extension GL_NV_gpu_shader5 : require

#define SHADER_PREDEFINE_TESC
#include </Common/STPSeparableShaderPredefine.glsl>
#include </Common/STPCameraInformation.glsl>

#define TWO_PI 6.283185307179586476925286766559

//patches output
layout(vertices = 3) out;

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
#if STP_WATER
//A dummy output acting as a shared variable among all invocations
patch out uint8_t WaterCulled[3];
#else
out VertexTCS{
	vec2 texCoord;
} tcs_out[];
#endif

#if STP_WATER
//The texture should remain the same as the terrain on which the water is rendered.
layout(binding = 0) uniform usampler2D Biomemap;
layout(binding = 1) uniform sampler2D Heightmap;
//Water level acts as a lookup table, given biomeID as index, obtain the height of the water.
//This lookup table should use nearest filtering and clamp to border.
layout(bindless_sampler) uniform sampler1D WaterLevel;

uniform TessellationSetting WaterTess;
uniform float MinLevel;

//Water plane culling test
uniform uint SampleCount;
uniform float SampleRadiusMul;
uniform float Altitude;
#else
//There are two terrain tessellation settings, one for regular rendering, the other low quality one is for depth rendering.
uniform TessellationSetting Tess[2];
//0: regular rendering; 1: depth pass
uniform uint TerrainRenderPass;
#endif//STP_WATER

//Calculate the level-of-detail for the mesh
float calcLoD(const TessellationSetting, const float, const float);

const uvec2 PatchEdgeIdx[3] = {
	{ 1u, 2u },
	{ 2u, 0u },
	{ 0u, 1u }
};

void main(){
	//copy pasting the input to output
	gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
#if !STP_WATER
	tcs_out[gl_InvocationID].texCoord = tcs_in[gl_InvocationID].texCoord;
#endif

#if STP_WATER
	{
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

		//calculate the starting and ending sample for the current invocation
		const uint invocSampleCount = SampleCount / 3u,
			invocStart = gl_InvocationID * invocSampleCount,
			//make sure the last invocation grabs all the remaining samples
			invocEnd = gl_InvocationID == 2u ? SampleCount : (gl_InvocationID + 1u) * invocSampleCount;
		//find the water level for the current patch
		float patchLevel = MinLevel;
		WaterCulled[gl_InvocationID] = uint8_t(0xFFu);
		//draw a circle at the centroid, sample at the circumference
		const float angleInc = TWO_PI / float(SampleCount);
		for(uint i = invocStart; i < invocEnd; i++){
			const float currentAngle = angleInc * i;
			const vec2 samplePos = SampleRadiusMul * vec2(cos(currentAngle), sin(currentAngle)) + triCentriod;

			//get the water level at the current texture coordinate
			const uint biome = textureLod(Biomemap, samplePos, 0.0f).r;
			const float sampleLevel = textureLod(WaterLevel, 1.0f * biome / (1.0f * textureSize(WaterLevel, 0)), 0.0f).r;

			//find the max water level among all samples
			//Here we assume all samples only lie within a single watery biome, or alternatively,
			//no two watery biomes are closed to each other, otherwise it will create water leakage.
			//TODO: develop about a better water plane placement algorithm in the future to handle this case.
			patchLevel = max(patchLevel, sampleLevel);
			//if we found any sample goes above min level or the terrain, water plane should not be culled
			WaterCulled[gl_InvocationID] &= uint8_t(sampleLevel < max(MinLevel, textureLod(Heightmap, samplePos, 0.0f).r));
		}

		//Invocation communication, find out if the water plane should be culled, 
		//only when all invocations say it should be culled.
		barrier();
		bvec3 cullPatch;
		for(int i = 0; i < 3; i++){
			cullPatch[i] = bool(WaterCulled[i]);
		}
		if(all(cullPatch)){
			gl_TessLevelOuter[gl_InvocationID] = -1.0f;
			return;
		}

		//Don't use thread ballot function, because invocation does not necessary match up SIMD group.
		//It is stupid that OpenGL does not provide shared variable in tessellation shader;
		//exploit to use the patch output as a shared variable, and exchange the water plane height.
		gl_TessLevelOuter[gl_InvocationID] = patchLevel;
		barrier();
		//now grab the water level from all invocations
		float commonLevel = MinLevel;
		for(int i = 0; i < 3; i++){
			commonLevel = max(commonLevel, gl_TessLevelOuter[i]);
		}
		//We can displace the water plane to the water level before tessellation because the displacement is constant per biome,
		//so we can save some memory bandwidth.
		gl_out[gl_InvocationID].gl_Position.y += commonLevel * Altitude;
	}

	const TessellationSetting lodControl = WaterTess;
#else
	const TessellationSetting lodControl = Tess[TerrainRenderPass];
#endif
	{
		//compute tessellation level in parallel
		const uvec2 currentEdge = PatchEdgeIdx[gl_InvocationID];
		float vertexDistance[2];
		//first calculate the distance from camera to each vertex in a patch
		for(int i = 0; i < vertexDistance.length; i++){
			//override the altitude of view position and vertex position
			//to make sure they are at the same height and will not be affected by displacement of vertices later.
			const vec2 vertexPos = gl_in[currentEdge[i]].gl_Position.xz,
				viewPos = Camera.Position.xz;

			//perform linear interpolation to the distance
			vertexDistance[i] = clamp((Camera.InvFar * distance(vertexPos, viewPos) - lodControl.MinDis) / (lodControl.MaxDis - lodControl.MinDis), 0.0f, 1.0f);
		}
		//each invocation (3 in total) is responsible for an outer level
		gl_TessLevelOuter[gl_InvocationID] = calcLoD(lodControl, vertexDistance[0], vertexDistance[1]);
		barrier();
		//the first invocation sync from other threads and compute the inner level
		if(gl_InvocationID == 0){
			gl_TessLevelInner[0] = (gl_TessLevelOuter[0] + gl_TessLevelOuter[1] + gl_TessLevelOuter[2]) / 3.0f;
		}
	}
}

float calcLoD(const TessellationSetting tess, const float v1, const float v2){
	return mix(tess.MaxLod, tess.MinLod, (v1 + v2) * 0.5f);
}