#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require

layout(early_fragment_tests) in;

//depth reconstruction to view space
#define EMIT_DEPTH_RECON_IMPL 1
#include </Common/STPCameraInformation.glsl>

/* ------------------- sampling controls ---------------------- */
#define SSAO_KERNEL_SAMPLE_SIZE 1

uniform vec3 KernelSample[SSAO_KERNEL_SAMPLE_SIZE];

uniform float KernelRadius;
uniform float SampleDepthBias;
//tile the noise texture over screen based on screen dimension to
//make tweaking the effect easier
uniform vec2 NoiseTexScale;

/* ----------------------------------------------------------------- */

//Input
in vec2 FragTexCoord;
//Output
layout (location = 0) out float OcclusionFactorOutput;

//Geometry buffer
layout (binding = 0) uniform sampler2D GeoDepth;
layout (binding = 1) uniform sampler2D GeoNormal;
//Noise
layout (bindless_sampler) uniform sampler2D NoiseVector;

void main(){
	//get inputs for SSAO
	const vec3 fragPos = fragDepthReconstruction(texture(GeoDepth, FragTexCoord).r, FragTexCoord),
		//our normal in the G-Buffer is in world space, we need to convert it to view space
		fragNormal = normalize(Camera.ViewNormal * texture(GeoNormal, FragTexCoord).rgb),
		//the random vector rotates normal around z-axis so the z component is zero
		randomVec = normalize(vec3(texture(NoiseVector, FragTexCoord * NoiseTexScale).rg, 0.0f)),
		//create TBN change of basis matrix: from tangent-space to view-space
		tangent = normalize(randomVec - fragNormal * dot(randomVec, fragNormal)),
		bitangent = cross(fragNormal, tangent);
	const mat3 TBN = mat3(tangent, bitangent, fragNormal);

	//iterate over the sample kernel and calculate occlusion factor
	float occlusion = 0.0f;
	for(int i = 0; i < KernelSample.length; i++){
		//get sample position
		//from tangent to view space
		const vec3 samplePos = fragPos + TBN * KernelSample[i] * KernelRadius;
		
		//project sample position to sample texture to get position on screen/texture
		//convert from view to clip space first
		vec4 offset = Camera.Projection * vec4(samplePos, 1.0f);
		//from clip space to NDC by perspective division
		offset.xyz /= offset.w;
		//range convert from [-1, 1] to [0, 1]
		offset.xyz = offset.xyz * 0.5f + 0.5f;

		//get sample depth
		//get depth value for kernel sample	
		//Depth reconstruction but in converts to view space and only grab the camera depth value,
		//however, this formula assumes a perspective camera
		const float sampleViewDepth = fragDepthReconstruction(texture(GeoDepth, offset.xy).r, offset.xy).z;

		//range check and accumulate
		const float rangeCheck = smoothstep(0.0f, 1.0f, KernelRadius / abs(fragPos.z - sampleViewDepth));
		occlusion += step(samplePos.z + SampleDepthBias, sampleViewDepth) * rangeCheck;
	}

	//write to output
	OcclusionFactorOutput = 1.0f - (occlusion / KernelSample.length);
}