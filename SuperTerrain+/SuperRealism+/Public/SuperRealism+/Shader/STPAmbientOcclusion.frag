#version 460 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_shading_language_include : require

layout(early_fragment_tests) in;

#include </Common/STPCameraInformation.glsl>

/* ------------------- algorithm control ---------------------- */
//choose which ambient occlusion algorithm to use
#define AO_ALGORITHM 0

#if AO_ALGORITHM == 0
#define AO_KERNEL_SAMPLE_SIZE 1
uniform vec3 KernelSample[AO_KERNEL_SAMPLE_SIZE];
#elif AO_ALGORITHM == 1
#define TWO_PI 6.283185307179586476925286766559
uniform uint DirectionStep, RayStep;
#endif

uniform float KernelRadius;
uniform float SampleDepthBias;

/* ------------------------------------------------------------- */
//Input
in vec2 FragTexCoord;
//Output
layout(location = 0) out float OcclusionFactorOutput;

//Geometry buffer
layout(binding = 0) uniform sampler2D GeoDepth;
layout(binding = 1) uniform sampler2D GeoNormal;
//Noise
layout(bindless_sampler) uniform sampler2D RandomRotationVector;
//tile the noise texture over screen based on screen dimension to
//make tweaking the effect easier
uniform vec2 RotationVectorScale;

//Compute occlusion factor.
//Output a value in range [0, 1] where 1 means no occlusion and vice versa.
float computeOcclusion(const vec3, const vec3);

void main(){
	//get inputs
	const vec3 position_view = fragDepthReconstructionView(textureLod(GeoDepth, FragTexCoord, 0.0f).r, FragTexCoord),
		//our normal in the G-Buffer is in world space, we need to convert it to view space
		normal_view = normalize(Camera.ViewNormal * textureLod(GeoNormal, FragTexCoord, 0.0f).rgb);

	//compute occlusion and write to output
	OcclusionFactorOutput = computeOcclusion(position_view, normal_view);
}

//Given a view position, return a view position which is on the geometry.
//Basically the x, y coordinate is unchanged but the depth will be on the geometry, if the geometry exists.
vec3 viewSnapToGeometry(const mat4x2 proj_xy, const vec3 viewPos){
	const vec2 position_ndc = fragViewToNDC(proj_xy, viewPos);
	//get the geometry depth at this coordinate and return
	return fragDepthReconstructionView(textureLod(GeoDepth, position_ndc, 0.0f).r, position_ndc);
}

float computeOcclusion(const vec3 position_view, const vec3 normal_view){
	//the output occlusion factor
	float occlusion = 0.0f;
	//the projection matrix that only deals with x and y component
	const mat4x2 xyProjection = mat4x2(Camera.Projection);

#if AO_ALGORITHM == 0
	/* ========================================== Screen-Space Ambient Occlusion ========================================= */
	//the random vector rotates normal around z-axis so the z component is zero
	const vec3 randomVec = normalize(vec3(textureLod(RandomRotationVector, FragTexCoord * RotationVectorScale, 0.0f).rg, 0.0f)),
		//create TBN change of basis matrix: from tangent-space to view-space
		tangent = normalize(randomVec - normal_view * dot(randomVec, normal_view)),
		bitangent = cross(normal_view, tangent);
	const mat3 TBN = mat3(tangent, bitangent, normal_view);

	//iterate over the sample kernel and calculate occlusion factor
	for(int i = 0; i < KernelSample.length; i++){
		//get sample position
		//from tangent to view space
		const vec3 samplePos = position_view + TBN * KernelSample[i] * KernelRadius;
		//project sample position to sample texture to get position on screen/texture
		//compute the screen-space UV coordinate using the view space sampling position
		//get depth value for kernel sample
		const float sampleViewDepth = viewSnapToGeometry(xyProjection, samplePos).z;

		//range check to avoid occlusion bleeding at geometry edge and accumulate
		const float rangeCheck = smoothstep(0.0f, 1.0f, KernelRadius / abs(position_view.z - sampleViewDepth));
		occlusion += step(samplePos.z + SampleDepthBias, sampleViewDepth) * rangeCheck;
	}

	return 1.0f - (occlusion / KernelSample.length);
#elif AO_ALGORITHM == 1
	/* ============================================== Horizon-Based Ambient Occlusion =========================================== */
	//the first two components of the random vector is a rotation on screen-space (2D) while third component is just a random number
	//to determine the starting point of ray marching.
	const vec3 randomVec = textureLod(RandomRotationVector, FragTexCoord * RotationVectorScale, 0.0f).rgb;
	//extract data
	const vec2 randomRotation = normalize(randomVec.xy);
	const float randomRayStart = randomVec.z;

	//pre-compute fixed constants
	//the angle accumulated at each step in the hemisphere
	const float delta = TWO_PI / float(DirectionStep),
		//add one to the number of ray step to avoid the final sample to be fully attenuated
		stepSize = KernelRadius / float(RayStep + 1u),
		negInvR2 = -1.0f / (KernelRadius * KernelRadius);

	//scan through the hemisphere above the horizon
	for(uint d = 0u; d < DirectionStep; d++){
		const float elevation = delta * d,
			cos_elev = cos(elevation),
			sin_elev = sin(elevation);
		const mat2 elevRotation = mat2(
			cos_elev, -sin_elev,
			sin_elev, cos_elev
		);

		//compute normalised 2D direction by rotating the current direction about a random angle.
		const vec2 direction = elevRotation * randomRotation;

		//jitter starting sample within the first step
		float rayLength = randomRayStart * stepSize;
		//for the current ray direction, ray march to the object
		for(uint r = 0u; r < RayStep; r++){
			//proceed to the next sampling point on the ray
			//We are scanning through the horizon which is parallel to the view plane, such that z-component is zero,
			//and advances to the next sampling point along the ray direction.
			const vec3 samplePos = position_view + vec3(rayLength * direction, 0.0f),
				//get the object depth of the current sampling point
				sampleGeoPos = viewSnapToGeometry(xyProjection, samplePos);

			const vec3 V = sampleGeoPos - position_view;
			const float VdotV = dot(V, V),
				NdotV = dot(normal_view, V) * inversesqrt(VdotV),
				//calculate quadratic AO attenuation such that the further away the weaker AO
				falloff = VdotV * negInvR2 + 1.0f;
			//accumulate AO factor
			occlusion += clamp(NdotV - SampleDepthBias, 0.0f, 1.0f) * clamp(falloff, 0.0f, 1.0f);

			//advance to the next sampling point
			rayLength += stepSize;
		}
	}

	const float aoMul = 1.0f / (1.0f - SampleDepthBias);
	occlusion *= aoMul / float(DirectionStep * RayStep);
	return clamp(1.0f - occlusion * 2.0f, 0.0f, 1.0f);
#else
#error No ambient occlusion algorithm has been selected
#endif//AO_ALGORITHM
}