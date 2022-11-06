#version 460 core
#extension GL_ARB_shading_language_include : require
#extension GL_ARB_bindless_texture : require

#include </Common/STPCameraInformation.glsl>
#include </Common/STPMaterialRegistry.glsl>

layout(early_fragment_tests) in;

//Input
in vec2 FragTexCoord;
//Output
layout(location = 0) out vec4 FragColor;

uniform float MaxDistance;
uniform float DepthBias;
uniform uint StepResolution, StepSize;

//The buffer here should contain the *scene* without the object where BSDF is applied
layout(bindless_sampler) uniform sampler2D SceneDepth;
layout(bindless_sampler) uniform sampler2D SceneColor;
//currently our BSDF only applies to mirror surface so roughness is not needed
//The buffer here should contain the *object* to be rendered with BSDF
layout(binding = 0) uniform sampler2D ObjectDepth;
layout(binding = 1) uniform sampler2D ObjectNormal;
layout(binding = 2) uniform usampler2D ObjectMaterial;

//find the closest hit on the geometry, returns the colour value at that point.
vec3 findClosestHitColor(vec3, vec3, mat4x2);
//sample linear depth from the scene
float getLinearDepthAt(vec2);

void main(){
	//find the material for the current fragment
	const uint object_mat_id = textureLod(ObjectMaterial, FragTexCoord, 0.0f).r;
	const STPMaterialProperty object_mat = Material[object_mat_id];
	const float object_depth = textureLod(ObjectDepth, FragTexCoord, 0.0f).r;
	
	//get the raw inputs
	const vec3 position_view = fragDepthReconstructionView(object_depth, FragTexCoord),
		normal_view = normalize(Camera.ViewNormal * textureLod(ObjectNormal, FragTexCoord, 0.0f).rgb),
		//calculate reflection vector from input
		incident_direction = normalize(position_view),
		reflection_direction = normalize(reflect(incident_direction, normal_view)),
		refraction_direction = normalize(refract(incident_direction, normal_view, 1.0f / object_mat.RefractiveIndex));

	const mat4x2 xyProjection = mat4x2(Camera.Projection);
	const vec3 reflection_color = findClosestHitColor(position_view, reflection_direction, xyProjection),
		refraction_color = findClosestHitColor(position_view, refraction_direction, xyProjection);

	//Fresnel equation
	const float absDepth = getLinearDepthAt(FragTexCoord) - lineariseDepth(object_depth),
		refractiveFactor = pow(abs(dot(incident_direction, normal_view)), object_mat.Reflexivity),
		//deeper -> more opaque
		displayOpacity = clamp(absDepth * object_mat.Opacity, 0.0f, 1.0f);
	const vec3 fresnelColor = mix(reflection_color, refraction_color, clamp(refractiveFactor, 0.0f, 1.0f));

	//make the object more transparent at lower depth
	FragColor = vec4(mix(textureLod(SceneColor, FragTexCoord, 0.0f).rgb, fresnelColor, displayOpacity), displayOpacity);
}

//compare the current sample depth with the actual depth on the depth buffer
float compareSampleDepth(float init_depth, float final_depth, float factor, vec2 position_ndc){
	//use linear interpolation to calculate the current ray depth
	//perspective correction for perspective projection
	const float sampleDepth_theoretical = init_depth * final_depth / mix(init_depth, final_depth, factor),
		//and the actual depth on the scene
		sampleDepth_actual = getLinearDepthAt(position_ndc);

	//test which depth is closer to the viewer
	return sampleDepth_theoretical - sampleDepth_actual;
}

//True if the sample point is inside the geometry, otherwise false
bool isDeltaDepthInside(float depth){
	return depth > 0.0f && depth < DepthBias;
}

vec3 findClosestHitColor(vec3 ray_origin, vec3 ray_dir, mat4x2 proj_xy){
	/* =============================================== Pass 1 ================================================== */
	/* ==== perform a rough ray marching to get an approximated ray-primitive intersection point, if exists ==== */
	const vec3 rayEndView = ray_origin + ray_dir * MaxDistance;
	//all operations are done in NDC
	const vec2 rayStart = FragTexCoord,
		rayEnd = fragViewToNDC(proj_xy, rayEndView),
		rayDelta = rayEnd - rayStart,
		rayDir = normalize(rayDelta);
	const float rayStartDepth = ray_origin.z,
		rayEndDepth = rayEndView.z,
		rayMaxLength = length(rayDelta),
		stepInc = rayMaxLength / StepResolution;

	bool hit_pass1 = false;
	//there is no need to test the first sampling point, start from the second
	float rayLength = stepInc;
	for(uint i = 0u; i < StepResolution; i++){
		const vec2 samplePosition = rayStart + rayDir * rayLength;
		const float delta_depth = compareSampleDepth(rayStartDepth, rayEndDepth, rayLength / rayMaxLength, samplePosition);

		if(isDeltaDepthInside(delta_depth)){
			hit_pass1 = true;
			break;
		}
		//advance to the next sampling point
		rayLength += stepInc;
	}
	if(!hit_pass1){
		//TODO: handle the case when pass 1 does not hit anything
	}

	/* =============================================== Pass 2 ================================================= */
	/* ==================================== search for a precise hit point ==================================== */
	float segStart = rayLength - stepInc,
		segEnd = rayLength;
	vec2 closestHit;
	bool hit_pass2 = false;
	//perform binary search
	for(uint i = 0u; i < StepSize; i++){
		//find the centre of the segment
		const float hitPoint = mix(segStart, segEnd, 0.5f);
		//the rest is the same as the previous pass
		closestHit = rayStart + rayDir * hitPoint;
		const float delta_depth = compareSampleDepth(rayStartDepth, rayEndDepth, hitPoint / rayMaxLength, closestHit);

		if(isDeltaDepthInside(delta_depth)){
			hit_pass2 = true;
			segEnd = hitPoint;
		}else{
			segStart = hitPoint;
		}
	}
	if(!hit_pass2){
		//TODO: handle the case when pass 2 does not failed, for some reasons
	}

	//read the colour value at this point
	return textureLod(SceneColor, closestHit, 0.0f).rgb;
}

float getLinearDepthAt(vec2 uv){
	return lineariseDepth(textureLod(SceneDepth, uv, 0.0f).r);
}