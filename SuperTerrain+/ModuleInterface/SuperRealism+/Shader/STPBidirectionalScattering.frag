#version 460 core
#extension GL_ARB_shading_language_include : require
#extension GL_ARB_bindless_texture : require

//This shader contains a few permutations of BSDF implementations,
//value defined by shader compiler depends on user settings.
#define BSDF_SHADER_TYPE_SELECTION 0

/* ----------------------------- global definition used by more than one shader stage ----------------------- */
layout(early_fragment_tests) in;

#include </Common/STPMaterialRegistry.glsl>

//Input
in vec2 FragTexCoord;

layout (binding = 0) uniform usampler2D ObjectMaterial;

/* globally used functions */
//get the material property data for the current texel
STPMaterialProperty getMaterialProps();
//compute the reflected and refracted ray direction, respectively, given incidence direction, normal and index of refraction
vec3[2] calcBSDFRayDirection(vec3, vec3, float);
//an inexpensive approximation for Fresnel equation, given the reflection coefficient and cosine of angle of incidence
float calcFresnelSchlickReflectance(float, float);
//given ray length, Fresnel reflectivity, reflection and refraction colour, compute the reflection and refraction outputs of the BSDF shading process
vec3[2] computeOutput(STPMaterialProperty, float, float, vec3, vec3);
/* ********************************************************************************************************* */

#if BSDF_SHADER_TYPE_SELECTION == 1
/* ------------------- BSDF ray data generation shader (for ray traced ray-primitive intersection) ---------- */
#define EMIT_DEPTH_RECON_WORLD_IMPL
#include </Common/STPCameraInformation.glsl>

//Output
layout (location = 0) out vec4 ReflectionRayDirection;
layout (location = 1) out vec4 RefractionRayDirection;
//precompute values to be used later to reduce memory read
layout (location = 2) out float FSReflectance;

//these texture will be used to calculate the output ray directions
layout (binding = 1) uniform sampler2D ObjectDepth;
layout (binding = 2) uniform sampler2D ObjectNormal;

void main(){
	const STPMaterialProperty obj_mat = getMaterialProps();
	//read raw data, ray tracing requires world space data
	const vec3 position_world = fragDepthReconstruction(textureLod(ObjectDepth, FragTexCoord, 0.0f).r, FragTexCoord),
		normal_world = normalize(textureLod(ObjectNormal, FragTexCoord, 0.0f).rgb),
		incident_direction = normalize(position_world - Camera.Position);
	const vec3[2] ray_direction = calcBSDFRayDirection(incident_direction, normal_world, obj_mat.RefractiveIndex);

	//write them to the output, remember to convert them to [0, 1] because we are using unsigned normalised texture
	ReflectionRayDirection = vec4(ray_direction[0] * 0.5f + 0.5f, 1.0f);
	RefractionRayDirection = vec4(ray_direction[1] * 0.5f + 0.5f, 1.0f);
	FSReflectance = calcFresnelSchlickReflectance(obj_mat.Reflectance, dot(incident_direction, normal_world));
}
#elif BSDF_SHADER_TYPE_SELECTION == 2
/* -------------------------------------------- BSDF colour texture resolution shader ------------------------------------------------
 * (applying shading equations on reflection and transmission colour outputs from ray tracer to combine them into single colour output)
*/
#include </Common/STPRayTracedIntersectionData.glsl>

//Output
layout (location = 0, index = 0) out vec4 FragReflection;
layout (location = 0, index = 1) out vec4 FragRefraction;

//computed value from the previous shader stage
layout (bindless_sampler) uniform sampler2D FSReflectance;
//other inputs
layout (binding = 1) uniform sampler2D ReflectionColor;
layout (binding = 2) uniform sampler2D RefractionColor;

void main(){
	const STPMaterialProperty obj_mat = getMaterialProps();
	//grab the data for the current pixel
	const float reflectiveFactor = textureLod(FSReflectance, FragTexCoord, 0.0f).r;
	const vec3 reflection_colour = textureLod(ReflectionColor, FragTexCoord, 0.0f).rgb,
		refraction_colour = textureLod(RefractionColor, FragTexCoord, 0.0f).rgb;
	//grab some more data from the current working intersection data
	const float rayTime = textureLod(sampler2D(RTIntersection.RayTime), FragTexCoord, 0.0f).r;

	const vec3[2] fragOutput = computeOutput(obj_mat, rayTime, reflectiveFactor, reflection_colour, refraction_colour);
	FragReflection = vec4(fragOutput[0], 1.0f);
	FragRefraction = vec4(fragOutput[1], 1.0f);
}
#else
/* ------------------ BSDF implemented with screen-space techniques (legacy rasterisation approach) --------- */
#define EMIT_DEPTH_RECON_VIEW_IMPL
#define EMIT_VIEW_TO_NDC_IMPL
#define EMIT_LINEARISE_DEPTH_IMPL
#include </Common/STPCameraInformation.glsl>

//Output
layout (location = 0) out vec4 FragColor;

uniform float MaxDistance;
uniform float DepthBias;
uniform unsigned int StepResolution, StepSize;

//The buffer here should contain the *scene* without the object where BSDF is applied
layout (bindless_sampler) uniform sampler2D SceneDepth;
layout (bindless_sampler) uniform sampler2D SceneColor;
//currently our BSDF only applies to mirror surface so roughness is not needed
//The buffer here should contain the *object* to be rendered with BSDF
layout (binding = 1) uniform sampler2D ObjectDepth;
layout (binding = 2) uniform sampler2D ObjectNormal;

//find the closest hit on the geometry, returns the colour value at that point.
vec3 findClosestHitColor(vec3, vec3, mat4x2);
//sample linear depth from the scene
float getLinearDepthAt(vec2);

void main(){
	const STPMaterialProperty object_mat = getMaterialProps();
	const float object_depth = textureLod(ObjectDepth, FragTexCoord, 0.0f).r;
	
	//get the raw inputs
	const vec3 position_view = fragDepthReconstruction(object_depth, FragTexCoord),
		normal_view = normalize(Camera.ViewNormal * textureLod(ObjectNormal, FragTexCoord, 0.0f).rgb),
		incident_direction = normalize(position_view);
	//calculate ray vectors
	const vec3[2] ray_direction = calcBSDFRayDirection(incident_direction, normal_view, object_mat.RefractiveIndex);

	const mat4x2 xyProjection = mat4x2(Camera.Projection);
	const vec3 reflection_color = findClosestHitColor(position_view, ray_direction[0], xyProjection),
		refraction_color = findClosestHitColor(position_view, ray_direction[1], xyProjection);

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
	const float sampleDepth_theoretical = (Camera.useOrtho ? 1.0f : init_depth * final_depth) / mix(init_depth, final_depth, factor),
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
#endif//BSDF_SHADER_TYPE_SELECTION

/* ------------------------------------- Global function definitions -------------------------- */
STPMaterialProperty getMaterialProps(){
	//find the material for the current fragment
	const unsigned int object_mat_id = textureLod(ObjectMaterial, FragTexCoord, 0.0f).r;
	return Material[object_mat_id];
}

vec3[2] calcBSDFRayDirection(vec3 incident_direction, vec3 normal, float ior){
	//calculate reflection and refraction vector from inputs
	const vec3 reflection_direction = normalize(reflect(incident_direction, normal)),
		refraction_direction = normalize(refract(incident_direction, normal, 1.0f / ior));

	return vec3[2](reflection_direction, refraction_direction);
}

float calcFresnelSchlickReflectance(float R0, float cosTheta){
	const float oneMdot = 1.0f - cosTheta;
	//experiment shows calculating integer power manually is approximately 10 times faster than using `pow` function
	return R0 + (1.0f - R0) * oneMdot * oneMdot * oneMdot * oneMdot * oneMdot;
}

vec3[2] computeOutput(STPMaterialProperty mat, float rayLength, float reflectivity, vec3 reflectionColour, vec3 refractionColour){
	//calculate refraction opacity
	const float opacity = smoothstep(mat.AttenuationStart, mat.AttenuationEnd, rayLength);
	//the `tint` of the material, scaled by the opacity; the deeper the ray travels, the stronger the `tint` is.
	const float intrinsicColour = textureLod(sampler1D(mat.IntrinsicSpectrum), opacity, 0.0f).r;
	
	//TODO: Use scattering equation like what we did for the atmosphere to calculate the optical depth rather than using ray length,
	//it can be pre-computed and store the colours to the spectrum.
	//-------------------------------------------------------------
	//combine all colours using the Fresnel equation:
	//output_colour = mix(under_surface_colour, over_surface_colour, reflectance)
	//where: under_surface_colour = mix(refraction_colour, intrinsic_colour, exp(-optical_depth))
	//and: over_surface_colour = reflection_colour * specular
	return vec3[2](reflectivity * reflectionColour, (1.0f - reflectivity) * mix(refractionColour, intrinsicColour, opacity));
}