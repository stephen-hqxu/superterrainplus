#version 460 core
#extension GL_ARB_shading_language_include : require

//Input
in VertexTES{
	vec3 position_world;
} fs_in;

//Output
#include </Common/STPGeometryBufferWriter.glsl>

/* -------------------------- Water Normal Mapping ---------------------- */
#include </Common/STPAnimatedWave.glsl>

uniform float WaveHeight;
uniform WaveFunction WaterWave;
//note that this is different from the iteration used in evaluation shader.
uniform uint WaveNormalIteration;
uniform float WaveTime;
//controls the distance between each sample
uniform float Epsilon;
/* ---------------------------------------------------------------------- */

#include </Common/STPCameraInformation.glsl>

uniform vec3 Tint;
uniform uint WaterMaterialID;

//compute the surface normal of the water at the current fragment
vec3 calcWaterNormal(const vec2);

void main(){
	const float waveDistance = distance(fs_in.position_world, Camera.Position);
	vec3 waveNormal = calcWaterNormal(fs_in.position_world.xz);

	//gradually flatten the water surface as it gets further away
	const float glossiness = smoothstep(0.0f, Camera.Far, waveDistance);
	waveNormal = normalize(mix(waveNormal, vec3(0.0f, 1.0f, 0.0f), glossiness));

	writeGeometryData(Tint, waveNormal, 1.0 - glossiness, 1.0f, WaterMaterialID);
}

vec3 getSamplePosition(const vec2 coord){
	return vec3(coord.x, getWaveHeight(coord, WaterWave, WaveNormalIteration, WaveTime) * WaveHeight, coord.y);
}

vec3 calcWaterNormal(const vec2 position){
	const vec2 eps = vec2(Epsilon, 0.0f);
	//get sample positions
	const vec3 centreHeight = getSamplePosition(position),
		NWHeight = getSamplePosition(position - eps.xy),
		SEHeight = getSamplePosition(position + eps.yx);

	//We use a simple cross product to compute an approximated the surface normal 
	//instead of a gradient filter like what has been done in the terrain shader
	//due to consideration of performance.
	const vec3 NWDelta = normalize(centreHeight - NWHeight),
		SEDelta = normalize(centreHeight - SEHeight);
	return normalize(cross(NWDelta, SEDelta));
}