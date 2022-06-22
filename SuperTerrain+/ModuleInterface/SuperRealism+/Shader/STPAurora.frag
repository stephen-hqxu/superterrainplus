#version 460 core
#extension GL_ARB_bindless_texture : require

layout(early_fragment_tests) in;

struct NoiseOctaveFunction{
	float iAmp, Pers, Lacu;
};

struct TriNoiseSetting{
	NoiseOctaveFunction fnNoise, fnDist;
	float iFreqDist;
	float Curv;
	mat2 octRot;
	float Spd;
	float C, maxInt;
	unsigned int Oct;
};

struct AuroraSetting{
	float Flat, stepSz, projRot;
	//x: the fading band length, equals to fading start - fading end
	//y: fading end, altitude below which aurora has zero intensity
	vec2 Fade;
	float LumMul;
	unsigned int Iter;
};

//Input
in vec3 FragRayDirection;
//Output
layout(location = 0) out vec4 FragColor;

layout(bindless_sampler) uniform sampler1D AuroraColorSpectrum;

//custom settings for procedural aurora
uniform AuroraSetting Aurora;
uniform TriNoiseSetting TriNoise;

uniform float AuroraTime;

//Triangular noise generator utilising the triangular wave function
float triangularNoise(vec2);

void main(){
	const vec3 rayDir = normalize(FragRayDirection);
	if(rayDir.y < Aurora.Fade.y){
		//aurora will complete fade out, no need to calculate all the expensive stuff
		FragColor = vec4(vec3(0.0f), 1.0f);
		return;
	}

	vec3 auroraColor = vec3(0.0f);
	float step_count = 0.0f, altitude = 0.0f;
	//minus one to ensure the entire altitude range [0.0, 1.0] is utilised.
	//Otherwise only [0.0, (Iter - 1) / (Iter)] is used.
	const float altitude_inc = 1.0f / float(Aurora.Iter - 1u);
	//We build the aurora using a bottom-up approach, and evaluate the noise function at each altitude step.
	for(uint i = 0u; i < Aurora.Iter; i++){
		//As the skybox is emulated as a dome, we want to project the aurora onto an infinite plane.
		//The altitude contracts the dome towards the centre, i.e., a sphere to a pie.
		//To avoid division by zero when the view direction altitude is zero, add a small bias.
		//The bias will rotate the projection sphere.
		const float projection_distance = (Aurora.Flat + step_count) / (rayDir.y + Aurora.projRot),
			//Don't care about altitude, aurora will fade out gradually as altitude increases, leaving a tail.
			//We want the intensity to decrease as the altitude increases
			step_intensity = triangularNoise((projection_distance * rayDir).xz) * (1.0f - altitude);

		//apply some colours to the aurora, based on the current altitude
		auroraColor += step_intensity * textureLod(AuroraColorSpectrum, altitude, 0.0f).rgb;

		//increment step
		step_count += Aurora.stepSz;
		altitude += altitude_inc;
	}

	//altitude fading, a simple linear interpolation
	auroraColor *= clamp((rayDir.y - Aurora.Fade.y) / Aurora.Fade.x, 0.0f, 1.0f);

	//output with intensity scaling
	FragColor = vec4(vec3(auroraColor * Aurora.LumMul), 1.0f);
}

//Construct a 2D rotation matrix, angle unit radians
mat2 doRotation(float theta){
	const float sinT = sin(theta),
		cosT = cos(theta);
	//as a reminder, GL uses column-major matrix
	return mat2(
		cosT, sinT,
		-sinT, cosT
	);
}

float triangularWave(float x){
	//if you sketch this function, it should give a shark-fin like function ranged [0.0, 0.5]
	return abs(fract(x) - 0.5f);
}

vec2 triangularWave(vec2 p){
	return vec2(triangularWave(p.x) + triangularWave(p.y), triangularWave(p.y + triangularWave(p.x)));
}

float triangularNoise(vec2 uv){
	const NoiseOctaveFunction func_noise = TriNoise.fnNoise,
		func_distortion = TriNoise.fnDist;

	float amplitude_noise = func_noise.iAmp,
		amplitude_distortion = func_distortion.iAmp,
		//cumulative noise intensity value
		value = 0.0f;
	//initial rotation to the input coordinate to create a curvature for the wave
	uv *= doRotation(uv.x * TriNoise.Curv);
	vec2 uv_distortion = uv * TriNoise.iFreqDist;

	for(uint i = 0u; i < TriNoise.Oct; i++){
		//domain wrapping to create a distortion coordinate
		const vec2 distortion = triangularWave(uv_distortion) * doRotation(AuroraTime * TriNoise.Spd);
		uv += distortion * amplitude_distortion;

		//progress octave parameters
		uv *= func_noise.Lacu;
		amplitude_noise *= func_noise.Pers;
		uv_distortion *= func_distortion.Lacu;
		amplitude_distortion *= func_distortion.Pers;

		value += triangularWave(uv.x + triangularWave(uv.y)) * amplitude_noise;
		//rotate the coordinate for the next iteration
		uv *= TriNoise.octRot;
	}

	//contrast adjustment
	return clamp(pow(value, -TriNoise.C), 0.0f, TriNoise.maxInt);
}