#ifndef _STP_ANIMATED_WAVE_GLSL_
#define _STP_ANIMATED_WAVE_GLSL_

struct WaveFunction {
	float iRot, iFreq, iAmp, iSpd;
	float octRot, Lacu, Pers, octSpd;
	float Drag;
};

//X: wave height; Y: wave height derivative
vec2 waveEquation(vec2, vec2, float, float, float);

float waveHeight(vec2 position, WaveFunction func, uint iteration, float time) {
	float rotation = func.iRot, 
		frequency = func.iFreq, 
		amplitude = func.iAmp, 
		speed = func.iSpd;
	float waveWeight = 0.0f, weightSum = 0.0f;

	//The wave is generated, essentially by compositing a bunch of trigonometric functions with fractal noise technique.
	for (uint i = 0u; i < iteration; i++) {
		//each octave rotates the wave a little bit
		const vec2 waveDir = vec2(cos(rotation), sin(rotation)),
			wave = waveEquation(position, waveDir, speed, frequency, time);
		//move the octave to the next derivative position
		position += waveDir * wave.y * waveWeight * func.Drag;
		waveWeight += wave.x * amplitude;
		weightSum += waveWeight;

		//prepare data for the next octave
		rotation += func.octRot;
		frequency *= func.Lacu;
		amplitude *= func.Pers;
		speed *= func.octSpd;
	}

	//normalise noise height
	return waveHeight / weightSum;
}

vec2 waveEquation(vec2 position, vec2 direction, float speed, float frequency, float phase) {
	const float x = dot(position, direction) * frequency + phase * speed,
		//the exp function makes the wave a bit more gentle; wave has range [exp(-2), 1]
		wave = exp(sin(x) - 1.0f),
		waveDx = wave * (-cos(x));
	return vec2(wave, waveDx);
}

#endif//_STP_ANIMATED_WAVE_GLSL_