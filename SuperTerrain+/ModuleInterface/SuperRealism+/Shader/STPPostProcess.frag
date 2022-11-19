#version 460 core
/* ------------------------- Post Process Setting ---------------------- */
//Tone mapping algorithm
#define TONE_MAPPING 0

#if TONE_MAPPING
struct FilmicToneSetting{
#if TONE_MAPPING == 1
	float P, a, m, c, b;
	//precomputed values based on the parameters above
	float l0;
	float S0, S1;
	float CP;
#elif TONE_MAPPING == 2
	float a, b, c, d;
#elif TONE_MAPPING == 3
	float A, B, C, D, E, F;
	float mW;
#endif
};
#endif//TONE_MAPPING
/* ------------------------------------------------------------------------ */

//Input
in vec2 FragTexCoord;
//Output
out vec4 FragColor;

layout(binding = 0) uniform sampler2D ScreenBuffer;

uniform float Gamma;

#if TONE_MAPPING
//Tone factor is interpreted differently based on choice of tone mapping algorithm
uniform FilmicToneSetting Tone;

//perform tone mapping for the input color
vec3 toneMapping(const vec3);
#endif//TONE_MAPPING

void main(){
	vec3 ScreenColor = textureLod(ScreenBuffer, FragTexCoord, 0.0f).rgb;

	//Tone Mapping
#if TONE_MAPPING
	ScreenColor = toneMapping(ScreenColor);
#endif

	//gamma correction
	ScreenColor = pow(ScreenColor, vec3(1.0f / Gamma));

	//write the final color
	FragColor = vec4(ScreenColor, 1.0f);
}

#if TONE_MAPPING
vec3 toneMapping(const vec3 x){
	//switch to different tone mapping function

#if TONE_MAPPING == 1
	//Gran Turismo(Hajime Uchimura, 2017)
	const vec3 w0 = 1.0f - smoothstep(0.0f, Tone.m, x),
		w2 = step(Tone.m + Tone.l0, x),
		w1 = 1.0f - w0 - w2;
	const vec3 T = Tone.m * pow(x / Tone.m, vec3(Tone.c)) + Tone.b,
		S = Tone.P - (Tone.P - Tone.S1) * exp(Tone.CP * (x - Tone.S0)),
		L = Tone.m + Tone.a * (x - Tone.m);

	return T * w0 + L * w1 + S * w2;
#elif TONE_MAPPING == 2
	//(Timothy Lottes, 2016)
	const vec3 va = vec3(Tone.a);
	return pow(x, va) / (pow(x, va * Tone.d) * Tone.b + Tone.c);
#elif TONE_MAPPING == 3
	//Uncharted2(John Hable, 2010)
	const vec3 product_a_x2 = Tone.A * x * x;
	return (((product_a_x2 + x * Tone.C * Tone.B + Tone.D * Tone.E) / (product_a_x2 + x * Tone.B + Tone.D * Tone.F)) - Tone.E / Tone.F) / Tone.mW;
#endif
}
#endif//TONE_MAPPING