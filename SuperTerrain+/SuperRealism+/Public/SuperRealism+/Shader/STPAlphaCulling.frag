#version 460 core
//Input
in vec2 FragTexCoord;
//And no output

//define to true so alpha test use two sub-expressions
#define USE_DUAL_EXPRESSIONS 0
#if USE_DUAL_EXPRESSIONS
#define ALPHA_COMPARATOR_A ==
#define ALPHA_CONNECTOR ||
#define ALPHA_COMPARATOR_B ==

uniform float LimA, LimB;
#else
//operator which can be defined by the user
#define ALPHA_COMPARATOR <=

uniform float Lim;
#endif//USE_DUAL_EXPRESSIONS

layout(binding = 0) uniform sampler2D ColorInput;

void main(){
	const float alpha = textureLod(ColorInput, FragTexCoord, 0.0f).a;

#if USE_DUAL_EXPRESSIONS
	if((alpha ALPHA_COMPARATOR_A LimA) ALPHA_CONNECTOR (alpha ALPHA_COMPARATOR_B LimB)){
		discard;
	}
#else
	if(alpha ALPHA_COMPARATOR Lim){
		discard;
	}
#endif
}