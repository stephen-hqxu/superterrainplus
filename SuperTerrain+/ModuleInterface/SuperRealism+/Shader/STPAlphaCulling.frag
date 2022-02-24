#version 460 core
//Input
in vec2 FragTexCoord;
//And no output

//Operator which can be defined by the user
#define ALPHA_TEST_OPERATOR <=

layout(binding = 0) uniform sampler2D ColorInput;

uniform float AlphaThreshold;

void main(){
	if(texture(ColorInput, FragTexCoord).a ALPHA_TEST_OPERATOR AlphaThreshold){
		discard;
	}
}