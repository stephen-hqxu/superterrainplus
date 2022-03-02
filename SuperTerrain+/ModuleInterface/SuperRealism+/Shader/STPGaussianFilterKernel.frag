#version 460 core

layout(early_fragment_tests) in;

//filter settings
#define GAUSSIAN_KERNEL_SIZE 1

//Input
in vec2 FragTexCoord;
layout(binding = 0) uniform sampler2D ImgInput;
//Output
layout(location = 0) out float FilterOutput;

//Uniform about the filter kernel
uniform float GaussianKernel[GAUSSIAN_KERNEL_SIZE];
uniform uint KernelRadius;

//The main filter function
//A separable filter allows breaking down a 2D filter into two 1D filters
subroutine float SeparableFilter(int);
layout(location = 0) subroutine uniform SeparableFilter filterImage;

void main(){
	FilterOutput = filterImage(int(KernelRadius));
}

#define CALCULATE_TEXEL_UNIT() const vec2 texel_unit = 1.0f / vec2(textureSize(ImgInput, 0))

layout(index = 0) subroutine(SeparableFilter) float horizontalPass(int radius){
	CALCULATE_TEXEL_UNIT();

	float acc = 0.0f;
	for(int x = -radius; x <= radius; x++){
		acc += textureLod(ImgInput, FragTexCoord + vec2(x * texel_unit.x, 0.0f), 0).r * GaussianKernel[x + radius];
	}

	return acc;
}

layout(index = 1) subroutine(SeparableFilter) float verticalPass(int radius){
	CALCULATE_TEXEL_UNIT();

	float acc = 0.0f;
	for(int y = -radius; y <= radius; y++){
		acc += textureLod(ImgInput, FragTexCoord + vec2(0, y * texel_unit.y), 0).r * GaussianKernel[y + radius];
	}

	return acc;
}