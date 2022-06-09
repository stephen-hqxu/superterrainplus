#version 460 core
#extension GL_ARB_shading_language_include : require

layout(early_fragment_tests) in;

//filter settings
#define GAUSSIAN_KERNEL_SIZE 1
#define GUASSIAN_KERNEL_VARIANT 0

//Input
in vec2 FragTexCoord;
layout(binding = 0) uniform sampler2D ImgInput;
#if GUASSIAN_KERNEL_VARIANT == 1
layout(binding = 1) uniform sampler2D ImgDepth;
#endif

//Output
layout(location = 0) out float FilterOutput;

//Gaussian kernel weight is L1-normalised
uniform float GaussianKernel[GAUSSIAN_KERNEL_SIZE];
uniform uint KernelRadius;
#if GUASSIAN_KERNEL_VARIANT == 1
//1.0f / (sqrt(2.0f * PI) * variance)
uniform float StandardDeviation;
//1.0f / (2.0f * variance * variance)
uniform float InvTwoVarSqr;
uniform float Sharpness;

#define EMIT_LINEARISE_DEPTH_IMPL
#include </Common/STPCameraInformation.glsl>
#endif

//The main filter function
//A separable filter allows breaking down a 2D filter into two 1D filters
subroutine float SeparableFilter(int);
layout(location = 0) subroutine uniform SeparableFilter filterImage;

void main(){
	FilterOutput = filterImage(int(KernelRadius));
}

float getImageValueOffset(vec2 uv_offset){
	return textureLod(ImgInput, FragTexCoord + uv_offset, 0).r;
}

#if GUASSIAN_KERNEL_VARIANT == 1
float getImageDepthOffset(vec2 uv_offset){
	return lineariseDepth(textureLod(ImgDepth, FragTexCoord + uv_offset, 0).r);
}

float getFilteredImageDepthOffset(float depth_centre, vec2 uv_offset){
	const float delta_depth = (getImageDepthOffset(uv_offset) - depth_centre) * Sharpness,
		freq_response = -delta_depth * delta_depth * InvTwoVarSqr;
	return StandardDeviation * exp(freq_response);
}
#endif

#define HORIZONTAL_PASS layout(index = 0) subroutine(SeparableFilter) float horizontalPass(int radius)
#define VERTICAL_PASS layout(index = 1) subroutine(SeparableFilter) float verticalPass(int radius)
#define KERNEL_LOOP(VAR) for(int VAR = -radius; VAR <= radius; VAR++)
#define UV_OFFSET_HORIZONTAL vec2(x * texel_unit.x, 0.0f)
#define UV_OFFSET_VERTICAL vec2(0.0f, y * texel_unit.y)

#define CALCULATE_TEXEL_UNIT() const vec2 texel_unit = 1.0f / vec2(textureSize(ImgInput, 0))

#if GUASSIAN_KERNEL_VARIANT == 0
//Classic Gaussian Filter
HORIZONTAL_PASS{
	CALCULATE_TEXEL_UNIT();

	float acc = 0.0f;
	KERNEL_LOOP(x){
		acc += getImageValueOffset(UV_OFFSET_HORIZONTAL) * GaussianKernel[x + radius];
	}

	return acc;
}
VERTICAL_PASS{
	CALCULATE_TEXEL_UNIT();

	float acc = 0.0f;
	KERNEL_LOOP(y){
		acc += getImageValueOffset(UV_OFFSET_VERTICAL) * GaussianKernel[y + radius];
	}

	return acc;
}
#elif GUASSIAN_KERNEL_VARIANT == 1
//Bilateral Filter
#define GET_CENTRE_DEPTH() const float depth_centre = getImageDepthOffset(vec2(0.0f))

HORIZONTAL_PASS{
	CALCULATE_TEXEL_UNIT();
	GET_CENTRE_DEPTH();

	float img_acc = 0.0f,
		weight_acc = 0.0f;
	KERNEL_LOOP(x){
		const vec2 uv_offset = UV_OFFSET_HORIZONTAL;
		const float weight = getFilteredImageDepthOffset(depth_centre, uv_offset) * GaussianKernel[x + radius];

		img_acc += getImageValueOffset(uv_offset) * weight;
		weight_acc += weight;
	}

	//normalise weight
	return img_acc / weight_acc;
}
VERTICAL_PASS{
	CALCULATE_TEXEL_UNIT();
	GET_CENTRE_DEPTH();

	float img_acc = 0.0f,
		weight_acc = 0.0f;
	KERNEL_LOOP(y){
		const vec2 uv_offset = UV_OFFSET_VERTICAL;
		const float weight = getFilteredImageDepthOffset(depth_centre, uv_offset) * GaussianKernel[y + radius];

		img_acc += getImageValueOffset(uv_offset) * weight;
		weight_acc += weight;
	}

	return img_acc / weight_acc;
}
#endif//GUASSIAN_KERNEL_VARIANT