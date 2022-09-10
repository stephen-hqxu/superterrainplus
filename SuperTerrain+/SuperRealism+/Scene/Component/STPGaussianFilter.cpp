#include <SuperRealism+/Scene/Component/STPGaussianFilter.h>
//Info
#include <SuperRealism+/STPRealismInfo.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>

//System
#include <memory>
#include <functional>
#include <numeric>
#include <algorithm>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/exponential.hpp>
#include <glm/gtc/constants.hpp>

using glm::uvec2;
using glm::uvec3;
using glm::vec3;
using glm::vec4;

using std::unique_ptr;
using std::make_unique;
using std::transform;

using namespace SuperTerrainPlus::STPRealism;

static constexpr auto FilterShaderFilename = 
	SuperTerrainPlus::STPFile::generateFilename(STPRealismInfo::ShaderPath, "/STPGaussianFilterKernel", ".frag");

/**
 * @brief Calculate the filter kernel extent length based on the radius.
 * @param radius The radius of the kernel.
 * @return The extent length of the kernel.
*/
inline static unsigned int calcExtentLength(unsigned int radius) {
	return radius * 2u + 1u;
}

//Gaussian standard deviation
inline static double calcGaussianStd(double variance) {
	return 1.0 / (glm::sqrt(2.0 * glm::pi<double>()) * variance);
}

inline static double calcGaussianResponseFactor(double variance) {
	return 1.0 / (2.0 * variance * variance);
}

/**
 * @brief Generate a Gaussian kernel weight table.
 * @param variance The variance of the Gaussian function.
 * @param sample_distance The distance between each sample within the kernel.
 * @param radius The radius of the kernel.
 * @param normalise Specifies if all weights in the kernel should sum up to one.
 * @return An array of Gaussian kernel weight with length of extent length of the kernel.
*/
static auto generateGaussianKernel(double variance, double sample_distance, unsigned int radius, bool normalise) {
	const unsigned int kernel_length = calcExtentLength(radius);
	//because we are using a separable filter, only a 1D kernel is needed
	unique_ptr<float[]> GaussianKernel = make_unique<float[]>(kernel_length);
	const double std_deviation = calcGaussianStd(variance),
		responseFactor = calcGaussianResponseFactor(variance);

	double kernel_sum = 0.0;
	for (int i = 0; i < static_cast<int>(kernel_length); i++) {
		const double x = sample_distance * (i - static_cast<double>(radius));
		//Gaussian probability
		const double response = -x * x * responseFactor,
			weight = std_deviation * glm::exp(response);

		GaussianKernel[i] = static_cast<float>(weight);
		kernel_sum += weight;
	}

	if (normalise) {
		float* const kernel_start = GaussianKernel.get(),
			*const kernel_end = kernel_start + kernel_length;
		transform(kernel_start, kernel_end, kernel_start,
			[&kernel_sum](auto val) { return static_cast<float>(val / kernel_sum); });
	}
	return GaussianKernel;
}

STPGaussianFilter::STPFilterExecution::STPFilterExecution(STPFilterVariant variant) :
	Variant(variant),
	Variance(1.0),
	SampleDistance(1.0),
	Radius(1u) {

}

void STPGaussianFilter::STPFilterExecution::operator()(STPProgramManager& program) const {
	if (this->Radius == 0u) {
		throw STPException::STPBadNumericRange("The radius of the filter kernel must be positive");
	}
	if (this->Variance <= 0.0f) {
		throw STPException::STPBadNumericRange("Non-positive variance is meaningless for Gaussian distribution");
	}

	//normalise naive Gaussian kernel; for bilateral filter we will normalise it in the shader.
	const unique_ptr<float[]> GaussianKernel = generateGaussianKernel(
		this->Variance, this->SampleDistance, this->Radius, this->Variant == STPFilterVariant::GaussianFilter);
	program.uniform(glProgramUniform1i, "ImgInput", 0)
		//kernel data
		.uniform(glProgramUniform1fv, "GaussianKernel", calcExtentLength(this->Radius), GaussianKernel.get())
		.uniform(glProgramUniform1ui, "KernelRadius", this->Radius);
}

STPGaussianFilter::STPGaussianFilter(const STPFilterExecution& execution, const STPScreen::STPScreenInitialiser& filter_init) :
	BorderColor(vec4(vec3(0.0f), 1.0f)) {
	//setup filter compute shader
	const char* const filter_source_file = FilterShaderFilename.data();
	//process source code
	STPShaderManager::STPShaderSource filter_source(filter_source_file, STPFile::read(filter_source_file));
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	Macro("GAUSSIAN_KERNEL_SIZE", calcExtentLength(execution.Radius))
		("GUASSIAN_KERNEL_VARIANT", static_cast<std::underlying_type_t<STPFilterVariant>>(execution.Variant));

	filter_source.define(Macro);

	STPShaderManager filter_shader(GL_FRAGMENT_SHADER);
	filter_shader(filter_source);
	this->GaussianQuad.initScreenRenderer(filter_shader, filter_init);

	//uniform
	execution(this->GaussianQuad.OffScreenRenderer);

	/* ------------------------------------- sampler for input data ---------------------------------- */
	this->InputImageSampler.wrap(GL_CLAMP_TO_BORDER);
	this->InputImageSampler.borderColor(this->BorderColor);
	this->InputImageSampler.filter(GL_NEAREST, GL_LINEAR);

	this->InputDepthSampler.wrap(GL_CLAMP_TO_EDGE);
	this->InputDepthSampler.filter(GL_NEAREST, GL_NEAREST);
}

void STPGaussianFilter::setFilterCacheDimension(STPTexture* stencil, uvec2 dimension) {
	this->IntermediateCache.setScreenBuffer(stencil, dimension, GL_R8);
}

void STPGaussianFilter::setBorderColor(vec4 border) {
	this->BorderColor = border;
	this->InputImageSampler.borderColor(this->BorderColor);
}

void STPGaussianFilter::filter(
	const STPTexture& depth, const STPTexture& input, STPFrameBuffer& output, bool output_blending) const {
	//only the input texture data requires sampler
	//output is an image object
	const STPSampler::STPSamplerUnitStateManager input_sampler_mgr[2] = {
		this->InputImageSampler.bindManaged(0u),
		this->InputDepthSampler.bindManaged(1u)
	};
	//clear old intermediate cache because convolutional filter reads data from neighbour pixels
	this->IntermediateCache.clearScreenBuffer(this->BorderColor);

	const STPScreen::STPScreenProgramExecutor perform_gaussian = this->GaussianQuad.drawScreenFromExecutor();
	GLuint filter_pass;
	/* ----------------------------- horizontal pass -------------------------------- */
	//for horizontal pass, read input from user and store output to the first buffer
	input.bind(0u);
	depth.bind(1u);
	this->IntermediateCache.capture();

	//enable horizontal filter subroutine
	filter_pass = 0u;
	glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &filter_pass);

	perform_gaussian();

	/* ------------------------------ vertical pass --------------------------------- */
	//for vertical pass, read input from the first buffer and output to the user-specified framebuffer
	this->IntermediateCache.ScreenColor.bind(0u);
	output.bind(GL_FRAMEBUFFER);

	//enable vertical filter subroutine
	filter_pass = 1u;
	glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &filter_pass);

	if (output_blending) {
		glEnable(GL_BLEND);

		perform_gaussian();

		glDisable(GL_BLEND);
	} else {
		perform_gaussian();
	}
}

#define FILTER_KERNEL_NAME(VAR) STPGaussianFilter::STPFilterKernel<STPGaussianFilter::STPFilterVariant::VAR>
#define FILTER_KERNEL_UNIFORM(VAR) void FILTER_KERNEL_NAME(VAR)::operator()(STPProgramManager& program) const

FILTER_KERNEL_NAME(GaussianFilter)::STPFilterKernel() : STPFilterExecution(STPFilterVariant::GaussianFilter) {

}

FILTER_KERNEL_NAME(BilateralFilter)::STPFilterKernel() :
	STPFilterExecution(STPFilterVariant::BilateralFilter), Sharpness(1.0f) {

}

FILTER_KERNEL_UNIFORM(BilateralFilter) {
	this->STPFilterExecution::operator()(program);

	program.uniform(glProgramUniform1i, "ImgDepth", 1)
		.uniform(glProgramUniform1f, "StandardDeviation", static_cast<float>(calcGaussianStd(this->Variance)))
		.uniform(glProgramUniform1f, "InvTwoVarSqr", static_cast<float>(calcGaussianResponseFactor(this->Variance)))
		.uniform(glProgramUniform1f, "Sharpness", this->Sharpness);
}