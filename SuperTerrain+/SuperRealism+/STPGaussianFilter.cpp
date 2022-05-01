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

using SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPRealism;

static constexpr auto FilterShaderFilename = 
	STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPGaussianFilterKernel", ".frag");

STPGaussianFilter::STPGaussianFilter(double variance, double sample_distance, unsigned int radius, const STPScreenInitialiser& filter_init) :
	BorderColor(vec4(vec3(0.0f), 1.0f)) {
	if (radius == 0u) {
		throw STPException::STPBadNumericRange("The radius of the filter kernel must be positive");
	}
	if (variance <= 0.0f) {
		throw STPException::STPBadNumericRange("Non-positive variance is meaningless for Gaussian distribution");
	}
	const unsigned int kernel_length = radius * 2u + 1u;

	//setup filter compute shader
	const char* const filter_source_file = FilterShaderFilename.data();
	//process source code
	STPShaderManager::STPShaderSource filter_source(filter_source_file, *STPFile(filter_source_file));
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	Macro("GAUSSIAN_KERNEL_SIZE", kernel_length);

	filter_source.define(Macro);

	STPShaderManager filter_shader(GL_FRAGMENT_SHADER);
	filter_shader(filter_source);
	this->initScreenRenderer(filter_shader, filter_init);

	/* -------------------------------------- Gaussian filter kernel generation ---------------------------- */
	//because we are using a separable filter, only a 1D kernel is needed
	unique_ptr<float[]> GaussianKernel = make_unique<float[]>(kernel_length);
	for (int i = 0; i < static_cast<int>(kernel_length); i++) {
		const double x = sample_distance * (i - static_cast<double>(radius));

		//Gaussian probability
		const double a = 1.0 / (glm::sqrt(2.0 * glm::pi<double>()) * variance),
			b = -x * x / (2.0 * variance * variance);
		GaussianKernel[i] = static_cast<float>(a * glm::exp(b));
	}

	//normalisation
	float* const kernel_start = GaussianKernel.get(), *const kernel_end = kernel_start + kernel_length;
	transform(kernel_start, kernel_end, kernel_start, 
		[kernel_sum = std::accumulate(kernel_start, kernel_end, 0.0f, std::plus<float>())](auto val) { return val / kernel_sum; });

	/* ------------------------------------------- uniform ------------------------------------------ */
	this->OffScreenRenderer.uniform(glProgramUniform1i, "ImgInput", 0)
		.uniform(glProgramUniform1i, "FilterOutput", 1)
		//kernel data
		.uniform(glProgramUniform1fv, "GaussianKernel", kernel_length, GaussianKernel.get())
		.uniform(glProgramUniform1ui, "KernelRadius", radius);

	/* ------------------------------------- sampler for input data ---------------------------------- */
	this->InputImageSampler.wrap(GL_CLAMP_TO_BORDER);
	this->InputImageSampler.borderColor(this->BorderColor);
	this->InputImageSampler.filter(GL_NEAREST, GL_LINEAR);
}

void STPGaussianFilter::setFilterCacheDimension(STPTexture* stencil, uvec2 dimension) {
	this->IntermediateCache.setScreenBuffer(stencil, dimension, GL_R8);
}

void STPGaussianFilter::setBorderColor(vec4 border) {
	this->BorderColor = border;
	this->InputImageSampler.borderColor(this->BorderColor);
}

void STPGaussianFilter::filter(const STPTexture& input, STPFrameBuffer& output) const {
	//only the input texture data requires sampler
	//output is an image object
	this->InputImageSampler.bind(0u);
	//clear old intermediate cache because convolutional filter reads data from neighbour pixels
	this->IntermediateCache.clearScreenBuffer(this->BorderColor);

	this->ScreenVertex->bind();
	this->OffScreenRenderer.use();
	GLuint filter_pass;
	/* ----------------------------- horizontal pass -------------------------------- */
	//for horizontal pass, read input from user and store output to the first buffer
	input.bind(0u);
	this->IntermediateCache.capture();

	//enable horizontal filter subroutine
	filter_pass = 0u;
	glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &filter_pass);

	this->drawScreen();

	/* ------------------------------ vertical pass --------------------------------- */
	//for vertical pass, read input from the first buffer and output to the user-specified framebuffer
	this->IntermediateCache.ScreenColor.bind(0u);
	output.bind(GL_FRAMEBUFFER);

	//enable vertical filter subroutine
	filter_pass = 1u;
	glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &filter_pass);

	this->drawScreen();

	/* ------------------------------------------------------------------------------ */
	//clear up
	STPProgramManager::unuse();
	STPSampler::unbind(0u);
	STPTexture::unbindImage(1u);
}