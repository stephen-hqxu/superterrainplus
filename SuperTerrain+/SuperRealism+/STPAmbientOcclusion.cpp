#include <SuperRealism+/Scene/Component/STPAmbientOcclusion.h>
//Info
#include <SuperRealism+/STPRealismInfo.h>
//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//File Reader
#include <SuperTerrain+/Utility/STPFile.h>

//GLAD
#include <glad/glad.h>

//System
#include <random>
#include <memory>
#include <functional>
#include <algorithm>

//GLM
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>

using std::uniform_real_distribution;
using std::mt19937_64;
using std::unique_ptr;
using std::make_unique;

using glm::uvec2;
using glm::vec2;
using glm::ivec3;
using glm::uvec3;
using glm::vec3;
using glm::normalize;
using glm::value_ptr;

using SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto SSAOShaderFilename = 
	STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPScreenSpaceAmbientOcclusion", ".frag");

STPAmbientOcclusion::STPAmbientOcclusion(const STPEnvironment::STPOcclusionKernelSetting& kernel_setting, STPAmbientOcclusionLog& log) :
	RandomRotationVector(GL_TEXTURE_2D), OcclusionResult(GL_TEXTURE_2D), NoiseDimension(kernel_setting.RotationVectorSize) {
	if (!kernel_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Occlusion kernel setting cannot be validated");
	}
	constexpr static auto lerp = [](float a, float b, float f) constexpr -> float {
		return a + f * (b - a);
	};

	STPShaderManager screen_shader(std::move(STPScreen::compileScreenVertexShader(log.QuadShader))),
		ao_shader(GL_FRAGMENT_SHADER);
	const char* const ssao_source_file = SSAOShaderFilename.data();
	STPShaderManager::STPShaderSource ssao_source(ssao_source_file, *STPFile(ssao_source_file));

	//setup ao fragment shader
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	Macro("SSAO_KERNEL_SAMPLE_SIZE", kernel_setting.KernelSize);

	ssao_source.define(Macro);
	log.AOShader.Log[0] = ao_shader(ssao_source);

	//setup program
	log.AOShader.Log[1] = this->OcclusionCalculator
		.attach(screen_shader)
		.attach(ao_shader)
		//link
		.finalise();

	/* ----------------------------------- setup AO kernel data ------------------------------------- */
	//setup random number generators
	const uniform_real_distribution dist(0.0f, 1.0f);//generates random floats between 0.0 and 1.0
	const mt19937_64 rng(kernel_setting.RandomSampleSeed);
	auto next_random = std::bind(dist, rng);

	//sample kernel generation
	const size_t ssaoKernel_size = kernel_setting.KernelSize;
	unique_ptr<vec3[]> ssaoKernel = make_unique<vec3[]>(ssaoKernel_size);
	for (unsigned int i = 0u; i < ssaoKernel_size; i++) {
		//get the current sample to be generated
		vec3& sample = ssaoKernel[i];

		sample = normalize(vec3(
			next_random() * 2.0f - 1.0f,
			next_random() * 2.0f - 1.0f,
			next_random()
		));
		sample *= next_random();

		//scale samples such that they are more aligned to centre of the kernel
		float scale = 1.0f * i / (1.0f * ssaoKernel_size);
		scale = lerp(0.1f, 1.0f, scale * scale);
		sample *= scale;
	}

	//noise texture generation
	const uvec2& rotVec = this->NoiseDimension;
	const size_t ssaoRotVec_size = rotVec.x * rotVec.y;
	unique_ptr<vec3[]> ssaoRotVec = make_unique<vec3[]>(ssaoRotVec_size);
	std::generate_n(ssaoRotVec.get(), ssaoRotVec_size, [&next_random]() {
		//rotate around z-axis (in tangent space)
		return vec3(
			next_random() * 2.0f - 1.0f,
			next_random() * 2.0f - 1.0f,
			0.0f
		);
	});

	/* ------------------------------------------ setup texture ------------------------------------ */
	this->GBufferSampler.wrap(GL_CLAMP_TO_EDGE);
	this->GBufferSampler.filter(GL_NEAREST, GL_NEAREST);

	const uvec3 rotVecDim = uvec3(rotVec, 1.0f);
	this->RandomRotationVector.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB32F, rotVecDim);
	this->RandomRotationVector.textureSubImage<STPTexture::STPDimension::TWO>(0, ivec3(0), rotVecDim, GL_RGB, GL_FLOAT, ssaoRotVec.get());
	this->RandomRotationVector.filter(GL_NEAREST, GL_NEAREST);
	this->RandomRotationVector.wrap(GL_REPEAT);
	//create handle
	this->RandomRotationVectorHandle.emplace(this->RandomRotationVector);

	/* ------------------------------------------ setup uniform ------------------------------------- */
	this->OcclusionCalculator.uniform(glProgramUniform3fv, "KernelSample", static_cast<GLsizei>(ssaoKernel_size), value_ptr(ssaoKernel[0]))
		.uniform(glProgramUniform1f, "KernelRadius", kernel_setting.SampleRadius)
		.uniform(glProgramUniform1f, "SampleDepthBias", kernel_setting.Bias)
		//sampler setup
		.uniform(glProgramUniform1i, "GeoDepth", 0)
		.uniform(glProgramUniform1i, "GeoNormal", 1)
		.uniform(glProgramUniformHandleui64ARB, "NoiseVector", **this->RandomRotationVectorHandle);
}

const STPTexture& STPAmbientOcclusion::operator*() const {
	return this->OcclusionResult;
}

void STPAmbientOcclusion::setScreenSpaceDimension(uvec2 dimension) {
	if (dimension.x == 0u || dimension.y == 0u) {
		throw STPException::STPBadNumericRange("Both component of a render target dimension must be positive");
	}
	//create new texture
	STPTexture occlusion(GL_TEXTURE_2D);
	//allocate memory
	occlusion.textureStorage<STPTexture::STPDimension::TWO>(1, GL_R8, uvec3(dimension, 1u));

	//attach new texture to framebuffer
	this->OcclusionContainer.attach(GL_COLOR_ATTACHMENT0, occlusion, 0);
	//depth buffer is not needed because we are doing off-screen rendering
	if (this->OcclusionContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw STPException::STPGLError("Ambient occlusion result framebuffer cannot be validated");
	}

	//store the new texture
	using std::move;
	this->OcclusionResult = move(occlusion);

	//update uniform
	//tile noise texture over screen based on screen dimensions divided by noise size
	const vec2 noise_scale = static_cast<vec2>(dimension) / static_cast<vec2>(this->NoiseDimension);
	this->OcclusionCalculator.uniform(glProgramUniform2fv, "NoiseTexScale", 1, value_ptr(noise_scale));
}

void STPAmbientOcclusion::occlude(const STPTexture& depth, const STPTexture& normal) const {
	//binding
	depth.bind(0);
	normal.bind(1);
	this->GBufferSampler.bind(0);
	this->GBufferSampler.bind(1);

	this->OcclusionCalculator.use();

	//capture data into the internal framebuffer
	this->OcclusionContainer.bind(GL_FRAMEBUFFER);
	//there is no need to clear the framebuffer because everything will be overdrawn
	//and there is no depth/stencil testing
	this->drawScreen();

	//finish up
	STPProgramManager::unuse();
	STPSampler::unbind(0);
	STPSampler::unbind(1);
}