#include <SuperRealism+/Scene/Component/STPAmbientOcclusion.h>
//Info
#include <SuperRealism+/STPRealismInfo.h>
//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

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
using glm::vec4;
using glm::normalize;
using glm::value_ptr;

using SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto SSAOShaderFilename = 
	STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPScreenSpaceAmbientOcclusion", ".frag");

STPAmbientOcclusion::STPAmbientOcclusion(const STPEnvironment::STPOcclusionKernelSetting& kernel_setting, STPGaussianFilter&& filter, 
	const STPScreenInitialiser& kernel_init) :
	STPScreen(*kernel_init.SharedVertexBuffer), RandomRotationVector(GL_TEXTURE_2D), NoiseDimension(kernel_setting.RotationVectorSize), BlurWorker(std::move(filter)) {
	if (!kernel_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Occlusion kernel setting cannot be validated");
	}
	constexpr static auto lerp = [](float a, float b, float f) constexpr -> float {
		return a + f * (b - a);
	};

	const char* const ssao_source_file = SSAOShaderFilename.data();
	STPShaderManager::STPShaderSource ssao_source(ssao_source_file, *STPFile(ssao_source_file));

	//setup ao fragment shader
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	Macro("SSAO_KERNEL_SAMPLE_SIZE", kernel_setting.KernelSize);

	ssao_source.define(Macro);
	this->initScreenRenderer(ssao_source, kernel_init);

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
	unique_ptr<vec2[]> ssaoRotVec = make_unique<vec2[]>(ssaoRotVec_size);
	std::generate_n(ssaoRotVec.get(), ssaoRotVec_size, [&next_random]() {
		//rotate around z-axis (in tangent space)
		return normalize(vec2(
			next_random() * 2.0f - 1.0f,
			next_random() * 2.0f - 1.0f
			//z component is zero, it can be ignored to save memory
		));
	});

	/* ------------------------------------------ setup texture ------------------------------------ */
	this->GBufferSampler.wrap(GL_CLAMP_TO_EDGE);
	this->GBufferSampler.filter(GL_NEAREST, GL_NEAREST);

	const uvec3 rotVecDim = uvec3(rotVec, 1.0f);
	this->RandomRotationVector.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RG16_SNORM, rotVecDim);
	this->RandomRotationVector.textureSubImage<STPTexture::STPDimension::TWO>(0, ivec3(0), rotVecDim, GL_RG, GL_FLOAT, ssaoRotVec.get());
	this->RandomRotationVector.filter(GL_NEAREST, GL_NEAREST);
	this->RandomRotationVector.wrap(GL_REPEAT);
	//create handle
	this->RandomRotationVectorHandle.emplace(this->RandomRotationVector);

	/* ------------------------------------------ setup uniform ------------------------------------- */
	this->OffScreenRenderer.uniform(glProgramUniform3fv, "KernelSample", static_cast<GLsizei>(ssaoKernel_size), value_ptr(ssaoKernel[0]))
		.uniform(glProgramUniform1f, "KernelRadius", kernel_setting.SampleRadius)
		.uniform(glProgramUniform1f, "SampleDepthBias", kernel_setting.Bias)
		//sampler setup
		.uniform(glProgramUniform1i, "GeoDepth", 0)
		.uniform(glProgramUniform1i, "GeoNormal", 1)
		.uniform(glProgramUniformHandleui64ARB, "NoiseVector", **this->RandomRotationVectorHandle);

	/* ------------------------------------------- setup output -------------------------------------- */
	//for ambient occlusion, "no data" should be 1.0
	this->BlurWorker.setBorderColor(vec4(1.0f));
}

void STPAmbientOcclusion::setScreenSpace(STPTexture* stencil, uvec2 dimension) {
	this->OcclusionResultContainer.setScreenBuffer(stencil, dimension, GL_R8);
	this->BlurWorker.setFilterCacheDimension(stencil, dimension);

	//update uniform
	//tile noise texture over screen based on screen dimensions divided by noise size
	const vec2 noise_scale = static_cast<vec2>(dimension) / static_cast<vec2>(this->NoiseDimension);
	this->OffScreenRenderer.uniform(glProgramUniform2fv, "NoiseTexScale", 1, value_ptr(noise_scale));
}

void STPAmbientOcclusion::occlude(const STPTexture& depth, const STPTexture& normal, STPFrameBuffer& output) const {
	//binding
	depth.bind(0);
	normal.bind(1);
	this->GBufferSampler.bind(0);
	this->GBufferSampler.bind(1);

	this->ScreenVertex->bind();
	this->OffScreenRenderer.use();

	//we need to clear the old ambient occlusion data
	//because when we blur it later, we might accidentally read the old data which were culled due to stencil testing
	this->OcclusionResultContainer.clearScreenBuffer(vec4(1.0f));
	//capture data into the internal framebuffer
	this->OcclusionResultContainer.capture();
	this->drawScreen();

	//clear up for ambient occlusion stage so it won't overwrite state later
	STPProgramManager::unuse();
	STPSampler::unbind(0);
	STPSampler::unbind(1);

	//blur the output to reduce noise
	this->BlurWorker.filter(this->OcclusionResultContainer.ScreenColor, output);
}