#include <SuperRealism+/Scene/Component/STPAmbientOcclusion.h>
//Info
#include <SuperRealism+/STPRealismInfo.h>
//
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

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
	STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPAmbientOcclusion", ".frag");

STPAmbientOcclusion::STPOcclusionKernelInstance::STPOcclusionKernelInstance
	(const STPEnvironment::STPOcclusionKernelSetting& kernel_setting, STPOcclusionAlgorithm algorithm) : Occluder(algorithm), Kernel(kernel_setting) {
	if (!this->Kernel.validate()) {
		throw STPException::STPInvalidEnvironment("Occlusion kernel setting cannot be validated");
	}
}

STPAmbientOcclusion::STPAmbientOcclusion(const STPOcclusionKernelInstance& kernel_instance, STPGaussianFilter&& filter, const STPScreenInitialiser& kernel_init) :
	STPScreen(*kernel_init.SharedVertexBuffer), RandomRotationVector(GL_TEXTURE_2D), 
	NoiseDimension(kernel_instance.Kernel.RotationVectorSize), BlurWorker(std::move(filter)) {
	const char* const ssao_source_file = SSAOShaderFilename.data();
	STPShaderManager::STPShaderSource ssao_source(ssao_source_file, *STPFile(ssao_source_file));

	//setup ao fragment shader
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	kernel_instance.compilerOption(Macro);
	Macro("AO_ALGORITHM", static_cast<std::underlying_type_t<STPOcclusionAlgorithm>>(kernel_instance.Occluder));

	ssao_source.define(Macro);
	this->initScreenRenderer(ssao_source, kernel_init);

	const auto& kernel_setting = kernel_instance.Kernel;
	/* --------------------------------- setup random number generators ------------------------------ */
	//setup random number generators
	uniform_real_distribution dist(0.0f, 1.0f);//generates random floats between 0.0 and 1.0
	mt19937_64 rngMachine(kernel_setting.RandomSampleSeed);
	const STPOcclusionKernelInstance::STPKernelRNG rng = std::bind(dist, rngMachine);

	/* ------------------------------------------ setup texture ------------------------------------ */
	this->GBufferSampler.wrap(GL_CLAMP_TO_EDGE);
	this->GBufferSampler.filter(GL_NEAREST, GL_NEAREST);

	//prepare data for random rotation vector
	kernel_instance.rotationVector(this->RandomRotationVector, rng);
	this->RandomRotationVector.filter(GL_NEAREST, GL_NEAREST);
	this->RandomRotationVector.wrap(GL_REPEAT);
	//create handle
	this->RandomRotationVectorHandle.emplace(this->RandomRotationVector);

	/* ------------------------------------------ setup uniform ------------------------------------- */
	this->OffScreenRenderer.uniform(glProgramUniform1f, "KernelRadius", kernel_setting.SampleRadius)
		.uniform(glProgramUniform1f, "SampleDepthBias", kernel_setting.Bias)
		//sampler setup
		.uniform(glProgramUniform1i, "GeoDepth", 0)
		.uniform(glProgramUniform1i, "GeoNormal", 1)
		.uniform(glProgramUniformHandleui64ARB, "RandomRotationVector", **this->RandomRotationVectorHandle);
	
	//setup kernel data based on chosen algorithm
	kernel_instance.uniformKernel(this->OffScreenRenderer, rng);

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
	this->OffScreenRenderer.uniform(glProgramUniform2fv, "RotationVectorScale", 1, value_ptr(noise_scale));
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

#define AO_KERNEL_NAME(ALG) STPAmbientOcclusion::STPOcclusionKernel<STPAmbientOcclusion::STPOcclusionAlgorithm::ALG>
#define AO_KERNEL_CSTR(ALG) AO_KERNEL_NAME(ALG)::STPOcclusionKernel(const STPEnvironment::STPOcclusionKernelSetting& kernel)
#define AO_KERNEL_VEC(ALG) void AO_KERNEL_NAME(ALG)::rotationVector(STPTexture& texture, const STPKernelRNG& rng) const
#define AO_KERNEL_OPT(ALG) void AO_KERNEL_NAME(ALG)::compilerOption(STPKernelOption& option) const
#define AO_KERNEL_UNI(ALG) void AO_KERNEL_NAME(ALG)::uniformKernel(STPProgramManager& program, const STPKernelRNG& rng) const

AO_KERNEL_CSTR(SSAO) : STPOcclusionKernelInstance(kernel, STPOcclusionAlgorithm::SSAO), KernelSize(1u) {

}

AO_KERNEL_VEC(SSAO) {
	//noise texture generation
	const uvec2& rotVecDim = this->Kernel.RotationVectorSize;
	const unsigned int RotVecCount = rotVecDim.x * rotVecDim.y;
	unique_ptr<vec2[]> RotVec = make_unique<vec2[]>(RotVecCount);
	std::generate_n(RotVec.get(), RotVecCount, [&rng]() {
		//rotate around z-axis (in tangent space)
		return vec2(
			rng(),
			rng()
			//z component is zero, it can be ignored to save memory
		) * 2.0f - 1.0f;
	});

	//submit data to texture
	const uvec3 rotVecTexDim = uvec3(rotVecDim, 1u);
	//the 2D random vector should be normalised because it represents a rotation direction.
	texture.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RG16_SNORM, rotVecTexDim);
	texture.textureSubImage<STPTexture::STPDimension::TWO>(0, ivec3(0), rotVecTexDim, GL_RG, GL_FLOAT, RotVec.get());
}

AO_KERNEL_OPT(SSAO) {
	if (this->KernelSize == 0u) {
		throw STPException::STPBadNumericRange("The ambient occlusion kernel must have positive size");
	}

	option("AO_KERNEL_SAMPLE_SIZE", this->KernelSize);
}

AO_KERNEL_UNI(SSAO) {
	constexpr static auto lerp = [](float a, float b, float f) constexpr -> float {
		return a + f * (b - a);
	};

	//kernel sample generation
	const size_t ssaoKernel_size = this->KernelSize;
	unique_ptr<vec3[]> ssaoKernel = make_unique<vec3[]>(ssaoKernel_size);
	for (unsigned int i = 0u; i < ssaoKernel_size; i++) {
		//get the current sample to be generated
		vec3& sample = ssaoKernel[i];

		sample = normalize(vec3(
			vec2(
				rng(),
				rng()
			) * 2.0f - 1.0f,
			//For SSAO, sample in a hemisphere because samples below the tangent plane are clipped through the ground,
			//so we only need to sample areas above the ground surface.
			//In tangent space, z-component range [0, 1] is above the surface whereas [-1, 0] is below.
			rng()
		));
		sample *= rng();

		//scale samples such that they are more aligned to centre of the kernel
		const float scale = 1.0f * i / (1.0f * ssaoKernel_size);
		sample *= lerp(0.1f, 1.0f, scale * scale);
	}

	//send samples to the program
	program.uniform(glProgramUniform3fv, "KernelSample", static_cast<GLsizei>(ssaoKernel_size), value_ptr(ssaoKernel[0]));
}

AO_KERNEL_CSTR(HBAO) : STPOcclusionKernelInstance(kernel, STPOcclusionAlgorithm::HBAO), DirectionStep(1u), RayStep(1u) {

}

AO_KERNEL_VEC(HBAO) {
	if (this->DirectionStep == 0u || this->RayStep == 0u) {
		throw STPException::STPBadNumericRange("The number of step for ambient occlusion kernel should be positive");
	}

	//for HBAO the random vector texture contains 2-component rotation vector and a random number as a random starting point for ray marching.
	const uvec2& randVecDim = this->Kernel.RotationVectorSize;
	const unsigned int randVecCount = randVecDim.x * randVecDim.y;
	unique_ptr<vec3[]> randVec = make_unique<vec3[]>(randVecCount);
	std::generate_n(randVec.get(), randVecCount, [&rng, dir_step = static_cast<float>(this->DirectionStep)]() {
		//Use random rotation angles in [0, 2PI/DirectionStep)
		const float angle = 2.0f * glm::pi<float>() * rng() / dir_step;

		return vec3(
			glm::cos(angle),
			glm::sin(angle),
			rng()
		);
	});

	//send to texture
	const uvec3 randVecTexDim = uvec3(randVecDim, 1u);
	texture.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB16_SNORM, randVecTexDim);
	texture.textureSubImage<STPTexture::STPDimension::TWO>(0, ivec3(0), randVecTexDim, GL_RGB, GL_FLOAT, randVec.get());
}

AO_KERNEL_UNI(HBAO) {
	program.uniform(glProgramUniform1ui, "DirectionStep", this->DirectionStep)
		.uniform(glProgramUniform1ui, "RayStep", this->RayStep);
}