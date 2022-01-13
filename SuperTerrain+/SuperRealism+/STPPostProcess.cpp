#include <SuperRealism+/Renderer/STPPostProcess.h>
#include <SuperRealism+/STPRealismInfo.h>
//Command
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>

//System
#include <array>
#include <type_traits>

using std::array;
using std::underlying_type_t;

using glm::vec2;
using glm::uvec2;
using glm::uvec3;
using glm::vec3;
using glm::ivec4;
using glm::vec4;

using SuperTerrainPlus::STPFile;
using SuperTerrainPlus::SuperRealismPlus_ShaderPath;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto OffScreenShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPPostProcess", ".vert", ".frag");

constexpr static array<signed char, 16ull> QuadVertex = {
	//Position		//TexCoord
	-1, +1,			0, 1,
	-1, -1,			0, 0,
	+1, -1,			1, 0,
	+1, +1,			1, 1
};
constexpr static array<unsigned char, 6ull> QuadIndex = {
	0, 1, 2,
	0, 2, 3
};
constexpr static STPIndirectCommand::STPDrawElement QuadDrawCommand = {
	static_cast<unsigned int>(QuadIndex.size()),
	1u,
	0u,
	0u,
	0u
};

STPPostProcess::STPToneMappingCurve::STPToneMappingCurve(STPToneMappingFunction function) : Function(function) {

}

STPPostProcess::STPPostProcess(const STPToneMappingCurve& tone_mapping, STPPostProcessLog& log) : Resolution(0u) {
	//send of off screen quad
	this->ScreenBuffer.bufferStorageSubData(QuadVertex.data(), QuadVertex.size() * sizeof(signed char), GL_NONE);
	this->ScreenIndex.bufferStorageSubData(QuadIndex.data(), QuadIndex.size() * sizeof(unsigned char), GL_NONE);
	//rendering command
	this->ScreenRenderCommand.bufferStorageSubData(&QuadDrawCommand, sizeof(QuadDrawCommand), GL_NONE);
	//vertex array
	STPVertexArray::STPVertexAttributeBuilder attr = this->ScreenArray.attribute();
	attr.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.vertexBuffer(this->ScreenBuffer, 0)
		.elementBuffer(this->ScreenIndex)
		.binding();
	this->ScreenArray.enable(0u, 2u);

	//setup post process shader
	STPShaderManager postprocess_shader[OffScreenShaderFilename.size()] = 
		{ GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
	for (unsigned int i = 0u; i < OffScreenShaderFilename.size(); i++) {
		STPShaderManager& current_shader = postprocess_shader[i];
		const char* const offscreen_filename = OffScreenShaderFilename[i].data();

		const STPFile shader_source = STPFile(offscreen_filename);
		if (i == 1u) {
			//fragment shader
			STPShaderManager::STPMacroValueDictionary Macro;

			//define post process macros
			Macro("TONE_MAPPING", static_cast<underlying_type_t<STPToneMappingFunction>>(tone_mapping.Function));

			//update macros in the source code
			current_shader.cache(*shader_source);
			current_shader.defineMacro(Macro);
			log.Log[i] = current_shader();
		}
		else {
			log.Log[i] = current_shader(*shader_source);
		}

		//add to program
		this->PostProcessor.attach(current_shader);
	}
	//program link
	log.Log[2] = this->PostProcessor.finalise();
	if (!this->PostProcessor) {
		throw STPException::STPGLError("Post processor program has error during compilation");
	}

	/* -------------------------------- setup sampler ---------------------------------- */
	this->RenderingSampler.filter(GL_NEAREST, GL_NEAREST);
	this->RenderingSampler.wrap(GL_CLAMP_TO_BORDER);
	this->RenderingSampler.borderColor(vec4(vec3(0.0f), 1.0f));

	/* -------------------------------- setup uniform ---------------------------------- */
	this->PostProcessor.uniform(glProgramUniform1i, "ScreenBuffer", 0);
	//prepare for tone mapping function definition
	tone_mapping(this->PostProcessor);
}

void STPPostProcess::capture() {
	this->SampleContainer.bind(GL_FRAMEBUFFER);
}

void STPPostProcess::setResolution(unsigned int sample, uvec2 resolution) {
	if (sample == 0u) {
		throw STPException::STPBadNumericRange("The number of sample should be greater than zero");
	}
	this->Resolution = resolution;

	const uvec3 dimension = uvec3(this->Resolution, 1u);
	//(re)allocate memory for texture
	STPTexture msTexture(GL_TEXTURE_2D_MULTISAMPLE), imageTexture(GL_TEXTURE_2D);
	//using a floating-point format allows color to go beyond the standard range of [0.0, 1.0]
	msTexture.textureStorageMultisample<STPTexture::STPDimension::TWO>(sample, GL_RGB16F, dimension, GL_TRUE);
	imageTexture.textureStorage<STPTexture::STPDimension::TWO>(1, GL_RGB16F, dimension);

	//do the same for render buffer
	STPRenderBuffer msBuffer;
	msBuffer.renderbufferStorageMultisample(sample, GL_DEPTH_COMPONENT24, this->Resolution);

	//attach to framebuffer
	STPFrameBuffer::unbind(GL_FRAMEBUFFER);
	this->SampleContainer.attach(GL_COLOR_ATTACHMENT0, msTexture, 0);
	this->SampleContainer.attach(GL_DEPTH_ATTACHMENT, msBuffer);

	this->PostProcessContainer.attach(GL_COLOR_ATTACHMENT0, imageTexture, 0);

	//verify
	if (this->SampleContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE 
		|| this->PostProcessContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw STPException::STPGLError("Post process framebuffer validation fails");
	}

	using std::move;
	//re-initialise the current objects
	this->RenderingSample.emplace(move(msTexture));
	this->RenderingImage.emplace(move(imageTexture));
	this->PostProcessBuffer.emplace(move(msBuffer));
}

#define SET_EFFECT(EFF, NAME) template<> STP_REALISM_API void STPPostProcess::setEffect<STPPostProcess::STPPostEffect::EFF>(float val) { \
	this->PostProcessor.uniform(glProgramUniform1f, NAME, val); \
}

SET_EFFECT(Gamma, "Gamma")

void STPPostProcess::clear() {
	static constexpr vec4 NoColor = vec4(vec3(0.0f), 1.0f);
	//clear color and depth buffer of all frame buffer
	this->SampleContainer.clearColor(0, NoColor);
	//the default clear depth value is 1.0
	this->SampleContainer.clearDepth(1.0f);
	//no need to clear the display framebuffer because it will be overwritten later anyway.
}

void STPPostProcess::operator()() {
	//multisample resolve
	const ivec4 bound = ivec4(0, 0, this->Resolution);
	this->PostProcessContainer.blitFrom(this->SampleContainer, bound, bound, GL_COLOR_BUFFER_BIT, GL_NEAREST);

	//prepare to render
	this->RenderingImage->bind(0);
	this->RenderingSampler.bind(0);

	this->ScreenArray.bind();
	this->ScreenRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);
	this->PostProcessor.use();

	glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, nullptr);

	STPProgramManager::unuse();
	//must unbind sampler otherwise it will affect texture in other renderer.
	STPSampler::unbind(0);
}

#define TONE_MAPPING_NAME(FUNC) STPPostProcess::STPToneMappingDefinition<STPPostProcess::STPToneMappingFunction::FUNC>
#define TONE_MAPPING_DEF(FUNC) void TONE_MAPPING_NAME(FUNC)::operator()(STPProgramManager& program) const

TONE_MAPPING_NAME(Disable)::STPToneMappingDefinition() : STPToneMappingCurve(STPToneMappingFunction::Disable) {

}

//For disabled tone mapping function, nothing needs to be defined in the functor.

TONE_MAPPING_NAME(GranTurismo)::STPToneMappingDefinition() : STPToneMappingCurve(STPToneMappingFunction::GranTurismo), 
	MaxBrightness(1.0f), Contrast(1.0f), LinearStart(0.22f), LinearLength(0.4f), BlackTightness(1.33f), Pedestal(0.0f) {

}

TONE_MAPPING_DEF(GranTurismo) {
	const float l0 = ((this->MaxBrightness - this->LinearStart) * this->LinearLength) / this->Contrast,
		S0 = this->LinearStart + l0,
		S1 = this->LinearStart + this->Contrast * l0,
		C2 = (this->Contrast * this->MaxBrightness) / (this->MaxBrightness - S1),
		CP = -C2 / this->MaxBrightness;

	program.uniform(glProgramUniform1f, "Tone.P", this->MaxBrightness)
		.uniform(glProgramUniform1f, "Tone.a", this->Contrast)
		.uniform(glProgramUniform1f, "Tone.m", this->LinearStart)
		.uniform(glProgramUniform1f, "Tone.c", this->BlackTightness)
		.uniform(glProgramUniform1f, "Tone.b", this->Pedestal)
		
		.uniform(glProgramUniform1f, "Tone.l0", l0)
		.uniform(glProgramUniform1f, "Tone.S0", S0)
		.uniform(glProgramUniform1f, "Tone.S1", S1)
		.uniform(glProgramUniform1f, "Tone.CP", CP);
}

TONE_MAPPING_NAME(Lottes)::STPToneMappingDefinition() : STPToneMappingCurve(STPToneMappingFunction::Lottes), 
	Contrast(1.6f), Shoulder(0.977f), HDRMax(8.0f), Middle(vec2(0.18f, 0.267f)) {

}

TONE_MAPPING_DEF(Lottes) {
	//product of contrast and shoulder
	const float product_ad = this->Contrast * this->Shoulder;
	const float b =
		(-glm::pow(this->Middle.x, this->Contrast) + glm::pow(this->HDRMax, this->Contrast) * this->Middle.y) /
		((glm::pow(this->HDRMax, product_ad) - glm::pow(this->Middle.x, product_ad)) * this->Middle.y),
				c =
		(glm::pow(this->HDRMax, product_ad) * glm::pow(this->Middle.x, this->Contrast) - glm::pow(this->HDRMax, this->Contrast) * glm::pow(this->Middle.x, product_ad) * this->Middle.y) /
		((glm::pow(this->HDRMax, product_ad) - glm::pow(this->Middle.x, product_ad)) * this->Middle.y);

	program.uniform(glProgramUniform1f, "Tone.a", this->Contrast)
		.uniform(glProgramUniform1f, "Tone.b", b)
		.uniform(glProgramUniform1f, "Tone.c", c)
		.uniform(glProgramUniform1f, "Tone.d", this->Shoulder);
}

TONE_MAPPING_NAME(Uncharted2)::STPToneMappingDefinition() : STPToneMappingCurve(STPToneMappingFunction::Uncharted2), 
	ShoulderStrength(0.22f), LinearStrength(0.3f), LinearAngle(0.1f), ToeStrength(0.2f), ToeNumerator(0.01f), ToeDenominator(0.3f), LinearWhite(11.2f) {

}

TONE_MAPPING_DEF(Uncharted2) {
	//compute mapped linear white point value.
	const float mW = 
		((this->LinearWhite * (this->ShoulderStrength * this->LinearWhite + this->LinearAngle * this->LinearStrength) + this->ToeStrength * this->ToeNumerator) / 
			(this->LinearWhite * (this->ShoulderStrength * this->LinearWhite + this->LinearStrength) + this->ToeStrength * this->ToeDenominator)) - this->ToeNumerator / this->ToeDenominator;

	program.uniform(glProgramUniform1f, "Tone.A", this->ShoulderStrength)
		.uniform(glProgramUniform1f, "Tone.B", this->LinearStrength)
		.uniform(glProgramUniform1f, "Tone.C", this->LinearAngle)
		.uniform(glProgramUniform1f, "Tone.D", this->ToeStrength)
		.uniform(glProgramUniform1f, "Tone.E", this->ToeNumerator)
		.uniform(glProgramUniform1f, "Tone.F", this->ToeDenominator)
		.uniform(glProgramUniform1f, "Tone.mW", mW);
}