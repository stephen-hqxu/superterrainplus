#include <SuperRealism+/Scene/Component/STPPostProcess.h>
#include <SuperRealism+/STPRealismInfo.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

//GLAD
#include <glad/glad.h>

//GLM
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>

//System
#include <type_traits>

using std::underlying_type_t;

using glm::vec2;
using glm::uvec2;
using glm::uvec3;
using glm::vec3;
using glm::ivec4;
using glm::vec4;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto PostProcessShaderFilename = 
	SuperTerrainPlus::STPFile::generateFilename(STPRealismInfo::ShaderPath, "/STPPostProcess", ".frag");

STPPostProcess::STPToneMappingCurve::STPToneMappingCurve(STPToneMappingFunction function) : Function(function) {

}

STPPostProcess::STPPostProcess(const STPToneMappingCurve& tone_mapping, const STPScreen::STPScreenInitialiser& post_process_init) {
	//setup post process shader
	const char* const source_file = PostProcessShaderFilename.data();
	STPShaderManager::STPShaderSource post_source(source_file, STPFile::read(source_file));

	//fragment shader
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;
	//define post process macros
	Macro("TONE_MAPPING", static_cast<underlying_type_t<STPToneMappingFunction>>(tone_mapping.Function));
	//update macros in the source code
	post_source.define(Macro);
	
	STPShaderManager post_shader(GL_FRAGMENT_SHADER);
	post_shader(post_source);
	this->PostProcessQuad.initScreenRenderer(post_shader, post_process_init);

	/* -------------------------------- setup sampler ---------------------------------- */
	this->ImageSampler.filter(GL_NEAREST, GL_NEAREST);
	this->ImageSampler.wrap(GL_CLAMP_TO_BORDER);
	this->ImageSampler.borderColor(vec4(vec3(0.0f), 1.0f));

	/* -------------------------------- setup uniform ---------------------------------- */
	this->PostProcessQuad.OffScreenRenderer.uniform(glProgramUniform1i, "ScreenBuffer", 0);
	//prepare for tone mapping function definition
	tone_mapping(this->PostProcessQuad.OffScreenRenderer);
}

const STPTexture& STPPostProcess::operator*() const {
	return this->PostProcessResultContainer.ScreenColor;
}

#define SET_EFFECT(EFF, NAME) template<> \
	STP_REALISM_API void STPPostProcess::setEffect<STPPostProcess::STPPostEffect::EFF>(float val) { \
		this->PostProcessQuad.OffScreenRenderer.uniform(glProgramUniform1f, NAME, val); \
}

SET_EFFECT(Gamma, "Gamma")

void STPPostProcess::setPostProcessBuffer(STPTexture* stencil, uvec2 dimension) {
	this->PostProcessResultContainer.setScreenBuffer(stencil, dimension, GL_RGBA16F);
}

void STPPostProcess::capture() const {
	this->PostProcessResultContainer.capture();
}

void STPPostProcess::process() const {
	//prepare to render
	this->PostProcessResultContainer.ScreenColor.bind(0);
	const STPSampler::STPSamplerUnitStateManager color_sampler_mgr = this->ImageSampler.bindManaged(0);

	this->PostProcessQuad.drawScreen();
}

#define TONE_MAPPING_NAME(FUNC) STPPostProcess::STPToneMappingDefinition<STPPostProcess::STPToneMappingFunction::FUNC>
#define TONE_MAPPING_DEF(FUNC) void TONE_MAPPING_NAME(FUNC)::operator()(STPProgramManager& program) const

TONE_MAPPING_NAME(Disable)::STPToneMappingDefinition() : STPToneMappingCurve(STPToneMappingFunction::Disable) {

}

//For disabled tone mapping function, nothing needs to be defined in the functor.

TONE_MAPPING_NAME(GranTurismo)::STPToneMappingDefinition() :
	STPToneMappingCurve(STPToneMappingFunction::GranTurismo), 
	MaxBrightness(1.0f),
	Contrast(1.0f),
	LinearStart(0.22f),
	LinearLength(0.4f),
	BlackTightness(1.33f),
	Pedestal(0.0f) {

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

TONE_MAPPING_NAME(Lottes)::STPToneMappingDefinition() :
	STPToneMappingCurve(STPToneMappingFunction::Lottes), 
	Contrast(1.6f),
	Shoulder(0.977f),
	HDRMax(8.0f),
	Middle(vec2(0.18f, 0.267f)) {

}

TONE_MAPPING_DEF(Lottes) {
	//product of contrast and shoulder
	const float product_ad = this->Contrast * this->Shoulder,
		pow_hdr_a = glm::pow(this->HDRMax, this->Contrast),
		pow_midin_a = glm::pow(this->Middle.x, this->Contrast),
		pow_hdr_ad = glm::pow(this->HDRMax, product_ad),
		pow_midin_ad = glm::pow(this->Middle.x, product_ad);
	const float b =
		(-pow_midin_a + pow_hdr_a * this->Middle.y) / ((pow_hdr_ad - pow_midin_ad) * this->Middle.y),
				c =
		(pow_hdr_ad * pow_midin_a - pow_hdr_a * pow_midin_ad * this->Middle.y) / ((pow_hdr_ad - pow_midin_ad) * this->Middle.y);

	program.uniform(glProgramUniform1f, "Tone.a", this->Contrast)
		.uniform(glProgramUniform1f, "Tone.b", b)
		.uniform(glProgramUniform1f, "Tone.c", c)
		.uniform(glProgramUniform1f, "Tone.d", this->Shoulder);
}

TONE_MAPPING_NAME(Uncharted2)::STPToneMappingDefinition() :
	STPToneMappingCurve(STPToneMappingFunction::Uncharted2), 
	ShoulderStrength(0.22f),
	LinearStrength(0.3f),
	LinearAngle(0.1f),
	ToeStrength(0.2f),
	ToeNumerator(0.01f),
	ToeDenominator(0.3f),
	LinearWhite(11.2f) {

}

TONE_MAPPING_DEF(Uncharted2) {
	//compute mapped linear white point value.
	const float product_a_w2 = this->ShoulderStrength * this->LinearWhite * this->LinearWhite,
		mW = ((product_a_w2 + this->LinearWhite * this->LinearAngle * this->LinearStrength + this->ToeStrength * this->ToeNumerator) /
			(product_a_w2 + this->LinearWhite * this->LinearStrength + this->ToeStrength * this->ToeDenominator))
			- this->ToeNumerator / this->ToeDenominator;

	program.uniform(glProgramUniform1f, "Tone.A", this->ShoulderStrength)
		.uniform(glProgramUniform1f, "Tone.B", this->LinearStrength)
		.uniform(glProgramUniform1f, "Tone.C", this->LinearAngle)
		.uniform(glProgramUniform1f, "Tone.D", this->ToeStrength)
		.uniform(glProgramUniform1f, "Tone.E", this->ToeNumerator)
		.uniform(glProgramUniform1f, "Tone.F", this->ToeDenominator)
		.uniform(glProgramUniform1f, "Tone.mW", mW);
}