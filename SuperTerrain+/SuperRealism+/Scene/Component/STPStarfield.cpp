#include <SuperRealism+/Scene/Component/STPStarfield.h>
#include <SuperRealism+/STPRealismInfo.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

#include <glad/glad.h>

using std::chrono::steady_clock;
using std::chrono::duration;

using glm::uvec3;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto StarfieldShaderFilename =
	SuperTerrainPlus::STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPStarfield", ".frag");

STPStarfield::STPStarfield(const STPStarfieldModel& starfield_model, const STPSkyboxInitialiser& starfield_init) :
	StarlightSpectrum(std::move(*starfield_model.StarlightSpectrum)) {
	//setup starfield renderer
	const char* const star_source_file = StarfieldShaderFilename.data();
	STPShaderManager::STPShaderSource star_source(star_source_file, STPFile::read(star_source_file));

	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;
	Macro("STAR_QUADRATIC_ATTENUATION", (starfield_model.UseQuadraticAttenuation ? 1 : 0));

	star_source.define(Macro);
	STPShaderManager star_fs(GL_FRAGMENT_SHADER);
	star_fs(star_source);

	this->initSkyboxRenderer(star_fs, starfield_init);

	/* -------------------------------- setup uniform --------------------------------- */
	this->ShineTimeLocation = this->SkyboxRenderer.uniformLocation("ShineTime");
	this->SkyboxRenderer.uniform(glProgramUniformHandleui64ARB, "StarColorSpectrum", this->StarlightSpectrum.spectrumHandle());
}

inline void STPStarfield::updateShineTime(double time) const {
	this->SkyboxRenderer.uniform(glProgramUniform1f, this->ShineTimeLocation, static_cast<float>(time));
}

void STPStarfield::setStarfield(const STPEnvironment::STPStarfieldSetting& starfield_setting, unsigned int rng_seed) {
	if (!starfield_setting.validate()) {
		throw STPException::STPInvalidEnvironment("The starfield setting fails to validate");
	}

	this->SkyboxRenderer.uniform(glProgramUniform1f, "Star.iLklh", starfield_setting.InitialLikelihood)
		.uniform(glProgramUniform1f, "Star.OctLklhMul", starfield_setting.OctaveLikelihoodMultiplier)
		.uniform(glProgramUniform1f, "Star.iScl", starfield_setting.InitialScale)
		.uniform(glProgramUniform1f, "Star.OctSclMul", starfield_setting.OctaveScaleMultiplier)
		.uniform(glProgramUniform1f, "Star.Thres", starfield_setting.EdgeDistanceFalloff)
		.uniform(glProgramUniform1f, "Star.spdShine", starfield_setting.ShineSpeed)
		.uniform(glProgramUniform1f, "Star.LumMul", starfield_setting.LuminosityMultiplier)
		.uniform(glProgramUniform1f, "Star.MinAlt", starfield_setting.MinimumAltitude)
		.uniform(glProgramUniform1ui, "Star.Oct", starfield_setting.Octave)
		.uniform(glProgramUniform1ui, "RandomSeed", rng_seed);

	//reset timer
	this->updateShineTime(0.0);
	this->ShineTimeStart = steady_clock::now();
}

void STPStarfield::render() const {
	//star animation
	const duration<double> elapsed = steady_clock::now() - this->ShineTimeStart;
	this->updateShineTime(elapsed.count());

	this->drawSkybox();
}