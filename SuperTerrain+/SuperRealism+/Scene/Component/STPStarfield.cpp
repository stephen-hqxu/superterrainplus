#include <SuperRealism+/Scene/Component/STPStarfield.h>
#include <SuperRealism+/STPRealismInfo.h>

#include <SuperTerrain+/Exception/STPBadNumericRange.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto StarfieldShaderFilename =
	SuperTerrainPlus::STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPStarfield", ".frag");

STPStarfield::STPStarfield(const STPStarfieldModel& starfield_model, const STPSkybox::STPSkyboxInitialiser& starfield_init) :
	StarlightSpectrum(std::move(*starfield_model.StarlightSpectrum)) {
	//setup starfield renderer
	const char* const star_source_file = StarfieldShaderFilename.data();
	STPShaderManager::STPShaderSource star_source(star_source_file, STPFile::read(star_source_file));

	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;
	Macro("STAR_QUADRATIC_ATTENUATION", (starfield_model.UseQuadraticAttenuation ? 1 : 0));

	star_source.define(Macro);
	STPShaderManager star_fs(GL_FRAGMENT_SHADER);
	star_fs(star_source);

	this->StarfieldBox.initSkyboxRenderer(star_fs, starfield_init);

	/* -------------------------------- setup uniform --------------------------------- */
	this->ShineTimeLocation = this->StarfieldBox.SkyboxRenderer.uniformLocation("ShineTime");
	this->StarfieldBox.SkyboxRenderer.uniform(glProgramUniformHandleui64ARB, "StarColorSpectrum", this->StarlightSpectrum.spectrumHandle());
}

void STPStarfield::setStarfield(const STPEnvironment::STPStarfieldSetting& starfield_setting, unsigned int rng_seed) {
	starfield_setting.validate();

	this->StarfieldBox.SkyboxRenderer.uniform(glProgramUniform1f, "Star.iLklh", starfield_setting.InitialLikelihood)
		.uniform(glProgramUniform1f, "Star.OctLklhMul", starfield_setting.OctaveLikelihoodMultiplier)
		.uniform(glProgramUniform1f, "Star.iScl", starfield_setting.InitialScale)
		.uniform(glProgramUniform1f, "Star.OctSclMul", starfield_setting.OctaveScaleMultiplier)
		.uniform(glProgramUniform1f, "Star.Thres", starfield_setting.EdgeDistanceFalloff)
		.uniform(glProgramUniform1f, "Star.spdShine", starfield_setting.ShineSpeed)
		.uniform(glProgramUniform1f, "Star.LumMul", starfield_setting.LuminosityMultiplier)
		.uniform(glProgramUniform1f, "Star.MinAlt", starfield_setting.MinimumAltitude)
		.uniform(glProgramUniform1ui, "Star.Oct", starfield_setting.Octave)
		.uniform(glProgramUniform1ui, "RandomSeed", rng_seed);
}

void STPStarfield::updateAnimationTimer(double second) {
	this->StarfieldBox.SkyboxRenderer.uniform(glProgramUniform1f, this->ShineTimeLocation, static_cast<float>(second));
}

void STPStarfield::render() const {
	this->StarfieldBox.drawSkybox();
}