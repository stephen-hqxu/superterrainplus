#include <SuperRealism+/Scene/Component/STPStarfield.h>
#include <SuperRealism+/STPRealismInfo.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

#include <glad/glad.h>

#include <glm/gtc/type_ptr.hpp>

#include <cstdint>
#include <random>
#include <limits>
#include <array>
#include <algorithm>
#include <functional>
#include <utility>

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
	const STPShaderManager::STPShader star_fs = STPShaderManager::make(GL_FRAGMENT_SHADER, star_source);
	this->StarfieldBox.initSkyboxRenderer(star_fs, starfield_init);

	/* -------------------------------- setup uniform --------------------------------- */
	this->ShineTimeLocation = this->StarfieldBox.SkyboxRenderer.uniformLocation("ShineTime");
	this->StarfieldBox.SkyboxRenderer.uniform(glProgramUniformHandleui64ARB, "StarColorSpectrum", this->StarlightSpectrum.spectrumHandle());
}

void STPStarfield::setStarfield(const STPEnvironment::STPStarfieldSetting& starfield_setting) {
	starfield_setting.validate();

	using std::numeric_limits;
	using glm::uvec3;
	//generate random vector seed
	std::mt19937_64 rng_engine(starfield_setting.Seed);
	std::uniform_int_distribution rng_dist(numeric_limits<std::uint32_t>::min(), numeric_limits<std::uint32_t>::max());
	
	std::array<std::uint32_t, uvec3::length()> random_arr;
	std::generate(random_arr.begin(), random_arr.end(),
		[next = std::bind(rng_dist, std::ref(rng_engine))]() { return next(); });
	const uvec3 random_vec = glm::make_vec3(random_arr.data());

	this->StarfieldBox.SkyboxRenderer.uniform(glProgramUniform1f, "Star.iLklh", starfield_setting.InitialLikelihood)
		.uniform(glProgramUniform1f, "Star.OctLklhMul", starfield_setting.OctaveLikelihoodMultiplier)
		.uniform(glProgramUniform1f, "Star.iScl", starfield_setting.InitialScale)
		.uniform(glProgramUniform1f, "Star.OctSclMul", starfield_setting.OctaveScaleMultiplier)
		.uniform(glProgramUniform1f, "Star.Thres", starfield_setting.EdgeDistanceFalloff)
		.uniform(glProgramUniform1f, "Star.spdShine", starfield_setting.ShineSpeed)
		.uniform(glProgramUniform1f, "Star.LumMul", starfield_setting.LuminosityMultiplier)
		.uniform(glProgramUniform1f, "Star.MinAlt", starfield_setting.MinimumAltitude)
		.uniform(glProgramUniform1ui, "Star.Oct", starfield_setting.Octave)
		.uniform(glProgramUniform3uiv, "RandomSeed", 1, glm::value_ptr(random_vec));
}

void STPStarfield::updateAnimationTimer(const double second) {
	this->StarfieldBox.SkyboxRenderer.uniform(glProgramUniform1f, this->ShineTimeLocation, static_cast<float>(second));
}

void STPStarfield::render() const {
	this->StarfieldBox.drawSkybox();
}