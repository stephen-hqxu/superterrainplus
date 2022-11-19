#include <SuperRealism+/Scene/Component/STPAurora.h>
#include <SuperRealism+/STPRealismInfo.h>

#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

#include <glad/glad.h>

//GLM
#include <glm/vec2.hpp>
#include <glm/mat2x2.hpp>
#include <glm/gtc/type_ptr.hpp>

using glm::vec2;
using glm::mat2;
using glm::value_ptr;

using namespace SuperTerrainPlus::STPRealism;

static constexpr auto AuroraShaderFilename =
	SuperTerrainPlus::STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPAurora", ".frag");

STPAurora::STPAurora(STPLightSpectrum&& aurora_spectrum, const STPSkybox::STPSkyboxInitialiser& aurora_init) : AuroraSpectrum(std::move(aurora_spectrum)) {
	const char* const aurora_source_file = AuroraShaderFilename.data();
	const STPShaderManager::STPShaderSource aurora_source(aurora_source_file, STPFile::read(aurora_source_file));

	const STPShaderManager::STPShader aurora_fs = STPShaderManager::make(GL_FRAGMENT_SHADER, aurora_source);
	this->AuroraBox.initSkyboxRenderer(aurora_fs, aurora_init);

	/* ----------------------------------- uniform time --------------------------------------- */
	this->AuroraTimeLocation = this->AuroraBox.SkyboxRenderer.uniformLocation("AuroraTime");
	this->AuroraBox.SkyboxRenderer.uniform(glProgramUniformHandleui64ARB, "AuroraColorSpectrum", this->AuroraSpectrum.spectrumHandle());
}

void STPAurora::setAurora(const STPEnvironment::STPAuroraSetting& aurora_setting) {
	aurora_setting.validate();

	//extract sub-settings
	const auto& tri_noise = aurora_setting.Noise;
	const auto& noise_func_main = tri_noise.MainNoise,
		&noise_func_distortion = tri_noise.DistortionNoise;

	const vec2 fade = vec2(aurora_setting.AltitudeFadeStart - aurora_setting.AltitudeFadeEnd, aurora_setting.AltitudeFadeEnd);
	//octave rotation matrix
	const float sinT = glm::sin(tri_noise.OctaveRotation),
		cosT = glm::cos(tri_noise.OctaveRotation);
	const mat2 octave_rotation = mat2(
		cosT, sinT,
		-sinT, cosT
	);

	//main aurora settings
	this->AuroraBox.SkyboxRenderer.uniform(glProgramUniform1f, "Aurora.Flat", aurora_setting.AuroraSphereFlatness)
		.uniform(glProgramUniform1f, "Aurora.stepSz", aurora_setting.StepSize)
		.uniform(glProgramUniform1f, "Aurora.projRot", aurora_setting.AuroraPlaneProjectionBias)
		.uniform(glProgramUniform2fv, "Aurora.Fade", 1, value_ptr(fade))
		.uniform(glProgramUniform1f, "Aurora.LumMul", aurora_setting.LuminosityMultiplier)
		.uniform(glProgramUniform1ui, "Aurora.Iter", aurora_setting.Iteration)
		//main noise function
		.uniform(glProgramUniform1f, "TriNoise.fnNoise.iAmp", noise_func_main.InitialAmplitude)
		.uniform(glProgramUniform1f, "TriNoise.fnNoise.Pers", noise_func_main.Persistence)
		.uniform(glProgramUniform1f, "TriNoise.fnNoise.Lacu", noise_func_main.Lacunarity)
		//distortion noise function
		.uniform(glProgramUniform1f, "TriNoise.fnDist.iAmp", noise_func_distortion.InitialAmplitude)
		.uniform(glProgramUniform1f, "TriNoise.fnDist.Pers", noise_func_distortion.Persistence)
		.uniform(glProgramUniform1f, "TriNoise.fnDist.Lacu", noise_func_distortion.Lacunarity)
		//triangular noise settings
		.uniform(glProgramUniform1f, "TriNoise.iFreqDist", tri_noise.InitialDistortionFrequency)
		.uniform(glProgramUniform1f, "TriNoise.Curv", tri_noise.Curvature)
		.uniform(glProgramUniformMatrix2fv, "TriNoise.octRot", 1, static_cast<GLboolean>(GL_FALSE), value_ptr(octave_rotation))
		.uniform(glProgramUniform1f, "TriNoise.Spd", tri_noise.AnimationSpeed)
		.uniform(glProgramUniform1f, "TriNoise.C", tri_noise.Contrast)
		.uniform(glProgramUniform1f, "TriNoise.maxInt", tri_noise.MaximumIntensity)
		.uniform(glProgramUniform1ui, "TriNoise.Oct", tri_noise.Octave);
}

void STPAurora::updateAnimationTimer(const double second) {
	this->AuroraBox.SkyboxRenderer.uniform(glProgramUniform1f, this->AuroraTimeLocation, static_cast<float>(second));
}

void STPAurora::render() const {
	this->AuroraBox.drawSkybox();
}