#include <SuperRealism+/Scene/Component/STPSun.h>
//Shader Dir
#include <SuperRealism+/STPRealismInfo.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>

//GLM
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>

//Container
#include <tuple>

#include <glad/glad.h>

using glm::normalize;
using glm::smoothstep;
using glm::radians;
using glm::degrees;
using glm::value_ptr;
using glm::clamp;
using glm::rotate;

using glm::uvec3;
using glm::vec3;
using glm::dvec3;
using glm::mat3;

using std::tuple;
using std::make_from_tuple;
using std::make_pair;
using std::make_tuple;

using SuperTerrainPlus::STPFile;
using SuperTerrainPlus::SuperRealismPlus_ShaderPath;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto SkyShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPSun", ".frag");
constexpr static auto SpectrumShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPSunSpectrum", ".comp");

STPSun::STPSun(const STPEnvironment::STPSunSetting& sun_setting, const STPBundledData<vec3>& spectrum_domain,
	const STPSkyboxInitialiser& sun_init) : SunSetting(sun_setting), SunDirectionCache(0.0) {
	const STPEnvironment::STPSunSetting& sun = this->SunSetting;
	//validate the setting
	if (!sun.validate()) {
		throw STPException::STPInvalidEnvironment("Sun setting provided is invalid");
	}
	//calculate starting LST
	this->Day = 1.0 * sun.DayStart / (1.0 * sun.DayLength) + sun.YearStart;

	//setup sky renderer
	STPShaderManager sky_shader(GL_FRAGMENT_SHADER);
	//build the shader filename
	const char* const sky_source_file = SkyShaderFilename.data();
	STPShaderManager::STPShaderSource sky_source(sky_source_file, *STPFile(sky_source_file));
	//compile
	sky_shader(sky_source);

	//initialise skybox renderer
	this->initSkyboxRenderer(sky_shader, sun_init);

	//uniform setup
	this->SunPositionLocation = this->SkyboxRenderer.uniformLocation("SunPosition");
	//calculate initial sun direction
	this->advanceTime(0.0);

	/* ---------------------------------------- sun spectrum emulator ----------------------------------- */
	//setup spectrum emulator
	STPShaderManager spectrum_shader(GL_COMPUTE_SHADER);
	const char* const spectrum_source_file = SpectrumShaderFilename.data();
	STPShaderManager::STPShaderSource spectrum_source(spectrum_source_file, *STPFile(spectrum_source_file));

	spectrum_shader(spectrum_source);
	this->SpectrumEmulator.attach(spectrum_shader);
	//link
	this->SpectrumEmulator.finalise();

	//record the sun direction domain
	const auto& [sunDir_start, sunDir_end] = spectrum_domain;
	this->SpectrumDomainElevation = make_pair(sunDir_start.y, sunDir_end.y);

	//sampler location
	this->SpectrumEmulator.uniform(glProgramUniform1i, "SkyLight", 0)
		.uniform(glProgramUniform1i, "SunLight", 1)
		//generator data
		.uniform(glProgramUniform3fv, "SunDirectionStart", 1, value_ptr(sunDir_start))
		.uniform(glProgramUniform3fv, "SunDirectionEnd", 1, value_ptr(sunDir_end));
}

inline void STPSun::updateAtmosphere(STPProgramManager& program, const STPEnvironment::STPAtmosphereSetting& atmo_setting) {
	program.uniform(glProgramUniform1f, "Atmo.iSun", atmo_setting.SunIntensity)
		.uniform(glProgramUniform1f, "Atmo.rPlanet", atmo_setting.PlanetRadius)
		.uniform(glProgramUniform1f, "Atmo.rAtmos", atmo_setting.AtmosphereRadius)
		.uniform(glProgramUniform1f, "Atmo.vAlt", atmo_setting.ViewAltitude)
		.uniform(glProgramUniform3fv, "Atmo.kRlh", 1, value_ptr(atmo_setting.RayleighCoefficient))
		.uniform(glProgramUniform1f, "Atmo.kMie", atmo_setting.MieCoefficient)
		.uniform(glProgramUniform1f, "Atmo.shRlh", atmo_setting.RayleighScale)
		.uniform(glProgramUniform1f, "Atmo.shMie", atmo_setting.MieScale)
		.uniform(glProgramUniform1f, "Atmo.g", atmo_setting.MieScatteringDirection)
		.uniform(glProgramUniform1ui, "Atmo.priStep", atmo_setting.PrimaryRayStep)
		.uniform(glProgramUniform1ui, "Atmo.secStep", atmo_setting.SecondaryRayStep);
}

const vec3& STPSun::sunDirection() const {
	return this->SunDirectionCache;
}

void STPSun::advanceTime(double delta_second) {
	const STPEnvironment::STPSunSetting& sun = this->SunSetting;

	//offset the timer
	//increment day count
	this->Day += delta_second / (1.0 * sun.DayLength);
	//wrap the day around if it is the next year
	if (this->Day >= 1.0 * sun.YearLength) {
		this->Day -= static_cast<double>(sun.YearLength);
	}

	//the old direction cache is no longer accurate, needs to recalculate
	static constexpr double TWO_PI = glm::pi<double>() * 2.0;
	static constexpr auto saturate = [](double val) constexpr -> double {
		return clamp(val, -1.0, 1.0);
	};
	//Calculate hour angle based on the local solar time (LST), i.e., current time in a day - noon time;
	//the noon time is essentially half of the day length.
	//The fraction of this->Day is the percentage time has passed in one day.
	//Consider one day as 360 degree, hour angle is the angle at LST in a day.
	const double HRA = TWO_PI * (glm::fract(this->Day) - 0.5);
	//calculate declination, the angle between the sun and the equator plane
	const double delta = sun.Obliquity * -glm::cos(TWO_PI * this->Day / (1.0 * sun.YearLength)),
		phi = sun.Latitude;

	//calculate sun direction
	const double sin_delta = glm::sin(delta),
		cos_delta = glm::cos(delta),
		sin_phi = glm::sin(phi),
		cos_phi = glm::cos(phi),
		cos_HRA = glm::cos(HRA);

	const double sin_Elevation = saturate(
		sin_delta * sin_phi +
		cos_delta * cos_phi * cos_HRA
	),
		cos_Elevation = glm::sqrt(1.0 - sin_Elevation * sin_Elevation);
	//azimuth angle: north=0, east=90, south=180, west=270 degree
	const double cos_Azimuth = saturate(
		(sin_delta * cos_phi - cos_delta * sin_phi * cos_HRA) /
		cos_Elevation
	),
		//azimuth correction for the afternoon
		//Azimuth angle starts from north, which is (1, 0, 0) in OpenGL coordinate system
		sin_Azimuth = glm::sqrt(1.0 - cos_Azimuth * cos_Azimuth) * ((HRA > 0.0) ? -1.0 : 1.0);

	//convert angles to direction vector
	//normalise the direction
	this->SunDirectionCache = static_cast<vec3>(normalize(dvec3(
		cos_Elevation * cos_Azimuth,
		sin_Elevation,
		cos_Elevation * sin_Azimuth
	)));

	//update sun position in the shader
	this->SkyboxRenderer.uniform(glProgramUniform3fv, this->SunPositionLocation, 1, value_ptr(this->SunDirectionCache));
}

void STPSun::setAtmoshpere(const STPEnvironment::STPAtmosphereSetting& atmo_setting) {
	//validate
	if (!atmo_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Atmosphere setting is invalid");
	}

	STPSun::updateAtmosphere(this->SkyboxRenderer, atmo_setting);
	STPSun::updateAtmosphere(this->SpectrumEmulator, atmo_setting);
}

STPSun::STPBundledData<STPLightSpectrum> STPSun::generateSunSpectrum(unsigned int spectrum_length, const mat3& ray_space) const {
	this->SpectrumEmulator.uniform(
		glProgramUniformMatrix3fv, "SunToRayDirection", 1, static_cast<GLboolean>(GL_FALSE), value_ptr(ray_space));
	//setup output
	const auto spectrum_creator = make_tuple(spectrum_length, GL_RGBA16F);
	STPBundledData<STPLightSpectrum> spectrum = make_pair(
		make_from_tuple<STPLightSpectrum>(spectrum_creator),
		make_from_tuple<STPLightSpectrum>(spectrum_creator)
	);
	auto& [sky_spec, sun_spec] = spectrum;

	//bind output
	sky_spec.spectrum().bindImage(0u, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
	sun_spec.spectrum().bindImage(1u, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

	//calculate launch configuration
	const unsigned int blockDim = static_cast<unsigned int>(this->SpectrumEmulator.workgroupSize().x),
		gridDim = (spectrum_length + blockDim - 1u) / blockDim;
	//compute
	this->SpectrumEmulator.use();
	glDispatchCompute(gridDim, 1u, 1u);
	//sync to ensure valid texture access later
	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

	//clear up
	STPTexture::unbindImage(0u);
	STPTexture::unbindImage(1u);
	STPProgramManager::unuse();

	return spectrum;
}

float STPSun::spectrumCoordinate() const {
	const auto [elev_start, elev_end] = this->SpectrumDomainElevation;

	//project current sun direction to the spectrum sun direction
	//we assume the atmosphere is uniform, meaning the spectrum is independent of Azimuth angle, and depends only on Elevation.
	return (this->SunDirectionCache.y - elev_start) / (elev_end - elev_start);
}

void STPSun::render() const {
	this->drawSkybox();
}