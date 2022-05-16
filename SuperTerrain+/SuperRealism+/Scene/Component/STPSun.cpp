#include <SuperRealism+/Scene/Component/STPSun.h>
//Shader Dir
#include <SuperRealism+/STPRealismInfo.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>
//Indirect
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//GLM
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>

//Container
#include <tuple>
#include <array>

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

using std::array;
using std::tuple;
using std::make_from_tuple;
using std::make_pair;
using std::make_tuple;

using SuperTerrainPlus::STPFile;
using SuperTerrainPlus::SuperRealismPlus_ShaderPath;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto SkyShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPSun", ".vert", ".frag");
constexpr static auto SpectrumShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPSunSpectrum", ".comp");

constexpr static array<signed char, 24ull> BoxVertex = { 
	-1, -1, -1, //origin
	+1, -1, -1, //x=1
	+1, -1, +1, //x=z=1
	-1, -1, +1, //z=1
	-1, +1, -1, //y=1
	+1, +1, -1, //x=y=1
	+1, +1, +1, //x=y=z=1
	-1, +1, +1  //y=z=1
};
constexpr static array<unsigned char, 36ull> BoxIndex = {
	0, 1, 2,
	0, 2, 3,

	0, 1, 5,
	0, 5, 4,

	1, 2, 6,
	1, 6, 5,

	2, 3, 7,
	2, 7, 6,

	3, 0, 4,
	3, 4, 7,

	4, 5, 6,
	4, 6, 7
};
constexpr static STPIndirectCommand::STPDrawElement SkyDrawCommand = {
	static_cast<unsigned int>(BoxIndex.size()),
	1u,
	0u,
	0u,
	0u
};

STPSun::STPSun(const STPEnvironment::STPSunSetting& sun_setting, const STPBundledData<vec3>& spectrum_domain) : SunSetting(sun_setting),
	AnglePerTick(radians(360.0 / (1.0 * sun_setting.DayLength))), NoonTime(sun_setting.DayLength / 2ull), SunDirectionCache(0.0) {
	//validate the setting
	if (!this->SunSetting.validate()) {
		throw STPException::STPInvalidEnvironment("Sun setting provided is invalid");
	}
	//calculate starting LST
	this->Day = 1.0 * this->SunSetting.DayStartOffset / (1.0 * this->SunSetting.DayLength);

	//setup sky rendering buffer
	this->RayDirectionBuffer.bufferStorageSubData(BoxVertex.data(), BoxVertex.size() * sizeof(signed char), GL_NONE);
	this->RayDirectionIndex.bufferStorageSubData(BoxIndex.data(), BoxIndex.size() * sizeof(unsigned char), GL_NONE);
	//setup rendering command
	this->SkyRenderCommand.bufferStorageSubData(&SkyDrawCommand, sizeof(SkyDrawCommand), GL_NONE);
	//setup vertex array
	STPVertexArray::STPVertexAttributeBuilder attr = this->RayDirectionArray.attribute();
	attr.format(3, GL_BYTE, GL_FALSE, sizeof(signed char))
		.vertexBuffer(this->RayDirectionBuffer, 0)
		.elementBuffer(this->RayDirectionIndex)
		.binding();
	this->RayDirectionArray.enable(0u);

	//setup sky renderer
	STPShaderManager sky_shader[SkyShaderFilename.size()] = 
		{ GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
	for (unsigned int i = 0u; i < SkyShaderFilename.size(); i++) {
		//build the shader filename
		const char* const sky_source_file = SkyShaderFilename[i].data();
		STPShaderManager::STPShaderSource sky_source(sky_source_file, *STPFile(sky_source_file));

		//compile
		sky_shader[i](sky_source);

		//put shader into the program
		this->SkyRenderer.attach(sky_shader[i]);
	}

	//link
	this->SkyRenderer.finalise();

	//uniform setup
	this->SunPositionLocation = this->SkyRenderer.uniformLocation("SunPosition");

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

void STPSun::advanceTick(unsigned long long tick) {
	const STPEnvironment::STPSunSetting& sun = this->SunSetting;

	//offset the timer
	//increment day count
	this->Day += 1.0 * tick / (1.0 * sun.DayLength);
	//wrap the day around if it is the next year
	if (this->Day >= 1.0 * sun.YearLength) {
		this->Day -= static_cast<double>(sun.YearLength);
	}
	//the fractional part of Day is the fraction in a day
	const unsigned long long LocalSolarTime = static_cast<unsigned long long>(glm::round(glm::fract(this->Day) * sun.DayLength));

	//the old direction cache is no longer accurate, needs to recalculate
	static constexpr double TWO_PI = glm::pi<double>() * 2.0;
	static constexpr auto saturate = [](double val) constexpr -> double {
		return clamp(val, -1.0, 1.0);
	};
	//calculate hour angle
	const double HRA = this->AnglePerTick * static_cast<long long>(LocalSolarTime - this->NoonTime);
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
	this->SkyRenderer.uniform(glProgramUniform3fv, this->SunPositionLocation, 1, value_ptr(this->SunDirectionCache));
}

void STPSun::setAtmoshpere(const STPEnvironment::STPAtmosphereSetting& atmo_setting) {
	//validate
	if (!atmo_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Atmosphere setting is invalid");
	}

	STPSun::updateAtmosphere(this->SkyRenderer, atmo_setting);
	STPSun::updateAtmosphere(this->SpectrumEmulator, atmo_setting);
}

STPSun::STPBundledData<STPLightSpectrum> STPSun::generateSunSpectrum(unsigned int spectrum_length, const mat3& ray_space) const {
	this->SpectrumEmulator.uniform(glProgramUniformMatrix3fv, "SunToRayDirection", 1, static_cast<GLboolean>(GL_FALSE), value_ptr(ray_space));
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
	//setup context
	this->SkyRenderer.use();
	this->RayDirectionArray.bind();
	this->SkyRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);

	//draw
	glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, nullptr);

	//clear up
	STPProgramManager::unuse();
}