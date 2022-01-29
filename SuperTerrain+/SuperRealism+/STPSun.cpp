#include <SuperRealism+/Renderer/STPSun.h>
//Shader Dir
#include <SuperRealism+/STPRealismInfo.h>

//Error
#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>
#include <SuperTerrain+/Exception/STPGLError.h>
//IO
#include <SuperTerrain+/Utility/STPFile.h>
//Indirect
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//GLM
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>

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

using std::array;
using std::make_pair;

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

STPSun<false>::STPSunSpectrum::STPSunSpectrum(unsigned int spectrum_length, const STPSun& sun, STPSpectrumLog& log) :
	STPLightSpectrum(spectrum_length, STPSpectrumType::Bitonic, GL_RGBA16F), SunElevation(sun.sunDirection().y) {
	//setup spectrum emulator
	STPShaderManager spectrum_shader(GL_COMPUTE_SHADER);
	STPShaderManager::STPShaderSource spectrum_source(*STPFile(SpectrumShaderFilename.data()));

	log.Log[0] = spectrum_shader(spectrum_source);
	this->SpectrumEmulator.attach(spectrum_shader);
	//link
	log.Log[1] = this->SpectrumEmulator.finalise();
	if (!this->SpectrumEmulator) {
		throw STPException::STPGLError("Spectrum generator program fails to validate");
	}

	//the number of iteration is a fixed number and does not allow to be changed
	this->SpectrumEmulator.uniform(glProgramUniform1ui, "SpectrumDimension", this->SpectrumLength);
}

void STPSun<false>::STPSunSpectrum::operator()(const STPSpectrumSpecification& spectrum_setting) {
	//send uniforms to compute shader
	STPSun::updateAtmosphere(this->SpectrumEmulator, *spectrum_setting.Atmosphere);

	//record the sun direction domain
	const auto& [sunDir_start, sunDir_end] = spectrum_setting.Domain;
	this->SpectrumEmulator.uniform(glProgramUniformMatrix3fv, "SunToRayDirection", 1, static_cast<GLboolean>(GL_FALSE), value_ptr(spectrum_setting.RaySpace))
		.uniform(glProgramUniform3fv, "SunDirectionStart", 1, value_ptr(sunDir_start))
		.uniform(glProgramUniform3fv, "SunDirectionEnd", 1, value_ptr(sunDir_end));
	this->DomainElevation = make_pair(sunDir_start.y, sunDir_end.y);

	//setup output
	this->Spectrum.bindImage(0u, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);

	//calculate launch configuration
	const unsigned int blockDim = static_cast<unsigned int>(this->SpectrumEmulator.workgroupSize().x),
		gridDim = (this->SpectrumLength + blockDim - 1u) / blockDim;
	//compute
	this->SpectrumEmulator.use();
	glDispatchCompute(gridDim, 1u, 1u);
	//sync
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	//clear up
	STPTexture::unbindImage(0u);
	STPProgramManager::unuse();
}

float STPSun<false>::STPSunSpectrum::coordinate() const {
	const auto [elev_start, elev_end] = this->DomainElevation;

	//project current sun direction to the spectrum sun direction
	//we assume the atmosphere is uniform, meaning the spectrum is independent of Azimuth angle, and depends only on Elevation.
	return (this->SunElevation - elev_start) / (elev_end - elev_start);
}

STPSun<false>::STPSun(const STPEnvironment::STPSunSetting& sun_setting, unsigned int spectrum_length, STPSunLog& raw_log) : SunSetting(sun_setting),
	AnglePerTick(radians(360.0 / (1.0 * sun_setting.DayLength))), NoonTime(sun_setting.DayLength / 2ull), SunDirectionCache(0.0), 
	SunSpectrum(spectrum_length, *this, raw_log.SpectrumGenerator) {
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

	auto& log = raw_log.SunRenderer;
	//setup sky renderer
	STPShaderManager sky_shader[SkyShaderFilename.size()] = 
		{ GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
	for (unsigned int i = 0u; i < SkyShaderFilename.size(); i++) {
		//build the shader filename
		STPShaderManager::STPShaderSource sky_source(*STPFile(SkyShaderFilename[i].data()));

		//compile
		log.Log[i] = sky_shader[i](sky_source);

		//put shader into the program
		this->SkyRenderer.attach(sky_shader[i]);
	}

	//link
	log.Log[2] = this->SkyRenderer.finalise();
	if (!this->SkyRenderer) {
		throw STPException::STPGLError("Sky renderer program returns a failed status");
	}
}

inline void STPSun<false>::updateAtmosphere(STPProgramManager& program, const STPEnvironment::STPAtmosphereSetting& atmo_setting) {
	//validate
	if (!atmo_setting.validate()) {
		throw STPException::STPInvalidEnvironment("Atmoshpere setting is invalid");
	}

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

const vec3& STPSun<false>::sunDirection() const {
	return this->SunDirectionCache;
}

void STPSun<false>::advanceTick(unsigned long long tick) {
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
	static auto saturate = [](double val) constexpr -> double {
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
		//Azimuth angle starts from north, which is (0, 0, -1) in OpenGL coordinate system
		sin_Azimuth = glm::sqrt(1.0 - cos_Azimuth * cos_Azimuth) * ((HRA > 0.0) ? -1.0 : 1.0);

	//convert angles to direction vector
	//normalise the direction
	this->SunDirectionCache = static_cast<vec3>(normalize(dvec3(
		cos_Elevation * cos_Azimuth,
		sin_Elevation,
		cos_Elevation * sin_Azimuth
	)));

	//update sun position in the shader
	this->SkyRenderer.uniform(glProgramUniform3fv, "SunPosition", 1, value_ptr(this->SunDirectionCache));
}

void STPSun<false>::setAtmoshpere(const STPEnvironment::STPAtmosphereSetting& atmo_setting) {
	STPSun::updateAtmosphere(this->SkyRenderer, atmo_setting);
}

const STPLightSpectrum& STPSun<false>::getLightSpectrum() const {
	return this->SunSpectrum;
}

void STPSun<false>::renderEnvironment() {
	//setup context
	this->SkyRenderer.use();
	this->RayDirectionArray.bind();
	this->SkyRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);

	//draw
	glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, nullptr);

	//clear up
	STPProgramManager::unuse();
}

STPSun<true>::STPSun(const STPEnvironment::STPSunSetting& sun_setting, unsigned int spectrum_length, 
	const STPCascadedShadowMap::STPLightFrustum& shadow_frustum, STPSunLog& log) :
	STPSun<false>(sun_setting, spectrum_length, log), STPEnvironmentLight<true>(shadow_frustum) {

}

void STPSun<true>::advanceTick(unsigned long long tick) {
	//call the base class method
	this->STPSun<false>::advanceTick(tick);

	//update sun direction in the shadow light space
	this->EnvironmentLightShadow.setDirection(this->SunDirectionCache);
}