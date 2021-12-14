#include <SuperRealism+/Renderer/STPSun.h>
//Shader Dir
#include <SuperRealism+/STPRealismInfo.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
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

using glm::vec3;
using glm::dvec3;

using std::array;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto SkyShaderFilename = STPFile::generateFilename(SuperRealismPlus_ShaderPath, "/STPSun", ".vert", ".frag");

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
	BoxIndex.size(),
	1u,
	0u,
	0u,
	0u
};

STPSun::STPSun(const STPEnvironment::STPSunSetting& sun_setting, STPSunLog& log) : SunSetting(sun_setting),
	AnglePerTick(radians(360.0 / (1.0 * sun_setting.DayLength))), NoonTime(sun_setting.DayLength / 2ull), DirectionOutdated(true), DirectionCache() {
	//validate the setting
	if (!this->SunSetting.validate()) {
		throw STPException::STPBadNumericRange("Sun setting provided is invalid");
	}
	//calculate starting LST
	this->LocalSolarTime = this->SunSetting.DayStartOffset;
	this->Day = 0u;

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
	STPShaderManager sky_shader[SkyShaderFilename.size()] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
	for (unsigned int i = 0u; i < SkyShaderFilename.size(); i++) {
		STPShaderManager& current_shader = sky_shader[i];
		//build the shader filename
		const char* const sky_filename = SkyShaderFilename[i].data();
		//compile with super realism + system include directory
		log.Log[i] = current_shader(*STPFile(sky_filename), { "/Common/STPCameraInformation.glsl" });

		//put shader into the program
		this->SkyRenderer.attach(current_shader);
	}

	//link
	this->SkyRenderer.finalise();
	log.Log[2] = this->SkyRenderer.lastLog(STPProgramManager::STPLogType::Link);
	log.Log[3] = this->SkyRenderer.lastLog(STPProgramManager::STPLogType::Validation);
}

const STPSun::STPSunDirection& STPSun::calcSunDirection() const {
	static constexpr double TWO_PI = glm::pi<double>() * 2.0;

	if (this->DirectionOutdated) {
		//the old direction cache is no longer accurate, needs to recalculate
		const STPEnvironment::STPSunSetting& sun = this->SunSetting;

		//calculate hour angle
		const double HRA = radians(this->AnglePerTick * (this->LocalSolarTime - this->NoonTime));
		//calculate declination, the angle between the sun and the equator plane
		const double delta = radians(sun.Obliquity * -glm::cos(TWO_PI * this->Day / (1.0 * sun.YearLength))),
			phi = radians(sun.Latitude);

		STPSunDirection& dir = this->DirectionCache;
		//calculate sun direction
		const double sin_delta = glm::sin(delta),
			cos_delta = glm::cos(delta),
			sin_phi = glm::sin(phi),
			cos_phi = glm::cos(phi),
			cos_HRA = glm::cos(HRA);
		//azimuth angle: north=0, east=90, south=180, west=270 degree
		dir.Direction.y =
			sin_delta * sin_phi +
			cos_delta * cos_phi * cos_HRA;
		dir.Elevation = glm::asin(dir.Direction.y);
		dir.Direction.z =
			-(sin_delta * cos_phi - cos_delta * sin_phi * cos_HRA) /
			glm::cos(dir.Elevation);
		dir.Azimuth = glm::acos(-dir.Direction.z);
		dir.Direction.x = glm::sin(dir.Azimuth);
		//normalise the direction
		dir.Direction = normalize(dir.Direction);

		this->DirectionOutdated = false;
	}
	return this->DirectionCache;
}

void STPSun::deltaTick(size_t delta) {
	const STPEnvironment::STPSunSetting& sun = this->SunSetting;

	this->LocalSolarTime += delta;
	const size_t deltaDay = this->LocalSolarTime / sun.DayLength;
	if (deltaDay > 0ull) {
		//wrap the time around if it is the next day
		this->LocalSolarTime %= sun.DayLength;

		this->Day += deltaDay;
		//wrap the day around if it is the next year
		this->Day %= sun.YearLength;
	}

	//time has changed, mark the direction cache as outdated
	this->DirectionOutdated = true;
}

double STPSun::status(double elevation) const {
	const STPEnvironment::STPSunSetting& sun = this->SunSetting;
	return smoothstep(sun.SunriseAngle, sun.SunsetAngle, degrees(elevation) + sun.CycleAngleOffset) * 2.0 - 1.0;
}

void STPSun::setAtomshpere(const STPEnvironment::STPAtomsphereSetting& sky_setting) {
	//validate
	if (!sky_setting.validate()) {
		throw STPException::STPBadNumericRange("Atomshpere setting is invalid");
	}

	this->SkyRenderer.uniform(glProgramUniform1f, "Sky.iSun", sky_setting.SunIntensity)
		.uniform(glProgramUniform1f, "Sky.rPlanet", sky_setting.PlanetRadius)
		.uniform(glProgramUniform1f, "Sky.rAtoms", sky_setting.AtomsphereRadius)
		.uniform(glProgramUniform1f, "Sky.vAlt", sky_setting.ViewAltitude)
		.uniform(glProgramUniform3fv, "Sky.kRlh", 1, value_ptr(sky_setting.RayleighCoefficient))
		.uniform(glProgramUniform1f, "Sky.kMie", sky_setting.MieCoefficient)
		.uniform(glProgramUniform1f, "Sky.shRlh", sky_setting.RayleighScale)
		.uniform(glProgramUniform1f, "Sky.shMie", sky_setting.MieScale)
		.uniform(glProgramUniform1f, "Sky.g", sky_setting.MieScatteringDirection)
		.uniform(glProgramUniform1ui, "Sky.priStep", sky_setting.PrimaryRayStep)
		.uniform(glProgramUniform1ui, "Sky.secStep", sky_setting.SecondaryRayStep);
}

void STPSun::operator()(const vec3& viewPos) const {
	//calculate the position/direction of the sun
	const STPSunDirection& sunInfo = this->calcSunDirection();
	const vec3 sun_dir = static_cast<vec3>(sunInfo.Direction);

	this->SkyRenderer.uniform(glProgramUniform3fv, "SunPosition", 1, value_ptr(sun_dir));

	//setup context
	this->SkyRenderer.use();
	this->RayDirectionArray.bind();
	this->SkyRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);

	//draw
	glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, nullptr);

	//clear up
	STPProgramManager::unuse();
}