#include <SuperRealism+/STPSun.h>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>

//GLM
#include <glm/trigonometric.hpp>
#include <glm/geometric.hpp>

using glm::normalize;
using glm::smoothstep;
using glm::radians;
using glm::degrees;
using glm::dvec3;

using namespace SuperTerrainPlus;
using namespace SuperTerrainPlus::STPRealism;

STPSun::STPSun(const STPEnvironment::STPSunSetting& sun_setting) : SunSetting(sun_setting), 
	AnglePerTick(radians(360.0 / (1.0 * sun_setting.DayLength))), NoonTime(sun_setting.DayLength / 2ull) {
	//validate the setting
	if (!this->SunSetting.validate()) {
		throw STPException::STPBadNumericRange("Sun setting provided is invalid");
	}
	//calculate starting LST
	this->LocalSolarTime = this->SunSetting.DayStartOffset;
	this->Day = 0u;
}

STPSun::STPSunDirection STPSun::currentDirection() const {
	static constexpr double TWO_PI = 6.283185307179586476925286766559;
	const STPEnvironment::STPSunSetting& sun = this->SunSetting;

	//calculate hour angle
	const double HRA = radians(this->AnglePerTick * (this->LocalSolarTime - this->NoonTime));
	//calculate declination, the angle between the sun and the equator plane
	const double delta = radians(sun.Obliquity * -glm::cos(TWO_PI * this->Day / (1.0 * sun.YearLength))),
		phi = radians(sun.Latitude);

	STPSunDirection dir;
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

	return dir;
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
}

double STPSun::status(double elevation) const {
	const STPEnvironment::STPSunSetting& sun = this->SunSetting;
	return smoothstep(sun.SunriseAngle, sun.SunsetAngle, degrees(elevation) + sun.CycleAngleOffset) * 2.0 - 1.0;
}