#include <SuperRealism+/Environment/STPAtomsphereSetting.h>
#include <SuperRealism+/Environment/STPSunSetting.h>

using glm::vec3;

using namespace SuperTerrainPlus::STPEnvironment;

//STPAtomsphereSetting.h

STPAtomsphereSetting::STPAtomsphereSetting() : 
	SunPosition(1.0f), 
	SunIntensity(1.0f), 
	PlanetRadius(1.0f), 
	AtomsphereRadius(1.0f), 

	RayleighCoefficient(1.0f), 
	MieCoefficient(1.0f), 
	RayleighScale(1.0f), 
	MieScale(1.0f), 
	MieScatteringDirection(1.0f), 

	PrimaryRayStep(1u), 
	SecondaryRayStep(1u) {

}

bool STPAtomsphereSetting::validate() const {
	constexpr static vec3 zeroVec3 = vec3(0.0f);

	return this->SunIntensity > 0.0f
		&& this->PlanetRadius > 0.0f
		&& this->AtomsphereRadius > 0.0f
		&& this->RayleighCoefficient != zeroVec3
		&& this->MieCoefficient > 0.0f
		&& this->RayleighScale > 0.0f
		&& this->MieScale > 0.0f
		&& this->MieScatteringDirection > 0.0f
		&& this->PrimaryRayStep > 0u
		&& this->SecondaryRayStep > 0u;
}

//STPSunSetting.h

STPSunSetting::STPSunSetting() : 
	DayLength(24000ull),
	DayStartOffset(0ull),
	YearLength(1u),

	Obliquity(0.0), 
	Latitude(0.0),

	SunsetAngle(1.0), 
	SunriseAngle(-1.0), 
	CycleAngleOffset(0.0) {

}

bool STPSunSetting::validate() const {
	static auto range_check = [](double val, double min, double max) constexpr -> bool {
		return val >= min && val <= max;
	};

	return this->DayLength > 0ull
		&& ((this->DayLength & 0x01ull) == 0x00ull)//must be an even number
		&& this->DayStartOffset < this->DayLength
		&& this->YearLength > 0u
		&& range_check(this->Obliquity, 0.0, 90.0)
		&& range_check(this->Latitude, -90.0, 90.0)
		&& range_check(this->SunsetAngle, -90.0, 90.0)
		&& range_check(this->SunriseAngle, -90.0, 90.0)
		&& this->SunsetAngle > this->SunriseAngle
		&& range_check(this->CycleAngleOffset, -90.0, 90.0);
}