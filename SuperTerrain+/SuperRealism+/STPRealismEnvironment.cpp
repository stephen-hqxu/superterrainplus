#include <SuperRealism+/Environment/STPSunSetting.h>

using namespace SuperTerrainPlus::STPEnvironment;

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