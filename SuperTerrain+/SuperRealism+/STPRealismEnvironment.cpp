#include <SuperRealism+/Environment/STPAtmosphereSetting.h>
#include <SuperRealism+/Environment/STPCameraSetting.h>
#include <SuperRealism+/Environment/STPLightSetting.h>
#include <SuperRealism+/Environment/STPMeshSetting.h>
#include <SuperRealism+/Environment/STPOrthographicCameraSetting.h>
#include <SuperRealism+/Environment/STPPerspectiveCameraSetting.h>
#include <SuperRealism+/Environment/STPSunSetting.h>

#include <glm/trigonometric.hpp>
#include <glm/ext/scalar_constants.hpp>

using glm::vec3;
using glm::radians;

using namespace SuperTerrainPlus::STPEnvironment;

//STPAtmosphereSetting.h

STPAtmosphereSetting::STPAtmosphereSetting() :
	SunIntensity(1.0f), 
	PlanetRadius(1.0f), 
	AtmosphereRadius(1.0f), 
	ViewAltitude(1.0f),

	RayleighCoefficient(1.0f), 
	MieCoefficient(1.0f), 
	RayleighScale(1.0f), 
	MieScale(1.0f), 
	MieScatteringDirection(1.0f), 

	PrimaryRayStep(1u), 
	SecondaryRayStep(1u) {

}

bool STPAtmosphereSetting::validate() const {
	constexpr static vec3 zeroVec3 = vec3(0.0f);

	return this->SunIntensity > 0.0f
		&& this->PlanetRadius > 0.0f
		&& this->AtmosphereRadius > 0.0f
		&& this->PlanetRadius <= this->AtmosphereRadius
		&& this->ViewAltitude > 0.0f
		&& this->ViewAltitude >= this->PlanetRadius
		&& this->RayleighCoefficient != zeroVec3
		&& this->MieCoefficient > 0.0f
		&& this->RayleighScale > 0.0f
		&& this->MieScale > 0.0f
		&& this->MieScatteringDirection > 0.0f
		&& this->PrimaryRayStep > 0u
		&& this->SecondaryRayStep > 0u;
}

//STPCameraSetting.h

STPCameraSetting::STPCameraSetting() : 
	Yaw(radians(-90.0f)), Pitch(0.0f),
	MovementSpeed(2.5f), RotationSensitivity(0.1f),
	Position(vec3(0.0f)), WorldUp(0.0f, 1.0f, 0.0f), 
	Near(0.1f), Far(1.0f), LogarithmicConstant(1.0f) {

}

bool STPCameraSetting::validate() const {
	static auto range = [](float val, float min, float max) constexpr -> bool {
		return val > min && val < max;
	};
	static constexpr float PI = glm::pi<float>(), PI_BY_2 = PI * 0.5f;

	return range(this->Yaw, -PI, PI) 
		&& range(this->Pitch, -PI_BY_2, PI_BY_2) 
		&& this->MovementSpeed > 0.0f 
		&& this->RotationSensitivity > 0.0f
		&& this->Near > 0.0f
		&& this->Far > 0.0f
		&& this->Near < this->Far
		&& this->LogarithmicConstant > 0.0f;
}

//STPLightSetting.h

STPLightSetting::STPAmbientLightSetting::STPAmbientLightSetting() : 
	AmbientStrength(0.1f) {

}

bool STPLightSetting::STPAmbientLightSetting::validate() const {
	return true;
}

STPLightSetting::STPDirectionalLightSetting::STPDirectionalLightSetting() : 
	DiffuseStrength(1.0f), SpecularStrength(1.0f) {

}

bool STPLightSetting::STPDirectionalLightSetting::validate() const {
	return true;
}

//STPMeshSetting.h

STPMeshSetting::STPTessellationSetting::STPTessellationSetting() :
	MaxTessLevel(0.0f),
	MinTessLevel(0.0f),
	FurthestTessDistance(0.0f),
	NearestTessDistance(0.0f) {

}

bool STPMeshSetting::STPTessellationSetting::validate() const {
	return this->MaxTessLevel >= 0.0f
		&& this->MinTessLevel >= 0.0f
		&& this->FurthestTessDistance >= 0.0f
		&& this->NearestTessDistance >= 0.0f
		//range check
		&& this->MaxTessLevel >= this->MinTessLevel
		&& this->FurthestTessDistance >= this->NearestTessDistance;
}

STPMeshSetting::STPTextureRegionSmoothSetting::STPTextureRegionSmoothSetting() :
	KernelRadius(1u), 
	KernelScale(1.0f), 
	NoiseScale(1.0f) {

}

bool STPMeshSetting::STPTextureRegionSmoothSetting::validate() const {
	return this->KernelRadius > 0u
		&& this->KernelScale > 0.0f;
}

STPMeshSetting::STPMeshSetting() :
	Strength(1.0f),
	Altitude(1.0f), 
	UVScaleFactor(1u) {

}

bool STPMeshSetting::validate() const {
	return this->Strength > 0.0f
		&& this->Altitude > 0.0f
		&& this->UVScaleFactor > 0u
		&& this->TessSetting.validate()
		&& this->RegionSmoothSetting.validate();
}

//STPOrthographicCameraSetting.h

STPOrthographicCameraSetting::STPOrthographicCameraSetting() : 
	Left(-1.0f), Right(1.0f), Bottom(-1.0f), Top(1.0f) {

}

bool STPOrthographicCameraSetting::validate() const {
	return this->Left < this->Right
		&& this->Bottom < this->Top;
}

//STPPerspectiveCameraSetting.h

STPPerspectiveCameraSetting::STPPerspectiveCameraSetting() :
	ViewAngle(radians(45.0f)), ZoomSensitivity(1.0f),
	ZoomLimit(radians(1.0f), radians(90.0f)),
	Aspect(1.0f) {

}

bool STPPerspectiveCameraSetting::validate() const {
	static auto range = [](float val, float min, float max) constexpr -> bool {
		return val > min && val < max;
	};
	static constexpr float TWO_PI = glm::pi<float>() * 2.0f;

	return range(this->ViewAngle, 0.0f, TWO_PI)
		&& this->ZoomSensitivity > 0.0f
		&& this->ZoomLimit.x > 0.0f
		&& this->ZoomLimit.y > 0.0f
		&& this->ZoomLimit.x <= this->ZoomLimit.y
		&& this->Aspect > 0.0f;
}

//STPSunSetting.h

STPSunSetting::STPSunSetting() : 
	DayLength(24000ull),
	DayStartOffset(0ull),
	YearLength(1u),

	Obliquity(0.0), 
	Latitude(0.0) {

}

bool STPSunSetting::validate() const {
	static constexpr double PI_BY_2 = glm::pi<double>() * 0.5;
	static auto range_check = [](double val, double min, double max) constexpr -> bool {
		return val >= min && val <= max;
	};

	return this->DayLength > 0ull
		&& ((this->DayLength & 0x01ull) == 0x00ull)//must be an even number
		&& this->DayStartOffset < this->DayLength
		&& this->YearLength > 0u
		&& range_check(this->Obliquity, 0.0, PI_BY_2)
		&& range_check(this->Latitude, -PI_BY_2, PI_BY_2);
}