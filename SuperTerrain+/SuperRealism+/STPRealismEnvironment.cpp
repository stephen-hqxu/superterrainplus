#include <SuperRealism+/Environment/STPAtmosphereSetting.h>
#include <SuperRealism+/Environment/STPBidirectionalScatteringSetting.h>
#include <SuperRealism+/Environment/STPCameraSetting.h>
#include <SuperRealism+/Environment/STPLightSetting.h>
#include <SuperRealism+/Environment/STPMeshSetting.h>
#include <SuperRealism+/Environment/STPOcclusionKernelSetting.h>
#include <SuperRealism+/Environment/STPOrthographicCameraSetting.h>
#include <SuperRealism+/Environment/STPPerspectiveCameraSetting.h>
#include <SuperRealism+/Environment/STPSunSetting.h>
#include <SuperRealism+/Environment/STPTessellationSetting.h>
#include <SuperRealism+/Environment/STPWaterSetting.h>

#include <glm/trigonometric.hpp>
#include <glm/ext/scalar_constants.hpp>

using glm::uvec2;
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

//STPBidirectionalScatteringSetting.h

STPBidirectionalScatteringSetting::STPBidirectionalScatteringSetting() : 
	MaxRayDistance(1.0f), 
	DepthBias(0.0f), 
	RayResolution(1u), 
	RayStep(1u) {

}

bool STPBidirectionalScatteringSetting::validate() const {
	return this->MaxRayDistance > 0.0f
		&& this->DepthBias > 0.0f
		&& this->RayResolution != 0u
		&& this->RayStep != 0u;
}

//STPCameraSetting.h

STPCameraSetting::STPCameraSetting() : 
	Yaw(radians(-90.0)), Pitch(0.0),
	MovementSpeed(2.5), RotationSensitivity(0.1),
	Position(vec3(0.0)), WorldUp(0.0, 1.0, 0.0), 
	Near(0.1), Far(1.0) {

}

bool STPCameraSetting::validate() const {
	static constexpr auto range = [](double val, double min, double max) constexpr -> double {
		return val > min && val < max;
	};
	static constexpr double PI = glm::pi<double>(), PI_BY_2 = PI * 0.5;

	return range(this->Yaw, -PI, PI)
		&& range(this->Pitch, -PI_BY_2, PI_BY_2)
		&& this->MovementSpeed > 0.0
		&& this->RotationSensitivity > 0.0
		&& this->Near > 0.0
		&& this->Far > 0.0
		&& this->Near < this->Far;
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

STPMeshSetting::STPTextureRegionSmoothSetting::STPTextureRegionSmoothSetting() :
	KernelRadius(1u), 
	KernelScale(1.0f), 
	NoiseScale(1u) {

}

bool STPMeshSetting::STPTextureRegionSmoothSetting::validate() const {
	return this->KernelRadius > 0u
		&& this->KernelScale > 0.0f;
}

STPMeshSetting::STPTextureScaleDistanceSetting::STPTextureScaleDistanceSetting() : 
	PrimaryFar(0.0f), SecondaryFar(0.0f), TertiaryFar(0.0f) {

}

bool STPMeshSetting::STPTextureScaleDistanceSetting::validate() const {
	return this->PrimaryFar > 0.0f
		&& this->SecondaryFar > 0.0f
		&& this->TertiaryFar > 0.0f
		&& this->PrimaryFar <= this->SecondaryFar
		&& this->SecondaryFar <= this->TertiaryFar;
}

STPMeshSetting::STPMeshSetting() :
	Strength(1.0f),
	Altitude(1.0f) {

}

bool STPMeshSetting::validate() const {
	return this->Strength > 0.0f
		&& this->Altitude > 0.0f
		&& this->TessSetting.validate()
		&& this->RegionSmoothSetting.validate()
		&& this->RegionScaleSetting.validate();
}

//STPOcclusionKernelSetting.h

STPOcclusionKernelSetting::STPOcclusionKernelSetting() : 
	RandomSampleSeed(0ull),
	RotationVectorSize(uvec2(1u)), 
	SampleRadius(1.0f), 
	Bias(0.0f) {

}

bool STPOcclusionKernelSetting::validate() const {
	return this->RotationVectorSize != uvec2(0u)
		&& this->SampleRadius > 0.0f
		&& this->Bias > 0.0f;
}

//STPOrthographicCameraSetting.h

STPOrthographicCameraSetting::STPOrthographicCameraSetting() : 
	Left(-1.0), Right(1.0), Bottom(-1.0), Top(1.0) {

}

bool STPOrthographicCameraSetting::validate() const {
	return this->Left < this->Right
		&& this->Bottom < this->Top;
}

//STPPerspectiveCameraSetting.h

STPPerspectiveCameraSetting::STPPerspectiveCameraSetting() :
	ViewAngle(radians(45.0)), ZoomSensitivity(1.0),
	ZoomLimit(radians(1.0), radians(90.0)),
	Aspect(1.0) {

}

bool STPPerspectiveCameraSetting::validate() const {
	static constexpr auto range = [](double val, double min, double max) constexpr -> double {
		return val > min && val < max;
	};

	return range(this->ViewAngle, 0.0, glm::pi<double>() * 2.0)
		&& this->ZoomSensitivity > 0.0
		&& this->ZoomLimit.x > 0.0
		&& this->ZoomLimit.y > 0.0
		&& this->ZoomLimit.x <= this->ZoomLimit.y
		&& this->Aspect > 0.0;
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
	static constexpr auto range_check = [](double val, double min, double max) constexpr -> bool {
		return val >= min && val <= max;
	};

	return this->DayLength > 0ull
		&& ((this->DayLength & 0x01ull) == 0x00ull)//must be an even number
		&& this->DayStartOffset < this->DayLength
		&& this->YearLength > 0u
		&& range_check(this->Obliquity, 0.0, PI_BY_2)
		&& range_check(this->Latitude, -PI_BY_2, PI_BY_2);
}

//STPTessellationSetting.h

STPTessellationSetting::STPTessellationSetting() :
	MaxTessLevel(0.0f),
	MinTessLevel(0.0f),
	FurthestTessDistance(0.0f),
	NearestTessDistance(0.0f) {

}

bool STPTessellationSetting::validate() const {
	return this->MaxTessLevel >= 0.0f
		&& this->MinTessLevel >= 0.0f
		&& this->FurthestTessDistance >= 0.0f
		&& this->NearestTessDistance >= 0.0f
		//range check
		&& this->MaxTessLevel >= this->MinTessLevel
		&& this->FurthestTessDistance >= this->NearestTessDistance;
}

//STPWaterSetting.h

STPWaterSetting::STPWaterWaveSetting::STPWaterWaveSetting() : 
	InitialRotation(0.0f), 
	InitialFrequency(1.0f), 
	InitialAmplitude(1.0f), 
	InitialSpeed(1.0f), 
	OctaveRotation(0.0f), 
	Lacunarity(1.0f), 
	Persistence(1.0f), 
	OctaveSpeed(1.0f), 
	WaveDrag(1.0f) {

}

bool STPWaterSetting::STPWaterWaveSetting::validate() const {
	return true;
}

STPWaterSetting::STPWaterSetting() : 
	MinimumWaterLevel(0.0f), 
	CullTestSample(1u), 
	CullTestRadius(1.0f), 
	Altitude(1.0f),
	WaveHeight(1.0f),
	WaterWaveIteration{ 1u, 1u }, 
	Tint(vec3(0.0f)), 
	NormalEpsilon(1.0f) {

}

bool STPWaterSetting::validate() const {
	return this->MinimumWaterLevel >= 0.0f
		&& this->CullTestSample != 0u
		&& this->CullTestRadius > 0.0f
		&& this->Altitude > 0.0f
		&& this->WaveHeight > 0.0f
		&& this->WaterWaveIteration.Geometry != 0u
		&& this->WaterWaveIteration.Normal != 0u
		&& this->WaterMeshTess.validate() && this->WaterWave.validate()
		&& this->NormalEpsilon > 0.0f;
}