#include <SuperRealism+/Environment/STPAtmosphereSetting.h>
#include <SuperRealism+/Environment/STPAuroraSetting.h>
#include <SuperRealism+/Environment/STPBidirectionalScatteringSetting.h>
#include <SuperRealism+/Environment/STPCameraSetting.h>
#include <SuperRealism+/Environment/STPLightSetting.h>
#include <SuperRealism+/Environment/STPMeshSetting.h>
#include <SuperRealism+/Environment/STPOcclusionKernelSetting.h>
#include <SuperRealism+/Environment/STPStarfieldSetting.h>
#include <SuperRealism+/Environment/STPSunSetting.h>
#include <SuperRealism+/Environment/STPTessellationSetting.h>
#include <SuperRealism+/Environment/STPWaterSetting.h>

#include <SuperTerrain+/Exception/STPInvalidEnvironment.h>

#include <glm/trigonometric.hpp>
#include <glm/ext/scalar_constants.hpp>

using glm::uvec2;
using glm::dvec2;
using glm::dvec3;
using glm::vec3;
using glm::radians;

using namespace SuperTerrainPlus::STPEnvironment;

//STPAtmosphereSetting.h

void STPAtmosphereSetting::validate() const {
	constexpr static vec3 zeroVec3 = vec3(0.0f);

	if (this->SunIntensity > 0.0f
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
		&& this->SecondaryRayStep > 0u) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPAtmosphereSetting validation fails");
}

//STPAuroraSetting.h

void STPAuroraSetting::STPTriangularNoiseSetting::STPNoiseFractalSetting::validate() const {
	if (this->InitialAmplitude > 0.0f
		&& this->Persistence > 0.0f
		&& this->Lacunarity > 0.0f) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPNoiseFractalSetting validation fails");
}

void STPAuroraSetting::STPTriangularNoiseSetting::validate() const {
	this->MainNoise.validate();
	this->DistortionNoise.validate();

	if (this->InitialDistortionFrequency > 0.0f
		&& this->AnimationSpeed > 0.0f
		&& this->Contrast > 0.0f
		&& this->MaximumIntensity > 0.0f
		&& this->Octave > 0u) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPTriangularNoiseSetting validation fails");
}

void STPAuroraSetting::validate() const {
	this->Noise.validate();

	if (this->AuroraSphereFlatness > 0.0f
		&& this->AuroraPlaneProjectionBias > 0.0f
		&& this->StepSize > 0.0f
		&& this->AltitudeFadeStart > this->AltitudeFadeEnd
		&& this->LuminosityMultiplier > 0.0f
		&& this->Iteration > 0u) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPAuroraSetting validation fails");
}

//STPBidirectionalScatteringSetting.h

void STPBidirectionalScatteringSetting::validate() const {
	if (this->MaxRayDistance > 0.0f
		&& this->DepthBias > 0.0f
		&& this->RayResolution != 0u
		&& this->RayStep != 0u) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPBidirectionalScatteringSetting validation fails");
}

//STPCameraSetting.h

void STPCameraSetting::validate() const {
	static constexpr auto range = [](const double val, const double min, const double max) constexpr -> double {
		return val > min && val < max;
	};
	static constexpr double Pi = glm::pi<double>(),
		PiByTwo = Pi * 0.5,
		TwoPi = Pi * 2.0;

	if (range(this->Yaw, -Pi, Pi)
		&& range(this->Pitch, -PiByTwo, PiByTwo)
		&& range(this->FoV, 0.0, TwoPi)
		&& this->MovementSpeed > 0.0
		&& this->RotationSensitivity > 0.0
		&& this->ZoomSensitivity > 0.0
		&& this->ZoomLimit.x > 0.0
		&& this->ZoomLimit.y > 0.0
		&& this->ZoomLimit.y < TwoPi
		&& this->ZoomLimit.x <= this->ZoomLimit.y
		&& this->Aspect > 0.0
		&& this->Near > 0.0
		&& this->Far > 0.0
		&& this->Near < this->Far) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPCameraSetting validation fails");
}

//STPLightSetting.h

void STPLightSetting::STPAmbientLightSetting::validate() const {
	if (this->AmbientStrength >= 0.0f) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPAmbientLightSetting validation fails");
}

void STPLightSetting::STPDirectionalLightSetting::validate() const {
	if (this->DiffuseStrength >= 0.0f
		&& this->SpecularStrength >= 0.0f) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPDirectionalLightSetting validation fails");
}

//STPMeshSetting.h

void STPMeshSetting::STPTextureRegionSmoothSetting::validate() const {
	if (this->KernelRadius > 0u
		&& this->KernelScale > 0.0f) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPTextureRegionSmoothSetting validation fails");
}

void STPMeshSetting::STPTextureScaleDistanceSetting::validate() const {
	constexpr static auto isNormalised = [](const float val) constexpr -> bool {
		return val > 0.0f && val <= 1.0f;
	};

	if (isNormalised(this->PrimaryFar)
		&& isNormalised(this->SecondaryFar)
		&& isNormalised(this->TertiaryFar)
		&& this->PrimaryFar <= this->SecondaryFar
		&& this->SecondaryFar <= this->TertiaryFar) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPTextureScaleDistanceSetting validation fails");
}

void STPMeshSetting::validate() const {
	this->TessSetting.validate();
	this->RegionSmoothSetting.validate();
	this->RegionScaleSetting.validate();

	if (this->Strength > 0.0f
		&& this->Altitude > 0.0f) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPMeshSetting validation fails");
}

//STPOcclusionKernelSetting.h

void STPOcclusionKernelSetting::validate() const {
	if (this->RotationVectorSize != uvec2(0u)
		&& this->SampleRadius > 0.0f
		&& this->Bias > 0.0f) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPOcclusionKernelSetting validation fails");
}

//STPStarfieldSetting.h

void STPStarfieldSetting::validate() const {
	static constexpr auto isNormalised = [](const float num) constexpr -> bool {
		return num > 0.0f && num < 1.0f;
	};

	if (isNormalised(this->InitialLikelihood)
		&& this->OctaveLikelihoodMultiplier > 0.0f
		&& this->InitialScale > 0.0f
		&& this->OctaveScaleMultiplier > 0.0f
		&& this->EdgeDistanceFalloff > 0.0f
		&& this->ShineSpeed > 0.0f
		&& this->LuminosityMultiplier > 0.0f
		&& this->Octave > 0u) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPStarfieldSetting validation fails");
}

//STPSunSetting.h

void STPSunSetting::validate() const {
	static constexpr double PiByTwo = glm::pi<double>() * 0.5;
	static constexpr auto range_check = [](const double val, const double min, const double max) constexpr -> bool {
		return val >= min && val <= max;
	};

	if (this->DayLength > 0u
		&& this->DayStart < this->DayLength
		&& this->YearLength> 0u
		&& this->YearStart < this->YearLength
		&& range_check(this->Obliquity, 0.0, PiByTwo)
		&& range_check(this->Latitude, -PiByTwo, PiByTwo)) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPSunSetting validation fails");
}

//STPTessellationSetting.h

void STPTessellationSetting::validate() const {
	static constexpr auto isNormalised = [](const float val) constexpr -> bool {
		return val >= 0.0f && val <= 1.0f;
	};

	if (this->MaxTessLevel >= 0.0f
		&& this->MinTessLevel >= 0.0f
		&& isNormalised(this->FurthestTessDistance)
		&& isNormalised(this->NearestTessDistance)
		//range check
		&& this->MaxTessLevel >= this->MinTessLevel
		&& this->FurthestTessDistance >= this->NearestTessDistance) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPTessellationSetting validation fails");
}

//STPWaterSetting.h

//STPWaterWaveSetting always validates

void STPWaterSetting::validate() const {
	this->WaterMeshTess.validate();

	if (this->MinimumWaterLevel >= 0.0f
		&& this->CullTestSample != 0u
		&& this->CullTestRadius > 0.0f
		&& this->Altitude > 0.0f
		&& this->WaveHeight > 0.0f
		&& this->WaveHeight <= 1.0f
		&& this->WaterWaveIteration.Geometry != 0u
		&& this->WaterWaveIteration.Normal != 0u
		&& this->NormalEpsilon > 0.0f) {
		return;
	}
	throw STPException::STPInvalidEnvironment("STPWaterSetting validation fails");
}