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

#define ASSERT_ATMO(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPAtmosphereSetting)

void STPAtmosphereSetting::validate() const {
	constexpr static vec3 zeroVec3 = vec3(0.0f);

	ASSERT_ATMO(this->SunIntensity > 0.0f);
	ASSERT_ATMO(this->PlanetRadius > 0.0f);
	ASSERT_ATMO(this->AtmosphereRadius > 0.0f);
	ASSERT_ATMO(this->PlanetRadius <= this->AtmosphereRadius);
	ASSERT_ATMO(this->ViewAltitude > 0.0f);
	ASSERT_ATMO(this->RayleighCoefficient != zeroVec3);
	ASSERT_ATMO(this->MieCoefficient > 0.0f);
	ASSERT_ATMO(this->RayleighScale > 0.0f);
	ASSERT_ATMO(this->MieScale > 0.0f);
	ASSERT_ATMO(this->MieScatteringDirection > 0.0f);
	ASSERT_ATMO(this->PrimaryRayStep > 0u);
	ASSERT_ATMO(this->SecondaryRayStep > 0u);
}

//STPAuroraSetting.h

#define ASSERT_AURORA_NOISE(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPAuroraSetting::STPTriangularNoiseSetting::STPNoiseFractalSetting)

void STPAuroraSetting::STPTriangularNoiseSetting::STPNoiseFractalSetting::validate() const {
	ASSERT_AURORA_NOISE(this->InitialAmplitude > 0.0f);
	ASSERT_AURORA_NOISE(this->Persistence > 0.0f);
	ASSERT_AURORA_NOISE(this->Lacunarity > 0.0f);
}

#define ASSERT_AURORA_TRINOISE(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPAuroraSetting::STPTriangularNoiseSetting)

void STPAuroraSetting::STPTriangularNoiseSetting::validate() const {
	this->MainNoise.validate();
	this->DistortionNoise.validate();

	ASSERT_AURORA_TRINOISE(this->InitialDistortionFrequency > 0.0f);
	ASSERT_AURORA_TRINOISE(this->AnimationSpeed > 0.0f);
	ASSERT_AURORA_TRINOISE(this->Contrast > 0.0f);
	ASSERT_AURORA_TRINOISE(this->MaximumIntensity > 0.0f);
	ASSERT_AURORA_TRINOISE(this->Octave > 0u);
}

#define ASSERT_AURORA(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPAuroraSetting)

void STPAuroraSetting::validate() const {
	this->Noise.validate();

	ASSERT_AURORA(this->AuroraSphereFlatness > 0.0f);
	ASSERT_AURORA(this->AuroraPlaneProjectionBias > 0.0f);
	ASSERT_AURORA(this->StepSize > 0.0f);
	ASSERT_AURORA(this->AltitudeFadeStart > this->AltitudeFadeEnd);
	ASSERT_AURORA(this->LuminosityMultiplier > 0.0f);
	ASSERT_AURORA(this->Iteration > 0u);
}

//STPBidirectionalScatteringSetting.h

#define ASSERT_BSDF(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPBidirectionalScatteringSetting)

void STPBidirectionalScatteringSetting::validate() const {
	ASSERT_BSDF(this->MaxRayDistance > 0.0f);
	ASSERT_BSDF(this->DepthBias > 0.0f);
	ASSERT_BSDF(this->RayResolution != 0u);
	ASSERT_BSDF(this->RayStep != 0u);
}

//STPCameraSetting.h

#define ASSERT_CAMERA(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPCameraSetting)

void STPCameraSetting::validate() const {
	static constexpr auto range = [](const double val, const double min, const double max) constexpr noexcept -> double {
		return val > min && val < max;
	};
	static constexpr double Pi = glm::pi<double>(),
		PiByTwo = Pi * 0.5,
		TwoPi = Pi * 2.0;

	ASSERT_CAMERA(range(this->Yaw, -Pi, Pi));
	ASSERT_CAMERA(range(this->Pitch, -PiByTwo, PiByTwo));
	ASSERT_CAMERA(range(this->FoV, 0.0, TwoPi));
	ASSERT_CAMERA(this->MovementSpeed > 0.0);
	ASSERT_CAMERA(this->RotationSensitivity > 0.0);
	ASSERT_CAMERA(this->ZoomSensitivity > 0.0);
	ASSERT_CAMERA(this->ZoomLimit.x > 0.0);
	ASSERT_CAMERA(this->ZoomLimit.y > 0.0);
	ASSERT_CAMERA(this->ZoomLimit.y < TwoPi);
	ASSERT_CAMERA(this->ZoomLimit.x <= this->ZoomLimit.y);
	ASSERT_CAMERA(this->Aspect > 0.0);
	ASSERT_CAMERA(this->Near > 0.0);
	ASSERT_CAMERA(this->Far > 0.0);
	ASSERT_CAMERA(this->Near < this->Far);
}

//STPLightSetting.h

#define ASSERT_AMBIENT_LIGHT(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPLightSetting::STPAmbientLightSetting)

void STPLightSetting::STPAmbientLightSetting::validate() const {
	ASSERT_AMBIENT_LIGHT(this->AmbientStrength >= 0.0f);
}

#define ASSERT_DIRECTIONAL_LIGHT(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPLightSetting::STPDirectionalLightSetting)

void STPLightSetting::STPDirectionalLightSetting::validate() const {
	ASSERT_DIRECTIONAL_LIGHT(this->DiffuseStrength >= 0.0f);
	ASSERT_DIRECTIONAL_LIGHT(this->SpecularStrength >= 0.0f);
}

//STPMeshSetting.h

#define ASSERT_MESH_SMOOTH(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPMeshSetting::STPTextureRegionSmoothSetting)

void STPMeshSetting::STPTextureRegionSmoothSetting::validate() const {
	ASSERT_MESH_SMOOTH(this->KernelRadius > 0u);
	ASSERT_MESH_SMOOTH(this->KernelScale > 0.0f);
}

#define ASSERT_MESH_SCALE(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPMeshSetting::STPTextureScaleDistanceSetting)

void STPMeshSetting::STPTextureScaleDistanceSetting::validate() const {
	constexpr static auto isNormalised = [](const float val) constexpr noexcept -> bool {
		return val > 0.0f && val <= 1.0f;
	};

	ASSERT_MESH_SCALE(isNormalised(this->PrimaryFar));
	ASSERT_MESH_SCALE(isNormalised(this->SecondaryFar));
	ASSERT_MESH_SCALE(isNormalised(this->TertiaryFar));
	ASSERT_MESH_SCALE(this->PrimaryFar <= this->SecondaryFar);
	ASSERT_MESH_SCALE(this->SecondaryFar <= this->TertiaryFar);
}

#define ASSERT_MESH(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPMeshSetting)

void STPMeshSetting::validate() const {
	this->TessSetting.validate();
	this->RegionSmoothSetting.validate();
	this->RegionScaleSetting.validate();

	ASSERT_MESH(this->Strength > 0.0f);
	ASSERT_MESH(this->Altitude > 0.0f);
}

//STPOcclusionKernelSetting.h

#define ASSERT_OCCLUSION(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPOcclusionKernelSetting)

void STPOcclusionKernelSetting::validate() const {
	ASSERT_OCCLUSION(this->RotationVectorSize != uvec2(0u));
	ASSERT_OCCLUSION(this->SampleRadius > 0.0f);
	ASSERT_OCCLUSION(this->Bias > 0.0f);
}

//STPStarfieldSetting.h

#define ASSERT_STAR(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPStarfieldSetting)

void STPStarfieldSetting::validate() const {
	static constexpr auto isNormalised = [](const float num) constexpr noexcept -> bool {
		return num > 0.0f && num < 1.0f;
	};

	ASSERT_STAR(isNormalised(this->InitialLikelihood));
	ASSERT_STAR(this->OctaveLikelihoodMultiplier > 0.0f);
	ASSERT_STAR(this->InitialScale > 0.0f);
	ASSERT_STAR(this->OctaveScaleMultiplier > 0.0f);
	ASSERT_STAR(this->EdgeDistanceFalloff > 0.0f);
	ASSERT_STAR(this->ShineSpeed > 0.0f);
	ASSERT_STAR(this->LuminosityMultiplier > 0.0f);
	ASSERT_STAR(this->Octave > 0u);
}

//STPSunSetting.h

#define ASSERT_SUN(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPSunSetting)

void STPSunSetting::validate() const {
	static constexpr double PiByTwo = glm::pi<double>() * 0.5;
	static constexpr auto range_check = [](const double val, const double min, const double max) constexpr noexcept -> bool {
		return val >= min && val <= max;
	};

	ASSERT_SUN(this->DayLength > 0u);
	ASSERT_SUN(this->DayStart < this->DayLength);
	ASSERT_SUN(this->YearLength> 0u);
	ASSERT_SUN(this->YearStart < this->YearLength);
	ASSERT_SUN(range_check(this->Obliquity, 0.0, PiByTwo));
	ASSERT_SUN(range_check(this->Latitude, -PiByTwo, PiByTwo));
}

//STPTessellationSetting.h

#define ASSERT_TESS(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPTessellationSetting)

void STPTessellationSetting::validate() const {
	static constexpr auto isNormalised = [](const float val) constexpr -> bool {
		return val >= 0.0f && val <= 1.0f;
	};

	ASSERT_TESS(this->MaxTessLevel >= 0.0f);
	ASSERT_TESS(this->MinTessLevel >= 0.0f);
	ASSERT_TESS(isNormalised(this->FurthestTessDistance));
	ASSERT_TESS(isNormalised(this->NearestTessDistance));
	//range check
	ASSERT_TESS(this->MaxTessLevel >= this->MinTessLevel);
	ASSERT_TESS(this->FurthestTessDistance >= this->NearestTessDistance);
}

//STPWaterSetting.h

//STPWaterWaveSetting always validates

#define ASSERT_WATER(EXPR) STP_ASSERTION_ENVIRONMENT(EXPR, STPWaterSetting)

void STPWaterSetting::validate() const {
	this->WaterMeshTess.validate();

	ASSERT_WATER(this->MinimumWaterLevel >= 0.0f);
	ASSERT_WATER(this->CullTestSample != 0u);
	ASSERT_WATER(this->CullTestRadius > 0.0f);
	ASSERT_WATER(this->Altitude > 0.0f);
	ASSERT_WATER(this->WaveHeight > 0.0f);
	ASSERT_WATER(this->WaveHeight <= 1.0f);
	ASSERT_WATER(this->WaterWaveIteration.Geometry != 0u);
	ASSERT_WATER(this->WaterWaveIteration.Normal != 0u);
	ASSERT_WATER(this->NormalEpsilon > 0.0f);
}