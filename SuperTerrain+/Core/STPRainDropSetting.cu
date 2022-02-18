//STPRainDropSetting header will be included by the private implementation header
//Private implementation for the erosion brush generator
#define STP_RAIN_DROP_SETTING_IMPL
#include <SuperTerrain+/Environment/STPErosionBrushGenerator.h>

using std::make_unique;

__host__ STPRainDropSetting::STPRainDropSetting() :
	STPSetting(),
	ErosionBrushRadius(0u),
	BrushSize(0u),
	RainDropCount(0u),
	Inertia(0.0f),
	SedimentCapacityFactor(1.0f),
	minSedimentCapacity(0.0f),
	initWaterVolume(1.0f),
	minWaterVolume(0.0f),
	Friction(0.0f),
	initSpeed(0.0f),
	ErodeSpeed(0.0f),
	DepositSpeed(0.0f),
	EvaporateSpeed(0.0f),
	Gravity(1.0f),
	ErosionBrushIndices(nullptr),
	ErosionBrushWeights(nullptr),
	//PIMPL
	ErosionBrushGenerator(make_unique<STPErosionBrushGenerator>(*this)) {

}

__host__ STPRainDropSetting::STPRainDropSetting(STPRainDropSetting&&) noexcept = default;

__host__ STPRainDropSetting& STPRainDropSetting::operator=(STPRainDropSetting&&) noexcept = default;

__host__ STPRainDropSetting::~STPRainDropSetting() {

}

__host__ bool STPRainDropSetting::validate() const {
	static constexpr auto checkRange = []__host__(float value, float lower, float upper) constexpr -> bool {
		return value >= lower && value <= upper;
	};

	return checkRange(this->Inertia, 0.0f, 1.0f)
		&& this->SedimentCapacityFactor > 0.0f
		&& this->minSedimentCapacity >= 0.0f
		&& this->initWaterVolume > 0.0f
		&& this->minWaterVolume >= 0.0f
		&& checkRange(this->Friction, 0.0f, 1.0f)
		&& this->initSpeed >= 0.0f
		&& checkRange(this->ErodeSpeed, 0.0f, 1.0f)
		&& checkRange(this->DepositSpeed, 0.0f, 1.0f)
		&& checkRange(this->EvaporateSpeed, 0.0f, 1.0f)
		&& this->Gravity > 0.0f
		&& this->ErosionBrushRadius != 0u
		&& this->BrushSize != 0u
		&& this->ErosionBrushIndices != nullptr
		&& this->ErosionBrushWeights != nullptr;
}

__host__ void STPRainDropSetting::setErosionBrushRadius(uvec2 slipRange, unsigned int erodeRadius) {
	(*this->ErosionBrushGenerator)(slipRange, erodeRadius);
}

__host__ __device__ unsigned int STPRainDropSetting::getErosionBrushRadius() const {
	return this->ErosionBrushRadius;
}

__host__ __device__ unsigned int STPRainDropSetting::getErosionBrushSize() const {
	return this->BrushSize;
}

__device__ int* STPRainDropSetting::getErosionBrushIndices() const {
	return this->ErosionBrushIndices;
}

__device__ float* STPRainDropSetting::getErosionBrushWeights() const {
	return this->ErosionBrushWeights;
}