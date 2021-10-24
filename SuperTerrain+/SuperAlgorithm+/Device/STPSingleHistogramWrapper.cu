#include <SuperAlgorithm+/Device/STPSingleHistogramWrapper.cuh>

using namespace SuperTerrainPlus::STPCompute;

using SuperTerrainPlus::STPDiversity::Sample;

__device__ STPSingleHistogramWrapper::STPSingleHistogramWrapper(const STPSingleHistogram& histogram) : Histogram(histogram) {
	
}

__device__ STPSingleHistogramWrapper::~STPSingleHistogramWrapper() {
	
}

__device__ unsigned int STPSingleHistogramWrapper::operator[](unsigned int pixel_index) const {
	return this->Histogram.HistogramStartOffset[pixel_index];
}

__device__ STPSingleHistogramWrapper::STPSingleHistogramBin_ct STPSingleHistogramWrapper::getBin(unsigned int bin_index) const {
	return this->Histogram.Bin[bin_index];
}

__device__ Sample STPSingleHistogramWrapper::getItem(STPSingleHistogramBin_ct bin) const {
	return bin.Item;
}

__device__ float STPSingleHistogramWrapper::getWeight(STPSingleHistogramBin_ct bin) const {
	return bin.Data.Weight;
}