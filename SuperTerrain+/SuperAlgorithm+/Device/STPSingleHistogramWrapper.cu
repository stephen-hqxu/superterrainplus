#pragma once
#include <SuperAlgorithm+/Device/STPSingleHistogramWrapper.cuh>

//Histogram
#include <SuperAlgorithm+/STPSingleHistogram.hpp>

using namespace SuperTerrainPlus::STPCompute;

template<typename I>
__device__ STPSingleHistogramWrapper<I>::STPSingleHistogramWrapper(const STPSingleHistogram_t histogram) : Histogram(histogram) {
	
}

template<typename I>
__device__ STPSingleHistogramWrapper<I>::~STPSingleHistogramWrapper() {
	
}

template<typename I>
__device__ unsigned int STPSingleHistogramWrapper<I>::operator[](unsigned int pixel_index) const {
	return this->Histogram->HistogramStartOffset[pixel_index];
}

template<typename I>
__device__ STPSingleHistogramWrapper<I>::STPSingleHistogramBin_ct STPSingleHistogramWrapper<I>::getBin(unsigned int bin_index) const {
	return static_cast<STPSingleHistogramBin_ct>(this->Histogram->Bin + bin_index);
}

template<typename I>
__device__ I STPSingleHistogramWrapper<I>::getItem(STPSingleHistogramBin_ct bin) const {
	return static_cast<const STPSingleHistogram::STPBin*>(bin)->Item;
}

template<typename I>
__device__ float STPSingleHistogramWrapper<I>::getWeight(STPSingleHistogramBin_ct bin) const {
	return static_cast<const STPSingleHistogram::STPBin*>(bin)->Data.Weight;
}

//Explicit instantiation
template class STPSingleHistogramWrapper<SuperTerrainPlus::STPDiversity::Sample>;