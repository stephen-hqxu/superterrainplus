#include <SuperAlgorithm+Device/STPTextureSplatRuleWrapper.cuh>

#include <cstddef>

using namespace SuperTerrainPlus::STPAlgorithm;

namespace STPTI = SuperTerrainPlus::STPDiversity::STPTextureInformation;
using SuperTerrainPlus::STPSample_t;

/**
 * @brief Perform lower bound binary search with custom comparator.
 * Please only perform lower bound search for small types.
 * @tparam It The type of the array.
 * @tparam T The type to be searched.
 * @tparam Comp The comparator, it must guarantee the object is strictly less than.
 * @param first The pointer to the beginning of the array.
 * @param last The pointer to the end of the array.
 * @param value The value to be searched.
 * @param comparator The custom comparator for "less than". See C++ specification to learn the comparator signature.
 * @return The pointer to the value pointing to the first element in the range [first, last) that is not less than (i.e.
 * greater or equal to) value, or last if no such element is found.
*/
template<class It, class T, class Comp>
__device__ static const It* lowerBound(const It* first, const It* const last, const T value, Comp&& comp) {
	//std::lower_bound implementation
	const It* it;
	std::ptrdiff_t count = last - first, step;

	while (count > 0) {
		it = first;
		step = count / 2;
		//advance
		it += step;
		if (std::forward<Comp>(comp)(*it, value)) {
			first = ++it;
			count -= step + 1;
		} else {
			count = step;
		}
	}
	return first;
}

/**
 * @brief Perform lower bound binary search.
 * @tparam T The type to be searched.
 * @param first The pointer to the beginning of the array.
 * @param last The pointer to the end of the array.
 * @param value The value to be searched.
 * @return The pointer to the value pointing to the first element in the range [first, last) that is not less than (i.e.
 * greater or equal to) value, or last if no such element is found.
*/
template<class T>
__device__ __forceinline__ static const T* lowerBound(const T* const first, const T* const last, const T value) {
	constexpr static auto lessThan = [] __device__(const T current, const T value) constexpr -> bool {
		return current < value;
	};
	return lowerBound(first, last, value, lessThan);
}

__device__ STPTextureSplatRuleWrapper::STPTextureSplatRuleWrapper(const STPTI::STPSplatRuleDatabase& splat_db) : SplatRule(splat_db) {

}

__device__ STPTextureSplatRuleWrapper::~STPTextureSplatRuleWrapper() {

}

__device__ const STPTI::STPSplatRegistry* STPTextureSplatRuleWrapper::findSplatRegistry(const STPSample_t sample) const {
	//binary search this sample in the registry dictionary
	//here because biomemap is generally a large scale texture, most threads in a half warp should have the same sample
	//so memory access should basically be aligned
	const STPSample_t* const dic_beg = this->SplatRule.SplatRegistryDictionary,
		*const dic_end = dic_beg + this->SplatRule.DictionaryEntryCount,
		*const dic_it = lowerBound(dic_beg, dic_end, sample);
	if (dic_it == dic_end) {
		//not found
		return nullptr;
	}

	//found, get index
	const std::ptrdiff_t registry_idx = dic_it - dic_beg;
	return this->SplatRule.SplatRegistry + registry_idx;
}

__device__ unsigned int STPTextureSplatRuleWrapper::altitudeRegion(const STPTI::STPSplatRegistry* const  splat_reg, const float alt) const {
	const STPTI::STPAltitudeNode* const alt_node = this->SplatRule.AltitudeRegistry;
	if (!splat_reg) {
		//no active region can be used
		return STPTextureSplatRuleWrapper::NoRegion;
	}

	//search for the altitude given
	//altitude upper bounds are all sorted, so we can use binary search
	const STPTI::STPAltitudeNode* const alt_beg = alt_node + splat_reg->AltitudeEntry,
		*const alt_end = alt_beg + splat_reg->AltitudeSize,
		*const alt_it = lowerBound(alt_beg, alt_end, alt, [] __device__(const auto& node, const auto val){ return node.UpperBound < val; });
	if (alt_it == alt_end) {
		//no altitude is found
		return STPTextureSplatRuleWrapper::NoRegion;
	}

	//get the region index for this altitude
	return alt_it->Reference.RegionIndex;
}

__device__ unsigned int STPTextureSplatRuleWrapper::gradientRegion(const STPTI::STPSplatRegistry* const splat_reg, const float gra, const float alt) const {
	const STPTI::STPGradientNode* const gra_node = this->SplatRule.GradientRegistry;
	if (!splat_reg) {
		return STPTextureSplatRuleWrapper::NoRegion;
	}

	//search for the gradient given
	const unsigned int begin = splat_reg->GradientEntry,
		end = begin + splat_reg->GradientSize;
	for (unsigned int i = begin; i < end; i++) {
		const STPTI::STPGradientNode& curr_gra = gra_node[i];
		//check all rules
		if (gra >= curr_gra.minGradient && gra <= curr_gra.maxGradient && alt >= curr_gra.LowerBound
			&& alt <= curr_gra.UpperBound) {
			//found a matched rule
			return curr_gra.Reference.RegionIndex;
		}
	}
	//otherwise no matched rule is found
	return STPTextureSplatRuleWrapper::NoRegion;
}