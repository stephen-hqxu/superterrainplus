#include <SuperAlgorithm+/Device/STPTextureSplatRuleWrapper.cuh>

//Math Utils
#include <SuperAlgorithm+/Device/STPKernelMath.cuh>

using namespace SuperTerrainPlus::STPCompute;

namespace STPTI = SuperTerrainPlus::STPDiversity::STPTextureInformation;
using SuperTerrainPlus::STPDiversity::Sample;

__device__ STPTextureSplatRuleWrapper::STPTextureSplatRuleWrapper(const STPTI::STPSplatRuleDatabase& splat_database) : SplatRule(splat_database) {

}

__device__ STPTextureSplatRuleWrapper::~STPTextureSplatRuleWrapper() {

}

__device__ const STPTI::STPSplatRegistry* STPTextureSplatRuleWrapper::getSplatRegistry(Sample sample) const {
	//binary search this sample in the registry dictionary
	//here because biomemap is generally a large scale texture, most threads in a half warp should have the same sample so
	//memory access should basically be aligned
	const Sample* dic_beg = this->SplatRule.SplatRegistryDictionary,
		*dic_end = dic_beg + this->SplatRule.DictionaryEntryCount,
		*dic_it = STPKernelMath::lower_bound(dic_beg, dic_end, sample);
	if(dic_it == dic_end) {
		//not found
		return nullptr;
	}
	
	//found, get the index
	const unsigned int registry_index = dic_it - dic_beg;
	return this->SplatRule.SplatRegistry + registry_index;
}

__device__ STPTextureSplatRuleWrapper::STPRegionIndex STPTextureSplatRuleWrapper::altitudeRegion(Sample sample, float alt) const {
	//get the splat registry first
	const STPTI::STPSplatRegistry* const reg = this->getSplatRegistry(sample);
	if(reg == nullptr) {
		//no active region can be used
		return STPTextureSplatRuleWrapper::NoRegion;
	}
	
	//search for the altitude given
	//altitude upper bounds are all sorted, so we can use binary search
	const STPTI::STPAltitudeNode* alt_beg = this->SplatRule.AltitudeRegistry + reg->AltitudeEntry,
		*alt_end = alt_beg + reg->AltitudeSize,
		*alt_it = STPKernelMath::lower_bound(alt_beg, alt_end, alt, 
			[]__device__(const auto& node, const auto& val) { return node.UpperBound < val; });
	if(alt_it == alt_end) {
		//no altitude is found
		return STPTextureSplatRuleWrapper::NoRegion;
	}
	
	//get the region index for this altitude
	return alt_it->Reference.RegionIndex;
}

__device__ STPTextureSplatRuleWrapper::STPRegionIndex STPTextureSplatRuleWrapper::gradientRegion
	(STPDiversity::Sample sample, float gra, float alt) const {
	//get registry
	const STPTI::STPSplatRegistry* const reg = this->getSplatRegistry(sample);
	if(reg == nullptr) {
		return STPTextureSplatRuleWrapper::NoRegion;
	}
	
	//search for the gradient given
	for(unsigned int i = reg->GradientEntry; i < reg->GradientSize; i++) {
		STPTI::STPGradientNode& curr_gra = this->SplatRule.GradientRegistry[i];
		//check all rules
		if(gra >= curr_gra.minGradient && gra <= curr_gra.maxGradient &&
			alt >= curr_gra.LowerBound && alt <= curr_gra.UpperBound) {
			//found a matched rule
			return curr_gra.Reference.RegionIndex;
		}
	}
	//otherwise no mathched rule is found
	return STPTextureSplatRuleWrapper::NoRegion;
}