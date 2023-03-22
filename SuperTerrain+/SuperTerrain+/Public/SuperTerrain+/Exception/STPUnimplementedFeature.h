#pragma once
#ifndef _STP_UNIMPLEMENTED_FEATURE_H_
#define _STP_UNIMPLEMENTED_FEATURE_H_

#include "STPFundamentalException.h"

//create an unimplemented feature error, given the description of the feature
#define STP_UNIMPLEMENTED_FEATURE_CREATE(FEAT) STP_STANDARD_EXCEPTION_CREATE(STPUnimplementedFeature, FEAT)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPUnimplementedFeature indicates that the error is caused by use of a feature that is currently unavailable.
	*/
	class STP_API STPUnimplementedFeature : public STPFundamentalException::STPBasic {
	public:

		//A description of which feature is not implemented
		const std::string Feature;

		/**
		 * @param feature_desc Description of the feature that is not implemented.
		*/
		STPUnimplementedFeature(const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPUnimplementedFeature() = default;

	};

}
#endif//_STP_UNIMPLEMENTED_FEATURE_H_