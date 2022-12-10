#pragma once
#ifndef _STP_UNSUPPORTED_SYSTEM_H_
#define _STP_UNSUPPORTED_SYSTEM_H_

#include "STPFundamentalException.h"

//create an unsupported system error, given the description of which requirement does not meet
#define STP_UNSUPPORTED_SYSTEM_CREATE(WHICH_REQ) STP_STANDARD_EXCEPTION_CREATE(STPUnsupportedSystem, WHICH_REQ)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPUnsupportedSystem indicates the executing system does not meet the minimum requirement of execution.
	*/
	class STP_API STPUnsupportedSystem : public STPFundamentalException::STPBasic {
	public:

		//A description of which system requirement is violated.
		const std::string ViolatedRequirement;

		/**
		 * @param violated_req Tell the user which requirement is not satisfied.
		*/
		STPUnsupportedSystem(const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPUnsupportedSystem() = default;

	};

}
#endif//_STP_UNSUPPORTED_SYSTEM_H_