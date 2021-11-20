#pragma once
#ifndef _STP_UNSUPPORTED_FUNCTIONALITY_H_
#define _STP_UNSUPPORTED_FUNCTIONALITY_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include "STPInvalidArgument.h"

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPUnsupportedFunctionality states the functionality that user is asking for is not (yet) supported or valid.
	*/
	class STP_API STPUnsupportedFunctionality : public STPInvalidArgument {
	public:

		/**
		 * @brief Init STPUnsupportedFunctionality
		 * @param msg The message for what functionality is not supported
		*/
		explicit STPUnsupportedFunctionality(const char*);

	};

}
#endif//_STP_UNSUPPORTED_FUNCTIONALITY_H_