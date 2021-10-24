#pragma once
#ifndef _STP_BAD_NUMERIC_RANGE_H_
#define _STP_BAD_NUMERIC_RANGE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include "STPInvalidArgument.h"

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPBadNumericRange defines number provided is out-of legal range and will cause fatal error if proceed
	*/
	class STP_API STPBadNumericRange : public STPInvalidArgument {
	public:

		/**
		 * @brief Init STPBadNumericRange
		 * @param msg The message about the numeric error
		*/
		explicit STPBadNumericRange(const char*);

	};
}
#endif//_STP_BAD_NUMERIC_RANGE_H_