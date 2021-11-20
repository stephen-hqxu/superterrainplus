#pragma once
#ifndef _STP_INVALID_ENVIRONMENT_H_
#define _STP_INVALID_ENVIRONMENT_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include "STPInvalidArgument.h"

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPInvalidEnvironment specifies the values from STPEnvironment are not validated
	*/
	class STP_API STPInvalidEnvironment : public STPInvalidArgument {
	public:

		/**
		 * @brief Init STPInvalidEnvironment
		 * @param msg The mssage about the invalid environment
		*/
		explicit STPInvalidEnvironment(const char*);

	};

}
#endif//_STP_INVALID_ENVIRONMENT_H_