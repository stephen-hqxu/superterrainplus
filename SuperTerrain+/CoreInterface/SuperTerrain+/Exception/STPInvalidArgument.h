#pragma once
#ifndef _STP_INVALID_ARGUMENT_H_
#define _STP_INVALID_ARGUMENT_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPInvalidArgument states the function parameter is not valid for the function call
	*/
	class STP_API STPInvalidArgument : public std::invalid_argument {
	public:

		/**
		 * @brief Init STPInvalidArgument
		 * @param msg Meesage about the invalid argument
		*/
		explicit STPInvalidArgument(const char*);

	};

}
#endif//_STP_INVALID_ARGUMENT_H_