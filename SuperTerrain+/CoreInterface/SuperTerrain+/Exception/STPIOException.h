#pragma once
#ifndef _STP_IO_EXCEPTION_H_
#define _STP_IO_EXCEPTION_H_

#include "STPFundamentalException.h"

#define STP_IO_EXCEPTION_CREATE(DESC) STP_STANDARD_EXCEPTION_CREATE(STPIOException, DESC)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPIOException indicates an IO operation failed.
	*/
	class STP_API STPIOException : public STPFundamentalException::STPBasic {
	public:

		//same as the base exception
		STPIOException(const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPIOException() = default;

	};

}
#endif//_STP_IO_EXCEPTION_H_