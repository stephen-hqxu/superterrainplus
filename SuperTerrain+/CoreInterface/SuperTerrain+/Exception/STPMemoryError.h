#pragma once
#ifndef _STP_MEMORY_ERROR_H_
#define _STP_MEMORY_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPMemoryError tells general error about memory allocation, access and deallocation
	*/
	class STP_API STPMemoryError : public std::logic_error {
	public:

		/**
		 * @brief Init STPMemoryError
		 * @param msg The message about memory error
		*/
		explicit STPMemoryError(const char*);

	};

}
#endif//_STP_MEMORY_ERROR_H_