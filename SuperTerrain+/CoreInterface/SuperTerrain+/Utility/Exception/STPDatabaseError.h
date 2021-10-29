#pragma once
#ifndef _STP_DATABASE_ERROR_H_
#define _STP_DATABASE_ERROR_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Exception
#include <stdexcept>

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPDatabaseError indicates error from database system
	*/
	class STP_API STPDatabaseError : public std::runtime_error {
	public:

		/**
		 * @brief Init STPDatabaseError
		 * @param msg Message to report related database error
		*/
		explicit STPDatabaseError(const char*);

	};

}
#endif//_STP_DATABASE_ERROR_H_