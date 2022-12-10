#pragma once
#ifndef _STP_SQL_ERROR_H_
#define _STP_SQL_ERROR_H_

#include "../STPFundamentalException.h"

//manually creating a SQL error with custom message
#define STP_SQL_ERROR_CREATE(MSG) STP_STANDARD_EXCEPTION_CREATE(STPSQLError, MSG)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPSQLError signals an error during execution of SQL API.
	*/
	class STP_API STPSQLError : public STPFundamentalException::STPBasic {
	public:

		/**
		 * @param err_str The error string from the SQL API.
		*/
		STPSQLError(const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPSQLError() = default;

	};

}
#endif//_STP_SQL_ERROR_H_