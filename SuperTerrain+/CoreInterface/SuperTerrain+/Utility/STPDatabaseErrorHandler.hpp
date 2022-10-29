#include <SuperTerrain+/Utility/STPGenericErrorHandlerBlueprint.hpp>

#ifdef SQLITE3_H
#define STP_HAS_SQLITE3
#endif

//#include <sqlite3.h>
#ifdef STP_HAS_SQLITE3
#ifndef _STP_DATABASE_ERROR_HANDLER_HPP_
#define _STP_DATABASE_ERROR_HANDLER_HPP_
#include <SuperTerrain+/Exception/STPDatabaseError.h>

STP_ERROR_DESCRIPTOR(assertSqlite3, int, SQLITE_OK, STPException::STPDatabaseError) {
	msg_str << "SQLite: " << sqlite3_errstr(error_code);
}
#define STP_CHECK_SQLITE3(ERR) STP_INVOKE_ERROR_DESCRIPTOR(assertSqlite3, ERR)

#endif//_STP_DATABASE_ERROR_HANDLER_HPP_
#endif//STP_HAS_SQLITE3

#undef STP_HAS_SQLITE3