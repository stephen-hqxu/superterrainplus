#pragma once
#ifndef STP_IMPLEMENTATION
#error __FILE__ is a helper header for SQLite include and we do not plan to expose the interfaces to developer
#endif//STP_IMPLEMENTATION

#ifndef _STP_SQLITE_H_
#define _STP_SQLITE_H_

//Please keep this header included in source files only, so client does not need to include sqlite

//Setup environment for sqlite3
//Enable importing visible external interfaces
#ifdef _WIN32
#define SQLITE_API __declspec(dllimport)
#endif//_WIN32

//Include sqlite3
#include <sqlite3.h>

#endif//_STP_SQLITE_H_