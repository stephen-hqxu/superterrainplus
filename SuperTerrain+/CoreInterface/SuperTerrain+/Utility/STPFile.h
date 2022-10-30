#pragma once
#ifndef _STP_FILE_H_
#define _STP_FILE_H_

#include <SuperTerrain+/STPCoreDefine.h>

#include <string>

namespace SuperTerrainPlus {

	/**
	 * @brief STPFile is a handy file IO utility.
	*/
	namespace STPFile {

		/**
		 * @brief Open a file and read all contents in the file.
		 * @param filename The filename of the file to be read.
		 * @return A string containing all text contents of the file.
		*/
		STP_API std::string read(const char*);

	}

}
#endif//_STP_FILE_H_