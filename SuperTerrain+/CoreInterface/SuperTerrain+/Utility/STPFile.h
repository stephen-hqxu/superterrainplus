#pragma once
#ifndef _STP_FILE_H_
#define _STP_FILE_H_

#include <SuperTerrain+/STPCoreDefine.h>

#include <string>

namespace SuperTerrainPlus {

	/**
	 * @brief STPFile is a handy file IO utility.
	*/
	class STP_API STPFile {
	private:

		std::string Content;

	public:

		/**
		 * @brief Open a file and read all contents in the file.
		 * @param filename The filename to be opened.
		*/
		STPFile(const char*);

		STPFile(const STPFile&) = default;

		STPFile(STPFile&&) noexcept = default;

		STPFile& operator=(const STPFile&) = default;

		STPFile& operator=(STPFile&&) noexcept = default;

		~STPFile() = default;

		/**
		 * @brief Get the whole content with the file being read.
		 * @return The pointer to the string to the content of the file.
		*/
		const std::string& operator*() const;

	};

}
#endif//_STP_FILE_H_