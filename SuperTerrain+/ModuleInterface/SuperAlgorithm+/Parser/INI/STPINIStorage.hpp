#pragma once
#ifndef _STP_INI_STORAGE_HPP_
#define _STP_INI_STORAGE_HPP_

#include "STPINISection.hpp"

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPINIStorage loads the sections and keys or properties from an INI file and store them in hash table.
	 * User can read the value of a specific key in a given section with a general run time of O(1).
	 * INI may optionally contain unnamed section, denoted by empty string section name,
	 * for which they are put to the top of the INI file without a section parent.
	*/
	using STPINIStorage = std::unordered_map<std::string, STPINISection>;

}
#endif//_STP_INI_STORAGE_HPP_