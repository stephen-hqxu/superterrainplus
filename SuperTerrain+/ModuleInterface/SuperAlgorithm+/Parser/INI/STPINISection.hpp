#pragma once
#ifndef _STP_INI_SECTION_HPP_
#define _STP_INI_SECTION_HPP_

//Container
#include <unordered_map>

#include "STPINIString.h"

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPINISection is a storage class for sections in INI file, each section contains multiple properties (or keys)
	 * Keys may (but need not) be grouped into arbitrarily named sections.
	 * The section name appears on a line by itself, in square brackets ([ and ]).
	 * All keys after the section declaration are associated with that section.
	 * There is no explicit "end of section" delimiter; sections end at the next section declaration, or the end of the file.
	 * Sections may not be nested.
	 * ----------------------------------------------------------------------------------------------------------------
	 * One section contains properties that store the property of the INI file.
	 * The basic element contained in an INI file is the key or property.
	 * Every key has a name and a value, delimited by an equals sign (=).
	 * The name appears to the left of the equals sign.
	 * In the Windows implementation the key cannot contain the characters equal sign ( = ) or semicolon ( ; ) as these are reserved characters.
	 * The value can contain any character.
	*/
	using STPINISection = std::unordered_map<std::string, STPINIString>;

}
#endif//_STP_INI_SECTION_HPP_