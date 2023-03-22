#pragma once
#ifndef _STP_INI_DATA_HPP_
#define _STP_INI_DATA_HPP_

#include "./Framework/STPBasicStringAdaptor.h"

#include <unordered_map>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPINIData defines data structure used by the INI reader and writer.
	*/
	namespace STPINIData {

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
		typedef std::unordered_map<std::string, STPStringAdaptor> STPINISection;
		//A non-owning version of STPINISection
		//@see STPINISection
		typedef std::unordered_map<std::string_view, STPStringViewAdaptor> STPINISectionView;

		/**
		 * @brief STPINIStorage loads the sections and keys or properties from an INI file and store them in hash table.
		 * User can read the value of a specific key in a given section with a general run time of O(1).
		 * INI may optionally contain unnamed section, denoted by empty string section name,
		 * for which they are put to the top of the INI file without a section parent.
		*/
		typedef std::unordered_map<std::string, STPINISection> STPINIStorage;
		//A non-owning version of STPINIStorage
		//@see STPINIStorage
		typedef std::unordered_map<std::string_view, STPINISectionView> STPINIStorageView;

		/**
		 * @brief STPINIndex maintains an indexed order for each entry, either INI section or key-value pairs.
		 * This is useful when writing INI data to INI string to specify the order of each entry.
		 * The index starts from 0, whose entry will be placed at the beginning, followed by the entries with index in increasing order.
		*/
		typedef std::unordered_map<std::string, size_t> STPINIEntryIndex;
		//A non-owning version of STPINIEntryIndex
		//@see STPINIEntryIndex
		typedef std::unordered_map<std::string_view, size_t> STPINIEntryIndexView;

	}

}
#endif//_STP_INI_DATA_HPP_