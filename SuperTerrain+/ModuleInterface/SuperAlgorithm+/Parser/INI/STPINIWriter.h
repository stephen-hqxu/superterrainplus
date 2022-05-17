#pragma once
#ifndef _STP_INI_WRITER_H_
#define _STP_INI_WRITER_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>

#include "STPINIStorage.hpp"

#include <string>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPINIWriter serialises an INI data structure into a string of standard INI format.
	*/
	class STP_ALGORITHM_HOST_API STPINIWriter {
	public:

		//STPWriterFlag controls how the writer should behave.
		typedef unsigned char STPWriterFlag;

		//Add newline between each section.
		static constexpr STPWriterFlag SectionNewline = 1u << 0u;
		//Add a space before and after the assignment of key-value pair.
		//E.g., key=value --> key = value
		static constexpr STPWriterFlag SpaceAroundAssignment = 1u << 1u;
		//Add a space before and after the section name.
		//E.g., [section] --> [ section ]
		static constexpr STPWriterFlag SpaceAroundSectionName = 1u << 2u;

	private:

		//Contains all INI data in a string format.
		std::string Data;

	public:

		/**
		 * @brief Initialise the writer and start formatting into INI string.
		 * @param storage - The storage class where all INI settings are stored.
		 * @param flag - The writer flag to control the behaviour.
		*/
		STPINIWriter(const STPINIStorage&, STPWriterFlag = 0u);

		STPINIWriter(const STPINIWriter&) = default;

		STPINIWriter(STPINIWriter&&) noexcept = default;

		STPINIWriter& operator=(const STPINIWriter&) = default;

		STPINIWriter& operator=(STPINIWriter&&) noexcept = default;

		~STPINIWriter() = default;

		/**
		 * @brief Get the formatted INI string.
		 * @return A pointer to the INI string.
		*/
		const std::string& operator*() const;

	};

}
#endif//_STP_INI_WRITER_H_