#pragma once
#ifndef _STP_TEXTURE_DEFINITION_LANGUAGE_H_
#define _STP_TEXTURE_DEFINITION_LANGUAGE_H_

#include <SuperAlgorithm+Host/STPAlgorithmDefine.h>
//Texture Database
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>

//System
#include <string_view>
#include <limits>
//Container
#include <vector>
#include <unordered_map>
#include <tuple>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPTextureDefinitionLanguage is a parser for SuperTerrain+ Texture Definition Language - TDL.
	 * TDL allow user to define texture splatting rules for terrain texturing.
	 * For more information please refer to the specification of the language.
	*/
	namespace STPTextureDefinitionLanguage {

		/**
		 * @brief Contains parsing result from an input TDL script.
		*/
		struct STP_ALGORITHM_HOST_API STPResult {
		public:

			//A table of texture variable, corresponds to texture ID and the belonging view group ID.
			typedef std::unordered_map<std::string_view, std::pair<
				STPDiversity::STPTextureInformation::STPTextureID,
				STPDiversity::STPTextureInformation::STPViewGroupID>
			> STPTextureVariable;

			//Indicates an index that is not pointing to any thing, i.e., null index.
			constexpr static size_t UnreferencedIndex = std::numeric_limits<size_t>::max();

			//View groups declared.
			std::vector<STPDiversity::STPTextureDatabase::STPViewGroupDescription> DeclaredViewGroup;
			//For each texture name, maps to an index to the view group data structure.
			std::unordered_map<std::string_view, size_t> DeclaredTexture;
			//The altitude rules for each sample.
			std::vector<std::tuple<STPDiversity::Sample, float, std::string_view>> Altitude;
			//The gradient rules for each sample.
			std::vector<std::tuple<STPDiversity::Sample, float, float, float, float, std::string_view>> Gradient;

			/**
			 * @brief Load all defined texture rules into a texture database.
			 * Note that if any texture rule has been defined in the database exception will be thrown since no duplicated rules can exist.
			 * @param database The pointer to the database for which rules will be added.
			 * @return A table of variable name to texture ID within this database, it can be used to uploaded texture data and assign texture group.
			 * The string key in the table is a view to the raw TDL string view provided by the user during parsing, and is valid
			 * only when the raw string which the view references from is alive.
			*/
			STPTextureVariable load(STPDiversity::STPTextureDatabase&) const;

		};

		/**
		 * @brief Read and parse a TDL source input.
		 * @param source The null-terminated string of the source code of TDL.
		 * The memory of the string is managed by the user, including all string view in the return result.
		 * @param source_name The name of the source code.
		 * This name will appear in the error message if any error is encountered during reading.
		 * @return The parsed result, whose string memory is a view of the input source code.
		*/
		STP_ALGORITHM_HOST_API STPResult read(const std::string_view&, const std::string_view&);

	}

}
#endif//_STP_TEXTURE_DEFINITION_LANGUAGE_H_