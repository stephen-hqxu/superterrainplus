#pragma once
#ifndef _STP_TEXTURE_DEFINITION_LANGUAGE_H_
#define _STP_TEXTURE_DEFINITION_LANGUAGE_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
//Texture Database
#include <SuperTerrain+/World/Diversity/Texture/STPTextureDatabase.h>

//Memory
#include <memory>
//System
#include <string>
#include <string_view>
#include <limits>
//Container
#include <vector>
#include <unordered_map>
#include <tuple>

namespace SuperTerrainPlus::STPAlgorithm {

	/**
	 * @brief STPTextureDefinitionLanguage is a lexer and parser for SuperTerrain+ Texture Definition Language - TDL.
	 * For more information please refer to the specification of the language.
	*/
	class STP_ALGORITHM_HOST_API STPTextureDefinitionLanguage {
	private:

		/**
		 * @brief STPTDLLexer is a lexical analysis tool and token generator for SuperTerrain+ Texture Definition Language.
		*/
		class STPTDLLexer;

		std::unique_ptr<STPTDLLexer> Lexer;

		//Indicates an index that is not pointing to any thing, i.e., null index.
		constexpr static size_t UnreferencedIndex = std::numeric_limits<size_t>::max();

		//Information from the source code after lexing and parsing
		std::vector<STPDiversity::STPTextureDatabase::STPViewGroupDescription> DeclaredViewGroup;
		//For each texture name, maps to an index to the view group data structure.
		std::unordered_map<std::string_view, size_t> DeclaredTexture;
		std::vector<std::tuple<STPDiversity::Sample, float, std::string_view>> Altitude;
		std::vector<std::tuple<STPDiversity::Sample, float, float, float, float, std::string_view>> Gradient;

		/**
		 * @brief Check if the parsing texture variable has been declared before it is used.
		 * If texture is not declared, exception will be thrown
		 * @param texture The texture variable being tesed.
		*/
		void checkTextureDeclared(const std::string_view&) const;

		/**
		 * @brief Process identifier texture.
		*/
		void processTexture();

		/**
		 * @brief Process identifier rule.
		*/
		void processRule();

		/**
		 * @brief Process identifier group.
		*/
		void processGroup();

	public:
		
		//A table of texture variable, corresponds to texture ID and the belonging view group ID
		typedef std::unordered_map<std::string_view, std::pair<STPDiversity::STPTextureInformation::STPTextureID, STPDiversity::STPTextureInformation::STPViewGroupID>> STPTextureVariable;

		/**
		 * @brief Construct a TDL parser with an input.
		 * @param source The pointer to the source code of TDL.
		 * No reference is retained after this function returns, source is copied by the compiler.
		*/
		STPTextureDefinitionLanguage(const std::string&);

		STPTextureDefinitionLanguage(const STPTextureDefinitionLanguage&) = delete;

		STPTextureDefinitionLanguage(STPTextureDefinitionLanguage&&) = delete;

		STPTextureDefinitionLanguage& operator=(const STPTextureDefinitionLanguage&) = delete;

		STPTextureDefinitionLanguage& operator=(STPTextureDefinitionLanguage&&) = delete;

		~STPTextureDefinitionLanguage();

		/**
		 * @brief Parse all defined texture rules into a texture database.
		 * Note that if any texture rule has been defined in the database exception will be thrown since no duplicated rules can exist.
		 * @param database The pointer to the database for which rules will be added.
		 * @return A table of variable name to texture ID within this database, it can be used to uploaded texture data and assign texture group.
		*/
		STPTextureVariable operator()(STPDiversity::STPTextureDatabase&) const;

	};

}
#endif//_STP_TEXTURE_DEFINITION_LANGUAGE_H_