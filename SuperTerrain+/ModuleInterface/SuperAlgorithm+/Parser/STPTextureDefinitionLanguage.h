#pragma once
#ifndef _STP_TEXTURE_DEFINITION_LANGUAGE_H_
#define _STP_TEXTURE_DEFINITION_LANGUAGE_H_

#include <SuperAlgorithm+/STPAlgorithmDefine.h>
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
	 * @brief STPTextureDefinitionLanguage is a lexer and parser for SuperTerrain+ Texture Definition Language - TDL.
	 * For more information please refer to the specification of the language.
	*/
	class STP_ALGORITHM_HOST_API STPTextureDefinitionLanguage {
	private:

		/**
		 * @brief STPTDLLexer is a lexical analysis tool and token generator for SuperTerrain+ Texture Definition Language.
		*/
		class STPTDLLexer;

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
		 * If texture is not declared, exception will be thrown.
		 * @param lexer The pointer to the lexer instance.
		 * @param texture The texture variable being tested.
		*/
		void checkTextureDeclared(const STPTDLLexer&, const std::string_view&) const;

		/**
		 * @brief Process identifier texture.
		 * @param lexer The pointer to the lexer instance.
		*/
		void processTexture(STPTDLLexer&);

		/**
		 * @brief Process identifier rule.
		 * @param lexer The pointer to the lexer instance.
		*/
		void processRule(STPTDLLexer&);

		/**
		 * @brief Process identifier group.
		 * @param lexer The pointer to the lexer instance.
		*/
		void processGroup(STPTDLLexer&);

	public:
		
		//A table of texture variable, corresponds to texture ID and the belonging view group ID
		typedef std::unordered_map<std::string_view, std::pair<STPDiversity::STPTextureInformation::STPTextureID, STPDiversity::STPTextureInformation::STPViewGroupID>> STPTextureVariable;

		/**
		 * @brief Construct a TDL parser with an input.
		 * @param source The pointer to the view of the source code of TDL. The source captured by the string view should be null-terminated.
		 * No reference is retained after this function returns, source is copied by the compiler.
		 * However, the TDL parser does not own the string, the memory of the original string should be managed by the user, 
		 * until the current instance and all returned view of string from the current instance to the user are destroyed.
		*/
		STPTextureDefinitionLanguage(const std::string_view&);

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
		 * The string key in the table is a view to the raw TDL string view provided by the user upon initialisation of this TDL parser instance, and is valid
		 * only when the raw string which the view references from is alive.
		*/
		STPTextureVariable operator()(STPDiversity::STPTextureDatabase&) const;

	};

}
#endif//_STP_TEXTURE_DEFINITION_LANGUAGE_H_