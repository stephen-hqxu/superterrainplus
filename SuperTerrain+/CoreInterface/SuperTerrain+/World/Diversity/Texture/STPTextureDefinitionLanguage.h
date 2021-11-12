#ifndef _STP_TEXTURE_DEFINITION_LANGUAGE_H_
#define _STP_TEXTURE_DEFINITION_LANGUAGE_H_

#include <SuperTerrain+/STPCoreDefine.h>
//Texture Database
#include "STPTextureDatabase.h"

//Memory
#include <memory>
//System
#include <string>
#include <string_view>
//Container
#include <vector>
#include <unordered_map>
#include <tuple>

namespace SuperTerrainPlus::STPDiversity {

	/**
	 * @brief STPTextureDefinitionLanguage is a lexer and parser for SuperTerrain+ Texture Definition Language - TDL.
	 * For more information please refer to the specification of the language.
	*/
	class STP_API STPTextureDefinitionLanguage {
	private:

		/**
		 * @brief STPTDLLexer is a lexical analysis tool and token generator for SuperTerrain+ Texture Definition Language.
		*/
		class STPTDLLexer;

		std::unique_ptr<STPTDLLexer> Lexer;

		//Information from the source code after lexing
		std::vector<std::string_view> DeclaredTextureVariable;
		std::vector<std::tuple<Sample, float, std::string_view>> Altitude;
		std::vector<std::tuple<Sample, float, float, float, float, std::string_view>> Gradient;

	public:

		//A table of texture variable, corresponds to texture ID
		typedef std::unordered_map<std::string_view, STPTextureInformation::STPTextureID> STPTextureVariable;

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

		STPTextureVariable operator()(const STPTextureDatabase&) const;

	};

}
#endif//_STP_TEXTURE_DEFINITION_LANGUAGE_H_