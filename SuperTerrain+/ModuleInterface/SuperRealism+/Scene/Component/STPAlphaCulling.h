#pragma once
#ifndef _STP_ALPHA_CULLING_H_
#define _STP_ALPHA_CULLING_H_

#include <SuperRealism+/STPRealismDefine.h>
#include "STPScreen.h"
//GL Object
#include "../../Object/STPTexture.h"
#include "../../Object/STPShaderManager.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPAlphaCulling is a simple utility shader that performs pixel discard operations 
	 * when an alpha equation returns true.
	 * The alpha test does not write any colour to the output if the equation fails,
	 * it is mainly used to modify stencil buffer based on alpha value.
	*/
	class STP_REALISM_API STPAlphaCulling : private STPScreen {
	public:

		/**
		 * @brief STPCullComparator specifies the comparison operator on the alpha equation.
		*/
		enum class STPCullComparator : unsigned char {
			Equal = 0x00u,
			NotEqual = 0x01u,
			Greater = 0x10u,
			GreaterEqual = 0x11u,
			Less = 0x12u,
			LessEqual = 0x13u
		};

		/**
		 * @brief STPCullConnector specifies the logical operator to connect multiple sub-expressions.
		*/
		enum class STPCullConnector : unsigned char {
			And = 0x00u,
			Or = 0xFFu
		};

	private:

		/**
		 * @brief Get the string representing a culling comparator.
		 * @param comp The comparator.
		 * @return A pointer to a compile-time string literal.
		*/
		static const char* comparatorString(STPCullComparator);

		/**
		 * @brief Get the string representing a culling connector.
		 * @param conn The connector.
		 * @return A pointer to a compile-time string literal.
		*/
		static const char* connectorString(STPCullConnector);

		/**
		 * @brief Compile the alpha shader and link into a complete shader program. Then setup basic uniforms.
		 * @param macro The pointer to the predefined macros.
		 * @param screen_init The screen initialiser.
		*/
		void prepareAlphaShader(const STPShaderManager::STPShaderSource::STPMacroValueDictionary&, const STPScreenInitialiser&);

	public:

		/**
		 * @brief Initialise a new alpha culler with a single culling expression.
		 * The alpha equation is defined as: InputAlpha `comp` `limit`.
		 * @param comp The comparator used in the culling equation.
		 * @param limit The new limit to the alpha equation.
		 * @param screen_init The screen initialiser for off-screen rendering.
		*/
		STPAlphaCulling(STPCullComparator, float, const STPScreenInitialiser&);

		/**
		 * @brief Initialise a new alpha culler with dual culling expressions.
		 * The alpha equation is defined as: (InputAlpha `comp1` `limit1`) `conn` (InputAlpha `comp2` `limit2`).
		 * @param comp1 The first comparator.
		 * @param limit1 The limit value for the first sub-expression.
		 * @param conn The logic connectivity operator.
		 * @param comp2 The second comparator.
		 * @param limit2 The limit value for the second sub-expression.
		 * @param screen_init The screen initialiser for off-screen rendering.
		*/
		STPAlphaCulling(STPCullComparator, float, STPCullConnector, STPCullComparator, float, const STPScreenInitialiser&);

		STPAlphaCulling(const STPAlphaCulling&) = delete;

		STPAlphaCulling(STPAlphaCulling&&) = delete;

		STPAlphaCulling& operator=(const STPAlphaCulling&) = delete;

		STPAlphaCulling& operator=(STPAlphaCulling&&) = delete;

		~STPAlphaCulling() = default;

		/**
		 * @brief Perform alpha culling operation.
		 * No colour will be written by the program.
		 * @param input The input colour texture to be used for culling.
		*/
		void cull(const STPTexture&) const;

	};

}
#endif//_STP_ALPHA_CULLING_H_