#pragma once
#ifndef _STP_ALPHA_CULLING_H_
#define _STP_ALPHA_CULLING_H_

#include <SuperRealism+/STPRealismDefine.h>
#include "STPScreen.h"
//GL Object
#include "../../Object/STPTexture.h"

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPAlphaCulling is a simple utility shader that performs pixel discard operations 
	 * when an alpha equation passes.
	 * The alpha equation is defined as: input.a OPERATOR user_limit.
	 * The alpha test does not write any color to the output if the equation fails,
	 * it is mainly used to modify stencil buffer based on alpha value.
	*/
	class STP_REALISM_API STPAlphaCulling : private STPScreen {
	public:

		/**
		 * @brief STPCullOperator specifies the operator on the alpha equation.
		*/
		enum class STPCullOperator : unsigned char {
			Equal = 0x00u,
			NotEqual = 0x01u,
			Greater = 0x10u,
			GreaterEqual = 0x11u,
			Less = 0x12u,
			LessEqual = 0x13u
		};

		/**
		 * @brief Initialise a new alpha culler.
		 * @param op The operator used in the culling equation.
		 * @param screen_init The screen initialiser for off-screen rendering.
		*/
		STPAlphaCulling(STPCullOperator, const STPScreenInitialiser&);

		STPAlphaCulling(const STPAlphaCulling&) = delete;

		STPAlphaCulling(STPAlphaCulling&&) = delete;

		STPAlphaCulling& operator=(const STPAlphaCulling&) = delete;

		STPAlphaCulling& operator=(STPAlphaCulling&&) = delete;

		~STPAlphaCulling() = default;

		/**
		 * @brief Set the user defined alpha limit for the alpha culling operation.
		 * @param limit The new limit to the alpha equation.
		*/
		void setAlphaLimit(float);

		/**
		 * @brief Perform alpha culling operation.
		 * No color will be written by the program.
		 * @param input The input color texture to be used for culling.
		*/
		void cull(const STPTexture&) const;

	};

}
#endif//_STP_ALPHA_CULLING_H_