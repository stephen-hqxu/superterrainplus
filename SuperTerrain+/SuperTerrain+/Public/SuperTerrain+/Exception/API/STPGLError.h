#pragma once
#ifndef _STP_GL_ERROR_H_
#define _STP_GL_ERROR_H_

#include "../STPFundamentalException.h"

//create an GL exception with encoded source information
#define STP_GL_ERROR_CREATE(DESC) STP_STANDARD_EXCEPTION_CREATE(STPGLError, DESC)

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPGLError signals an error during execution of GL API.
	*/
	class STP_API STPGLError : public STPFundamentalException::STPBasic {
	public:

		//same function arguments as the base class
		STPGLError(const std::string&, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPGLError() = default;

	};

}
#endif//_STP_GL_ERROR_H_