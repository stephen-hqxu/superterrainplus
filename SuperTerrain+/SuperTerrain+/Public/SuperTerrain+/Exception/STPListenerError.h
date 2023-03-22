#pragma once
#ifndef _STP_LISTENER_ERROR_H_
#define _STP_LISTENER_ERROR_H_

#include "STPFundamentalException.h"

//create a listener-related error, with the pointer to the listener and the type of this error.y
#define STP_LISTENER_ERROR_CREATE(LIS_PTR, TYPE) STP_STANDARD_EXCEPTION_CREATE(STPListenerError, LIS_PTR, SuperTerrainPlus::STPException::STPListenerError::TYPE { })

namespace SuperTerrainPlus::STPException {

	/**
	 * @brief STPListenerError signals any error related to usage of subscriber and listener pattern.
	*/
	class STP_API STPListenerError : public STPFundamentalException::STPBasic {
	public:

		//Specifies which listener object causes the problem.
		const void* const GuiltyListener;

		//Unable to add this listener as this has previously added.
		struct STPRepeatedListener { explicit STPRepeatedListener() = default; };
		//Unable to remove this listener because it is not registered.
		struct STPListenerNotFound { explicit STPListenerNotFound() = default; };

		/**
		 * @param listener_ptr The pointer to the guilty listener object.
		 * @param error_type Specifies the type of error of the listener error.
		*/
		STPListenerError(const void*, STPRepeatedListener, STP_EXCEPTION_SOURCE_INFO_DECL);
		STPListenerError(const void*, STPListenerNotFound, STP_EXCEPTION_SOURCE_INFO_DECL);

		~STPListenerError() = default;

	};

}
#endif//_STP_LISTENER_ERROR_H_