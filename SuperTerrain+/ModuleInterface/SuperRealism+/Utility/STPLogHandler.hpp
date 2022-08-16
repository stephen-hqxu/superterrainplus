#pragma once
#ifndef _STP_LOG_HANDLER_HPP_
#define _STP_LOG_HANDLER_HPP_

#include <SuperRealism+/STPRealismDefine.h>
#include <string>

namespace SuperTerrainPlus::STPRealism {

	/**
	 * @brief STPLogHandler handles logs from shader compilers.
	*/
	namespace STPLogHandler {

		/**
		 * @brief STPLogHandlerSolution decides how a log should be handled.
		*/
		class STPLogHandlerSolution {
		public:

			/**
			 * @brief Handle an incoming log.
			 * @param log A pointer to a constant log string instance.
			 * Note that the pointer is not guaranteed to be available after the function return,
			 * retaining the pointer will lead to UB. In that case, user should copy the string.
			*/
			virtual void handle(const std::string&) { }

		};

		//The log handler default by the system.
		STP_REALISM_API extern STPLogHandlerSolution DefaultLogHandler;
		//Set the active log handler.
		//If user is intended to use a custom log handler, the lifetime of the handle must persist until the end of the application
		//or the new handle is emplaced.
		//The pointer should always be a valid pointer.
		STP_REALISM_API extern STPLogHandlerSolution* ActiveLogHandler;

		/* Note: don't declare them as inline to ensure unique address and instance across dynamic library on Windows platform;
		their definitions reside in STPShaderManager.cpp. */

	}

}
#endif//_STP_LOG_HANDLER_HPP_