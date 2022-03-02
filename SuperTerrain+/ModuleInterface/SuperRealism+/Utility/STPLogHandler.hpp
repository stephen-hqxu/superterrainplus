#pragma once
#ifndef _STP_LOG_HANDLER_HPP_
#define _STP_LOG_HANDLER_HPP_

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
			 * @param log A rvalue reference to a log string instance.
			*/
			virtual void handle(std::string&&) { }

		};

		//The log handler default by the system.
		inline static STPLogHandlerSolution DefaultLogHandler;
		//Set the active log handler.
		//If user is intended to use a custom log handler, the lifetime of the handle must persist until the end of the application
		//or the new handle is emplaced.
		//The pointer should always be a valid pointer.
		inline static STPLogHandlerSolution* ActiveLogHandler = &DefaultLogHandler;

	}

}
#endif//_STP_LOG_HANDLER_HPP_