#pragma once
#ifndef _STP_COMMAND_LINE_OPTION_H_
#define _STP_COMMAND_LINE_OPTION_H_

//Data
#include <tuple>

namespace STPDemo {

	/**
	 * @brief STPCommandLineOption processes command line arguments for the demo application.
	*/
	namespace STPCommandLineOption {

		/**
		 * @brief STPResult stores parsed command line argument.
		*/
		struct STPResult {
		public:

			//The speed-up in the sprint mode.
			double SprintSpeedMultiplier;

			//Specify the X and Y rendering resolution of the displayed window.
			//This value is ignored if run under full-screen.
			std::tuple<unsigned int, unsigned int> WindowResolution;
			//True to make the application run at full-screen mode.
			//The rendering resolution will be on native resolution.
			bool UseFullScreen;

		};

		/**
		 * @brief Read command line argument.
		 * @param argc Argument count.
		 * @param argv Argument value.
		 * @return Result of parsed arguments.
		*/
		STPResult read(int, const char* const*);

	}

}
#endif//_STP_COMMAND_LINE_OPTION_H_