#pragma once
#ifndef _STP_COMMAND_LINE_OPTION_H_
#define _STP_COMMAND_LINE_OPTION_H_

//Data
#include <tuple>
#include <optional>

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
			//To make the application run at full-screen mode.
			//The rendering resolution will be on native resolution.
			//The value will be the index to the monitor to use, or no value if don't use full screen mode.
			std::optional<unsigned int> UseFullScreen;
			//The rendering FPS limit.
			//Not specifying this option, or an option with non-positive value will make the program to use the default FPS.
			std::optional<double> FrameRate;

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