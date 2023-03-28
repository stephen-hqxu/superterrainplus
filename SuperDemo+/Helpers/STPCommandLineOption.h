#pragma once
#ifndef _STP_COMMAND_LINE_OPTION_H_
#define _STP_COMMAND_LINE_OPTION_H_

#include <SuperTerrain+/World/STPWorldMapPixelFormat.hpp>

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

			//The seed value for the whole generator.
			//If not specified, use the application's default.
			std::optional<SuperTerrainPlus::STPSeed_t> GeneratorSeed;

			//Set the starting time in a day at the beginning of the program.
			std::optional<unsigned int> DayStart;
			//Similarly, the start of year.
			std::optional<unsigned int> YearStart;

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