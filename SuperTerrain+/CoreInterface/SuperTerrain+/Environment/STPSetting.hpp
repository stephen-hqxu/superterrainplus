#pragma once
#ifndef _STP_SETTING_HPP_
#define _STP_SETTING_HPP_

namespace SuperTerrainPlus::STPEnvironment {

	/**
	 * @brief A base class for each Super Terrain + settings
	*/
	struct STPSetting {
	protected:

		/**
		 * @brief Init settings
		*/
		STPSetting() = default;

		virtual ~STPSetting() = default;

	public:

		/**
		 * @brief Validate each setting values and check if all settings are legal
		 * @return True if all settings are legal.
		*/
		virtual bool validate() const = 0;
	};

}
#endif//_STP_SETTING_HPP_