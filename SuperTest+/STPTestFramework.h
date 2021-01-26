#pragma once
#ifndef _STP_TEST_FRAMEWORK_H_
#define _STP_TEST_FRAMEWORK_H_

//Add your precompiled headers here
#define CATCH_CONFIG_DEFAULT_REPORTER "stp"
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_CONSOLE_WIDTH 100
#include "catch2/catch.hpp"
#include "STPTestReporter.hpp"

#endif//_STP_TEST_FRAMEWORK_H_