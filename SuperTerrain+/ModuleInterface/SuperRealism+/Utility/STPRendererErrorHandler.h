#pragma once
#ifndef _STP_RENDERER_ERROR_HANDLER_H_
#define _STP_RENDERER_ERROR_HANDLER_H_

#include <SuperRealism+/STPRealismDefine.h>

#define STP_ENGINE_ASSERT_QUAL STP_REALISM_API
#include <SuperTerrain+/Utility/STPDeviceErrorHandlerBlueprint.hpp>

#define STPoptixCheckErr(EC) STP_ASSERT_ENGINE_BASIC(EC)

#endif//_STP_RENDERER_ERROR_HANDLER_H_