target_sources(${STP_REALISM_ENGINE} PRIVATE
STPRealismDefine.h
STPScenePipeline.h
# /Environment
Environment/STPAtomsphereSetting.h
Environment/STPSunSetting.h
# /Object
Object/STPBuffer.h
Object/STPPipelineManager.h
Object/STPProgramManager.h
Object/STPProgramManager.inl
Object/STPShaderManager.h
Object/STPVertexArray.h
# /Renderer
Renderer/STPHeightfieldTerrain.h
Renderer/STPSun.h
# /Utility/Camera
Utility/Camera/STPCamera.h
Utility/Camera/STPPerspectiveCamera.h
# /Utility
Utility/STPDebugCallback.h
Utility/STPIndirectCommand.hpp
Utility/STPLogStorage.hpp
Utility/STPShaderIncludeManager.h
)

# config file for the realism rendering engine
# output the path to the GLSL shader code
set(SuperRealism+ShaderPath ${SuperRealism+Include}/Shader)
setupConfigTemplate(${STP_REALISM_ENGINE} STPRealismInfo.h.in Include/STPRealismInfo.h STPRealismInfo.h)