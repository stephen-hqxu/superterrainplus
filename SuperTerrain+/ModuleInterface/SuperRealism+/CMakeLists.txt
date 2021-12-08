target_sources(${STP_REALISM_ENGINE} PRIVATE
STPRealismDefine.h
STPSun.h
# /Environment
Environment/STPAtomsphereSetting.h
Environment/STPSunSetting.h
# /Object
Object/STPPipelineManager.h
Object/STPProgramManager.h
Object/STPProgramManager.inl
Object/STPShaderManager.h
)

# config file for the realism rendering engine
# output the path to the GLSL shader code
set(SuperRealism+ShaderPath ${SuperRealism+Include}/Shader)
setupConfigTemplate(${STP_REALISM_ENGINE} STPRealismInfo.h.in Include/STPRealismInfo.h STPRealismInfo.h)