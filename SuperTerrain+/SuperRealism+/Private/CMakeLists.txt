target_sources(${STP_REALISM_ENGINE} PRIVATE
STPExtendedScenePipeline.cpp
STPRealismEnvironment.cpp
STPRendererInitialiser.cpp
STPScenePipeline.cpp
# /Geometry
Geometry/STPPlaneGeometry.cpp
# /Object
Object/STPBuffer.cpp
Object/STPFrameBuffer.cpp
Object/STPPipelineManager.cpp
Object/STPProgramManager.cpp
Object/STPRenderBuffer.cpp
Object/STPSampler.cpp
Object/STPShaderManager.cpp
Object/STPTexture.cpp
Object/STPVertexArray.cpp
# /Scene/Component
Scene/Component/STPAlphaCulling.cpp
Scene/Component/STPAmbientOcclusion.cpp
Scene/Component/STPAurora.cpp
Scene/Component/STPBidirectionalScattering.cpp
Scene/Component/STPGaussianFilter.cpp
Scene/Component/STPHeightfieldTerrain.cpp
Scene/Component/STPPostProcess.cpp
Scene/Component/STPScreen.cpp
Scene/Component/STPSkybox.cpp
Scene/Component/STPStarfield.cpp
Scene/Component/STPSun.cpp
Scene/Component/STPWater.cpp
# /Scene/Light
Scene/Light/STPCascadedShadowMap.cpp
Scene/Light/STPLightShadow.cpp
Scene/Light/STPLightSpectrum.cpp
# /Scene
Scene/STPMaterialLibrary.cpp
Scene/STPSceneLight.cpp
# /Utility
Utility/STPAsyncAccelBuilder.cpp
Utility/STPCamera.cpp
Utility/STPDebugCallback.cpp

Utility/STPRandomTextureGenerator.cu
)