#include <SuperRealism+/STPRendererInitialiser.h>

//Shader Manager
#include <SuperRealism+/Object/STPShaderManager.h>

using namespace SuperTerrainPlus::STPRealism;

void STPRendererInitialiser::init() {
	STPShaderManager::initialise();
}