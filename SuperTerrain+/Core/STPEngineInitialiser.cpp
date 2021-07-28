#pragma once
#include <STPEngineInitialiser.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus;

bool STPEngineInitialiser::Inited = false;

bool STPEngineInitialiser::initGLcurrent() {
	if (!gladLoadGL()) {
		return false;
	}
	STPEngineInitialiser::Inited = true;
	return true;
}

bool STPEngineInitialiser::initGLexplicit(STPglProc process) {
	if (!gladLoadGLLoader(process)) {
		return false;
	}
	STPEngineInitialiser::Inited = true;
	return true;
}

bool STPEngineInitialiser::hasInit() {
	return STPEngineInitialiser::Inited;
}