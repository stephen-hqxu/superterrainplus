#include <SuperRealism+/Scene/Component/STPAlphaCulling.h>
#include <SuperRealism+/STPRealismInfo.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

//GLAD
#include <glad/glad.h>

using SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto AlphaCullingFilename = STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPAlphaCulling", ".frag");

STPAlphaCulling::STPAlphaCulling(STPCullOperator op, const STPScreenInitialiser& screen_init) : STPScreen(*screen_init.SharedVertexBuffer) {
	//setup alpha culling shading program
	const char* const culling_shader_file = AlphaCullingFilename.data();
	STPShaderManager::STPShaderSource cull_source(culling_shader_file, *STPFile(culling_shader_file));
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	//determine operator string
	const char* op_str;
	switch (op) {
	case STPCullOperator::Equal:
		op_str = "=";
		break;
	case STPCullOperator::NotEqual:
		op_str = "!=";
		break;
	case STPCullOperator::Greater:
		op_str = ">";
		break;
	case STPCullOperator::GreaterEqual:
		op_str = ">=";
		break;
	case STPCullOperator::Less:
		op_str = "<";
		break;
	case STPCullOperator::LessEqual:
		op_str = "<=";
		break;
	default:
		//impossible
		break;
	}
	Macro("ALPHA_TEST_OPERATOR", op_str);
	cull_source.define(Macro);

	//build the program
	STPShaderManager cull_shader(GL_FRAGMENT_SHADER);
	cull_shader(cull_source);
	this->initScreenRenderer(cull_shader, screen_init);

	//sampler
	this->OffScreenRenderer.uniform(glProgramUniform1i, "ColorInput", 0);
}

void STPAlphaCulling::setAlphaLimit(float limit) {
	this->OffScreenRenderer.uniform(glProgramUniform1f, "AlphaThreshold", limit);
}

void STPAlphaCulling::cull(const STPTexture& input) const {
	//prepare for rendering
	this->ScreenVertex->bind();
	this->OffScreenRenderer.use();

	input.bind(0);

	//draw
	this->drawScreen();

	//finish up
	STPProgramManager::unuse();
}