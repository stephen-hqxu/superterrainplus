#include <SuperRealism+/Scene/Component/STPAlphaCulling.h>
#include <SuperRealism+/STPRealismInfo.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

//GLAD
#include <glad/glad.h>

using SuperTerrainPlus::STPFile;
using namespace SuperTerrainPlus::STPRealism;

constexpr static auto AlphaCullingFilename = STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPAlphaCulling", ".frag");

STPAlphaCulling::STPAlphaCulling(STPCullComparator comp, float limit, const STPScreenInitialiser& screen_init) {
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;
	
	Macro("ALPHA_COMPARATOR", STPAlphaCulling::comparatorString(comp));
	
	this->prepareAlphaShader(Macro, screen_init);

	this->OffScreenRenderer.uniform(glProgramUniform1f, "Lim", limit);
}

STPAlphaCulling::STPAlphaCulling(STPCullComparator comp1, float limit1, 
	STPCullConnector conn, STPCullComparator comp2, float limit2, const STPScreenInitialiser& screen_init) {
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	Macro("USE_DUAL_EXPRESSIONS", 1)
		("ALPHA_COMPARATOR_A", STPAlphaCulling::comparatorString(comp1))
		("ALPHA_CONNECTOR", STPAlphaCulling::connectorString(conn))
		("ALPHA_COMPARATOR_B", STPAlphaCulling::comparatorString(comp2));

	this->prepareAlphaShader(Macro, screen_init);

	this->OffScreenRenderer.uniform(glProgramUniform1f, "LimA", limit1)
		.uniform(glProgramUniform1f, "LimB", limit2);
}

inline const char* STPAlphaCulling::comparatorString(STPCullComparator comp) {
	switch (comp) {
	case STPCullComparator::Equal: return "=";
	case STPCullComparator::NotEqual: return "!=";
	case STPCullComparator::Greater: return ">";
	case STPCullComparator::GreaterEqual: return ">=";
	case STPCullComparator::Less: return "<";
	case STPCullComparator::LessEqual: return "<=";
	default: return "impossible";
	}
}

inline const char* STPAlphaCulling::connectorString(STPCullConnector conn) {
	switch (conn) {
	case STPCullConnector::And: return "&&";
	case STPCullConnector::Or: return "||";
	default: return "impossible";
	}
}

inline void STPAlphaCulling::prepareAlphaShader(const STPShaderManager::STPShaderSource::STPMacroValueDictionary& macro, const STPScreenInitialiser& screen_init) {
	//setup alpha culling shading program
	const char* const culling_shader_file = AlphaCullingFilename.data();
	STPShaderManager::STPShaderSource cull_source(culling_shader_file, *STPFile(culling_shader_file));
	cull_source.define(macro);

	//build the program
	STPShaderManager cull_shader(GL_FRAGMENT_SHADER);
	cull_shader(cull_source);
	this->initScreenRenderer(cull_shader, screen_init);

	//sampler
	this->OffScreenRenderer.uniform(glProgramUniform1i, "ColorInput", 0);
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