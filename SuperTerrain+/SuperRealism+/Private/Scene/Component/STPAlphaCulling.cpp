#include <SuperRealism+/Scene/Component/STPAlphaCulling.h>
#include <SuperRealism+/STPRealismInfo.h>

#include <SuperTerrain+/Exception/STPInvalidEnum.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

//GLAD
#include <glad/glad.h>

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto AlphaCullingFilename =
	SuperTerrainPlus::STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPAlphaCulling", ".frag");

STPAlphaCulling::STPAlphaCulling(const STPCullComparator comp, const float limit, const STPScreen::STPScreenInitialiser& screen_init) {
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;
	
	Macro("ALPHA_COMPARATOR", STPAlphaCulling::comparatorString(comp));
	
	this->prepareAlphaShader(Macro, screen_init);

	this->CullingQuad.OffScreenRenderer.uniform(glProgramUniform1f, "Lim", limit);
}

STPAlphaCulling::STPAlphaCulling(const STPCullComparator comp1, const float limit1, 
	const STPCullConnector conn, const STPCullComparator comp2, const float limit2, const STPScreen::STPScreenInitialiser& screen_init) {
	STPShaderManager::STPShaderSource::STPMacroValueDictionary Macro;

	Macro("USE_DUAL_EXPRESSIONS", 1)
		("ALPHA_COMPARATOR_A", STPAlphaCulling::comparatorString(comp1))
		("ALPHA_CONNECTOR", STPAlphaCulling::connectorString(conn))
		("ALPHA_COMPARATOR_B", STPAlphaCulling::comparatorString(comp2));

	this->prepareAlphaShader(Macro, screen_init);

	this->CullingQuad.OffScreenRenderer.uniform(glProgramUniform1f, "LimA", limit1)
		.uniform(glProgramUniform1f, "LimB", limit2);
}

inline const char* STPAlphaCulling::comparatorString(const STPCullComparator comp) {
	switch (comp) {
	case STPCullComparator::Equal: return "=";
	case STPCullComparator::NotEqual: return "!=";
	case STPCullComparator::Greater: return ">";
	case STPCullComparator::GreaterEqual: return ">=";
	case STPCullComparator::Less: return "<";
	case STPCullComparator::LessEqual: return "<=";
	default: throw STP_INVALID_ENUM_CREATE(comp, STPCullComparator);
	}
}

inline const char* STPAlphaCulling::connectorString(const STPCullConnector conn) {
	switch (conn) {
	case STPCullConnector::And: return "&&";
	case STPCullConnector::Or: return "||";
	default: throw STP_INVALID_ENUM_CREATE(conn, STPCullConnector);
	}
}

inline void STPAlphaCulling::prepareAlphaShader(const STPShaderManager::STPShaderSource::STPMacroValueDictionary& macro,
	const STPScreen::STPScreenInitialiser& screen_init) {
	//setup alpha culling shading program
	const char* const culling_shader_file = AlphaCullingFilename.data();
	STPShaderManager::STPShaderSource cull_source(culling_shader_file, STPFile::read(culling_shader_file));
	cull_source.define(macro);

	//build the program
	const STPShaderManager::STPShader cull_shader = STPShaderManager::make(GL_FRAGMENT_SHADER, cull_source);
	this->CullingQuad.initScreenRenderer(cull_shader, screen_init);

	//sampler
	this->CullingQuad.OffScreenRenderer.uniform(glProgramUniform1i, "ColorInput", 0);
}

void STPAlphaCulling::cull(const STPTexture& input) const {
	input.bind(0);
	this->CullingQuad.drawScreen();
}