#include <SuperRealism+/Scene/Component/STPScreen.h>
#include <SuperRealism+/STPRealismInfo.h>
//Command
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//Error
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>

//GLAD
#include <glad/glad.h>

//System
#include <array>

using std::array;

using glm::uvec2;
using glm::uvec3;
using glm::vec4;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto ScreenShaderFilename =
	SuperTerrainPlus::STPFile::generateFilename(SuperTerrainPlus::SuperRealismPlus_ShaderPath, "/STPScreen", ".vert");

constexpr static array<signed char, 16ull> QuadVertex = {
	//Position		//TexCoord
	-1, +1,			0, 1,
	-1, -1,			0, 0,
	+1, -1,			1, 0,
	+1, +1,			1, 1
};
constexpr static array<unsigned char, 6ull> QuadIndex = {
	0, 1, 2,
	0, 2, 3
};
constexpr static STPIndirectCommand::STPDrawElement QuadDrawCommand = {
	static_cast<unsigned int>(QuadIndex.size()),
	1u,
	0u,
	0u,
	0u
};

STPScreen::STPSimpleScreenFrameBuffer::STPSimpleScreenFrameBuffer() : ScreenColor(GL_TEXTURE_2D) {

}

void STPScreen::STPSimpleScreenFrameBuffer::setScreenBuffer(STPTexture* stencil, const uvec2& dimension, STPOpenGL::STPenum internal) {
	if (dimension.x == 0u || dimension.y == 0u) {
		throw STPException::STPBadNumericRange("Both component of a screen buffer dimension must be positive");
	}
	//create new texture
	STPTexture new_screen_color(GL_TEXTURE_2D);
	//allocate memory
	new_screen_color.textureStorage<STPTexture::STPDimension::TWO>(1, internal, uvec3(dimension, 1u));

	//attach new texture to framebuffer
	this->ScreenColorContainer.attach(GL_COLOR_ATTACHMENT0, new_screen_color, 0);
	if (stencil) {
		this->ScreenColorContainer.attach(GL_STENCIL_ATTACHMENT, *stencil, 0);
	}
	else {
		//detach stencil buffer if none is provided because the framebuffer needs to be resized
		this->ScreenColorContainer.detachTexture(GL_STENCIL_ATTACHMENT);
	}
	//depth buffer is not needed because we are doing off-screen rendering
	if (this->ScreenColorContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw STPException::STPGLError("Screen framebuffer cannot be validated");
	}

	//store the new texture
	using std::move;
	this->ScreenColor = move(new_screen_color);
}

void STPScreen::STPSimpleScreenFrameBuffer::clearScreenBuffer(const vec4& color) {
	this->ScreenColorContainer.clearColor(0, color);
}

void STPScreen::STPSimpleScreenFrameBuffer::capture() const {
	this->ScreenColorContainer.bind(GL_FRAMEBUFFER);
}

STPScreen::STPScreenVertexShader::STPScreenVertexShader() : ScreenVertexShader(GL_VERTEX_SHADER) {
	//read source
	const char* const screen_source_file = ScreenShaderFilename.data();
	STPShaderManager::STPShaderSource shader_source(screen_source_file, *STPFile(screen_source_file));
	//compile
	this->ScreenVertexShader(shader_source);
}

const STPShaderManager& STPScreen::STPScreenVertexShader::operator*() const {
	return this->ScreenVertexShader;
}

STPScreen::STPScreenVertexBuffer::STPScreenVertexBuffer() {
	//send of off screen quad
	this->ScreenBuffer.bufferStorageSubData(QuadVertex.data(), QuadVertex.size() * sizeof(signed char), GL_NONE);
	this->ScreenIndex.bufferStorageSubData(QuadIndex.data(), QuadIndex.size() * sizeof(unsigned char), GL_NONE);
	//rendering command
	this->ScreenRenderCommand.bufferStorageSubData(&QuadDrawCommand, sizeof(QuadDrawCommand), GL_NONE);

	//vertex array
	STPVertexArray::STPVertexAttributeBuilder attr = this->ScreenArray.attribute();
	attr.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.vertexBuffer(this->ScreenBuffer, 0)
		.elementBuffer(this->ScreenIndex)
		.binding();
	this->ScreenArray.enable(0u, 2u);
}

inline void STPScreen::STPScreenVertexBuffer::bind() const {
	//bind vertex buffer
	this->ScreenArray.bind();
	this->ScreenRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);
}

STPScreen::STPScreen(const STPSharableScreenVertexBuffer& screen_vb) : ScreenVertex(screen_vb) {
	
}

void STPScreen::initScreenRenderer(const STPShaderManager::STPShaderSource& screen_fs_source, const STPScreenInitialiser& screen_init) {
	const auto [vs, vb] = screen_init;

	STPShaderManager screen_fs(GL_FRAGMENT_SHADER);
	screen_fs(screen_fs_source);

	//setup compute program
	this->OffScreenRenderer
		.attach(**vs)
		.attach(screen_fs)
		.finalise();
}

void STPScreen::drawScreen() const {
	glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_BYTE, nullptr);
}