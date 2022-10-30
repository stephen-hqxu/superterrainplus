#include <SuperRealism+/Scene/Component/STPScreen.h>
#include <SuperRealism+/STPRealismInfo.h>
//Command
#include <SuperRealism+/Utility/STPIndirectCommand.hpp>

//Error
#include <SuperTerrain+/Exception/STPInvalidArgument.h>
#include <SuperTerrain+/Exception/STPBadNumericRange.h>
#include <SuperTerrain+/Exception/STPGLError.h>

//IO
#include <SuperTerrain+/Utility/STPFile.h>
#include <SuperTerrain+/Utility/STPStringUtility.h>

//GLAD
#include <glad/glad.h>

//System
#include <array>

using std::array;
using std::shared_ptr;

using glm::uvec2;
using glm::vec4;

using namespace SuperTerrainPlus::STPRealism;

constexpr static auto ScreenShaderFilename =
	SuperTerrainPlus::STPStringUtility::generateFilename(STPRealismInfo::ShaderPath, "/STPScreen", ".vert");

constexpr static array<signed char, 16ull> QuadVertex = {
	//Position		//TexCoord
	-1, +1,			0, 1,
	-1, -1,			0, 0,
	+1, -1,			1, 0,
	+1, +1,			1, 1
};
constexpr static STPIndirectCommand::STPDrawArray QuadDrawCommand = {
	4u,
	1u,
	0u,
	0u
};

STPScreen::STPSimpleScreenFrameBuffer::STPSimpleScreenFrameBuffer() : ScreenColor(GL_TEXTURE_2D) {

}

STPTexture STPScreen::STPSimpleScreenFrameBuffer::updateScreenFrameBuffer(
	STPTexture* stencil, const uvec2& dimension, STPOpenGL::STPenum internal) {
	if (dimension.x == 0u || dimension.y == 0u) {
		throw STPException::STPBadNumericRange("Both component of a screen buffer dimension must be positive");
	}
	//create new texture
	STPTexture new_screen_color(GL_TEXTURE_2D);
	//allocate memory
	new_screen_color.textureStorage2D(1, internal, dimension);

	//attach new texture to framebuffer
	this->ScreenColorContainer.attach(GL_COLOR_ATTACHMENT0, new_screen_color, 0);
	if (stencil) {
		this->ScreenColorContainer.attach(GL_STENCIL_ATTACHMENT, *stencil, 0);
	} else {
		//detach stencil buffer if none is provided because the framebuffer needs to be resized
		this->ScreenColorContainer.detachTexture(GL_STENCIL_ATTACHMENT);
	}
	//depth buffer is not needed because we are doing off-screen rendering
	if (this->ScreenColorContainer.status(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw STPException::STPGLError("Screen framebuffer cannot be validated");
	}

	return new_screen_color;
}

void STPScreen::STPSimpleScreenFrameBuffer::setScreenBuffer(STPTexture* stencil, const uvec2& dimension, STPOpenGL::STPenum internal) {
	//store the new texture
	this->ScreenColor = std::move(this->updateScreenFrameBuffer(stencil, dimension, internal));
}

void STPScreen::STPSimpleScreenFrameBuffer::clearScreenBuffer(const vec4& color) {
	this->ScreenColorContainer.clearColor(0, color);
}

void STPScreen::STPSimpleScreenFrameBuffer::capture() const {
	this->ScreenColorContainer.bind(GL_FRAMEBUFFER);
}

STPScreen::STPSimpleScreenBindlessFrameBuffer::STPSimpleScreenBindlessFrameBuffer() : STPSimpleScreenFrameBuffer() {

}

void STPScreen::STPSimpleScreenBindlessFrameBuffer::setScreenBuffer(
	STPTexture* stencil, const uvec2& dimension, STPOpenGL::STPenum internal) {
	using std::move;
	STPTexture new_screen_color = this->updateScreenFrameBuffer(stencil, dimension, internal);
	//update bindless handle
	this->ScreenColorHandle = move(STPBindlessTexture(new_screen_color, this->ScreenColorSampler));
	this->ScreenColor = move(new_screen_color);
}

STPScreen::STPScreenVertexShader::STPScreenVertexShader() : ScreenVertexShader(GL_VERTEX_SHADER) {
	//read source
	const char* const screen_source_file = ScreenShaderFilename.data();
	STPShaderManager::STPShaderSource shader_source(screen_source_file, STPFile::read(screen_source_file));
	//compile
	this->ScreenVertexShader(shader_source);
}

const STPShaderManager& STPScreen::STPScreenVertexShader::operator*() const {
	return this->ScreenVertexShader;
}

STPScreen::STPScreenVertexBuffer::STPScreenVertexBuffer() {
	//send of off screen quad
	this->ScreenBuffer.bufferStorageSubData(QuadVertex.data(), QuadVertex.size() * sizeof(signed char), GL_NONE);
	//rendering command
	this->ScreenRenderCommand.bufferStorageSubData(&QuadDrawCommand, sizeof(QuadDrawCommand), GL_NONE);

	//vertex array
	STPVertexArray::STPVertexAttributeBuilder attr = this->ScreenArray.attribute();
	attr.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.format(2, GL_BYTE, GL_FALSE, sizeof(signed char))
		.vertexBuffer(this->ScreenBuffer, 0)
		.binding();
	this->ScreenArray.enable(0u, 2u);
}

void STPScreen::STPScreenVertexBuffer::bind() const {
	//bind vertex buffer
	this->ScreenArray.bind();
	this->ScreenRenderCommand.bind(GL_DRAW_INDIRECT_BUFFER);
}

STPScreen::STPScreenProgramExecutor::STPScreenProgramExecutor(const STPScreen& screen) {
	screen.ScreenVertex->bind();
	screen.OffScreenRenderer.use();
}

STPScreen::STPScreenProgramExecutor::~STPScreenProgramExecutor() {
	STPProgramManager::unuse();
}

void STPScreen::STPScreenProgramExecutor::operator()() const {
	glDrawArraysIndirect(GL_TRIANGLE_FAN, nullptr);
}

void STPScreen::initScreenRenderer(const STPShaderManager& screen_fs, const STPScreenInitialiser& screen_init) {
	if (screen_fs.Type != GL_FRAGMENT_SHADER) {
		throw STPException::STPInvalidArgument("The shader initialised for off-screen rendering must be a fragment shader");
	}
	const auto& [screen_vs, screen_buf] = screen_init;

	//initialise screen vertex buffer
	this->ScreenVertex = shared_ptr(screen_buf);

	//setup screen program
	this->OffScreenRenderer
		.attach(**screen_vs)
		.attach(screen_fs)
		.finalise();
}

void STPScreen::drawScreen() const {
	const STPScreenProgramExecutor executor = this->drawScreenFromExecutor();
	executor();
}

STPScreen::STPScreenProgramExecutor STPScreen::drawScreenFromExecutor() const {
	return STPScreenProgramExecutor(*this);
}