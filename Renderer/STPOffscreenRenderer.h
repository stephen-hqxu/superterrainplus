#pragma once

/**
 * @brief Super Terrain + is an open source, procedural terrain engine running on OpenGL 4.6, which utilises most modern terrain rendering techniques
 * including perlin noise generated height map, hydrology processing and marching cube algorithm.
 * Super Terrain + uses GLFW library for display and GLAD for opengl contexting.
*/
namespace SuperTerrainPlus {

	/**
	 * @brief Render the scene in offscreen with framebuffer
	*/
	class STPOffscreenRenderer {
	private:

		//drawing command
		const void* const command;
		//buffers
		GLuint quad_vbo, quad_vao, quad_indirect;

		SglToolkit::SgTShaderProc quadShader;

		/**
		 * @brief Get uniform location of the quad renderer
		 * @param name Name of the uniform
		 * @return The uniform location
		*/
		GLint getLoc(const GLchar* const name) {
			return glGetUniformLocation(this->quadShader.getP(), name);
		}

		/**
		 * @brief Compile the shader for quad
		 * @return True if compiled
		*/
		const bool compileShader() {
			//log
			GLchar* log = new GLchar[1024];

			this->quadShader.addShader(GL_VERTEX_SHADER, "./GLSL/STPquad.vert");
			this->quadShader.addShader(GL_FRAGMENT_SHADER, "./GLSL/STPquad.frag");
			if (this->quadShader.linkShader(log, 1024) != SglToolkit::SgTShaderProc::OK) {
				cerr << "\n---------------STPOffscreenRender fails to load the shaders :(---------------------" << endl;
				cerr << log << endl;
				cerr << "-------------------------------------------------------------------------------------" << endl;
				//exit
				delete[] log;
				return false;
			}
			//linking textures
			glProgramUniform1i(this->quadShader.getP(), this->getLoc("screen"), 0);

			//clearup
			delete[] log;
			return true;
		}

		/**
		 * @brief Load the quad model
		*/
		void loadQuad() {
			//create
			glCreateBuffers(1, &this->quad_indirect);
			glCreateBuffers(1, &this->quad_vbo);
			glCreateVertexArrays(1, &this->quad_vao);

			//creating indirect buffer
			glNamedBufferStorage(this->quad_indirect, sizeof(DrawArraysIndirectCommand), this->command, GL_NONE);
			//vbo
			glNamedBufferStorage(this->quad_vbo, sizeof(SglToolkit::SgTUtils::FRAMEBUFFER_QUAD), SglToolkit::SgTUtils::FRAMEBUFFER_QUAD, GL_NONE);
			//vao
			glVertexArrayVertexBuffer(this->quad_vao, 0, this->quad_vbo, 0, 4 * sizeof(int));
			//attributing
			glEnableVertexArrayAttrib(this->quad_vao, 0);
			glEnableVertexArrayAttrib(this->quad_vao, 1);
			glVertexArrayAttribFormat(this->quad_vao, 0, 2, GL_INT, GL_FALSE, 0);
			glVertexArrayAttribFormat(this->quad_vao, 1, 2, GL_INT, GL_FALSE, 2 * sizeof(int));
			glVertexArrayAttribBinding(this->quad_vao, 0, 0);
			glVertexArrayAttribBinding(this->quad_vao, 1, 0);
		}


		/**
		 * @brief Delete buffers and clearup
		*/
		void clearup() {
			this->quadShader.deleteShader();
			glDeleteBuffers(1, &this->quad_indirect);
			glDeleteBuffers(1, &this->quad_vbo);
			glDeleteVertexArrays(1, &this->quad_vao);
		}

	public:

		/**
		 * @brief Init the offscreen renderer
		 * @param quad_cmd The indrect rendering command for quad renderer
		*/
		STPOffscreenRenderer(DrawArraysIndirectCommand* const quad_cmd) : command(reinterpret_cast<void*>(quad_cmd)) {
			cout << "....Loading STPQuadRenderer....";
			if (this->compileShader()) {
				cout << "Shader loaded :)" << endl;
			}

			this->loadQuad();
			cout << "....Done...." << endl;
		}

		~STPOffscreenRenderer() {
			this->clearup();
		}

		/**
		 * @brief Render the screen with screen texture
		 * @param screen_tex Screen texture
		*/
		void render(const GLuint& screen_tex) {
			glBindTextureUnit(0, screen_tex);
			glBindVertexArray(this->quad_vao);
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, this->quad_indirect);
			glUseProgram(this->quadShader.getP());

			glDrawArraysIndirect(GL_TRIANGLES, nullptr);
			glUseProgram(0);
		}

	};
}

