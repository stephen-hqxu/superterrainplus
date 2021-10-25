#pragma once

namespace STPDemo {

	/**
	 * @brief STPRendererCommander is used to generate rendering commands for all renderers in multithreads, and can be reused over time
	*/
	struct STPRendererCommander {
	public:

		//Commands
		DrawElementsIndirectCommand* Command_SkyRenderer = nullptr;
		DrawArraysIndirectCommand* Command_Quad = nullptr;
		DrawElementsIndirectCommand* Command_Procedural2DINF = nullptr;

		/**
		 * @brief Init the STPRendererCommander, and getting all drawing commands ready
		 * @param terrain2d_unitplane_count specify the number of unit plane for the procedural 2d infinite terrain renderer.
		 * The value is equavalent to CHUNK_SIZE.x * CHUNK_SIZE.y * RENDERED_CHUNK.x * RENDERED_CHUNK.y;
		*/
		STPRendererCommander(unsigned int terrain2d_unitplane_count) {
			//Initialise rendering command
			this->Command_SkyRenderer = new DrawElementsIndirectCommand{
					SglToolkit::SgTUtil::UNITBOX_INDICES_SIZE,
					1,
					0,
					0,
					0
			};
			this->Command_Quad = new DrawArraysIndirectCommand{
					6,
					1,
					0,
					0
			};
			this->Command_Procedural2DINF = new DrawElementsIndirectCommand{
					SglToolkit::SgTUtil::UNITPLANE_INDICES_SIZE,
					terrain2d_unitplane_count,
					0,
					0,
					0
			};
		}

		~STPRendererCommander() {
			//delete all commands
			delete this->Command_SkyRenderer;
			delete this->Command_Quad;
			delete this->Command_Procedural2DINF;
		}

	};
}

