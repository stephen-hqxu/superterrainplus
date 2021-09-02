#pragma once

/**
 * @brief STPDemo is a sample implementation of super terrain + application, it's not part of the super terrain + api library.
 * Every thing in the STPDemo namespace is modifiable and re-implementable by developers.
*/
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
		 * @param pool The thread pool for multi-threaded rendering command
		 * @param terrain2d_unitplane_count specify the number of unit plane for the procedural 2d infinite terrain renderer.
		 * The value is equavalent to CHUNK_SIZE.x * CHUNK_SIZE.y * RENDERED_CHUNK.x * RENDERED_CHUNK.y;
		*/
		STPRendererCommander(SuperTerrainPlus::STPThreadPool& pool, unsigned int terrain2d_unitplane_count) {
			//Initialise rendering command in multi-thread
			//Sky renderer
			std::future<DrawElementsIndirectCommand*> skycmd_generator = pool.enqueue_future([]() -> DrawElementsIndirectCommand* {
				DrawElementsIndirectCommand* skycmd = new DrawElementsIndirectCommand{
					SglToolkit::SgTUtils::UNITBOX_INDICES_SIZE,
					1,
					0,
					0,
					0
				};
				return skycmd;
				});
			//Quad renderer
			std::future<DrawArraysIndirectCommand*> quadcmd_generator = pool.enqueue_future([]() -> DrawArraysIndirectCommand* {
				DrawArraysIndirectCommand* quadcmd = new DrawArraysIndirectCommand{
					6,
					1,
					0,
					0
				};
				return quadcmd;
				});
			//Procedural 2D infinite terrain renderer
			std::future<DrawElementsIndirectCommand*> procedural2dinf_generator = pool.enqueue_future([terrain2d_unitplane_count]() -> DrawElementsIndirectCommand* {
				DrawElementsIndirectCommand* procedural2dinfcmd = new DrawElementsIndirectCommand{
					SglToolkit::SgTUtils::UNITPLANE_INDICES_SIZE,
					terrain2d_unitplane_count,
					0,
					0,
					0
				};
				return  procedural2dinfcmd;
				});

			//retrieve all results
			this->Command_SkyRenderer = skycmd_generator.get();
			this->Command_Quad = quadcmd_generator.get();
			this->Command_Procedural2DINF = procedural2dinf_generator.get();
		}

		~STPRendererCommander() {
			//delete all commands
			delete this->Command_SkyRenderer;
			delete this->Command_Quad;
			delete this->Command_Procedural2DINF;
		}

	};
}

