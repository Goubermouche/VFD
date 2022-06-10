#ifndef RENDERER_H_
#define RENDERER_H_

#include "RendererAPI.h"

namespace fe {
	/// <summary>
	/// Base renderer class. Enables us to interact with the current renderer API. 
	/// </summary>
	class Renderer
	{
	public:
		/// <summary>
		/// Initializes the renderer with the currently selected API.
		/// </summary>
		static void Init();

		/// <summary>
		/// Gets the currently used API.
		/// </summary>
		/// <returns>Currently used API.</returns>
		inline static RendererAPIType GetAPI() {
			return RendererAPI::GetAPI();
		}

		/// <summary>
		/// Sets a new viewport size and position.
		/// </summary>
		/// <param name="x">Viewport position X.</param>
		/// <param name="y">Viewport position Y.</param>
		/// <param name="width">Viewport width.</param>
		/// <param name="height">Viewport height.</param>
		static void SetViewport(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

		/// <summary>
		/// Sets the new clear color.
		/// </summary>
		/// <param name="color">New clear color.</param>
		static void SetClearColor(const glm::vec4& color);

		/// <summary>
		/// Clears the screen buffer using the current clear color.
		/// </summary>
		static void Clear();
	private:
		/// <summary>
		/// Current renderer API.
		/// </summary>
		static RendererAPI* s_RendererAPI;
	};
}

#endif // !RENDERER_H_