#ifndef INPUT_H
#define INPUT_H

#include "KeyCodes.h"

namespace vfd {
	class Input {
	public:
		/// <summary>
		/// Checks whether a given key is pressed on the current frame. 
		/// </summary>
		/// <param name="key">ID of the requested key. </param>
		/// <returns>True/False based on key state.</returns>
		static bool IsKeyPressed(KeyCode key);

		/// <summary>
		/// Checks whether a given mouse button is pressed on the current frame. 
		/// </summary>
		/// <param name="button">ID of the requested mouse button.</param>
		/// <returns>True/False based on mouse button state. </returns>
		static bool IsMouseButtonPressed(MouseButton button);

		/// <summary>
		/// Returns the current mouse position relative to the app window (X axis). 
		/// </summary>
		/// <returns>Mouse position (X axis).</returns>
		static float GetMouseX();

		/// <summary>
		/// Returns the current mouse position relative to the app window (Y axis). 
		/// </summary>
		/// <returns>Mouse position(Y axis).</returns>
		static float GetMouseY();

		/// <summary>
		/// Returns the current mouse position relative to the app window. 
		/// </summary>
		/// <returns>Mouse position.</returns>
		static const glm::vec2& GetMousePosition();
	};
}

#endif // !INPUT_H