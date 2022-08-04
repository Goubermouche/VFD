#ifndef INPUT_H
#define INPUT_H

#include "KeyCodes.h"

namespace fe {
	class Input {
	public:
		static bool IsKeyPressed(KeyCode key);
		static bool IsMouseButtonPressed(MouseButton button);
		static float GetMouseX();
		static float GetMouseY();
		static const glm::vec2& GetMousePosition();
	};
}

#endif // !INPUT_H