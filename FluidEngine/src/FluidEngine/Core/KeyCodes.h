#ifndef KEY_CODES_H_
#define KEY_CODES_H_

namespace fe {
	typedef enum class KeyCode : uint16_t
	{
		// From glfw3.h
		Space = 32,
		Apostrophe = 39, /* ' */
		Comma = 44, /* , */
		Minus = 45, /* - */
		Period = 46, /* . */
		Slash = 47, /* / */

		D0 = 48, /* 0 */
		D1 = 49, /* 1 */
		D2 = 50, /* 2 */
		D3 = 51, /* 3 */
		D4 = 52, /* 4 */
		D5 = 53, /* 5 */
		D6 = 54, /* 6 */
		D7 = 55, /* 7 */
		D8 = 56, /* 8 */
		D9 = 57, /* 9 */

		Semicolon = 59, /* ; */
		Equal = 61, /* = */

		A = 65,
		B = 66,
		C = 67,
		D = 68,
		E = 69,
		F = 70,
		G = 71,
		H = 72,
		I = 73,
		J = 74,
		K = 75,
		L = 76,
		M = 77,
		N = 78,
		O = 79,
		P = 80,
		Q = 81,
		R = 82,
		S = 83,
		T = 84,
		U = 85,
		V = 86,
		W = 87,
		X = 88,
		Y = 89,
		Z = 90,

		LeftBracket = 91,  /* [ */
		Backslash = 92,  /* \ */
		RightBracket = 93,  /* ] */
		GraveAccent = 96,  /* ` */

		World1 = 161, /* non-US #1 */
		World2 = 162, /* non-US #2 */

		/* Function keys */
		Escape = 256,
		Enter = 257,
		Tab = 258,
		Backspace = 259,
		Insert = 260,
		Delete = 261,
		Right = 262,
		Left = 263,
		Down = 264,
		Up = 265,
		PageUp = 266,
		PageDown = 267,
		Home = 268,
		End = 269,
		CapsLock = 280,
		ScrollLock = 281,
		NumLock = 282,
		PrintScreen = 283,
		Pause = 284,
		F1 = 290,
		F2 = 291,
		F3 = 292,
		F4 = 293,
		F5 = 294,
		F6 = 295,
		F7 = 296,
		F8 = 297,
		F9 = 298,
		F10 = 299,
		F11 = 300,
		F12 = 301,
		F13 = 302,
		F14 = 303,
		F15 = 304,
		F16 = 305,
		F17 = 306,
		F18 = 307,
		F19 = 308,
		F20 = 309,
		F21 = 310,
		F22 = 311,
		F23 = 312,
		F24 = 313,
		F25 = 314,

		/* Keypad */
		KP0 = 320,
		KP1 = 321,
		KP2 = 322,
		KP3 = 323,
		KP4 = 324,
		KP5 = 325,
		KP6 = 326,
		KP7 = 327,
		KP8 = 328,
		KP9 = 329,
		KPDecimal = 330,
		KPDivide = 331,
		KPMultiply = 332,
		KPSubtract = 333,
		KPAdd = 334,
		KPEnter = 335,
		KPEqual = 336,

		LeftShift = 340,
		LeftControl = 341,
		LeftAlt = 342,
		LeftSuper = 343,
		RightShift = 344,
		RightControl = 345,
		RightAlt = 346,
		RightSuper = 347,
		Menu = 348
	} Key;

	enum class CursorMode
	{
		Normal = 0,
		Hidden = 1,
		Locked = 2
	};

	typedef enum class MouseButton : uint16_t
	{
		Button0 = 0,
		Button1 = 1,
		Button2 = 2,
		Button3 = 3,
		Button4 = 4,
		Button5 = 5,
		Left = Button0,
		Right = Button1,
		Middle = Button2
	} Button;

	inline std::ostream& operator<<(std::ostream& os, KeyCode keyCode)
	{
		os << static_cast<int32_t>(keyCode);
		return os;
	}

	inline std::ostream& operator<<(std::ostream& os, MouseButton button)
	{
		os << static_cast<int32_t>(button);
		return os;
	}
}

#endif // !KEY_CODES_H_

// From glfw3.h
#define FE_KEY_SPACE           ::Hazel::Key::Space
#define FE_KEY_APOSTROPHE      ::Hazel::Key::Apostrophe    /* ' */
#define FE_KEY_COMMA           ::Hazel::Key::Comma         /* , */
#define FE_KEY_MINUS           ::Hazel::Key::Minus         /* - */
#define FE_KEY_PERIOD          ::Hazel::Key::Period        /* . */
#define FE_KEY_SLASH           ::Hazel::Key::Slash         /* / */
#define FE_KEY_0               ::Hazel::Key::D0
#define FE_KEY_1               ::Hazel::Key::D1
#define FE_KEY_2               ::Hazel::Key::D2
#define FE_KEY_3               ::Hazel::Key::D3
#define FE_KEY_4               ::Hazel::Key::D4
#define FE_KEY_5               ::Hazel::Key::D5
#define FE_KEY_6               ::Hazel::Key::D6
#define FE_KEY_7               ::Hazel::Key::D7
#define FE_KEY_8               ::Hazel::Key::D8
#define FE_KEY_9               ::Hazel::Key::D9
#define FE_KEY_SEMICOLON       ::Hazel::Key::Semicolon     /* ; */
#define FE_KEY_EQUAL           ::Hazel::Key::Equal         /* = */
#define FE_KEY_A               ::Hazel::Key::A
#define FE_KEY_B               ::Hazel::Key::B
#define FE_KEY_C               ::Hazel::Key::C
#define FE_KEY_D               ::Hazel::Key::D
#define FE_KEY_E               ::Hazel::Key::E
#define FE_KEY_F               ::Hazel::Key::F
#define FE_KEY_G               ::Hazel::Key::G
#define FE_KEY_H               ::Hazel::Key::H
#define FE_KEY_I               ::Hazel::Key::I
#define FE_KEY_J               ::Hazel::Key::J
#define FE_KEY_K               ::Hazel::Key::K
#define FE_KEY_L               ::Hazel::Key::L
#define FE_KEY_M               ::Hazel::Key::M
#define FE_KEY_N               ::Hazel::Key::N
#define FE_KEY_O               ::Hazel::Key::O
#define FE_KEY_P               ::Hazel::Key::P
#define FE_KEY_Q               ::Hazel::Key::Q
#define FE_KEY_R               ::Hazel::Key::R
#define FE_KEY_S               ::Hazel::Key::S
#define FE_KEY_T               ::Hazel::Key::T
#define FE_KEY_U               ::Hazel::Key::U
#define FE_KEY_V               ::Hazel::Key::V
#define FE_KEY_W               ::Hazel::Key::W
#define FE_KEY_X               ::Hazel::Key::X
#define FE_KEY_Y               ::Hazel::Key::Y
#define FE_KEY_Z               ::Hazel::Key::Z
#define FE_KEY_LEFT_BRACKET    ::Hazel::Key::LeftBracket   /* [ */
#define FE_KEY_BACKSLASH       ::Hazel::Key::Backslash     /* \ */
#define FE_KEY_RIGHT_BRACKET   ::Hazel::Key::RightBracket  /* ] */
#define FE_KEY_GRAVE_ACCENT    ::Hazel::Key::GraveAccent   /* ` */
#define FE_KEY_WORLD_1         ::Hazel::Key::World1        /* non-US #1 */
#define FE_KEY_WORLD_2         ::Hazel::Key::World2        /* non-US #2 */

/* Function keys */
#define FE_KEY_ESCAPE          ::Hazel::Key::Escape
#define FE_KEY_ENTER           ::Hazel::Key::Enter
#define FE_KEY_TAB             ::Hazel::Key::Tab
#define FE_KEY_BACKSPACE       ::Hazel::Key::Backspace
#define FE_KEY_INSERT          ::Hazel::Key::Insert
#define FE_KEY_DELETE          ::Hazel::Key::Delete
#define FE_KEY_RIGHT           ::Hazel::Key::Right
#define FE_KEY_LEFT            ::Hazel::Key::Left
#define FE_KEY_DOWN            ::Hazel::Key::Down
#define FE_KEY_UP              ::Hazel::Key::Up
#define FE_KEY_PAGE_UP         ::Hazel::Key::PageUp
#define FE_KEY_PAGE_DOWN       ::Hazel::Key::PageDown
#define FE_KEY_HOME            ::Hazel::Key::Home
#define FE_KEY_END             ::Hazel::Key::End
#define FE_KEY_CAPS_LOCK       ::Hazel::Key::CapsLock
#define FE_KEY_SCROLL_LOCK     ::Hazel::Key::ScrollLock
#define FE_KEY_NUM_LOCK        ::Hazel::Key::NumLock
#define FE_KEY_PRINT_SCREEN    ::Hazel::Key::PrintScreen
#define FE_KEY_PAUSE           ::Hazel::Key::Pause
#define FE_KEY_F1              ::Hazel::Key::F1
#define FE_KEY_F2              ::Hazel::Key::F2
#define FE_KEY_F3              ::Hazel::Key::F3
#define FE_KEY_F4              ::Hazel::Key::F4
#define FE_KEY_F5              ::Hazel::Key::F5
#define FE_KEY_F6              ::Hazel::Key::F6
#define FE_KEY_F7              ::Hazel::Key::F7
#define FE_KEY_F8              ::Hazel::Key::F8
#define FE_KEY_F9              ::Hazel::Key::F9
#define FE_KEY_F10             ::Hazel::Key::F10
#define FE_KEY_F11             ::Hazel::Key::F11
#define FE_KEY_F12             ::Hazel::Key::F12
#define FE_KEY_F13             ::Hazel::Key::F13
#define FE_KEY_F14             ::Hazel::Key::F14
#define FE_KEY_F15             ::Hazel::Key::F15
#define FE_KEY_F16             ::Hazel::Key::F16
#define FE_KEY_F17             ::Hazel::Key::F17
#define FE_KEY_F18             ::Hazel::Key::F18
#define FE_KEY_F19             ::Hazel::Key::F19
#define FE_KEY_F20             ::Hazel::Key::F20
#define FE_KEY_F21             ::Hazel::Key::F21
#define FE_KEY_F22             ::Hazel::Key::F22
#define FE_KEY_F23             ::Hazel::Key::F23
#define FE_KEY_F24             ::Hazel::Key::F24
#define FE_KEY_F25             ::Hazel::Key::F25

/* Keypad */
#define FE_KEY_KP_0            ::Hazel::Key::KP0
#define FE_KEY_KP_1            ::Hazel::Key::KP1
#define FE_KEY_KP_2            ::Hazel::Key::KP2
#define FE_KEY_KP_3            ::Hazel::Key::KP3
#define FE_KEY_KP_4            ::Hazel::Key::KP4
#define FE_KEY_KP_5            ::Hazel::Key::KP5
#define FE_KEY_KP_6            ::Hazel::Key::KP6
#define FE_KEY_KP_7            ::Hazel::Key::KP7
#define FE_KEY_KP_8            ::Hazel::Key::KP8
#define FE_KEY_KP_9            ::Hazel::Key::KP9
#define FE_KEY_KP_DECIMAL      ::Hazel::Key::KPDecimal
#define FE_KEY_KP_DIVIDE       ::Hazel::Key::KPDivide
#define FE_KEY_KP_MULTIPLY     ::Hazel::Key::KPMultiply
#define FE_KEY_KP_SUBTRACT     ::Hazel::Key::KPSubtract
#define FE_KEY_KP_ADD          ::Hazel::Key::KPAdd
#define FE_KEY_KP_ENTER        ::Hazel::Key::KPEnter
#define FE_KEY_KP_EQUAL        ::Hazel::Key::KPEqual

#define FE_KEY_LEFT_SHIFT      ::Hazel::Key::LeftShift
#define FE_KEY_LEFT_CONTROL    ::Hazel::Key::LeftControl
#define FE_KEY_LEFT_ALT        ::Hazel::Key::LeftAlt
#define FE_KEY_LEFT_SUPER      ::Hazel::Key::LeftSuper
#define FE_KEY_RIGHT_SHIFT     ::Hazel::Key::RightShift
#define FE_KEY_RIGHT_CONTROL   ::Hazel::Key::RightControl
#define FE_KEY_RIGHT_ALT       ::Hazel::Key::RightAlt
#define FE_KEY_RIGHT_SUPER     ::Hazel::Key::RightSuper
#define FE_KEY_MENU            ::Hazel::Key::Menu

// Mouse
#define FE_MOUSE_BUTTON_LEFT    ::Hazel::Button::Left
#define FE_MOUSE_BUTTON_RIGHT   ::Hazel::Button::Right
#define FE_MOUSE_BUTTON_MIDDLE  ::Hazel::Button::Middle
