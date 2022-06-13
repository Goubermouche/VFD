#ifndef KEY_CODES_H_
#define KEY_CODES_H_

namespace fe {
	typedef enum class KeyCode : uint16_t
	{
		Space = 32,
		Apostrophe = 39,   /* ' */
		Comma = 44,        /* , */
		Minus = 45,        /* - */
		Period = 46,       /* . */
		Slash = 47,        /* / */

		D0 = 48,		   /* 0 */
		D1 = 49,		   /* 1 */
		D2 = 50,		   /* 2 */
		D3 = 51,		   /* 3 */
		D4 = 52,		   /* 4 */
		D5 = 53,		   /* 5 */
		D6 = 54,		   /* 6 */
		D7 = 55,		   /* 7 */
		D8 = 56,		   /* 8 */
		D9 = 57,		   /* 9 */

		Semicolon = 59,    /* ; */
		Equal = 61,        /* = */

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
		Backslash = 92,    /* \ */
		RightBracket = 93, /* ] */
		GraveAccent = 96,  /* ` */

		World2 = 162,	   /* non-US #2 */
		World1 = 161,	   /* non-US #1 */

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

#define FE_KEY_SPACE           Key::Space
#define FE_KEY_APOSTROPHE      Key::Apostrophe   /* ' */
#define FE_KEY_COMMA           Key::Comma        /* , */
#define FE_KEY_MINUS           Key::Minus        /* - */
#define FE_KEY_PERIOD          Key::Period       /* . */
#define FE_KEY_SLASH           Key::Slash        /* / */
#define FE_KEY_0               Key::D0
#define FE_KEY_1               Key::D1
#define FE_KEY_2               Key::D2
#define FE_KEY_3               Key::D3
#define FE_KEY_4               Key::D4
#define FE_KEY_5               Key::D5
#define FE_KEY_6               Key::D6
#define FE_KEY_7               Key::D7
#define FE_KEY_8               Key::D8
#define FE_KEY_9               Key::D9
#define FE_KEY_SEMICOLON       Key::Semicolon    /* ; */
#define FE_KEY_EQUAL           Key::Equal        /* = */
#define FE_KEY_A               Key::A
#define FE_KEY_B               Key::B
#define FE_KEY_C               Key::C
#define FE_KEY_D               Key::D
#define FE_KEY_E               Key::E
#define FE_KEY_F               Key::F
#define FE_KEY_G               Key::G
#define FE_KEY_H               Key::H
#define FE_KEY_I               Key::I
#define FE_KEY_J               Key::J
#define FE_KEY_K               Key::K
#define FE_KEY_L               Key::L
#define FE_KEY_M               Key::M
#define FE_KEY_N               Key::N
#define FE_KEY_O               Key::O
#define FE_KEY_P               Key::P
#define FE_KEY_Q               Key::Q
#define FE_KEY_R               Key::R
#define FE_KEY_S               Key::S
#define FE_KEY_T               Key::T
#define FE_KEY_U               Key::U
#define FE_KEY_V               Key::V
#define FE_KEY_W               Key::W
#define FE_KEY_X               Key::X
#define FE_KEY_Y               Key::Y
#define FE_KEY_Z               Key::Z
#define FE_KEY_LEFT_BRACKET    Key::LeftBracket  /* [ */
#define FE_KEY_BACKSLASH       Key::Backslash    /* \ */
#define FE_KEY_RIGHT_BRACKET   Key::RightBracket /* ] */
#define FE_KEY_GRAVE_ACCENT    Key::GraveAccent  /* ` */
#define FE_KEY_WORLD_1         Key::World1       /* non-US #1 */
#define FE_KEY_WORLD_2         Key::World2       /* non-US #2 */

#define FE_KEY_ESCAPE          Key::Escape
#define FE_KEY_ENTER           Key::Enter
#define FE_KEY_TAB             Key::Tab
#define FE_KEY_BACKSPACE       Key::Backspace
#define FE_KEY_INSERT          Key::Insert
#define FE_KEY_DELETE          Key::Delete
#define FE_KEY_RIGHT           Key::Right
#define FE_KEY_LEFT            Key::Left
#define FE_KEY_DOWN            Key::Down
#define FE_KEY_UP              Key::Up
#define FE_KEY_PAGE_UP         Key::PageUp
#define FE_KEY_PAGE_DOWN       Key::PageDown
#define FE_KEY_HOME            Key::Home
#define FE_KEY_END             Key::End
#define FE_KEY_CAPS_LOCK       Key::CapsLock
#define FE_KEY_SCROLL_LOCK     Key::ScrollLock
#define FE_KEY_NUM_LOCK        Key::NumLock
#define FE_KEY_PRINT_SCREEN    Key::PrintScreen
#define FE_KEY_PAUSE           Key::Pause
#define FE_KEY_F1              Key::F1
#define FE_KEY_F2              Key::F2
#define FE_KEY_F3              Key::F3
#define FE_KEY_F4              Key::F4
#define FE_KEY_F5              Key::F5
#define FE_KEY_F6              Key::F6
#define FE_KEY_F7              Key::F7
#define FE_KEY_F8              Key::F8
#define FE_KEY_F9              Key::F9
#define FE_KEY_F10             Key::F10
#define FE_KEY_F11             Key::F11
#define FE_KEY_F12             Key::F12
#define FE_KEY_F13             Key::F13
#define FE_KEY_F14             Key::F14
#define FE_KEY_F15             Key::F15
#define FE_KEY_F16             Key::F16
#define FE_KEY_F17             Key::F17
#define FE_KEY_F18             Key::F18
#define FE_KEY_F19             Key::F19
#define FE_KEY_F20             Key::F20
#define FE_KEY_F21             Key::F21
#define FE_KEY_F22             Key::F22
#define FE_KEY_F23             Key::F23
#define FE_KEY_F24             Key::F24
#define FE_KEY_F25             Key::F25

#define FE_KEY_KP_0            Key::KP0
#define FE_KEY_KP_1            Key::KP1
#define FE_KEY_KP_2            Key::KP2
#define FE_KEY_KP_3            Key::KP3
#define FE_KEY_KP_4            Key::KP4
#define FE_KEY_KP_5            Key::KP5
#define FE_KEY_KP_6            Key::KP6
#define FE_KEY_KP_7            Key::KP7
#define FE_KEY_KP_8            Key::KP8
#define FE_KEY_KP_9            Key::KP9
#define FE_KEY_KP_DECIMAL      Key::KPDecimal
#define FE_KEY_KP_DIVIDE       Key::KPDivide
#define FE_KEY_KP_MULTIPLY     Key::KPMultiply
#define FE_KEY_KP_SUBTRACT     Key::KPSubtract
#define FE_KEY_KP_ADD          Key::KPAdd
#define FE_KEY_KP_ENTER        Key::KPEnter
#define FE_KEY_KP_EQUAL        Key::KPEqual

#define FE_KEY_LEFT_SHIFT      Key::LeftShift
#define FE_KEY_LEFT_CONTROL    Key::LeftControl
#define FE_KEY_LEFT_ALT        Key::LeftAlt
#define FE_KEY_LEFT_SUPER      Key::LeftSuper
#define FE_KEY_RIGHT_SHIFT     Key::RightShift
#define FE_KEY_RIGHT_CONTROL   Key::RightControl
#define FE_KEY_RIGHT_ALT       Key::RightAlt
#define FE_KEY_RIGHT_SUPER     Key::RightSuper
#define FE_KEY_MENU            Key::Menu

#define FE_MOUSE_BUTTON_LEFT   Button::Left
#define FE_MOUSE_BUTTON_RIGHT  Button::Right
#define FE_MOUSE_BUTTON_MIDDLE Button::Middle
}

#endif // !KEY_CODES_H_