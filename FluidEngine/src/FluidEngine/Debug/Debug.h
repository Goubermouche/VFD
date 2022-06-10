#ifndef DEBUG_H_
#define DEBUG_H_

#include "pch.h"
#include <span>

namespace fe::debug {

	const static HANDLE s_ConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);

	const enum ConsoleColor {
		Blue = 9,
		Green = 10,
		Red = 12,
		Yellow = 14,
		White = 15,
		RedBackground = 64
	};

	/// <summary>
	/// Basic debugging function used in macros for greater ease of use.
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="message"></param>
	/// <param name="color"></param>
	template<typename T>
	inline static void Log(T message, const std::string& origin = "engine", long lineNumber = 0, ConsoleColor color = Blue) {
		SetConsoleTextAttribute(s_ConsoleHandle, White);
		std::cout << "[" << origin << "][" << lineNumber << "]";
		SetConsoleTextAttribute(s_ConsoleHandle, color);
		std::cout << message << std::endl;
		SetConsoleTextAttribute(s_ConsoleHandle, White);
	}

	/// <summary>
	/// Checks the specified conditions value, if the resulting value is false a breakpoint is triggered and the supplied 
	/// message is shown.
	/// </summary>
	/// <param name="result"></param>
	/// <param name="message"></param>
	/// <returns></returns>
	inline static bool Assert(bool result, const std::string& message, const std::string& origin = "engine") {
		if (!result) {
			SetConsoleTextAttribute(s_ConsoleHandle, RedBackground);
			std::cout << "[" << origin << "]";
			SetConsoleTextAttribute(s_ConsoleHandle, Red);
			std::cout << " assertion failed!: " << message << std::endl;
			SetConsoleTextAttribute(s_ConsoleHandle, White);
			return false;
		}

		return true;
	}

	/// <summary>
	/// A simple scope-based profiler, used in fe_PROFILE, the resulting time value may be retrieved at any time by
	/// calling the GetTimings() function.
	/// </summary>
	/// TODO: clear s_Timings every frame (?)
	class Profiler {
	public:
		Profiler(const std::string& caller)
			: m_Caller(caller), m_Begin(std::chrono::steady_clock::now()) {}
		~Profiler() {
			const auto duration = std::chrono::steady_clock::now() - m_Begin;

			float time = ((float)std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) / 1000.0f;

			s_Timings.insert_or_assign(m_Caller, time);
		}

		static inline const std::unordered_map< std::string, float>& GetTimings() {
			return s_Timings;
		}
	private:
		static inline std::unordered_map<std::string, float> s_Timings;
		const std::chrono::steady_clock::time_point m_Begin;
		std::string m_Caller;
	};
}

#pragma region Ostream overloads
inline std::ostream& operator<< (std::ostream& out, const glm::vec2& vec) {
	out << "{" << vec.x << ", " << vec.y << ", " << "}";
	return out;
}

inline std::ostream& operator<< (std::ostream& out, const glm::vec3& vec) {
	out << "{" << vec.x << ", " << vec.y << ", " << vec.z << "}";
	return out;
}

inline std::ostream& operator<< (std::ostream& out, const glm::vec4& vec) {
	out << "{" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << "}";
	return out;
}
#pragma endregion 

// Expansion macros
#define EXPAND(x) x
#define GET_MACRO(_2, _1, NAME, ...) NAME

#pragma region Log
#define LOG1(message)             fe::debug::Log(message, __FILENAME__, __LINE__ );
#define LOG2(message, origin)     fe::debug::Log(message, origin, __LINE__ );
#define LOG(...)     EXPAND(GET_MACRO(__VA_ARGS__, LOG2,     LOG1)(__VA_ARGS__))
#pragma endregion

// Debug warn, highlights the logged message in yellow.
#pragma region Warn
#define WARN1(message)            fe::debug::Log(message, __FILENAME__, __LINE__ , fe::debug::Yellow);
#define WARN2(message, origin)    fe::debug::Log(message, origin, __LINE__ , fe::debug::Yellow);
#define WARN(...)    EXPAND(GET_MACRO(__VA_ARGS__, WARN2,    WARN1)(__VA_ARGS__))
#pragma endregion

// Debug error, highlights the logged message in red.
#pragma region Error
#define ERR1(message)             fe::debug::Log(message, __FILENAME__, __LINE__ , fe::debug::Red);
#define ERR2(message, origin)     fe::debug::Log(message, origin, __LINE__ , fe::debug::Red);
#define ERR(...)   EXPAND(GET_MACRO(__VA_ARGS__,   ERR2,     ERR1)(__VA_ARGS__))
#pragma endregion

// Custom assertion macro, checks if an expression is true, in case it isn't it creates a breakpoint and prints the error message.
#pragma region Assert
#define ASSERT1(...){  if(!(fe::debug::Assert(false, __VA_ARGS__, __FILENAME__))){__debugbreak();}}
#define ASSERT2(...){  if(!(fe::debug::Assert(__VA_ARGS__, __FILENAME__))){__debugbreak();}}
#define ASSERT(...) EXPAND(GET_MACRO(__VA_ARGS__, ASSERT2, ASSERT1)(__VA_ARGS__))
#pragma endregion

// Basic profiler function that measures the time a scope took to execute in miliseconds, resulting 
// values can then be retrieved using the Profiler::GetTimings() function.
#define PROFILE const fe::debug::Profiler profiler(__FUNCTION__);

#endif // !DEBUG_H_