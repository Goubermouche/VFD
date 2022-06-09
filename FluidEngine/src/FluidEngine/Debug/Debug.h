#ifndef DEBUG_H_
#define DEBUG_H_

#include "pch.h"

namespace fe::debug {
	const static HANDLE s_ConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);

	const enum ConsoleColor {
		BLUE = 9,
		GREEN = 10,
		RED = 12,
		YELLOW = 14,
		WHITE = 15
	};

	/// <summary>
	/// Basic debugging function used in macros for greater ease of use.
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="message"></param>
	/// <param name="color"></param>
	template<typename T>
	inline static void Log(T message, const std::string& origin = "x", ConsoleColor color = BLUE) {
		SetConsoleTextAttribute(s_ConsoleHandle, WHITE);
		std::cout << "[" << origin << "] ";
		SetConsoleTextAttribute(s_ConsoleHandle, color);
		std::cout << message << std::endl;
		SetConsoleTextAttribute(s_ConsoleHandle, WHITE);
	}

	/// <summary>
	/// Checks the specified conditions value, if the resulting value is false a breakpoint is triggered and the supplied 
	/// message is shown.
	/// </summary>
	/// <param name="result"></param>
	/// <param name="message"></param>
	/// <returns></returns>
	inline static bool Assert(bool result, const std::string& message) {
		if (!result) {
			SetConsoleTextAttribute(s_ConsoleHandle, RED);
			std::cout << "[engine] assertion failed!: " << message << std::endl;
			SetConsoleTextAttribute(s_ConsoleHandle, 15);

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

#define LOG1(message)             fe::debug::Log(message, "engine");
#define LOG2(message, origin)     fe::debug::Log(message, origin);

#define WARN1(message)            fe::debug::Log(message, "engine", fe::debug::YELLOW);
#define WARN2(message, origin)    fe::debug::Log(message, origin, fe::debug::YELLOW);

#define ERROR1(message)           fe::debug::Log(message, "engine", fe::debug::RED);
#define ERROR2(message, origin)   fe::debug::Log(message, origin, fe::debug::RED);

#define SUCCESS1(message)         fe::debug::Log(message, "engine", fe::debug::GREEN);
#define SUCCESS2(message, origin) fe::debug::Log(message, origin, fe::debug::GREEN);

#define EXPAND(x) x
#define GET_MACRO(_2, _1, NAME, ...) NAME

// Log using these functions, the above ones use the EXPAND macro so we can use operator overloading.
#define LOG(...)     EXPAND(GET_MACRO(__VA_ARGS__, LOG2,     LOG1)(__VA_ARGS__))
#define WARN(...)    EXPAND(GET_MACRO(__VA_ARGS__, WARN2,    WARN1)(__VA_ARGS__))
#define ERROR(...)   EXPAND(GET_MACRO(__VA_ARGS__, ERROR2,   ERROR1)(__VA_ARGS__))
#define SUCCESS(...) EXPAND(GET_MACRO(__VA_ARGS__, SUCCESS2, SUCCESS1)(__VA_ARGS__))

// Custom assertion macro, checks if an expression is true, in case it isn't it creates a breakpoint
// and prints the error message.
// TODO: Optional error message with overload macros.
#define ASSERT(...){  if(!(fe::debug::Assert(__VA_ARGS__))){__debugbreak();}}

// Basic profiler function that measures the time a scope took to execute in miliseconds, resulting 
// values can then be retrieved using the Profiler::GetTimings() function.
// TODO: Optional error message with overload macros.
#define PROFILE const fe::debug::Profiler profiler(__FUNCTION__);

#endif // !DEBUG_H_