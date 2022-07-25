#ifndef DEBUG_H_
#define DEBUG_H_

#include "pch.h"
#include "FluidEngine/Compute/Utility/CUDA/cutil.h"

namespace fe {
	const enum ConsoleColor {
		Blue = 9,
		Green = 10,
		Cyan = 11,
		Red = 12,
		Purple = 13,
		Yellow = 14,
		White = 15,
		RedBackground = 64
	};

	namespace debug {
		const static HANDLE s_ConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
		const static uint32_t s_IndentSize = 20;


		// Example usage: 

		//LOG(1);
		//LOG(1, "source");
		//LOG(1, ConsoleColor::Cyan);
		//LOG(1, "source", ConsoleColor::Cyan);

		/// <summary>
		/// Log wrapper function.
		/// </summary>
		/// <typeparam name="T">Message type.</typeparam>
		/// <param name="message">Message.</param>
		/// <param name="origin">Origin of the message. </param>
		/// <param name="color">Color of the printed text. </param> 
		template<typename T>
		inline static void Log(T message, const std::string& origin = "", ConsoleColor color = ConsoleColor::White) {
			SetConsoleTextAttribute(s_ConsoleHandle, color);
			if (origin.empty()) {
				std::cout << std::string(s_IndentSize + 2, ' ') << message << std::endl;
			}
			else {
				std::cout << "[" << origin << "]" << std::string(s_IndentSize - origin.size(), ' ') << message << std::endl;
			}
			SetConsoleTextAttribute(s_ConsoleHandle, White);
		}

		template<typename T>
		inline static void Log(T message, ConsoleColor color) {
			Log(message, "", color);
		}

		/// <summary>
		/// Checks the specified conditions value, if the resulting value is false a breakpoint is triggered and the supplied 
		/// message is shown.
		/// </summary>
		/// <param name="result"></param>;;
		/// <param name="message"></param>
		/// <returns></returns>
		inline static bool Assert(bool result, const std::string& message, const std::string& origin = "") {
			if (!result) {
				SetConsoleTextAttribute(s_ConsoleHandle, RedBackground);
				std::cout << "[" << origin << "]" << std::string(s_IndentSize - origin.size(), ' ') << message << std::endl;
				SetConsoleTextAttribute(s_ConsoleHandle, White);
				return false;
			}

			return true;
		}

		class Timer {
		public:
			Timer(const std::string& origin)
				: m_Origin(origin), m_Begin(std::chrono::steady_clock::now())
			{}

			~Timer() {
				const auto duration = std::chrono::steady_clock::now() - m_Begin;
				float time = ((float)std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) / 1000.0f;
				// Log(std::to_string(time) + "ms", m_Origin, ConsoleColor::Cyan);
			}
		private:
			const std::chrono::steady_clock::time_point m_Begin;
			std::string m_Origin;
		};
	}

}

// Whether debug macros such as Assert() or Log() should be enabled in release
#define ENABLE_DEBUG_MACROS_RELEASE true


// File name macro, simplifies the __FILE__ macro so that it only returns the file name instead of the entire path.
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#ifndef NDEBUG
#define DEBUG true
#else
#define DEBUG false
#endif // !NDEBUG

#if DEBUG || ENABLE_DEBUG_MACROS_RELEASE
// Expansion macros
#define EXPAND(x) x
#define GET_MACRO(_2, _1, NAME, ...) NAME
// Cuda assert
#define COMPUTE_SAFE(call) CUDA_SAFE_CALL(call)
#define COMPUTE_CHECK(errorMessage) cudaGetLastError();
// Logging
#define LOG(...) fe::debug::Log(__VA_ARGS__);
#define WARN(...) fe::debug::Log(__VA_ARGS__, ConsoleColor::Yellow);
#define ERR(...) fe::debug::Log(__VA_ARGS__, ConsoleColor::Red);

#else
// Expansion macros
#define EXPAND(x)
#define GET_MACRO(_2, _1, NAME, ...)
// Cuda assert
#define COMPUTE_SAFE(call) call
#define COMPUTE_CHECK(errorMessage)
// Logging
#define LOG(...)
#define WARN(...)
#define ERR(...)
#endif // DEBUG || ENABLE_DEBUG_MACROS_RELEASE

// Custom assertion macro, checks if an expression is true, in case it isn't it creates a breakpoint and prints the error message.
#pragma region Assert
#define ASSERT1(...) { if(!(fe::debug::Assert(false, __VA_ARGS__, __FILENAME__))){__debugbreak(); }}
#define ASSERT2(...) { if(!(fe::debug::Assert(__VA_ARGS__, __FILENAME__))){__debugbreak(); }}
#define ASSERT(...)              EXPAND(GET_MACRO(__VA_ARGS__, ASSERT2, ASSERT1)(__VA_ARGS__))
#pragma endregion

// Basic profiler function that measures the time a scope took to execute in miliseconds, resulting 
// values can then be retrieved using the Profiler::GetTimings() function.
#define PROFILE_SCOPE /*const fe::debug::Profiler profiler(__FUNCTION__);*/

#define TIME_SCOPE const fe::debug::Timer timer(__FUNCTION__);
#endif // !DEBUG_H_