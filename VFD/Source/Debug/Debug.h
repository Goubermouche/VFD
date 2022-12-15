#ifndef DEBUG_H_
#define DEBUG_H_

#include "pch.h"
#include "Compute/Utility/CUDA/cutil.h"

namespace vfd {
	enum ConsoleColor {
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
		const static uint32_t s_IndentSize = 30;

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
		static void Log(T message, const std::string& origin = "", const ConsoleColor color = ConsoleColor::White) {
			SetConsoleTextAttribute(s_ConsoleHandle, color);

			if (origin.empty()) {
				std::cout << std::string(s_IndentSize + 2, ' ') << message << std::endl;
			}
			else {
				std::cout << "[" << origin << "]" << std::string(s_IndentSize - origin.size(), ' ') << message << std::endl;
			}

			SetConsoleTextAttribute(s_ConsoleHandle, ConsoleColor::White);
		}

		template<typename T>
		static void Log(T message, ConsoleColor color) {
			Log(message, "", color);
		}

		/// <summary>
		/// Checks the specified conditions value, if the resulting value is false a breakpoint is triggered and the supplied 
		/// message is shown.
		/// </summary>
		/// <param name="result"></param>
		/// <param name="message"></param>
		/// <param name="origin"></param>
		/// <returns></returns>
		static bool Assert(const bool result, const std::string& message, const std::string& origin = "") {
			if (!result) {
				SetConsoleTextAttribute(s_ConsoleHandle, RedBackground);
				std::cout << "[" << origin << "]" << std::string(s_IndentSize - origin.size(), ' ') << message << std::endl;
				SetConsoleTextAttribute(s_ConsoleHandle, ConsoleColor::White);
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

				SetConsoleTextAttribute(s_ConsoleHandle, ConsoleColor::Cyan);
				std::cout << "[" << m_Origin << "]" << std::string(s_IndentSize - m_Origin.size(), ' ') << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << " ms" << std::endl;
				SetConsoleTextAttribute(s_ConsoleHandle, ConsoleColor::White);
			}
		private:
			const std::chrono::steady_clock::time_point m_Begin;
			std::string m_Origin;
		};
	}
}

// File name macro, simplifies the __FILE__ macro so that it only returns the file name instead of the entire path.
#define FILENAME (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#ifndef NDEBUG
#define DEBUG true
#else
#define DEBUG false
#endif // !NDEBUG

// Debug logs
#if DEBUG || defined(ENABLE_LOGS_RELEASE)
// Logging
#define LOG(...) vfd::debug::Log(__VA_ARGS__);
#define WARN(...) vfd::debug::Log(__VA_ARGS__, ConsoleColor::Yellow);
#define ERR(...) vfd::debug::Log(__VA_ARGS__, ConsoleColor::Red);
#else
// Logging
#define LOG(...)
#define WARN(...)
#define ERR(...)
#endif

// Asserts
// TODO: add OpenGL asserts
#if DEBUG || defined(ENABLE_LOGS_RELEASE)
// Expansion macros
#define EXPAND(x) x
#define GET_MACRO(_2, _1, NAME, ...) NAME
// CUDA assert
#define COMPUTE_SAFE(call) CUDA_SAFE_CALL(call);
#define COMPUTE_CHECK(errorMessage) cudaGetLastError();
#else
// Expansion macros
#define EXPAND(x)
#define GET_MACRO(_2, _1, NAME, ...)
// CUDA assert
#define COMPUTE_SAFE(call) call
#define COMPUTE_CHECK(errorMessage)
#endif

// Custom assertion macro, checks if an expression is true, in case it isn't it creates a breakpoint and prints the error message.
#pragma region Assert
#define ASSERT1(...) { if(!(vfd::debug::Assert(false, __VA_ARGS__, FILENAME))){__debugbreak(); }}
#define ASSERT2(...) { if(!(vfd::debug::Assert(__VA_ARGS__, FILENAME))){__debugbreak(); }}
#define ASSERT(...)              EXPAND(GET_MACRO(__VA_ARGS__, ASSERT2, ASSERT1)(__VA_ARGS__))
#pragma endregion

// Basic profiler function that measures the time a scope took to execute in milliseconds, resulting 
// values can then be retrieved using the Profiler::GetTimings() function.
#define TIME_SCOPE(name) const vfd::debug::Timer timer(name);
#endif // !DEBUG_H_