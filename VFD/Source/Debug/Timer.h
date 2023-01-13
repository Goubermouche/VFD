#ifndef TIMER_H
#define TIMER_H

#include "pch.h"

namespace vfd
{
	class Timer
	{
	public:
		Timer() = default;

		void Start()
		{
			m_Start = std::chrono::steady_clock::now();
		}

		void Stop()
		{
			m_End = std::chrono::steady_clock::now();
		}

		template<typename Units = std::chrono::nanoseconds>
		float GetElapsed() const
		{
			return static_cast<float>(std::chrono::duration_cast<Units>(m_End - m_Start).count());
		}
	private:
		std::chrono::steady_clock::time_point m_Start;
		std::chrono::steady_clock::time_point m_End;
	};
}

#endif TIMER_H