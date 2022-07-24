#ifndef TIME_H_
#define TIME_H_

#include <GLFW/glfw3.h>

namespace fe {
	class Time
	{
	public:
		static float Get() {
			return s_Time;
		}

		static float GetDeltaTime() {
			return s_DeltaTime;
		}

		static float GetLastFrameTime() {
			return s_LastFrameTime;
		}
	private:
		// This function should only be called by the Application class once per frame.
		static void OnUpdate();
	private:
		static float s_Time;
		static float s_DeltaTime;
		static float s_LastFrameTime;

		friend class Application;
	};
}

#endif // !TIME_H_

