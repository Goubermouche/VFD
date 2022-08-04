#ifndef TIME_H
#define TIME_H

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

#endif // !TIME_H

