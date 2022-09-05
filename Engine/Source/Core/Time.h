#ifndef TIME_H
#define TIME_H

namespace fe {
	/// <summary>
	/// Global tme wrapper. 
	/// </summary>
	class Time
	{
	public:
		/// <summary>
		/// Gets the current time value. 
		/// </summary>
		/// <returns>Time. </returns>
		static float Get() {
			return s_Time;
		}

		/// <summary>
		/// Gets the time difference between the current and last frame. 
		/// </summary>
		/// <returns>Difference between the current and last frame. </returns>
		static float GetDeltaTime() {
			return s_DeltaTime;
		}

		/// <summary>
		/// Gets the last frame time. 
		/// </summary>
		/// <returns>The last frame time</returns>
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