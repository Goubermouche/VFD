#include "pch.h"
#include "Time.h"

#include <GLFW/glfw3.h>

namespace fe {
	float Time::s_Time = 0.0f;
	float Time::s_DeltaTime = 0.0f;
	float Time::s_LastFrameTime = 0.0f;

    void Time::OnUpdate()
    {
		s_Time = glfwGetTime();
		s_DeltaTime = s_Time > 0.0f ? (s_Time - s_LastFrameTime) : (1.0f / 60.0f);
		s_LastFrameTime = s_Time;
    }
}