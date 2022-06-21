#include "pch.h"
#include "SystemInfoPanel.h"
#include "FluidEngine/Core/Time.h"

#include "psapi.h"

namespace fe {
	SystemInfoPanel::SystemInfoPanel()
	{
		SYSTEM_INFO sysInfo;
		FILETIME ftime, fsys, fuser;

		GetSystemInfo(&sysInfo);
		m_NumProcessors = sysInfo.dwNumberOfProcessors;

		GetSystemTimeAsFileTime(&ftime);
		memcpy(&m_LastCPU, &ftime, sizeof(FILETIME));

		m_ProcessHandle = GetCurrentProcess();
		GetProcessTimes(m_ProcessHandle, &ftime, &ftime, &fsys, &fuser);
		memcpy(&m_LastSysCPU, &fsys, sizeof(FILETIME));
		memcpy(&m_LastUserCPU, &fuser, sizeof(FILETIME));
	}

	void SystemInfoPanel::OnUpdate()
	{
		// CPU usage
		FILETIME ftime, fsys, fuser;
		ULARGE_INTEGER now, sys, user;
		float CPUUsagePercent;

		GetSystemTimeAsFileTime(&ftime);
		memcpy(&now, &ftime, sizeof(FILETIME));

		GetProcessTimes(m_ProcessHandle, &ftime, &ftime, &fsys, &fuser);
		memcpy(&sys, &fsys, sizeof(FILETIME));
		memcpy(&user, &fuser, sizeof(FILETIME));
		CPUUsagePercent = (sys.QuadPart - m_LastSysCPU.QuadPart) + (user.QuadPart - m_LastUserCPU.QuadPart);
		CPUUsagePercent /= (now.QuadPart - m_LastCPU.QuadPart);
		CPUUsagePercent /= m_NumProcessors;
		CPUUsagePercent *= 100;
		m_LastCPU = now;
		m_LastUserCPU = user;
		m_LastSysCPU = sys;

		// Memory usage
		PROCESS_MEMORY_COUNTERS_EX pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		int physMemUsedByMe = (pmc.WorkingSetSize / 1024) / 1024;

		// Render
		ImGui::Text(("CPU usage: " + std::to_string((int)CPUUsagePercent) + "%%").c_str());
		ImGui::Text(("Memory usage: " + std::to_string((pmc.PrivateUsage / 1024) / 1024) + " MB").c_str());
		ImGui::Separator();
		ImGui::Text((std::to_string((int)(1.0f / Time::GetDeltaTime())) + " FPS (" + std::to_string(Time::GetDeltaTime() * 1000) + " ms)").c_str());
	}
}