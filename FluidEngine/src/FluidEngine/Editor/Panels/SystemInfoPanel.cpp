#include "pch.h"
#include "SystemInfoPanel.h"
#include "FluidEngine/Core/Time.h"

namespace fe {
	SystemInfoPanel::SystemInfoPanel()
	{
		int CPUInfo[4] = { -1 };
		unsigned   nExIds, i = 0;
		char CPUBrandString[0x40];
		__cpuid(CPUInfo, 0x80000000);
		nExIds = CPUInfo[0];

		for (i = 0x80000000; i <= nExIds; ++i)
		{
			__cpuid(CPUInfo, i);
			// Interpret CPU brand string
			if (i == 0x80000002) {
				memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
			}
			else if (i == 0x80000003) {
				memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
			}
			else if (i == 0x80000004) {
				memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
			}
		}
		m_CPUInfo = CPUBrandString;
		MEMORYSTATUSEX statex;
		statex.dwLength = sizeof(statex);
		GlobalMemoryStatusEx(&statex);
		m_SystemMemory = std::to_string( (statex.ullTotalPhys / 1024) / 1024);
	}

	void SystemInfoPanel::OnUpdate()
	{
		ImGui::Text(("CPU: " + m_CPUInfo).c_str());
		ImGui::Text(("System memory: " + m_SystemMemory + "MB").c_str());
		ImGui::Separator();
		ImGui::Text(("FPS: " + std::to_string((int)(1.0f / Time::GetDeltaTime())) + " (" + std::to_string(Time::GetDeltaTime()) + "ms)").c_str());
	}
}