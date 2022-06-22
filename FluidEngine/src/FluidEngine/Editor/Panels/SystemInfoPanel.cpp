#include "pch.h"
#include "SystemInfoPanel.h"
#include "FluidEngine/Core/Time.h"
#include "FluidEngine/Compute/Compute.h"


namespace fe {
	SystemInfoPanel::SystemInfoPanel()
	{
		int CPUInfo[4] = { -1 };
		unsigned   nExIds, i = 0;
		char CPUBrandString[0x40];
		// Get the information associated with each extended ID.
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
		
		// CPU name
		m_CPUName = CPUBrandString;

		// CPU core count
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		m_CPUCoreCount = sysInfo.dwNumberOfProcessors;
	}

	void SystemInfoPanel::OnUpdate()
	{
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 0.0f, 2.0f });
		// FPS info
		ImGui::Text("%0.f FPS (%0.3f ms)", 1.0f / Time::GetDeltaTime(), Time::GetDeltaTime() * 1000.0f);

		// CPU info
		ImGui::Separator();
		ImGui::Text("CPU: %s", m_CPUName.c_str());
		ImGui::Indent();
		ImGui::Text("Core count: %d", m_CPUCoreCount);
		ImGui::Unindent();

		// GPU / Compute info
		ImGui::Separator();
		if (Compute::GetInitState()) {
			DeviceInfo info = Compute::GetDeviceInfo();
			
			ImGui::Text("GPU: %s", info.name.c_str());
			ImGui::Indent();
			ImGui::Text("Clock rate: %d MHz", info.clockRate / 1024);
			ImGui::Text("Global memory: %.0f MB", (float)info.globalMemory / 1024.0f / 1024.0f);
			ImGui::Text("Concurrent kernels: %s", info.concurrentKernels ? "yes" : "no");
			ImGui::Text("Core count: %d", info.coreCount);
			ImGui::Unindent();
		}
		else {
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(255, 0, 0, 255));
			ImGui::Text("failed to initialize compute context!");
			ImGui::PopStyleColor();
		}
		ImGui::PopStyleVar();
	}
}