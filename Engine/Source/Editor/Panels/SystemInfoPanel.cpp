#include "pch.h"
#include "SystemInfoPanel.h"

#include "Core/Time.h"
#include "Compute/GPUCompute.h"

namespace fe {
	SystemInfoPanel::SystemInfoPanel()
	{
		int CPUInfo[4] = { -1 };
		unsigned nExIds = CPUInfo[0];
		unsigned i = 0;
		char CPUBrandString[0x40];
		// Get the information associated with each extended ID.
		__cpuid(CPUInfo, 0x80000000);
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
		if (ImGui::CollapsingHeader("CPU info")) {
			ImGui::Indent();
			ImGui::Text("CPU: %s", m_CPUName.c_str());
			ImGui::Text("Core count: %d", m_CPUCoreCount);
			ImGui::Unindent();
		}
		
		// GPU / Compute info
		if(ImGui::CollapsingHeader("GPU info")) {
			if (GPUCompute::GetInitState()) {
				const DeviceInfo info = GPUCompute::GetDeviceInfo();

				ImGui::Indent();
				ImGui::Text("GPU: %s", info.name.c_str());
				ImGui::Text("Clock rate: %d MHz", info.clockRate / 1024);
				ImGui::Text("Global memory: %.0f MB", (float)info.globalMemory / 1024.0f / 1024.0f);
				ImGui::Text("Concurrent kernels: %s", info.concurrentKernels ? "yes" : "no");
				// ImGui::Text("Core count: %d", info.coreCount);
				ImGui::Unindent();
			}
			else {
				ImGui::Indent();
				ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(255, 0, 0, 255));
				ImGui::Text("failed to initialize compute context!");
				ImGui::PopStyleColor();
				ImGui::Unindent();
			}
		}

		// Profiler
		if (ImGui::CollapsingHeader("Profiler")){
			//auto t = debug::Profiler::GetTimings();

			//ImGui::Indent();
			//if (ImGui::BeginTable("##ProfilerTable", 4)) {
			//	const float callColumnWidth = 180;
			//	float cellWidth = (ImGui::GetContentRegionAvail().x - callColumnWidth) / 3.0f - 8;

			//	ImGui::TableSetupColumn("API Call", ImGuiTableColumnFlags_WidthFixed, callColumnWidth);
			//	ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, cellWidth);
			//	ImGui::TableSetupColumn("\xCE\xBC CPU", ImGuiTableColumnFlags_WidthFixed, cellWidth);
			//	ImGui::TableSetupColumn("\xE2\x88\x91 CPU", ImGuiTableColumnFlags_WidthFixed, cellWidth);
			//	ImGui::TableHeadersRow();

			//	ImGuiListClipper clipper;
			//	clipper.Begin(t.size());

			//	for (auto& [key, value] : t) {
			//		float combinedTime = 0.0f;

			//		for (size_t i = 0; i < value.size(); i++)
			//		{
			//			combinedTime += value[i];
			//		}

			//		ImGui::TableNextRow();

			//		ImGui::TableNextColumn();
			//		ImGui::Text("%s()", key.c_str());

			//		ImGui::TableNextColumn();
			//		ImGui::Text("%d", value.size());

			//		ImGui::TableNextColumn();
			//		ImGui::Text("%0.1f ms", combinedTime / value.size());

			//		ImGui::TableNextColumn();
			//		ImGui::Text("%0.1f ms", combinedTime);
			//	}

			//	ImGui::EndTable();
			//}

			ImGui::Unindent();
		}

		ImGui::PopStyleVar();
	}
}