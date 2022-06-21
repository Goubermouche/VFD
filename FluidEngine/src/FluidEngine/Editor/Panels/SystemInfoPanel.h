#ifndef SYSTEM_INFO_PANEL_H_
#define SYSTEM_INFO_PANEL_H_

#include "FluidEngine/Editor/Panels/EditorPanel.h"

namespace fe {

	class SystemInfoPanel : public EditorPanel
	{
	public:
		SystemInfoPanel();

		virtual void OnUpdate() override;
	private:
		ULARGE_INTEGER m_LastCPU, m_LastSysCPU, m_LastUserCPU;
		int m_NumProcessors;
		HANDLE m_ProcessHandle; // rename to process
	};
}

#endif // !SYSTEM_INFO_PANEL_H_