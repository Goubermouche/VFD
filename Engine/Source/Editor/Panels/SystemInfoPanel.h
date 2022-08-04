#ifndef SYSTEM_INFO_PANEL_H
#define SYSTEM_INFO_PANEL_H

#include "Editor/Panels/EditorPanel.h"

namespace fe {
	class SystemInfoPanel : public EditorPanel
	{
	public:
		SystemInfoPanel();

		void OnUpdate() override;
	private:
		std::string m_CPUName;
		int m_CPUCoreCount;
	};
}

#endif // !SYSTEM_INFO_PANEL_H