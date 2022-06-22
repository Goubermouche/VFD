#ifndef SYSTEM_INFO_PANEL_H_
#define SYSTEM_INFO_PANEL_H_

#include "FluidEngine/Editor/Panels/EditorPanel.h"

namespace fe {
	enum MyItemColumnID
	{
		MyItemColumnID_Call,
		MyItemColumnID_Count,
		MyItemColumnID_Average,
		MyItemColumnID_Combined
	};
	

	class SystemInfoPanel : public EditorPanel
	{
	public:
		SystemInfoPanel();

		virtual void OnUpdate() override;
	private:
		std::string m_CPUName;
		int m_CPUCoreCount;
	};
}

#endif // !SYSTEM_INFO_PANEL_H_