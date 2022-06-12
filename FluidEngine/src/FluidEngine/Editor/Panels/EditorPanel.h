#ifndef PANEL_H_
#define PANEL_H_

namespace fe {
	class EditorPanel : public RefCounted {
	public:
		virtual void OnUpdate() = 0;
		virtual void OnEvent(Event& e) = 0;
	protected:
		std::string m_Name;

		friend class PanelManager;
	};
}

#endif // !PANEL_H_