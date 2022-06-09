#include "pch.h"
#include "Application.h"

namespace fe {
	Application* Application::s_Instance = nullptr;

	Application::Application()
	{
		s_Instance = this;

		Run();
	}

	Application::~Application()
	{
	}

	void Application::Run()
	{
		while (m_Running)
		{
			std::cout << "run" << std::endl;
		}
	}
}


