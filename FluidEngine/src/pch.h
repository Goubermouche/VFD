#ifndef PCH_H_
#define PCH_H_

#include <iostream>

// glm
#include <glm/glm.hpp>

// OpenGL
// TODO: after the renderer gets sufficiently developed move this to the OpenGLRenderer 
#include <Glad/glad.h>
#include <GLFW/glfw3.h>

// ImGui
// TODO: after the UI renderer gets sufficiently developed move this to the ImGuiUIRenderer (?)
#include "imgui.h"
//#include "Platform/ImGui/ImGuiRenderer.h"
//#include "Platform/ImGui/ImGuiGLFWBackend.h"
#include <imgui_internal.h>

// Bind function to a specific event
#define BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }
#endif