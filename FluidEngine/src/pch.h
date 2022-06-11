#ifndef PCH_H_
#define PCH_H_

// std includes
#include <Windows.h>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <mutex>
#include <functional>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <random>
#include <sstream>
#include <queue>
#include <cstdio>

// glm
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

// OpenGL
// TODO: after the renderer gets sufficiently developed move this to the OpenGLRenderer 
#include <Glad/glad.h>
#include <GLFW/glfw3.h>

// ImGui
// TODO: after the UI renderer gets sufficiently developed move this to the ImGuiUIRenderer (?)
#include "imgui.h"
#include "FluidEngine/Platform/ImGui/ImGuiRenderer.h"
#include "FluidEngine/Platform/ImGui/ImGuiGLFWBackend.h"
#include <imgui_internal.h>

// Fluid Engine
#include "FluidEngine/Core/Input.h"
#include "FluidEngine/Core/Ref.h"
#include "FluidEngine/Debug/Debug.h"
#include "FluidEngine/Core/Application.h"
#endif // !PCH_H_

#pragma region Macros
// File name macro, simplifies the __FILE__ macro so that it only returns the file name instead of the entire path.
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

// Bit shifts the specified variable.
#define BIT(x) (1u << x)
// Binds function to a specific event.

#define BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }
#pragma region