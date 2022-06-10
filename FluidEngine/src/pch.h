#ifndef PCH_H_
#define PCH_H_

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
//#include "Platform/ImGui/ImGuiRenderer.h"
//#include "Platform/ImGui/ImGuiGLFWBackend.h"
#include <imgui_internal.h>

#include "FluidEngine/Core/Ref.h"
#include "FluidEngine/Debug/Debug.h"


#endif // !PCH_H_

#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
// Bit shift number
#define BIT(x) (1u << x)
// Bind function to a specific event
#define BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }
