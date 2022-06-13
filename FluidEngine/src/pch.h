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
#include <array>

// glm
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

// Fluid Engine
#include <FluidEngine/Debug/Debug.h>
#include <FluidEngine/Core/Input.h>
#include <FluidEngine/Core/Ref.h>
#include "FluidEngine/Core/Events/Event.h"
#include "FluidEngine/Core/Events/ApplicationEvent.h"
#include "FluidEngine/Core/Events/KeyEvent.h"
#include "FluidEngine/Core/Events/MouseEvent.h"

#endif // !PCH_H_

// File name macro, simplifies the __FILE__ macro so that it only returns the file name instead of the entire path.
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
// Bit shifts the specified variable.
#define BIT(x) (1u << x)
// Binds function to a specific event.
#define BIND_EVENT_FN(fn) [this](auto&&... args) -> decltype(auto) { return this->fn(std::forward<decltype(args)>(args)...); }