#ifndef PCH_H_
#define PCH_H_

// std includes
#include <Windows.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <functional>
#include <future>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <algorithm>
#include <random>
#include <sstream>
#include <queue>
#include <cstdio>
#include <array>
#include <numeric>
#include <regex>
#include <filesystem>

// cuda
#include <cuda.h>

// glm
#define GLM_FORCE_CUDA // Has to be defined after #include <cuda.h>
#include <glm/glm.hpp>
#include <types/vector.hpp>	
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>

// Fluid Engine
#include <FluidEngine/Debug/Debug.h>
#include <FluidEngine/Core/Input.h>
#include <FluidEngine/Core/Ref.h>
#include "FluidEngine/Core/Events/Event.h"
#include "FluidEngine/Core/Events/ApplicationEvent.h"
#include "FluidEngine/Core/Events/KeyEvent.h"
#include "FluidEngine/Core/Events/MouseEvent.h"

#endif // !PCH_H_