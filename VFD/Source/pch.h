#ifndef PCH_H
#define PCH_H

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
#include <immintrin.h>

// CUDA
#include <cuda.h>

// glm
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA // Has to be defined after #include <cuda.h>
#include <glm/glm.hpp>
#include <types/vector.hpp>	
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/matrix_decompose.hpp>

// Fluid Engine
#define ENABLE_LOGS_RELEASE    // Enable logs in the release configuration
#define ENABLE_ASSERTS_RELEASE // Enable asserts in the release configuration (triggered asserts still log relevant information)
#include <Debug/Debug.h>
#include <Core/Input.h>
#include <Core/Ref.h>
#include "Core/Events/Event.h"
#include "Core/Events/ApplicationEvent.h"
#include "Core/Events/KeyEvent.h"
#include "Core/Events/MouseEvent.h"
#include "Core/Events/EditorEvent.h"

#endif // !PCH_H