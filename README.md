# Fluid Engine
FE is a simple, real-time fluid simulation tool for computer graphics and simulation enthusiasts. The current focus is to get the basic structure of the engine to a working state. The engine is currently in development and is not yet ready for use. 

<!-- <img src="https://github.com/Goubermouche/FluidEngine/blob/master/Images/1.png" alt="Image" width="50%"></img><img src="https://github.com/Goubermouche/FluidEngine/blob/master/Images/2.png" alt="Image" width="50%"></img>
<img src="https://github.com/Goubermouche/FluidEngine/blob/master/Images/3.png" alt="Image" width="50%"></img><img src="https://github.com/Goubermouche/FluidEngine/blob/master/Images/4.png" alt="Image" width="50%"></img> -->

## Getting up and running
Visual Studio 2022 is recommended, but older versions should work after a few updates to the [Setup.bat](https://github.com/Goubermouche/FluidEngine/blob/master/Setup.bat) and [premake5.lua](https://github.com/Goubermouche/FluidEngine/blob/master/premake5.lua) files. Additionally, an **Nvidia GPU** is required for running the CUDA code.

<ins>**1. Downloading CUDA**</ins>   
Download the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and follow the setup instructions for your system. The project runs on CUDA toolkit version 11.7, however, older or newer versions will probably work aswell. Older versions of the toolkit can be found [here](https://developer.nvidia.com/cuda-toolkit-archive). Note that if you're using a different version of CUDA you will have to update the [premake5.lua](https://github.com/Goubermouche/FluidEngine/blob/master/premake5.lua) file.

<ins>**2. Downloading Vulkan**</ins>   
Download the [Vulkan SDK](https://vulkan.lunarg.com/) and follow the setup instructions for your system. 

<ins>**3. Downloading the repository:**</ins>   
Clone the repository with `git clone --recursive https://github.com/Goubermouche/FluidEngine.git`.
If the repository was cloned non-recursively previously, use `git submodule update --init` to clone the necessary submodules.

<ins>**4. Configuring the dependencies:**</ins>   
Run the [Setup.bat](https://github.com/Goubermouche/FluidEngine/blob/master/Setup.bat) file found in the root directory. This will create project files for VS2022.

Note: it is recommended that you use the Release configuration for optimal performance (SPIR-V and CUDA performance suffer greatly in Debug).

## Plans
The current plans and known issues can be found [here](https://trello.com/b/WBXdDTXZ/fluidengine). 
### Long term plans
* ~~Engine core, editor & basic renderer~~
* ~~CUDA integration~~
* ~~First fluid simulation~~
* ~~Better component integration and renderer extension~~
* ~~Editor style pass~~
* Viscous FLIP sim
* ...

## Dependencies & Requirements
Note that CUDA and Vulkan have to be installed on the target system in order for the project to compile successfully. All other dependencies are already included and will be downloaded and set up automatically.

|Dependency|Description & Usage|
|---|---|
|[**CUDA**](https://developer.nvidia.com/cuda-downloads)|Used as a parallel computing platform and application programming interface.|
|[**GLFW**](https://github.com/TheCherno/GLFW)|OpenGL window API|
|[**GLUT**](https://www.opengl.org/resources/libraries/glut/glut_downloads.php)|OpenGL API wrapper|
|[**Vulkan**](https://www.lunarg.com/vulkan-sdk/)|Rendering API, currently only used for parsing shaders.|
|[**ImGui**](https://github.com/TheCherno/imgui)|Stateless GUI, works as the core of the editor.|
|[**glm**](https://github.com/g-truc/glm)|Math functions and utilities.|
|[**entt**](https://github.com/skypjack/entt)|Fast ECS, core of the entire application.|
|[**cereal**](https://uscilab.github.io/cereal/)|Simple serialization library used to save & load ECS scenes.|
|[**tinyobjloader**](https://github.com/tinyobjloader/tinyobjloader)|.obj Model loader.|
|[**stb**](https://github.com/nothings/stb)|Utility libraries (stb_image.h, stb_image_write.h)|

## Release Notes
## Screenshots
