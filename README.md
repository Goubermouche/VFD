## Introduction
FE is a simple and fast fluid simulation tool for computer graphics and simulation enthusiasts. The current focus is to get the basic structure of the engine to a working state. The engine is currently in development and is not yet ready for use. 

## Getting started
Visual Studio 2022 is recommended, but older versions should work ater a few updates to the [Setup.bat](https://github.com/Goubermouche/FluidEngine/blob/master/Setup.bat) and [premake5.lua](https://github.com/Goubermouche/FluidEngine/blob/master/premake5.lua) files. Additionally, an Nvidia GPU is 
required for compiling and running the CUDA code.

<ins>**1. Downloading the repository:**</ins>   
Start by cloning the repository with `git clone --recursive https://github.com/Goubermouche/FluidEngine.git`.
If the repository was cloned non-recursively previously, use `git submodule update --init` to clone the necessary submodules.

<ins>**2. Configuring the dependencies:**</ins>   
Run the [Setup.bat](https://github.com/Goubermouche/FluidEngine/blob/master/Setup.bat) file found in the root directory. This will create project files for VS2022.

## Plans
The current plans and known issues can be found [here](https://trello.com/b/WBXdDTXZ/fluidengine). Right now the main goal is to get the engine to a very basic, but stable and working state. After this goal is reached, the engine development will shift focus to fluid simulation. 

## Dependencies & Requirements
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [OpenCL](https://www.khronos.org/opencl/)
* [GLFW](https://github.com/TheCherno/GLFW)
* [GLUT](https://www.opengl.org/resources/libraries/glut/glut_downloads.php)
* [ImGui](https://github.com/TheCherno/imgui)
* [glm](https://github.com/g-truc/glm)
* [entt](https://github.com/skypjack/entt)
* [cereal](https://uscilab.github.io/cereal/)

## Release Notes
