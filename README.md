## Getting started
<ins>**1. Downloading the repository:**</ins>
Start by cloning the repository with `git clone --recursive https://github.com/Goubermouche/FluidEngine`.
If the repository was cloned non-recursively previously, use `git submodule update --init` to clone the necessary submodules.

<ins>**2. Configuring the dependencies:**</ins>
run the [Setup.bat](https://github.com/Goubermouche/FluidEngine/blob/master/Setup.bat) file found in the root directory. This will create project files for VS2022.

## Dependencies & Requirements

- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [OpenCL](https://www.khronos.org/opencl/)
- [GLFW](https://github.com/TheCherno/GLFW)
- [GLUT](https://www.opengl.org/resources/libraries/glut/glut_downloads.php)
- [ImGui](https://github.com/TheCherno/imgui)
- [glm](https://github.com/g-truc/glm)

- To run & compile this project an Nvidia GPU is required.
- The project officially only supports Visual Studio 2022, but other versions may compile after installing the required dependencies & editing the [Setup.bat](https://github.com/Goubermouche/FluidEngine/blob/master/Setup.bat) file 
