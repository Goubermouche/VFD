# VFD
VFD (**V**iscous **F**luid **D**ynamics) is a simple, real-time fluid simulation tool for computer graphics and simulation enthusiasts with special focus on highly-viscous simulations. The current focus is to get the basic structure of the engine to a working state. The engine is currently in development and is not yet ready for use. 

<!-- <img src="https://github.com/Goubermouche/VFD/blob/master/Images/1.png" alt="Image" width="50%"></img><img src="https://github.com/Goubermouche/VFD/blob/master/Images/2.png" alt="Image" width="50%"></img>
<img src="https://github.com/Goubermouche/VFD/blob/master/Images/3.png" alt="Image" width="50%"></img><img src="https://github.com/Goubermouche/VFD/blob/master/Images/4.png" alt="Image" width="50%"></img> -->

<a href="https://github.com/Goubermouche/VFD/blob/52535c243e1d6b4a52dbbd385d0c8c1011b6d6e0/Media/Images/Viscosity1.gif"><img src="https://github.com/Goubermouche/VFD/blob/52535c243e1d6b4a52dbbd385d0c8c1011b6d6e0/Media/Images/Viscosity1.gif" alt="Viscous fluid simulation" width="50%"></img></a><a href="https://github.com/Goubermouche/VFD/blob/188f161ccc3a53a5cda57489aec61b618904425a/Media/Images/Viscosity2.gif"><img src="https://github.com/Goubermouche/VFD/blob/188f161ccc3a53a5cda57489aec61b618904425a/Media/Images/Viscosity2.gif" alt="Viscous fluid simulation" width="50%"></img></a>

## Getting up and running
Visual Studio 2022 is recommended, but older versions should work after a few updates to the [Setup.bat](https://github.com/Goubermouche/VFD/blob/master/Setup.bat) and [premake5.lua](https://github.com/Goubermouche/VFD/blob/master/premake5.lua) files. Additionally, an **Nvidia GPU** is required for running the CUDA code.

<ins>**1. Downloading CUDA**</ins>   
Download the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and follow the setup instructions for your system. The project runs on CUDA toolkit version 11.7, however, older or newer versions will probably work aswell. Older versions of the toolkit can be found [here](https://developer.nvidia.com/cuda-toolkit-archive). Note that if you're using a different version of CUDA you will have to update the [premake5.lua](https://github.com/Goubermouche/VFD/blob/master/premake5.lua) file.

<ins>**2. Downloading Vulkan**</ins>   
Download the [Vulkan SDK](https://vulkan.lunarg.com/) and follow the setup instructions for your system. 

<ins>**3. Downloading the repository:**</ins>   
Clone the repository with `git clone --recursive https://github.com/Goubermouche/VFD.git`.
If the repository was cloned non-recursively previously, use `git submodule update --init` to clone the necessary submodules.

<ins>**4. Configuring the dependencies:**</ins>   
Run the [Setup.bat](https://github.com/Goubermouche/VFD/blob/master/Setup.bat) file found in the root directory. This will create project files for VS2022.

## Plans
The current plans and known issues can be found [here](https://trello.com/b/WBXdDTXZ/fluidengine). 
### Long term plans
For more detailed descriptions and a full list of plans see the [SRS](https://github.com/Goubermouche/VFD/blob/master/Documents/SoftwareRequirementSpecification.md) and [FS](https://github.com/Goubermouche/VFD/blob/master/Documents/FunctionalSpecification.md) documents. 
* ~~Engine core, editor & basic renderer~~
* ~~CUDA integration~~
* ~~First fluid simulation~~
* ~~Better component integration and renderer extension~~
* ~~Editor style pass~~
* ~~DFSPH solver~~
* ~~Viscosity solver~~
* ~~Surface tension solver~~
* Offline DFSPH implementation (GPU based)
* Better simulation GUI
* ...

## Usage
Note that the application is currently in its infancty, and as such won't provide a mature toolset. 

- Once you've starting the application you can modify the current scene; to do so you have to right click the scene hierarchy panel and, using the right mouse button, create an empty entity, you can the select this entity by left clicking it and, by pressing the 'Add Component' button, add various components to it. 

### Controls 
<table>
<thead>
<tr>
<th width="608px">Action</th>
<th width="608px">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>
<kbd>Space</kbd>
</td>
<td>
Pause/Unpause active simulations [Deprecated]
</td>
</tr>
<tr>
<td>
<kbd>RMB</kbd>
</td>
<td>
Open context menu
</td>
</tr>
<tr>
<td>
<kbd>LMB</kbd>
</td>
<td>
Select, interact
</td>
</tr>
<tr>
<td>
<kbd>MMB</kbd> + Drag (while hovering the viewport)
</td>
<td>
Camera orbit
</td>
</tr>
</tr>
<tr>
<td>
<kbd>MMB</kbd> + <kbd>Shift</kbd> (while hovering the viewport)
</td>
<td>
Camera pan
</td>
</tr>
</tr>
<tr>
<td>
Scroll (while hovering the viewport)
</td>
<td>
Modify camera zoom
</td>
</tr>
</tr>
</tbody>
</table>


## Dependencies & Requirements
Note that CUDA and Vulkan have to be installed on the target system in order for the project to compile successfully. All other dependencies are already included and will be downloaded and set up automatically.

<table>
<thead>
<tr>
<th width="608px">Dependency</th>
<th width="608px">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>
<a href="https://developer.nvidia.com/cuda-downloads">CUDA</a>
</td>
<td>
Used as a parallel computing platform and application programming interface.
</td>
</tr>
<tr>
<td>
<a href="https://github.com/TheCherno/GLFW">GLFW</a>
</td>
<td>
OpenGL window API
</td>
</tr>
<tr>
<td>
<a href="https://www.opengl.org/resources/libraries/glut/glut_downloads.php">GLUT</a>
</td>
<td>
OpenGL API wrapper
</td>
</tr>
<tr>
<td>
<a href="https://www.lunarg.com/vulkan-sdk/">Vulkan</a>
</td>
<td>
Rendering API, currently only used for parsing shaders.
</td>
</tr>
<tr>
<td>
<a href="https://github.com/TheCherno/imgui">ImGui</a>
</td>
<td>
Stateless GUI, works as the core of the editor.
</td>
</tr>
<tr>
<td>
<a href="https://github.com/g-truc/glm">glm</a>
</td>
<td>
Math functions and utilities.
</td>
</tr>
<tr>
<td>
<a href="https://github.com/skypjack/entt">entt</a>
</td>
<td>
Fast ECS, core of the entire application.
</td>
</tr>
<tr>
<td>
<a href="https://uscilab.github.io/cereal/">Cereal</a>
</td>
<td>
Simple serialization library used to save & load ECS scenes.
</td>
</tr>
<tr>
<td>
<a href="https://github.com/tinyobjloader/tinyobjloader">tinyobjloader</a>
</td>
<td>
.obj Model loader.
</td>
</tr>
<tr>
<td>
<a href="https://github.com/nothings/stb">STB</a>
</td>
<td>
Utility image libraries (stb_image.h, stb_image_write.h)
</td>
</tr>

</tbody>
</table>

## Release Notes
## Screenshots

<p align="center">
  <img src="https://github.com/Goubermouche/VFD/blob/fc7dacfe8f62635d15a3d51c82fc1ecddc6167fa/Media/Images/6.png" />
</p>

## Acknowledgements
- Implementation of the DFSPH simulator is largely based off of [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH).   

The project currently implements techniques from the following publications: 
- Jan Bender, Tassilo Kugelstadt, Marcel Weiler, Dan Koschier. *Volume Maps: An Implicit Boundary Representation for SPH*. ACM SIGGRAPH Conference on Motion. Interaction and Games, 2019
- Marcel Weiler, Dan Koschier, Magnus Brand, Jan Bender. *A Physically Consistent Implicit Viscosity Solver for SPH Fluids*. Computer Graphics Forum (Eurographics). 37(2). 2018
- F. Zorilla, M. Ritter, J. Sappl, W. Rauch, M. Harders. *Accelerating Surface Tension Calculation in SPH via Particle Classification and Monte Carlo Integration*. Computers 9, 23, 2020.
