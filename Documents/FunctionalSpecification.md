# Functional Specification
## Fluid Engine

Version 0.1  
Prepared by Simon Tupý 4.C  
SSPŠaG  
September 2, 2022

Table of Contents
================
* 1 [Introduction](#1-introduction)
  * 1.1 [Document Purpose](#11-document-purpose)
  * 1.2 [Definitions, Acronyms and Abbreviations](#12-definitions-acronyms-and-abbreviations)
  * 1.3 [Target Audience](#13-target-audience)
  * 1.4 [References](#14-references)
* 2 [Scenarios](#2-scenarios)
  * 2.1 [Usecases](#21-usecases)
  * 2.2 [Personas](#22-personas)
  * 2.3 [Details, Motivation and Live Examples](#23-details-motivation-and-live-examples)
  * 2.4 [Product Scope](#24-product-scope)
  * 2.5 [Unimportant Functions and Properties](#25-unimportant-functions-and-properties)
* 3 [Architecture Overview](#3-architecture-overview)
  * 3.1 [Work Flow](#31-work-flow)
    * 3.1.1 [Viewport Window](#311-viewport-window)
      * 3.1.1.1 [Saving and Loading Scenes](#3111-saving-and-loading-scenes)
      * 3.1.1.2 [Camera Controls](#3112-camera-controls)
    * 3.1.2 [Profiler Window](#312-profiler-window)
      * 3.1.2.1 [Frame Time Graph](#3121-frame-time-graph)
      * 3.1.2.2 [Frame Time Counter](#3122-frame-time-counter)
      * 3.1.2.3 [Renderer Statistics](#3123-renderer-statistics)
    * 3.1.3 [Scene Hierarchy Window](#313-scene-hierarchy-window)
      * 3.1.3.1 [Entity List](#3131-entity-list)
    * 3.1.4 [README Window](#314-readme-window)
  * 3.2 [Main Modules](#32-main-modules)
    * 3.2.1 [Core](#321-core)
    * 3.2.1 [Editor](#322-editor)
  * 3.3 [Details](#33-details)
  * 3.4 [Possible Program Flows](#34-possible-program-flows)

# 1. Introduction  
  ## 1.1 Document Purpose
  The purpose of this document is to present a description of the functions and interfaces of the final product. Furthermore, it will introduce and discuss commonly used concepts, interfaces and usecases, including a basic overview of the application's structure and inner workings.
  ## 1.2 Definitions, Acronyms and Abbreviations
| Term | Definition    |
| ---- | ------- |
| Software Requirements Specification  |  A document that completely describes all of the functions of a proposed system and the constraints under which it must operate. For example, this document. |
| CFD | Computational fluid dynamics - the use of applied mathematics, physics and computational software to visualize how a gas or liquid flows, as well as how it affects objects as it flows past.  |
|Advection|The evolution of mass forwards in time using a velocity field. |
|Convection|The process of transferring heat via circulation of movement of the fluid. |
|Lagrangian (methods)|Methods that move a fluid volume (ie. by using advection), most commonly used with particles. |
|Eulerian (methods)|Methods utilizing a grid-based approach to fluid simulation. |
| SPH | Smoothed particle hydrodynamics - the most common form of CFD. |
| FLIP | Fluid-Implicit-Particle method used in CFD. Utilizes fully Lagrangian particles to eliminate convective transport.|
| Device | A device capable of running CUDA code (ie. an Nvidia GPU) |
| Host | The CPU and CPU related code |
| <kbd>Key</kbd> | A keyboard key or mouse button declaration. |
  ## 1.3 Target Audience
This document is intended mainly for testers, developers, the marketing department, and other parties that may be involved. 
  ## 1.4 References
* Software Requirement Specification *Šimon Tupý* https://github.com/Goubermouche/FluidEngine/blob/master/Documents/SoftwareRequirementSpecification.md
# 2. Scenarios
## 2.1 Usecases
The main use case of the product will be the process of viewing and interacting with a real-time fluid simulation. In the future we hope to expand this by providing options of exporting generated fluid simulations. 
## 2.2 Personas
The main audience this product is intended for are computer graphics, simulation and GPU compute programmers. 
## 2.3 Details, Motivation and Live Examples
The original idea was inspired by the realtime toolset provided by [JangaFX](https://jangafx.com/), whom have incidentally also introduces a real-time fluid simulation tool a couple months after we began working on the product. 
The individual products however target different audiences. 
## 2.4 Product Scope
Due to the limited time frame the scale of this project is limited to just the engine and 2 fluid simulation methods (SPH, FLIP). It is expected that both methods will be accelerated using CUDA. 
## 2.5 Unimportant Functions and Properties
N/A
# 3. Architecture Overview
## 3.1 Work Flow
Once the user launches the application an empty scene, viewport, profiler and scene hierarchy panels will be opened. The user will be able to edit the scene by deleting or creating new entities and adding various components to them. The currently available components are:     
  - DFSPHSimulationComponent
  - SPHSimulationComponent
  - IDComponent
  - MaterialComponent
  - MeshComponent
  - TagComponent
  - TransformComponent
  - RelationshipComponent   
  - ColliderComponent

An example worflow may looks something like this: The user launches the application; an empty scene appears. After the application loads the user creates a new entity by pressing the <kbd>RMB</kbd> in the entity hierarchy panel, then he selects it by pressing the <kbd>LMB</kbd>. After selecting the entity the user adds a new 'DFSPHSimulationComponent' component to the entity via the 'Add Component' button located in the component panel, the user uses the default simulation parameters without changing anything. After creating the simulation entity the user creates a new entity by following the entity creation process described above, however, this time a 'ColliderComponent' is added instead. The user then selects the component in the component panel and chooses a desired base mesh for it. Afterwads the user re-selects the simulation entity and presses the 'Bake' button, after the simulation ends baking the user will see a live replay of the simulation with the ability to scrub through it, 

Furthermore the user will have the ability to open and save scenes (the application will be shipped with an assortment of example scenes showcasing its functionality). The application will provide a simple window-based layout that the user can edit and transform to their liking. 
<!--VIEWPORT-->
### 3.1.1 Viewport Window
The viewport window contains an OpenGL framebuffer texture, that displays the current scene. 
#### 3.1.1.1 Saving and Loading Scenes
To save and load scenes the user can right click the viewport and select either the "Save Scene" or "Load Scene" option. The save scene option also provides a simple shortcut - <kbd>Ctrl</kbd> + <kbd>S</kbd> - which will save the currently loaded scene, if a default filepath is provided. 

#### 3.1.1.2 Camera Controls 
The built-in arc ball camera has three movement functions: orbit (<kbd>MMB</kbd>), pan (<kbd>MMB</kbd>+<kbd>Shift</kbd>) and zoom (<kbd>Scroll</kbd>)

<!--PROFILER-->
### 3.1.2 Profiler Window

<p align="center">
  <img src="https://github.com/Goubermouche/VFD/blob/e1c764e41c9e7348378be72898ad8fc2903b4d89/Media/Images/diagram4.png" />
</p>

The profiler window displays useful information about the current scene. 
#### 3.1.2.1 Frame Time Graph 
The frame time graph is a basic graph UI component that displays the time each frame took to compute. The graph is comprised of many rectangular shapes, that are scaled by the current delta time value. 
#### 3.1.2.2 Frame Time Counter
Since the frame time graph by itself does not provide exact information we need another UI component - the frame time counter displays 3 values: max, min and current delta time values in milliseconds. 
#### 3.1.2.3 Renderer Statistics 
The profiler will additionally provide a simple renderer statistics: the current count of all vertices that are being renderer in this frame, the draw call count and whether VSync is enabled. 

<!--SCENE HIERARCHY PANEL-->
### 3.1.3 Scene Hierarchy Window

<p align="center">
  <img src="https://github.com/Goubermouche/VFD/blob/e1c764e41c9e7348378be72898ad8fc2903b4d89/Media/Images/diagram3.png" />
</p>

The scene hierarchy window displays a list of all entities in the current scene. 
#### 3.1.3.1 Entity List 
The individual entities are displayed using a tree diagram. The individual tree nodes respond to <kbd>M2</kbd> events and produce a simple context menu containing the following options: 
- Delete - Deletes the entity. 
- Rename - Creates a rename input field and renames the entity. 
- Create Empty - Creates an empty child entity parented to the entity.    
  
In the case of the <kbd>RMB</kbd> event not being handled by any specific tree node, the list creates a different context menu containing the following options: 
- Create Empty - Creates an empty entity. 
- Save Scene - Opens a save file dialog window and saves the current scene. 
- Load Scene - Opens a load file dialog window and loads the selected scene. 

### 3.1.4 README Window
Some scenes may also provide a simple README panel containing relevant information.
## 3.2 Main Modules
### 3.2.1 Core
The application's core module will provide the essential functionality including but not limited to: math operations, GPU compute, scene and entity management and various utilities. The main loop will also be located here. 
### 3.2.2 Editor
The editor will provide users with a simple GUI built on top of the core module that will enable them to manipulate the underlying data (scenes, entities), for more details see the image below: 

<p align="center">
  <img src="https://github.com/Goubermouche/VFD/blob/e1c764e41c9e7348378be72898ad8fc2903b4d89/Media/Images/diagram1.png" />
</p>

The image contains a preview of what the main editor page might look like (note that this is currently the default layout of the application and that the application supports dynamically moving and reorganizing it's windows). The current layout displays a simple viewport window, which acts as a partent to a profiler (top) and scene window (bottom). Note that the viewport window contains a transformation gizmo.    
Since the user can manipulate the window layout at will they may create their own variations of the default setup; below you can see a simple layout containg the same layout with an additional viewport panel (note that the viewport panel on the right side is currently focused, hence why the gizmo is only visible on this side, also note that the sphere that is currently being rendered is displayed from different angles in both viewports, this showcases the multiple cameras these two viewports work with). 

<p align="center">
  <img src="https://github.com/Goubermouche/VFD/blob/e1c764e41c9e7348378be72898ad8fc2903b4d89/Media/Images/diagram2.png" />
</p>

## 3.3 Details
The fluid simulations and GPU compute-related functionalities rely on a working Nvidia GPU - if a valid compute device is not detected on the target system all of these capabilities will be disabled and not available, however, the engine will continue to function. 
## 3.4 Possible Program Flows
The application functions similarly to [Blender](https://www.blender.org/) - it enables the user to freely interact with a given scene.
