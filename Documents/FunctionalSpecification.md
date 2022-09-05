# Functional Specification
## Fluid Engine

Version 0.1  
Prepared by Simon Tupý 3.C  
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
Once the user launches the application an empty scene, viewport, profiler and scene hierarchy panels will be opened. The user will be free to edit the scene by deleting and c new entities and adding various components to them. The currently available components are:     
  - FLIPSimulationComponent
  - SPHSimulationComponent
  - IDComponent
  - MaterialComponent
  - MeshComponent
  - TagComponent
  - TransformComponent
  - RelationshipComponent   

Furthermore the user will have the ability to open and save scenes (the application will be shipped with an assortment of example scenes showcasing its functionality). 
## 3.2 Main Modules
### 3.2.1 Core
The application's core module will provide the essential functionality including but not limited to: math operations, GPU compute, scene and entity management and various utilities. The main loop will also be located here. 
### 3.2.2 Editor
The editor will provide users with a simple GUI that will enable them to manipulate the underlying data (scenes, entities). 
## 3.3 Details
The fluid simulations and GPU compute-related functionalities rely on a working Nvidia GPU - if a valid compute device is not detected on the target system all of these capabilities will be disabled and not available, however, the engine will continue to function. 
## 3.4 Possible Program Flows
The application functions similarly to [Blender](https://www.blender.org/) - it enables the user to freely interact with a given scene.