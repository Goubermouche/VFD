#ifndef COMPONENTS_H_
#define COMPONENTS_H_

#include "FluidEngine/Core/Math/Math.h"
#include "FluidEngine/Core/Math/GlmSerialization.h"
#include "FluidEngine/Core/Cryptography/UUID.h"
#include "FluidEngine/Renderer/Renderer.h"
#include "FluidEngine/Renderer/Mesh/TriangleMesh.h"

// Components
#include "FluidEngine/Scene/Components/IDComponent.h"
#include "FluidEngine/Scene/Components/TagComponent.h"
#include "FluidEngine/Scene/Components/RelationshipComponent.h"
#include "FluidEngine/Scene/Components/TransformComponent.h"
#include "FluidEngine/Scene/Components/MaterialComponent.h"
#include "FluidEngine/Scene/Components/MeshComponent.h"
#include "FluidEngine/Scene/Components/SPHSimulationComponent.h"

// How to add new components: 
// 1. Create a new component.
// 2. Add a serialize() function to it. 
// 3. Add the component to Save() and Load() functions in Scene.cpp, so they can be saved and loaded properly.

// TODO: serialization
// TODO: mehs factory

#endif // !COMPONENTS_H_