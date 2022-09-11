#ifndef COMPONENTS_H
#define COMPONENTS_H

#include "Core/Math/Math.h"
#include "Core/Cryptography/UUID.h"
#include "Renderer/Renderer.h"
#include "Renderer/Mesh/TriangleMesh.h"

// Components
#include "Scene/Components/IDComponent.h"
#include "Scene/Components/TagComponent.h"
#include "Scene/Components/RelationshipComponent.h"
#include "Scene/Components/TransformComponent.h"
#include "Scene/Components/MaterialComponent.h"
#include "Scene/Components/MeshComponent.h"
#include "Scene/Components/SPHSimulationComponent.h"
#include "Scene/Components/DFSPHSimulationComponent.h"

// How to add new components: 
// 1. Create a new component.
// 2. Add a serialization method to it. 
// 3. Add the component to Save() and Load() functions in Scene.cpp, so they can be saved and loaded properly.

// Utility serialization functions for glm variables. 
namespace glm {
	template<class Archive, class T>
	void serialize(Archive& archive, glm::vec<2, T, glm::defaultp>& v)
	{
		archive(
			cereal::make_nvp("x", v.x),
			cereal::make_nvp("y", v.y)
		);
	}

	template<class Archive, class T>
	void serialize(Archive& archive, glm::vec<3, T, glm::defaultp>& v)
	{
		archive(
			cereal::make_nvp("x", v.x),
			cereal::make_nvp("y", v.y),
			cereal::make_nvp("z", v.z)
		);
	}

	template<class Archive, class T>
	void serialize(Archive& archive, glm::vec<4, T, glm::defaultp>& v)
	{
		archive(
			cereal::make_nvp("x", v.x),
			cereal::make_nvp("y", v.y),
			cereal::make_nvp("z", v.z),
			cereal::make_nvp("w", v.w)
		);
	}

	// Glm matrix serialization
	template<class Archive, class T>
	void serialize(Archive& archive, glm::mat<2, 2, T, glm::defaultp>& m) {
		archive(m[0], m[1]);
	}

	template<class Archive, class T>
	void serialize(Archive& archive, glm::mat<3, 3, T, glm::defaultp>& m) {
		archive(m[0], m[1], m[2]);
	}

	template<class Archive, class T>
	void serialize(Archive& archive, glm::mat<4, 4, T, glm::defaultp>& m) {
		archive(m[0], m[1], m[2], m[3]);
	}

	template<class Archive, class T>
	void serialize(Archive& archive, glm::qua<T, glm::defaultp>& q)
	{
		archive(
			cereal::make_nvp("x", q.x),
			cereal::make_nvp("y", q.y),
			cereal::make_nvp("z", q.z),
			cereal::make_nvp("w", q.w)
		);
	}
}

#endif // !COMPONENTS_H