#ifndef ID_COMPONENT_H
#define ID_COMPONENT_H

#include "pch.h"
#include "Core/Cryptography/UUID.h"

namespace fe {
	struct IDComponent
	{
		UUID32 ID = 0;

		template<class Archive>
		void serialize(Archive& archive)
		{
			archive(cereal::make_nvp("id", ID));
		}
	};
}

#endif // !ID_COMPONENT_H