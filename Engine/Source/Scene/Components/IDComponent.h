#ifndef ID_COMPONENT_H_
#define ID_COMPONENT_H_

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

#endif // !ID_COMPONENT_H_