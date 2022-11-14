#ifndef REF_H
#define REF_H

namespace fe {
	class RefCounted
	{
	public:
		void IncRefCount() const;
		void DecRefCount() const;

		uint32_t GetRefCount() const;
	private:
		mutable std::atomic<uint32_t> m_RefCount = 0;
	};

	void AddToLiveReferences(void* instance);
	void RemoveFromLiveReferences(void* instance);
	bool IsAlive(void* instance);

	/// <summary>
	/// Intrusive reference counter, similar to std::shared_ptr, unlike 
	/// std::shared_ptr utilization of atomics not is omitted. 
	/// </summary>
	/// <typeparam name="T">Class type.</typeparam>
	template<typename T>
	class Ref
	{
	public:
		Ref();
		Ref(std::nullptr_t n);
		Ref(T* instance);

		template<typename T2>
		Ref(const Ref<T2>& other)
		{
			m_Instance = (T*)other.m_Instance;
			IncRef();
		}

		template<typename T2>
		Ref(Ref<T2>&& other)
		{
			m_Instance = (T*)other.m_Instance;
			other.m_Instance = nullptr;
		}

		static Ref<T> CopyWithoutIncrement(const Ref<T>& other);
		~Ref();

		Ref(const Ref<T>& other)
			: m_Instance(other.m_Instance)
		{
			IncRef();
		}

		Ref& operator=(std::nullptr_t)
		{
			DecRef();
			m_Instance = nullptr;
			return *this;
		}

		Ref& operator=(const Ref<T>& other)
		{
			other.IncRef();
			DecRef();

			m_Instance = other.m_Instance;
			return *this;
		}

		template<typename T2>
		Ref& operator=(const Ref<T2>& other)
		{
			other.IncRef();
			DecRef();

			m_Instance = other.m_Instance;
			return *this;
		}

		template<typename T2>
		Ref& operator=(Ref<T2>&& other)
		{
			DecRef();

			m_Instance = other.m_Instance;
			other.m_Instance = nullptr;
			return *this;
		}

		operator bool() {
			return m_Instance != nullptr; 
		}

		operator bool() const { 
			return m_Instance != nullptr; 
		}

		T* operator->() { 
			return m_Instance;
		}

		const T* operator->() const {
			return m_Instance; 
		}

		T& operator*() {
			return *m_Instance; 
		}

		const T& operator*() const { 
			return *m_Instance; 
		}

		T* Raw();
		const T* Raw() const;

		void Reset(T* instance = nullptr);

		template<typename T2>
		Ref<T2> As() const
		{
			return Ref<T2>(*this);
		}

		template<typename... Args>
		static Ref<T> Create(Args&&... args)
		{
			return Ref<T>(new T(std::forward<Args>(args)...));
		}

		bool operator==(const Ref<T>& other) const
		{
			return m_Instance == other.m_Instance;
		}

		bool operator!=(const Ref<T>& other) const
		{
			return !(*this == other);
		}

		bool EqualsObject(const Ref<T>& other);
	private:
		void IncRef() const;
		void DecRef() const;

		template<class T2>
		friend class Ref;
		mutable T* m_Instance;
	};

	template<typename T>
	class WeakRef
	{
	public:
		WeakRef() = default;
		WeakRef(Ref<T> ref);
		WeakRef(T* instance);

		T* operator->() { 
			return m_Instance; 
		}

		const T* operator->() const { 
			return m_Instance;
		}

		T& operator*() { 
			return *m_Instance;
		}

		const T& operator*() const { 
			return *m_Instance; 
		}

		bool IsValid() const;

		operator bool() const {
			return IsValid();
		}
	private:
		T* m_Instance = nullptr;
	};

	template<typename T>
	inline Ref<T>::Ref()
		: m_Instance(nullptr)
	{}

	template<typename T>
	inline Ref<T>::Ref(std::nullptr_t n)
		: m_Instance(nullptr)
	{}

	template<typename T>
	inline Ref<T>::Ref(T * instance)
		: m_Instance(instance)
	{
		static_assert(std::is_base_of<RefCounted, T>::value, "Class is not RefCounted!");

		IncRef();
	}

	template<typename T>
	inline Ref<T> Ref<T>::CopyWithoutIncrement(const Ref<T>& other)
	{
		Ref<T> result = nullptr;
		result->m_Instance = other.m_Instance;
		return result;
	}

	template<typename T>
	inline Ref<T>::~Ref()
	{
		DecRef();
	}

	template<typename T>
	inline T* Ref<T>::Raw()
	{
		return  m_Instance;
	}

	template<typename T>
	inline const T* Ref<T>::Raw() const
	{
		return  m_Instance;
	}

	template<typename T>
	inline void Ref<T>::Reset(T* instance)
	{
		DecRef();
		m_Instance = instance;
	}

	template<typename T>
	inline bool Ref<T>::EqualsObject(const Ref<T>& other)
	{
		if (!m_Instance || !other.m_Instance) {
			return false;
		}

		return *m_Instance == *other.m_Instance;
	}

	template<typename T>
	inline void Ref<T>::IncRef() const
	{
		if (m_Instance)
		{
			m_Instance->IncRefCount();
			AddToLiveReferences((void*)m_Instance);
		}
	}

	template<typename T>
	inline void Ref<T>::DecRef() const
	{
		if (m_Instance)
		{
			m_Instance->DecRefCount();

			if (m_Instance->GetRefCount() == 0)
			{
				delete m_Instance;
				RemoveFromLiveReferences((void*)m_Instance);
				m_Instance = nullptr;
			}
		}
	}

	template<typename T>
	inline WeakRef<T>::WeakRef(Ref<T> ref)
	{
		m_Instance = ref.Raw();
	}

	template<typename T>
	inline WeakRef<T>::WeakRef(T* instance)
	{
		m_Instance = instance;
	}

	template<typename T>
	inline bool WeakRef<T>::IsValid() const
	{
		return m_Instance ? IsAlive(m_Instance) : false;
	}
}

#endif // !REF_H