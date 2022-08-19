#ifndef CACHE_H
#define CACHE_H

namespace fe {
	template<typename Key, typename Value>
	class Cache {
	public:
		Cache(const std::function<Value(const Key&)>& f, const uint32_t c)
			: m_EvaluationFunction(f), m_Capacity(c)
		{
			ASSERT(m_Capacity != 0, "cache capacity cannot be lower than 0!");
		}

		~Cache() = default;

		Value operator()(const Key& key) {
			auto it = m_KeyToValue.find(key);

			if (it == m_KeyToValue.end()) {
				auto value = m_EvaluationFunction(key);
				Insert(key, value);
				return value;
			}
			else {
				m_KeyTracker.splice(m_KeyTracker.end(), m_KeyTracker, (*it).second.second);
				return(*it).second.first;
			}
		}
	private:
		void Insert(Key const& key, Value const& value) {
			if (m_KeyToValue.size() == m_Capacity) {
				Evict();
			}

			auto it = m_KeyTracker.insert(m_KeyTracker.end(), key);
			m_KeyToValue.insert(std::make_pair(key, std::make_pair(value, it)));
		}

		void Evict() {
			ASSERT(!m_KeyTracker.empty(), "cannot evict item!, key tracker is already empty!");
			auto it = m_KeyToValue.find(m_KeyTracker.front());
			ASSERT(it != m_KeyToValue.end(), "unable to find key in cache!");
			m_KeyToValue.erase(it);
			m_KeyTracker.pop_front();
		}
	private:
		uint32_t m_Capacity = 0;
		std::function<Value(const Key&)> m_EvaluationFunction;
		std::list<Key> m_KeyTracker;
		std::map<Key, std::pair<Value, typename std::list<Key>::iterator>> m_KeyToValue;
	};
}

#endif // !CACHE_H