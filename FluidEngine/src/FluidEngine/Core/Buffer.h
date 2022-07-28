#ifndef BUFFER_H_
#define BUFFER_H_

#include "pch.h"

namespace fe {
	class Buffer
	{
	public:
		Buffer()
			: Data(nullptr), Size(0)
		{
		}

		Buffer(void* data, uint32_t size)
			: Data(data), Size(size)
		{
		}

		static Buffer Copy(const void* data, uint32_t size)
		{
			Buffer buffer;
			buffer.Allocate(size);
			memcpy(buffer.Data, data, size);
			return buffer;
		}

		void Allocate(uint32_t size)
		{
			delete[](byte*)Data;
			Data = nullptr;

			if (size == 0) {
				return;
			}

			Data = new byte[size];
			Size = size;
		}

		void Release()
		{
			delete[](byte*)Data;
			Data = nullptr;
			Size = 0;
		}

		void Fill(const int value)
		{
			if (Data) {
				memset(Data, value, Size);
			}
		}

		template<typename T>
		T& Read(uint32_t offset = 0)
		{
			return *(T*)((byte*)Data + offset);
		}

		byte* ReadBytes(uint32_t size, uint32_t offset)
		{
			ASSERT(offset + size <= Size, "buffer overflow!");
			byte* buffer = new byte[size];
			memcpy(buffer, (byte*)Data + offset, size);
			return buffer;
		}

		void Write(void* data, uint32_t size, uint32_t offset = 0)
		{
			ASSERT(offset + size <= Size, "buffer overflow!");
			memcpy((byte*)Data + offset, data, size);
		}

		operator bool() const
		{
			return Data;
		}

		byte& operator[](int index)
		{
			return ((byte*)Data)[index];
		}

		byte operator[](int index) const
		{
			return ((byte*)Data)[index];
		}

		template<typename T>
		T* As() const
		{
			return (T*)Data;
		}
	public:
		void* Data;
		uint32_t Size;
	};
}

#endif // !BUFFER_H_