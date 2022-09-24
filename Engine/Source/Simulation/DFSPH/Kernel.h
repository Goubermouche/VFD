#ifndef KERNEL_H
#define KERNEL_H

#include "Core/Math/Math.h"

namespace fe {
	class CubicKernel {
	protected: 
		inline static float m_Radius;
		inline static float m_K;
		inline static float m_L;
		inline static float m_WZero;
	public:
		static float GetRadius() { return m_Radius; }

		static void SetRadius(float value) {
			m_Radius = value;
			const float pi = static_cast<float>(PI);
			const float h3 = m_Radius * m_Radius * m_Radius;
			m_K = static_cast<float>(8.0) / (pi * h3);
			m_L = static_cast<float>(48.0) / (pi * h3);
			m_WZero = W({0, 0, 0});
		}

		static float W(const float r) {
			float res = 0.0;
			const float q = r / m_Radius;
			if (q <= 1.0) {
				if (q <= 0.5) {
					const float q2 = q * q;
					const float q3 = q2 * q;
					res = m_K * (static_cast<float>(6.0) * q3 - static_cast<float>(6.0) * q2 + static_cast<float>(1.0));
				}
				else {
					res = m_K * (static_cast<float>(2.0) * glm::pow(static_cast<float>(1.0) - q, static_cast<float>(3.0)));
				}
			}
			return res;
		}

		static float W(const glm::vec3& r) {
			return W(glm::length(r));
		}

		static glm::vec3 GradientW(const glm::vec3& r) {
			glm::vec3 res;
			const float rl = glm::length(r);
			const float q = rl / m_Radius;
			if ((rl > 1.0e-5) && (q <= 1.0))
			{
				const glm::vec3 gradq = r * (static_cast<float>(1.0) / rl * m_Radius);
				if (q <= 0.5)
				{
					res = m_L * q * ((float)3.0 * q - static_cast<float>(2.0)) * gradq;
				}
				else
				{
					const float factor = static_cast<float>(1.0) - q;
					res = m_L * (-factor * factor) * gradq;
				}
			}
			else {
				res = { 0, 0, 0 };
			}
			return res;
		}

		static float WZero() {
			return m_WZero;
		}
	};

	template <typename KernelType, unsigned int resolution = 10000u>
	class PrecomputedKernel {
	protected: 
		inline static float m_W[resolution];
		inline static float m_GradientW[resolution + 1];
		inline static float m_Radius;
		inline static float m_Radius2;
		inline static float m_InvStepSize;
		inline static float m_WZero;
	public:
		static float GetRadius() { return m_Radius; }

		static void SetRadius(float value) {
			m_Radius = value;
			m_Radius2 = m_Radius * m_Radius;
			KernelType::SetRadius(value);
			const float stepSize = m_Radius / (float)(resolution - 1);
			m_InvStepSize = static_cast<float>(1.0) / stepSize;
			for (unsigned int i = 0; i < resolution; i++)
			{
				const float posX = stepSize * (float)i;
				m_W[i] = KernelType::W(posX);
				KernelType::SetRadius(value);
				if (posX > 1.0e-9) {
					m_GradientW[i] = KernelType::GradientW({ posX, 0.0, 0.0 })[0] / posX;
				}
				else {
					m_GradientW[i] = 0.0;
				}
			}
			m_GradientW[resolution] = 0.0;
			m_WZero = W(static_cast<float>(0));
		}

		static float W(const float r) {
			float res = 0.0;
			if (r <= m_Radius) {
				const unsigned int pos = std::min<unsigned int>((unsigned int)(r * m_InvStepSize), resolution - 2u);
				res = static_cast<float>(0.5) * (m_W[pos] + m_W[pos + 1]);
			}
			return res;
		}

		static float W(const glm::vec3& r) {
			float res = 0.0;
			const float r2 = glm::dot(r, r);
			if (r2 < m_Radius2) {
				const float rl = sqrt(r2);
				const unsigned int pos = std::min<unsigned int>((unsigned int)(rl * m_InvStepSize), resolution - 2u);
				res = static_cast<float>(0.5) * (m_W[pos] + m_W[pos + 1]);
			}
			return res;
		}

		static glm::vec3 GradientW(const glm::vec3& r) {
			glm::vec3 res;
			const float rl = glm::length(r);
			if (rl <= m_Radius) {
				const unsigned int pos = std::min<unsigned int>(static_cast<unsigned int>(rl * m_InvStepSize), resolution - 2u);
				res = static_cast<float>(0.5) * (m_GradientW[pos] + m_GradientW[pos + 1]) * r;
			}
			else {
				res = { 0, 0, 0 };
			}

			return res;
		}

		static float WZero() {
			return m_WZero;
		}
	};
}

#endif // !KERNEL_H