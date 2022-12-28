#ifndef DFPSH_KERNELS_H
#define DFPSH_KERNELS_H

#include "Core/Math/Math.h"

namespace vfd {
	template<unsigned int Resolution>
	struct DFSPHKernel {
		__host__ __device__ float GetRadius() const {
			return m_Radius;
		}

		__host__ __device__ void SetRadius(const float radius) {
			constexpr float pi = static_cast<float>(PI);

			m_Radius = radius;
			m_Radius2 = m_Radius * m_Radius;
			const float radius3 = m_Radius * m_Radius * m_Radius;

			m_K = static_cast<float>(8.0) / (pi * radius3);
			m_L = static_cast<float>(48.0) / (pi * radius3);

			m_WZero = CalculateW({0.0f, 0.0f, 0.0f});

			const float stepSize = m_Radius / static_cast<float>(Resolution - 1);
			m_InvStepSize = static_cast<float>(1.0) / stepSize;

			for (unsigned int i = 0; i < Resolution; i++) {
				const float posX = stepSize * static_cast<float>(i);
				m_W[i] = CalculateW(posX);

				if (posX > 1.0e-9) {
					m_GradientW[i] = CalculateGradientW({ posX, 0.0f, 0.0f }).x / posX;
				}
				else {
					m_GradientW[i] = 0.0f;
				}
			}

			m_GradientW[Resolution] = 0.0f;
		}

		__host__ __device__ float GetW(const float r) {
			float res = 0.0f;

			if (r <= m_Radius) {
				const unsigned int pos = glm::min(static_cast<unsigned int>(r * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_W[pos] + m_W[pos + 1]);
			}

			return res;
		}

		__host__ __device__ float GetW(const glm::vec3& r) {
			float res = 0.0;
			const float r2 = dot(r, r);

			if (r2 <= m_Radius2) {
				const float rl = sqrt(r2);
				const unsigned int pos = glm::min(static_cast<unsigned int>(rl * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_W[pos] + m_W[pos + 1]);
			}

			return res;
		}

		__host__ __device__ glm::vec3 GetGradientW(const glm::vec3& r) {
			glm::vec3 res;
			const float rl = sqrt(glm::dot(r, r));

			if (rl <= m_Radius) {
				const unsigned int pos = glm::min(static_cast<unsigned int>(rl * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_GradientW[pos] + m_GradientW[pos + 1]) * r;
			}
			else {
				res = { 0.0f, 0.0f, 0.0f };
			}

			return res;
		}

		__host__ __device__ float GetWZero() {
			return m_WZero;
		}
	private:
		__host__ __device__ float CalculateW(const float r) {
			float res = 0.0;
			const float q = r / m_Radius;

			if (q <= 1.0f) {
				if (q <= 0.5f) {
					const float q2 = q * q;
					const float q3 = q2 * q;
					res = m_K * (static_cast<float>(6.0) * q3 - static_cast<float>(6.0) * q2 + static_cast<float>(1.0));
				}
				else {
					res = m_K * (static_cast<float>(2.0) * pow(static_cast<float>(1.0) - q, static_cast<float>(3.0)));
				}
			}

			return res;
		}

		__host__ __device__ float CalculateW(const glm::vec3& r) {
			return CalculateW(sqrt(dot(r, r)));
		}

		__host__ __device__ glm::vec3 CalculateGradientW(const glm::vec3& r) {
			glm::vec3 res;
			const float rl = sqrt(dot(r, r));
			const float q = rl / m_Radius;
			if (rl > 1.0e-5 && (q <= 1.0f))
			{
				const glm::vec3 gradientQ = r * (static_cast<float>(1.0) / (rl * m_Radius));
				if (q <= 0.5f)
				{
					res = m_L * q * ((float)3.0 * q - static_cast<float>(2.0)) * gradientQ;
				}
				else
				{
					const float factor = static_cast<float>(1.0) - q;
					res = m_L * (-factor * factor) * gradientQ;
				}
			}
			else {
				res = { 0.0f, 0.0f, 0.0f };
			}
			return res;
		}
	private:
		float m_W[Resolution];
		float m_GradientW[Resolution + 1];
		float m_Radius;
		float m_Radius2;
		float m_InvStepSize;
		float m_WZero;

		float m_K;
		float m_L;
	};

	typedef DFSPHKernel<10000u> PrecomputedDFSPHCubicKernel;
}

#endif // !DFPSH_KERNELS_H