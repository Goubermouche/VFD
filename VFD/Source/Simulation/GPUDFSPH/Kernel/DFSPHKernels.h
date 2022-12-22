#ifndef DFPSH_KERNELS_H
#define DFPSH_KERNELS_H

#include "Core/Math/Math.h"

namespace vfd {
	struct TestKernel {
		int data[2];
	};

	struct DFSPHCubicKernel {
		__host__ __device__ float GetRadius() {
			return m_Radius;
		}

		__host__ __device__ void SetRadius(const float value) {
			m_Radius = value;
			constexpr float pi = static_cast<float>(PI);

			const float h3 = m_Radius * m_Radius * m_Radius;
			m_K = static_cast<float>(8.0) / (pi * h3);
			m_L = static_cast<float>(48.0) / (pi * h3);
			m_WZero = W({ 0.0f, 0.0f, 0.0f });
		}

		__host__ __device__ float W(const float r) {
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

		__host__ __device__ float W(const glm::vec3& r) {
			return W(sqrt(dot(r, r)));
		}

		__host__ __device__ glm::vec3 GradientW(const glm::vec3& r) {
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

		__host__ __device__ float WZero() {
			return m_WZero;
		}
	protected:
		float m_Radius;
		float m_K;
		float m_L;
		float m_WZero;
	};

	template <typename KernelType, unsigned int Resolution = 10000u>
	struct DFSPHPrecomputedKernel {
		__host__ __device__ float GetRadius() {
			return m_Radius;
		}

		__host__ __device__ void SetRadius(const float value, KernelType& kernel) {
			m_Radius = value;
			m_Radius2 = m_Radius * m_Radius;
			kernel.SetRadius(value);
			const float stepSize = m_Radius / (float)(Resolution - 1);
			m_InvStepSize = static_cast<float>(1.0) / stepSize;
			for (unsigned int i = 0; i < Resolution; i++)
			{
				const float posX = stepSize * (float)i;
				m_W[i] = kernel.W(posX);
				kernel.SetRadius(value);
				if (posX > 1.0e-9) {
					m_GradientW[i] = kernel.GradientW(glm::vec3(posX, 0.0f, 0.0f))[0] / posX;
				}
				else {
					m_GradientW[i] = 0.0f;
				}
			}
			m_GradientW[Resolution] = 0.0f;
			m_WZero = W(static_cast<float>(0.0));
		}

		__host__ __device__ float W(const float r) {
			float res = 0.0f;
			if (r <= m_Radius) {
				const unsigned int pos = std::min<unsigned int>((unsigned int)(r * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_W[pos] + m_W[pos + 1]);
			}
			return res;
		}

		__host__ __device__ float W(const glm::vec3& r) {
			float res = 0.0;
			const float r2 = dot(r, r);
			if (r2 < m_Radius2) {
				const float rl = sqrt(r2);
				const unsigned int pos = std::min<unsigned int>((unsigned int)(rl * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_W[pos] + m_W[pos + 1]);
			}
			return res;
		}

		__host__ __device__ glm::vec3 GradientW(const glm::vec3& r) {
			glm::vec3 res;
			const float rl = sqrt(glm::dot(r, r));
			if (rl <= m_Radius) {
				const unsigned int pos = std::min<unsigned int>(static_cast<unsigned int>(rl * m_InvStepSize), Resolution - 2u);
				res = static_cast<float>(0.5) * (m_GradientW[pos] + m_GradientW[pos + 1]) * r;
			}
			else {
				res = { 0.0f, 0.0f, 0.0f };
			}

			return res;
		}

		__host__ __device__ float WZero() {
			return m_WZero;
		}
	protected:
		float m_W[Resolution];
		float m_GradientW[Resolution + 1];
		float m_Radius;
		float m_Radius2;
		float m_InvStepSize;
		float m_WZero;
	};

	typedef DFSPHPrecomputedKernel<DFSPHCubicKernel, 10000u> PrecomputedDFSPHCubicKernel;
}

#endif // !DFPSH_KERNELS_H