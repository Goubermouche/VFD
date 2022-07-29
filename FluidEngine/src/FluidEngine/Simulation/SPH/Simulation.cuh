#ifndef SPH_SIMULATION_CUH_
#define SPH_SIMULATION_CUH_

#include "SimulationParameters.cuh"

namespace fe {
	extern "C" {
		/// <summary>
		/// Copies the specified parameters to the GPU. 
		/// </summary>
		/// <param name="params">Simulation parameters.</param>
		void SetParameters(SimulationData& params);

		/// <summary>
		/// Main particle update function. Takes the current position and velocity and adds the delta velocity to it.
		/// </summary>
		/// <param name="oldPositionVBO">VBO containing the position from the last frame.</param>
		/// <param name="newPositionVBO">VBO containing the current position.</param>
		/// <param name="oldVelocity">Array containing the velocity from the last frame (delta velocity).</param>
		/// <param name="newVelocity">Array containing the current velocity.</param>
		/// <param name="particleCount">Particle count.</param>
		void Integrate(unsigned int oldPositionVBO, unsigned int newPositionVBO, glm::vec4* oldVelocity, glm::vec4* newVelocity, int particleCount);

		/// <summary>
		/// This function calculates the hash for each particle. It takes the current position and calculates the hash for each particle.
		/// </summary>
		/// <param name="positionVBO">VBO containing the current position.</param>
		/// <param name="particleHash">Destination array for the newly generated hash.</param>
		/// <param name="particleCount">Particle count.</param>
		void CalculateHash(unsigned int positionVBO, glm::uvec2* particleHash, int particleCount);

		/// <summary>
		/// This function reorders the particles based on the sorted hash. It takes the current position and velocity and the sorted position and velocity. The reordered particles are stored in the current position and velocity.
		/// </summary>
		/// <param name="oldPositionVBO">VBO containing the position from the last frame.</param>
		/// <param name="oldVelocity">Array containing the velocity from the last frame (delta velocity).</param>
		/// <param name="sortedPosition">Array containing the sorted position.</param>
		/// <param name="sortedVelocity">Array containing the sorted velocity.</param>
		/// <param name="particleHash">Array containing hash and index pairs for every particle.</param>
		/// <param name="cellStart">Array containing starting points for every cell.</param>
		/// <param name="particleCount">Particle count.</param>
		/// <param name="cellCount">Cell count.</param>
		void Reorder(unsigned int oldPositionVBO, glm::vec4* oldVelocity, glm::vec4* sortedPosition, glm::vec4* sortedVelocity,
			glm::uvec2* particleHash, unsigned int* cellStart, unsigned int particleCount, unsigned int cellCount);

		/// <summary>
		/// This function updates the velocity and position of the particles. It takes the current position and velocity and the delta velocity. The delta velocity is calculated in the previous frame.
		/// </summary>
		/// <param name="positionVBO">VBO containing the current position.</param>
		/// <param name="sortedPosition">Array containing the sorted position.</param>
		/// <param name="sortedVelocity">Array containing the sorted velocity.</param>
		/// <param name="oldVelocity">Array containing the velocity from the last frame (delta velocity).</param>
		/// <param name="newVelocity">Array containing the current velocity.</param>
		/// <param name="pressure">Array containing per particle pressure values.</param>
		/// <param name="density">Array containing per particle density values.</param>
		/// <param name="particleHash">Array containing hash and index pairs for every particle.</param>
		/// <param name="cellStart">Array containing starting points for every cell.</param>
		/// <param name="particleCount">Particle count.</param>
		/// <param name="cellCount">Cell count.</param>
		void Collide(unsigned int positionVBO, glm::vec4* sortedPosition, glm::vec4* sortedVelocity,
			glm::vec4* oldVelocity, glm::vec4* newVelocity, float* pressure, float* density,
			glm::uvec2* particleHash, unsigned int* cellStart, unsigned int particleCount, unsigned int cellCount);
	}
}
#endif // !SPH_SIMULATION_CUH_