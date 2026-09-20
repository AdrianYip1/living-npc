#pragma once
#include "defines.hpp"
#include "enginemath/mat4.hpp"
#include <vector>

namespace HTN {
	struct NPCConfig {
		f32 animOffset = 0.0f;
	};

	inline const std::vector<NPCConfig> NPC_LIST = {
		{ 0.0f },
		{ 1.5f },
		{ 0.5f },
		{ 1.0f },
		{ 2.0f },
		{ 2.5f }
	};

	inline u32 npcCount() { return static_cast<u32>(NPC_LIST.size()); }

	struct WorldBounds {
		f32 minX = 0, maxX = 0;
		f32 minZ = 0, maxZ = 0;

		f32 toWorldX(f32 nx) const { return minX + nx * (maxX - minX); }
		f32 toWorldZ(f32 nz) const { return minZ + nz * (maxZ - minZ); }
		f32 toNormX(f32 wx) const { return (maxX != minX) ? (wx - minX) / (maxX - minX) : 0.0f; }
		f32 toNormZ(f32 wz) const { return (maxZ != minZ) ? (wz - minZ) / (maxZ - minZ) : 0.0f; }
	};

	struct Obstacle {
		f32 x, z, radius;
	};

	inline const WorldBounds WORLD_BOUNDS = { -23.7901f, 23.8101f, -23.7713f, 23.7758f };

	inline const std::vector<Obstacle> OBSTACLES = {
		{ -15.1f, -15.9f, 9.6f },
		{ -14.9f, -15.4f, 10.3f },
		{  19.7f, -20.0f, 10.1f },
		{ -21.0f, -21.7f, 12.4f },
		// jacarandas (approximate centers from gltf)
		{  -5.0f,  10.0f, 5.0f },
		{   8.0f,  15.0f, 4.0f },
		{ -12.0f,   5.0f, 3.5f },
		{  15.0f,   8.0f, 4.5f },
		{   0.0f, -10.0f, 3.0f },
	};
}
