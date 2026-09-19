#pragma once
#include <enginemath/vec3.hpp>

#include "../../defines.hpp"
#include "../Model/model.hpp"

#include <cmath>
#include <vector>
#include <unordered_map>

namespace HTN {
	struct GroundGrid {
		f32 cellSize = 4.0f;
		std::unordered_map<i64, std::vector<u32>> cells;

		void build(const CollisionMesh& collision);
		i64 cellKey(f32 x, f32 z) const;
	};

	class Raycast {
	public:
		static f32 getGround(const enginemath::Vec3& pos, const CollisionMesh& collision, const GroundGrid& grid);

	private:
		static bool rayIntersection(const enginemath::Vec3& origin, const enginemath::Vec3& dir,
									const enginemath::Vec3& v0, const enginemath::Vec3& v1,
									const enginemath::Vec3& v2, f32& t);
	};
} // namespace HTN
