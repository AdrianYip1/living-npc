#include "raycast.hpp"
#include <cfloat>
#include <algorithm>

HTN::i64 HTN::GroundGrid::cellKey(HTN::f32 x, HTN::f32 z) const {
	HTN::i32 cx = static_cast<HTN::i32>(std::floor(x / cellSize));
	HTN::i32 cz = static_cast<HTN::i32>(std::floor(z / cellSize));
	return (static_cast<HTN::i64>(cx) << 32) | static_cast<HTN::u32>(cz);
}

void HTN::GroundGrid::build(const CollisionMesh& collision) {
	cells.clear();
	auto& verts = collision.verts;
	auto& indices = collision.indices;
	for (HTN::u32 i = 0; i + 2 < static_cast<HTN::u32>(indices.size()); i += 3) {
		const auto& v0 = verts[indices[i]];
		const auto& v1 = verts[indices[i + 1]];
		const auto& v2 = verts[indices[i + 2]];

		HTN::f32 minX = std::min({v0.x, v1.x, v2.x});
		HTN::f32 maxX = std::max({v0.x, v1.x, v2.x});
		HTN::f32 minZ = std::min({v0.z, v1.z, v2.z});
		HTN::f32 maxZ = std::max({v0.z, v1.z, v2.z});

		HTN::i32 x0 = static_cast<HTN::i32>(std::floor(minX / cellSize));
		HTN::i32 x1 = static_cast<HTN::i32>(std::floor(maxX / cellSize));
		HTN::i32 z0 = static_cast<HTN::i32>(std::floor(minZ / cellSize));
		HTN::i32 z1 = static_cast<HTN::i32>(std::floor(maxZ / cellSize));

		for (HTN::i32 cx = x0; cx <= x1; cx++) {
			for (HTN::i32 cz = z0; cz <= z1; cz++) {
				HTN::i64 key = (static_cast<HTN::i64>(cx) << 32) | static_cast<HTN::u32>(cz);
				cells[key].push_back(i);
			}
		}
	}
}

HTN::f32 HTN::Raycast::getGround(const enginemath::Vec3& pos, const CollisionMesh& collision, const GroundGrid& grid) {
	enginemath::Vec3 origin = pos;
	enginemath::Vec3 down = enginemath::Vec3(0.0f, -1.0f, 0.0f);
	HTN::f32 closest = FLT_MAX;

	auto& verts = collision.verts;
	auto& indices = collision.indices;

	HTN::i64 key = grid.cellKey(origin.x, origin.z);
	auto it = grid.cells.find(key);
	if (it == grid.cells.end()) return -999.0f;

	for (HTN::u32 triStart : it->second) {
		const auto& v0 = verts[indices[triStart]];
		const auto& v1 = verts[indices[triStart + 1]];
		const auto& v2 = verts[indices[triStart + 2]];

		HTN::f32 t;
		if (rayIntersection(origin, down, v0, v1, v2, t)) {
			if (t < closest) closest = t;
		}
	}
	if (closest == FLT_MAX) return -999.0f;
	return origin.y - closest;
}

bool HTN::Raycast::rayIntersection(const enginemath::Vec3& origin, const enginemath::Vec3& dir,
	const enginemath::Vec3& v0, const enginemath::Vec3& v1,
	const enginemath::Vec3& v2, HTN::f32& t) {

	enginemath::Vec3 edge1 = v1 - v0;
	enginemath::Vec3 edge2 = v2 - v0;

	enginemath::Vec3 h = dir.cross(edge2);
	HTN::f32 a = edge1.dot(h);
	if (fabs(a) < 1e-6f) return false;

	HTN::f32 f = 1.0f / a;
	enginemath::Vec3 s = origin - v0;

	HTN::f32 u = f * s.dot(h);
	if (u < 0.0f || u > 1.0f) return false;

	HTN::f32 v = f * dir.dot(s.cross(edge1));
	if (v < 0.0f || u + v > 1.0f) return false;

	t = f * edge2.dot(s.cross(edge1));
	return t > 0.0f;
}
