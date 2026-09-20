#include "defines.hpp"
#include "clock.hpp"
#include "npcConfig.hpp"
#include "RenderingEngine/Platform/window.hpp"
#include "RenderingEngine/Core/camera.hpp"
#include "RenderingEngine/Core/input.hpp"
#include "RenderingEngine/Core/cameraControls.hpp"
#include "RenderingEngine/renderer.hpp"
#include "RenderingEngine/Utility/raycast.hpp"
#include "outputText/parseOutput.hpp"
#include "speechListener.hpp"
#include "httpClient.hpp"
#include "serverLauncher.hpp"
#include "json.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <queue>
#include <map>
#include <set>
#include <cctype>
#include <cmath>
#include <thread>

namespace {
	const std::string VOICE_FEMALE = "en-US-Ava:DragonHDLatestNeural";
	const std::string VOICE_MALE = "en-US-Andrew:DragonHDLatestNeural";
	const std::string SERVER_HOST = "127.0.0.1";
	const int SERVER_PORT = 8765;

	std::string voiceForGender(const std::string& gender) {
		std::string g;
		for (char c : gender) g += (char)std::tolower((unsigned char)c);
		return g == "male" ? VOICE_MALE : VOICE_FEMALE;
	}

	// The player's position, going the other way from npc_state.json: the
	// camera's spot in the scene, normalized 0..1 across the world bounds,
	// and its facing about the up axis -- exactly the terms the sim sends
	// NPC bodies in, so it can turn them straight back into 2D minimap
	// coordinates. Written every poll even standing still: the file's age
	// is how the sim knows this program is still running.
	void writePlayerState(const HTN::WorldBounds& bounds, const enginemath::Vec3& pos,
						  const enginemath::Vec3& forward) {
		std::string dir = CONVO_LOG_DIR;
		std::ofstream out(dir + "/player_state.json.tmp", std::ios::trunc);
		if (!out) return;
		out << "{\"x\": " << bounds.toNormX(pos.x)
			<< ", \"z\": " << bounds.toNormZ(pos.z)
			<< ", \"rot\": " << std::atan2(forward.x, forward.z) << "}";
		out.close();
		std::error_code ec;
		std::filesystem::rename(dir + "/player_state.json.tmp", dir + "/player_state.json", ec);
	}

	void writeSpokenAck(const std::string& conversation, int seq) {
		std::string dir = CONVO_LOG_DIR;
		std::ofstream out(dir + "/spoken.json.tmp", std::ios::trunc);
		if (!out) return;
		out << "{\"conversation\": \"" << conversation << "\", \"seq\": " << seq << "}";
		out.close();
		std::error_code ec;
		std::filesystem::rename(dir + "/spoken.json.tmp", dir + "/spoken.json", ec);
	}

	std::string escapeJson(const std::string& s) {
		std::string out;
		for (char c : s) {
			switch (c) {
			case '"':  out += "\\\""; break;
			case '\\': out += "\\\\"; break;
			case '\n': out += "\\n";  break;
			case '\r': out += "\\r";  break;
			case '\t': out += "\\t";  break;
			default:   out += c;
			}
		}
		return out;
	}

	float lerpAngle(float from, float to, float t) {
		float diff = fmodf(to - from + 3.0f * 3.14159265f, 2.0f * 3.14159265f) - 3.14159265f;
		return from + diff * t;
	}

	int findNearestNPC(const enginemath::Vec3& playerPos,
					   const std::vector<enginemath::Vec3>& npcPositions,
					   const std::vector<bool>& active) {
		int nearest = -1;
		float minDist = 999999.0f;
		for (int i = 0; i < (int)npcPositions.size(); i++) {
			if (!active[i]) continue;
			float dx = playerPos.x - npcPositions[i].x;
			float dz = playerPos.z - npcPositions[i].z;
			float dist = dx * dx + dz * dz;
			if (dist < minDist) {
				minDist = dist;
				nearest = i;
			}
		}
		if (minDist > 10.0f * 10.0f) return -1;
		return nearest;
	}
}

int main() {
	try {
		HTN::ServerLauncher server;
		server.start("python3", PYTHON_SRC_DIR, SERVER_PORT);

		std::cout << "Waiting for Python server..." << std::endl;
		for (int attempt = 0; attempt < 30; attempt++) {
			Sleep(500);
			std::string r = HTN::httpPost(SERVER_HOST, SERVER_PORT, "/api/pause", "{}");
			if (!r.empty()) {
				HTN::httpPost(SERVER_HOST, SERVER_PORT, "/api/resume", "{}");
				break;
			}
		}
		std::cout << "Server ready." << std::endl;

		HTN::Window window(0, 0, "Living Npc: Face Model");
		HTN::Camera camera(enginemath::Vec3(0.0f, 1.56f, 0.5f),
						   static_cast<float>(window.getWidth()) / window.getHeight());
		HTN::Renderer renderer(window, camera);
		HTN::Clock clock;
		HTN::Input input(window);
		HTN::CameraControls controls(clock, window, input, camera);
		HTN::SpeechListener listener;

		HTN::Clock pollClock;
		pollClock.resetTime();
		HTN::Clock fpsClock;
		int frameCount = 0;
		HTN::outputParser outputText(CONVO_LOG_DIR);
		std::queue<HTN::SpokenLine> speechQueue;
		std::string npcStatePath = std::string(CONVO_LOG_DIR) + "/npc_state.json";
		const HTN::WorldBounds& bounds = renderer.getWorldBounds();

		std::string ackConversation;
		int ackSeq = -1;
		std::map<int, std::string> slotConversation;

		std::vector<float> prevNpcX(renderer.faceCount(), -9999.0f);
		std::vector<float> prevNpcZ(renderer.faceCount(), -9999.0f);
		std::vector<bool> npcMoving(renderer.faceCount(), false);
		std::vector<bool> npcActive(renderer.faceCount(), false);
		std::vector<std::string> npcNames(renderer.faceCount());
		std::vector<enginemath::Vec3> npcWorldPositions(renderer.faceCount());
		std::vector<enginemath::Vec3> npcTargetPositions(renderer.faceCount());
		std::vector<float> npcCurrentRot(renderer.faceCount(), 0.0f);
		std::vector<float> npcTargetRot(renderer.faceCount(), 0.0f);

		HTN::Clock dtClock;
		bool talkKeyWasDown = false;
		std::string talkingToNPC;
		bool playerConversationActive = false;
		while (!window.checkClose()) {
			controls.accumulateMovement();
			controls.accumulateRotation();
			controls.updatePos();

			HTN::f32 groundY = 0.0f;
			{
				HTN::f32 floor = HTN::Raycast::getGround(camera.getPos(), renderer.getCollision(), renderer.getGroundGrid());
				if (floor != -999.0f) {
					groundY = floor;
					enginemath::Vec3 pos = camera.getPos();
					pos.y = floor + 3.3f;
					camera.setPos(pos);
				}
			}

			bool talkKeyDown = !ImGui::GetIO().WantCaptureKeyboard && input.keyPressed(GLFW_KEY_T);
			if (talkKeyDown && !talkKeyWasDown) {
				int nearest = findNearestNPC(camera.getPos(), npcWorldPositions, npcActive);
				if (nearest >= 0 && !npcNames[nearest].empty()) {
					std::string targetNPC = npcNames[nearest];
					if (targetNPC != talkingToNPC) {
						std::string oldNPC = talkingToNPC;
						talkingToNPC = targetNPC;
						std::thread([oldNPC, targetNPC, wasActive = playerConversationActive] {
							if (wasActive && !oldNPC.empty())
								HTN::httpPost(SERVER_HOST, SERVER_PORT, "/api/conversation/end",
									"{\"name\":\"" + oldNPC + "\"}");
							HTN::httpPost(SERVER_HOST, SERVER_PORT, "/api/conversation/start",
								"{\"name\":\"" + targetNPC + "\"}");
						}).detach();
						playerConversationActive = true;
					} else if (!playerConversationActive) {
						std::string body = "{\"name\":\"" + talkingToNPC + "\"}";
						std::thread([body] {
							HTN::httpPost(SERVER_HOST, SERVER_PORT, "/api/conversation/start", body);
						}).detach();
						playerConversationActive = true;
					}
					listener.startListening();
					std::cout << "[PTT] Listening... (talking to " << talkingToNPC << ")" << std::endl;
				}
			}
			if (!talkKeyDown && talkKeyWasDown && listener.isListening()) {
				listener.stopListening();
				std::cout << "[PTT] Stopped listening." << std::endl;
			}
			talkKeyWasDown = talkKeyDown;

			if (listener.hasResult()) {
				std::string text = listener.takeResult();
				if (!text.empty() && !talkingToNPC.empty()) {
					std::cout << "[PTT] You said: " << text << std::endl;
					std::string name = talkingToNPC;
					std::thread([name, text] {
						std::string body = "{\"name\":\"" + escapeJson(name) + "\",\"text\":\"" + escapeJson(text) + "\"}";
						std::string response = HTN::httpPost(SERVER_HOST, SERVER_PORT,
							"/api/conversation/say", body);
						std::cout << "[PTT] NPC response: " << response << std::endl;
					}).detach();
				}
			}

			if (pollClock.elapsedMs() >= 250) {
				std::vector<HTN::SpokenLine> allLines = outputText.getAllOutputText();
				for (const auto& line : allLines) {
					if (line.slot >= 0 && !line.text.empty())
						speechQueue.push(line);
				}

				writeSpokenAck(ackConversation, ackSeq);
				writePlayerState(bounds, camera.getPos(), camera.getFlatForward());

				std::ifstream npcIn(npcStatePath);
				if (npcIn) {
					try {
						nlohmann::json state;
						npcIn >> state;
						if (state.contains("time")) {
								renderer.setDayFraction(state["time"].value("day_fraction", 0.25f));
							}
							if (state.contains("npcs")) {
							std::fill(npcActive.begin(), npcActive.end(), false);
							for (const auto& npc : state["npcs"]) {
								int slot = npc.value("slot", -1);
								if (slot < 0 || slot >= (int)renderer.faceCount()) continue;
								npcActive[slot] = true;
								float nx = npc.value("x", 0.5f);
								float nz = npc.value("z", 0.5f);
								HTN::f32 wx = bounds.toWorldX(nx);
								HTN::f32 wz = bounds.toWorldZ(nz);
								npcTargetRot[slot] = npc.value("rot", 0.0f);

								std::string name = npc.value("name", "");
								bool newNpc = prevNpcX[slot] < -9000.0f || name != npcNames[slot];
								npcNames[slot] = name;
								npcTargetPositions[slot] = {wx, 0.0f, wz};
								if (newNpc)
									npcWorldPositions[slot] = npcTargetPositions[slot];

								float dx = nx - prevNpcX[slot];
								float dz = nz - prevNpcZ[slot];
								npcMoving[slot] = (dx * dx + dz * dz) > 1e-8f;
								prevNpcX[slot] = nx;
								prevNpcZ[slot] = nz;
							}
						}
					} catch (const nlohmann::json::parse_error&) {}
				}

				pollClock.resetTime();
			}

			float dt = dtClock.elapsedMs() / 1000.0f;
			dtClock.resetTime();
			const float moveSpeed = 8.0f;
			const float turnSpeed = 5.0f;
			float t = turnSpeed * dt;
			if (t > 1.0f) t = 1.0f;
			float mt = moveSpeed * dt;
			if (mt > 1.0f) mt = 1.0f;

			for (HTN::u32 s = 0; s < renderer.faceCount(); s++) {
				if (!npcActive[s]) {
					renderer.setNPCTransform(s,
						enginemath::Mat4::translationM(0.0f, -1000.0f, 0.0f));
					prevNpcX[s] = -9999.0f;
					continue;
				}

				if (npcMoving[s])
					renderer.setNPCAnimState(s, HTN::AnimState::WALK);
				else if (renderer.getFace(s).isBusy())
					renderer.setNPCAnimState(s, HTN::AnimState::TALK);
				else
					renderer.setNPCAnimState(s, HTN::AnimState::IDLE);

				npcWorldPositions[s].x += (npcTargetPositions[s].x - npcWorldPositions[s].x) * mt;
				npcWorldPositions[s].z += (npcTargetPositions[s].z - npcWorldPositions[s].z) * mt;

				enginemath::Vec3 npcRayOrigin = {npcWorldPositions[s].x, 500.0f, npcWorldPositions[s].z};
				HTN::f32 npcFloor = HTN::Raycast::getGround(npcRayOrigin, renderer.getCollision(), renderer.getGroundGrid());
				if (npcFloor != -999.0f)
					npcWorldPositions[s].y = npcFloor;

				float goal = npcTargetRot[s];
				if (playerConversationActive && npcNames[s] == talkingToNPC) {
					enginemath::Vec3 cam = camera.getPos();
					enginemath::Vec3 npc = npcWorldPositions[s];
					goal = atan2f(cam.x - npc.x, cam.z - npc.z);
				}
				npcCurrentRot[s] = lerpAngle(npcCurrentRot[s], goal, t);
				renderer.setNPCTransform(s,
					enginemath::Mat4::translationM(npcWorldPositions[s].x, npcWorldPositions[s].y, npcWorldPositions[s].z)
					* enginemath::Mat4::rotateY(npcCurrentRot[s]));

				float dx = camera.getPos().x - npcWorldPositions[s].x;
				float dz = camera.getPos().z - npcWorldPositions[s].z;
				float dist = sqrtf(dx * dx + dz * dz);
				const float fullVolDist = 5.0f;
				const float fadeOutDist = 30.0f;
				float vol = 1.0f - std::clamp((dist - fullVolDist) / (fadeOutDist - fullVolDist), 0.0f, 1.0f);
				renderer.getFace(s).setVolume(vol * vol);
			}

			for (auto it = slotConversation.begin(); it != slotConversation.end();) {
				if (!renderer.getFace(it->first).isBusy()) {
					it = slotConversation.erase(it);
				} else {
					++it;
				}
			}

			{
				std::set<std::string> busyConvos;
				for (const auto& [slot, convo] : slotConversation)
					busyConvos.insert(convo);

				std::queue<HTN::SpokenLine> retry;
				while (!speechQueue.empty()) {
					HTN::SpokenLine next = speechQueue.front();
					speechQueue.pop();

					int slot = -1;
					for (HTN::u32 i = 0; i < renderer.faceCount(); i++) {
						if (npcNames[i] == next.speakerName) { slot = (int)i; break; }
					}
					if (slot >= 0 && !renderer.getFace(slot).isBusy() &&
						busyConvos.find(next.conversation) == busyConvos.end()) {
						renderer.getFace(slot).setVoice(voiceForGender(next.gender));
						renderer.getFace(slot).startSpeaking(next.text);
						slotConversation[slot] = next.conversation;
						busyConvos.insert(next.conversation);
						ackConversation = next.conversation;
						ackSeq = next.seq;
						writeSpokenAck(ackConversation, ackSeq);
					} else if (slot >= 0) {
						retry.push(next);
					}
				}
				std::swap(speechQueue, retry);
			}

			window.pollWindowEvents();
			renderer.drawFrame();
			frameCount++;
			if (fpsClock.elapsedMs() >= 1000) {
				std::string title = "Living Npc | " + std::to_string(frameCount) + " FPS";
				glfwSetWindowTitle(window.getWindow(), title.c_str());
				frameCount = 0;
				fpsClock.resetTime();
			}
		}

		if (playerConversationActive && !talkingToNPC.empty()) {
			std::string body = "{\"name\":\"" + talkingToNPC + "\"}";
			HTN::httpPost(SERVER_HOST, SERVER_PORT, "/api/conversation/end", body);
		}

		renderer.wait();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	return 0;
}
