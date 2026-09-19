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
#include <iostream>
#include <fstream>
#include <filesystem>
#include <queue>
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
					   const std::vector<enginemath::Vec3>& npcPositions) {
		int nearest = -1;
		float minDist = 999999.0f;
		for (int i = 0; i < (int)npcPositions.size(); i++) {
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

		HTN::Window window(1600, 900, "Living Npc: Face Model");
		HTN::Camera camera(enginemath::Vec3(0.0f, 1.56f, 0.5f),
						   1600.0f / 900.0f);
		HTN::Renderer renderer(window, camera);
		HTN::Clock clock;
		HTN::Input input(window);
		HTN::CameraControls controls(clock, window, input, camera);
		HTN::SpeechListener listener;

		HTN::Clock pollClock;
		pollClock.resetTime();
		HTN::outputParser outputText(CONVO_LOG_DIR);
		std::queue<HTN::SpokenLine> speechQueue;
		std::string npcStatePath = std::string(CONVO_LOG_DIR) + "/npc_state.json";
		const HTN::WorldBounds& bounds = renderer.getWorldBounds();

		bool inFlight = false;
		std::string inFlightConversation;
		int inFlightSeq = -1;
		std::string ackConversation;
		int ackSeq = -1;

		std::vector<float> prevNpcX(renderer.faceCount(), -9999.0f);
		std::vector<float> prevNpcZ(renderer.faceCount(), -9999.0f);
		std::vector<bool> npcMoving(renderer.faceCount(), false);
		std::vector<std::string> npcNames(renderer.faceCount());
		std::vector<enginemath::Vec3> npcWorldPositions(renderer.faceCount());
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

			HTN::f32 floor = HTN::Raycast::getGround(camera.getPos(), renderer.getCollision(), renderer.getGroundGrid());
			if (floor != -999.0f) {
				enginemath::Vec3 pos = camera.getPos();
				pos.y = floor + 3.3f;
				camera.setPos(pos);
			}

			bool talkKeyDown = input.keyPressed(GLFW_KEY_T);
			if (talkKeyDown && !talkKeyWasDown) {
				int nearest = findNearestNPC(camera.getPos(), npcWorldPositions);
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
				HTN::SpokenLine line = outputText.getOutputText();

				if (outputText.startedNewConversation()) {
					std::queue<HTN::SpokenLine> empty;
					std::swap(speechQueue, empty);
				}

				if (line.slot >= 0 && !line.text.empty()) {
					speechQueue.push(line);
				}

				writeSpokenAck(ackConversation, ackSeq);

				std::ifstream npcIn(npcStatePath);
				if (npcIn) {
					try {
						nlohmann::json state;
						npcIn >> state;
						if (state.contains("npcs")) {
							for (const auto& npc : state["npcs"]) {
								int slot = npc.value("slot", -1);
								if (slot < 0 || slot >= (int)renderer.faceCount()) continue;
								float nx = npc.value("x", 0.5f);
								float nz = npc.value("z", 0.5f);
								HTN::f32 wx = bounds.toWorldX(nx);
								HTN::f32 wz = bounds.toWorldZ(nz);
								npcTargetRot[slot] = npc.value("rot", 0.0f);

								npcNames[slot] = npc.value("name", "");
								npcWorldPositions[slot] = {wx, 0.0f, wz};

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
			const float turnSpeed = 5.0f;
			float t = turnSpeed * dt;
			if (t > 1.0f) t = 1.0f;

			for (HTN::u32 s = 0; s < renderer.faceCount(); s++) {
				if (npcMoving[s])
					renderer.setNPCAnimState(s, HTN::AnimState::WALK);
				else if (renderer.getFace(s).isBusy())
					renderer.setNPCAnimState(s, HTN::AnimState::TALK);
				else
					renderer.setNPCAnimState(s, HTN::AnimState::IDLE);

				float goal = npcTargetRot[s];
				if (playerConversationActive && npcNames[s] == talkingToNPC) {
					enginemath::Vec3 cam = camera.getPos();
					enginemath::Vec3 npc = npcWorldPositions[s];
					goal = atan2f(cam.x - npc.x, cam.z - npc.z);
				}
				npcCurrentRot[s] = lerpAngle(npcCurrentRot[s], goal, t);
				renderer.setNPCTransform(s,
					enginemath::Mat4::translationM(npcWorldPositions[s].x, 0.0f, npcWorldPositions[s].z)
					* enginemath::Mat4::rotateY(npcCurrentRot[s]));
			}

			if (inFlight && !renderer.anyBusy()) {
				ackConversation = inFlightConversation;
				ackSeq = inFlightSeq;
				inFlight = false;
				writeSpokenAck(ackConversation, ackSeq);
			}

			if (!renderer.anyBusy() && !speechQueue.empty()) {
				HTN::SpokenLine next = speechQueue.front();
				speechQueue.pop();
				int slot = -1;
				for (HTN::u32 i = 0; i < renderer.faceCount(); i++) {
					if (npcNames[i] == next.speakerName) { slot = (int)i; break; }
				}
				if (slot >= 0) {
					renderer.getFace(slot).setVoice(voiceForGender(next.gender));
					renderer.getFace(slot).startSpeaking(next.text);
				}
				inFlight = true;
				inFlightConversation = next.conversation;
				inFlightSeq = next.seq;
			}

			window.pollWindowEvents();
			renderer.drawFrame();
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
