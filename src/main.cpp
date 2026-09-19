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
#include "json.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <queue>
#include <cctype>

namespace {
	const std::string VOICE_FEMALE = "en-US-Ava:DragonHDLatestNeural";
	const std::string VOICE_MALE = "en-US-Andrew:DragonHDLatestNeural";

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
}

int main() {
	try {
		HTN::Window window(1600, 900, "Living Npc: Face Model");
		HTN::Camera camera(enginemath::Vec3(0.0f, 1.56f, 0.5f),
						   1600.0f / 900.0f);
		HTN::Renderer renderer(window, camera);
		HTN::Clock clock;
		HTN::Input input(window);
		HTN::CameraControls controls(clock, window, input, camera);

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
								HTN::f32 rot = npc.value("rot", 0.0f);
								renderer.setNPCTransform(slot,
									enginemath::Mat4::translationM(wx, 0.0f, wz)
									* enginemath::Mat4::rotateY(rot));

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

			for (HTN::u32 s = 0; s < renderer.faceCount(); s++) {
				if (npcMoving[s])
					renderer.setNPCAnimState(s, HTN::AnimState::WALK);
				else if (renderer.getFace(s).isBusy())
					renderer.setNPCAnimState(s, HTN::AnimState::TALK);
				else
					renderer.setNPCAnimState(s, HTN::AnimState::IDLE);
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
				HTN::u32 slot = static_cast<HTN::u32>(next.slot);
				if (slot < renderer.faceCount()) {
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

		renderer.wait();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	return 0;
}
