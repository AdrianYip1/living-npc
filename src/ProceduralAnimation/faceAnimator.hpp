#pragma once
#include "enginemath/vec3.hpp"
#include "enginemath/vec4.hpp"
#include "enginemath/mat4.hpp"

#include "../defines.hpp"
#include "../clock.hpp"

#include "visemeDataTable.hpp"

// std
#include <utility>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>
#include <memory>
#include <cstdint>
#include <random>

#include <speechapi_cxx.h>

struct ma_device;
struct ma_decoder;

namespace HTN {
	struct visemeEntry {
		u32 visemeID;
		u32 audioOffset;
	};

	class faceAnim {
	public:
		faceAnim();
		~faceAnim();
		faceAnim(const faceAnim& other) = delete;
		faceAnim& operator=(const faceAnim& other) = delete;

		std::vector<f32> sample();
		std::vector<f32> weightsForViseme(u32 id);
		std::vector<f32> weightChange(const std::vector<f32>& oldW, const std::vector<f32>& finalW, u32 transitionTime, u32 timeElapsed);

		void pushViseme(u32 ID, u32 offset);
		size entriesSize() { return entries.size(); };

		u32 getViseme(size entryIndex) { return entries[entryIndex].visemeID; };
		u32 getAudioOffset(size entryIndex) { return entries[entryIndex].audioOffset; };

		void setupSpeech();
		void startSpeaking(const std::string& text);
		void setVoice(const std::string& v) { voiceName = v; };
		void setVolume(f32 vol);

		bool isBusy();
		void feedAudio(void* output, u32 frameCount);
	private:
		Clock clock;
		Clock eyeClock;
		bool started = false;
		std::string voiceName = "en-US-Ava:DragonHDLatestNeural";

		u32 lastFrameMs = 0;
		std::vector<f32> displayedWeights = std::vector<f32>(MAX_WEIGHTS, 0.0f);

		std::string getSSML(const std::string& input);
		void exponentialSmoothing(std::vector<f32>& newWeights, f32 alpha);
		void applyBilabialDominance(u32 nowMs, size current);
		void applyLipHeavyTiming(u32 nowMs, size current);
		void applyEyeMovementWeights(u32 nowMs, f32 alpha);
		void applyEyebrowMovementWeights(u32 nowMs, size current);

		std::vector<visemeEntry> entries;
		std::shared_ptr<Microsoft::CognitiveServices::Speech::SpeechSynthesizer> synth;

		std::shared_ptr<std::vector<uint8_t>> audioData;
		std::unique_ptr<ma_decoder> decoder;
		std::unique_ptr<ma_device> device;
		std::atomic<bool> busy{false};

		std::thread speakThread;
		std::mutex mtx;

		std::random_device rd;
		std::mt19937 gen;
		std::string eyeState = "FOCUS";
		bool isBlinking = false;
		f32 blinkTimer = 0.0f;
		f32 nextBlinkMs = 1000.0f;
		u32 nextTransitionTimeMs = 0;
		u32 cascadeTransitionTime = 0;

		std::vector<f32> targetEyeWeights = std::vector<f32>(MAX_WEIGHTS, 0.0f);
		std::vector<f32> currentEyeWeights = std::vector<f32>(MAX_WEIGHTS, 0.0f);
		std::vector<u32> eyeIndices;
		u32 focusWaitTime();
		u32 avertWaitTime();
		void pickAvertEyeLocation(f32 scale);
		void focusLocation();
		void blinking(u32 frameMs);
	};
} // namespace HTN
