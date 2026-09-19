#include "faceAnimator.hpp"

#include "../secrets.hpp"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <iostream>
#include <algorithm>

HTN::faceAnim::faceAnim() {
}

HTN::faceAnim::~faceAnim() {
	if (speakThread.joinable()) speakThread.join();
	if (device) ma_device_uninit(device.get());
	if (decoder) ma_decoder_uninit(decoder.get());
}

void HTN::faceAnim::pushViseme(u32 ID, u32 offset) {
	std::lock_guard<std::mutex> lock(mtx);
	visemeEntry entry{ ID, offset };
	entries.push_back(entry);
}

std::vector<HTN::f32> HTN::faceAnim::sample() {
	std::lock_guard<std::mutex> lock(mtx);

	u32 frameMs = clock.elapsedMs();
	u32 dt = frameMs - lastFrameMs;
	lastFrameMs = frameMs;

	f32 alpha = 1.0f - std::exp(-(f32)dt / 40.0f);

	if (!started) {
		exponentialSmoothing(std::vector<f32>(MAX_WEIGHTS, 0.0f), alpha);
		return displayedWeights;
	}

	u32 nowMs = clock.elapsedMs();
	u32 leadMsOffset = 60;
	u32 sampleMs = nowMs + leadMsOffset;

	size current = 0;
	bool found = false;
	for (size i = 0; i < entries.size(); i++) {
		if (getAudioOffset(i) <= sampleMs) {
			current = i;
			found = true;
		}
	}

	if (!found) {
		exponentialSmoothing(std::vector<f32>(MAX_WEIGHTS, 0.0f), alpha);
		return displayedWeights;
	}

	if (current + 1 >= entries.size()) {
		exponentialSmoothing(weightsForViseme(getViseme(current)), alpha);
		return displayedWeights;
	}

	std::vector<f32> srcWeights = weightsForViseme(getViseme(current));
	std::vector<f32> dstWeights = weightsForViseme(getViseme(current + 1));
	u32 transitionTime = getAudioOffset(current + 1) - getAudioOffset(current);
	u32 timeElapsed = sampleMs - getAudioOffset(current);

	exponentialSmoothing(weightChange(srcWeights, dstWeights, transitionTime, timeElapsed), alpha);

	return displayedWeights;
}

void HTN::faceAnim::setupSpeech() {
	using namespace Microsoft::CognitiveServices::Speech;

	auto speechConfig = SpeechConfig::FromSubscription(SPEECH_KEY, SPEECH_REGION);
	speechConfig->SetSpeechSynthesisLanguage("en-US");
	speechConfig->SetSpeechSynthesisVoiceName("en-US-Ava:DragonHDLatestNeural");

	synth = SpeechSynthesizer::FromConfig(speechConfig, nullptr);
	synth->VisemeReceived += [this](const SpeechSynthesisVisemeEventArgs& e) {
		auto offset = e.AudioOffset / 10000;
		auto viseme = e.VisemeId;
		std::cout << offset << " " << viseme << std::endl;
		pushViseme((u32)viseme, (u32)offset);
	};
}

bool HTN::faceAnim::isBusy() {
	return busy;
}

void HTN::faceAnim::feedAudio(void* output, u32 frameCount) {
	ma_uint64 framesRead = 0;
	if (decoder) ma_decoder_read_pcm_frames(decoder.get(), output, frameCount, &framesRead);
	if (framesRead < frameCount) busy = false;
}

static void audioCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
	HTN::faceAnim* self = (HTN::faceAnim*)pDevice->pUserData;
	if (self != nullptr) self->feedAudio(pOutput, frameCount);
	(void)pInput;
}

void HTN::faceAnim::startSpeaking(const std::string& text) {
	busy = true;

	if (speakThread.joinable()) speakThread.join();
	if (device) { ma_device_uninit(device.get()); device.reset(); }
	if (decoder) { ma_decoder_uninit(decoder.get()); decoder.reset(); }
	{
		std::lock_guard<std::mutex> lock(mtx);
		entries.clear();
		started = false;
	}

	speakThread = std::thread([this, text] {
		auto result = synth->SpeakSsmlAsync(getSSML(text)).get();
		audioData = result->GetAudioData();

		decoder = std::make_unique<ma_decoder>();
		if (ma_decoder_init_memory(audioData->data(), audioData->size(), nullptr, decoder.get()) != MA_SUCCESS) {
			std::cerr << "faceAnim: failed to init audio decoder" << std::endl;
			busy = false;
			return;
		}

		ma_device_config config = ma_device_config_init(ma_device_type_playback);
		config.playback.format = decoder->outputFormat;
		config.playback.channels = decoder->outputChannels;
		config.sampleRate = decoder->outputSampleRate;
		config.dataCallback = audioCallback;
		config.pUserData = this;

		device = std::make_unique<ma_device>();
		if (ma_device_init(nullptr, &config, device.get()) != MA_SUCCESS) {
			std::cerr << "faceAnim: failed to init audio device" << std::endl;
			busy = false;
			return;
		}

		{
			std::lock_guard<std::mutex> lock(mtx);
			clock.resetTime();
			started = true;
		}
		ma_device_start(device.get());
	});
}

std::string HTN::faceAnim::getSSML(const std::string& input) {
	std::string begin = "<speak version = \"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" xml:lang=\"en-US\">";
	std::string voiceTag = "<voice name=\"" + voiceName + "\">";

	return begin + voiceTag + input + "</voice>" + "</speak>";
}

std::vector<HTN::f32> HTN::faceAnim::weightsForViseme(u32 id) {
	std::vector<f32> newWeight = std::vector<f32>(MAX_WEIGHTS, 0.0f);

	if (id < 22) {
		for (size i = 0; i < weightChanges[id].size(); i++) {
			newWeight[weightChanges[id][i].weightIndex] = weightChanges[id][i].valueWanted;
		}
	}

	return newWeight;
}

std::vector<HTN::f32> HTN::faceAnim::weightChange(const std::vector<f32>& oldW, const std::vector<f32>& finalW, u32 transitionTime, u32 timeElapsed) {
	if (timeElapsed >= transitionTime) return finalW;

	f32 rawT = (f32)timeElapsed / transitionTime;
	f32 holdRatio = 0.6f;
	f32 t = (rawT < holdRatio) ? 0.0f : (rawT - holdRatio) / (1.0f - holdRatio);
	t = t * t * (3.0f - 2.0f * t);

	std::vector<f32> result(MAX_WEIGHTS, 0.0f);
	for (size i = 0; i < result.size(); i++) {
		result[i] = oldW[i] + t * (finalW[i] - oldW[i]);
	}
	return result;
}

void HTN::faceAnim::exponentialSmoothing(std::vector<f32>& newWeights, f32 alpha) {
	for (size i = 0; i < MAX_WEIGHTS; i++) {
		displayedWeights[i] += alpha * (newWeights[i] - displayedWeights[i]);
	}
}
