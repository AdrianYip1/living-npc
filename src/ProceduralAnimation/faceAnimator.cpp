#include "faceAnimator.hpp"

#include "../secrets.hpp"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <iostream>
#include <algorithm>

HTN::faceAnim::faceAnim() {
	gen.seed(rd());

	for (size p = 0; p < 5; p++)
		for (size i = 0; i < eyeWeightChanges[p].size(); i++) {
			u32 idx = eyeWeightChanges[p][i].weightIndex;
			if (std::find(eyeIndices.begin(), eyeIndices.end(), idx) == eyeIndices.end())
				eyeIndices.push_back(idx);
		}
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

	u32 frameMs = eyeClock.elapsedMs();
	u32 dt = frameMs - lastFrameMs;
	lastFrameMs = frameMs;

	f32 alpha = 1.0f - std::exp(-(f32)dt / 40.0f);

	if (!started) {
		exponentialSmoothing(std::vector<f32>(MAX_WEIGHTS, 0.0f), alpha);
		applyEyeMovementWeights(frameMs, alpha);
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
		applyEyeMovementWeights(frameMs, alpha);
		blinking(frameMs);
		return displayedWeights;
	}

	if (current + 1 >= entries.size()) {
		exponentialSmoothing(weightsForViseme(getViseme(current)), alpha);
		applyBilabialDominance(sampleMs, current);
		applyLipHeavyTiming(sampleMs, current);
		applyEyeMovementWeights(frameMs, alpha);
		blinking(frameMs);
		return displayedWeights;
	}

	std::vector<f32> srcWeights = weightsForViseme(getViseme(current));
	std::vector<f32> dstWeights = weightsForViseme(getViseme(current + 1));
	u32 transitionTime = getAudioOffset(current + 1) - getAudioOffset(current);
	u32 timeElapsed = sampleMs - getAudioOffset(current);

	exponentialSmoothing(weightChange(srcWeights, dstWeights, transitionTime, timeElapsed), alpha);

	applyBilabialDominance(sampleMs, current);
	applyLipHeavyTiming(sampleMs, current);
	applyEyeMovementWeights(frameMs, alpha);
	blinking(frameMs);

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

void HTN::faceAnim::applyBilabialDominance(u32 nowMs, size current) {
	const f32 onsetMs = 80.0f;
	const f32 decayMs = 120.0f;

	u32 apex;
	if (getViseme(current) == 21) {
		apex = getAudioOffset(current);
	} else if (current + 1 < entries.size() && getViseme(current + 1) == 21) {
		apex = getAudioOffset(current + 1);
	} else {
		return;
	}

	i32 dtApex = (i32)nowMs - (i32)apex;
	f32 d;

	if (dtApex < 0) {
		d = 1.0f + (f32)dtApex / onsetMs;
	}
	else {
		d = 1.0f - (f32)dtApex / decayMs;
	}

	d = std::clamp(d, 0.0f, 0.65f);

	std::vector<f32> closed = weightsForViseme(21);
	for (size i = 0; i < MAX_WEIGHTS; i++)
		displayedWeights[i] += d * (closed[i] - displayedWeights[i]);
}

void HTN::faceAnim::applyLipHeavyTiming(u32 nowMs, size current) {
	const f32 onsetMs = 150.0f;
	const f32 decayMs = 150.0f;

	size lo = (current >= 3) ? current - 3 : 0;
	size hi = min(current + 3, entries.size() - 1);
	for (size i = lo; i <= hi; i++) {
		u32 v = getViseme(i);
		if (v != 7 && v != 8 && v != 10 && v != 16) continue;

		i32 dtApex = (i32)nowMs - (i32)getAudioOffset(i);

		f32 d = (dtApex > 0) ? 1 - (f32)dtApex / decayMs :
							   1 + (f32)dtApex / onsetMs;

		d = std::clamp(d, 0.0f, 1.0f);
		if (d <= 0.0f) continue;

		std::vector<f32> pose = weightsForViseme(getViseme(i));
		for (size j = 0; j < MAX_WEIGHTS; j++) {
			displayedWeights[j] += d * (pose[j] - displayedWeights[j]);
		}
	}
}

void HTN::faceAnim::applyEyeMovementWeights(u32 nowMs, f32 alpha) {
	if (nowMs >= nextTransitionTimeMs) {
		std::uniform_real_distribution<> distr(0, 1);
		if (eyeState == "FOCUS") {
			f32 chance = isBusy() ? 0.6f : 0.3f;
			if (distr(gen) < chance) {
				eyeState = "AVERT";
				pickAvertEyeLocation(1.0f);
				nextTransitionTimeMs = avertWaitTime() + nowMs;
			}
			else {
				focusLocation();
				nextTransitionTimeMs = focusWaitTime() + nowMs;
			}
		}
		else {
			eyeState = "FOCUS";
			focusLocation();
			nextTransitionTimeMs = focusWaitTime() + nowMs;
		}
	}
	if (nowMs >= cascadeTransitionTime && eyeState == "FOCUS") {
		focusLocation();
		std::uniform_real_distribution<> distr(0, 2);
		pickAvertEyeLocation(0.05f * distr(gen));
		cascadeTransitionTime = nowMs + 300.0f * distr(gen);
	}

	for (u32 idx : eyeIndices) {
		currentEyeWeights[idx] += alpha * (targetEyeWeights[idx] - currentEyeWeights[idx]);
		displayedWeights[idx] = currentEyeWeights[idx];
	}
}

HTN::u32 HTN::faceAnim::focusWaitTime() {
	std::uniform_real_distribution<> distr(1.0f, 2.0f);
	return distr(gen) * 1000;
}

HTN::u32 HTN::faceAnim::avertWaitTime() {
	std::uniform_real_distribution<> distr(0.7f, 1.6f);
	return distr(gen) * 1000;
}

void HTN::faceAnim::pickAvertEyeLocation(f32 scale) {
	for (u32 idx : eyeIndices) targetEyeWeights[idx] = 0.0f;

	std::uniform_real_distribution<> distr(0.0f, 1.0f);
	std::uniform_real_distribution<> amount(0.6f, 1.0f);

	float yAmt = amount(gen);
	size yPose = distr(gen) > 0.5f ? 1 : 2;
	for (size i = 0; i < eyeWeightChanges[yPose].size(); i++)
		targetEyeWeights[eyeWeightChanges[yPose][i].weightIndex] = yAmt * eyeWeightChanges[yPose][i].valueWanted * scale;

	float xAmt = amount(gen);
	size xPose = distr(gen) > 0.5f ? 3 : 4;
	for (size i = 0; i < eyeWeightChanges[xPose].size(); i++)
		targetEyeWeights[eyeWeightChanges[xPose][i].weightIndex] = xAmt * eyeWeightChanges[xPose][i].valueWanted * scale;
}

void HTN::faceAnim::focusLocation() {
	for (u32 idx : eyeIndices) targetEyeWeights[idx] = 0.0f;
	for (size i = 0; i < eyeWeightChanges[0].size(); i++) {
		targetEyeWeights[eyeWeightChanges[0][i].weightIndex] = eyeWeightChanges[0][i].valueWanted;
	}
}

void HTN::faceAnim::blinking(u32 frameMs) {
	if (!isBlinking && frameMs >= nextBlinkMs) {
		isBlinking = true;
		blinkTimer = frameMs;
	}
	if (!isBlinking) return;

	const f32 onsetMs = 50.0f;
	const f32 decayMs = 50.0f;

	f32 elapsed = (f32)(frameMs - blinkTimer);
	f32 d = (elapsed < 50.0f) ? elapsed / 50.0f : (100.0f - elapsed) / 50.0f;
	d = std::clamp(d, 0.0f, 1.0f) * 0.75f;

	for (size i = 0; i < blinkWeightChanges.size(); i++)
		displayedWeights[blinkWeightChanges[i].weightIndex] = d * blinkWeightChanges[i].valueWanted;

	if (elapsed >= 80.0f) {
		isBlinking = false;
		std::uniform_real_distribution<> gap(500.0f, 2000.0f);
		nextBlinkMs = frameMs + (u32)gap(gen);
	}
}

void HTN::faceAnim::applyEyebrowMovementWeights(u32 nowMs, size current) {
}
