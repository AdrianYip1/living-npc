#include "speechListener.hpp"
#include "secrets.hpp"

#include <iostream>

HTN::SpeechListener::SpeechListener() {
	using namespace Microsoft::CognitiveServices::Speech;
	using namespace Microsoft::CognitiveServices::Speech::Audio;

	auto config = SpeechConfig::FromSubscription(SPEECH_KEY, SPEECH_REGION);
	config->SetSpeechRecognitionLanguage("en-US");

	auto audioConfig = AudioConfig::FromDefaultMicrophoneInput();
	recognizer = SpeechRecognizer::FromConfig(config, audioConfig);

	recognizer->Recognized += [this](const SpeechRecognitionEventArgs& e) {
		if (e.Result->Reason == ResultReason::RecognizedSpeech) {
			std::lock_guard<std::mutex> lock(mtx);
			if (!recognizedText.empty()) recognizedText += " ";
			recognizedText += e.Result->Text;
		}
	};
}

HTN::SpeechListener::~SpeechListener() {
	if (listening) {
		recognizer->StopContinuousRecognitionAsync().get();
	}
}

void HTN::SpeechListener::startListening() {
	if (listening) return;
	{
		std::lock_guard<std::mutex> lock(mtx);
		recognizedText.clear();
	}
	resultReady = false;
	listening = true;
	recognizer->StartContinuousRecognitionAsync().get();
}

void HTN::SpeechListener::stopListening() {
	if (!listening) return;
	recognizer->StopContinuousRecognitionAsync().get();
	listening = false;
	resultReady = true;
}

std::string HTN::SpeechListener::takeResult() {
	std::lock_guard<std::mutex> lock(mtx);
	resultReady = false;
	std::string result = std::move(recognizedText);
	recognizedText.clear();
	return result;
}
