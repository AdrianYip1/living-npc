#pragma once

#include "defines.hpp"

#include <speechapi_cxx.h>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <functional>

namespace HTN {
	class SpeechListener {
	public:
		SpeechListener();
		~SpeechListener();
		SpeechListener(const SpeechListener&) = delete;
		SpeechListener& operator=(const SpeechListener&) = delete;

		void startListening();
		void stopListening();
		bool isListening() const { return listening; }
		bool hasResult() const { return resultReady; }
		std::string takeResult();

	private:
		std::shared_ptr<Microsoft::CognitiveServices::Speech::SpeechRecognizer> recognizer;
		std::atomic<bool> listening{false};
		std::atomic<bool> resultReady{false};
		std::mutex mtx;
		std::string recognizedText;
	};
} // namespace HTN
