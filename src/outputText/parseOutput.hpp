#pragma once

#include "../defines.hpp"

#include <string>
#include <fstream>
#include <vector>

namespace HTN {
	struct SpokenLine {
		int slot = -1;
		std::string text;
		std::string gender;
		int seq = -1;
		std::string conversation;
	};

	class outputParser {
	public:
		outputParser(const std::string& dirPath);
		~outputParser();
		outputParser(const outputParser& other) = delete;
		outputParser& operator=(const outputParser& other) = delete;

		SpokenLine getOutputText();
		bool startedNewConversation();
		int activeSpeakerCount() const { return (int)speakers.size(); }

	private:
		std::string readActivePointer(std::vector<std::string>& speakersOut);

		std::string dirPath;
		std::string currentFile;
		std::string pointerName;
		int pointerSeq = 0;
		std::vector<std::string> speakers;
		int turnIndex = 0;
		bool switched = false;
	};
} // namespace HTN
