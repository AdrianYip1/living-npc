#include "parseOutput.hpp"

#include "json.hpp"

#include <filesystem>

namespace fs = std::filesystem;

HTN::outputParser::outputParser(const std::string& _dirPath) :
	dirPath(_dirPath) {

}

HTN::outputParser::~outputParser() {

}

std::string HTN::outputParser::readActivePointer(std::vector<std::string>& speakersOut) {
	speakersOut.clear();
	pointerName = "";
	pointerSeq = 0;

	std::ifstream in(dirPath + "/active.json");
	if (!in) return "";

	nlohmann::json obj;
	try {
		in >> obj;
	}
	catch (const nlohmann::json::parse_error&) {
		return "";
	}

	if (!obj.contains("conversation") || obj["conversation"].is_null()) return "";
	std::string conversation = obj.value("conversation", "");
	if (conversation.empty()) return "";

	if (obj.contains("speakers")) {
		for (const auto& s : obj["speakers"]) speakersOut.push_back(s.get<std::string>());
	}
	pointerName = conversation;
	pointerSeq = obj.value("seq", 0);
	return dirPath + "/" + conversation;
}

bool HTN::outputParser::startedNewConversation() {
	bool result = switched;
	switched = false;
	return result;
}

HTN::SpokenLine HTN::outputParser::getOutputText() {
	std::vector<std::string> newSpeakers;
	std::string relevant = readActivePointer(newSpeakers);

	speakers = newSpeakers;
	if (relevant.empty()) return SpokenLine{};

	if (relevant != currentFile) {
		currentFile = relevant;
		turnIndex = pointerSeq;
		switched = true;
	}

	std::ifstream in(currentFile);
	if (!in) return SpokenLine{};

	nlohmann::json obj;
	do {
		try {
			in >> obj;
		}
		catch (const nlohmann::json::parse_error&) {
			return SpokenLine{};
		}
		if (obj.value("phase", "") == "end") {
			return SpokenLine{};
		}
	} while (obj.value("seq", -1) < turnIndex);

	std::string speaker = obj.value("speaker_id", "");

	int slot = -1;
	for (size i = 0; i < speakers.size(); i++) {
		if (speakers[i] == speaker) {
			slot = (int)i;
			break;
		}
	}
	if (slot < 0) {
		turnIndex = obj.value("seq", -1) + 1;
		return SpokenLine{};
	}

	if (obj.value("text", "") != "") {
		turnIndex = obj.value("seq", -1) + 1;
		SpokenLine line;
		line.slot = slot;
		line.speakerName = speaker;
		line.text = obj.value("text", "");
		line.gender = obj.value("gender", "");
		line.seq = obj.value("seq", -1);
		line.conversation = pointerName;
		return line;
	}
	return SpokenLine{};
}
