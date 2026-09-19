#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>

namespace HTN {
	class ServerLauncher {
	public:
		ServerLauncher() = default;
		~ServerLauncher();
		ServerLauncher(const ServerLauncher&) = delete;
		ServerLauncher& operator=(const ServerLauncher&) = delete;

		bool start(const std::string& pythonExe = "python",
				   const std::string& srcDir = "src",
				   int port = 8765);
		void stop();
		bool isRunning() const;

	private:
		HANDLE processHandle = nullptr;
		HANDLE jobHandle = nullptr;
	};
} // namespace HTN
