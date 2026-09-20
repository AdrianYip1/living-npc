#include "serverLauncher.hpp"
#include <iostream>

HTN::ServerLauncher::~ServerLauncher() {
	stop();
}

bool HTN::ServerLauncher::start(const std::string& pythonExe,
								const std::string& srcDir, int port) {
	jobHandle = CreateJobObjectA(nullptr, nullptr);
	if (jobHandle) {
		JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
		jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
		SetInformationJobObject(jobHandle, JobObjectExtendedLimitInformation,
								&jeli, sizeof(jeli));
	}

	// Convert Windows path to WSL path for the working directory
	std::string wslSrcDir = srcDir;
	if (wslSrcDir.size() >= 2 && wslSrcDir[1] == ':') {
		char drive = (char)tolower((unsigned char)wslSrcDir[0]);
		wslSrcDir = "/mnt/" + std::string(1, drive) + wslSrcDir.substr(2);
	}
	for (char& c : wslSrcDir) if (c == '\\') c = '/';

	std::string cmd = "wsl bash -c \"cd '" + wslSrcDir +
		"' && " + pythonExe + " -m mini_map.game --port " +
		std::to_string(port) + "\"";

	STARTUPINFOA si{};
	si.cb = sizeof(si);
	PROCESS_INFORMATION pi{};

	BOOL ok = CreateProcessA(
		nullptr,
		const_cast<char*>(cmd.c_str()),
		nullptr, nullptr, FALSE,
		CREATE_NO_WINDOW,
		nullptr,
		nullptr,
		&si, &pi);

	if (!ok) {
		std::cerr << "[ServerLauncher] Failed to start Python server (error "
				  << GetLastError() << ")" << std::endl;
		return false;
	}

	processHandle = pi.hProcess;
	if (jobHandle)
		AssignProcessToJobObject(jobHandle, processHandle);
	CloseHandle(pi.hThread);

	std::cout << "[ServerLauncher] Python server started (PID " << pi.dwProcessId << ")" << std::endl;
	return true;
}

void HTN::ServerLauncher::stop() {
	if (processHandle) {
		TerminateProcess(processHandle, 0);
		WaitForSingleObject(processHandle, 3000);
		CloseHandle(processHandle);
		processHandle = nullptr;
	}
	if (jobHandle) {
		CloseHandle(jobHandle);
		jobHandle = nullptr;
	}
}

bool HTN::ServerLauncher::isRunning() const {
	if (!processHandle) return false;
	DWORD exitCode;
	GetExitCodeProcess(processHandle, &exitCode);
	return exitCode == STILL_ACTIVE;
}
