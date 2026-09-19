#pragma once

#include <string>

namespace HTN {
	std::string httpPost(const std::string& host, int port,
						 const std::string& path, const std::string& jsonBody);
} // namespace HTN
