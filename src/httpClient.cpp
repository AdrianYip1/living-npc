#include "httpClient.hpp"

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#include <sstream>
#include <iostream>

namespace {
	struct WsaInit {
		WsaInit() { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); }
		~WsaInit() { WSACleanup(); }
	};
	static WsaInit wsaInit;
}

std::string HTN::httpPost(const std::string& host, int port,
						  const std::string& path, const std::string& jsonBody) {
	struct addrinfo hints{}, *result = nullptr;
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;

	if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &result) != 0)
		return "";

	SOCKET sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
	if (sock == INVALID_SOCKET) { freeaddrinfo(result); return ""; }

	if (connect(sock, result->ai_addr, (int)result->ai_addrlen) == SOCKET_ERROR) {
		closesocket(sock);
		freeaddrinfo(result);
		return "";
	}
	freeaddrinfo(result);

	std::ostringstream req;
	req << "POST " << path << " HTTP/1.1\r\n"
		<< "Host: " << host << ":" << port << "\r\n"
		<< "Content-Type: application/json\r\n"
		<< "Content-Length: " << jsonBody.size() << "\r\n"
		<< "Connection: close\r\n\r\n"
		<< jsonBody;

	std::string request = req.str();
	send(sock, request.c_str(), (int)request.size(), 0);

	std::string response;
	char buf[1024];
	int n;
	while ((n = recv(sock, buf, sizeof(buf) - 1, 0)) > 0) {
		buf[n] = '\0';
		response += buf;
	}

	closesocket(sock);

	auto bodyPos = response.find("\r\n\r\n");
	if (bodyPos != std::string::npos)
		return response.substr(bodyPos + 4);
	return response;
}
