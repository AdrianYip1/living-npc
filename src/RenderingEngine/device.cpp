#include "device.hpp"

namespace HTN {

Device::Device(Window& window) : window(window) {
	createInstance();
}

Device::~Device() {}

void Device::createInstance() {}

} // namespace HTN
