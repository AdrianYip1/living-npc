#pragma once

#include "../Platform/window.hpp"
#include "../../defines.hpp"

namespace HTN {
	class Input {
	public:
		Input(Window& _window);
		~Input() = default;
		Input(const Input&) = delete;
		Input& operator=(const Input&) = delete;

		bool keyPressed(int key);
		bool mousePressed(int button);
	private:
		Window& window;
	};
} // namespace HTN
