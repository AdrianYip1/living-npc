"""Manual smoke test. Run with `python -m environment_agent` from src/ to
watch the deterministic tick loop: each Enter prints the next EnvironmentEvent.
"""
from __future__ import annotations

from .agent import EnvironmentAgent


def main() -> None:
    env = EnvironmentAgent()
    print(f"Starting weather: {env.weather.value}. Press Enter to tick, Ctrl+C to quit.")
    try:
        while True:
            input()
            event = env.tick()
            line = (
                f"clock: {event.clock} ({event.time_of_day.value}) | weather: {event.weather.value}"
                f" | temperature: {event.temperature}F"
            ) + (" (changed)" if event.weather_changed else "")
            if event.travelers_arrived:
                for traveler in event.travelers_arrived:
                    line += f" | traveler arrived: {traveler.name}, {traveler.origin}, {traveler.reason} ({', '.join(traveler.traits)})"
            print(line)
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
