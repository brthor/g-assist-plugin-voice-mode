import argparse
import sys
import time
from typing import List, Optional

import plugin


def ensure_initialized(skip_init: bool) -> None:
    if skip_init:
        return
    response = plugin.execute_initialize_command()
    if not response.get("success"):
        print("Initialization failed:", response)
        sys.exit(1)


def run_text_to_audio_pipeline(user_text: str) -> None:
    audio_bytes = plugin.synthesize_voice_with_gemini(user_text)
    # audio_bytes = plugin.synthesize_voice_streaming(user_text)
    if not audio_bytes:
        print("No audio returned from Gemini.")
        sys.exit(1)
    plugin.play_audio_bytes(audio_bytes)
    time.sleep(10)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test the Voice Mode text → Gemini → audio pipeline outside G-Assist."
    )
    parser.add_argument("--text", "-t", required=True, help="User input to synthesize.")
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Assume Voice Mode is already initialized.",
    )
    parser.add_argument(
        "--skip-shutdown",
        action="store_true",
        help="Leave Voice Mode initialized after running.",
    )
    args = parser.parse_args(argv)


    ensure_initialized(args.skip_init)
    print(plugin.execute_list_gemini_voice_options_command())
    try:
        run_text_to_audio_pipeline(args.text)
    finally:
        if not args.skip_shutdown:
            plugin.execute_shutdown_command()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
