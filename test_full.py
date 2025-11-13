import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

import plugin


def ensure_initialized(skip_init: bool) -> None:
    if skip_init:
        return
    response = plugin.execute_initialize_command()
    if not response.get("success"):
        print("Initialization failed:", response)
        sys.exit(1)


def build_context(user_text: str, prior_messages: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    context = prior_messages[:] if prior_messages else []
    context.append({"role": "user", "content": user_text})
    return context


def parse_history(history_json: Optional[str]) -> Optional[List[Dict[str, str]]]:
    if not history_json:
        return None
    try:
        history = json.loads(history_json)
        if isinstance(history, list):
            return history
    except json.JSONDecodeError as exc:
        print("Failed to parse history JSON:", exc)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exercise the full Voice Mode pipeline (Gemini planning, optional tool use, TTS)."
    )
    parser.add_argument("--text", "-t", required=True, help="User utterance to send to Voice Mode.")
    parser.add_argument(
        "--system-info",
        help="Optional JSON string describing system info to include in the request.",
    )
    parser.add_argument(
        "--history",
        help="Optional JSON list of prior messages to seed the conversation context.",
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Assume the plugin is already initialized.",
    )
    parser.add_argument(
        "--skip-shutdown",
        action="store_true",
        help="Leave the plugin initialized after the test run.",
    )
    args = parser.parse_args(argv)

    prior_messages = parse_history(args.history)
    if args.system_info:
        try:
            system_info = json.loads(args.system_info)
        except json.JSONDecodeError as exc:
            print("Failed to parse system-info JSON:", exc)
            sys.exit(1)
    else:
        system_info = None

    ensure_initialized(args.skip_init)

    try:
        response = plugin.execute_query_gemini_voice_command(
            params={"input": args.text},
            context=build_context(args.text, prior_messages),
            system_info=system_info,
        )
        print(json.dumps(response, indent=2))

        print('waiting for voice playback.')

        time.sleep(60)

    finally:
        if not args.skip_shutdown:
            plugin.execute_shutdown_command()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
