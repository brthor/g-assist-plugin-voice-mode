import base64
import datetime
import io
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
import winsound
from ctypes import byref, windll, wintypes
from typing import Any, Dict, List, Optional, Tuple

LOG_FILE = os.path.join(os.environ.get("USERPROFILE", "."), "voice-mode-plugin.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# logging.getLogger().addHandler(logging.StreamHandler())

file_dir = os.path.dirname(os.path.realpath(__file__))
logging.info(f"script exec: {__file__}")

try:
    from google import genai
    from google.genai.types import ModelContent, Part, UserContent, GenerateContentConfig
except ImportError:  # pragma: no cover - handled at runtime
    genai = None  # type: ignore
    ModelContent = Part = UserContent = GenerateContentConfig = object  # type: ignore
    logging.error("Unable to import google genai")

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    np = None  # type: ignore
    logging.error("Unable to import numpy")
    logging.error(exc)

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    sd = None  # type: ignore
    logging.error("Unable to import sounddevice")
    logging.error(exc)


Response = Dict[str, Any]

PLUGIN_STORAGE = os.path.join(
    os.environ.get("PROGRAMDATA", "."),
    "NVIDIA Corporation",
    "nvtopps",
    "rise",
    "plugins",
    "v",
)
API_KEY_FILE = os.path.join(PLUGIN_STORAGE, "gemini.key")
CONFIG_FILE = os.path.join(PLUGIN_STORAGE, "config.json")
PLUGINS_ROOT = os.path.dirname(PLUGIN_STORAGE)

DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_VOICE_MODEL = "gemini-2.5-flash-tts"
DEFAULT_VOICE_NAME = "Despina"
DEFAULT_AUDIO_FORMAT = "WAV"

TOOL_CALLS_PROPERTY = "tool_calls"
FUNCTION_PROPERTY = "func"
PARAMS_PROPERTY = "properties"
ALT_PARAMS_PROPERTY = "params"
CONTEXT_PROPERTY = "messages"
SYSTEM_INFO_PROPERTY = "system_info"

INITIALIZE_COMMAND = "initialize"
SHUTDOWN_COMMAND = "shutdown"
VOICE_COMMAND = "voice_mode"
SET_VOICE_COMMAND = "set_gemini_voice"
EXCLUDED_PLUGINS = {"gemini", "google", "voice-mode", "v", "voice_mode"}

HIT_API_LIMIT = False

api_key = None
client: Optional["genai.Client"] = None  # type: ignore[name-defined]
chat_model = DEFAULT_CHAT_MODEL
voice_model = DEFAULT_VOICE_MODEL
voice_name = DEFAULT_VOICE_NAME
voice_format = DEFAULT_AUDIO_FORMAT
config_cache: Dict[str, Any] = {}
available_plugins: Dict[str, Dict[str, Any]] = {}
speech_worker: Optional["SpeechWorker"] = None

RESPONSE_SETUP_API_KEY = f"""
-- API KEY REQUIRED --

- To use voice mode, you need a gemini api key.
- Get it from https://aistudio.google.com/apikey
- We recommend linking a billing account to partake in the more generous free tier.

After getting the api key, add it to this plugin's key file:
{API_KEY_FILE}
"""

RESPONSE_INTRO = f"""Welcome to voice mode. 

Voice mode adds a voice response to all user requests and can call any of your installed g-assist plugins.

- To add voice mode to any request type: "/v YOUR REQUEST"
- See available voice options: https://ai.google.dev/gemini-api/docs/speech-generation#voices
- To change the current voice, adjust the voice name in the config file: {CONFIG_FILE}

"""

RESPONSE_HIT_API_LIMIT = f"""-- USAGE LIMIT EXHAUSTED --

- You have exhausted the text-to-speech usage limit for your api key.
- Either wait, add more usage to your api key, or change your api key in the file:
{API_KEY_FILE}
"""

def wrap_main():
    try:
        main()
    except Exception as exc:
        logging.exception(exc)
        write_response(generate_failure_response("Encountered an error. Check the log for more infomation."))

def main() -> int:
    global client, HIT_API_LIMIT
    logging.info("Voice Mode plugin started.")

    # os.makedirs(PLUGIN_STORAGE, exist_ok=True)
    commands = {
        INITIALIZE_COMMAND: execute_initialize_command,
        SHUTDOWN_COMMAND: execute_shutdown_command,
        VOICE_COMMAND: execute_query_gemini_voice_command,
    }

    write_response(generate_message_response(RESPONSE_INTRO))

    logging.info("reading commands")
    logging.info(client)
    while True:
        request = read_command()
        if request is None:
            logging.error("Malformed request received.")
            continue

        if client is None:
            logging.info("initializing without initialize command")
            response = execute_initialize_command()
            logging.info(client)
            if not response.get("success"):
                logging.error("Initialization failed:", response)

                # - return a success response so the user can retry
                # write_response(generate_failure_response(response.get("message", None)))
                write_response(response)
                continue

        if HIT_API_LIMIT:
            write_response(generate_failure_response(RESPONSE_HIT_API_LIMIT))
            HIT_API_LIMIT = False
            continue

        tool_calls = request.get(TOOL_CALLS_PROPERTY, [])
        if not isinstance(tool_calls, list):
            write_response(generate_failure_response("Invalid tool call payload."))
            continue

        for tool_call in tool_calls:
            command_name = tool_call.get(FUNCTION_PROPERTY)
            handler = commands.get(command_name)
            if handler is None:
                write_response(generate_failure_response(f"Unknown command: {command_name}"))
                continue

            params = _extract_params(tool_call)
            context = request.get(CONTEXT_PROPERTY)
            system_info = request.get(SYSTEM_INFO_PROPERTY)

            try:
                response = handler(params, context, system_info)
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Handler %s failed: %s", command_name, exc)
                response = generate_failure_response("Unhandled plugin exception.")

            write_response(response)

            if command_name == SHUTDOWN_COMMAND:
                logging.info("Shutdown command processed; exiting.")
                return 0


def execute_initialize_command(
    params: Optional[Dict[str, Any]] = None,
    context: Optional[Any] = None,
    system_info: Optional[Any] = None,
) -> Response:
    global client, chat_model, voice_model, voice_name, voice_format, config_cache, available_plugins, API_KEY_FILE, api_key

    if genai is None:
        logging.error("google-genai is not installed.")
        return generate_failure_response("google-genai dependency missing.")

    # if not os.path.exists(API_KEY_FILE):
    #     API_KEY_FILE = f'{file_dir}/gemini.key'

    api_key = _read_file(API_KEY_FILE)
    if not api_key:
        logging.error("API key not found at %s", API_KEY_FILE)
        return generate_failure_response(RESPONSE_SETUP_API_KEY)

    config_cache = _load_config()
    chat_model = config_cache.get("chat_model", DEFAULT_CHAT_MODEL)
    voice_model = config_cache.get("voice_model", DEFAULT_VOICE_MODEL)
    voice_name = config_cache.get("voice_name", DEFAULT_VOICE_NAME)
    # voice_format = config_cache.get("voice_format", DEFAULT_AUDIO_FORMAT)

    try:
        client = genai.Client(api_key=api_key)
        logging.info("Configured Gemini client (chat: %s, voice: %s).", chat_model, voice_model)
        available_plugins = load_available_plugins()
        ensure_speech_worker()
        logging.info("Discovered %d external plugins for tool use.", len(available_plugins))
        return generate_success_response("Voice Mode initialized.")
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to configure Gemini client: %s", exc)
        client = None
        return generate_failure_response("Unable to configure Gemini client.")


def execute_shutdown_command(
    params: Optional[Dict[str, Any]] = None,
    context: Optional[Any] = None,
    system_info: Optional[Any] = None,
) -> Response:
    global client, speech_worker
    client = None
    if speech_worker:
        speech_worker.stop()
        speech_worker = None
    logging.info("Voice Mode plugin shutdown.")
    return generate_success_response("Voice Mode shutdown complete.")


def execute_query_gemini_voice_command(
    params: Optional[Dict[str, Any]] = None,
    context: Optional[Any] = None,
    system_info: Optional[Any] = None,
) -> Response:
    global client

    if client is None:
        logging.error("Gemini client not initialized.")
        return generate_failure_response("Gemini client not initialized.")

    prompt = ""
    if isinstance(params, dict):
        prompt = params.get("input", "") or ""

    if not prompt:
        prompt = extract_latest_user_message(context) or ""

    if not prompt.strip():
        logging.error("No prompt available for Voice Mode request.")
        return generate_failure_response("No prompt provided.")

    ensure_speech_worker()
    history = convert_openai_history_to_google_gemini(context, exclude_latest=True)

    try:
        chat_session = client.chats.create(model=chat_model, history=history)
        planner_prompt = build_planner_prompt(prompt, system_info)
        planner_response = chat_session.send_message(planner_prompt)
        plan = parse_planner_decision(extract_response_text(planner_response))

        tool_result = None
        if plan and plan.get("action") == "tool_call":
            plugin_name = plan["plugin"]
            function_name = plan["function"]
            invocation_request = {
                "plugin": plugin_name,
                "function": function_name,
                "arguments": plan.get("arguments", {}),
            }
            tool_result, tool_success = invoke_plugin_tool(
                invocation_request,
                context=context,
                system_info=system_info,
            )
            if not tool_success:
                logging.error("Tool invocation failed: %s", tool_result)
                write_response(generate_message_response(tool_result))
                return generate_failure_response(tool_result)
            summary_prompt = build_summary_prompt(prompt, plugin_name, function_name, tool_result, system_info)
            final_text = stream_final_response(chat_session, summary_prompt)
        else:
            response_prompt = build_direct_response_prompt(prompt, system_info)
            final_text = stream_final_response(chat_session, response_prompt)

        if not final_text.strip():
            logging.warning("Voice Mode produced no final text.")
            return generate_failure_response("No response available.")

        enqueue_speech(final_text)
        return generate_success_response()

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Voice Mode request failed: %s", exc)
        return generate_failure_response("Voice Mode request failed.")


def execute_list_gemini_voice_options_command(
    params: Optional[Dict[str, Any]] = None,
    context: Optional[Any] = None,
    system_info: Optional[Any] = None,
) -> Response:
    if client is None:
        return generate_failure_response("Gemini client not initialized.")

    voices = fetch_available_voice_metadata()
    if not voices:
        return generate_failure_response("No Gemini voices could be retrieved.")

    description_lines = [
        f"- {voice['voice_name']} (model: {voice['model']})"
        for voice in voices
    ]
    description = "Available Gemini voices:\n" + "\n".join(description_lines)
    description += f"\n\nCurrent voice: {voice_name}"
    return generate_success_response(description)

def synthesize_voice_streaming(text: str, locale: str="en-us"):
    """Synthesizes speech from the input text.

    Args:
        prompt: Styling instructions on how to synthesize the content in
          the text field.
        text_chunks: Text chunks to synthesize. Note that The synthesis will
          start when the client initiates half-close.
        model: gemini tts model name. gemini-2.5-flash-tts, gemini-2.5-pro-tts
        voice: voice name. example: leda, kore. Refer to available voices
        locale: locale name, example: en-us. Refer to available locales.
    """

    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    config_request = texttospeech.StreamingSynthesizeRequest(
        streaming_config=texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(
                name=voice_name,
                language_code=locale,
                model_name=voice_model
            )
        )
    )

    text_chunks = [text]

    # Example request generator. A function like this can be linked to an LLM
    # text generator and the text can be passed to the TTS API asynchronously.
    def request_generator():
      yield config_request

      for i, text in enumerate(text_chunks):
        yield texttospeech.StreamingSynthesizeRequest(
            input=texttospeech.StreamingSynthesisInput(
              text=text,
              # Prompt is only supported in the first input chunk.
              # prompt=prompt if i == 0 else None,
            )
        )

    request_start_time = datetime.datetime.now()
    streaming_responses = client.streaming_synthesize(request_generator())

    is_first_chunk_received = False
    final_audio_data = np.array([])
    num_chunks_received = 0
    for response in streaming_responses:
        # just a simple progress indicator
        num_chunks_received += 1
        print(".", end="")
        if num_chunks_received % 40 == 0:
            print("")

        # measuring time to first audio
        if not is_first_chunk_received:
            is_first_chunk_received = True
            first_chunk_received_time = datetime.datetime.now()

        # accumulating audio. In a web-server scenario, you would want to
        # "emit" audio to the frontend as soon as it arrives.
        #
        # For example using flask socketio, you could do the following
        # from flask_socketio import SocketIO, emit
        # emit("audio", response.audio_content)
        # socketio.sleep(0)
        audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
        final_audio_data = np.concatenate((final_audio_data, audio_data))

    time_to_first_audio = first_chunk_received_time - request_start_time
    time_to_completion = datetime.datetime.now() - request_start_time
    audio_duration = len(final_audio_data) / 24_000  # default sampling rate.

    print("\n")
    print(f"Time to first audio: {time_to_first_audio.total_seconds()} seconds")
    print(f"Time to completion: {time_to_completion.total_seconds()} seconds")
    print(f"Audio duration: {audio_duration} seconds")

    return final_audio_data


def synthesize_voice_with_gemini(text: str) -> Optional[bytes]:
    if client is None:
        return None

    try:
        from google.genai import types

        response = client.models.generate_content(
            model=voice_model,
            contents=f"Please read the following aloud at a quick pace:\n{text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    ),

                ),
            )
        )

        pcm_data = response.candidates[0].content.parts[0].inline_data.data

        return pcm_data
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Gemini audio generation failed: %s", exc)

        global HIT_API_LIMIT
        if 'RESOURCE_EXHAUSTED' in str(exc):
            HIT_API_LIMIT = True

        return None


def decode_audio_payload(response: Any) -> Optional[bytes]:
    if response is None:
        return None

    chunks: List[bytes] = []

    output_audio = getattr(response, "output_audio", None)
    if output_audio:
        chunks.extend(_extract_audio_chunks(output_audio))

    if not chunks:
        if hasattr(response, "to_dict"):
            response_dict = response.to_dict()
        else:
            response_dict = response
        inline_data_values = _collect_inline_data(response_dict)
        for value in inline_data_values:
            chunks.append(_coerce_bytes(value))

    if not chunks:
        return None

    return b"".join(chunks)


def play_audio_bytes(pcm_bytes: bytes, cancel_event: Optional[threading.Event] = None) -> None:
    if sd is None or np is None:
        logging.warning("sounddevice/numpy not available; skipping playback.")
        return

    samplewidth = 2
    channels = 1
    samplerate = 24000

    # def create_temp_wave_file(pcm, channels=1, rate=24000, sample_width=2):
    #     # Create a named temporary file.
    #     # - suffix='.wav' gives it the correct extension.
    #     # - delete=False is CRUCIAL. It prevents the file from being deleted
    #     #   when the 'with' block is exited, allowing us to return its path.
    #     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
    #         file_path = tf.name
    #         # The tempfile is already open, but we use wave.open on its path
    #         # to get the wave-specific writer object.
    #         with wave.open(file_path, "wb") as wf:
    #             wf.setnchannels(channels)
    #             wf.setsampwidth(sample_width)
    #             wf.setframerate(rate)
    #             wf.writeframes(pcm)
    #
    #     return file_path
    #
    # wav_file = create_temp_wave_file(pcm_bytes)

    # winsound.PlaySound(wav_file, winsound.SND_ASYNC | winsound.SND_FILENAME)
    # audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    # sd.play(audio_array, samplerate=samplerate)

    # Set the NumPy data type based on the sample width.
    # For a samplewidth of 2, int16 is appropriate.
    dtype = np.int16

    try:
        frames_per_chunk = 2048
        bytes_per_frame = channels * samplewidth
        bytes_per_chunk = frames_per_chunk * bytes_per_frame

        with sd.OutputStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                blocksize=frames_per_chunk,
        ) as stream:
            logging.info("Starting audio playback.")
            # Create a memory view of the bytes for efficient slicing
            pcm_view = memoryview(pcm_bytes)

            for i in range(0, len(pcm_view), bytes_per_chunk):
                if cancel_event and cancel_event.is_set():
                    logging.info("Audio playback cancelled.")
                    break

                chunk = pcm_view[i:i + bytes_per_chunk]

                if not chunk:
                    break

                # Convert the chunk of bytes to a NumPy array. [8, 12]
                data = np.frombuffer(chunk, dtype=dtype)

                # The sounddevice library expects the array to have a shape of
                # (number_of_frames, number_of_channels) for multi-channel audio. [9]
                if channels > 1:
                    data = np.reshape(data, (-1, channels))

                stream.write(data)

            logging.info("Audio playback finished.")

    except Exception as e:
        logging.error("An error occurred during audio playback: %s", e)


class SpeechWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.current_cancel = threading.Event()
        self.shutdown = threading.Event()

    def run(self) -> None:
        while not self.shutdown.is_set():
            try:
                text = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.shutdown.is_set():
                break

            self.current_cancel.clear()

            try:
                pcm_bytes = synthesize_voice_with_gemini(text)
                if pcm_bytes and not self.current_cancel.is_set():
                    play_audio_bytes(pcm_bytes, cancel_event=self.current_cancel)
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Speech worker failed: %s", exc)
            finally:
                self.queue.task_done()

        logging.info("Speech worker shutting down.")

    def submit(self, text: str) -> None:
        self.current_cancel.set()
        self._clear_queue()
        self.queue.put(text)

    def stop(self) -> None:
        self.shutdown.set()
        self.current_cancel.set()

    def _clear_queue(self) -> None:
        try:
            while True:
                self.queue.get_nowait()
                self.queue.task_done()
        except queue.Empty:
            pass


def ensure_speech_worker() -> None:
    global speech_worker
    if speech_worker is None:
        speech_worker = SpeechWorker()
        speech_worker.start()


def enqueue_speech(text: str) -> None:
    if not text:
        return
    ensure_speech_worker()
    assert speech_worker is not None
    speech_worker.submit(text)


def convert_openai_history_to_google_gemini(
    context: Optional[Any],
    exclude_latest: bool = False,
) -> List[Any]:
    if not isinstance(context, list):
        return []

    entries = context[:-1] if exclude_latest else context
    gemini_history: List[Any] = []
    for message in entries:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        role = message.get("role")
        if not isinstance(content, str):
            continue
        part = Part(text=content)
        if role == "assistant":
            gemini_history.append(ModelContent(parts=[part]))
        else:
            gemini_history.append(UserContent(parts=[part]))
    return gemini_history


def extract_latest_user_message(context: Optional[Any]) -> Optional[str]:
    if not isinstance(context, list):
        return None

    for entry in reversed(context):
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role == "user" and isinstance(content, str):
            return content
    return None


def read_command() -> Optional[Dict[str, Any]]:
    try:
        pipe = windll.kernel32.GetStdHandle(-10)
        chunks: List[str] = []
        while True:
            buffer = bytes(4096)
            read = wintypes.DWORD()
            success = windll.kernel32.ReadFile(
                pipe,
                buffer,
                len(buffer),
                byref(read),
                None,
            )
            if not success:
                logging.error("Unable to read from stdin pipe.")
                return None
            segment = buffer[: read.value].decode("utf-8")
            chunks.append(segment)
            if read.value < len(buffer):
                break
        raw = "".join(chunks)
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logging.error("Invalid JSON received: %s", exc)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("read_command failed: %s", exc)
    return None


def write_response(response: Response) -> None:
    logging.info("write_response %s", json.dumps(response))
    try:
        pipe = windll.kernel32.GetStdHandle(-11)
        payload = json.dumps(response) + "<<END>>"
        data = payload.encode("utf-8")
        written = wintypes.DWORD()
        windll.kernel32.WriteFile(
            pipe,
            data,
            len(data),
            byref(written),
            None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("write_response failed: %s", exc)


def generate_success_response(message: Optional[str] = None) -> Response:
    result: Response = {"success": True}
    if message:
        result["message"] = message
    return result


def generate_failure_response(message: Optional[str] = None) -> Response:
    result: Response = {"success": False}
    if message:
        result["message"] = message
    return result


def generate_message_response(message: str) -> Response:
    return {"message": message}


def build_planner_prompt(user_prompt: str, system_info: Optional[Any]) -> str:
    tools_description = build_tool_prompt()
    system_line = f"System info: {system_info}" if system_info else ""
    instructions = [
        "You are Voice Mode, acting as NVIDIA G-Assist's voice interface.",
        "Decide whether to call an external tool or answer directly.",
        "Respond ONLY with valid JSON using one of these formats:",
        '{"action":"tool_call","plugin":"plugin_name","function":"function_name","arguments":{"param":"value"}}',
        '{"action":"respond"}',
        "Choose tool_call only when the tool is clearly better suited.",
    ]
    if tools_description:
        instructions.append(tools_description)
    instructions.append(system_line)
    instructions.append(f"User request: {user_prompt}")
    return "\n".join(filter(None, instructions))


def parse_planner_decision(response_text: str) -> Optional[Dict[str, Any]]:
    if not response_text:
        return None
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        logging.warning("Planner response was not valid JSON: %s", response_text)
        return None

    action = payload.get("action")
    if action == "tool_call":
        plugin = payload.get("plugin")
        function = payload.get("function")
        arguments = payload.get("arguments", {})
        if isinstance(plugin, str) and isinstance(function, str):
            if not isinstance(arguments, dict):
                arguments = {}
            return {
                "action": "tool_call",
                "plugin": plugin,
                "function": function,
                "arguments": arguments,
            }
    elif action == "respond":
        return {"action": "respond"}

    return None


def build_summary_prompt(
    user_prompt: str,
    plugin_name: str,
    function_name: str,
    tool_result: str,
    system_info: Optional[Any],
) -> str:
    return (
        f"You invoked the G-Assist tool {plugin_name}.{function_name} in response to the user request:\n"
        f"{user_prompt}\n\n"
        f"The tool returned:\n{tool_result}\n\n"
        f"{'System info: ' + str(system_info) if system_info else ''}\n"
        "Summarize these results for the user in a concise, conversational tone."
    )


def build_direct_response_prompt(user_prompt: str, system_info: Optional[Any]) -> str:
    return (
        "Now you must construct your response for the respond action. Reply in plain text, not json for this message."
        "Provide a helpful, concise response to the following user request. "
        "Speak as NVIDIA G-Assist's built-in assistant.\n"
        f"System info: {system_info}\n"
        f"User request: {user_prompt}"
    )


def stream_final_response(chat_session: Any, prompt: str) -> str:
    final_segments: List[str] = []
    stream = chat_session.send_message_stream(prompt)
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if text:
            final_segments.append(text)
            write_response(generate_message_response(text))
    return "".join(final_segments)


def _extract_params(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    params = tool_call.get(PARAMS_PROPERTY)
    if isinstance(params, dict):
        return params
    params = tool_call.get(ALT_PARAMS_PROPERTY)
    if isinstance(params, dict):
        return params
    return {}


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return None
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to read %s: %s", path, exc)
        return None


def _load_config() -> Dict[str, str]:
    if not os.path.exists(CONFIG_FILE) and os.path.exists(f'{file_dir}/config.json'):
        try:
            with open(f'{file_dir}/config.json', "r", encoding="utf-8") as cfg:
                return json.load(cfg)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to read config: %s", exc)
            return {}

    if not os.path.isfile(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as cfg:
            return json.load(cfg)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to read config: %s", exc)
        return {}


def _save_config(config: Dict[str, Any]) -> None:
    try:
        # os.makedirs(PLUGIN_STORAGE, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as cfg:
            json.dump(config, cfg, indent=2)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to write config: %s", exc)


def load_available_plugins() -> Dict[str, Dict[str, Any]]:
    plugins: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(PLUGINS_ROOT):
        logging.warning("Plugins root %s not found; no internal tools available.", PLUGINS_ROOT)
        return plugins

    for entry in os.listdir(PLUGINS_ROOT):
        entry_path = os.path.join(PLUGINS_ROOT, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.lower() in EXCLUDED_PLUGINS:
            continue

        manifest_path = os.path.join(entry_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue

        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Failed to parse manifest for %s: %s", entry, exc)
            continue

        executable = manifest.get("executable")
        functions = manifest.get("functions", [])
        if not executable or not functions:
            continue

        normalized_executable = executable
        if normalized_executable.startswith("./") or normalized_executable.startswith(".\\"):
            normalized_executable = normalized_executable[2:]
        if not os.path.isabs(normalized_executable):
            normalized_executable = os.path.normpath(os.path.join(entry_path, normalized_executable))

        function_map = {
            func["name"]: func
            for func in functions
            if isinstance(func, dict) and func.get("name")
        }
        if not function_map:
            continue

        plugins[entry] = {
            "name": entry,
            "path": entry_path,
            "manifest": manifest,
            "executable": normalized_executable,
            "functions": function_map,
        }

    return plugins


def build_tool_prompt() -> str:
    if not available_plugins:
        return ""

    lines = [
        "Available tools:",
    ]

    for plugin_name, descriptor in available_plugins.items():
        lines.append(f"- Plugin '{plugin_name}':")
        lines.append(f"    - {json.dumps(descriptor['manifest'])}")
        # for function_name, function_details in descriptor["functions"].items():
        #     description = function_details.get("description", "No description provided.")
        #     lines.append(f"    - {function_name}: {description}")

    lines.append("Only call a tool when it is clearly helpful; otherwise answer normally.")
    return "\n".join(lines)

def kill_proc_group(process: subprocess.Popen):
    # Terminate the process and its entire process tree
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], check=True)
        # print(f"Successfully terminated process with PID: {process.pid} and its children.")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to kill process: %s", e)
        # print(f"Error terminating process: {e}")

def invoke_plugin_tool(
    request: Dict[str, Any],
    context: Optional[Any],
    system_info: Optional[Any],
) -> Tuple[str, bool]:
    plugin_name = request["plugin"]
    function_name = request["function"]
    arguments = request.get("arguments", {})

    descriptor = None
    for key, value in available_plugins.items():
        if key.lower() == plugin_name.lower():
            descriptor = value
            plugin_name = key  # normalize casing
            break

    if descriptor is None:
        return (f"Plugin '{plugin_name}' is not available.", False)

    if function_name not in descriptor["functions"]:
        return (f"Plugin '{plugin_name}' has no function '{function_name}'.", False)

    executable = descriptor["executable"]
    if not os.path.isfile(executable):
        return (f"Plugin executable not found for '{plugin_name}'.", False)

    payload_context = context if isinstance(context, list) else []
    tool_payload: Dict[str, Any] = {
        "tool_calls": [
            {
                "func": function_name,
                "properties": arguments,
                "params": arguments,
            }
        ],
        "messages": payload_context,
    }
    if system_info is not None:
        tool_payload["system_info"] = system_info

    commands = [
        # {"tool_calls": [{"func": "initialize"}]},
        tool_payload,
        # {"tool_calls": [{"func": "shutdown"}]},
    ]

    try:
        proc = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=descriptor["path"],
            bufsize=0,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to start plugin %s: %s", plugin_name, exc)
        return (f"Unable to start plugin '{plugin_name}'.", False)

    responses: List[List[str]] = []
    try:
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("Plugin pipes unavailable.")

        for payload in commands:
            payload_data = json.dumps(payload).encode("utf-8")
            proc.stdin.write(payload_data)
            proc.stdin.flush()
            time.sleep(0.01)
            response_batch = _collect_plugin_responses(proc.stdout)
            if response_batch:
                responses.append(response_batch)
    # except Exception as exc:  # pylint: disable=broad-except
    #     logging.exception("Error communicating with plugin %s: %s", plugin_name, exc)
    #     raise
    #     return (f"Communication error with plugin '{plugin_name}'.", False)
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        if proc.stdout:
            try:
                proc.stderr.close()
            except Exception:
                pass
        # if proc.stderr:
        #     try:
        #         stderr_output = proc.stderr.read()
        #         if stderr_output:
        #             try:
        #                 decoded = stderr_output.decode("utf-8", errors="replace")
        #             except AttributeError:
        #                 decoded = str(stderr_output)
        #             logging.debug("Plugin %s stderr: %s", plugin_name, decoded)
        #         proc.stderr.close()
        #     except Exception:
        #         pass
        try:
            proc.wait(timeout=0.1)
        except Exception:
            logging.info(f"kill proc: {plugin_name}")
            # proc.kill()
            kill_proc_group(proc)

    if len(responses) < 1:
        return ("Plugin returned no data.", False)

    command_responses = responses[0]
    tool_messages: List[str] = []
    success_payload: Dict[str, Any] = {}

    for response_text in command_responses:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            tool_messages.append(response_text)
            continue

        if "message" in parsed:
            msg = parsed.get("message")
            if isinstance(msg, str):
                tool_messages.append(msg)

        if "success" in parsed:
            success_payload = parsed

    tool_response_raw = command_responses[-1] if command_responses else ""
    try:
        tool_response = json.loads(tool_response_raw)
    except json.JSONDecodeError:
        tool_response = success_payload or {"message": tool_response_raw}

    message = "\n".join(tool_messages) or tool_response.get("message") or tool_response_raw
    success = bool(tool_response.get("success", True))
    return (message, success)


def _read_plugin_response(stream: Any) -> str:
    buffer = bytearray()
    while True:
        chunk = stream.read(1)
        if not chunk:
            break
        buffer.extend(chunk)
        if buffer.endswith(b"<<END>>"):
            break
    payload = buffer.decode("utf-8", errors="replace")
    if payload.endswith("<<END>>"):
        payload = payload[:-7]
    return payload


def _collect_plugin_responses(stream: Any, max_messages: int = 64) -> List[str]:
    messages: List[str] = []
    for _ in range(max_messages):
        chunk = _read_plugin_response(stream)
        if not chunk:
            break
        messages.append(chunk)
        try:
            parsed = json.loads(chunk)
            if "success" in parsed:
                break
        except json.JSONDecodeError:
            logging.error("invalid json response from toolcall.")
            return messages
    return messages


def build_candidate_text(candidate: Any) -> str:
    text = getattr(candidate, "text", "")
    if text:
        return text
    if hasattr(candidate, "content"):
        parts = getattr(candidate.content, "parts", [])
        strings = [getattr(part, "text", "") for part in parts if getattr(part, "text", "")]
        return "".join(strings)
    return ""


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    text = getattr(response, "text", "")
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            candidate_text = build_candidate_text(candidate)
            if candidate_text:
                return candidate_text
    return ""


def _extract_audio_chunks(audio_parts: Any) -> List[bytes]:
    chunks: List[bytes] = []
    for part in audio_parts:
        inline_data = getattr(part, "inline_data", None)
        data = getattr(inline_data, "data", None) if inline_data else None
        if data:
            chunks.append(_coerce_bytes(data))
    return chunks


def _collect_inline_data(node: Any) -> List[Any]:
    results: List[Any] = []
    if isinstance(node, dict):
        if "inlineData" in node and isinstance(node["inlineData"], dict):
            inline_data = node["inlineData"].get("data")
            if inline_data:
                results.append(inline_data)
        for value in node.values():
            results.extend(_collect_inline_data(value))
    elif isinstance(node, list):
        for item in node:
            results.extend(_collect_inline_data(item))
    return results


def _coerce_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        try:
            return base64.b64decode(value)
        except Exception:
            return value.encode("utf-8")
    return bytes(value)


def fetch_available_voice_metadata() -> List[Dict[str, str]]:
    if client is None:
        return []

    voices: List[Dict[str, str]] = []
    try:
        try:
            models_iter = client.models.list(page_size=100, view="FULL")
        except TypeError:
            models_iter = client.models.list(page_size=100)

        for model in models_iter:
            model_name = getattr(model, "name", "unknown")
            for voice in _extract_voice_names_from_model(model):
                voices.append({
                    "voice_name": voice,
                    "model": model_name,
                })

        if not voices:
            voices.append({
                "voice_name": voice_name,
                "model": voice_model,
            })
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to list Gemini voices: %s", exc)
        if not voices:
            voices.append({
                "voice_name": voice_name,
                "model": voice_model,
            })
    return voices


def _extract_voice_names_from_model(model: Any) -> List[str]:
    voice_names: List[str] = []
    voice_config = getattr(model, "voice_config", None)
    if voice_config:
        candidates = getattr(voice_config, "voices", None) or getattr(voice_config, "supported_voices", None)
        if candidates:
            for entry in candidates:
                candidate_name = getattr(entry, "name", None) or getattr(entry, "voice_name", None)
                if isinstance(candidate_name, str):
                    voice_names.append(candidate_name)
    supported_voice_styles = getattr(model, "supported_voice_styles", None)
    if supported_voice_styles:
        for entry in supported_voice_styles:
            if isinstance(entry, str):
                voice_names.append(entry)
            else:
                candidate = getattr(entry, "name", None) or getattr(entry, "voice_name", None)
                if isinstance(candidate, str):
                    voice_names.append(candidate)
    return voice_names


if __name__ == "__main__":
    raise SystemExit(wrap_main())
