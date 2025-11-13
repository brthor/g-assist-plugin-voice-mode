![Voice Mode](cover-art.jpg?raw=true "Voice Mode")

# Voice Mode Plugin for G-Assist

Bring your G-Assist experience to life with natural, high-quality voice responses. 
Voice Mode seamlessly integrates with your existing setup, adding a voice to any and every AI interaction. 
Whether you're getting game tips, checking the weather, or just looking for a chat, Voice Mode makes it a conversation.

## What Can It Do?
- High-Quality Voice Responses: Get natural and expressive audio responses for any query.
- Universal Integration: Automatically adds voice to standard G-Assist responses and even the output from other installed plugins.
- Powered by Gemini: Leverages Google's state-of-the-art AI for both text processing and voice generation.
- Easy Setup: Get up and running in minutes with a simple setup process.

## Before You Start
Make sure you have:
- A [Google Cloud API key](https://aistudio.google.com/api-keys) with Gemini access.
- G-Assist installed on your system.

💡 **Tip**: You'll need a Google Cloud API key enabled for Gemini. You can get one from the [Google AI Studio](https://aistudio.google.com/api-keys)!

## Installation & Setup Guide

### Step 1: Get the Plugin Files
Download the latest `VoiceMode.zip` release from the [project's release page](https://github.com/brthor/g-assist-plugin-voice-mode/releases) and extract its contents.

### Step 2: Install the Plugin
1. Open File Explorer and navigate to the G-Assist plugins directory. If it doesn't exist, create it. The path is:
   ```
   C:\ProgramData\NVIDIA Corporation\nvtopps\rise\plugins\v
   ```
   💡 **Tip**: You can copy and paste this path directly into your File Explorer's address bar and press Enter.

2. Copy all the extracted files into the `v` folder you just opened or created.

### Step 3: Configure Your API Key
1. In the `C:\ProgramData\NVIDIA Corporation\nvtopps\rise\plugins\v` folder, there is an empty file named `gemini.key`.
2. Open the file and paste your Google Gemini API key into it.

   **`gemini.key`**
   ```
   your_api_key_here
   ```
   ⚠️ **Important**: Ensure there are no extra spaces or characters before or after your key. The file should contain *only* the key itself.

That's it! The plugin will automatically detect the updated API Key without needing a restart and will let you know if there is an issue.

## How to Use

To use Voice Mode, simply start your G-Assist query with `/v`.

### Basic Voice Queries
- `/v Hey Google, explain quantum computing like I'm five.`
- `/v What were the main causes of the fall of the Roman Empire?`
- `/v Tell me a joke about programming.`

### Using Voice Mode with Other Plugins (Important!)
To ensure Voice Mode works correctly with other plugins (like the weather plugin), you should send a simple "warmup" query first. This prevents the system from getting confused or calling the other plugin directly without voice.

**The process is two steps:**

1.  **Warmup Query:** Send a simple, conversational query first.
    - `/v Hey, how are you?`

2.  **Plugin Query:** Immediately after, send your command for the other plugin.
    - `/v Hey, can you check the weather in Santa Clara for me?`

Following this two-step process ensures Voice Mode properly intercepts the command and reads the final response aloud.

## Limitations
- Requires an active internet connection to reach Google's APIs.
- Subject to Google's Gemini API usage and rate limits.
- The quality and speed of the response depend on API latency.

### Logging
For troubleshooting, the plugin logs all its activity to the following file:
```
%USERPROFILE%\voicemode.log
```
Check this file for detailed error messages and debugging information.

## Troubleshooting Tips
- **No Voice Response?**
  1.  Make sure you are starting your query with `/v`.
  2.  Check the `%USERPROFILE%\voicemode.log` file for any errors.

- **API Key Not Working?**
  1.  Verify your API key is correctly copied into `gemini.key` and that the file is in the correct directory.
  2.  Make sure there are absolutely no extra spaces or new lines in the `gemini.key` file.

- **Plugin Commands Don't Work with Voice?**
  - Remember to send a "warmup" query first! Send a simple command like `/v Hello` before you send the plugin command (e.g., `/v What's the weather?`). This is the most common reason for issues when combining Voice Mode with other plugins.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.