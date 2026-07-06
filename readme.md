# Real-Time Translator

A desktop application that captures system audio, translates spoken English to Vietnamese in real time using Gemini Live, and displays subtitles as an overlay on your screen.

## Requirements

- Python 3.10 or later
- A Google AI Studio API key

## Installation

1. Clone this repository or download the project.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```text
google-genai
PySide6
soundcard
numpy
python-dotenv
```

## Configuration

Create a `.env` file in the project root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

## Running the Application

Start the translator by running:

```bash
python main1.py
```

## Usage

### Move the Subtitle Overlay

Left-click and drag anywhere on the subtitle window to reposition it.

### Close the Application

Right-click the subtitle overlay and select **"Close Translator"**. This safely disconnects from the Gemini Live API and shuts down all background tasks.

## Notes

- The application captures **system audio (loopback)** rather than microphone input.
- Ensure your speakers or headphones are set as the default Windows playback device.
- An active internet connection is required to communicate with the Gemini Live API.