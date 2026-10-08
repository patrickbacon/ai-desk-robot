# AI Desk Robot

![Demo](docs/demo.gif)

A desk robot head with an animated OLED face, a two-axis servo neck, and a real-time voice conversation pipeline. A Raspberry Pi Pico drives the face and servos, while a PC handles vision, speech, and conversation over a USB serial link.

## Features
- Custom wake word with a transcription-based backup check
- Streaming voice pipeline: speech-to-text, conversational LLM, and text-to-speech, spoken sentence by sentence for low latency
- Animated OLED eyes with mood expressions, blinking, saccades, and a talking mouth
- Face detection and recognition (YuNet + SFace) to greet known users by name
- Head tracking that follows faces, turns toward whoever is speaking, and reacts to motion
- Gesture animations (nod, shake, tilt, startle) driven by keyframe interpolation
- Mood engine that shifts emotional state from context and fades over time, with matching voice settings
- Tool use: time, weather, timers, reminders, calendar, and web search
- Persistent long-term memory with automatic post-conversation review
- Startle reflex triggered by sudden loud sounds
- Servo-noise rejection so head movement doesn't trigger false speech detection
- Software-controlled speaker amp to eliminate idle buzz
- Rocker switch power sensing through a voltage divider

## Hardware
| Component | Purpose |
|---|---|
| Raspberry Pi Pico | Face, servo, and amp control (MicroPython) |
| 1.3" SH1106 OLED (SPI) | Animated face |
| 2x SG90 servos | Pan and tilt neck |
| Audio amplifier | Speaker drive with shutdown control |
| DW-0845 3W speaker | Voice output |
| USB webcam | Face tracking and recognition |
| Microphone | Voice input |
| Rocker switch | Power with voltage-divider sensing |
| Custom CAD enclosure | Head and base |

## Architecture
```
PC (pc/robot_brain.py)                     Pico (pico/main.py)
 mic -> wake word -> speech-to-text          OLED face rendering
 LLM (streaming) -> text-to-speech -> spkr   servo easing + gestures
 webcam -> face detect/recognize -> aim      amp on/off, power sense
            |                                       ^
            +----------- USB serial ----------------+
              STATE / MOOD / LOOK / GESTURE / AMP
```

## Serial Protocol
| Command | Example | Action |
|---|---|---|
| STATE | `STATE thinking` | Sets the face state |
| MOOD | `MOOD happy` | Sets the eye expression |
| LOOK | `LOOK 110 85` | Moves the head to pan/tilt angles |
| GESTURE | `GESTURE nod` | Plays a head gesture |
| AMP | `AMP on` | Switches the speaker amp |
| STATUS | `STATUS` | Replies with power switch state |

## Setup
1. Flash MicroPython to the Pico and copy `pico/main.py` to it.
2. On the PC: `pip install -r requirements.txt`
3. Copy `env.example` to `.env` and fill in your API keys and settings.
4. Enroll your face: `python pc/robot_brain.py --enroll YourName`
5. Run: `python pc/robot_brain.py`

## Repository Structure
- `pc/` – PC-side voice, vision, and conversation pipeline
- `pico/` – MicroPython firmware for the head
- `env.example` – configuration template
- `requirements.txt` – Python dependencies
- `docs/` – photos and demo media
