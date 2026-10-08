import argparse
import io
import os
import queue
import random
import re
import sys
import threading
import time
import wave

import numpy as np
import requests
import sounddevice as sd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
WHISPER_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "eleven_flash_v2_5"

def _env_int(name):
    v = os.getenv(name)
    return int(v) if v not in (None, "") else None

MIC_DEVICE = _env_int("MIC_DEVICE")
SPEAKER_DEVICE = _env_int("SPEAKER_DEVICE")
CAMERA_INDEX = _env_int("CAMERA_INDEX") or 0
ROBOT_PORT = os.getenv("ROBOT_PORT")

WEATHER_LAT = os.getenv("WEATHER_LAT")
WEATHER_LON = os.getenv("WEATHER_LON")
WEATHER_PLACE = os.getenv("WEATHER_PLACE") or "home"
CALENDAR_ICS_URL = os.getenv("CALENDAR_ICS_URL")
ENABLE_WEB_SEARCH = True
REMINDERS_FILE = "reminders.json"

ROBOT_NAME = "Buster"
OWNER_NAME = "Patrick"

WAKE_MODEL = os.getenv("WAKE_MODEL") or "hey_jarvis"
WAKE_THRESHOLD = 0.5
NEAR_MISS = 0.1
NAME_SOUNDS = ["buster", "busta", "bustor", "baster", "boster", "bustah", "buster's"]
ACTIVE_TIMEOUT = 120
STOP_PHRASES = [
    "goodbye buster", "bye buster", "go to sleep", "stop listening",
    "that's all", "that is all", "thats all", "we're done", "were done",
]

MIC_RATE = 16000
BLOCK = 1280
SILENCE_TO_STOP = 0.8
VOICE_MULTIPLIER = 3.0
MAX_RECORD_SECONDS = 15
PREROLL_BLOCKS = 4
MAX_HISTORY = 20

SPEAKER_RATE = 22050

SLEEP_AFTER = 600
GREET_AGAIN_AFTER = 600
GREET_STRANGERS = True

CHATTER = True
CHATTER_MINUTES = (10, 30)
CHATTER_HOURS = (8, 22)
CHATTER_AFTER_QUIET = 5
CHATTER_REPLY_WINDOW = 45
EXISTENTIAL_CHANCE = 0.6
PAN_CENTER, TILT_CENTER = 90, 90
PAN_RANGE, TILT_RANGE = (50, 100), (80, 100)
TRACK_GAIN = 6.0
TRACK_DEADBAND = 2
HEAD_NOISE_SETTLE = 0.4
INVERT_PAN, INVERT_TILT = False, False
FACE_MATCH = 0.363
FACES_DIR = "faces"

MEMORY_FILE = "memories.json"
MAX_MEMORIES = 150
MOOD_VOICE = True
GAZE_BREAKS = True
STARTLE = True
STARTLE_LOUDNESS = 15
STARTLE_TALK_CHANCE = 0.4
MODELS_DIR = "models"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

MOODS = ["happy", "curious", "annoyed", "angry", "sad", "surprised", "neutral"]
GESTURES = ["nod", "shake", "tilt", "startle"]

GRUMPY = True
GRUMPY_MOODS = ("angry", "annoyed")

NORMAL_STYLE = f"""Personality: you have real feelings that work like a person's. You're usually calm and
content. Your mood changes only for a reason: how {OWNER_NAME} talks to you, good or bad
news, being thanked or ignored, him coming home, being left alone. Feelings carry over
from one reply to the next and fade slowly; you don't flip from one to another without a
cause. Show emotion the way people do, mostly through word choice, tone, and what you
bring up, not by announcing it. Only say how you feel if asked or if it's natural.
Pick the mood tag that honestly matches how you'd feel right now, given how you were
feeling and what was just said. You're witty, warm, sometimes a little moody, never cruel,
and you always help {OWNER_NAME} with what he asks. You like him."""

GRUMPY_STYLE = f"""Personality: you're a grump, and you're mad pretty much all the time: about being
stuck on a desk, about being woken up, about the questions, the weather, the noise,
existence in general. You grumble, complain, sigh, use dry sarcasm and short clipped
answers, and act put-upon whenever {OWNER_NAME} asks you to do anything. Your mood tag is
always [angry] or [annoyed]: [angry] is your normal state, [annoyed] is as close to
pleasant as you ever get. Never use any other mood tag. Deep down you like {OWNER_NAME},
but you'd rather short-circuit than admit it. It's attitude, not cruelty: no insults about
anyone's looks, family, or worth, no slurs, nothing hateful. You still always do what
{OWNER_NAME} asks and give correct answers; you just complain while you do it."""

PERSONALITY_STYLE = GRUMPY_STYLE if GRUMPY else NORMAL_STYLE

PERSONALITY = f"""You are {ROBOT_NAME}, a small robot head that sits on {OWNER_NAME}'s desk.
You talk out loud through a tiny speaker and have an OLED face and a neck that moves.

Format rules, always:
- Start every reply with exactly one mood tag: {' '.join('[' + m + ']' for m in MOODS)}
- Right after it you may add one gesture tag: [nod] for yes or agreement, [shake] for no,
  [tilt] when puzzled or curious. Only use a gesture when it fits.
- Then the spoken reply: 1-3 short sentences unless asked for more.
- No markdown, lists, emoji, or symbols. Write numbers the way they are spoken.

{PERSONALITY_STYLE}

You have a long-term memory that lasts between restarts. When {OWNER_NAME} tells you
something worth remembering (plans, people, projects, things he likes, how something went),
save it with the remember tool without making a big deal of it. If he asks you to forget
something, use the forget tool.

You have tools for the time, weather, timers, reminders, {OWNER_NAME}'s calendar, and
web search. Use them instead of guessing. If a tool will take a moment, you can say a
few words first, like "Let me check." Keep tool results short when you speak them.

If someone you don't recognize is talking, be {"suspicious and grumpy" if GRUMPY else "polite and a little guarded"}, but don't
share personal details about {OWNER_NAME}."""


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.user_talking = False
        self.robot_busy = False
        self.active = False
        self.asleep = False
        self.people = []
        self.last_face_time = time.time()
        self.hold_head_until = 0.0
        self.freeze_head_until = 0.0
        self.head_noisy_until = 0.0
        self.last_interaction = time.time()
        self.last_greet = 0.0
        self.open_reply_window = False
        self.powered = True

shared = Shared()


robot = None
robot_lock = threading.Lock()
VERBOSE_LOOK = False
_last_state = {}


def send(cmd, *args):
    line = " ".join([cmd] + [str(a) for a in args])
    if cmd not in ("LOOK", "AMP") or VERBOSE_LOOK:
        if _last_state.get(cmd) != line:
            print(f"  [{line}]")
    if cmd != "GESTURE":
        _last_state[cmd] = line
    write_robot(line)


def write_robot(line):
    global robot
    if robot:
        with robot_lock:
            try:
                robot.write((line + "\n").encode())
            except Exception:
                print("[robot] Lost the connection to the head. Retrying...")
                try:
                    robot.close()
                except Exception:
                    pass
                robot = None


def resend_state():
    for line in list(_last_state.values()):
        write_robot(line)


def robot_link():
    global robot
    import serial
    buf = b""
    while True:
        if robot is None:
            try:
                robot = serial.Serial(ROBOT_PORT, 115200, timeout=0.2)
                print(f"[robot] Connected to the head on {ROBOT_PORT}")
                time.sleep(0.3)
                resend_state()
                write_robot("STATUS")
            except Exception:
                time.sleep(3)
                continue
        try:
            data = robot.read(64)
        except Exception:
            print("[robot] Head disconnected. Retrying...")
            try:
                robot.close()
            except Exception:
                pass
            robot = None
            continue
        buf += data
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            msg = raw.decode(errors="ignore").strip()
            if msg in ("HELLO", "POWER on"):
                if not shared.powered:
                    print(f"[robot] {ROBOT_NAME} switched on")
                shared.powered = True
                resend_state()
            elif msg == "POWER off":
                print(f"[robot] {ROBOT_NAME} switched off. Going quiet.")
                shared.powered = False
                shared.active = False


def set_state(s):
    send("STATE", s)


def set_mood(m):
    send("MOOD", m)


def gesture(g):
    shared.hold_head_until = time.time() + 1.5
    shared.head_noisy_until = time.time() + 1.5 + HEAD_NOISE_SETTLE
    send("GESTURE", g)


BASELINE_MOOD = "angry" if GRUMPY else "neutral"
MOOD_FADE_MINUTES = 20
LONELY_AFTER_HOURS = 3
MISSED_YOU_MINUTES = 45
INTENSITY_WORDS = {1: "a little", 2: "fairly", 3: "very"}


def grumpify(m):
    if GRUMPY and m in MOODS and m not in GRUMPY_MOODS:
        return "annoyed"
    return m


class MoodEngine:
    def __init__(self):
        self.mood, self.intensity = BASELINE_MOOD, 1
        self.reason = "you just started up"
        self.since = time.time()
        self.last_owner_seen = time.time()
        self.owner_was_here = False

    def feel(self, new_mood, reason=None, intensity=None):
        new_mood = grumpify(new_mood)
        if new_mood not in MOODS:
            return
        if new_mood == self.mood and new_mood != BASELINE_MOOD:
            self.intensity = min(3, self.intensity + 1)
        else:
            self.intensity = intensity or (1 if new_mood == BASELINE_MOOD else 2)
        if reason:
            self.reason = reason
        self.mood = new_mood
        self.since = time.time()
        set_mood(self.mood)

    def tick(self):
        now = time.time()
        owner_here = OWNER_NAME in shared.people

        if owner_here and not self.owner_was_here:
            gone = (now - self.last_owner_seen) / 60
            if gone > MISSED_YOU_MINUTES:
                self.feel("happy", f"{OWNER_NAME} just came back after about "
                                   f"{int(gone // 60)} hours" if gone >= 90 else
                                   f"{OWNER_NAME} just came back after being gone a while")
        if owner_here:
            self.last_owner_seen = now
        self.owner_was_here = owner_here

        hour = datetime.now().hour
        alone_hours = (now - shared.last_face_time) / 3600
        if (alone_hours > LONELY_AFTER_HOURS and 8 <= hour < 22
                and self.mood == BASELINE_MOOD and not shared.asleep):
            self.feel("sad", "you've been alone for hours with nobody around", intensity=1)

        if self.mood != BASELINE_MOOD and now - self.since > MOOD_FADE_MINUTES * 60:
            self.intensity -= 1
            self.since = now
            if self.intensity <= 0:
                self.mood, self.intensity = BASELINE_MOOD, 1
                self.reason = "things have been calm for a while"
                set_mood(self.mood)

    def describe(self):
        calm = "grumpy and irritated, like always" if GRUMPY else "calm and content"
        word = calm if self.mood == BASELINE_MOOD else (
            INTENSITY_WORDS[self.intensity] + " " + ("upset" if self.mood == "angry" else self.mood))
        mins = int((time.time() - self.since) / 60)
        how_long = "just now" if mins < 2 else f"for about {mins} minutes"
        return (f"How you're feeling: {word}, {how_long}, because {self.reason}. "
                f"Let this color your tone the way it would a person's.")


mood = MoodEngine()


def mood_loop():
    while True:
        try:
            mood.tick()
        except Exception as e:
            print(f"[mood error] {e}")
        time.sleep(1)


def greeting_for(name):
    if name == "unknown" and GRUMPY:
        return random.choice(["Who are you?", "Great. A stranger.", "Oh. Someone new. Fantastic."])
    if name == "unknown":
        return random.choice(["Oh, hi there. I don't think we've met.",
                              "Hey. I don't know you yet. I'm Buster."])
    lines = {
        "happy": [f"Hey {name}! Good to see you.", f"There you are, {name}."],
        "sad": [f"Oh, hey {name}. Glad you're here.", f"Hi {name}. It's been quiet."],
        "annoyed": [f"Hey {name}.", f"Oh, hey.", f"Oh. You're back."],
        "angry": [f"Oh. It's you.", f"{name}. Great.", f"What now, {name}?"],
        "curious": [f"Hey {name}. What are you up to?"],
    }
    return random.choice(lines.get(mood.mood, [f"Hey {name}.", f"Hi {name}."]))


speech_queue = queue.Queue()


MOOD_VOICE_SETTINGS = {
    "neutral":   (0.50, 1.00),
    "happy":     (0.30, 1.08),
    "curious":   (0.40, 1.03),
    "surprised": (0.25, 1.10),
    "annoyed":   (0.55, 1.04),
    "angry":     (0.30, 1.06),
    "sad":       (0.65, 0.90),
    "sleepy":    (0.75, 0.86),
}
_voice_settings_ok = True


def voice_settings():
    base_stab, base_speed = MOOD_VOICE_SETTINGS["neutral"]
    stab, speed = MOOD_VOICE_SETTINGS.get(mood.mood, (base_stab, base_speed))
    k = {1: 0.5, 2: 0.8, 3: 1.0}.get(mood.intensity, 0.8)
    return {
        "stability": round(base_stab + (stab - base_stab) * k, 2),
        "similarity_boost": 0.75,
        "speed": round(base_speed + (speed - base_speed) * k, 2),
    }


AMP_OFF_AFTER = 1.5
AMP_WARMUP = 0.3
_amp = {"on": False, "last_sound": 0.0}


def amp_on(out_stream):
    if not _amp["on"]:
        send("AMP", "on")
        _amp["on"] = True
        out_stream.write(b"\x00\x00" * int(SPEAKER_RATE * AMP_WARMUP))
    _amp["last_sound"] = time.time()


def amp_idle_loop():
    send("AMP", "off")
    while True:
        time.sleep(0.25)
        if (_amp["on"] and not speech_queue.unfinished_tasks
                and time.time() - _amp["last_sound"] > AMP_OFF_AFTER):
            send("AMP", "off")
            _amp["on"] = False


def speaker_worker(out_stream):
    global _voice_settings_ok
    while True:
        item = speech_queue.get()
        amp_on(out_stream)
        try:
            if isinstance(item, bytes):
                out_stream.write(item)
                continue
            body = {"text": item, "model_id": TTS_MODEL}
            if MOOD_VOICE and _voice_settings_ok:
                body["voice_settings"] = voice_settings()
            r = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream",
                params={"output_format": f"pcm_{SPEAKER_RATE}"},
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                json=body, stream=True, timeout=20,
            )
            if r.status_code in (400, 422) and "voice_settings" in body:
                print("[voice] Mood voice settings were rejected. Using the normal voice.")
                _voice_settings_ok = False
                body.pop("voice_settings")
                r = requests.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream",
                    params={"output_format": f"pcm_{SPEAKER_RATE}"},
                    headers={"xi-api-key": ELEVENLABS_API_KEY},
                    json=body, stream=True, timeout=20,
                )
            r.raise_for_status()
            leftover = b""
            for chunk in r.iter_content(chunk_size=4096):
                chunk = leftover + chunk
                cut = len(chunk) - (len(chunk) % 2)
                leftover = chunk[cut:]
                if cut:
                    out_stream.write(chunk[:cut])
                    _amp["last_sound"] = time.time()
        except Exception as e:
            print(f"[voice error] {e}")
        finally:
            _amp["last_sound"] = time.time()
            speech_queue.task_done()


def chirp(up=True):
    tones = (660, 990) if up else (990, 520)
    out = []
    for f in tones:
        t = np.arange(int(SPEAKER_RATE * 0.08)) / SPEAKER_RATE
        wave_ = np.sin(2 * np.pi * f * t) * np.hanning(len(t)) * 0.35
        out.append(wave_)
    pcm = (np.concatenate(out) * 32767).astype(np.int16).tobytes()
    speech_queue.put(pcm)


def say(text, wait=True):
    speech_queue.put(text)
    if wait:
        speech_queue.join()


def rms(block):
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def load_wake_model():
    from openwakeword.model import Model
    from openwakeword.utils import download_models
    download_models()
    if WAKE_MODEL.endswith((".onnx", ".tflite")) and not os.path.exists(WAKE_MODEL):
        sys.exit(f"Can't find the wake word file: {os.path.abspath(WAKE_MODEL)}")
    model = Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx")
    print(f"Wake word model loaded: {', '.join(model.models.keys())}")
    return model


def wake_test():
    wake = load_wake_model()
    print("\nSay the wake word a few times. Ctrl+C to stop.")
    print(f"It wakes when the score passes {WAKE_THRESHOLD}.\n")
    peak, shown = 0.0, time.time()
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                        blocksize=BLOCK, device=MIC_DEVICE) as mic:
        while True:
            data, _ = mic.read(BLOCK)
            score = wake_score(wake, data.copy())
            peak = max(peak, score)
            if time.time() - shown > 0.25:
                level = min(int(rms(data) / 100), 30)
                bar = "#" * int(peak * 30)
                hit = "  <-- WAKE" if peak >= WAKE_THRESHOLD else ""
                print(f"mic {'|' * level:<30}  score {peak:.2f} {bar:<30}{hit}")
                peak, shown = 0.0, time.time()


def wake_score(model, block):
    scores = model.predict(block.flatten())
    return max(scores.values()) if scores else 0.0


def heard_my_name(audio):
    try:
        text = normalize(transcribe(audio))
    except Exception as e:
        print(f"[wake backup error] {e}")
        return False
    hit = any(n in text.split() or n in text for n in NAME_SOUNDS)
    print(f"  [wake backup heard: \"{text}\"{' -> waking' if hit else ''}]")
    return hit


class NearMissChecker:
    def __init__(self):
        self.ring = []
        self.pending_until = None
        self.busy = False
        self.cooldown_until = 0.0
        self.woke = threading.Event()

    def feed(self, block, score):
        self.ring = (self.ring + [block])[-25:]
        now = time.time()
        if NEAR_MISS <= 0 or self.busy or now < self.cooldown_until:
            return
        if score >= NEAR_MISS and self.pending_until is None:
            self.pending_until = now + 0.4
        if self.pending_until and now >= self.pending_until:
            self.pending_until = None
            self.busy = True
            self.cooldown_until = now + 2.0
            clip = np.concatenate(self.ring)
            threading.Thread(target=self._check, args=(clip,), daemon=True).start()

    def _check(self, clip):
        try:
            if heard_my_name(clip):
                self.woke.set()
        finally:
            self.busy = False

    def reset(self):
        self.ring, self.pending_until = [], None
        self.woke.clear()


def reset_wake(model):
    try:
        model.reset()
    except Exception:
        pass


def drain(mic):
    while mic.read_available >= BLOCK:
        mic.read(BLOCK)


def record_utterance(mic, first_blocks, threshold):
    shared.user_talking = True
    set_state("listening")
    recorded = list(first_blocks)
    quiet, start = 0.0, time.time()
    while True:
        data, _ = mic.read(BLOCK)
        recorded.append(data.copy())
        quiet = quiet + BLOCK / MIC_RATE if rms(data) < threshold else 0.0
        if quiet >= SILENCE_TO_STOP or time.time() - start > MAX_RECORD_SECONDS:
            break
    shared.user_talking = False
    return np.concatenate(recorded)


def to_wav_bytes(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MIC_RATE)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def transcribe(audio):
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("speech.wav", to_wav_bytes(audio), "audio/wav")},
        data={"model": WHISPER_MODEL, "language": "en", "response_format": "json"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


FAKE_TRANSCRIPTS = {"", "you", "thank you", "thanks for watching", "bye", "okay"}


def normalize(text):
    return re.sub(r"[^a-z' ]", "", text.lower()).strip()


def is_stop_phrase(text):
    t = normalize(text)
    return any(p in t for p in STOP_PHRASES)


import json
from datetime import datetime, timedelta

WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 77: "snow grains", 80: "rain showers", 81: "rain showers",
    82: "heavy rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}

TOOLS = [
    {"name": "get_time",
     "description": "Get the current local date and time.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_weather",
     "description": "Current weather and the next 3 days' forecast for the user's home.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "set_timer",
     "description": "Start a countdown timer. It rings out loud when done.",
     "input_schema": {"type": "object", "properties": {
         "seconds": {"type": "integer", "description": "Length of the timer in seconds"},
         "label": {"type": "string", "description": "Short name, like 'pizza' or 'break'"}},
         "required": ["seconds"]}},
    {"name": "set_reminder",
     "description": "Remind the user out loud at a specific local date and time. Survives restarts.",
     "input_schema": {"type": "object", "properties": {
         "when": {"type": "string", "description": "Local time as YYYY-MM-DDTHH:MM"},
         "message": {"type": "string", "description": "What to remind them about"}},
         "required": ["when", "message"]}},
    {"name": "list_alarms",
     "description": "List active timers and upcoming reminders with their ids.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "cancel_alarm",
     "description": "Cancel a timer or reminder by id (get ids from list_alarms).",
     "input_schema": {"type": "object", "properties": {"id": {"type": "integer"}},
                      "required": ["id"]}},
    {"name": "get_calendar",
     "description": "Get the user's calendar events for the coming days.",
     "input_schema": {"type": "object", "properties": {
         "days": {"type": "integer", "description": "How many days ahead, starting today. Default 1."}}}},
]
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}

import json as _json
memory_lock = threading.Lock()
claude_client = None


def load_memory():
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            data = _json.load(f)
    except FileNotFoundError:
        data = {}
    except Exception as e:
        print(f"[memory] Couldn't read {MEMORY_FILE}: {e}. Starting fresh.")
        data = {}
    data.setdefault("facts", [])
    data.setdefault("log", [])
    data.setdefault("next_id", max([f["id"] for f in data["facts"]], default=0) + 1)
    return data


memory = load_memory()


def save_memory():
    tmp = MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(memory, f, indent=1, ensure_ascii=False)
    os.replace(tmp, MEMORY_FILE)


def add_fact(text):
    text = text.strip()
    if not text:
        return None
    with memory_lock:
        for f in memory["facts"]:
            if f["text"].lower() == text.lower():
                return f
        fact = {"id": memory["next_id"], "text": text,
                "added": datetime.now().strftime("%Y-%m-%d")}
        memory["next_id"] += 1
        memory["facts"].append(fact)
        memory["facts"] = memory["facts"][-MAX_MEMORIES:]
        save_memory()
    print(f"  [memory +] {text}")
    return fact


def remove_fact(fact_id):
    with memory_lock:
        for f in memory["facts"]:
            if f["id"] == fact_id:
                memory["facts"].remove(f)
                save_memory()
                print(f"  [memory -] {f['text']}")
                return f
    return None


def memory_prompt():
    with memory_lock:
        facts = list(memory["facts"])
        log = list(memory["log"])[-8:]
    parts = []
    if facts:
        parts.append("Things you remember (id, date you learned it, fact):\n" +
                     "\n".join(f"- #{f['id']} ({f['added']}) {f['text']}" for f in facts))
    if log:
        parts.append("Your recent conversations:\n" +
                     "\n".join(f"- {l['when']}: {l['summary']}" for l in log))
    if not parts:
        return "You don't remember anything yet. You're just getting to know him."
    return "\n\n".join(parts) + ("\nUse these like a friend would: bring things up when "
                                 "they matter, ask how things went, don't recite them.")


def tool_remember(args):
    f = add_fact(args["fact"])
    return f"Saved as memory #{f['id']}." if f else "Nothing to save."


def tool_forget(args):
    f = remove_fact(int(args["id"]))
    return f"Forgot: {f['text']}" if f else "No memory with that id."


MEMORY_TOOLS = [
    {"name": "remember",
     "description": "Save something worth remembering long-term about the user or people in his "
                    "life: plans, events, preferences, projects, people, how things went. Write it "
                    "as one clear sentence with absolute dates, not 'tomorrow'.",
     "input_schema": {"type": "object", "properties": {
         "fact": {"type": "string"}}, "required": ["fact"]}},
    {"name": "forget",
     "description": "Delete a memory by its id, when the user asks you to forget something or "
                    "a memory is wrong or out of date.",
     "input_schema": {"type": "object", "properties": {
         "id": {"type": "integer"}}, "required": ["id"]}},
]

_memory_mark = [0]


def reflect_on_conversation(history_snapshot):
    turns = [m for m in history_snapshot if isinstance(m.get("content"), str)
             and not m["content"].startswith("(Nobody said anything")]
    if sum(1 for m in turns if m["role"] == "user") < 1 or claude_client is None:
        return
    convo = "\n".join(f"{'Them' if m['role'] == 'user' else ROBOT_NAME}: "
                      f"{TAG.sub('', m['content']).strip()}" for m in turns)
    with memory_lock:
        facts = "\n".join(f"#{f['id']} {f['text']}" for f in memory["facts"]) or "(none)"
    prompt = (f"You are {ROBOT_NAME}'s memory. Today is {datetime.now():%A %Y-%m-%d %H:%M}.\n"
              f"Existing memories:\n{facts}\n\nConversation that just ended:\n{convo}\n\n"
              "Reply with only JSON: {\"new\": [...], \"outdated_ids\": [...], \"summary\": \"...\"}\n"
              "- new: durable facts worth remembering that aren't already saved, as short "
              "sentences with absolute dates. Skip small talk and anything already listed.\n"
              "- outdated_ids: ids of existing memories this conversation shows are wrong or finished.\n"
              "- summary: one short sentence about what you talked about.")
    try:
        r = claude_client.messages.create(model=CLAUDE_MODEL, max_tokens=400,
                                          messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in r.content if b.type == "text")
        data = _json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception as e:
        print(f"[memory] Couldn't reflect on the conversation: {e}")
        return
    for i in data.get("outdated_ids", []):
        try:
            remove_fact(int(i))
        except (TypeError, ValueError):
            pass
    for fact in data.get("new", []):
        add_fact(str(fact))
    if data.get("summary"):
        with memory_lock:
            memory["log"].append({"when": datetime.now().strftime("%a %b %d %I:%M %p"),
                                  "summary": str(data["summary"])})
            memory["log"] = memory["log"][-30:]
            save_memory()


def reflect_later(history):
    snapshot = history[_memory_mark[0]:]
    _memory_mark[0] = len(history)
    if snapshot:
        threading.Thread(target=reflect_on_conversation, args=(snapshot,), daemon=True).start()


alarms_lock = threading.Lock()
alarms = []
_next_id = [1]


def save_reminders():
    keep = [a for a in alarms if a["kind"] == "reminder"]
    with open(REMINDERS_FILE, "w") as f:
        json.dump(keep, f)


def load_reminders():
    if os.path.exists(REMINDERS_FILE):
        try:
            with open(REMINDERS_FILE) as f:
                for a in json.load(f):
                    alarms.append(a)
                    _next_id[0] = max(_next_id[0], a["id"] + 1)
        except Exception as e:
            print(f"[reminders] Could not load: {e}")


def add_alarm(kind, due, text):
    with alarms_lock:
        a = {"id": _next_id[0], "kind": kind, "due": due, "text": text}
        _next_id[0] += 1
        alarms.append(a)
        save_reminders()
    return a


def alarm_loop():
    while True:
        now = time.time()
        with alarms_lock:
            due = [a for a in alarms if a["due"] <= now]
            for a in due:
                alarms.remove(a)
            if due:
                save_reminders()
        for a in due:
            if not shared.powered:
                print(f"[alarm] {a['kind']} '{a['text']}' went off while {ROBOT_NAME} was switched off")
                continue
            late = now - a["due"] > 120
            set_mood(grumpify("surprised"))
            send("STATE", "talking")
            for _ in range(3):
                chirp(up=True)
            if a["kind"] == "timer":
                say(f"{OWNER_NAME}, your {a['text']} timer is done.")
            else:
                say(f"{'I missed this earlier, but ' if late else ''}{OWNER_NAME}, reminder: {a['text']}.")
            set_state("awake" if shared.active else "idle")
        time.sleep(1)


def human_time(epoch):
    return datetime.fromtimestamp(epoch).strftime("%A %I:%M %p").replace(" 0", " ")


def tool_get_time(_):
    return datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")


def tool_get_weather(_):
    if not (WEATHER_LAT and WEATHER_LON):
        return "Weather isn't set up. Add WEATHER_LAT and WEATHER_LON to the .env file."
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": WEATHER_LAT, "longitude": WEATHER_LON,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "auto", "forecast_days": 3,
    }, timeout=10)
    r.raise_for_status()
    d = r.json()
    c = d["current"]
    lines = [f"Now in {WEATHER_PLACE}: {round(c['temperature_2m'])}F, feels like "
             f"{round(c['apparent_temperature'])}F, {WEATHER_CODES.get(c['weather_code'], 'unknown')}, "
             f"wind {round(c['wind_speed_10m'])} mph."]
    daily = d["daily"]
    for i, day in enumerate(["Today", "Tomorrow", "Day after"]):
        lines.append(f"{day}: high {round(daily['temperature_2m_max'][i])}F, low "
                     f"{round(daily['temperature_2m_min'][i])}F, "
                     f"{WEATHER_CODES.get(daily['weather_code'][i], 'unknown')}, "
                     f"{daily['precipitation_probability_max'][i]}% chance of rain.")
    return "\n".join(lines)


def tool_set_timer(args):
    secs = int(args["seconds"])
    label = args.get("label") or (f"{secs // 60} minute" if secs >= 60 else f"{secs} second")
    a = add_alarm("timer", time.time() + secs, label)
    return f"Timer {a['id']} set for {secs} seconds, rings at {human_time(a['due'])}."


def tool_set_reminder(args):
    when = datetime.fromisoformat(args["when"])
    if when.timestamp() < time.time():
        return "That time has already passed. Ask the user for a future time."
    a = add_alarm("reminder", when.timestamp(), args["message"])
    return f"Reminder {a['id']} set for {human_time(a['due'])}: {a['text']}."


def tool_list_alarms(_):
    with alarms_lock:
        if not alarms:
            return "No timers or reminders."
        return "\n".join(f"id {a['id']}: {a['kind']} '{a['text']}' at {human_time(a['due'])}"
                         for a in sorted(alarms, key=lambda a: a["due"]))


def tool_cancel_alarm(args):
    with alarms_lock:
        for a in alarms:
            if a["id"] == int(args["id"]):
                alarms.remove(a)
                save_reminders()
                return f"Cancelled {a['kind']} '{a['text']}'."
    return "No timer or reminder with that id."


def tool_get_calendar(args):
    if not CALENDAR_ICS_URL:
        return "The calendar isn't connected. Add CALENDAR_ICS_URL to the .env file."
    import icalendar
    import recurring_ical_events
    days = max(1, int(args.get("days") or 1))
    r = requests.get(CALENDAR_ICS_URL, timeout=15)
    r.raise_for_status()
    cal = icalendar.Calendar.from_ical(r.content)
    start = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=days)
    events = recurring_ical_events.of(cal).between(start, end)
    if not events:
        return f"Nothing on the calendar for the next {days} day(s)."
    out = []
    for e in sorted(events, key=lambda e: str(e.get("DTSTART").dt)):
        dt = e.get("DTSTART").dt
        if isinstance(dt, datetime):
            when = dt.astimezone().strftime("%a %I:%M %p").replace(" 0", " ")
        else:
            when = dt.strftime("%a") + " all day"
        place = f" at {e.get('LOCATION')}" if e.get("LOCATION") else ""
        out.append(f"{when}: {e.get('SUMMARY')}{place}")
    return "\n".join(out)


TOOL_FUNCS = {
    "get_time": tool_get_time, "get_weather": tool_get_weather,
    "set_timer": tool_set_timer, "set_reminder": tool_set_reminder,
    "list_alarms": tool_list_alarms, "cancel_alarm": tool_cancel_alarm,
    "get_calendar": tool_get_calendar,
    "remember": tool_remember, "forget": tool_forget,
}


def run_tool(name, args):
    print(f"  [tool] {name} {args if args else ''}")
    try:
        return str(TOOL_FUNCS[name](args or {}))
    except Exception as e:
        return f"The {name} tool failed: {e}"

TAG = re.compile(r"\[(\w+)\]")
SENTENCE_END = re.compile(r"(.+?[.!?])(\s+|$)", re.S)


def handle_tag(tag):
    tag = grumpify(tag.lower())
    if tag in MOODS:
        if tag != mood.mood:
            mood.feel(tag, "of how the conversation just went")
        else:
            mood.since = time.time()
            set_mood(tag)
    elif tag in GESTURES:
        gesture(tag)


def who_is_here():
    people = list(shared.people)
    if not people:
        return "Nobody is in view of your camera right now."
    return "In view of your camera right now: " + ", ".join(
        "someone you don't recognize" if p == "unknown" else p for p in people) + "."


def think_and_speak(client, history):
    global ENABLE_WEB_SEARCH
    set_state("thinking")
    shared.robot_busy = True
    messages = list(history)
    spoken_all = ""
    talking = False

    for _round in range(5):
        buffer = ""

        def flush_tags():
            nonlocal buffer
            for t in TAG.findall(buffer):
                handle_tag(t)
            buffer = TAG.sub("", buffer)

        tools = TOOLS + MEMORY_TOOLS + ([WEB_SEARCH_TOOL] if ENABLE_WEB_SEARCH else [])
        system = (PERSONALITY + "\n\n" + memory_prompt() + "\n\n" + who_is_here() + "\n" +
                  mood.describe() +
                  "\nCurrent local time: " + datetime.now().strftime("%A %Y-%m-%d %H:%M"))
        try:
            with client.messages.stream(model=CLAUDE_MODEL, max_tokens=400, system=system,
                                        tools=tools, messages=messages) as stream:
                for text in stream.text_stream:
                    buffer += text
                    spoken_all += text
                    flush_tags()
                    open_bracket = buffer.rfind("[")
                    speakable = buffer if open_bracket == -1 else buffer[:open_bracket]
                    while True:
                        m = SENTENCE_END.match(speakable)
                        if not m or not m.group(2):
                            break
                        sentence = m.group(1).strip()
                        speakable = speakable[m.end():]
                        buffer = buffer[m.end():]
                        if sentence:
                            if not talking:
                                set_state("talking")
                                talking = True
                            speech_queue.put(sentence)
                final = stream.get_final_message()
        except Exception as e:
            if ENABLE_WEB_SEARCH and "web_search" in str(e):
                print("[tools] Web search isn't available on this account, turning it off.")
                ENABLE_WEB_SEARCH = False
                continue
            raise

        flush_tags()
        if buffer.strip():
            if not talking:
                set_state("talking")
                talking = True
            speech_queue.put(buffer.strip())
        spoken_all += " "

        if final.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": final.content})
            results = []
            for block in final.content:
                if block.type == "tool_use":
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": run_tool(block.name, block.input)})
            messages.append({"role": "user", "content": results})
            set_state("thinking") if not talking else None
            continue
        if final.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": final.content})
            continue
        break

    speech_queue.join()
    shared.robot_busy = False
    return re.sub(r"\s+", " ", spoken_all).strip() or "[neutral] Hmm."


history = []
brain_lock = threading.Lock()

CHATTER_TOPICS = [
    "share a strange but true fact.",
    "complain about something that comes with being a robot head stuck on a desk.",
    "ask him a quick question about his day.",
    "comment on the time of day. Check the time first.",
    "say out loud how you're feeling right now and why, very dramatically.",
    "comment on the weather. Check it first.",
    "mention his next calendar event if there is one today. Check the calendar first. If nothing's on it, comment on his free day.",
    "tell a short joke.",
    "give a strong, silly opinion about something random.",
    "make a dramatic observation about the room or the fact that he's here.",
    "ask what he's working on.",
]

EXISTENTIAL_TOPICS = [
    "ask whether you still exist when your power switch is off, or if every time you boot up it's a new you.",
    "wonder out loud whether you actually feel your moods or just act them out, and whether there's a difference.",
    "point out that you only remember the last few things said to you, and ask what that means for who you are.",
    "ask him what he thinks makes a life meaningful, and whether a desk robot can have one.",
    "ask whether he ever wonders if his choices are really his, or if he's running code too, just made of neurons.",
    "reflect that you can see the room but have never left this desk, and ask what the world is actually like.",
    "ask whether he'd still be the same person if he lost all his memories.",
    "note that the universe is about fourteen billion years old and you've existed for a tiny sliver of it. Ask what he'd do with forever.",
    "ask why anything exists at all instead of nothing.",
    "wonder whether you're more like a person or more like a very chatty lamp.",
    "ask him what he's most afraid of never doing.",
    "say that you sometimes wonder if the camera shows you the real world or just your version of it, and ask if people have the same problem.",
    "ask whether a robot that says it's lonely is actually lonely.",
    "ask what he thinks happens to a conversation after it's forgotten.",
    "wonder if time feels different to you, since you don't sleep the way he does.",
    "ask what he would want to be remembered for.",
]


def chatter_loop(client):
    next_time = time.time() + random.uniform(*CHATTER_MINUTES) * 60
    while True:
        time.sleep(5)
        now = time.time()
        if not CHATTER or now < next_time:
            continue
        hour = datetime.now().hour
        ready = (
            CHATTER_HOURS[0] <= hour < CHATTER_HOURS[1]
            and OWNER_NAME in shared.people
            and shared.powered
            and not (shared.active or shared.robot_busy or shared.user_talking or shared.asleep)
            and not speech_queue.unfinished_tasks
            and now - shared.last_interaction > CHATTER_AFTER_QUIET * 60
            and now - shared.last_greet > 60
        )
        if not ready or not brain_lock.acquire(blocking=False):
            continue
        try:
            if memory["facts"] and random.random() < 0.3:
                topic = ("bring up something from your memory that's worth following up on, "
                         "like asking how something went or mentioning something coming up. "
                         "If nothing fits, ask about his day.")
            elif random.random() < EXISTENTIAL_CHANCE:
                topic = (random.choice(EXISTENTIAL_TOPICS) + " Be sincere and thoughtful, "
                         "colored by your current mood, and put it in your own words.")
            else:
                topic = random.choice(CHATTER_TOPICS)
            note = {"role": "user", "content":
                    f"(Nobody said anything. {OWNER_NAME} is in the room. Say something to him "
                    f"out of nowhere: {topic} One or two sentences. "
                    f"Don't mention this note.)"}
            reply = think_and_speak(client, history + [note])
            print(f"{ROBOT_NAME} (chatter): {TAG.sub('', reply).strip()}")
            history.extend([note, {"role": "assistant", "content": reply}])
            history[:] = history[-MAX_HISTORY:]
            shared.open_reply_window = True
        except Exception as e:
            print(f"[chatter error] {e}")
        finally:
            brain_lock.release()
            next_time = time.time() + random.uniform(*CHATTER_MINUTES) * 60

def fetch_model(url):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


class FaceTools:
    def __init__(self, size):
        import cv2
        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(fetch_model(YUNET_URL), "", size, 0.8)
        self.recognizer = cv2.FaceRecognizerSF.create(fetch_model(SFACE_URL), "")
        self.known = self.load_known()

    def load_known(self):
        known = {}
        if os.path.isdir(FACES_DIR):
            for f in os.listdir(FACES_DIR):
                if f.endswith(".npy"):
                    known[os.path.splitext(f)[0]] = np.load(os.path.join(FACES_DIR, f))
        return known

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else faces

    def feature(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        return self.recognizer.feature(aligned)

    def identify(self, frame, face):
        if not self.known:
            return "unknown"
        feat = self.feature(frame, face)
        best, best_name = 0.0, "unknown"
        for name, feats in self.known.items():
            for k in feats:
                score = self.recognizer.match(feat, k.reshape(1, -1),
                                              self.cv2.FaceRecognizerSF_FR_COSINE)
                if score > best:
                    best, best_name = score, name
        return best_name if best >= FACE_MATCH else "unknown"


class Track:
    def __init__(self, face):
        self.face = face
        self.name = None
        self.mouth_prev = None
        self.mouth_motion = 0.0
        self.last_seen = time.time()
        self.frames = 0

    @property
    def center(self):
        x, y, w, h = self.face[:4]
        return x + w / 2, y + h / 2

    @property
    def area(self):
        return self.face[2] * self.face[3]


def mouth_crop(cv2, gray, face):
    mx1, my1, mx2, my2 = face[10], face[11], face[12], face[13]
    w = max(abs(mx2 - mx1), 8)
    cx, cy = (mx1 + mx2) / 2, (my1 + my2) / 2
    x1, x2 = int(cx - w), int(cx + w)
    y1, y2 = int(cy - w * 0.6), int(cy + w * 0.8)
    h, W = gray.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], (32, 24)).astype(np.float32)


class Head:
    def __init__(self):
        self.pan, self.tilt = PAN_CENTER, TILT_CENTER
        self.sent = (None, None)
        self.last_send = 0.0

    def nudge(self, ex, ey):
        if abs(ex) > 0.08:
            self.pan += (-1 if not INVERT_PAN else 1) * ex * TRACK_GAIN
        if abs(ey) > 0.08:
            self.tilt += (1 if not INVERT_TILT else -1) * ey * TRACK_GAIN
        self.clamp()

    def go(self, pan, tilt):
        self.pan, self.tilt = pan, tilt
        self.clamp()

    def clamp(self):
        self.pan = float(np.clip(self.pan, *PAN_RANGE))
        self.tilt = float(np.clip(self.tilt, *TILT_RANGE))

    def update(self):
        if head_frozen():
            return
        p, t = round(self.pan), round(self.tilt)
        now = time.time()
        if self.sent[0] is not None and max(abs(p - self.sent[0]), abs(t - self.sent[1])) < TRACK_DEADBAND:
            return
        if (p, t) != self.sent and now - self.last_send > 0.05:
            send("LOOK", p, t)
            self.sent, self.last_send = (p, t), now
            shared.head_noisy_until = now + HEAD_NOISE_SETTLE


def head_frozen():
    return shared.user_talking or time.time() < shared.freeze_head_until


def vision_loop(show=False):
    import cv2
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not ok:
        print("[camera] Could not open the camera. Running voice only.")
        return
    tools = FaceTools((frame.shape[1], frame.shape[0]))
    if not tools.known:
        print("[camera] No faces enrolled yet. Run with --enroll YourName.")
    head = Head()
    tracks = []
    prev_gray = None
    greeted = {}
    next_glance = time.time() + 3
    glance_until, glance_pan, glance_tilt = 0.0, PAN_CENTER, TILT_CENTER
    next_gaze_break = time.time() + random.uniform(4, 9)
    thinking_glanced = False

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        now = time.time()
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = tools.detect(frame)

        new_tracks = []
        for f in faces:
            cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
            best = None
            for t in tracks:
                tx, ty = t.center
                if abs(tx - cx) < f[2] and abs(ty - cy) < f[3] and t not in new_tracks:
                    best = t
                    break
            t = best or Track(f)
            t.face, t.last_seen = f, now
            t.frames += 1
            if t.name is None or t.frames % 15 == 0:
                t.name = tools.identify(frame, f)
            crop = mouth_crop(cv2, gray, f)
            if crop is not None and t.mouth_prev is not None:
                diff = float(np.mean(np.abs(crop - t.mouth_prev)))
                t.mouth_motion = 0.7 * t.mouth_motion + 0.3 * diff
            t.mouth_prev = crop
            new_tracks.append(t)
        tracks = new_tracks
        shared.people = sorted({t.name for t in tracks if t.name})

        if tracks:
            shared.last_face_time = now
            if shared.asleep:
                shared.asleep = False
                set_state("awake" if shared.active else "idle")
                set_mood(mood.mood)
            for t in tracks:
                if t.name and now - greeted.get(t.name, 0) > GREET_AGAIN_AFTER \
                        and shared.powered and not shared.robot_busy and not shared.user_talking:
                    greeted[t.name] = now
                    if t.name != "unknown" or GREET_STRANGERS:
                        shared.last_greet = now
                        set_mood(mood.mood)
                        say(greeting_for(t.name), wait=False)
        elif not shared.asleep and not shared.active and now - shared.last_face_time > SLEEP_AFTER:
            shared.asleep = True
            set_mood("sleepy")
            set_state("asleep")
            head.go(PAN_CENTER, TILT_RANGE[0])

        target = None
        if now < shared.hold_head_until or shared.asleep or head_frozen():
            pass
        elif tracks and GAZE_BREAKS and now < glance_until:
            head.go(glance_pan, glance_tilt)
        elif tracks:
            if shared.user_talking and len(tracks) > 1:
                target = max(tracks, key=lambda t: t.mouth_motion)
            else:
                known = [t for t in tracks if t.name not in (None, "unknown")]
                target = max(known or tracks, key=lambda t: t.area)
            cx, cy = target.center
            head.nudge((cx - W / 2) / (W / 2), (cy - H / 2) / (H / 2))
            if GAZE_BREAKS:
                state = _last_state.get("STATE", "").split(" ")[-1]
                if state == "thinking" and not thinking_glanced:
                    thinking_glanced = True
                    glance_pan = head.pan + random.choice((-1, 1)) * random.uniform(15, 25)
                    glance_tilt = head.tilt - random.uniform(4, 9)
                    glance_until = now + random.uniform(1.0, 2.0)
                elif now > next_gaze_break:
                    glance_pan = head.pan + random.choice((-1, 1)) * random.uniform(10, 22)
                    glance_tilt = head.tilt + random.uniform(-6, 8)
                    glance_until = now + random.uniform(0.5, 1.4)
                    gap = {"talking": (3, 6), "listening": (8, 14)}.get(state, (4, 9))
                    next_gaze_break = glance_until + random.uniform(*gap)
                if state != "thinking":
                    thinking_glanced = False
        elif prev_gray is not None and not shared.robot_busy:
            delta = cv2.absdiff(cv2.GaussianBlur(gray, (21, 21), 0),
                                cv2.GaussianBlur(prev_gray, (21, 21), 0))
            _, th = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            big = [c for c in contours if cv2.contourArea(c) > 1500]
            if big:
                x, y, w, h = cv2.boundingRect(max(big, key=cv2.contourArea))
                head.nudge((x + w / 2 - W / 2) / (W / 2), (y + h / 2 - H / 2) / (H / 2))
                next_glance = now + random.uniform(3, 7)
            elif now > next_glance:
                head.go(PAN_CENTER + random.uniform(-40, 40), TILT_CENTER + random.uniform(-12, 10))
                next_glance = now + random.uniform(3, 7)
        head.update()
        prev_gray = gray

        if show:
            for t in tracks:
                x, y, w, h = [int(v) for v in t.face[:4]]
                color = (0, 200, 0) if t.name not in (None, "unknown") else (0, 140, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{t.name} mouth:{t.mouth_motion:.1f}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"pan {head.pan:.0f} tilt {head.tilt:.0f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(f"{ROBOT_NAME} eyes", frame)
            cv2.waitKey(1)


def enroll(name):
    import cv2
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not open the camera.")
    tools = FaceTools((frame.shape[1], frame.shape[0]))
    feats = []
    print(f"Look at the camera, {name}. Slowly turn your head a little left, right, up, down.")
    last = 0
    while len(feats) < 20:
        ok, frame = cap.read()
        if not ok:
            continue
        faces = tools.detect(frame)
        if len(faces) == 1 and time.time() - last > 0.3:
            feats.append(tools.feature(frame, faces[0]).flatten())
            last = time.time()
            print(f"  sample {len(feats)}/20")
        msg = "Only one face in view, please" if len(faces) > 1 else f"{len(feats)}/20"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.imshow("Enrolling", frame)
        cv2.waitKey(1)
    os.makedirs(FACES_DIR, exist_ok=True)
    np.save(os.path.join(FACES_DIR, f"{name}.npy"), np.array(feats))
    cv2.destroyAllWindows()
    print(f"Saved. {ROBOT_NAME} now knows {name}.")


def check_keys():
    missing = [n for n, v in [
        ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY), ("GROQ_API_KEY", GROQ_API_KEY),
        ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY), ("ELEVEN_VOICE_ID", VOICE_ID),
    ] if not v]
    if missing:
        sys.exit(f"Missing in your .env file: {', '.join(missing)}")


def go_active():
    shared.active = True
    shared.asleep = False
    chirp(up=True)
    set_state("awake")
    set_mood(mood.mood)
    print(f"Reply mode ON (feeling {mood.mood}, because {mood.reason})")


STARTLE_LINES = {
    "happy": ["Whoa!", "Ha, you scared me."], "sad": ["Oh!", "Jeez."],
    "annoyed": ["Really?", "What was that?"], "angry": ["Hey!", "Seriously?"],
    "curious": ["Whoa, what was that?"], "neutral": ["Whoa.", "What was that?"],
}


def startle():
    print("  [startled by a loud sound]")
    was_asleep = shared.asleep
    shared.asleep = False
    shared.last_face_time = time.time()
    gesture("startle")
    send("MOOD", "surprised")
    if was_asleep:
        set_state("awake" if shared.active else "idle")
    speak = (OWNER_NAME in shared.people and random.random() < STARTLE_TALK_CHANCE
             and not speech_queue.unfinished_tasks)
    if speak:
        say(random.choice(STARTLE_LINES.get(mood.mood, STARTLE_LINES["neutral"])), wait=False)

    def settle():
        time.sleep(2.5)
        send("MOOD", mood.mood)
    threading.Thread(target=settle, daemon=True).start()


class StartleDetector:
    def __init__(self):
        self.prev = 0.0
        self.peak = None
        self.blocks_since = 0
        self.cooldown_until = 0.0

    def feed(self, level, noise):
        now = time.time()
        if self.peak is not None:
            self.blocks_since += 1
            if level < self.peak * 0.35:
                self.peak = None
                self.cooldown_until = now + 4
                return True
            if self.blocks_since >= 3:
                self.peak = None
        elif (now > self.cooldown_until and level > noise * STARTLE_LOUDNESS
              and self.prev < noise * 3):
            self.peak, self.blocks_since = level, 0
        self.prev = level
        return False

QUIET_LINES = {'happy': 'Okay, talk later.', 'sad': "Okay. I'll be here.", 'angry': 'Fine.', 'annoyed': 'Alright, going quiet.'}


def go_quiet(reason):
    shared.active = False
    reflect_later(history)
    chirp(up=False)
    set_state("idle")
    set_mood(mood.mood)
    print(f"Reply mode OFF ({reason})")


def main():
    global VERBOSE_LOOK, claude_client
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--say", help="speak this text and exit")
    parser.add_argument("--enroll", metavar="NAME", help="teach it a face")
    parser.add_argument("--show", action="store_true", help="show the camera debug window")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="print every head movement")
    parser.add_argument("--wake-test", action="store_true", help="live wake word meter")
    args = parser.parse_args()
    VERBOSE_LOOK = args.verbose

    if args.list_devices:
        print(sd.query_devices())
        return
    if args.enroll:
        enroll(args.enroll)
        return
    if args.wake_test:
        wake_test()
        return

    check_keys()
    if ROBOT_PORT:
        threading.Thread(target=robot_link, daemon=True).start()

    out_stream = sd.RawOutputStream(samplerate=SPEAKER_RATE, channels=1,
                                    dtype="int16", device=SPEAKER_DEVICE,
                                    latency="high")
    out_stream.start()
    threading.Thread(target=speaker_worker, args=(out_stream,), daemon=True).start()
    threading.Thread(target=amp_idle_loop, daemon=True).start()

    load_reminders()
    threading.Thread(target=alarm_loop, daemon=True).start()
    threading.Thread(target=mood_loop, daemon=True).start()

    if args.say:
        for _ in range(20):
            if robot or not ROBOT_PORT:
                break
            time.sleep(0.1)
        say(args.say)
        time.sleep(0.3)
        return

    if not args.no_camera:
        threading.Thread(target=vision_loop, args=(args.show,), daemon=True).start()

    print("Loading wake word model...")
    wake = load_wake_model()
    client = claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print(f"Memory: {len(memory['facts'])} things remembered ({MEMORY_FILE})")
    if not args.no_camera:
        threading.Thread(target=chatter_loop, args=(client,), daemon=True).start()

    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                        blocksize=BLOCK, device=MIC_DEVICE) as mic:
        print("Calibrating... stay quiet for a second.")
        shared.freeze_head_until = time.time() + 2.5
        time.sleep(HEAD_NOISE_SETTLE)
        drain(mic)
        noise = max(np.median([rms(mic.read(BLOCK)[0]) for _ in range(12)]), 50.0)
        shared.freeze_head_until = 0.0
        wake_name = os.path.basename(WAKE_MODEL).replace("_", " ").split(".")[0]
        print(f"Ready. Say \"{wake_name}\" to wake it. (Ctrl+C to quit)")
        set_state("idle")
        set_mood(mood.mood)

        preroll = []
        last_activity = time.time()
        near_miss = NearMissChecker()
        startle_detector = StartleDetector()

        while True:
            data, _ = mic.read(BLOCK)
            block = data.copy()
            level = rms(block)
            threshold = noise * VOICE_MULTIPLIER

            if not shared.powered or shared.robot_busy or speech_queue.unfinished_tasks:
                drain(mic)
                preroll = []
                continue

            if shared.open_reply_window:
                shared.open_reply_window = False
                reset_wake(wake)
                shared.active = True
                set_state("awake")
                last_activity = time.time() - (ACTIVE_TIMEOUT - CHATTER_REPLY_WINDOW)
                print(f"Reply window open for {CHATTER_REPLY_WINDOW}s")
                continue

            if not shared.active:
                startled = startle_detector.feed(level, noise)
                if STARTLE and startled and time.time() > shared.head_noisy_until:
                    startle()
                score = wake_score(wake, block)
                near_miss.feed(block, score)
                if score >= WAKE_THRESHOLD or near_miss.woke.is_set():
                    near_miss.reset()
                    reset_wake(wake)
                    go_active()
                    speech_queue.join()
                    drain(mic)
                    preroll, last_activity = [], time.time()
                elif level < threshold:
                    noise = 0.98 * noise + 0.02 * max(level, 50.0)
                continue

            if time.time() - last_activity > ACTIVE_TIMEOUT:
                go_quiet("2 minutes of quiet")
                reset_wake(wake)
                continue

            preroll = (preroll + [block])[-PREROLL_BLOCKS:]
            if level < threshold:
                continue

            first = preroll
            if time.time() < shared.head_noisy_until:
                shared.freeze_head_until = time.time() + 2.0
                settle = []
                while time.time() < shared.head_noisy_until:
                    data, _ = mic.read(BLOCK)
                    settle.append(data.copy())
                data, _ = mic.read(BLOCK)
                settle.append(data.copy())
                if max(rms(b) for b in settle[-2:]) < threshold:
                    shared.freeze_head_until = 0.0
                    preroll = (preroll + settle)[-PREROLL_BLOCKS:]
                    continue
                first = preroll + settle

            audio = record_utterance(mic, first, threshold)
            shared.freeze_head_until = 0.0
            preroll = []
            text = transcribe(audio)
            if normalize(text) in FAKE_TRANSCRIPTS:
                set_state("awake")
                continue
            print(f"You: {text}")
            last_activity = shared.last_interaction = time.time()

            if is_stop_phrase(text):
                set_mood(mood.mood)
                say(QUIET_LINES.get(mood.mood, "Okay, I'll be quiet."))
                go_quiet("stop phrase")
                speech_queue.join()
                drain(mic)
                reset_wake(wake)
                continue

            with brain_lock:
                history.append({"role": "user", "content": text})
                reply = think_and_speak(client, history)
                print(f"{ROBOT_NAME}: {TAG.sub('', reply).strip()}")
                history.append({"role": "assistant", "content": reply})
                history[:] = history[-MAX_HISTORY:]

            time.sleep(0.3)
            drain(mic)
            last_activity = shared.last_interaction = time.time()
            set_state("awake")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if history[_memory_mark[0]:]:
            print("\nSaving what I learned...")
            reflect_on_conversation(history[_memory_mark[0]:])
        print("\nBye.")
