import sys
import time
import random
import select
import framebuf
from array import array
from machine import Pin, PWM, I2C, SPI, ADC

DEMO = False

OLED_BUS = "spi"
OLED_CLK = 18
OLED_MOSI = 19
OLED_RES = 20
OLED_DC = 16
OLED_CS = 17
I2C_SDA = 4
I2C_SCL = 5
PAN_PIN = 15
TILT_PIN = 14
AMP_SD_PIN = 21
SENSE_PIN = None

OLED_ADDR = 0x3C
COLUMN_OFFSET = 2
FLIP_SCREEN = False
EYES_FOLLOW_X = 1

PAN_LIMITS = (20, 160)
TILT_LIMITS = (60, 120)
PAN_START, TILT_START = 90, 90
SERVO_MIN_US, SERVO_MAX_US = 500, 2500
EASE = 0.15
MAX_STEP = 3.0

W, H = 128, 64
LEFT_X, RIGHT_X, EYE_Y = 40, 88, 27
EYE_RX, EYE_RY = 15, 17
MOUTH_Y = 56


class SH1106(framebuf.FrameBuffer):
    def __init__(self):
        if OLED_BUS == "spi":
            self.spi = SPI(0, baudrate=4_000_000, polarity=0, phase=0,
                           sck=Pin(OLED_CLK), mosi=Pin(OLED_MOSI))
            self.dc = Pin(OLED_DC, Pin.OUT, value=0)
            self.cs = Pin(OLED_CS, Pin.OUT, value=1)
            res = Pin(OLED_RES, Pin.OUT, value=1)
            time.sleep_ms(1)
            res.value(0)
            time.sleep_ms(10)
            res.value(1)
            time.sleep_ms(10)
        else:
            self.i2c = I2C(0, sda=Pin(I2C_SDA), scl=Pin(I2C_SCL), freq=400000)
            self.addr = OLED_ADDR
        self.buf = bytearray(W * H // 8)
        super().__init__(self.buf, W, H, framebuf.MONO_VLSB)
        seg, com = (0xA0, 0xC0) if FLIP_SCREEN else (0xA1, 0xC8)
        for c in (0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0xAD, 0x8B,
                  seg, com, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0x22, 0xDB, 0x35,
                  0xA4, 0xA6, 0xAF):
            self.cmd(c)
        self.on = True

    def cmd(self, c):
        if OLED_BUS == "spi":
            self.dc.value(0)
            self.cs.value(0)
            self.spi.write(bytes((c,)))
            self.cs.value(1)
        else:
            self.i2c.writeto(self.addr, bytes((0x80, c)))

    def data(self, d):
        if OLED_BUS == "spi":
            self.dc.value(1)
            self.cs.value(0)
            self.spi.write(d)
            self.cs.value(1)
        else:
            self.i2c.writeto(self.addr, b"\x40" + d)

    def show(self):
        for page in range(8):
            self.cmd(0xB0 + page)
            self.cmd(COLUMN_OFFSET & 0x0F)
            self.cmd(0x10 | (COLUMN_OFFSET >> 4))
            self.data(self.buf[page * W:(page + 1) * W])

    def power(self, on):
        if on != self.on:
            self.cmd(0xAF if on else 0xAE)
            self.on = on


class Servo:
    def __init__(self, pin, angle):
        self.pwm = PWM(Pin(pin))
        self.pwm.freq(50)
        self.last = None
        self.write(angle)

    def write(self, angle):
        a = round(angle * 2) / 2
        if a == self.last:
            return
        self.last = a
        us = SERVO_MIN_US + (SERVO_MAX_US - SERVO_MIN_US) * a / 180
        self.pwm.duty_u16(int(us * 65535 / 20000))

    def relax(self):
        self.pwm.duty_u16(0)
        self.last = None


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


GESTURES = {
    "nod":   [(0, 0, 0), (0.15, 0, 12), (0.35, 0, -6), (0.5, 0, 10), (0.7, 0, 0)],
    "shake": [(0, 0, 0), (0.15, 14, 0), (0.35, -14, 0), (0.55, 12, 0), (0.75, -8, 0), (0.9, 0, 0)],
    "tilt":  [(0, 0, 0), (0.25, 10, 6), (1.2, 10, 6), (1.5, 0, 0)],
    "startle": [(0, 0, 0), (0.06, 3, -12), (0.12, -2, -10), (0.5, 0, -8), (1.0, 0, 0)],
}


def gesture_offset(name, t):
    keys = GESTURES[name]
    if t >= keys[-1][0]:
        return None
    for (t0, p0, q0), (t1, p1, q1) in zip(keys, keys[1:]):
        if t0 <= t < t1:
            f = (t - t0) / (t1 - t0)
            f = f * f * (3 - 2 * f)
            return p0 + (p1 - p0) * f, q0 + (q1 - q0) * f
    return 0, 0


def lid(fb, pts):
    fb.poly(0, 0, array("h", pts), 0, True)


def draw_eye(fb, x, y, rx, ry, side, mood, state):
    fb.ellipse(x, y, rx, ry, 1, True)
    top = y - ry - 2
    inner = x - side * (rx + 2)
    outer = x + side * (rx + 2)

    if mood == "happy":
        fb.ellipse(x, y + ry + 3, rx + 5, ry, 0, True)
    elif mood == "angry":
        lid(fb, [outer, top, inner, top, inner, y, outer, top + 3])
    elif mood == "sad":
        lid(fb, [outer, top, inner, top, inner, top + 4, outer, y - ry // 5])
    elif mood == "annoyed":
        fb.fill_rect(x - rx - 2, top, 2 * rx + 5, ry + 2, 0)
    elif mood == "sleepy" or state == "thinking":
        cover = int(ry * (1.6 if mood == "sleepy" else 0.3))
        fb.fill_rect(x - rx - 2, top, 2 * rx + 5, cover + 2, 0)


def draw_closed(fb, x, y, rx):
    fb.fill_rect(x - rx, y + 2, 2 * rx, 3, 1)


def draw_face(fb, f, now):
    fb.fill(0)
    ox, oy = f.eye_x, f.eye_y

    if f.state == "asleep":
        for cx in (LEFT_X, RIGHT_X):
            draw_closed(fb, cx, EYE_Y + 4, EYE_RX - 2)
        for i in range(3):
            phase = ((now / 1.4) + i / 3) % 1
            fb.text("z", 102 + int(phase * 8), 30 - int(phase * 26), 1)
        return

    rx, ry = EYE_RX, EYE_RY
    if f.state == "listening":
        rx, ry = rx + 1, ry + 2
    if f.mood == "surprised":
        rx, ry = rx - 1, ry + 3

    for side, cx in ((-1, LEFT_X), (1, RIGHT_X)):
        erx, ery = rx, ry
        if f.mood == "curious":
            ery = ry + 2 if side == 1 else ry - 4
        ery = max(1, int(ery * f.openness))
        if ery <= 2:
            draw_closed(fb, cx + ox, EYE_Y + oy - 2, erx)
        else:
            draw_eye(fb, cx + ox, EYE_Y + oy, erx, ery, side, f.mood, f.state)

    if f.state == "talking":
        mw, mh = f.mouth
        fb.ellipse(64 + ox // 2, MOUTH_Y, mw, mh, 1, True)
    elif f.state == "thinking":
        lit = int(now * 3) % 3
        for i in range(3):
            r = 2 if i == lit else 1
            fb.ellipse(56 + i * 8, MOUTH_Y + 2, r, r, 1, True)


class Amp:
    def __init__(self):
        self.pin = None if AMP_SD_PIN is None else Pin(AMP_SD_PIN, Pin.OUT, value=0)
        self.on = False

    def set(self, on):
        if self.pin is None:
            return
        self.pin.value(1 if on else 0)
        self.on = on


class Face:
    def __init__(self):
        self.state, self.mood = "idle", "neutral"
        self.openness = 1.0
        self.eye_x, self.eye_y = 0, 0
        self.mouth = (4, 1)


def main():
    oled = SH1106()
    pan_servo, tilt_servo = Servo(PAN_PIN, PAN_START), Servo(TILT_PIN, TILT_START)
    sense = ADC(SENSE_PIN) if SENSE_PIN is not None else None
    amp = Amp()
    if DEMO:
        amp.set(True)

    face = Face()
    pan, tilt = float(PAN_START), float(TILT_START)
    target_pan, target_tilt = pan, tilt
    gesture, gesture_start = None, 0
    powered = True
    power_change_at = None

    blink_at = time.ticks_add(time.ticks_ms(), random.randint(1500, 4000))
    blink_start = None
    saccade_at, sacc_x, sacc_y = 0, 0, 0
    mouth_at = 0
    demo_at, demo_i = 0, 0
    demo_moods = ["neutral", "happy", "curious", "annoyed", "angry", "sad", "surprised"]

    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)
    line = ""

    print("HELLO")

    while True:
        ms = time.ticks_ms()
        now = ms / 1000

        while poll.poll(0):
            ch = sys.stdin.read(1)
            if ch in "\r\n":
                parts = line.strip().split()
                line = ""
                if not parts:
                    continue
                cmd = parts[0].upper()
                try:
                    if cmd == "STATE":
                        face.state = parts[1].lower()
                    elif cmd == "MOOD":
                        face.mood = parts[1].lower()
                    elif cmd == "LOOK":
                        target_pan = clamp(float(parts[1]), *PAN_LIMITS)
                        target_tilt = clamp(float(parts[2]), *TILT_LIMITS)
                    elif cmd == "GESTURE" and parts[1].lower() in GESTURES:
                        gesture, gesture_start = parts[1].lower(), now
                    elif cmd == "PING":
                        print("PONG")
                    elif cmd == "AMP":
                        amp.set(parts[1].lower() == "on" and powered)
                    elif cmd == "STATUS":
                        print("POWER on" if powered else "POWER off")
                except (IndexError, ValueError):
                    pass
            else:
                line = (line + ch)[-64:]

        if sense is not None:
            rail_on = sense.read_u16() * 3.3 / 65535 > 2.0
            if rail_on != powered:
                if power_change_at is None:
                    power_change_at = ms
                elif time.ticks_diff(ms, power_change_at) > 300:
                    powered = rail_on
                    power_change_at = None
                    oled.power(powered)
                    if not powered:
                        pan_servo.relax()
                        tilt_servo.relax()
                        amp.set(False)
                    print("POWER on" if powered else "POWER off")
            else:
                power_change_at = None
        if not powered:
            time.sleep_ms(50)
            continue

        if DEMO and now > demo_at:
            demo_at = now + 3
            face.mood = demo_moods[demo_i % len(demo_moods)]
            face.state = ("idle", "listening", "thinking", "talking")[demo_i % 4]
            target_pan = random.uniform(50, 130)
            target_tilt = random.uniform(75, 105)
            if demo_i % 3 == 2:
                gesture, gesture_start = ("nod", "shake", "tilt")[demo_i % 3], now
            demo_i += 1

        dp = clamp((target_pan - pan) * EASE, -MAX_STEP, MAX_STEP)
        dt = clamp((target_tilt - tilt) * EASE, -MAX_STEP, MAX_STEP)
        pan, tilt = pan + dp, tilt + dt
        gp, gt = 0, 0
        if gesture:
            off = gesture_offset(gesture, now - gesture_start)
            if off is None:
                gesture = None
            else:
                gp, gt = off
        pan_servo.write(clamp(pan + gp, 0, 180))
        tilt_servo.write(clamp(tilt + gt, 0, 180))

        if now > saccade_at:
            saccade_at = now + random.uniform(1.5, 4)
            sacc_x, sacc_y = random.randint(-3, 3), random.randint(-2, 2)
        lead_x = clamp((target_pan - pan) * 0.4, -10, 10) * EYES_FOLLOW_X
        lead_y = clamp((target_tilt - tilt) * 0.3, -6, 6)
        if face.state == "thinking":
            lead_x, lead_y, sacc_x, sacc_y = 7, -5, 0, 0
        face.eye_x = int(lead_x + sacc_x)
        face.eye_y = int(lead_y + sacc_y)

        if face.state != "asleep":
            if blink_start is None and time.ticks_diff(ms, blink_at) > 0:
                blink_start = ms
            if blink_start is not None:
                t = time.ticks_diff(ms, blink_start) / 160
                if t >= 1:
                    blink_start = None
                    face.openness = 1.0
                    gap = 180 if random.random() < 0.2 else random.randint(2000, 6000)
                    blink_at = time.ticks_add(ms, gap)
                else:
                    face.openness = abs(t * 2 - 1)
            else:
                face.openness = 1.0

        if face.state == "talking" and now > mouth_at:
            mouth_at = now + random.uniform(0.06, 0.14)
            face.mouth = (random.randint(5, 12), random.randint(1, 4))

        draw_face(oled, face, now)
        oled.show()


main()
