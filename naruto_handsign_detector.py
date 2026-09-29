"""
Naruto hand-sign -> jutsu detector, with REAL 3D-rendered, textured
Rasenshuriken/Chidori VFX

Uses your webcam (OpenCV) + MediaPipe's HandLandmarker (Tasks API) to track
hand landmarks and classify hand seals. Holding Open Palm or a closed Fist
charges an actual 3D-rendered effect (OpenGL: real geometry, procedurally
generated chakra/electricity textures, lighting, specular highlights, depth
buffer) positioned/oriented in 3D to sit in your palm, then composited onto
the webcam frame.

Install first:
    pip install opencv-python mediapipe numpy PyOpenGL PyOpenGL_accelerate pygame

The first run auto-downloads the hand landmark model (hand_landmarker.task,
~8MB) into the same folder as this script.

Run:
    python naruto_handsign_detector.py

Controls:
    q       - quit
    h       - toggle the on-screen hand-sign book (also printed at startup)
    c       - clear the whole current sequence / cancel charge
    z or BACKSPACE - remove just the last sign from the sequence (undo)
    SPACE   - manually commit the currently-held sign to the sequence

Notes:
    - This opens a hidden OpenGL context (via pygame) purely to render the
      3D effect off-screen; only the webcam window is shown.
    - If your GPU/driver doesn't support hidden GL windows, set
      GL_WINDOW_HIDDEN = False below to use a tiny visible debug window
      instead (the composited result still only appears in the main window).
"""

import os
import time
import math
import threading
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import sounddevice as sd

try:
    import torch
    ANIME_MODEL_AVAILABLE = True
except ImportError:
    ANIME_MODEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Jutsu database (hand-sign sequence -> jutsu name, text-only trigger)
# ---------------------------------------------------------------------------
# canon=True entries use the hand-seal sequence as documented on the
# Narutopedia/official sources. canon=False entries are techniques that
# either have no single agreed-upon seal sequence across the series, or -
# in the case of the bloodline/dojutsu abilities (Amaterasu, Susanoo, Kamui,
# Byakugan, Sharingan, Eight Gates) - use NO hand seals at all in canon
# (they're activated through eye/chakra techniques or internal gates, not
# hand signs). Those are given a sequence here purely so this hand-sign
# app has a way to trigger them; see print_handsign_book() for the note
# shown alongside each one.
JUTSU_DB = [
    {"seq": ["Snake", "Ram", "Monkey", "Boar", "Horse", "Tiger"], "name": "Fireball jutsu", "canon": True},
    {"seq": ["Ram", "Snake", "Tiger"], "name": "Shadow clone jutsu", "canon": True},
    {"seq": ["Tiger", "Boar", "Dog", "Monkey", "Ram"], "name": "Water dragon jutsu", "canon": False},
    {"seq": ["Rat", "Boar", "Dog"], "name": "Lightning style: false darkness", "canon": False},
    {"seq": ["Dragon", "Hare", "Ox"], "name": "Earth style: mud wall", "canon": False},
    {"seq": ["Tiger", "Boar", "Ox", "Dog", "Snake"], "name": "Substitution jutsu", "canon": True},
    {"seq": ["Boar", "Dog", "Bird", "Monkey", "Ram"], "name": "Summoning jutsu", "canon": True},
    {"seq": ["Dog", "Boar", "Tiger"], "name": "Transformation jutsu", "canon": True},
    {"seq": ["Tiger"], "name": "Body flicker jutsu", "canon": True},
    {"seq": ["Snake", "Ram", "Monkey", "Boar", "Horse", "Tiger", "Boar"], "name": "Great fireball jutsu", "canon": False},
    {"seq": ["Ox", "Monkey", "Boar"], "name": "Earth style: headhunter jutsu", "canon": False},
    {"seq": ["Snake", "Dragon", "Rat", "Bird", "Horse"], "name": "Water style: water wall", "canon": False},
    {"seq": ["Bird", "Horse", "Snake"], "name": "Wind style: great breakthrough", "canon": False},
    {"seq": ["Boar", "Ram", "Dog", "Bird"], "name": "Summoning: toad expansion jutsu", "canon": False},
    {"seq": ["Ox", "Tiger", "Rat"], "name": "Chakra sensing jutsu", "canon": False},
    {"seq": ["Dragon", "Ram", "Tiger"], "name": "Genjutsu release", "canon": False},
    {"seq": ["Snake", "Boar", "Rat", "Ram"], "name": "Multi shadow clone jutsu", "canon": False},
    {"seq": ["Tiger", "Ram", "Snake", "Bird"], "name": "Amaterasu", "canon": False},
    {"seq": ["Ox", "Ram", "Monkey", "Boar", "Horse"], "name": "Susanoo", "canon": False},
    {"seq": ["Dragon", "Bird", "Rat"], "name": "Kamui", "canon": False},
    {"seq": ["Ram", "Boar", "Ox", "Dog"], "name": "Sage mode", "canon": False},
    {"seq": ["Horse", "Ram", "Dragon", "Boar", "Dog"], "name": "Reanimation jutsu", "canon": False},
    {"seq": ["Tiger", "Dog", "Ram"], "name": "Byakugan", "canon": False},
    {"seq": ["Snake", "Ox", "Dragon"], "name": "Sharingan", "canon": False},
    {"seq": ["Bird", "Snake", "Tiger", "Ram"], "name": "Eight gates release", "canon": False},
    {"seq": ["Monkey", "Rat", "Bird"], "name": "Puppet technique", "canon": False},
    {"seq": ["Dog", "Hare", "Monkey", "Ram"], "name": "Reaper death seal", "canon": False},
]
MAX_SEQ_LEN = max((len(j["seq"]) for j in JUTSU_DB), default=1)

RASEN_TRIGGER_SIGN = "Open Palm"
RASEN_CHARGE_SECONDS = 1.2
CHIDORI_TRIGGER_SIGN = "Fist / Rat"
CHIDORI_CHARGE_SECONDS = 1.0

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

SAMPLE_RATE = 44100


def _envelope(n, attack=0.05, release=0.3):
    t = np.linspace(0, 1, n)
    env = np.ones(n)
    a_n = int(n * attack)
    r_n = int(n * release)
    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n)
    if r_n > 0:
        env[-r_n:] *= np.linspace(1, 0, r_n)
    return env


def make_charge_sound(duration=0.9, f0=180, f1=950):
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n)
    freq = np.linspace(f0, f1, n)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    tone = np.sin(phase) * 0.5 + np.sin(phase * 2.01) * 0.2
    tone *= _envelope(n, attack=0.15, release=0.1)
    tone *= np.linspace(0.15, 1.0, n)  # swell in volume as it charges
    return (tone * 0.35).astype(np.float32)


def make_release_boom(duration=0.6, noisy=False):
    n = int(SAMPLE_RATE * duration)
    rng = np.random.default_rng()
    if noisy:
        # crackly electric zap: filtered noise bursts
        noise = rng.uniform(-1, 1, n)
        spikes = (rng.uniform(0, 1, n) > 0.985).astype(np.float32)
        sig = noise * 0.4 + spikes * rng.uniform(-1, 1, n) * 1.5
        env = _envelope(n, attack=0.01, release=0.7)
        sig *= env
    else:
        t = np.linspace(0, duration, n)
        freq = np.linspace(700, 90, n)
        phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
        sig = np.sin(phase) * 0.6 + np.sin(phase * 1.5) * 0.3
        sig *= _envelope(n, attack=0.005, release=0.85)
    return (np.clip(sig, -1, 1) * 0.5).astype(np.float32)


def make_poof_sound(duration=0.35):
    n = int(SAMPLE_RATE * duration)
    rng = np.random.default_rng()
    noise = rng.uniform(-1, 1, n)
    t = np.linspace(0, duration, n)
    lp = np.convolve(noise, np.ones(30) / 30, mode="same")  # crude low-pass
    sig = lp * _envelope(n, attack=0.02, release=0.75)
    return (sig * 0.4).astype(np.float32)


def play_sound(data):
    try:
        sd.play(data, SAMPLE_RATE, latency="low")
    except Exception:
        try:
            sd.play(data, SAMPLE_RATE)  # fall back if the device rejects 'low'
        except Exception:
            pass  # never let audio issues crash the visual effect


def play_sound_synced(data, frame_delay_sec):
    """Play a pre-generated sound, delayed to roughly line up with when the
    CURRENT frame's visual actually reaches the screen. Heavy per-frame work
    (MediaPipe inference + OpenGL render + readback) means the frame being
    processed right now typically shows up one frame-interval later than the
    moment the trigger condition was detected - since the sound itself is
    instant (pre-generated), playing it immediately makes it arrive early
    relative to the visual. Delaying it by the current measured frame time
    compensates for that gap and keeps them roughly in sync even as
    framerate drifts."""
    delay = max(0.0, min(0.5, frame_delay_sec))  # clamp - never delay more than 0.5s
    if delay < 0.005:
        play_sound(data)
    else:
        threading.Timer(delay, play_sound, args=(data,)).start()


def pregenerate_sounds():
    """Synthesize every sound clip once up front. Triggering a sound during
    the main loop is then just sd.play() on an already-built buffer, so it
    can't stall a frame and throw off sync with the visual effect."""
    return {
        "rasen_charge": make_charge_sound(duration=RASEN_CHARGE_SECONDS + 0.2, f0=180, f1=950),
        "rasen_release": make_release_boom(noisy=False),
        "chidori_charge": make_charge_sound(duration=CHIDORI_CHARGE_SECONDS + 0.15, f0=260, f1=1400),
        "chidori_release": make_release_boom(noisy=True),
        "poof": make_poof_sound(),
    }


TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

FOVY_DEG = 50.0
OBJECT_DEPTH = -260.0
GL_WINDOW_HIDDEN = True


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(landmarks, handedness_label):
    up = {}

    # Thumb: compare how far the tip sits from the index knuckle versus how
    # far the thumb's own IP joint sits from that same point. This is a
    # distance-based (rotation-tolerant) check - much more reliable than
    # comparing raw x-coordinates, which only works when the hand is close
    # to upright and facing the camera straight-on.
    index_mcp = landmarks[INDEX_MCP]
    thumb_tip_dist = _dist(landmarks[TIP_IDS[0]], index_mcp)
    thumb_ip_dist = _dist(landmarks[PIP_IDS[0]], index_mcp)
    up["thumb"] = thumb_tip_dist > thumb_ip_dist * 1.15  # small margin against jitter

    for i in range(1, 5):
        margin = 0.02  # small hysteresis so near-straight fingers don't flicker
        up[FINGER_NAMES[i]] = landmarks[TIP_IDS[i]].y < (landmarks[PIP_IDS[i]].y - margin)
    return up


def classify_sign(up):
    pattern = tuple(up[f] for f in FINGER_NAMES)
    mapping = {
        (False, False, False, False, False): CHIDORI_TRIGGER_SIGN,
        (True, False, False, False, False): "Ox",
        (True, True, False, False, False): "Tiger",
        (False, True, False, False, False): "Dragon",
        (True, True, True, False, False): "Hare",
        (False, True, True, False, False): "Snake",
        (True, False, False, False, True): "Horse",
        (False, False, False, False, True): "Ram",
        (True, True, False, False, True): "Monkey",
        (False, True, True, True, False): "Bird",
        (True, False, True, True, True): "Dog",
        (False, True, True, True, True): "Boar",
        (True, True, True, True, True): RASEN_TRIGGER_SIGN,
    }
    return mapping.get(pattern, None)


def match_jutsu(seq):
    for j in JUTSU_DB:
        n = len(j["seq"])
        if len(seq) >= n and seq[-n:] == j["seq"]:
            return j["name"]
    return None


def palm_orientation_normal(landmarks):
    w = landmarks[WRIST]
    i = landmarks[INDEX_MCP]
    p = landmarks[PINKY_MCP]
    v1 = np.array([i.x - w.x, i.y - w.y, i.z - w.z])
    v2 = np.array([p.x - w.x, p.y - w.y, p.z - w.z])
    normal = np.cross(v1, v2)
    n = np.linalg.norm(normal)
    if n < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return normal / n


# ---------------------------------------------------------------------------
# Procedural textures (no external image files needed)
# ---------------------------------------------------------------------------
def _to_gl_texture(rgba_uint8):
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    h, w = rgba_uint8.shape[:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_uint8)
    return tex_id


def make_chakra_texture(size=256):
    """Swirling blue/cyan/white chakra energy: radial glow + spiral streak
    noise, alpha falls off toward the edge for a wispy, non-solid look."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2.0
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) / (size / 2.0)
    ang = np.arctan2(dy, dx)

    swirl = np.sin(ang * 6 + r * 18) * 0.5 + 0.5
    swirl2 = np.sin(ang * -10 + r * 30) * 0.5 + 0.5
    streaks = (swirl * 0.6 + swirl2 * 0.4)

    core_glow = np.clip(1.0 - r * 1.1, 0, 1) ** 1.5
    brightness = np.clip(core_glow * 1.3 + streaks * 0.55, 0, 1)

    blue = np.clip(0.55 + brightness * 0.45, 0, 1)
    green = np.clip(0.55 + brightness * 0.4, 0, 1)
    redc = np.clip(0.35 + brightness * 0.55, 0, 1)

    alpha = np.clip((1.0 - r) * (0.55 + 0.45 * streaks), 0, 1)
    alpha = np.clip(alpha * 1.15, 0, 1)

    rgba = np.stack([
        (redc * 255),
        (green * 255),
        (blue * 255),
        (alpha * 255),
    ], axis=-1).astype(np.uint8)
    return rgba


def make_core_texture(size=128):
    """White-hot core with faint spiral hint, fully opaque center fading at
    the rim so it blends smoothly into the blade texture."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2.0
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) / (size / 2.0)
    ang = np.arctan2(dy, dx)
    swirl = np.sin(ang * 5 + r * 10) * 0.15
    glow = np.clip(1.0 - r + swirl, 0, 1) ** 0.8
    val = np.clip(0.6 + glow * 0.6, 0, 1)
    alpha = np.clip(1.2 - r, 0, 1)
    rgba = np.stack([val * 255, val * 255, 255 * np.ones_like(val), alpha * 255], axis=-1).astype(np.uint8)
    return rgba


def make_lightning_texture(width=64, height=256):
    """Electric blue/white streaks running lengthwise (t axis) with jagged
    alpha cutouts around the circumference (s axis), for a crackling,
    non-solid tube surface."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = yy / height
    s = xx / width

    n_streaks = 5
    val = np.zeros((height, width), dtype=np.float32)
    for _ in range(n_streaks):
        phase = rng.uniform(0, 2 * math.pi)
        freq = rng.uniform(2, 5)
        centerline = 0.5 + 0.3 * np.sin(t * freq * 2 * math.pi + phase)
        dist = np.minimum(np.abs(s - centerline), 1 - np.abs(s - centerline))
        val += np.clip(1 - dist * 10, 0, 1)
    val = np.clip(val, 0, 1)

    noise = rng.uniform(0.85, 1.15, size=val.shape).astype(np.float32)
    val = np.clip(val * noise, 0, 1)

    alpha = np.clip(val * 1.3, 0, 1)
    blue = np.clip(0.7 + val * 0.3, 0, 1)
    white = np.clip(0.6 + val * 0.4, 0, 1)

    rgba = np.stack([white * 255, white * 255, blue * 255, alpha * 255], axis=-1).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# OpenGL offscreen renderer
# ---------------------------------------------------------------------------
class GLRenderer:
    def __init__(self, width, height):
        self.w, self.h = width, height
        pygame.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if GL_WINDOW_HIDDEN:
            flags |= pygame.HIDDEN
        pygame.display.set_mode((width, height), flags)

        glViewport(0, 0, width, height)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 0.6, 1.0, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.15, 0.15, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))

        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)
        gluQuadricTexture(self.quadric, GL_TRUE)

        self.tex_chakra = _to_gl_texture(make_chakra_texture())
        self.tex_core = _to_gl_texture(make_core_texture())
        self.tex_lightning = _to_gl_texture(make_lightning_texture())

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOVY_DEG, width / float(height), 10.0, 2000.0)
        glMatrixMode(GL_MODELVIEW)

    def begin_frame(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

    def read_rgba(self):
        glFlush()
        data = glReadPixels(0, 0, self.w, self.h, GL_RGBA, GL_UNSIGNED_BYTE)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(self.h, self.w, 4)
        arr = np.flipud(arr)
        return arr

    def px_to_world(self, px, py, radius_px, z=OBJECT_DEPTH):
        top = math.tan(math.radians(FOVY_DEG) / 2.0) * abs(z)
        right = top * (self.w / float(self.h))
        ndc_x = (px / self.w) * 2 - 1
        ndc_y = 1 - (py / self.h) * 2
        X = ndc_x * right
        Y = ndc_y * top
        world_radius = radius_px * abs(z) * math.tan(math.radians(FOVY_DEG) / 2.0) / (self.h / 2.0)
        return X, Y, z, world_radius

    def set_object_transform(self, px, py, radius_px, normal_xyz, spin_deg):
        X, Y, Z, world_r = self.px_to_world(px, py, radius_px)
        glTranslatef(X, Y, Z)

        nx, ny, nz = normal_xyz
        n_gl = np.array([nx, -ny, -nz])
        n_gl /= (np.linalg.norm(n_gl) + 1e-9)
        z_axis = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z_axis, n_gl)
        axis_len = np.linalg.norm(axis)
        dot = float(np.clip(np.dot(z_axis, n_gl), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        if axis_len > 1e-6:
            axis = axis / axis_len
            glRotatef(angle_deg, axis[0], axis[1], axis[2])
        glRotatef(spin_deg, 0, 0, 1)
        return world_r

    def draw_core_sphere(self, radius, texture_id):
        glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS, (90.0,))
        glColor4f(1.0, 1.0, 1.0, 1.0)
        gluSphere(self.quadric, max(1.0, radius * 0.22), 14, 14)
        glDisable(GL_TEXTURE_2D)
        glPopAttrib()

    def draw_glow_halo(self, radius, color=(1.0, 1.0, 1.0), scale=1.7):
        """Soft, additive-blended bloom disc behind the geometry, anime-style
        overexposed glow. Foreshortens with the object since it shares the
        object's current rotation, which is physically correct for a disc
        lying flat in the same plane."""
        glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex_core)

        r = radius * scale
        n = 24
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(color[0], color[1], color[2], 0.9)
        glTexCoord2f(0.5, 0.5)
        glVertex3f(0, 0, 0)
        glColor4f(color[0], color[1], color[2], 0.0)
        for i in range(n + 1):
            ang = 2 * math.pi * i / n
            glTexCoord2f(0.5 + 0.5 * math.cos(ang), 0.5 + 0.5 * math.sin(ang))
            glVertex3f(r * math.cos(ang), r * math.sin(ang), 0)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopAttrib()

    def draw_rasenshuriken(self, radius):
        """4 extruded, twisted spiral blades with a procedural chakra
        texture + per-vertex color gradient (white near the core, deep blue
        toward the tips), plus a textured lit sphere core."""
        self.draw_glow_halo(radius, color=(0.85, 0.93, 1.0), scale=1.9)

        glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex_chakra)
        glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS, (70.0,))

        n_blades = 4
        n_pts = 12
        half_thick = max(1.0, radius * 0.05)

        def color_for(t):
            # near-white throughout (anime rasenshuriken reads as blown-out
            # white with only a light blue tint), slightly cooler at the tip
            r = 0.90 + 0.10 * (1 - t)
            g = 0.95 + 0.05 * (1 - t)
            b = 1.0
            return (r, g, b)

        for b in range(n_blades):
            base_angle = b * (2 * math.pi / n_blades)
            centerline = []
            widths = []
            for i in range(n_pts):
                t = i / (n_pts - 1)
                r = radius * (0.12 + 0.88 * t)
                ang = base_angle + t * 1.6
                x = r * math.cos(ang)
                y = r * math.sin(ang)
                centerline.append(np.array([x, y, 0.0]))
                widths.append((1.0 - t * 0.8) * radius * 0.16)

            n_wraps = 3.0  # how many times the texture repeats along the blade

            glBegin(GL_TRIANGLE_STRIP)
            for i in range(n_pts):
                t = i / (n_pts - 1)
                p = centerline[i]
                tang = (centerline[min(i + 1, n_pts - 1)] - centerline[max(i - 1, 0)])
                tang /= (np.linalg.norm(tang) + 1e-9)
                side = np.array([-tang[1], tang[0], 0.0]) * widths[i]
                glColor4f(*color_for(t), 1.0)
                glNormal3f(0, 0, 1)
                glTexCoord2f(t * n_wraps, 0.0)
                glVertex3f(*(p + side))
                glTexCoord2f(t * n_wraps, 1.0)
                glVertex3f(*(p - side))
            glEnd()

            glBegin(GL_TRIANGLE_STRIP)
            for i in range(n_pts):
                t = i / (n_pts - 1)
                p = centerline[i]
                tang = (centerline[min(i + 1, n_pts - 1)] - centerline[max(i - 1, 0)])
                tang /= (np.linalg.norm(tang) + 1e-9)
                side = np.array([-tang[1], tang[0], 0.0]) * widths[i]
                off = np.array([0, 0, -half_thick])
                glColor4f(*color_for(t), 1.0)
                glNormal3f(0, 0, -1)
                glTexCoord2f(t * n_wraps, 0.0)
                glVertex3f(*(p + side + off))
                glTexCoord2f(t * n_wraps, 1.0)
                glVertex3f(*(p - side + off))
            glEnd()

            glBegin(GL_QUAD_STRIP)
            for i in range(n_pts):
                t = i / (n_pts - 1)
                p = centerline[i]
                tang = (centerline[min(i + 1, n_pts - 1)] - centerline[max(i - 1, 0)])
                tang /= (np.linalg.norm(tang) + 1e-9)
                side = np.array([-tang[1], tang[0], 0.0]) * widths[i]
                out_top = p + side
                out_bot = p + side + np.array([0, 0, -half_thick])
                edge_n = side / (np.linalg.norm(side) + 1e-9)
                glColor4f(*color_for(t), 1.0)
                glNormal3f(*edge_n)
                glTexCoord2f(t * n_wraps, 0.3)
                glVertex3f(*out_top)
                glTexCoord2f(t * n_wraps, 0.7)
                glVertex3f(*out_bot)
            glEnd()

        # small serrated "teeth" ring around the perimeter, between the
        # main blades, for the gear-like silhouette seen in the anime
        n_teeth = 16
        glBegin(GL_TRIANGLES)
        for i in range(n_teeth):
            ang0 = 2 * math.pi * i / n_teeth
            ang1 = ang0 + (2 * math.pi / n_teeth) * 0.5
            r_out = radius * 1.02
            r_in = radius * 0.88
            p0 = np.array([r_in * math.cos(ang0), r_in * math.sin(ang0), 0.0])
            p1 = np.array([r_out * math.cos((ang0 + ang1) / 2), r_out * math.sin((ang0 + ang1) / 2), 0.0])
            p2 = np.array([r_in * math.cos(ang1), r_in * math.sin(ang1), 0.0])
            glColor4f(0.95, 0.97, 1.0, 1.0)
            glNormal3f(0, 0, 1)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(*p0)
            glTexCoord2f(0.5, 1.0)
            glVertex3f(*p1)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(*p2)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glPopAttrib()
        self.draw_core_sphere(radius, self.tex_core)

    def draw_chidori(self, radius, rng):
        """Jagged lightning bolts as textured, lit 3D tubes radiating in 3D
        around the palm, regenerated each call for flicker."""
        self.draw_glow_halo(radius, color=(0.75, 0.88, 1.0), scale=2.2)

        glPushAttrib(GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_ENABLE_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex_lightning)
        glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS, (95.0,))
        glColor4f(0.94, 0.97, 1.0, 1.0)

        n_bolts = 7
        n_segs = 4
        for _ in range(n_bolts):
            theta = rng.uniform(0, 2 * math.pi)
            phi = rng.uniform(-0.6, 0.6)
            length = radius * rng.uniform(0.7, 1.5)
            pos = np.array([0.0, 0.0, 0.0])
            direction = np.array([math.cos(theta) * math.cos(phi),
                                   math.sin(theta) * math.cos(phi),
                                   math.sin(phi)])
            for s in range(n_segs):
                seg_len = length / n_segs
                jitter = np.array([rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45)])
                direction = direction + jitter
                direction /= (np.linalg.norm(direction) + 1e-9)
                nxt = pos + direction * seg_len
                thickness = max(0.4, radius * 0.032 * (1 - s / n_segs))
                self._draw_cylinder_segment(pos, nxt, thickness)
                pos = nxt

        glDisable(GL_TEXTURE_2D)
        glPopAttrib()
        self.draw_core_sphere(radius * 1.1, self.tex_core)

    def _draw_cylinder_segment(self, p0, p1, radius):
        seg = p1 - p0
        length = np.linalg.norm(seg)
        if length < 1e-6:
            return
        glPushMatrix()
        glTranslatef(*p0)
        z_axis = np.array([0.0, 0.0, 1.0])
        dirn = seg / length
        axis = np.cross(z_axis, dirn)
        axis_len = np.linalg.norm(axis)
        dot = float(np.clip(np.dot(z_axis, dirn), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        if axis_len > 1e-6:
            axis = axis / axis_len
            glRotatef(angle_deg, axis[0], axis[1], axis[2])
        gluCylinder(self.quadric, radius, radius * 0.5, length, 6, 1)
        glPopMatrix()

    def end_frame(self):
        pygame.display.flip()


def composite_rgba(frame, rgba, alpha_mult):
    rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)
    a = (rgba[:, :, 3:4].astype(np.float32) / 255.0) * alpha_mult
    frame[:] = (frame.astype(np.float32) * (1 - a) + rgb * a).astype(np.uint8)


EXECUTE_BUTTON_RECT = None  # (x0, y0, x1, y1) - set once frame size is known


def compute_execute_button_rect(frame_w, frame_h):
    bw, bh = 150, 46
    x0 = frame_w - bw - 16
    y0 = frame_h - bh - 16
    return (x0, y0, x0 + bw, y0 + bh)


def point_in_rect(px, py, rect):
    x0, y0, x1, y1 = rect
    return x0 <= px <= x1 and y0 <= py <= y1


def draw_execute_button(frame, rect, pressed=False):
    x0, y0, x1, y1 = rect
    base_color = (40, 160, 90) if not pressed else (30, 110, 60)
    cv2.rectangle(frame, (x0, y0), (x1, y1), base_color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 220, 140), 2, lineType=cv2.LINE_AA)
    text = "EXECUTE"
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    tx = x0 + (x1 - x0 - ts[0]) // 2
    ty = y0 + (y1 - y0 + ts[1]) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_leaf_emblem(frame, center, radius, color=(90, 200, 255)):
    """A simple stylized swirl/leaf mark - a nod to the Hidden Leaf
    headband emblem, drawn procedurally (no image assets needed)."""
    cx, cy = center
    for i in range(3):
        ang0 = i * 120
        ang1 = ang0 + 95
        cv2.ellipse(frame, (cx, cy), (radius, int(radius * 0.62)), ang0,
                    0, 95, color, thickness=max(2, radius // 6), lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), max(2, radius // 5), color, -1, lineType=cv2.LINE_AA)


def draw_hud(frame, current_sign, sequence, jutsu_text, jutsu_timer, charge, charge_label="Charging...", fps=None, finger_state=None):
    h, w = frame.shape[:2]
    hud_h = 112 if finger_state else 90

    # dark navy "shinobi scanner" panel with a thin gold accent line, instead
    # of a flat black debug bar
    panel = frame[0:hud_h, 0:w].astype(np.float32)
    tint = np.full_like(panel, (30, 22, 14))  # BGR - warm dark brown-navy
    frame[0:hud_h, 0:w] = cv2.addWeighted(panel, 0.15, tint, 0.85, 0).astype(np.uint8)
    cv2.line(frame, (0, hud_h - 1), (w, hud_h - 1), (60, 150, 210), 2, cv2.LINE_AA)

    draw_leaf_emblem(frame, (w - 34, 36), 16)

    cv2.putText(frame, f"Sign: {current_sign or '-'}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    seq_str = " -> ".join(sequence[-6:]) if sequence else "-"
    cv2.putText(frame, f"Seq: {seq_str}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "[c] clear all   [z] undo last", (12, hud_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    if finger_state is not None:
        parts = []
        for label, up in zip(("T", "I", "M", "R", "P"), finger_state):
            parts.append((f"{label}:{'UP' if up else 'dn'}", (120, 255, 120) if up else (140, 140, 140)))
        x = 12
        y = 82
        for text, color in parts:
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            x += 62
    if fps is not None:
        cv2.putText(frame, f"{fps:.0f} fps", (w - 130, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 255, 150), 1)

    if charge > 0:
        bar_w = 200
        filled = int(bar_w * min(1.0, charge))
        cv2.rectangle(frame, (w - bar_w - 20, 20), (w - 20, 40), (80, 80, 80), 1)
        cv2.rectangle(frame, (w - bar_w - 20, 20), (w - bar_w - 20 + filled, 40), (255, 210, 120), -1)
        cv2.putText(frame, charge_label, (w - bar_w - 20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 190), 1)

    if jutsu_text and time.time() < jutsu_timer:
        elapsed = 2.5 - (jutsu_timer - time.time())  # matches the 2.5s popup window set at trigger
        pop_dur = 0.22
        pop_t = min(1.0, max(0.0, elapsed) / pop_dur)
        overshoot = 1.0 + 0.35 * math.sin(pop_t * math.pi) * (1 - pop_t)  # quick overshoot-then-settle
        base_scale = 1.5
        scale = base_scale * (0.4 + 0.6 * pop_t) * overshoot

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness_fill = max(2, int(3 * scale / base_scale))
        thickness_outline = thickness_fill + 5
        upper_text = jutsu_text.upper() + "!!"
        text_size = cv2.getTextSize(upper_text, font, scale, thickness_fill)[0]
        tx = (w - text_size[0]) // 2
        ty = h - 46

        fade = min(1.0, max(0.0, (jutsu_timer - time.time()) / 0.4)) if (jutsu_timer - time.time()) < 0.4 else 1.0

        text_layer = frame.copy()
        # thick black comic outline, then bright fill on top
        cv2.putText(text_layer, upper_text, (tx, ty), font, scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
        cv2.putText(text_layer, upper_text, (tx, ty), font, scale, (0, 210, 255), thickness_fill, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(frame, 1 - fade, text_layer, fade, 0)
    return frame


def build_handsign_book():
    """One combined reference list: charge-and-hold moves first, then every
    hand-sign sequence jutsu, in the order they appear in JUTSU_DB. canon
    marks whether the sequence matches documented Narutopedia/official
    sources, or is a gameplay simplification for this app."""
    book = [
        (RASEN_TRIGGER_SIGN + " (hold)", "Rasenshuriken", False),
        (CHIDORI_TRIGGER_SIGN + " (hold)", "Chidori", False),
    ]
    for j in JUTSU_DB:
        book.append((" -> ".join(j["seq"]), j["name"], j.get("canon", False)))
    return book


HANDSIGN_BOOK = build_handsign_book()


def print_handsign_book():
    print("\n" + "=" * 60)
    print("  KONOHA NINJA ARCHIVES - Scroll of Sealing Techniques")
    print("=" * 60)
    name_w = max(len(name) for _, name, _ in HANDSIGN_BOOK) + 2
    for seq, name, canon in HANDSIGN_BOOK:
        tag = "[canon]" if canon else "[simplified]"
        print(f"  {name:<{name_w}} {tag:<13} {seq}")
    print("-" * 60)
    print("  [canon] = matches the documented anime/manga hand-seal sequence")
    print("  [simplified] = adapted for this app (no single canon sequence")
    print("                 exists, or the real technique uses no hand seals)")
    print("  (press 'h' in the app window to toggle this list on-screen)")
    print("=" * 60 + "\n")


def draw_handsign_book(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # dark scroll-brown wash instead of flat black
    cv2.rectangle(overlay, (0, 0), (w, h), (18, 24, 34), -1)
    frame[:] = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)

    cv2.putText(frame, "Scroll of Sealing Techniques", (24, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (110, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "(h to close)", (24, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(frame, (24, 70), (w - 24, 70), (60, 120, 180), 1, cv2.LINE_AA)

    y = 98
    line_h = 25
    for seq, name, canon in HANDSIGN_BOOK:
        if y > h - 40:
            break
        tag_color = (110, 220, 130) if canon else (150, 150, 150)
        tag_text = "canon" if canon else "app"
        cv2.putText(frame, tag_text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    tag_color, 1, cv2.LINE_AA)
        cv2.putText(frame, name, (74, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 210, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, seq, (w // 2 - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (215, 215, 215), 1, cv2.LINE_AA)
        y += line_h

    cv2.putText(frame, "green 'canon' = official sequence   grey 'app' = simplified for this scanner",
                (24, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    return frame


def get_jutsu_effect(name):
    """Map a jutsu name to a (effect_type, color) pair used for its
    post-trigger visual. color is BGR."""
    lname = name.lower()
    if "genjutsu" in lname or "sharingan" in lname:
        return "genjutsu", None                # zoom + red Sharingan eye
    if "amaterasu" in lname:
        return "flame", (5, 5, 5)               # jet-black licking flames
    if "susanoo" in lname:
        return "susanoo", (200, 60, 160)        # glowing purple ribcage aura
    if "kamui" in lname:
        return "vortex", (220, 150, 255)        # spatial swirl warp
    if any(k in lname for k in ("clone", "transformation", "substitution", "flicker")):
        return "smoke", (235, 235, 255)
    if any(k in lname for k in ("summon", "toad")):
        return "smoke", (200, 255, 210)
    if "fire" in lname:
        return "burst", (50, 90, 255)
    if "water" in lname:
        return "burst", (255, 150, 40)
    if "earth" in lname or "mud" in lname or "headhunter" in lname:
        return "burst", (35, 95, 140)
    if "wind" in lname or "breakthrough" in lname:
        return "burst", (225, 255, 225)
    if "lightning" in lname or "darkness" in lname:
        return "burst", (255, 220, 170)
    if "chakra sensing" in lname:
        return "burst", (255, 210, 90)
    if "sage mode" in lname:
        return "burst", (90, 200, 255)          # orange-yellow (BGR)
    if "reanimation" in lname or "reaper" in lname:
        return "smoke", (150, 150, 150)         # ashen grey
    if "byakugan" in lname:
        return "byakugan", (240, 230, 255)      # pale lavender-white veins
    if "eight gates" in lname:
        return "eightgates", (40, 90, 235)      # intense red-orange aura
    if "puppet" in lname:
        return "smoke", (210, 210, 210)
    return "burst", (255, 255, 255)


def apply_burst_effect(frame, color, t):
    """Quick full-frame color flash + an expanding ring with a few sparks -
    a lightweight, distinct visual per elemental jutsu."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    flash_alpha = max(0.0, 1.0 - t * 2.2) * 0.32
    if flash_alpha > 0:
        overlay = np.full_like(frame, color, dtype=np.uint8)
        frame[:] = cv2.addWeighted(frame, 1 - flash_alpha, overlay, flash_alpha, 0)

    ring_r = int(min(w, h) * 0.5 * t)
    if ring_r > 0:
        cv2.circle(frame, (cx, cy), ring_r, color, thickness=max(2, int(8 * (1 - t))),
                   lineType=cv2.LINE_AA)
        add_ink_outline(frame, (cx, cy), ring_r, thickness=max(2, int(3 * (1 - t))))
        rng = np.random.default_rng(int(t * 1000))
        for _ in range(14):
            ang = rng.uniform(0, 2 * math.pi)
            r = ring_r * rng.uniform(0.85, 1.1)
            px, py = int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(frame, (px, py), 2, color, -1, lineType=cv2.LINE_AA)
    return frame


def apply_smoke_effect(frame, color, t):
    """Soft expanding puff(s) of smoke fading out - used for clone/
    transformation/substitution/summoning/body-flicker style jutsu."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    alpha = max(0.0, (1.0 - t)) * 0.5
    r = int(min(w, h) * (0.12 + 0.22 * t))

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r, color, -1, lineType=cv2.LINE_AA)
    rng = np.random.default_rng(7)
    for _ in range(5):
        ang = rng.uniform(0, 2 * math.pi)
        dist = r * rng.uniform(0.5, 1.0)
        px, py = int(cx + dist * math.cos(ang)), int(cy + dist * math.sin(ang))
        cv2.circle(overlay, (px, py), int(r * rng.uniform(0.3, 0.6)), color, -1, lineType=cv2.LINE_AA)
    frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return frame


def draw_sharingan(frame, center, radius, rotation_deg, alpha):
    """Procedural red Sharingan eye: dark sclera, red iris, black pupil,
    three tomoe orbiting at mid-radius, a ring outline and a glint."""
    overlay = frame.copy()
    cx, cy = center

    cv2.circle(overlay, (cx, cy), radius, (10, 10, 10), -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), int(radius * 0.9), (20, 20, 195), -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), int(radius * 0.28), (5, 5, 5), -1, lineType=cv2.LINE_AA)

    for i in range(3):
        ang_deg = rotation_deg + i * 120
        ang = math.radians(ang_deg)
        dist = radius * 0.5
        px, py = int(cx + dist * math.cos(ang)), int(cy + dist * math.sin(ang))
        size = max(3, int(radius * 0.13))
        cv2.circle(overlay, (px, py), size, (5, 5, 5), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(overlay, (px, py), (size * 2, size), ang_deg, 0, 150,
                    (5, 5, 5), thickness=max(2, size // 2 + 1), lineType=cv2.LINE_AA)

    cv2.circle(overlay, (cx, cy), radius, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    glint_r = max(2, int(radius * 0.09))
    cv2.circle(overlay, (cx - int(radius * 0.35), cy - int(radius * 0.35)), glint_r,
               (255, 255, 255), -1, lineType=cv2.LINE_AA)

    frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return frame


def apply_genjutsu_effect(frame, t):
    """Dolly zoom-in + red tint + a Sharingan eye fading in at center - the
    signature 'caught in genjutsu' visual."""
    h, w = frame.shape[:2]
    zoom = 1.0 + 1.1 * min(t * 1.6, 1.0)
    nw, nh = max(2, int(w / zoom)), max(2, int(h / zoom))
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    crop = frame[y0:y0 + nh, x0:x0 + nw]
    frame[:] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    tint = frame.astype(np.float32)
    tint[:, :, 2] = np.clip(tint[:, :, 2] * 1.35 + 20, 0, 255)  # boost red
    tint[:, :, 0] *= 0.55                                       # cut blue
    tint[:, :, 1] *= 0.55                                       # cut green
    frame[:] = tint.astype(np.uint8)

    eye_alpha = min(1.0, t * 2.2)
    radius = int(min(w, h) * 0.22)
    frame = draw_sharingan(frame, (w // 2, h // 2), radius, rotation_deg=t * 160, alpha=eye_alpha)
    return frame


def apply_flame_effect(frame, color, t):
    """Jet-black licking flame tongues rising from center (Amaterasu) -
    flickering teardrop shapes rather than a plain colored ring."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rng = np.random.default_rng(int(t * 500))
    overlay = frame.copy()

    max_r = min(w, h) * 0.32 * min(1.0, t * 1.6)
    n_flames = 10
    for i in range(n_flames):
        base_ang = 2 * math.pi * i / n_flames + rng.uniform(-0.15, 0.15)
        flicker = rng.uniform(0.7, 1.3)
        length = max_r * flicker
        bx = cx + math.cos(base_ang) * max_r * 0.15
        by = cy + math.sin(base_ang) * max_r * 0.15
        pts = [(bx, by)]
        n_seg = 5
        for s in range(1, n_seg + 1):
            tt2 = s / n_seg
            wobble = rng.uniform(-0.35, 0.35) * (1 - tt2)
            ang = base_ang + wobble
            r = length * tt2
            pts.append((cx + math.cos(ang) * r * 0.3 + math.cos(base_ang) * r * 0.7,
                        cy + math.sin(ang) * r * 0.3 + math.sin(base_ang) * r * 0.7))
        pts_arr = np.array(pts, dtype=np.int32)
        thickness = max(2, int(max_r * 0.09))
        for s in range(len(pts_arr) - 1):
            tt3 = s / max(1, len(pts_arr) - 2)
            th = max(1, int(thickness * (1 - tt3 * 0.8)))
            cv2.line(overlay, tuple(pts_arr[s]), tuple(pts_arr[s + 1]), (25, 10, 35), th, lineType=cv2.LINE_AA)
        cv2.circle(overlay, tuple(pts_arr[-1]), max(1, thickness // 2), (60, 30, 90), -1, lineType=cv2.LINE_AA)

    cv2.circle(overlay, (cx, cy), max(2, int(max_r * 0.18)), (10, 5, 15), -1, lineType=cv2.LINE_AA)
    add_ink_outline(overlay, (cx, cy), int(max_r), thickness=max(2, int(max_r * 0.03)))
    alpha = min(1.0, t * 1.8) * 0.85
    frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return frame


def apply_susanoo_effect(frame, color, t):
    """Glowing purple skeletal ribcage-like aura pulsing outward, evoking
    Susanoo's energy form rather than a plain color flash."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    overlay = np.zeros_like(frame)

    scale = min(1.0, t * 1.4)
    max_r = min(w, h) * 0.42 * scale
    n_ribs = 6
    for i in range(n_ribs):
        frac = (i + 1) / n_ribs
        r = max_r * frac
        ang0 = -70 - i * 4
        ang1 = 70 + i * 4
        cv2.ellipse(overlay, (cx, cy - int(max_r * 0.1)), (int(r * 0.55), int(r)), 0,
                    ang0, ang1, color, thickness=max(2, int(6 * (1 - frac * 0.5))), lineType=cv2.LINE_AA)
    cv2.ellipse(overlay, (cx, cy - int(max_r * 0.1)), (int(max_r * 0.2), int(max_r * 0.55)), 0,
                0, 360, (255, 220, 255), thickness=3, lineType=cv2.LINE_AA)

    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=3)
    alpha = min(1.0, t * 1.6) * 0.8
    frame[:] = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)
    return frame


def apply_vortex_effect(frame, t):
    """An actual spatial swirl/warp of the frame pixels sucking inward
    (Kamui), not just a colored overlay - built with a polar-coordinate
    remap so the image itself visibly twists."""
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    strength = min(1.0, t * 1.3) * 9.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    max_r = math.hypot(cx, cy)
    swirl_amount = strength * np.exp(-(r / (max_r * 0.5)) ** 2)
    theta = np.arctan2(dy, dx) + swirl_amount * 0.35
    pull = 1.0 - 0.25 * np.exp(-(r / (max_r * 0.4)) ** 2) * min(1.0, t * 1.3)
    map_x = (cx + np.cos(theta) * r * pull).astype(np.float32)
    map_y = (cy + np.sin(theta) * r * pull).astype(np.float32)

    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    frame[:] = warped

    tint = np.full_like(frame, (220, 150, 255))
    a = 0.12 * min(1.0, t * 1.5)
    frame[:] = cv2.addWeighted(frame, 1 - a, tint, a, 0)
    return frame


def apply_byakugan_effect(frame, color, t):
    """Pale radiating vein lines bursting outward from the eye area, plus a
    soft glow flash - evokes the Byakugan's bulging-vein activation rather
    than a flat colored flash."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.42)
    overlay = frame.copy()

    flash_a = max(0.0, 1.0 - t * 2.0) * 0.25
    if flash_a > 0:
        tintlayer = np.full_like(frame, color, dtype=np.uint8)
        frame[:] = cv2.addWeighted(frame, 1 - flash_a, tintlayer, flash_a, 0)

    n_veins = 22
    max_len = min(w, h) * 0.3 * min(1.0, t * 1.8)
    rng = np.random.default_rng(3)
    for i in range(n_veins):
        ang = 2 * math.pi * i / n_veins + rng.uniform(-0.05, 0.05)
        length = max_len * rng.uniform(0.6, 1.0)
        x0, y0 = cx, cy
        n_seg = 3
        for s in range(n_seg):
            t2 = (s + 1) / n_seg
            jitter = rng.uniform(-0.12, 0.12)
            x1 = int(cx + math.cos(ang + jitter) * length * t2)
            y1 = int(cy + math.sin(ang + jitter) * length * t2)
            th = max(1, int(3 * (1 - t2)))
            cv2.line(overlay, (x0, y0), (x1, y1), color, th, lineType=cv2.LINE_AA)
            x0, y0 = x1, y1

    for ex in (cx - int(min(w, h) * 0.05), cx + int(min(w, h) * 0.05)):
        cv2.circle(overlay, (ex, cy), max(2, int(min(w, h) * 0.02)), (250, 250, 250), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (ex, cy), max(2, int(min(w, h) * 0.02)), (100, 90, 120), 2, lineType=cv2.LINE_AA)

    alpha = min(1.0, t * 2.0) * 0.9
    frame[:] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return frame


def apply_eightgates_effect(frame, color, t):
    """Intense red-orange aura/steam bursting off the whole frame edges,
    with rising heat-haze wisps, evoking the Eight Gates' overwhelming
    chakra output rather than a flat color flash."""
    h, w = frame.shape[:2]
    pulse = 0.5 + 0.5 * math.sin(t * 28)  # fast pulsing aura

    vignette = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.hypot(cx, cy)
    vignette = np.clip(r - 0.35, 0, 1) ** 1.5
    edge_alpha = (vignette * (0.35 + 0.25 * pulse) * min(1.0, t * 1.6))[:, :, None]
    tint = np.full_like(frame, color, dtype=np.float32)
    frame[:] = (frame.astype(np.float32) * (1 - edge_alpha) + tint * edge_alpha).astype(np.uint8)

    overlay = np.zeros_like(frame)
    rng = np.random.default_rng(int(t * 200))
    n_wisps = 10
    for _ in range(n_wisps):
        bx = rng.uniform(0, w)
        by = h - rng.uniform(0, h * 0.3)
        length = rng.uniform(0.15, 0.35) * h
        sway = rng.uniform(-25, 25)
        pts = np.array([[bx, by], [bx + sway * 0.5, by - length * 0.5], [bx + sway, by - length]], dtype=np.int32)
        cv2.polylines(overlay, [pts], False, color, thickness=max(2, int(w * 0.006)), lineType=cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=4)
    frame[:] = cv2.addWeighted(frame, 1.0, overlay, 0.5 * min(1.0, t * 1.5), 0)
    return frame


def apply_screen_shake(frame, tt, duration_frac=0.22, magnitude=14):
    """Brief camera-shake jolt right at activation - decaying random jitter
    via an affine translation, another staple of anime impact moments."""
    if tt > duration_frac:
        return frame
    p = 1.0 - (tt / duration_frac)
    amp = magnitude * (p ** 1.5)
    if amp < 0.5:
        return frame
    rng = np.random.default_rng(int(tt * 9973))
    dx = rng.uniform(-amp, amp)
    dy = rng.uniform(-amp, amp)
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    frame[:] = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return frame


def apply_radial_zoom_blur(frame, center, tt, duration_frac=0.16, n_layers=6, max_scale=1.22):
    """Layers several progressively scaled-up, fading copies of the frame on
    top of itself - a cheap but very effective fake of the 'zoom punch'
    radial motion blur used for impact frames in anime."""
    if tt > duration_frac:
        return frame
    p = 1.0 - (tt / duration_frac)
    if p < 0.03:
        return frame
    h, w = frame.shape[:2]
    cx, cy = center
    acc = frame.astype(np.float32)
    base_weight = 0.85
    for i in range(1, n_layers + 1):
        s = 1.0 + (max_scale - 1.0) * (i / n_layers)
        M = cv2.getRotationMatrix2D((cx, cy), 0, s)
        layer = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        wgt = base_weight * (i / n_layers) * p * 0.12
        acc = acc * (1 - wgt) + layer.astype(np.float32) * wgt
    frame[:] = np.clip(acc, 0, 255).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Real neural anime style transfer (AnimeGANv2), used only as a brief
# freeze-frame "finishing move" splash at the moment a jutsu lands - too
# heavy to run every frame on a CPU, but a single frame is fast enough to
# feel snappy. Falls back silently (procedural effects only) if torch/the
# model aren't available or fail to load, so the app still works without it.
# First use downloads the pretrained weights via torch.hub (needs internet).
# ---------------------------------------------------------------------------
_anime_model = None
_anime_model_load_failed = False
_anime_result_lock = threading.Lock()
_anime_result = {"image": None, "requested_at": 0.0}


def _load_anime_model():
    global _anime_model, _anime_model_load_failed
    if _anime_model is not None or _anime_model_load_failed:
        return _anime_model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator",
                                pretrained="face_paint_512_v2", device=device)
        model.eval()
        _anime_model = (model, device)
        print(f"[anime style] AnimeGANv2 loaded on {device}.")
    except Exception as e:
        _anime_model_load_failed = True
        print(f"[anime style] Could not load AnimeGANv2 ({e}). "
              f"Finishing-move splash will be skipped; procedural effects still run.")
    return _anime_model


def _stylize_with_animegan(frame_bgr):
    model, device = _anime_model
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # model expects manageable input size; downscale for speed then upscale result
    target_w = 512
    scale = target_w / w
    small = cv2.resize(rgb, (target_w, int(h * scale) // 4 * 4))
    tensor = torch.from_numpy(small).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    tensor = tensor.to(device)
    with torch.no_grad():
        out = model(tensor)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = ((out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def request_anime_stylize(frame_bgr, request_id):
    """Kick off stylization on a background thread so it never blocks the
    main webcam loop. Only the most recent request's result is kept."""
    if not ANIME_MODEL_AVAILABLE:
        return

    def _worker():
        model = _load_anime_model()
        if model is None:
            return
        try:
            styled = _stylize_with_animegan(frame_bgr)
        except Exception as e:
            print(f"[anime style] stylization failed: {e}")
            return
        with _anime_result_lock:
            if request_id >= _anime_result["requested_at"]:
                _anime_result["image"] = styled
                _anime_result["requested_at"] = request_id

    threading.Thread(target=_worker, daemon=True).start()


def get_anime_result():
    with _anime_result_lock:
        return _anime_result["image"]


def clear_anime_result():
    with _anime_result_lock:
        _anime_result["image"] = None


# ---------------------------------------------------------------------------
# Optional royalty-free video overlays (speed lines, lightning, energy
# bursts). This app draws all of its own effects procedurally and needs no
# external files to run - but if you want real stock-footage overlays layered
# on top, download free clips (Pixabay/Vecteezy - royalty-free, no
# attribution required) and drop them in an "overlays" folder next to this
# script, named to match what they should be used for:
#   overlays/speed_lines.mp4   - converging speed-line burst
#   overlays/lightning.mp4     - electric/crackle overlay (used for Chidori)
#   overlays/energy.mp4        - glowing energy burst (used for Rasenshuriken)
#   overlays/explosion.mp4     - explosion burst (used for fire/burst jutsu)
# Any file you don't provide is simply skipped - nothing else changes.
# ---------------------------------------------------------------------------
OVERLAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlays")
OVERLAY_FILES = {
    "speed_lines": "speed_lines.mp4",
    "lightning": "lightning.mp4",
    "energy": "energy.mp4",
    "explosion": "explosion.mp4",
}


class VideoOverlay:
    """Loops a short video file and returns frames resized to the target
    size. Used to screen-blend real stock-footage FX on top of the
    procedural effects when available."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.available = self.cap.isOpened()

    def get_frame(self, w, h):
        if not self.available:
            return None
        ok, frame = self.cap.read()
        if not ok:  # loop back to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        return cv2.resize(frame, (w, h))


_overlay_cache = {}
_overlay_hint_printed = False


def get_overlay(name):
    global _overlay_hint_printed
    if name not in _overlay_cache:
        path = os.path.join(OVERLAY_DIR, OVERLAY_FILES.get(name, ""))
        if os.path.exists(path):
            _overlay_cache[name] = VideoOverlay(path)
        else:
            _overlay_cache[name] = None
            if not _overlay_hint_printed:
                print(f"[overlays] No files found in {OVERLAY_DIR}/ - running with procedural "
                      f"effects only. Drop free clips there (see script header) to layer real "
                      f"stock-footage FX on top.")
                _overlay_hint_printed = True
    return _overlay_cache[name]


def screen_blend_overlay(frame, overlay_name, strength=0.7):
    """Screen-blends a looping video overlay on top of frame, if that
    overlay file is present. Screen blending naturally ignores black
    background in the source clip, which is how most FX overlay footage is
    authored. No-op (returns frame unchanged) if the file isn't available."""
    ov = get_overlay(overlay_name)
    if ov is None:
        return frame
    h, w = frame.shape[:2]
    ov_frame = ov.get_frame(w, h)
    if ov_frame is None:
        return frame
    f = frame.astype(np.float32) / 255.0
    o = ov_frame.astype(np.float32) / 255.0 * strength
    screened = 1.0 - (1.0 - f) * (1.0 - o)
    frame[:] = np.clip(screened * 255.0, 0, 255).astype(np.uint8)
    return frame


def apply_toon_filter(frame, strength):
    """Cheap real-time cartoon/cel-shade filter: smooth flat color regions
    (via a small bilateral filter at reduced resolution, for speed) + black
    ink edge-lines from an adaptive threshold on the luminance - the classic
    'cartoonizer' recipe. strength is 0..1, blended against the original so
    it can fade in/out rather than snapping the whole feed into toon mode."""
    if strength <= 0.01:
        return frame
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    for _ in range(2):
        small = cv2.bilateralFilter(small, d=7, sigmaColor=55, sigmaSpace=55)
    color = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=9, C=3)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    toon = cv2.bitwise_and(color, edges_color)

    # posterize the flat-color layer slightly for a cel-shaded step look
    levels = 5
    toon = (np.round(toon.astype(np.float32) / 255 * (levels - 1)) / (levels - 1) * 255).astype(np.uint8)

    out = cv2.addWeighted(frame, 1 - strength, toon, strength, 0)
    frame[:] = out
    return frame


def add_ink_outline(frame, center, radius, color=(0, 0, 0), thickness=None):
    """A bold black ring stroke around an effect's silhouette - the graphic
    ink-linework anime effects are drawn with, instead of a soft painterly
    edge."""
    if thickness is None:
        thickness = max(2, int(radius * 0.045))
    cv2.circle(frame, center, max(1, int(radius)), color, thickness, lineType=cv2.LINE_AA)
    return frame


def apply_impact_flash(frame, tt, duration_frac=0.14):
    """A hard white flash at the instant of activation - the classic anime
    'impact frame' that sells power/weight to a technique landing."""
    if tt > duration_frac:
        return frame
    p = tt / duration_frac
    flash_a = (1.0 - p) ** 2  # fast decay
    if flash_a > 0.01:
        white = np.full_like(frame, 255, dtype=np.uint8)
        frame[:] = cv2.addWeighted(frame, 1 - flash_a, white, flash_a, 0)
    return frame


def draw_speed_lines(frame, center, tt, color=(255, 255, 255), duration_frac=0.45, n_lines=44):
    """Manga-style radiating focus/speed lines converging on the effect's
    center, strongest right after activation and fading out - the other
    core piece of 'straight out of the anime' visual language alongside the
    impact flash."""
    if tt > duration_frac:
        return frame
    p = tt / duration_frac
    alpha = (1.0 - p) ** 1.5
    if alpha < 0.02:
        return frame

    h, w = frame.shape[:2]
    cx, cy = center
    overlay = np.zeros_like(frame)
    diag = math.hypot(w, h)
    rng = np.random.default_rng(1234)
    for i in range(n_lines):
        ang = 2 * math.pi * i / n_lines + rng.uniform(-0.02, 0.02)
        inner = diag * (0.06 + 0.05 * rng.uniform(0, 1))
        outer = diag * rng.uniform(0.55, 0.95)
        x0 = int(cx + math.cos(ang) * inner)
        y0 = int(cy + math.sin(ang) * inner)
        x1 = int(cx + math.cos(ang) * outer)
        y1 = int(cy + math.sin(ang) * outer)
        th = 1 if i % 3 else 2
        cv2.line(overlay, (x0, y0), (x1, y1), color, th, lineType=cv2.LINE_AA)

    frame[:] = cv2.addWeighted(frame, 1.0, overlay, alpha * 0.8, 0)
    return frame


def apply_anime_punch(frame, tt, duration_frac=0.5):
    """Brief contrast/saturation boost right after activation - cheap but
    effective cel-shaded 'pop' rather than the flatter natural-video look."""
    if tt > duration_frac:
        return frame
    p = 1.0 - (tt / duration_frac)
    amount = 0.35 * p
    if amount < 0.02:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + amount * 0.6), 0, 255)
    frame[:] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    frame[:] = np.clip(frame.astype(np.float32) * (1 + amount * 0.15) - amount * 8, 0, 255).astype(np.uint8)
    return frame


def apply_jutsu_vfx(frame, name, t, duration):
    tt = min(1.0, t / duration)
    eff_type, color = get_jutsu_effect(name)
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    if eff_type == "genjutsu":
        frame = apply_genjutsu_effect(frame, tt)
    elif eff_type == "smoke":
        frame = apply_smoke_effect(frame, color, tt)
    elif eff_type == "flame":
        frame = apply_flame_effect(frame, color, tt)
    elif eff_type == "susanoo":
        frame = apply_susanoo_effect(frame, color, tt)
    elif eff_type == "vortex":
        frame = apply_vortex_effect(frame, tt)
    elif eff_type == "byakugan":
        frame = apply_byakugan_effect(frame, color, tt)
    elif eff_type == "eightgates":
        frame = apply_eightgates_effect(frame, color, tt)
    else:
        frame = apply_burst_effect(frame, color, tt)

    line_color = (255, 255, 255) if eff_type in ("flame", "susanoo", "vortex") else (245, 245, 245)
    frame = draw_speed_lines(frame, center, tt, color=line_color)

    overlay_alpha = min(1.0, tt * 3) * min(1.0, (1.0 - tt) * 2.2)
    if overlay_alpha > 0.03:
        frame = screen_blend_overlay(frame, "speed_lines", strength=0.5 * overlay_alpha)
        if "lightning" in name.lower() or "darkness" in name.lower():
            frame = screen_blend_overlay(frame, "lightning", strength=0.6 * overlay_alpha)
        elif eff_type in ("flame", "burst"):
            frame = screen_blend_overlay(frame, "explosion", strength=0.5 * overlay_alpha)

    frame = apply_radial_zoom_blur(frame, center, tt)
    frame = apply_anime_punch(frame, tt)
    frame = apply_screen_shake(frame, tt)

    toon_strength = min(1.0, tt * 4) * min(1.0, (1.0 - tt) * 3) * 0.8
    frame = apply_toon_filter(frame, toon_strength)

    frame = apply_impact_flash(frame, tt)
    return frame


def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    print_handsign_book()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    gl = GLRenderer(frame_w, frame_h)
    rng = np.random.default_rng()
    sounds = pregenerate_sounds()

    mouse_state = {"execute_clicked": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and EXECUTE_BUTTON_RECT is not None:
            if point_in_rect(x, y, EXECUTE_BUTTON_RECT):
                mouse_state["execute_clicked"] = True

    cv2.namedWindow("Konoha Ninja Scanner")
    cv2.setMouseCallback("Konoha Ninja Scanner", _on_mouse)
    # Pre-warm the audio backend: the very first sd.play() call on many
    # systems pays a one-time cost to open the output device, which shows up
    # as a delayed first sound relative to the visual trigger. Eat that cost
    # now, before the loop starts, with a near-silent buffer.
    try:
        sd.play(np.zeros(int(SAMPLE_RATE * 0.05), dtype=np.float32), SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass

    sequence = []
    current_sign = None
    last_committed = None
    last_change_time = time.time()
    hold_seconds = 0.6

    jutsu_text = ""
    jutsu_timer = 0.0

    rasen_charge_start = None
    rasen_spin_angle = 0.0
    chidori_charge_start = None
    rasen_charge_sound_played = False
    rasen_release_sound_played = False
    chidori_charge_sound_played = False
    chidori_release_sound_played = False
    rasen_release_flash_start = None
    chidori_release_flash_start = None
    show_book = False
    fps_smooth = None
    prev_frame_time = time.time()
    jutsu_vfx_name = None
    jutsu_vfx_start = 0.0
    anime_splash_start = None
    anime_splash_requested = False
    JUTSU_VFX_DURATION = {"genjutsu": 1.8, "smoke": 1.0, "burst": 0.9,
                           "flame": 1.3, "susanoo": 1.6, "vortex": 1.5,
                           "byakugan": 1.2, "eightgates": 1.4}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _now_ft = time.time()
            _dt = _now_ft - prev_frame_time
            prev_frame_time = _now_ft
            if _dt > 0:
                inst_fps = 1.0 / _dt
                fps_smooth = inst_fps if fps_smooth is None else (fps_smooth * 0.9 + inst_fps * 0.1)
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_sign = None
            palm_center = None
            hand_scale_px = 80
            normal = np.array([0.0, 0.0, 1.0])
            finger_state_debug = None
            h, w = frame.shape[:2]

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness_label = "Right"
                if result.handedness:
                    handedness_label = result.handedness[0][0].category_name

                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

                wrist = landmarks[WRIST]
                mcp = landmarks[MIDDLE_MCP]
                palm_center = ((wrist.x + mcp.x) / 2 * w, (wrist.y + mcp.y) / 2 * h)
                hand_scale_px = max(40, math.hypot((wrist.x - mcp.x) * w, (wrist.y - mcp.y) * h) * 2.2)
                normal = palm_orientation_normal(landmarks)

                up = fingers_up(landmarks, handedness_label)
                current_sign = classify_sign(up)
                finger_state_debug = tuple(up[f] for f in FINGER_NAMES)

            now = time.time()
            charge = 0.0
            charge_label = "Charging..."
            gl_rgba = None

            if current_sign == RASEN_TRIGGER_SIGN and palm_center is not None:
                chidori_charge_start = None
                chidori_charge_sound_played = False
                chidori_release_sound_played = False
                if rasen_charge_start is None:
                    rasen_charge_start = now
                if not rasen_charge_sound_played:
                    play_sound_synced(sounds["rasen_charge"], 1.0 / fps_smooth if fps_smooth else 0.0)
                    rasen_charge_sound_played = True
                charge = (now - rasen_charge_start) / RASEN_CHARGE_SECONDS
                rasen_spin_angle += 12 + 16 * min(1.0, charge)
                radius_px = hand_scale_px * (0.3 + 0.9 * min(1.0, charge))

                gl.begin_frame()
                world_r = gl.set_object_transform(palm_center[0], palm_center[1], radius_px,
                                                   normal, rasen_spin_angle)
                gl.draw_rasenshuriken(world_r)
                gl_rgba = gl.read_rgba()
                gl.end_frame()

                charge_label = "Rasenshuriken!" if charge >= 1.0 else "Charging..."
                if charge >= 1.0:
                    jutsu_text = "Rasenshuriken"
                    jutsu_timer = now + 1.0
                    if not rasen_release_sound_played:
                        play_sound_synced(sounds["rasen_release"], 1.0 / fps_smooth if fps_smooth else 0.0)
                        rasen_release_sound_played = True
                        rasen_release_flash_start = now
                        if ANIME_MODEL_AVAILABLE:
                            request_anime_stylize(frame.copy(), now)
                            anime_splash_requested = True

            elif current_sign == CHIDORI_TRIGGER_SIGN and palm_center is not None:
                rasen_charge_start = None
                rasen_charge_sound_played = False
                rasen_release_sound_played = False
                if chidori_charge_start is None:
                    chidori_charge_start = now
                if not chidori_charge_sound_played:
                    play_sound_synced(sounds["chidori_charge"], 1.0 / fps_smooth if fps_smooth else 0.0)
                    chidori_charge_sound_played = True
                charge = (now - chidori_charge_start) / CHIDORI_CHARGE_SECONDS
                radius_px = hand_scale_px * (0.5 + 0.7 * min(1.0, charge))

                gl.begin_frame()
                world_r = gl.set_object_transform(palm_center[0], palm_center[1], radius_px,
                                                   normal, 0.0)
                gl.draw_chidori(world_r, rng)
                gl_rgba = gl.read_rgba()
                gl.end_frame()

                charge_label = "Chidori!" if charge >= 1.0 else "Charging..."
                if charge >= 1.0:
                    jutsu_text = "Chidori"
                    jutsu_timer = now + 1.0
                    if not chidori_release_sound_played:
                        play_sound_synced(sounds["chidori_release"], 1.0 / fps_smooth if fps_smooth else 0.0)
                        chidori_release_sound_played = True
                        chidori_release_flash_start = now
                        if ANIME_MODEL_AVAILABLE:
                            request_anime_stylize(frame.copy(), now)
                            anime_splash_requested = True
            else:
                rasen_charge_start = None
                chidori_charge_start = None
                rasen_charge_sound_played = False
                rasen_release_sound_played = False
                chidori_charge_sound_played = False
                chidori_release_sound_played = False
                rasen_release_flash_start = None
                chidori_release_flash_start = None

            if gl_rgba is not None:
                composite_rgba(frame, gl_rgba, min(1.0, charge))

            if rasen_release_flash_start is not None:
                rt = now - rasen_release_flash_start
                if rt <= 0.5:
                    ptt = rt / 0.5
                    ov_a = min(1.0, ptt * 3) * min(1.0, (1.0 - ptt) * 2.2)
                    frame = draw_speed_lines(frame, (w // 2, h // 2), ptt, color=(255, 255, 255))
                    if ov_a > 0.03:
                        frame = screen_blend_overlay(frame, "speed_lines", strength=0.5 * ov_a)
                        frame = screen_blend_overlay(frame, "energy", strength=0.6 * ov_a)
                    frame = apply_radial_zoom_blur(frame, (w // 2, h // 2), ptt)
                    frame = apply_screen_shake(frame, ptt)
                    toon_s = min(1.0, ptt * 4) * min(1.0, (1.0 - ptt) * 3) * 0.8
                    frame = apply_toon_filter(frame, toon_s)
                    frame = apply_impact_flash(frame, ptt)
                else:
                    rasen_release_flash_start = None
            if chidori_release_flash_start is not None:
                ct = now - chidori_release_flash_start
                if ct <= 0.5:
                    ptt = ct / 0.5
                    ov_a = min(1.0, ptt * 3) * min(1.0, (1.0 - ptt) * 2.2)
                    frame = draw_speed_lines(frame, (w // 2, h // 2), ptt, color=(220, 235, 255))
                    if ov_a > 0.03:
                        frame = screen_blend_overlay(frame, "speed_lines", strength=0.5 * ov_a)
                        frame = screen_blend_overlay(frame, "lightning", strength=0.6 * ov_a)
                    frame = apply_radial_zoom_blur(frame, (w // 2, h // 2), ptt)
                    frame = apply_screen_shake(frame, ptt)
                    toon_s = min(1.0, ptt * 4) * min(1.0, (1.0 - ptt) * 3) * 0.8
                    frame = apply_toon_filter(frame, toon_s)
                    frame = apply_impact_flash(frame, ptt)
                else:
                    chidori_release_flash_start = None

            if current_sign != last_committed:
                last_change_time = now
                last_committed = current_sign
            elif current_sign and current_sign not in (RASEN_TRIGGER_SIGN, CHIDORI_TRIGGER_SIGN) and (now - last_change_time) > hold_seconds:
                if not sequence or sequence[-1] != current_sign:
                    sequence.append(current_sign)
                    sequence = sequence[-MAX_SEQ_LEN:]
                    match = match_jutsu(sequence)
                    if match:
                        jutsu_text = match
                        jutsu_timer = now + 2.5
                        jutsu_vfx_name = match
                        jutsu_vfx_start = now
                        sequence = []
                        play_sound_synced(sounds["poof"], 1.0 / fps_smooth if fps_smooth else 0.0)
                        if ANIME_MODEL_AVAILABLE:
                            request_anime_stylize(frame.copy(), now)
                            anime_splash_requested = True
                last_change_time = now + 999

            if jutsu_vfx_name is not None:
                eff_type, _ = get_jutsu_effect(jutsu_vfx_name)
                dur = JUTSU_VFX_DURATION[eff_type]
                elapsed = now - jutsu_vfx_start
                if elapsed <= dur:
                    frame = apply_jutsu_vfx(frame, jutsu_vfx_name, elapsed, dur)
                else:
                    jutsu_vfx_name = None

            # Neural anime-style finishing-move splash: once the background
            # stylization finishes, show it full-screen for a couple seconds
            # (it arrives asynchronously, so this may appear slightly after
            # the procedural impact effects rather than in perfect lockstep).
            if anime_splash_requested:
                styled = get_anime_result()
                if styled is not None:
                    if anime_splash_start is None:
                        anime_splash_start = now
                    splash_elapsed = now - anime_splash_start
                    if splash_elapsed <= 2.0:
                        splash_alpha = min(1.0, splash_elapsed * 6) * min(1.0, (2.0 - splash_elapsed) * 2)
                        resized = cv2.resize(styled, (w, h))
                        frame[:] = cv2.addWeighted(frame, 1 - splash_alpha, resized, splash_alpha, 0)
                        cv2.putText(frame, "ANIME CUT-IN", (16, h - 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        anime_splash_requested = False
                        anime_splash_start = None
                        clear_anime_result()

            EXECUTE_BUTTON_RECT = compute_execute_button_rect(w, h)
            frame = draw_hud(frame, current_sign, sequence, jutsu_text, jutsu_timer, charge, charge_label, fps_smooth, finger_state_debug)
            if show_book:
                frame = draw_handsign_book(frame)
            can_execute = bool(current_sign) and current_sign not in (RASEN_TRIGGER_SIGN, CHIDORI_TRIGGER_SIGN)
            frame = draw_execute_button(frame, EXECUTE_BUTTON_RECT, pressed=not can_execute)
            cv2.imshow("Konoha Ninja Scanner", frame)

            key = cv2.waitKey(1) & 0xFF
            execute_pressed = mouse_state["execute_clicked"] or key == ord(" ")
            mouse_state["execute_clicked"] = False

            if key == ord("q"):
                break
            elif key == ord("h"):
                show_book = not show_book
            elif key == ord("c"):
                sequence = []
                rasen_charge_start = None
                chidori_charge_start = None
            elif key in (ord("z"), 8):  # 8 = Backspace
                if sequence:
                    sequence.pop()
            elif execute_pressed and can_execute:
                sequence.append(current_sign)
                match = match_jutsu(sequence)
                if match:
                    jutsu_text = match
                    jutsu_timer = time.time() + 2.5
                    jutsu_vfx_name = match
                    jutsu_vfx_start = time.time()
                    sequence = []
                    play_sound_synced(sounds["poof"], 1.0 / fps_smooth if fps_smooth else 0.0)
                    if ANIME_MODEL_AVAILABLE:
                        request_anime_stylize(frame.copy(), time.time())
                        anime_splash_requested = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        pygame.quit()


if __name__ == "__main__":
    main()
