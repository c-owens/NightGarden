"""
Night Garden: backing track
Rendered entirely from code: additive-synth pads, FM glass bells, synthesized
crickets / cicadas / owl / wind, convolution reverb. Key: A minor pentatonic.

    python compose.py            -> night_garden.wav
"""
import numpy as np
from scipy import signal
import wave

SR = 44100
rng = np.random.default_rng(5023)          # fixed seed: same track every render

def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)

# ---------------- form ----------------
INTRO = 24.0                 # night sounds, pad fades in at the end
CHORD = 12.0                 # seconds per chord (60 bpm, 12 beats)
PROG_A = [[45, 52, 59, 60, 64],   # Am9
          [41, 48, 57, 60, 64],   # Fmaj7
          [48, 55, 62, 64, 67],   # Cadd9
          [40, 47, 55, 59, 62]]   # Em7
PROG_B = PROG_A[:3] + [[43, 50, 59, 62, 67]]   # ...ends on G for lift
CYCLES = [PROG_A, PROG_A, PROG_B, PROG_A]
SECTION_LEN = CHORD * 4
MUSIC_END = INTRO + SECTION_LEN * len(CYCLES)
OUTRO = 34.0
DUR = MUSIC_END + OUTRO
N = int(DUR * SR)

L = np.zeros(N, np.float32); R = np.zeros(N, np.float32)          # dry
bellL = np.zeros(N, np.float32); bellR = np.zeros(N, np.float32)  # goes through echo
nightL = np.zeros(N, np.float32); nightR = np.zeros(N, np.float32)

def add(bufL, bufR, start, sig, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i >= N: return
    sig = sig[: N - i] * gain
    a = (pan + 1) * np.pi / 4                      # equal-power pan
    bufL[i:i + len(sig)] += sig * np.cos(a)
    bufR[i:i + len(sig)] += sig * np.sin(a)

def t_axis(sec):
    return np.arange(int(sec * SR)) / SR

def env_adsr(n, a, r, sus_end):
    t = np.arange(n) / SR
    e = np.clip(t / a, 0, 1)
    e = np.where(t > sus_end, e * np.clip(1 - (t - sus_end) / r, 0, 1), e)
    return (np.sin(e * np.pi / 2) ** 2).astype(np.float32)

# ---------------- pad (additive, detuned, warm) ----------------
def pad_note(m, length, bright):
    tail = 6.0
    t = t_axis(length + tail)
    out = np.zeros(len(t), np.float32)
    f0 = mtof(m)
    for det in (-7, 0, 6):                         # cents
        f = f0 * 2 ** (det / 1200)
        ph = rng.uniform(0, 2 * np.pi)
        for h in range(1, 11):
            if f * h > 7000: break
            amp = (1 / h ** 1.4) * np.exp(-h / bright)
            out += (amp * np.sin(2 * np.pi * f * h * t + ph * h)).astype(np.float32)
    # slow breathing tremolo
    out *= (0.85 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.05, 0.11) * t + rng.uniform(0, 6))).astype(np.float32)
    return out * env_adsr(len(t), 3.5, tail, length) / 3

def sub(m, length):
    t = t_axis(length + 4)
    s = np.sin(2 * np.pi * mtof(m - 12) * t).astype(np.float32)
    return s * env_adsr(len(t), 2.5, 4, length)

# ---------------- glass bell (2-op FM) ----------------
def bell(m, decay=3.2, index=1.6, ratio=2.0):
    t = t_axis(decay * 2.2)
    f = mtof(m)
    I = index * np.exp(-t / (decay * 0.35))
    s = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t))
    s += 0.18 * np.sin(2 * np.pi * f * 3.51 * t) * np.exp(-t / 0.4)   # shimmer partial
    e = np.exp(-t / decay) * np.clip(t / 0.004, 0, 1)
    return (s * e).astype(np.float32)

# ---------------- build the music ----------------
brightness = [2.2, 3.0, 4.0, 2.6]
pad_gain   = [0.11, 0.12, 0.13, 0.10]

MELODY = {   # beat offsets within a chord (1 beat = 1 s) -> midi
    0: [(0, 76), (1.5, 74), (3, 72), (6, 69), (9, 72), (10.5, 74)],
    1: [(0, 72), (1.5, 76), (3, 79), (6, 76), (9, 74)],
    2: [(0, 76), (1.5, 79), (3, 81), (6, 79), (8, 76), (10, 74)],
    3: [(0, 74), (1.5, 71), (3, 67), (6, 71), (9, 74), (10.5, 76)],
}
PENTA = [57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84]

for c, prog in enumerate(CYCLES):
    for k, chord in enumerate(prog):
        t0 = INTRO + c * SECTION_LEN + k * CHORD
        # pad: lower notes centered, upper notes spread
        for j, m in enumerate(chord):
            add(L, R, t0, pad_note(m, CHORD, brightness[c]), pan=(j - 2) * 0.28, gain=pad_gain[c])
        if c in (1, 2):
            add(L, R, t0, sub(chord[0], CHORD), gain=0.03)

        # melody: second time through, then an octave-shimmer variation
        if c in (1, 2):
            for beat, m in MELODY[k]:
                if c == 2 and k == 3 and m == 76: m = 79            # lift into the G
                add(bellL, bellR, t0 + beat + rng.uniform(-0.02, 0.02),
                    bell(m, decay=2.8, index=1.3), pan=rng.uniform(-0.3, 0.3), gain=0.13)
        # arpeggio in the fullest section
        if c == 2:
            tones = sorted(chord[1:])
            tones = [x + 12 for x in tones]
            for b in range(int(CHORD * 2)):
                m = tones[b % len(tones)] if (b // len(tones)) % 2 == 0 else tones[-1 - b % len(tones)]
                add(bellL, bellR, t0 + b * 0.5 + 0.25, bell(m + 12, decay=1.1, index=0.8, ratio=3.0),
                    pan=0.55 if b % 2 else -0.55, gain=0.035)
        # scattered chimes, like flowers being planted
        density = [0.25, 0.15, 0.1, 0.35][c]
        for b in range(int(CHORD)):
            if rng.random() < density:
                m = int(rng.choice(PENTA)) + 12
                add(bellL, bellR, t0 + b + rng.uniform(0, 1), bell(m, decay=2.0, index=2.0),
                    pan=rng.uniform(-0.8, 0.8), gain=0.06)

# outro: final Am9, high and far away
t0 = MUSIC_END
for j, m in enumerate([45, 57, 64, 71, 72]):
    add(L, R, t0, pad_note(m, 14, 1.8), pan=(j - 2) * 0.3, gain=0.09)
for i, m in enumerate([81, 79, 76, 72, 69]):
    add(bellL, bellR, t0 + 2 + i * 2.2, bell(m, decay=3.5, index=1.0), pan=0.3 - i * 0.15, gain=0.09)

# ---------------- night sounds ----------------
def cricket(freq, period, pulses, start, end, amp, pan):
    t = start
    pulse_n = int(0.022 * SR)
    pt = np.arange(pulse_n) / SR
    shape = np.clip(pt / 0.005, 0, 1) * np.clip((0.022 - pt) / 0.017, 0, 1)
    while t < end:
        if rng.random() < 0.05:
            t += rng.uniform(2, 6); continue
        for k in range(pulses):
            ts = t + k * 0.042
            i = int(ts * SR)
            if i + pulse_n >= N: break
            ph = 2 * np.pi * freq * (ts + pt)
            s = (np.sin(ph) * shape * amp).astype(np.float32)
            a = (pan + 1) * np.pi / 4
            nightL[i:i + pulse_n] += s * np.cos(a); nightR[i:i + pulse_n] += s * np.sin(a)
        t += period * rng.uniform(0.92, 1.08)

for i, pan in enumerate([-0.75, -0.25, 0.3, 0.8]):
    cricket(rng.uniform(3900, 5200), rng.uniform(0.55, 1.1), int(rng.integers(2, 5)),
            start=rng.uniform(0.5, 6), end=DUR - rng.uniform(1, 5), amp=rng.uniform(0.018, 0.035), pan=pan)

# noise source helpers
def noise(sec):
    return rng.standard_normal(int(sec * SR)).astype(np.float32)

def cicada(start, up, hold, down, freq, rate, peak, pan):
    length = up + hold + down
    t = t_axis(length)
    sos = signal.butter(2, [freq * 0.9, freq * 1.1], btype='band', fs=SR, output='sos')
    s = signal.sosfilt(sos, noise(length)).astype(np.float32)
    buzz = (0.5 + 0.5 * np.sin(2 * np.pi * rate * t)) ** 2
    env = np.interp(t, [0, up, up + hold, length], [0, 1, 1, 0])
    add(nightL, nightR, start, (s * buzz * env).astype(np.float32), pan=pan, gain=peak)

cicada(40, 6, 7, 6, 4400, 92, 0.30, -0.5)
cicada(98, 5, 9, 7, 5600, 127, 0.24, 0.6)
cicada(150, 6, 8, 6, 4700, 104, 0.28, -0.2)
cicada(178, 5, 6, 8, 5300, 118, 0.22, 0.4)

def owl(start, pan, base):
    for at, length in [(0, 0.5), (0.95, 0.22), (1.25, 0.22), (1.55, 0.45)]:
        t = t_axis(length)
        f = base * 1.04 * (0.94 / 1.04) ** (t / length)
        ph = 2 * np.pi * np.cumsum(f) / SR
        e = np.clip(t / 0.06, 0, 1) * np.clip((length - t) / 0.08, 0, 1)
        add(bellL, bellR, start + at, (np.sin(ph) * e).astype(np.float32), pan=pan, gain=0.05)

owl(66, -0.6, 355)
owl(205, 0.5, 340)
owl(DUR - 22, -0.3, 365)

# wind: dark noise with slow gusts
wsec = DUR
w = signal.sosfilt(signal.butter(2, [90, 520], btype='band', fs=SR, output='sos'), noise(wsec))
gust = np.interp(np.arange(len(w)), np.linspace(0, len(w), 40), 0.3 + 0.7 * rng.random(40) ** 2)
gust = signal.sosfilt(signal.butter(1, 0.3, fs=SR, output='sos'), gust)
w2 = signal.sosfilt(signal.butter(2, [90, 520], btype='band', fs=SR, output='sos'), noise(wsec))
nightL += (w * gust * 0.07).astype(np.float32)
nightR += (w2 * gust * 0.07).astype(np.float32)

# ---------------- effects ----------------
# ping-pong echo on the bells (dotted-eighth-ish at 60 bpm)
def echo(xL, xR, d=0.75, fb=0.42, taps=5):
    yL, yR = xL.copy(), xR.copy()
    D = int(d * SR)
    lp = signal.butter(1, 3500, fs=SR, output='sos')
    srcL, srcR = xL, xR
    for k in range(1, taps + 1):
        srcL, srcR = signal.sosfilt(lp, srcR).astype(np.float32), signal.sosfilt(lp, srcL).astype(np.float32)  # swap = ping-pong
        g = fb ** k
        yL[D * k:] += srcL[:-D * k] * g
        yR[D * k:] += srcR[:-D * k] * g
    return yL, yR

bellL, bellR = echo(bellL, bellR)

# convolution reverb: decaying stereo noise impulse response
def make_ir(sec=4.5, rt60=4.2):
    t = t_axis(sec)
    dec = np.exp(-6.9 * t / rt60)
    lp = signal.butter(2, 5500, fs=SR, output='sos')
    irs = []
    for _ in range(2):
        n = signal.sosfilt(lp, rng.standard_normal(len(t))) * dec
        n[: int(0.02 * SR)] = 0                                   # pre-delay
        irs.append((n / np.sqrt(np.sum(n ** 2))).astype(np.float32))
    return irs

irL, irR = make_ir()
sendL = L * 0.6 + bellL * 0.8 + nightL * 0.25
sendR = R * 0.6 + bellR * 0.8 + nightR * 0.25
revL = signal.oaconvolve(sendL, irL)[:N].astype(np.float32)
revR = signal.oaconvolve(sendR, irR)[:N].astype(np.float32)

mixL = L + bellL + nightL * 1.7 + revL * 0.55
mixR = R + bellR + nightR * 1.7 + revR * 0.55

# clean up rumble below the music
hp = signal.butter(2, 40, btype='high', fs=SR, output='sos')
mixL = signal.sosfilt(hp, mixL).astype(np.float32); mixR = signal.sosfilt(hp, mixR).astype(np.float32)

# master fades
t = np.arange(N) / SR
fade = np.clip(t / 4, 0, 1) * np.clip((DUR - t) / 8, 0, 1)
mixL *= fade; mixR *= fade

# gentle glue + normalize
peak = max(np.abs(mixL).max(), np.abs(mixR).max())
mixL, mixR = np.tanh(mixL / peak * 1.1) / np.tanh(1.1), np.tanh(mixR / peak * 1.1) / np.tanh(1.1)
out = np.stack([mixL, mixR], axis=1) * 0.89

with wave.open('night_garden.wav', 'wb') as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes((out * 32767).astype('<i2').tobytes())
print(f"rendered {DUR:.1f}s")
