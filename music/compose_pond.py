"""
Moon Pond: backing track for pond.html
Rendered entirely from code: chorused glass pads, kalimba tines, water-drop
plips, synthesized frogs / spring peepers / lapping water, convolution reverb.
Key: D major pentatonic, with a Lydian G for the moonlight.

    python compose_pond.py       -> moon_pond.wav (and moon_pond.mp3 if lameenc is installed)
"""
import numpy as np
from scipy import signal
import wave

SR = 44100
rng = np.random.default_rng(1147)          # fixed seed: same track every render

def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)

# ---------------- form ----------------
BEAT = 60 / 72                    # 72 bpm, but everything floats over it
CHORD = BEAT * 12                 # 10 s per chord
INTRO = 20.0                      # water and frogs, pad drifts in at the end
DMAJ9  = [50, 57, 61, 64, 66]     # D  A  C# E  F#
BM11   = [47, 54, 57, 62, 64]     # B  F# A  D  E
GLYD   = [43, 50, 54, 59, 61]     # G  D  F# B  C#   (Gmaj7#11, the moonlit one)
ASUS2  = [45, 52, 57, 59, 64]     # A  E  A  B  E
EM9    = [40, 47, 54, 55, 62]     # E  B  F# G  D
PROG_A = [DMAJ9, BM11, GLYD, ASUS2]
PROG_B = [DMAJ9, BM11, GLYD, EM9]                 # ...settles on Em before coming home
SECTIONS = [PROG_A, PROG_A, PROG_B, PROG_A, PROG_A]
SECTION_LEN = CHORD * 4
MUSIC_END = INTRO + SECTION_LEN * len(SECTIONS)
OUTRO = 30.0
DUR = MUSIC_END + OUTRO
N = int(DUR * SR)

padL = np.zeros(N, np.float32); padR = np.zeros(N, np.float32)    # goes through chorus
L = np.zeros(N, np.float32); R = np.zeros(N, np.float32)          # dry
tineL = np.zeros(N, np.float32); tineR = np.zeros(N, np.float32)  # goes through echo
natL = np.zeros(N, np.float32); natR = np.zeros(N, np.float32)

def add(bufL, bufR, start, sig, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i >= N or i < 0: return
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

def noise(sec):
    return rng.standard_normal(int(sec * SR)).astype(np.float32)

# ---------------- glass pad (few partials, slow vibrato, chorused later) ----------------
def glass(m, length, air):
    tail = 7.0
    t = t_axis(length + tail)
    f0 = mtof(m)
    vib = 1 + 0.0025 * np.sin(2 * np.pi * rng.uniform(0.18, 0.32) * t + rng.uniform(0, 6))
    ph = 2 * np.pi * np.cumsum(f0 * vib) / SR
    out = np.sin(ph) + 0.32 * np.sin(2 * ph + 1.1) + air * 0.1 * np.sin(3 * ph + 0.4) + air * 0.05 * np.sin(4 * ph)
    # each note swells on its own slow tide
    out *= 0.75 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.04, 0.09) * t + rng.uniform(0, 6))
    return (out * env_adsr(len(t), 4.5, tail, length) / 2.2).astype(np.float32)

def hum(m, length):
    t = t_axis(length + 5)
    s = np.sin(2 * np.pi * mtof(m - 12) * t) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.11 * t))
    return (s * env_adsr(len(t), 3.0, 5, length)).astype(np.float32)

# ---------------- kalimba tine ----------------
def tine(m, decay=1.8):
    t = t_axis(decay * 3)
    f = mtof(m)
    body = np.sin(2 * np.pi * f * t) * np.exp(-t / decay)
    body += 0.12 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t / (decay * 0.4))
    click = 0.3 * np.sin(2 * np.pi * f * 5.43 * t) * np.exp(-t / 0.035)    # the metal tongue
    return ((body + click) * np.clip(t / 0.003, 0, 1)).astype(np.float32)

# ---------------- water drop: a pitch that falls into place ----------------
def plip(m, decay=0.5):
    t = t_axis(decay * 4)
    f = mtof(m) * (1 + 0.6 * np.exp(-t / 0.012))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) * np.exp(-t / decay) * np.clip(t / 0.002, 0, 1)).astype(np.float32)

# ---------------- build the music ----------------
pad_gain = [0.10, 0.12, 0.13, 0.12, 0.09]
air      = [0.6, 1.0, 1.6, 1.2, 0.5]

MELODY = {   # beat offsets within a chord -> midi (D major pentatonic, C# over the G)
    0: [(0, 78), (2, 76), (3, 74), (6, 81), (9, 78)],
    1: [(0, 74), (1.5, 76), (3, 78), (6, 71), (8, 74)],
    2: [(0, 78), (2, 79), (3, 81), (6, 85), (8, 83), (10, 81)],
    3: [(0, 81), (1.5, 83), (3, 76), (6, 74), (9, 76), (10.5, 78)],
}
PENTA = [62, 64, 66, 69, 71, 74, 76, 78, 81, 83, 86]

for c, prog in enumerate(SECTIONS):
    for k, chord in enumerate(prog):
        t0 = INTRO + c * SECTION_LEN + k * CHORD
        for j, m in enumerate(chord):
            add(padL, padR, t0, glass(m, CHORD, air[c]), pan=(j - 2) * 0.3, gain=pad_gain[c])
        if c in (1, 2, 3):
            add(L, R, t0, hum(chord[0], CHORD), gain=0.035)

        # kalimba melody: sections 2 and 4, an octave higher and sparser in 4
        if c in (1, 3):
            for beat, m in MELODY[k]:
                if c == 3 and rng.random() < 0.3: continue
                add(tineL, tineR, t0 + beat * BEAT + rng.uniform(-0.015, 0.015),
                    tine(m + (12 if c == 3 else 0), decay=1.6), pan=rng.uniform(-0.25, 0.25), gain=0.11)

        # the fullest section: rings of triplets spreading outward, left to right, like a ripple
        if c == 2:
            tones = sorted(x + 12 for x in chord[1:])
            steps = int(12 * 3)
            for b in range(steps):
                up = (b // len(tones)) % 2 == 0
                m = tones[b % len(tones)] if up else tones[-1 - b % len(tones)]
                pan = np.sin(2 * np.pi * (b / steps) + k) * 0.7
                add(tineL, tineR, t0 + b * BEAT / 3, tine(m + 12, decay=0.6), pan=pan,
                    gain=0.028 * (0.7 + 0.3 * np.sin(b * 0.4)))
            for beat, m in MELODY[k]:
                add(tineL, tineR, t0 + beat * BEAT, tine(m, decay=2.0), pan=rng.uniform(-0.2, 0.2), gain=0.1)

        # drops falling into the pond, in key
        density = [0.3, 0.12, 0.08, 0.15, 0.35][c]
        for b in range(12):
            if rng.random() < density:
                add(tineL, tineR, t0 + b * BEAT + rng.uniform(0, BEAT), plip(int(rng.choice(PENTA)) + 12),
                    pan=rng.uniform(-0.85, 0.85), gain=0.05)

# a few drops in the intro, before anything else
for k in range(7):
    add(tineL, tineR, rng.uniform(3, INTRO - 2), plip(int(rng.choice(PENTA)) + 12), pan=rng.uniform(-0.8, 0.8), gain=0.045)

# outro: Dmaj9 opened up and left hanging over the water
t0 = MUSIC_END
for j, m in enumerate([38, 50, 57, 64, 66, 73]):
    add(padL, padR, t0, glass(m, 13, 0.5), pan=(j - 2.5) * 0.28, gain=0.085)
for i, m in enumerate([81, 78, 76, 74, 69, 66]):
    add(tineL, tineR, t0 + 2 + i * 2.4, tine(m, decay=2.6), pan=-0.3 + i * 0.12, gain=0.085)

# ---------------- night at the water's edge ----------------
# lapping: dark noise in slow, uneven waves, each side on its own tide
for buf, seed_ph in ((natL, 0.0), (natR, 1.9)):
    lp = signal.sosfilt(signal.butter(2, [60, 380], btype='band', fs=SR, output='sos'), noise(DUR))
    t = np.arange(N) / SR
    tide = (0.55 + 0.25 * np.sin(2 * np.pi * 0.13 * t + seed_ph) + 0.2 * np.sin(2 * np.pi * 0.21 * t + seed_ph * 2)) ** 2
    buf += (lp * tide * 0.09).astype(np.float32)

def frog(start, f, pulses, pan, gain):
    bp = signal.butter(2, [f * 3, f * 5.5], btype='band', fs=SR, output='sos')
    for k in range(pulses):
        t = t_axis(0.12)
        ff = f * 1.06 * (0.87 ** (t / 0.09))
        saw = 2 * ((np.cumsum(ff) / SR) % 1) - 1
        e = np.clip(t / 0.015, 0, 1) * np.exp(-t / 0.035)
        add(natL, natR, start + k * rng.uniform(0.13, 0.18), signal.sosfilt(bp, saw * e).astype(np.float32), pan=pan, gain=gain)

frogs = [(rng.uniform(95, 120), -0.7), (rng.uniform(130, 150), 0.5), (rng.uniform(155, 175), 0.1)]
for f, pan in frogs:
    t = rng.uniform(2, 10)
    while t < DUR - 6:
        frog(t, f, int(rng.integers(2, 6)), pan, rng.uniform(0.14, 0.22))
        t += rng.uniform(6, 18)

# spring peepers: tiny rising whistles, far off across the water
def peeper_run(start, end, f, pan):
    t = start
    while t < end:
        if rng.random() < 0.15:
            t += rng.uniform(3, 9); continue
        tt = t_axis(0.11)
        ff = f * (1 + 0.14 * tt / 0.11)
        s = np.sin(2 * np.pi * np.cumsum(ff) / SR) * np.sin(np.pi * tt / 0.11) ** 2
        add(natL, natR, t, s.astype(np.float32), pan=pan, gain=0.012)
        t += rng.uniform(0.85, 1.25)

for pan in (-0.85, 0.75, -0.2):
    peeper_run(rng.uniform(0.5, 8), DUR - rng.uniform(2, 8), rng.uniform(2500, 3000), pan)

# ---------------- effects ----------------
# chorus on the pads: two slowly wandering delay lines, the shimmer of light on water
def chorus(x, rate, depth_ms=6.0, base_ms=18.0, ph=0.0):
    n = np.arange(len(x))
    d = (base_ms + depth_ms * np.sin(2 * np.pi * rate * n / SR + ph)) * SR / 1000
    return np.interp(n - d, n, x, left=0).astype(np.float32)

padL, padR = (padL * 0.7 + chorus(padL, 0.17) * 0.5 + chorus(padR, 0.23, ph=2.0) * 0.3,
              padR * 0.7 + chorus(padR, 0.19, ph=1.0) * 0.5 + chorus(padL, 0.29, ph=3.0) * 0.3)

# ping-pong echo on the tines, a dotted quarter apart
def echo(xL, xR, d=BEAT * 1.5, fb=0.38, taps=5):
    yL, yR = xL.copy(), xR.copy()
    D = int(d * SR)
    lp = signal.butter(1, 3000, fs=SR, output='sos')
    srcL, srcR = xL, xR
    for k in range(1, taps + 1):
        srcL, srcR = signal.sosfilt(lp, srcR).astype(np.float32), signal.sosfilt(lp, srcL).astype(np.float32)  # swap = ping-pong
        g = fb ** k
        yL[D * k:] += srcL[:-D * k] * g
        yR[D * k:] += srcR[:-D * k] * g
    return yL, yR

tineL, tineR = echo(tineL, tineR)

# convolution reverb: long and dark, a wide open night over still water
def make_ir(sec=6.0, rt60=5.5):
    t = t_axis(sec)
    dec = np.exp(-6.9 * t / rt60)
    lp = signal.butter(2, 4200, fs=SR, output='sos')
    irs = []
    for _ in range(2):
        n = signal.sosfilt(lp, rng.standard_normal(len(t))) * dec
        n[: int(0.035 * SR)] = 0                                  # pre-delay
        irs.append((n / np.sqrt(np.sum(n ** 2))).astype(np.float32))
    return irs

irL, irR = make_ir()
sendL = padL * 0.7 + L * 0.4 + tineL * 0.9 + natL * 0.35
sendR = padR * 0.7 + R * 0.4 + tineR * 0.9 + natR * 0.35
revL = signal.oaconvolve(sendL, irL)[:N].astype(np.float32)
revR = signal.oaconvolve(sendR, irR)[:N].astype(np.float32)

mixL = padL + L + tineL + natL * 1.5 + revL * 0.6
mixR = padR + R + tineR + natR * 1.5 + revR * 0.6

hp = signal.butter(2, 35, btype='high', fs=SR, output='sos')
mixL = signal.sosfilt(hp, mixL).astype(np.float32); mixR = signal.sosfilt(hp, mixR).astype(np.float32)

# master fades
t = np.arange(N) / SR
fade = np.clip(t / 4, 0, 1) * np.clip((DUR - t) / 9, 0, 1)
mixL *= fade; mixR *= fade

# gentle glue + normalize
peak = max(np.abs(mixL).max(), np.abs(mixR).max())
mixL, mixR = np.tanh(mixL / peak * 1.1) / np.tanh(1.1), np.tanh(mixR / peak * 1.1) / np.tanh(1.1)
out = np.stack([mixL, mixR], axis=1) * 0.89
pcm = (out * 32767).astype('<i2')

with wave.open('moon_pond.wav', 'wb') as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes(pcm.tobytes())
try:
    import lameenc
    enc = lameenc.Encoder()
    enc.set_bit_rate(192); enc.set_in_sample_rate(SR); enc.set_channels(2); enc.set_quality(2)
    with open('moon_pond.mp3', 'wb') as f:
        f.write(enc.encode(pcm.tobytes()) + enc.flush())
except ImportError:
    pass
print(f"rendered {DUR:.1f}s")
