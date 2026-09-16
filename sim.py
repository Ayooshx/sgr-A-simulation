"""
╔══════════════════════════════════════════════════════════════════════════╗
║       SAGITTARIUS A* — Real-Time Ray-Marching Simulation                ║
║                                                                        ║
║  Physics : Sgr A* mass scale, reduced Kerr lensing, Doppler beaming,   ║
║            frame-dragging (Lense-Thirring), Kerr ISCO / horizon         ║
║  Render  : Adaptive-step ray-march, procedural accretion filaments,    ║
║            volumetric disk glow, image-based Milky Way skybox,        ║
║            ACES tone-mapped HDR                                        ║
║  Target  : Sagittarius A* — 4.154×10⁶ M☉ compact object                          ║
║                                                                        ║
║  Controls:                                                             ║
║    Left-drag     →  orbit camera around the black hole                 ║
║    Right-drag    →  zoom in / out                                      ║
║    W / S         →  zoom in / out (keyboard)                           ║
║    A / D         →  orbit left / right (keyboard)                      ║
║    GUI sliders   →  spin, exposure, quality, camera, Milky Way, etc.   ║
║                                                                        ║
║  Launch:                                                               ║
║    python sim.py              (Vulkan — best for Intel iGPU)           ║
║    python sim.py --opengl     (OpenGL fallback)                        ║
║    python sim.py --cpu        (CPU — slowest, universal)               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import taichi as ti
import numpy as np
import math
import os
import time as _clock
import sys

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §1  TAICHI BACKEND SELECTION                                          ║
# ║                                                                        ║
# ║  Vulkan is optimal for Intel Iris Xe / UHD (12th-gen+).               ║
# ║  Pass --opengl or --cpu on the command line to switch backends.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_arch = ti.vulkan                          # Default: best for Intel iGPU
if "--opengl" in sys.argv:
    _arch = ti.opengl
elif "--cpu" in sys.argv:
    _arch = ti.cpu

ti.init(arch=_arch, default_fp=ti.f32, default_ip=ti.i32)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §1b  MILKY WAY BACKGROUND IMAGE                                       ║
# ║                                                                        ║
# ║  Instead of a procedurally-generated band, the sky background is now  ║
# ║  sourced from a real equirectangular Milky Way photo/render on disk.   ║
# ║  Place the image next to this script (or set MILKYWAY_IMAGE_PATH) and  ║
# ║  it will be loaded once at startup and used to texture the skybox.     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MILKYWAY_IMAGE_PATH = os.environ.get(
    "MILKYWAY_IMAGE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "milkyway.jpg")
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §2  CONSTANTS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

WIN_W, WIN_H = 1920, 1080                  # Display / Window resolution
COMP_W, COMP_H = 960, 540                  # Internal Compute resolution (for 60 FPS)
ASPECT  = WIN_W / WIN_H                    # 16∶9 aspect ratio

# Bloom buffers — a small, heavily-downsampled copy of the display frame
# used for a cheap bright-pass + separable-blur glow. Downsampling this
# much keeps the extra passes essentially free relative to the ray-march.
BLOOM_W, BLOOM_H = WIN_W // 4, WIN_H // 4

# Physics — geometric units where G = c = 1.
#
# Sagittarius A* is represented in normalized gravitational units:
#   M = 1 -> one gravitational radius GM/c² in the renderer.
# The physical Sgr A* scale is retained for telemetry and unit conversion.
M = 1.0

# Physical parameters for Sagittarius A*
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SUN_KG = 1.98847e30
M_SUN_KM = 1.4766250385                  # GM_sun / c²
SgrA_MASS_SOLAR = 4.154e6                # adopted mass for this simulation

SgrA_RG_KM = M_SUN_KM * SgrA_MASS_SOLAR
SgrA_RS_KM = 2.0 * SgrA_RG_KM
SgrA_RG_LIGHT_SECONDS = SgrA_RG_KM / 299.792458
SgrA_MASS_KG = SgrA_MASS_SOLAR * M_SUN_KG
SgrA_TIME_UNIT_S = SgrA_MASS_KG * G_SI / (C_SI ** 3)

# Normalized Kerr horizon uses r+ = M + sqrt(M²-a²).
RS = 2.0 * M

# Compact hot accretion-flow region for a Sgr A*-scale viewer.
R_OUT = 24.0
R_MAX = 96.0
MAX_MARCH = 500

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §2b  S301 ORBIT — ASTROPHYSICAL PARAMETERS                           ║
# ║                                                                        ║
# ║  S301 is integrated in its own physical unit system (AU, years,        ║
# ║  solar masses) using RK4, exactly like an Earth-around-the-Sun         ║
# ║  simulation — then its position is converted into the renderer's       ║
# ║  code units (1 unit = 1 gravitational radius r_g of Sgr A*) purely     ║
# ║  as a length conversion, so it can be drawn in the same 3-D scene       ║
# ║  as the ray-marched horizon and accretion disk.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

AU_KM   = 1.495978707e8          # 1 AU in km
YEAR_S  = 365.25 * 86400.0       # Julian year, in seconds
C_KM_S  = 299792.458             # speed of light, km/s
C_AU_YR = C_KM_S * YEAR_S / AU_KM        # speed of light in AU/yr (~63,241)

# S301's fitted central-mass solution (independent of the renderer's own
# adopted Sgr A* mass above — different stellar orbits around Sgr A* are
# fit with slightly different central-mass values in the literature).
# Kepler's 3rd law in (AU, yr, M_sun) units — GM = 4*pi^2*M — is satisfied
# by these numbers: P = sqrt(a^3 / M) = sqrt(687^3 / 4.3e6) = 8.686 yr.
S301_M_BH_MSUN   = 4.3e6
S301_GM_BH       = 4.0 * math.pi ** 2 * S301_M_BH_MSUN     # AU^3 / yr^2

S301_M_STAR_MSUN = 1.5                       # negligible -> restricted 2-body
S301_A_AU        = 687.0                     # semi-major axis
S301_ECC         = 0.983                     # eccentricity
S301_PERIOD_YR   = 8.68
S301_R_PERI_AU   = S301_A_AU * (1.0 - S301_ECC)     # ≈ 11.68 AU
S301_R_APO_AU    = S301_A_AU * (1.0 + S301_ECC)     # ≈ 1362.3 AU

# NOTE: periapsis speed is intentionally NOT hardcoded here. For a Keplerian
# orbit, r_peri, a, and ecc already fully determine v_peri via vis-viva:
#   v_peri^2 = GM/a * (1+e)/(1-e)
# A previous version of this file hardcoded v_peri from an independent
# "25,000 km/s" estimate. That value was ~1.8% off from the vis-viva speed
# for (a=687 AU, e=0.983, M=4.3e6 Msun) — and because a = 1/(2/r - v^2/GM)
# is a small difference of two nearly-equal large terms, that tiny velocity
# error blew up into a wildly different *actual* integrated orbit (a ~ 135 AU,
# e ~ 0.91, P ~ 0.76 yr instead of the labeled 687 AU / 0.983 / 8.68 yr).
# TestParticleOrbit.__init__ already derives v_peri from (a, ecc, gm_bh) via
# vis-viva whenever v_peri_auyr is not explicitly supplied, so S301 is now
# constructed without an override to guarantee the displayed elements match
# the physics actually being integrated.

# Length conversion: AU -> renderer code units (r_g of the rendered Sgr A*)
AU_TO_CODE = AU_KM / SgrA_RG_KM

# Trail buffer size (number of sampled orbit points kept & drawn on screen)
S301_TRAIL_MAX = 700

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §2c  MILKY WAY SKYBOX — CONSTANTS                                     ║
# ║                                                                        ║
# ║  Sgr A* sits inside the Galactic disk (at the very center of the       ║
# ║  Milky Way), so an observer near it looking outward at "infinity"      ║
# ║  sees the Galaxy's disk stars/dust as a broad glowing band circling    ║
# ║  the whole sky, not a single galaxy-shaped blob off in the distance.   ║
# ║  The skybox texture is now loaded from a real equirectangular image    ║
# ║  (see §1b) instead of being procedurally generated, and is optionally  ║
# ║  rotated/tilted relative to the render's Y (spin) axis since the       ║
# ║  galactic plane and the hole's spin axis are not the same axis.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SKY_W, SKY_H = 1024, 512
PI = math.pi

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §2d  REAL STARS — YALE BRIGHT STAR CATALOGUE (BSC5) SUBSET             ║
# ║                                                                        ║
# ║  A curated subset of the ~170 brightest naked-eye stars from the Yale  ║
# ║  Bright Star Catalogue (Harvard Revised numbers), with true J2000      ║
# ║  right ascension, declination, and visual magnitude. This gives the    ║
# ║  sky real, recognisable stars/constellations rather than only          ║
# ║  procedural noise stars. RA is given in decimal hours, Dec in decimal  ║
# ║  degrees, V is apparent visual magnitude (lower = brighter).           ║
# ║  Source: Yale BSC5 (Hoffleit & Warren, 1991), http://tdc-www.harvard.  ║
# ║  edu/catalogs/bsc5.html                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# (HR, Name, RA_hours, Dec_deg, Vmag, B-V) — each star listed once.
REAL_STARS = [
    (2491, "Sirius",          6.7525,  -16.7161, -1.46,  0.00),
    (2326, "Canopus",         6.3992,  -52.6957, -0.74,  0.15),
    (5459, "Alpha Centauri",  14.6614, -60.8340, -0.27,  0.71),
    (5340, "Arcturus",        14.2610,  19.1825, -0.05,  1.23),
    (7001, "Vega",            18.6156,  38.7837,  0.03,  0.00),
    (1708, "Capella",         5.2782,   45.9980,  0.08,  0.80),
    (1713, "Rigel",           5.2423,  -8.2017,   0.13, -0.03),
    (2943, "Procyon",         7.6550,   5.2250,   0.34,  0.42),
    (472,  "Achernar",        1.6286,  -57.2367,  0.46, -0.16),
    (2061, "Betelgeuse",      5.9195,   7.4071,   0.50,  1.85),
    (5267, "Hadar",           14.0637, -60.3730,  0.61, -0.23),
    (7557, "Altair",          19.8464,  8.8683,   0.77,  0.22),
    (4730, "Acrux",           12.4433, -63.0990,  0.77, -0.20),
    (1457, "Aldebaran",       4.5987,  16.5093,   0.85,  1.54),
    (6134, "Antares",         16.4901, -26.4320,  0.96,  1.83),
    (5056, "Spica",           13.4199, -11.1613,  0.98, -0.24),
    (2990, "Pollux",          7.7553,  28.0262,   1.14,  1.00),
    (8728, "Fomalhaut",       22.9608, -29.6222,  1.16,  0.09),
    (7924, "Deneb",           20.6905,  45.2803,  1.25,  0.09),
    (4853, "Mimosa",          12.7953, -59.6888,  1.25, -0.24),
    (3982, "Regulus",         10.1395,  11.9672,  1.35, -0.11),
    (2618, "Adhara",          6.9770,  -28.9721,  1.50, -0.21),
    (2891, "Castor",          7.5766,  31.8883,   1.58,  0.03),
    (4763, "Gacrux",          12.5194, -57.1133,  1.63,  1.60),
    (6527, "Shaula",          17.5601, -37.1038,  1.62, -0.22),
    (1790, "Bellatrix",       5.4188,   6.3497,   1.64, -0.22),
    (1791, "Elnath",          5.4382,  28.6075,   1.65, -0.13),
    (3685, "Miaplacidus",     9.2199,  -69.7172,  1.69,  0.00),
    (1903, "Alnilam",         5.6036,  -1.2019,   1.69, -0.18),
    (8425, "Alnair",          22.1372, -46.9611,  1.73, -0.07),
    (4905, "Alioth",          12.9005,  55.9598,  1.76, -0.02),
    (1948, "Alnitak",         5.6793,  -1.9426,   1.79, -0.20),
    (3207, "Regor",           8.1596,  -47.3365,  1.75, -0.14),
    (4301, "Dubhe",           11.0621,  61.7510,  1.79,  1.07),
    (1017, "Mirfak",          3.4054,  49.8613,   1.79,  0.48),
    (2693, "Wezen",           7.1398,  -26.3932,  1.83,  0.71),
    (6879, "Kaus Australis",  18.4028, -34.3846,  1.85, -0.03),
    (3307, "Avior",           8.3752,  -59.5095,  1.86,  1.28),
    (5191, "Alkaid",          13.7923,  49.3133,  1.86, -0.19),
    (6553, "Sargas",          17.6222, -42.9978,  1.87,  0.40),
    (2088, "Menkalinan",      5.9922,  44.9474,   1.90,  0.03),
    (6217, "Atria",           16.8111, -69.0277,  1.91,  1.44),
    (2985, "Alhena",          6.6285,  16.3993,   1.93,  0.00),
    (8322, "Peacock",         20.4275, -56.7350,  1.94, -0.17),
    (2422, "Mirzam",          6.3783,  -17.9558,  1.98, -0.24),
    (424,  "Polaris",         2.5303,  89.2641,   1.98,  0.60),
    (3748, "Alphard",         9.4599,  -8.6586,   1.98,  1.44),
    (617,  "Hamal",           2.1194,  23.4624,   2.00,  1.15),
    (188,  "Diphda",          0.7265,  -17.9866,  2.04,  1.02),
    (7121, "Nunki",           18.9210, -26.2967,  2.05, -0.18),
    (337,  "Mirach",          1.1620,  35.6206,   2.06,  1.58),
    (15,   "Alpheratz",       0.1397,  29.0906,   2.06, -0.11),
    (5793, "Rasalhague",      17.5822, 12.5601,   2.08,  0.15),
    (6220, "Kochab",          14.8451, 74.1555,   2.08,  1.47),
    (5054, "Mizar",           13.3988, 54.9254,   2.23,  0.02),
    (7106, "Eltanin",         17.9434,  51.4889,  2.24,  1.53),
    (168,  "Schedar",         0.6752,  56.5372,   2.23,  1.17),
    (21,   "Caph",            0.1530,  59.1497,   2.27,  0.38),
    (8308, "Enif",            21.7364,  9.8750,   2.39,  1.52),
    (4295, "Merak",           11.0307,  56.3824,  2.37,  0.03),
    (4554, "Phecda",          11.8971,  53.6948,  2.44,  0.00),
    (39,   "Scheat",          23.0629,  28.0828,  2.42,  1.67),
    (39,   "Markab",          23.0794,  15.2053,  2.49, -0.04),
    (5288, "Menkent",         14.1114, -36.3700,  2.06,  1.02),
    (5947, "Sabik",           17.1730, -15.7249,  2.43, -0.19),
    (2827, "Naos",            8.0592,  -40.0031,  2.21, -0.27),
    (5107, "Zubenelgenubi",   14.8479, -16.0418,  2.75,  0.15),
    (403,  "Ruchbah",         1.4303,  60.2353,   2.68,  0.16),
    (7573, "Tarazed",         19.7709,  10.6133,  2.72,  1.52),
    (6580, "Kaus Media",      18.3502, -29.8281,  2.72,  1.03),
    (8781, "Deneb Algedi",    21.7844, -16.1272,  2.87,  0.29),
    (6410, "Vindemiatrix",    13.0361,  10.9592,  2.85,  0.92),
    (39,   "Algenib",         0.2207,  15.1836,   2.83, -0.19),
    (7602, "Sheliak",         18.8347,  33.3627,  3.52, -0.06),
    (7377, "Albireo",         19.5121,  27.9597,  3.18,  1.11),
    (4660, "Megrez",          12.2570,  57.0326,  3.31,  0.08),
    (4534, "Chertan",         11.2373,  15.4295,  3.34,  0.13),
]

def _compute_star_dirs():
    dirs = np.zeros((len(REAL_STARS), 3), dtype=np.float64)
    for k, (hr, name, ra_h, dec_deg, vmag, bv) in enumerate(REAL_STARS):
        ra_rad = ra_h * (math.pi / 12.0)
        dec_rad = math.radians(dec_deg)
        dirs[k, 0] = math.cos(dec_rad) * math.cos(ra_rad)
        dirs[k, 1] = math.sin(dec_rad)
        dirs[k, 2] = math.cos(dec_rad) * math.sin(ra_rad)
    return dirs


STAR_DIRS = _compute_star_dirs()   # (N, 3) unit vectors, row-matched to REAL_STARS


NEUTRAL_STAR_COLOR = (1.0, 0.97, 0.92)


def bv_to_temperature(bv):
    if bv is None:
        return None
    try:
        bv = float(bv)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bv):
        return None
    bv = max(-0.4, min(bv, 2.0))
    denom_a = 0.92 * bv + 1.7
    denom_b = 0.92 * bv + 0.62
    if abs(denom_a) < 1e-6 or abs(denom_b) < 1e-6:
        return None
    temp = 4600.0 * (1.0 / denom_a + 1.0 / denom_b)
    if not math.isfinite(temp) or temp <= 0.0:
        return None
    return temp


def temperature_to_rgb(temp_k):
    t = max(1000.0, min(temp_k, 40000.0)) / 100.0

    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)

    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)

    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307

    r = max(0.0, min(r, 255.0)) / 255.0
    g = max(0.0, min(g, 255.0)) / 255.0
    b = max(0.0, min(b, 255.0)) / 255.0
    return (r, g, b)


def bv_to_star_color(bv):
    temp = bv_to_temperature(bv)
    if temp is None:
        return NEUTRAL_STAR_COLOR
    return temperature_to_rgb(temp)


pixels_compute = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))
pixels_display = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))

bloom_a = ti.Vector.field(3, dtype=ti.f32, shape=(BLOOM_W, BLOOM_H))
bloom_b = ti.Vector.field(3, dtype=ti.f32, shape=(BLOOM_W, BLOOM_H))

cam_origin  = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_right   = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_up      = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_forward = ti.Vector.field(3, dtype=ti.f32, shape=())

MAX_TEST_PARTICLES = 7   # S301 (slot 0) + up to 6 user-spawned test particles

PARTICLE_LENS_RADIUS     = 0.55   # code units (comparable to a few r_hz)
PARTICLE_LENS_BRIGHTNESS = 3.2

particle_pos          = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_TEST_PARTICLES,))
particle_trail_data   = ti.Vector.field(4, dtype=ti.f32, shape=(MAX_TEST_PARTICLES, S301_TRAIL_MAX))
particle_trail_count  = ti.field(dtype=ti.i32, shape=(MAX_TEST_PARTICLES,))
particle_color        = ti.Vector.field(3, dtype=ti.f32, shape=(MAX_TEST_PARTICLES,))
particle_active       = ti.field(dtype=ti.i32, shape=(MAX_TEST_PARTICLES,))

_particle_pos_np      = np.zeros((MAX_TEST_PARTICLES, 3), dtype=np.float32)
_particle_trail_np    = np.zeros((MAX_TEST_PARTICLES, S301_TRAIL_MAX, 4), dtype=np.float32)
_particle_trailcnt_np = np.zeros((MAX_TEST_PARTICLES,), dtype=np.int32)
_particle_color_np    = np.zeros((MAX_TEST_PARTICLES, 3), dtype=np.float32)
_particle_active_np   = np.zeros((MAX_TEST_PARTICLES,), dtype=np.int32)


def sync_particle_fields():
    particle_pos.from_numpy(_particle_pos_np)
    particle_color.from_numpy(_particle_color_np)
    particle_active.from_numpy(_particle_active_np)
    particle_trail_count.from_numpy(_particle_trailcnt_np)
    particle_trail_data.from_numpy(_particle_trail_np)


skybox_source = ti.Vector.field(3, dtype=ti.f32, shape=(SKY_W, SKY_H))
skybox        = ti.Vector.field(3, dtype=ti.f32, shape=(SKY_W, SKY_H))

real_stars_tex = ti.Vector.field(3, dtype=ti.f32, shape=(SKY_W, SKY_H))

real_stars_twinkle = ti.Vector.field(2, dtype=ti.f32, shape=(SKY_W, SKY_H))

@ti.func
def fract(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)


@ti.func
def smoothstep(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.min(ti.max((x - edge0) / (edge1 - edge0 + 1e-8), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def hash21(p: ti.math.vec2) -> ti.f32:
    return fract(ti.sin(p.dot(ti.Vector([127.1, 311.7]))) * 43758.5453)


@ti.func
def hash22(p: ti.math.vec2) -> ti.math.vec2:
    return ti.Vector([
        hash21(p),
        hash21(p + ti.Vector([269.5, 183.3]))
    ])


@ti.func
def noise2d(p: ti.math.vec2) -> ti.f32:
    i = ti.floor(p)
    f = fract(p)
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    a = hash21(i + ti.Vector([0.0, 0.0]))
    b = hash21(i + ti.Vector([1.0, 0.0]))
    c = hash21(i + ti.Vector([0.0, 1.0]))
    d = hash21(i + ti.Vector([1.0, 1.0]))
    return a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y

@ti.func
def fbm2d(p: ti.math.vec2) -> ti.f32:
    v = 0.0
    a = 0.5
    shift = ti.Vector([100.0, 100.0])
    for _ in ti.static(range(4)):
        v += a * noise2d(p)
        px = p.x * 0.8 - p.y * 0.6
        py = p.x * 0.6 + p.y * 0.8
        p = ti.Vector([px, py]) * 2.0 + shift
        a *= 0.5
    return v

@ti.func
def horizon_radius(a: ti.f32) -> ti.f32:
    return M + ti.sqrt(ti.max(M * M - a * a, 0.0))


@ti.func
def isco_radius(a: ti.f32) -> ti.f32:
    aa = ti.abs(a)
    z1 = 1.0 + ti.pow(ti.max(1.0 - aa * aa, 1e-12), 1.0 / 3.0) * (
        ti.pow(1.0 + aa, 1.0 / 3.0) +
        ti.pow(ti.max(1.0 - aa, 1e-12), 1.0 / 3.0)
    )
    z2 = ti.sqrt(3.0 * aa * aa + z1 * z1)
    return M * (3.0 + z2 - ti.sqrt(
        ti.max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)
    ))


@ti.func
def gravity_accel(
    pos: ti.math.vec3,
    vel: ti.math.vec3,
    a:   ti.f32
) -> ti.math.vec3:
    r      = pos.norm()
    r_safe = ti.max(r, 0.5)
    r2     = r_safe * r_safe
    r4     = r2 * r2

    h  = pos.cross(vel)
    h2 = h.dot(h)

    pos_hat = pos / r_safe
    accel   = -1.5 * RS * h2 / r4 * pos_hat

    if a > 1e-4:
        spin_axis = ti.Vector([0.0, 1.0, 0.0])
        omega_fd  = 2.0 * a * M / (r2 * r_safe)
        fd_dir = pos.cross(spin_axis)
        fd_len = fd_dir.norm()
        if fd_len > 1e-6:
            fd_dir /= fd_len
        accel += omega_fd * fd_dir

    return accel


@ti.func
def hsv_to_rgb(h: ti.f32, s: ti.f32, v: ti.f32) -> ti.math.vec3:
    r = ti.abs(h * 6.0 - 3.0) - 1.0
    g = 2.0 - ti.abs(h * 6.0 - 2.0)
    b = 2.0 - ti.abs(h * 6.0 - 4.0)
    
    rgb = ti.Vector([
        ti.min(ti.max(r, 0.0), 1.0),
        ti.min(ti.max(g, 0.0), 1.0),
        ti.min(ti.max(b, 0.0), 1.0)
    ])
    return v * ((rgb - 1.0) * s + 1.0)


@ti.func
def disk_emission(
    hit_pos:  ti.math.vec3,
    ray_vel:  ti.math.vec3,
    r:        ti.f32,
    r_inner:  ti.f32,
    a:        ti.f32,
    t:        ti.f32,
    hue:      ti.f32,
    sat:      ti.f32,
    val:      ti.f32,
    doppler:  ti.f32
) -> ti.math.vec3:
    phi = ti.atan2(hit_pos.z, hit_pos.x)

    t_r = (r - r_inner) / (R_OUT - r_inner + 1e-4)

    dist = ti.max(r - r_inner, 0.01)
    compact_core = ti.exp(-dist / 3.0)
    extended_flow = 0.35 * ti.exp(-dist / 9.0)
    radial = (0.80 * compact_core + extended_flow) / (1.0 + 0.16 * dist)

    radial += 0.45 * ti.exp(-dist * 1.6)

    radial *= smoothstep(r_inner - 0.12, r_inner + 0.25, r)

    radial *= (1.0 - smoothstep(R_OUT - 7.0, R_OUT, r))

    PI2 = 6.283185307
    p = phi if phi >= 0.0 else phi + PI2
    
    twist_factor = 7.5
    phi_sheared = p - (twist_factor / ti.pow(ti.max(r, r_inner), 1.5)) - (t * 0.85)
    
    # Height coupling: fold the sample's vertical offset (hit_pos.y) into
    # the noise domain so the turbulent structure genuinely changes as a
    # ray samples through the thickness of the flow, rather than just
    # repeating the same (r, phi) pattern at every height — this is what
    # gives the volumetric render its puffed-up, MHD-turbulence look
    # instead of a flat plane's texture painted straight up and down.
    u = ti.log(r) * 2.5 + hit_pos.y * 2.2
    v = phi_sheared * 8.0
    
    noise = fbm2d(ti.Vector([u, v]))
    
    if p > 4.5:
        p_wrap = p - PI2
        phi_sheared_wrap = p_wrap - (twist_factor / ti.pow(ti.max(r, r_inner), 1.5)) - (t * 0.85)
        v_wrap = phi_sheared_wrap * 8.0
        noise_wrap = fbm2d(ti.Vector([u, v_wrap]))
        
        weight = smoothstep(4.5, PI2, p)
        noise = noise * (1.0 - weight) + noise_wrap * weight
    
    base_glow = 0.12
    strands = smoothstep(0.2, 0.8, noise)
    filament = base_glow + (1.0 - base_glow) * ti.pow(strands, 2.2)
    r_vel = ti.max(r, r_inner)
    omega_kerr = 1.0 / (ti.pow(r_vel, 1.5) + ti.max(a, 0.0))
    v_orb = ti.min(r_vel * omega_kerr, 0.97)

    v_dir  = ti.Vector([-ti.sin(phi), 0.0, ti.cos(phi)])
    v_disk = v_dir * v_orb

    # ── Directional Doppler only ─────────────────────────────────────────
    # This term carries *only* the approach/recession asymmetry, i.e. the
    # part that flips sign as gas swings from the approaching limb to the
    # receding one. The time-dilation part (the Lorentz gamma that used to
    # live here) has been moved into the orbital-geodesic factor below, so
    # that the isotropic energy loss is not counted twice and so it stays
    # in force even when the Doppler slider is turned down to zero.
    ray_hat = ray_vel / ti.max(ray_vel.norm(), 1e-8)
    v_proj  = v_disk.dot(ray_hat)
    g_dopp  = 1.0 / ti.max(1.0 - v_proj, 0.05)
    g_dopp  = ti.min(ti.max(g_dopp, 0.06), 9.0)

    beaming = g_dopp * g_dopp * g_dopp
    beaming_factor = 1.0 * (1.0 - doppler) + beaming * doppler

    # ── Gravitational redshift (independent of kinematic Doppler) ────────
    # Photons climbing out of the potential well lose energy purely as a
    # function of areal radius r, regardless of the local gas velocity's
    # direction relative to the camera. Using the Schwarzschild form
    # sqrt(1 - r_s/r) as a stand-in redshift factor for the Kerr metric
    # (exact for a=0, a reasonable proxy near the equatorial plane for
    # small-to-moderate spin): gas skimming the inner edge is dimmed and
    # reddened even when it happens to be moving exactly transverse to
    # the line of sight, which the beaming term alone would leave bright.
    # Kerr equatorial circular geodesic: the proper-time rate of an orbiting
    # emitter relative to a distant observer is
    #     g_orb = sqrt(1 - 3M/r + 2a*sqrt(M)/r^(3/2))
    # which in the a=0 limit reduces to the familiar sqrt(1 - 3M/r) of a
    # Schwarzschild circular orbit, and which vanishes at the photon orbit.
    # It folds together the gravitational redshift of climbing out of the
    # well and the transverse (second-order) time dilation of the orbital
    # motion, both of which are isotropic — they do not care which way the
    # gas happens to be moving across the line of sight. Prograde spin
    # (a > 0) makes the term *less* severe at fixed r, which is why a
    # rapidly spinning hole can keep a bright, stable inner edge much
    # closer in than a static one.
    r_g   = ti.max(r, 1.05 * horizon_radius(a))
    g_orb = ti.sqrt(ti.max(
        1.0 - 3.0 * M / r_g + 2.0 * a * ti.sqrt(M) / ti.pow(r_g, 1.5),
        0.0
    ))
    redshift_1z = ti.min(ti.max(g_orb, 0.02), 1.0)
    grav_dim    = ti.pow(redshift_1z, 4.0)   # flux ~ g^4

    intensity = radial * filament * beaming_factor * grav_dim

    base_color = hsv_to_rgb(hue, sat, val)

    # Redden the intrinsic color as the local redshift deepens, on top of
    # the flux dimming above — mimics the spectrum shifting out of the
    # visible band toward the red/IR as photons climb out of the well.
    redshift_color_mix = 1.0 - smoothstep(0.0, 0.35, redshift_1z)
    reddened = ti.Vector([base_color.x, base_color.y * 0.5, base_color.z * 0.25])
    base_color = base_color * (1.0 - redshift_color_mix) + reddened * redshift_color_mix

    core_mix = ti.pow(smoothstep(r_inner + 1.2, r_inner, r), 3.0)
    white_hot = ti.Vector([val, val, val]) * 1.2
    
    outer_mix = smoothstep(R_OUT - 8.0, R_OUT, r)
    dim_grey = ti.Vector([val, val, val]) * 0.5
    
    final_color = base_color
    final_color = final_color * (1.0 - core_mix) + white_hot * core_mix
    final_color = final_color * (1.0 - outer_mix) + dim_grey * outer_mix

    return intensity * final_color


@ti.func
def background_stars(ray_dir: ti.math.vec3) -> ti.math.vec3:
    color = ti.Vector([0.0, 0.0, 0.0])

    theta = ti.acos(ti.min(ti.max(ray_dir.y, -0.9999), 0.9999))
    phi   = ti.atan2(ray_dir.z, ray_dir.x)

    cs = 0.04
    ci = ti.floor(theta / cs)
    cj = ti.floor(phi / cs)

    for di in ti.static(range(-1, 2)):
        for dj in ti.static(range(-1, 2)):
            cell = ti.Vector([ci + di, cj + dj])
            h1   = hash21(cell * 1.0)

            if h1 < 0.75:
                sub = hash22(cell + ti.Vector([42.0, 17.0]))
                s_theta = (cell.x + sub.x) * cs
                s_phi   = (cell.y + sub.y) * cs

                st = ti.sin(s_theta)
                star_d = ti.Vector([
                    st * ti.cos(s_phi),
                    ti.cos(s_theta),
                    st * ti.sin(s_phi)
                ])

                cos_a = ti.min(ti.max(ray_dir.dot(star_d), -1.0), 1.0)
                ang   = ti.acos(cos_a)
                sigma = 0.0012 + h1 * 0.0008

                if ang < sigma * 5.0:
                    bright = ti.exp(-0.5 * (ang / sigma) * (ang / sigma))

                    mag = ti.pow(sub.x, 2.0) * 0.7

                    temp   = 0.75 + sub.y * 0.25
                    star_c = (ti.Vector([temp * 0.9, temp * 0.95, 1.0])
                              * mag * bright * 0.35)
                    color += star_c

    return color


def load_milkyway_skybox_from_image(path: str):
    data = None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if img.size != (SKY_W, SKY_H):
            img = img.resize((SKY_W, SKY_H), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), 2.2)
        data = np.transpose(arr, (1, 0, 2)).astype(np.float32)
    except Exception as e:
        print(f"[Milky Way] Could not load '{path}': {e}")
        print("[Milky Way] Falling back to a blank skybox.")
        data = np.zeros((SKY_W, SKY_H, 3), dtype=np.float32)

    skybox_source.from_numpy(data)


def bake_real_stars():
    img = np.zeros((SKY_H, SKY_W, 3), dtype=np.float32)
    twinkle = np.zeros((SKY_H, SKY_W, 2), dtype=np.float32)
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:SKY_H, 0:SKY_W]

    for hr, name, ra_h, dec_deg, vmag, bv in REAL_STARS:
        ra_rad = ra_h * (PI / 12.0)
        dec_rad = math.radians(dec_deg)

        dx = math.cos(dec_rad) * math.cos(ra_rad)
        dy = math.sin(dec_rad)
        dz = math.cos(dec_rad) * math.sin(ra_rad)

        theta = math.acos(max(-1.0, min(1.0, dy)))
        phi = math.atan2(dz, dx)
        u = (phi + PI) / (2.0 * PI)
        v = theta / PI

        px = u * SKY_W
        py = v * SKY_H

        bright = 10.0 ** (-0.4 * (vmag - 1.0))
        bright = max(0.15, min(bright, 6.0))

        r_c, g_c, b_c = bv_to_star_color(bv)

        sigma = 0.9
        phase = rng.uniform(0.0, 2.0 * PI)
        speed = rng.uniform(0.6, 1.8) * (1.0 + 0.25 * min(max(vmag, 0.0), 3.0))

        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                cx = px + ox * SKY_W
                cy = py + oy * SKY_H
                d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                mask = d2 < (sigma * 6.0) ** 2
                if np.any(mask):
                    falloff = np.exp(-0.5 * d2[mask] / (sigma * sigma))
                    color = np.array([r_c, g_c, b_c], dtype=np.float32) * bright
                    img[mask] += falloff[:, None] * color

                disc_mask = d2 < (sigma * 3.0) ** 2
                if np.any(disc_mask):
                    twinkle[..., 0][disc_mask] = phase
                    twinkle[..., 1][disc_mask] = speed

    data = np.transpose(img, (1, 0, 2)).astype(np.float32)
    real_stars_tex.from_numpy(data)
    twinkle_data = np.transpose(twinkle, (1, 0, 2)).astype(np.float32)
    real_stars_twinkle.from_numpy(twinkle_data)


@ti.kernel
def bake_milkyway_skybox(tilt: ti.f32, mw_hue: ti.f32, mw_intensity: ti.f32, tint_amount: ti.f32):
    for i, j in skybox:
        u = (ti.cast(i, ti.f32) + 0.5) / SKY_W
        v = (ti.cast(j, ti.f32) + 0.5) / SKY_H

        phi   = u * 2.0 * PI - PI
        theta = v * PI

        st = ti.sin(theta)
        dirx = st * ti.cos(phi)
        diry = ti.cos(theta)
        dirz = st * ti.sin(phi)

        ct = ti.cos(-tilt)
        stt = ti.sin(-tilt)
        y_rot = diry * ct - dirz * stt
        z_rot = diry * stt + dirz * ct
        x_rot = dirx

        theta_src = ti.acos(ti.min(ti.max(y_rot, -1.0), 1.0))
        phi_src   = ti.atan2(z_rot, x_rot)

        u_src = (phi_src + PI) / (2.0 * PI)
        v_src = theta_src / PI

        fx = u_src * SKY_W - 0.5
        fy = v_src * SKY_H - 0.5

        x0 = ti.floor(fx)
        y0 = ti.floor(fy)
        tx = fx - x0
        ty = fy - y0

        i0 = ti.cast(x0, ti.i32) % SKY_W
        i1 = (i0 + 1) % SKY_W
        j0 = ti.min(ti.max(ti.cast(y0, ti.i32), 0), SKY_H - 1)
        j1 = ti.min(ti.max(j0 + 1, 0), SKY_H - 1)
        if i0 < 0:
            i0 += SKY_W
        if i1 < 0:
            i1 += SKY_W

        c00 = skybox_source[i0, j0]
        c10 = skybox_source[i1, j0]
        c01 = skybox_source[i0, j1]
        c11 = skybox_source[i1, j1]

        c0 = c00 * (1.0 - tx) + c10 * tx
        c1 = c01 * (1.0 - tx) + c11 * tx
        c = c0 * (1.0 - ty) + c1 * ty

        c = c * mw_intensity

        luma = c.dot(ti.Vector([0.299, 0.587, 0.114]))
        tint_col = hsv_to_rgb(mw_hue, 0.32, luma)
        c = c * (1.0 - tint_amount) + tint_col * tint_amount

        skybox[i, j] = c


@ti.func
def sample_skybox(dir_in: ti.math.vec3) -> ti.math.vec3:
    d = dir_in / ti.max(dir_in.norm(), 1e-8)

    theta = ti.acos(ti.min(ti.max(d.y, -0.9999), 0.9999))
    phi   = ti.atan2(d.z, d.x)

    u = (phi + PI) / (2.0 * PI)
    v = theta / PI

    fx = u * SKY_W - 0.5
    fy = v * SKY_H - 0.5

    x0 = ti.floor(fx)
    y0 = ti.floor(fy)
    tx = fx - x0
    ty = fy - y0

    i0 = ti.cast(x0, ti.i32) % SKY_W
    i1 = (i0 + 1) % SKY_W
    j0 = ti.min(ti.max(ti.cast(y0, ti.i32), 0), SKY_H - 1)
    j1 = ti.min(ti.max(j0 + 1, 0), SKY_H - 1)
    if i0 < 0:
        i0 += SKY_W
    if i1 < 0:
        i1 += SKY_W

    c00 = skybox[i0, j0]
    c10 = skybox[i1, j0]
    c01 = skybox[i0, j1]
    c11 = skybox[i1, j1]

    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    return c0 * (1.0 - ty) + c1 * ty


@ti.func
def sample_real_stars(dir_in: ti.math.vec3) -> ti.math.vec3:
    d = dir_in / ti.max(dir_in.norm(), 1e-8)
    theta = ti.acos(ti.min(ti.max(d.y, -0.9999), 0.9999))
    phi = ti.atan2(d.z, d.x)
    u = (phi + PI) / (2.0 * PI)
    v = theta / PI

    fx = u * SKY_W - 0.5
    fy = v * SKY_H - 0.5
    x0 = ti.floor(fx)
    y0 = ti.floor(fy)
    tx = fx - x0
    ty = fy - y0
    i0 = ti.cast(x0, ti.i32) % SKY_W
    i1 = (i0 + 1) % SKY_W
    j0 = ti.min(ti.max(ti.cast(y0, ti.i32), 0), SKY_H - 1)
    j1 = ti.min(ti.max(j0 + 1, 0), SKY_H - 1)
    if i0 < 0:
        i0 += SKY_W
    if i1 < 0:
        i1 += SKY_W

    c00 = real_stars_tex[i0, j0]
    c10 = real_stars_tex[i1, j0]
    c01 = real_stars_tex[i0, j1]
    c11 = real_stars_tex[i1, j1]
    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    return c0 * (1.0 - ty) + c1 * ty


@ti.func
def star_twinkle_factor(dir_in: ti.math.vec3, sim_time: ti.f32) -> ti.f32:
    d = dir_in / ti.max(dir_in.norm(), 1e-8)
    theta = ti.acos(ti.min(ti.max(d.y, -0.9999), 0.9999))
    phi = ti.atan2(d.z, d.x)
    u = (phi + PI) / (2.0 * PI)
    v = theta / PI

    i0 = ti.cast(u * SKY_W, ti.i32) % SKY_W
    j0 = ti.min(ti.max(ti.cast(v * SKY_H, ti.i32), 0), SKY_H - 1)
    if i0 < 0:
        i0 += SKY_W

    ps = real_stars_twinkle[i0, j0]
    phase = ps.x
    speed = ps.y
    osc = 0.6 * ti.sin(sim_time * speed + phase) \
        + 0.4 * ti.sin(sim_time * speed * 1.9 + phase * 2.3)
    return 0.89 + 0.11 * osc


@ti.func
def tone_map(c: ti.math.vec3, exposure: ti.f32) -> ti.math.vec3:
    x = c * exposure
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return ti.Vector([
        ti.min(ti.max(mapped.x, 0.0), 1.0),
        ti.min(ti.max(mapped.y, 0.0), 1.0),
        ti.min(ti.max(mapped.z, 0.0), 1.0)
    ])


@ti.func
def photon_ring_boost(accum_angle: ti.f32) -> ti.math.vec3:
    # A ray that winds >= ~1.5*2pi around the hole before escaping has
    # been trapped orbiting near the photon sphere's critical impact
    # parameter — the boundary of the black hole "shadow". Real photon
    # rings are exponentially thin in impact-parameter space because the
    # winding angle diverges logarithmically as b -> b_crit, so a small
    # extra bit of winding maps to an extremely narrow band of pixels.
    # We approximate that divergence directly in angle-space: any ray
    # that manages 1.5+ full winds gets a sharp, narrow brightness spike
    # that decays fast with additional winding (rays winding *even more*
    # are vanishingly rare/thin — deeper into a self-similar ladder of
    # fainter sub-images — so the falloff keeps the ring thin instead of
    # smearing brightness over every multiply-wound ray).
    winding_frac = accum_angle / (2.0 * PI)
    result = ti.Vector([0.0, 0.0, 0.0])
    if winding_frac > 1.5:
        ring_boost = ti.exp(-(winding_frac - 1.5) * 6.0) * 9.0
        result = ti.Vector([1.0, 0.96, 0.88]) * ring_boost
    return result


@ti.func
def kerr_shadow_asymmetry(spin: ti.f32, psi: ti.f32, psi_pro: ti.f32) -> ti.f32:
    # Approximate D-shaped flattening of the Kerr shadow boundary as a
    # function of the azimuthal angle psi around the projected shadow
    # rim (psi=0 taken as the side of the shadow toward the hole's
    # prograde-rotation direction as seen on screen). For a=0 (no spin)
    # this returns 1.0 everywhere, i.e. a perfectly circular Schwarzschild
    # shadow. For a>0, the prograde side of the shadow flattens inward
    # (photon orbits co-rotating with the hole have a smaller critical
    # impact parameter) while the retrograde side bulges slightly
    # outward, producing the classic asymmetric "D" silhouette. This is
    # a low-order cos(psi) fit to the qualitative Bardeen shadow shape,
    # not an exact geodesic solution — a full match requires integrating
    # null geodesics in Boyer-Lindquist coordinates with the Carter
    # constant, which this fast approximate-GR renderer does not do.
    # psi_pro is the screen-space azimuth of the hole's prograde direction
    # (the side whose co-rotating photon orbits — and whose Doppler-boosted
    # gas — face the camera). Measuring the flattening relative to psi_pro
    # rather than relative to the raw screen +u axis means the D keeps its
    # correct physical orientation as the camera orbits: swing round to the
    # far side and the flat edge swaps to the other limb, as it must.
    #
    # The sign is negative because co-rotating photons have a *smaller*
    # critical impact parameter, so the shadow edge cuts inward on the
    # prograde side and bulges outward on the retrograde side.
    asym = spin * 0.18
    return 1.0 - asym * ti.cos(psi - psi_pro)


@ti.kernel
def render_frame(
    spin:      ti.f32,
    exposure:  ti.f32,
    max_steps: ti.i32,
    time:      ti.f32,
    fov:       ti.f32,
    hue:       ti.f32,
    sat:       ti.f32,
    val:       ti.f32,
    doppler:   ti.f32,
    render_w:  ti.i32,
    render_h:  ti.i32
):
    origin = cam_origin[None]
    right  = cam_right[None]
    up     = cam_up[None]
    fwd    = cam_forward[None]

    r_hz  = horizon_radius(spin)
    r_isc = isco_radius(spin)

    # ── Screen-space orientation of the hole's prograde side ─────────────
    # The disk (and the hole) rotate about +Y, with gas velocity at a point
    # p given by Omega * (Y_hat x p). The limb moving toward the camera is
    # therefore the one lying along d_pro = Y_hat x fwd, where fwd points
    # from the camera toward the hole. Projecting d_pro onto the camera's
    # (right, up) basis gives the screen angle of that limb, which the Kerr
    # shadow-asymmetry term uses to orient the flat edge of the "D".
    spin_axis = ti.Vector([0.0, 1.0, 0.0])
    d_pro     = spin_axis.cross(fwd)
    psi_pro   = ti.atan2(d_pro.dot(up), d_pro.dot(right))

    for i, j in ti.ndrange(render_w, render_h):

        u = (2.0 * (ti.cast(i, ti.f32) + 0.5) / render_w - 1.0) * ASPECT * fov
        v = (2.0 * (ti.cast(j, ti.f32) + 0.5) / render_h - 1.0) * fov

        # Screen-space azimuth of this pixel around the image center,
        # used only for the Kerr shadow-boundary asymmetry shaping below.
        psi = ti.atan2(v, u)
        shadow_shape = kerr_shadow_asymmetry(spin, psi, psi_pro)

        rd = fwd + u * right + v * up
        rd_len = rd.norm()
        ray_dir = rd / ti.max(rd_len, 1e-8)

        pos      = origin
        vel      = ray_dir
        color    = ti.Vector([0.003, 0.004, 0.010])
        prev_y   = pos.y
        absorbed = False

        # ── Photon-ring winding tracker ───────────────────────────────
        # Accumulates the total angle turned by the ray's velocity
        # vector across the march. Rays that orbit near the photon
        # sphere before escaping wind through a large angle; used below
        # to paint the thin, bright photon ring at the shadow boundary.
        accum_angle = 0.0
        prev_vel = vel

        p_hit = [0] * MAX_TEST_PARTICLES

        for step in range(MAX_MARCH):
            if step >= max_steps:
                break

            r = pos.norm()

            # Kerr shadow asymmetry: shrink/expand the effective capture
            # horizon slightly with screen-space azimuth so the
            # silhouette (and hence the photon ring traced around it)
            # comes out as a flattened "D" for spinning holes instead of
            # a perfect circle.
            if r < r_hz * 1.02 * shadow_shape:
                absorbed = True
                break

            if r > R_MAX:
                break

            dr = r - r_hz
            h = 0.03 * ti.max(dr, 0.01) + 0.008 * dr * dr + 0.001
            h = ti.min(h, 2.5)

            prev_pos = pos
            prev_y   = pos.y

            acc  = gravity_accel(pos, vel, spin)
            vel += acc * h

            v_len = vel.norm()
            if v_len > 1e-8:
                vel /= v_len

            cos_turn = ti.min(ti.max(vel.dot(prev_vel), -1.0), 1.0)
            accum_angle += ti.acos(cos_turn)
            prev_vel = vel

            pos += vel * h

            # ── Volumetric hot-flow integration ───────────────────────
            # Rather than treating the accretion flow as an infinitely
            # thin geometric plane sampled only at the exact y=0
            # crossing, integrate emission through a flared slab around
            # the midplane: scale height h_scale(r) grows with radius
            # (a puffed-up, geometrically-thick hot flow, appropriate
            # for a radiatively inefficient accretor like Sgr A*, rather
            # than a razor-thin standard disk), and the local density
            # falls off as a Gaussian in y/h_scale. Every march step that
            # samples inside this slab contributes emission weighted by
            # both that density and the step length, so a ray grazing
            # through the flow at a shallow angle picks up contributions
            # from many steps — turbulent structure and all — instead of
            # one instantaneous plane-crossing spike.
            r_disk = pos.norm()
            if r_disk > r_isc and r_disk < R_OUT:
                h_scale = 0.10 * r_disk + 0.22
                y_norm  = pos.y / h_scale
                dens    = ti.exp(-0.5 * y_norm * y_norm)
                if dens > 0.004:
                    em = disk_emission(
                             pos, vel, r_disk, r_isc,
                             spin, time, hue, sat, val, doppler)
                    color += em * dens * ti.min(h, 0.6) * 0.9

            seg     = pos - prev_pos
            seg_len2 = seg.dot(seg)
            for p in ti.static(range(MAX_TEST_PARTICLES)):
                if particle_active[p] == 1 and p_hit[p] == 0:
                    ppos = particle_pos[p]
                    t_p = 0.0
                    if seg_len2 > 1e-12:
                        t_p = (ppos - prev_pos).dot(seg) / seg_len2
                        t_p = ti.min(ti.max(t_p, 0.0), 1.0)
                    closest  = prev_pos + seg * t_p
                    dist     = (closest - ppos).norm()
                    capture_r = PARTICLE_LENS_RADIUS + 0.15 * h
                    if dist < capture_r:
                        falloff = ti.exp(-(dist / capture_r) ** 2 * 3.0)
                        color  += particle_color[p] * PARTICLE_LENS_BRIGHTNESS * falloff
                        p_hit[p] = 1

        if not absorbed and pos.norm() >= R_MAX:
            color += sample_skybox(vel)
            color += sample_real_stars(vel) * star_twinkle_factor(vel, time)
            color += background_stars(vel) * 0.4
            # Photon ring: rays that wound >= 1.5 full turns around the
            # hole before escaping sit at (or extremely near) the
            # critical impact parameter — paint the thin bright ring.
            color += photon_ring_boost(accum_angle)

        pixels_compute[i, j] = tone_map(color, exposure)


@ti.kernel
def upscale_bilinear(render_w: ti.i32, render_h: ti.i32):
    for i, j in pixels_display:
        u = (ti.cast(i, ti.f32) + 0.5) / WIN_W * render_w - 0.5
        v = (ti.cast(j, ti.f32) + 0.5) / WIN_H * render_h - 0.5

        i_fl = ti.floor(u)
        j_fl = ti.floor(v)
        
        i0 = ti.max(0, ti.min(ti.cast(i_fl, ti.i32), render_w - 1))
        j0 = ti.max(0, ti.min(ti.cast(j_fl, ti.i32), render_h - 1))
        i1 = ti.min(i0 + 1, render_w - 1)
        j1 = ti.min(j0 + 1, render_h - 1)

        wu = u - i_fl
        wv = v - j_fl

        c00 = pixels_compute[i0, j0]
        c10 = pixels_compute[i1, j0]
        c01 = pixels_compute[i0, j1]
        c11 = pixels_compute[i1, j1]

        c0 = c00 * (1.0 - wu) + c10 * wu
        c1 = c01 * (1.0 - wu) + c11 * wu
        c = c0 * (1.0 - wv) + c1 * wv

        pixels_display[i, j] = c


BLOOM_THRESHOLD = 0.55
BLOOM_BLUR_RADIUS = 4


@ti.kernel
def bloom_extract():
    sx = ti.cast(WIN_W, ti.f32) / BLOOM_W
    sy = ti.cast(WIN_H, ti.f32) / BLOOM_H
    for i, j in bloom_a:
        xi = ti.min(ti.cast((ti.cast(i, ti.f32) + 0.5) * sx, ti.i32), WIN_W - 1)
        yj = ti.min(ti.cast((ti.cast(j, ti.f32) + 0.5) * sy, ti.i32), WIN_H - 1)
        xi1 = ti.min(xi + 1, WIN_W - 1)
        yj1 = ti.min(yj + 1, WIN_H - 1)

        c = (pixels_display[xi, yj] + pixels_display[xi1, yj]
             + pixels_display[xi, yj1] + pixels_display[xi1, yj1]) * 0.25

        luma = c.dot(ti.Vector([0.299, 0.587, 0.114]))
        knee = ti.max(luma - BLOOM_THRESHOLD, 0.0)
        factor = knee / ti.max(luma, 1e-4)
        bloom_a[i, j] = c * factor


@ti.kernel
def bloom_blur_h():
    for i, j in bloom_b:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for k in range(-BLOOM_BLUR_RADIUS, BLOOM_BLUR_RADIUS + 1):
            w = ti.exp(-0.5 * (ti.cast(k, ti.f32) / 2.0) ** 2)
            xi = ti.min(ti.max(i + k, 0), BLOOM_W - 1)
            acc += bloom_a[xi, j] * w
            wsum += w
        bloom_b[i, j] = acc / wsum


@ti.kernel
def bloom_blur_v():
    for i, j in bloom_a:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for k in range(-BLOOM_BLUR_RADIUS, BLOOM_BLUR_RADIUS + 1):
            w = ti.exp(-0.5 * (ti.cast(k, ti.f32) / 2.0) ** 2)
            yj = ti.min(ti.max(j + k, 0), BLOOM_H - 1)
            acc += bloom_a[i, yj] * w
            wsum += w
        bloom_b[i, j] = acc / wsum


@ti.kernel
def bloom_composite(strength: ti.f32):
    for i, j in pixels_display:
        u = (ti.cast(i, ti.f32) + 0.5) / WIN_W * BLOOM_W - 0.5
        v = (ti.cast(j, ti.f32) + 0.5) / WIN_H * BLOOM_H - 0.5

        i_fl = ti.floor(u)
        j_fl = ti.floor(v)
        i0 = ti.max(0, ti.min(ti.cast(i_fl, ti.i32), BLOOM_W - 1))
        j0 = ti.max(0, ti.min(ti.cast(j_fl, ti.i32), BLOOM_H - 1))
        i1 = ti.min(i0 + 1, BLOOM_W - 1)
        j1 = ti.min(j0 + 1, BLOOM_H - 1)

        wu = u - i_fl
        wv = v - j_fl

        c00 = bloom_b[i0, j0]
        c10 = bloom_b[i1, j0]
        c01 = bloom_b[i0, j1]
        c11 = bloom_b[i1, j1]

        c0 = c00 * (1.0 - wu) + c10 * wu
        c1 = c01 * (1.0 - wu) + c11 * wu
        glow = c0 * (1.0 - wv) + c1 * wv

        pixels_display[i, j] = ti.min(pixels_display[i, j] + glow * strength, 1.0)


def apply_bloom(strength: float = 0.6):
    bloom_extract()
    bloom_blur_h()
    bloom_blur_v()
    bloom_composite(strength)


class TestParticleOrbit:
    def __init__(self, a_au, ecc, inclination_deg=25.0, gr_enabled=True,
                 color=(1.0, 0.75, 0.45), name="S301",
                 gm_bh=S301_GM_BH, m_bh_msun=S301_M_BH_MSUN,
                 v_peri_auyr=None):
        self.a_au        = a_au
        self.ecc         = ecc
        self.inclination_deg = inclination_deg
        self.gr_enabled  = gr_enabled
        self.color       = color
        self.name        = name
        self.gm_bh       = gm_bh
        self.m_bh_msun   = m_bh_msun

        self.r_peri_au   = a_au * (1.0 - ecc)
        self.r_apo_au    = a_au * (1.0 + ecc)
        self.period_yr   = math.sqrt(a_au ** 3 / m_bh_msun)
        if v_peri_auyr is not None:
            self.v_peri_auyr = v_peri_auyr
        else:
            self.v_peri_auyr = math.sqrt(
                gm_bh / a_au * (1.0 + ecc) / max(1.0 - ecc, 1e-6)
            )

        self.state = np.array(
            [self.r_peri_au, 0.0, 0.0, self.v_peri_auyr], dtype=np.float64
        )
        self.t_years = 0.0

        self.trail = np.zeros((S301_TRAIL_MAX, 4), dtype=np.float32)
        self.trail_len = 0
        self.trail_head = 0

        self._sample_interval_yr = self.period_yr / 260.0
        self._t_since_sample = 1e9

    @staticmethod
    def _accel(state, gm, c, gr_enabled):
        x, y, vx, vy = state
        r2 = x * x + y * y
        r = math.sqrt(r2)
        r3 = r2 * r

        L = x * vy - y * vx

        pn_factor = 1.0
        if gr_enabled:
            pn_factor = 1.0 + 3.0 * (L * L) / (c * c * r2)

        a_mag = -gm / r3 * pn_factor
        return np.array([vx, vy, a_mag * x, a_mag * y])

    def _rk4_step(self, state, dt):
        gm, c, gr = self.gm_bh, C_AU_YR, self.gr_enabled
        k1 = self._accel(state, gm, c, gr)
        k2 = self._accel(state + 0.5 * dt * k1, gm, c, gr)
        k3 = self._accel(state + 0.5 * dt * k2, gm, c, gr)
        k4 = self._accel(state + dt * k3, gm, c, gr)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, years):
        remaining = years
        max_substep = self.period_yr / 400.0

        while remaining > 1e-12:
            x, y, vx, vy = self.state
            r = math.sqrt(x * x + y * y)

            dt = 2.5e-5 * self.period_yr * (r / self.r_peri_au) ** 1.5
            dt = min(dt, max_substep, remaining)

            self.state = self._rk4_step(self.state, dt)
            self.t_years += dt
            remaining -= dt
            self._t_since_sample += dt

            if self._t_since_sample >= self._sample_interval_yr:
                self._t_since_sample = 0.0
                self._push_trail_point()

    def _push_trail_point(self):
        x_au, y_au, vx, vy = self.state
        incl = math.radians(self.inclination_deg)
        x_code = x_au * AU_TO_CODE
        z_code = y_au * AU_TO_CODE
        y_code = z_code * math.sin(incl)
        z_code = z_code * math.cos(incl)

        idx = self.trail_head
        self.trail[idx, 0] = x_code
        self.trail[idx, 1] = y_code
        self.trail[idx, 2] = z_code
        self.trail[idx, 3] = 1.0
        self.trail_head = (self.trail_head + 1) % S301_TRAIL_MAX
        self.trail_len = min(self.trail_len + 1, S301_TRAIL_MAX)

    def position_code_units(self):
        x_au, y_au, vx, vy = self.state
        incl = math.radians(self.inclination_deg)
        x_code = x_au * AU_TO_CODE
        z_raw = y_au * AU_TO_CODE
        y_code = z_raw * math.sin(incl)
        z_code = z_raw * math.cos(incl)
        return np.array([x_code, y_code, z_code], dtype=np.float32)

    def distance_au(self):
        x, y, vx, vy = self.state
        return math.sqrt(x * x + y * y)

    def speed_km_s(self):
        x, y, vx, vy = self.state
        v_auyr = math.sqrt(vx * vx + vy * vy)
        return v_auyr * AU_KM / YEAR_S

    def precession_deg_per_orbit(self):
        return math.degrees(
            6.0 * math.pi * self.gm_bh /
            (C_AU_YR ** 2 * self.a_au * (1.0 - self.ecc ** 2))
        )

    def upload_to(self, i):
        _particle_pos_np[i]    = self.position_code_units()
        _particle_color_np[i]  = np.array(self.color, dtype=np.float32)
        _particle_active_np[i] = 1

        n = self.trail_len
        _particle_trailcnt_np[i] = n
        _particle_trail_np[i]    = 0.0
        if n == 0:
            return
        if n < S301_TRAIL_MAX:
            ordered = self.trail[:n].copy()
        else:
            ordered = np.concatenate(
                [self.trail[self.trail_head:], self.trail[:self.trail_head]], axis=0
            )
        ordered[:, 3] = np.linspace(0.05, 1.0, n, dtype=np.float32)
        _particle_trail_np[i, :n] = ordered


@ti.func
def photon_shadow_cos_half_angle(r_obs: ti.f32) -> ti.f32:
    cos_psi = -1.0
    if r_obs > 2.0 * M:
        s = (3.0 * ti.sqrt(3.0) * M / r_obs) * ti.sqrt(ti.max(1.0 - 2.0 * M / r_obs, 0.0))
        s = ti.min(s, 1.0)
        cos_psi = ti.sqrt(ti.max(1.0 - s * s, 0.0))
    return cos_psi


@ti.func
def lensed_apparent_direction(
    rel:    ti.math.vec3,
    origin: ti.math.vec3
) -> ti.math.vec3:
    """Approximate the *lensed* apparent direction of a point at world
    position (origin + rel) as seen by an observer at `origin`, so that
    screen-space overlays (particle markers/trails) bend around Sgr A*
    the same way the ray-marched disk/background/star layers already do.

    This uses the classic weak-field point-mass deflection angle
        alpha ≈ 4M / b
    where b is the photon's impact parameter relative to the lens
    (the black hole, sitting at the world origin) — the same formula
    behind the historic 1919 solar-eclipse light-bending measurement,
    generalized here to Sgr A*'s mass in geometric units (M=1).

    `theta` is the angular separation, as seen from the camera, between
    "straight at the black hole" and "straight at the particle" — i.e.
    where the particle sits on the sky relative to the hole's silhouette.
    The impact parameter of a straight ray from the camera toward the
    particle, relative to the lens at the origin, is then
        b ≈ D_L * sin(theta)
    with D_L = |origin| the camera's distance from the lens. Increasing
    theta by alpha pushes the apparent image further from the black
    hole's screen position — exactly the outward-bending look of real
    gravitational lensing — and the bend strength grows sharply as the
    line of sight passes closer to the hole (small b), matching the
    photon-sphere behaviour used elsewhere in the renderer.
    """
    r = rel.norm()
    d = rel / ti.max(r, 1e-8)

    d_l = origin.norm()
    result = d

    if d_l > 1e-4:
        to_bh = -origin / d_l
        cos_theta = ti.min(ti.max(d.dot(to_bh), -1.0), 1.0)
        theta = ti.acos(cos_theta)

        b = d_l * ti.sin(theta)
        b_safe = ti.max(b, 0.35)

        alpha = 4.0 * M / b_safe
        alpha = ti.min(alpha, 2.4)   # cap so near-axis points don't invert

        theta_new = theta + alpha

        perp = d - to_bh * cos_theta
        perp_len = perp.norm()
        if perp_len > 1e-6:
            perp_hat = perp / perp_len
            result = to_bh * ti.cos(theta_new) + perp_hat * ti.sin(theta_new)

    return result


@ti.func
def lensed_secondary_image(
    rel:    ti.math.vec3,
    origin: ti.math.vec3
) -> ti.math.vec4:
    """Approximate the *secondary* (ghost) image of a point-source formed
    by a point-mass lens: the fainter image that appears on the opposite
    side of the black hole from the source's true position, formed by
    light that passes the lens on the other side before reaching the
    observer. Real point-mass lensing always produces this second image
    together with the primary one computed by lensed_apparent_direction;
    without it, a marker/trail passing behind Sgr A* would just vanish
    into the shadow instead of throwing a dim mirrored image out the
    other side, the way the real ray-marched disk/background can.

    Returned as a vec4: (dir.x, dir.y, dir.z, weight), where `weight`
    is a 0..1 brightness factor for this ghost image — it fades out as
    the true impact parameter b grows, since a distant, weakly-lensed
    source barely produces a visible secondary image at all, and it
    strengthens the closer the sightline passes to the photon sphere.
    """
    r = rel.norm()
    d = rel / ti.max(r, 1e-8)

    d_l = origin.norm()
    result = ti.Vector([d.x, d.y, d.z, 0.0])

    if d_l > 1e-4:
        to_bh = -origin / d_l
        cos_theta = ti.min(ti.max(d.dot(to_bh), -1.0), 1.0)
        theta = ti.acos(cos_theta)

        b = d_l * ti.sin(theta)
        b_safe = ti.max(b, 0.35)

        alpha = 4.0 * M / b_safe
        alpha = ti.min(alpha, 2.4)
        theta_new = theta + alpha

        perp = d - to_bh * cos_theta
        perp_len = perp.norm()
        img_dir = d
        if perp_len > 1e-6:
            perp_hat = perp / perp_len
            # Mirror to the opposite side of the lens axis — the
            # ghost image sits diametrically across the black hole
            # from the primary bent image.
            img_dir = to_bh * ti.cos(theta_new) - perp_hat * ti.sin(theta_new)

        # Weight fades out for large impact parameters (weak lensing
        # barely produces a visible secondary image) and strengthens
        # near the photon sphere, capped well below the primary image's
        # brightness since real secondary images are always fainter.
        weight = ti.exp(-b_safe * 0.6) * 0.6

        result = ti.Vector([img_dir.x, img_dir.y, img_dir.z, weight])

    return result


@ti.kernel
def draw_particles(
    fov: ti.f32,
    spin: ti.f32,
    star_brightness: ti.f32
):
    """Screen-space splat of every active test particle's trail + current
    position onto the display framebuffer. Both the marker and every
    trail point are first passed through lensed_apparent_direction() so
    their on-screen position bends around the hole exactly like the
    ray-marched disk/star layers, instead of being drawn at their flat
    (unlensed) straight-line screen projection."""
    origin = cam_origin[None]
    right  = cam_right[None]
    up     = cam_up[None]
    fwd    = cam_forward[None]
    r_obs    = origin.norm()
    cos_psi  = photon_shadow_cos_half_angle(r_obs)

    for p, idx in ti.ndrange(MAX_TEST_PARTICLES, S301_TRAIL_MAX):
        if particle_active[p] == 1 and idx < particle_trail_count[p]:
            tp = particle_trail_data[p, idx]
            pos_w = ti.Vector([tp[0], tp[1], tp[2]])
            bright = tp[3]
            base_col = particle_color[p]
            rel = pos_w - origin
            rel_len = rel.norm()
            depth_flat = rel.dot(fwd)
            in_shadow = rel_len > 1e-6 and (depth_flat / rel_len) > cos_psi
            if depth_flat > 1e-3 and not in_shadow:
                bright_boosted = 0.35 + 0.65 * bright
                col = base_col * bright_boosted * 1.6

                ld = lensed_apparent_direction(rel, origin)
                depth = ld.dot(fwd)
                if depth > 1e-4:
                    uu = ld.dot(right) / depth
                    vv = ld.dot(up) / depth
                    u_ndc = uu / (ASPECT * fov)
                    v_ndc = vv / fov
                    if -1.05 < u_ndc < 1.05 and -1.05 < v_ndc < 1.05:
                        px = (u_ndc * 0.5 + 0.5) * WIN_W
                        py = (v_ndc * 0.5 + 0.5) * WIN_H
                        radius = 4
                        for di in range(-radius, radius + 1):
                            for dj in range(-radius, radius + 1):
                                xi = ti.cast(px, ti.i32) + di
                                yj = ti.cast(py, ti.i32) + dj
                                if 0 <= xi < WIN_W and 0 <= yj < WIN_H:
                                    d2 = ti.cast(di * di + dj * dj, ti.f32)
                                    falloff = ti.exp(-d2 * 0.35)
                                    ti.atomic_add(pixels_display[xi, yj], col * falloff * 0.55)

                # Secondary (ghost) image — dim mirror on the far side
                # of the hole, only meaningfully visible when the true
                # sightline passes close to the photon sphere.
                sec = lensed_secondary_image(rel, origin)
                sec_dir = ti.Vector([sec.x, sec.y, sec.z])
                sec_w   = sec.w
                if sec_w > 0.01:
                    sdepth = sec_dir.dot(fwd)
                    if sdepth > 1e-4:
                        suu = sec_dir.dot(right) / sdepth
                        svv = sec_dir.dot(up) / sdepth
                        su_ndc = suu / (ASPECT * fov)
                        sv_ndc = svv / fov
                        if -1.05 < su_ndc < 1.05 and -1.05 < sv_ndc < 1.05:
                            spx = (su_ndc * 0.5 + 0.5) * WIN_W
                            spy = (sv_ndc * 0.5 + 0.5) * WIN_H
                            sradius = 4
                            scol = col * sec_w
                            for di in range(-sradius, sradius + 1):
                                for dj in range(-sradius, sradius + 1):
                                    xi = ti.cast(spx, ti.i32) + di
                                    yj = ti.cast(spy, ti.i32) + dj
                                    if 0 <= xi < WIN_W and 0 <= yj < WIN_H:
                                        d2 = ti.cast(di * di + dj * dj, ti.f32)
                                        falloff = ti.exp(-d2 * 0.35)
                                        ti.atomic_add(pixels_display[xi, yj], scol * falloff * 0.55)

    for p in range(MAX_TEST_PARTICLES):
        if particle_active[p] == 1:
            sp = particle_pos[p]
            base_col = particle_color[p]
            rel = sp - origin
            rel_len = rel.norm()
            depth_flat = rel.dot(fwd)
            in_shadow = rel_len > 1e-6 and (depth_flat / rel_len) > cos_psi
            if depth_flat > 1e-3 and not in_shadow:
                ld = lensed_apparent_direction(rel, origin)
                depth = ld.dot(fwd)
                if depth > 1e-4:
                    uu = ld.dot(right) / depth
                    vv = ld.dot(up) / depth
                    u_ndc = uu / (ASPECT * fov)
                    v_ndc = vv / fov
                    if -1.15 < u_ndc < 1.15 and -1.15 < v_ndc < 1.15:
                        px = (u_ndc * 0.5 + 0.5) * WIN_W
                        py = (v_ndc * 0.5 + 0.5) * WIN_H

                        halo_radius = 22
                        halo_col = base_col * star_brightness
                        for di in range(-halo_radius, halo_radius + 1):
                            for dj in range(-halo_radius, halo_radius + 1):
                                xi = ti.cast(px, ti.i32) + di
                                yj = ti.cast(py, ti.i32) + dj
                                if 0 <= xi < WIN_W and 0 <= yj < WIN_H:
                                    d2 = ti.cast(di * di + dj * dj, ti.f32)
                                    falloff = ti.exp(-d2 * 0.02)
                                    ti.atomic_add(pixels_display[xi, yj], halo_col * falloff * 0.16)

                        core_radius = 10
                        core = (base_col * 0.55 + ti.Vector([1.0, 0.96, 0.85]) * 0.45) * star_brightness * 1.3
                        for di in range(-core_radius, core_radius + 1):
                            for dj in range(-core_radius, core_radius + 1):
                                xi = ti.cast(px, ti.i32) + di
                                yj = ti.cast(py, ti.i32) + dj
                                if 0 <= xi < WIN_W and 0 <= yj < WIN_H:
                                    d2 = ti.cast(di * di + dj * dj, ti.f32)
                                    falloff = ti.exp(-d2 * 0.09)
                                    ti.atomic_add(pixels_display[xi, yj], core * falloff * 0.65)

                        ring_radius  = 34
                        ring_thick   = 2.4
                        ring_col     = (base_col * 0.6 + ti.Vector([1.0, 0.92, 0.55]) * 0.4) * star_brightness
                        ring_span    = ring_radius + 4
                        for di in range(-ring_span, ring_span + 1):
                            for dj in range(-ring_span, ring_span + 1):
                                xi = ti.cast(px, ti.i32) + di
                                yj = ti.cast(py, ti.i32) + dj
                                if 0 <= xi < WIN_W and 0 <= yj < WIN_H:
                                    dist = ti.sqrt(ti.cast(di * di + dj * dj, ti.f32))
                                    ring_falloff = ti.exp(-((dist - ring_radius) ** 2) / (2.0 * ring_thick * ring_thick))
                                    ti.atomic_add(pixels_display[xi, yj], ring_col * ring_falloff * 0.5)

                        # Secondary (ghost) marker image — a dimmer,
                        # smaller halo+core echo mirrored to the far
                        # side of the black hole (no ID ring, since
                        # this is a lensing artifact, not the "real"
                        # tracked position).
                        sec = lensed_secondary_image(rel, origin)
                        sec_dir = ti.Vector([sec.x, sec.y, sec.z])
                        sec_w   = sec.w
                        if sec_w > 0.01:
                            sdepth = sec_dir.dot(fwd)
                            if sdepth > 1e-4:
                                suu = sec_dir.dot(right) / sdepth
                                svv = sec_dir.dot(up) / sdepth
                                su_ndc = suu / (ASPECT * fov)
                                sv_ndc = svv / fov
                                if -1.15 < su_ndc < 1.15 and -1.15 < sv_ndc < 1.15:
                                    spx = (su_ndc * 0.5 + 0.5) * WIN_W
                                    spy = (sv_ndc * 0.5 + 0.5) * WIN_H

                                    ghost_halo_radius = 14
                                    ghost_halo_col = base_col * star_brightness * sec_w
                                    for di in range(-ghost_halo_radius, ghost_halo_radius + 1):
                                        for dj in range(-ghost_halo_radius, ghost_halo_radius + 1):
                                            xi = ti.cast(spx, ti.i32) + di
                                            yj = ti.cast(spy, ti.i32) + dj
                                            if 0 <= xi < WIN_W and 0 <= yj < WIN_H:
                                                d2 = ti.cast(di * di + dj * dj, ti.f32)
                                                falloff = ti.exp(-d2 * 0.03)
                                                ti.atomic_add(pixels_display[xi, yj], ghost_halo_col * falloff * 0.16)

                                    ghost_core_radius = 6
                                    ghost_core = (base_col * 0.55 + ti.Vector([1.0, 0.96, 0.85]) * 0.45) * star_brightness * 1.3 * sec_w
                                    for di in range(-ghost_core_radius, ghost_core_radius + 1):
                                        for dj in range(-ghost_core_radius, ghost_core_radius + 1):
                                            xi = ti.cast(spx, ti.i32) + di
                                            yj = ti.cast(spy, ti.i32) + dj
                                            if 0 <= xi < WIN_W and 0 <= yj < WIN_H:
                                                d2 = ti.cast(di * di + dj * dj, ti.f32)
                                                falloff = ti.exp(-d2 * 0.1)
                                                ti.atomic_add(pixels_display[xi, yj], ghost_core * falloff * 0.65)


class OrbitalCamera:
    def __init__(self):
        self.azimuth   = 0.0
        self.elevation = 0.101
        self.distance  = 44.800
        self.fov       = 0.212
        self.sens      = 3.5
        self.zoom_sens = 25.0
        self.min_dist  = 4.0
        self.max_dist  = 60.0
        self._prev     = None

        self.transitioning = False
        self._t_start    = None
        self._t_target    = None
        self._t_daz       = 0.0
        self._t_elapsed   = 0.0
        self._t_duration  = 2.2

    def update(self, window):
        cur = window.get_cursor_pos()
        mx, my = cur[0], cur[1]

        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)
        if (lmb or rmb) and self.transitioning:
            self.transitioning = False

        if self._prev is not None and not self.transitioning:
            dx = mx - self._prev[0]
            dy = my - self._prev[1]

            if lmb:
                self.azimuth   -= dx * self.sens
                self.elevation += dy * self.sens
                self.elevation  = max(-1.5, min(1.5, self.elevation))

            if rmb:
                self.distance -= dy * self.zoom_sens
                self.distance  = max(self.min_dist,
                                     min(self.max_dist, self.distance))

        self._prev = (mx, my)

        if not self.transitioning:
            if window.is_pressed('w'):
                self.distance = max(self.min_dist, self.distance - 0.25)
            if window.is_pressed('s'):
                self.distance = min(self.max_dist, self.distance + 0.25)
            if window.is_pressed('a'):
                self.azimuth += 0.03
            if window.is_pressed('d'):
                self.azimuth -= 0.03

    def fly_to(self, bookmark, duration=2.2):
        self._t_start = {
            "azimuth": self.azimuth, "elevation": self.elevation,
            "distance": self.distance, "fov": self.fov,
        }
        daz = bookmark["azimuth"] - self.azimuth
        daz = (daz + math.pi) % (2.0 * math.pi) - math.pi
        self._t_daz      = daz
        self._t_target   = bookmark
        self._t_elapsed  = 0.0
        self._t_duration = max(0.05, duration)
        self.transitioning = True

    def update_transition(self, dt):
        if not self.transitioning:
            return
        self._t_elapsed += dt
        t = min(1.0, self._t_elapsed / self._t_duration)
        e = t * t * (3.0 - 2.0 * t)
        s, tg = self._t_start, self._t_target
        self.azimuth   = s["azimuth"] + self._t_daz * e
        self.elevation = s["elevation"] + (tg["elevation"] - s["elevation"]) * e
        self.distance  = s["distance"]  + (tg["distance"]  - s["distance"])  * e
        self.fov       = s["fov"]       + (tg["fov"]       - s["fov"])       * e
        if t >= 1.0:
            self.transitioning = False

    def upload(self):
        ce = np.cos(self.elevation);  se = np.sin(self.elevation)
        ca = np.cos(self.azimuth);    sa = np.sin(self.azimuth)

        pos = np.array([
            self.distance * ce * sa,
            self.distance * se,
            self.distance * ce * ca
        ], dtype=np.float32)

        fwd = -pos / (np.linalg.norm(pos) + 1e-12)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right    = np.cross(fwd, world_up)
        rn       = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= rn

        up = np.cross(right, fwd)
        up /= (np.linalg.norm(up) + 1e-12)

        cam_origin[None]  = pos.tolist()
        cam_right[None]   = right.tolist()
        cam_up[None]      = up.tolist()
        cam_forward[None] = fwd.tolist()


def pick_star(mx: float, my: float, fov: float, max_angle_deg: float = None):
    right = np.array(cam_right[None],   dtype=np.float64)
    up    = np.array(cam_up[None],      dtype=np.float64)
    fwd   = np.array(cam_forward[None], dtype=np.float64)

    u = (2.0 * mx - 1.0) * ASPECT * fov
    v = (2.0 * my - 1.0) * fov

    rd = fwd + u * right + v * up
    rd_norm = np.linalg.norm(rd)
    if rd_norm < 1e-12:
        return None
    ray_dir = rd / rd_norm

    dots = STAR_DIRS @ ray_dir
    idx = int(np.argmax(dots))
    best_dot = float(np.clip(dots[idx], -1.0, 1.0))
    angle_deg = math.degrees(math.acos(best_dot))

    if max_angle_deg is None:
        max_angle_deg = math.degrees(max(0.010, fov * 0.09))

    if angle_deg <= max_angle_deg:
        hr, name, ra_h, dec_deg, vmag, bv = REAL_STARS[idx]
        return {
            "hr": hr, "name": name, "ra_h": ra_h, "dec_deg": dec_deg,
            "vmag": vmag, "bv": bv,
            "temp_k": bv_to_temperature(bv),
            "angle_deg": angle_deg,
        }
    return None


CAMERA_BOOKMARKS = [
    {
        "name": "ISCO Close-up",
        "azimuth": 0.35, "elevation": 0.12, "distance": 4.6, "fov": 0.16,
    },
    {
        "name": "Wide Skybox View",
        "azimuth": 2.40, "elevation": 0.55, "distance": 58.0, "fov": 1.25,
    },
    {
        "name": "S301 Apoapsis",
        "azimuth": math.pi / 2.0, "elevation": 0.05, "distance": 42.0, "fov": 0.42,
    },
]


def main():
    print()
    print("===============================================================")
    print("|  O  SAGITTARIUS A* SIMULATION                              |")
    print("===============================================================")
    print("|  Left-drag   ->  orbit        Right-drag  ->  zoom          |")
    print("|  W / S       ->  zoom         A / D       ->  orbit         |")
    print("|  GUI sliders ->  physics, Milky Way & visual parameters     |")
    print("===============================================================")
    print()

    window = ti.ui.Window("Sagittarius A* — Relativistic Control Deck", (WIN_W, WIN_H), vsync=True)
    
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, "Sagittarius A* — Relativistic Control Deck")
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 3)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(ctypes.c_int(0x00100C0B)), 4)
    except Exception:
        pass

    canvas = window.get_canvas()
    camera = OrbitalCamera()

    PARTICLE_PALETTE = [
        (1.00, 0.75, 0.45),
        (0.35, 0.85, 1.00),
        (0.65, 1.00, 0.35),
        (1.00, 0.40, 0.85),
        (0.55, 0.55, 1.00),
        (1.00, 0.55, 0.15),
        (0.55, 1.00, 0.80),
    ]
    particles = [
        TestParticleOrbit(
            a_au=S301_A_AU, ecc=S301_ECC, inclination_deg=25.0,
            gr_enabled=True, color=PARTICLE_PALETTE[0], name="S301"
            # v_peri intentionally omitted -> derived via vis-viva so the
            # integrated orbit matches the labeled (a, ecc, period) elements.
        )
    ]
    particle_spawn_count = 0

    spin       = 0.300
    exposure   = 4.20
    max_steps  = 500
    sim_time   = 0.0
    flow_speed = 0.20
    doppler_on = 1.000
    hue        = 0.055
    sat        = 0.72
    val        = 4.20
    render_scale = 1.000
    auto_orbit   = False
    bloom_strength = 0.600

    orbit_time_warp  = 0.35
    orbit_show_trail = True
    orbit_paused     = False

    spawn_a_au   = 400.0
    spawn_ecc    = 0.90
    spawn_incl   = 25.0
    spawn_gr_on  = True

    mw_tilt_deg   = 63.0
    mw_intensity  = 0.55
    mw_hue        = 0.62
    mw_tint       = 0.0

    load_milkyway_skybox_from_image(MILKYWAY_IMAGE_PATH)
    bake_milkyway_skybox(math.radians(mw_tilt_deg), mw_hue, mw_intensity, mw_tint)
    bake_real_stars()

    fps_smooth = 30.0
    t_prev     = _clock.perf_counter()

    GUI_PANEL_FRAC   = 0.18
    CLICK_DRAG_TOL   = 0.004
    lmb_was_down     = False
    lmb_down_pos     = (0.0, 0.0)
    selected_star    = None

    while window.running:

        t_now  = _clock.perf_counter()
        dt     = t_now - t_prev
        t_prev = t_now
        sim_time += dt * 0.5 * flow_speed
        if dt > 0:
            fps_smooth = fps_smooth * 0.92 + (1.0 / dt) * 0.08

        camera.update(window)
        camera.update_transition(dt)

        if auto_orbit and not camera.transitioning:
            camera.azimuth += 0.005

        camera.upload()

        cur_pos  = window.get_cursor_pos()
        lmb_down = window.is_pressed(ti.ui.LMB)
        if lmb_down and not lmb_was_down:
            lmb_down_pos = cur_pos
        if (not lmb_down) and lmb_was_down:
            ddx = cur_pos[0] - lmb_down_pos[0]
            ddy = cur_pos[1] - lmb_down_pos[1]
            moved = math.hypot(ddx, ddy)
            if moved < CLICK_DRAG_TOL and lmb_down_pos[0] > GUI_PANEL_FRAC:
                picked = pick_star(lmb_down_pos[0], lmb_down_pos[1], camera.fov)
                if picked is not None:
                    selected_star = picked
        lmb_was_down = lmb_down

        if not orbit_paused and dt > 0:
            for pt in particles:
                pt.step(dt * orbit_time_warp)
        for i in range(MAX_TEST_PARTICLES):
            if i < len(particles):
                particles[i].upload_to(i)
                if not orbit_show_trail:
                    _particle_trailcnt_np[i] = 0
            else:
                _particle_active_np[i] = 0
                _particle_trailcnt_np[i] = 0
        sync_particle_fields()
        

        gui = window.get_gui()
        
        render_w = max(1, int(WIN_W * render_scale))
        render_h = max(1, int(WIN_H * render_scale))
        
        with gui.sub_window("Telemetry Dashboard", 0.0, 0.0, 0.18, 1.0):
            gui.text(f"FPS: {fps_smooth:.0f} (Internal Res: {render_w}x{render_h})")
            gui.text("")

            gui.text("--- SGR A* TELEMETRY ---")
            gui.text("Target: Sagittarius A*")
            gui.text(f"Mass: {SgrA_MASS_SOLAR/1e6:.3f} million M_sun")
            gui.text(f"r_g = GM/c²: {SgrA_RG_KM/1e6:.3f} million km")
            gui.text(f"r_s = 2GM/c²: {SgrA_RS_KM/1e6:.3f} million km")
            gui.text(f"r_g light-time: {SgrA_RG_LIGHT_SECONDS:.2f} s")
            gui.text(f"Spin parameter a*: {spin:.3f} (model)")
            r_plus = M + math.sqrt(max(M*M - spin*spin, 0.0))
            aa = abs(spin)
            z1 = 1.0 + max(1.0-aa*aa, 1e-12)**(1.0/3.0) * (
                (1.0+aa)**(1.0/3.0) + max(1.0-aa, 1e-12)**(1.0/3.0)
            )
            z2 = math.sqrt(3.0*aa*aa + z1*z1)
            r_isco_py = 3.0 + z2 - math.sqrt(
                max((3.0-z1)*(3.0+z1+2.0*z2), 0.0)
            )
            gui.text(f"Kerr horizon r+ = {r_plus:.3f} M")
            gui.text(f"Prograde ISCO = {r_isco_py:.3f} M")
            gui.text(f"ISCO scale: {r_isco_py*SgrA_RG_KM/1e6:.3f} million km")
            gui.text("")

            gui.text("--- CLICK-TO-IDENTIFY STAR ---")
            if selected_star is None:
                gui.text("Click any star in the sky to ID it.")
            else:
                s = selected_star
                gui.text(f"{s['name']}  (HR {s['hr']})")
                gui.text(f"RA {s['ra_h']:.4f}h   Dec {s['dec_deg']:+.4f} deg")
                gui.text(f"V mag: {s['vmag']:+.2f}")
                bv_txt = f"{s['bv']:+.2f}" if s['bv'] is not None else "N/A"
                gui.text(f"B-V color index: {bv_txt}")
                if s['temp_k'] is not None:
                    gui.text(f"Est. temperature: {s['temp_k']:,.0f} K")
                else:
                    gui.text("Est. temperature: N/A")
                gui.text(f"(picked {s['angle_deg']:.2f} deg off cursor)")
                if gui.button("Clear Selection"):
                    selected_star = None
            gui.text("")

            gui.text("--- SPECTRAL VIEWS ---")
            if gui.button("1. Hot Plasma"):
                hue, sat, val = 0.055, 0.72, 4.20
            if gui.button("2. Thermal IR"):
                hue, sat, val = 0.035, 0.48, 3.20
            if gui.button("3. X-Ray"):
                hue, sat, val = 0.575, 0.55, 4.60
            if gui.button("4. Radio / Sub-mm"):
                hue, sat, val = 0.62, 0.30, 2.20
            gui.text("")

            gui.text("=== SGR A* PHYSICS ===")
            spin       = gui.slider_float("Kerr Spin a*", spin, 0.0, 0.99)
            flow_speed = gui.slider_float("Flow Evolution", flow_speed, 0.0, 5.0)
            doppler_on = gui.slider_float("Relativistic Doppler", doppler_on, 0.0, 1.0)
            gui.text("")

            gui.text("=== CAMERA & NAVIGATION ===")
            camera.distance  = gui.slider_float("Distance", camera.distance, camera.min_dist, camera.max_dist)
            camera.elevation = gui.slider_float("Inclination", camera.elevation, -1.5, 1.5)
            camera.fov       = gui.slider_float("FOV", camera.fov, 0.2, 1.5)
            if gui.button("Toggle Cinematic Auto-Orbit"):
                auto_orbit = not auto_orbit
            gui.text("")

            gui.text("--- CAMERA BOOKMARKS (eased fly-to) ---")
            if camera.transitioning:
                gui.text("  ...flying to saved view (drag to cancel)")
            else:
                gui.text("  ")
            for bm in CAMERA_BOOKMARKS:
                if gui.button(bm["name"]):
                    camera.fly_to(bm, duration=2.2)
            gui.text("")

            gui.text("=== ARTISTIC PALETTE ===")
            hue = gui.slider_float("Hue", hue, 0.0, 1.0)
            sat = gui.slider_float("Saturation", sat, 0.0, 1.0)
            val = gui.slider_float("Glow/Value", val, 0.0, 10.0)
            gui.text("")

            gui.text("=== MILKY WAY BACKGROUND (from image) ===")
            gui.text("Sgr A* sits inside the Galactic disk, so the sky")
            gui.text("shows the Milky Way's own band, not a distant galaxy.")
            gui.text(os.path.basename(MILKYWAY_IMAGE_PATH))
            gui.text(f"+ {len(REAL_STARS)} real stars (Yale BSC5)")
            gui.text("  gravitationally lensed by the same ray-march")
            new_tilt = gui.slider_float("Galactic Plane Tilt", mw_tilt_deg, 0.0, 90.0)
            new_intensity = gui.slider_float("Milky Way Brightness", mw_intensity, 0.0, 2.0)
            new_tint = gui.slider_float("Tint Amount", mw_tint, 0.0, 1.0)
            new_hue = gui.slider_float("Tint Hue", mw_hue, 0.0, 1.0)
            if (new_tilt != mw_tilt_deg or new_intensity != mw_intensity
                    or new_hue != mw_hue or new_tint != mw_tint):
                mw_tilt_deg, mw_intensity, mw_hue, mw_tint = new_tilt, new_intensity, new_hue, new_tint
                bake_milkyway_skybox(math.radians(mw_tilt_deg), mw_hue, mw_intensity, mw_tint)
            if gui.button("Reload Milky Way Image"):
                load_milkyway_skybox_from_image(MILKYWAY_IMAGE_PATH)
                bake_milkyway_skybox(math.radians(mw_tilt_deg), mw_hue, mw_intensity, mw_tint)
            gui.text("")
            
            gui.text("=== RENDER ENGINE SPECS ===")
            render_scale = gui.slider_float("Render Scale", render_scale, 0.25, 1.0)
            exposure     = gui.slider_float("Exposure",  exposure,  0.1, 10.0)
            ms_f         = gui.slider_float("Ray Steps", float(max_steps), 48.0, 500.0)
            max_steps    = int(ms_f)
            bloom_strength = gui.slider_float("Bloom Strength", bloom_strength, 0.0, 1.5)
            gui.text("")

            gui.text("--- ORBITAL LAB: S301 + TEST PARTICLES (RK4+1PN) ---")
            gui.text(f"Central mass: {S301_M_BH_MSUN/1e6:.2f} million M_sun")
            gui.text(f"Active bodies: {len(particles)} / {MAX_TEST_PARTICLES}")
            orbit_time_warp = gui.slider_float("Time Warp (yr/s)", orbit_time_warp, 0.0, 3.0)
            if gui.button("Toggle Trail Visibility (all)"):
                orbit_show_trail = not orbit_show_trail
            if gui.button("Pause / Resume All"):
                orbit_paused = not orbit_paused
            gui.text("")

            s301 = particles[0]
            gui.text(f"[{s301.name}] a={s301.a_au:.0f} AU  e={s301.ecc:.3f}  "
                      f"P={s301.period_yr:.2f} yr")
            gui.text(f"  r={s301.distance_au():8.2f} AU   "
                      f"v={s301.speed_km_s():8.0f} km/s "
                      f"({s301.speed_km_s()/C_KM_S*100:.2f}% c)")
            gui.text(f"  t={s301.t_years:7.3f} yr  "
                      f"(orbit #{s301.t_years/s301.period_yr:5.2f})  "
                      f"precession={s301.precession_deg_per_orbit():.3f} deg/orbit")
            s301.inclination_deg = gui.slider_float(
                "S301 Inclination", s301.inclination_deg, 0.0, 90.0)
            if gui.button("Toggle S301 GR Precession"):
                s301.gr_enabled = not s301.gr_enabled
            if gui.button("Reset S301 to Periapsis"):
                particles[0] = TestParticleOrbit(
                    a_au=S301_A_AU, ecc=S301_ECC,
                    inclination_deg=s301.inclination_deg,
                    gr_enabled=s301.gr_enabled,
                    color=PARTICLE_PALETTE[0], name="S301"
                    # v_peri intentionally omitted -> derived via vis-viva,
                    # same reasoning as the initial construction above.
                )
            gui.text("")

            gui.text("--- COMPARISON TEST PARTICLES ---")
            remove_idx = None
            for i in range(1, len(particles)):
                pt = particles[i]
                gui.text(f"[{pt.name}] a={pt.a_au:.0f} AU e={pt.ecc:.3f} "
                          f"i={pt.inclination_deg:.0f}deg "
                          f"{'GR' if pt.gr_enabled else 'Newton'}")
                gui.text(f"  r={pt.distance_au():7.1f} AU  "
                          f"P={pt.period_yr:5.2f} yr  "
                          f"precession={pt.precession_deg_per_orbit():.3f} deg/orbit")
                if gui.button(f"Remove {pt.name}"):
                    remove_idx = i
            if remove_idx is not None:
                del particles[remove_idx]
            if len(particles) > 1 and gui.button("Remove All Test Particles"):
                particles = [particles[0]]
            gui.text("")

            gui.text("--- SPAWN NEW TEST PARTICLE ---")
            if len(particles) >= MAX_TEST_PARTICLES:
                gui.text("Slot limit reached — remove one to add another.")
            else:
                spawn_a_au = gui.slider_float("New: Semi-major Axis (AU)", spawn_a_au, 20.0, 1300.0)
                spawn_ecc  = gui.slider_float("New: Eccentricity", spawn_ecc, 0.0, 0.99)
                spawn_incl = gui.slider_float("New: Inclination (deg)", spawn_incl, 0.0, 90.0)
                if gui.button(f"GR Precession: {'ON' if spawn_gr_on else 'OFF'} (toggle)"):
                    spawn_gr_on = not spawn_gr_on
                if gui.button("Spawn Test Particle"):
                    particle_spawn_count += 1
                    color = PARTICLE_PALETTE[len(particles) % len(PARTICLE_PALETTE)]
                    particles.append(TestParticleOrbit(
                        a_au=spawn_a_au, ecc=spawn_ecc,
                        inclination_deg=spawn_incl, gr_enabled=spawn_gr_on,
                        color=color, name=f"TP{particle_spawn_count}"
                    ))

        render_frame(
            spin, exposure, max_steps,
            sim_time, camera.fov,
            hue, sat, val, doppler_on,
            render_w, render_h
        )

        upscale_bilinear(render_w, render_h)

        draw_particles(camera.fov, spin, 5.5)

        if bloom_strength > 0.0:
            apply_bloom(bloom_strength)

        canvas.set_image(pixels_display)
        window.show()


if __name__ == "__main__":
    main()
