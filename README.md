# Sagittarius A* — Real-Time Ray-Marched Black Hole Simulator

A GPU-accelerated, real-time renderer that simulates the appearance and physics of **Sagittarius A\***, the supermassive black hole at the center of the Milky Way, complete with gravitational lensing, an orbiting star (S2/S301-class), a real starry sky, and interactive relativistic physics controls.

Built with [Taichi](https://www.taichi-lang.org/) for cross-platform GPU compute (Vulkan / OpenGL / CPU), the simulator runs a full ray-marcher per pixel that bends light around a spinning (Kerr) black hole, renders a volumetric accretion disk, and integrates a real orbital-mechanics model side-by-side in the same 3D scene — all interactively, at real-time frame rates.

---

## ✨ Highlights

- **Physically-motivated Kerr black hole**: horizon and ISCO radii computed from the Kerr metric, reduced-order gravitational lensing, relativistic Doppler beaming, and frame-dragging (Lense–Thirring precession).
- **Real astrophysical scale**: mass is calibrated to Sagittarius A*'s measured mass of **4.154 × 10⁶ M☉**, with full unit conversion between geometric (G = c = 1) render units and physical units (km, seconds, solar masses).
- **Independent orbital-mechanics lab**: a test-particle integrator (RK4 + 1PN relativistic precession) simulates a star (modeled on the S2/S0-2-class "S301" orbit: a = 687 AU, e = 0.983, P ≈ 8.69 yr) around the black hole in its own physical unit system, then projects it into the render scene — so what you see orbiting the hole is a real, independently-verified Keplerian/relativistic solution, not a scripted animation.
- **Volumetric, procedural accretion disk**: animated turbulent plasma filaments (fBm noise), Doppler-shifted color and brightness, and a photon-ring brightness boost near the shadow edge.
- **Real sky**: an equirectangular Milky Way image is used as the skybox background (tiltable relative to the black hole's spin axis), overlaid with real stars from the **Yale Bright Star Catalogue (BSC5)** at their true J2000 coordinates, colored by their actual B–V color index / blackbody temperature, and gravitationally lensed by the same ray-marcher as everything else in the scene.
- **Cinematic render pipeline**: adaptive-step ray marching, bilinear upscaling from a lower internal compute resolution for performance, a separable bright-pass + blur bloom pass, and ACES filmic tone mapping for an HDR, cinema-grade look.
- **Fully interactive GUI**: live sliders/buttons for spin, exposure, render quality, camera, color palette, Milky Way appearance, and the orbital lab (spawn test particles, toggle GR precession, compare orbits).
- **Multi-backend support**: runs on Vulkan, OpenGL, or pure CPU, so it works across discrete GPUs, integrated GPUs (e.g. Intel Iris Xe/UHD), and machines with no GPU at all.

---

## 🧠 The Physics

| Component | What's simulated |
|---|---|
| **Horizon & ISCO** | Kerr outer horizon `r₊ = M + √(M² − a²)` and innermost stable circular orbit, both as functions of spin `a*` |
| **Light bending** | Reduced-order Kerr gravitational lensing applied per-ray during the march, plus a closed-form photon-shadow half-angle and secondary (lensed) image approximation for point objects like the orbiting star |
| **Frame dragging** | Lense–Thirring–style angular drag term added to the geodesic-like ray/particle acceleration when spin `a > 0` |
| **Doppler beaming** | Disk emission color/brightness shifted based on the local orbital velocity relative to the camera ray |
| **Orbital dynamics** | RK4 integration of a test particle around a 4.3 × 10⁶ M☉ central mass in AU/year/M☉ units, with optional 1PN relativistic apsidal precession, vis-viva-derived periapsis velocity (so the labeled orbital elements a, e, and P are always internally self-consistent), and live telemetry (distance, speed as % of c, precession per orbit) |
| **Tone mapping** | ACES filmic curve applied after physically-motivated exposure, so bright disk/photon-ring regions roll off naturally instead of clipping |

The simulation deliberately keeps two unit systems in sync: the renderer's geometric units (`G = c = M = 1`) for the ray-marched black hole, and a real AU/year/M☉ system for the orbiting star — bridged by an explicit length conversion (`AU_TO_CODE`) — so the physics can be checked and trusted independently of how it's drawn.

---

## 🖥️ Tech Stack

- **Python 3** + **[Taichi](https://www.taichi-lang.org/)** — JIT-compiled GPU kernels for the ray marcher, skybox baking, bloom, and particle rendering
- **NumPy** — star catalogue math, buffer staging between Python and Taichi fields
- **Taichi GGUI** — real-time window, canvas, and immediate-mode GUI controls
- Backends: **Vulkan** (default, best for integrated GPUs), **OpenGL** (fallback), **CPU** (universal)

---

## 🚀 Getting Started

### Requirements
```bash
pip install taichi numpy
```

### Run it
```bash
python sim.py                # Vulkan (recommended, best for Intel iGPUs)
python sim.py --opengl       # OpenGL fallback
python sim.py --cpu          # CPU — slowest, but works everywhere
```

Place a Milky Way equirectangular image named `milkyway.jpg` next to the script (or set the `MILKYWAY_IMAGE_PATH` environment variable) to enable the real sky background.

### Controls

| Input | Action |
|---|---|
| Left-drag | Orbit camera around the black hole |
| Right-drag | Zoom in / out |
| `W` / `S` | Zoom in / out (keyboard) |
| `A` / `D` | Orbit left / right (keyboard) |
| GUI panel | Spin, exposure, render quality, camera bookmarks, palette, Milky Way, orbital lab |

---

## 🪐 The Orbital Lab

Alongside the visual renderer, the simulation includes a small self-contained orbital-mechanics sandbox:

- The default body, **S301**, reproduces a highly eccentric star orbit around Sgr A* (semi-major axis 687 AU, eccentricity 0.983, ~8.7-year period) — the same kind of orbit that historically provided direct dynamical proof of a supermassive compact object at the Galactic Center.
- Users can spawn up to 6 additional comparison test particles with custom semi-major axis, eccentricity, and inclination, and toggle 1PN relativistic precession on/off per body to visually compare Newtonian vs. relativistic orbits in real time.
- All orbital elements are Kepler/vis-viva self-consistent by construction — the periapsis velocity is *derived* from `(a, e, GM)` rather than hardcoded, avoiding the classic "slightly-off velocity → wildly different actual orbit" trap of naive N-body demos.

---

## 📁 Project Structure

```
sim.py     # Single-file simulation: physics, rendering, GUI, and orbital lab
milkyway.jpg         # (user-provided) equirectangular Milky Way background image
```

## 🔭 Why This Project

Most "black hole visualizers" are either pure Hollywood-style shaders with no real physics, or offline scientific ray-tracers (like the tools used for *Interstellar*) that take minutes per frame and aren't interactive. This project sits in between: it keeps genuine general-relativistic quantities (horizon, ISCO, spin-dependent lensing, frame dragging, relativistic precession) driving what's on screen, while still rendering interactively on consumer and integrated GPUs — turning a normally research-grade or VFX-grade problem into something anyone can explore live with a mouse and a slider.


## 📜 Credits & Data Sources

- Star positions/magnitudes/colors: **Yale Bright Star Catalogue (BSC5)**, Hoffleit & Warren (1991)
- Physical constants: CODATA / IAU standard values (G, c, M☉)
- Sgr A* mass: adopted value of 4.154 × 10⁶ M☉ from published measurements of the Galactic Center
