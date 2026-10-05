# Deploying PARIKSHAK — backend on Railway, frontend on Vercel

```
 Browser (HTTPS)                     Vercel  (static: frontend/)                 Railway  (Docker: FastAPI)
 ───────────────  GET /, /console ─▶  index.html, console.html, static/   
                  POST /api/frame ─▶  rewrite /api/*  ───────────────────────▶  backend/server.py
                                      rewrite /recordings, /reports, /tags ─▶   MediaPipe + YOLO + AprilTag + Groq
```

The browser only ever talks to the Vercel domain. Vercel forwards `/api/*`, `/recordings/*`,
`/reports/*` and `/tags` to Railway (a *rewrite* = server-side proxy), so there is no CORS and
no mixed-content problem, and the frontend code needs no changes.

> Shortcut: Railway alone also serves the whole site (`https://<backend>.up.railway.app/` and
> `/console`). Use that URL if you need the site live before Vercel is set up.

---

## 0. Before you start

1. **Rotate the Groq key.** The old key was pasted in chat — create a new one at
   console.groq.com → API Keys, and delete the old one. It goes into Railway as a variable
   (step 2), never into git. `.env` is git-ignored.
2. Code is on GitHub: `PriyamMishra853/Pari`, branch `main`.
3. Accounts: railway.com (Hobby plan recommended — see "Memory" below) and vercel.com, both
   signed in with GitHub.

## 1. Backend on Railway

1. railway.com → **New Project → Deploy from GitHub repo → PriyamMishra853/Pari**.
2. Railway finds `railway.json` + `Dockerfile` and builds the image (first build ≈ 8–15 min:
   CPU PyTorch, MediaPipe, OpenCV). No build settings to change.
3. Service → **Variables** → add:

   | Variable | Value | Required |
   |---|---|---|
   | `GROQ_API_KEY` | your new key `gsk_…` | yes (else the copilot uses local rules) |
   | `GROQ_MODEL` | `openai/gpt-oss-120b` | optional (default) |
   | `GROQ_FALLBACK_MODEL` | `openai/gpt-oss-20b` | optional |
   | `GROQ_STT_MODEL` | `whisper-large-v3-turbo` | optional (push-to-talk voice) |
   | `PARIKSHAK_DATA_DIR` | `/data` | only with a volume (step 5) |

   Do **not** set `PORT` — Railway injects it; the Dockerfile binds `0.0.0.0:$PORT`.
4. Service → **Settings → Networking → Generate Domain**. You get
   `https://<something>.up.railway.app`. Copy it.
5. (Recommended) **Persistent storage:** Service → **Volumes → New Volume**, mount path `/data`,
   then set `PARIKSHAK_DATA_DIR=/data`. Without it, recordings, reports, flight logs, datasets
   and custom experiments are deleted on every redeploy.
6. Check it: open `https://<backend>.up.railway.app/api/system` → JSON with
   `"pose3d": {"status": "RUNNING"}` and `"groq": {"available": true}`. Then open
   `https://<backend>.up.railway.app/console`.

## 2. Frontend on Vercel

1. Edit **`vercel.json`** in the repo: replace all four `https://YOUR-BACKEND.up.railway.app`
   with your Railway domain (no trailing slash). Commit and push:
   ```bash
   git add vercel.json
   git commit -m "deploy: point Vercel rewrites at Railway"
   git push
   ```
2. vercel.com → **Add New → Project → Import** `PriyamMishra853/Pari`.
3. In the import screen: **Framework Preset: Other**, **Root Directory: `./`** (repo root).
   Leave build/output settings empty — `vercel.json` sets `outputDirectory: frontend` and no build.
4. Environment variables: **none** needed on Vercel (all secrets live on Railway).
5. **Deploy.** Open `https://<project>.vercel.app/` (landing) and `/console`.
6. Every later `git push` redeploys both services automatically.

## 3. Smoke test after deploy

| Check | Expected |
|---|---|
| `https://<vercel>/` | landing page, scroll → ship lands on the moon |
| `https://<vercel>/console` | console loads, experiment list has 10 items |
| Top bar pill **AI** | `AI GROQ` (green). `AI LOCAL` = `GROQ_API_KEY` missing/wrong |
| **Start camera** | browser asks for camera; skeleton + mesh appear over you |
| **System** tab | pose backend RUNNING, SAM 3D Body MODEL UNAVAILABLE (expected, no GPU) |
| `https://<vercel>/tags` | AprilTag sheet (served from Railway) |
| **What's next** button | spoken guidance, source tag `Groq · openai/gpt-oss-120b` |

## 4. Environment variables — full list

| Where | Variable | Purpose |
|---|---|---|
| Railway | `GROQ_API_KEY` | Groq reasoning + Whisper speech-to-text |
| Railway | `GROQ_MODEL` / `GROQ_FALLBACK_MODEL` | chat models (defaults shown above) |
| Railway | `GROQ_STT_MODEL` | voice-command transcription model |
| Railway | `PARIKSHAK_DATA_DIR` | volume path for recordings/reports/logs/datasets/custom experiments |
| Railway | `PORT` | injected by Railway — do not set |
| Local only | `.env` | same `GROQ_*` keys for `python -m backend.server` (see `.env.example`) |
| Vercel | — | none; the backend URL lives in `vercel.json` rewrites |

## 5. Problems you can hit (and the fix)

| Symptom | Cause | Fix |
|---|---|---|
| Railway build fails at `pip install mediapipe` / torch | wheel missing for the image's Python/CPU | keep `python:3.11-slim` (tested 3.11); re-run the build (PyTorch index occasionally times out) |
| Deploy "unhealthy", logs stop at model loading | models take 30–60 s to load; or out of memory | healthcheck timeout is already 300 s; check **Metrics → Memory** |
| Process killed / restarts, `Killed` in logs | RAM cap: MediaPipe + YOLO + torch need ~1–1.5 GB | use a plan with ≥2 GB RAM (Railway trial limits are too small) |
| `ImportError: libGL.so.1` | OpenCV GUI build without system libs | Dockerfile installs `libgl1 libglib2.0-0`; don't remove them |
| `module 'cv2' has no attribute 'aruco'` → rack never locks | `opencv-python` overwrote the contrib build | Dockerfile reinstalls `opencv-contrib-python` last; keep that line |
| Console loads but every button fails (404/502 on `/api/...`) | `vercel.json` still has `YOUR-BACKEND` or a wrong domain | fix the 4 URLs, push, wait for the Vercel redeploy |
| Video upload fails for big files through Vercel | Vercel proxy body limit ≈ 4.5 MB | upload large videos on the Railway URL (`https://<backend>/console`), or trim the clip |
| Camera / mic never asks for permission | page not on HTTPS, or opened inside an in-app browser | use the `https://` Vercel/Railway URL in Chrome/Edge |
| Hands-free voice does nothing | Web Speech API exists only in Chrome/Edge and needs internet | use **Hold to talk** (Groq Whisper) — works in all browsers |
| `AI LOCAL` pill, copilot replies tagged `local-rules` | `GROQ_API_KEY` missing/invalid, or Groq timed out (>4 s) | set/rotate the key on Railway; a slow network falls back to local rules by design |
| `model_not_found` from Groq | Groq retired a model | set `GROQ_MODEL` to a current model from console.groq.com/docs/models |
| Two people use the site at once and steps jump | the server holds **one** live session (one operator, one rack) | demo with one browser tab at a time; reset the procedure before your run |
| Recordings/logs vanish after redeploy | no volume | add the volume + `PARIKSHAK_DATA_DIR` (section 1, step 5) |
| Low FPS online (1–3 fps) vs local | every frame travels browser → Vercel → Railway on shared CPU | expected on a shared CPU; for the demo video record against the **local** server |
| Rack frame never LOCKED | no AprilTag in view, tag too small/blurred, or wrong tag size | print tag 10 from `/tags`, keep it near the middle of the view, set **Tag size** in System tab |

## 6. Run locally (unchanged)

```bash
.venv/Scripts/python.exe -m backend.server --port 8766
```
Open http://localhost:8766/console.
