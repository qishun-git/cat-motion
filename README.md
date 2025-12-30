# cat-motion

Motion-enabled Raspberry Pi pipeline that captures clips, runs cat detection/recognition, and serves a phone-friendly dashboard.

## 1. Components

1. **Motion daemon** – native `motion` service streams the camera and records MP4s when motion is detected.  
2. **Collector** – watches Motion’s target directory, waits for files to finish writing, copies clips into `data/clips/`, and add JSON sidecars.  
3. **Processor** – runs YOLO detection + embedding recognition, trims clips, writes summaries, routes output to recognized/unknown folders, exports unlabeled crops, and can write compressed proxies.  
4. **Web UI** – FastAPI app that shows live stream, clip lists, unlabeled triage cards, and buttons to trigger processing/refresh.  
5. **Trainer** – CLI that rebuilds embeddings/labels from `data/training/` so the Pi can learn new cats locally.

## 2. Directory layout

```
cat-motion/
├── motion_config/          # Motion daemon templates
├── configs/                # YAML runtime config
├── collector/              # ingestion service
├── processor/              # detection + recognition
├── web/                    # FastAPI app + templates/static
├── models/                 # embeddings + labels
└── data/
    ├── clips/
    ├── recognized_clips/
    ├── unknown_clips/
    ├── compressed_clips/
    ├── unlabeled/
    ├── training/
    └── reject/
```

All Python modules live inside this repo; no external project imports are required.

## 3. Motion daemon configuration (Pi)

1. Install Motion:
   ```bash
   sudo apt install motion
   sudo systemctl disable motion
   ```
2. Copy `motion_config/motion.conf.example` to `/etc/motion/motion.conf` and adjust:
   - `daemon on`
   - `width`/`height`/`framerate` for your Pi camera
   - `stream_port`, `webcontrol_port`
   - `target_dir /var/lib/motion-clips`
   - `movie_filename %Y%m%d-%H%M%S`
   - `movie_codec mp4`, `movie_quality` or `ffmpeg_bps`
   - detection tuning (`threshold`, `noise_level`, masks)
3. Enable and start:
   ```bash
   sudo systemctl enable motion
   sudo systemctl start motion
   ```
Document these values because the collector relies on them.

## 4. Configure cat-motion

- Copy `configs/cat_motion.yml`, edit to match your Pi:  
  - `recorder.motion_drop_dir` = Motion `target_dir`.  
  - `detection.model` = path to YOLO ONNX.  
  - `recognition.embeddings/labels` = outputs created by `cat-motion-train`.  
  - `processing` settings = frame stride, detection interval, trimming, unlabeled export behavior.  
- Point `CAT_MOTION_CONFIG` at your YAML file (or use `--config` on every CLI).

## 5. Run the stack

1. **Install**  
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .
   ```
2. **Collector** – `cat-motion-collector --config configs/cat_motion.yml` (runs forever; add a systemd unit later).  
3. **Web UI** – `cat-motion-web serve --config configs/cat_motion.yml`; visit from phone/tablet.  
4. **Processor** – run `cat-motion-processor` manually or tap “Process Clips” in the UI.  
5. **Training loop** – label unlabeled clips in the UI → `cat-motion-train --config configs/cat_motion.yml` → rerun processor.

## 6. Self-training workflow

- Recognized clips automatically promote high-confidence face crops into `data/training/<label>/`.  
- Unknown clips write per-second crops into `data/unlabeled/<clip>/`.  
- In the UI, assign unlabeled folders to a label (moves frames into training) or reject them (moves to `data/reject/`).  
- After adding training images, run `cat-motion-train` to regenerate embeddings/labels; recognition is disabled until those files exist.

## 7. Web interface

Features:
- Live stream embed (MJPEG from Motion).  
- Recognized/unknown clip cards with file size + JSON summaries.  
- Mobile-friendly unlabeled triage forms with assign/reject actions.  
- Buttons to trigger processing and refresh configuration remotely.

## 8. CLI reference

- `cat-motion-collector` – ingest Motion clips into `data/clips/`.  
- `cat-motion-processor` – detection + recognition pass, trimming, exports, compression.  
- `cat-motion-web serve` – FastAPI dashboard.  
- `cat-motion-train` – rebuild embedding centroids + label map from `data/training/`.  
- `CAT_MOTION_CONFIG=/path/to/config.yml` – override config path for any command.
