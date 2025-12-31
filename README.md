# cat-motion

Motion-enabled Raspberry Pi pipeline that captures clips, runs cat detection/recognition, and serves a phone-friendly dashboard.

## 1. Components

1. **Motion daemon** – native `motion` service streams the camera and records MP4s when motion is detected.  
2. **Collector** – watches Motion’s target directory, waits for files to finish writing, copies clips into `data/clips/`, and add JSON sidecars.  
3. **Processor** – runs YOLO detection + embedding recognition, trims clips, writes summaries, routes output to recognized/unknown folders, and exports unlabeled crops.  
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
    ├── unlabeled/
    └── training/
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

- Copy `cat_motion.yml`, edit to match your Pi:  
  - `recorder.motion_drop_dir` = Motion `target_dir`.  
  - `detection.model` = path to YOLO ONNX.  
  - `recognition.embeddings/labels` = outputs created by `cat-motion-train`.  
  - `processing` settings = frame stride, detection interval, trimming, unlabeled export behavior.  
  - `web.stream_url` (optional) – leave empty to auto-build a URL using the dashboard host + `web.stream_port`.  
  - `auth` block – provide SMTP credentials and list the `allowed_emails` that can log in.  
- Point `CAT_MOTION_CONFIG` at your YAML file (or use `--config` on every CLI).

## 5. Run the stack

1. **Install** (reuse Pi system packages such as `opencv`/`picamera`)  
   ```bash
   python -m venv .venv --system-site-packages
   source .venv/bin/activate
   pip install -e .
   ```
2. **Collector** – `cat-motion-collector --config cat_motion.yml` (runs forever; add a systemd unit later).  
3. **Web UI** – `cat-motion-web --config cat_motion.yml`; visit from phone/tablet.  
4. **Processor** – run `cat-motion-processor` manually or tap “Process Clips” in the UI.  
5. **Training loop** – label unlabeled clips in the UI → `cat-motion-train --config cat_motion.yml` → rerun processor.

### Optional: run the processor automatically (systemd timer)

Sample unit definitions live in `systemd/`. To install on the Pi:

1. Create the service file:

   ```bash
   sudo nano /etc/systemd/system/cat-motion-processor.service
   ```

   Paste the contents of `systemd/cat-motion-processor.service` (update paths), save, exit.

2. Create the timer:

   ```bash
   sudo nano /etc/systemd/system/cat-motion-processor.timer
   ```

   Paste `systemd/cat-motion-processor.timer`, save, exit.

3. Enable:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now cat-motion-processor.timer
   ```

The timer wakes up a few minutes after boot and then every 5 minutes to run `cat-motion-processor --config cat_motion.yml`. Check status/logs via `systemctl status cat-motion-processor.timer` or `journalctl -u cat-motion-processor.service`.

### Running everything in the background on a Pi

Use systemd services so collector + web UI start on boot (and combine with the processor timer above):

1. **Collector service** – edit directly with nano:

   ```bash
   sudo nano /etc/systemd/system/cat-motion-collector.service
   ```

   Paste:

   ```
   [Unit]
   Description=Cat Motion collector
   After=network.target motion.service

   [Service]
   Type=simple
   WorkingDirectory=/home/pi/cat-motion
   ExecStart=/home/pi/cat-motion/.venv/bin/cat-motion-collector --config /home/pi/cat-motion/cat_motion.yml
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

   Save (Ctrl+O) and exit (Ctrl+X).

2. **Web service**:

   ```bash
   sudo nano /etc/systemd/system/cat-motion-web.service
   ```

   Paste:

   ```
   [Unit]
   Description=Cat Motion web UI
   After=network.target

   [Service]
   WorkingDirectory=/home/pi/cat-motion
   ExecStart=/home/pi/cat-motion/.venv/bin/cat-motion-web --config /home/pi/cat-motion/cat_motion.yml --host 0.0.0.0 --port 8000
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

   Save/exit.

3. Enable everything:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now cat-motion-collector.service
   sudo systemctl enable --now cat-motion-web.service
   sudo systemctl enable --now cat-motion-processor.timer
   ```

Use `journalctl -u <unit>` or `systemctl status <unit>` to monitor logs. Adjust the paths if your checkout lives somewhere else.

## 6. Self-training workflow

- Recognized clips automatically promote high-confidence face crops into `data/training/<label>/`.  
- Unknown clips write per-second crops into `data/unlabeled/<clip>/`.  
- In the UI, assign unlabeled folders to a label or delete them if they’re junk.  
- When you run `cat-motion-train`, it now prunes near-duplicate crops and enforces a 1,000-image cap per label (oldest photos deleted first) before rebuilding embeddings. Recognition is disabled until embeddings/labels exist.

## 7. Web interface

Features:
- Live stream embed (MJPEG from Motion).  
- Recognized/unknown clip cards with file size + JSON summaries.  
- Clips page to assign labels, download, or delete recordings.  
- Unlabeled page to preview each extracted image, assign labels individually, or delete them.  
- Training page to preview and prune training images.  
- Mobile-friendly unlabeled triage forms with assign/delete actions.  
- Buttons to trigger processing and refresh configuration remotely.
- Email-based login: every page/API requires a signed session cookie; request a magic link from the login page and only addresses in `auth.allowed_emails` receive it.

## 8. Authentication flow

1. Configure `auth.mail` with an SMTP server that can deliver outbound mail from the Pi (Gmail app passwords work, or a local postfix relay).  
2. Add every allowed operator email to `auth.allowed_emails`. Unknown addresses are rejected before an email is sent.  
3. When you browse to the dashboard you’ll see the login form. Enter an allowed email → check your inbox for the one-time link → click it to establish a cookie-backed session (default lifetime: 7 days). The login link is built from the host you used to reach the login page, so no extra config is required.  
4. Use the “Log out” button in the header to clear the cookie immediately on any device.  
5. Restarting the web service rotates the secret automatically, which invalidates existing sessions—log in again via email.

## 9. CLI reference

- `cat-motion-collector` – ingest Motion clips into `data/clips/`.  
- `cat-motion-processor` – detection + recognition pass with trimming + exports.  
- `cat-motion-web` – FastAPI dashboard.  
- `cat-motion-train` – rebuild embedding centroids + label map from `data/training/`.  
- `CAT_MOTION_CONFIG=/path/to/config.yml` – override config path for any command.
