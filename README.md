# Camera Stability Diff Tools

This workspace contains two main entry points:

- `diff_analyzer.py`: CLI for camera capture, frame-difference analysis, and offline image analysis.
- `diff_gui_qt5.py`: Qt5 GUI wrapper for the same workflow.

## What the tools do

- Lock key camera parameters before capture.
- Capture a fixed number of frames.
- Save the frames and generate a difference report.
- Analyze an existing folder of images in offline mode.
- Show the current camera parameter snapshot in the GUI.

## Recommended defaults

- Exposure: `10000 us`
- Gain: `20`
- Frame rate: camera maximum if available
- Capture count: `20`
- Prewarm discard frames: `5`

## Environment notes

This machine uses the system OpenCV / PyQt5 packages together with the project `.venv`.

The current working combination is:

- `.venv` Python
- system `cv2` from `/usr/lib/python3/dist-packages`
- system `PyQt5` from `/usr/lib/python3/dist-packages`
- `numpy 1.26.x`

If you see `ModuleNotFoundError: No module named 'cv2'`, run the scripts with the project `.venv` after the code has prepended the system `dist-packages` path.

## Run GUI

```bash
cd /home/nvidia/zangjiayu/carmera/Python
/home/nvidia/zangjiayu/carmera/Python/.venv/bin/python diff_gui_qt5.py
```

## Run CLI capture

```bash
cd /home/nvidia/zangjiayu/carmera/Python
/home/nvidia/zangjiayu/carmera/Python/.venv/bin/python diff_analyzer.py
```

## Run offline analysis

```bash
cd /home/nvidia/zangjiayu/carmera/Python
/home/nvidia/zangjiayu/carmera/Python/.venv/bin/python diff_analyzer.py --input-dir ./capture_data --output-report ./difference.txt
```

## Suggested capture workflow

1. Open the GUI.
2. Scan devices.
3. Confirm the parameter snapshot panel.
4. Start capture and wait for 20 frames.
5. Review the generated report in `difference.txt`.

## Camera parameters exposed in the GUI

Editable controls:

- Device index
- Capture count
- Prewarm discard frames
- Timeout
- Exposure time
- Gain
- Manual frame rate override
- Trigger mode toggle
- Output directory
- Report path
- Offline image directory

Read-only snapshot fields:

- Width
- Height
- OffsetX
- OffsetY
- PayloadSize
- GevSCPSPacketSize
- ExposureTime
- Gain
- AcquisitionFrameRate
- ResultingFrameRate
- AcquisitionFrameRateEnable
- GammaEnable
- PixelFormat
- TriggerMode
- TriggerSource
- BalanceWhiteAuto
- ExposureAuto
- GainAuto
- SharpnessMode

## Files of interest

- `diff_analyzer.py`
- `diff_gui_qt5.py`
- `difference.txt`
- `capture_data/`
- `AreaScanCamera/MultipleCameras/MultipleCameras.py`

## Notes

- The GUI and CLI are designed to work with the Hikvision / MVS Python SDK bindings in `MvImport/`.
- The capture path now prefers SDK BGR retrieval for stability and falls back to raw conversion if needed.
- If camera-side conversion still fails, check the pixel format in the snapshot panel and the report log.
