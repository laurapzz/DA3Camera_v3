# DA3Camera v3 — On-device bike-lane obstacle detection for Android

[![Android](https://img.shields.io/badge/Android-9.0%2B%20(API%2028)-brightgreen.svg)](https://developer.android.com/)
[![Kotlin](https://img.shields.io/badge/Kotlin-Android-blue.svg)](https://kotlinlang.org/)
[![ExecuTorch](https://img.shields.io/badge/ExecuTorch-1.5.1-ee4c2c.svg)](https://pytorch.org/executorch/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.31.1-005fed.svg)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-5C3EE8.svg)](https://opencv.org/)

DA3Camera v3 (shown on the phone as **CameraDepthLane v3**) is an Android app. It takes a single photo of a street,
finds the **bike lane** in it, works out the **3D shape of the scene** from that one image,
and **detects obstacles inside the bike lane**. For each obstacle it reports **how far away it is and how tall it is, in meters**.

Everything runs **on the phone**. There is no server and no network access, and the app writes no output files.

```
 photo ──► depth network (DA3) ──► metric depth map ─┐
   │                                                 ├─► 3D point cloud ─► ground plane (RANSAC)
   └────► lane network (UFLDv2) ─► bike-lane mask ───┘                         │
                                                                               ▼
                                         obstacles = lane points that stick out above the ground
                                                                               │
                                                                               ▼
                                         photo with red boxes:  "Dist: 4.2m | H: 1.1m"
```

---

## Table of contents

1. [What problem does it solve?](#1-what-problem-does-it-solve)
2. [Key concepts (short primer)](#2-key-concepts-short-primer)
3. [What the user sees](#3-what-the-user-sees)
4. [How the pipeline works, step by step](#4-how-the-pipeline-works-step-by-step)
5. [Models](#5-models)
6. [Code tour](#6-code-tour)
7. [Building and running](#7-building-and-running)
8. [Camera intrinsics: where they come from](#8-camera-intrinsics-where-they-come-from)
9. [Timing and logging](#9-timing-and-logging)
10. [Tunable parameters](#10-tunable-parameters)
11. [Known limitations and caveats](#11-known-limitations-and-caveats)
12. [Differences from v2](#12-differences-from-v2)

---

## 1. What problem does it solve?

The goal is to check, using only a phone camera, whether a bike lane ahead is clear. If it is not,
the app shows where the obstacles are, how far away they are and how big they are.

A normal photo is 2D, so it cannot tell a parked scooter 5 m away from a painted symbol on the
asphalt. To separate the two, the app:

1. Estimates **depth** (distance from the camera) for every pixel, using a neural network.
2. Converts the depth into a **3D point cloud** using the camera's optics (its *intrinsics*).
3. Finds the **road surface** as a flat plane in 3D.
4. Treats anything inside the bike lane that rises **well above that plane** as an obstacle.

Restricting the search to the bike lane (found by a second neural network) avoids reporting
cars in other lanes, buildings, trees and so on.

## 2. Key concepts (short primer)

| Term | Meaning in this project |
|---|---|
| **Monocular depth estimation** | Predicting per-pixel depth from a *single* image (no stereo, no LiDAR). Here this is done by the **DA3** model (a Depth-Anything-style network). |
| **Metric depth** | Depth in real meters, not just "nearer / farther". DA3 outputs a depth that is converted to meters using the camera focal length (see step 5 below). |
| **Camera intrinsics** (`fx, fy, cx, cy`) | The focal length in pixels (`fx`, `fy`) and the optical center (`cx`, `cy`). They are needed to convert a pixel plus a depth into a 3D point, and to get the metric scale. |
| **Back-projection** | Converting pixel `(u, v)` with depth `Z` into 3D: `X = -(u-cx)·Z/fx`, `Y = -(v-cy)·Z/fy`, `Z = Z`. |
| **UFLDv2** | *Ultra-Fast Lane Detection v2*, a lane-line detector. It predicts where lane lines cross a fixed set of image rows ("row anchors"). It was trained on the **TuSimple** highway dataset. |
| **Ego lane** | The lane directly in front of the camera, bounded by the "ego-left" and "ego-right" lines. Here it is treated as the bike lane. |
| **ROI mask** | *Region of interest*: a black/white image where white = "inside the bike lane". |
| **RANSAC** | A robust fitting algorithm. It repeatedly picks 3 random 3D points, builds a plane from them, counts how many other points lie close to that plane, and keeps the best plane. This finds the road surface even when obstacles are present. |
| **Connected components** | Grouping neighbouring "obstacle" pixels into separate blobs, one per object. |
| **ExecuTorch / `.pte`** | PyTorch's on-device runtime and its model file format. Used for DA3. |
| **ONNX Runtime / `.onnx`** | Microsoft's cross-platform inference runtime. Used for UFLDv2. |

## 3. What the user sees

The app has a single main screen ([activity_main.xml](app/src/main/res/layout/activity_main.xml)), from top to bottom:

- **Take Photo**: opens an in-app camera ([CameraActivity](app/src/main/java/com/example/da3camera/CameraActivity.kt)) with **Capture** and **Back** buttons.
- **Choose from Gallery**: picks any existing image.
- A **status line** and a **progress bar**.
- The **image preview**. When processing finishes it shows the original photo with **red boxes** around each obstacle, labelled `Dist: X.Xm | H: Y.Ym`.
- A **result line**, for example "2 objects detected in the bike lane", plus the scene class predicted by DA3 (for example "Scene: unidirectional bicycle lane").
- An **Execution time** table listing every pipeline step in milliseconds.
- An **App startup** table showing how long model loading took.
- A **Camera intrinsics** section (`fx`, `fy`, `cx`, `cy` fields plus **Import Intrinsics from .txt**), used for gallery images.

### Typical flow

1. Open the app. Models start loading in the background right away. Status shows *Loading models…*, then *Ready*.
2. **Take a photo.** Processing starts automatically. The phone's own lens calibration is used for the intrinsics.
3. **Or pick a gallery image.** If intrinsics are already filled in (they are remembered between sessions), processing starts right away.
   If not, the image waits. Type the four values and press **Done** on the keyboard, or import a `.txt` file, and processing then starts automatically.
4. Read the result image and the timing breakdown.

There is no "Process" button and nothing is saved. Each run is: input image → final detection image.

## 4. How the pipeline works, step by step

The main logic is in `runDetection()` in
[MainActivity.kt](app/src/main/java/com/example/da3camera/MainActivity.kt). It runs on a
single background worker thread, and each step is wrapped in a timer.

### Step 0: Startup (once per app launch)
`startModelLoading()` runs on its own thread:
1. **Copy the DA3 model** from the APK assets to internal storage (`filesDir`). ExecuTorch needs a real file path.
   The copy goes through a `.tmp` file and is skipped if a copy of the same size already exists, so only the first launch is slow.
2. **Initialise UFLDv2** (an ONNX Runtime session with 2 intra-op and 2 inter-op threads).
3. **Load DA3** with `org.pytorch.executorch.Module.load()`.

OpenCV is initialised in parallel on the UI thread. A `CountDownLatch` (`modelsReady`) lets the
pipeline start before loading finishes and wait only when the models are actually needed.

### Step 1: Get the image
- **Camera:** CameraX captures a JPEG (low-latency mode) into the app's private `filesDir`.
  `MainActivity` reads the camera calibration (see §8), decodes the JPEG, applies the EXIF rotation,
  then **deletes the JPEG**.
- **Gallery:** decoded with `ImageDecoder` (software bitmap).

### Step 2: DA3 preprocessing ([DA3Preprocessor.kt](app/src/main/java/com/example/da3camera/DA3Preprocessor.kt))
The image is turned into a `1 × 3 × 504 × 504` float tensor:
1. Resize so the **longest side is 504 px**, keeping the aspect ratio (INTER_AREA when shrinking).
2. Resize each side to the **nearest multiple of 14** (the ViT patch size).
3. **Zero-pad** to 504 × 504, centered (letterboxing).
4. Normalise with ImageNet mean/std and lay out as NCHW.

The **intrinsics are updated to match**: they are scaled by both resizes, and `cx`/`cy` are shifted by
the padding. These "corrected intrinsics" describe the 504 × 504 tensor, not the original photo.

### Step 3: Lane detection in parallel ([Ufldv2Utils.kt](app/src/main/java/com/example/da3camera/Ufldv2Utils.kt), [LaneDetector.kt](app/src/main/java/com/example/da3camera/LaneDetector.kt))
Started with `CompletableFuture.supplyAsync`, so it **overlaps with DA3 inference**:
1. Crop the top `min(3222, height)` rows of the image (`DETECTION_LINE_POINT`, carried over from the original Python prototype).
2. Resize to 800 × 400, crop rows 80–400 to get an **800 × 320** input, then normalise with ImageNet stats.
3. Run the ONNX model. The outputs are `loc_row` (grid logits: *where* the lane is along each row anchor) and `exist_row` (*is* the lane present at each anchor).
4. Decode lanes **1 (ego-left)** and **2 (ego-right)**:
   - A lane is kept only if it is present in more than half of the 56 row anchors.
   - Per anchor, keep the point only if `P(exist) > 0.95`, then take a soft-argmax over the ±1 grid cells around the peak to get sub-cell x precision.
   - Row anchors follow TuSimple: `linspace(160, 710, 56) / 720`.
5. **Fit a straight line** `x = slope·y + intercept` to each lane (least squares) and sample it at 7 fixed heights (15 %–100 % of the image height).

### Step 4: DA3 inference
`module.forward(tensor)` on ExecuTorch. The model returns several outputs:
- `outputs[0]`: the **depth map** (504 × 504).
- `outputs[2]` (if present): **scene classification logits** for 5 classes:
  `bidirectional bicycle lane`, `unidirectional bicycle lane`, `roadway`, `shared streets`, `sidewalk`.
  (`outputs[1]`, a segmentation head, is **not used** in v3.)

### Step 5: Metric depth
DA3 uses a focal-length convention: its raw output corresponds to a camera with focal length 300 px.
It is converted to meters with:
```
metric_depth = raw_depth × ((fx + fy) / 2) / 300        # fx, fy = corrected 504×504 intrinsics
```
The scene class is the `argmax` of the classification logits.

### Step 6: Bike-lane ROI mask ([RoiMaskGenerator.kt](app/src/main/java/com/example/da3camera/RoiMaskGenerator.kt))
1. Starting from the bottom of the image, pair left/right lane points while `left.x < right.x`, so it stops where the two lines would cross.
2. Build a polygon (left points → bottom-left corner → bottom-right corner → right points in reverse order) and fill it white on black.
3. Rescale the mask to **504 × 504** (nearest neighbour) so it lines up with the depth map.
4. **Cut at 10 m**: mask pixels with depth ≤ 0 or > 10 m are removed.

If lane detection fails completely, the ROI step falls back to an all-white mask. If no lanes are
found, the mask is empty, so no objects are reported.

### Step 7: Ground plane with RANSAC ([RansacUtils.kt](app/src/main/java/com/example/da3camera/RansacUtils.kt))
1. Back-project ROI pixels (sub-sampled to about 10,000 points, Z < 20 m) into 3D using the corrected intrinsics.
2. RANSAC: **400 iterations**, inlier threshold **5 cm**, fixed random seed `42` (so results are reproducible).
3. Output: the plane `a·x + b·y + c·z + d = 0`, the inlier ratio, and derived tilt, roll and pitch angles in degrees (logged under `DA3_RANSAC`).

### Step 8: Obstacle detection ([ObjectDetector.kt](app/src/main/java/com/example/da3camera/ObjectDetector.kt))
Runs in the 504 × 504 model space. This saves memory compared with working on full-resolution photos.
1. For every ROI pixel with depth 0.5–15 m, mark it as an obstacle pixel if its 3D point is **more than 15 cm from the ground plane**.
2. Clean the mask with a morphological **open** then **close** (3 × 3 kernel).
3. `connectedComponentsWithStats` gives one blob per candidate object. Blobs under **30 px** are dropped.
4. For each blob, back-project a sample of its pixels and compute:
   - **Distance** = median Euclidean distance from the camera.
   - **Height** = `max(Y) − min(Y)` of its 3D points.
5. Keep objects with **0.2 m ≤ height ≤ 2.5 m**.
6. Scale the bounding boxes from 504 × 504 back to the original image size.

> Note: the box scaling multiplies by `originalWidth/504` and `originalHeight/504` directly and does not
> remove the letterbox padding from Step 2. For non-square photos, boxes may therefore be offset or stretched
> along the padded axis. See [§11](#11-known-limitations-and-caveats).

### Step 9: Draw and display
Red boxes with a black label `Dist: X.Xm | H: Y.Ym` are drawn on a copy of the original bitmap,
which is then shown on screen together with the object count, the scene class and the timing table.

## 5. Models

The two model files are **not stored in git** because they exceed GitHub's 100 MB file limit.
They are attached to this repository's **[GitHub Releases](../../releases)** and must be placed in
`app/src/main/assets/` before building.

| File | Size | Runtime | Input | Purpose |
|---|---|---|---|---|
| `da3_baseline72_fixed504.pte` | ~987 MB | ExecuTorch 1.5.1 | `1×3×504×504` float, ImageNet-normalised | Depth (output 0), segmentation (output 1, unused), 5-class scene classification (output 2) |
| `tusimple_res18.onnx` | ~385 MB | ONNX Runtime 1.31.1 | `1×3×320×800` float named `input` | UFLDv2 (ResNet-18 backbone, TuSimple), outputs `loc_row` `[1,G,56,4]` and `exist_row` `[1,2,56,4]` |

Both files are listed under `noCompress` in [app/build.gradle.kts](app/build.gradle.kts) so the APK
stores them uncompressed. On first launch they are copied to the app's internal storage, so the installed app needs about 1.4 GB of free space.

```bash
# Download the models from the latest release (needs the GitHub CLI and repo access)
gh release download --repo laurapzz/DA3Camera_v3 --pattern '*.pte' --pattern '*.onnx' --dir app/src/main/assets/
```

## 6. Code tour

```
DA3Camera_v3/
├── app/
│   ├── build.gradle.kts              # Dependencies, NDK/CMake config, ABI filters, noCompress for models
│   └── src/main/
│       ├── AndroidManifest.xml       # CAMERA permission; MainActivity (launcher) + CameraActivity
│       ├── assets/                   # ⚠ put the .pte and .onnx models here (git-ignored)
│       ├── cpp/
│       │   ├── CMakeLists.txt        # Builds libnative-lib.so against OpenCV (see §7 / §11)
│       │   └── native-lib.cpp        # Legacy C++ ROI-mask generator (built, not called in v3)
│       ├── jniLibs/{arm64-v8a,x86_64}/
│       │   ├── libopencv_java4.so    # OpenCV native lib linked by the C++ code
│       │   └── libc++_shared.so      # C++ runtime
│       ├── java/com/example/da3camera/
│       │   ├── MainActivity.kt       # UI, model startup, the full pipeline (runDetection), intrinsics handling
│       │   ├── CameraActivity.kt     # CameraX preview + capture; records Camera2 metadata; returns JPEG path
│       │   ├── DA3ModelLoader.kt     # Copies .pte from assets to filesDir and loads it with ExecuTorch
│       │   ├── DA3Preprocessor.kt    # Resize / patch-align / pad to 504², normalise, correct intrinsics
│       │   ├── LaneDetector.kt       # ONNX Runtime UFLDv2: preprocess, run, decode ego lanes, line-fit
│       │   ├── Ufldv2Utils.kt        # Lane-detection facade: crop, call LaneDetector, build ROI mask bytes
│       │   ├── RoiMaskGenerator.kt   # Lane points → filled polygon mask (+ visual overlay, unused in v3)
│       │   ├── RansacUtils.kt        # Back-projection, RANSAC ground plane, tilt/roll/pitch (+ PLY export, unused)
│       │   ├── ObjectDetector.kt     # Above-ground blobs → distance/height → boxes; draws the overlay
│       │   └── PipelineTimer.kt      # Per-step timing, logcat "DA3_TIMING", on-screen summary table
│       └── res/layout/
│           ├── activity_main.xml     # Main screen
│           └── activity_camera.xml   # Camera screen
├── gradle/libs.versions.toml         # Version catalog (AGP, AndroidX)
├── build.gradle.kts / settings.gradle.kts
└── README.md
```

**Threads at a glance**

| Thread | Work |
|---|---|
| UI thread | Views, OpenCV init, result display |
| Startup thread | Model copy plus UFLDv2 and DA3 loading (once) |
| `pipelineExecutor` (single thread) | The full `runDetection()` pipeline, one image at a time |
| `CompletableFuture` common pool | UFLDv2 inference, in parallel with DA3 |

## 7. Building and running

### Requirements
- **Android Studio** (recent version; the project uses AGP 9.4.1 and Gradle 9.6.0)
- **Android SDK** with `compileSdk`/`targetSdk` 37, `minSdk` 28
- **Android NDK `26.1.10909125`** and **CMake `3.22.1`** (install via *SDK Manager → SDK Tools*)
- **JDK**: Gradle uses a toolchain (configured in `gradle/gradle-daemon-jvm.properties`, auto-downloaded via foojay). The code targets Java 11.
- **OpenCV Android SDK 4.10.0**, used only for the C++ headers (see below)
- A **physical Android phone** (arm64) is recommended. The models are large, and the emulator (x86_64) is very slow and has no real camera calibration.

### Steps

```bash
# 1. Clone
git clone https://github.com/laurapzz/DA3Camera_v3.git
cd DA3Camera_v3

# 2. Get the models into app/src/main/assets/   (see §5)
gh release download --repo laurapzz/DA3Camera_v3 --pattern '*.pte' --pattern '*.onnx' --dir app/src/main/assets/

# 3. Point CMake to your OpenCV Android SDK headers (see note below)

# 4. Build and install on a connected device
./gradlew installDebug
```

> **Important: OpenCV include path.** [app/src/main/cpp/CMakeLists.txt](app/src/main/cpp/CMakeLists.txt) currently
> contains a hard-coded absolute path to the original author's machine:
> ```cmake
> include_directories("/Users/lauraperez/Downloads/OpenCV-android-sdk/sdk/native/jni/include")
> ```
> Download the [OpenCV 4.10.0 Android SDK](https://github.com/opencv/opencv/releases/tag/4.10.0), unzip it,
> and change this line to `<your path>/OpenCV-android-sdk/sdk/native/jni/include`. Without this, the native build fails.
> (The Kotlin code gets OpenCV from the Maven dependency `org.opencv:opencv:4.10.0`. Only the C++ library needs the SDK.)

The application ID is `com.example.cameradepthlane.v3`, so v3 can be installed alongside v2.
The first launch takes noticeably longer while the ~1 GB DA3 model is copied to internal storage.

## 8. Camera intrinsics: where they come from

Accurate intrinsics matter. They set the **metric scale** of the depth (Step 5) and the 3D
geometry (Steps 7–8). Wrong values lead to wrong distances and heights.

| Image source | Intrinsics used |
|---|---|
| **Photo taken in the app** | Read from Camera2 `LENS_INTRINSIC_CALIBRATION` for the camera CameraX selected. If the JPEG's EXIF orientation is `ROTATE_90`, they are rotated to match the upright image (`fx↔fy`, `cx' = (H−1) − cy`, `cy' = cx`). If the phone does not report a calibration, the manual values are used. |
| **Gallery image** | The manual fields (`fx`, `fy`, `cx`, `cy`). They are saved in `SharedPreferences` (`DA3CameraV3Prefs`) and pre-filled on the next launch. |

Values must be **in pixels of the image you are processing** (full resolution). The app scales them to
504 × 504 internally. Validation: all four must be numbers, and `fx`, `fy` > 0.

**Import from `.txt`.** The file can contain either:
- 4 numbers: `fx fy cx cy`, or
- a 3 × 3 camera matrix `K` (9 numbers, row-major). The app reads `K[0][0]`, `K[1][1]`, `K[0][2]`, `K[1][2]`.

Any separators work (spaces, commas, brackets, new lines), because the app simply extracts all the numbers in the file. Example:
```
1450.2 0 960.0
0 1450.2 540.0
0 0 1
```

## 9. Timing and logging

Every step is timed by [PipelineTimer](app/src/main/java/com/example/da3camera/PipelineTimer.kt). The results are
shown on screen and logged:

```bash
adb logcat -s DA3_TIMING          # per-step and total times
adb logcat -s DA3_TIMING DA3_PIPELINE DA3_RANSAC DA3_OBJECTS DA3_STARTUP UFLDV2 DA3_MODEL DA3_INTRINSICS DA3_CAMERA
```

Example timing table (the numbers are illustrative):
```
Read camera intrinsics                     2 ms
Decode JPEG + EXIF rotation              180 ms
DA3 preprocessing                         60 ms
Wait for model loading                     0 ms
UFLDv2 lane detection                    250 ms  (parallel)
DA3 inference (ExecuTorch)              ...  ms
...
TOTAL                                   ...  ms
```
- `(parallel)`: overlaps with other steps, so it does not add directly to the total.
- `(not in total)`: the camera capture time, from shutter press to JPEG, measured in `CameraActivity`.

| Log tag | Content |
|---|---|
| `DA3_TIMING` | Step durations and totals |
| `DA3_PIPELINE` | Image size, intrinsics used, failures |
| `DA3_RANSAC` | Plane equation and inlier ratio |
| `DA3_OBJECTS` | Distance and height of each detected object |
| `DA3_STARTUP`, `DA3_MODEL` | Model copy and loading |
| `UFLDV2` | Lane detector initialisation and errors |
| `DA3_CAMERA`, `DA3_CAPTURE_MATCH`, `DA3_INTRINSICS` | Camera capture and calibration details |

## 10. Tunable parameters

| Parameter | Value | Location |
|---|---|---|
| Model input size | 504 × 504, patch 14 | `DA3Preprocessor` |
| DA3 reference focal | 300 px | `MainActivity.runDetection` |
| ROI depth cut-off | 10 m | `MainActivity.runDetection` |
| UFLDv2 input | 800 × 320 (from 800 × 400, crop top 80) | `LaneDetector` |
| Lane existence threshold | 0.95 (per anchor); lane needs > 28 / 56 anchors | `LaneDetector` |
| Lane crop line | 3222 px from top | `Ufldv2Utils.DETECTION_LINE_POINT` |
| RANSAC iterations / threshold | 400 / 0.05 m, seed 42 | `RansacUtils.estimateGroundPlane` |
| RANSAC point cap | ~10,000 points, Z < 20 m | `RansacUtils.estimateGroundPlane` |
| Obstacle height above plane | > 0.15 m | `ObjectDetector.detectObjects504` |
| Obstacle depth range | 0.5–15 m | `ObjectDetector.detectObjects504` |
| Min blob area | 30 px (in 504² space) | `ObjectDetector.detectObjects504` |
| Object height filter | 0.2–2.5 m | `ObjectDetector.detectObjects504` |

## 11. Known limitations and caveats

- **Lane model domain.** UFLDv2 was trained on TuSimple (US highways). Its "ego lane" is used as a stand-in for the bike lane, which works best when the bike lane has clear painted borders and the camera is roughly centered on it.
- **Straight-line lanes only.** Each lane is fitted with a straight line, so curved lanes are approximated.
- **No lanes → no detections.** If either ego line is not found, the ROI is empty and the app reports 0 objects.
- **Box placement on non-square images.** Bounding boxes are scaled from 504 × 504 to the original size without removing the letterbox padding (Step 8).
- **Depth range.** The ROI is cut at 10 m and obstacles are only searched between 0.5 and 15 m.
- **Legacy / unused code.** `native-lib.cpp` (C++ ROI mask) is compiled, but nothing loads it from Kotlin (there is no `System.loadLibrary` or `external fun`). `RansacUtils.exportPly`, `RoiMaskGenerator.isObstacleInLane`, the ROI `visualOverlay`, and DA3's segmentation output are also unused in v3. They were kept from v2.
- **Hard-coded OpenCV path** in `CMakeLists.txt` (see §7).
- **Memory and storage.** About 1.4 GB of models are copied to internal storage, and DA3 needs a recent phone with enough RAM.
- **One image at a time.** The camera and gallery buttons are disabled while a run is in progress.

## 12. Differences from v2

v3 is a streamlined, timed version of v2:
- **Automatic flow:** take or select an image → processing → final detection image. There is no *Process Image* button.
- **No output files:** no TIFF, PNG, TXT or PLY export, no output-folder picker, and no intermediate depth or lane previews. The only files written are the one-time model copy and the temporary camera JPEG, which is deleted right after decoding.
- **Classification only as text:** the scene class is displayed, and the segmentation output is ignored.
- **Execution time** of every step is shown on screen and in logcat (`DA3_TIMING`).
- **Remembered intrinsics** for gallery images. Images without intrinsics wait until they are provided.
- **Separate application ID** (`com.example.cameradepthlane.v3`), so v2 and v3 can be installed side by side.
