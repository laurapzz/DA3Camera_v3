package com.example.da3camera

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.util.Locale
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * v3 flow: take / select image -> pipeline starts automatically -> final detection image.
 *
 * No files are written (except the temporary camera JPEG hand-off, deleted right after
 * decoding, and the one-time model copy that ExecuTorch needs). Every step is timed
 * (logcat tag "DA3_TIMING") and the breakdown is shown on screen.
 */
class MainActivity : AppCompatActivity() {

    // ---- Models -------------------------------------------------------------------------

    @Volatile
    private var da3Module: Module? = null

    @Volatile
    private var modelLoadError: Throwable? = null

    /** Released once DA3 + UFLDv2 have finished loading (successfully or not). */
    private val modelsReady = CountDownLatch(1)

    private val startupTimer = PipelineTimer("startup")

    // ---- Pipeline state (UI thread only) ------------------------------------------------

    private val pipelineExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var isBusy = false

    /** Image waiting for the user to provide intrinsics (gallery image, no calibration). */
    private var pendingBitmap: Bitmap? = null

    // ---- Views --------------------------------------------------------------------------

    private lateinit var imagePreview: ImageView
    private lateinit var textStatus: TextView
    private lateinit var textResult: TextView
    private lateinit var textTiming: TextView
    private lateinit var textStartupTiming: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var btnCamera: Button
    private lateinit var btnGallery: Button
    private lateinit var inputFx: EditText
    private lateinit var inputFy: EditText
    private lateinit var inputCx: EditText
    private lateinit var inputCy: EditText

    // ---- Activity result launchers ------------------------------------------------------

    private val requestCameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                cameraLauncher.launch(Intent(this, CameraActivity::class.java))
            } else {
                Toast.makeText(
                    this,
                    "Camera permission is required to take photos",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    private val cameraLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode != RESULT_OK) return@registerForActivityResult

            val data = result.data
            val photoPath = data?.getStringExtra("captured_photo_path")

            if (photoPath == null || !File(photoPath).exists()) {
                showError("Camera did not return a photograph")
                return@registerForActivityResult
            }

            startCameraPipeline(
                photoPath = photoPath,
                cameraId = data.getStringExtra("captured_camera_id"),
                captureMillis = data.getLongExtra("captured_duration_ms", -1L)
            )
        }

    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri != null) startGalleryPipeline(uri)
        }

    private val pickIntrinsicsFileLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri != null) importIntrinsicsFile(uri)
        }

    // ---- Lifecycle ----------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imagePreview = findViewById(R.id.imagePreview)
        textStatus = findViewById(R.id.textStatus)
        textResult = findViewById(R.id.textResult)
        textTiming = findViewById(R.id.textTiming)
        textStartupTiming = findViewById(R.id.textStartupTiming)
        progressBar = findViewById(R.id.progressBar)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        inputFx = findViewById(R.id.inputFx)
        inputFy = findViewById(R.id.inputFy)
        inputCx = findViewById(R.id.inputCx)
        inputCy = findViewById(R.id.inputCy)

        loadSavedIntrinsics()

        startModelLoading()

        val openCvLoaded = startupTimer.measure("OpenCV init") {
            org.opencv.android.OpenCVLoader.initLocal()
        }
        if (!openCvLoaded) {
            Log.e("DA3_STARTUP", "OpenCV failed to initialize")
            textStatus.text = "OpenCV failed to initialize"
        }

        btnCamera.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
            ) {
                cameraLauncher.launch(Intent(this, CameraActivity::class.java))
            } else {
                requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        btnGallery.setOnClickListener {
            pickImageLauncher.launch(arrayOf("image/*"))
        }

        findViewById<Button>(R.id.btnImportIntrinsics).setOnClickListener {
            pickIntrinsicsFileLauncher.launch(arrayOf("text/plain", "*/*"))
        }

        // Pressing "Done" on the keyboard resumes an image that was waiting for intrinsics.
        for (field in listOf(inputFx, inputFy, inputCx, inputCy)) {
            field.setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_DONE) {
                    readManualIntrinsics()?.let { saveIntrinsics(it) }
                    tryResumePending()
                }
                false
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipelineExecutor.shutdown()
    }

    // ---- Startup: model loading ---------------------------------------------------------

    private fun startModelLoading() {
        Thread {
            try {
                val modelFile = startupTimer.measure("DA3 model copy / check") {
                    DA3ModelLoader.prepareModelFile(applicationContext)
                }

                startupTimer.measure("UFLDv2 init (ONNX Runtime)") {
                    Ufldv2Utils.initialize(applicationContext)
                }

                da3Module = startupTimer.measure("DA3 load (ExecuTorch)") {
                    DA3ModelLoader.loadModel(modelFile)
                }

                val total = startupTimer.logTotal()
                runOnUiThread {
                    textStartupTiming.text = startupTimer.summary(total)
                    if (!isBusy && pendingBitmap == null) {
                        textStatus.text = "Ready. Take or choose a photo."
                    }
                }

            } catch (e: Throwable) {
                modelLoadError = e
                Log.e("DA3_STARTUP", "Model preparation failed", e)
                runOnUiThread {
                    textStartupTiming.text = "Model loading failed: ${e.message}"
                    textStatus.text = "Model loading failed: ${e.message}"
                }
            } finally {
                modelsReady.countDown()
            }
        }.start()
    }

    // ---- Entry points -------------------------------------------------------------------

    private fun startCameraPipeline(photoPath: String, cameraId: String?, captureMillis: Long) {
        if (isBusy) return
        val manualIntrinsics = readManualIntrinsics()

        beginRun("Processing photo…")

        pipelineExecutor.execute {
            val timer = PipelineTimer("camera")
            if (captureMillis >= 0) {
                timer.record(
                    "Camera capture (shutter→JPEG)",
                    captureMillis.toDouble(),
                    outsideTotal = true
                )
            }

            try {
                val cameraIntrinsics = cameraId?.let { id ->
                    timer.measure("Read camera intrinsics") {
                        readCameraIntrinsics(photoPath, id)
                    }
                }

                val bitmap = try {
                    timer.measure("Decode JPEG + EXIF rotation") {
                        loadCorrectlyOrientedBitmap(photoPath)
                    } ?: error("Failed to decode captured photograph")
                } finally {
                    // The JPEG was only a hand-off from CameraActivity; don't keep it.
                    File(photoPath).delete()
                }

                val intrinsics = cameraIntrinsics ?: manualIntrinsics
                val source = if (cameraIntrinsics != null) "camera calibration" else "manual"

                continueWithBitmap(bitmap, intrinsics, source, timer)

            } catch (e: Throwable) {
                failRun(e)
            }
        }
    }

    private fun startGalleryPipeline(uri: Uri) {
        if (isBusy) return
        val manualIntrinsics = readManualIntrinsics()
        manualIntrinsics?.let { saveIntrinsics(it) }

        beginRun("Processing image…")

        pipelineExecutor.execute {
            val timer = PipelineTimer("gallery")
            try {
                val bitmap = timer.measure("Decode image") { decodeUri(uri) }
                continueWithBitmap(bitmap, manualIntrinsics, "manual", timer)
            } catch (e: Throwable) {
                failRun(e)
            }
        }
    }

    /** Called when intrinsics are provided for an image that was waiting for them. */
    private fun tryResumePending() {
        val bitmap = pendingBitmap ?: return
        if (isBusy) return

        val intrinsics = readManualIntrinsics()
        if (intrinsics == null) {
            textStatus.text = "Intrinsics are not valid yet (fx, fy > 0; all four required)."
            return
        }
        saveIntrinsics(intrinsics)

        beginRun("Processing image…")

        pipelineExecutor.execute {
            val timer = PipelineTimer("resume")
            try {
                runDetection(bitmap, intrinsics, "manual", timer)
            } catch (e: Throwable) {
                failRun(e)
            }
        }
    }

    /** Worker thread. Shows the input image, then runs detection or waits for intrinsics. */
    private fun continueWithBitmap(
        bitmap: Bitmap,
        intrinsics: CameraIntrinsics?,
        intrinsicsSource: String,
        timer: PipelineTimer
    ) {
        runOnUiThread { imagePreview.setImageBitmap(bitmap) }

        if (intrinsics == null) {
            runOnUiThread {
                endRun()
                pendingBitmap = bitmap
                textStatus.text = "Enter or import camera intrinsics below. " +
                        "Processing starts automatically (press Done on the keyboard)."
                textTiming.text = "Waiting for intrinsics"
            }
            return
        }

        runDetection(bitmap, intrinsics, intrinsicsSource, timer)
    }

    // ---- The pipeline -------------------------------------------------------------------

    /** Worker thread. Runs the complete detection pipeline and displays the final image. */
    private fun runDetection(
        bitmap: Bitmap,
        intrinsics: CameraIntrinsics,
        intrinsicsSource: String,
        timer: PipelineTimer
    ) {
        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        Log.d(
            "DA3_PIPELINE",
            "Image ${originalWidth}x$originalHeight, intrinsics ($intrinsicsSource): $intrinsics"
        )

        // 1. DA3 preprocessing (only needs OpenCV, so it can run while models finish loading).
        setStatus("Preprocessing…")
        val prep = timer.measure("DA3 preprocessing") {
            DA3Preprocessor.preprocess(bitmap, intrinsics)
        }

        // 2. Make sure both models are loaded (only waits if the app has just started).
        if (modelsReady.count > 0) setStatus("Waiting for models to finish loading…")
        val module = timer.measure("Wait for model loading") {
            modelsReady.await()
            da3Module
        } ?: error("DA3 model failed to load: ${modelLoadError?.message}")

        // 3. UFLDv2 lane detection, in parallel with DA3 inference.
        val laneFuture = CompletableFuture.supplyAsync {
            timer.measure("UFLDv2 lane detection", parallel = true) {
                try {
                    Ufldv2Utils.generateLaneResult(bitmap, originalWidth, originalHeight)
                } catch (e: Exception) {
                    Log.e("UFLDV2", "Lane detection failed", e)
                    emptyLaneResult(originalWidth, originalHeight)
                }
            }
        }

        // 4. DA3 inference.
        setStatus("Running depth model…")
        val outputs = timer.measure("DA3 inference (ExecuTorch)") {
            val tensor = Tensor.fromBlob(prep.inputTensor, longArrayOf(1, 3, 504, 504))
            module.forward(EValue.from(tensor))
        }

        // 5. Metric depth (DA3 focal-length convention) + scene classification.
        val k = prep.correctedIntrinsics
        val (metricDepthValues, predictedClassName) = timer.measure("Metric depth + classification") {
            val depthValues = outputs[0].toTensor().dataAsFloatArray
            val depthScale = ((k.fx + k.fy) / 2f) / 300f
            val metric = FloatArray(depthValues.size) { i -> depthValues[i] * depthScale }

            val className = if (outputs.size > 2) {
                val logits = outputs[2].toTensor().dataAsFloatArray
                logits.indices.maxByOrNull { logits[it] }
                    ?.let { CLASSIFICATION_LABELS.getOrNull(it) }
            } else {
                null
            }
            metric to className
        }

        // 6. Lane result (usually already finished while DA3 was running).
        setStatus("Detecting objects…")
        val laneResult = timer.measure("Wait for lane detection") {
            try {
                laneFuture.get()
            } catch (e: Exception) {
                Log.e("UFLDV2", "Failed to get parallel lane result", e)
                emptyLaneResult(originalWidth, originalHeight)
            }
        }

        // 7. 504x504 bike-lane ROI mask, cut at ~10 m.
        val roiMaskBytes = timer.measure("Bike-lane ROI mask (504x504)") {
            val roi = try {
                Ufldv2Utils.generateRoiMaskFromLanes(
                    bitmap,
                    laneResult.leftLane,
                    laneResult.rightLane,
                    504,
                    504
                )
            } catch (e: Exception) {
                Log.e("UFLDV2", "Failed to generate bike lane ROI mask", e)
                ByteArray(504 * 504) { 255.toByte() }
            }

            for (i in roi.indices) {
                val maskVal = roi[i].toInt() and 0xFF
                val depth = metricDepthValues[i]
                if (maskVal > 0 && (depth <= 0f || depth > 10.0f)) {
                    roi[i] = 0
                }
            }
            roi
        }

        // 8. RANSAC ground plane on the ROI point cloud.
        val modelIntrinsics = CameraIntrinsics(k.fx, k.fy, k.cx, k.cy)
        val planeResult = timer.measure("RANSAC ground plane") {
            val maskedDepthValues = FloatArray(metricDepthValues.size) { i ->
                if ((roiMaskBytes[i].toInt() and 0xFF) > 0) metricDepthValues[i] else 0f
            }
            RansacUtils.estimateGroundPlane(
                metricDepth = maskedDepthValues,
                width = 504,
                height = 504,
                intrinsics = modelIntrinsics
            )
        }

        Log.d(
            "DA3_RANSAC",
            "Plane: ${planeResult.a}x + ${planeResult.b}y + ${planeResult.c}z + ${planeResult.d} = 0, " +
                    "inliers=${planeResult.inlierRatio}"
        )

        // 9. Object detection (ground removed, connected components), scaled to full resolution.
        val detectedObjects = timer.measure("Object detection") {
            ObjectDetector.detectObjects504(
                metricDepthValues = metricDepthValues,
                modelIntrinsics = modelIntrinsics,
                plane = planeResult,
                roiMaskBytes = roiMaskBytes,
                originalWidth = originalWidth,
                originalHeight = originalHeight
            )
        }

        for ((idx, obj) in detectedObjects.withIndex()) {
            Log.d(
                "DA3_OBJECTS",
                "Object $idx: distance=${obj.distanceMeters}m, height=${obj.heightMeters}m"
            )
        }

        // 10. Final detection image.
        val overlay = timer.measure("Draw detection overlay") {
            ObjectDetector.drawObjectsOverlay(bitmap, detectedObjects)
        }

        runOnUiThread {
            timer.measure("Display result") {
                imagePreview.setImageBitmap(overlay)
            }
            val total = timer.logTotal()

            val objectsText = when (detectedObjects.size) {
                1 -> "1 object detected in the bike lane"
                else -> "${detectedObjects.size} objects detected in the bike lane"
            }
            textResult.text = buildString {
                append(objectsText)
                if (predictedClassName != null) append("\nScene: $predictedClassName")
            }
            imagePreview.contentDescription = objectsText
            textTiming.text = timer.summary(total) + "\n\nIntrinsics: $intrinsicsSource"

            endRun()
            textStatus.text = String.format(Locale.US, "Done in %.2f s", total / 1000.0)
        }
    }

    // ---- Helpers: UI state --------------------------------------------------------------

    private fun beginRun(status: String) {
        isBusy = true
        pendingBitmap = null
        btnCamera.isEnabled = false
        btnGallery.isEnabled = false
        progressBar.visibility = View.VISIBLE
        textStatus.text = status
        textResult.text = ""
        textTiming.text = "Running…"
    }

    private fun endRun() {
        isBusy = false
        btnCamera.isEnabled = true
        btnGallery.isEnabled = true
        progressBar.visibility = View.GONE
    }

    private fun setStatus(message: String) {
        runOnUiThread { textStatus.text = message }
    }

    private fun failRun(e: Throwable) {
        Log.e("DA3_PIPELINE", "Pipeline failed", e)
        runOnUiThread {
            endRun()
            showError("Processing failed: ${e.message}")
            textTiming.text = "Run failed"
        }
    }

    private fun showError(message: String) {
        textStatus.text = message
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    // ---- Helpers: images ----------------------------------------------------------------

    private fun decodeUri(uri: Uri): Bitmap {
        return if (android.os.Build.VERSION.SDK_INT >= 28) {
            val source = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    }

    private fun loadCorrectlyOrientedBitmap(photoPath: String): Bitmap? {
        val bitmap = BitmapFactory.decodeFile(photoPath) ?: return null

        val orientation = ExifInterface(photoPath).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

        if (orientation == ExifInterface.ORIENTATION_NORMAL ||
            orientation == ExifInterface.ORIENTATION_UNDEFINED
        ) {
            return bitmap
        }

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1f, 1f)
            }
        }

        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }

    private fun emptyLaneResult(width: Int, height: Int) =
        Ufldv2Utils.LaneResult(ByteArray(width * height), ArrayList(), ArrayList(), 0f, 0f)

    // ---- Helpers: intrinsics ------------------------------------------------------------

    /**
     * Intrinsics from Camera2 LENS_INTRINSIC_CALIBRATION, rotated to match the
     * EXIF-rotated image when the JPEG is ROTATE_90 (same convention as v2).
     */
    private fun readCameraIntrinsics(photoPath: String, cameraId: String): CameraIntrinsics? {
        return try {
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(photoPath, options)

            val orientation = ExifInterface(photoPath).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED
            )

            val cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
            val calibration = cameraManager.getCameraCharacteristics(cameraId)
                .get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)

            if (calibration == null || calibration.size < 5) {
                Log.w("DA3_INTRINSICS", "No intrinsic calibration for camera $cameraId")
                return null
            }

            val fx = calibration[0]
            val fy = calibration[1]
            val cx = calibration[2]
            val cy = calibration[3]

            if (options.outWidth > 0 && options.outHeight > 0 &&
                orientation == ExifInterface.ORIENTATION_ROTATE_90
            ) {
                CameraIntrinsics(
                    fx = fy,
                    fy = fx,
                    cx = (options.outHeight - 1) - cy,
                    cy = cx
                )
            } else {
                CameraIntrinsics(fx = fx, fy = fy, cx = cx, cy = cy)
            }
        } catch (e: Exception) {
            Log.e("DA3_INTRINSICS", "Failed to read camera calibration", e)
            null
        }
    }

    private fun readManualIntrinsics(): CameraIntrinsics? {
        val fx = inputFx.text.toString().toFloatOrNull()
        val fy = inputFy.text.toString().toFloatOrNull()
        val cx = inputCx.text.toString().toFloatOrNull()
        val cy = inputCy.text.toString().toFloatOrNull()

        if (fx == null || fy == null || cx == null || cy == null) return null
        if (!fx.isFinite() || !fy.isFinite() || !cx.isFinite() || !cy.isFinite()) return null
        if (fx <= 0f || fy <= 0f) return null

        return CameraIntrinsics(fx, fy, cx, cy)
    }

    private fun setIntrinsicFields(fx: Float, fy: Float, cx: Float, cy: Float) {
        inputFx.setText(fx.toString())
        inputFy.setText(fy.toString())
        inputCx.setText(cx.toString())
        inputCy.setText(cy.toString())
    }

    private fun saveIntrinsics(k: CameraIntrinsics) {
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putFloat("fx", k.fx)
            .putFloat("fy", k.fy)
            .putFloat("cx", k.cx)
            .putFloat("cy", k.cy)
            .apply()
    }

    private fun loadSavedIntrinsics() {
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        if (prefs.contains("fx")) {
            setIntrinsicFields(
                prefs.getFloat("fx", 0f),
                prefs.getFloat("fy", 0f),
                prefs.getFloat("cx", 0f),
                prefs.getFloat("cy", 0f)
            )
        }
    }

    /** Accepts a 3x3 K matrix (9 numbers) or "fx fy cx cy" (4 numbers), as in v2. */
    private fun importIntrinsicsFile(uri: Uri) {
        try {
            val content = contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                ?: error("Cannot read file")

            val numbers = Regex("[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?")
                .findAll(content).map { it.value.toFloatOrNull() }.toList()

            val values = when {
                numbers.size >= 9 -> listOf(numbers[0], numbers[4], numbers[2], numbers[5])
                numbers.size >= 4 -> listOf(numbers[0], numbers[1], numbers[2], numbers[3])
                else -> {
                    Toast.makeText(
                        this,
                        "Expected at least 4 or 9 numeric values, found ${numbers.size}",
                        Toast.LENGTH_LONG
                    ).show()
                    return
                }
            }

            if (values.any { it == null }) {
                Toast.makeText(this, "Failed to parse intrinsic values", Toast.LENGTH_LONG).show()
                return
            }

            setIntrinsicFields(values[0]!!, values[1]!!, values[2]!!, values[3]!!)
            readManualIntrinsics()?.let { saveIntrinsics(it) }
            Toast.makeText(this, "Intrinsics imported", Toast.LENGTH_SHORT).show()

            tryResumePending()

        } catch (e: Exception) {
            Toast.makeText(this, "Error reading file: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    companion object {
        private const val PREFS = "DA3CameraV3Prefs"

        /** Classification labels in model output order. */
        private val CLASSIFICATION_LABELS = arrayOf(
            "bidirectional bicycle lane",
            "unidirectional bicycle lane",
            "roadway",
            "shared streets",
            "sidewalk"
        )
    }
}
