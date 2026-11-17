package com.example.da3camera

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.camera.core.ImageCapture
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import androidx.camera.core.ImageCaptureException
import android.app.Activity
import android.content.Intent
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import androidx.camera.camera2.interop.Camera2CameraInfo
import android.hardware.camera2.CaptureResult
import androidx.camera.camera2.interop.Camera2Interop
import android.hardware.camera2.TotalCaptureResult
import androidx.exifinterface.media.ExifInterface
import androidx.camera.core.ImageProxy
import java.util.concurrent.ConcurrentHashMap
import java.io.FileOutputStream

class CameraActivity : AppCompatActivity() {

    private lateinit var cameraPreview: PreviewView

    private lateinit var imageCapture: ImageCapture

    // Camera2 ID of the camera selected by CameraX.
    private var selectedCameraId: String? = null

    // Camera2 metadata indexed by sensor timestamp.
    private val stillCaptureResults =
        ConcurrentHashMap<Long, TotalCaptureResult>()

    private fun saveCapturedJpeg(
        imageProxy: ImageProxy,
        photoFile: File
    ) {
        try {
            val buffer = imageProxy.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)

            FileOutputStream(photoFile).use { output ->
                output.write(bytes)
            }

            Log.d(
                "DA3_CAMERA",
                "In-memory JPEG saved: ${photoFile.absolutePath}"
            )

            Log.d(
                "DA3_CAMERA",
                "Saved JPEG size: ${photoFile.length()} bytes"
            )

        } finally {
            imageProxy.close()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_camera)

        cameraPreview = findViewById(R.id.cameraPreview)

        // Return to MainActivity.
        findViewById<Button>(R.id.btnBack).setOnClickListener {
            finish()
        }

        // Start the live camera preview.
        startCamera()

        // Inspect camera calibration.
        inspectCameraIntrinsics()

        // Capture a photograph.
        findViewById<Button>(R.id.btnTakePhoto).setOnClickListener {
            takePhoto()
        }
    }

    private fun takePhoto() {

        if (!::imageCapture.isInitialized) {
            Log.e("DA3_CAMERA", "ImageCapture is not initialized")
            return
        }

        // Generate a unique filename.
        val timestamp = SimpleDateFormat(
            "yyyyMMdd_HHmmss_SSS",
            Locale.US
        ).format(System.currentTimeMillis())

        val photoFile = File(
            filesDir,
            "DA3_$timestamp.jpg"
        )

        Log.d("DA3_CAMERA", "Capturing photograph...")

        // Timing: shutter press -> JPEG handed to MainActivity.
        val captureStartNs = System.nanoTime()

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),

            object : ImageCapture.OnImageCapturedCallback() {

                override fun onCaptureSuccess(imageProxy: ImageProxy) {

                    // Timestamp of the actual captured image.
                    val imageTimestamp = imageProxy.imageInfo.timestamp

                    Log.d(
                        "DA3_CAPTURE_MATCH",
                        "Captured image timestamp: $imageTimestamp"
                    )

                    Log.d(
                        "DA3_CAPTURE_MATCH",
                        "Captured using Camera2 camera ID: $selectedCameraId"
                    )

                    // Check that CameraX returned JPEG data.
                    if (imageProxy.format != android.graphics.ImageFormat.JPEG) {

                        Log.e(
                            "DA3_CAMERA",
                            "Unexpected image format: ${imageProxy.format}"
                        )

                        imageProxy.close()
                        return
                    }

                    // Match the image to its Camera2 capture result.
                    val captureResult =
                        stillCaptureResults.remove(imageTimestamp)

                    val capturedCropRegion = captureResult?.get(
                        CaptureResult.SCALER_CROP_REGION
                    )

                    val capturedDistortionMode = captureResult?.get(
                        CaptureResult.DISTORTION_CORRECTION_MODE
                    )

                    val capturedZoomRatio = captureResult?.get(
                        CaptureResult.CONTROL_ZOOM_RATIO
                    )

                    if (captureResult != null) {

                        val cropRegion = captureResult.get(
                            CaptureResult.SCALER_CROP_REGION
                        )

                        val distortionMode = captureResult.get(
                            CaptureResult.DISTORTION_CORRECTION_MODE
                        )

                        val zoomRatio = captureResult.get(
                            CaptureResult.CONTROL_ZOOM_RATIO
                        )

                        Log.d(
                            "DA3_CAPTURE_MATCH",
                            "MATCH FOUND: " +
                                    "timestamp=$imageTimestamp, " +
                                    "crop=$cropRegion, " +
                                    "distortion=$distortionMode, " +
                                    "zoom=$zoomRatio"
                        )

                    } else {

                        Log.w(
                            "DA3_CAPTURE_MATCH",
                            "No matching Camera2 result for timestamp=$imageTimestamp"
                        )
                    }

                    // Save the JPEG and close ImageProxy.
                    try {

                        // Temporary hand-off file; MainActivity deletes it after decoding.
                        saveCapturedJpeg(imageProxy, photoFile)

                        val captureDurationMs =
                            (System.nanoTime() - captureStartNs) / 1_000_000L

                        Log.d(
                            "DA3_TIMING",
                            "[camera] Camera capture (shutter→JPEG): $captureDurationMs ms"
                        )

                        // Return the photograph to MainActivity.
                        val resultIntent = Intent().apply {

                            putExtra("captured_duration_ms", captureDurationMs)

                            putExtra(
                                "captured_photo_path",
                                photoFile.absolutePath
                            )

                            putExtra(
                                "captured_camera_id",
                                selectedCameraId
                            )

                            putExtra("captured_sensor_timestamp", imageTimestamp)

                            capturedCropRegion?.let { crop ->
                                putExtra("captured_crop_left", crop.left)
                                putExtra("captured_crop_top", crop.top)
                                putExtra("captured_crop_right", crop.right)
                                putExtra("captured_crop_bottom", crop.bottom)
                            }

                            capturedDistortionMode?.let { mode ->
                                putExtra("captured_distortion_mode", mode)
                            }

                            capturedZoomRatio?.let { zoom ->
                                putExtra("captured_zoom_ratio", zoom)
                            }
                        }

                        setResult(Activity.RESULT_OK, resultIntent)

                        Log.d(
                            "DA3_CAMERA",
                            "Returning captured photograph to MainActivity"
                        )

                        finish()

                    } catch (e: Exception) {

                        Log.e(
                            "DA3_CAMERA",
                            "Failed to save captured JPEG",
                            e
                        )

                        Toast.makeText(
                            this@CameraActivity,
                            "Failed to save photograph: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }

                override fun onError(exception: ImageCaptureException) {

                    Log.e(
                        "DA3_CAMERA",
                        "Photograph capture failed",
                        exception
                    )

                    Toast.makeText(
                        this@CameraActivity,
                        "Capture failed: ${exception.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        )
    }

    private fun startCamera() {

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            try {
                val cameraProvider = cameraProviderFuture.get()

                // Configure the live preview.
                val previewBuilder = Preview.Builder()

                // Observe Camera2 capture results.
                val camera2Interop = Camera2Interop.Extender(previewBuilder)

                camera2Interop.setSessionCaptureCallback(
                    object : android.hardware.camera2.CameraCaptureSession.CaptureCallback() {

                        private var metadataLogged = false

                        override fun onCaptureCompleted(
                            session: android.hardware.camera2.CameraCaptureSession,
                            request: android.hardware.camera2.CaptureRequest,
                            result: TotalCaptureResult
                        ) {
                            super.onCaptureCompleted(session, request, result)

                            // Log one preview result to avoid flooding Logcat.
                            if (metadataLogged) return
                            metadataLogged = true

                            val cropRegion = result.get(
                                CaptureResult.SCALER_CROP_REGION
                            )

                            val distortionMode = result.get(
                                CaptureResult.DISTORTION_CORRECTION_MODE
                            )

                            val zoomRatio = result.get(
                                CaptureResult.CONTROL_ZOOM_RATIO
                            )

                            Log.d(
                                "DA3_CAPTURE",
                                "Preview crop region: $cropRegion"
                            )

                            Log.d(
                                "DA3_CAPTURE",
                                "Preview distortion correction mode: $distortionMode"
                            )

                            Log.d(
                                "DA3_CAPTURE",
                                "Preview zoom ratio: $zoomRatio"
                            )
                        }
                    }
                )

                val preview = previewBuilder
                    .build()
                    .also {
                        it.surfaceProvider = cameraPreview.surfaceProvider
                    }

                // Configure image capture.
                val imageCaptureBuilder = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)

                // Attach a Camera2 callback to the ImageCapture use case.
                val imageCaptureInterop = Camera2Interop.Extender(imageCaptureBuilder)

                imageCaptureInterop.setSessionCaptureCallback(
                    object : android.hardware.camera2.CameraCaptureSession.CaptureCallback() {

                        override fun onCaptureCompleted(
                            session: android.hardware.camera2.CameraCaptureSession,
                            request: android.hardware.camera2.CaptureRequest,
                            result: TotalCaptureResult
                        ) {
                            super.onCaptureCompleted(session, request, result)

                            val cropRegion = result.get(
                                CaptureResult.SCALER_CROP_REGION
                            )

                            val distortionMode = result.get(
                                CaptureResult.DISTORTION_CORRECTION_MODE
                            )

                            val zoomRatio = result.get(
                                CaptureResult.CONTROL_ZOOM_RATIO
                            )

                            val sensorTimestamp = result.get(
                                CaptureResult.SENSOR_TIMESTAMP
                            )

                            val requestTemplate = request.get(
                                android.hardware.camera2.CaptureRequest.CONTROL_CAPTURE_INTENT
                            )

                            val captureType = when (requestTemplate) {

                                android.hardware.camera2.CaptureRequest.CONTROL_CAPTURE_INTENT_STILL_CAPTURE ->
                                    "STILL_CAPTURE"

                                android.hardware.camera2.CaptureRequest.CONTROL_CAPTURE_INTENT_PREVIEW ->
                                    "PREVIEW"

                                android.hardware.camera2.CaptureRequest.CONTROL_CAPTURE_INTENT_VIDEO_RECORD ->
                                    "VIDEO_RECORD"

                                else ->
                                    "OTHER($requestTemplate)"
                            }

                            if (captureType == "STILL_CAPTURE") {

                                if (sensorTimestamp != null) {

                                    if (android.os.Build.VERSION.SDK_INT >= 28) {

                                        val physicalResults = result.physicalCameraResults

                                        Log.d(
                                            "DA3_PHYSICAL_CAMERA",
                                            "Physical camera results: ${physicalResults.keys}"
                                        )

                                        for ((physicalId, physicalResult) in physicalResults) {

                                            val physicalTimestamp = physicalResult.get(
                                                CaptureResult.SENSOR_TIMESTAMP
                                            )

                                            Log.d(
                                                "DA3_PHYSICAL_CAMERA",
                                                "Physical camera ID=$physicalId, " +
                                                        "timestamp=$physicalTimestamp"
                                            )
                                        }
                                    }

                                    // Store the complete Camera2 result for later matching.
                                    stillCaptureResults[sensorTimestamp] = result

                                    Log.d(
                                        "DA3_STILL_CAPTURE",
                                        "Stored still-capture metadata: " +
                                                "timestamp=$sensorTimestamp, " +
                                                "crop=$cropRegion, " +
                                                "distortion=$distortionMode, " +
                                                "zoom=$zoomRatio"
                                    )

                                } else {

                                    Log.w(
                                        "DA3_STILL_CAPTURE",
                                        "Still capture has no sensor timestamp"
                                    )
                                }
                            }
                        }
                    }
                )

                imageCapture = imageCaptureBuilder.build()

                // Use the rear camera.
                val cameraSelector =
                    CameraSelector.DEFAULT_BACK_CAMERA

                // Remove any previous camera use cases.
                cameraProvider.unbindAll()

                // Connect the camera to this activity.
                // Bind the preview and image capture to the selected camera.
                val camera = cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageCapture
                )

                // Identify the actual Camera2 camera used by CameraX.
                val cameraId = Camera2CameraInfo.from(
                    camera.cameraInfo
                ).cameraId

                // Remember which camera was selected for this session.
                selectedCameraId = cameraId

                // Retrieve calibration for the camera actually selected by CameraX.
                val cameraManager =
                    getSystemService(CAMERA_SERVICE) as CameraManager

                val characteristics =
                    cameraManager.getCameraCharacteristics(cameraId)

                val intrinsics = characteristics.get(
                    CameraCharacteristics.LENS_INTRINSIC_CALIBRATION
                )

                // Sensor dimensions in the pre-correction coordinate system.
                val preCorrectionArray = characteristics.get(
                    CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE
                )

                Log.d(
                    "DA3_INTRINSICS",
                    "Selected camera pre-correction array: $preCorrectionArray"
                )

                // Sensor dimensions in the active-array coordinate system.
                val activeArray = characteristics.get(
                    CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE
                )

                Log.d(
                    "DA3_INTRINSICS",
                    "Selected camera active array: $activeArray"
                )

                // Distortion-correction modes supported by this camera.
                val distortionModes = characteristics.get(
                    CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES
                )

                Log.d(
                    "DA3_INTRINSICS",
                    "Available distortion correction modes: " +
                            (distortionModes?.contentToString() ?: "Not available")
                )

                // Sensor orientation relative to the device.
                val sensorOrientation = characteristics.get(
                    CameraCharacteristics.SENSOR_ORIENTATION
                )

                Log.d(
                    "DA3_INTRINSICS",
                    "Selected camera sensor orientation: $sensorOrientation"
                )

                if (intrinsics != null && intrinsics.size == 5) {

                    Log.d(
                        "DA3_INTRINSICS",
                        "Selected camera calibration: " +
                                "fx=${intrinsics[0]}, " +
                                "fy=${intrinsics[1]}, " +
                                "cx=${intrinsics[2]}, " +
                                "cy=${intrinsics[3]}, " +
                                "skew=${intrinsics[4]}"
                    )

                } else {

                    Log.w(
                        "DA3_INTRINSICS",
                        "Selected camera $cameraId does not expose intrinsic calibration"
                    )
                }

                Log.d(
                    "DA3_INTRINSICS",
                    "CameraX selected camera ID: $cameraId"
                )

                Log.d(
                    "DA3_CAMERA",
                    "CameraX live preview started successfully"
                )

            } catch (e: Exception) {

                Log.e(
                    "DA3_CAMERA",
                    "Failed to start CameraX preview",
                    e
                )

                Toast.makeText(
                    this,
                    "Camera preview failed: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun inspectCameraIntrinsics() {

        try {
            val cameraManager =
                getSystemService(CAMERA_SERVICE) as CameraManager

            for (cameraId in cameraManager.cameraIdList) {

                val characteristics =
                    cameraManager.getCameraCharacteristics(cameraId)

                val lensFacing = characteristics.get(
                    CameraCharacteristics.LENS_FACING
                )

                // Inspect rear-facing cameras only.
                if (lensFacing != CameraCharacteristics.LENS_FACING_BACK) {
                    continue
                }

                val intrinsics = characteristics.get(
                    CameraCharacteristics.LENS_INTRINSIC_CALIBRATION
                )

                Log.d(
                    "DA3_INTRINSICS",
                    "Rear camera ID: $cameraId"
                )

                if (intrinsics != null && intrinsics.size == 5) {

                    Log.d(
                        "DA3_INTRINSICS",
                        "Sensor intrinsics: " +
                                "fx=${intrinsics[0]}, " +
                                "fy=${intrinsics[1]}, " +
                                "cx=${intrinsics[2]}, " +
                                "cy=${intrinsics[3]}, " +
                                "skew=${intrinsics[4]}"
                    )

                } else {

                    Log.w(
                        "DA3_INTRINSICS",
                        "Camera $cameraId does not provide " +
                                "LENS_INTRINSIC_CALIBRATION"
                    )
                }

                val activeArray = characteristics.get(
                    CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE
                )

                Log.d(
                    "DA3_INTRINSICS",
                    "Sensor active array: $activeArray"
                )

                val preCorrectionArray = characteristics.get(
                    CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE
                )

                Log.d(
                    "DA3_INTRINSICS",
                    "Sensor pre-correction active array: $preCorrectionArray"
                )
            }

        } catch (e: Exception) {

            Log.e(
                "DA3_INTRINSICS",
                "Failed to inspect camera intrinsics",
                e
            )
        }
    }
}