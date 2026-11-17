package com.example.da3camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log

object Ufldv2Utils {
    private const val TAG = "UFLDV2"
    private const val DETECTION_LINE_POINT = 3222
    private var laneDetector: LaneDetector? = null

    fun initialize(context: Context) {
        if (laneDetector != null) return
        try {
            laneDetector = LaneDetector(context)
            Log.d(TAG, "LaneDetector initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LaneDetector", e)
        }
    }

    data class LaneResult(
        val maskBytes: ByteArray,
        val leftLane: ArrayList<PointF>,
        val rightLane: ArrayList<PointF>,
        val leftBottomX: Float,
        val rightBottomX: Float,
        val detectedLeftLane: List<PointF> = leftLane,
        val detectedRightLane: List<PointF> = rightLane
    )

    fun generateLaneResult(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): LaneResult {
        val detector = laneDetector ?: run {
            Log.w(TAG, "LaneDetector was null during generateLaneResult, attempting lazy init")
            return LaneResult(
                ByteArray(targetWidth * targetHeight),
                ArrayList(),
                ArrayList(),
                0f,
                0f,
                emptyList(),
                emptyList()
            )
        }

        val argbBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap.copy(Bitmap.Config.ARGB_8888, true)
        } else {
            bitmap
        }

        var detectionCrop: Bitmap? = null
        try {
            val cropHeight = minOf(DETECTION_LINE_POINT, argbBitmap.height)
            detectionCrop = Bitmap.createBitmap(
                argbBitmap,
                0,
                0,
                argbBitmap.width,
                cropHeight
            )

            val detectionResult = detector.detectLanes(detectionCrop)
            val leftPts = detectionResult.leftLane
            val rightPts = detectionResult.rightLane
            val rawLeft = detectionResult.rawLeftLane
            val rawRight = detectionResult.rawRightLane

            if (leftPts.isEmpty() || rightPts.isEmpty()) {
                val emptyList = ArrayList<PointF>()
                return LaneResult(
                    ByteArray(targetWidth * targetHeight),
                    emptyList,
                    emptyList,
                    0f,
                    0f,
                    rawLeft,
                    rawRight
                )
            }

            val roiResult = RoiMaskGenerator.generateLaneRoi(argbBitmap, leftPts, rightPts)
            val maskBytes = if (roiResult != null) {
                bitmapToByteArray(roiResult.binaryMask, targetWidth, targetHeight)
            } else {
                ByteArray(targetWidth * targetHeight)
            }

            val leftArrayList = ArrayList(leftPts)
            val rightArrayList = ArrayList(rightPts)
            val leftBottomX = leftPts.maxByOrNull { it.y }?.x ?: 0f
            val rightBottomX = rightPts.maxByOrNull { it.y }?.x ?: 0f

            return LaneResult(
                maskBytes = maskBytes,
                leftLane = leftArrayList,
                rightLane = rightArrayList,
                leftBottomX = leftBottomX,
                rightBottomX = rightBottomX,
                detectedLeftLane = rawLeft,
                detectedRightLane = rawRight
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error in generateLaneResult using LaneDetector", e)
            val emptyList = ArrayList<PointF>()
            return LaneResult(
                ByteArray(targetWidth * targetHeight),
                emptyList,
                emptyList,
                0f,
                0f,
                emptyList,
                emptyList
            )
        } finally {
            if (detectionCrop !== argbBitmap && detectionCrop != null) {
                detectionCrop.recycle()
            }
        }
    }

    fun generateRoiMask(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): ByteArray {
        return generateLaneResult(bitmap, targetWidth, targetHeight).maskBytes
    }

    fun generateRoiMaskFromLanes(
        bitmap: Bitmap,
        leftPts: List<PointF>,
        rightPts: List<PointF>,
        targetWidth: Int,
        targetHeight: Int
    ): ByteArray {
        if (leftPts.isEmpty() || rightPts.isEmpty()) {
            return ByteArray(targetWidth * targetHeight)
        }
        val roiResult = RoiMaskGenerator.generateLaneRoi(bitmap, leftPts, rightPts)
        return if (roiResult != null) {
            bitmapToByteArray(roiResult.binaryMask, targetWidth, targetHeight)
        } else {
            ByteArray(targetWidth * targetHeight)
        }
    }

    private fun bitmapToByteArray(maskBitmap: Bitmap, targetWidth: Int, targetHeight: Int): ByteArray {
        val scaled = if (maskBitmap.width != targetWidth || maskBitmap.height != targetHeight) {
            Bitmap.createScaledBitmap(maskBitmap, targetWidth, targetHeight, false)
        } else {
            maskBitmap
        }

        val bytes = ByteArray(targetWidth * targetHeight)
        val pixels = IntArray(targetWidth * targetHeight)
        scaled.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            bytes[i] = if (r > 128) 0xFF.toByte() else 0
        }

        if (scaled !== maskBitmap) {
            scaled.recycle()
        }
        return bytes
    }
}
