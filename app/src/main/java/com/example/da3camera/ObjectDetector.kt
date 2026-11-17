package com.example.da3camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.sqrt

data class DetectedObject(
    val boundingBox: Rect, // In original image coordinates
    val distanceMeters: Float,
    val heightMeters: Float,
    val pixelCount: Int
)

object ObjectDetector {

    /**
     * Performs object detection in 504x504 model space (ground plane removed, connected components),
     * avoiding OutOfMemory errors on high-res photos, and scales the resulting bounding boxes
     * to original image resolution.
     */
    fun detectObjects504(
        metricDepthValues: FloatArray,
        modelIntrinsics: CameraIntrinsics,
        plane: PlaneResult,
        roiMaskBytes: ByteArray,
        originalWidth: Int,
        originalHeight: Int,
        groundDistThresh: Float = 0.15f,
        minBlobArea: Int = 30,
        maxHeightMeters: Float = 2.5f,
        minHeightMeters: Float = 0.2f
    ): List<DetectedObject> {
        val width = 504
        val height = 504
        val obstacleMask = Mat.zeros(height, width, CvType.CV_8UC1)
        val maskData = ByteArray(width * height)

        for (v in 0 until height) {
            for (u in 0 until width) {
                val idx = v * width + u
                val roiVal = if (idx < roiMaskBytes.size) roiMaskBytes[idx].toInt() and 0xFF else 0
                if (roiVal > 0 && idx < metricDepthValues.size) {
                    val d = metricDepthValues[idx]
                    val p = RansacUtils.backprojectPoint(u.toFloat(), v.toFloat(), d, modelIntrinsics)
                    if (p != null && p.z in 0.5f..15.0f) {
                        val distToPlane = abs(plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d)
                        if (distToPlane > groundDistThresh) {
                            maskData[idx] = 255.toByte()
                        }
                    }
                }
            }
        }
        obstacleMask.put(0, 0, maskData)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
        Imgproc.morphologyEx(obstacleMask, obstacleMask, Imgproc.MORPH_OPEN, kernel)
        Imgproc.morphologyEx(obstacleMask, obstacleMask, Imgproc.MORPH_CLOSE, kernel)

        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val numLabels = Imgproc.connectedComponentsWithStats(obstacleMask, labels, stats, centroids)

        val detectedObjects = mutableListOf<DetectedObject>()
        val labelsData = IntArray(width * height)
        labels.get(0, 0, labelsData)

        val scaleX = originalWidth.toFloat() / width.toFloat()
        val scaleY = originalHeight.toFloat() / height.toFloat()

        for (i in 1 until numLabels) {
            val left504 = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val top504 = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val w504 = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val h504 = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()

            if (area < minBlobArea) continue

            val componentPoints = mutableListOf<RansacUtils.Point3D>()
            val step = maxOf(1, (w504 * h504) / 200)
            for (v in top504 until (top504 + h504) step maxOf(1, step / 10)) {
                for (u in left504 until (left504 + w504) step maxOf(1, step / 10)) {
                    if (u in 0 until width && v in 0 until height) {
                        val idx = v * width + u
                        if (labelsData[idx] == i && idx < metricDepthValues.size) {
                            val d = metricDepthValues[idx]
                            val p = RansacUtils.backprojectPoint(u.toFloat(), v.toFloat(), d, modelIntrinsics)
                            if (p != null) {
                                componentPoints.add(p)
                            }
                        }
                    }
                }
            }

            if (componentPoints.isEmpty()) continue

            val distances = componentPoints.map { sqrt(it.x * it.x + it.y * it.y + it.z * it.z) }.sorted()
            val medianDist = distances[distances.size / 2]

            val ys = componentPoints.map { it.y }
            val objHeight = abs((ys.maxOrNull() ?: 0f) - (ys.minOrNull() ?: 0f))

            if (objHeight !in minHeightMeters..maxHeightMeters) {
                continue
            }

            val origLeft = (left504 * scaleX).toInt().coerceIn(0, originalWidth - 1)
            val origTop = (top504 * scaleY).toInt().coerceIn(0, originalHeight - 1)
            val origRight = ((left504 + w504) * scaleX).toInt().coerceIn(0, originalWidth)
            val origBottom = ((top504 + h504) * scaleY).toInt().coerceIn(0, originalHeight)

            if (origRight <= origLeft || origBottom <= origTop) continue

            detectedObjects.add(
                DetectedObject(
                    boundingBox = Rect(origLeft, origTop, origRight, origBottom),
                    distanceMeters = medianDist,
                    heightMeters = objHeight,
                    pixelCount = area
                )
            )
        }

        obstacleMask.release()
        labels.release()
        stats.release()
        centroids.release()
        kernel.release()

        return detectedObjects
    }

    fun drawObjectsOverlay(
        bitmap: Bitmap,
        objects: List<DetectedObject>
    ): Bitmap {
        val resultBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(resultBitmap)

        val paintRect = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 6f
            isAntiAlias = true
        }

        val paintTextBg = Paint().apply {
            color = Color.argb(220, 0, 0, 0)
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        val paintText = Paint().apply {
            color = Color.WHITE
            textSize = 32f
            isAntiAlias = true
            isFakeBoldText = true
        }

        for (obj in objects) {
            val rect = obj.boundingBox
            canvas.drawRect(rect.left.toFloat(), rect.top.toFloat(), rect.right.toFloat(), rect.bottom.toFloat(), paintRect)

            val infoText = "Dist: %.1fm | H: %.1fm".format(obj.distanceMeters, obj.heightMeters)
            val textX = rect.left.toFloat().coerceAtLeast(10f)
            val textY = (rect.top - 12f).coerceAtLeast(40f)

            canvas.drawRect(textX - 6f, textY - 32f, textX + 260f, textY + 10f, paintTextBg)
            canvas.drawText(infoText, textX, textY, paintText)
        }

        return resultBitmap
    }
}
