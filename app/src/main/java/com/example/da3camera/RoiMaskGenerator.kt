package com.example.da3camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF

data class RoiResult(
    val binaryMask: Bitmap,
    val visualOverlay: Bitmap,
    val polygonPoints: List<PointF>
)

object RoiMaskGenerator {

    /**
     * Kotlin implementation of generate_lane_roi()
     * from the Python app.py.
     *
     * Input coordinates must be in the original
     * frame's coordinate system.
     */
    fun generateLaneRoi(
        frame: Bitmap,
        leftPtsRaw: List<PointF>,
        rightPtsRaw: List<PointF>
    ): RoiResult? {

        if (leftPtsRaw.isEmpty() || rightPtsRaw.isEmpty()) {
            return null
        }

        val width = frame.width
        val height = frame.height

        if (width <= 0 || height <= 0) {
            return null
        }

        val leftPts = leftPtsRaw
            .filter { it.x.isFinite() && it.y.isFinite() }
            .map {
                PointF(
                    it.x.toInt().toFloat(),
                    it.y.toInt().toFloat()
                )
            }
            .sortedBy { it.y }

        val rightPts = rightPtsRaw
            .filter { it.x.isFinite() && it.y.isFinite() }
            .map {
                PointF(
                    it.x.toInt().toFloat(),
                    it.y.toInt().toFloat()
                )
            }
            .sortedBy { it.y }

        if (leftPts.isEmpty() || rightPts.isEmpty()) {
            return null
        }

        val collectedLeft = mutableListOf<PointF>()
        val collectedRight = mutableListOf<PointF>()
        for (i in (minOf(leftPts.size, rightPts.size) - 1) downTo 0) {
            val l = leftPts[i]
            val r = rightPts[i]
            if (l.x < r.x) {
                collectedLeft.add(l)
                collectedRight.add(r)
            } else {
                break
            }
        }

        if (collectedLeft.isEmpty() || collectedRight.isEmpty()) {
            return null
        }

        collectedLeft.reverse()
        collectedRight.reverse()

        val bottomLeftX = collectedLeft.last().x
        val bottomRightX = collectedRight.last().x

        val extBottomLeft = PointF(
            bottomLeftX,
            (height - 1).toFloat()
        )

        val extBottomRight = PointF(
            bottomRightX,
            (height - 1).toFloat()
        )

        val polygonPoints = ArrayList<PointF>()

        polygonPoints.addAll(collectedLeft)
        polygonPoints.add(extBottomLeft)
        polygonPoints.add(extBottomRight)
        polygonPoints.addAll(collectedRight.asReversed())

        val roiPath = Path().apply {

            moveTo(
                polygonPoints[0].x,
                polygonPoints[0].y
            )

            for (i in 1 until polygonPoints.size) {
                lineTo(
                    polygonPoints[i].x,
                    polygonPoints[i].y
                )
            }

            close()
        }

        val binaryMask = Bitmap.createBitmap(
            width,
            height,
            Bitmap.Config.ARGB_8888
        )

        val maskCanvas = Canvas(binaryMask)

        maskCanvas.drawColor(Color.BLACK)

        val maskPaint = Paint().apply {
            isAntiAlias = false
            color = Color.WHITE
            style = Paint.Style.FILL
        }

        maskCanvas.drawPath(
            roiPath,
            maskPaint
        )

        val visualOverlay = frame.copy(
            Bitmap.Config.ARGB_8888,
            true
        )

        val overlayCanvas = Canvas(visualOverlay)

        val fillPaint = Paint().apply {
            isAntiAlias = false
            color = Color.argb(
                115,
                100,
                200,
                0
            )
            style = Paint.Style.FILL
        }

        overlayCanvas.drawPath(
            roiPath,
            fillPaint
        )

        val borderPaint = Paint().apply {
            isAntiAlias = false
            color = Color.rgb(255, 0, 0)
            style = Paint.Style.STROKE
            strokeWidth = 2f
            strokeJoin = Paint.Join.ROUND
        }

        overlayCanvas.drawPath(
            roiPath,
            borderPaint
        )

        return RoiResult(
            binaryMask = binaryMask,
            visualOverlay = visualOverlay,
            polygonPoints = polygonPoints
        )
    }

    fun isObstacleInLane(
        binaryMask: Bitmap,
        box: RectF
    ): Boolean {

        if (
            binaryMask.width <= 0 ||
            binaryMask.height <= 0
        ) {
            return false
        }

        val contactX = box.centerX()
            .toInt()
            .coerceIn(
                0,
                binaryMask.width - 1
            )

        val contactY = box.bottom
            .toInt()
            .coerceIn(
                0,
                binaryMask.height - 1
            )

        val pixel = binaryMask.getPixel(
            contactX,
            contactY
        )

        return Color.red(pixel) > 128
    }
}
