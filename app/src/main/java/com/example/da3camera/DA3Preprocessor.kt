package com.example.da3camera

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.roundToInt

data class CameraIntrinsics(
    val fx: Float,
    val fy: Float,
    val cx: Float,
    val cy: Float
)

data class DA3PreprocessResult(
    val inputTensor: FloatArray,
    val correctedIntrinsics: CameraIntrinsics,
    val originalWidth: Int,
    val originalHeight: Int,
    val processedWidth: Int,
    val processedHeight: Int,
    val padLeft: Int,
    val padRight: Int,
    val padTop: Int,
    val padBottom: Int
)

object DA3Preprocessor {

    private const val INPUT_SIZE = 504
    private const val PATCH_SIZE = 14

    private val MEAN = floatArrayOf(
        0.485f, 0.456f, 0.406f
    )

    private val STD = floatArrayOf(
        0.229f, 0.224f, 0.225f
    )

    private fun nearestMultiple(value: Int): Int {
        val down = (value / PATCH_SIZE) * PATCH_SIZE
        val up = down + PATCH_SIZE

        return if (abs(up - value) <= abs(value - down)) {
            up
        } else {
            down
        }
    }

    fun preprocess(
        bitmap: Bitmap,
        intrinsics: CameraIntrinsics
    ): DA3PreprocessResult {

        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        // Convert Android Bitmap to OpenCV Mat.
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)

        val rgb = Mat()
        Imgproc.cvtColor(
            rgba,
            rgb,
            Imgproc.COLOR_RGBA2RGB
        )

        rgba.release()

        // --------------------------------------------------
        // STEP 1: Resize longest side to 504
        // --------------------------------------------------

        val longestSide = maxOf(originalWidth, originalHeight)

        val scale = INPUT_SIZE.toDouble() / longestSide

        val resizedWidth =
            (originalWidth * scale).roundToInt()

        val resizedHeight =
            (originalHeight * scale).roundToInt()

        val resized = Mat()

        val interpolation1 =
            if (scale < 1.0) {
                Imgproc.INTER_AREA
            } else {
                Imgproc.INTER_CUBIC
            }

        Imgproc.resize(
            rgb,
            resized,
            Size(
                resizedWidth.toDouble(),
                resizedHeight.toDouble()
            ),
            0.0,
            0.0,
            interpolation1
        )

        rgb.release()

        // Update intrinsics after first resize.
        var fx = intrinsics.fx *
                resizedWidth.toFloat() / originalWidth

        var fy = intrinsics.fy *
                resizedHeight.toFloat() / originalHeight

        var cx = intrinsics.cx *
                resizedWidth.toFloat() / originalWidth

        var cy = intrinsics.cy *
                resizedHeight.toFloat() / originalHeight

        // --------------------------------------------------
        // STEP 2: Resize to multiples of patch size (14)
        // --------------------------------------------------

        val processedWidth =
            nearestMultiple(resizedWidth)

        val processedHeight =
            nearestMultiple(resizedHeight)

        val aligned = Mat()

        val interpolation2 =
            if (
                processedWidth > resizedWidth ||
                processedHeight > resizedHeight
            ) {
                Imgproc.INTER_CUBIC
            } else {
                Imgproc.INTER_AREA
            }

        Imgproc.resize(
            resized,
            aligned,
            Size(
                processedWidth.toDouble(),
                processedHeight.toDouble()
            ),
            0.0,
            0.0,
            interpolation2
        )

        resized.release()

        // Update intrinsics after second resize.
        fx *= processedWidth.toFloat() / resizedWidth
        fy *= processedHeight.toFloat() / resizedHeight
        cx *= processedWidth.toFloat() / resizedWidth
        cy *= processedHeight.toFloat() / resizedHeight

        // --------------------------------------------------
        // STEP 3: Calculate zero-padding
        // --------------------------------------------------

        val padLeft =
            (INPUT_SIZE - processedWidth) / 2

        val padRight =
            INPUT_SIZE - processedWidth - padLeft

        val padTop =
            (INPUT_SIZE - processedHeight) / 2

        val padBottom =
            INPUT_SIZE - processedHeight - padTop

        // --------------------------------------------------
        // STEP 4: Convert to normalized NCHW tensor
        // --------------------------------------------------

        val inputTensor = FloatArray(
            1 * 3 * INPUT_SIZE * INPUT_SIZE
        )

        val pixelData = ByteArray(
            processedWidth * processedHeight * 3
        )

        aligned.get(0, 0, pixelData)
        aligned.release()

        val channelSize = INPUT_SIZE * INPUT_SIZE

        for (y in 0 until processedHeight) {
            for (x in 0 until processedWidth) {

                val pixelIndex =
                    (y * processedWidth + x) * 3

                val tensorY = y + padTop
                val tensorX = x + padLeft

                val tensorIndex =
                    tensorY * INPUT_SIZE + tensorX

                for (c in 0 until 3) {

                    val value =
                        (pixelData[pixelIndex + c].toInt() and 0xFF) / 255.0f

                    val normalized =
                        (value - MEAN[c]) / STD[c]

                    inputTensor[
                        c * channelSize + tensorIndex
                    ] = normalized
                }
            }
        }

        // Padding stays zero because FloatArray is initialized with zeros.

        // --------------------------------------------------
        // STEP 5: Correct intrinsics for padding
        // --------------------------------------------------

        cx += padLeft
        cy += padTop

        val correctedIntrinsics = CameraIntrinsics(
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy
        )

        return DA3PreprocessResult(
            inputTensor = inputTensor,
            correctedIntrinsics = correctedIntrinsics,
            originalWidth = originalWidth,
            originalHeight = originalHeight,
            processedWidth = processedWidth,
            processedHeight = processedHeight,
            padLeft = padLeft,
            padRight = padRight,
            padTop = padTop,
            padBottom = padBottom
        )
    }
}
