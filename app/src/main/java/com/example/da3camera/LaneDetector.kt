package com.example.da3camera

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.exp

data class LaneDetectionResult(
    val leftLane: List<PointF>,
    val rightLane: List<PointF>,
    val rawLeftLane: List<PointF> = emptyList(),
    val rawRightLane: List<PointF> = emptyList()
)

class LaneDetector(context: Context) : AutoCloseable {

    companion object {
        private const val MODEL_FILE = "tusimple_res18.onnx"

        private const val INPUT_WIDTH = 800
        private const val INPUT_HEIGHT = 320

        private const val NUM_CLS_ROW = 56
        private const val NUM_LANES = 4

        private const val LEFT_EGO_LANE_INDEX = 1
        private const val RIGHT_EGO_LANE_INDEX = 2

        private const val EXISTENCE_THRESHOLD = 0.95f
        private const val LOCAL_WIDTH = 1

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // Exact TuSimple anchors from common.py:
        // np.linspace(160, 710, 56) / 720
        private val ROW_ANCHORS = FloatArray(NUM_CLS_ROW) { i ->
            ((160.0 + i * (710.0 - 160.0) / (NUM_CLS_ROW - 1)) / 720.0).toFloat()
        }
    }

    private val ortEnv: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private val ortSession: OrtSession

    init {
        val modelFile = prepareModelFile(context)
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(2)
        }
        ortSession = ortEnv.createSession(modelFile.absolutePath, sessionOptions)
    }

    private fun prepareModelFile(context: Context): File {
        val destination = File(context.filesDir, MODEL_FILE)
        val assetFd = try {
            context.assets.openFd(MODEL_FILE)
        } catch (e: Exception) {
            context.assets.open(MODEL_FILE).use { input ->
                destination.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            return destination
        }
        val assetSize = assetFd.length
        assetFd.close()

        if (destination.exists() && destination.length() == assetSize) {
            return destination
        }

        context.assets.open(MODEL_FILE).use { input ->
            destination.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return destination
    }

    /**
     * The input bitmap must already have been cropped as in Python:
     *
     * cropped_frame = frame[:LINE_POINT, :]
     *
     * Coordinates returned by this function are relative to that cropped
     * image. Since the crop starts at y=0, they also retain the original
     * image's horizontal and vertical coordinate origin.
     */
    fun detectLanes(croppedBitmap: Bitmap): LaneDetectionResult {

        require(croppedBitmap.width > 0 && croppedBitmap.height > 0) {
            "Input bitmap must have positive dimensions."
        }

        val imageWidth = croppedBitmap.width
        val imageHeight = croppedBitmap.height

        preprocess(croppedBitmap).use { inputTensor ->

            ortSession.run(mapOf("input" to inputTensor)).use { results ->

                val locTensor =
                    results.get("loc_row").get() as OnnxTensor

                val existTensor =
                    results.get("exist_row").get() as OnnxTensor

                @Suppress("UNCHECKED_CAST")
                val locRow =
                    (locTensor.value as Array<Array<Array<FloatArray>>>)[0]

                @Suppress("UNCHECKED_CAST")
                val existRow =
                    (existTensor.value as Array<Array<Array<FloatArray>>>)[0]

                validateOutputShapes(locRow, existRow)

                val rawLeft = decodeLane(
                    locRow = locRow,
                    existRow = existRow,
                    laneIndex = LEFT_EGO_LANE_INDEX,
                    imageWidth = imageWidth,
                    imageHeight = imageHeight
                )

                val rawRight = decodeLane(
                    locRow = locRow,
                    existRow = existRow,
                    laneIndex = RIGHT_EGO_LANE_INDEX,
                    imageWidth = imageWidth,
                    imageHeight = imageHeight
                )

                val targetY = intArrayOf(
                    imageHeight - 10,
                    (imageHeight * 0.85).toInt(),
                    (imageHeight * 0.70).toInt(),
                    (imageHeight * 0.55).toInt(),
                    (imageHeight * 0.40).toInt(),
                    (imageHeight * 0.25).toInt(),
                    (imageHeight * 0.15).toInt()
                )

                val fittedLeft =
                    fitLikePythonLineFit(rawLeft, targetY)

                val fittedRight =
                    fitLikePythonLineFit(rawRight, targetY)

                return LaneDetectionResult(
                    leftLane = fittedLeft,
                    rightLane = fittedRight,
                    rawLeftLane = rawLeft,
                    rawRightLane = rawRight
                )
            }
        }
    }

    private fun validateOutputShapes(
        locRow: Array<Array<FloatArray>>,
        existRow: Array<Array<FloatArray>>
    ) {

        require(locRow.size > 1) {
            "Invalid loc_row grid dimension: ${locRow.size}"
        }

        require(locRow[0].size == NUM_CLS_ROW) {
            "Unexpected loc_row anchor dimension: ${locRow[0].size}"
        }

        require(locRow[0][0].size == NUM_LANES) {
            "Unexpected loc_row lane dimension: ${locRow[0][0].size}"
        }

        require(existRow.size == 2) {
            "Unexpected exist_row class dimension: ${existRow.size}"
        }

        require(existRow[0].size == NUM_CLS_ROW) {
            "Unexpected exist_row anchor dimension: ${existRow[0].size}"
        }

        require(existRow[0][0].size == NUM_LANES) {
            "Unexpected exist_row lane dimension: ${existRow[0][0].size}"
        }
    }

    /**
     * Equivalent to the row-lane section of Python pred2coords().
     */
    private fun decodeLane(
        locRow: Array<Array<FloatArray>>,
        existRow: Array<Array<FloatArray>>,
        laneIndex: Int,
        imageWidth: Int,
        imageHeight: Int
    ): List<PointF> {

        val numGrid = locRow.size
        val numAnchors = locRow[0].size

        var validAnchorCount = 0

        for (rowIndex in 0 until numAnchors) {

            val noLaneLogit =
                existRow[0][rowIndex][laneIndex]

            val hasLaneLogit =
                existRow[1][rowIndex][laneIndex]

            if (hasLaneLogit > noLaneLogit) {
                validAnchorCount++
            }
        }

        if (validAnchorCount <= numAnchors / 2) {
            return emptyList()
        }

        val points = ArrayList<PointF>(numAnchors)

        for (rowIndex in 0 until numAnchors) {

            val noLaneLogit =
                existRow[0][rowIndex][laneIndex]

            val hasLaneLogit =
                existRow[1][rowIndex][laneIndex]

            val maxExistLogit =
                maxOf(noLaneLogit, hasLaneLogit)

            val noLaneExp =
                exp(noLaneLogit - maxExistLogit)

            val hasLaneExp =
                exp(hasLaneLogit - maxExistLogit)

            val hasLaneProbability =
                hasLaneExp / (noLaneExp + hasLaneExp)

            if (hasLaneProbability <= EXISTENCE_THRESHOLD) {
                continue
            }

            var maxGridIndex = 0
            var maxGridLogit = Float.NEGATIVE_INFINITY

            for (gridIndex in 0 until numGrid) {

                val value =
                    locRow[gridIndex][rowIndex][laneIndex]

                if (value > maxGridLogit) {
                    maxGridLogit = value
                    maxGridIndex = gridIndex
                }
            }

            val start =
                (maxGridIndex - LOCAL_WIDTH).coerceAtLeast(0)

            val end =
                (maxGridIndex + LOCAL_WIDTH).coerceAtMost(numGrid - 1)

            var expSum = 0.0
            var weightedGridSum = 0.0

            for (gridIndex in start..end) {

                val logit =
                    locRow[gridIndex][rowIndex][laneIndex]

                val value = exp(
                    (logit - maxGridLogit).toDouble()
                )

                expSum += value
                weightedGridSum += value * gridIndex
            }

            if (expSum <= 0.0) {
                continue
            }

            val expectedGrid =
                weightedGridSum / expSum + 0.5

            val x =
                expectedGrid / (numGrid - 1).toDouble() * imageWidth

            val y =
                (ROW_ANCHORS[rowIndex] * imageHeight).toInt()

            points.add(
                PointF(x.toFloat(), y.toFloat())
            )
        }

        return points
    }

    private fun fitLikePythonLineFit(
        points: List<PointF>,
        targetY: IntArray
    ): List<PointF> {

        if (points.size < 2) {
            return points
        }

        val n = points.size.toDouble()

        var sumY = 0.0
        var sumX = 0.0
        var sumYY = 0.0
        var sumYX = 0.0

        for (point in points) {
            val x = point.x.toInt().toDouble()
            val y = point.y.toDouble()

            sumY += y
            sumX += x
            sumYY += y * y
            sumYX += y * x
        }

        val denominator =
            n * sumYY - sumY * sumY

        if (abs(denominator) < 1e-12) {
            return points
        }

        val slope =
            (n * sumYX - sumY * sumX) / denominator

        val intercept =
            (sumX - slope * sumY) / n

        val fitted = targetY.map { y ->

            val x =
                slope * y + intercept

            PointF(
                x.toFloat(),
                y.toFloat()
            )
        }

        if (fitted.any {
                !it.x.isFinite() || !it.y.isFinite()
            }
        ) {
            return points
        }

        return fitted
    }

    private fun preprocess(bitmap: Bitmap): OnnxTensor {

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            INPUT_WIDTH,
            400,
            true
        )

        val inputCrop = Bitmap.createBitmap(
            resized,
            0,
            80,
            INPUT_WIDTH,
            INPUT_HEIGHT
        )

        if (resized !== bitmap) {
            resized.recycle()
        }

        val pixels = IntArray(
            INPUT_WIDTH * INPUT_HEIGHT
        )

        inputCrop.getPixels(
            pixels,
            0,
            INPUT_WIDTH,
            0,
            0,
            INPUT_WIDTH,
            INPUT_HEIGHT
        )

        inputCrop.recycle()

        val channelSize =
            INPUT_WIDTH * INPUT_HEIGHT

        val floatBuffer =
            FloatBuffer.allocate(3 * channelSize)

        for (i in 0 until channelSize) {

            val pixel = pixels[i]

            val red =
                ((pixel shr 16) and 0xFF) / 255.0f

            val green =
                ((pixel shr 8) and 0xFF) / 255.0f

            val blue =
                (pixel and 0xFF) / 255.0f

            floatBuffer.put(
                i,
                (red - MEAN[0]) / STD[0]
            )

            floatBuffer.put(
                channelSize + i,
                (green - MEAN[1]) / STD[1]
            )

            floatBuffer.put(
                2 * channelSize + i,
                (blue - MEAN[2]) / STD[2]
            )
        }

        floatBuffer.rewind()

        val shape = longArrayOf(
            1,
            3,
            INPUT_HEIGHT.toLong(),
            INPUT_WIDTH.toLong()
        )

        return OnnxTensor.createTensor(
            ortEnv,
            floatBuffer,
            shape
        )
    }

    override fun close() {
        ortSession.close()
    }
}
