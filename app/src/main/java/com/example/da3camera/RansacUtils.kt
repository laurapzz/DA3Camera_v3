package com.example.da3camera

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.PI
import kotlin.math.sqrt
import kotlin.random.Random

data class PlaneResult(
    val a: Float,
    val b: Float,
    val c: Float,
    val d: Float,
    val inlierRatio: Float,
    val tiltDeg: Float,
    val rollDeg: Float,
    val pitchDeg: Float
)

object RansacUtils {

    data class Point3D(val x: Float, val y: Float, val z: Float)
    class Vertex(val point: Point3D, val color: IntArray)

    fun backprojectPoint(
        u: Float,
        v: Float,
        depth: Float,
        intrinsics: CameraIntrinsics
    ): Point3D? {
        if (depth <= 0f || !depth.isFinite()) return null
        val x = -(u - intrinsics.cx) * depth / intrinsics.fx
        val y = -(v - intrinsics.cy) * depth / intrinsics.fy
        val z = depth
        return Point3D(x, y, z)
    }

    fun estimateGroundPlane(
        metricDepth: FloatArray,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics,
        maxIters: Int = 400,
        distThresh: Float = 0.05f // provar d'incrementar per contrarrestar imperfeccions a l'asfalt o error d'escala a depth
    ): PlaneResult {
        val points = ArrayList<Point3D>()
        val step = max(1, (width * height) / 10000)

        for (y in 0 until height step step) {
            for (x in 0 until width step step) {
                val idx = y * width + x
                if (idx < metricDepth.size) {
                    val d = metricDepth[idx]
                    val p = backprojectPoint(x.toFloat(), y.toFloat(), d, intrinsics)
                    if (p != null && p.z < 20.0f) {
                        points.add(p)
                    }
                }
            }
        }

        if (points.size < 3) {
            return PlaneResult(0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f)
        }

        val random = Random(42)
        var bestA = 0f
        var bestB = 1f
        var bestC = 0f
        var bestD = 0f
        var bestInliers = 0
        val nPoints = points.size

        for (iter in 0 until maxIters) {
            val i1 = random.nextInt(nPoints)
            val i2 = random.nextInt(nPoints)
            val i3 = random.nextInt(nPoints)
            if (i1 == i2 || i2 == i3 || i1 == i3) continue

            val p1 = points[i1]
            val p2 = points[i2]
            val p3 = points[i3]

            val v1x = p2.x - p1.x
            val v1y = p2.y - p1.y
            val v1z = p2.z - p1.z

            val v2x = p3.x - p1.x
            val v2y = p3.y - p1.y
            val v2z = p3.z - p1.z

            var nx = v1y * v2z - v1z * v2y
            var ny = v1z * v2x - v1x * v2z
            var nz = v1x * v2y - v1y * v2x

            val len = sqrt(nx * nx + ny * ny + nz * nz)
            if (len < 1e-6f) continue
            nx /= len
            ny /= len
            nz /= len

            val d = -(nx * p1.x + ny * p1.y + nz * p1.z)

            var inliers = 0
            for (p in points) {
                val dist = abs(nx * p.x + ny * p.y + nz * p.z + d)
                if (dist < distThresh) {
                    inliers++
                }
            }

            if (inliers > bestInliers) {
                bestInliers = inliers
                bestA = nx
                bestB = ny
                bestC = nz
                bestD = d
            }
        }

        val inlierRatio = if (nPoints > 0) bestInliers.toFloat() / nPoints.toFloat() else 0f

        val normalLen = sqrt(bestA * bestA + bestB * bestB + bestC * bestC)
        val ax = if (normalLen > 0f) bestA / normalLen else 0f
        val ay = if (normalLen > 0f) bestB / normalLen else 0f
        val az = if (normalLen > 0f) bestC / normalLen else 0f

        val dot = (ax * 0f + ay * 0f + az * 1f).coerceIn(-1f, 1f)
        val tiltRad = acos(abs(dot))
        val tiltDeg = (tiltRad * 180f / PI).toFloat()

        val rollDeg = (atan2(ax, az) * 180f / PI).toFloat()
        val pitchDeg = (atan2(ay, az) * 180f / PI).toFloat()

        return PlaneResult(
            a = bestA,
            b = bestB,
            c = bestC,
            d = bestD,
            inlierRatio = inlierRatio,
            tiltDeg = tiltDeg,
            rollDeg = rollDeg,
            pitchDeg = pitchDeg
        )
    }

    fun exportPly(
        metricDepth: FloatArray,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics,
        plane: PlaneResult,
        distThresh: Float = 0.05f
    ): ByteArray {
        val vertices = ArrayList<Vertex>()
        val step = max(1, (width * height) / 40000)

        for (y in 0 until height step step) {
            for (x in 0 until width step step) {
                val idx = y * width + x
                if (idx < metricDepth.size) {
                    val d = metricDepth[idx]
                    val p = backprojectPoint(x.toFloat(), y.toFloat(), d, intrinsics)
                    if (p != null && p.z < 20.0f) {
                        val dist = abs(plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d)
                        val color = if (dist < distThresh) {
                            intArrayOf(25, 204, 25)
                        } else {
                            intArrayOf(230, 25, 25)
                        }
                        vertices.add(Vertex(p, color))
                    }
                }
            }
        }

        val sb = StringBuilder()
        sb.append("ply\n")
        sb.append("format ascii 1.0\n")
        sb.append("element vertex ${vertices.size}\n")
        sb.append("property float x\n")
        sb.append("property float y\n")
        sb.append("property float z\n")
        sb.append("property uchar red\n")
        sb.append("property uchar green\n")
        sb.append("property uchar blue\n")
        sb.append("end_header\n")

        for (v in vertices) {
            sb.append("${v.point.x} ${v.point.y} ${v.point.z} ${v.color[0]} ${v.color[1]} ${v.color[2]}\n")
        }

        return sb.toString().toByteArray(Charsets.UTF_8)
    }
}
