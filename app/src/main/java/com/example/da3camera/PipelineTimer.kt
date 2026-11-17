package com.example.da3camera

import android.util.Log
import java.util.Locale

/**
 * Records how long each step of a run takes.
 *
 * Every step is logged immediately under the "DA3_TIMING" tag, and [summary]
 * builds a readable table for the UI. Steps can be recorded from several
 * threads (lane detection runs in parallel with DA3).
 */
class PipelineTimer(private val runLabel: String) {

    data class Step(
        val name: String,
        val millis: Double,
        val parallel: Boolean,
        val outsideTotal: Boolean
    )

    private val startNs = System.nanoTime()
    private val steps = mutableListOf<Step>()

    fun <T> measure(name: String, parallel: Boolean = false, block: () -> T): T {
        val t0 = System.nanoTime()
        try {
            return block()
        } finally {
            record(name, (System.nanoTime() - t0) / 1e6, parallel)
        }
    }

    fun record(
        name: String,
        millis: Double,
        parallel: Boolean = false,
        outsideTotal: Boolean = false
    ) {
        synchronized(steps) {
            steps.add(Step(name, millis, parallel, outsideTotal))
        }
        Log.d(TAG, String.format(Locale.US, "[%s] %s: %.1f ms", runLabel, name, millis))
    }

    /** Wall-clock milliseconds since this timer was created. */
    fun elapsedMillis(): Double = (System.nanoTime() - startNs) / 1e6

    fun logTotal(): Double {
        val total = elapsedMillis()
        Log.d(TAG, String.format(Locale.US, "[%s] TOTAL: %.1f ms", runLabel, total))
        return total
    }

    fun summary(totalMillis: Double): String {
        val snapshot = synchronized(steps) { steps.toList() }
        val sb = StringBuilder()
        for (step in snapshot) {
            val suffix = when {
                step.outsideTotal -> "  (not in total)"
                step.parallel -> "  (parallel)"
                else -> ""
            }
            sb.append(String.format(Locale.US, "%-34s %8.0f ms%s\n", step.name, step.millis, suffix))
        }
        sb.append(String.format(Locale.US, "%-34s %8.0f ms", "TOTAL", totalMillis))
        return sb.toString()
    }

    companion object {
        const val TAG = "DA3_TIMING"
    }
}
