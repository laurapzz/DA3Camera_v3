package com.example.da3camera

import android.content.Context
import android.util.Log
import java.io.File
import org.pytorch.executorch.Module

object DA3ModelLoader {

    private const val TAG = "DA3_MODEL"
    private const val MODEL_NAME = "da3_baseline72_fixed504.pte"

    fun loadModel(modelFile: File): Module {

        Log.d(TAG, "Starting ExecuTorch model loading...")

        val startTime = System.currentTimeMillis()

        val module = Module.load(
            modelFile.absolutePath
        )

        val elapsedSeconds =
            (System.currentTimeMillis() - startTime) / 1000.0

        Log.d(
            TAG,
            "ExecuTorch model loaded successfully!"
        )

        Log.d(
            TAG,
            "Model loading time: $elapsedSeconds seconds"
        )

        return module
    }

    fun prepareModelFile(context: Context): File {

        val destination = File(
            context.filesDir,
            MODEL_NAME
        )

        // Reuse the file if it has already been copied.
        val assetSize = context.assets.openFd(MODEL_NAME).use {
            it.length
        }

        if (destination.exists() &&
            destination.length() == assetSize
        ) {
            Log.d(TAG, "Model file already prepared")
            return destination
        }

        Log.d(TAG, "Copying model from assets...")

        // Copy through a temporary file to avoid leaving a
        // partially copied model if the operation fails.
        val temporaryFile = File(
            context.filesDir,
            "$MODEL_NAME.tmp"
        )

        try {
            context.assets.open(MODEL_NAME).use { input ->
                temporaryFile.outputStream().buffered().use { output ->
                    input.copyTo(output, bufferSize = 1024 * 1024)
                }
            }

            if (temporaryFile.length() != assetSize) {
                error("Model copy size mismatch")
            }

            if (destination.exists()) {
                destination.delete()
            }

            check(temporaryFile.renameTo(destination)) {
                "Could not finalize model copy"
            }

        } catch (e: Exception) {
            temporaryFile.delete()
            throw e
        }

        Log.d(
            TAG,
            "Model prepared: ${destination.length()} bytes"
        )

        return destination
    }
}