package com.mangaocr.sdk

import android.graphics.Bitmap

class OnnxMangaOcrEngine : MangaOcrEngine {
    override fun runPage(bitmap: Bitmap): List<OcrBlock> {
        throw UnsupportedOperationException(
            "ONNX Runtime Mobile integration is not wired yet. " +
                "This scaffold preserves the stable Android API while the model export pipeline is built."
        )
    }
}
