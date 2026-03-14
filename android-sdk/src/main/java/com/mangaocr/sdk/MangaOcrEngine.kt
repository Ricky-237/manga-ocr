package com.mangaocr.sdk

import android.graphics.Bitmap

interface MangaOcrEngine {
    fun runPage(bitmap: Bitmap): List<OcrBlock>
}
