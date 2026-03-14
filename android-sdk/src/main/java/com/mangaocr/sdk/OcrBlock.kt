package com.mangaocr.sdk

data class OcrBlock(
    val polygon: List<OcrPoint>,
    val text: String,
    val lang: String,
    val direction: String,
    val confidence: Float,
    val orderIndex: Int,
)
