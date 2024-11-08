package com.hh.flowsync.data.network

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

interface ApiService {

    @POST("predict")
    suspend fun getPrediction(@Body features: Map<String, Any>): Response<List<Float>>

    // Add other API endpoints as needed
}
