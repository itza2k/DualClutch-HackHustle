package com.itza2k.transiq

import retrofit2.Call
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Body

interface ApiService {
    @GET("/predict")
    fun getPredictions(): Call<List<Double>>

    @POST("/update")
    fun updateModel(@Body newData: List<Map<String, Any>>): Call<Map<String, String>>
}