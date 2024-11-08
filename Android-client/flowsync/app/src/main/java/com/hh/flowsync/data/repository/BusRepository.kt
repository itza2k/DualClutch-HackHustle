package com.hh.flowsync.data.repository

import com.hh.flowsync.data.network.ApiService
import com.hh.flowsync.data.model.Bus
import javax.inject.Inject

class BusRepository @Inject constructor(
    private val apiService: ApiService
) {
    suspend fun getPrediction(features: Map<String, Any>): List<Float>? {
        val response = apiService.getPrediction(features)
        return if (response.isSuccessful) {
            response.body()
        } else {
            null
        }
    }

    // Example method to get a list of buses
    fun getBuses(): List<Bus> {
        // This could be fetched from a local database or API
        return listOf(
            Bus("1A", "Route 1", 50, 30, "Next Stop 1", "10 mins"),
            Bus("2B", "Route 2", 60, 45, "Next Stop 2", "5 mins")
        )
    }
}
