package com.hh.flowsync.data.model

data class Bus(
    val number: String,
    val route: String,
    val capacity: Int,
    val currentLoad: Int,
    val nextStop: String,
    val eta: String
)
