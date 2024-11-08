package com.hh.flowsync.ui.screens

import androidx.compose.material.Scaffold
import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.hh.flowsync.ui.components.BottomNavBar

@Composable
fun MainScreen() {
    val navController = rememberNavController()
    Scaffold(
        bottomBar = { BottomNavBar(navController) }
    ) {
        NavHost(navController = navController, startDestination = "home") {
            composable("home") { HomeScreen() }
            composable("buses") { BusesScreen(busList = listOf()) } // Pass the actual bus list here
            composable("settings") { SettingsScreen() }
            composable("predictions") { PredictionsScreen() }
        }
    }
}
