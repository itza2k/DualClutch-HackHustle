package com.dualclutch.hackhustle

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.dualclutch.hackhustle.screens.HomeScreen
import com.dualclutch.hackhustle.screens.ScheduleScreen
import com.dualclutch.hackhustle.screens.SettingsScreen
import com.dualclutch.hackhustle.ui.theme.TransIQTheme
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TransIQTheme {
                MainScreen()
            }
        }
    }
}

@Composable
fun MainScreen() {
    val navController = rememberNavController()
    val coroutineScope = rememberCoroutineScope()

    Scaffold(
        bottomBar = {
            NavigationBar {
                val items = listOf("Home", "Schedule", "Settings")
                val icons = listOf(Icons.Filled.Home, Icons.Filled.DateRange, Icons.Filled.Settings)
                items.forEachIndexed { index, item ->
                    NavigationBarItem(
                        icon = { Icon(icons[index], contentDescription = item) },
                        label = { Text(item) },
                        selected = false,
                        onClick = {
                            coroutineScope.launch {
                                navController.navigate(item)
                            }
                        }
                    )
                }
            }
        },
        modifier = Modifier.fillMaxSize()
    ) { innerPadding ->
        NavHost(navController, startDestination = "Home", Modifier.padding(innerPadding)) {
            composable("Home") { HomeScreen() }
            composable("Schedule") { ScheduleScreen() }
            composable("Settings") { SettingsScreen() }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
    TransIQTheme {
        MainScreen()
    }
}