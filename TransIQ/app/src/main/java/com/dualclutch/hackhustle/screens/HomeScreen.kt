package com.dualclutch.hackhustle.screens

import android.Manifest
import android.location.Location
import androidx.activity.ComponentActivity
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import com.dualclutch.hackhustle.utils.LocationUtils

@Composable
fun HomeScreen() {
    val context = LocalContext.current
    var location by remember { mutableStateOf<Location?>(null) }
    val locationUtils = remember { LocationUtils(context) }

    LaunchedEffect(Unit) {
        ActivityCompat.requestPermissions(
            context as ComponentActivity,
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
            1
        )
        locationUtils.getLastLocation { loc ->
            location = loc
        }
    }

    val recentActivities = remember { listOf("Activity 1", "Activity 2", "Activity 3") }
    val notifications = remember { listOf("Notification 1", "Notification 2") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Welcome to HackHustle",
            style = MaterialTheme.typography.headlineLarge,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        Text(
            text = "Hello, User!",
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        location?.let {
            Text(
                text = "Location: ${it.latitude}, ${it.longitude}",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(bottom = 16.dp)
            )
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(onClick = { /* TODO: Navigate to Schedule */ }) {
                Text(text = "Schedule")
            }
            Button(onClick = { /* TODO: Navigate to Settings */ }) {
                Text(text = "Settings")
            }
        }
        Text(
            text = "Recent Activities",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        ) {
            items(recentActivities) { activity ->
                Text(
                    text = activity,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }
        }
        Text(
            text = "Notifications",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        LazyColumn(
            modifier = Modifier.fillMaxWidth()
        ) {
            items(notifications) { notification ->
                Text(
                    text = notification,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }
        }
    }
}