package com.dualclutch.hackhustle.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.google.android.gms.maps.CameraUpdateFactory
import com.google.android.gms.maps.model.LatLng
import com.google.maps.android.compose.*
import com.google.android.gms.maps.model.CameraPosition

@Composable
fun ScheduleScreen() {
    val vehicles = remember { listOf("Bus 1", "Train A", "Bus 2") }
    val events = remember { listOf("Concert at 7 PM", "Road closure on 5th Ave") }
    val feedback = remember { mutableStateOf("") }
    val mapProperties = remember { mutableStateOf(MapProperties(mapType = MapType.NORMAL)) }
    val mapUiSettings = remember { mutableStateOf(MapUiSettings(zoomControlsEnabled = false)) }
    val cameraPositionState = rememberCameraPositionState {
        position = CameraPosition.fromLatLngZoom(LatLng(37.7749, -122.4194), 10f)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Schedule Screen",
            style = MaterialTheme.typography.headlineLarge,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        Text(
            text = "Real-Time Vehicle Tracking",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        GoogleMap(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp),
            properties = mapProperties.value,
            uiSettings = mapUiSettings.value,
            cameraPositionState = cameraPositionState
        ) {
            // Add markers for vehicles
            vehicles.forEach { vehicle ->
                Marker(
                    state = MarkerState(position = LatLng(37.7749, -122.4194)),
                    title = vehicle
                )
            }
        }
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Predictive Scheduling",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        ) {
            items(vehicles) { vehicle ->
                Text(
                    text = "$vehicle - Predicted Arrival: 5 mins",
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }
        }
        Text(
            text = "Event Alerts",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 16.dp)
        ) {
            items(events) { event ->
                Text(
                    text = event,
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }
        }
        Text(
            text = "User Feedback",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        OutlinedTextField(
            value = feedback.value,
            onValueChange = { feedback.value = it },
            label = { Text("Feedback") },
            modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)
        )
        Button(
            onClick = { /* TODO: Handle feedback submission */ },
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text(text = "Submit Feedback")
        }
    }
}