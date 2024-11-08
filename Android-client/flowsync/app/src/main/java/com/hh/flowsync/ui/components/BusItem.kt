package com.hh.flowsync.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material.Card
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.hh.flowsync.data.model.Bus

@Composable
fun BusItem(bus: Bus) {
    Card(modifier = Modifier.padding(8.dp)) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Bus Number: ${bus.number}", style = MaterialTheme.typography.h6)
            Text("Route: ${bus.route}")
            Text("Capacity: ${bus.capacity}")
            Text("Current Load: ${bus.currentLoad}")
            Text("Next Stop: ${bus.nextStop}")
            Text("ETA: ${bus.eta}")
        }
    }
}
