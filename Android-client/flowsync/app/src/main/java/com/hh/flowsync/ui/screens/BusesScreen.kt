package com.hh.flowsync.ui.screens

import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.runtime.Composable
import com.hh.flowsync.data.model.Bus
import com.hh.flowsync.ui.components.BusItem

@Composable
fun BusesScreen(busList: List<Bus>) {
    LazyColumn {
        items(busList) { bus ->
            BusItem(bus)
        }
    }
}
