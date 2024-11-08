package com.hh.flowsync.ui.screens

import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.hilt.navigation.compose.hiltViewModel
import com.hh.flowsync.ui.viewmodel.BusViewModel

@Composable
fun PredictionsScreen(viewModel: BusViewModel = hiltViewModel()) {
    val predictions by viewModel.predictions.collectAsState(initial = emptyList())

    LazyColumn {
        items(predictions) { prediction ->
            Text("Prediction: $prediction", style = MaterialTheme.typography.body1)
        }
    }
}
