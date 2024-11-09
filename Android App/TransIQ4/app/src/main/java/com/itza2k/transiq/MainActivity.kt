package com.itza2k.transiq

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.itza2k.transiq.ui.theme.TransIQTheme
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            TransIQTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    PredictionScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun PredictionScreen(modifier: Modifier = Modifier) {
    var predictions by remember { mutableStateOf<List<Double>?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        RetrofitInstance.apiService.getPredictions().enqueue(object : Callback<List<Double>> {
            override fun onResponse(call: Call<List<Double>>, response: Response<List<Double>>) {
                if (response.isSuccessful) {
                    predictions = response.body()
                } else {
                    errorMessage = "Failed to get predictions"
                }
            }

            override fun onFailure(call: Call<List<Double>>, t: Throwable) {
                errorMessage = t.message
            }
        })
    }

    Column(modifier = modifier.padding(16.dp)) {
        Text(text = "Predictions for Next Day", style = MaterialTheme.typography.h6)
        Spacer(modifier = Modifier.height(8.dp))
        if (predictions != null) {
            predictions!!.forEachIndexed { hour, prediction ->
                Text(text = "Hour $hour: $prediction passengers")
            }
        } else if (errorMessage != null) {
            Text(text = "Error: $errorMessage", color = MaterialTheme.colorScheme.error)
        } else {
            CircularProgressIndicator()
        }
    }
}

@Preview(showBackground = true)
@Composable
fun PredictionScreenPreview() {
    TransIQTheme {
        PredictionScreen()
    }
}