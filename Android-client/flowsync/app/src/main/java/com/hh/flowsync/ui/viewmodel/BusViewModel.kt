package com.hh.flowsync.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.liveData
import com.hh.flowsync.data.repository.BusRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class BusViewModel @Inject constructor(
    private val repository: BusRepository
) : ViewModel() {

    val buses = liveData {
        emit(repository.getBuses())
    }

    fun getPrediction(features: Map<String, Any>) = liveData {
        val prediction = repository.getPrediction(features)
        emit(prediction)
    }
}
