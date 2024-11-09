// src/services/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000/api', // Backend URL
});

// Interfaces for type safety
export interface DemandData {
  time: string;
  day: number;
  presentDemand: number | null;
  predictedDemand: number | null;
}

export interface ConditionsUpdate {
  rain: 0 | 1;
  event: 0 | 1;
  passengers_bus_stop_1?: number;
  passengers_bus_stop_2?: number;
  passengers_bus_stop_3?: number;
  passengers_bus_stop_4?: number;
}

export interface CurrentConditions {
  rain: 0 | 1;
  event: 0 | 1;
}

export const fetchDemandData = (busStop: string) => 
  api.get<DemandData[]>(`/demand/${busStop}`);

export const updateConditions = (conditions: ConditionsUpdate) => 
  api.post('/update-conditions', conditions);

export const getCurrentConditions = () => 
  api.get<CurrentConditions>('/current-conditions');

export const addBus = (busNumber: string) => 
  api.post('/buses', { busNumber });