// src/services/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5001/api', // Replace with your backend URL
});

export const fetchDemandData = () => api.get('/demand');
export const addBus = (busNumber: string) => api.post('/buses', { busNumber });
