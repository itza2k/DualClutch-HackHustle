
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

let demandData = [
  { time: '10:00', presentDemand: 100, predictedDemand: 120 },
  { time: '11:00', presentDemand: 130, predictedDemand: 140 },
  // ...more data
];

app.get('/api/demand', (req, res) => {
  res.json(demandData);
});

let buses = [];

app.post('/api/buses', (req, res) => {
  const { busNumber } = req.body;
  buses.push({ busNumber });
  res.status(201).json({ message: 'Bus added' });
});

app.listen(5001, () => console.log('Server running on port 5001'));