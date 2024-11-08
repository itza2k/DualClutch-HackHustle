import React, { useState, useEffect } from "react";
import "./App.css";

type Location = {
  id: number;
  name: string;
};

const App: React.FC = () => {
  const [busNumbers, setBusNumbers] = useState<string[]>([]);
  const [locations, setLocations] = useState<Location[]>([]);
  const [selectedBusNo, setSelectedBusNo] = useState<string>("");
  const [startLocation, setStartLocation] = useState<number | null>(null);
  const [stopLocation, setStopLocation] = useState<number | null>(null);

  useEffect(() => {
    // Fetch available bus numbers from the Flask API
    const fetchBusNumbers = async () => {
      try {
        const response = await fetch("http://localhost:5000/api/busNumbers");
        const data = await response.json();
        setBusNumbers(data.busNumbers);
      } catch (error) {
        console.error("Error fetching bus numbers:", error);
      }
    };

    // Fetch available locations from the Flask API
    const fetchLocations = async () => {
      try {
        const response = await fetch("http://localhost:5000/api/locations");
        const data = await response.json();
        setLocations(data.locations);
      } catch (error) {
        console.error("Error fetching locations:", error);
      }
    };

    fetchBusNumbers();
    fetchLocations();
  }, []);

  const handleSubmit = async () => {
    if (!selectedBusNo || startLocation === null || stopLocation === null) {
      alert("Please fill out all fields.");
      return;
    }

    try {
      const response = await fetch(
        `http://localhost:5000/api/route?busNo=${selectedBusNo}&start=${startLocation}&stop=${stopLocation}`
      );
      const data = await response.json();
      console.log("Server Response:", data);
    } catch (error) {
      console.error("Error sending request:", error);
    }
  };

  const startLocationOptions = locations.filter(
    (location) => location.id !== startLocation
  );

  const stopLocationOptions = locations.filter(
    (location) =>
      location.id !== startLocation && location.id > (startLocation || 0)
  );

  return (
    <div className="app">
      <h1>Bus Route Finder</h1>
      <div className="form">
        <div className="input-group">
          <label htmlFor="busNo">Select Bus Number:</label>
          <select
            id="busNo"
            value={selectedBusNo}
            onChange={(e) => setSelectedBusNo(e.target.value)}
          >
            <option value="">Select Bus Number</option>
            {busNumbers.map((busNo) => (
              <option key={busNo} value={busNo}>
                {busNo}
              </option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label htmlFor="startLocation">Select Start Location:</label>
          <select
            id="startLocation"
            value={startLocation ?? ""}
            onChange={(e) => setStartLocation(Number(e.target.value))}
          >
            <option value="">Select Start Location</option>
            {locations.map((location) => (
              <option key={location.id} value={location.id}>
                {location.name}
              </option>
            ))}
          </select>
        </div>

        <div className="input-group">
          <label htmlFor="stopLocation">Select Stop Location:</label>
          <select
            id="stopLocation"
            value={stopLocation ?? ""}
            onChange={(e) => setStopLocation(Number(e.target.value))}
          >
            <option value="">Select Stop Location</option>
            {stopLocationOptions.map((location) => (
              <option key={location.id} value={location.id}>
                {location.name}
              </option>
            ))}
          </select>
        </div>

        <button onClick={handleSubmit}>Submit</button>
      </div>
    </div>
  );
};

export default App;
