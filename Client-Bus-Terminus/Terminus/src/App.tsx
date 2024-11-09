// src/pages/App.tsx
/*
import React from "react";
import LineChartComponent from "./components/LineChartComponent";
import MapComponent from "./components/MapComponent";
import AddBusForm from "./components/AddBusForm";

const App: React.FC = () => {
  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>City Transport Department Dashboard</h1>
      <LineChartComponent />
      <MapComponent />
      <AddBusForm />
    </div>
  );
};

export default App;
*/

import React, { useState, useEffect } from "react";
import LineChartComponent from "./components/LineChartComponent";
import MapComponent from "./components/MapComponent";
import AddBusForm from "./components/AddBusForm";
import ConditionsForm from "./components/ConditionsForm";
import {
  fetchDemandData,
  updateConditions,
  getCurrentConditions,
  DemandData,
  CurrentConditions,
  ConditionsUpdate,
} from "./services/api";

const App: React.FC = () => {
  const [selectedBusStop, setSelectedBusStop] = useState("bus_stop_1");
  const [demandData, setDemandData] = useState<DemandData[]>([]);
  const [currentConditions, setCurrentConditions] = useState<CurrentConditions>(
    {
      rain: 0,
      event: 0,
    }
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch demand data when bus stop or component mounts
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const demandResponse = await fetchDemandData(selectedBusStop);
        const conditionsResponse = await getCurrentConditions();

        setDemandData(demandResponse.data);
        setCurrentConditions(conditionsResponse.data);
        setLoading(false);
      } catch (err) {
        setError("Failed to fetch data");
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedBusStop]);

  // Handler for updating conditions
  const handleConditionsUpdate = async (update: ConditionsUpdate) => {
    try {
      const response = await updateConditions(update);

      // Update local state with new conditions and demand data
      setCurrentConditions({
        rain: update.rain,
        event: update.event,
      });

      // Optionally, you might want to refresh demand data here
      const demandResponse = await fetchDemandData(selectedBusStop);
      setDemandData(demandResponse.data);
    } catch (err) {
      setError("Failed to update conditions");
    }
  };

  // Bus stop selection handler
  const handleBusStopChange = (busStop: string) => {
    setSelectedBusStop(busStop);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>City Transport Department Dashboard</h1>

      {/* Bus Stop Selector */}
      <div style={{ marginBottom: "20px" }}>
        <label>Select Bus Stop: </label>
        <select
          value={selectedBusStop}
          onChange={(e) => handleBusStopChange(e.target.value)}
        >
          <option value="bus_stop_1">Bus Stop 1</option>
          <option value="bus_stop_2">Bus Stop 2</option>
          <option value="bus_stop_3">Bus Stop 3</option>
          <option value="bus_stop_4">Bus Stop 4</option>
        </select>
      </div>

      {/* Conditional Rendering based on loading/error states */}
      {loading ? (
        <p>Loading...</p>
      ) : error ? (
        <p>Error: {error}</p>
      ) : (
        <>
          {/* Demand Chart */}
          <LineChartComponent
            demandData={demandData}
            busStop={selectedBusStop}
          />

          {/* Map Component */}
          <MapComponent busStop={selectedBusStop} demandData={demandData} />
        </>
      )}

      {/* Conditions Update Form */}
      <ConditionsForm
        currentConditions={currentConditions}
        onUpdateConditions={handleConditionsUpdate}
      />

      {/* Add Bus Form */}
      <AddBusForm />
    </div>
  );
};

export default App;
