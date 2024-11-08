// src/pages/App.tsx
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
