// src/components/LineChartComponent.tsx
/*
import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import { fetchDemandData } from "../services/api.ts";

const LineChartComponent: React.FC = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetchDemandData().then((response) => setData(response.data));
  }, []);

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data}>
        <CartesianGrid stroke="#ccc" />
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="presentDemand"
          stroke="#8884d8"
          name="Present Demand"
        />
        <Line
          type="monotone"
          dataKey="predictedDemand"
          stroke="#82ca9d"
          name="Predicted Demand"
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default LineChartComponent;
*/
// src/components/LineChartComponent.tsx
// src/components/LineChartComponent.tsx

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// Define the type for demand data
interface DemandData {
  time: string;
  day: number;
  presentDemand: number | null;
  predictedDemand: number | null;
}

interface LineChartComponentProps {
  demandData: DemandData[];
  busStop: string;
}

const LineChartComponent: React.FC<LineChartComponentProps> = ({
  demandData,
  busStop,
}) => {
  // Prepare data for the chart, handling null values
  const chartData = demandData.map((item) => ({
    time: item.time,
    presentDemand: item.presentDemand ?? 0,
    predictedDemand: item.predictedDemand ?? 0,
  }));

  return (
    <div style={{ width: "100%", height: 400 }}>
      <h2>{busStop.replace("_", " ").toUpperCase()} Demand Forecast</h2>
      <ResponsiveContainer width="100%" height="80%">
        <LineChart
          data={chartData}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />

          <XAxis
            dataKey="time"
            label={{
              value: "Time",
              position: "insideBottom",
              offset: -10,
            }}
          />

          <YAxis
            label={{
              value: "Passengers",
              angle: -90,
              position: "insideLeft",
            }}
          />

          <Tooltip
            contentStyle={{ backgroundColor: "#f5f5f5" }}
            labelStyle={{ fontWeight: "bold" }}
          />

          <Legend verticalAlign="top" height={36} />

          {/* Present Demand Line */}
          <Line
            type="monotone"
            dataKey="presentDemand"
            stroke="#8884d8"
            activeDot={{ r: 8 }}
            name="Present Demand"
          />

          {/* Predicted Demand Line */}
          <Line
            type="monotone"
            dataKey="predictedDemand"
            stroke="#82ca9d"
            activeDot={{ r: 8 }}
            name="Predicted Demand"
            strokeDasharray="5 5" // Dashed line for predictions
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LineChartComponent;
