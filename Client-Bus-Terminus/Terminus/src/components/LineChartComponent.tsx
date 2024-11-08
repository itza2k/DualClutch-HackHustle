// src/components/LineChartComponent.tsx
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
