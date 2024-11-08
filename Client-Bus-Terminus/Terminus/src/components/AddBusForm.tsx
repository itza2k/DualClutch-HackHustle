// src/components/AddBusForm.tsx
import React, { useState } from "react";
import { addBus } from "../services/api";

const AddBusForm: React.FC = () => {
  const [busNumber, setBusNumber] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busNumber) {
      await addBus(busNumber);
      setBusNumber("");
      alert("Bus added successfully");
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: "20px" }}>
      <input
        type="text"
        placeholder="Enter Bus Number"
        value={busNumber}
        onChange={(e) => setBusNumber(e.target.value)}
      />
      <button type="submit">Add Bus</button>
    </form>
  );
};

export default AddBusForm;
