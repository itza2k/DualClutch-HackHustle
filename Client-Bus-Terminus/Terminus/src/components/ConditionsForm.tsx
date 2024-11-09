import React, { useState } from "react";
import { CurrentConditions, ConditionsUpdate } from "../services/api";

interface ConditionsFormProps {
  currentConditions: CurrentConditions;
  onUpdateConditions: (conditions: ConditionsUpdate) => void;
}

const ConditionsForm: React.FC<ConditionsFormProps> = ({
  currentConditions,
  onUpdateConditions,
}) => {
  const [rain, setRain] = useState<0 | 1>(currentConditions.rain);
  const [event, setEvent] = useState<0 | 1>(currentConditions.event);
  const [passengers, setPassengers] = useState({
    bus_stop_1: 0,
    bus_stop_2: 0,
    bus_stop_3: 0,
    bus_stop_4: 0,
  });

  const handlePassengerChange = (busStop: string, value: number) => {
    setPassengers((prev) => ({
      ...prev,
      [busStop]: value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const update: ConditionsUpdate = {
      rain,
      event,
      passengers_bus_stop_1: passengers.bus_stop_1,
      passengers_bus_stop_2: passengers.bus_stop_2,
      passengers_bus_stop_3: passengers.bus_stop_3,
      passengers_bus_stop_4: passengers.bus_stop_4,
    };

    onUpdateConditions(update);
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: "20px" }}>
      <h2>Update Conditions</h2>

      {/* Rain Condition */}
      <div>
        <label>
          Rain:
          <select
            value={rain}
            onChange={(e) => setRain(Number(e.target.value) as 0 | 1)}
          >
            <option value={0}>No Rain</option>
            <option value={1}>Raining</option>
          </select>
        </label>
      </div>

      {/* Event Condition */}
      <div>
        <label>
          Event:
          <select
            value={event}
            onChange={(e) => setEvent(Number(e.target.value) as 0 | 1)}
          >
            <option value={0}>No Event</option>
            <option value={1}>Event Happening</option>
          </select>
        </label>
      </div>

      {/* Passengers Input for Each Bus Stop */}
      <h3>Passengers Information</h3>
      <div>
        <label>
          Passengers at Bus Stop 1:
          <input
            type="number"
            value={passengers.bus_stop_1}
            onChange={(e) =>
              handlePassengerChange("bus_stop_1", Number(e.target.value))
            }
            min="0"
          />
        </label>
      </div>
      <div>
        <label>
          Passengers at Bus Stop 2:
          <input
            type="number"
            value={passengers.bus_stop_2}
            onChange={(e) =>
              handlePassengerChange("bus_stop_2", Number(e.target.value))
            }
            min="0"
          />
        </label>
      </div>
      <div>
        <label>
          Passengers at Bus Stop 3:
          <input
            type="number"
            value={passengers.bus_stop_3}
            onChange={(e) =>
              handlePassengerChange("bus_stop_3", Number(e.target.value))
            }
            min="0"
          />
        </label>
      </div>
      <div>
        <label>
          Passengers at Bus Stop 4:
          <input
            type="number"
            value={passengers.bus_stop_4}
            onChange={(e) =>
              handlePassengerChange("bus_stop_4", Number(e.target.value))
            }
            min="0"
          />
        </label>
      </div>

      {/* Submit Button */}
      <button type="submit">Update Conditions</button>
    </form>
  );
};

export default ConditionsForm;
