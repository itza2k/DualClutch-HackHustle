// src/components/MapComponent.tsx
/*
import React from "react";
import { GoogleMap, LoadScript, Marker } from "@react-google-maps/api";

const mapContainerStyle = {
  width: "100%",
  height: "400px",
};

const center = { lat: 40.7128, lng: -74.006 }; // Adjust to your city's center

const MapComponent: React.FC = () => {
  return (
    <LoadScript googleMapsApiKey="YOUR_GOOGLE_MAPS_API_KEY">
      <GoogleMap
        mapContainerStyle={mapContainerStyle}
        center={center}
        zoom={10}
      >

        <Marker position={{ lat: 40.7128, lng: -74.006 }} />
        <Marker position={{ lat: 40.73, lng: -73.935 }} />
      </GoogleMap>
    </LoadScript>
  );
};

export default MapComponent;
*/
// src/components/MapComponent.tsx

// src/components/MapComponent.tsx

// src/components/MapComponent.tsx

// src/components/MapComponent.tsx

import React from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import { LatLngExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

// Fix for default marker icon
import L from "leaflet";
import icon from "leaflet/dist/images/marker-icon.png";
import iconShadow from "leaflet/dist/images/marker-shadow.png";

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

// Define the props interface
interface MapComponentProps {
  busStop: string;
  demandData: {
    time: string;
    day: number;
    presentDemand: number | null;
    predictedDemand: number | null;
  }[];
}

// Define the coordinates for each bus stop
const busStopCoordinates: Record<string, LatLngExpression> = {
  bus_stop_1: [51.505, -0.09],
  bus_stop_2: [51.51, -0.1],
  bus_stop_3: [51.51, -0.12],
  bus_stop_4: [51.52, -0.1],
};

const MapComponent: React.FC<MapComponentProps> = ({ busStop, demandData }) => {
  // Get the position of the selected bus stop
  const position = busStopCoordinates[busStop];

  // Calculate current demand
  const currentDemand =
    demandData.length > 0
      ? demandData[demandData.length - 1].presentDemand ?? 0
      : 0;

  return (
    <MapContainer
      center={position}
      zoom={13}
      style={{ height: "400px", width: "100%" }}
      scrollWheelZoom={false}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <Marker position={position}>
        <Popup>
          {busStop} <br />
          Current Demand: {currentDemand}
        </Popup>
      </Marker>
    </MapContainer>
  );
};

export default MapComponent;
