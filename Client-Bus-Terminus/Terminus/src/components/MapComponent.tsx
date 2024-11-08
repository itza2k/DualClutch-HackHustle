// src/components/MapComponent.tsx
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
        {/* Example markers - replace with dynamic data */}
        <Marker position={{ lat: 40.7128, lng: -74.006 }} />
        <Marker position={{ lat: 40.73, lng: -73.935 }} />
      </GoogleMap>
    </LoadScript>
  );
};

export default MapComponent;
