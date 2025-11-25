import { useEffect, useState } from "react";

export function useCityFromGeolocation() {
  const [coords, setCoords] = useState<GeolocationCoordinates | null>(null);
  const [city, setCity] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // STEP 1 — get geolocation
  useEffect(() => {
    if (!("geolocation" in navigator)) {
      setError("Geolocation is not supported by your browser.");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => setCoords(pos.coords),
      (err) => setError(err.message),
      { enableHighAccuracy: true }
    );
  }, []);

  // STEP 2 — reverse geocoding when coords arrive
  useEffect(() => {
    const fetchCity = async () => {
      if (!coords) return;

      try {
        const url = `https://nominatim.openstreetmap.org/reverse?lat=${coords.latitude}&lon=${coords.longitude}&format=json&addressdetails=1`;

        const res = await fetch(url, {
          headers: {
            "User-Agent": "rag-chat/1.0 (audheon.nicolas@gmail.com)" // Required by Nominatim's usage policy
          }
        });

        const data = await res.json();

        // best candidates for "city name"
        const addr = data.address || {};
        const name =
          addr.city ||
          addr.town ||
          addr.village ||
          addr.hamlet ||
          addr.municipality ||
          null;

        setCity(name);
      } catch (err: any) {
        setError("Failed to fetch city name");
      }
    };

    fetchCity();
  }, [coords]);

  return { coords, city, error };
}