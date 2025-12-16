import { Injectable } from '@angular/core';

export interface GeoPosition {
  lat: number;
  lng: number;
  accuracy: number;
}

@Injectable({ providedIn: 'root' })
export class GeolocationService {
  getCurrentPosition(): Promise<GeoPosition> {
    return new Promise((resolve, reject) => {
      if (!('geolocation' in navigator)) {
        reject(new Error('Geolocation is not supported by this browser.'));
        return;
      }

      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const { latitude, longitude, accuracy } = pos.coords;
          resolve({ lat: latitude, lng: longitude, accuracy });
        },
        (err) => {
          reject(new Error(err.message || 'Unable to get position'));
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 0,
        }
      );
    });
  }

  /**
   * Frontend-only city lookup using a public reverse-geocoding API.
   * Here I use BigDataCloud's free endpoint (no key needed for light usage).
   */
  async getCityFromBrowser(): Promise<string> {
    const pos = await this.getCurrentPosition();

    const url =
      `https://api.bigdatacloud.net/data/reverse-geocode-client` +
      `?latitude=${pos.lat}&longitude=${pos.lng}&localityLanguage=fr`;

    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Reverse geocoding failed: ${resp.status}`);
    }

    const data = await resp.json();

    // BigDataCloud returns several locality fields
    const city =
      data.city ||
      data.locality ||
      data.principalSubdivision ||
      data.countryName;

    if (!city) {
      throw new Error('City not found');
    }

    return city as string;
  }
}