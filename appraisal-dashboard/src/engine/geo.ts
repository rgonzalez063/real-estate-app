import type { PropertyAttributes } from '../types';

/** Great-circle distance in miles between two coordinate pairs. */
export function haversineMiles(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const R = 3958.7613;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

/**
 * Distance to the subject: the reported distance field is used when present; otherwise
 * it is computed from coordinates when both subject and comparable have them. Never invented.
 */
export function distanceToSubject(subject: PropertyAttributes, comp: PropertyAttributes): { miles: number | null; basis: 'reported' | 'coordinates' | null } {
  if (comp.distance !== null) return { miles: comp.distance, basis: 'reported' };
  if (subject.latitude !== null && subject.longitude !== null && comp.latitude !== null && comp.longitude !== null) {
    return { miles: haversineMiles(subject.latitude, subject.longitude, comp.latitude, comp.longitude), basis: 'coordinates' };
  }
  return { miles: null, basis: null };
}
