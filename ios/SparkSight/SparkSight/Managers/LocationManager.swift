// LocationManager.swift
// Spark Sight — iOS Client
//
// CoreLocation wrapper that provides GPS coordinates for the server's
// NYC Open Data spatial queries. Sends location updates every 5 seconds
// or on significant change.

import CoreLocation
import Combine

@MainActor
final class LocationManager: NSObject, ObservableObject {

    // MARK: - Published State

    @Published private(set) var currentLocation: CLLocationCoordinate2D?
    @Published private(set) var currentAddress: String = "Locating..."
    @Published private(set) var isAuthorized = false

    // MARK: - Callback

    /// Called every ~5 seconds with the latest coordinates.
    var onLocationUpdate: ((Double, Double) -> Void)?

    // MARK: - Private

    private let manager = CLLocationManager()
    private let geocoder = CLGeocoder()
    private var updateTimer: Timer?
    private var lastSentLocation: CLLocationCoordinate2D?

    // MARK: - Setup

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.distanceFilter = 5  // meters — update on significant movement
    }

    func requestPermission() {
        manager.requestWhenInUseAuthorization()
    }

    func startUpdating() {
        manager.startUpdatingLocation()

        // Periodic send timer (every 5 seconds).
        updateTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) {
            [weak self] _ in
            Task { @MainActor in
                self?.sendCurrentLocation()
            }
        }
    }

    func stopUpdating() {
        manager.stopUpdatingLocation()
        updateTimer?.invalidate()
        updateTimer = nil
    }

    // MARK: - On-Device Responses

    /// Reverse-geocode the current location into a street address.
    func describeCurrentLocation() async -> String {
        guard let location = currentLocation else {
            return "I don't have your location yet."
        }

        let clLocation = CLLocation(
            latitude: location.latitude,
            longitude: location.longitude
        )

        do {
            let placemarks = try await geocoder.reverseGeocodeLocation(clLocation)
            if let place = placemarks.first {
                var parts: [String] = []
                if let street = place.thoroughfare {
                    if let number = place.subThoroughfare {
                        parts.append("\(number) \(street)")
                    } else {
                        parts.append(street)
                    }
                }
                if let neighborhood = place.subLocality {
                    parts.append(neighborhood)
                }
                if let city = place.locality {
                    parts.append(city)
                }
                let address = parts.joined(separator: ", ")
                return "You are near \(address)."
            }
        } catch {
            print("[Location] Geocoding failed: \(error)")
        }

        return String(
            format: "You are at latitude %.4f, longitude %.4f.",
            location.latitude, location.longitude
        )
    }

    // MARK: - Private

    private func sendCurrentLocation() {
        guard let coord = currentLocation else { return }

        // Only send if moved significantly (>2m) or first send.
        if let last = lastSentLocation {
            let distance = CLLocation(latitude: coord.latitude, longitude: coord.longitude)
                .distance(from: CLLocation(latitude: last.latitude, longitude: last.longitude))
            guard distance > 2.0 else { return }
        }

        lastSentLocation = coord
        onLocationUpdate?(coord.latitude, coord.longitude)
    }

    private func updateAddress(from location: CLLocation) {
        geocoder.reverseGeocodeLocation(location) { [weak self] placemarks, _ in
            guard let place = placemarks?.first else { return }
            Task { @MainActor in
                if let street = place.thoroughfare {
                    self?.currentAddress = street
                } else if let area = place.subLocality {
                    self?.currentAddress = area
                }
            }
        }
    }
}

// MARK: - CLLocationManagerDelegate

extension LocationManager: CLLocationManagerDelegate {

    nonisolated func locationManager(
        _ manager: CLLocationManager,
        didUpdateLocations locations: [CLLocation]
    ) {
        guard let location = locations.last else { return }
        Task { @MainActor in
            self.currentLocation = location.coordinate
            self.updateAddress(from: location)
        }
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            switch manager.authorizationStatus {
            case .authorizedWhenInUse, .authorizedAlways:
                self.isAuthorized = true
                self.startUpdating()
            case .denied, .restricted:
                self.isAuthorized = false
            case .notDetermined:
                break
            @unknown default:
                break
            }
        }
    }
}
