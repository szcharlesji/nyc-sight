// MainView.swift
// Spark Sight — iOS Client
//
// The single-screen accessible interface for blind and visually impaired
// users. Designed for full VoiceOver support with large touch targets,
// simple gesture model, and zero visual-only feedback.
//
// Gesture model:
//   - Single tap anywhere: start/stop listening
//   - Double tap: repeat last response
//   - Long press: "Where am I?" (reverse geocode)
//   - Two-finger tap: toggle torch (flashlight)

import SwiftUI

struct MainView: View {

    @EnvironmentObject var coordinator: AppCoordinator

    var body: some View {
        ZStack {
            // Full-screen background — the entire screen is a touch target.
            backgroundColor
                .ignoresSafeArea()

            VStack(spacing: 24) {
                Spacer()

                // Connection status indicator.
                connectionBadge

                // Current mode display.
                modeDisplay

                Spacer()

                // Central status area.
                statusArea

                Spacer()

                // Bottom info bar.
                bottomBar

                Spacer()
            }
            .padding(.horizontal, 24)
        }
        // Accessibility: the entire view is one large interactive region.
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityDescription)
        .accessibilityHint("Tap to start or stop listening. Double tap to repeat last response. Triple tap to open settings. Long press to hear your location.")
        .accessibilityAddTraits(.allowsDirectInteraction)
        // Gesture handlers (highest count first for correct SwiftUI priority).
        .onTapGesture(count: 3) {
            coordinator.showSettings = true
        }
        .onTapGesture(count: 2) {
            coordinator.repeatLastResponse()
        }
        .onTapGesture(count: 1) {
            coordinator.toggleListening()
        }
        .onLongPressGesture(minimumDuration: 0.5) {
            coordinator.speakCurrentLocation()
        }
        .simultaneousGesture(
            // Two-finger tap for torch.
            TapGesture(count: 1)
                .simultaneously(with: TapGesture(count: 1))
                .onEnded { _ in
                    coordinator.toggleTorch()
                }
        )
    }

    // MARK: - Subviews

    private var connectionBadge: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(connectionColor)
                .frame(width: 16, height: 16)
                .accessibilityHidden(true)

            Text(connectionText)
                .font(.headline)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(
            Capsule()
                .fill(Color.black.opacity(0.3))
        )
        .accessibilityLabel("Connection status: \(connectionText)")
    }

    private var modeDisplay: some View {
        VStack(spacing: 8) {
            Text(coordinator.currentMode.uppercased())
                .font(.system(size: 18, weight: .bold, design: .monospaced))
                .foregroundColor(.white.opacity(0.7))
                .accessibilityLabel("Mode: \(coordinator.currentMode)")

            if let goal = coordinator.activeGoal {
                Text(goal)
                    .font(.body)
                    .foregroundColor(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .accessibilityLabel("Active goal: \(goal)")
            }
        }
    }

    private var statusArea: some View {
        VStack(spacing: 16) {
            // Large status icon.
            Image(systemName: statusIcon)
                .font(.system(size: 80))
                .foregroundColor(.white)
                .accessibilityHidden(true)

            // Status text.
            Text(statusText)
                .font(.title2)
                .fontWeight(.medium)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .accessibilityLabel(statusText)

            // Transcript display (if listening).
            if coordinator.isListening && !coordinator.currentTranscript.isEmpty {
                Text(coordinator.currentTranscript)
                    .font(.body)
                    .foregroundColor(.white.opacity(0.8))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.white.opacity(0.1))
                    )
                    .accessibilityLabel("You said: \(coordinator.currentTranscript)")
            }
        }
        .frame(minHeight: 200)
    }

    private var bottomBar: some View {
        HStack {
            // Location.
            Label(coordinator.currentAddress, systemImage: "location.fill")
                .font(.caption)
                .foregroundColor(.white.opacity(0.6))
                .accessibilityLabel("Location: \(coordinator.currentAddress)")

            Spacer()

            // Frames delivered.
            Label("\(coordinator.framesDelivered)", systemImage: "camera.fill")
                .font(.caption)
                .foregroundColor(.white.opacity(0.6))
                .accessibilityLabel("\(coordinator.framesDelivered) frames sent")
        }
    }

    // MARK: - Computed Properties

    private var backgroundColor: Color {
        switch coordinator.connectionState {
        case .connected:
            return coordinator.isListening ? Color(red: 0.1, green: 0.2, blue: 0.4) : Color(red: 0.1, green: 0.15, blue: 0.25)
        case .connecting, .reconnecting:
            return Color(red: 0.2, green: 0.15, blue: 0.1)
        case .disconnected:
            return Color(red: 0.15, green: 0.1, blue: 0.1)
        }
    }

    private var connectionColor: Color {
        switch coordinator.connectionState {
        case .connected: return .green
        case .connecting, .reconnecting: return .orange
        case .disconnected: return .red
        }
    }

    private var connectionText: String {
        switch coordinator.connectionState {
        case .connected: return "Connected"
        case .connecting: return "Connecting..."
        case .reconnecting: return "Reconnecting..."
        case .disconnected: return "Disconnected"
        }
    }

    private var statusIcon: String {
        if coordinator.isSpeaking { return "speaker.wave.3.fill" }
        if coordinator.isListening { return "mic.fill" }
        if coordinator.connectionState == .connected { return "eye.fill" }
        return "wifi.slash"
    }

    private var statusText: String {
        if coordinator.isSpeaking { return "Speaking..." }
        if coordinator.isListening { return "Listening..." }
        if coordinator.connectionState == .connected { return "Tap to speak" }
        if coordinator.connectionState == .connecting { return "Connecting to Spark..." }
        if coordinator.connectionState == .reconnecting { return "Reconnecting..." }
        return "Not connected"
    }

    private var accessibilityDescription: String {
        var parts: [String] = [
            "Spark Sight. \(connectionText).",
            "Mode: \(coordinator.currentMode).",
        ]
        if let goal = coordinator.activeGoal {
            parts.append("Goal: \(goal).")
        }
        parts.append(statusText)
        return parts.joined(separator: " ")
    }
}

// MARK: - Settings View (minimal, for server IP configuration)

struct SettingsView: View {
    @Binding var serverHost: String
    @Binding var serverPort: String
    var onConnect: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Text("Server Settings")
                .font(.title2)
                .fontWeight(.bold)
                .accessibilityAddTraits(.isHeader)

            TextField("Server IP", text: $serverHost)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.numbersAndPunctuation)
                .autocorrectionDisabled()
                .accessibilityLabel("Server IP address")
                .frame(minHeight: 88)  // Large touch target

            TextField("Port", text: $serverPort)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.numberPad)
                .accessibilityLabel("Server port number")
                .frame(minHeight: 88)

            Button("Connect", action: onConnect)
                .font(.title3)
                .fontWeight(.semibold)
                .frame(maxWidth: .infinity, minHeight: 88)
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(16)
                .accessibilityLabel("Connect to server")
        }
        .padding(24)
    }
}
