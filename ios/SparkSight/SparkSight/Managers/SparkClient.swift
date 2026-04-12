// SparkClient.swift
// Spark Sight — iOS Client
//
// WebSocket client that communicates with the DGX Spark server using
// the binary type-prefixed protocol. Sends camera frames, transcripts,
// and GPS; receives text responses and warnings for on-device TTS.

import Foundation
import Combine

/// Connection state for the WebSocket link to the DGX Spark.
enum ConnectionState: String {
    case disconnected
    case connecting
    case connected
    case reconnecting
}

@MainActor
final class SparkClient: ObservableObject {

    // MARK: - Published State

    @Published private(set) var connectionState: ConnectionState = .disconnected
    @Published private(set) var framesDelivered: Int = 0

    // MARK: - Callbacks

    /// Called when a text response arrives (for on-device TTS).
    var onTextResponse: ((String, Bool) -> Void)?

    /// Called when an urgent warning arrives (interrupts speech).
    var onWarning: ((String, String) -> Void)?

    /// Called when a status update arrives (ambient agent signals).
    var onStatus: ((StatusUpdate) -> Void)?

    // MARK: - Private State

    private var webSocket: URLSessionWebSocketTask?
    private var session: URLSession?
    private var host: String = ""
    private var port: Int = 3000
    private var reconnectAttempt = 0
    private let maxReconnectDelay: TimeInterval = 30
    private var isIntentionalDisconnect = false

    // MARK: - Connection

    /// Connect to the DGX Spark WebSocket server.
    func connect(host: String, port: Int = 3000) {
        self.host = host
        self.port = port
        isIntentionalDisconnect = false
        reconnectAttempt = 0
        establishConnection()
    }

    /// Disconnect gracefully.
    func disconnect() {
        isIntentionalDisconnect = true
        webSocket?.cancel(with: .goingAway, reason: nil)
        webSocket = nil
        connectionState = .disconnected
    }

    // MARK: - Send Methods

    /// Send a JPEG camera frame to the server.
    func sendFrame(_ jpegData: Data) {
        guard connectionState == .connected else { return }
        let message = packFrame(jpegData)
        webSocket?.send(.data(message)) { [weak self] error in
            if error == nil {
                Task { @MainActor in
                    self?.framesDelivered += 1
                }
            }
        }
    }

    /// Send a pre-transcribed text (from WhisperKit) to the server.
    func sendTranscript(_ text: String) {
        guard connectionState == .connected, !text.isEmpty else { return }
        let message = packTranscript(text)
        webSocket?.send(.data(message)) { error in
            if let error {
                print("[SparkClient] Transcript send failed: \(error)")
            }
        }
    }

    /// Send a GPS location update to the server.
    func sendLocation(lat: Double, lng: Double) {
        guard connectionState == .connected else { return }
        let message = packLocation(lat: lat, lng: lng)
        webSocket?.send(.data(message)) { error in
            if let error {
                print("[SparkClient] Location send failed: \(error)")
            }
        }
    }

    // MARK: - Private

    private func establishConnection() {
        let urlString = "ws://\(host):\(port)/ws"
        guard let url = URL(string: urlString) else {
            print("[SparkClient] Invalid URL: \(urlString)")
            return
        }

        connectionState = reconnectAttempt > 0 ? .reconnecting : .connecting

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 10
        session = URLSession(configuration: config)

        webSocket = session?.webSocketTask(with: url)
        webSocket?.resume()

        // The server expects binary frames; confirm connection by
        // starting the receive loop.
        connectionState = .connected
        reconnectAttempt = 0
        print("[SparkClient] Connected to \(urlString)")

        receiveLoop()
        startPing()
    }

    private func receiveLoop() {
        webSocket?.receive { [weak self] result in
            guard let self else { return }

            switch result {
            case .success(let message):
                Task { @MainActor in
                    self.handleMessage(message)
                }
                // Continue listening.
                self.receiveLoop()

            case .failure(let error):
                print("[SparkClient] Receive error: \(error)")
                Task { @MainActor in
                    self.handleDisconnection()
                }
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        let data: Data
        switch message {
        case .data(let d):
            data = d
        case .string(let s):
            data = Data(s.utf8)
        @unknown default:
            return
        }

        guard let (msgType, payload) = unpackMessage(data) else {
            print("[SparkClient] Unknown message type")
            return
        }

        switch msgType {
        case .textResponse:
            // Text for on-device AVSpeechSynthesizer.
            if let resp = try? JSONDecoder().decode(TextResponse.self, from: payload) {
                onTextResponse?(resp.text, resp.final)
            }

        case .warningText:
            // Urgent warning — must interrupt current TTS.
            if let warn = try? JSONDecoder().decode(WarningResponse.self, from: payload) {
                onWarning?(warn.text, warn.urgency)
            }

        case .status:
            // Ambient agent status update.
            if let status = try? JSONDecoder().decode(StatusUpdate.self, from: payload) {
                onStatus?(status)
            }

        case .tts:
            // WAV audio — ignore on iOS (we use on-device TTS).
            break

        default:
            break
        }
    }

    // MARK: - Reconnection (Exponential Backoff)

    private func handleDisconnection() {
        guard !isIntentionalDisconnect else { return }
        connectionState = .reconnecting
        reconnectAttempt += 1

        let delay = min(
            pow(2.0, Double(reconnectAttempt)) + Double.random(in: 0...1),
            maxReconnectDelay
        )
        print("[SparkClient] Reconnecting in \(String(format: "%.1f", delay))s (attempt \(reconnectAttempt))")

        Task {
            try? await Task.sleep(for: .seconds(delay))
            guard !isIntentionalDisconnect else { return }
            establishConnection()
        }
    }

    // MARK: - Keep-Alive Ping

    private func startPing() {
        Task {
            while connectionState == .connected {
                try? await Task.sleep(for: .seconds(15))
                webSocket?.sendPing { [weak self] error in
                    if let error {
                        print("[SparkClient] Ping failed: \(error)")
                        Task { @MainActor in
                            self?.handleDisconnection()
                        }
                    }
                }
            }
        }
    }
}
