// MessageModels.swift
// Spark Sight — iOS Client
//
// Binary protocol message types matching the server's protocol.py.
// All WebSocket messages are prefixed with a one-byte type tag.

import Foundation

// MARK: - Protocol Message Types

/// One-byte type tags matching ``spark_sight.server.protocol.MessageType``.
enum MessageType: UInt8 {
    // Upstream (iOS → Server)
    case frame      = 0x01   // JPEG camera frame
    case audio      = 0x02   // PCM audio (web client only)
    case transcript = 0x05   // UTF-8 text from WhisperKit
    case location   = 0x08   // JSON: {"lat": Double, "lng": Double}

    // Downstream (Server → iOS)
    case tts          = 0x03  // WAV audio (web client only)
    case status       = 0x04  // JSON status update
    case textResponse = 0x06  // Text for on-device TTS
    case warningText  = 0x07  // Urgent warning text
}

// MARK: - Pack Helpers

/// Prefix payload with a one-byte type tag.
func packMessage(type: MessageType, payload: Data) -> Data {
    var data = Data([type.rawValue])
    data.append(payload)
    return data
}

/// Pack a JPEG frame for sending to the server.
func packFrame(_ jpeg: Data) -> Data {
    packMessage(type: .frame, payload: jpeg)
}

/// Pack a transcript string for sending to the server.
func packTranscript(_ text: String) -> Data {
    let payload = Data(text.utf8)
    return packMessage(type: .transcript, payload: payload)
}

/// Pack a GPS location update for sending to the server.
func packLocation(lat: Double, lng: Double) -> Data {
    let json = #"{"lat":\#(lat),"lng":\#(lng)}"#
    let payload = Data(json.utf8)
    return packMessage(type: .location, payload: payload)
}

// MARK: - Unpack Helpers

/// Split a raw WebSocket message into (type, payload).
func unpackMessage(_ data: Data) -> (MessageType, Data)? {
    guard let first = data.first,
          let msgType = MessageType(rawValue: first) else {
        return nil
    }
    let payload = data.dropFirst()
    return (msgType, Data(payload))
}

// MARK: - Server Response Models

/// A text response from the server for on-device TTS.
struct TextResponse: Decodable {
    let text: String
    let final: Bool
}

/// A warning from the active agent (interrupts current speech).
struct WarningResponse: Decodable {
    let text: String
    let urgency: String  // "critical", "high", "medium"
}

/// A status update from the server (ambient agent signals).
struct StatusUpdate: Decodable {
    let type: String
    let signal: String?
    let message: String?
    let mode: String?
    let goal: String?
    let ts: Double?
}

// MARK: - Message Priority

/// Priority levels for speech output. Matches server ``SpeechPriority``.
enum MessagePriority: Comparable {
    case normal     // Passive agent responses — queue after current speech
    case high       // Ambient agent progress/corrections
    case critical   // Active agent warnings — interrupt immediately
}

// MARK: - Query Type Classification

/// Classifies a user query for contextual filler phrases.
enum QueryType {
    case vision
    case navigation
    case information
    case general

    /// Simple keyword-based classification (no model needed).
    static func classify(_ text: String) -> QueryType {
        let lower = text.lowercased()

        let visionKeywords = ["see", "look", "front", "ahead", "sign", "read",
                              "what is", "what's", "describe", "show", "door",
                              "building", "store", "color"]
        let navKeywords = ["find", "where", "subway", "get to", "nearest",
                           "route", "directions", "walk", "cross", "street",
                           "intersection", "turn", "go to", "take me", "navigate"]
        let infoKeywords = ["construction", "time", "weather", "accessible",
                            "elevator", "how many", "open", "closed", "permit",
                            "hours", "schedule"]

        if visionKeywords.contains(where: { lower.contains($0) }) {
            return .vision
        }
        if navKeywords.contains(where: { lower.contains($0) }) {
            return .navigation
        }
        if infoKeywords.contains(where: { lower.contains($0) }) {
            return .information
        }
        return .general
    }
}
