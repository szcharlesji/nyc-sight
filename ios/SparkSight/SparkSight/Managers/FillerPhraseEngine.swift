// FillerPhraseEngine.swift
// Spark Sight — iOS Client
//
// Generates contextual filler phrases to mask VLM inference latency.
// Speaks immediately after the user's utterance is detected, before
// the server response arrives. Prevents awkward silence.

import Foundation

@MainActor
final class FillerPhraseEngine: ObservableObject {

    // MARK: - Private

    private var lastUsedPhrase: String = ""

    // Filler phrase pools indexed by query type.
    private let phrasePools: [QueryType: [String]] = [
        .vision: [
            "Let me take a look.",
            "Looking at that now.",
            "Checking what's ahead.",
            "Scanning the scene.",
            "Let me see.",
        ],
        .navigation: [
            "Let me find that for you.",
            "Checking the route.",
            "Looking up directions.",
            "Finding the way.",
            "Let me locate that.",
        ],
        .information: [
            "Let me check.",
            "One moment.",
            "Looking that up.",
            "Checking on that.",
            "Let me find out.",
        ],
        .general: [
            "Sure, one moment.",
            "Let me think about that.",
            "Working on it.",
            "Give me a second.",
            "Processing that.",
        ],
    ]

    // MARK: - On-Device Quick Responses

    /// Queries that can be answered entirely on-device without the VLM.
    /// Returns nil if the query requires the server.
    func localResponse(for text: String) -> String? {
        let lower = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        // Time queries.
        if lower.contains("what time") || lower == "time" {
            let formatter = DateFormatter()
            formatter.dateFormat = "h:mm a"
            return "It's \(formatter.string(from: Date()))."
        }

        // Battery queries.
        if lower.contains("battery") {
            let level = Int(UIDevice.current.batteryLevel * 100)
            if UIDevice.current.batteryState == .unknown {
                return "I can't check the battery level right now."
            }
            let charging = UIDevice.current.batteryState == .charging ? ", and charging" : ""
            return "Battery is at \(level) percent\(charging)."
        }

        // Repeat last response — handled by TTSManager.repeatLast().
        if lower.contains("repeat") || lower.contains("say that again") ||
           lower.contains("what did you say") {
            return nil  // Signals to coordinator: call ttsManager.repeatLast()
        }

        return nil
    }

    /// Check if this is a "repeat" request (handled specially by coordinator).
    func isRepeatRequest(_ text: String) -> Bool {
        let lower = text.lowercased()
        return lower.contains("repeat") ||
               lower.contains("say that again") ||
               lower.contains("what did you say") ||
               lower.contains("say again")
    }

    // MARK: - Filler Phrase Generation

    /// Generate a contextual filler phrase for the given query.
    /// Avoids repeating the last phrase used.
    func fillerPhrase(for text: String) -> String {
        let queryType = QueryType.classify(text)
        let pool = phrasePools[queryType] ?? phrasePools[.general]!

        // Pick a random phrase, avoiding repetition.
        var phrase: String
        repeat {
            phrase = pool.randomElement()!
        } while phrase == lastUsedPhrase && pool.count > 1

        lastUsedPhrase = phrase
        return phrase
    }

    /// Generate a "wake word echo" — compressed restatement of the user's intent.
    /// This confirms understanding and buys 1-2 seconds of perceived latency.
    func wakeWordEcho(for text: String) -> String? {
        let lower = text.lowercased()
        let queryType = QueryType.classify(text)

        switch queryType {
        case .vision:
            if lower.contains("sign") { return "Reading the sign." }
            if lower.contains("door") { return "Looking for a door." }
            if lower.contains("what") { return "Checking what's there." }
            return nil

        case .navigation:
            if lower.contains("subway") { return "Finding the nearest subway." }
            if lower.contains("cross") { return "Checking the crossing." }
            // Extract the destination if it follows "find" or "get to".
            if let range = lower.range(of: "find ") {
                let dest = String(lower[range.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .prefix(30)
                if !dest.isEmpty {
                    return "Looking for \(dest)."
                }
            }
            return nil

        case .information:
            if lower.contains("construction") { return "Checking for construction ahead." }
            if lower.contains("elevator") { return "Looking for an elevator." }
            return nil

        case .general:
            return nil
        }
    }
}

// UIDevice battery monitoring needs to be enabled.
import UIKit
