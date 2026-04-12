// swift-tools-version: 5.9
// Package.swift
//
// This file defines the Swift Package Manager dependencies for SparkSight.
// When creating the Xcode project, add WhisperKit via SPM:
//   https://github.com/argmaxinc/WhisperKit (branch: main)
//
// NOTE: This Package.swift is for reference. The actual Xcode project
// should be created via Xcode → File → New → Project → iOS App, then
// add the source files and WhisperKit dependency via SPM.

import PackageDescription

let package = Package(
    name: "SparkSight",
    platforms: [
        .iOS(.v17),
    ],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.9.0"),
    ],
    targets: [
        .executableTarget(
            name: "SparkSight",
            dependencies: [
                .product(name: "WhisperKit", package: "WhisperKit"),
            ],
            path: "SparkSight"
        ),
    ]
)
