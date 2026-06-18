// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CV2VisionPose",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "CV2VisionPose",
            targets: ["CV2VisionPose"]
        ),
        .executable(
            name: "PosePipeline",
            targets: ["PosePipeline"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "CV2VisionPose",
            path: "Sources/CV2VisionPose"
        ),
        .executableTarget(
            name: "PosePipeline",
            dependencies: ["CV2VisionPose"],
            path: "Sources/PosePipeline"
        ),
        .testTarget(
            name: "CV2VisionPoseTests",
            dependencies: ["CV2VisionPose"],
            path: "Tests/CV2VisionPoseTests",
            resources: [
                .process("Fixtures")
            ]
        )
    ]
)
