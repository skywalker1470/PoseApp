import Foundation
import Vision
import CoreGraphics
import ImageIO

public struct PosePoint: Codable, Equatable {
    public let name: String
    public let x: Double?
    public let y: Double?
    public let confidence: Float
}

public struct PersonPose: Codable, Equatable {
    public let personIndex: Int
    public let points: [PosePoint]
}

public enum VisionPoseExtractor {
    // COCO-style 17-keypoint output.
    // Apple Vision may expose more joints, but this normalizes the output
    // to the 17 points required for CV2-101.
    private static let coco17Joints: [(String, VNHumanBodyPoseObservation.JointName)] = [
        ("nose", .nose),
        ("left_eye", .leftEye),
        ("right_eye", .rightEye),
        ("left_ear", .leftEar),
        ("right_ear", .rightEar),
        ("left_shoulder", .leftShoulder),
        ("right_shoulder", .rightShoulder),
        ("left_elbow", .leftElbow),
        ("right_elbow", .rightElbow),
        ("left_wrist", .leftWrist),
        ("right_wrist", .rightWrist),
        ("left_hip", .leftHip),
        ("right_hip", .rightHip),
        ("left_knee", .leftKnee),
        ("right_knee", .rightKnee),
        ("left_ankle", .leftAnkle),
        ("right_ankle", .rightAnkle)
    ]

    public static func detect(
        in image: CGImage,
        orientation: CGImagePropertyOrientation = .up,
        maxPeople: Int = 4
    ) throws -> [PersonPose] {
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(
            cgImage: image,
            orientation: orientation,
            options: [:]
        )

        try handler.perform([request])

        let observations = request.results ?? []

        return try observations.prefix(maxPeople).enumerated().map { personIndex, observation in
            let recognizedPoints = try observation.recognizedPoints(.all)

            let points = coco17Joints.map { name, joint in
                guard let point = recognizedPoints[joint], point.confidence > 0 else {
                    return PosePoint(
                        name: name,
                        x: nil,
                        y: nil,
                        confidence: 0
                    )
                }

                return PosePoint(
                    name: name,
                    x: Double(point.location.x),
                    y: Double(point.location.y),
                    confidence: point.confidence
                )
            }

            return PersonPose(
                personIndex: personIndex,
                points: points
            )
        }
    }
}