import XCTest
import Foundation
import CoreGraphics
import ImageIO
@testable import CV2VisionPose

final class VisionAnnotationTests: XCTestCase {
    private let skeleton: [(String, String)] = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear")
    ]

    func testAnnotateCapturedFramesWithAppleVision() throws {
        let frameURLs = try findWebcamFrames()

        XCTAssertFalse(
            frameURLs.isEmpty,
            "No captured JPG frames found in the Swift Package resource bundle."
        )

        guard !frameURLs.isEmpty else { return }

        let outputDir = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["ANNOTATION_OUTPUT_DIR"] ?? "vision-output",
            isDirectory: true
        )

        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        var totalPeople = 0
        var totalMs = 0.0

        for frameURL in frameURLs {
            let (image, orientation) = try loadCGImage(from: frameURL)

            let start = Date()
            let poses = try VisionPoseExtractor.detect(in: image, orientation: orientation, maxPeople: 4)
            let elapsedMs = Date().timeIntervalSince(start) * 1000.0

            totalPeople += poses.count
            totalMs += elapsedMs

            for pose in poses {
                XCTAssertEqual(pose.points.count, 17)
            }

            let annotated = try drawPoses(poses, on: image)
            let outputURL = outputDir.appendingPathComponent(
                frameURL.deletingPathExtension().lastPathComponent + "_annotated.png"
            )

            try writePNG(annotated, to: outputURL)
            print("\(frameURL.lastPathComponent): people=\(poses.count), timeMs=\(String(format: "%.2f", elapsedMs))")
        }

        let averageMs = totalMs / Double(frameURLs.count)
        print("Frames annotated: \(frameURLs.count)")
        print("Total people detections: \(totalPeople)")
        print("Average Apple Vision time per frame: \(String(format: "%.2f", averageMs)) ms")
    }

    // MARK: - Helpers

    private func findWebcamFrames() throws -> [URL] {
        guard let resourceURL = Bundle.module.resourceURL else { return [] }
        guard let enumerator = FileManager.default.enumerator(
            at: resourceURL, includingPropertiesForKeys: nil
        ) else { return [] }

        return enumerator
            .compactMap { $0 as? URL }
            .filter { isCapturedFrame($0) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func isCapturedFrame(_ url: URL) -> Bool {
        guard url.pathExtension.lowercased() == "jpg" else { return false }
        let filename = url.deletingPathExtension().lastPathComponent
        guard filename.hasPrefix("frame_") else { return false }
        let frameNumber = filename.dropFirst("frame_".count)
        return frameNumber.count == 4 && frameNumber.allSatisfy { $0.isNumber }
    }

    private func loadCGImage(from url: URL) throws -> (image: CGImage, orientation: CGImagePropertyOrientation) {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image  = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw NSError(domain: "VisionAnnotationTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not load image: \(url.path)"])
        }

        var orientation = CGImagePropertyOrientation.up
        if
            let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let raw   = props[kCGImagePropertyOrientation] as? UInt32,
            let exif  = CGImagePropertyOrientation(rawValue: raw)
        {
            orientation = exif
        }

        return (image, orientation)
    }

    private func drawPoses(_ poses: [PersonPose], on image: CGImage) throws -> CGImage {
        let width  = image.width
        let height = image.height

        guard let context = CGContext(
            data: nil,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw NSError(domain: "VisionAnnotationTests", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Could not create drawing context."])
        }

        // CGContext has top-left origin by default. Flip it so the image
        // draws right-side up (CGImage data is stored top-to-bottom).
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Undo the flip so keypoint coordinates are in top-left origin space,
        // matching the (1.0 - y) conversion in pointMap below.
        context.scaleBy(x: 1, y: -1)
        context.translateBy(x: 0, y: -CGFloat(height))

        context.setLineWidth(3)
        context.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))

        for pose in poses {
            let points = pointMap(for: pose, imageWidth: width, imageHeight: height)

            for (a, b) in skeleton {
                guard let p1 = points[a], let p2 = points[b] else { continue }
                context.move(to: p1)
                context.addLine(to: p2)
                context.strokePath()
            }

            for point in points.values {
                let radius: CGFloat = 5
                context.fillEllipse(in: CGRect(
                    x: point.x - radius, y: point.y - radius,
                    width: radius * 2,   height: radius * 2
                ))
            }
        }

        guard let raw = context.makeImage() else {
            throw NSError(domain: "VisionAnnotationTests", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Could not create annotated image."])
        }

        // Rotate the final composited image 180° so it saves the right way up.
        guard let rotateContext = CGContext(
            data: nil,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return raw
        }
        rotateContext.translateBy(x: CGFloat(width), y: CGFloat(height))
        rotateContext.rotate(by: .pi)
        rotateContext.translateBy(x: CGFloat(width), y: 0)
        rotateContext.scaleBy(x: -1, y: 1)
        rotateContext.draw(raw, in: CGRect(x: 0, y: 0, width: width, height: height))

        return rotateContext.makeImage() ?? raw
    }

    // Vision: bottom-left origin (y=0 at bottom)
    // CGContext after flip undo: top-left origin (y=0 at top)
    // So: py = (1.0 - y) * height
    private func pointMap(
        for pose: PersonPose,
        imageWidth: Int,
        imageHeight: Int
    ) -> [String: CGPoint] {
        var result: [String: CGPoint] = [:]
        for point in pose.points {
            guard point.confidence >= 0.2, let x = point.x, let y = point.y else { continue }
            result[point.name] = CGPoint(
                x: CGFloat(x) * CGFloat(imageWidth),
                y: CGFloat(1.0 - y) * CGFloat(imageHeight)
            )
        }
        return result
    }

    private func writePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil
        ) else {
            throw NSError(domain: "VisionAnnotationTests", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Could not create PNG destination."])
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NSError(domain: "VisionAnnotationTests", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "Could not write PNG: \(url.path)"])
        }
    }
}
