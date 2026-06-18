import SwiftUI

// MARK: - Skeleton definition

private let skeleton: [(String, String)] = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("nose",           "left_eye"),
    ("nose",           "right_eye"),
    ("left_eye",       "left_ear"),
    ("right_eye",      "right_ear")
]

// MARK: - PoseOverlayView

/// Draws skeleton overlays for all detected people on top of the camera preview.
/// Vision returns normalised coordinates with origin at bottom-left.
/// SwiftUI Canvas has origin at top-left, so Y is flipped: py = (1 - y) * height.
struct PoseOverlayView: View {
    let poses: [PersonPose]

    var body: some View {
        Canvas { ctx, size in
            for pose in poses {
                let pts = pointMap(pose, size: size)
                drawBones(ctx, pts: pts)
                drawJoints(ctx, pts: pts)
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: - Drawing

    private func drawBones(_ ctx: GraphicsContext, pts: [String: CGPoint]) {
        for (a, b) in skeleton {
            guard let p1 = pts[a], let p2 = pts[b] else { continue }
            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            ctx.stroke(path, with: .color(.green), lineWidth: 3)
        }
    }

    private func drawJoints(_ ctx: GraphicsContext, pts: [String: CGPoint]) {
        for pt in pts.values {
            let rect = CGRect(x: pt.x - 5, y: pt.y - 5, width: 10, height: 10)
            ctx.fill(Path(ellipseIn: rect), with: .color(.red))
        }
    }

    // MARK: - Coordinate conversion

    /// Vision: bottom-left origin, normalised 0–1.
    /// SwiftUI Canvas: top-left origin.
    /// Conversion: px = x * width,  py = (1 - y) * height
    private func pointMap(_ pose: PersonPose, size: CGSize) -> [String: CGPoint] {
        var result: [String: CGPoint] = [:]
        for pt in pose.points {
            result[pt.name] = CGPoint(
                x: pt.x * size.width,
                y: (1.0 - pt.y) * size.height
            )
        }
        return result
    }
}
