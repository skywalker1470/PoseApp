import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var camera = CameraManager()

    var body: some View {
        ZStack {
            // Camera feed
            CameraPreview(manager: camera)
                .ignoresSafeArea()

            // Skeleton overlay
            PoseOverlayView(poses: camera.poses)
                .ignoresSafeArea()

            // HUD
            VStack {
                HStack {
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text(String(format: "%.0f FPS", camera.fps))
                            .font(.system(.caption, design: .monospaced).bold())
                            .foregroundStyle(.green)
                        Text("\(camera.poses.count) person\(camera.poses.count == 1 ? "" : "s")")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.white)
                    }
                    .padding(8)
                    .background(.black.opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding()
                }
                Spacer()
            }
        }
        .onAppear {
            requestCameraPermission {
                camera.start()
            }
        }
        .onDisappear {
            camera.stop()
        }
    }

    private func requestCameraPermission(granted: @escaping () -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            granted()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { ok in
                if ok { DispatchQueue.main.async { granted() } }
            }
        default:
            break
        }
    }
}

// MARK: - Camera preview bridge (UIKit → SwiftUI)

struct CameraPreview: UIViewRepresentable {
    let manager: CameraManager

    func makeUIView(context: Context) -> PreviewView {
        PreviewView()
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        if let layer = manager.previewLayer {
            uiView.setPreviewLayer(layer)
        }
    }
}

final class PreviewView: UIView {
    private var previewLayer: AVCaptureVideoPreviewLayer?

    func setPreviewLayer(_ layer: AVCaptureVideoPreviewLayer) {
        previewLayer?.removeFromSuperlayer()
        layer.frame = bounds
        self.layer.insertSublayer(layer, at: 0)
        previewLayer = layer
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}
