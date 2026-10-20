# Intended Architecture

## UML diagram of the architecture
   ```mermaid

    classDiagram
        %% ==========================================
        %% HARDWARE ABSTRACTION (CAMERA)
        %% ==========================================
        class ICamera {
            <<interface>>
            +open() bool
            +close() void
            +get_frame() ndarray
            +set_parameter(name: str, value: Any) void
            +get_parameter(name: str) Any
        }
        class IDSCamera {
            -device: Any
            -nodemap: Any
            +open() bool
            +close() void
            +get_frame() ndarray
            +set_parameter(name: str, value: Any) void
        }

        class OpenCVCamera {
            -cap: VideoCapture
            +open() bool
            +close() void
            +get_frame() ndarray
            +set_parameter(name: str, value: Any) void
        }
        class MockCamera {
            -frames_dir: str
            +open() bool
            +close() void
            +get_frame() ndarray
        }

        class CameraFactory {
            +create_camera(camera_type: str, config: dict) ICamera
        }

        ICamera <|.. IDSCamera : implements
        ICamera <|.. OpenCVCamera : implements
        ICamera <|.. MockCamera : implements
        CameraFactory ..> ICamera : creates
        %% ==========================================
        %% CONFIG & GUI ANNOTATION
        %% ==========================================
        class ROI {
            <<abstract>>
            +label: str
            +shape: str
            +crop(image: ndarray) ndarray
        }
        class CircleROI {
            +center_x: int
            +center_y: int
            +radius: int
            +crop(image: ndarray) ndarray
        }
        class RectROI {
            +x: int
            +y: int
            +width: int
            +height: int
            +crop(image: ndarray) ndarray
        }

        class ROIManager {
            +load_config(path: str) dict
            +save_config(path: str, rois: dict) void
        }

        class ROIAnnotator {
            -window_name: str
            +select_single(image: ndarray, shape: str) ROI
            +select_multiple(image: ndarray, initial_rois: dict) dict
        }

        ROI <|-- CircleROI
        ROI <|-- RectROI
        ROIManager o-- ROI
        ROIAnnotator ..> ROI : creates

        %% ==========================================
        %% RECOGNITION & ML (INFERENCE RUNTIME)
        %% ==========================================
        class FeatureExtractor {
            +extract(roi_image: ndarray) ndarray
        }

        class IRecognizer {
            <<interface>>
            +predict(roi_image: ndarray) str
            +predict_proba(roi_image: ndarray) dict
        }
        class CNNRecognizer {
            -model_session: Any
            +predict(roi_image: ndarray) str
        }
        class KNNRecognizer {
            -classifier: Any
            -feature_extractor: FeatureExtractor
            +predict(roi_image: ndarray) str
        }
        class DecisionTreeRecognizer {
            -classifier: Any
            -feature_extractor: FeatureExtractor
            +predict(roi_image: ndarray) str
        }
        class RecognizerFactory {
            +create_recognizer(model_type: str, model_path: str) IRecognizer
        }
        IRecognizer <|.. CNNRecognizer : implements
        IRecognizer <|.. KNNRecognizer : implements
        IRecognizer <|.. DecisionTreeRecognizer : implements
        KNNRecognizer --> FeatureExtractor : uses
        DecisionTreeRecognizer --> FeatureExtractor : uses
        RecognizerFactory ..> IRecognizer : creates

        %% ==========================================
        %% TRAINING PIPELINE
        %% ==========================================
        class IModelTrainer {
            <<interface>>
            +train(data_dir: str) TrainMetrics
            +save_model(output_path: str) void
            +evaluate(test_dir: str) dict
        }

        class CNNTrainer {
            -augmenter: Any
            -model: Any
            +train(data_dir: str) TrainMetrics
            +save_model(output_path: str) void
        }

        class TabularTrainer {
            -feature_extractor: FeatureExtractor
            -model_type: str
            +train(data_dir: str) TrainMetrics
            +save_model(output_path: str) void
        }

        class TrainerFactory {
            +create_trainer(model_type: str, config: dict) IModelTrainer
        }

        IModelTrainer <|.. CNNTrainer : implements
        IModelTrainer <|.. TabularTrainer : implements
        TabularTrainer --> FeatureExtractor : uses
        TrainerFactory ..> IModelTrainer : creates

        %% ==========================================
        %% PIPELINE & BUFFERING
        %% ==========================================
        class FrameBuffer {
            -camera: ICamera
            -buffer: deque
            -lock: Lock
            -worker_thread: Thread
            -is_running: bool
            +start() void
            +stop() void
            +get_recent_frames(seconds: float) list
            +get_latest_frame() ndarray
        }
        class TemporalAnalyzer {
            +evaluate_blinking(state_series: list) dict
        }
        FrameBuffer o-- ICamera : captures from

        %% ==========================================
        %% ORCHESTRATION & HIGH-LEVEL APIS
        %% ==========================================
        class DatasetGenerator {
            -camera: ICamera
            -roi: ROI
            +capture_dataset(output_dir: str, variations: list) void
        }

        class LEDRecognizer {
            -camera: ICamera
            -buffer: FrameBuffer
            -recognizer: IRecognizer
            -rois: dict
            -analyzer: TemporalAnalyzer
            +initialize(config_path: str, interactive: bool) void
            +get_current_state(led_name: str) str
            +get_dynamic_state(led_name: str, window_sec: float) dict
            +close() void
        }

        DatasetGenerator o-- ICamera
        DatasetGenerator o-- ROI
        DatasetGenerator ..> CameraFactory
        DatasetGenerator ..> ROIAnnotator

        LEDRecognizer *-- FrameBuffer
        LEDRecognizer *-- IRecognizer
        LEDRecognizer *-- TemporalAnalyzer
        LEDRecognizer o-- ROI
        LEDRecognizer ..> CameraFactory
        LEDRecognizer ..> RecognizerFactory
        LEDRecognizer ..> ROIManager
        LEDRecognizer ..> ROIAnnotator
  ```

##  Architecture description
### 1. Architectural Layers & Design Patterns

- **Hardware Abstraction (`src/camera/`):**
  - `ICamera` (Interface): Declares unified camera control (`open`, `close`, `get_frame`, `set_parameter`).
  - Implementations: `IDSCamera` (IDS Peak SDK for IDS U3-3880CP-C-HQ), `OpenCVCamera` (webcam/DirectShow), and `MockCamera` (offline testing).
  - `CameraFactory` (Factory Method): Instantiates the appropriate camera driver based on configuration.

- **ROI & GUI Annotation (`src/config/`, `src/gui/`):**
  - `ROI` (`CircleROI`, `RectROI`): Encapsulates coordinate geometry and masking/cropping logic.
  - `ROIManager`: Handles JSON serialization and loading of LED board configurations.
  - `ROIAnnotator`: Reusable GUI component for single-LED marking (dataset generation) and multi-LED calibration (recognizer initialization).

- **Recognition & Inference Runtime (`src/recognition/`):**
  - `IRecognizer` (Strategy): Common lightweight inference interface (`predict`, `predict_proba`).
  - `CNNRecognizer`: Direct inference on cropped image tensors using ONNX Runtime.
  - `FeatureExtractor`: Computes color and intensity metrics (HSV, color ratios, histograms) shared between training and inference.
  - `KNNRecognizer` & `DecisionTreeRecognizer`: Lightweight classifiers trained on extracted feature vectors, loaded via `joblib`.
  - `RecognizerFactory` (Factory Method): Dynamically loads the selected recognition model for inference.

- **Model Training Pipeline (`src/training/`, `src/train.py`):**
  - **Dependency Segregation (Inference vs. Training):** HIL test stations running automated tests with Robot Framework only require a lightweight runtime (`onnxruntime`, `joblib`, `opencv-python`). Heavy training libraries (`PyTorch`, `Albumentations`, `Matplotlib`, scikit-learn model fitters) remain strictly isolated to the training environment.
  - `IModelTrainer` (Interface): Unified contract for training, model serialization, and evaluation (`train`, `save_model`, `evaluate`).
  - `CNNTrainer`: Loads raw images from `data/raw_dataset/`, applies in-memory augmentation on-the-fly (`DataLoader` with Albumentations/Torchvision), trains the neural network, and saves the final weights to `models/cnn_model.onnx`.
  - `TabularTrainer` (kNN & Decision Tree): Uses `FeatureExtractor` to transform images into a feature matrix, trains scikit-learn models, and serializes them via `joblib.dump` into `models/knn_model.joblib` and `models/dt_model.joblib`.
  - `TrainerFactory` (Factory Method): Instantiates the requested trainer (`cnn`, `knn`, `dt`) based on configuration.
  - `evaluation.py`: Handles model evaluation (confusion matrix, accuracy, classification report).
  - `src/train.py`: CLI training script orchestrating dataset loading from `data/raw_dataset/`, invokes the configured trainer, evaluates metrics, and outputs serialized artifacts into `models/`.

- **Pipeline & Temporal Analysis (`src/pipeline/`):**
  - `FrameBuffer`: Thread-safe circular buffer (`deque`) in a background thread, caching the last $N$ seconds of video frames with timestamps.
  - `TemporalAnalyzer`: Evaluates state transitions over time from buffered frames to determine blinking frequency, duty cycle, and stability.

- **Orchestration & High-Level APIs:**
  - `DatasetGenerator` (`src/dataset/`): Controls camera parameter variations to capture raw training images without on-disk augmentation.
  - `LEDRecognizer` (`src/led_recognizer.py` - Facade): Primary API integrating buffering, ROI mapping, classification, and temporal analysis for Robot Framework test suites.

---

### 2. Directory Structure

```text
dp_led_recognizer/
├── src/
│   ├── camera/             # Camera abstraction (ICamera, CameraFactory, IDS, OpenCV)
│   ├── config/             # ROI models and configuration manager (JSON persistence)
│   ├── gui/                # Interactive GUI tools (ROIAnnotator for single & multi ROI)
│   ├── dataset/            # Dataset acquisition generator (DatasetGenerator)
│   ├── recognition/        # Lightweight inference runtime (IRecognizer, CNN, kNN, DT, FeatureExtractor)
│   ├── training/           # Dedicated model training pipeline (isolated from inference runtime)
│   │   ├── base.py         # IModelTrainer (common interface) & TrainMetrics
│   │   ├── factory.py      # TrainerFactory (selection of cnn / knn / dt)
│   │   ├── cnn_trainer.py  # CNN training + on-the-fly augmentation -> export to .onnx
│   │   ├── tabular_trainer.py # Feature extraction -> kNN/DT training -> export to .joblib
│   │   └── evaluation.py   # Evaluation (confusion matrix, accuracy, classification report)
│   ├── pipeline/           # Thread-safe FrameBuffer and TemporalAnalyzer (blinking logic)
│   ├── led_recognizer.py   # High-level Facade API for automated test integration
│   └── train.py            # CLI training script (loads data from data/raw_dataset/, saves to models/)
├── data/
│   ├── configs/            # Board ROI configuration files (e.g., BPS, ODS JSONs)
│   ├── raw_dataset/        # Raw captured image datasets organized by class
│   └── features/           # Extracted tabular feature sets for kNN/DT (.csv, .parquet)
├── models/                 # Serialized weights and models (.onnx, .joblib)
├── tests/                  # Unit and integration test suites
├── requirements.txt        # Runtime dependencies (lightweight for HIL / Robot Framework)
└── README.md
```
