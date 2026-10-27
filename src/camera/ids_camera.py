"""IDS Camera driver for IDS U3-3880CP-C-HQ Rev.2.2.

Implements the ICamera interface using the IDS Peak Python SDK (ids_peak and ids_peak_ipl)
for high-performance industrial vision. Supports direct configuration via GenApi
persistence files (.cset) such as ids_setup/auto_setup.cset.
"""

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple, Union, override

import numpy as np

try:
    from .camera import ICamera
except ImportError:
    from camera import ICamera

logger = logging.getLogger(__name__)

# Lazy import flag for IDS Peak SDK
try:
    from ids_peak import ids_peak as peak
    from ids_peak_ipl import ids_peak_ipl as ipl
    import ids_peak
    HAS_IDS_PEAK = True
except ImportError:
    HAS_IDS_PEAK = False
    peak = None
    ipl = None
    ids_peak = None


class IDSCamera(ICamera):
    """Driver for IDS cameras (specifically IDS U3-3880CP-C-HQ Rev.2.2).

    Integrates with IDS Peak SDK (ids_peak and ids_peak_ipl), managing:
      - Device enumeration and connection by model name, serial, or index
      - GenApi configuration loading from .cset persistence files (e.g. auto_setup.cset)
      - Stream allocation and continuous ring buffer acquisition
      - Automatic queue flushing to ensure fresh frame delivery after parameter changes
      - On-the-fly pixel format conversion to standard OpenCV BGR8
      - Dynamic parameter access via GenICam NodeMap nodes
    """

    # Global library reference counter & lock to ensure safe initialize/close
    _lib_ref_count: int = 0
    _lib_lock: threading.Lock = threading.Lock()

    # Parameter aliases: generic names -> GenICam SFNC node names
    PARAMETER_ALIASES: Dict[str, str] = {
        "exposure": "ExposureTime",
        "exposure_time": "ExposureTime",
        "exposure_clocks": "SensorExposureTimeClocks",
        "gain": "Gain",
        "fps": "AcquisitionFrameRate",
        "frame_rate": "AcquisitionFrameRate",
        "pixel_format": "PixelFormat",
        "width": "Width",
        "height": "Height",
        "black_level": "BlackLevel",
        "auto_exposure": "ExposureAuto",
        "auto_gain": "GainAuto",
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        camera_index: int = 0,
        serial_number: Optional[str] = None,
        device_id: Optional[str] = None,
        model_name: Optional[str] = "U3-3880CP-C-HQ",
        buffer_count: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize the IDS camera driver.

        Args:
            config_path: Optional path to a GenApi persistence file (.cset).
            camera_index: Index of device to open if multiple are present (default: 0).
            serial_number: Optional camera serial number to match.
            device_id: Optional camera key / device ID to match.
            model_name: Target model name substring to filter/match (default: 'U3-3880CP-C-HQ').
            buffer_count: Number of ring buffers to allocate for acquisition (default: 4).
            **kwargs: Extra parameters passed to base class.
        """
        super().__init__(config_path=config_path, **kwargs)

        self.camera_index = camera_index
        self.serial_number = serial_number
        self.device_id = device_id
        self.model_name = model_name
        self.buffer_count = max(2, buffer_count)

        self._device: Any = None
        self._nodemap: Any = None
        self._datastream: Any = None
        self._image_converter: Any = None
        self._is_opened: bool = False
        self._is_acquiring: bool = False
        self._parameter_changed: bool = False

    # -------------------------------------------------------------------------
    # Library lifecycle helpers
    # -------------------------------------------------------------------------
    @classmethod
    def _ensure_library_initialized(cls) -> None:
        """Initializes the IDS Peak library if not already initialized."""
        if not HAS_IDS_PEAK:
            raise ImportError(
                "The 'ids_peak' and 'ids_peak_ipl' packages are required to use IDSCamera. "
                "Install them with: pip install ids_peak ids_peak_ipl"
            )
        with cls._lib_lock:
            if cls._lib_ref_count == 0:
                logger.debug("Initializing IDS Peak Library...")
                peak.Library.Initialize()
            cls._lib_ref_count += 1

    @classmethod
    def _release_library(cls) -> None:
        """Decrements reference count and closes IDS Peak library when 0."""
        if not HAS_IDS_PEAK:
            return
        with cls._lib_lock:
            if cls._lib_ref_count > 0:
                cls._lib_ref_count -= 1
                if cls._lib_ref_count == 0:
                    logger.debug("Closing IDS Peak Library...")
                    peak.Library.Close()

    # -------------------------------------------------------------------------
    # Device connection & lifecycle
    # -------------------------------------------------------------------------
    @override
    def open(self) -> bool:
        """Opens the IDS camera, loads configuration if given, and starts streaming.

        Returns:
            True if opened and streaming successfully, False otherwise.
        """
        if self._is_opened:
            logger.warning("IDSCamera is already opened.")
            return True

        try:
            self._ensure_library_initialized()
        except ImportError as e:
            logger.error(str(e))
            return False

        try:
            # Update device manager and query devices
            device_manager = peak.DeviceManager.Instance()
            device_manager.Update()
            devices = device_manager.Devices()

            if not devices or len(devices) == 0:
                logger.error(
                    "No IDS devices found. Ensure the camera (U3-3880CP-C-HQ) is connected "
                    "via USB 3.0 and the IDS driver is installed."
                )
                self._release_library()
                return False

            logger.info("Found %d IDS device(s). Selecting target camera...", len(devices))
            target_device_descriptor = self._select_device_descriptor(devices)

            if target_device_descriptor is None:
                logger.error("Could not find an IDS camera matching specified criteria.")
                self._release_library()
                return False

            # Open target device with Control access
            logger.info(
                "Opening IDS device: %s (%s, S/N: %s)...",
                target_device_descriptor.ModelName(),
                target_device_descriptor.Key(),
                target_device_descriptor.SerialNumber(),
            )
            self._device = target_device_descriptor.OpenDevice(peak.DeviceAccessType_Control)
            self._nodemap = self._device.RemoteDevice().NodeMaps()[0]

            # Load configuration file (.cset) if specified
            if self.config_path:
                self._apply_cset_file(self.config_path)

            # Initialize ImageConverter for OpenCV BGR8 conversion
            self._image_converter = ipl.ImageConverter()

            # Start acquisition pipeline
            if not self._start_acquisition():
                logger.error("Failed to start acquisition stream on IDS camera.")
                self.close()
                return False

            self._is_opened = True
            self._parameter_changed = False
            logger.info("IDSCamera opened successfully and ready for capture.")
            return True

        except Exception as e:
            logger.exception("Failed to open IDSCamera: %s", e)
            self.close()
            return False

    @override
    def close(self) -> None:
        """Stops streaming, revokes buffers, closes device and cleans up library."""
        if not self._is_opened and self._device is None:
            return

        logger.info("Closing IDSCamera...")

        # 1. Stop acquisition pipeline
        self._stop_acquisition()

        # 2. Close DataStream
        if self._datastream is not None:
            try:
                del self._datastream
            except Exception as e:
                logger.debug("Error releasing datastream: %s", e)
            self._datastream = None

        # 3. Close Device
        if self._device is not None:
            try:
                del self._device
            except Exception as e:
                logger.debug("Error releasing device: %s", e)
            self._device = None

        self._nodemap = None
        self._image_converter = None
        self._is_opened = False
        self._parameter_changed = False

        # 4. Release Library reference
        self._release_library()
        logger.info("IDSCamera closed successfully.")

    @override
    def is_opened(self) -> bool:
        """Returns True if the camera is opened and actively acquiring frames."""
        return self._is_opened and self._is_acquiring

    # -------------------------------------------------------------------------
    # Frame Acquisition & Buffer Flushing
    # -------------------------------------------------------------------------
    @override
    def flush_buffers(self, discard_count: int = 2, timeout_ms: int = 500) -> None:
        """Drains queued buffers and discards in-flight frames after parameter changes.

        CMOS sensors (e.g. Sony IMX178) take 1-2 integration cycles for exposure or
        gain register updates to take effect in analog readout. Furthermore, continuous
        acquisition buffers sitting in the output queue must be discarded to prevent
        stale frame delivery.

        Args:
            discard_count: Number of subsequent frames to discard to allow the CMOS sensor
                registers and analog circuitry to settle (default: 2).
            timeout_ms: Timeout in ms for waiting for transition frames.
        """
        if not self.is_opened() or self._datastream is None:
            return

        try:
            # 1. Drain all pending finished buffers currently queued in the output pool
            drained_stale = 0
            while hasattr(self._datastream, "NumBuffersAwaitDelivery"):
                try:
                    num_await = int(self._datastream.NumBuffersAwaitDelivery())
                except (TypeError, ValueError):
                    num_await = 0
                if num_await <= 0:
                    break
                buf = self._datastream.WaitForFinishedBuffer(50)
                self._datastream.QueueBuffer(buf)
                drained_stale += 1

            if drained_stale > 0:
                logger.debug("Drained %d stale buffer(s) from output queue.", drained_stale)

            # 2. Discard transition frames
            for _ in range(discard_count):
                buf = self._datastream.WaitForFinishedBuffer(timeout_ms)
                self._datastream.QueueBuffer(buf)

        except Exception as e:
            logger.debug("Buffer flush completed or timed out: %s", e)

    @override
    def get_frame(self, timeout_ms: int = 1000, latest_only: bool = True) -> Optional[np.ndarray]:
        """Captures a frame from the IDS camera and converts it to OpenCV BGR8 format.

        Args:
            timeout_ms: Maximum time to wait for a finished buffer in milliseconds.
            latest_only: If True, drains any older buffers that accumulated in the output
                queue while the application was idle, and flushes transition frames if
                parameters (e.g. exposure/gain) were recently modified.

        Returns:
            Captured image as a BGR uint8 NumPy array (H x W x 3), or None on timeout/error.
        """
        if not self.is_opened() or self._datastream is None:
            logger.error("Cannot get frame: IDSCamera is not acquiring.")
            return None

        # Resolve timeout exception class safely across real and mocked environments
        timeout_exc = getattr(peak, "TimeoutException", None)
        if not (isinstance(timeout_exc, type) and issubclass(timeout_exc, BaseException)):
            timeout_exc = ()

        try:
            # 1. If camera parameters were modified, flush stale and transition frames
            if self._parameter_changed:
                self.flush_buffers(discard_count=2, timeout_ms=timeout_ms)
                self._parameter_changed = False
            elif latest_only:
                # 2. Drain accumulated stale frames, keeping only the most recent incoming buffer
                while hasattr(self._datastream, "NumBuffersAwaitDelivery"):
                    try:
                        num_await = int(self._datastream.NumBuffersAwaitDelivery())
                    except (TypeError, ValueError):
                        num_await = 0
                    if num_await <= 1:
                        break
                    stale_buffer = self._datastream.WaitForFinishedBuffer(50)
                    self._datastream.QueueBuffer(stale_buffer)

            # Wait for finished buffer from data stream
            buffer = self._datastream.WaitForFinishedBuffer(timeout_ms)

            # Convert buffer to BGR8 using ImageView + ImageConverter
            image_view = ids_peak.ImageView(buffer)
            ipl_image = ipl.Image.from_image_view(image_view)
            converted_image = self._image_converter.Convert(ipl_image, ipl.PixelFormatName_BGR8)

            # Extract NumPy array (deep copy to decouple from internal buffer memory)
            frame_bgr = converted_image.get_numpy_3D().copy()

            # Re-queue buffer back to input pool
            self._datastream.QueueBuffer(buffer)

            return frame_bgr

        except timeout_exc:
            logger.warning("Frame acquisition timed out after %d ms.", timeout_ms)
            return None
        except Exception as e:
            logger.exception("Error capturing frame from IDSCamera: %s", e)
            return None

    # -------------------------------------------------------------------------
    # Configuration Loading (.cset)
    # -------------------------------------------------------------------------
    @override
    def load_configuration(self, config_path: str) -> bool:
        """Loads and applies a GenApi persistence file (.cset).

        Can be called before open() to specify the configuration file to load,
        or dynamically while opened to reconfigure the hardware.

        Args:
            config_path: Path to the .cset configuration file.

        Returns:
            True if successfully loaded, False otherwise.
        """
        abs_path = os.path.abspath(config_path)
        if not os.path.isfile(abs_path):
            logger.error("Configuration file not found: %s", abs_path)
            return False

        self.config_path = abs_path

        # If camera is already opened, apply configuration immediately
        if self._is_opened and self._nodemap is not None:
            was_acquiring = self._is_acquiring
            if was_acquiring:
                self._stop_acquisition()

            success = self._apply_cset_file(abs_path)

            if was_acquiring:
                self._start_acquisition()

            self._parameter_changed = True
            return success

        return True

    def _apply_cset_file(self, config_path: str) -> bool:
        """Helper to invoke nodemap.LoadFromFile for GenApi persistence files."""
        logger.info("Applying GenApi configuration (.cset) from: %s", config_path)
        try:
            self._nodemap.LoadFromFile(config_path)
            logger.info("GenApi configuration successfully applied.")
            return True
        except Exception as e:
            logger.error("Failed to load GenApi configuration file '%s': %s", config_path, e)
            return False

    @staticmethod
    def _get_node_type(node: Any) -> int:
        """Safely extracts the GenICam node type enum integer.

        In IDS Peak Python SDK, Node.Type() is the method returning the NodeType integer,
        while node.NodeType is an integer constant on node classes (e.g. peak.IntegerNode.NodeType = 0).
        """
        if hasattr(node, "Type") and callable(node.Type):
            return node.Type()
        if hasattr(node, "NodeType"):
            nt = node.NodeType
            return nt() if callable(nt) else int(nt)
        return -1

    @classmethod
    def _is_node_kind(cls, node: Any, node_type: int, target_int: int, target_cls_name: str) -> bool:
        """Safely checks if a node matches a GenICam type."""
        if node_type == target_int:
            return True
        target_cls = getattr(peak, target_cls_name, None) if peak is not None else None
        if isinstance(target_cls, type) and isinstance(node, target_cls):
            return True
        return False

    # -------------------------------------------------------------------------
    # Parameter Access (GenICam NodeMap)
    # -------------------------------------------------------------------------
    @override
    def set_parameter(self, name: str, value: Any) -> bool:
        """Sets a parameter via the camera GenICam NodeMap.

        Supports normalized aliases ('exposure', 'gain', 'fps', etc.) or direct
        GenICam SFNC feature node names (e.g. 'ExposureTime', 'Gain').

        Args:
            name: Parameter or node name.
            value: Target value (float, int, bool, str, or enum entry).

        Returns:
            True if successfully set, False otherwise.
        """
        if self._nodemap is None:
            logger.error("Cannot set parameter '%s': NodeMap not available.", name)
            return False

        node_name = self.PARAMETER_ALIASES.get(name.lower(), name)

        if not self._nodemap.HasNode(node_name):
            # Fallback check for alternative clocks node for exposure
            if node_name == "ExposureTime" and self._nodemap.HasNode("SensorExposureTimeClocks"):
                node_name = "SensorExposureTimeClocks"
            else:
                logger.error("NodeMap does not contain node: '%s'", node_name)
                return False

        try:
            node = self._nodemap.FindNode(node_name)
            if not node.IsWriteable():
                logger.warning("Node '%s' is currently not writeable (access status).", node_name)
                return False

            node_type = self._get_node_type(node)

            if self._is_node_kind(node, node_type, getattr(peak, "NodeType_Float", 3), "FloatNode"):
                target_val = float(value)
                try:
                    target_val = max(node.Minimum(), min(node.Maximum(), target_val))
                except Exception:
                    pass
                node.SetValue(target_val)
            elif self._is_node_kind(node, node_type, getattr(peak, "NodeType_Integer", 0), "IntegerNode"):
                target_val = int(value)
                try:
                    target_val = max(node.Minimum(), min(node.Maximum(), target_val))
                except Exception:
                    pass
                node.SetValue(target_val)
            elif self._is_node_kind(node, node_type, getattr(peak, "NodeType_Boolean", 1), "BooleanNode"):
                node.SetValue(bool(value))
            elif self._is_node_kind(node, node_type, getattr(peak, "NodeType_Enumeration", 7), "EnumerationNode"):
                if isinstance(value, int):
                    node.SetCurrentEntry(value)
                else:
                    node.SetCurrentEntry(str(value))
            elif self._is_node_kind(node, node_type, getattr(peak, "NodeType_String", 4), "StringNode"):
                node.SetValue(str(value))
            else:
                logger.warning("Unsupported node type %d for node '%s'.", node_type, node_name)
                return False

            self._parameter_changed = True
            logger.debug("Set parameter '%s' (%s) = %s", name, node_name, value)
            return True

        except Exception as e:
            logger.error("Error setting parameter '%s' to '%s': %s", node_name, value, e)
            return False

    @override
    def get_parameter(self, name: str) -> Any:
        """Reads a parameter value from the GenICam NodeMap.

        Args:
            name: Parameter or node name.

        Returns:
            Current parameter value, or None if node is not found/readable.
        """
        if self._nodemap is None:
            logger.error("Cannot get parameter '%s': NodeMap not available.", name)
            return None

        node_name = self.PARAMETER_ALIASES.get(name.lower(), name)

        if not self._nodemap.HasNode(node_name):
            if node_name == "ExposureTime" and self._nodemap.HasNode("SensorExposureTimeClocks"):
                node_name = "SensorExposureTimeClocks"
            else:
                logger.error("NodeMap does not contain node: '%s'", node_name)
                return None

        try:
            node = self._nodemap.FindNode(node_name)
            if not node.IsReadable():
                logger.warning("Node '%s' is not readable.", node_name)
                return None

            node_type = self._get_node_type(node)

            is_simple_val = (
                self._is_node_kind(node, node_type, getattr(peak, "NodeType_Float", 3), "FloatNode")
                or self._is_node_kind(node, node_type, getattr(peak, "NodeType_Integer", 0), "IntegerNode")
                or self._is_node_kind(node, node_type, getattr(peak, "NodeType_Boolean", 1), "BooleanNode")
                or self._is_node_kind(node, node_type, getattr(peak, "NodeType_String", 4), "StringNode")
            )

            if is_simple_val:
                return node.Value()
            elif self._is_node_kind(node, node_type, getattr(peak, "NodeType_Enumeration", 7), "EnumerationNode"):
                return node.CurrentEntry().SymbolicValue()
            else:
                return None

        except Exception as e:
            logger.error("Error getting parameter '%s': %s", node_name, e)
            return None

    # -------------------------------------------------------------------------
    # Acquisition Engine Internals
    # -------------------------------------------------------------------------
    def _select_device_descriptor(self, devices: Any) -> Any:
        """Selects the best matching DeviceDescriptor from available devices."""
        # 1. Match by serial number if provided
        if self.serial_number:
            for dev in devices:
                if dev.SerialNumber() == self.serial_number:
                    return dev

        # 2. Match by device_id / key if provided
        if self.device_id:
            for dev in devices:
                if dev.Key() == self.device_id:
                    return dev

        # 3. Match by model name if provided
        if self.model_name:
            for dev in devices:
                if self.model_name in dev.ModelName():
                    return dev

        # 4. Fallback to index
        if 0 <= self.camera_index < len(devices):
            return devices[self.camera_index]

        return None

    def _start_acquisition(self) -> bool:
        """Configures data stream, allocates ring buffers, and starts capture."""
        try:
            if self._datastream is None:
                datastreams = self._device.DataStreams()
                if not datastreams or len(datastreams) == 0:
                    logger.error("No DataStreams available on device.")
                    return False

                self._datastream = datastreams[0].OpenDataStream()

            # Retrieve required payload buffer size from nodemap
            payload_size = self._nodemap.FindNode("PayloadSize").Value()

            # Allocate and announce buffers to input pool
            try:
                min_buffers = int(self._datastream.NumBuffersAnnouncedMinRequired())
            except (TypeError, ValueError):
                min_buffers = 2
            alloc_count = max(min_buffers, self.buffer_count)

            logger.debug("Allocating %d stream buffers (payload size: %d bytes)...", alloc_count, payload_size)
            for _ in range(alloc_count):
                buf = self._datastream.AllocAndAnnounceBuffer(payload_size)
                self._datastream.QueueBuffer(buf)

            # Start acquisition engine
            self._datastream.StartAcquisition()

            # Lock transport layer parameters
            if self._nodemap.HasNode("TLParamsLocked"):
                tl_locked = self._nodemap.FindNode("TLParamsLocked")
                if tl_locked.IsWriteable():
                    tl_locked.SetValue(1)

            # Execute AcquisitionStart command
            if self._nodemap.HasNode("AcquisitionStart"):
                acq_start = self._nodemap.FindNode("AcquisitionStart")
                acq_start.Execute()
                acq_start.WaitUntilDone()

            self._is_acquiring = True
            logger.debug("Acquisition started successfully.")
            return True

        except Exception as e:
            logger.exception("Failed to start acquisition: %s", e)
            self._stop_acquisition()
            return False

    def _stop_acquisition(self) -> None:
        """Stops capture engine and revokes all allocated buffers."""
        if not self._is_acquiring and self._datastream is None:
            return

        logger.debug("Stopping acquisition and flushing buffers...")
        try:
            # 1. Execute AcquisitionStop
            if self._nodemap is not None and self._nodemap.HasNode("AcquisitionStop"):
                acq_stop = self._nodemap.FindNode("AcquisitionStop")
                if acq_stop.IsWriteable():
                    acq_stop.Execute()
                    acq_stop.WaitUntilDone()

            # 2. Unlock transport layer parameters
            if self._nodemap is not None and self._nodemap.HasNode("TLParamsLocked"):
                tl_locked = self._nodemap.FindNode("TLParamsLocked")
                if tl_locked.IsWriteable():
                    tl_locked.SetValue(0)

            # 3. Stop data stream
            if self._datastream is not None:
                try:
                    is_grabbing = (
                        self._datastream.IsGrabbing()
                        if callable(getattr(self._datastream, "IsGrabbing", None))
                        else self._is_acquiring
                    )
                except Exception:
                    is_grabbing = self._is_acquiring

                if is_grabbing:
                    self._datastream.StopAcquisition(peak.AcquisitionStopMode_Default)

                # 4. Flush and revoke all buffers
                if hasattr(self._datastream, "FlushAndRevokeAllBuffers"):
                    self._datastream.FlushAndRevokeAllBuffers()
                else:
                    self._datastream.Flush(peak.DataStreamFlushMode_DiscardAll)
                    for buf in self._datastream.AnnouncedBuffers():
                        self._datastream.RevokeBuffer(buf)

        except Exception as e:
            logger.warning("Exception during acquisition stop: %s", e)
        finally:
            self._is_acquiring = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cam = IDSCamera(config_path="ids_setup/factory_setup.cset")
    if cam.open():
        import cv2

        print("Capturing Frame 1...")
        frame1 = cam.get_frame(timeout_ms=2000)
        if frame1 is not None:
            cv2.imshow("Captured Frame 1", frame1)
            print("Press any key in the window to change exposure...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Change exposure
        print("Changing exposure (SensorExposureTimeClocks = 191365)...")
        cam.set_parameter("SensorExposureTimeClocks", 191365)

        print("Capturing Frame 2 with new exposure...")
        frame2 = cam.get_frame(timeout_ms=2000)
        if frame2 is not None:
            cv2.imshow("Captured Frame 2 (New Exposure)", frame2)
            print("Press any key to finish...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cam.close()
