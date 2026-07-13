"""GLAS: Granular Lab Acquisition System.

A production-quality acquisition and analysis platform built around a
Basler ace acA640-750um camera, for granular-material physics experiments.
"""

from glas.acquisition import Acquisition, AcquisitionStats
from glas.camera import Camera
from glas.camera_info import CameraInfo, UsbDiagnostics, detect_cameras
from glas.camera_validator import ROI
from glas.controller import RecorderController
from glas.dataset import (
    Dataset,
    DatasetValidationResult,
    create_experiment_folder,
    resolve_dataset_format,
    validate_dataset,
)
from glas.exceptions import (
    AcquisitionError,
    CameraConfigurationError,
    CameraConnectionError,
    CameraDriverError,
    CameraError,
    CameraFeatureUnavailableError,
    CameraNotFoundError,
    ConfigurationError,
    DatasetError,
    DatasetFormatError,
    DatasetIOError,
    GLASError,
    JSONValidationError,
    LoggingError,
    RecorderError,
    SettingsError,
    WriterError,
)
from glas.frame import Frame
from glas.logger import configure_logging, get_logger
from glas.metadata import DatasetMetadata
from glas.recorder import Recorder, RecorderProgress, RecorderState
from glas.ringbuffer import RingBuffer, RingBufferStats
from glas.settings import Settings
from glas.timestamps import TimestampLog, WallClockReference
from glas.version import VERSION_INFO, __version__
from glas.writer import DatasetWriter, WriterStats

__all__ = [
    "__version__",
    "VERSION_INFO",
    "GLASError",
    "ConfigurationError",
    "JSONValidationError",
    "LoggingError",
    "SettingsError",
    "CameraError",
    "CameraDriverError",
    "CameraNotFoundError",
    "CameraConnectionError",
    "CameraConfigurationError",
    "CameraFeatureUnavailableError",
    "AcquisitionError",
    "DatasetError",
    "DatasetFormatError",
    "DatasetIOError",
    "WriterError",
    "RecorderError",
    "configure_logging",
    "get_logger",
    "Settings",
    "Camera",
    "CameraInfo",
    "UsbDiagnostics",
    "detect_cameras",
    "ROI",
    "Frame",
    "RingBuffer",
    "RingBufferStats",
    "Acquisition",
    "AcquisitionStats",
    "DatasetMetadata",
    "TimestampLog",
    "WallClockReference",
    "Dataset",
    "DatasetValidationResult",
    "create_experiment_folder",
    "resolve_dataset_format",
    "validate_dataset",
    "DatasetWriter",
    "WriterStats",
    "Recorder",
    "RecorderState",
    "RecorderProgress",
    "RecorderController",
]
