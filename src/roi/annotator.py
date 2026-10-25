"""Interactive PySide6 ROI Annotator for LED recognition.

Provides an interactive GUI canvas for defining, adjusting, and previewing
circular and rectangular Regions of Interest with real-time crop preview,
side panel LED management, and direct integration with ROIManager and ICamera.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from PySide6.QtCore import QPointF, QRectF, Qt, Signal
    from PySide6.QtGui import (
        QAction,
        QBrush,
        QColor,
        QFont,
        QImage,
        QKeySequence,
        QPainter,
        QPen,
        QPixmap,
    )
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QFileDialog,
        QGraphicsEllipseItem,
        QGraphicsItem,
        QGraphicsPixmapItem,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

# Ensure repository root is on sys.path for direct script execution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from .manager import ROIManager
    from .roi import CircleROI, RectROI, ROI
except (ImportError, ValueError):
    from manager import ROIManager
    from roi import CircleROI, RectROI, ROI

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Image Conversion Utilities
# -----------------------------------------------------------------------------
def ndarray_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Converts an OpenCV BGR or RGB NumPy array into a QPixmap."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")

    if image.ndim == 2:
        # Grayscale
        h, w = image.shape
        qimg = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif image.ndim == 3:
        h, w, c = image.shape
        if c == 3:
            # OpenCV standard is BGR -> convert to RGB for QImage
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        elif c == 4:
            bgra = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            bytes_per_line = 4 * w
            qimg = QImage(bgra.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()
        else:
            raise ValueError(f"Unsupported channel count: {c}")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

    return QPixmap.fromImage(qimg)


if HAS_PYSIDE6:

    class ResizeHandle(QGraphicsRectItem):
        """Interactive handle placed at corner for resizing graphics items."""

        def __init__(self, parent: QGraphicsItem) -> None:
            super().__init__(-6, -6, 12, 12, parent)
            self.setBrush(QBrush(QColor(255, 165, 0)))  # Orange
            self.setPen(QPen(Qt.GlobalColor.black, 1))
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            self.setZValue(10)

        def mousePressEvent(self, event: Any) -> None:
            parent = self.parentItem()
            if parent is not None:
                scene = parent.scene()
                if scene is not None:
                    scene.clearSelection()
                parent.setSelected(True)
            event.accept()

        def mouseMoveEvent(self, event: Any) -> None:
            parent = self.parentItem()
            if parent is None:
                return

            pos = parent.mapFromScene(event.scenePos())
            rect = parent.rect()

            new_w = max(10.0, pos.x() - rect.left())
            new_h = max(10.0, pos.y() - rect.top())

            # For circles, constrain to 1:1 aspect ratio
            if getattr(parent, "shape_type", "") == "circle":
                size = max(new_w, new_h)
                rect.setWidth(size)
                rect.setHeight(size)
            else:
                rect.setWidth(new_w)
                rect.setHeight(new_h)

            parent.setRect(rect)
            parent.update_dependents()
            event.accept()

        def mouseReleaseEvent(self, event: Any) -> None:
            parent = self.parentItem()
            if parent is not None:
                parent.update_dependents()
            event.accept()


    class BaseGraphicsROIItem:
        """Mixin for graphics items representing an ROI."""

        def setup_roi_visuals(self, label: str) -> None:
            self.setFlags(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )
            self.setPen(QPen(QColor(0, 255, 0), 2))
            self.setBrush(QBrush(QColor(0, 255, 0, 45)))

            # Text label above item
            self.text_item = QGraphicsSimpleTextItem(label, self)
            self.text_item.setBrush(QBrush(QColor(255, 235, 59)))  # Yellow text
            font = QFont("Arial", 50, QFont.Weight.Bold)
            self.text_item.setFont(font)

            # Resize handle
            self.handle = ResizeHandle(self)
            self.update_dependents()

        def set_label(self, label: str) -> None:
            self.label = label
            self.text_item.setText(label)
            self.update_dependents()

        def update_dependents(self) -> None:
            r = self.rect()
            self.text_item.setPos(r.left() - 50, r.top() - 100)
            self.handle.setPos(r.bottomRight())
            # Notify scene/window of geometry change if needed
            scene = self.scene()
            if scene is not None and hasattr(scene, "on_item_geometry_changed"):
                scene.on_item_geometry_changed(self)

        def itemChange(self, change: Any, value: Any) -> Any:
            if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
                scene = self.scene()
                if scene is not None and hasattr(scene, "on_item_geometry_changed"):
                    scene.on_item_geometry_changed(self)
            return super().itemChange(change, value)


    class GraphicsRectROIItem(BaseGraphicsROIItem, QGraphicsRectItem):
        """QGraphicsItem representing a rectangular ROI."""

        def __init__(self, rect: QRectF, label: str = "") -> None:
            super().__init__(rect)
            self.label = label
            self.shape_type = "rect"
            self.setup_roi_visuals(label)

        def to_domain_roi(self) -> RectROI:
            scene_rect = self.mapRectToScene(self.rect())
            return RectROI(
                x=int(round(scene_rect.x())),
                y=int(round(scene_rect.y())),
                width=int(round(scene_rect.width())),
                height=int(round(scene_rect.height())),
                label=self.label,
            )


    class GraphicsCircleROIItem(BaseGraphicsROIItem, QGraphicsEllipseItem):
        """QGraphicsItem representing a circular ROI."""

        def __init__(self, rect: QRectF, label: str = "") -> None:
            super().__init__(rect)
            self.label = label
            self.shape_type = "circle"
            self.setup_roi_visuals(label)

        def to_domain_roi(self) -> CircleROI:
            scene_rect = self.mapRectToScene(self.rect())
            w = scene_rect.width()
            h = scene_rect.height()
            radius = int(round(min(w, h) / 2.0))
            cx = int(round(scene_rect.x() + radius))
            cy = int(round(scene_rect.y() + radius))
            return CircleROI(
                center_x=cx,
                center_y=cy,
                radius=radius,
                label=self.label,
                mask_outside=True,
            )


    class InteractiveROIScene(QGraphicsScene):
        """Custom QGraphicsScene handling user mouse gestures to draw new ROIs."""

        item_selected_signal = Signal(object)
        roi_changed_signal = Signal(object)
        roi_added_signal = Signal(object)

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.mode: str = "select"  # 'select', 'add_rect', 'add_circle'
            self.start_point: Optional[QPointF] = None
            self.temp_item: Optional[QGraphicsItem] = None
            self.pixmap_item: Optional[QGraphicsPixmapItem] = None
            self.selectionChanged.connect(self._on_selection_changed)

        def _on_selection_changed(self) -> None:
            selected = self.selectedItems()
            if selected and isinstance(selected[0], (GraphicsRectROIItem, GraphicsCircleROIItem)):
                self.item_selected_signal.emit(selected[0])
            else:
                self.item_selected_signal.emit(None)

        def on_item_geometry_changed(self, item: Any) -> None:
            self.roi_changed_signal.emit(item)

        def keyPressEvent(self, event: Any) -> None:
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                selected = self.selectedItems()
                roi_items = [it for it in selected if isinstance(it, (GraphicsRectROIItem, GraphicsCircleROIItem))]
                if roi_items:
                    for it in roi_items:
                        self.removeItem(it)
                    parent_window = self.views()[0].window() if self.views() else None
                    if parent_window and hasattr(parent_window, "_update_table"):
                        parent_window._update_table()
                    if parent_window and hasattr(parent_window, "preview_label"):
                        parent_window.preview_label.setText("No selection")
                        parent_window.preview_label.setPixmap(QPixmap())
                    event.accept()
                    return
            super().keyPressEvent(event)

        def mousePressEvent(self, event: Any) -> None:
            if event.button() == Qt.MouseButton.LeftButton and self.mode in ("add_rect", "add_circle"):
                self.start_point = event.scenePos()
                rect = QRectF(self.start_point, self.start_point)
                pen = QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine)

                if self.mode == "add_rect":
                    self.temp_item = self.addRect(rect, pen)
                else:
                    self.temp_item = self.addEllipse(rect, pen)
                event.accept()
            else:
                super().mousePressEvent(event)

        def mouseMoveEvent(self, event: Any) -> None:
            if self.temp_item and self.start_point:
                rect = QRectF(self.start_point, event.scenePos()).normalized()
                if self.mode == "add_circle":
                    size = max(rect.width(), rect.height())
                    rect.setWidth(size)
                    rect.setHeight(size)
                self.temp_item.setRect(rect)
                event.accept()
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: Any) -> None:
            if event.button() == Qt.MouseButton.LeftButton and self.temp_item:
                rect = self.temp_item.rect().normalized()
                self.removeItem(self.temp_item)
                self.temp_item = None

                if rect.width() >= 8 and rect.height() >= 8:
                    # Generate candidate label
                    parent_window = self.views()[0].window() if self.views() else None
                    candidate_label = "led"
                    if hasattr(parent_window, "generate_next_label"):
                        candidate_label = parent_window.generate_next_label()

                    label, ok = QInputDialog.getText(
                        parent_window,
                        "Define LED ROI",
                        "Enter LED Label / Name:",
                        text=candidate_label,
                    )
                    if ok and label.strip():
                        label_clean = label.strip()
                        local_rect = QRectF(0, 0, rect.width(), rect.height())
                        if self.mode == "add_rect":
                            item = GraphicsRectROIItem(local_rect, label=label_clean)
                        else:
                            item = GraphicsCircleROIItem(local_rect, label=label_clean)
                        item.setPos(rect.topLeft())
                        self.addItem(item)
                        self.roi_added_signal.emit(item)
                        self.clearSelection()
                        item.setSelected(True)

                self.mode = "select"
                if self.views() and hasattr(self.views()[0].window(), "reset_tool_actions"):
                    self.views()[0].window().reset_tool_actions()
                event.accept()
            else:
                super().mouseReleaseEvent(event)


    class ROIAnnotatorWindow(QMainWindow):
        """Full-featured GUI window for interactive ROI annotation and configuration."""

        def __init__(
            self,
            image: Optional[Union[np.ndarray, str]] = None,
            initial_rois: Optional[Union[Dict[str, ROI], str]] = None,
            config_path: Optional[str] = None,
            camera: Optional[Any] = None,
            window_name: str = "LED ROI Annotator",
        ) -> None:
            super().__init__()
            self.setWindowTitle(window_name)
            self.resize(1200, 800)

            self.camera = camera
            self.current_image: Optional[np.ndarray] = None
            self.current_config_path: Optional[str] = config_path or (
                initial_rois if isinstance(initial_rois, str) else None
            )
            self.confirmed: bool = False

            # Central layout with splitter: View on left, Side Panel on right
            splitter = QSplitter(Qt.Orientation.Horizontal, self)
            self.setCentralWidget(splitter)

            # 1. Left: Graphics View & Scene
            self.scene = InteractiveROIScene(self)
            self.view = QGraphicsView(self.scene)
            self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
            self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            splitter.addWidget(self.view)

            # 2. Right: Side Panel
            side_panel = QWidget(self)
            side_layout = QVBoxLayout(side_panel)
            side_layout.setContentsMargins(8, 8, 8, 8)

            # LED list table
            lbl_table = QLabel("<b>Defined ROIs</b>", self)
            side_layout.addWidget(lbl_table)

            self.table = QTableWidget(0, 3, self)
            self.table.setHorizontalHeaderLabels(["Label", "Shape", "Coordinates"])
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
            side_layout.addWidget(self.table)

            # Table action buttons
            btn_box = QHBoxLayout()
            self.btn_rename = QPushButton("Rename", self)
            self.btn_rename.clicked.connect(self._rename_selected_roi)
            self.btn_delete = QPushButton("Delete", self)
            self.btn_delete.clicked.connect(self._delete_selected_roi)
            btn_box.addWidget(self.btn_rename)
            btn_box.addWidget(self.btn_delete)
            side_layout.addLayout(btn_box)

            # Live Crop Preview
            lbl_preview = QLabel("<b>Live Crop Preview</b>", self)
            side_layout.addWidget(lbl_preview)

            self.preview_label = QLabel(self)
            self.preview_label.setFixedSize(140, 140)
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setStyleSheet("border: 2px dashed #888; background-color: #222; color: #aaa;")
            self.preview_label.setText("No selection")
            side_layout.addWidget(self.preview_label)

            # Info / resolution label
            self.info_label = QLabel("Resolution: N/A", self)
            side_layout.addWidget(self.info_label)

            # Confirm / Finish button
            self.btn_confirm = QPushButton("Confirm & Use ROIs", self)
            self.btn_confirm.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;")
            self.btn_confirm.clicked.connect(self._confirm_and_close)
            side_layout.addWidget(self.btn_confirm)

            splitter.addWidget(side_panel)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)
            # 75% width for scene view, 25% for side panel (default window width: 1200)
            splitter.setSizes([900, 300])

            # Connect scene signals
            self.scene.item_selected_signal.connect(self._on_scene_item_selected)
            self.scene.roi_changed_signal.connect(self._on_scene_item_geometry_changed)
            self.scene.roi_added_signal.connect(self._on_roi_added)

            # Setup Toolbar
            self._create_toolbar()

            # Load initial image if provided
            if image is not None:
                if isinstance(image, str):
                    if os.path.isfile(image):
                        loaded_img = cv2.imread(image)
                        if loaded_img is not None:
                            self.set_image(loaded_img)
                        else:
                            logger.error("Failed to load image from %s", image)
                    else:
                        logger.error("Image file not found: %s", image)
                elif isinstance(image, np.ndarray):
                    self.set_image(image)

            # Load initial ROIs if provided
            effective_config = config_path or (initial_rois if isinstance(initial_rois, str) else None)
            if effective_config and os.path.isfile(effective_config):
                try:
                    manager = ROIManager()
                    loaded_rois = manager.load_config(effective_config)
                    self.load_rois_dict(loaded_rois)
                except Exception as e:
                    logger.error("Failed to load ROI config from %s: %s", effective_config, e)
            elif isinstance(initial_rois, dict):
                self.load_rois_dict(initial_rois)

        def _create_toolbar(self) -> None:
            toolbar = QToolBar("ROI Tools", self)
            self.addToolBar(toolbar)

            # File / Camera actions
            act_open = QAction("Open Image", self)
            act_open.triggered.connect(self.open_image_dialog)
            toolbar.addAction(act_open)

            if self.camera is not None:
                act_cam = QAction("Capture Frame", self)
                act_cam.triggered.connect(self.capture_from_camera)
                toolbar.addAction(act_cam)

            toolbar.addSeparator()

            # Modes
            self.act_select = QAction("Select / Move", self)
            self.act_select.setCheckable(True)
            self.act_select.setChecked(True)
            self.act_select.triggered.connect(lambda: self.set_mode("select"))
            toolbar.addAction(self.act_select)

            self.act_rect = QAction("Add Rectangle", self)
            self.act_rect.setCheckable(True)
            self.act_rect.triggered.connect(lambda: self.set_mode("add_rect"))
            toolbar.addAction(self.act_rect)

            self.act_circle = QAction("Add Circle", self)
            self.act_circle.setCheckable(True)
            self.act_circle.triggered.connect(lambda: self.set_mode("add_circle"))
            toolbar.addAction(self.act_circle)

            toolbar.addSeparator()

            # Save / Load JSON
            act_save = QAction("Save Config (JSON)", self)
            act_save.triggered.connect(self.save_config_dialog)
            toolbar.addAction(act_save)

            act_load = QAction("Load Config (JSON)", self)
            act_load.triggered.connect(self.load_config_dialog)
            toolbar.addAction(act_load)

            toolbar.addSeparator()

            act_zoom_fit = QAction("Fit to View", self)
            act_zoom_fit.triggered.connect(self.fit_view)
            toolbar.addAction(act_zoom_fit)

        def set_mode(self, mode: str) -> None:
            self.scene.mode = mode
            self.act_select.setChecked(mode == "select")
            self.act_rect.setChecked(mode == "add_rect")
            self.act_circle.setChecked(mode == "add_circle")

        def reset_tool_actions(self) -> None:
            self.set_mode("select")

        def generate_next_label(self) -> str:
            existing = {item.label for item in self.get_all_roi_items()}
            idx = 1
            while f"led_{idx}" in existing:
                idx += 1
            return f"led_{idx}"

        def set_image(self, image: np.ndarray) -> None:
            """Loads an image into the annotator scene."""
            self.current_image = image
            pixmap = ndarray_to_qpixmap(image)

            # Preserve existing ROI items while replacing the background pixmap
            roi_items = self.get_all_roi_items()
            for it in roi_items:
                self.scene.removeItem(it)

            self.scene.clear()
            self.scene.pixmap_item = self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))

            # Re-add items
            for it in roi_items:
                self.scene.addItem(it)

            h, w = image.shape[:2]
            self.info_label.setText(f"Resolution: {w} x {h}")
            self.fit_view()
            self._update_table()

        def fit_view(self) -> None:
            if self.scene.pixmap_item:
                self.view.fitInView(self.scene.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        def open_image_dialog(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
            )
            if path:
                img = cv2.imread(path)
                if img is not None:
                    self.set_image(img)
                else:
                    QMessageBox.critical(self, "Error", f"Failed to load image from {path}")

        def capture_from_camera(self) -> None:
            if self.camera is None:
                QMessageBox.warning(self, "No Camera", "No ICamera instance provided.")
                return
            frame = self.camera.get_frame()
            if frame is not None:
                self.set_image(frame)
            else:
                QMessageBox.warning(self, "Capture Failed", "Failed to capture frame from camera.")

        def get_all_roi_items(self) -> List[Union[GraphicsRectROIItem, GraphicsCircleROIItem]]:
            return [it for it in self.scene.items() if isinstance(it, (GraphicsRectROIItem, GraphicsCircleROIItem))]

        def get_rois_dict(self) -> Dict[str, ROI]:
            """Extracts all items as a dictionary of domain ROI objects."""
            results: Dict[str, ROI] = {}
            for item in self.get_all_roi_items():
                domain_roi = item.to_domain_roi()
                results[domain_roi.label] = domain_roi
            return results

        def load_rois_dict(self, rois: Dict[str, ROI]) -> None:
            """Populates the scene with given domain ROI objects."""
            for item in self.get_all_roi_items():
                self.scene.removeItem(item)

            for label, roi in rois.items():
                if roi.shape == "circle" and isinstance(roi, CircleROI):
                    rect = QRectF(0, 0, 2 * roi.radius, 2 * roi.radius)
                    item = GraphicsCircleROIItem(rect, label=label)
                    item.setPos(roi.center_x - roi.radius, roi.center_y - roi.radius)
                    self.scene.addItem(item)
                elif roi.shape == "rect" and isinstance(roi, RectROI):
                    rect = QRectF(0, 0, roi.width, roi.height)
                    item = GraphicsRectROIItem(rect, label=label)
                    item.setPos(roi.x, roi.y)
                    self.scene.addItem(item)

            self._update_table()

        def _update_table(self, active_label: Optional[str] = None) -> None:
            # Preserve existing table selection or prioritize active_label
            selected_items = self.scene.selectedItems()
            selected_label = active_label or (
                selected_items[0].label if selected_items and hasattr(selected_items[0], "label") else None
            )

            self.table.blockSignals(True)
            self.table.setRowCount(0)
            items = self.get_all_roi_items()

            target_row = -1
            for row, item in enumerate(items):
                self.table.insertRow(row)
                lbl_item = QTableWidgetItem(item.label)
                shape_item = QTableWidgetItem(item.shape_type)

                domain_roi = item.to_domain_roi()
                coords_str = str(domain_roi.bbox)
                coord_item = QTableWidgetItem(coords_str)

                self.table.setItem(row, 0, lbl_item)
                self.table.setItem(row, 1, shape_item)
                self.table.setItem(row, 2, coord_item)

                if selected_label and item.label == selected_label:
                    target_row = row

            if target_row >= 0:
                self.table.selectRow(target_row)

            self.table.blockSignals(False)

        def _on_roi_added(self, item: Any) -> None:
            item_label = getattr(item, "label", None)
            self._update_table(active_label=item_label)
            self._update_crop_preview(item)

        def _on_table_selection_changed(self) -> None:
            selected_rows = self.table.selectionModel().selectedRows()
            if not selected_rows:
                return
            row = selected_rows[0].row()
            target_label = self.table.item(row, 0).text()

            for item in self.get_all_roi_items():
                if item.label == target_label:
                    self.scene.blockSignals(True)
                    self.scene.clearSelection()
                    item.setSelected(True)
                    self.scene.blockSignals(False)
                    self._update_crop_preview(item)
                    break

        def _on_scene_item_selected(self, item: Optional[Any]) -> None:
            if item is None:
                self.preview_label.setText("No selection")
                self.preview_label.setPixmap(QPixmap())
                return

            self._update_crop_preview(item)
            # Sync selection in table
            self.table.blockSignals(True)
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0).text() == item.label:
                    self.table.selectRow(row)
                    break
            self.table.blockSignals(False)

        def _on_scene_item_geometry_changed(self, item: Any) -> None:
            item_label = getattr(item, "label", None)
            self._update_table(active_label=item_label)
            self._update_crop_preview(item)

        def _update_crop_preview(self, item: Any) -> None:
            if self.current_image is None or not hasattr(item, "to_domain_roi"):
                return
            domain_roi = item.to_domain_roi()
            try:
                crop = domain_roi.crop(self.current_image)
                if crop.size > 0:
                    pix = ndarray_to_qpixmap(crop)
                    scaled = pix.scaled(
                        self.preview_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.preview_label.setPixmap(scaled)
                    self.preview_label.setText("")
            except Exception as e:
                logger.debug("Error generating preview crop: %s", e)

        def _rename_selected_roi(self) -> None:
            selected = self.scene.selectedItems()
            if not selected or not hasattr(selected[0], "set_label"):
                return
            item = selected[0]
            new_label, ok = QInputDialog.getText(
                self, "Rename ROI", "New LED Label:", text=item.label
            )
            if ok and new_label.strip():
                item.set_label(new_label.strip())
                self._update_table()

        def _delete_selected_roi(self) -> None:
            selected = self.scene.selectedItems()
            for it in selected:
                self.scene.removeItem(it)
            self._update_table()
            self.preview_label.setText("No selection")
            self.preview_label.setPixmap(QPixmap())

        def save_config_dialog(self) -> None:
            default_path = self.current_config_path or ""
            path, _ = QFileDialog.getSaveFileName(self, "Save ROI Config", default_path, "JSON (*.json)")
            if path:
                self.current_config_path = path
                manager = ROIManager(self.get_rois_dict())
                if self.current_image is not None:
                    h, w = self.current_image.shape[:2]
                    manager.set_reference_resolution(w, h)
                manager.save_config(path)
                QMessageBox.information(self, "Saved", f"Configuration saved to {path}")

        def load_config_dialog(self) -> None:
            default_path = self.current_config_path or ""
            path, _ = QFileDialog.getOpenFileName(self, "Load ROI Config", default_path, "JSON (*.json)")
            if path:
                self.current_config_path = path
                manager = ROIManager()
                rois = manager.load_config(path)
                self.load_rois_dict(rois)
                QMessageBox.information(self, "Loaded", f"Loaded {len(rois)} ROI(s) from {path}")

        def _confirm_and_close(self) -> None:
            self.confirmed = True
            self.close()


class ROIAnnotator:
    """High-level annotator orchestrator conforming to system architecture UML."""

    def __init__(self, window_name: str = "LED ROI Annotator") -> None:
        """Initialize ROIAnnotator.

        Args:
            window_name: Window title string.
        """
        self._window_name = window_name

    @property
    def window_name(self) -> str:
        return self._window_name

    def _ensure_qapp(self) -> "QApplication":
        if not HAS_PYSIDE6:
            raise ImportError(
                "PySide6 is required to use ROIAnnotator GUI. Install it with: pip install PySide6"
            )
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        return app

    def select_single(
        self,
        image: Optional[Union[np.ndarray, str]] = None,
        shape: str = "rect",
        config_path: Optional[str] = None,
        camera: Optional[Any] = None,
        window_name: Optional[str] = None,
    ) -> Optional[ROI]:
        """Interactively allows the user to define a single ROI on an image.

        Args:
            image: Source frame (NumPy array) or file path to an image.
            shape: Default shape ('rect' or 'circle').
            config_path: Optional path to a JSON configuration file.
            camera: Optional ICamera instance for live frame capture.
            window_name: Optional override for window title.

        Returns:
            The selected ROI object, or None if cancelled/empty.
        """
        app = self._ensure_qapp()
        title = window_name or f"{self._window_name} - Select Single {shape.capitalize()}"
        win = ROIAnnotatorWindow(
            image=image,
            config_path=config_path,
            camera=camera,
            window_name=title,
        )
        win.set_mode(f"add_{shape}" if shape in ("rect", "circle") else "select")
        win.show()
        app.exec()

        rois = win.get_rois_dict()
        if not rois:
            return None
        # Return first created ROI
        return next(iter(rois.values()))

    def select_multiple(
        self,
        image: Optional[Union[np.ndarray, str]] = None,
        initial_rois: Optional[Union[Dict[str, ROI], str]] = None,
        config_path: Optional[str] = None,
        camera: Optional[Any] = None,
        window_name: Optional[str] = None,
    ) -> Dict[str, ROI]:
        """Interactively allows user to view, edit, and define multiple ROIs on an image.

        Args:
            image: Source frame (NumPy array) or file path to an image.
            initial_rois: Optional dictionary of existing ROIs or file path to a JSON config.
            config_path: Optional file path to a JSON config file to preload.
            camera: Optional ICamera instance to support direct frame capture.
            window_name: Optional override for window title.

        Returns:
            Dictionary mapping labels to configured ROI objects.
        """
        app = self._ensure_qapp()
        title = window_name or self._window_name
        win = ROIAnnotatorWindow(
            image=image,
            initial_rois=initial_rois,
            config_path=config_path,
            camera=camera,
            window_name=title,
        )
        win.show()
        app.exec()

        return win.get_rois_dict()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive LED ROI Annotator")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image file")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to existing ROI JSON configuration")
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        choices=["ids", "opencv", "mock"],
        help="Optional camera driver type to connect for live capture",
    )
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    loaded_image = None
    if args.image:
        if os.path.isfile(args.image):
            loaded_image = cv2.imread(args.image)
            if loaded_image is None:
                print(f"[Warning] Failed to read image from {args.image}")
        else:
            print(f"[Warning] Image file not found: {args.image}")

    initial_rois = None
    if args.config:
        if os.path.isfile(args.config):
            manager = ROIManager()
            initial_rois = manager.load_config(args.config)
            print(f"Loaded {len(initial_rois)} ROI(s) from {args.config}")
        else:
            print(f"[Warning] Config file not found: {args.config}")

    camera_instance = None
    if args.camera:
        try:
            from src.camera import CameraFactory

            camera_instance = CameraFactory.create_camera(args.camera)
            if not camera_instance.open():
                print(f"[Warning] Could not open camera '{args.camera}'")
                camera_instance = None
            else:
                print(f"Camera '{args.camera}' connected successfully.")
        except Exception as e:
            print(f"[Warning] Could not initialize camera: {e}")

    window = ROIAnnotatorWindow(
        image=loaded_image,
        initial_rois=initial_rois,
        camera=camera_instance,
    )
    window.show()
    exit_code = app.exec()

    if camera_instance is not None:
        camera_instance.close()

    sys.exit(exit_code)