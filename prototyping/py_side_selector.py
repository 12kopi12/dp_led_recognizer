import sys
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsSimpleTextItem,
    QToolBar, QInputDialog, QFileDialog, QMessageBox, QGraphicsItem
)
from PySide6.QtGui import QPixmap, QPen, QColor, QBrush, QAction, QFont
from PySide6.QtCore import Qt, QRectF


class ResizeHandle(QGraphicsRectItem):
    """Úchyt v pravém dolním rohu pro změnu velikosti objektu."""
    def __init__(self, parent):
        super().__init__(-5, -5, 10, 10, parent)
        self.setBrush(QBrush(QColor(255, 165, 0)))
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)

    def mousePressEvent(self, event):
        event.accept()

    def mouseMoveEvent(self, event):
        parent = self.parentItem()
        # Pozice myši v souřadnicích rodičovského tvaru
        pos = parent.mapFromScene(event.scenePos())
        rect = parent.rect()

        # Minimální velikost 10x10 px
        new_w = max(10.0, pos.x() - rect.left())
        new_h = max(10.0, pos.y() - rect.top())

        rect.setWidth(new_w)
        rect.setHeight(new_h)
        parent.setRect(rect)
        parent.update_dependents()
        event.accept()


def setup_led_item(item, label):
    """Pomocná funkce pro nastavení vzhledu, textu a úchytu."""
    item.setFlags(
        QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
        QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
        QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
    )
    item.setPen(QPen(QColor(0, 255, 0), 2))
    item.setBrush(QBrush(QColor(0, 255, 0, 40)))

    # Textový popisek jako potomek (hýbe se automaticky s objektem)
    item.text_item = QGraphicsSimpleTextItem(label, item)
    item.text_item.setBrush(QBrush(QColor(255, 255, 0)))  # Žlutý text
    font = QFont("Arial", 10, QFont.Weight.Bold)
    item.text_item.setFont(font)

    # Úchyt pro změnu velikosti
    item.handle = ResizeHandle(item)
    item.update_dependents()


class LedRectItem(QGraphicsRectItem):
    def __init__(self, rect, label=""):
        super().__init__(rect)
        self.label = label
        self.shape_type = "rect"
        setup_led_item(self, label)

    def update_dependents(self):
        r = self.rect()
        self.text_item.setPos(r.left(), r.top() - 20)
        self.handle.setPos(r.bottomRight())


class LedCircleItem(QGraphicsEllipseItem):
    def __init__(self, rect, label=""):
        super().__init__(rect)
        self.label = label
        self.shape_type = "circle"
        setup_led_item(self, label)

    def update_dependents(self):
        r = self.rect()
        self.text_item.setPos(r.left(), r.top() - 20)
        self.handle.setPos(r.bottomRight())


class InteractiveScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "select"  # "select", "add_rect", "add_circle"
        self.start_point = None
        self.temp_item = None

    def mousePressEvent(self, event):
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

    def mouseMoveEvent(self, event):
        if self.temp_item and self.start_point:
            rect = QRectF(self.start_point, event.scenePos()).normalized()
            self.temp_item.setRect(rect)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.temp_item:
            rect = self.temp_item.rect().normalized()
            self.removeItem(self.temp_item)
            self.temp_item = None

            if rect.width() > 5 and rect.height() > 5:
                label, ok = QInputDialog.getText(None, "Označení LED", "Zadejte název / label diody:")
                if ok and label.strip():
                    # Lokální souřadnice začínají na (0,0), pozice ve scéně je nastavena přes setPos
                    local_rect = QRectF(0, 0, rect.width(), rect.height())
                    if self.mode == "add_rect":
                        item = LedRectItem(local_rect, label=label.strip())
                    else:
                        item = LedCircleItem(local_rect, label=label.strip())
                    item.setPos(rect.topLeft())
                    self.addItem(item)

            self.mode = "select"
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LED ROI Annotator")
        self.resize(1000, 700)

        self.scene = InteractiveScene(self)
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        toolbar = QToolBar("Nástroje")
        self.addToolBar(toolbar)

        act_open = QAction("Otevřít obrázek", self)
        act_open.triggered.connect(self.load_image)
        toolbar.addAction(act_open)

        toolbar.addSeparator()

        self.act_rect = QAction("Přidat obdélník", self)
        self.act_rect.triggered.connect(lambda: self.set_mode("add_rect"))
        toolbar.addAction(self.act_rect)

        self.act_circle = QAction("Přidat kruh", self)
        self.act_circle.triggered.connect(lambda: self.set_mode("add_circle"))
        toolbar.addAction(self.act_circle)

        toolbar.addSeparator()

        act_export = QAction("Exportovat souřadnice", self)
        act_export.triggered.connect(self.export_data)
        toolbar.addAction(act_export)

    def set_mode(self, mode):
        self.scene.mode = mode

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vyberte obrázek", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.scene.clear()
            pixmap = QPixmap(path)
            self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QRectF(pixmap.rect()))

    def export_data(self):
        results = {}
        for item in self.scene.items():
            if isinstance(item, (LedRectItem, LedCircleItem)):
                # mapRectToScene vezme pouze čistý tvar LED (ignoruje přesahující text a úchyt)
                rect = item.mapRectToScene(item.rect())
                results[item.label] = {
                    "type": item.shape_type,
                    "x": int(rect.x()),
                    "y": int(rect.y()),
                    "width": int(rect.width()),
                    "height": int(rect.height())
                }
        print("--- Nalezené ROI zóny ---")
        print(json.dumps(results, indent=2))
        QMessageBox.information(self, "Export hotov", f"Exportováno {len(results)} prvků do konzole.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())