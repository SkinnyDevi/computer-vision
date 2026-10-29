"""Etiquetador de matrículas para YOLO.

Uso: python etiquetator.py [carpeta_con_imagenes]

- Arrastra con el ratón para dibujar un rectángulo (puedes dibujar varios).
- Enter: se abre un popup por cada rectángulo nuevo pidiendo la matrícula,
  se guarda labels/<imagen>.json y se pasa a la siguiente imagen.
- Clic derecho sobre una etiqueta guardada: editar su matrícula o eliminarla (botón Remove).
- Ctrl + rueda: zoom (sobre el cursor). Rueda: desplazar. Botón central y arrastrar: mover.
  Con zoom se puede seguir dibujando rectángulos. Tecla 0: restablecer zoom.
- Clic derecho sobre un rectángulo amarillo (sin matrícula): eliminarlo (Remove).
- Líneas guía en cruz siguen al cursor para ajustar mejor el rectángulo.
- Ctrl+Z / Backspace: deshacer el último rectángulo.
- Flechas izquierda/derecha: imagen anterior / siguiente (sin guardar).
- Esc: cierra la aplicación.

Formato del JSON (rectangle = [cx, cy, w, h] normalizados entre 0 y 1):
[{"rectangle": [cx, cy, w, h], "plate": "0000AAA"}, ...]
Si el popup se deja vacío se guarda "None".
"""
import json
import sys
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QApplication, QDialog, QDialogButtonBox,
                             QFileDialog, QInputDialog, QLabel, QLineEdit,
                             QMessageBox, QVBoxLayout, QWidget)

EXTENSIONES = {".jpg", ".jpeg", ".heic"}
OTRAS_IMAGENES = {".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


class DialogoEditar(QDialog):
    """Edita la matrícula; el botón Remove elimina la etiqueta."""
    ELIMINAR = 2

    def __init__(self, parent, texto):
        super().__init__(parent)
        self.setWindowTitle("Editar matrícula")
        self.campo = QLineEdit(texto)
        botones = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        remove = botones.addButton("Remove",
                                   QDialogButtonBox.ButtonRole.DestructiveRole)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        remove.clicked.connect(lambda: self.done(self.ELIMINAR))
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Nueva matrícula (vacío = None):"))
        lay.addWidget(self.campo)
        lay.addWidget(botones)


class Etiquetador(QWidget):
    def __init__(self, carpeta: Path):
        super().__init__()
        todas = sorted(p for p in carpeta.iterdir()
                       if p.suffix.lower() in OTRAS_IMAGENES | EXTENSIONES)
        self.imagenes = [p for p in todas if p.suffix.lower() in EXTENSIONES]
        rechazadas = [p.name for p in todas if p.suffix.lower() not in EXTENSIONES]
        if rechazadas:
            QMessageBox.warning(
                None, "Formato no válido",
                "Las imágenes tienen que estar en formato JPG. "
                "Se ignoran estos archivos:\n\n" + "\n".join(rechazadas))
        self.imagenes = [p for p in self.imagenes if not QImage(str(p)).isNull()]
        if not self.imagenes:
            QMessageBox.critical(None, "Sin imágenes",
                                 f"No hay imágenes JPG en {carpeta}")
            raise SystemExit(1)
        self.indice = 0
        self.imagen = QImage()
        self.guardadas = []   # [{"rect": QRectF normalizado, "plate": str}]
        self.nuevas = []      # QRectF normalizados pendientes de matrícula
        self.inicio = None    # QPointF en coords de imagen normalizadas
        self.actual = None    # QRectF normalizado mientras se arrastra
        self.zoom = 1.0       # 1 = imagen ajustada a la ventana
        self.pan = QPointF(0, 0)  # desplazamiento en px respecto al centro
        self.pan_ini = None   # posición del ratón al empezar a mover con el botón central
        self.cursor_pos = None  # QPointF del ratón, para las líneas guía
        self.preguntando = None  # QRectF cuya matrícula se está pidiendo (rojo)
        self.setMouseTracking(True)
        self.resize(1200, 800)
        self.cargar()

    # ---------- carga / guardado ----------
    def ruta_json(self) -> Path:
        ruta = self.imagenes[self.indice]
        return ruta.parent / "annotations" / (ruta.stem + ".json")

    def cargar(self):
        ruta = self.imagenes[self.indice]
        self.imagen = QImage(str(ruta))
        self.guardadas, self.nuevas = [], []
        self.inicio = self.actual = None
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        if self.ruta_json().exists():
            try:
                datos = json.loads(self.ruta_json().read_text())
                for d in datos:
                    cx, cy, w, h = d["rectangle"]
                    self.guardadas.append({
                        "rect": QRectF(cx - w / 2, cy - h / 2, w, h),
                        "plate": d["plate"]})
            except (ValueError, KeyError, TypeError):
                QMessageBox.warning(self, "Aviso",
                                    f"No se pudo leer {self.ruta_json().name}")
        self.setWindowTitle(f"[{self.indice + 1}/{len(self.imagenes)}] {ruta.name}")
        self.update()

    def guardar(self):
        datos = []
        for g in self.guardadas:
            r = g["rect"]
            datos.append({"rectangle": [r.center().x(), r.center().y(),
                                        r.width(), r.height()],
                          "plate": g["plate"]})
        self.ruta_json().parent.mkdir(exist_ok=True)
        self.ruta_json().write_text(json.dumps(datos, indent=2))

    def confirmar(self):
        """Enter: pide la matrícula de cada rectángulo nuevo y guarda."""
        for i, rect in enumerate(self.nuevas):
            self.preguntando = rect
            self.repaint()
            texto, ok = QInputDialog.getText(
                self, "Matrícula",
                f"Matrícula del rectángulo {i + 1}/{len(self.nuevas)} "
                "(vacío = None):")
            if not ok:
                # cancelado: se conservan los que faltan por etiquetar
                self.nuevas = self.nuevas[i:]
                self.preguntando = None
                self.update()
                return
            texto = texto.strip().upper().replace(" ", "")
            self.guardadas.append({"rect": rect, "plate": texto or "None"})
        self.nuevas, self.preguntando = [], None
        if self.guardadas:
            self.guardar()
        self.siguiente()

    def siguiente(self):
        if self.indice < len(self.imagenes) - 1:
            self.indice += 1
            self.cargar()
        else:
            QMessageBox.information(self, "Fin", "No hay más imágenes.")

    def anterior(self):
        if self.indice > 0:
            self.indice -= 1
            self.cargar()

    # ---------- geometría ----------
    def area_imagen(self) -> QRectF:
        """Rectángulo del widget donde se pinta la imagen (mantiene aspecto)."""
        if self.imagen.isNull():
            return QRectF()
        w, h = self.imagen.width(), self.imagen.height()
        escala = min(self.width() / w, self.height() / h) * self.zoom
        iw, ih = w * escala, h * escala
        self.pan = self.limitar_pan(self.pan, iw, ih)
        return QRectF((self.width() - iw) / 2 + self.pan.x(),
                      (self.height() - ih) / 2 + self.pan.y(), iw, ih)

    def limitar_pan(self, pan, iw, ih) -> QPointF:
        """El desplazamiento solo existe en los ejes donde la imagen desborda."""
        mx = max(0.0, (iw - self.width()) / 2)
        my = max(0.0, (ih - self.height()) / 2)
        return QPointF(min(max(pan.x(), -mx), mx), min(max(pan.y(), -my), my))

    def a_norm_libre(self, p) -> QPointF:
        """Como a_norm pero sin recortar a [0, 1] (puede caer fuera de la imagen)."""
        a = self.area_imagen()
        if a.width() <= 0 or a.height() <= 0:
            return QPointF(-1.0, -1.0)
        return QPointF((p.x() - a.x()) / a.width(), (p.y() - a.y()) / a.height())

    def a_norm(self, p) -> QPointF:
        a = self.area_imagen()
        if a.width() <= 0 or a.height() <= 0:
            return QPointF(0.0, 0.0)
        x = min(max((p.x() - a.x()) / a.width(), 0.0), 1.0)
        y = min(max((p.y() - a.y()) / a.height(), 0.0), 1.0)
        return QPointF(x, y)

    def a_pantalla(self, r: QRectF) -> QRectF:
        a = self.area_imagen()
        return QRectF(a.x() + r.x() * a.width(), a.y() + r.y() * a.height(),
                      r.width() * a.width(), r.height() * a.height())

    # ---------- eventos ----------
    def editar_etiqueta(self, pos):
        """Clic derecho: modifica la matrícula de la etiqueta bajo el cursor."""
        p = self.a_norm_libre(pos)
        candidatas = [g for g in self.guardadas if g["rect"].contains(p)]
        if not candidatas:
            self.eliminar_pendiente(p)
            return
        # si hay solapadas, la más pequeña
        g = min(candidatas, key=lambda g: g["rect"].width() * g["rect"].height())
        self.preguntando = g["rect"]
        self.repaint()
        dlg = DialogoEditar(self, g["plate"])
        res = dlg.exec()
        self.preguntando = None
        if res == QDialog.DialogCode.Accepted:
            texto = dlg.campo.text().strip().upper().replace(" ", "")
            g["plate"] = texto or "None"
            self.guardar()
        elif res == DialogoEditar.ELIMINAR:
            self.guardadas.remove(g)
            self.guardar()
        self.update()

    def eliminar_pendiente(self, p):
        """Clic derecho sobre un rectángulo aún sin matrícula: permite quitarlo."""
        candidatas = [r for r in self.nuevas if r.contains(p)]
        if not candidatas:
            return
        r = min(candidatas, key=lambda r: r.width() * r.height())
        self.preguntando = r
        self.repaint()
        caja = QMessageBox(self)
        caja.setWindowTitle("Rectángulo sin etiquetar")
        caja.setText("Este rectángulo aún no tiene matrícula.")
        remove = caja.addButton("Remove", QMessageBox.ButtonRole.DestructiveRole)
        caja.addButton(QMessageBox.StandardButton.Cancel)
        caja.exec()
        self.preguntando = None
        if caja.clickedButton() is remove:
            self.nuevas.remove(r)
        self.update()

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            antes = self.a_norm_libre(e.position())
            self.zoom = min(max(self.zoom * (1.25 if e.angleDelta().y() > 0 else 0.8), 1.0), 40.0)
            if self.zoom == 1.0:
                self.pan = QPointF(0, 0)
            else:
                # mantiene bajo el cursor el mismo punto de la imagen
                a = self.area_imagen()
                self.pan += QPointF(e.position().x() - (a.x() + antes.x() * a.width()),
                                    e.position().y() - (a.y() + antes.y() * a.height()))
            self.update()
        else:
            self.pan += QPointF(e.angleDelta().x(), e.angleDelta().y())
            self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self.pan_ini = e.position()
        elif e.button() == Qt.MouseButton.RightButton:
            self.editar_etiqueta(e.position())
        elif e.button() == Qt.MouseButton.LeftButton:
            self.inicio = self.a_norm(e.position())
            self.actual = QRectF(self.inicio, self.inicio)

    def leaveEvent(self, _):
        self.cursor_pos = None
        self.update()

    def mouseMoveEvent(self, e):
        self.cursor_pos = e.position()
        self.update()
        if self.pan_ini is not None:
            self.pan += e.position() - self.pan_ini
            self.pan_ini = e.position()
            self.update()
        elif self.inicio is not None:
            self.actual = QRectF(self.inicio, self.a_norm(e.position())).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self.pan_ini = None
        elif e.button() == Qt.MouseButton.LeftButton and self.inicio is not None:
            r = QRectF(self.inicio, self.a_norm(e.position())).normalized()
            a = self.area_imagen()
            # ignora clics sueltos (menos de 3 px)
            if r.width() * a.width() > 3 and r.height() * a.height() > 3:
                self.nuevas.append(r)
            self.inicio = self.actual = None
            self.update()

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.nuevas:
                self.confirmar()
            else:
                self.siguiente()
        elif k == Qt.Key.Key_Backspace or (
                k == Qt.Key.Key_Z and e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if self.nuevas:
                self.nuevas.pop()
            elif self.guardadas:
                self.guardadas.pop()
                self.guardar()
            self.update()
        elif k == Qt.Key.Key_0:
            self.zoom, self.pan = 1.0, QPointF(0, 0)
            self.update()
        elif k == Qt.Key.Key_Right:
            self.siguiente()
        elif k == Qt.Key.Key_Left:
            self.anterior()
        elif k == Qt.Key.Key_Escape:
            self.close()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self.imagen.isNull():
            return
        p.drawImage(self.area_imagen(), self.imagen)
        p.setFont(QFont("monospace", 12, QFont.Weight.Bold))

        p.setPen(QPen(QColor(0, 220, 0), 2))
        for g in self.guardadas:
            r = self.a_pantalla(g["rect"])
            p.drawRect(r)
            p.drawText(r.topLeft() + QPointF(2, -4), g["plate"])

        p.setPen(QPen(QColor(255, 200, 0), 2))
        for r in self.nuevas:
            p.drawRect(self.a_pantalla(r))

        if self.preguntando is not None:
            p.setPen(QPen(QColor(255, 0, 0), 4))
            p.drawRect(self.a_pantalla(self.preguntando))

        if self.inicio is not None and self.actual is not None:
            p.setPen(QPen(QColor(255, 60, 60), 2, Qt.PenStyle.DashLine))
            p.drawRect(self.a_pantalla(self.actual))

        if self.cursor_pos is not None:
            # líneas guía horizontal y vertical que cruzan toda la ventana
            p.setPen(QPen(QColor(0, 255, 255, 200), 1))
            x, y = self.cursor_pos.x(), self.cursor_pos.y()
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.drawLine(QPointF(0, y), QPointF(self.width(), y))


def main():
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        carpeta = Path(sys.argv[1])
    else:
        d = QFileDialog.getExistingDirectory(None, "Carpeta con imágenes")
        if not d:
            return
        carpeta = Path(d)
    ventana = Etiquetador(carpeta)
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
