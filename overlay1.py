import logging
from typing import Optional
from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QContextMenuEvent, QFont, QMouseEvent, QScreen
from PySide6.QtWidgets import QApplication, QGraphicsDropShadowEffect, QLabel, QMenu, QVBoxLayout, QWidget

logger = logging.getLogger("realtime_translator.overlay")

class OverlayWindow(QWidget):
    caption_updated = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 20, 20, 20)
        
        self.caption_label = QLabel(self)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet("color: white; font-weight: bold;")

        self.caption_label.setText("Waiting for Gemini Live audio translation...")

        self.current_sentence = ""
        
        font = QFont("Arial", 24)
        self.caption_label.setFont(font)

        shadow = QGraphicsDropShadowEffect(self.caption_label)
        shadow.setBlurRadius(18.0)
        shadow.setOffset(0, 2.0)
        shadow.setColor(QColor(0, 0, 0, 220))
        self.caption_label.setGraphicsEffect(shadow)
        
        self._layout.addWidget(self.caption_label)

        self.setWindowOpacity(0.01)
        self._fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self._fade_animation.setDuration(220)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.caption_updated.connect(self._on_caption_updated)
        
        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.setInterval(4000)
        self._clear_timer.timeout.connect(self.hide_text)

        self._clear_timer.start()
        self.resize(800, 150)
        self._center_on_screen_bottom()

    def _on_caption_updated(self, text: str):
        if not text: return
        
        # Reset the phrase builder if the window has fully faded out
        if self.windowOpacity() <= 0.05:
            self.current_sentence = ""

        # Intelligently stitch incoming chunks keeping sentence spacing pristine
        if self.current_sentence and not text.startswith(" "):
            self.current_sentence += " " + text
        else:
            self.current_sentence += text

        # ==================== YOUTUBE-STYLE 3-LINE ROLLGUARD ====================
        import textwrap
        
        # Wrap the accumulated text cleanly based on word boundaries.
        # At font size 24 and window width 800, ~50 characters fit per line.
        wrapped_lines = textwrap.wrap(self.current_sentence.strip(), width=50)
        
        # Keep only the last 3 lines to simulate a scrolling subtitle display
        if len(wrapped_lines) > 3:
            wrapped_lines = wrapped_lines[-3:]
            # Optional: Update our internal state so the hidden history doesn't grow infinitely
            self.current_sentence = " ".join(wrapped_lines)

        # Re-join the lines with actual line breaks for the QLabel
        formatted_caption = "\n".join(wrapped_lines)
        # ========================================================================

        # Output the bounded lines directly to the desktop window
        self.caption_label.setText(formatted_caption)
        self._clear_timer.start()
        
        # Smoothly trigger the fade-in animation if hidden
        if self.windowOpacity() < 0.75:
            self._fade_animation.stop()
            self._fade_animation.setStartValue(self.windowOpacity())
            self._fade_animation.setEndValue(0.75)
            self._fade_animation.start()

    def hide_text(self):
        self._fade_animation.stop()
        self._fade_animation.setStartValue(self.windowOpacity())
        self._fade_animation.setEndValue(0.01)
        self._fade_animation.start()
        # Clear the memory string when the silence timer triggers this method
        self.current_sentence = ""

    def _center_on_screen_bottom(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = screen.height() - self.height() - 60
        self.move(x, y)

    # Basic dragging support
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPos() - self._drag_position)
            event.accept()

    def contextMenuEvent(self, event: QContextMenuEvent):
        menu = QMenu(self)
        exit_act = QAction("Close Translator", self)
        exit_act.triggered.connect(QApplication.quit)
        menu.addAction(exit_act)
        menu.exec_(event.globalPos())