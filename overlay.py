import logging
from typing import List, Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QFont,
    QMouseEvent,
    QScreen,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("realtime_translator.overlay")

# --------------------------------------------------------------------------
# Configuration constants (no magic numbers below this point)
# --------------------------------------------------------------------------
DEFAULT_FONT_SIZE_PX: int = 24
MIN_FONT_SIZE_PX: int = 14
MAX_FONT_SIZE_PX: int = 64
FONT_SIZE_STEP_PX: int = 2

DEFAULT_CONTAINER_OPACITY: float = 0.75
OPACITY_STEP: float = 0.1

FADE_DURATION_MS: int = 220
FADE_EASING_CURVE = QEasingCurve.Type.InOutQuad

MIN_UPDATE_INTERVAL_MS: int = 60          # coalesces rapid caption_updated bursts to avoid flicker
MAX_VISIBLE_LINES: int = 2
MAX_LINE_CHARS: int = 42

SHADOW_BLUR_RADIUS_PX: float = 18.0
SHADOW_OFFSET_PX: float = 2.0
SHADOW_COLOR = QColor(0, 0, 0, 220)

WINDOW_WIDTH_RATIO: float = 0.7
WINDOW_BOTTOM_MARGIN_PX: int = 60
WINDOW_HEIGHT_PX: int = 130
MIN_WINDOW_WIDTH_PX: int = 600
MIN_WINDOW_HEIGHT_PX: int = 80

REFERENCE_DPI: float = 96.0               # baseline DPI that DEFAULT_FONT_SIZE_PX is tuned for


def wrap_caption(text: str, max_chars: int = MAX_LINE_CHARS, max_lines: int = MAX_VISIBLE_LINES) -> str:
    """Wraps caption text into a bounded set of lines without ever splitting inside a word.

    Because wrapping only ever occurs on whitespace boundaries, numbers, names, abbreviations,
    and URLs (all single contiguous tokens) are never broken mid-token. When the wrapped text
    would exceed `max_lines`, only the most recent `max_lines` lines are kept, since live
    captions favor showing the newest content.

    Args:
        text: The raw caption text (already flattened to a single line internally).
        max_chars: Approximate maximum characters per rendered line.
        max_lines: Maximum number of lines to display simultaneously.

    Returns:
        The text reformatted with newline separators, trimmed to at most `max_lines` lines.
    """
    words = text.split()
    if not words:
        return ""

    lines: List[str] = []
    current_line = ""

    for word in words:
        candidate = f"{current_line} {word}".strip()
        if len(candidate) <= max_chars or not current_line:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    return "\n".join(lines)


class OverlayWindow(QWidget):
    """A transparent, always-on-top, draggable desktop overlay for displaying live captions.

    Provides a stylized subtitle display mimicking YouTube Live Captions, with smooth
    fade-in/fade-out transitions, a soft drop shadow, automatic DPI-aware font scaling,
    multi-monitor awareness, and word-boundary-safe line wrapping capped at two visible
    lines. Built with thread-safe PySide6 signals to receive real-time streaming text
    updates from asynchronous background pipelines without stalling UI rendering.
    """

    # Thread-safe signal to push caption text updates directly to the UI thread main loop
    caption_updated = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initializes the transparent caption overlay widget with default dimensions and geometry."""
        super().__init__(parent)

        # Configuration properties
        self.base_font_size: int = DEFAULT_FONT_SIZE_PX
        self.font_size: int = DEFAULT_FONT_SIZE_PX
        self.container_opacity: float = DEFAULT_CONTAINER_OPACITY
        self._drag_position: QPoint = QPoint()
        self._current_screen: Optional[QScreen] = None
        self._is_visible_text: bool = False

        # Window optimization configuration flags
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # UI Elements layout initialization
        self._layout: QVBoxLayout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 20, 20, 20)

        self.caption_label: QLabel = QLabel(self)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label.setWordWrap(True)

        shadow = QGraphicsDropShadowEffect(self.caption_label)
        shadow.setBlurRadius(SHADOW_BLUR_RADIUS_PX)
        shadow.setOffset(0, SHADOW_OFFSET_PX)
        shadow.setColor(SHADOW_COLOR)
        self.caption_label.setGraphicsEffect(shadow)

        self._layout.addWidget(self.caption_label)

        # Fade animation operates on the window's native opacity so the label, its shadow,
        # and the background box all cross-fade together as a single unit.
        self.setWindowOpacity(0.01)
        self._fade_animation: QPropertyAnimation = QPropertyAnimation(self, b"windowOpacity")
        self._fade_animation.setDuration(FADE_DURATION_MS)
        self._fade_animation.setEasingCurve(FADE_EASING_CURVE)

        # Coalesces bursts of rapid caption_updated emissions into smooth, throttled UI writes.
        self._pending_text: Optional[str] = None
        self._update_throttle: QTimer = QTimer(self)
        self._update_throttle.setSingleShot(True)
        self._update_throttle.setInterval(MIN_UPDATE_INTERVAL_MS)
        self._update_throttle.timeout.connect(self._apply_pending_text)

        # Apply visual styles and anchor placement
        self._update_stylesheets()
        self._center_on_screen_bottom()
        self._sync_dpi_scaling()

        # Connect internal signal handling safely
        self.caption_updated.connect(self.set_caption)

        logger.info("OverlayWindow initialized successfully with transparent WASAPI layout configuration.")

    def _update_stylesheets(self) -> None:
        """Dynamically builds and reapplies CSS stylesheets mapping font sizes and container opacities."""
        rgba_color = f"rgba(0, 0, 0, {self.container_opacity})"
        style = f"""
            QLabel {{
                color: #FFFFFF;
                background-color: {rgba_color};
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: {self.font_size}px;
                font-weight: bold;
                border-radius: 8px;
                padding: 14px 24px;
            }}
        """
        self.caption_label.setStyleSheet(style)
        self.caption_label.adjustSize()
        self.adjustSize()

    def _center_on_screen_bottom(self) -> None:
        """Initializes geometry to position the caption bar at the safe lower-third of the current screen."""
        screen = self._active_screen()
        self._current_screen = screen
        if not screen:
            self.resize(1000, WINDOW_HEIGHT_PX)
            return

        screen_geometry = screen.availableGeometry()
        width = int(screen_geometry.width() * WINDOW_WIDTH_RATIO)
        height = WINDOW_HEIGHT_PX

        x = screen_geometry.x() + int((screen_geometry.width() - width) / 2)
        y = screen_geometry.y() + screen_geometry.height() - height - WINDOW_BOTTOM_MARGIN_PX

        self.setGeometry(x, y, width, height)
        self.setMinimumWidth(MIN_WINDOW_WIDTH_PX)
        self.setMinimumHeight(MIN_WINDOW_HEIGHT_PX)

    def _active_screen(self) -> Optional[QScreen]:
        """Resolves the screen the overlay currently occupies, falling back to the primary display.

        Returns:
            The QScreen under the overlay's current position, or the primary screen if unset.
        """
        if self.isVisible():
            screen_at_pos = QApplication.screenAt(self.geometry().center())
            if screen_at_pos:
                return screen_at_pos
        return QApplication.primaryScreen()

    def _sync_dpi_scaling(self) -> None:
        """Recomputes the effective font size based on the current screen's DPI.

        Keeps caption legibility consistent when the overlay is dragged between monitors
        with different scaling factors (e.g. a 4K 150%-scaled display next to a 1080p display).
        """
        screen = self._active_screen()
        if not screen:
            return

        dpi_ratio = screen.logicalDotsPerInch() / REFERENCE_DPI if screen.logicalDotsPerInch() > 0 else 1.0
        scaled_size = int(round(self.base_font_size * dpi_ratio))
        new_font_size = max(MIN_FONT_SIZE_PX, min(scaled_size, MAX_FONT_SIZE_PX))

        if new_font_size != self.font_size:
            self.font_size = new_font_size
            self._update_stylesheets()
            logger.debug(f"DPI-aware font scaling applied: {self.font_size}px (ratio {dpi_ratio:.2f}).")

    @Slot(str)
    def set_caption(self, text: str) -> None:
        """Queues a caption text update for smooth, throttled application to the UI.

        Args:
            text: Pure text block or translated sentence segment to display.
        """
        self._pending_text = text
        if not self._update_throttle.isActive():
            self._update_throttle.start()

    def _apply_pending_text(self) -> None:
        """Applies the most recently queued caption text, handling fade transitions and wrapping."""
        if self._pending_text is None:
            return

        cleaned_text = self._pending_text.strip()
        self._pending_text = None

        wrapped_text = wrap_caption(cleaned_text) if cleaned_text else ""
        self.caption_label.setText(wrapped_text)
        self.caption_label.adjustSize()
        self.adjustSize()

        now_has_text = bool(wrapped_text)
        if now_has_text and not self._is_visible_text:
            self._fade_to(1.0)
        elif not now_has_text and self._is_visible_text:
            self._fade_to(0.0)
        # Text-to-text updates (both non-empty) intentionally skip the fade animation entirely
        # to avoid a flicker/pulse on every incremental word; the label swap alone is smooth
        # since it happens at most once per MIN_UPDATE_INTERVAL_MS.

        self._is_visible_text = now_has_text

    def _fade_to(self, target_opacity: float) -> None:
        """Animates the overlay's window opacity to the given target value.

        Args:
            target_opacity: Destination opacity in [0.0, 1.0].
        """
        self._fade_animation.stop()
        self._fade_animation.setStartValue(self.windowOpacity())
        self._fade_animation.setEndValue(target_opacity)
        self._fade_animation.start()

    def update_caption(self, text: str, is_final: bool = False) -> None:
        self.caption_updated.emit(text)

    def clear_caption(self) -> None:
        """Compatibility shim to clear the currently displayed caption text."""
        self.set_caption("")

    def set_font_size(self, size: int) -> None:
        """Public configuration modifier to change the caption display font sizing layout.

        Args:
            size: Linear font size value in pixels, used as the DPI-scaling baseline.
                Bound between 14px and 64px.
        """
        self.base_font_size = max(MIN_FONT_SIZE_PX, min(size, MAX_FONT_SIZE_PX))
        self._sync_dpi_scaling()
        self._update_stylesheets()
        logger.debug(f"Overlay base font size adjusted to: {self.base_font_size}px")

    def set_overlay_opacity(self, opacity: float) -> None:
        """Public configuration modifier to change the alpha transparency of the backing label container.

        Args:
            opacity: Floating percentage scalar parameter [0.0 (Invisible) to 1.0 (Solid Opaque)].
        """
        self.container_opacity = max(0.0, min(opacity, 1.0))
        self._update_stylesheets()
        logger.debug(f"Overlay background opacity configured to: {self.container_opacity}")

    def move_to_next_monitor(self) -> None:
        """Cycles the overlay to the next available screen, wrapping back to the first."""
        screens = QApplication.screens()
        if len(screens) <= 1:
            logger.debug("Only one screen detected; monitor cycling skipped.")
            return

        current = self._active_screen()
        try:
            current_index = screens.index(current) if current else 0
        except ValueError:
            current_index = 0

        next_screen = screens[(current_index + 1) % len(screens)]
        target_geometry = next_screen.availableGeometry()
        width = int(target_geometry.width() * WINDOW_WIDTH_RATIO)
        x = target_geometry.x() + int((target_geometry.width() - width) / 2)
        y = target_geometry.y() + target_geometry.height() - WINDOW_HEIGHT_PX - WINDOW_BOTTOM_MARGIN_PX

        self.setGeometry(x, y, width, WINDOW_HEIGHT_PX)
        self._current_screen = next_screen
        self._sync_dpi_scaling()
        logger.info(f"Overlay moved to monitor: {next_screen.name()}")

    def moveEvent(self, event) -> None:
        """Detects cross-monitor drags and refreshes DPI-dependent metrics accordingly.

        Args:
            event: Native PySide6 move event.
        """
        super().moveEvent(event)
        screen = self._active_screen()
        if screen is not None and screen is not self._current_screen:
            self._current_screen = screen
            self._sync_dpi_scaling()
            logger.debug(f"Overlay crossed onto screen: {screen.name()}")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Intercepts hardware mouse clicks to register positional vectors for custom drag actions.

        Args:
            event: Native PySide6 mouse state parameters.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Recalculates display tracking coordinates when dragging the frameless overlay window.

        Args:
            event: Native PySide6 tracking state configuration.
        """
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        """Assembles and displays an interactive context control menu on right-click actions.

        Args:
            event: Intercepted hardware event triggers.
        """
        context_menu = QMenu(self)
        context_menu.setStyleSheet("""
            QMenu {
                background-color: #242424;
                color: #FFFFFF;
                border: 1px solid #444444;
                padding: 5px;
            }
            QMenu::item:selected {
                background-color: #0078D4;
            }
        """)

        inc_font_act = QAction("Increase Font Size (+)", self)
        inc_font_act.triggered.connect(lambda: self.set_font_size(self.base_font_size + FONT_SIZE_STEP_PX))

        dec_font_act = QAction("Decrease Font Size (-)", self)
        dec_font_act.triggered.connect(lambda: self.set_font_size(self.base_font_size - FONT_SIZE_STEP_PX))

        inc_opac_act = QAction("Increase Opacity", self)
        inc_opac_act.triggered.connect(lambda: self.set_overlay_opacity(self.container_opacity + OPACITY_STEP))

        dec_opac_act = QAction("Decrease Opacity", self)
        dec_opac_act.triggered.connect(lambda: self.set_overlay_opacity(self.container_opacity - OPACITY_STEP))

        reset_act = QAction("Reset Window Position", self)
        reset_act.triggered.connect(self._center_on_screen_bottom)

        next_monitor_act = QAction("Move to Next Monitor", self)
        next_monitor_act.triggered.connect(self.move_to_next_monitor)

        exit_act = QAction("Close App", self)
        exit_act.triggered.connect(QApplication.quit)

        context_menu.addAction(inc_font_act)
        context_menu.addAction(dec_font_act)
        context_menu.addSeparator()
        context_menu.addAction(inc_opac_act)
        context_menu.addAction(dec_opac_act)
        context_menu.addSeparator()
        context_menu.addAction(reset_act)
        context_menu.addAction(next_monitor_act)
        context_menu.addSeparator()
        context_menu.addAction(exit_act)

        context_menu.exec(event.globalPos())