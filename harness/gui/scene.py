"""QGraphicsView node graph — Simulink-like canvas.

Custom scene/items without external node-editor deps. The model
(`harness.gui.model.ExperimentGraph`) is authoritative; the scene is a
projection that emits `graph_changed` so the MainWindow can sync inspector,
validation, and file-dirty state.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal, QLineF
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from .model import ExperimentGraph, NodeSpec, PORT_TYPES

# -- colors (dark-lab theme) -----------------------------------------------

COLORS = {
    "canvas_bg": QColor("#0f1115"),
    "canvas_grid": QColor("#1e222a"),
    "node_bg": QColor("#1b1e24"),
    "node_border": QColor("#2a2f3a"),
    "node_selected": QColor("#3b82f6"),
    "node_text": QColor("#e2e8f0"),
    "node_subtext": QColor("#94a3b8"),
    "port_material": QColor("#f59e0b"),
    "port_profile": QColor("#06b6d4"),
    "port_problem": QColor("#8b5cf6"),
    "port_objective": QColor("#ec4899"),
    "port_result": QColor("#22c55e"),
    "port_trace": QColor("#eab308"),
    "edge": QColor("#475569"),
    "edge_selected": QColor("#38bdf8"),
    "edge_bad": QColor("#ef4444"),
}

PORT_COLOR = {
    "material": COLORS["port_material"],
    "profile": COLORS["port_profile"],
    "problem": COLORS["port_problem"],
    "objective": COLORS["port_objective"],
    "result": COLORS["port_result"],
    "trace": COLORS["port_trace"],
    "schedule": COLORS["port_profile"],
}

# Node visuals
NODE_W = 160
NODE_H = 70
PORT_R = 6
PORT_SPACING = 14

TYPE_SHORT = {
    "Material": "MAT",
    "Profile": "PROF",
    "Physics": "PHYS",
    "Objective": "OBJ",
    "Optimizer": "OPT",
    "Scope": "SCOPE",
    "Sweep": "SWEEP",
}


class PortItem(QGraphicsEllipseItem):
    def __init__(self, parent: QGraphicsItem, name: str, is_out: bool) -> None:
        r = PORT_R
        super().__init__(-r, -r, 2 * r, 2 * r, parent)
        self.port_name = name
        self.is_out = is_out
        self.setBrush(QBrush(PORT_COLOR.get(name, QColor("#64748b"))))
        self.setPen(QPen(QColor("#0f1115"), 1.2))
        kind = "out" if is_out else "in"
        self.setToolTip(f"{kind} · {name}")
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

    def scenePosCenter(self) -> QPointF:
        return self.scenePos() + QPointF(0, 0)


class NodeItem(QGraphicsRectItem):
    def __init__(self, spec: NodeSpec) -> None:
        super().__init__(0, 0, NODE_W, NODE_H)
        self.spec = spec
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        # label
        self.title = QGraphicsSimpleTextItem(self)
        self.title.setBrush(QBrush(COLORS["node_text"]))
        f = QFont("Inter, sans-serif", 9)
        f.setBold(True)
        self.title.setFont(f)
        self.subtitle = QGraphicsSimpleTextItem(self)
        self.subtitle.setBrush(QBrush(COLORS["node_subtext"]))
        sf = QFont("Inter, sans-serif", 7)
        self.subtitle.setFont(sf)
        self.status_dot = QGraphicsEllipseItem(self)
        self.status_dot.setRect(NODE_W - 14, 6, 8, 8)
        self.status_dot.setPen(QPen(Qt.PenStyle.NoPen))
        self.status_dot.setBrush(QBrush(QColor("#334155")))
        # ports
        self.in_ports: dict[str, PortItem] = {}
        self.out_ports: dict[str, PortItem] = {}
        self._build_ports()
        self._update_labels()
        self.setPos(spec.x, spec.y)
        self.setZValue(1)

    def _build_ports(self) -> None:
        # clear old
        for p in list(self.in_ports.values()) + list(self.out_ports.values()):
            if p.scene():
                p.scene().removeItem(p)
        self.in_ports.clear()
        self.out_ports.clear()
        ports = PORT_TYPES.get(self.spec.type, {})
        ins = ports.get("in", [])
        outs = ports.get("out", [])
        # layout: inputs on left, outputs on right
        for i, name in enumerate(ins):
            port = PortItem(self, name, is_out=False)
            port.setPos(0, 18 + i * PORT_SPACING)
            self.in_ports[name] = port
        for i, name in enumerate(outs):
            port = PortItem(self, name, is_out=True)
            port.setPos(NODE_W, 18 + i * PORT_SPACING)
            self.out_ports[name] = port

    def _update_labels(self) -> None:
        short = TYPE_SHORT.get(self.spec.type, self.spec.type[:4].upper())
        self.title.setText(f"{short}  {self.spec.display_label()[:18]}")
        self.title.setPos(8, 6)
        kind = self.spec.type
        if kind == "Physics":
            sub = self.spec.data.get("kind", "")
        elif kind == "Material":
            sub = self.spec.data.get("ref", "")[:22]
        elif kind == "Profile":
            sub = self.spec.data.get("ref", "")[:22]
        else:
            sub = ""
        self.subtitle.setText(sub)
        self.subtitle.setPos(8, 26)

    def refresh(self) -> None:
        self._build_ports()
        self._update_labels()
        self.update()

    def set_status(self, color: QColor | None) -> None:
        if color is None:
            self.status_dot.setBrush(QBrush(QColor("#334155")))
        else:
            self.status_dot.setBrush(QBrush(color))

    def paint(self, painter: QPainter, option: Any, widget: Any = None) -> None:  # type: ignore[override]
        rect = self.rect()
        # shadow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 60)))
        painter.drawRoundedRect(rect.translated(1.5, 1.5), 8, 8)
        # body
        bg = COLORS["node_bg"]
        border = COLORS["node_selected"] if self.isSelected() else COLORS["node_border"]
        painter.setBrush(QBrush(bg))
        pen = QPen(border, 1.4 if self.isSelected() else 1.0)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 8, 8)
        # top accent bar
        accent = PORT_COLOR.get("material")
        if self.spec.type == "Profile":
            accent = PORT_COLOR.get("profile")
        elif self.spec.type == "Physics":
            accent = PORT_COLOR.get("problem")
        elif self.spec.type == "Objective":
            accent = PORT_COLOR.get("objective")
        elif self.spec.type == "Optimizer":
            accent = PORT_COLOR.get("result")
        elif self.spec.type == "Scope":
            accent = PORT_COLOR.get("trace")
        elif self.spec.type == "Sweep":
            accent = QColor("#a855f7")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(accent))
        painter.drawRoundedRect(QRectF(0, 0, NODE_W, 4), 4, 4)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:  # type: ignore[override]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # sync model pos (caller will persist)
            self.spec.x = float(self.pos().x())
            self.spec.y = float(self.pos().y())
            # notify scene to update edges
            if self.scene():
                # scene is CanvasScene
                try:
                    self.scene().on_node_moved(self)  # type: ignore[attr-defined]
                except Exception:
                    pass
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # type: ignore[override]
        # bubble to scene
        if self.scene():
            try:
                self.scene().node_double_clicked.emit(self.spec.id)  # type: ignore[attr-defined]
            except Exception:
                pass
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: Any) -> None:  # type: ignore[override]
        from PyQt6.QtWidgets import QMenu  # noqa: WPS433
        menu = QMenu()
        a_edit = menu.addAction("Edit in Inspector")
        a_dup = menu.addAction("Duplicate")
        a_del = menu.addAction("Delete")
        chosen = menu.exec(event.screenPos())
        if chosen == a_edit:
            if self.scene():
                try:
                    self.scene().node_double_clicked.emit(self.spec.id)  # type: ignore[attr-defined]
                except Exception:
                    pass
        elif chosen == a_dup:
            if self.scene():
                try:
                    self.scene().duplicate_node(self.spec.id)  # type: ignore[attr-defined]
                    self.scene().graph_changed.emit()  # type: ignore[attr-defined]
                except Exception:
                    pass
        elif chosen == a_del:
            if self.scene():
                try:
                    self.scene().remove_node(self.spec.id)  # type: ignore[attr-defined]
                except Exception:
                    pass
        event.accept()


class EdgeItem(QGraphicsPathItem):
    def __init__(self, from_item: NodeItem, from_port: str, to_item: NodeItem, to_port: str, edge_id: str) -> None:
        super().__init__()
        self.edge_id = edge_id
        self.from_item = from_item
        self.from_port_name = from_port
        self.to_item = to_item
        self.to_port_name = to_port
        self.setZValue(0)
        self.setPen(QPen(COLORS["edge"], 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self._bad = False
        self.update_path()

    def set_bad(self, bad: bool) -> None:
        self._bad = bad
        c = COLORS["edge_bad"] if bad else (COLORS["edge_selected"] if self.isSelected() else COLORS["edge"])
        self.setPen(QPen(c, 2.6 if bad or self.isSelected() else 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

    def update_path(self) -> None:
        src_port = self.from_item.out_ports.get(self.from_port_name)
        dst_port = self.to_item.in_ports.get(self.to_port_name)
        if src_port is None or dst_port is None:
            return
        p0 = src_port.scenePos()
        p1 = dst_port.scenePos()
        # bezier with horizontal tangents
        dx = max(40, abs(p1.x() - p0.x()) * 0.5)
        path = QPainterPath(p0)
        c0 = QPointF(p0.x() + dx, p0.y())
        c1 = QPointF(p1.x() - dx, p1.y())
        path.cubicTo(c0, c1, p1)
        self.setPath(path)

    def paint(self, painter: QPainter, option: Any, widget: Any = None) -> None:  # type: ignore[override]
        super().paint(painter, option, widget)
        # arrow head at dst
        dst_port = self.to_item.in_ports.get(self.to_port_name)
        if dst_port is None:
            return
        p1 = dst_port.scenePos()
        path = self.path()
        if path.elementCount() < 2:
            return
        # tangent near end
        e1 = path.elementAt(path.elementCount() - 1)
        e0 = path.elementAt(path.elementCount() - 2)
        line = QLineF(e0.x, e0.y, e1.x, e1.y)
        ang = line.angle()  # 0 = east
        painter.setPen(Qt.PenStyle.NoPen)
        c = COLORS["edge_bad"] if self._bad else (COLORS["edge_selected"] if self.isSelected() else COLORS["edge"])
        painter.setBrush(QBrush(c))
        # small triangle
        painter.save()
        painter.translate(p1)
        painter.rotate(-ang)
        tri = QPainterPath()
        tri.moveTo(0, 0)
        tri.lineTo(-7, -3.5)
        tri.lineTo(-7, 3.5)
        tri.closeSubpath()
        painter.drawPath(tri)
        painter.restore()


class CanvasScene(QGraphicsScene):
    node_double_clicked = pyqtSignal(str)
    selection_changed = pyqtSignal()
    graph_changed = pyqtSignal()

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(COLORS["canvas_bg"]))
        self._graph = ExperimentGraph()
        self._nodes: dict[str, NodeItem] = {}
        self._edges: dict[str, EdgeItem] = {}
        self._pending_wire: tuple[NodeItem, str] | None = None
        self._temp_edge: QGraphicsPathItem | None = None
        self.selectionChanged.connect(self._forward_selection)

    def _forward_selection(self) -> None:
        try:
            self.selection_changed.emit()
        except RuntimeError:
            pass

    # -- model sync ---------------------------------------------------------

    def set_graph(self, graph: ExperimentGraph) -> None:
        self.clear()
        self._nodes.clear()
        self._edges.clear()
        self._graph = graph
        for spec in graph.nodes:
            item = NodeItem(spec)
            self.addItem(item)
            self._nodes[spec.id] = item
        for e in graph.edges:
            src = self._nodes.get(e.from_node)
            dst = self._nodes.get(e.to_node)
            if src and dst:
                edge = EdgeItem(src, e.from_port, dst, e.to_port, e.id)
                self.addItem(edge)
                self._edges[e.id] = edge
        self.setSceneRect(self.itemsBoundingRect().adjusted(-200, -200, 200, 200))

    def graph(self) -> ExperimentGraph:
        return self._graph

    def add_node(self, spec: NodeSpec) -> NodeItem:
        item = NodeItem(spec)
        self.addItem(item)
        self._nodes[spec.id] = item
        self._graph.nodes.append(spec)
        self.graph_changed.emit()
        return item

    def remove_node(self, nid: str) -> None:
        item = self._nodes.pop(nid, None)
        if item:
            self.removeItem(item)
        # remove incident edges
        for eid, edge in list(self._edges.items()):
            if edge.from_item.spec.id == nid or edge.to_item.spec.id == nid:
                self.removeItem(edge)
                del self._edges[eid]
        self._graph.remove_node(nid)
        self.graph_changed.emit()

    def duplicate_node(self, nid: str) -> NodeItem | None:
        src_spec = self._graph.node_by_id(nid)
        if not src_spec:
            return None
        import copy  # noqa: WPS433
        new_data = copy.deepcopy(src_spec.data)
        # offset so duplicate is visible next to original
        new_spec = self._graph.add_node(src_spec.type, x=src_spec.x + 40, y=src_spec.y + 40, data=new_data, label=src_spec.label)
        item = NodeItem(new_spec)
        self.addItem(item)
        self._nodes[new_spec.id] = item
        self.graph_changed.emit()
        return item

    def add_edge(self, from_node: str, from_port: str, to_node: str, to_port: str) -> EdgeItem | None:
        src = self._nodes.get(from_node)
        dst = self._nodes.get(to_node)
        if not src or not dst:
            return None
        # model first
        espec = self._graph.add_edge(from_node, from_port, to_node, to_port)
        edge = EdgeItem(src, from_port, dst, to_port, espec.id)
        self.addItem(edge)
        self._edges[espec.id] = edge
        self.graph_changed.emit()
        return edge

    def on_node_moved(self, node: NodeItem) -> None:
        for edge in self._edges.values():
            if edge.from_item is node or edge.to_item is node:
                edge.update_path()

    def refresh_node(self, nid: str) -> None:
        item = self._nodes.get(nid)
        if item:
            item.refresh()
            # update incident edges (ports may have changed)
            for edge in list(self._edges.values()):
                if edge.from_item is item or edge.to_item is item:
                    # if port no longer exists, remove edge
                    if edge.from_port_name not in edge.from_item.out_ports or edge.to_port_name not in edge.to_item.in_ports:
                        self.removeItem(edge)
                        del self._edges[edge.edge_id]
                        # also remove from model
                        self._graph.remove_edge(edge.edge_id)
                    else:
                        edge.update_path()
            self.graph_changed.emit()

    def selected_node_ids(self) -> list[str]:
        return [item.spec.id for item in self.selectedItems() if isinstance(item, NodeItem)]

    def selected_node(self) -> NodeItem | None:
        for item in self.selectedItems():
            if isinstance(item, NodeItem):
                return item
        return None

    # -- wire interaction ---------------------------------------------------

    def start_wire(self, node: NodeItem, port: str) -> None:
        self._pending_wire = (node, port)
        if self._temp_edge:
            self.removeItem(self._temp_edge)
            self._temp_edge = None
        self._temp_edge = QGraphicsPathItem()
        self._temp_edge.setPen(QPen(COLORS["edge_selected"], 2.0, Qt.PenStyle.DashLine))
        self._temp_edge.setZValue(10)
        self.addItem(self._temp_edge)

    def update_temp_wire(self, scene_pos: QPointF) -> None:
        if not self._pending_wire or not self._temp_edge:
            return
        src, port = self._pending_wire
        p0 = src.out_ports[port].scenePos() if port in src.out_ports else src.scenePos()
        path = QPainterPath(p0)
        dx = max(40, abs(scene_pos.x() - p0.x()) * 0.5)
        path.cubicTo(QPointF(p0.x() + dx, p0.y()), QPointF(scene_pos.x() - dx, scene_pos.y()), scene_pos)
        self._temp_edge.setPath(path)

    def finish_wire(self, scene_pos: QPointF) -> None:
        if not self._pending_wire:
            return
        src, from_port = self._pending_wire
        # hit test for dst port
        items = self.items(scene_pos)
        dst_item: NodeItem | None = None
        dst_port: str | None = None
        for it in items:
            if isinstance(it, PortItem) and not it.is_out:
                # parent is NodeItem
                parent = it.parentItem()
                if isinstance(parent, NodeItem):
                    dst_item = parent
                    dst_port = it.port_name
                    break
            if isinstance(it, NodeItem):
                # generic hit on node -> pick first in-port
                if it is not src and it.in_ports:
                    dst_item = it
                    dst_port = next(iter(it.in_ports))
                    break
        if dst_item and dst_port:
            self.add_edge(src.spec.id, from_port, dst_item.spec.id, dst_port)
        # cleanup
        if self._temp_edge:
            self.removeItem(self._temp_edge)
            self._temp_edge = None
        self._pending_wire = None

    def cancel_wire(self) -> None:
        if self._temp_edge:
            self.removeItem(self._temp_edge)
            self._temp_edge = None
        self._pending_wire = None

    def mark_bad_edges(self, bad_ids: set[str]) -> None:
        for eid, edge in self._edges.items():
            edge.set_bad(eid in bad_ids)

    # -- painting grid ------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        super().drawBackground(painter, rect)
        grid = 24
        left = int(rect.left()) - int(rect.left()) % grid
        top = int(rect.top()) - int(rect.top()) % grid
        painter.setPen(QPen(COLORS["canvas_grid"], 0.6))
        # dots
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(COLORS["canvas_grid"]))
        r = 1.0
        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                painter.drawEllipse(QRectF(x - r, y - r, 2 * r, 2 * r))
                y += grid
            x += grid


class CanvasView(QGraphicsView):
    request_inspect = pyqtSignal(str)

    def __init__(self, scene: CanvasScene, parent: Any = None) -> None:
        super().__init__(scene, parent)
        self._scene = scene
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.TextAntialiasing
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._panning = False
        self._pan_start: QPointF | None = None
        self._space_down = False

    def wheelEvent(self, event: Any) -> None:  # type: ignore[override]
        # zoom with wheel (Ctrl or plain)
        delta = event.angleDelta().y()
        factor = 1.0015 ** delta
        factor = max(0.5, min(2.0, factor))
        self.scale(factor, factor)

    def keyPressEvent(self, event: Any) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for nid in self._scene.selected_node_ids():
                self._scene.remove_node(nid)
            # also remove selected edges
            for eid, edge in list(self._scene._edges.items()):
                if edge.isSelected():
                    self._scene.removeItem(edge)
                    del self._scene._edges[eid]
                    self._scene.graph().remove_edge(eid)
            self._scene.graph_changed.emit()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: Any) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: Any) -> None:  # type: ignore[override]
        pos = self.mapToScene(event.pos())
        items = self._scene.items(pos)
        # port click starts wire
        for it in items:
            if isinstance(it, PortItem) and it.is_out:
                parent = it.parentItem()
                if isinstance(parent, NodeItem):
                    self._scene.start_wire(parent, it.port_name)
                    event.accept()
                    return
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and self._space_down):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # type: ignore[override]
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._scene._pending_wire:
            self._scene.update_temp_wire(self.mapToScene(event.pos()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # type: ignore[override]
        if self._panning:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        if self._scene._pending_wire and event.button() == Qt.MouseButton.LeftButton:
            self._scene.finish_wire(self.mapToScene(event.pos()))
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton and self._scene._pending_wire:
            self._scene.cancel_wire()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # type: ignore[override]
        pos = self.mapToScene(event.pos())
        for it in self._scene.items(pos):
            if isinstance(it, NodeItem):
                self.request_inspect.emit(it.spec.id)
                break
        super().mouseDoubleClickEvent(event)
