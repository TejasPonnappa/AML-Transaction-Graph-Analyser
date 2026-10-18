"""Shared pyvis rendering used by every network view, styled to match the theme.

Nodes and edges at or above the alert threshold are red; everything else is a
neutral grey-blue. Nodes touching a labelled positive get a light border.
"""
import html

import streamlit as st
from pyvis.network import Network

from ui import theme

_NODE_BASE = "#4a5565"
_EDGE_BASE = "#3a4452"
_LABELLED_BORDER = "#e6e9ee"

_OPTIONS = """
var options = {
  "physics": {
    "solver": "repulsion",
    "repulsion": { "nodeDistance": 160, "centralGravity": 0.05, "springLength": 180,
                   "springConstant": 0.02, "damping": 0.09 },
    "minVelocity": 0.75,
    "stabilization": { "enabled": true, "iterations": 300, "fit": true }
  },
  "nodes": { "font": { "size": 11 } },
  "interaction": { "hover": true, "zoomView": true, "dragView": true, "dragNodes": true }
}
"""

_CSS = f"""
<style>
  body {{ margin: 0; background: {theme.BG}; }}
  .card {{ background: {theme.BG}; border: none; }}
  #mynetwork {{ border: 1px solid {theme.BORDER} !important; }}
</style>
"""

_FREEZE = """
<script type="text/javascript">
  network.once("stabilizationIterationsDone", function () { network.setOptions({ physics: false }); });
</script>
"""


def render_graph(nodes, edges, threshold, height=560):
    """
    nodes: {node_id: {"score": float, "tooltip": str, "labelled": bool}}
    edges: iterable of (source_id, target_id, score, tooltip)
    """
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor=theme.BG,
                  font_color=theme.TEXT, cdn_resources="remote")
    net.set_options(_OPTIONS)

    for node_id, info in nodes.items():
        alert = info["score"] >= threshold
        net.add_node(
            str(node_id), label=str(node_id), title=html.escape(info["tooltip"]),
            color={"background": theme.HIGH if alert else _NODE_BASE,
                   "border": _LABELLED_BORDER if info.get("labelled") else (theme.HIGH if alert else _NODE_BASE)},
            borderWidth=3 if info.get("labelled") else 1,
            size=14 + 10 * info["score"],
        )
    for source, target, score, tooltip in edges:
        net.add_edge(str(source), str(target), title=html.escape(tooltip),
                     color=theme.HIGH if score >= threshold else _EDGE_BASE,
                     width=1 + 3 * float(score))

    page = net.generate_html().replace("</head>", _CSS + "</head>").replace("</body>", _FREEZE + "</body>")
    st.components.v1.html(page, height=height + 20, scrolling=False)
    st.caption("Red: model score at or above the alert threshold. Light outline: labelled positive. "
               "Larger nodes have higher scores. Drag to rearrange; scroll to zoom.")
