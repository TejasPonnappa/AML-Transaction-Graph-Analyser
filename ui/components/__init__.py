"""Streamlit wrappers around the single JavaScript frontend (frontend/index.html): the network and the decision tree.

Both return what the page sends back (a clicked node), or None. `sig` is a short string that changes whenever the data
changes; the frontend keeps its layout and animation untouched while it stays the same.
"""
import os

import streamlit.components.v1 as components

_viz = components.declare_component("aml_viz", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"))


def network(nodes, edges, *, key, sig, layout="cose", height=560):
    """nodes: dicts with id, label, score, alert, size, center, labelled, badges, tip.
    edges: dicts with id, source, target, score, alert, kind ('normal' | 'cycle' | 'pass'), w, tip."""
    return _viz(mode="graph", nodes=nodes, edges=edges, layout=layout, height=height, sig=sig, key=key, default=None)


def decision_tree(tree, path, *, key, sig, height=680):
    """tree: the exported surrogate tree; path: [{id, text}, ...] for the transaction to animate (may be empty)."""
    return _viz(mode="tree", tree=tree, path=path, height=height, sig=sig, key=key, default=None)
