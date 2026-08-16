"""
cleanup_outputs.py — deletes stray generated HTML/graph files that piled up
from the old per-account filename bug. Safe to run once, then never needed
again since the new code reuses fixed filenames.
"""
import os
import glob

PATTERNS_TO_DELETE = [
    'outputs/egonet_graph_*.html',
    'outputs/network_graph.html',
    'outputs/network_graph.png',
    'outputs/gephi_export.graphml',
    'outputs/global_network_graph*.html',
    'outputs/egonet_graph.html',
    'outputs/egonet_graph_filtered.html',
    'ui/static/*.html',
]

if __name__ == '__main__':
    deleted = 0
    for pattern in PATTERNS_TO_DELETE:
        for filepath in glob.glob(pattern):
            os.remove(filepath)
            print(f"Deleted: {filepath}")
            deleted += 1
    print(f"\nCleaned up {deleted} stray files.")
    print("Going forward, graphs are written to outputs/.graph_cache/ and reused, not accumulated.")