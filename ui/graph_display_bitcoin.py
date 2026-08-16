import networkx as nx
from pyvis.network import Network
import matplotlib.colors as mcolors
import streamlit as st
import codecs
import os

GRAPH_CACHE_DIR = 'outputs/.graph_cache'


def display_network_graph(df_account_edges, df_account_nodes, view_key="bitcoin_global"):
    """
    UPDATED: physics runs briefly to settle into a natural layout, then
    auto-freezes once stabilized. Reusable filename per view_key.
    """
    if df_account_edges.empty:
        st.info("No connections to display for this address.")
        return

    G = nx.from_pandas_edgelist(
        df_account_edges, source='source', target='target', create_using=nx.DiGraph()
    )

    net = Network(height='500px', width='100%', directed=True, cdn_resources='remote')
    net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "centralGravity": 0.3,
              "springLength": 100,
              "springConstant": 0.04,
              "damping": 0.9
            },
            "minVelocity": 0.75,
            "stabilization": { "enabled": true, "iterations": 200, "fit": true }
          },
          "interaction": { "hover": true, "zoomView": true, "dragView": true, "dragNodes": true }
        }
    """)

    cmap = mcolors.LinearSegmentedColormap.from_list("suspicion_cmap", ["#112A66", "#dc2626"])

    for node_id in G.nodes():
        node_data_list = df_account_nodes[df_account_nodes['txId'] == node_id]
        if node_data_list.empty:
            continue
        node_data = node_data_list.iloc[0]
        score = node_data['SUSPICION_SCORE']
        is_illicit = node_data['class'] == 'Illicit'

        color_val = min(score * 1.5, 1.0)
        hex_color = mcolors.to_hex(cmap(color_val))

        title_html = (
            f"Address ID: {node_id}<br>"
            f"Score: {score:.6f}<br>"
            f"Truth: {node_data['class']}"
        )

        net.add_node(
            n_id=node_id,
            label=node_id,
            title=title_html,
            size=15 + score * 20,
            color={'border': '#dc2626' if is_illicit else '#000000', 'background': hex_color},
            borderWidth=3 if is_illicit else 1
        )

    for source, target in G.edges():
        net.add_edge(source, target, color='#999999', width=1.5)

    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    html_file_path = os.path.join(GRAPH_CACHE_DIR, f"{view_key}.html")
    html_content = net.generate_html()

    freeze_script = """
    <script type="text/javascript">
      network.once("stabilizationIterationsDone", function () {
        network.setOptions({ physics: false });
      });
    </script>
    """
    html_content = html_content.replace("</body>", freeze_script + "</body>")

    with codecs.open(html_file_path, "w", encoding="utf-8") as out:
        out.write(html_content)

    with open(html_file_path, 'r', encoding='utf-8') as f:
        final_html_content = f.read()

    st.components.v1.html(final_html_content, height=550, scrolling=True)