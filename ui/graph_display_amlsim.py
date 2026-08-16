import networkx as nx
from pyvis.network import Network
import matplotlib.colors as mcolors
import streamlit as st
import codecs
import os

GRAPH_CACHE_DIR = 'outputs/.graph_cache'

# repulsion = spread-out, non-clumped layout (this is what you want for AMLSim + IBM)
# barnesHut = original dense/clustered look (kept available, NOT the default anymore)
PHYSICS_PROFILES = {
    "barnesHut": """
        {
          "physics": {
            "solver": "barnesHut",
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
    """,
    "repulsion": """
        {
          "physics": {
            "solver": "repulsion",
            "repulsion": {
              "nodeDistance": 180,
              "centralGravity": 0.05,
              "springLength": 200,
              "springConstant": 0.02,
              "damping": 0.09
            },
            "minVelocity": 0.75,
            "stabilization": { "enabled": true, "iterations": 300, "fit": true }
          },
          "interaction": { "hover": true, "zoomView": true, "dragView": true, "dragNodes": true }
        }
    """,
}


def display_network_graph(df_filtered, view_key="global", physics_profile="repulsion"):
    """
    physics_profile: "repulsion" (default) = spread-out, readable layout, no dense clumping.
                      "barnesHut" = original tighter clustered look, available if ever wanted back.
    Physics runs briefly to produce a layout, then auto-freezes once stabilized.
    """
    if df_filtered.empty:
        st.info("No transactions to display for this view.")
        return

    G = nx.from_pandas_edgelist(
        df_filtered,
        source='SENDER_ACCOUNT_ID',
        target='RECEIVER_ACCOUNT_ID',
        edge_attr=['TX_AMOUNT', 'SUSPICION_SCORE', 'TX_TYPE', 'IS_FRAUD'],
        create_using=nx.DiGraph()
    )

    net = Network(height='600px', width='100%', directed=True, notebook=False, cdn_resources='remote')
    options_json = PHYSICS_PROFILES.get(physics_profile, PHYSICS_PROFILES["repulsion"])
    net.set_options("var options = " + options_json)

    node_scores = {}
    for _, row in df_filtered.iterrows():
        node_scores[row['SENDER_ACCOUNT_ID']] = max(node_scores.get(row['SENDER_ACCOUNT_ID'], 0), row['SUSPICION_SCORE'])
        node_scores[row['RECEIVER_ACCOUNT_ID']] = max(node_scores.get(row['RECEIVER_ACCOUNT_ID'], 0), row['SUSPICION_SCORE'])

    fraud_accounts = set(df_filtered[df_filtered['IS_FRAUD'] == 1]['SENDER_ACCOUNT_ID']).union(
        set(df_filtered[df_filtered['IS_FRAUD'] == 1]['RECEIVER_ACCOUNT_ID'])
    )

    cmap = mcolors.LinearSegmentedColormap.from_list("suspicion_cmap", ["#2563eb", "#dc2626"])

    for node in G.nodes():
        score = node_scores.get(node, 0)
        color_val = min(score * 1.5, 1.0)
        hex_color = mcolors.to_hex(cmap(color_val))
        border_width = 3 if node in fraud_accounts else 1
        title_html = f"Account ID: {node}<br>Max Edge Score: {score:.4f}<br>Known Fraud: {'Yes' if node in fraud_accounts else 'No'}"
        net.add_node(
            n_id=str(node),
            label=str(node),
            title=title_html,
            color={'border': '#000000' if node not in fraud_accounts else '#dc2626', 'background': hex_color},
            borderWidth=border_width
        )

    for source, target, edge_data in G.edges(data=True):
        score = edge_data['SUSPICION_SCORE']
        line_thickness = max(0.5, score * 10)  # reverted to original
        line_color = mcolors.to_hex(cmap(min(score * 1.5, 1.0)))
        title_html = (
            f"Score: {score:.4f}<br>"
            f"Amount: {edge_data['TX_AMOUNT']:,.2f}<br>"
            f"Type: {edge_data['TX_TYPE']}<br>"
            f"Known Fraud: {'YES' if edge_data['IS_FRAUD'] == 1 else 'No'}"
        )
        net.add_edge(source=str(source), to=str(target), title=title_html, value=line_thickness, color=line_color)

    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    html_file_path = os.path.join(GRAPH_CACHE_DIR, f"{view_key}.html")

    html_str = net.generate_html()

    freeze_script = """
    <script type="text/javascript">
      network.once("stabilizationIterationsDone", function () {
        network.setOptions({ physics: false });
      });
    </script>
    """
    html_str = html_str.replace("</body>", freeze_script + "</body>")

    with codecs.open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=650, scrolling=True)