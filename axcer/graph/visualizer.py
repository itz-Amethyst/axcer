import matplotlib.pyplot as plt
import matplotlib as mpl
import rustworkx as rx
from rustworkx.visualization import mpl_draw
from rustworkx.visualization.matplotlib import draw_edge_labels
from axcer.utils.custom_logger import logger


class GraphVisualizeMixIn:
    def show_graph(self):
        try:
            # Publication‑grade defaults
            mpl.rcParams.update(
                {
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                    "font.size": 22,  # base font size
                    "axes.titlesize": 22,
                    "axes.labelsize": 22,
                    "xtick.labelsize": 20,
                    "ytick.labelsize": 20,
                    "figure.titlesize": 22,
                }
            )

            fig, ax = plt.subplots(figsize=(12, 9), dpi=300)

            pos = rx.spring_layout(self.graph)

            node_colors = [self.node_to_color.get(i, "gray") for i in range(self.graph.num_nodes())]

            # 1) Draw graph with node labels but no edge labels
            mpl_draw(
                self.graph,
                pos=pos,
                ax=ax,
                with_labels=True,
                edge_labels=None,  # no edge labels here
                node_size=200,
                font_size=6,  # node text size
                node_color=node_colors,
                labels=lambda node: node,
                width=0.2,
            )

            # 2) Build a dict of edge labels for draw_edge_labels()
            edge_label_dict = {(u, v): str(weight) for u, v, weight in self.graph.weighted_edge_list()}

            # 3) Draw edge labels with smaller font
            draw_edge_labels(
                self.graph,
                pos=pos,
                edge_labels=edge_label_dict,  # correct dict
                font_size=3,  # smaller edge font
                ax=ax,
            )

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.error("Visualization requires matplotlib and rustworkx.visualization")
