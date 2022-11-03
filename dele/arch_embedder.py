from typing import Tuple, List, Dict, Union, Optional

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from . import utils


class ArchEmbedder(nn.Module):
    """
    Base class for architecture encoder.
    The encoder encode architectures to the latent space with shape [batch_size, dimension].
    """
    def __init__(self):
        nn.Module.__init__(self)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location = torch.device("cpu")))


class NasBench201SearchSpace(object):
    def __init__(
        self,
        num_layers: int = 17,
        vertices: int = 4,
        ops_choices: Tuple[str] = (
            "none",
            "skip_connect",
            "nor_conv_1x1",
            "nor_conv_3x3",
            "avg_pool_3x3",
        ),
    ):
        self.ops_choices = ops_choices
        self.ops_choice_to_idx = {
            choice: i for i, choice in enumerate(self.ops_choices)
        }

        self.num_vertices = vertices
        self.num_layers = num_layers
        self.none_op_ind = self.ops_choices.index("none")
        self.num_possible_edges = self.num_vertices * (self.num_vertices - 1) // 2
        self.num_op_choices = len(self.ops_choices)  # 5
        self.num_ops = self.num_vertices * (self.num_vertices - 1) // 2
        self.idx = np.tril_indices(self.num_vertices, k = -1)
        self.genotype_type = str

    # optional API
    def genotype_from_str(self, genotype_str):
        return genotype_str

    def genotype(self, arch):
        # return the corresponding ModelSpec
        # edges, ops = arch
        return self.matrix2str(arch)

    def plot_arch(self, genotypes, filename, label, plot_format = "pdf", **kwargs):
        matrix = self.str2matrix(genotypes)

        from graphviz import Digraph

        graph = Digraph(
            format = plot_format,
            # https://stackoverflow.com/questions/4714262/graphviz-dot-captions
            body=['label="{l}"'.format(l = label), "labelloc=top", "labeljust=left"],
            edge_attr = dict(fontsize = "20", fontname = "times"),
            node_attr = dict(
                style = "filled",
                shape = "rect",
                align = "center",
                fontsize = "20",
                height = "0.5",
                width = "0.5",
                penwidth = "2",
                fontname = "times",
            ),
            engine = "dot",
        )
        graph.body.extend(["rankdir=LR"])
        graph.node(str(0), fillcolor = "darkseagreen2")
        graph.node(str(self.num_vertices - 1), fillcolor = "palegoldenrod")
        [
            graph.node(str(i), fillcolor = "lightblue")
            for i in range(1, self.num_vertices - 1)
        ]

        for to_, from_ in zip(*self.idx):
            op_name = self.ops_choices[int(matrix[to_, from_])]
            if op_name == "none":
                continue
            graph.edge(str(from_), str(to_), label = op_name, fillcolor = "gray")

        graph.render(filename, view = False)
        fnames = []
        fnames.append(("cell", filename + ".{}".format(plot_format)))
        return fnames

    # ---- helpers ----
    def matrix2str(self, arch):
        node_strs = []
        for i_node in range(1, self.num_vertices):
            node_strs.append(
                "|"
                + "|".join(
                    [
                        "{}~{}".format(
                            self.ops_choices[int(arch[i_node, i_input])], i_input
                        )
                        for i_input in range(0, i_node)
                    ]
                )
                + "|"
            )
        return "+".join(node_strs)

    def str2matrix(self, str_):
        arch = np.zeros((self.num_vertices, self.num_vertices))
        split_str = str_.split("+")
        for ind, s in enumerate(split_str):
            geno = [name for name in s.split("|") if name != ""]
            for g in geno:
                name, conn = g.split("~")
                to_ = ind + 1
                from_ = int(conn)
                arch[to_][from_] = self.ops_choices.index(name)
        return arch

    def op_to_idx(self, ops):
        return [self.ops_choice_to_idx[op] for op in ops]

    def random_sample_arch(self):
        arch = np.zeros((self.num_vertices, self.num_vertices))
        arch[np.tril_indices(self.num_vertices, k = -1)] = np.random.randint(
            low = 0, high = self.num_op_choices, size = self.num_ops
        )
        return arch


class NasBench201_LSTMSeqEmbedder(ArchEmbedder):

    def __init__(
        self,
        search_space = NasBench201SearchSpace(),
        num_hid: int = 100,
        emb_hid: int = 100,
        num_layers: int = 1,
        use_mean: bool = False,
        use_hid: bool = False
    ):
        super(NasBench201_LSTMSeqEmbedder, self).__init__()

        self.search_space = search_space
        self.num_hid = num_hid
        self.num_layers = num_layers
        self.emb_hid = emb_hid
        self.use_mean = use_mean
        self.use_hid = use_hid

        self.op_emb = nn.Embedding(self.search_space.num_op_choices, self.emb_hid)

        self.rnn = nn.LSTM(
            input_size = self.emb_hid,
            hidden_size = self.num_hid,
            num_layers = self.num_layers,
            batch_first = True
        )

        self.out_dim = num_hid
        self._tril_indices = np.tril_indices(self.search_space.num_vertices, k = -1)

    def forward(self, archs):
        x = [arch[self._tril_indices] for arch in archs]
        embs = self.op_emb(torch.LongTensor(x).to(self.op_emb.weight.device))
        out, (h_n, _) = self.rnn(embs)

        if self.use_hid:
            y = h_n[0]
        else:
            if self.use_mean:
                y = torch.mean(out, dim=1)
            else:
                # return final output
                y = out[:, -1, :]
        return y
