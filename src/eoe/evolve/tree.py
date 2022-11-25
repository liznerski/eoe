import json
import os
import os.path as pt
import uuid
from collections import deque
from typing import List, Tuple, Union, Callable
from uuid import uuid4

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import tqdm
from deap.base import Fitness
from networkx.drawing.nx_agraph import graphviz_layout
from torchvision.transforms import Compose

from eoe.datasets import MSM, TRAIN_OE_ID
from eoe.utils.logger import Logger
from eoe.utils.transformations import GPU_TRANSFORMS


class Node(object):
    def __init__(self, content: object,):
        """
        A node in a tree. The node has links to its parents and children.
        It also has a content, which can be anything serializable.
        """
        self.children = []
        self.parents = []
        self.content = content
        self.id = uuid4()

    def bfs(self) -> List['Node']:
        """ returns a list of all descendants via breadth-first search """
        nodes, queue = [], deque()
        queue.appendleft(self)
        while len(queue) > 0:
            node = queue.pop()
            nodes.append(node)
            queue.extendleft([c for c in node.children if c not in nodes and c not in queue])
        return nodes

    def dfs(self) -> List['Node']:
        """ returns a list of all descendants via depth-first search """
        visited = set()
        def _dfs(node: Node):
            if node in visited:
                return []
            else:
                visited.add(node)
            return [node] + [cc for c in node.children for cc in _dfs(c)]
        return _dfs(self)

    def add_children(self, *nodes: 'Node', add_parent=True):
        """ adds a list of child nodes """
        self.children.extend(nodes)
        if add_parent:
            for n in nodes:
                n.add_parents(self)

    def add_parents(self, *nodes: 'Node'):
        """ adds a list of parent nodes """
        self.parents.extend(nodes)

    def __repr__(self):
        return repr(self.content)

    def __getstate__(self) -> dict:
        """ serialize the node """
        return {
            'content': self.content, 'id': self.id.hex,
            'children': [c.id.hex for c in self.children], 'parents': [p.id.hex for p in self.parents],
            'class': "Node"
        }

    @staticmethod
    def _get_content_from_state(state: dict) -> object:
        return state['content']

    def __setstate__(self, state: dict):
        raise NotImplementedError()


class Individual(object):
    def __init__(self, values: List[int], file: str = None, fitness: float = None):
        """
        The Individual class used by DEAP to create DEAP individuals.
        For our purposes, an individual corresponds to an OE subset.
        It consists of a list of indices pointing to images in the complete OE dataset.
        For instance, an individual with values==[2, 8] denotes an OE subset with OE images being oe_dataset[2] and
        oe_dataset[8]. Further, contains the fitness (i.e., mean test AUC; might be None if not yet evaluated) and a path to
        the corresponding image file on the disk (if this image files has been created already by the logger).

        @param values: Indices to images in the complete OE dataset.
        @param file: Path to a logged image file corresponding to the individual.
        @param fitness: Fitness of the individual.
        """
        self.values = values
        self.file = file
        self.fitness = fitness

    def __repr__(self):
        return repr(self.values)

    def __eq__(self, other: Union['Individual', object]):
        if isinstance(other, Individual):
            return self.values == other.values
        else:  # assuming individuals as in deap here
            return self.values == list(other)


class EvolNode(Node):
    def __init__(self, content: Individual, ):
        """ A node in a genealogical tree. The node's content is an Individual as defined above (see :class:`Individual`). """
        super().__init__(content)

    def __getstate__(self) -> dict:
        return {
            'content': self.content.__dict__, 'id': self.id.hex,
            'children': [c.id.hex for c in self.children], 'parents': [p.id.hex for p in self.parents],
            'class': "EvolNode"
        }

    @staticmethod
    def _get_content_from_state(state: dict) -> object:
        content = Individual([])
        for k, v in state['content'].items():
            setattr(content, k, v)
        return content

    def __repr__(self):
        return repr(self.content)


class Tree(object):
    def __init__(self, *roots: EvolNode):
        """
        A tree consisting of nodes that are linked to each other.
        Each node contains an individual of the evolutionary algorithm.
        An individual corresponds to an OE subset.
        """
        self.meta_root = Node("METAROOT")
        self.meta_root.add_children(*roots)

    def bfs(self):
        """ returns a list of all nodes via breadth-first search """
        return self.meta_root.bfs()

    def dfs(self):
        """ returns a list of all nodes via depth-first search """
        return self.meta_root.dfs()

    def __getstate__(self) -> List[dict]:
        """ serializes the tree """
        return [n.__getstate__() for n in self.bfs()]

    def __setstate__(self, state: List[dict]):
        """ loads a serialized tree and deserializes it """
        nodes, idmap = [], {}
        for node_state in state:
            id, NodeClass = uuid.UUID(node_state['id']), {'Node': Node, 'EvolNode': EvolNode}[node_state['class']]
            nodes.append(NodeClass(NodeClass._get_content_from_state(node_state)))
            nodes[-1].id = id
            idmap[id] = nodes[-1]
        for node_state in state:
            id, children_ids, parents_ids = uuid.UUID(node_state['id']), node_state['children'], node_state['parents']
            node = idmap[id]
            node.children = [idmap[uuid.UUID(cid)] for cid in children_ids]
            node.parents = [idmap[uuid.UUID(pid)] for pid in parents_ids]
        self.meta_root = nodes[0]

    def save(self, file: str):
        """ serializes the tree and stores it to the given filepath """
        file = os.path.abspath(file)
        if not file.endswith('.json'):
            file = f"{file}.json"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w") as writer:
            writer.write(json.dumps(self.__getstate__()))

    def load(self, file: str):
        """ loads a serialized tree from the given filepath """
        file = os.path.abspath(file)
        with open(file, "r") as reader:
            state = json.load(reader)
        self.__setstate__(state)
        return self

    def vis(self, outfile: str, image_dir: str = None, figsize=(324, 224), scale=2, label_offset=0):
        """
        Visualizes the tree. Nodes are visualized using the individual's logged images.
        The filepath to these images is part of the node's content. The fitness is also visualized.
        Requires GraphViz. Might take a long time.
        @param outfile: Where to save the visualization in the form of a pdf.
        @param image_dir: Optional. If None, uses the full paths stored in the nodes to retrieve the corresponding OE image files.
            If not None, redefines the directory where all the images of the individuals were logged.
            That is, uses the file name from the node's file paths but looks at a different directory.
        @param figsize: The final size of the visualization (in height, width).
        @param scale: some scaling factor.
        @param label_offset: an offset (in height) for the label of the node.
        """
        g = nx.Graph()
        queue, edges, visited = deque(), [], []
        queue.appendleft(self.meta_root)
        while len(queue) > 0:
            node = queue.pop()
            if isinstance(node, EvolNode):
                file = node.content.file
                if file is not None and image_dir is not None:
                    file = pt.join(image_dir, pt.basename(file))
                fitness = node.content.fitness.values[0] if isinstance(node.content.fitness, Fitness) else node.content.fitness
                fitness = np.nan if fitness is None else fitness
                if file is not None:
                    g.add_node(node.id, image=file, text=f"{fitness*100:06.3f}")
                else:
                    g.add_node(node.id, text=f"{node.content.values}\n{fitness*100:06.3f}")
            elif hasattr(node.content, 'file'):
                g.add_node(node.id, text=node.content.values)
            else:
                g.add_node(node.id, text=node.content)
            edges.extend([(node.id, c.id) for c in node.children])
            visited.append(node)
            queue.extendleft([c for c in node.children if c not in visited and c not in queue])
        for n1, n2 in edges:
            g.add_edge(n1, n2)

        # nx.nx_agraph.write_dot(g, 'test.dot')
        pos = graphviz_layout(g, prog='dot')
        node_size = max(5000-4*len(g.nodes), 10)
        piesize = scale / len(g.nodes)  # this is the image size
        p2 = piesize / 2.0
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        nx.draw(
            g, pos, with_labels=False, arrows=False, node_size=node_size, font_size=2,
        )
        offset = trans((0, p2 * fig.get_figheight()))[1] - trans((0, 0))[1] + label_offset
        nx.draw_networkx_labels(
            g, {
                k: (vx, vy - (offset if g.nodes[k].get('image', None) is not None else 0))
                for k, (vx, vy) in pos.items()
            },
            labels={node: g.nodes[node].get('text', '') for node in g.nodes}, font_size=3, font_color='red'
        )

        for n in tqdm.tqdm(g, desc='Drawing graph...'):
            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            a = plt.axes([xa - p2, ya - p2, piesize, piesize])
            a.set_aspect('equal')
            if 'image' in g.nodes[n] and g.nodes[n]['image'] is not None:
                a.imshow(cv2.cvtColor(cv2.imread(g.nodes[n]['image']), cv2.COLOR_BGR2RGB))
            a.axis('off')
        ax.axis('off')
        plt.savefig(f'{outfile}.pdf')
        plt.close()

    def scores_best(self, k=20, reverse=False, return_nodes=False) -> Union[List[int], Tuple[List[int], List[EvolNode]]]:
        """
        Returns the fitness of the k best scoring nodes.
        @param k: see above.
        @param reverse: whether to reverse this; i.e., return the fitness of the k worst scoring nodes.
        @param return_nodes: whether to return a tuple instead with (fitness_values, corresponding_nodes).
        @return: either the best fitness values or a tuple with (fitness, nodes). See `return_nodes` above.
        """
        nodes = [node for node in self.bfs() if isinstance(node, EvolNode) and node.content.fitness is not None]
        # remove duplicates
        nodes = sorted(nodes, key=lambda x: x.content.values)
        nodes = [nodes[i] for i in range(len(nodes)) if i == 0 or nodes[i].content.values != nodes[i-1].content.values]
        # sort for fitness
        nodes = sorted(nodes, key=lambda x: x.content.fitness)
        nodes = nodes[-k:] if not reverse else nodes[:k]
        fitnesses = [node.content.fitness if node.content.fitness is not None else np.nan for node in nodes]
        if return_nodes:
            return fitnesses, nodes
        else:
            return fitnesses

    def imsave_best(self, logger: Logger, name: str, k=20, reverse=False, print_fitness=False, image_dir: str = None,
                    img_transform: Callable[[torch.Tensor], torch.Tensor] = None):
        """
        Creates and saves an overview figure that contains the best k individuals (~ OE subsets).
        Each OE subset is a row with each cell showing the fitness (mean test AUC) and the concatenated OE images.
        @param logger: The logger that is used to save the figure.
        @param name: The name under which the figure is saved.
        @param k: Defines how many individuals are visualized. See above.
        @param reverse: Whether to reverse this; i.e., visualize the k worst-performing OE subsets instead.
        @param print_fitness: Whether to visualize the fitness or only the concatenated OE images.
        @param image_dir: Optional. If None, uses the full paths stored in the nodes to retrieve the corresponding OE image files.
            If not None, redefines the directory where all the images of the individuals were logged.
            That is, uses the file name from the node's file paths but looks at a different directory.
        @param img_transform: some image transformation that are to be applied to the images for visualization.
        """
        nodes = [node for node in self.bfs() if isinstance(node, EvolNode) and node.content.fitness is not None]
        # remove duplicates
        nodes = sorted(nodes, key=lambda x: x.content.values)
        nodes = [nodes[i] for i in range(len(nodes)) if i == 0 or nodes[i].content.values != nodes[i-1].content.values]
        # sort for fitness
        nodes = sorted(nodes, key=lambda x: x.content.fitness)
        nodes = nodes[-k:] if not reverse else nodes[:k]
        files = [node.content.file if image_dir is None else pt.join(image_dir, pt.basename(node.content.file)) for node in nodes]
        images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]
        fitnesses = [node.content.fitness if node.content.fitness is not None else np.nan for node in nodes]
        timg = torch.stack([torch.from_numpy(img) for img in images]).permute(0, 3, 1, 2).float().div(255)
        if img_transform is not None:
            if 224 > timg.size(-1) > 32:
                os = timg.size(-1)
                timg = torch.nn.functional.interpolate(timg, 224, mode='bilinear')
                timg = img_transform(timg.cuda()).cpu()
                timg = torch.nn.functional.interpolate(timg, os, mode='bilinear')
            else:
                timg = img_transform(timg.cuda()).cpu()
        if print_fitness:
            logger.logimg(name, timg, nrow=1, rowheaders=[f'{f * 100:06.3f}' for f in fitnesses], maxres=1024)
        else:
            logger.logimg(name, timg, nrow=k, maxres=1024,)

    def imsave_collection_best(self, logger: Logger, msm: List[MSM] = None, image_dir: str = None, k: int = 20):
        """
        Creates several overview figures as defined by :method:`imsave_best` above.
        - A figure with the best performing individuals including a visualization of the fitness at final/best/
        - A figure with the best performing individuals at final/best_raw/
        - A figure with the worst performing individuals including a visualization of the fitness at final/worst/
        - A figure with the worst performing individuals at final/worst_raw/
        If a list of MSMs is given (see :class:`eoe.datasets.MSM`), also creates the same set of four figures
        with the images being transformed by the MSMs.

        @param logger: A logger instance for saving the figures.
        @param msm: Optional. A list of MSMs.
        @param image_dir: Optional. See :method:`imsave_best` above.
        @param k: See :method:`imsave_best` above.
        """
        self.imsave_best(logger, os.path.join('final', 'best_raw'), image_dir=image_dir, k=k)
        self.imsave_best(logger, os.path.join('final', 'best'), print_fitness=True, image_dir=image_dir, k=k)
        self.imsave_best(logger, os.path.join('final', 'worst_raw'), reverse=True, image_dir=image_dir, k=k)
        self.imsave_best(logger, os.path.join('final', 'worst'), reverse=True, print_fitness=True, image_dir=image_dir, k=k)
        if msm is not None:
            oemsm = Compose([
                GPU_TRANSFORMS[type(m.get_transform())](m.get_transform()) for m in msm if m.ds_part == TRAIN_OE_ID
            ])
            self.imsave_best(
                logger, os.path.join('final-transformed', 'best_raw'), img_transform=oemsm, image_dir=image_dir, k=k
            )
            self.imsave_best(
                logger, os.path.join('final-transformed', 'best'), print_fitness=True, img_transform=oemsm, image_dir=image_dir,
                k=k
            )
            self.imsave_best(
                logger, os.path.join('final-transformed', 'worst_raw'), reverse=True, img_transform=oemsm, image_dir=image_dir,
                k=k
            )
            self.imsave_best(
                logger, os.path.join('final-transformed', 'worst'), reverse=True, print_fitness=True, img_transform=oemsm,
                image_dir=image_dir, k=k
            )

    def __repr__(self):
        return repr(self.bfs())

    def get(self, node: Union[uuid.UUID, object]):
        """ Get the node corresponding to the given UUID or object """
        nodes = self.bfs()
        if isinstance(node, uuid.UUID):
            return nodes[[n.id for n in nodes].index(node)]
        else:
            return nodes[[n.content for n in nodes].index(node)]

