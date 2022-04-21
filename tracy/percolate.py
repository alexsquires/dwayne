"""
Code for studying percolation related properties

The percolating network is represented as a tree-like structure. The network is obtained from 
growing and merging the trees. 

The algorithm works roughly as the following:

- First, we instantiate all SiteNode using the atomic positions, as store them as a list
- Iterate though each SiteNode and add the bonds according to the local geometry
- Because adding bonds has the effect of merging existing trees, once all of the bonds are added, the fully network should be grown
- Search for the unique '_root_node' from the list of nodes


TODO:
- Separate the definitions for `Bond` and `Link`
    - Link is for constructing the tree structure for holding the network
    - Bond is the actual entity linking between the two Nodes
    - Two bonded nodes will have share a bond, but there may not be a link between them.

"""

from re import L
from uuid import uuid4, UUID
import numpy as np


class SiteNode:
    """
    Class for a 'Node' in an percolation network. It is essentially an atomic site with
    likes to the next site. The data is stored in a tree - each node has one incoming nodes and
    multiple outgoing links (leaf). In addition, the root node of the tree is stored.
    """

    def __init__(
        self,
        pos: np.ndarray,
        incoming=None,
        outgoing=None,
        root_node=None,
        metadata=None,
        uuid=None,
    ):
        """
        Instantiate an SiteNode Object
        """
        # The incoming/outgoing attributes are a dictionary maps the inputs/output nodes to the links
        self.incoming = incoming
        self.outgoing = outgoing if outgoing else {}
        self._root_node = root_node
        self.pos = np.asarray(pos)
        self.metadata = metadata if metadata else {}
        self._uuid = uuid4() if uuid is None else uuid
        self._children_as_root = set()
        self._network_data = None

    def add_child(self, child):
        if not self.is_root:
            return ValueError("Calling the method for a non-root node")
        self._children_as_root.add(child)
        # If the child is an root node, clean its children list
        if child.is_root:
            child._children_as_root = set()
        child._root_node = self

    @property
    def network_data(self):
        if self.root_node._network_data is None:
            self.root_node._network_data = {}
        return self.root_node._network_data

    @network_data.setter
    def network_data(self, value):
        self.root_node._network_data = value

    @property
    def atom(self):
        """The ase.Atom associated with the SiteNode"""
        return self.metadata.get("atom")

    @property
    def atoms(self):
        """The ase.Atoms associated with the SiteNode"""
        if self.atom:
            return self.atom.atoms
        return None

    @property
    def root_node(self):
        """The root node of the tree-like network"""
        if self._root_node is None:
            assert (
                self.incoming is None
            ), "Inconsistent data: no 'root_node' is recorded but has incoming link"
            return self
        else:
            return self._root_node

    @root_node.setter
    def root_node(self, value):
        """Set the root node"""
        if not value.is_root:
            raise ValueError("The target node is not a root node itself")
        elif value is self:
            print("Warning - setting root_node to myself - this should not happen!")
            self._root_node = None
        else:
            self._root_node = value

    @property
    def is_root(self) -> bool:
        """Is this node the root node of the tree?"""
        return self._root_node is None

    @property
    def tot_num_nodes(self) -> int:
        """Tot number of nodes of the tree that this node belongs to"""
        return len(self.root_node._children_as_root) + 1

    @property
    def non_root_nodes_in_the_tree(self) -> list:
        return self.root_node._children_as_root

    @property
    def all_bonds_in_the_tree(self) -> list:
        return [node.incoming for node in self.non_root_nodes_in_the_tree]

    def __hash__(self) -> int:
        """The has of the node is its UUID"""
        return self._uuid.int

    def __repr__(self) -> str:
        if not self.is_root:
            return f"SiteNode(pos={self.pos}, is_root=False), and {len(self.outgoing)} out links"
        return f"SiteNode(pos={self.pos}, is_root=True, tot_num_nodes={self.tot_num_nodes}), and {len(self.outgoing)} out links"

    def __str__(self) -> str:
        return f"SiteNode(pos={self.pos}), uuid={self.id})"

    @property
    def positions(self) -> np.ndarray:
        """The position of the site"""
        return self.pos

    @property
    def id(self) -> str:
        """Return the UUID representation of the node"""
        return self._uuid.hex

    @property
    def uuid(self) -> UUID:
        """Return the UUID representation of the node"""
        return self._uuid

    def process_bond_to(self, other_site):
        """
        Process the existence of a bond to the other site.

        If the other site is a root node, add the network as the whole subtree,
        otherwise, add the root node of that network as a subtree to the root node directly.
        NOTE: this is not quite the `path compression`, as the depth of  the tree will grow with time!
        Otherwise, all of the nodes of the subtress should be added as the direct child of the root node,
        which will return computation of the link lengths....
        """
        # Case if the other_site is a root node of another network

        # Not doing anything if the nodes are from the same network
        if self == other_site:
            raise RuntimeError("Cannot bond to myself!")
        if other_site.root_node == self.root_node:
            print("Two sites are alrady in the same network")
            return

        self.root_node.network_data.update(other_site.root_node.network_data)

        # Always add the root node of the other node as the subtree of the head node ()
        other_root = other_site.root_node
        all_children = other_root._children_as_root
        for child in all_children:
            self.root_node.add_child(child)
        self.root_node.add_child(other_root)
        # Update the other root node - add a link between it and the corrent root node
        link = BondLink(self.root_node, other_root)
        other_root.incoming = link
        self.root_node.outgoing[other_root] = link
        other_root.root_node = self.root_node

    def get_all_ancestors(self) -> list:
        """Recursively find all ancestors"""
        ancestors = []
        if self.incoming:
            ancestors.extend(self.incoming.inp.get_all_ancestors())
            ancestors.append([self.incoming, self.incoming.inp])
        return ancestors

    def get_displacement_to_root(self):
        """
        Compute the distancement vector from this node to the root
        node by follwing the links
        """
        vector = np.zeros(3)
        for bond, _ in self.get_all_ancestors():
            vector[:] -= bond.vector
        return vector


class BondLink:
    """
    Class representing an 'Link' in a percolation network. It is essentially a Bond between different sites
    """

    def __init__(self, inp: SiteNode, out: SiteNode, metadata=None):
        """
        Links between two SiteNode
        """
        self._inp = inp
        self._out = out
        assert inp.uuid != out.uuid, "Cannot add a link between a site and itself"
        self.metadata = metadata if metadata else {}
        # Store the displacement vector
        if inp.atom:
            atoms = inp.atoms
            self.vd = atoms.get_distance(
                inp.atom.index, out.atom.index, vector=True, mic=True
            )
        else:
            self.vd = None

    @property
    def inp(self):
        return self._inp

    @property
    def out(self):
        return self._out

    def __repr__(self) -> str:
        return f"BondLink(inp={self.inp.id}, out={self.out.id})"

    @property
    def position(self):
        """Position of the link is the mid point of the two ends"""
        return (self.inp.pos + self.out.pos) / 2

    @property
    def vector(self):
        return self.vd
