# sim_graph.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
import inspect
import numpy as np
from graphviz import Digraph


@dataclass
class _Node:
    nid: str
    title: str
    ports_in: List[str] = field(default_factory=list)
    ports_out: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class SimGraph:
    """
    Auto-build a dataflow graph of Simulation.of_dynamical_system_block calls.

    Usage
    -----
    >>> # from your code: import compute  # has compute.Simulation
    >>> G = SimGraph("MetaF")
    >>> with G.patch_simulation(compute.Simulation):
    ...     # run code that calls Simulation.of_dynamical_system_block
    ...     meta_f(xk0, tk=0.0)
    ...
    >>> G.render("meta_f_graph")  # -> meta_f_graph.svg

    How it works
    ------------
    - Monkey-patches Simulation.of_dynamical_system_block with a wrapper.
    - Each call becomes a node. Node titles use block.name or type(block).__name__.
    - The wrapper records outputs; later, if those exact Python objects are passed
      as inputs (in arg_dict/kwarg_dict/x0), edges are drawn based on identity().
    - Ports are inferred with simple heuristics and you can post-tweak them via
      `rename_port` if needed.
    """

    def __init__(self, name: str = "SimGraph"):
        self.name = name
        self._nodes: Dict[str, _Node] = {}
        self._edges: List[Tuple[str, str, str, str, str]] = (
            []
        )  # (src, sport, dst, dport, label)
        self._produced: Dict[int, Tuple[str, str]] = (
            {}
        )  # id(obj) -> (node_id, out_port)
        self._seq = 0

        # patching state
        self._orig_method = None
        self._sim_cls = None

    # -------- patching / unpatching -------------------------------------------------

    def patch_simulation(self, simulation_cls):
        """
        Context manager to patch `simulation_cls.of_dynamical_system_block`.
        """
        self._sim_cls = simulation_cls
        self._orig_method = simulation_cls.of_dynamical_system_block

        def _wrapper(*args, **kwargs):
            # Expect signature: (block, *, ...). Block is first arg.
            block = args[0] if args else kwargs.get("block")
            # Create node for this call
            node_id = self._make_node_id(block)
            title = self._title_for(block)
            self._ensure_node(node_id, title)

            # Extract inputs the best we can
            t = kwargs.get("tk")
            dt = kwargs.get("dt")
            arg_dict = kwargs.get("arg_dict") or {}
            kwarg_dict = kwargs.get("kwarg_dict") or {}
            x0 = kwargs.get("x0", None)

            # Record input ports (heuristics)
            self._add_in_port(node_id, "in_t")
            if x0 is not None:
                self._add_in_port(node_id, "in_x")
            # Add ports for keys in arg/kwarg dicts
            for k in sorted(arg_dict.keys()):
                self._add_in_port(node_id, f"in:{k}")
            for k in sorted(kwarg_dict.keys()):
                self._add_in_port(node_id, f"in:{k}")

            # Wire edges for any inputs that match previously produced objects
            self._maybe_wire_input(node_id, "in_x", x0)
            for k, v in arg_dict.items():
                self._maybe_wire_input(node_id, f"in:{k}", v)
            for k, v in kwarg_dict.items():
                self._maybe_wire_input(node_id, f"in:{k}", v)

            # Call through to the original method
            out = self._orig_method(*args, **kwargs)

            # Interpret outputs & register producers
            self._register_outputs(node_id, out)

            # Store some meta for nice labels
            self._nodes[node_id].meta.update(
                {
                    "tk": float(t) if t is not None else None,
                    "dt": float(dt) if dt is not None else None,
                }
            )

            return out

        # Bind wrapper as a function (not descriptor)
        def __enter__():
            self._sim_cls.of_dynamical_system_block = _wrapper
            return self

        def __exit__(exc_type, exc, tb):
            # Restore original
            self._sim_cls.of_dynamical_system_block = self._orig_method  # type: ignore
            self._orig_method = None
            self._sim_cls = None
            return False  # don't swallow exceptions

        # Simple context manager object
        class _CM:
            def __enter__(_self):
                return __enter__()

            def __exit__(_self, et, ex, tb):
                return __exit__(et, ex, tb)

        return _CM()

    # -------- graph construction helpers -------------------------------------------

    def _make_node_id(self, block) -> str:
        self._seq += 1
        return f"n{self._seq}"

    def _title_for(self, block) -> str:
        title = getattr(block, "name", None)
        if not title:
            title = type(block).__name__
        return title

    def _ensure_node(self, nid: str, title: str):
        if nid not in self._nodes:
            self._nodes[nid] = _Node(nid=nid, title=title)

    def _add_in_port(self, nid: str, port: str):
        n = self._nodes[nid]
        if port not in n.ports_in:
            n.ports_in.append(port)

    def _add_out_port(self, nid: str, port: str):
        n = self._nodes[nid]
        if port not in n.ports_out:
            n.ports_out.append(port)

    def _edge(self, src: str, sport: str, dst: str, dport: str, label: str):
        self._edges.append((src, sport, dst, dport, label))

    def _maybe_wire_input(self, dst_id: str, dport: str, val: Any):
        if val is None:
            return
        # We try: identity match, then for numpy arrays: identity of object
        key = id(val)
        if key in self._produced:
            src_id, sport = self._produced[key]
            self._edge(src_id, sport, dst_id, dport, dport.replace("in:", ""))

    def _register_outputs(self, nid: str, out: Any):
        """
        Heuristics:
          - If out is a tuple/list of length 2: treat as (x, y) -> ports out_x, out_y
          - Else single output: treat as out_y (often measurement) but allow override:
            * If node title contains 'u' or 'control' -> out_u
        """

        # Helper to remember an object as produced by a node/port
        def remember(obj, port):
            self._add_out_port(nid, port)
            try:
                self._produced[id(obj)] = (nid, port)
            except Exception:
                pass

        # Numpy arrays come in many shapes — store the object itself
        if isinstance(out, (tuple, list)) and len(out) == 2:
            x, y = out
            remember(x, "out_x")
            remember(y, "out_y")
        else:
            # Single output — guess name
            title = self._nodes[nid].title.lower()
            port = (
                "out_u"
                if ("u" in title and "turtle" not in title) or ("control" in title)
                else "out_y"
            )
            remember(out, port)

    # -------- public API niceties ----------------------------------------------------

    def rename_port(self, nid: str, old: str, new: str):
        n = self._nodes[nid]
        if old in n.ports_in:
            n.ports_in[n.ports_in.index(old)] = new
        if old in n.ports_out:
            n.ports_out[n.ports_out.index(old)] = new
        # Update edges
        new_edges = []
        for s, sp, d, dp, lab in self._edges:
            if sp == old and s == nid:
                sp = new
            if dp == old and d == nid:
                dp = new
            new_edges.append((s, sp, d, dp, lab if lab != old else new))
        self._edges = new_edges

    def render(self, filename: Optional[str] = None):
        """
        Render to SVG if filename given; otherwise return a Graphviz Digraph for inline display.
        """
        g = Digraph(self.name, format="svg")
        g.attr(rankdir="LR", fontsize="11")
        g.attr(
            "node",
            shape="record",
            style="rounded,filled",
            fillcolor="white",
            fontsize="11",
        )

        # Build record labels
        for n in self._nodes.values():
            header_lines = [n.title]
            # small meta in header
            tk = n.meta.get("tk")
            dt = n.meta.get("dt")
            meta_bits = []
            if tk is not None:
                meta_bits.append(f"t={tk:g}")
            if dt is not None:
                meta_bits.append(f"dt={dt:g}")
            if meta_bits:
                header_lines.append("\\n" + ", ".join(meta_bits))
            header = "".join(header_lines)

            left = "|".join(f"<{p}>{p}" for p in n.ports_in)
            right = "|".join(f"<{p}>{p}" for p in n.ports_out)
            if left and right:
                label = "{" + header + "|" + "{" + left + "}|{" + right + "}" + "}"
            elif left:
                label = "{" + header + "|" + left + "}"
            elif right:
                label = "{" + header + "|" + right + "}"
            else:
                label = header
            g.node(n.nid, label=label)

        # Edges
        for s, sp, d, dp, lab in self._edges:
            g.edge(f"{s}:{sp}", f"{d}:{dp}", label=lab)

        if filename:
            g.render(filename, cleanup=True)
            return filename + ".svg"
        return g
