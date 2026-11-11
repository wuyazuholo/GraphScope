#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging

from graphscope.framework.app import AppAssets
from graphscope.framework.app import not_compatible_for
from graphscope.framework.app import project_to_simple

__all__ = ["pagerank", "pagerank_push", "pagerank_push_opt", "pagerank_nx"]

logger = logging.getLogger("graphscope")


@project_to_simple
@not_compatible_for("arrow_property", "dynamic_property")
def pagerank(graph, delta=0.85, max_round=None, tol=None):
    """Evaluate PageRank on a graph.

    Directed graphs stop when the global L1 rank change falls below tol.
    Undirected graphs retain the fixed-round implementation.

    Args:
        graph (:class:`graphscope.Graph`): A simple graph.
        delta (float, optional): Damping factor. Defaults to 0.85.
        max_round (int, optional): Maximum number of rounds. Defaults to 500
            for directed graphs and 10 for undirected graphs.
        tol (float, optional): Directed graph convergence threshold.
            Defaults to 1e-6; unsupported for undirected graphs.

    Returns:
        :class:`graphscope.framework.context.VertexDataContextDAGNode`:
            A context with each vertex assigned with the pagerank value, evaluated in eager mode.

    Examples:

    .. code:: python

        >>> import graphscope
        >>> from graphscope.dataset import load_p2p_network
        >>> sess = graphscope.session(cluster_type="hosts", mode="eager")
        >>> g = load_p2p_network(sess)
        >>> # project to a simple graph (if needed)
        >>> pg = g.project(vertices={"host": ["id"]}, edges={"connect": ["dist"]})
        >>> c = graphscope.pagerank(pg, delta=0.85, max_round=10)
        >>> sess.close()
    """
    delta = float(delta)
    if graph.is_directed():
        max_round = 500 if max_round is None else int(max_round)
        tol = 1e-6 if tol is None else float(tol)
        return AppAssets(algo="pagerank_directed", context="vertex_data")(
            graph, delta, max_round, tol
        )

    if tol is not None:
        raise ValueError("tol is only supported for directed graphs")
    max_round = 10 if max_round is None else int(max_round)
    return AppAssets(algo="pagerank", context="vertex_data")(
        graph, delta, max_round
    )


@project_to_simple
@not_compatible_for("arrow_property", "dynamic_property")
def pagerank_push(graph, delta=0.85, max_round=10):
    """Evaluate PageRank on a graph."""
    if graph.is_directed():
        logger.warning("PageRankPush is not designed for directed graph.")
    delta = float(delta)
    max_round = int(max_round)
    return AppAssets(algo="pagerank_push", context="vertex_data")(
        graph, delta, max_round
    )


@project_to_simple
@not_compatible_for("arrow_property", "dynamic_property")
def pagerank_push_opt(graph, delta=0.85, max_round=10):
    """Evaluate PageRank on a graph."""
    if graph.is_directed():
        logger.warning("PageRankPushOpt is not designed for directed graph.")
    delta = float(delta)
    max_round = int(max_round)
    return AppAssets(algo="pagerank_push_opt", context="vertex_data")(
        graph, delta, max_round
    )


@project_to_simple
@not_compatible_for("arrow_property", "dynamic_property")
def pagerank_nx(graph, alpha=0.85, max_iter=100, tol=1e-06):
    """Evaluate pagerank on a graph using algorithm exactly follows the implemented in NetworkX library.

    Args:
        graph (:class:`graphscope.Graph`): A simple graph.
        alpha (float, optional): Dumping factor. Defaults to 0.85.
        max_iter (int, optional): Maximum number of iteration. Defaults to 100.
        tol (float, optional): Error tolerance used to check convergence in power method solver.

    Returns:
        :class:`graphscope.framework.context.VertexDataContextDAGNode`:
            A context with each vertex assigned with the pagerank value, evaluated in eager mode.

    Examples:

    .. code:: python

        >>> import graphscope
        >>> from graphscope.dataset import load_p2p_network
        >>> sess = graphscope.session(cluster_type="hosts", mode="eager")
        >>> g = load_p2p_network(sess)
        >>> # project to a simple graph (if needed)
        >>> pg = g.project(vertices={"host": ["id"]}, edges={"connect": ["dist"]})
        >>> c = graphscope.pagerank_nx(pg,  alpha=0.85, max_iter=10, tol=1e-06)
        >>> sess.close()
    """
    alpha = float(alpha)
    max_iter = int(max_iter)
    return AppAssets(algo="pagerank_nx", context="vertex_data")(
        graph, alpha, max_iter, tol
    )
