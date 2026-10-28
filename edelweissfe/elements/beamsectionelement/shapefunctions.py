#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#  Paul Hofer paul.hofer@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
# Created on Tue Oct 27 2026

# @author: Paul Hofer
"""
Shape functions and quadrature rules of the cross section elements.

Node numbering follows Abaqus: corner nodes counterclockwise, then the mid-side nodes (the first one
between nodes 1 and 2), then the center node of the 9-node quadrilateral.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SectionElementType:
    """The interpolation of a cross section element type, evaluated at its quadrature points."""

    nNodes: int
    #: 1 for the thin-walled line element, 2 for the plane elements
    nParametricDimensions: int
    ensightType: str
    #: the nodes used for visualization (the center node of the 9-node quadrilateral is omitted)
    nVisualizationNodes: int
    #: quadrature point coordinates in the parent domain, shape (nInt, nParametricDimensions)
    xi: np.ndarray
    #: quadrature weights, shape (nInt,)
    w: np.ndarray
    #: shape functions at the quadrature points, shape (nInt, nNodes)
    N: np.ndarray
    #: parametric derivatives at the quadrature points, shape (nInt, nParametricDimensions, nNodes)
    dNdXi: np.ndarray

    @property
    def nInt(self) -> int:
        return len(self.w)


def _lagrange1D(xi: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """1D Lagrange polynomials (and derivatives) on [-1, 1] with nodes ordered -1, (0,) 1."""
    xi = np.atleast_1d(xi)
    if order == 1:
        return np.array([(1 - xi) / 2, (1 + xi) / 2]).T, np.array([-0.5 + 0 * xi, 0.5 + 0 * xi]).T
    return (
        np.array([xi * (xi - 1) / 2, 1 - xi**2, xi * (xi + 1) / 2]).T,
        np.array([xi - 0.5, -2 * xi, xi + 0.5]).T,
    )


def shapeFunctionsLine2(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N, dN = _lagrange1D(xi[:, 0], 1)
    return N, dN[:, None, :]


def shapeFunctionsTri3(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, s = xi[:, 0], xi[:, 1]
    N = np.array([1 - r - s, r, s]).T
    dN = np.tile(np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]), (len(r), 1, 1))
    return N, dN


def shapeFunctionsTri6(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, s = xi[:, 0], xi[:, 1]
    L = np.array([1 - r - s, r, s])
    # derivatives of the area coordinates w.r.t. (r, s)
    dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    N = np.array(
        [
            L[0] * (2 * L[0] - 1),
            L[1] * (2 * L[1] - 1),
            L[2] * (2 * L[2] - 1),
            4 * L[0] * L[1],
            4 * L[1] * L[2],
            4 * L[2] * L[0],
        ]
    ).T
    dN = np.zeros((len(r), 2, 6))
    for d in range(2):
        dN[:, d, 0] = (4 * L[0] - 1) * dL[0, d]
        dN[:, d, 1] = (4 * L[1] - 1) * dL[1, d]
        dN[:, d, 2] = (4 * L[2] - 1) * dL[2, d]
        dN[:, d, 3] = 4 * (dL[0, d] * L[1] + L[0] * dL[1, d])
        dN[:, d, 4] = 4 * (dL[1, d] * L[2] + L[1] * dL[2, d])
        dN[:, d, 5] = 4 * (dL[2, d] * L[0] + L[2] * dL[0, d])
    return N, dN


def shapeFunctionsQuad4(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _lagrangeQuad(xi, 1, [(0, 0), (1, 0), (1, 1), (0, 1)])


def shapeFunctionsQuad9(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # (index in xi direction, index in eta direction) of the 1D polynomials, nodes at -1, 0, 1
    return _lagrangeQuad(xi, 2, [(0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1)])


def _lagrangeQuad(xi: np.ndarray, order: int, nodeIndices: list) -> tuple[np.ndarray, np.ndarray]:
    Nr, dNr = _lagrange1D(xi[:, 0], order)
    Ns, dNs = _lagrange1D(xi[:, 1], order)
    N = np.array([Nr[:, i] * Ns[:, j] for i, j in nodeIndices]).T
    dN = np.stack(
        [
            np.array([dNr[:, i] * Ns[:, j] for i, j in nodeIndices]).T,
            np.array([Nr[:, i] * dNs[:, j] for i, j in nodeIndices]).T,
        ],
        axis=1,
    )
    return N, dN


def shapeFunctionsQuad8(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, s = xi[:, 0], xi[:, 1]
    rc = np.array([-1, 1, 1, -1])
    sc = np.array([-1, -1, 1, 1])
    N = np.zeros((len(r), 8))
    dN = np.zeros((len(r), 2, 8))
    for i in range(4):
        a, b = rc[i], sc[i]
        N[:, i] = 0.25 * (1 + a * r) * (1 + b * s) * (a * r + b * s - 1)
        dN[:, 0, i] = 0.25 * a * (1 + b * s) * (2 * a * r + b * s)
        dN[:, 1, i] = 0.25 * b * (1 + a * r) * (a * r + 2 * b * s)
    # mid-side nodes at s = -1, r = 1, s = 1, r = -1
    for i, (a, b) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)], start=4):
        if a == 0:
            N[:, i] = 0.5 * (1 - r**2) * (1 + b * s)
            dN[:, 0, i] = -r * (1 + b * s)
            dN[:, 1, i] = 0.5 * b * (1 - r**2)
        else:
            N[:, i] = 0.5 * (1 + a * r) * (1 - s**2)
            dN[:, 0, i] = 0.5 * a * (1 - s**2)
            dN[:, 1, i] = -s * (1 + a * r)
    return N, dN


def gaussLegendreQuad(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()]), np.outer(w, w).ravel()


def dunavantTri6() -> tuple[np.ndarray, np.ndarray]:
    """6-point rule on the unit triangle, exact for polynomials of degree 4."""
    a, b = 0.445948490915965, 0.091576213509771
    wa, wb = 0.223381589678011 / 2, 0.109951743655322 / 2
    xi = np.array([[a, a], [1 - 2 * a, a], [a, 1 - 2 * a], [b, b], [1 - 2 * b, b], [b, 1 - 2 * b]])
    return xi, np.array([wa, wa, wa, wb, wb, wb])


def _makeType(nNodes, nParametricDimensions, ensightType, nVisualizationNodes, rule, shapeFunctions):
    xi, w = rule
    N, dNdXi = shapeFunctions(xi)
    return SectionElementType(nNodes, nParametricDimensions, ensightType, nVisualizationNodes, xi, w, N, dNdXi)


_line2Rule = np.polynomial.legendre.leggauss(2)

#: The cross section element types; the name follows Abaqus' WARP2D elements for meshed beam sections.
sectionElementTypes = {
    "WARP2D2": _makeType(2, 1, "bar2", 2, (_line2Rule[0][:, None], _line2Rule[1]), shapeFunctionsLine2),
    "WARP2D3": _makeType(3, 2, "tria3", 3, dunavantTri6(), shapeFunctionsTri3),
    "WARP2D6": _makeType(6, 2, "tria6", 6, dunavantTri6(), shapeFunctionsTri6),
    "WARP2D4": _makeType(4, 2, "quad4", 4, gaussLegendreQuad(2), shapeFunctionsQuad4),
    "WARP2D8": _makeType(8, 2, "quad8", 8, gaussLegendreQuad(3), shapeFunctionsQuad8),
    "WARP2D9": _makeType(9, 2, "quad8", 8, gaussLegendreQuad(3), shapeFunctionsQuad9),
}

#: Shape function evaluation by element type, for evaluating results at arbitrary parent coordinates.
shapeFunctions = {
    "WARP2D2": shapeFunctionsLine2,
    "WARP2D3": shapeFunctionsTri3,
    "WARP2D6": shapeFunctionsTri6,
    "WARP2D4": shapeFunctionsQuad4,
    "WARP2D8": shapeFunctionsQuad8,
    "WARP2D9": shapeFunctionsQuad9,
}
