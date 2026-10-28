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
Finite elements for the cross sections of beams, after Kindmann & Kraus, *Finite-Elemente-Methoden
im Stahlbau*, chapter 7.

Every node carries two scalar fields, both displacements u in the beam axis direction x:

- ``warping``: the warping ordinate (unit warping) omega, i.e., u for a twist rate of -1,
  from the virtual work (7.8) -- (7.41), (7.51) -- (7.53),
- ``shear deformation``: u due to the shear forces Vy, Vz and the secondary torsion moment Mxs,
  from the virtual work (7.12) -- (7.42), (7.54) -- (7.56).

Both problems share the stiffness matrix G * int(grad(f)^T grad(f)) dA and differ only in their load
vectors. They are solved by the :class:`~edelweissfe.solvers.beamsection.BeamSection` solver
through :class:`~edelweissfe.elements.beamsectionelement.sectionanalysis.BeamSectionAnalysis`, which
uses the kernels of this element.

Element types:

- ``WARP2D2``: 2-node line element for thin-walled sections, (7.24), (7.26); the wall thickness is
  given by a ``plane`` section,
- ``WARP2D3``, ``WARP2D6``: 3- and 6-node triangles for thick-walled sections,
- ``WARP2D4``, ``WARP2D8``, ``WARP2D9``: 4-, 8- and 9-node quadrilaterals for thick-walled sections.

The shear modulus G is derived from a linear elastic material (E, nu). It scales the shear
deformations only; warping ordinates, section values and stresses do not depend on it in a
homogeneous section.

.. code-block:: edelweiss
    :caption: Example:

    *element, type=WARP2D4, provider=edelweiss
        1, 1, 2, 3, 4
"""

import numpy as np

from edelweissfe.elements.base.baseelement import BaseElement
from edelweissfe.elements.beamsectionelement.shapefunctions import (
    sectionElementTypes,
    shapeFunctions,
)
from edelweissfe.points.node import Node
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict


class BeamSectionElement(BaseElement):
    """A cross section element.

    Parameters
    ----------
    elementType
        The element type, see the module documentation.
    elNumber
        A unique integer label.
    """

    #: the results available at the quadrature points, and their dimensions
    resultDimensions = {
        "warping": 1,
        "shear deformation": 1,
        "normal stress": 1,
        "shear stress": 2,
        "shear stress open": 1,
    }

    def __init__(self, elementType: str, elNumber: int):
        self._elType = elementType
        self._type = sectionElementTypes[elementType.upper()]
        self._shapeFunctions = shapeFunctions[elementType.upper()]
        self._elNumber = elNumber
        self._nNodes = self._type.nNodes
        self.nSpatialDimensions = 2
        self._isLine = self._type.nParametricDimensions == 1
        self._t = 1.0
        self._G = 1.0
        self._hasMaterial = False
        self._results = CaseInsensitiveDict(
            {name: np.zeros((self._type.nInt, dim)) for name, dim in self.resultDimensions.items()}
        )

    @property
    def elNumber(self) -> int:
        return self._elNumber

    @property
    def elType(self) -> str:
        return self._elType

    @property
    def nNodes(self) -> int:
        return self._nNodes

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @property
    def nDof(self) -> int:
        return 2 * self._nNodes

    @property
    def fields(self) -> list[list[str]]:
        return [["warping", "shear deformation"] for _ in range(self._nNodes)]

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        return np.arange(self.nDof)

    @property
    def ensightType(self) -> str:
        return self._type.ensightType

    @property
    def visualizationNodes(self) -> list[Node]:
        return self._nodes[: self._type.nVisualizationNodes]

    @property
    def hasMaterial(self) -> bool:
        return self._hasMaterial

    @property
    def isThinWalled(self) -> bool:
        """True for the line elements of thin-walled sections."""
        return self._isLine

    @property
    def thickness(self) -> float:
        return self._t

    @property
    def length(self) -> float:
        """The length of thin-walled (line) elements."""
        return self._length

    @property
    def shearModulus(self) -> float:
        return self._G

    @property
    def sectionCoordinates(self) -> np.ndarray:
        """The nodal coordinates (y, z) the section kernels currently refer to, shape (nNodes, 2)."""
        return self._yz

    def setNodes(self, nodes: list[Node]):
        self._nodes = nodes
        self._nodesCoordinates = np.array([n.coordinates[:2] for n in nodes], dtype=float)

    def setProperties(self, elementProperties: np.ndarray):
        """The first property is the wall thickness of thin-walled (line) elements; ignored otherwise."""
        if self._isLine and len(elementProperties):
            self._t = float(elementProperties[0])

    def assignProperty(self, propertyName: str, properties: np.ndarray):
        if propertyName.lower() == "thickness" and self._isLine:
            self._t = float(properties[0])
            self.initializeSection(self._yz)
        else:
            raise NotImplementedError(f"Property '{propertyName}' is not supported by element {self._elType}.")

    def getPropertyNames(self) -> list[str]:
        return ["thickness"] if self._isLine else []

    def initializeElement(self):
        self.initializeSection(self._nodesCoordinates)

    def setMaterial(self, material, materialProperties: np.ndarray = None):
        """Take the shear modulus from a linear elastic material (E, nu), given as an EdelweissFE
        material or, Marmot-style, as a name and its properties."""
        properties = material.materialProperties if materialProperties is None else materialProperties
        E, nu = properties[0], properties[1]
        self.setShearModulus(E / (2 * (1 + nu)))

    def setShearModulus(self, G: float):
        self._G = float(G)
        self._hasMaterial = True

    def initializeSection(self, coordinates: np.ndarray):
        """Evaluate the geometry at the quadrature points for nodal coordinates (y, z) in the
        coordinate system of the section analysis, e.g., the principal axes.

        Parameters
        ----------
        coordinates
            The nodal coordinates, shape (nNodes, 2).
        """
        t = self._type
        self._yz = np.asarray(coordinates, dtype=float)
        if self._isLine:
            d = self._yz[1] - self._yz[0]
            self._length = np.linalg.norm(d)
            e = d / self._length
            dNds = t.dNdXi[:, 0, :] * 2 / self._length
            # the gradient of a thin-walled element only has a component in the profile direction s
            self._dNdY = e[None, :, None] * dNds[:, None, :]
            self._dA = t.w * self._length / 2 * self._t
            self._projector = np.outer(e, e)
            self._tangent = e
        else:
            J = np.einsum("qan,nb->qab", t.dNdXi, self._yz)
            detJ = np.linalg.det(J)
            if np.any(detJ <= 0):
                raise Exception(f"Element {self._elNumber} has a non-positive Jacobian; check the node ordering.")
            self._dNdY = np.linalg.solve(J, t.dNdXi)
            self._dA = t.w * detJ
            self._projector = np.eye(2)
        self._yzQP = t.N @ self._yz

    def _rotationField(self, yM: float, zM: float) -> np.ndarray:
        """The term p = (z - zM, -(y - yM)) of the warping problem at the quadrature points, projected
        onto the directions a thin-walled element resolves, shape (nInt, 2)."""
        p = np.column_stack([self._yzQP[:, 1] - zM, -(self._yzQP[:, 0] - yM)])
        return p @ self._projector

    def computeSectionStiffness(self) -> np.ndarray:
        """The element stiffness matrix (7.24), (7.52), (7.55), shape (nNodes, nNodes)."""
        return self._G * np.einsum("q,qan,qam->nm", self._dA, self._dNdY, self._dNdY)

    def computeWarpingLoad(self, yD: float, zD: float) -> np.ndarray:
        """The load vector for a twist rate of -1 about the pole D (7.24), (7.53)."""
        return -self._G * np.einsum("q,qan,qa->n", self._dA, self._dNdY, self._rotationField(yD, zD))

    def computeShearLoad(self, sigmaPrimeX: np.ndarray) -> np.ndarray:
        """The load vector (7.26), (7.56) for the nodal derivatives of the normal stress
        sigma'x = Vy/Iz*y + Vz/Iy*z + Mxs/Iw*omega."""
        return np.einsum("q,qn,q->n", self._dA, self._type.N, self._type.N @ sigmaPrimeX)

    def integrate(self, a: np.ndarray = None, b: np.ndarray = None) -> float:
        """The integral of the product of two interpolated nodal fields over the element area;
        a field left None is 1, so integrate() is the area."""
        fa = 1.0 if a is None else self._type.N @ a
        fb = 1.0 if b is None else self._type.N @ b
        return float(np.sum(self._dA * fa * fb))

    def computeTorsionConstant(self, omega: np.ndarray, yM: float, zM: float) -> float:
        """The contribution to the torsion constant IT (7.62), (7.31), for the normalized warping
        ordinates omega; for thin-walled elements, plus the open part l*t^3/3 (7.32)."""
        p = self._rotationField(yM, zM)
        gradOmega = np.einsum("qan,n->qa", self._dNdY, omega)
        IT = float(np.sum(self._dA * np.einsum("qa,qa->q", gradOmega + p, p)))
        if self._isLine:
            IT += self._length * self._t**3 / 3
        return IT

    def computePrimaryTorsionShearStresses(self, omega: np.ndarray, yM: float, zM: float) -> np.ndarray:
        """The shear stresses (tau_xy, tau_xz) at the quadrature points for Mxp / IT = 1 (7.63), (7.36).
        For thin-walled elements, this is the part constant over the wall thickness only."""
        return -(np.einsum("qan,n->qa", self._dNdY, omega) + self._rotationField(yM, zM))

    def computeShearStresses(self, u: np.ndarray, sigmaPrimeX: np.ndarray) -> np.ndarray:
        """The shear stresses (tau_xy, tau_xz) at the quadrature points from the shear deformations u
        (7.64); for thin-walled elements from the cubic distribution (7.39) along the element."""
        if not self._isLine:
            return self._G * np.einsum("qan,n->qa", self._dNdY, u)
        xi = self._type.xi[:, 0]
        Fa, Fb = sigmaPrimeX
        tau = self._G / self._length * (u[1] - u[0]) + self._length / 24 * (
            Fa * (-1 - 6 * xi + 3 * xi**2) + Fb * (1 - 6 * xi - 3 * xi**2)
        )
        return tau[:, None] * self._tangent[None, :]

    def interpolateAtQuadraturePoints(self, nodalValues: np.ndarray) -> np.ndarray:
        return self._type.N @ nodalValues

    def evaluateShapeFunctions(self, xi: np.ndarray) -> np.ndarray:
        """The shape functions at parent coordinates xi, shape (nPoints, nParametricDimensions)."""
        return self._shapeFunctions(np.atleast_2d(xi))[0]

    def setResult(self, result: str, values: np.ndarray):
        """Store a result at the quadrature points, shape (nInt,) or (nInt, dim)."""
        self._results[result][:] = np.reshape(values, self._results[result].shape)

    def getResultArray(self, result: str, quadraturePoint: int, getPersistentView: bool = True) -> np.ndarray:
        return self._results[result][quadraturePoint]

    def getCoordinatesAtCenter(self) -> np.ndarray:
        return np.average(self._nodesCoordinates, axis=0)

    def getCoordinatesAtQuadraturePoints(self) -> np.ndarray:
        return self._type.N @ self._nodesCoordinates

    def getNumberOfQuadraturePoints(self) -> int:
        return self._type.nInt

    def computeKernels(self, K: np.ndarray, P: np.ndarray, U: np.ndarray, dU: np.ndarray, time, dTime):
        """Both fields as uncoupled potential problems without loads; the section problems themselves
        are solved by the BeamSection solver."""
        Ke = self.computeSectionStiffness()
        KFull = np.kron(Ke, np.eye(2))
        K += KFull.ravel() if K.ndim == 1 else KFull
        P += KFull @ U

    def computeKernelsExplicit(self, *args, **kwargs):
        raise NotImplementedError(f"{self._elType} is not intended for explicit simulations.")

    def computeDistributedLoad(self, *args, **kwargs):
        raise NotImplementedError(f"{self._elType} does not support distributed loads.")

    def computeBodyForce(self, *args, **kwargs):
        raise NotImplementedError(f"{self._elType} does not support body forces.")

    def computeLumpedInertia(self, *args, **kwargs):
        raise NotImplementedError(f"{self._elType} has no inertia.")

    def computeCriticalTimeStepForExplicitDynamics(self, *args, **kwargs) -> float:
        return np.inf

    def computeInternalEnergy(self) -> float:
        return 0.0

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        raise NotImplementedError(f"{self._elType} has no state.")

    def acceptLastState(self):
        pass

    def resetToLastValidState(self):
        pass
