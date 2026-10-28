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
Section values and stresses of beam cross sections with the FEM, after Kindmann & Kraus,
*Finite-Elemente-Methoden im Stahlbau*, chapter 7 (calculation procedure of section 7.6):

1. normalization part I (table 7.2): area A, centroid S, principal axes (angle alpha), Iy, Iz,
2. warping ordinates for an arbitrary pole D = S from K * omega_D = f (7.28a), one node per
   connected part held (7.29a),
3. normalization part II (tables 7.3, 7.4, 7.8): shear center M and normalized warping ordinate omega,
4. warping constant Iw (7.30), (7.61) and torsion constant IT (7.31) -- (7.33), (7.62),
5. shear deformations u for unit Vy, Vz and Mxs from K * u = f (7.28b),
6. stresses: normal stresses sigma_x (figure 7.6) and shear stresses for given section forces
   (7.36) -- (7.39), (7.63), (7.64).

All quantities refer to the principal axes y, z through the centroid unless named otherwise;
the axes of the mesh are denoted by a bar. Additionally, the shear areas Asy, Asz follow from the
complementary energy of the shear stresses due to Vy and Vz, Vy^2 / (G * Asy) = int(tau^2 / G) dA.

The section is assumed homogeneous, i.e., all elements have the same shear modulus.

.. code-block:: python
    :caption: Example:

    analysis = BeamSectionAnalysis(model.elements.values())
    values = analysis.sectionValues
    analysis.computeStresses(Vz=100.0, Mxp=5.0)
"""

from dataclasses import dataclass, fields

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from edelweissfe.elements.beamsectionelement.element import BeamSectionElement


@dataclass(frozen=True)
class BeamSectionValues:
    """The section values; coordinates refer to the principal axes through the centroid, except
    those with the suffix Bar, which refer to the axes of the mesh."""

    A: float
    yBarS: float
    zBarS: float
    #: angle from the mesh axes to the principal axes (rad), y = cos(alpha) * yBar + sin(alpha) * zBar
    alpha: float
    Iy: float
    Iz: float
    yM: float
    zM: float
    yBarM: float
    zBarM: float
    Iw: float
    IT: float
    #: the part of IT from the single plates of thin-walled elements (7.32)
    ITOpen: float
    Asy: float
    Asz: float
    G: float

    def asDict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class BeamSectionAnalysis:
    """Section values, warping ordinates, shear deformations and stresses of a meshed cross section.

    Parameters
    ----------
    elements
        The cross section elements; their nodal coordinates are (yBar, zBar).
    """

    def __init__(self, elements):
        self.elements = list(elements)
        if not all(isinstance(el, BeamSectionElement) for el in self.elements):
            raise TypeError("A beam section analysis requires cross section elements (WARP2D...) only.")

        G = np.array([el.shearModulus for el in self.elements])
        if not np.allclose(G, G[0]):
            raise ValueError("The beam section analysis assumes a homogeneous section (a single shear modulus).")
        self.G = float(G[0])

        nodeIndex = {}
        self.nodes = []
        for el in self.elements:
            for node in el.nodes:
                if id(node) not in nodeIndex:
                    nodeIndex[id(node)] = len(self.nodes)
                    self.nodes.append(node)
        #: the global node indices of each element
        self.elementNodeIndices = [np.array([nodeIndex[id(n)] for n in el.nodes]) for el in self.elements]
        #: the nodal coordinates in the mesh axes (yBar, zBar)
        self.coordinatesBar = np.array([n.coordinates[:2] for n in self.nodes], dtype=float)

        self._normalizePartI()
        self._assembleStiffness()
        self._computeWarping()
        self._computeShearDeformations()

        self.sectionValues = BeamSectionValues(
            A=self.A,
            yBarS=self.centroidBar[0],
            zBarS=self.centroidBar[1],
            alpha=self.alpha,
            Iy=self.Iy,
            Iz=self.Iz,
            yM=self.yM,
            zM=self.zM,
            yBarM=self.shearCenterBar[0],
            zBarM=self.shearCenterBar[1],
            Iw=self.Iw,
            IT=self.IT,
            ITOpen=self.ITOpen,
            Asy=self.Asy,
            Asz=self.Asz,
            G=self.G,
        )

    @property
    def nNodes(self) -> int:
        return len(self.nodes)

    def _integrate(self, a: np.ndarray = None, b: np.ndarray = None) -> float:
        """The integral of the product of two nodal fields over the section."""
        return sum(
            el.integrate(None if a is None else a[idx], None if b is None else b[idx])
            for el, idx in zip(self.elements, self.elementNodeIndices)
        )

    def _assembleVector(self, elementVector) -> np.ndarray:
        f = np.zeros(self.nNodes)
        for el, idx in zip(self.elements, self.elementNodeIndices):
            np.add.at(f, idx, elementVector(el, idx))
        return f

    def _normalizePartI(self):
        """Table 7.2: centroid, principal axes, and the principal coordinates of the nodes."""
        for el, idx in zip(self.elements, self.elementNodeIndices):
            el.initializeSection(self.coordinatesBar[idx])

        yBar, zBar = self.coordinatesBar.T
        self.A = self._integrate()
        self.centroidBar = np.array([self._integrate(yBar), self._integrate(zBar)]) / self.A

        yT, zT = (self.coordinatesBar - self.centroidBar).T
        Ayy, Azz, Ayz = self._integrate(zT, zT), self._integrate(yT, yT), self._integrate(yT, zT)
        # tan(2 alpha) = 2 Ayz / (Azz - Ayy), with |alpha| <= 45 deg
        if abs(Ayz) <= 1e-12 * (Ayy + Azz):
            self.alpha = 0.0
        elif abs(Azz - Ayy) <= 1e-12 * (Ayy + Azz):
            self.alpha = np.sign(Ayz) * np.pi / 4
        else:
            self.alpha = 0.5 * np.arctan(2 * Ayz / (Azz - Ayy))
        c, s = np.cos(self.alpha), np.sin(self.alpha)
        #: rotation from the mesh axes to the principal axes
        self.rotation = np.array([[c, s], [-s, c]])
        #: the nodal coordinates in the principal axes (y, z)
        self.coordinates = (self.coordinatesBar - self.centroidBar) @ self.rotation.T

        for el, idx in zip(self.elements, self.elementNodeIndices):
            el.initializeSection(self.coordinates[idx])
        y, z = self.coordinates.T
        self.Iy, self.Iz = self._integrate(z, z), self._integrate(y, y)

    def _assembleStiffness(self):
        rows, cols, vals = [], [], []
        for el, idx in zip(self.elements, self.elementNodeIndices):
            rows.append(np.repeat(idx, len(idx)))
            cols.append(np.tile(idx, len(idx)))
            vals.append(el.computeSectionStiffness().ravel())
        K = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.nNodes, self.nNodes)
        )

        # the section is held against rigid body displacements in x by one node per connected part (7.29)
        nParts, labels = connected_components(K, directed=False)
        heldNodes = np.array([np.flatnonzero(labels == part)[0] for part in range(nParts)])
        self._freeNodes = np.setdiff1d(np.arange(self.nNodes), heldNodes)
        self.K = K
        self._factorization = splu(K[self._freeNodes][:, self._freeNodes].tocsc())

    def _solve(self, f: np.ndarray) -> np.ndarray:
        u = np.zeros(self.nNodes)
        u[self._freeNodes] = self._factorization.solve(f[self._freeNodes])
        return u

    def _computeWarping(self):
        """Warping about the centroid as pole D, then normalization part II (table 7.3)."""
        y, z = self.coordinates.T
        f = self._assembleVector(lambda el, idx: el.computeWarpingLoad(0.0, 0.0))
        #: the warping ordinates about the pole D = S, with an arbitrary constant
        self.omegaD = self._solve(f)

        A_w = self._integrate(self.omegaD)
        A_yw = self._integrate(y, self.omegaD)
        A_zw = self._integrate(z, self.omegaD)
        omegaK = A_w / self.A
        self.yM = A_zw / self.Iy
        self.zM = -A_yw / self.Iz
        self.shearCenterBar = self.centroidBar + self.rotation.T @ np.array([self.yM, self.zM])

        #: the normalized warping ordinates
        self.omega = self.omegaD - omegaK - z * self.yM + y * self.zM
        self.Iw = self._integrate(self.omega, self.omega)

        self.ITOpen = sum(el.length * el.thickness**3 / 3 for el in self.elements if el.isThinWalled)
        self.IT = sum(
            el.computeTorsionConstant(self.omega[idx], self.yM, self.zM)
            for el, idx in zip(self.elements, self.elementNodeIndices)
        )

    def sigmaPrimeX(self, Vy: float = 0.0, Vz: float = 0.0, Mxs: float = 0.0) -> np.ndarray:
        """The nodal derivative of the normal stress along the beam axis (figure 7.6)."""
        y, z = self.coordinates.T
        sigmaPrime = Vy / self.Iz * y + Vz / self.Iy * z
        if Mxs:
            sigmaPrime = sigmaPrime + Mxs / self.Iw * self.omega
        return sigmaPrime

    def _computeShearDeformations(self):
        """The shear deformations for unit Vy, Vz and Mxs, free of their mean value."""
        self.unitShearDeformations = {}
        for name, forces in (("Vy", dict(Vy=1.0)), ("Vz", dict(Vz=1.0)), ("Mxs", dict(Mxs=1.0))):
            if name == "Mxs" and self.Iw <= 1e-12 * self.Iy * self.Iz / self.A:
                # warping-free section (e.g., circle, thin-walled star): no secondary torsion
                self.unitShearDeformations[name] = np.zeros(self.nNodes)
                continue
            sigmaPrime = self.sigmaPrimeX(**forces)
            f = self._assembleVector(lambda el, idx: el.computeShearLoad(sigmaPrime[idx]))
            u = self._solve(f)
            u -= self._integrate(u) / self.A
            self.unitShearDeformations[name] = u
            if name == "Vy":
                self.Asy = 1.0 / (self.G * u @ f)
            elif name == "Vz":
                self.Asz = 1.0 / (self.G * u @ f)

    def shearDeformation(self, Vy: float = 0.0, Vz: float = 0.0, Mxs: float = 0.0) -> np.ndarray:
        """The nodal shear deformations u due to Vy, Vz and Mxs."""
        u = self.unitShearDeformations
        return Vy * u["Vy"] + Vz * u["Vz"] + Mxs * u["Mxs"]

    def computeStresses(
        self,
        N: float = 0.0,
        Vy: float = 0.0,
        Vz: float = 0.0,
        My: float = 0.0,
        Mz: float = 0.0,
        Mxp: float = 0.0,
        Mxs: float = 0.0,
        Mw: float = 0.0,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The stresses for given section forces at the quadrature points of each element, also
        stored as the element results ``normal stress``, ``shear stress`` (tau_xy, tau_xz),
        ``shear stress open`` (thin-walled: the maximum due to Mxp of the single plate (7.37)),
        ``warping`` and ``shear deformation``.

        The section forces refer to the principal axes; Mxp, Mxs and Mw to the shear center.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            Per element, sigma_x (nInt,) and (tau_xy, tau_xz) (nInt, 2).
        """
        u = self.shearDeformation(Vy, Vz, Mxs)
        sigmaPrime = self.sigmaPrimeX(Vy, Vz, Mxs)
        y, z = self.coordinates.T
        sigma = N / self.A + My / self.Iy * z - Mz / self.Iz * y
        if Mw:
            sigma = sigma + Mw / self.Iw * self.omega

        stresses = []
        for el, idx in zip(self.elements, self.elementNodeIndices):
            sigmaQP = el.interpolateAtQuadraturePoints(sigma[idx])
            tauQP = el.computeShearStresses(u[idx], sigmaPrime[idx])
            if Mxp:
                tauQP = tauQP + Mxp / self.IT * el.computePrimaryTorsionShearStresses(self.omega[idx], self.yM, self.zM)
            el.setResult("normal stress", sigmaQP)
            el.setResult("shear stress", tauQP)
            el.setResult(
                "shear stress open",
                np.full(el.getNumberOfQuadraturePoints(), Mxp / self.IT * el.thickness if el.isThinWalled else 0.0),
            )
            el.setResult("warping", el.interpolateAtQuadraturePoints(self.omega[idx]))
            el.setResult("shear deformation", el.interpolateAtQuadraturePoints(u[idx]))
            stresses.append((sigmaQP, tauQP))
        return stresses
