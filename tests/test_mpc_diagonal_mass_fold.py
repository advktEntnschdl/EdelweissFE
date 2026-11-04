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
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""Explicit dynamics folds the lumped slave masses of a multi-point constraint onto its masters by the
row sum of :math:`T^T M T` (interpolation weights, e.g. a tie) or by its diagonal (lever arms, e.g. the
linearized rigid body), see
:attr:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase.lumpedMassFolding`.

Pinned here: the diagonal fold gives a rigid body reference point the total mass and the rotational
inertia :math:`\\sum m |d|^2`, where the row sum would give the signed static moment; the row sum stays
the default for every other slave.
"""

import numpy as np

from edelweissfe.constraints.linearizedrigidbody import rigidBodyKinematicsCoefficients
from edelweissfe.numerics.mpctransformation import MultiPointConstraintTransformation

# 2D: slave nodes 0, 1, 2 (DOFs 0-5), RP displacement DOFs 6, 7, RP rotation DOF 8, tie slave DOF 9
# interpolating DOFs 10, 11
_DISTANCES = [np.array([-1.0, 0.5]), np.array([0.0, 0.5]), np.array([1.0, 0.5])]
_MASSES = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0])


def _rigidBodyRecords():
    records = []
    for s, d in enumerate(_DISTANCES):
        for i, rotationTerms in enumerate(rigidBodyKinematicsCoefficients(d, 2)):
            records.append((2 * s + i, [(6 + i, 1.0)] + [(8, c) for _, c in rotationTerms]))
    return records


def _fold(diagonalMassFoldSlaveDofs):
    records = _rigidBodyRecords() + [(9, [(10, 0.25), (11, 0.75)])]
    mpc = MultiPointConstraintTransformation(records, 12, diagonalMassFoldSlaveDofs=diagonalMassFoldSlaveDofs)
    M = _MASSES.copy()
    mpc.foldLumpedMass(M)
    return M


def test_diagonal_fold_gives_rigid_body_mass_and_rotational_inertia():
    M = _fold(frozenset(range(6)))
    nodeMasses = _MASSES[0:6:2]

    np.testing.assert_allclose(M[:6], 0.0)
    np.testing.assert_allclose(M[6:8], nodeMasses.sum())
    np.testing.assert_allclose(M[8], sum(m * d @ d for m, d in zip(nodeMasses, _DISTANCES)))
    # the tie slave is still folded by the row sum
    np.testing.assert_allclose(M[9:], [0.0, 5.0 + 0.25 * 4.0, 6.0 + 0.75 * 4.0])


def test_row_sum_fold_gives_the_static_moment_on_the_rotation():
    M = _fold(frozenset())
    nodeMasses = _MASSES[0:6:2]

    # sum m (-d_y) + sum m d_x: the static moment, of either sign and possibly zero
    np.testing.assert_allclose(M[8], sum(m * (d[0] - d[1]) for m, d in zip(nodeMasses, _DISTANCES)))
