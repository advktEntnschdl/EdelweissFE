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
#  Paul Hofer Paul.Hofer@uibk.ac.at
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

import numpy as np

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.misc import convertLinesToStringDictionary

"""
A lagrangian multiplier based constraint for single quadrature point
simulations of undrained conditions
"""

documentation = {
    "nSet": "The node set to be constrained.",
}


class Constraint(ConstraintBase):
    def __init__(self, name: str, definitionLines: list, model: FEModel):
        definition = convertLinesToStringDictionary(definitionLines)

        theField = "strain symmetric"
        self.sizeField = getFieldSize(theField, model.domainSize)
        # self.component = int(definition["component"])
        self._name = name
        self._nodes = model.nodeSets[definition["nSet"]]
        self.nNodes = len(self._nodes)
        self.nMultipliers = len(self._nodes)

        self._nDof = self.sizeField * self.nNodes + self.nMultipliers

        self._fieldsOnNodes = [
            [
                theField,
            ]
        ] * self.nNodes

        self.active = True

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def getNumberOfAdditionalNeededScalarVariables(self):
        return self.nNodes

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
        timeStep: TimeStep,
    ):
        if not self.active:
            return

        K[0, -1] += 1
        K[1, -1] += 1
        K[2, -1] += 1

        K[-1, 0] += 1
        K[-1, 1] += 1
        K[-1, 2] += 1

        PExt[0] -= U_np[-1]
        PExt[1] -= U_np[-1]
        PExt[2] -= U_np[-1]
