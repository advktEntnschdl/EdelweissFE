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
The section forces for the stresses computed by the :class:`~edelweissfe.solvers.beamsection.BeamSection`
solver. They refer to the principal axes of the section; Mxp, Mxs and Mw to the shear center.
Unspecified forces are zero; within a step, the forces change linearly from those of the previous step.

.. code-block:: edelweiss
    :caption: Example:

    *step, solver=theSectionSolver
        >>sectionforces, name=forces, Vz=100.0, Mxp=2.5
"""

from dataclasses import dataclass, fields

from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.misc import withoutParserBookkeepingKeys
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField


@dataclass(frozen=True)
class SectionForcesSchema:
    """The options of the ``sectionforces`` step action."""

    name: str | None = schemaField(
        description="Name of the step action.", dtype=str, default=None, required=True, structuralOnly=True
    )
    N: float = schemaField(description="Normal force.", dtype=float, default=0.0)
    Vy: float = schemaField(description="Shear force in y direction.", dtype=float, default=0.0)
    Vz: float = schemaField(description="Shear force in z direction.", dtype=float, default=0.0)
    My: float = schemaField(description="Bending moment about the y axis.", dtype=float, default=0.0)
    Mz: float = schemaField(description="Bending moment about the z axis.", dtype=float, default=0.0)
    Mxp: float = schemaField(description="Primary torsion moment.", dtype=float, default=0.0)
    Mxs: float = schemaField(description="Secondary torsion moment.", dtype=float, default=0.0)
    Mw: float = schemaField(description="Warping bimoment.", dtype=float, default=0.0)


#: the names of the section forces
sectionForceNames = [f.name for f in fields(SectionForcesSchema) if f.name != "name"]


class StepAction(StepActionBase):
    """Section forces acting on a beam cross section.

    Parameters
    ----------
    name
        The name of this step action.
    forces
        The section forces at the end of the step, by name.
    """

    schema = SectionForcesSchema

    #: The forces at the start and the end of the step, by name.
    checkpointedState = {"forcesAtStepStart": dict, "forces": dict}

    def __init__(self, name: str, forces: dict[str, float]):
        self.name = name
        self.forces = {force: 0.0 for force in sectionForceNames} | forces
        self.forcesAtStepStart = {force: 0.0 for force in sectionForceNames}

    @staticmethod
    def _forcesFromDefinition(definition: dict) -> dict[str, float]:
        definition = CaseInsensitiveDict(withoutParserBookkeepingKeys(definition))
        definition.pop("name", None)
        options = buildSchemaFromOptions(SectionForcesSchema, definition)
        return {force: getattr(options, force) for force in sectionForceNames}

    @classmethod
    def fromStepActionDefinition(cls, name, definition, jobInfo, model, fieldOutputController, journal):
        return cls(name, cls._forcesFromDefinition(definition))

    def updateStepAction(self, definition, jobInfo, model, fieldOutputController, journal):
        self.forcesAtStepStart = self.forces
        self.forces = self._forcesFromDefinition(definition)

    def applyAtStepEnd(self, model, stepMagnitude=None):
        self.forcesAtStepStart = self.forces

    def getCurrentForces(self, stepProgress: float) -> dict[str, float]:
        """The section forces at a relative progress of the step."""
        return {
            force: self.forcesAtStepStart[force] + stepProgress * (self.forces[force] - self.forcesAtStepStart[force])
            for force in sectionForceNames
        }
