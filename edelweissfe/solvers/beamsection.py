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
The BeamSection solver computes the section values of a beam cross section meshed with the
cross section elements (``WARP2D...``), see
:mod:`~edelweissfe.elements.beamsectionelement.sectionanalysis`, and the stresses for the section
forces of a ``>>sectionforces`` step action.

Results:

- the section values, written to the journal, to ``model.additionalParameters["beamSectionValues"]``
  and, optionally, to a csv file (option ``sectionValuesFile``),
- the node fields ``warping`` (the normalized warping ordinate omega) and ``shear deformation``
  (u due to Vy, Vz and Mxs),
- the element results ``normal stress``, ``shear stress`` (tau_xy, tau_xz), ``shear stress open``,
  ``warping`` and ``shear deformation`` at the quadrature points.

.. code-block:: edelweiss
    :caption: Example:

    *solver, solver=BeamSection, name=theSectionSolver
        sectionValuesFile=sectionValues.csv

    *step, solver=theSectionSolver
        maxInc=1.0
        >>sectionforces, name=forces, Vz=100.0, Mxp=2.5
"""

import numpy as np

from edelweissfe.elements.beamsectionelement.sectionanalysis import BeamSectionAnalysis
from edelweissfe.models.femodel import FEModel
from edelweissfe.solvers.base.nonlinearsolverbase import NonlinearSolverBase
from edelweissfe.stepactions.sectionforces import sectionForceNames
from edelweissfe.timesteppers.timestep import TimeStep


class BeamSection(NonlinearSolverBase):
    """Linear analysis of a beam cross section.

    Parameters
    ----------
    jobInfo
        A dictionary containing the job information.
    journal
        The journal instance for logging.
    sectionValuesFile
        An optional csv file for the section values.
    """

    identification = "BeamSection"

    #: Nothing is carried between increments: each increment is a complete linear analysis, set up in beginStep.
    checkpointedState = {}

    def __init__(self, jobInfo, journal, sectionValuesFile: str = None, **kwargs):
        if kwargs:
            raise KeyError(f"Unknown options for the {self.identification} solver: {', '.join(kwargs)}")
        self.journal = journal
        self.sectionValuesFile = sectionValuesFile
        self.analysis = None
        self._analysedElements = None
        self.incrementStatus = {}

    def beginStep(self, step, model: FEModel, fieldOutputController, outputmanagers):
        self.validateModelCapabilities(model)
        self.applyStepActionsAtStepStart(model, step)

        elements = list(model.elements.values())
        if self._analysedElements != [id(el) for el in elements]:
            self.analysis = BeamSectionAnalysis(elements)
            self._analysedElements = [id(el) for el in elements]
            self._reportSectionValues(model)

    def _reportSectionValues(self, model: FEModel):
        values = self.analysis.sectionValues.asDict()
        model.additionalParameters["beamSectionValues"] = values
        self.journal.printTable(
            [["section value", "value"]] + [[name, "{:.6e}".format(v)] for name, v in values.items()],
            self.identification,
        )
        if self.sectionValuesFile:
            with open(self.sectionValuesFile, "w") as f:
                f.write("name,value\n")
                f.writelines("{:},{:.12e}\n".format(name, v) for name, v in values.items())

    def prepareIncrement(self, step, model: FEModel, isRetry: bool):
        pass

    def attemptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        self.incrementStatus = {
            "step": step.number,
            "inc": timeStep.number,
            "iters": 1,
            "converged": True,
            "time inc": timeStep.timeIncrement,
            "time end": timeStep.totalTime,
            "notes": "",
        }

        forces = {force: 0.0 for force in sectionForceNames}
        for action in step.actions["sectionforces"].values():
            for force, value in action.getCurrentForces(timeStep.stepProgress).items():
                forces[force] += value
        self.forces = forces

        self.journal.message(
            "increment {:}, step progress {:.4f}; section forces: {:}".format(
                timeStep.number,
                timeStep.stepProgress,
                ", ".join("{:}={:.4e}".format(f, v) for f, v in forces.items() if v) or "none",
            ),
            self.identification,
            1,
        )
        self.analysis.computeStresses(**forces)

    def acceptIncrement(self, step, model: FEModel, timeStep: TimeStep):
        f = self.forces
        nodalValues = {
            "warping": self.analysis.omega,
            "shear deformation": self.analysis.shearDeformation(f["Vy"], f["Vz"], f["Mxs"]),
        }
        for field, values in nodalValues.items():
            nodeField = model.nodeFields[field]
            indices = np.array([nodeField.indexOfNode(node) for node in self.analysis.nodes])
            nodeField["U"][indices, 0] = values

        model.advanceToTime(timeStep.totalTime)

    def endStep(self, step, model: FEModel):
        pass

    def solveIncrement(self, *args):
        raise NotImplementedError(f"The {self.identification} solver solves each increment as a whole.")
