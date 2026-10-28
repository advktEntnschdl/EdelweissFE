"""Section values and stresses of the cross section elements against closed-form solutions."""

import numpy as np
import pytest

from edelweissfe.elements.beamsectionelement.element import BeamSectionElement
from edelweissfe.elements.beamsectionelement.sectionanalysis import BeamSectionAnalysis
from edelweissfe.elements.beamsectionelement.shapefunctions import (
    sectionElementTypes,
    shapeFunctions,
)
from edelweissfe.points.node import Node

# parent coordinates of the nodes, in Abaqus order
_parentNodes = {
    "WARP2D2": [[-1], [1]],
    "WARP2D3": [[0, 0], [1, 0], [0, 1]],
    "WARP2D6": [[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]],
    "WARP2D4": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    "WARP2D8": [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]],
    "WARP2D9": [[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]],
}


def rectangleMesh(b, h, ny, nz, elType, origin=(0.0, 0.0), angle=0.0):
    """A structured mesh of the rectangle b x h, rotated by angle about its corner at origin."""
    elType = elType.upper()
    order = 2 if elType in ("WARP2D6", "WARP2D8", "WARP2D9") else 1
    ys, zs = np.linspace(0, b, order * ny + 1), np.linspace(0, h, order * nz + 1)
    c, s = np.cos(angle), np.sin(angle)
    nodes = {}

    def node(i, j):
        if (i, j) not in nodes:
            x = np.array([c * ys[i] - s * zs[j], s * ys[i] + c * zs[j]]) + origin
            nodes[(i, j)] = Node(len(nodes) + 1, x)
        return nodes[(i, j)]

    connectivities = []
    for i in range(ny):
        for j in range(nz):
            a, k = order * i, order * j
            corners = [(a, k), (a + order, k), (a + order, k + order), (a, k + order)]
            if elType == "WARP2D3":
                connectivities += [[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]]
            elif elType == "WARP2D4":
                connectivities.append(corners)
            else:
                mids = [(a + 1, k), (a + 2, k + 1), (a + 1, k + 2), (a, k + 1)]
                center = (a + 1, k + 1)
                if elType == "WARP2D6":
                    connectivities += [
                        [corners[0], corners[1], corners[2], mids[0], mids[1], center],
                        [corners[0], corners[2], corners[3], center, mids[2], mids[3]],
                    ]
                else:
                    connectivities.append(corners + mids + ([center] if elType == "WARP2D9" else []))

    elements = []
    for label, connectivity in enumerate(connectivities, start=1):
        el = BeamSectionElement(elType, label)
        el.setNodes([node(*ij) for ij in connectivity])
        el.initializeElement()
        el.setShearModulus(1.0)
        elements.append(el)
    return elements


def thinWalledMesh(points, segments, thicknesses, nPerSegment=6):
    nodes = {k: Node(k, np.array(v, dtype=float)) for k, v in points.items()}
    label = max(nodes) + 1
    elements = []
    for (a, b), t in zip(segments, thicknesses):
        pa, pb = nodes[a].coordinates, nodes[b].coordinates
        chain = [nodes[a]]
        for i in range(1, nPerSegment):
            chain.append(Node(label, pa + (pb - pa) * i / nPerSegment))
            label += 1
        chain.append(nodes[b])
        for n1, n2 in zip(chain[:-1], chain[1:]):
            el = BeamSectionElement("WARP2D2", len(elements) + 1)
            el.setNodes([n1, n2])
            el.setProperties([t])
            el.initializeElement()
            el.setShearModulus(1.0)
            elements.append(el)
    return elements


def stressResultants(analysis, **forces):
    """Vy, Vz and the torsion moment about the shear center of the shear stresses."""
    Vy = Vz = Mx = 0.0
    for el, (_, tau) in zip(analysis.elements, analysis.computeStresses(**forces)):
        dA = el._dA
        y, z = el._yzQP.T
        Vy += dA @ tau[:, 0]
        Vz += dA @ tau[:, 1]
        Mx += dA @ ((y - analysis.yM) * tau[:, 1] - (z - analysis.zM) * tau[:, 0])
    return np.array([Vy, Vz, Mx])


def torsionConstantRectangle(b, h):
    """Series solution for the rectangle b x h with b <= h."""
    n = np.arange(1, 400, 2)
    return h * b**3 / 3 * (1 - 192 * b / (np.pi**5 * h) * np.sum(np.tanh(n * np.pi * h / (2 * b)) / n**5))


@pytest.mark.parametrize("elType", list(sectionElementTypes))
def test_shape_functions_are_interpolating(elType):
    N = shapeFunctions[elType](np.array(_parentNodes[elType], dtype=float))[0]
    assert np.allclose(N, np.eye(len(N)))


@pytest.mark.parametrize("elType", ["WARP2D3", "WARP2D4", "WARP2D6", "WARP2D8", "WARP2D9"])
def test_rectangle(elType):
    b, h = 1.0, 2.0
    analysis = BeamSectionAnalysis(rectangleMesh(b, h, 8, 16, elType, origin=(3.0, -1.0)))
    v = analysis.sectionValues

    assert v.A == pytest.approx(b * h)
    assert (v.yBarS, v.zBarS) == pytest.approx((3.5, 0.0))
    assert v.alpha == pytest.approx(0.0)
    assert (v.Iy, v.Iz) == pytest.approx((b * h**3 / 12, h * b**3 / 12))
    assert (v.yM, v.zM) == pytest.approx((0.0, 0.0), abs=1e-10)
    assert v.IT == pytest.approx(torsionConstantRectangle(b, h), rel=2e-2)
    # without transverse normal stresses, the shear stresses due to Vz are parabolic: A_s = 5/6 A
    assert v.Asz / v.A == pytest.approx(5 / 6, rel=5e-3)

    assert stressResultants(analysis, Vy=1.0) == pytest.approx([1, 0, 0], abs=1e-8)
    assert stressResultants(analysis, Vz=1.0) == pytest.approx([0, 1, 0], abs=1e-8)
    assert stressResultants(analysis, Mxp=1.0) == pytest.approx([0, 0, 1], abs=1e-8)
    assert stressResultants(analysis, Mxs=1.0) == pytest.approx([0, 0, 1], abs=1e-8)


def test_rectangle_quadratic_converges():
    analysis = BeamSectionAnalysis(rectangleMesh(1.0, 2.0, 8, 16, "WARP2D9"))
    assert analysis.sectionValues.IT == pytest.approx(torsionConstantRectangle(1.0, 2.0), rel=1e-4)


def test_rotated_rectangle_principal_axes():
    angle = np.radians(30.0)
    analysis = BeamSectionAnalysis(rectangleMesh(1.0, 3.0, 4, 12, "WARP2D8", origin=(1.0, 2.0), angle=angle))
    v = analysis.sectionValues
    # the strong axis of the rotated rectangle is its long side, rotated by 30 degrees
    assert np.degrees(v.alpha) == pytest.approx(30.0)
    assert (v.Iy, v.Iz) == pytest.approx((3.0**3 / 12, 3.0 / 12))
    center = np.array([1.0, 2.0]) + np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ [
        0.5,
        1.5,
    ]
    assert (v.yBarS, v.zBarS) == pytest.approx(tuple(center))
    assert (v.yBarM, v.zBarM) == pytest.approx(tuple(center))


def test_thin_walled_i_section():
    bf, tf, h, tw = 0.2, 0.015, 0.3, 0.01
    points = {
        1: (-bf / 2, -h / 2),
        2: (0, -h / 2),
        3: (bf / 2, -h / 2),
        4: (-bf / 2, h / 2),
        5: (0, h / 2),
        6: (bf / 2, h / 2),
    }
    analysis = BeamSectionAnalysis(
        thinWalledMesh(points, [(1, 2), (2, 3), (4, 5), (5, 6), (2, 5)], [tf, tf, tf, tf, tw])
    )
    v = analysis.sectionValues
    assert v.Iw == pytest.approx(tf * bf**3 * h**2 / 24)
    assert v.IT == pytest.approx((2 * bf * tf**3 + h * tw**3) / 3)
    assert v.Iy == pytest.approx(2 * bf * tf * (h / 2) ** 2 + tw * h**3 / 12)
    assert stressResultants(analysis, Vz=1.0) == pytest.approx([0, 1, 0], abs=1e-10)
    assert stressResultants(analysis, Mxs=1.0) == pytest.approx([0, 0, 1], abs=1e-10)


def test_thin_walled_channel_shear_center():
    b, tf, h, tw = 0.1, 0.015, 0.3, 0.01
    points = {1: (b, -h / 2), 2: (0, -h / 2), 3: (0, h / 2), 4: (b, h / 2)}
    analysis = BeamSectionAnalysis(thinWalledMesh(points, [(1, 2), (2, 3), (3, 4)], [tf, tw, tf]))
    v = analysis.sectionValues
    assert v.yBarM == pytest.approx(-3 * b**2 * tf / (6 * b * tf + h * tw))
    assert v.Iw == pytest.approx(tf * b**3 * h**2 / 12 * (3 * b * tf + 2 * h * tw) / (6 * b * tf + h * tw))
    # the shear flow due to Vz is parabolic in the web and linear in the flanges: exact for the cubic interpolation
    assert stressResultants(analysis, Vz=1.0) == pytest.approx([0, 1, 0], abs=1e-10)


def test_thin_walled_box_bredt():
    b, h, t = 0.3, 0.2, 0.01
    points = {1: (0, 0), 2: (b, 0), 3: (b, h), 4: (0, h)}
    analysis = BeamSectionAnalysis(thinWalledMesh(points, [(1, 2), (2, 3), (3, 4), (4, 1)], [t] * 4))
    v = analysis.sectionValues
    ITBredt = 4 * (b * h) ** 2 / (2 * (b + h) / t)
    assert v.IT - v.ITOpen == pytest.approx(ITBredt)
    # the closed part carries the share of the torsion moment of the closed part
    assert stressResultants(analysis, Mxp=1.0)[2] == pytest.approx(ITBredt / v.IT)


def test_shear_deformation_scales_with_shear_modulus():
    elements = rectangleMesh(1.0, 2.0, 2, 4, "WARP2D8")
    uRef = BeamSectionAnalysis(elements).shearDeformation(Vz=1.0)
    for el in elements:
        el.setShearModulus(4.0)
    assert BeamSectionAnalysis(elements).shearDeformation(Vz=1.0) == pytest.approx(uRef / 4)
