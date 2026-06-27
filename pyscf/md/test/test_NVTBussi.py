#!/usr/bin/env python
# Copyright; Samik Bose and Nathan Emeott (Michigan State University)
# Not implemented in pyscf
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
"""Unit tests for the CSVR (Bussi-Donadio-Parrinello) thermostat.

These tests exercise NVTBussi._scale_velocities in isolation -- they do NOT
run an SCF/gradient calculation. A bare instance is built with __new__ and the
handful of attributes the rescale step touches are set by hand, so the test is
fast and deterministic and only depends on the thermostat math.
"""

import unittest
import numpy as np

from pyscf import data
from pyscf.md.integrators import NVTBussi


def _bare_bussi(natm=10, T=300.0, dt=20.0, taut=None, seed=12345):
    """Construct an NVTBussi instance without invoking _Integrator.__init__
    (which would require a real gradient method / SCF)."""
    if taut is None:
        taut = 10.0 * dt                      # stiff-ish: low autocorrelation
    inst = NVTBussi.__new__(NVTBussi)
    inst.mol = type("Mol", (), {"natm": natm})()
    inst.T = T
    inst.dt = dt
    inst.taut = taut
    inst.rng = np.random.default_rng(seed)
    inst.thermostat_work = 0.0
    inst.epot = None
    inst.ekin = None
    inst._masses = np.ones(natm)              # masses irrelevant to the test
    inst.veloc = inst.rng.standard_normal((natm, 3)) * 1e-3
    return inst


def _sample_ke(inst, N, burn):
    """Iterate the rescale step as a Markov chain and collect the kinetic
    energy after burn-in."""
    ks = np.empty(N - burn)
    for i in range(N):
        inst._scale_velocities()
        if i >= burn:
            ks[i - burn] = inst.compute_kinetic_energy()
    return ks


class KnownValues(unittest.TestCase):

    def test_canonical_distribution(self):
        """The stationary KE distribution must be the canonical one:
        K ~ Gamma(shape=dof/2, scale=kT), i.e.
            <K>    = dof/2 * kT          (= sigma)
            Var(K) = dof/2 * kT**2
        Matching the *variance* (not just the mean) is what distinguishes
        CSVR from Berendsen."""
        inst = _bare_bussi()                   # natm=10 -> dof=30
        dof = 3 * inst.mol.natm
        kT = inst.T * data.nist.BOLTZMANN / data.nist.HARTREE2J
        sigma = 0.5 * dof * kT
        var_target = 0.5 * dof * kT * kT

        ks = _sample_ke(inst, N=500000, burn=20000)

        # Stochastic test with a fixed seed -> deterministic; tolerances have
        # ample margin (observed ~0.05% / ~1% with this seed and sample count).
        self.assertAlmostEqual(ks.mean() / sigma, 1.0, delta=0.01)
        self.assertAlmostEqual(ks.var() / var_target, 1.0, delta=0.05)

    def test_canonical_distribution_small_nf(self):
        """Regression guard for the well-known 2007-paper typo (matsci.org
        'error in Bussi thermostat'): the sum of (Nf-1) squared normals is a
        chi-square (gamma scale 2), NOT gamma scale 1. Following the paper
        literally halves that term's mean; approximating the chi-square by a
        normal fails at small Nf. This code draws np.random.chisquare(dof-1),
        an exact chi-square at any Nf -- so the canonical variance must be
        reproduced even at very low dof, which is exactly where the typo bites.
        """
        inst = _bare_bussi(natm=2)             # dof = 6, a small-Nf stressor
        dof = 3 * inst.mol.natm
        kT = inst.T * data.nist.BOLTZMANN / data.nist.HARTREE2J
        sigma = 0.5 * dof * kT
        var_target = 0.5 * dof * kT * kT

        ks = _sample_ke(inst, N=1500000, burn=50000)

        # A paper-literal (scale-1) implementation would miss the variance by
        # ~4x here; these tolerances easily separate correct from typo'd.
        self.assertAlmostEqual(ks.mean() / sigma, 1.0, delta=0.015)
        self.assertAlmostEqual(ks.var() / var_target, 1.0, delta=0.06)

    def test_taut_zero_instantaneous(self):
        """taut == 0 is the documented CSVR limit: the decay factor c -> 0 and
        every step draws K' directly from the canonical distribution
        (instantaneous stochastic rescaling), GROMACS style. This must not
        divide by zero, and the samples -- now i.i.d. canonical draws -- must
        reproduce the Gamma(dof/2, kT) mean and variance."""
        inst = _bare_bussi(taut=0.0)
        dof = 3 * inst.mol.natm
        kT = inst.T * data.nist.BOLTZMANN / data.nist.HARTREE2J
        sigma = 0.5 * dof * kT
        var_target = 0.5 * dof * kT * kT

        ks = _sample_ke(inst, N=200000, burn=2000)   # i.i.d. -> fast convergence
        self.assertAlmostEqual(ks.mean() / sigma, 1.0, delta=0.01)
        self.assertAlmostEqual(ks.var() / var_target, 1.0, delta=0.03)

    def test_negative_taut_raises(self):
        """A negative relaxation time is unphysical; the constructor rejects it
        before any gradient evaluation (so method can be a placeholder)."""
        self.assertRaises(ValueError, NVTBussi, None, 300.0, -1.0)

    def test_thermostat_work_bookkeeping(self):
        """In isolation the thermostat is the ONLY thing changing the
        velocities, so the accumulated thermostat_work telescopes exactly to
        (K_final - K_initial). (In a real run the intervening Verlet step also
        changes K, so this exact equality holds only for this isolated test.)"""
        inst = _bare_bussi(seed=7)
        k0 = inst.compute_kinetic_energy()
        for _ in range(5000):
            inst._scale_velocities()
        k_final = inst.compute_kinetic_energy()
        self.assertAlmostEqual(inst.thermostat_work, k_final - k0,
                               delta=1e-12 * max(1.0, abs(k_final)))

    def test_zero_velocity_is_noop(self):
        """With (near) zero kinetic energy the rescale factor is undefined;
        GROMACS-style behaviour is a silent no-op that touches neither the
        velocities nor the conserved-quantity bookkeeping."""
        inst = _bare_bussi()
        inst.veloc = np.zeros((inst.mol.natm, 3))
        inst.thermostat_work = 0.0
        inst._scale_velocities()
        self.assertEqual(inst.thermostat_work, 0.0)
        self.assertTrue(np.all(inst.veloc == 0.0))

    def test_econs_none_until_run(self):
        """econs is undefined before epot/ekin are populated by a step."""
        inst = _bare_bussi()
        self.assertIsNone(inst.econs)
        inst.epot, inst.ekin = -1.0, 0.5
        self.assertAlmostEqual(inst.econs,
                               inst.epot + inst.ekin - inst.thermostat_work)


if __name__ == "__main__":
    print("Full Tests for CSVR (Bussi) thermostat")
    unittest.main()
