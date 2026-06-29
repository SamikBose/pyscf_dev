#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Authors: Matthew Hennefarth <matthew.hennefarth@gmail.com>,
#          Aniruddha Seal <aniruddhaseal2011@gmail.com>

import os
from math import sqrt
import numpy as np

from pyscf import data, md
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad.rhf import GradientsBase


class Frame:
    '''Basic class to hold information at each time step of a MD simulation

    Attributes:
        ekin : float
            Kinetic energy
        epot : float
            Potential energy (electronic energy)
        etot : float
            Total energy, sum of the potential and kinetic energy
        coord : 2D array with shape (natm, 3)
            Geometry of the system at the current time step
        veloc : 2D array with shape (natm, 3)
            Velocities of the system at the current time step
        time : float
            Time for which this frame represents
    '''

    def __init__(self,
                 ekin=None,
                 epot=None,
                 coord=None,
                 veloc=None,
                 time=None):
        self.ekin = ekin
        self.epot = epot
        self.etot = self.ekin + self.epot
        self.coord = coord
        self.veloc = veloc
        self.time = time


def _toframe(integrator):
    '''Convert an _Integrator to a Frame given current saved data.

    Args:
        integrator : md.integrator._Integrator object

    Returns:
        Frame with all data taken from the integrator.
    '''
    return Frame(ekin=integrator.ekin,
                 epot=integrator.epot,
                 coord=integrator.mol.atom_coords(),
                 veloc=integrator.veloc,
                 time=integrator.time)


def _write(dev, mol, vec, atmlst=None):
    '''Format output of molecular vector quantity.

    Args:
        dev : lib.logger.Logger object
        mol : gto.mol object
        vec : 2D array with shape (mol.natm, 3)
        atmlst : array of indices to pull atoms from.
            Must be smaller than mol.natm.
    '''
    if atmlst is None:
        atmlst = range(mol.natm)
    ia_width = len(str(max(atmlst)))
    dev.stdout.write(' ' * ia_width + '        x                y                z\n')
    fmt = f'%{ia_width}d %s  %15.10f  %15.10f  %15.10f\n'
    for k, ia in enumerate(atmlst):
        dev.stdout.write(
            fmt %
            (ia, mol.atom_symbol(ia), vec[k, 0], vec[k, 1], vec[k, 2]))


def kernel(integrator, verbose=logger.NOTE):
    log = logger.new_logger(integrator, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start BOMD')

    integrator.mol = integrator.mol.copy()

    # Begin the iterations to run molecular dynamics
    t1 = t0
    for iteration, frame in enumerate(integrator):
        log.note('----------- %s final geometry -----------',
                 integrator.__class__.__name__)
        _write(integrator, integrator.mol, frame.coord)
        log.note('----------------------------------------------')
        log.note('------------ %s final velocity -----------',
                 integrator.__class__.__name__)
        _write(integrator, integrator.mol, frame.veloc)
        log.note('----------------------------------------------')
        log.note('Ekin = %17.13f', frame.ekin)
        log.note('Epot = %17.13f', frame.epot)
        log.note('Etot = %17.13f', frame.etot)

        t1 = log.timer('BOMD iteration %d' % iteration, *t1)

    t0 = log.timer('BOMD', *t0)
    return integrator


class _Integrator(lib.StreamObject):
    '''Integrator abstract base class. Should never be directly constructed,
    but inherited from.

    Args:
        method : lib.GradScanner, rhf.GradientsBase instance, or
        has nuc_grad_method method.
            Method by which to compute the energy gradients and energies
            in order to propagate the equations of motion. Realistically,
            it can be any callable object such that it returns the energy
            and potential energy gradient when given a mol.

    Attributes:
        incore_anyway : bool
            If true, then it will save every frame in memory.
            False, no frames are saved.

        veloc : ndarray
            Initial velocity for the simulation. Values should be given
            in atomic units (Bohr/a.u.). Dimensions should be (natm, 3) such as

             [[x1, y1, z1],
             [x2, y2, z2],
             [x3, y3, z3]]

        verbose : int
            Print level

        steps : int
            Number of steps to take when the kernel or run function is called.

        dt : float
            Time between steps. Given in atomic units.

        stdout : file object
            Default is self.scanner.mol.stdout.

        data_output : file object
            Stream to write energy and temperature to
            during the course of the simulation.

        trajectory_output : file object
            Stream to write the trajectory to during the course of the
            simulation. Written in xyz format.

        frames : ndarray of Frames or None
            If incore_anyway is true, then this will hold a list of frames
            corresponding to the simulation trajectory.

        epot : float
            Potential energy of the last time step during the simulation.

        ekin : float
            Kinetic energy of the last time step during the simulation

        time : float
            Time of the last step during the simulation.

        callback : function(envs_dict) => None
            Callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            environment.
    '''

    def __init__(self, method, **kwargs):

        if isinstance(method, lib.GradScanner):
            self.scanner = method
        elif isinstance(method, GradientsBase):
            self.scanner = method.as_scanner()
        elif getattr(method, 'nuc_grad_method', None):
            self.scanner = method.nuc_grad_method().as_scanner()
        else:
            raise NotImplementedError('Nuclear gradients of %s not available' %
                                      method)

        self.mol = self.scanner.mol
        self.stdout = self.mol.stdout
        self.incore_anyway = self.mol.incore_anyway
        self.veloc = None
        self.verbose = self.mol.verbose
        self.steps = 1
        self.dt = 10
        self.frames = None
        self.epot = None
        self.ekin = None
        self.time = 0
        self.data_output = None
        self.trajectory_output = None
        self.callback = None

        # Cache the masses into a list, they will be in atomic units
        self._masses = None

        self.__dict__.update(kwargs)

    def kernel(self, veloc=None, steps=None, dump_flags=True, verbose=None):
        '''Runs the molecular dynamics simulation.

        Args:
            veloc : ndarray
                Initial velocity for the simulation. Values should be given
                in atomic units (Bohr/a.u.). Dimensions should be (natm, 3)
                such as

                [[x1, y1, z1],
                 [x2, y2, z2],
                 [x3, y3, z3]]

            steps : int
                Number of steps to take when the kernel or run function
                is called.

            dump_flags : bool
                Print flags to output.

            verbose : int
                Print level

        Returns:
            _Integrator with final epot, ekin, temp,
            mol, and veloc of the simulation.
        '''

        if veloc is not None:
            self.veloc = veloc

        if steps is not None:
            self.steps = steps

        if verbose is None:
            verbose = self.verbose

        # Default velocities are 0 if none specified
        if self.veloc is None:
            self.veloc = np.full((self.mol.natm, 3), 0.0)

        # Store the masses into a cached variable,
        # so we don't have to keep looking them up
        self._masses = np.array([
            data.elements.COMMON_ISOTOPE_MASSES[m] * data.nist.AMU2AU
            for m in self.mol.atom_charges()])

        # avoid opening data_output file twice
        if type(self.data_output) is str:
            if self.verbose > logger.QUIET:
                if os.path.isfile(self.data_output):
                    print('overwrite data output file: %s' %
                          self.data_output)
                else:
                    print('data output file: %s' % self.data_output)

            if self.data_output == '/dev/null':
                self.data_output = open(os.devnull, 'w')

            else:
                self.data_output = open(self.data_output, 'w')
                self.data_output.write(
                    'time          Epot                 Ekin                 '
                    'Etot                 T\n'
                )

        # avoid opening trajectory_output file twice
        if type(self.trajectory_output) is str:
            if self.verbose > logger.QUIET:
                if os.path.isfile(self.trajectory_output):
                    print('overwrite energy output file: %s' %
                          self.trajectory_output)
                else:
                    print('trajectory output file: %s' %
                          self.trajectory_output)

            if self.trajectory_output == '/dev/null':
                self.trajectory_output = open(os.devnull, 'w')

            else:
                self.trajectory_output = open(self.trajectory_output, 'w')

        log = logger.new_logger(self, verbose)
        self.check_sanity()

        if dump_flags and self.verbose > logger.NOTE:
            self.dump_input()

        return kernel(self, verbose=log)

    def dump_input(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** BOMD flags ********')
        log.info('dt = %f', self.dt)
        log.info('Iterations = %d', self.steps)
        log.info('                   Initial Velocity                  ')
        log.info('             vx              vy              vz')
        for i, (e, v) in enumerate(zip(self.mol.elements, self.veloc)):
            log.info('%d %s    %.8E  %.8E  %.8E', i, e, *v)

    def check_sanity(self):
        assert self.time >= 0
        assert self.dt > 0
        assert self.steps > 0
        assert self.veloc is not None
        assert self.veloc.shape == (self.mol.natm, 3)
        assert self.scanner is not None

        return self

    def compute_kinetic_energy(self):
        '''Compute the kinetic energy of the current frame.'''
        return 0.5 * np.dot(self._masses, np.einsum('ij,ij->i', self.veloc, self.veloc))

    def temperature(self):
        '''Returns the temperature of the system'''
        # checked against ORCA for linear and non-linear molecules
        dof = 3 * len(self.mol.atom_coords())

        # Temp = 2/(3*k*N_f) * KE
        #      = 2/(3*k*N_f)*\sum_i (1/2 m_i v_i^2)
        return ((2 * self.ekin) / (
                dof * data.nist.BOLTZMANN / data.nist.HARTREE2J))

    def __iter__(self):
        self._step = 0
        self._log = logger.new_logger(self, self.verbose)
        return self

    def __next__(self):
        if self._step < self.steps:
            if self._log.verbose >= lib.logger.NOTE:
                self._log.note('\nBOMD Time %.2f', self.time)

            current_frame = self._next()

            if self.incore_anyway:
                self.frames.append(current_frame)

            if self.data_output is not None:
                self._write_data()

            if self.trajectory_output is not None:
                self._write_coord()

            if callable(self.callback):
                mol = self.mol
                scanner = self.scanner
                self.callback(locals())

            self._step += 1
            self.time += self.dt

            return current_frame

        else:
            raise StopIteration

    def _next(self):
        '''Determines the next step in the molecular dynamics simulation.
        Integrates to the next time step. Must be implemented in derived
        classes.

        Returns: 'Frame' which contains the new geometry, velocity,
        time step, and energy.
        '''
        raise NotImplementedError('Method Not Implemented')

    def _write_data(self):
        '''Writes out the potential, kinetic, and total energy, temperature to the
        self.data_output stream. '''

        output = '%8.2f  %.12E  %.12E  %.12E %3.4f' % (self.time,
                                                       self.epot,
                                                       self.ekin,
                                                       self.ekin + self.epot,
                                                       self.temperature())

        # We follow OM of writing all the states at the end of the line
        if getattr(self.scanner.base, 'e_states', None) is not None:
            if len(self.scanner.base.e_states) > 1:
                for e in self.scanner.base.e_states:
                    output += '  %.12E' % e

        self.data_output.write(output + '\n')

        # If we don't flush, there is a possibility of losing data
        self.data_output.flush()

    def _write_coord(self):
        '''Writes out the current geometry to the self.trajectory_output
        stream in xyz format. '''
        self.trajectory_output.write('%s\nMD Time %.2f\n' %
                                     (self.mol.natm, self.time))
        self.trajectory_output.write(self.mol.tostring(format='raw') + '\n')

        # If we don't flush, there is a possibility of losing data
        self.trajectory_output.flush()


class VelocityVerlet(_Integrator):
    '''Velocity Verlet algorithm

    Args:
        method : lib.GradScanner or rhf.GradientsBase instance, or
        has nuc_grad_method method.
            Method by which to compute the energy gradients and energies
            in order to propagate the equations of motion. Realistically,
            it can be any callable object such that it returns the energy
            and potential energy gradient when given a mol.

    Attributes:
        accel : ndarray
            Current acceleration for the simulation. Values are given
            in atomic units (Bohr/a.u.^2). Dimensions is (natm, 3) such as

             [[x1, y1, z1],
             [x2, y2, z2],
             [x3, y3, z3]]
    '''

    def __init__(self, method, **kwargs):
        super().__init__(method, **kwargs)
        self.accel = None

    def _next(self):
        '''Computes the next frame of the simulation and sets all internal
         variables to this new frame. First computes the new geometry,
         then the next acceleration, and finally the velocity, all according
         to the Velocity Verlet algorithm.

        Returns:
            The next frame of the simulation.
        '''

        # If no acceleration, compute that first, and then go
        # onto the next step
        if self.accel is None:
            next_epot, next_accel = self._compute_accel()

        else:
            self.mol.set_geom_(self._next_geometry(), unit='B')
            self.mol.build()
            next_epot, next_accel = self._compute_accel()
            self.veloc = self._next_velocity(next_accel)

        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel

        return _toframe(self)

    def _compute_accel(self):
        '''Given the current geometry, computes the acceleration
        for each atom.'''
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge!')

        a = -1 * grad / self._masses.reshape(-1, 1)
        return e_tot, a

    def _next_geometry(self):
        '''Computes the next geometry using the Velocity Verlet algorithm. The
        necessary equations of motion for the position is
            r(t_i+1) = r(t_i) + /delta t * v(t_i) + 0.5(/delta t)^2 a(t_i)
        '''
        return self.mol.atom_coords() + self.dt * self.veloc + \
            0.5 * (self.dt ** 2) * self.accel

    def _next_velocity(self, next_accel):
        '''Compute the next velocity using the Velocity Verlet algorithm. The
        necessary equations of motion for the velocity is
            v(t_i+1) = v(t_i) + 0.5(a(t_i+1) + a(t_i))'''
        return self.veloc + 0.5 * self.dt * (self.accel + next_accel)


class RandomNoiseVelocityVerlet(VelocityVerlet):
    '''Velocity Verlet algorithm with random noise applied to acceleration

    Args:
        method : lib.GradScanner or rhf.GradientsBase instance, or
        has nuc_grad_method method.
            Method by which to compute the energy gradients and energies
            in order to propagate the equations of motion. Realistically,
            it can be any callable object such that it returns the energy
            and potential energy gradient when given a mol.

        std_dev : float
            Standard deviation of the normal distribution.

        rng : np.random.Generator
            Random number generator to sample from. Must contain a method
            `normal`. Default is to use the md.rng which is a
            np.random.Generator

    Attributes:
        accel : ndarray
            Current acceleration for the simulation. Values are given
            in atomic units (Bohr/a.u.^2). Dimensions is (natm, 3) such as

             [[x1, y1, z1],
             [x2, y2, z2],
             [x3, y3, z3]]
    '''

    def __init__(self, method, std_dev=1e-4, rng=md.rng, **kwargs):
        self.std_dev = std_dev
        self.rng = rng
        super().__init__(method, **kwargs)

    def _compute_accel(self):
        '''Given the current geometry, computes the acceleration
        for each atom.'''
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge!')
        noise = self.rng.normal(0, self.std_dev, size=(self.mol.natm, 3))
        return e_tot, -grad / self._masses.reshape(-1, 1) + noise


class NVTBerendson(_Integrator):
    '''Berendsen (constant N, V, T) molecular dynamics

    Args:
        method : lib.GradScanner or rhf.GradientsMixin instance, or
        has nuc_grad_method method.
            Method by which to compute the energy gradients and energies
            in order to propagate the equations of motion. Realistically,
            it can be any callable object such that it returns the energy
            and potential energy gradient when given a mol.

        T      : float
            Target temperature for the NVT Ensemble. Given in K.

        taut   : float
            Time constant for Berendsen temperature coupling.
            Given in atomic units.

    Attributes:
        accel : ndarray
            Current acceleration for the simulation. Values are given
            in atomic units (Bohr/a.u.^2). Dimensions is (natm, 3) such as

             [[x1, y1, z1],
             [x2, y2, z2],
             [x3, y3, z3]]
    '''

    def __init__(self, method, T, taut, **kwargs):
        self.T = T
        self.taut = taut
        self.accel = None
        super().__init__(method, **kwargs)

    def _next(self):
        '''Computes the next frame of the simulation and sets all internal
         variables to this new frame. First computes the new geometry,
         then the next acceleration, and finally the velocity, all according
         to the Velocity Verlet algorithm.

        Returns:
            The next frame of the simulation.
        '''

        # If no acceleration, compute that first, and then go
        # onto the next step
        if self.accel is None:
            next_epot, next_accel = self._compute_accel()

        else:
            self._scale_velocities()
            self.mol.set_geom_(self._next_geometry(), unit='B')
            self.mol.build()
            next_epot, next_accel = self._compute_accel()
            self.veloc = self._next_velocity(next_accel)

        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel

        return _toframe(self)

    def _compute_accel(self):
        '''Given the current geometry, computes the acceleration
        for each atom.'''
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge!')

        a = -1 * grad / self._masses.reshape(-1, 1)
        return e_tot, a

    def _scale_velocities(self):
        '''NVT Berendsen velocity scaling
        v_rescale(t) = v(t) * (1 + ((T_target/T - 1)
                            * (/delta t / taut)))^(0.5)
        '''
        tautscl = self.dt / self.taut
        scl_temp = np.sqrt(1.0 + (self.T / self.temperature() - 1.0) * tautscl)

        # Limit the velocity scaling to reasonable values
        # (taken from ase md/nvtberendson.py)
        if scl_temp > 1.1:
            scl_temp = 1.1
        if scl_temp < 0.9:
            scl_temp = 0.9

        self.veloc = self.veloc * scl_temp
        return

    def _next_geometry(self):
        '''Computes the next geometry using the Velocity Verlet algorithm. The
        necessary equations of motion for the position is
            r(t_i+1) = r(t_i) + /delta t * v(t_i) + 0.5(/delta t)^2 a(t_i)
        '''
        return self.mol.atom_coords() + self.dt * self.veloc + \
            0.5 * (self.dt ** 2) * self.accel

    def _next_velocity(self, next_accel):
        '''Compute the next velocity using the Velocity Verlet algorithm. The
        necessary equations of motion for the velocity is
            v(t_i+1) = v(t_i) + /delta t * 0.5(a(t_i+1) + a(t_i))'''
        return self.veloc + 0.5 * self.dt * (self.accel + next_accel)


class NVTBussi(NVTBerendson):
    '''Bussi-Donadio-Parrinello (CSVR) constant N, V, T molecular dynamics.

    Velocity-Verlet propagation with a global stochastic velocity rescaling
    applied once per step. Compared to Berendsen, CSVR samples the correct
    canonical kinetic-energy distribution rather than just driving the mean
    temperature, so it is the right choice for free-energy / enhanced-sampling
    work where the ensemble matters.

    The kinetic-energy update is a literal transcription of GROMACS'
    `vrescale_resamplekin` (Bussi et al., J. Chem. Phys. 126, 014101 (2007),
    Eq. A7) and is algebraically identical to HOOMD-blue's BussiThermostat.
    The conserved-quantity bookkeeping (thermostat_work / econs) follows the
    same sign convention as GROMACS' temperature-coupling integral.

    Args:
        method : lib.GradScanner, rhf.GradientsBase instance, or object with
            a nuc_grad_method. Returns (e_tot, grad) when called on a mol.

        T : float
            Target temperature in K.

        taut : float
            Thermostat relaxation time, in atomic units of time (same
            convention as NVTBerendson.taut). The decay factor per step is
            exp(-dt / taut). Larger taut == weaker coupling; taut -> inf
            recovers NVE. The taut == 0 limit is the documented CSVR special
            case of *instantaneous* stochastic rescaling (decay factor 0),
            handled here the GROMACS way (see _scale_velocities). taut < 0 is
            unphysical and raises ValueError.

        rng : numpy random Generator, optional
            Source of randomness. Defaults to None, which resolves to md.rng
            *inside __init__* (so md.set_seed() is honoured -- a bare
            `rng=md.rng` default would capture the generator object at class
            definition time and ignore later reseeding). For weighted-ensemble
            (wepy) runs, pass a distinctly seeded per-walker Generator, exactly
            as the Langevin integrators in this module expect -- e.g.

                rng = np.random.default_rng([walker_id, cycle_idx])
                myint = NVTBussi(mf, T=300, taut=413.4, dt=20, rng=rng)

            Cloned/parallel walker segments correlate only if they replay the
            same stream; distinct seeds remove that risk.

    Attributes:
        accel : ndarray (natm, 3)
            Current acceleration in atomic units.

        thermostat_work : float
            Running sum of kinetic energy injected/removed by the thermostat
            (Hartree). The CSVR conserved quantity is
                econs = epot + ekin - thermostat_work
            which should be flat (up to integrator error) over a trajectory --
            a useful sanity check / benchmark diagnostic. This matches the
            quantity GROMACS exposes as the temperature-coupling energy.
    '''

    def __init__(self, method, T, taut, rng=None, **kwargs):
        # Validate before touching anything else so the error is raised at
        # construction without requiring a gradient evaluation.
        if taut < 0:
            raise ValueError(
                'taut (thermostat relaxation time) must be >= 0; got %r' % (taut,))
        # Resolve the RNG default dynamically (see Args note on rng).
        self.rng = md.rng if rng is None else rng # TODO: Use pyscf convention
        self.thermostat_work = 0.0
        super().__init__(method, T, taut, **kwargs)

    def _scale_velocities(self):
        r'''Canonical Sampling through Velocity Rescaling.

        Reproduces Bussi's reference `resamplekin` routine. Given the current
        kinetic energy K and target kinetic energy
            sigma = (Nf/2) k_B T,
        draw a new kinetic energy K' from the canonical distribution and
        rescale all velocities by alpha = sqrt(K'/K):

            K' = K + (1-c) * ( sigma * (chi_{Nf-1} + R1^2) / Nf - K )
                   + 2 * R1 * sqrt( c (1-c) * K * sigma / Nf )

        where c = exp(-dt/taut), R1 ~ N(0,1), and chi_{Nf-1} is a draw from a
        chi-square distribution with (Nf-1) degrees of freedom (== the sum of
        Nf-1 squared standard normals).
        '''
        ekin_old = self.compute_kinetic_energy()

        # Zero kinetic energy: a rescale factor sqrt(K'/K) is undefined.
        # GROMACS handles this by simply not scaling (lambda = 1) and leaving
        # the conserved-quantity integral untouched; we do the same (silent
        # no-op, no thermostat_work accumulated). Initialise velocities from a
        # Maxwell-Boltzmann distribution rather than from rest to avoid this.
        if ekin_old <= 1e-12:
            return

        # Degrees of freedom: use 3*natm to stay consistent with
        # _Integrator.temperature() in this codebase (it does not remove
        # COM / rotational dof). If you subtract those elsewhere, subtract
        # them here too so the target temperature is reproduced.
        dof = 3 * self.mol.natm

        kT = self.T * data.nist.BOLTZMANN / data.nist.HARTREE2J   # Hartree
        sigma = 0.5 * dof * kT                                    # target KE

        # Decay factor, GROMACS style. GROMACS computes the relaxation time in
        # units of steps (taut/dt) and uses exp(-dt/taut) only when that ratio
        # exceeds 0.1, taking c = 0 otherwise. This both implements the
        # documented taut -> 0 instantaneous-resampling limit and guards the
        # division/overflow for very stiff coupling. (taut < 0 is rejected in
        # __init__.)
        if self.taut > 0.1 * self.dt:
            c = np.exp(-self.dt / self.taut)
        else:
            c = 0.0

        r1 = self.rng.standard_normal()
        chi = self.rng.chisquare(dof - 1)        # sum of (dof-1) squared N(0,1)

        ekin_new = (ekin_old
                    + (1.0 - c) * (sigma * (chi + r1 * r1) / dof - ekin_old)
                    + 2.0 * r1 * np.sqrt(c * (1.0 - c) * ekin_old * sigma / dof))

        # Numerical floor; ekin_new is positive with probability 1 but guard
        # against tiny negative round-off.
        ekin_new = max(ekin_new, 0.0)

        alpha = np.sqrt(ekin_new / ekin_old)
        self.veloc = self.veloc * alpha

        # Track the conserved quantity (heat exchanged with the bath).
        self.thermostat_work += (ekin_new - ekin_old)

    # Convenience: the CSVR conserved quantity, useful as a benchmark check.
    @property
    def econs(self):
        if self.epot is None or self.ekin is None:
            return None
        return self.epot + self.ekin - self.thermostat_work


# TODO: Units for friction coeff

# Copyright; Samik Bose and Nathan Emeott (Michigan State University)
# Not implemented in pyscf

class Langevin(_Integrator):
    '''Langevin algorithm

    Args:
        method        : lib.GradScanner or rhf.GradientsBase instance, or
        has nuc_grad_method method.
            Method by which to compute the energy gradients and energies
            in order to propagate the equations of motion. Realistically,
            it can be any callable object such that it returns the energy
            and potential energy gradient when given a mol.

        T             : float
        Temperature of the heat bath for the Langevin thermostat. Given in K.

        friction_coef : float
            Friction coefficient (gamma) for the Langevin thermostat. Default of 1.0.

    Attributes:
        accel : ndarray
            Current acceleration for the simulation. Values are given
            in atomic units (Bohr/a.u.^2). Dimensions is (natm, 3) such as

             [[x1, y1, z1],
             [x2, y2, z2],
             [x3, y3, z3]]
    '''

    def __init__(self, method, T: float, friction_coef: float = 1.0, rng=md.rng, **kwargs):
        self.T: float = T
        self.friction_coef = friction_coef
        self.rng = rng
        self.accel = None
        super().__init__(method, **kwargs)
        self._std_dev = np.sqrt(2.0 * friction_coef * data.nist.BOLTZMANN / data.nist.HARTREE2J * T)

    def _generate_R_noise(self, rng=md.rng):
        '''Generate random noise for the Langevin Thermostat.

        The noise is generated from a normal distribution with mean 0 and 2*gamma*k_B*T variance,
        where gamma is the friction coefficient, k_B is the Boltzmann constant, and T is the temperature of the heat bath.

        Args:
            rng : np.random.Generator
                Random number generator to sample from. Must contain a method
                `normal`. Default is to use the md.rng which is a
                np.random.Generator

        Returns:
            (n, 3) array of random noise for each atom in the system.
        '''
        return rng.normal(0, self._std_dev, size=(self.mol.natm, 3))

    def _next(self):
        '''Computes the next frame of the simulation and sets all internal
         variables to this new frame. First computes the new velocity,
         then the next geometry, and finally the next acceleration, all according
         to the Langevin algorithm.

        Returns:
            The next frame of the simulation.
        '''

        # If no acceleration, compute that first, and then go
        # onto the next step
        if self.accel is None:
            next_epot, next_accel = self._compute_init_accel()

        else:
            next_veloc = self._next_velocity()
            self.mol.set_geom_(self._next_geometry(), unit='B')
            self.mol.build()
            next_epot, next_accel = self._compute_accel(next_veloc)

            self.veloc = next_veloc

        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel

        return _toframe(self)

    def _compute_init_accel(self):
        '''Given the initial geometry, computes the acceleration
        for each atom.'''
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge!')

        a = -1 * grad / self._masses.reshape(-1, 1)
        return e_tot, a

    def _next_velocity(self):
        '''Computes the next velocity using the Langevin algorithm. The
        necessary equations of motion for the next velocity is
            v_i(t + delta_t) = v_i(t) + f_i(t)*delta_t/m_i
        '''
        return self.veloc + self.dt * self.accel

    def _next_geometry(self):
        '''Computes the next geometry using the Langevin algorithm. The
        necessary equations of motion for the next geometry is
            r_i(t + delta_t) = r_i(t) + v_i(t)*delta_t
        '''
        return self.mol.atom_coords() + self.dt * self.veloc

    def _compute_accel(self, next_veloc):
        '''Given the current geometry, computes the acceleration
        for each atom with noise from the Langevin equation
            a(t + delta_t) = (-grad(t + delta_t) + R_i)/m_i - gamma*v(t + delta_t)
        '''
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge!')

        a = (-grad + self._generate_R_noise()) / self._masses.reshape(-1, 1) - \
            self.friction_coef * next_veloc
        return e_tot, a


class LangevinMiddle(_Integrator):
    '''Langevin Middle algorithm with Velocity Verlet components

    Args:
        method        : lib.GradScanner or rhf.GradientsBase instance, or
        has nuc_grad_method method.
            Method by which to compute the energy gradients and energies
            in order to propagate the equations of motion. Realistically,
            it can be any callable object such that it returns the energy
            and potential energy gradient when given a mol.

        T             : float
        Temperature of the heat bath for the Langevin Middle thermostat. Given in K.

        friction_coef : float
            Friction coefficient (gamma) for the Langevin Middle thermostat. Default of 1.0.

    Attributes:
        accel : ndarray
            Current acceleration for the simulation. Values are given
            in atomic units (Bohr/a.u.^2). Dimensions is (natm, 3) such as

             [[x1, y1, z1],
             [x2, y2, z2],
             [x3, y3, z3]]
    '''

    def __init__(self, method, T: float, friction_coef: float = 1.0, rng=md.rng, **kwargs):
        self.T: float = T
        self.friction_coef = friction_coef
        self.rng = rng
        self.accel = None
        super().__init__(method, **kwargs)
        self.alpha = np.exp(-friction_coef * self.dt)
        self.mid_veloc = None

    def _generate_R_noise(self):
        '''Generate random noise for the Langevin Middle Thermostat.

        The noise is generated from a standard normal distribution.

        Args:
            rng : np.random.Generator
                Random number generator to sample from. Must contain a method
                `normal`. Default is to use the md.rng which is a
                np.random.Generator

        Returns:
            (n, 3) array of random noise for each atom in the system.
        '''
        return self.rng.normal(0, 1, size=(self.mol.natm, 3))

    def _next(self):
        '''Computes the next frame of the simulation and sets all internal
         variables to this new frame. First computes the new velocity,
         then the next geometry, and finally the acceleration, all according
         to the Langevin Middle algorithm.

        Returns:
            The next frame of the simulation.
        '''

        # If no acceleration, compute that first, and then go
        # onto the next step
        if self.accel is None:
            next_epot, next_accel = self._compute_accel()
            self.mid_veloc = self.veloc - 0.5 * self.dt * next_accel # Initialize half-step veloc

        else:
            mid_veloc = self._mid_velocity()
            next_veloc = self._next_velocity(mid_veloc)
            self.mol.set_geom_(self._next_geometry(mid_veloc, next_veloc), unit='B')
            self.mol.build()
            next_epot, next_accel = self._compute_accel()

            self.mid_veloc = mid_veloc
            self.veloc = next_veloc

        self.epot = next_epot
        self.ekin = self.compute_kinetic_energy()
        self.accel = next_accel

        return _toframe(self)

    def _compute_accel(self):
        '''Given the current geometry, computes the acceleration
        for each atom.'''
        e_tot, grad = self.scanner(self.mol)
        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge!')

        a = -1 * grad / self._masses.reshape(-1, 1)
        return e_tot, a

    def _mid_velocity(self):
        '''Computes the middle velocity using the Langevin Middle algorithm. The
        necessary equations of motion for the middle velocity is
            v_i(t + delta_t/2) = v_i(t - delta_t/2) + f_i(t)*delta_t/m_i
        '''
        return self.mid_veloc + self.dt * self.accel

    def _next_velocity(self, mid_veloc):
        '''Computes the next velocity using the Langevin Middle algorithm. The
        necessary equations of motion for the next velocity is
            v'_i(t + delta_t/2) = v_i(t + delta_t/2)*alpha + sqrt(k*T*(1-alpha^2)/m)*R
        '''
        return mid_veloc * self.alpha + np.sqrt(
            data.nist.BOLTZMANN / data.nist.HARTREE2J * self.T * (1 - self.alpha ** 2) / self._masses.reshape(-1, 1) 
        ) * self._generate_R_noise()

    def _next_geometry(self, mid_veloc, next_veloc):
        '''Computes the next geometry using the Langevin Middle algorithm. The
        necessary equations of motion for the next geometry is
            r_i(t + delta_t/2) = r_i(t) + v_i(t + delta_t/2)*delta_t/2
            r_i(t + delta_t) = r_i(t + delta_t/2) + v'_i(t + delta_t/2)*delta_t/2
        '''
        mid_geom = self.mol.atom_coords() + 0.5 * self.dt * mid_veloc
        return mid_geom + 0.5 * self.dt * next_veloc
