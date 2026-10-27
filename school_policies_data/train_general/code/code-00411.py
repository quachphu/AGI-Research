        self._check_objects(joints, [], Joint, 'Joints', 'joints')
        self._joints = []
        self.add_joints(*joints)

    @property
    def loads(self):
        """Tuple of loads that have been applied on the system."""
        return tuple(self._loads)

    @loads.setter
    @_reset_eom_method
    def loads(self, loads):
        loads = self._objects_to_list(loads)
        self._loads = [_parse_load(load) for load in loads]

    @property
    def actuators(self):
        """Tuple of actuators present in the system."""
        return tuple(self._actuators)

    @actuators.setter
    @_reset_eom_method
    def actuators(self, actuators):
        actuators = self._objects_to_list(actuators)
        self._check_objects(actuators, [], ActuatorBase, 'Actuators',
                            'actuators')
        self._actuators = actuators

    @property
    def q(self):
        """Matrix of all the generalized coordinates with the independent
        stacked upon the dependent."""
        return self._q_ind.col_join(self._q_dep)

    @property
    def u(self):
        """Matrix of all the generalized speeds with the independent stacked
        upon the dependent."""
        return self._u_ind.col_join(self._u_dep)

    @property
    def q_ind(self):
        """Matrix of the independent generalized coordinates."""
        return self._q_ind

    @q_ind.setter
    @_reset_eom_method
    def q_ind(self, q_ind):
        self._q_ind, self._q_dep = self._parse_coordinates(
            self._objects_to_list(q_ind), True, [], self.q_dep, 'coordinates')

    @property
    def q_dep(self):
        """Matrix of the dependent generalized coordinates."""
        return self._q_dep

    @q_dep.setter
    @_reset_eom_method
    def q_dep(self, q_dep):
        self._q_ind, self._q_dep = self._parse_coordinates(
            self._objects_to_list(q_dep), False, self.q_ind, [], 'coordinates')

    @property
    def u_ind(self):
        """Matrix of the independent generalized speeds."""
        return self._u_ind

    @u_ind.setter
    @_reset_eom_method
    def u_ind(self, u_ind):
        self._u_ind, self._u_dep = self._parse_coordinates(
            self._objects_to_list(u_ind), True, [], self.u_dep, 'speeds')

    @property
    def u_dep(self):
        """Matrix of the dependent generalized speeds."""
        return self._u_dep

    @u_dep.setter
    @_reset_eom_method
    def u_dep(self, u_dep):
        self._u_ind, self._u_dep = self._parse_coordinates(
            self._objects_to_list(u_dep), False, self.u_ind, [], 'speeds')

    @property
    def u_aux(self):
        """Matrix of auxiliary generalized speeds."""
        return self._u_aux

    @u_aux.setter
    @_reset_eom_method
    def u_aux(self, u_aux):
        self._u_aux = self._parse_coordinates(
            self._objects_to_list(u_aux), True, [], [], 'u_auxiliary')[0]

    @property
    def kdes(self):
        """Kinematical differential equations as expressions equated to the zero
        matrix. These equations describe the coupling between the generalized
        coordinates and the generalized speeds."""
        return self._kdes

    @kdes.setter
    @_reset_eom_method
    def kdes(self, kdes):
        kdes = self._objects_to_list(kdes)
        self._kdes = self._parse_expressions(
            kdes, [], 'kinematic differential equations')

    @property
    def holonomic_constraints(self):
        """Matrix with the holonomic constraints as expressions equated to the
        zero matrix."""
        return self._hol_coneqs

    @holonomic_constraints.setter
    @_reset_eom_method
    def holonomic_constraints(self, constraints):
        constraints = self._objects_to_list(constraints)
        self._hol_coneqs = self._parse_expressions(
            constraints, [], 'holonomic constraints')

    @property
    def nonholonomic_constraints(self):
        """Matrix with the nonholonomic constraints as expressions equated to
        the zero matrix."""
        return self._nonhol_coneqs

    @nonholonomic_constraints.setter
    @_reset_eom_method
    def nonholonomic_constraints(self, constraints):
        constraints = self._objects_to_list(constraints)
        self._nonhol_coneqs = self._parse_expressions(
            constraints, [], 'nonholonomic constraints')

    @property
    def velocity_constraints(self):
        """Matrix with the velocity constraints as expressions equated to the
        zero matrix. The velocity constraints are by default derived from the
        holonomic and nonholonomic constraints unless they are explicitly set.
        """
        if self._vel_constrs is None:
            return self.holonomic_constraints.diff(dynamicsymbols._t).col_join(
                self.nonholonomic_constraints)
        return self._vel_constrs

    @velocity_constraints.setter
    @_reset_eom_method
    def velocity_constraints(self, constraints):
        if constraints is None:
            self._vel_constrs = None
            return
        constraints = self._objects_to_list(constraints)
        self._vel_constrs = self._parse_expressions(
            constraints, [], 'velocity constraints')

    @property
    def eom_method(self):
        """Backend for forming the equations of motion."""
        return self._eom_method

    @staticmethod
    def _objects_to_list(lst):
        """Helper to convert passed objects to a list."""
        if not iterable(lst):  # Only one object
            return [lst]
        return list(lst[:])  # converts Matrix and tuple to flattened list

    @staticmethod
    def _check_objects(objects, obj_lst, expected_type, obj_name, type_name):
        """Helper to check the objects that are being added to the system.

        Explanation
        ===========
        This method checks that the objects that are being added to the system
        are of the correct type and have not already been added. If any of the
        objects are not of the correct type or have already been added, then
        an error is raised.

        Parameters
        ==========
        objects : iterable
            The objects that would be added to the system.
        obj_lst : list
            The list of objects that are already in the system.
        expected_type : type
            The type that the objects should be.
        obj_name : str
            The name of the category of objects. This string is used to
            formulate the error message for the user.
        type_name : str
            The name of the type that the objects should be. This string is used
            to formulate the error message for the user.

        """
        seen = set(obj_lst)
        duplicates = set()
        wrong_types = set()
        for obj in objects:
            if not isinstance(obj, expected_type):
                wrong_types.add(obj)
            if obj in seen:
                duplicates.add(obj)
            else:
                seen.add(obj)
        if wrong_types:
            raise TypeError(f'{obj_name} {wrong_types} are not {type_name}.')
        if duplicates:
            raise ValueError(f'{obj_name} {duplicates} have already been added '
                             f'to the system.')

    def _parse_coordinates(self, new_coords, independent, old_coords_ind,
                           old_coords_dep, coord_type='coordinates'):
        """Helper to parse coordinates and speeds."""
        # Construct lists of the independent and dependent coordinates
        coords_ind, coords_dep = old_coords_ind[:], old_coords_dep[:]
        if not iterable(independent):
            independent = [independent] * len(new_coords)
        for coord, indep in zip(new_coords, independent):
            if indep:
                coords_ind.append(coord)
            else:
                coords_dep.append(coord)
        # Check types and duplicates
        current = {'coordinates': self.q_ind[:] + self.q_dep[:],
                   'speeds': self.u_ind[:] + self.u_dep[:],
                   'u_auxiliary': self._u_aux[:],
                   coord_type: coords_ind + coords_dep}
        _validate_coordinates(**current)
        return (ImmutableMatrix(1, len(coords_ind), coords_ind).T,
                ImmutableMatrix(1, len(coords_dep), coords_dep).T)

    @staticmethod
    def _parse_expressions(new_expressions, old_expressions, name,
                           check_negatives=False):
        """Helper to parse expressions like constraints."""
        old_expressions = old_expressions[:]
        new_expressions = list(new_expressions)  # Converts a possible tuple
        if check_negatives:
            check_exprs = old_expressions + [-expr for expr in old_expressions]
        else:
            check_exprs = old_expressions
        System._check_objects(new_expressions, check_exprs, Basic, name,
                              'expressions')
        for expr in new_expressions:
            if expr == 0:
                raise ValueError(f'Parsed {name} are zero.')
        return ImmutableMatrix(1, len(old_expressions) + len(new_expressions),
                               old_expressions + new_expressions).T

    @_reset_eom_method
    def add_coordinates(self, *coordinates, independent=True):
        """Add generalized coordinate(s) to the system.

        Parameters
        ==========

        *coordinates : dynamicsymbols
            One or more generalized coordinates to be added to the system.
        independent : bool or list of bool, optional
            Boolean whether a coordinate is dependent or independent. The
            default is True, so the coordinates are added as independent by
            default.

        """
        self._q_ind, self._q_dep = self._parse_coordinates(
            coordinates, independent, self.q_ind, self.q_dep, 'coordinates')

    @_reset_eom_method
    def add_speeds(self, *speeds, independent=True):
        """Add generalized speed(s) to the system.

        Parameters
        ==========

        *speeds : dynamicsymbols
            One or more generalized speeds to be added to the system.
        independent : bool or list of bool, optional
            Boolean whether a speed is dependent or independent. The default is
            True, so the speeds are added as independent by default.

        """
        self._u_ind, self._u_dep = self._parse_coordinates(
            speeds, independent, self.u_ind, self.u_dep, 'speeds')

    @_reset_eom_method
    def add_auxiliary_speeds(self, *speeds):
        """Add auxiliary speed(s) to the system.

        Parameters
        ==========

        *speeds : dynamicsymbols
            One or more auxiliary speeds to be added to the system.

        """
        self._u_aux = self._parse_coordinates(
            speeds, True, self._u_aux, [], 'u_auxiliary')[0]

    @_reset_eom_method
    def add_kdes(self, *kdes):
        """Add kinematic differential equation(s) to the system.

        Parameters
        ==========

        *kdes : Expr
            One or more kinematic differential equations.

        """
        self._kdes = self._parse_expressions(
            kdes, self.kdes, 'kinematic differential equations',
            check_negatives=True)

    @_reset_eom_method
    def add_holonomic_constraints(self, *constraints):
        """Add holonomic constraint(s) to the system.

        Parameters
        ==========

        *constraints : Expr
            One or more holonomic constraints, which are expressions that should
            be zero.

        """
        self._hol_coneqs = self._parse_expressions(
            constraints, self._hol_coneqs, 'holonomic constraints',
            check_negatives=True)

    @_reset_eom_method
    def add_nonholonomic_constraints(self, *constraints):
        """Add nonholonomic constraint(s) to the system.

        Parameters
        ==========

        *constraints : Expr
            One or more nonholonomic constraints, which are expressions that
            should be zero.

        """
        self._nonhol_coneqs = self._parse_expressions(
            constraints, self._nonhol_coneqs, 'nonholonomic constraints',
            check_negatives=True)

    @_reset_eom_method
    def add_bodies(self, *bodies):
        """Add body(ies) to the system.

        Parameters
        ==========

        bodies : Particle or RigidBody
            One or more bodies.

        """
        self._check_objects(bodies, self.bodies, BodyBase, 'Bodies', 'bodies')
        self._bodies.extend(bodies)

    @_reset_eom_method
    def add_loads(self, *loads):
        """Add load(s) to the system.

        Parameters
        ==========

        *loads : Force or Torque
            One or more loads.

        """
        loads = [_parse_load(load) for load in loads]  # Checks the loads
        self._loads.extend(loads)

    @_reset_eom_method
    def apply_uniform_gravity(self, acceleration):
        """Apply uniform gravity to all bodies in the system by adding loads.

        Parameters
        ==========

        acceleration : Vector
            The acceleration due to gravity.

        """
        self.add_loads(*gravity(acceleration, *self.bodies))

    @_reset_eom_method
    def add_actuators(self, *actuators):
        """Add actuator(s) to the system.

        Parameters
        ==========

        *actuators : subclass of ActuatorBase
            One or more actuators.

        """
        self._check_objects(actuators, self.actuators, ActuatorBase,
                            'Actuators', 'actuators')
        self._actuators.extend(actuators)

    @_reset_eom_method
    def add_joints(self, *joints):
        """Add joint(s) to the system.

        Explanation
        ===========

        This methods adds one or more joints to the system including its
        associated objects, i.e. generalized coordinates, generalized speeds,
        kinematic differential equations and the bodies.

        Parameters
        ==========

        *joints : subclass of Joint
            One or more joints.

        Notes
        =====

        For the generalized coordinates, generalized speeds and bodies it is
        checked whether they are already known by the system instance. If they
        are, then they are not added. The kinematic differential equations are
        however always added to the system, so you should not also manually add
        those on beforehand.

        """
        self._check_objects(joints, self.joints, Joint, 'Joints', 'joints')
        self._joints.extend(joints)
        coordinates, speeds, kdes, bodies = (OrderedSet() for _ in range(4))
        for joint in joints:
            coordinates.update(joint.coordinates)
            speeds.update(joint.speeds)
            kdes.update(joint.kdes)
            bodies.update((joint.parent, joint.child))
        coordinates = coordinates.difference(self.q)
        speeds = speeds.difference(self.u)
        kdes = kdes.difference(self.kdes[:] + (-self.kdes)[:])
        bodies = bodies.difference(self.bodies)
        self.add_coordinates(*tuple(coordinates))
        self.add_speeds(*tuple(speeds))
        self.add_kdes(*(kde for kde in tuple(kdes) if not kde == 0))
        self.add_bodies(*tuple(bodies))

    def get_body(self, name):
        """Retrieve a body from the system by name.

        Parameters
        ==========

        name : str
            The name of the body to retrieve.

        Returns
        =======

        RigidBody or Particle
            The body with the given name, or None if no such body exists.

        """
        for body in self._bodies:
            if body.name == name:
                return body

    def get_joint(self, name):
        """Retrieve a joint from the system by name.

        Parameters
        ==========

        name : str
            The name of the joint to retrieve.

        Returns
        =======

        subclass of Joint
            The joint with the given name, or None if no such joint exists.

        """
        for joint in self._joints:
            if joint.name == name:
                return joint

    def _form_eoms(self):
        return self.form_eoms()

    def form_eoms(self, eom_method=KanesMethod, **kwargs):
        """Form the equations of motion of the system.

        Parameters
        ==========

        eom_method : subclass of KanesMethod or LagrangesMethod
            Backend class to be used for forming the equations of motion. The
            default is ``KanesMethod``.

        Returns
        ========

        ImmutableMatrix
            Vector of equations of motions.

        Examples
        ========

        This is a simple example for a one degree of freedom translational
        spring-mass-damper.

        >>> from sympy import S, symbols
        >>> from sympy.physics.mechanics import (
        ...     LagrangesMethod, dynamicsymbols, PrismaticJoint, Particle,
        ...     RigidBody, System)
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> wall = RigidBody('W')
        >>> system = System.from_newtonian(wall)
        >>> bob = Particle('P', mass=m)
        >>> bob.potential_energy = S.Half * k * q**2
        >>> system.add_joints(PrismaticJoint('J', wall, bob, q, qd))
        >>> system.add_loads((bob.masscenter, b * qd * system.x))
        >>> system.form_eoms(LagrangesMethod)
        Matrix([[-b*Derivative(q(t), t) + k*q(t) + m*Derivative(q(t), (t, 2))]])

        We can also solve for the states using the 'rhs' method.

        >>> system.rhs()
        Matrix([
        [               Derivative(q(t), t)],
        [(b*Derivative(q(t), t) - k*q(t))/m]])

        """
        # KanesMethod does not accept empty iterables
        loads = self.loads + tuple(
            load for act in self.actuators for load in act.to_loads())
        loads = loads if loads else None
        if issubclass(eom_method, KanesMethod):
            disallowed_kwargs = {
                "frame", "q_ind", "u_ind", "kd_eqs", "q_dependent",
                "u_dependent", "u_auxiliary", "configuration_constraints",
                "velocity_constraints", "forcelist", "bodies"}
            wrong_kwargs = disallowed_kwargs.intersection(kwargs)
            if wrong_kwargs:
                raise ValueError(
                    f"The following keyword arguments are not allowed to be "
                    f"overwritten in {eom_method.__name__}: {wrong_kwargs}.")
            kwargs = {"frame": self.frame, "q_ind": self.q_ind,
                      "u_ind": self.u_ind, "kd_eqs": self.kdes,
                      "q_dependent": self.q_dep, "u_dependent": self.u_dep,
                      "configuration_constraints": self.holonomic_constraints,
                      "velocity_constraints": self.velocity_constraints,
                      "u_auxiliary": self.u_aux,
                      "forcelist": loads, "bodies": self.bodies,
                      "explicit_kinematics": False, **kwargs}
            self._eom_method = eom_method(**kwargs)
        elif issubclass(eom_method, LagrangesMethod):
            disallowed_kwargs = {
                "frame", "qs", "forcelist", "bodies", "hol_coneqs",
                "nonhol_coneqs", "Lagrangian"}
            wrong_kwargs = disallowed_kwargs.intersection(kwargs)
            if wrong_kwargs:
                raise ValueError(
                    f"The following keyword arguments are not allowed to be "
                    f"overwritten in {eom_method.__name__}: {wrong_kwargs}.")
            kwargs = {"frame": self.frame, "qs": self.q, "forcelist": loads,
                      "bodies": self.bodies,
                      "hol_coneqs": self.holonomic_constraints,
                      "nonhol_coneqs": self.nonholonomic_constraints, **kwargs}
            if "Lagrangian" not in kwargs:
                kwargs["Lagrangian"] = Lagrangian(kwargs["frame"],
                                                  *kwargs["bodies"])
            self._eom_method = eom_method(**kwargs)
        else:
            raise NotImplementedError(f'{eom_method} has not been implemented.')
        return self.eom_method._form_eoms()

    def rhs(self, inv_method=None):
        """Compute the equations of motion in the explicit form.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        Returns
        ========

        ImmutableMatrix
            Equations of motion in the explicit form.

        See Also
        ========

        sympy.physics.mechanics.kane.KanesMethod.rhs:
            KanesMethod's ``rhs`` function.
        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:
            LagrangesMethod's ``rhs`` function.

        """
        return self.eom_method.rhs(inv_method=inv_method)

    @property
    def mass_matrix(self):
        r"""The mass matrix of the system.

        Explanation
        ===========

        The mass matrix $M_d$ and the forcing vector $f_d$ of a system describe
        the system's dynamics according to the following equations:

        .. math::
            M_d \dot{u} = f_d

        where $\dot{u}$ is the time derivative of the generalized speeds.

        """
        return self.eom_method.mass_matrix

    @property
    def mass_matrix_full(self):
        r"""The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form.

        Explanation
        ===========

        The full mass matrix $M_m$ and the full forcing vector $f_m$ of a system
        describe the dynamics and kinematics according to the following
        equation:

        .. math::
            M_m \dot{x} = f_m

        where $x$ is the state vector stacking $q$ and $u$.

        """
        return self.eom_method.mass_matrix_full

    @property
    def forcing(self):
        """The forcing vector of the system."""
        return self.eom_method.forcing

    @property
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        return self.eom_method.forcing_full

    def validate_system(self, eom_method=KanesMethod, check_duplicates=False):
        """Validates the system using some basic checks.

        Explanation
        ===========

        This method validates the system based on the following checks:

        - The number of dependent generalized coordinates should equal the
          number of holonomic constraints.
        - All generalized coordinates defined by the joints should also be known
          to the system.
        - If ``KanesMethod`` is used as a ``eom_method``:
            - All generalized speeds and kinematic differential equations
              defined by the joints should also be known to the system.
            - The number of dependent generalized speeds should equal the number
              of velocity constraints.
            - The number of generalized coordinates should be less than or equal
              to the number of generalized speeds.
            - The number of generalized coordinates should equal the number of
              kinematic differential equations.
        - If ``LagrangesMethod`` is used as ``eom_method``:
            - There should not be any generalized speeds that are not
              derivatives of the generalized coordinates (this includes the
              generalized speeds defined by the joints).

        Parameters
        ==========

        eom_method : subclass of KanesMethod or LagrangesMethod
            Backend class that will be used for forming the equations of motion.
            There are different checks for the different backends. The default
            is ``KanesMethod``.
        check_duplicates : bool
            Boolean whether the system should be checked for duplicate
            definitions. The default is False, because duplicates are already
            checked when adding objects to the system.

        Notes
        =====

        This method is not guaranteed to be backwards compatible as it may
        improve over time. The method can become both more and less strict in
        certain areas. However a well-defined system should always pass all
        these tests.

        """
        msgs = []
        # Save some data in variables
        n_hc = self.holonomic_constraints.shape[0]
        n_vc = self.velocity_constraints.shape[0]
        n_q_dep, n_u_dep = self.q_dep.shape[0], self.u_dep.shape[0]
        q_set, u_set = set(self.q), set(self.u)
        n_q, n_u = len(q_set), len(u_set)
        # Check number of holonomic constraints
        if n_q_dep != n_hc:
            msgs.append(filldedent(f"""
            The number of dependent generalized coordinates {n_q_dep} should be
            equal to the number of holonomic constraints {n_hc}."""))
        # Check if all joint coordinates and speeds are present
        missing_q = set()
        for joint in self.joints:
            missing_q.update(set(joint.coordinates).difference(q_set))
        if missing_q:
            msgs.append(filldedent(f"""
            The generalized coordinates {missing_q} used in joints are not added
            to the system."""))
        # Method dependent checks
        if issubclass(eom_method, KanesMethod):
            n_kdes = len(self.kdes)
            missing_kdes, missing_u = set(), set()
            for joint in self.joints:
                missing_u.update(set(joint.speeds).difference(u_set))
                missing_kdes.update(set(joint.kdes).difference(
                    self.kdes[:] + (-self.kdes)[:]))
            if missing_u:
                msgs.append(filldedent(f"""
                The generalized speeds {missing_u} used in joints are not added
                to the system."""))
            if missing_kdes:
                msgs.append(filldedent(f"""
                The kinematic differential equations {missing_kdes} used in
                joints are not added to the system."""))
            if n_u_dep != n_vc:
                msgs.append(filldedent(f"""
                The number of dependent generalized speeds {n_u_dep} should be
                equal to the number of velocity constraints {n_vc}."""))
            if n_q > n_u:
                msgs.append(filldedent(f"""
                The number of generalized coordinates {n_q} should be less than
                or equal to the number of generalized speeds {n_u}."""))
            if n_u != n_kdes:
                msgs.append(filldedent(f"""
                The number of generalized speeds {n_u} should be equal to the
                number of kinematic differential equations {n_kdes}."""))
        elif issubclass(eom_method, LagrangesMethod):
            not_qdots = set(self.u).difference(self.q.diff(dynamicsymbols._t))
            for joint in self.joints:
                not_qdots.update(set(
                    joint.speeds).difference(self.q.diff(dynamicsymbols._t)))
            if not_qdots:
                msgs.append(filldedent(f"""
                The generalized speeds {not_qdots} are not supported by this
                method. Only derivatives of the generalized coordinates are
                supported. If these symbols are used in your expressions, then
                this will result in wrong equations of motion."""))
            if self.u_aux:
                msgs.append(filldedent(f"""
                This method does not support auxiliary speeds. If these symbols
                are used in your expressions, then this will result in wrong
                equations of motion. The auxiliary speeds are {self.u_aux}."""))
        else:
            raise NotImplementedError(f'{eom_method} has not been implemented.')
        if check_duplicates:  # Should be redundant
            duplicates_to_check = [('generalized coordinates', self.q),
                                   ('generalized speeds', self.u),
                                   ('auxiliary speeds', self.u_aux),
                                   ('bodies', self.bodies),
                                   ('joints', self.joints)]
            for name, lst in duplicates_to_check:
                seen = set()
                duplicates = {x for x in lst if x in seen or seen.add(x)}
                if duplicates:
                    msgs.append(filldedent(f"""
                    The {name} {duplicates} exist multiple times within the
                    system."""))
        if msgs:
            raise ValueError('\n'.join(msgs))


class SymbolicSystem:
    """SymbolicSystem is a class that contains all the information about a
    system in a symbolic format such as the equations of motions and the bodies
    and loads in the system.

    There are three ways that the equations of motion can be described for
    Symbolic System:


        [1] Explicit form where the kinematics and dynamics are combined
            x' = F_1(x, t, r, p)

        [2] Implicit form where the kinematics and dynamics are combined
            M_2(x, p) x' = F_2(x, t, r, p)

        [3] Implicit form where the kinematics and dynamics are separate
            M_3(q, p) u' = F_3(q, u, t, r, p)
            q' = G(q, u, t, r, p)

    where

    x : states, e.g. [q, u]
    t : time
    r : specified (exogenous) inputs
    p : constants
    q : generalized coordinates
    u : generalized speeds
    F_1 : right hand side of the combined equations in explicit form
    F_2 : right hand side of the combined equations in implicit form
    F_3 : right hand side of the dynamical equations in implicit form
    M_2 : mass matrix of the combined equations in implicit form
    M_3 : mass matrix of the dynamical equations in implicit form
    G : right hand side of the kinematical differential equations

        Parameters
        ==========

        coord_states : ordered iterable of functions of time
            This input will either be a collection of the coordinates or states
            of the system depending on whether or not the speeds are also
            given. If speeds are specified this input will be assumed to
            be the coordinates otherwise this input will be assumed to
            be the states.

        right_hand_side : Matrix
            This variable is the right hand side of the equations of motion in
            any of the forms. The specific form will be assumed depending on
            whether a mass matrix or coordinate derivatives are given.

        speeds : ordered iterable of functions of time, optional
            This is a collection of the generalized speeds of the system. If
            given it will be assumed that the first argument (coord_states)
            will represent the generalized coordinates of the system.

        mass_matrix : Matrix, optional
            The matrix of the implicit forms of the equations of motion (forms
            [2] and [3]). The distinction between the forms is determined by
            whether or not the coordinate derivatives are passed in. If
            they are given form [3] will be assumed otherwise form [2] is
            assumed.

        coordinate_derivatives : Matrix, optional
            The right hand side of the kinematical equations in explicit form.
            If given it will be assumed that the equations of motion are being
            entered in form [3].

        alg_con : Iterable, optional
            The indexes of the rows in the equations of motion that contain
            algebraic constraints instead of differential equations. If the
            equations are input in form [3], it will be assumed the indexes are
            referencing the mass_matrix/right_hand_side combination and not the
            coordinate_derivatives.

        output_eqns : Dictionary, optional
            Any output equations that are desired to be tracked are stored in a
            dictionary where the key corresponds to the name given for the
            specific equation and the value is the equation itself in symbolic
            form

        coord_idxs : Iterable, optional
            If coord_states corresponds to the states rather than the
            coordinates this variable will tell SymbolicSystem which indexes of
            the states correspond to generalized coordinates.

        speed_idxs : Iterable, optional
            If coord_states corresponds to the states rather than the
            coordinates this variable will tell SymbolicSystem which indexes of
            the states correspond to generalized speeds.

        bodies : iterable of Body/Rigidbody objects, optional
            Iterable containing the bodies of the system

        loads : iterable of load instances (described below), optional
            Iterable containing the loads of the system where forces are given
            by (point of application, force vector) and torques are given by
            (reference frame acting upon, torque vector). Ex [(point, force),
            (ref_frame, torque)]

    Attributes
    ==========

    coordinates : Matrix, shape(n, 1)
        This is a matrix containing the generalized coordinates of the system

    speeds : Matrix, shape(m, 1)
        This is a matrix containing the generalized speeds of the system

    states : Matrix, shape(o, 1)
        This is a matrix containing the state variables of the system

    alg_con : List
        This list contains the indices of the algebraic constraints in the
        combined equations of motion. The presence of these constraints
        requires that a DAE solver be used instead of an ODE solver.
        If the system is given in form [3] the alg_con variable will be
        adjusted such that it is a representation of the combined kinematics
        and dynamics thus make sure it always matches the mass matrix
        entered.

    dyn_implicit_mat : Matrix, shape(m, m)
        This is the M matrix in form [3] of the equations of motion (the mass
        matrix or generalized inertia matrix of the dynamical equations of
        motion in implicit form).

    dyn_implicit_rhs : Matrix, shape(m, 1)
        This is the F vector in form [3] of the equations of motion (the right
        hand side of the dynamical equations of motion in implicit form).

    comb_implicit_mat : Matrix, shape(o, o)
        This is the M matrix in form [2] of the equations of motion.
        This matrix contains a block diagonal structure where the top
        left block (the first rows) represent the matrix in the
        implicit form of the kinematical equations and the bottom right
        block (the last rows) represent the matrix in the implicit form
        of the dynamical equations.

    comb_implicit_rhs : Matrix, shape(o, 1)
        This is the F vector in form [2] of the equations of motion. The top
        part of the vector represents the right hand side of the implicit form
        of the kinemaical equations and the bottom of the vector represents the
        right hand side of the implicit form of the dynamical equations of
        motion.

    comb_explicit_rhs : Matrix, shape(o, 1)
        This vector represents the right hand side of the combined equations of
        motion in explicit form (form [1] from above).

    kin_explicit_rhs : Matrix, shape(m, 1)
        This is the right hand side of the explicit form of the kinematical
        equations of motion as can be seen in form [3] (the G matrix).

    output_eqns : Dictionary
        If output equations were given they are stored in a dictionary where
        the key corresponds to the name given for the specific equation and
        the value is the equation itself in symbolic form

    bodies : Tuple
        If the bodies in the system were given they are stored in a tuple for
        future access

    loads : Tuple
        If the loads in the system were given they are stored in a tuple for
        future access. This includes forces and torques where forces are given
        by (point of application, force vector) and torques are given by
        (reference frame acted upon, torque vector).

    Example
    =======

    As a simple example, the dynamics of a simple pendulum will be input into a
    SymbolicSystem object manually. First some imports will be needed and then
    symbols will be set up for the length of the pendulum (l), mass at the end
    of the pendulum (m), and a constant for gravity (g). ::

        >>> from sympy import Matrix, sin, symbols
        >>> from sympy.physics.mechanics import dynamicsymbols, SymbolicSystem
        >>> l, m, g = symbols('l m g')

    The system will be defined by an angle of theta from the vertical and a
    generalized speed of omega will be used where omega = theta_dot. ::

        >>> theta, omega = dynamicsymbols('theta omega')

    Now the equations of motion are ready to be formed and passed to the
    SymbolicSystem object. ::

        >>> kin_explicit_rhs = Matrix([omega])
        >>> dyn_implicit_mat = Matrix([l**2 * m])
        >>> dyn_implicit_rhs = Matrix([-g * l * m * sin(theta)])
        >>> symsystem = SymbolicSystem([theta], dyn_implicit_rhs, [omega],
        ...                            dyn_implicit_mat)

    Notes
    =====

    m : number of generalized speeds
    n : number of generalized coordinates
    o : number of states

    """

    def __init__(self, coord_states, right_hand_side, speeds=None,
                 mass_matrix=None, coordinate_derivatives=None, alg_con=None,
                 output_eqns={}, coord_idxs=None, speed_idxs=None, bodies=None,
                 loads=None):
        """Initializes a SymbolicSystem object"""

        # Extract information on speeds, coordinates and states
        if speeds is None:
            self._states = Matrix(coord_states)

            if coord_idxs is None:
                self._coordinates = None
            else:
                coords = [coord_states[i] for i in coord_idxs]
                self._coordinates = Matrix(coords)

            if speed_idxs is None:
                self._speeds = None
            else:
                speeds_inter = [coord_states[i] for i in speed_idxs]
                self._speeds = Matrix(speeds_inter)
        else:
            self._coordinates = Matrix(coord_states)
            self._speeds = Matrix(speeds)
            self._states = self._coordinates.col_join(self._speeds)

        # Extract equations of motion form
        if coordinate_derivatives is not None:
            self._kin_explicit_rhs = coordinate_derivatives
            self._dyn_implicit_rhs = right_hand_side
            self._dyn_implicit_mat = mass_matrix
            self._comb_implicit_rhs = None
            self._comb_implicit_mat = None
            self._comb_explicit_rhs = None
        elif mass_matrix is not None:
            self._kin_explicit_rhs = None
            self._dyn_implicit_rhs = None
            self._dyn_implicit_mat = None
            self._comb_implicit_rhs = right_hand_side
            self._comb_implicit_mat = mass_matrix
            self._comb_explicit_rhs = None
        else:
            self._kin_explicit_rhs = None
            self._dyn_implicit_rhs = None
            self._dyn_implicit_mat = None
            self._comb_implicit_rhs = None
            self._comb_implicit_mat = None
            self._comb_explicit_rhs = right_hand_side

        # Set the remainder of the inputs as instance attributes
        if alg_con is not None and coordinate_derivatives is not None:
            alg_con = [i + len(coordinate_derivatives) for i in alg_con]
        self._alg_con = alg_con
        self.output_eqns = output_eqns

        # Change the body and loads iterables to tuples if they are not tuples
        # already
        if not isinstance(bodies, tuple) and bodies is not None:
            bodies = tuple(bodies)
        if not isinstance(loads, tuple) and loads is not None:
            loads = tuple(loads)
        self._bodies = bodies
        self._loads = loads

    @property
    def coordinates(self):
        """Returns the column matrix of the generalized coordinates"""
        if self._coordinates is None:
            raise AttributeError("The coordinates were not specified.")
        else:
            return self._coordinates

    @property
    def speeds(self):
        """Returns the column matrix of generalized speeds"""
        if self._speeds is None:
            raise AttributeError("The speeds were not specified.")
        else:
            return self._speeds

    @property
    def states(self):
        """Returns the column matrix of the state variables"""
        return self._states

    @property
    def alg_con(self):
        """Returns a list with the indices of the rows containing algebraic
        constraints in the combined form of the equations of motion"""
        return self._alg_con

    @property
    def dyn_implicit_mat(self):
        """Returns the matrix, M, corresponding to the dynamic equations in
        implicit form, M x' = F, where the kinematical equations are not
        included"""
        if self._dyn_implicit_mat is None:
            raise AttributeError("dyn_implicit_mat is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            return self._dyn_implicit_mat

    @property
    def dyn_implicit_rhs(self):
        """Returns the column matrix, F, corresponding to the dynamic equations
        in implicit form, M x' = F, where the kinematical equations are not
        included"""
        if self._dyn_implicit_rhs is None:
            raise AttributeError("dyn_implicit_rhs is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            return self._dyn_implicit_rhs

    @property
    def comb_implicit_mat(self):
        """Returns the matrix, M, corresponding to the equations of motion in
        implicit form (form [2]), M x' = F, where the kinematical equations are
        included"""
        if self._comb_implicit_mat is None:
            if self._dyn_implicit_mat is not None:
                num_kin_eqns = len(self._kin_explicit_rhs)
                num_dyn_eqns = len(self._dyn_implicit_rhs)
                zeros1 = zeros(num_kin_eqns, num_dyn_eqns)
                zeros2 = zeros(num_dyn_eqns, num_kin_eqns)
                inter1 = eye(num_kin_eqns).row_join(zeros1)
                inter2 = zeros2.row_join(self._dyn_implicit_mat)
                self._comb_implicit_mat = inter1.col_join(inter2)
                return self._comb_implicit_mat
            else:
                raise AttributeError("comb_implicit_mat is not specified for "
                                     "equations of motion form [1].")
        else:
            return self._comb_implicit_mat

    @property
    def comb_implicit_rhs(self):
        """Returns the column matrix, F, corresponding to the equations of
        motion in implicit form (form [2]), M x' = F, where the kinematical
        equations are included"""
        if self._comb_implicit_rhs is None:
            if self._dyn_implicit_rhs is not None:
                kin_inter = self._kin_explicit_rhs
                dyn_inter = self._dyn_implicit_rhs
                self._comb_implicit_rhs = kin_inter.col_join(dyn_inter)
                return self._comb_implicit_rhs
            else:
                raise AttributeError("comb_implicit_mat is not specified for "
                                     "equations of motion in form [1].")
        else:
            return self._comb_implicit_rhs

    def compute_explicit_form(self):
        """If the explicit right hand side of the combined equations of motion
        is to provided upon initialization, this method will calculate it. This
        calculation can potentially take awhile to compute."""
        if self._comb_explicit_rhs is not None:
            raise AttributeError("comb_explicit_rhs is already formed.")

        inter1 = getattr(self, 'kin_explicit_rhs', None)
        if inter1 is not None:
            inter2 = self._dyn_implicit_mat.LUsolve(self._dyn_implicit_rhs)
            out = inter1.col_join(inter2)
        else:
            out = self._comb_implicit_mat.LUsolve(self._comb_implicit_rhs)

        self._comb_explicit_rhs = out

    @property
    def comb_explicit_rhs(self):
        """Returns the right hand side of the equations of motion in explicit
        form, x' = F, where the kinematical equations are included"""
        if self._comb_explicit_rhs is None:
            raise AttributeError("Please run .combute_explicit_form before "
                                 "attempting to access comb_explicit_rhs.")
        else:
            return self._comb_explicit_rhs

    @property
    def kin_explicit_rhs(self):
        """Returns the right hand side of the kinematical equations in explicit
        form, q' = G"""
        if self._kin_explicit_rhs is None:
            raise AttributeError("kin_explicit_rhs is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            return self._kin_explicit_rhs

    def dynamic_symbols(self):
        """Returns a column matrix containing all of the symbols in the system
        that depend on time"""
        # Create a list of all of the expressions in the equations of motion
        if self._comb_explicit_rhs is None:
            eom_expressions = (self.comb_implicit_mat[:] +
                               self.comb_implicit_rhs[:])
        else:
            eom_expressions = (self._comb_explicit_rhs[:])

        functions_of_time = set()
        for expr in eom_expressions:
            functions_of_time = functions_of_time.union(
                find_dynamicsymbols(expr))
        functions_of_time = functions_of_time.union(self._states)

        return tuple(functions_of_time)

    def constant_symbols(self):
        """Returns a column matrix containing all of the symbols in the system
        that do not depend on time"""
        # Create a list of all of the expressions in the equations of motion
        if self._comb_explicit_rhs is None:
            eom_expressions = (self.comb_implicit_mat[:] +
                               self.comb_implicit_rhs[:])
        else:
            eom_expressions = (self._comb_explicit_rhs[:])

        constants = set()
        for expr in eom_expressions:
            constants = constants.union(expr.free_symbols)
        constants.remove(dynamicsymbols._t)

        return tuple(constants)

    @property
    def bodies(self):
        """Returns the bodies in the system"""
        if self._bodies is None:
            raise AttributeError("bodies were not specified for the system.")
        else:
            return self._bodies

    @property
    def loads(self):
        """Returns the loads in the system"""
        if self._loads is None:
            raise AttributeError("loads were not specified for the system.")
        else:
            return self._loads
<|endoftext|>"""Geometry objects for use by wrapping pathways."""

from abc import ABC, abstractmethod

from sympy import Integer, acos, pi, sqrt, sympify, tan
from sympy.core.relational import Eq
from sympy.functions.elementary.trigonometric import atan2
from sympy.polys.polytools import cancel
from sympy.physics.vector import Vector, dot
from sympy.simplify.simplify import trigsimp


__all__ = [
    'WrappingGeometryBase',
    'WrappingCylinder',
    'WrappingSphere',
]


class WrappingGeometryBase(ABC):
    """Abstract base class for all geometry classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom geometry types through subclassing.

    """

    @property
    @abstractmethod
    def point(cls):
        """The point with which the geometry is associated."""
        pass

    @abstractmethod
    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the geometry's surface.

        Parameters
        ==========
        point : Point
            The point for which it's to be ascertained if it's on the
            geometry's surface or not.

        """
        pass

    @abstractmethod
    def geodesic_length(self, point_1, point_2):
        """Returns the shortest distance between two points on a geometry's
        surface.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic length should be calculated.
        point_2 : Point
            The point to which the geodesic length should be calculated.

        """
        pass

    @abstractmethod
    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        pass

    def __repr__(self):
        """Default representation of a geometry object."""
        return f'{self.__class__.__name__}()'


class WrappingSphere(WrappingGeometryBase):
    """A solid spherical object.

    Explanation
    ===========

    A wrapping geometry that allows for circular arcs to be defined between
    pairs of points. These paths are always geodetic (the shortest possible).

    Examples
    ========

    To create a ``WrappingSphere`` instance, a ``Symbol`` denoting its radius
    and ``Point`` at which its center will be located are needed:

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Point, WrappingSphere
    >>> r = symbols('r')
    >>> pO = Point('pO')

    A sphere with radius ``r`` centered on ``pO`` can be instantiated with:

    >>> WrappingSphere(r, pO)
    WrappingSphere(radius=r, point=pO)

    Parameters
    ==========

    radius : Symbol
        Radius of the sphere. This symbol must represent a value that is
        positive and constant, i.e. it cannot be a dynamic symbol, nor can it
        be an expression.
    point : Point
        A point at which the sphere is centered.

    See Also
    ========

    WrappingCylinder: Cylindrical geometry where the wrapping direction can be
        defined.

    """

    def __init__(self, radius, point):
        """Initializer for ``WrappingSphere``.

        Parameters
        ==========

        radius : Symbol
            The radius of the sphere.
        point : Point
            A point on which the sphere is centered.

        """
        self.radius = radius
        self.point = point

    @property
    def radius(self):
        """Radius of the sphere."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def point(self):
        """A point on which the sphere is centered."""
        return self._point

    @point.setter
    def point(self, point):
        self._point = point

    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the sphere's surface.

        Parameters
        ==========

        point : Point
            The point for which it's to be ascertained if it's on the sphere's
            surface or not. This point's position relative to the sphere's
            center must be a simple expression involving the radius of the
            sphere, otherwise this check will likely not work.

        """
        point_vector = point.pos_from(self.point)
        if isinstance(point_vector, Vector):
            point_radius_squared = dot(point_vector, point_vector)
        else:
            point_radius_squared = point_vector**2
        return Eq(point_radius_squared, self.radius**2) == True

    def geodesic_length(self, point_1, point_2):
        r"""Returns the shortest distance between two points on the sphere's
        surface.

        Explanation
        ===========

        The geodesic length, i.e. the shortest arc along the surface of a
        sphere, connecting two points can be calculated using the formula:

        .. math::

           l = \arccos\left(\mathbf{v}_1 \cdot \mathbf{v}_2\right)

        where $\mathbf{v}_1$ and $\mathbf{v}_2$ are the unit vectors from the
        sphere's center to the first and second points on the sphere's surface
        respectively. Note that the actual path that the geodesic will take is
        undefined when the two points are directly opposite one another.

        Examples
        ========

        A geodesic length can only be calculated between two points on the
        sphere's surface. Firstly, a ``WrappingSphere`` instance must be
        created along with two points that will lie on its surface:

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
        ...     WrappingSphere)
        >>> N = ReferenceFrame('N')
        >>> r = symbols('r')
        >>> pO = Point('pO')
        >>> pO.set_vel(N, 0)
        >>> sphere = WrappingSphere(r, pO)
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')

        Let's assume that ``p1`` lies at a distance of ``r`` in the ``N.x``
        direction from ``pO`` and that ``p2`` is located on the sphere's
        surface in the ``N.y + N.z`` direction from ``pO``. These positions can
        be set with:

        >>> p1.set_pos(pO, r*N.x)
        >>> p1.pos_from(pO)
        r*N.x
        >>> p2.set_pos(pO, r*(N.y + N.z).normalize())
        >>> p2.pos_from(pO)
        sqrt(2)*r/2*N.y + sqrt(2)*r/2*N.z

        The geodesic length, which is in this case is a quarter of the sphere's
        circumference, can be calculated using the ``geodesic_length`` method:

        >>> sphere.geodesic_length(p1, p2)
        pi*r/2

        If the ``geodesic_length`` method is passed an argument, the ``Point``
        that doesn't lie on the sphere's surface then a ``ValueError`` is
        raised because it's not possible to calculate a value in this case.

        Parameters
        ==========

        point_1 : Point
            Point from which the geodesic length should be calculated.
        point_2 : Point
            Point to which the geodesic length should be calculated.

        """
        for point in (point_1, point_2):
            if not self.point_on_surface(point):
                msg = (
                    f'Geodesic length cannot be calculated as point {point} '
                    f'with radius {point.pos_from(self.point).magnitude()} '
                    f'from the sphere\'s center {self.point} does not lie on '
                    f'the surface of {self} with radius {self.radius}.'
                )
                raise ValueError(msg)
        point_1_vector = point_1.pos_from(self.point).normalize()
        point_2_vector = point_2.pos_from(self.point).normalize()
        central_angle = acos(point_2_vector.dot(point_1_vector))
        geodesic_length = self.radius*central_angle
        return geodesic_length

    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        pA, pB = point_1, point_2
        pO = self.point
        pA_vec = pA.pos_from(pO)
        pB_vec = pB.pos_from(pO)

        if pA_vec.cross(pB_vec) == 0:
            msg = (
                f'Can\'t compute geodesic end vectors for the pair of points '
                f'{pA} and {pB} on a sphere {self} as they are diametrically '
                f'opposed, thus the geodesic is not defined.'
            )
            raise ValueError(msg)

        return (
            pA_vec.cross(pB.pos_from(pA)).cross(pA_vec).normalize(),
            pB_vec.cross(pA.pos_from(pB)).cross(pB_vec).normalize(),
        )

    def __repr__(self):
        """Representation of a ``WrappingSphere``."""
        return (
            f'{self.__class__.__name__}(radius={self.radius}, '
            f'point={self.point})'
        )


class WrappingCylinder(WrappingGeometryBase):
    """A solid (infinite) cylindrical object.

    Explanation
    ===========

    A wrapping geometry that allows for circular arcs to be defined between
    pairs of points. These paths are always geodetic (the shortest possible) in
    the sense that they will be a straight line on the unwrapped cylinder's
    surface. However, it is also possible for a direction to be specified, i.e.
    paths can be influenced such that they either wrap along the shortest side
    or the longest side of the cylinder. To define these directions, rotations
    are in the positive direction following the right-hand rule.

    Examples
    ========

    To create a ``WrappingCylinder`` instance, a ``Symbol`` denoting its
    radius, a ``Vector`` defining its axis, and a ``Point`` through which its
    axis passes are needed:

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
    ...     WrappingCylinder)
    >>> N = ReferenceFrame('N')
    >>> r = symbols('r')
    >>> pO = Point('pO')
    >>> ax = N.x

    A cylinder with radius ``r``, and axis parallel to ``N.x`` passing through
    ``pO`` can be instantiated with:

    >>> WrappingCylinder(r, pO, ax)
    WrappingCylinder(radius=r, point=pO, axis=N.x)

    Parameters
    ==========

    radius : Symbol
        The radius of the cylinder.
    point : Point
        A point through which the cylinder's axis passes.
    axis : Vector
        The axis along which the cylinder is aligned.

    See Also
    ========

    WrappingSphere: Spherical geometry where the wrapping direction is always
        geodetic.

    """

    def __init__(self, radius, point, axis):
        """Initializer for ``WrappingCylinder``.

        Parameters
        ==========

        radius : Symbol
            The radius of the cylinder. This symbol must represent a value that
            is positive and constant, i.e. it cannot be a dynamic symbol.
        point : Point
            A point through which the cylinder's axis passes.
        axis : Vector
            The axis along which the cylinder is aligned.

        """
        self.radius = radius
        self.point = point
        self.axis = axis

    @property
    def radius(self):
        """Radius of the cylinder."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def point(self):
        """A point through which the cylinder's axis passes."""
        return self._point

    @point.setter
    def point(self, point):
        self._point = point

    @property
    def axis(self):
        """Axis along which the cylinder is aligned."""
        return self._axis

    @axis.setter
    def axis(self, axis):
        self._axis = axis.normalize()

    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the cylinder's surface.

        Parameters
        ==========

        point : Point
            The point for which it's to be ascertained if it's on the
            cylinder's surface or not. This point's position relative to the
            cylinder's axis must be a simple expression involving the radius of
            the sphere, otherwise this check will likely not work.

        """
        relative_position = point.pos_from(self.point)
        parallel = relative_position.dot(self.axis) * self.axis
        point_vector = relative_position - parallel
        if isinstance(point_vector, Vector):
            point_radius_squared = dot(point_vector, point_vector)
        else:
            point_radius_squared = point_vector**2
        return Eq(trigsimp(point_radius_squared), self.radius**2) == True

    def geodesic_length(self, point_1, point_2):
        """The shortest distance between two points on a geometry's surface.

        Explanation
        ===========

        The geodesic length, i.e. the shortest arc along the surface of a
        cylinder, connecting two points. It can be calculated using Pythagoras'
        theorem. The first short side is the distance between the two points on
        the cylinder's surface parallel to the cylinder's axis. The second
        short side is the arc of a circle between the two points of the
        cylinder's surface perpendicular to the cylinder's axis. The resulting
        hypotenuse is the geodesic length.

        Examples
        ========

        A geodesic length can only be calculated between two points on the
        cylinder's surface. Firstly, a ``WrappingCylinder`` instance must be
        created along with two points that will lie on its surface:

        >>> from sympy import symbols, cos, sin
        >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
        ...     WrappingCylinder, dynamicsymbols)
        >>> N = ReferenceFrame('N')
        >>> r = symbols('r')
        >>> pO = Point('pO')
        >>> pO.set_vel(N, 0)
        >>> cylinder = WrappingCylinder(r, pO, N.x)
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')

        Let's assume that ``p1`` is located at ``N.x + r*N.y`` relative to
        ``pO`` and that ``p2`` is located at ``r*(cos(q)*N.y + sin(q)*N.z)``
        relative to ``pO``, where ``q(t)`` is a generalized coordinate
        specifying the angle rotated around the ``N.x`` axis according to the
        right-hand rule where ``N.y`` is zero. These positions can be set with:

        >>> q = dynamicsymbols('q')
        >>> p1.set_pos(pO, N.x + r*N.y)
        >>> p1.pos_from(pO)
        N.x + r*N.y
        >>> p2.set_pos(pO, r*(cos(q)*N.y + sin(q)*N.z).normalize())
        >>> p2.pos_from(pO).simplify()
        r*cos(q(t))*N.y + r*sin(q(t))*N.z

        The geodesic length, which is in this case a is the hypotenuse of a
        right triangle where the other two side lengths are ``1`` (parallel to
        the cylinder's axis) and ``r*q(t)`` (parallel to the cylinder's cross
        section), can be calculated using the ``geodesic_length`` method:

        >>> cylinder.geodesic_length(p1, p2).simplify()
        sqrt(r**2*q(t)**2 + 1)

        If the ``geodesic_length`` method is passed an argument ``Point`` that
        doesn't lie on the sphere's surface then a ``ValueError`` is raised
        because it's not possible to calculate a value in this case.

        Parameters
        ==========

        point_1 : Point
            Point from which the geodesic length should be calculated.
        point_2 : Point
            Point to which the geodesic length should be calculated.

        """
        for point in (point_1, point_2):
            if not self.point_on_surface(point):
                msg = (
                    f'Geodesic length cannot be calculated as point {point} '
                    f'with radius {point.pos_from(self.point).magnitude()} '
                    f'from the cylinder\'s center {self.point} does not lie on '
                    f'the surface of {self} with radius {self.radius} and axis '
                    f'{self.axis}.'
                )
                raise ValueError(msg)

        relative_position = point_2.pos_from(point_1)
        parallel_length = relative_position.dot(self.axis)

        point_1_relative_position = point_1.pos_from(self.point)
        point_1_perpendicular_vector = (
            point_1_relative_position
            - point_1_relative_position.dot(self.axis)*self.axis
        ).normalize()

        point_2_relative_position = point_2.pos_from(self.point)
        point_2_perpendicular_vector = (
            point_2_relative_position
            - point_2_relative_position.dot(self.axis)*self.axis
        ).normalize()

        central_angle = _directional_atan(
            cancel(point_1_perpendicular_vector
                .cross(point_2_perpendicular_vector)
                .dot(self.axis)),
            cancel(point_1_perpendicular_vector.dot(point_2_perpendicular_vector)),
        )

        planar_arc_length = self.radius*central_angle
        geodesic_length = sqrt(parallel_length**2 + planar_arc_length**2)
        return geodesic_length

    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        point_1_from_origin_point = point_1.pos_from(self.point)
        point_2_from_origin_point = point_2.pos_from(self.point)

        if point_1_from_origin_point == point_2_from_origin_point:
            msg = (
                f'Cannot compute geodesic end vectors for coincident points '
                f'{point_1} and {point_2} as no geodesic exists.'
            )
            raise ValueError(msg)

        point_1_parallel = point_1_from_origin_point.dot(self.axis) * self.axis
        point_2_parallel = point_2_from_origin_point.dot(self.axis) * self.axis
        point_1_normal = (point_1_from_origin_point - point_1_parallel)
        point_2_normal = (point_2_from_origin_point - point_2_parallel)

        if point_1_normal == point_2_normal:
            point_1_perpendicular = Vector(0)
            point_2_perpendicular = Vector(0)
        else:
            point_1_perpendicular = self.axis.cross(point_1_normal).normalize()
            point_2_perpendicular = -self.axis.cross(point_2_normal).normalize()

        geodesic_length = self.geodesic_length(point_1, point_2)
        relative_position = point_2.pos_from(point_1)
        parallel_length = relative_position.dot(self.axis)
        planar_arc_length = sqrt(geodesic_length**2 - parallel_length**2)

        point_1_vector = (
            planar_arc_length * point_1_perpendicular
            + parallel_length * self.axis
        ).normalize()
        point_2_vector = (
            planar_arc_length * point_2_perpendicular
            - parallel_length * self.axis
        ).normalize()

        return (point_1_vector, point_2_vector)

    def __repr__(self):
        """Representation of a ``WrappingCylinder``."""
        return (
            f'{self.__class__.__name__}(radius={self.radius}, '
            f'point={self.point}, axis={self.axis})'
        )


def _directional_atan(numerator, denominator):
    """Compute atan in a directional sense as required for geodesics.

    Explanation
    ===========

    To be able to control the direction of the geodesic length along the
    surface of a cylinder a dedicated arctangent function is needed that
    properly handles the directionality of different case. This function
    ensures that the central angle is always positive but shifting the case
    where ``atan2`` would return a negative angle to be centered around
    ``2*pi``.

    Notes
    =====

    This function only handles very specific cases, i.e. the ones that are
    expected to be encountered when calculating symbolic geodesics on uniformly
    curved surfaces. As such, ``NotImplemented`` errors can be raised in many
    cases. This function is named with a leader underscore to indicate that it
    only aims to provide very specific functionality within the private scope
    of this module.

    """

    if numerator.is_number and denominator.is_number:
        angle = atan2(numerator, denominator)
        if angle < 0:
            angle += 2 * pi
    elif numerator.is_number:
        msg = (
            f'Cannot compute a directional atan when the numerator {numerator} '
            f'is numeric and the denominator {denominator} is symbolic.'
        )
        raise NotImplementedError(msg)
    elif denominator.is_number:
        msg = (
            f'Cannot compute a directional atan when the numerator {numerator} '
            f'is symbolic and the denominator {denominator} is numeric.'
        )
        raise NotImplementedError(msg)
    else:
        ratio = sympify(trigsimp(numerator / denominator))
        if isinstance(ratio, tan):
            angle = ratio.args[0]
        elif (
            ratio.is_Mul
            and ratio.args[0] == Integer(-1)
            and isinstance(ratio.args[1], tan)
        ):
            angle = 2 * pi - ratio.args[1].args[0]
        else:
            msg = f'Cannot compute a directional atan for the value {ratio}.'
            raise NotImplementedError(msg)

    return angle
<|endoftext|>from sympy import S
from sympy.physics.vector import cross, dot
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.inertia import inertia_of_point_mass
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Particle']


class Particle(BodyBase):
    """A particle.

    Explanation
    ===========

    Particles have a non-zero mass and lack spatial extension; they take up no
    space.

    Values need to be supplied on initialization, but can be changed later.

    Parameters
    ==========

    name : str
        Name of particle
    point : Point
        A physics/mechanics Point which represents the position, velocity, and
        acceleration of this Particle
    mass : Sympifyable
        A SymPy expression representing the Particle's mass
    potential_energy : Sympifyable
        The potential energy of the Particle.

    Examples
    ========

    >>> from sympy.physics.mechanics import Particle, Point
    >>> from sympy import Symbol
    >>> po = Point('po')
    >>> m = Symbol('m')
    >>> pa = Particle('pa', po, m)
    >>> # Or you could change these later
    >>> pa.mass = m
    >>> pa.point = po

    """
    point = BodyBase.masscenter

    def __init__(self, name, point=None, mass=None):
        super().__init__(name, point, mass)

    def linear_momentum(self, frame):
        """Linear momentum of the particle.

        Explanation
        ===========

        The linear momentum L, of a particle P, with respect to frame N is
        given by:

        L = m * v

        where m is the mass of the particle, and v is the velocity of the
        particle in the frame N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v = dynamicsymbols('m v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> A = Particle('A', P, m)
        >>> P.set_vel(N, v * N.x)
        >>> A.linear_momentum(N)
        m*v*N.x

        """

        return self.mass * self.point.vel(frame)

    def angular_momentum(self, point, frame):
        """Angular momentum of the particle about the point.

        Explanation
        ===========

        The angular momentum H, about some point O of a particle, P, is given
        by:

        ``H = cross(r, m * v)``

        where r is the position vector from point O to the particle P, m is
        the mass of the particle, and v is the velocity of the particle in
        the inertial frame, N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum of the particle is desired.

        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r = dynamicsymbols('m v r')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> A = O.locatenew('A', r * N.x)
        >>> P = Particle('P', A, m)
        >>> P.point.set_vel(N, v * N.y)
        >>> P.angular_momentum(O, N)
        m*r*v*N.z

        """

        return cross(self.point.pos_from(point),
                     self.mass * self.point.vel(frame))

    def kinetic_energy(self, frame):
        """Kinetic energy of the particle.

        Explanation
        ===========

        The kinetic energy, T, of a particle, P, is given by:

        ``T = 1/2 (dot(m * v, v))``

        where m is the mass of particle P, and v is the velocity of the
        particle in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The Particle's velocity is typically defined with respect to
            an inertial frame but any relevant frame in which the velocity is
            known can be supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy import symbols
        >>> m, v, r = symbols('m v r')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> P = Particle('P', O, m)
        >>> P.point.set_vel(N, v * N.y)
        >>> P.kinetic_energy(N)
        m*v**2/2

        """

        return S.Half * self.mass * dot(self.point.vel(frame),
                                        self.point.vel(frame))

    def set_potential_energy(self, scalar):
        sympy_deprecation_warning(
            """
The sympy.physics.mechanics.Particle.set_potential_energy()
method is deprecated. Instead use

    P.potential_energy = scalar
            """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-set-potential-energy",
        )
        self.potential_energy = scalar

    def parallel_axis(self, point, frame):
        """Returns an inertia dyadic of the particle with respect to another
        point and frame.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the particle expressed about the provided
            point and frame.

        """
        return inertia_of_point_mass(self.mass, self.point.pos_from(point),
                                     frame)
<|endoftext|>"""Implementations of pathways for use by actuators."""

from abc import ABC, abstractmethod

from sympy.core.singleton import S
from sympy.physics.mechanics.loads import Force
from sympy.physics.mechanics.wrapping_geometry import WrappingGeometryBase
from sympy.physics.vector import Point, dynamicsymbols


__all__ = ['PathwayBase', 'LinearPathway', 'ObstacleSetPathway',
           'WrappingPathway']


class PathwayBase(ABC):
    """Abstract base class for all pathway classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom pathway types through subclassing.

    """

    def __init__(self, *attachments):
        """Initializer for ``PathwayBase``."""
        self.attachments = attachments

    @property
    def attachments(self):
        """The pair of points defining a pathway's ends."""
        return self._attachments

    @attachments.setter
    def attachments(self, attachments):
        if hasattr(self, '_attachments'):
            msg = (
                f'Can\'t set attribute `attachments` to {repr(attachments)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if len(attachments) != 2:
            msg = (
                f'Value {repr(attachments)} passed to `attachments` was an '
                f'iterable of length {len(attachments)}, must be an iterable '
                f'of length 2.'
            )
            raise ValueError(msg)
        for i, point in enumerate(attachments):
            if not isinstance(point, Point):
                msg = (
                    f'Value {repr(point)} passed to `attachments` at index '
                    f'{i} was of type {type(point)}, must be {Point}.'
                )
                raise TypeError(msg)
        self._attachments = tuple(attachments)

    @property
    @abstractmethod
    def length(self):
        """An expression representing the pathway's length."""
        pass

    @property
    @abstractmethod
    def extension_velocity(self):
        """An expression representing the pathway's extension velocity."""
        pass

    @abstractmethod
    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        """
        pass

    def __repr__(self):
        """Default representation of a pathway."""
        attachments = ', '.join(str(a) for a in self.attachments)
        return f'{self.__class__.__name__}({attachments})'


class LinearPathway(PathwayBase):
    """Linear pathway between a pair of attachment points.

    Explanation
    ===========

    A linear pathway forms a straight-line segment between two points and is
    the simplest pathway that can be formed. It will not interact with any
    other objects in the system, i.e. a ``LinearPathway`` will intersect other
    objects to ensure that the path between its two ends (its attachments) is
    the shortest possible.

    A linear pathway is made up of two points that can move relative to each
    other, and a pair of equal and opposite forces acting on the points. If the
    positive time-varying Euclidean distance between the two points is defined,
    then the "extension velocity" is the time derivative of this distance. The
    extension velocity is positive when the two points are moving away from
    each other and negative when moving closer to each other. The direction for
    the force acting on either point is determined by constructing a unit
    vector directed from the other point to this point. This establishes a sign
    convention such that a positive force magnitude tends to push the points
    apart. The following diagram shows the positive force sense and the
    distance between the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    >>> from sympy.physics.mechanics import LinearPathway

    To construct a pathway, two points are required to be passed to the
    ``attachments`` parameter as a ``tuple``.

    >>> from sympy.physics.mechanics import Point
    >>> pA, pB = Point('pA'), Point('pB')
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> linear_pathway
    LinearPathway(pA, pB)

    The pathway created above isn't very interesting without the positions and
    velocities of its attachment points being described. Without this its not
    possible to describe how the pathway moves, i.e. its length or its
    extension velocity.

    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x

    A pathway's length can be accessed via its ``length`` attribute.

    >>> linear_pathway.length
    sqrt(q(t)**2)

    Note how what appears to be an overly-complex expression is returned. This
    is actually required as it ensures that a pathway's length is always
    positive.

    A pathway's extension velocity can be accessed similarly via its
    ``extension_velocity`` attribute.

    >>> linear_pathway.extension_velocity
    sqrt(q(t)**2)*Derivative(q(t), t)/q(t)

    Parameters
    ==========

    attachments : tuple[Point, Point]
        Pair of ``Point`` objects between which the linear pathway spans.
        Constructor expects two points to be passed, e.g.
        ``LinearPathway(Point('pA'), Point('pB'))``. More or fewer points will
        cause an error to be thrown.

    """

    def __init__(self, *attachments):
        """Initializer for ``LinearPathway``.

        Parameters
        ==========

        attachments : Point
            Pair of ``Point`` objects between which the linear pathway spans.
            Constructor expects two points to be passed, e.g.
            ``LinearPathway(Point('pA'), Point('pB'))``. More or fewer points
            will cause an error to be thrown.

        """
        super().__init__(*attachments)

    @property
    def length(self):
        """Exact analytical expression for the pathway's length."""
        return _point_pair_length(*self.attachments)

    @property
    def extension_velocity(self):
        """Exact analytical expression for the pathway's extension velocity."""
        return _point_pair_extension_velocity(*self.attachments)

    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced in a linear
        actuator that produces an expansile force ``F``. First, create a linear
        actuator between two points separated by the coordinate ``q`` in the
        ``x`` direction of the global frame ``N``.

        >>> from sympy.physics.mechanics import (LinearPathway, Point,
        ...     ReferenceFrame)
        >>> from sympy.physics.vector import dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> N = ReferenceFrame('N')
        >>> pA, pB = Point('pA'), Point('pB')
        >>> pB.set_pos(pA, q*N.x)
        >>> linear_pathway = LinearPathway(pA, pB)

        Now create a symbol ``F`` to describe the magnitude of the (expansile)
        force that will be produced along the pathway. The list of loads that
        ``KanesMethod`` requires can be produced by calling the pathway's
        ``to_loads`` method with ``F`` passed as the only argument.

        >>> from sympy import symbols
        >>> F = symbols('F')
        >>> linear_pathway.to_loads(F)
        [(pA, - F*q(t)/sqrt(q(t)**2)*N.x), (pB, F*q(t)/sqrt(q(t)**2)*N.x)]

        Parameters
        ==========

        force : Expr
            Magnitude of the force acting along the length of the pathway. As
            per the sign conventions for the pathway length, pathway extension
            velocity, and pair of point forces, if this ``Expr`` is positive
            then the force will act to push the pair of points away from one
            another (it is expansile).

        """
        relative_position = _point_pair_relative_position(*self.attachments)
        loads = [
            Force(self.attachments[0], -force*relative_position/self.length),
            Force(self.attachments[-1], force*relative_position/self.length),
        ]
        return loads


class ObstacleSetPathway(PathwayBase):
    """Obstacle-set pathway between a set of attachment points.

    Explanation
    ===========

    An obstacle-set pathway forms a series of straight-line segment between
    pairs of consecutive points in a set of points. It is similar to multiple
    linear pathways joined end-to-end. It will not interact with any other
    objects in the system, i.e. an ``ObstacleSetPathway`` will intersect other
    objects to ensure that the path between its pairs of points (its
    attachments) is the shortest possible.

    Examples
    ========

    To construct an obstacle-set pathway, three or more points are required to
    be passed to the ``attachments`` parameter as a ``tuple``.

    >>> from sympy.physics.mechanics import ObstacleSetPathway, Point
    >>> pA, pB, pC, pD = Point('pA'), Point('pB'), Point('pC'), Point('pD')
    >>> obstacle_set_pathway = ObstacleSetPathway(pA, pB, pC, pD)
    >>> obstacle_set_pathway
    ObstacleSetPathway(pA, pB, pC, pD)

    The pathway created above isn't very interesting without the positions and
    velocities of its attachment points being described. Without this its not
    possible to describe how the pathway moves, i.e. its length or its
    extension velocity.

    >>> from sympy import cos, sin
    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> pO = Point('pO')
    >>> pA.set_pos(pO, N.y)
    >>> pB.set_pos(pO, -N.x)
    >>> pC.set_pos(pA, cos(q) * N.x - (sin(q) + 1) * N.y)
    >>> pD.set_pos(pA, sin(q) * N.x + (cos(q) - 1) * N.y)
    >>> pB.pos_from(pA)
    - N.x - N.y
    >>> pC.pos_from(pA)
    cos(q(t))*N.x + (-sin(q(t)) - 1)*N.y
    >>> pD.pos_from(pA)
    sin(q(t))*N.x + (cos(q(t)) - 1)*N.y

    A pathway's length can be accessed via its ``length`` attribute.

    >>> obstacle_set_pathway.length.simplify()
    sqrt(2)*(sqrt(cos(q(t)) + 1) + 2)

    A pathway's extension velocity can be accessed similarly via its
    ``extension_velocity`` attribute.

    >>> obstacle_set_pathway.extension_velocity.simplify()
    -sqrt(2)*sin(q(t))*Derivative(q(t), t)/(2*sqrt(cos(q(t)) + 1))

    Parameters
    ==========

    attachments : tuple[Point, ...]
        The set of ``Point`` objects that define the segmented obstacle-set
        pathway.

    """

    def __init__(self, *attachments):
        """Initializer for ``ObstacleSetPathway``.

        Parameters
        ==========

        attachments : tuple[Point, ...]
            The set of ``Point`` objects that define the segmented obstacle-set
            pathway.

        """
        super().__init__(*attachments)

    @property
    def attachments(self):
        """The set of points defining a pathway's segmented path."""
        return self._attachments

    @attachments.setter
    def attachments(self, attachments):
        if hasattr(self, '_attachments'):
            msg = (
                f'Can\'t set attribute `attachments` to {repr(attachments)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if len(attachments) <= 2:
            msg = (
                f'Value {repr(attachments)} passed to `attachments` was an '
                f'iterable of length {len(attachments)}, must be an iterable '
                f'of length 3 or greater.'
            )
            raise ValueError(msg)
        for i, point in enumerate(attachments):
            if not isinstance(point, Point):
                msg = (
                    f'Value {repr(point)} passed to `attachments` at index '
                    f'{i} was of type {type(point)}, must be {Point}.'
                )
                raise TypeError(msg)
        self._attachments = tuple(attachments)

    @property
    def length(self):
        """Exact analytical expression for the pathway's length."""
        length = S.Zero
        attachment_pairs = zip(self.attachments[:-1], self.attachments[1:])
        for attachment_pair in attachment_pairs:
            length += _point_pair_length(*attachment_pair)
        return length

    @property
    def extension_velocity(self):
        """Exact analytical expression for the pathway's extension velocity."""
        extension_velocity = S.Zero
        attachment_pairs = zip(self.attachments[:-1], self.attachments[1:])
        for attachment_pair in attachment_pairs:
            extension_velocity += _point_pair_extension_velocity(*attachment_pair)
        return extension_velocity

    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced in an
        actuator that follows an obstacle-set pathway between four points and
        produces an expansile force ``F``. First, create a pair of reference
        frames, ``A`` and ``B``, in which the four points ``pA``, ``pB``,
        ``pC``, and ``pD`` will be located. The first two points in frame ``A``
        and the second two in frame ``B``. Frame ``B`` will also be oriented
        such that it relates to ``A`` via a rotation of ``q`` about an axis
        ``N.z`` in a global frame (``N.z``, ``A.z``, and ``B.z`` are parallel).

        >>> from sympy.physics.mechanics import (ObstacleSetPathway, Point,
        ...     ReferenceFrame)
        >>> from sympy.physics.vector import dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> N = ReferenceFrame('N')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'axis', (0, N.x))
        >>> B = A.orientnew('B', 'axis', (q, N.z))
        >>> pO = Point('pO')
        >>> pA, pB, pC, pD = Point('pA'), Point('pB'), Point('pC'), Point('pD')
        >>> pA.set_pos(pO, A.x)
        >>> pB.set_pos(pO, -A.y)
        >>> pC.set_pos(pO, B.y)
        >>> pD.set_pos(pO, B.x)
        >>> obstacle_set_pathway = ObstacleSetPathway(pA, pB, pC, pD)

        Now create a symbol ``F`` to describe the magnitude of the (expansile)
        force that will be produced along the pathway. The list of loads that
        ``KanesMethod`` requires can be produced by calling the pathway's
        ``to_loads`` method with ``F`` passed as the only argument.

        >>> from sympy import Symbol
        >>> F = Symbol('F')
        >>> obstacle_set_pathway.to_loads(F)
        [(pA, sqrt(2)*F/2*A.x + sqrt(2)*F/2*A.y),
         (pB, - sqrt(2)*F/2*A.x - sqrt(2)*F/2*A.y),
         (pB, - F/sqrt(2*cos(q(t)) + 2)*A.y - F/sqrt(2*cos(q(t)) + 2)*B.y),
         (pC, F/sqrt(2*cos(q(t)) + 2)*A.y + F/sqrt(2*cos(q(t)) + 2)*B.y),
         (pC, - sqrt(2)*F/2*B.x + sqrt(2)*F/2*B.y),
         (pD, sqrt(2)*F/2*B.x - sqrt(2)*F/2*B.y)]

        Parameters
        ==========

        force : Expr
            The force acting along the length of the pathway. It is assumed
            that this ``Expr`` represents an expansile force.

        """
        loads = []
        attachment_pairs = zip(self.attachments[:-1], self.attachments[1:])
        for attachment_pair in attachment_pairs:
            relative_position = _point_pair_relative_position(*attachment_pair)
            length = _point_pair_length(*attachment_pair)
            loads.extend([
                Force(attachment_pair[0], -force*relative_position/length),
                Force(attachment_pair[1], force*relative_position/length),
            ])
        return loads


class WrappingPathway(PathwayBase):
    """Pathway that wraps a geometry object.

    Explanation
    ===========

    A wrapping pathway interacts with a geometry object and forms a path that
    wraps smoothly along its surface. The wrapping pathway along the geometry
    object will be the geodesic that the geometry object defines based on the
    two points. It will not interact with any other objects in the system, i.e.
    a ``WrappingPathway`` will intersect other objects to ensure that the path
    between its two ends (its attachments) is the shortest possible.

    To explain the sign conventions used for pathway length, extension
    velocity, and direction of applied forces, we can ignore the geometry with
    which the wrapping pathway interacts. A wrapping pathway is made up of two
    points that can move relative to each other, and a pair of equal and
    opposite forces acting on the points. If the positive time-varying
    Euclidean distance between the two points is defined, then the "extension
    velocity" is the time derivative of this distance. The extension velocity
    is positive when the two points are moving away from each other and
    negative when moving closer to each other. The direction for the force
    acting on either point is determined by constructing a unit vector directed
    from the other point to this point. This establishes a sign convention such
    that a positive force magnitude tends to push the points apart. The
    following diagram shows the positive force sense and the distance between
    the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    >>> from sympy.physics.mechanics import WrappingPathway

    To construct a wrapping pathway, like other pathways, a pair of points must
    be passed, followed by an instance of a wrapping geometry class as a
    keyword argument. We'll use a cylinder with radius ``r`` and its axis
    parallel to ``N.x`` passing through a point ``pO``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Point, ReferenceFrame, WrappingCylinder
    >>> r = symbols('r')
    >>> N = ReferenceFrame('N')
    >>> pA, pB, pO = Point('pA'), Point('pB'), Point('pO')
    >>> cylinder = WrappingCylinder(r, pO, N.x)
    >>> wrapping_pathway = WrappingPathway(pA, pB, cylinder)
    >>> wrapping_pathway
    WrappingPathway(pA, pB, geometry=WrappingCylinder(radius=r, point=pO,
        axis=N.x))

    Parameters
    ==========

    attachment_1 : Point
        First of the pair of ``Point`` objects between which the wrapping
        pathway spans.
    attachment_2 : Point
        Second of the pair of ``Point`` objects between which the wrapping
        pathway spans.
    geometry : WrappingGeometryBase
        Geometry about which the pathway wraps.

    """

    def __init__(self, attachment_1, attachment_2, geometry):
        """Initializer for ``WrappingPathway``.

        Parameters
        ==========

        attachment_1 : Point
            First of the pair of ``Point`` objects between which the wrapping
            pathway spans.
        attachment_2 : Point
            Second of the pair of ``Point`` objects between which the wrapping
            pathway spans.
        geometry : WrappingGeometryBase
            Geometry about which the pathway wraps.
            The geometry about which the pathway wraps.

        """
        super().__init__(attachment_1, attachment_2)
        self.geometry = geometry

    @property
    def geometry(self):
        """Geometry around which the pathway wraps."""
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        if hasattr(self, '_geometry'):
            msg = (
                f'Can\'t set attribute `geometry` to {repr(geometry)} as it '
                f'is immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(geometry, WrappingGeometryBase):
            msg = (
                f'Value {repr(geometry)} passed to `geometry` was of type '
                f'{type(geometry)}, must be {WrappingGeometryBase}.'
            )
            raise TypeError(msg)
        self._geometry = geometry

    @property
    def length(self):
        """Exact analytical expression for the pathway's length."""
        return self.geometry.geodesic_length(*self.attachments)

    @property
    def extension_velocity(self):
        """Exact analytical expression for the pathway's extension velocity."""
        return self.length.diff(dynamicsymbols._t)

    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced in an
        actuator that produces an expansile force ``F`` while wrapping around a
        cylinder. First, create a cylinder with radius ``r`` and an axis
        parallel to the ``N.z`` direction of the global frame ``N`` that also
        passes through a point ``pO``.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
        ...     WrappingCylinder)
        >>> N = ReferenceFrame('N')
        >>> r = symbols('r', positive=True)
        >>> pO = Point('pO')
        >>> cylinder = WrappingCylinder(r, pO, N.z)

        Create the pathway of the actuator using the ``WrappingPathway`` class,
        defined to span between two points ``pA`` and ``pB``. Both points lie
        on the surface of the cylinder and the location of ``pB`` is defined
        relative to ``pA`` by the dynamics symbol ``q``.

        >>> from sympy import cos, sin
        >>> from sympy.physics.mechanics import WrappingPathway, dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> pA = Point('pA')
        >>> pB = Point('pB')
        >>> pA.set_pos(pO, r*N.x)
        >>> pB.set_pos(pO, r*(cos(q)*N.x + sin(q)*N.y))
        >>> pB.pos_from(pA)
        (r*cos(q(t)) - r)*N.x + r*sin(q(t))*N.y
        >>> pathway = WrappingPathway(pA, pB, cylinder)

        Now create a symbol ``F`` to describe the magnitude of the (expansile)
        force that will be produced along the pathway. The list of loads that
        ``KanesMethod`` requires can be produced by calling the pathway's
        ``to_loads`` method with ``F`` passed as the only argument.

        >>> F = symbols('F')
        >>> loads = pathway.to_loads(F)
        >>> [load.__class__(load.location, load.vector.simplify()) for load in loads]
        [(pA, F*N.y), (pB, F*sin(q(t))*N.x - F*cos(q(t))*N.y),
         (pO, - F*sin(q(t))*N.x + F*(cos(q(t)) - 1)*N.y)]

        Parameters
        ==========

        force : Expr
            Magnitude of the force acting along the length of the pathway. It
            is assumed that this ``Expr`` represents an expansile force.

        """
        pA, pB = self.attachments
        pO = self.geometry.point
        pA_force, pB_force = self.geometry.geodesic_end_vectors(pA, pB)
        pO_force = -(pA_force + pB_force)

        loads = [
            Force(pA, force * pA_force),
            Force(pB, force * pB_force),
            Force(pO, force * pO_force),
        ]
        return loads

    def __repr__(self):
        """Representation of a ``WrappingPathway``."""
        attachments = ', '.join(str(a) for a in self.attachments)
        return (
            f'{self.__class__.__name__}({attachments}, '
            f'geometry={self.geometry})'
        )


def _point_pair_relative_position(point_1, point_2):
    """The relative position between a pair of points."""
    return point_2.pos_from(point_1)


def _point_pair_length(point_1, point_2):
    """The length of the direct linear path between two points."""
    return _point_pair_relative_position(point_1, point_2).magnitude()


def _point_pair_extension_velocity(point_1, point_2):
    """The extension velocity of the direct linear path between two points."""
    return _point_pair_length(point_1, point_2).diff(dynamicsymbols._t)
<|endoftext|>from sympy import Symbol, S
from sympy.physics.vector import ReferenceFrame, Dyadic, Point, dot
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.inertia import inertia_of_point_mass, Inertia
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['RigidBody']


class RigidBody(BodyBase):
    """An idealized rigid body.

    Explanation
    ===========

    This is essentially a container which holds the various components which
    describe a rigid body: a name, mass, center of mass, reference frame, and
    inertia.

    All of these need to be supplied on creation, but can be changed
    afterwards.

    Attributes
    ==========

    name : string
        The body's name.
    masscenter : Point
        The point which represents the center of mass of the rigid body.
    frame : ReferenceFrame
        The ReferenceFrame which the rigid body is fixed in.
    mass : Sympifyable
        The body's mass.
    inertia : (Dyadic, Point)
        The body's inertia about a point; stored in a tuple as shown above.
    potential_energy : Sympifyable
        The potential energy of the RigidBody.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.physics.mechanics import ReferenceFrame, Point, RigidBody
    >>> from sympy.physics.mechanics import outer
    >>> m = Symbol('m')
    >>> A = ReferenceFrame('A')
    >>> P = Point('P')
    >>> I = outer (A.x, A.x)
    >>> inertia_tuple = (I, P)
    >>> B = RigidBody('B', P, A, m, inertia_tuple)
    >>> # Or you could change them afterwards
    >>> m2 = Symbol('m2')
    >>> B.mass = m2

    """

    def __init__(self, name, masscenter=None, frame=None, mass=None,
                 inertia=None):
        super().__init__(name, masscenter, mass)
        if frame is None:
            frame = ReferenceFrame(f'{name}_frame')
        self.frame = frame
        if inertia is None:
            ixx = Symbol(f'{name}_ixx')
            iyy = Symbol(f'{name}_iyy')
            izz = Symbol(f'{name}_izz')
            izx = Symbol(f'{name}_izx')
            ixy = Symbol(f'{name}_ixy')
            iyz = Symbol(f'{name}_iyz')
            inertia = Inertia.from_inertia_scalars(self.masscenter, self.frame,
                                                   ixx, iyy, izz, ixy, iyz, izx)
        self.inertia = inertia

    def __repr__(self):
        return (f'{self.__class__.__name__}({repr(self.name)}, masscenter='
                f'{repr(self.masscenter)}, frame={repr(self.frame)}, mass='
                f'{repr(self.mass)}, inertia={repr(self.inertia)})')

    @property
    def frame(self):
        """The ReferenceFrame fixed to the body."""
        return self._frame

    @frame.setter
    def frame(self, F):
        if not isinstance(F, ReferenceFrame):
            raise TypeError("RigidBody frame must be a ReferenceFrame object.")
        self._frame = F

    @property
    def x(self):
        """The basis Vector for the body, in the x direction. """
        return self.frame.x

    @property
    def y(self):
        """The basis Vector for the body, in the y direction. """
        return self.frame.y

    @property
    def z(self):
        """The basis Vector for the body, in the z direction. """
        return self.frame.z

    @property
    def inertia(self):
        """The body's inertia about a point; stored as (Dyadic, Point)."""
        return self._inertia

    @inertia.setter
    def inertia(self, I):
        # check if I is of the form (Dyadic, Point)
        if len(I) != 2 or not isinstance(I[0], Dyadic) or not isinstance(I[1], Point):
            raise TypeError("RigidBody inertia must be a tuple of the form (Dyadic, Point).")

        self._inertia = Inertia(I[0], I[1])
        # have I S/O, want I S/S*
        # I S/O = I S/S* + I S*/O; I S/S* = I S/O - I S*/O
        # I_S/S* = I_S/O - I_S*/O
        I_Ss_O = inertia_of_point_mass(self.mass,
                                       self.masscenter.pos_from(I[1]),
                                       self.frame)
        self._central_inertia = I[0] - I_Ss_O

    @property
    def central_inertia(self):
        """The body's central inertia dyadic."""
        return self._central_inertia

    @central_inertia.setter
    def central_inertia(self, I):
        if not isinstance(I, Dyadic):
            raise TypeError("RigidBody inertia must be a Dyadic object.")
        self.inertia = Inertia(I, self.masscenter)

    def linear_momentum(self, frame):
        """ Linear momentum of the rigid body.

        Explanation
        ===========

        The linear momentum L, of a rigid body B, with respect to frame N is
        given by:

        ``L = m * v``

        where m is the mass of the rigid body, and v is the velocity of the mass
        center of B in the frame N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v = dynamicsymbols('m v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (N.x, N.x)
        >>> Inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, N, m, Inertia_tuple)
        >>> B.linear_momentum(N)
        m*v*N.x

        """

        return self.mass * self.masscenter.vel(frame)

    def angular_momentum(self, point, frame):
        """Returns the angular momentum of the rigid body about a point in the
        given frame.

        Explanation
        ===========

        The angular momentum H of a rigid body B about some point O in a frame N
        is given by:

        ``H = dot(I, w) + cross(r, m * v)``

        where I and m are the central inertia dyadic and mass of rigid body B, w
        is the angular velocity of body B in the frame N, r is the position
        vector from point O to the mass center of B, and v is the velocity of
        the mass center in the frame N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum is desired.
        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r, omega = dynamicsymbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, 1 * N.x)
        >>> I = outer(b.x, b.x)
        >>> B = RigidBody('B', P, b, m, (I, P))
        >>> B.angular_momentum(P, N)
        omega*b.x

        """
        I = self.central_inertia
        w = self.frame.ang_vel_in(frame)
        m = self.mass
        r = self.masscenter.pos_from(point)
        v = self.masscenter.vel(frame)

        return I.dot(w) + r.cross(m * v)

    def kinetic_energy(self, frame):
        """Kinetic energy of the rigid body.

        Explanation
        ===========

        The kinetic energy, T, of a rigid body, B, is given by:

        ``T = 1/2 * (dot(dot(I, w), w) + dot(m * v, v))``

        where I and m are the central inertia dyadic and mass of rigid body B
        respectively, w is the body's angular velocity, and v is the velocity of
        the body's mass center in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The RigidBody's angular velocity and the velocity of it's mass
            center are typically defined with respect to an inertial frame but
            any relevant frame in which the velocities are known can be
            supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody
        >>> from sympy import symbols
        >>> m, v, r, omega = symbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (b.x, b.x)
        >>> inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, b, m, inertia_tuple)
        >>> B.kinetic_energy(N)
        m*v**2/2 + omega**2/2

        """

        rotational_KE = S.Half * dot(
            self.frame.ang_vel_in(frame),
            dot(self.central_inertia, self.frame.ang_vel_in(frame)))
        translational_KE = S.Half * self.mass * dot(self.masscenter.vel(frame),
                                                    self.masscenter.vel(frame))
        return rotational_KE + translational_KE

    def set_potential_energy(self, scalar):
        sympy_deprecation_warning(
            """
The sympy.physics.mechanics.RigidBody.set_potential_energy()
method is deprecated. Instead use

    B.potential_energy = scalar
            """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-set-potential-energy",
        )
        self.potential_energy = scalar

    def parallel_axis(self, point, frame=None):
        """Returns the inertia dyadic of the body with respect to another point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        """
        if frame is None:
            frame = self.frame
        return self.central_inertia + inertia_of_point_mass(
            self.mass, self.masscenter.pos_from(point), frame)
<|endoftext|>from sympy import sympify
from sympy.physics.vector import Point, Dyadic, ReferenceFrame, outer
from collections import namedtuple

__all__ = ['inertia', 'inertia_of_point_mass', 'Inertia']


def inertia(frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0):
    """Simple way to create inertia Dyadic object.

    Explanation
    ===========

    Creates an inertia Dyadic based on the given tensor values and a body-fixed
    reference frame.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame the inertia is defined in.
    ixx : Sympifyable
        The xx element in the inertia dyadic.
    iyy : Sympifyable
        The yy element in the inertia dyadic.
    izz : Sympifyable
        The zz element in the inertia dyadic.
    ixy : Sympifyable
        The xy element in the inertia dyadic.
    iyz : Sympifyable
        The yz element in the inertia dyadic.
    izx : Sympifyable
        The zx element in the inertia dyadic.

    Examples
    ========

    >>> from sympy.physics.mechanics import ReferenceFrame, inertia
    >>> N = ReferenceFrame('N')
    >>> inertia(N, 1, 2, 3)
    (N.x|N.x) + 2*(N.y|N.y) + 3*(N.z|N.z)

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Need to define the inertia in a frame')
    ixx, iyy, izz = sympify(ixx), sympify(iyy), sympify(izz)
    ixy, iyz, izx = sympify(ixy), sympify(iyz), sympify(izx)
    return (ixx*outer(frame.x, frame.x) + ixy*outer(frame.x, frame.y) +
            izx*outer(frame.x, frame.z) + ixy*outer(frame.y, frame.x) +
            iyy*outer(frame.y, frame.y) + iyz*outer(frame.y, frame.z) +
            izx*outer(frame.z, frame.x) + iyz*outer(frame.z, frame.y) +
            izz*outer(frame.z, frame.z))


def inertia_of_point_mass(mass, pos_vec, frame):
    """Inertia dyadic of a point mass relative to point O.

    Parameters
    ==========

    mass : Sympifyable
        Mass of the point mass
    pos_vec : Vector
        Position from point O to point mass
    frame : ReferenceFrame
        Reference frame to express the dyadic in

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import ReferenceFrame, inertia_of_point_mass
    >>> N = ReferenceFrame('N')
    >>> r, m = symbols('r m')
    >>> px = r * N.x
    >>> inertia_of_point_mass(m, px, N)
    m*r**2*(N.y|N.y) + m*r**2*(N.z|N.z)

    """

    return mass*(
        (outer(frame.x, frame.x) +
         outer(frame.y, frame.y) +
         outer(frame.z, frame.z)) *
        (pos_vec.dot(pos_vec)) - outer(pos_vec, pos_vec))


class Inertia(namedtuple('Inertia', ['dyadic', 'point'])):
    """Inertia object consisting of a Dyadic and a Point of reference.

    Explanation
    ===========

    This is a simple class to store the Point and Dyadic, belonging to an
    inertia.

    Attributes
    ==========

    dyadic : Dyadic
        The dyadic of the inertia.
    point : Point
        The reference point of the inertia.

    Examples
    ========

    >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Inertia(N.x.outer(N.x) + N.y.outer(N.y) + N.z.outer(N.z), Po)
    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    In the example above the Dyadic was created manually, one can however also
    use the ``inertia`` function for this or the class method ``from_tensor`` as
    shown below.

    >>> Inertia.from_inertia_scalars(Po, N, 1, 1, 1)
    ((N.x|N.x) + (N.y|N.y) + (N.z|N.z), Po)

    """
    __slots__ = ()

    def __new__(cls, dyadic, point):
        # Switch order if given in the wrong order
        if isinstance(dyadic, Point) and isinstance(point, Dyadic):
            point, dyadic = dyadic, point
        if not isinstance(point, Point):
            raise TypeError('Reference point should be of type Point')
        if not isinstance(dyadic, Dyadic):
            raise TypeError('Inertia value should be expressed as a Dyadic')
        return super().__new__(cls, dyadic, point)

    @classmethod
    def from_inertia_scalars(cls, point, frame, ixx, iyy, izz, ixy=0, iyz=0,
                             izx=0):
        """Simple way to create an Inertia object based on the tensor values.

        Explanation
        ===========

        This class method uses the :func`~.inertia` to create the Dyadic based
        on the tensor values.

        Parameters
        ==========

        point : Point
            The reference point of the inertia.
        frame : ReferenceFrame
            The frame the inertia is defined in.
        ixx : Sympifyable
            The xx element in the inertia dyadic.
        iyy : Sympifyable
            The yy element in the inertia dyadic.
        izz : Sympifyable
            The zz element in the inertia dyadic.
        ixy : Sympifyable
            The xy element in the inertia dyadic.
        iyz : Sympifyable
            The yz element in the inertia dyadic.
        izx : Sympifyable
            The zx element in the inertia dyadic.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import ReferenceFrame, Point, Inertia
        >>> ixx, iyy, izz, ixy, iyz, izx = symbols('ixx iyy izz ixy iyz izx')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> I = Inertia.from_inertia_scalars(P, N, ixx, iyy, izz, ixy, iyz, izx)

        The tensor values can easily be seen when converting the dyadic to a
        matrix.

        >>> I.dyadic.to_matrix(N)
        Matrix([
        [ixx, ixy, izx],
        [ixy, iyy, iyz],
        [izx, iyz, izz]])

        """
        return cls(inertia(frame, ixx, iyy, izz, ixy, iyz, izx), point)

    def __add__(self, other):
        raise TypeError(f"unsupported operand type(s) for +: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    def __mul__(self, other):
        raise TypeError(f"unsupported operand type(s) for *: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    __radd__ = __add__
    __rmul__ = __mul__
<|endoftext|>from sympy import Symbol
from sympy.physics.vector import Point, Vector, ReferenceFrame, Dyadic
from sympy.physics.mechanics import RigidBody, Particle, Inertia
from sympy.physics.mechanics.body_base import BodyBase
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Body']


# XXX: We use type:ignore because the classes RigidBody and Particle have
# inconsistent parallel axis methods that take different numbers of arguments.
class Body(RigidBody, Particle):  # type: ignore
    """
    Body is a common representation of either a RigidBody or a Particle SymPy
    object depending on what is passed in during initialization. If a mass is
    passed in and central_inertia is left as None, the Particle object is
    created. Otherwise a RigidBody object will be created.

    .. deprecated:: 1.13
        The Body class is deprecated. Its functionality is captured by
        :class:`~.RigidBody` and :class:`~.Particle`.

    Explanation
    ===========

    The attributes that Body possesses will be the same as a Particle instance
    or a Rigid Body instance depending on which was created. Additional
    attributes are listed below.

    Attributes
    ==========

    name : string
        The body's name
    masscenter : Point
        The point which represents the center of mass of the rigid body
    frame : ReferenceFrame
        The reference frame which the body is fixed in
    mass : Sympifyable
        The body's mass
    inertia : (Dyadic, Point)
        The body's inertia around its center of mass. This attribute is specific
        to the rigid body form of Body and is left undefined for the Particle
        form
    loads : iterable
        This list contains information on the different loads acting on the
        Body. Forces are listed as a (point, vector) tuple and torques are
        listed as (reference frame, vector) tuples.

    Parameters
    ==========

    name : String
        Defines the name of the body. It is used as the base for defining
        body specific properties.
    masscenter : Point, optional
        A point that represents the center of mass of the body or particle.
        If no point is given, a point is generated.
    mass : Sympifyable, optional
        A Sympifyable object which represents the mass of the body. If no
        mass is passed, one is generated.
    frame : ReferenceFrame, optional
        The ReferenceFrame that represents the reference frame of the body.
        If no frame is given, a frame is generated.
    central_inertia : Dyadic, optional
        Central inertia dyadic of the body. If none is passed while creating
        RigidBody, a default inertia is generated.

    Examples
    ========

    As Body has been deprecated, the following examples are for illustrative
    purposes only. The functionality of Body is fully captured by
    :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
    warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings

    Default behaviour. This results in the creation of a RigidBody object for
    which the mass, mass center, frame and inertia attributes are given default
    values. ::

        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     body = Body('name_of_body')

    This next example demonstrates the code required to specify all of the
    values of the Body object. Note this will also create a RigidBody version of
    the Body object. ::

        >>> from sympy import Symbol
        >>> from sympy.physics.mechanics import ReferenceFrame, Point, inertia
        >>> from sympy.physics.mechanics import Body
        >>> mass = Symbol('mass')
        >>> masscenter = Point('masscenter')
        >>> frame = ReferenceFrame('frame')
        >>> ixx = Symbol('ixx')
        >>> body_inertia = inertia(frame, ixx, 0, 0)
        >>> with ignore_warnings(DeprecationWarning):
        ...     body = Body('name_of_body', masscenter, mass, frame, body_inertia)

    The minimal code required to create a Particle version of the Body object
    involves simply passing in a name and a mass. ::

        >>> from sympy import Symbol
        >>> from sympy.physics.mechanics import Body
        >>> mass = Symbol('mass')
        >>> with ignore_warnings(DeprecationWarning):
        ...     body = Body('name_of_body', mass=mass)

    The Particle version of the Body object can also receive a masscenter point
    and a reference frame, just not an inertia.
    """

    def __init__(self, name, masscenter=None, mass=None, frame=None,
                 central_inertia=None):
        sympy_deprecation_warning(
            """
            Support for the Body class has been removed, as its functionality is
            fully captured by RigidBody and Particle.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-mechanics-body-class"
        )

        self._loads = []

        if frame is None:
            frame = ReferenceFrame(name + '_frame')

        if masscenter is None:
            masscenter = Point(name + '_masscenter')

        if central_inertia is None and mass is None:
            ixx = Symbol(name + '_ixx')
            iyy = Symbol(name + '_iyy')
            izz = Symbol(name + '_izz')
            izx = Symbol(name + '_izx')
            ixy = Symbol(name + '_ixy')
            iyz = Symbol(name + '_iyz')
            _inertia = Inertia.from_inertia_scalars(masscenter, frame, ixx, iyy,
                                                    izz, ixy, iyz, izx)
        else:
            _inertia = (central_inertia, masscenter)

        if mass is None:
            _mass = Symbol(name + '_mass')
        else:
            _mass = mass

        masscenter.set_vel(frame, 0)

        # If user passes masscenter and mass then a particle is created
        # otherwise a rigidbody. As a result a body may or may not have inertia.
        # Note: BodyBase.__init__ is used to prevent problems with super() calls in
        # Particle and RigidBody arising due to multiple inheritance.
        if central_inertia is None and mass is not None:
            BodyBase.__init__(self, name, masscenter, _mass)
            self.frame = frame
            self._central_inertia = Dyadic(0)
        else:
            BodyBase.__init__(self, name, masscenter, _mass)
            self.frame = frame
            self.inertia = _inertia

    def __repr__(self):
        if self.is_rigidbody:
            return RigidBody.__repr__(self)
        return Particle.__repr__(self)

    @property
    def loads(self):
        return self._loads

    @property
    def x(self):
        """The basis Vector for the Body, in the x direction."""
        return self.frame.x

    @property
    def y(self):
        """The basis Vector for the Body, in the y direction."""
        return self.frame.y

    @property
    def z(self):
        """The basis Vector for the Body, in the z direction."""
        return self.frame.z

    @property
    def inertia(self):
        """The body's inertia about a point; stored as (Dyadic, Point)."""
        if self.is_rigidbody:
            return RigidBody.inertia.fget(self)
        return (self.central_inertia, self.masscenter)

    @inertia.setter
    def inertia(self, I):
        RigidBody.inertia.fset(self, I)

    @property
    def is_rigidbody(self):
        if hasattr(self, '_inertia'):
            return True
        return False

    def kinetic_energy(self, frame):
        """Kinetic energy of the body.

        Parameters
        ==========

        frame : ReferenceFrame or Body
            The Body's angular velocity and the velocity of it's mass
            center are typically defined with respect to an inertial frame but
            any relevant frame in which the velocities are known can be supplied.

        Examples
        ========

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, ReferenceFrame, Point
        >>> from sympy import symbols
        >>> m, v, r, omega = symbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> with ignore_warnings(DeprecationWarning):
        ...     P = Body('P', masscenter=O, mass=m)
        >>> P.masscenter.set_vel(N, v * N.y)
        >>> P.kinetic_energy(N)
        m*v**2/2

        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B', masscenter=P, frame=b)
        >>> B.kinetic_energy(N)
        B_ixx*omega**2/2 + B_mass*v**2/2

        See Also
        ========

        sympy.physics.mechanics : Particle, RigidBody

        """
        if isinstance(frame, Body):
            frame = Body.frame
        if self.is_rigidbody:
            return RigidBody(self.name, self.masscenter, self.frame, self.mass,
                            (self.central_inertia, self.masscenter)).kinetic_energy(frame)
        return Particle(self.name, self.masscenter, self.mass).kinetic_energy(frame)

    def apply_force(self, force, point=None, reaction_body=None, reaction_point=None):
        """Add force to the body(s).

        Explanation
        ===========

        Applies the force on self or equal and opposite forces on
        self and other body if both are given on the desired point on the bodies.
        The force applied on other body is taken opposite of self, i.e, -force.

        Parameters
        ==========

        force: Vector
            The force to be applied.
        point: Point, optional
            The point on self on which force is applied.
            By default self's masscenter.
        reaction_body: Body, optional
            Second body on which equal and opposite force
            is to be applied.
        reaction_point : Point, optional
            The point on other body on which equal and opposite
            force is applied. By default masscenter of other body.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import Body, Point, dynamicsymbols
        >>> m, g = symbols('m g')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> force1 = m*g*B.z
        >>> B.apply_force(force1) #Applying force on B's masscenter
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z)]

        We can also remove some part of force from any point on the body by
        adding the opposite force to the body on that point.

        >>> f1, f2 = dynamicsymbols('f1 f2')
        >>> P = Point('P') #Considering point P on body B
        >>> B.apply_force(f1*B.x + f2*B.y, P)
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z), (P, f1(t)*B_frame.x + f2(t)*B_frame.y)]

        Let's remove f1 from point P on body B.

        >>> B.apply_force(-f1*B.x, P)
        >>> B.loads
        [(B_masscenter, g*m*B_frame.z), (P, f2(t)*B_frame.y)]

        To further demonstrate the use of ``apply_force`` attribute,
        consider two bodies connected through a spring.

        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> with ignore_warnings(DeprecationWarning):
        ...     N = Body('N') #Newtonion Frame
        >>> x = dynamicsymbols('x')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B1 = Body('B1')
        ...     B2 = Body('B2')
        >>> spring_force = x*N.x

        Now let's apply equal and opposite spring force to the bodies.

        >>> P1 = Point('P1')
        >>> P2 = Point('P2')
        >>> B1.apply_force(spring_force, point=P1, reaction_body=B2, reaction_point=P2)

        We can check the loads(forces) applied to bodies now.

        >>> B1.loads
        [(P1, x(t)*N_frame.x)]
        >>> B2.loads
        [(P2, - x(t)*N_frame.x)]

        Notes
        =====

        If a new force is applied to a body on a point which already has some
        force applied on it, then the new force is added to the already applied
        force on that point.

        """

        if not isinstance(point, Point):
            if point is None:
                point = self.masscenter  # masscenter
            else:
                raise TypeError("Force must be applied to a point on the body.")
        if not isinstance(force, Vector):
            raise TypeError("Force must be a vector.")

        if reaction_body is not None:
            reaction_body.apply_force(-force, point=reaction_point)

        for load in self._loads:
            if point in load:
                force += load[1]
                self._loads.remove(load)
                break

        self._loads.append((point, force))

    def apply_torque(self, torque, reaction_body=None):
        """Add torque to the body(s).

        Explanation
        ===========

        Applies the torque on self or equal and opposite torques on
        self and other body if both are given.
        The torque applied on other body is taken opposite of self,
        i.e, -torque.

        Parameters
        ==========

        torque: Vector
            The torque to be applied.
        reaction_body: Body, optional
            Second body on which equal and opposite torque
            is to be applied.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> t = symbols('t')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> torque1 = t*B.z
        >>> B.apply_torque(torque1)
        >>> B.loads
        [(B_frame, t*B_frame.z)]

        We can also remove some part of torque from the body by
        adding the opposite torque to the body.

        >>> t1, t2 = dynamicsymbols('t1 t2')
        >>> B.apply_torque(t1*B.x + t2*B.y)
        >>> B.loads
        [(B_frame, t1(t)*B_frame.x + t2(t)*B_frame.y + t*B_frame.z)]

        Let's remove t1 from Body B.

        >>> B.apply_torque(-t1*B.x)
        >>> B.loads
        [(B_frame, t2(t)*B_frame.y + t*B_frame.z)]

        To further demonstrate the use, let us consider two bodies such that
        a torque `T` is acting on one body, and `-T` on the other.

        >>> from sympy.physics.mechanics import Body, dynamicsymbols
        >>> with ignore_warnings(DeprecationWarning):
        ...     N = Body('N') #Newtonion frame
        ...     B1 = Body('B1')
        ...     B2 = Body('B2')
        >>> v = dynamicsymbols('v')
        >>> T = v*N.y #Torque

        Now let's apply equal and opposite torque to the bodies.

        >>> B1.apply_torque(T, B2)

        We can check the loads (torques) applied to bodies now.

        >>> B1.loads
        [(B1_frame, v(t)*N_frame.y)]
        >>> B2.loads
        [(B2_frame, - v(t)*N_frame.y)]

        Notes
        =====

        If a new torque is applied on body which already has some torque applied on it,
        then the new torque is added to the previous torque about the body's frame.

        """

        if not isinstance(torque, Vector):
            raise TypeError("A Vector must be supplied to add torque.")

        if reaction_body is not None:
            reaction_body.apply_torque(-torque)

        for load in self._loads:
            if self.frame in load:
                torque += load[1]
                self._loads.remove(load)
                break
        self._loads.append((self.frame, torque))

    def clear_loads(self):
        """
        Clears the Body's loads list.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> force = B.x + B.y
        >>> B.apply_force(force)
        >>> B.loads
        [(B_masscenter, B_frame.x + B_frame.y)]
        >>> B.clear_loads()
        >>> B.loads
        []

        """

        self._loads = []

    def remove_load(self, about=None):
        """
        Remove load about a point or frame.

        Parameters
        ==========

        about : Point or ReferenceFrame, optional
            The point about which force is applied,
            and is to be removed.
            If about is None, then the torque about
            self's frame is removed.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, Point
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B')
        >>> P = Point('P')
        >>> f1 = B.x
        >>> f2 = B.y
        >>> B.apply_force(f1)
        >>> B.apply_force(f2, P)
        >>> B.loads
        [(B_masscenter, B_frame.x), (P, B_frame.y)]

        >>> B.remove_load(P)
        >>> B.loads
        [(B_masscenter, B_frame.x)]

        """

        if about is not None:
            if not isinstance(about, Point):
                raise TypeError('Load is applied about Point or ReferenceFrame.')
        else:
            about = self.frame

        for load in self._loads:
            if about in load:
                self._loads.remove(load)
                break

    def masscenter_vel(self, body):
        """
        Returns the velocity of the mass center with respect to the provided
        rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the velocity in.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        ...     B = Body('B')
        >>> A.masscenter.set_vel(B.frame, 5*B.frame.x)
        >>> A.masscenter_vel(B)
        5*B_frame.x
        >>> A.masscenter_vel(B.frame)
        5*B_frame.x

        """

        if isinstance(body,  ReferenceFrame):
            frame=body
        elif isinstance(body, Body):
            frame = body.frame
        return self.masscenter.vel(frame)

    def ang_vel_in(self, body):
        """
        Returns this body's angular velocity with respect to the provided
        rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the angular velocity in.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body, ReferenceFrame
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        >>> N = ReferenceFrame('N')
        >>> with ignore_warnings(DeprecationWarning):
        ...     B = Body('B', frame=N)
        >>> A.frame.set_ang_vel(N, 5*N.x)
        >>> A.ang_vel_in(B)
        5*N.x
        >>> A.ang_vel_in(N)
        5*N.x

        """

        if isinstance(body,  ReferenceFrame):
            frame=body
        elif isinstance(body, Body):
            frame = body.frame
        return self.frame.ang_vel_in(frame)

    def dcm(self, body):
        """
        Returns the direction cosine matrix of this body relative to the
        provided rigid body or reference frame.

        Parameters
        ==========

        body: Body or ReferenceFrame
            The rigid body or reference frame to calculate the dcm.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        ...     B = Body('B')
        >>> A.frame.orient_axis(B.frame, B.frame.x, 5)
        >>> A.dcm(B)
        Matrix([
        [1,       0,      0],
        [0,  cos(5), sin(5)],
        [0, -sin(5), cos(5)]])
        >>> A.dcm(B.frame)
        Matrix([
        [1,       0,      0],
        [0,  cos(5), sin(5)],
        [0, -sin(5), cos(5)]])

        """

        if isinstance(body,  ReferenceFrame):
            frame=body
        elif isinstance(body, Body):
            frame = body.frame
        return self.frame.dcm(frame)

    def parallel_axis(self, point, frame=None):
        """Returns the inertia dyadic of the body with respect to another
        point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        Example
        =======

        As Body has been deprecated, the following examples are for illustrative
        purposes only. The functionality of Body is fully captured by
        :class:`~.RigidBody` and :class:`~.Particle`. To ignore the deprecation
        warning we can use the ignore_warnings context manager.

        >>> from sympy.utilities.exceptions import ignore_warnings
        >>> from sympy.physics.mechanics import Body
        >>> with ignore_warnings(DeprecationWarning):
        ...     A = Body('A')
        >>> P = A.masscenter.locatenew('point', 3 * A.x + 5 * A.y)
        >>> A.parallel_axis(P).to_matrix(A.frame)
        Matrix([
        [A_ixx + 25*A_mass, A_ixy - 15*A_mass,             A_izx],
        [A_ixy - 15*A_mass,  A_iyy + 9*A_mass,             A_iyz],
        [            A_izx,             A_iyz, A_izz + 34*A_mass]])

        """
        if self.is_rigidbody:
            return RigidBody.parallel_axis(self, point, frame)
        return Particle.parallel_axis(self, point, frame)
<|endoftext|>#!/usr/bin/env python
"""This module contains some sample symbolic models used for testing and
examples."""

# Internal imports
from sympy.core import backend as sm
import sympy.physics.mechanics as me


def multi_mass_spring_damper(n=1, apply_gravity=False,
                             apply_external_forces=False):
    r"""Returns a system containing the symbolic equations of motion and
    associated variables for a simple multi-degree of freedom point mass,
    spring, damper system with optional gravitational and external
    specified forces. For example, a two mass system under the influence of
    gravity and external forces looks like:

    ::

        ----------------
         |     |     |   | g
         \    | |    |   V
      k0 /    --- c0 |
         |     |     | x0, v0
        ---------    V
        |  m0   | -----
        ---------    |
         | |   |     |
         \ v  | |    |
      k1 / f0 --- c1 |
         |     |     | x1, v1
        ---------    V
        |  m1   | -----
        ---------
           | f1
           V

    Parameters
    ==========

    n : integer
        The number of masses in the serial chain.
    apply_gravity : boolean
        If true, gravity will be applied to each mass.
    apply_external_forces : boolean
        If true, a time varying external force will be applied to each mass.

    Returns
    =======

    kane : sympy.physics.mechanics.kane.KanesMethod
        A KanesMethod object.

    """

    mass = sm.symbols('m:{}'.format(n))
    stiffness = sm.symbols('k:{}'.format(n))
    damping = sm.symbols('c:{}'.format(n))

    acceleration_due_to_gravity = sm.symbols('g')

    coordinates = me.dynamicsymbols('x:{}'.format(n))
    speeds = me.dynamicsymbols('v:{}'.format(n))
    specifieds = me.dynamicsymbols('f:{}'.format(n))

    ceiling = me.ReferenceFrame('N')
    origin = me.Point('origin')
    origin.set_vel(ceiling, 0)

    points = [origin]
    kinematic_equations = []
    particles = []
    forces = []

    for i in range(n):

        center = points[-1].locatenew('center{}'.format(i),
                                      coordinates[i] * ceiling.x)
        center.set_vel(ceiling, points[-1].vel(ceiling) +
                       speeds[i] * ceiling.x)
        points.append(center)

        block = me.Particle('block{}'.format(i), center, mass[i])

        kinematic_equations.append(speeds[i] - coordinates[i].diff())

        total_force = (-stiffness[i] * coordinates[i] -
                       damping[i] * speeds[i])
        try:
            total_force += (stiffness[i + 1] * coordinates[i + 1] +
                            damping[i + 1] * speeds[i + 1])
        except IndexError:  # no force from below on last mass
            pass

        if apply_gravity:
            total_force += mass[i] * acceleration_due_to_gravity

        if apply_external_forces:
            total_force += specifieds[i]

        forces.append((center, total_force * ceiling.x))

        particles.append(block)

    kane = me.KanesMethod(ceiling, q_ind=coordinates, u_ind=speeds,
                          kd_eqs=kinematic_equations)
    kane.kanes_equations(particles, forces)

    return kane


def n_link_pendulum_on_cart(n=1, cart_force=True, joint_torques=False):
    r"""Returns the system containing the symbolic first order equations of
    motion for a 2D n-link pendulum on a sliding cart under the influence of
    gravity.

    ::

                  |
         o    y   v
          \ 0 ^   g
           \  |
          --\-|----
          |  \|   |
      F-> |   o --|---> x
          |       |
          ---------
           o     o

    Parameters
    ==========

    n : integer
        The number of links in the pendulum.
    cart_force : boolean, default=True
        If true an external specified lateral force is applied to the cart.
    joint_torques : boolean, default=False
        If true joint torques will be added as specified inputs at each
        joint.

    Returns
    =======

    kane : sympy.physics.mechanics.kane.KanesMethod
        A KanesMethod object.

    Notes
    =====

    The degrees of freedom of the system are n + 1, i.e. one for each
    pendulum link and one for the lateral motion of the cart.

    M x' = F, where x = [u0, ..., un+1, q0, ..., qn+1]

    The joint angles are all defined relative to the ground where the x axis
    defines the ground line and the y axis points up. The joint torques are
    applied between each adjacent link and the between the cart and the
    lower link where a positive torque corresponds to positive angle.

    """
    if n <= 0:
        raise ValueError('The number of links must be a positive integer.')

    q = me.dynamicsymbols('q:{}'.format(n + 1))
    u = me.dynamicsymbols('u:{}'.format(n + 1))

    if joint_torques is True:
        T = me.dynamicsymbols('T1:{}'.format(n + 1))

    m = sm.symbols('m:{}'.format(n + 1))
    l = sm.symbols('l:{}'.format(n))
    g, t = sm.symbols('g t')

    I = me.ReferenceFrame('I')
    O = me.Point('O')
    O.set_vel(I, 0)

    P0 = me.Point('P0')
    P0.set_pos(O, q[0] * I.x)
    P0.set_vel(I, u[0] * I.x)
    Pa0 = me.Particle('Pa0', P0, m[0])

    frames = [I]
    points = [P0]
    particles = [Pa0]
    forces = [(P0, -m[0] * g * I.y)]
    kindiffs = [q[0].diff(t) - u[0]]

    if cart_force is True or joint_torques is True:
        specified = []
    else:
        specified = None

    for i in range(n):
        Bi = I.orientnew('B{}'.format(i), 'Axis', [q[i + 1], I.z])
        Bi.set_ang_vel(I, u[i + 1] * I.z)
        frames.append(Bi)

        Pi = points[-1].locatenew('P{}'.format(i + 1), l[i] * Bi.y)
        Pi.v2pt_theory(points[-1], I, Bi)
        points.append(Pi)

        Pai = me.Particle('Pa' + str(i + 1), Pi, m[i + 1])
        particles.append(Pai)

        forces.append((Pi, -m[i + 1] * g * I.y))

        if joint_torques is True:

            specified.append(T[i])

            if i == 0:
                forces.append((I, -T[i] * I.z))

            if i == n - 1:
                forces.append((Bi, T[i] * I.z))
            else:
                forces.append((Bi, T[i] * I.z - T[i + 1] * I.z))

        kindiffs.append(q[i + 1].diff(t) - u[i + 1])

    if cart_force is True:
        F = me.dynamicsymbols('F')
        forces.append((P0, F * I.x))
        specified.append(F)

    kane = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
    kane.kanes_equations(particles, forces)

    return kane
<|endoftext|>from abc import ABC, abstractmethod
from sympy import Symbol, sympify
from sympy.physics.vector import Point

__all__ = ['BodyBase']


class BodyBase(ABC):
    """Abstract class for body type objects."""
    def __init__(self, name, masscenter=None, mass=None):
        # Note: If frame=None, no auto-generated frame is created, because a
        # Particle does not need to have a frame by default.
        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name
        if mass is None:
            mass = Symbol(f'{name}_mass')
        if masscenter is None:
            masscenter = Point(f'{name}_masscenter')
        self.mass = mass
        self.masscenter = masscenter
        self.potential_energy = 0
        self.points = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return (f'{self.__class__.__name__}({repr(self.name)}, masscenter='
                f'{repr(self.masscenter)}, mass={repr(self.mass)})')

    @property
    def name(self):
        """The name of the body."""
        return self._name

    @property
    def masscenter(self):
        """The body's center of mass."""
        return self._masscenter

    @masscenter.setter
    def masscenter(self, point):
        if not isinstance(point, Point):
            raise TypeError("The body's center of mass must be a Point object.")
        self._masscenter = point

    @property
    def mass(self):
        """The body's mass."""
        return self._mass

    @mass.setter
    def mass(self, mass):
        self._mass = sympify(mass)

    @property
    def potential_energy(self):
        """The potential energy of the body.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point
        >>> from sympy import symbols
        >>> m, g, h = symbols('m g h')
        >>> O = Point('O')
        >>> P = Particle('P', O, m)
        >>> P.potential_energy = m * g * h
        >>> P.potential_energy
        g*h*m

        """
        return self._potential_energy

    @potential_energy.setter
    def potential_energy(self, scalar):
        self._potential_energy = sympify(scalar)

    @abstractmethod
    def kinetic_energy(self, frame):
        pass

    @abstractmethod
    def linear_momentum(self, frame):
        pass

    @abstractmethod
    def angular_momentum(self, point, frame):
        pass

    @abstractmethod
    def parallel_axis(self, point, frame):
        pass
<|endoftext|># coding=utf-8

from abc import ABC, abstractmethod

from sympy import pi, Derivative, Matrix
from sympy.core.function import AppliedUndef
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
                                  ReferenceFrame)
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Joint', 'PinJoint', 'PrismaticJoint', 'CylindricalJoint',
           'PlanarJoint', 'SphericalJoint', 'WeldJoint']


class Joint(ABC):
    """Abstract base class for all specific joints.

    Explanation
    ===========

    A joint subtracts degrees of freedom from a body. This is the base class
    for all specific joints and holds all common methods acting as an interface
    for all joints. Custom joint can be created by inheriting Joint class and
    defining all abstract functions.

    The abstract methods are:

    - ``_generate_coordinates``
    - ``_generate_speeds``
    - ``_orient_frames``
    - ``_set_angular_velocity``
    - ``_set_linear_velocity``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    coordinates : iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Notes
    =====

    When providing a vector as the intermediate frame, a new intermediate frame
    is created which aligns its X axis with the provided vector. This is done
    with a single fixed rotation about a rotation axis. This rotation axis is
    determined by taking the cross product of the ``body.x`` axis with the
    provided vector. In the case where the provided vector is in the ``-body.x``
    direction, the rotation is done about the ``body.y`` axis.

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 parent_joint_pos=None, child_joint_pos=None):

        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name

        if not isinstance(parent, BodyBase):
            raise TypeError('Parent must be a body.')
        self._parent = parent

        if not isinstance(child, BodyBase):
            raise TypeError('Child must be a body.')
        self._child = child

        if parent_axis is not None or child_axis is not None:
            sympy_deprecation_warning(
                """
                The parent_axis and child_axis arguments for the Joint classes
                are deprecated. Instead use parent_interframe, child_interframe.
                """,
                deprecated_since_version="1.12",
                active_deprecations_target="deprecated-mechanics-joint-axis",
                stacklevel=4
            )
            if parent_interframe is None:
                parent_interframe = parent_axis
            if child_interframe is None:
                child_interframe = child_axis

        # Set parent and child frame attributes
        if hasattr(self._parent, 'frame'):
            self._parent_frame = self._parent.frame
        else:
            if isinstance(parent_interframe, ReferenceFrame):
                self._parent_frame = parent_interframe
            else:
                self._parent_frame = ReferenceFrame(
                    f'{self.name}_{self._parent.name}_frame')
        if hasattr(self._child, 'frame'):
            self._child_frame = self._child.frame
        else:
            if isinstance(child_interframe, ReferenceFrame):
                self._child_frame = child_interframe
            else:
                self._child_frame = ReferenceFrame(
                    f'{self.name}_{self._child.name}_frame')

        self._parent_interframe = self._locate_joint_frame(
            self._parent, parent_interframe, self._parent_frame)
        self._child_interframe = self._locate_joint_frame(
            self._child, child_interframe, self._child_frame)
        self._parent_axis = self._axis(parent_axis, self._parent_frame)
        self._child_axis = self._axis(child_axis, self._child_frame)

        if parent_joint_pos is not None or child_joint_pos is not None:
            sympy_deprecation_warning(
                """
                The parent_joint_pos and child_joint_pos arguments for the Joint
                classes are deprecated. Instead use parent_point and child_point.
                """,
                deprecated_since_version="1.12",
                active_deprecations_target="deprecated-mechanics-joint-pos",
                stacklevel=4
            )
            if parent_point is None:
                parent_point = parent_joint_pos
            if child_point is None:
                child_point = child_joint_pos
        self._parent_point = self._locate_joint_pos(
            self._parent, parent_point, self._parent_frame)
        self._child_point = self._locate_joint_pos(
            self._child, child_point, self._child_frame)

        self._coordinates = self._generate_coordinates(coordinates)
        self._speeds = self._generate_speeds(speeds)
        _validate_coordinates(self.coordinates, self.speeds)
        self._kdes = self._generate_kdes()

        self._orient_frames()
        self._set_angular_velocity()
        self._set_linear_velocity()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        """Name of the joint."""
        return self._name

    @property
    def parent(self):
        """Parent body of Joint."""
        return self._parent

    @property
    def child(self):
        """Child body of Joint."""
        return self._child

    @property
    def coordinates(self):
        """Matrix of the joint's generalized coordinates."""
        return self._coordinates

    @property
    def speeds(self):
        """Matrix of the joint's generalized speeds."""
        return self._speeds

    @property
    def kdes(self):
        """Kinematical differential equations of the joint."""
        return self._kdes

    @property
    def parent_axis(self):
        """The axis of parent frame."""
        # Will be removed with `deprecated-mechanics-joint-axis`
        return self._parent_axis

    @property
    def child_axis(self):
        """The axis of child frame."""
        # Will be removed with `deprecated-mechanics-joint-axis`
        return self._child_axis

    @property
    def parent_point(self):
        """Attachment point where the joint is fixed to the parent body."""
        return self._parent_point

    @property
    def child_point(self):
        """Attachment point where the joint is fixed to the child body."""
        return self._child_point

    @property
    def parent_interframe(self):
        return self._parent_interframe

    @property
    def child_interframe(self):
        return self._child_interframe

    @abstractmethod
    def _generate_coordinates(self, coordinates):
        """Generate Matrix of the joint's generalized coordinates."""
        pass

    @abstractmethod
    def _generate_speeds(self, speeds):
        """Generate Matrix of the joint's generalized speeds."""
        pass

    @abstractmethod
    def _orient_frames(self):
        """Orient frames as per the joint."""
        pass

    @abstractmethod
    def _set_angular_velocity(self):
        """Set angular velocity of the joint related frames."""
        pass

    @abstractmethod
    def _set_linear_velocity(self):
        """Set velocity of related points to the joint."""
        pass

    @staticmethod
    def _to_vector(matrix, frame):
        """Converts a matrix to a vector in the given frame."""
        return Vector([(matrix, frame)])

    @staticmethod
    def _axis(ax, *frames):
        """Check whether an axis is fixed in one of the frames."""
        if ax is None:
            ax = frames[0].x
            return ax
        if not isinstance(ax, Vector):
            raise TypeError("Axis must be a Vector.")
        ref_frame = None  # Find a body in which the axis can be expressed
        for frame in frames:
            try:
                ax.to_matrix(frame)
            except ValueError:
                pass
            else:
                ref_frame = frame
                break
        if ref_frame is None:
            raise ValueError("Axis cannot be expressed in one of the body's "
                             "frames.")
        if not ax.dt(ref_frame) == 0:
            raise ValueError('Axis cannot be time-varying when viewed from the '
                             'associated body.')
        return ax

    @staticmethod
    def _choose_rotation_axis(frame, axis):
        components = axis.to_matrix(frame)
        x, y, z = components[0], components[1], components[2]

        if x != 0:
            if y != 0:
                if z != 0:
                    return cross(axis, frame.x)
            if z != 0:
                return frame.y
            return frame.z
        else:
            if y != 0:
                return frame.x
            return frame.y

    @staticmethod
    def _create_aligned_interframe(frame, align_axis, frame_axis=None,
                                   frame_name=None):
        """
        Returns an intermediate frame, where the ``frame_axis`` defined in
        ``frame`` is aligned with ``axis``. By default this means that the X
        axis will be aligned with ``axis``.

        Parameters
        ==========

        frame : BodyBase or ReferenceFrame
            The body or reference frame with respect to which the intermediate
            frame is oriented.
        align_axis : Vector
            The vector with respect to which the intermediate frame will be
            aligned.
        frame_axis : Vector
            The vector of the frame which should get aligned with ``axis``. The
            default is the X axis of the frame.
        frame_name : string
            Name of the to be created intermediate frame. The default adds
            "_int_frame" to the name of ``frame``.

        Example
        =======

        An intermediate frame, where the X axis of the parent becomes aligned
        with ``parent.y + parent.z`` can be created as follows:

        >>> from sympy.physics.mechanics.joint import Joint
        >>> from sympy.physics.mechanics import RigidBody
        >>> parent = RigidBody('parent')
        >>> parent_interframe = Joint._create_aligned_interframe(
        ...     parent, parent.y + parent.z)
        >>> parent_interframe
        parent_int_frame
        >>> parent.frame.dcm(parent_interframe)
        Matrix([
        [        0, -sqrt(2)/2, -sqrt(2)/2],
        [sqrt(2)/2,        1/2,       -1/2],
        [sqrt(2)/2,       -1/2,        1/2]])
        >>> (parent.y + parent.z).express(parent_interframe)
        sqrt(2)*parent_int_frame.x

        Notes
        =====

        The direction cosine matrix between the given frame and intermediate
        frame is formed using a simple rotation about an axis that is normal to
        both ``align_axis`` and ``frame_axis``. In general, the normal axis is
        formed by crossing the ``frame_axis`` with the ``align_axis``. The
        exception is if the axes are parallel with opposite directions, in which
        case the rotation vector is chosen using the rules in the following
        table with the vectors expressed in the given frame:

        .. list-table::
           :header-rows: 1

           * - ``align_axis``
             - ``frame_axis``
             - ``rotation_axis``
           * - ``-x``
             - ``x``
             - ``z``
           * - ``-y``
             - ``y``
             - ``x``
           * - ``-z``
             - ``z``
             - ``y``
           * - ``-x-y``
             - ``x+y``
             - ``z``
           * - ``-y-z``
             - ``y+z``
             - ``x``
           * - ``-x-z``
             - ``x+z``
             - ``y``
           * - ``-x-y-z``
             - ``x+y+z``
             - ``(x+y+z) × x``

        """
        if isinstance(frame, BodyBase):
            frame = frame.frame
        if frame_axis is None:
            frame_axis = frame.x
        if frame_name is None:
            if frame.name[-6:] == '_frame':
                frame_name = f'{frame.name[:-6]}_int_frame'
            else:
                frame_name = f'{frame.name}_int_frame'
        angle = frame_axis.angle_between(align_axis)
        rotation_axis = cross(frame_axis, align_axis)
        if rotation_axis == Vector(0) and angle == 0:
            return frame
        if angle == pi:
            rotation_axis = Joint._choose_rotation_axis(frame, align_axis)

        int_frame = ReferenceFrame(frame_name)
        int_frame.orient_axis(frame, rotation_axis, angle)
        int_frame.set_ang_vel(frame, 0 * rotation_axis)
        return int_frame

    def _generate_kdes(self):
        """Generate kinematical differential equations."""
        kdes = []
        t = dynamicsymbols._t
        for i in range(len(self.coordinates)):
            kdes.append(-self.coordinates[i].diff(t) + self.speeds[i])
        return Matrix(kdes)

    def _locate_joint_pos(self, body, joint_pos, body_frame=None):
        """Returns the attachment point of a body."""
        if body_frame is None:
            body_frame = body.frame
        if joint_pos is None:
            return body.masscenter
        if not isinstance(joint_pos, (Point, Vector)):
            raise TypeError('Attachment point must be a Point or Vector.')
        if isinstance(joint_pos, Vector):
            point_name = f'{self.name}_{body.name}_joint'
            joint_pos = body.masscenter.locatenew(point_name, joint_pos)
        if not joint_pos.pos_from(body.masscenter).dt(body_frame) == 0:
            raise ValueError('Attachment point must be fixed to the associated '
                             'body.')
        return joint_pos

    def _locate_joint_frame(self, body, interframe, body_frame=None):
        """Returns the attachment frame of a body."""
        if body_frame is None:
            body_frame = body.frame
        if interframe is None:
            return body_frame
        if isinstance(interframe, Vector):
            interframe = Joint._create_aligned_interframe(
                body_frame, interframe,
                frame_name=f'{self.name}_{body.name}_int_frame')
        elif not isinstance(interframe, ReferenceFrame):
            raise TypeError('Interframe must be a ReferenceFrame.')
        if not interframe.ang_vel_in(body_frame) == 0:
            raise ValueError(f'Interframe {interframe} is not fixed to body '
                             f'{body}.')
        body.masscenter.set_vel(interframe, 0)  # Fixate interframe to body
        return interframe

    def _fill_coordinate_list(self, coordinates, n_coords, label='q', offset=0,
                              number_single=False):
        """Helper method for _generate_coordinates and _generate_speeds.

        Parameters
        ==========

        coordinates : iterable
            Iterable of coordinates or speeds that have been provided.
        n_coords : Integer
            Number of coordinates that should be returned.
        label : String, optional
            Coordinate type either 'q' (coordinates) or 'u' (speeds). The
            Default is 'q'.
        offset : Integer
            Count offset when creating new dynamicsymbols. The default is 0.
        number_single : Boolean
            Boolean whether if n_coords == 1, number should still be used. The
            default is False.

        """

        def create_symbol(number):
            if n_coords == 1 and not number_single:
                return dynamicsymbols(f'{label}_{self.name}')
            return dynamicsymbols(f'{label}{number}_{self.name}')

        name = 'generalized coordinate' if label == 'q' else 'generalized speed'
        generated_coordinates = []
        if coordinates is None:
            coordinates = []
        elif not iterable(coordinates):
            coordinates = [coordinates]
        if not (len(coordinates) == 0 or len(coordinates) == n_coords):
            raise ValueError(f'Expected {n_coords} {name}s, instead got '
                             f'{len(coordinates)} {name}s.')
        # Supports more iterables, also Matrix
        for i, coord in enumerate(coordinates):
            if coord is None:
                generated_coordinates.append(create_symbol(i + offset))
            elif isinstance(coord, (AppliedUndef, Derivative)):
                generated_coordinates.append(coord)
            else:
                raise TypeError(f'The {name} {coord} should have been a '
                                f'dynamicsymbol.')
        for i in range(len(coordinates) + offset, n_coords + offset):
            generated_coordinates.append(create_symbol(i))
        return Matrix(generated_coordinates)


class PinJoint(Joint):
    """Pin (Revolute) Joint.

    .. raw:: html
        :file: ../../../doc/src/explanation/modules/physics/mechanics/PinJoint.svg

    Explanation
    ===========

    A pin joint is defined such that the joint rotation axis is fixed in both
    the child and parent and the location of the joint is relative to the mass
    center of each body. The child rotates an angle, θ, from the parent about
    the rotation axis and has a simple angular speed, ω, relative to the
    parent. The direction cosine matrix between the child interframe and
    parent interframe is formed using a simple rotation about the joint axis.
    The page on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    coordinates : dynamicsymbol, optional
        Generalized coordinates of the joint.
    speeds : dynamicsymbol, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector
        The axis about which the rotation occurs. Note that the components
        of this axis are the same in the parent_interframe and child_interframe.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : Matrix
        Matrix of the joint's generalized speeds. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    joint_axis : Vector
        The axis about which the rotation occurs. Note that the components of
        this axis are the same in the parent_interframe and child_interframe.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single pin joint is created from two bodies and has the following basic
    attributes:

    >>> from sympy.physics.mechanics import RigidBody, PinJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PinJoint('PC', parent, child)
    >>> joint
    PinJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([[q_PC(t)]])
    >>> joint.speeds
    Matrix([[u_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,             0,            0],
    [0,  cos(q_PC(t)), sin(q_PC(t))],
    [0, -sin(q_PC(t)), cos(q_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the pin joint, the kinematics of simple
    double pendulum that rotates about the Z axis of each connected body can be
    created as follows.

    >>> from sympy import symbols, trigsimp
    >>> from sympy.physics.mechanics import RigidBody, PinJoint
    >>> l1, l2 = symbols('l1 l2')

    First create bodies to represent the fixed ceiling and one to represent
    each pendulum bob.

    >>> ceiling = RigidBody('C')
    >>> upper_bob = RigidBody('U')
    >>> lower_bob = RigidBody('L')

    The first joint will connect the upper bob to the ceiling by a distance of
    ``l1`` and the joint axis will be about the Z axis for each body.

    >>> ceiling_joint = PinJoint('P1', ceiling, upper_bob,
    ... child_point=-l1*upper_bob.frame.x,
    ... joint_axis=ceiling.frame.z)

    The second joint will connect the lower bob to the upper bob by a distance
    of ``l2`` and the joint axis will also be about the Z axis for each body.

    >>> pendulum_joint = PinJoint('P2', upper_bob, lower_bob,
    ... child_point=-l2*lower_bob.frame.x,
    ... joint_axis=upper_bob.frame.z)

    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of pendulum link relative
    to the ceiling are found:

    >>> upper_bob.frame.dcm(ceiling.frame)
    Matrix([
    [ cos(q_P1(t)), sin(q_P1(t)), 0],
    [-sin(q_P1(t)), cos(q_P1(t)), 0],
    [            0,            0, 1]])
    >>> trigsimp(lower_bob.frame.dcm(ceiling.frame))
    Matrix([
    [ cos(q_P1(t) + q_P2(t)), sin(q_P1(t) + q_P2(t)), 0],
    [-sin(q_P1(t) + q_P2(t)), cos(q_P1(t) + q_P2(t)), 0],
    [                      0,                      0, 1]])

    The position of the lower bob's masscenter is found with:

    >>> lower_bob.masscenter.pos_from(ceiling.masscenter)
    l1*U_frame.x + l2*L_frame.x

    The angular velocities of the two pendulum links can be computed with
    respect to the ceiling.

    >>> upper_bob.frame.ang_vel_in(ceiling.frame)
    u_P1(t)*C_frame.z
    >>> lower_bob.frame.ang_vel_in(ceiling.frame)
    u_P1(t)*C_frame.z + u_P2(t)*U_frame.z

    And finally, the linear velocities of the two pendulum bobs can be computed
    with respect to the ceiling.

    >>> upper_bob.masscenter.vel(ceiling.frame)
    l1*u_P1(t)*U_frame.y
    >>> lower_bob.masscenter.vel(ceiling.frame)
    l1*u_P1(t)*U_frame.y + l2*(u_P1(t) + u_P2(t))*L_frame.y

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 joint_axis=None, parent_joint_pos=None, child_joint_pos=None):

        self._joint_axis = joint_axis
        super().__init__(name, parent, child, coordinates, speeds, parent_point,
                         child_point, parent_interframe, child_interframe,
                         parent_axis, child_axis, parent_joint_pos,
                         child_joint_pos)

    def __str__(self):
        return (f'PinJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis about which the child rotates with respect to the parent."""
        return self._joint_axis

    def _generate_coordinates(self, coordinate):
        return self._fill_coordinate_list(coordinate, 1, 'q')

    def _generate_speeds(self, speed):
        return self._fill_coordinate_list(speed, 1, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, self.coordinates[0])

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, self.speeds[
            0] * self.joint_axis.normalize())

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.v2pt_theory(self.parent_point,
                                          self._parent_frame, self._child_frame)


class PrismaticJoint(Joint):
    """Prismatic (Sliding) Joint.

    .. image:: PrismaticJoint.svg

    Explanation
    ===========

    It is defined such that the child body translates with respect to the parent
    body along the body-fixed joint axis. The location of the joint is defined
    by two points, one in each body, which coincide when the generalized
    coordinate is zero. The direction cosine matrix between the
    parent_interframe and child_interframe is the identity matrix. Therefore,
    the direction cosine matrix between the parent and child frames is fully
    defined by the definition of the intermediate frames. The page on the joints
    framework gives a more detailed explanation of the intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    coordinates : dynamicsymbol, optional
        Generalized coordinates of the joint. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : dynamicsymbol, optional
        Generalized speeds of joint. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector
        The axis along which the translation occurs. Note that the components
        of this axis are the same in the parent_interframe and child_interframe.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single prismatic joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, PrismaticJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PrismaticJoint('PC', parent, child)
    >>> joint
    PrismaticJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([[q_PC(t)]])
    >>> joint.speeds
    Matrix([[u_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    0
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q_PC(t)*P_frame.x

    To further demonstrate the use of the prismatic joint, the kinematics of two
    masses sliding, one moving relative to a fixed body and the other relative
    to the moving body. about the X axis of each connected body can be created
    as follows.

    >>> from sympy.physics.mechanics import PrismaticJoint, RigidBody

    First create bodies to represent the fixed ceiling and one to represent
    a particle.

    >>> wall = RigidBody('W')
    >>> Part1 = RigidBody('P1')
    >>> Part2 = RigidBody('P2')

    The first joint will connect the particle to the ceiling and the
    joint axis will be about the X axis for each body.

    >>> J1 = PrismaticJoint('J1', wall, Part1)

    The second joint will connect the second particle to the first particle
    and the joint axis will also be about the X axis for each body.

    >>> J2 = PrismaticJoint('J2', Part1, Part2)

    Once the joint is established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of Part relative
    to the ceiling are found:

    >>> Part1.frame.dcm(wall.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    >>> Part2.frame.dcm(wall.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    The position of the particles' masscenter is found with:

    >>> Part1.masscenter.pos_from(wall.masscenter)
    q_J1(t)*W_frame.x

    >>> Part2.masscenter.pos_from(wall.masscenter)
    q_J1(t)*W_frame.x + q_J2(t)*P1_frame.x

    The angular velocities of the two particle links can be computed with
    respect to the ceiling.

    >>> Part1.frame.ang_vel_in(wall.frame)
    0

    >>> Part2.frame.ang_vel_in(wall.frame)
    0

    And finally, the linear velocities of the two particles can be computed
    with respect to the ceiling.

    >>> Part1.masscenter.vel(wall.frame)
    u_J1(t)*W_frame.x

    >>> Part2.masscenter.vel(wall.frame)
    u_J1(t)*W_frame.x + Derivative(q_J2(t), t)*P1_frame.x

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 joint_axis=None, parent_joint_pos=None, child_joint_pos=None):

        self._joint_axis = joint_axis
        super().__init__(name, parent, child, coordinates, speeds, parent_point,
                         child_point, parent_interframe, child_interframe,
                         parent_axis, child_axis, parent_joint_pos,
                         child_joint_pos)

    def __str__(self):
        return (f'PrismaticJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis along which the child translates with respect to the parent."""
        return self._joint_axis

    def _generate_coordinates(self, coordinate):
        return self._fill_coordinate_list(coordinate, 1, 'q')

    def _generate_speeds(self, speed):
        return self._fill_coordinate_list(speed, 1, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, 0)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, 0)

    def _set_linear_velocity(self):
        axis = self.joint_axis.normalize()
        self.child_point.set_pos(self.parent_point, self.coordinates[0] * axis)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child_point.set_vel(self._parent_frame, self.speeds[0] * axis)
        self.child.masscenter.set_vel(self._parent_frame, self.speeds[0] * axis)


class CylindricalJoint(Joint):
    """Cylindrical Joint.

    .. image:: CylindricalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A cylindrical joint is defined such that the child body both rotates about
    and translates along the body-fixed joint axis with respect to the parent
    body. The joint axis is both the rotation axis and translation axis. The
    location of the joint is defined by two points, one in each body, which
    coincide when the generalized coordinate corresponding to the translation is
    zero. The direction cosine matrix between the child interframe and parent
    interframe is formed using a simple rotation about the joint axis. The page
    on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    translation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the translation distance. The
        default value is ``dynamicsymbols(f'q1_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    translation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the translation velocity. The default
        value is ``dynamicsymbols(f'u1_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector, optional
        The rotation as well as translation axis. Note that the components of
        this axis are the same in the parent_interframe and child_interframe.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody
        The joint's parent body.
    child : Particle or RigidBody
        The joint's child body.
    rotation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the rotation angle.
    translation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the translation distance.
    rotation_speed : dynamicsymbol
        Generalized speed corresponding to the angular velocity.
    translation_speed : dynamicsymbol
        Generalized speed corresponding to the translation velocity.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.
    joint_axis : Vector
        The axis of rotation and translation.

    Examples
    =========

    A single cylindrical joint is created between two bodies and has the
    following basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, CylindricalJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = CylindricalJoint('PC', parent, child)
    >>> joint
    CylindricalJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u0_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.x
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.x

    To further demonstrate the use of the cylindrical joint, the kinematics of
    two cylindrical joints perpendicular to each other can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import RigidBody, CylindricalJoint
    >>> r, l, w = symbols('r l w')

    First create bodies to represent the fixed floor with a fixed pole on it.
    The second body represents a freely moving tube around that pole. The third
    body represents a solid flag freely translating along and rotating around
    the Y axis of the tube.

    >>> floor = RigidBody('floor')
    >>> tube = RigidBody('tube')
    >>> flag = RigidBody('flag')

    The first joint will connect the first tube to the floor with it translating
    along and rotating around the Z axis of both bodies.

    >>> floor_joint = CylindricalJoint('C1', floor, tube, joint_axis=floor.z)

    The second joint will connect the tube perpendicular to the flag along the Y
    axis of both the tube and the flag, with the joint located at a distance
    ``r`` from the tube's center of mass and a combination of the distances
    ``l`` and ``w`` from the flag's center of mass.

    >>> flag_joint = CylindricalJoint('C2', tube, flag,
    ...                               parent_point=r * tube.y,
    ...                               child_point=-w * flag.y + l * flag.z,
    ...                               joint_axis=tube.y)

