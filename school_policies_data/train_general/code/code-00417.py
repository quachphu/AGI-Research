            Axis along which moment is applied.
        order : Integer
            The order of the applied load.
            - For point moments, order=-2
            - For constant distributed moment, order=-1
            - For ramp moments, order=0
            - For parabolic ramp moments, order=1
            - ... so on.
        """
        x = self.variable
        value = sympify(value)
        start = sympify(start)
        order = sympify(order)

        if dir == "x":
            if not order == -2:
                self._moment_load_vector[0] += value
            else:
                if start in list(self._torsion_moment):
                    self._torsion_moment[start] += value
                else:
                    self._torsion_moment[start] = value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)
        elif dir == "y":
            if not order == -2:
                self._moment_load_vector[1] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)
        else:
            if not order == -2:
                self._moment_load_vector[2] += value
            self._load_Singularity[0] += value*SingularityFunction(x, start, order)

    def apply_support(self, loc, type="fixed"):
        if type in ("pin", "roller"):
            reaction_load = Symbol('R_'+str(loc))
            self._reaction_loads[reaction_load] = reaction_load
            self.bc_deflection.append((loc, [0, 0, 0]))
        else:
            reaction_load = Symbol('R_'+str(loc))
            reaction_moment = Symbol('M_'+str(loc))
            self._reaction_loads[reaction_load] = [reaction_load, reaction_moment]
            self.bc_deflection.append((loc, [0, 0, 0]))
            self.bc_slope.append((loc, [0, 0, 0]))

    def solve_for_reaction_loads(self, *reaction):
        """
        Solves for the reaction forces.

        Examples
        ========
        There is a beam of length 30 meters. It it supported by rollers at
        of its end. A constant distributed load of magnitude 8 N is applied
        from start till its end along y-axis. Another linear load having
        slope equal to 9 is applied along z-axis.

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(30, E, G, I, A, x)
        >>> b.apply_load(8, start=0, order=0, dir="y")
        >>> b.apply_load(9*x, start=0, order=0, dir="z")
        >>> b.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]
        >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
        >>> b.apply_load(R1, start=0, order=-1, dir="y")
        >>> b.apply_load(R2, start=30, order=-1, dir="y")
        >>> b.apply_load(R3, start=0, order=-1, dir="z")
        >>> b.apply_load(R4, start=30, order=-1, dir="z")
        >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
        >>> b.reaction_loads
        {R1: -120, R2: -120, R3: -1350, R4: -2700}
        """
        x = self.variable
        l = self.length
        q = self._load_Singularity
        shear_curves = [integrate(load, x) for load in q]
        moment_curves = [integrate(shear, x) for shear in shear_curves]
        for i in range(3):
            react = [r for r in reaction if (shear_curves[i].has(r) or moment_curves[i].has(r))]
            if len(react) == 0:
                continue
            shear_curve = limit(shear_curves[i], x, l)
            moment_curve = limit(moment_curves[i], x, l)
            sol = list((linsolve([shear_curve, moment_curve], react).args)[0])
            sol_dict = dict(zip(react, sol))
            reaction_loads = self._reaction_loads
            # Check if any of the evaluated reaction exists in another direction
            # and if it exists then it should have same value.
            for key in sol_dict:
                if key in reaction_loads and sol_dict[key] != reaction_loads[key]:
                    raise ValueError("Ambiguous solution for %s in different directions." % key)
            self._reaction_loads.update(sol_dict)

    def shear_force(self):
        """
        Returns a list of three expressions which represents the shear force
        curve of the Beam object along all three axes.
        """
        x = self.variable
        q = self._load_vector
        return [integrate(-q[0], x), integrate(-q[1], x), integrate(-q[2], x)]

    def axial_force(self):
        """
        Returns expression of Axial shear force present inside the Beam object.
        """
        return self.shear_force()[0]

    def shear_stress(self):
        """
        Returns a list of three expressions which represents the shear stress
        curve of the Beam object along all three axes.
        """
        return [self.shear_force()[0]/self._area, self.shear_force()[1]/self._area, self.shear_force()[2]/self._area]

    def axial_stress(self):
        """
        Returns expression of Axial stress present inside the Beam object.
        """
        return self.axial_force()/self._area

    def bending_moment(self):
        """
        Returns a list of three expressions which represents the bending moment
        curve of the Beam object along all three axes.
        """
        x = self.variable
        m = self._moment_load_vector
        shear = self.shear_force()

        return [integrate(-m[0], x), integrate(-m[1] + shear[2], x),
                integrate(-m[2] - shear[1], x) ]

    def torsional_moment(self):
        """
        Returns expression of Torsional moment present inside the Beam object.
        """
        return self.bending_moment()[0]

    def solve_for_torsion(self):
        """
        Solves for the angular deflection due to the torsional effects of
        moments being applied in the x-direction i.e. out of or into the beam.

        Here, a positive torque means the direction of the torque is positive
        i.e. out of the beam along the beam-axis. Likewise, a negative torque
        signifies a torque into the beam cross-section.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.beam import Beam3D
        >>> from sympy import symbols
        >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
        >>> b = Beam3D(20, E, G, I, A, x)
        >>> b.apply_moment_load(4, 4, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.apply_moment_load(4, 8, -2, dir='x')
        >>> b.solve_for_torsion()
        >>> b.angular_deflection().subs(x, 3)
        18/(G*I)
        """
        x = self.variable
        sum_moments = 0
        for point in list(self._torsion_moment):
            sum_moments += self._torsion_moment[point]
        list(self._torsion_moment).sort()
        pointsList = list(self._torsion_moment)
        torque_diagram = Piecewise((sum_moments, x<=pointsList[0]), (0, x>=pointsList[0]))
        for i in range(len(pointsList))[1:]:
            sum_moments -= self._torsion_moment[pointsList[i-1]]
            torque_diagram += Piecewise((0, x<=pointsList[i-1]), (sum_moments, x<=pointsList[i]), (0, x>=pointsList[i]))
        integrated_torque_diagram = integrate(torque_diagram)
        self._angular_deflection =  integrated_torque_diagram/(self.shear_modulus*self.polar_moment())

    def solve_slope_deflection(self):
        x = self.variable
        l = self.length
        E = self.elastic_modulus
        G = self.shear_modulus
        I = self.second_moment
        if isinstance(I, list):
            I_y, I_z = I[0], I[1]
        else:
            I_y = I_z = I
        A = self._area
        load = self._load_vector
        moment = self._moment_load_vector
        defl = Function('defl')
        theta = Function('theta')

        # Finding deflection along x-axis(and corresponding slope value by differentiating it)
        # Equation used: Derivative(E*A*Derivative(def_x(x), x), x) + load_x = 0
        eq = Derivative(E*A*Derivative(defl(x), x), x) + load[0]
        def_x = dsolve(Eq(eq, 0), defl(x)).args[1]
        # Solving constants originated from dsolve
        C1 = Symbol('C1')
        C2 = Symbol('C2')
        constants = list((linsolve([def_x.subs(x, 0), def_x.subs(x, l)], C1, C2).args)[0])
        def_x = def_x.subs({C1:constants[0], C2:constants[1]})
        slope_x = def_x.diff(x)
        self._deflection[0] = def_x
        self._slope[0] = slope_x

        # Finding deflection along y-axis and slope across z-axis. System of equation involved:
        # 1: Derivative(E*I_z*Derivative(theta_z(x), x), x) + G*A*(Derivative(defl_y(x), x) - theta_z(x)) + moment_z = 0
        # 2: Derivative(G*A*(Derivative(defl_y(x), x) - theta_z(x)), x) + load_y = 0
        C_i = Symbol('C_i')
        # Substitute value of `G*A*(Derivative(defl_y(x), x) - theta_z(x))` from (2) in (1)
        eq1 = Derivative(E*I_z*Derivative(theta(x), x), x) + (integrate(-load[1], x) + C_i) + moment[2]
        slope_z = dsolve(Eq(eq1, 0)).args[1]

        # Solve for constants originated from using dsolve on eq1
        constants = list((linsolve([slope_z.subs(x, 0), slope_z.subs(x, l)], C1, C2).args)[0])
        slope_z = slope_z.subs({C1:constants[0], C2:constants[1]})

        # Put value of slope obtained back in (2) to solve for `C_i` and find deflection across y-axis
        eq2 = G*A*(Derivative(defl(x), x)) + load[1]*x - C_i - G*A*slope_z
        def_y = dsolve(Eq(eq2, 0), defl(x)).args[1]
        # Solve for constants originated from using dsolve on eq2
        constants = list((linsolve([def_y.subs(x, 0), def_y.subs(x, l)], C1, C_i).args)[0])
        self._deflection[1] = def_y.subs({C1:constants[0], C_i:constants[1]})
        self._slope[2] = slope_z.subs(C_i, constants[1])

        # Finding deflection along z-axis and slope across y-axis. System of equation involved:
        # 1: Derivative(E*I_y*Derivative(theta_y(x), x), x) - G*A*(Derivative(defl_z(x), x) + theta_y(x)) + moment_y = 0
        # 2: Derivative(G*A*(Derivative(defl_z(x), x) + theta_y(x)), x) + load_z = 0

        # Substitute value of `G*A*(Derivative(defl_y(x), x) + theta_z(x))` from (2) in (1)
        eq1 = Derivative(E*I_y*Derivative(theta(x), x), x) + (integrate(load[2], x) - C_i) + moment[1]
        slope_y = dsolve(Eq(eq1, 0)).args[1]
        # Solve for constants originated from using dsolve on eq1
        constants = list((linsolve([slope_y.subs(x, 0), slope_y.subs(x, l)], C1, C2).args)[0])
        slope_y = slope_y.subs({C1:constants[0], C2:constants[1]})

        # Put value of slope obtained back in (2) to solve for `C_i` and find deflection across z-axis
        eq2 = G*A*(Derivative(defl(x), x)) + load[2]*x - C_i + G*A*slope_y
        def_z = dsolve(Eq(eq2,0)).args[1]
        # Solve for constants originated from using dsolve on eq2
        constants = list((linsolve([def_z.subs(x, 0), def_z.subs(x, l)], C1, C_i).args)[0])
        self._deflection[2] = def_z.subs({C1:constants[0], C_i:constants[1]})
        self._slope[1] = slope_y.subs(C_i, constants[1])

    def slope(self):
        """
        Returns a three element list representing slope of deflection curve
        along all the three axes.
        """
        return self._slope

    def deflection(self):
        """
        Returns a three element list representing deflection curve along all
        the three axes.
        """
        return self._deflection

    def angular_deflection(self):
        """
        Returns a function in x depicting how the angular deflection, due to moments
        in the x-axis on the beam, varies with x.
        """
        return self._angular_deflection

    def _plot_shear_force(self, dir, subs=None):

        shear_force = self.shear_force()

        if dir == 'x':
            dir_num = 0
            color = 'r'

        elif dir == 'y':
            dir_num = 1
            color = 'g'

        elif dir == 'z':
            dir_num = 2
            color = 'b'

        if subs is None:
            subs = {}

        for sym in shear_force[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(shear_force[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Shear Force along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{V(%c)}$'%dir, line_color=color)

    def plot_shear_force(self, dir="all", subs=None):

        """

        Returns a plot for Shear force along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear force plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_force()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For shear force along x direction
        if dir == "x":
            Px = self._plot_shear_force('x', subs)
            return Px.show()
        # For shear force along y direction
        elif dir == "y":
            Py = self._plot_shear_force('y', subs)
            return Py.show()
        # For shear force along z direction
        elif dir == "z":
            Pz = self._plot_shear_force('z', subs)
            return Pz.show()
        # For shear force along all direction
        else:
            Px = self._plot_shear_force('x', subs)
            Py = self._plot_shear_force('y', subs)
            Pz = self._plot_shear_force('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_bending_moment(self, dir, subs=None):

        bending_moment = self.bending_moment()

        if dir == 'x':
            dir_num = 0
            color = 'g'

        elif dir == 'y':
            dir_num = 1
            color = 'c'

        elif dir == 'z':
            dir_num = 2
            color = 'm'

        if subs is None:
            subs = {}

        for sym in bending_moment[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(bending_moment[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Bending Moment along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{M(%c)}$'%dir, line_color=color)

    def plot_bending_moment(self, dir="all", subs=None):

        """

        Returns a plot for bending moment along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which bending moment plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_bending_moment()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: 2*x**3 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For bending moment along x direction
        if dir == "x":
            Px = self._plot_bending_moment('x', subs)
            return Px.show()
        # For bending moment along y direction
        elif dir == "y":
            Py = self._plot_bending_moment('y', subs)
            return Py.show()
        # For bending moment along z direction
        elif dir == "z":
            Pz = self._plot_bending_moment('z', subs)
            return Pz.show()
        # For bending moment along all direction
        else:
            Px = self._plot_bending_moment('x', subs)
            Py = self._plot_bending_moment('y', subs)
            Pz = self._plot_bending_moment('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_slope(self, dir, subs=None):

        slope = self.slope()

        if dir == 'x':
            dir_num = 0
            color = 'b'

        elif dir == 'y':
            dir_num = 1
            color = 'm'

        elif dir == 'z':
            dir_num = 2
            color = 'g'

        if subs is None:
            subs = {}

        for sym in slope[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length


        return plot(slope[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Slope along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{\theta(%c)}$'%dir, line_color=color)

    def plot_slope(self, dir="all", subs=None):

        """

        Returns a plot for Slope along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which Slope plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_slope()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/8000 - 19*x**2/172 + 52*x/43 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For Slope along x direction
        if dir == "x":
            Px = self._plot_slope('x', subs)
            return Px.show()
        # For Slope along y direction
        elif dir == "y":
            Py = self._plot_slope('y', subs)
            return Py.show()
        # For Slope along z direction
        elif dir == "z":
            Pz = self._plot_slope('z', subs)
            return Pz.show()
        # For Slope along all direction
        else:
            Px = self._plot_slope('x', subs)
            Py = self._plot_slope('y', subs)
            Pz = self._plot_slope('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _plot_deflection(self, dir, subs=None):

        deflection = self.deflection()

        if dir == 'x':
            dir_num = 0
            color = 'm'

        elif dir == 'y':
            dir_num = 1
            color = 'r'

        elif dir == 'z':
            dir_num = 2
            color = 'c'

        if subs is None:
            subs = {}

        for sym in deflection[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(deflection[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Deflection along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\mathrm{\delta(%c)}$'%dir, line_color=color)

    def plot_deflection(self, dir="all", subs=None):

        """

        Returns a plot for Deflection along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which deflection plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as keys and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_deflection()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: x**4/6400 - x**3/160 + 27*x**2/560 + 2*x/7 for x over (0.0, 20.0)


        """

        dir = dir.lower()
        # For deflection along x direction
        if dir == "x":
            Px = self._plot_deflection('x', subs)
            return Px.show()
        # For deflection along y direction
        elif dir == "y":
            Py = self._plot_deflection('y', subs)
            return Py.show()
        # For deflection along z direction
        elif dir == "z":
            Pz = self._plot_deflection('z', subs)
            return Pz.show()
        # For deflection along all direction
        else:
            Px = self._plot_deflection('x', subs)
            Py = self._plot_deflection('y', subs)
            Pz = self._plot_deflection('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def plot_loading_results(self, dir='x', subs=None):

        """

        Returns a subplot of Shear Force, Bending Moment,
        Slope and Deflection of the Beam object along the direction specified.

        Parameters
        ==========

        dir : string (default : "x")
               Direction along which plots are required.
               If no direction is specified, plots along x-axis are displayed.
        subs : dictionary
               Python dictionary containing Symbols as key and their
               corresponding values.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, A, x)
            >>> subs = {E:40, G:21, I:100, A:25}
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.plot_loading_results('y',subs)
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: -6*x**2 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -15*x**2/2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -x**3/1600 + 3*x**2/160 - x/8 for x over (0.0, 20.0)
            Plot[3]:Plot object containing:
            [0]: cartesian line: x**5/40000 - 4013*x**3/90300 + 26*x**2/43 + 1520*x/903 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        if subs is None:
            subs = {}

        ax1 = self._plot_shear_force(dir, subs)
        ax2 = self._plot_bending_moment(dir, subs)
        ax3 = self._plot_slope(dir, subs)
        ax4 = self._plot_deflection(dir, subs)

        return PlotGrid(4, 1, ax1, ax2, ax3, ax4)

    def _plot_shear_stress(self, dir, subs=None):

        shear_stress = self.shear_stress()

        if dir == 'x':
            dir_num = 0
            color = 'r'

        elif dir == 'y':
            dir_num = 1
            color = 'g'

        elif dir == 'z':
            dir_num = 2
            color = 'b'

        if subs is None:
            subs = {}

        for sym in shear_stress[dir_num].atoms(Symbol):
            if sym != self.variable and sym not in subs:
                raise ValueError('Value of %s was not passed.' %sym)
        if self.length in subs:
            length = subs[self.length]
        else:
            length = self.length

        return plot(shear_stress[dir_num].subs(subs), (self.variable, 0, length), show = False, title='Shear stress along %c direction'%dir,
                xlabel=r'$\mathrm{X}$', ylabel=r'$\tau(%c)$'%dir, line_color=color)

    def plot_shear_stress(self, dir="all", subs=None):

        """

        Returns a plot for Shear Stress along all three directions
        present in the Beam object.

        Parameters
        ==========
        dir : string (default : "all")
            Direction along which shear stress plot is required.
            If no direction is specified, all plots are displayed.
        subs : dictionary
            Python dictionary containing Symbols as key and their
            corresponding values.

        Examples
        ========
        There is a beam of length 20 meters and area of cross section 2 square
        meters. It is supported by rollers at both of its ends. A linear load having
        slope equal to 12 is applied along y-axis. A constant distributed load
        of magnitude 15 N is applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, E, G, I, 2, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.plot_shear_stress()
            PlotGrid object containing:
            Plot[0]:Plot object containing:
            [0]: cartesian line: 0 for x over (0.0, 20.0)
            Plot[1]:Plot object containing:
            [0]: cartesian line: -3*x**2 for x over (0.0, 20.0)
            Plot[2]:Plot object containing:
            [0]: cartesian line: -15*x/2 for x over (0.0, 20.0)

        """

        dir = dir.lower()
        # For shear stress along x direction
        if dir == "x":
            Px = self._plot_shear_stress('x', subs)
            return Px.show()
        # For shear stress along y direction
        elif dir == "y":
            Py = self._plot_shear_stress('y', subs)
            return Py.show()
        # For shear stress along z direction
        elif dir == "z":
            Pz = self._plot_shear_stress('z', subs)
            return Pz.show()
        # For shear stress along all direction
        else:
            Px = self._plot_shear_stress('x', subs)
            Py = self._plot_shear_stress('y', subs)
            Pz = self._plot_shear_stress('z', subs)
            return PlotGrid(3, 1, Px, Py, Pz)

    def _max_shear_force(self, dir):
        """
        Helper function for max_shear_force().
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.shear_force()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        load_curve = Piecewise((float("nan"), self.variable<=0),
                (self._load_vector[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(load_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self.length)
        shear_curve = self.shear_force()[dir_num]
        shear_values = [shear_curve.subs(self.variable, x) for x in points]
        shear_values = list(map(abs, shear_values))

        max_shear = max(shear_values)
        return (points[shear_values.index(max_shear)], max_shear)

    def max_shear_force(self):
        """
        Returns point of max shear force and its corresponding shear value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_shear_force()
            [(0, 0), (20, 2400), (20, 300)]
        """

        max_shear = []
        max_shear.append(self._max_shear_force('x'))
        max_shear.append(self._max_shear_force('y'))
        max_shear.append(self._max_shear_force('z'))
        return max_shear

    def _max_bending_moment(self, dir):
        """
        Helper function for max_bending_moment().
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.bending_moment()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        shear_curve = Piecewise((float("nan"), self.variable<=0),
                (self.shear_force()[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(shear_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self.length)
        bending_moment_curve = self.bending_moment()[dir_num]
        bending_moments = [bending_moment_curve.subs(self.variable, x) for x in points]
        bending_moments = list(map(abs, bending_moments))

        max_bending_moment = max(bending_moments)
        return (points[bending_moments.index(max_bending_moment)], max_bending_moment)

    def max_bending_moment(self):
        """
        Returns point of max bending moment and its corresponding bending moment value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() must be called before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.max_bending_moment()
            [(0, 0), (20, 3000), (20, 16000)]
        """

        max_bmoment = []
        max_bmoment.append(self._max_bending_moment('x'))
        max_bmoment.append(self._max_bending_moment('y'))
        max_bmoment.append(self._max_bending_moment('z'))
        return max_bmoment

    max_bmoment = max_bending_moment

    def _max_deflection(self, dir):
        """
        Helper function for max_Deflection()
        """

        dir = dir.lower()

        if dir == 'x':
            dir_num = 0

        elif dir == 'y':
            dir_num = 1

        elif dir == 'z':
            dir_num = 2

        if not self.deflection()[dir_num]:
            return (0,0)
        # To restrict the range within length of the Beam
        slope_curve = Piecewise((float("nan"), self.variable<=0),
                (self.slope()[dir_num], self.variable<self.length),
                (float("nan"), True))

        points = solve(slope_curve.rewrite(Piecewise), self.variable,
                        domain=S.Reals)
        points.append(0)
        points.append(self._length)
        deflection_curve = self.deflection()[dir_num]
        deflections = [deflection_curve.subs(self.variable, x) for x in points]
        deflections = list(map(abs, deflections))

        max_def = max(deflections)
        return (points[deflections.index(max_def)], max_def)

    def max_deflection(self):
        """
        Returns point of max deflection and its corresponding deflection value
        along all directions in a Beam object as a list.
        solve_for_reaction_loads() and solve_slope_deflection() must be called
        before using this function.

        Examples
        ========
        There is a beam of length 20 meters. It is supported by rollers
        at both of its ends. A linear load having slope equal to 12 is applied
        along y-axis. A constant distributed load of magnitude 15 N is
        applied from start till its end along z-axis.

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.beam import Beam3D
            >>> from sympy import symbols
            >>> l, E, G, I, A, x = symbols('l, E, G, I, A, x')
            >>> b = Beam3D(20, 40, 21, 100, 25, x)
            >>> b.apply_load(15, start=0, order=0, dir="z")
            >>> b.apply_load(12*x, start=0, order=0, dir="y")
            >>> b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
            >>> R1, R2, R3, R4 = symbols('R1, R2, R3, R4')
            >>> b.apply_load(R1, start=0, order=-1, dir="z")
            >>> b.apply_load(R2, start=20, order=-1, dir="z")
            >>> b.apply_load(R3, start=0, order=-1, dir="y")
            >>> b.apply_load(R4, start=20, order=-1, dir="y")
            >>> b.solve_for_reaction_loads(R1, R2, R3, R4)
            >>> b.solve_slope_deflection()
            >>> b.max_deflection()
            [(0, 0), (10, 495/14), (-10 + 10*sqrt(10793)/43, (10 - 10*sqrt(10793)/43)**3/160 - 20/7 + (10 - 10*sqrt(10793)/43)**4/6400 + 20*sqrt(10793)/301 + 27*(10 - 10*sqrt(10793)/43)**2/560)]
        """

        max_def = []
        max_def.append(self._max_deflection('x'))
        max_def.append(self._max_deflection('y'))
        max_def.append(self._max_deflection('z'))
        return max_def
<|endoftext|>"""
This module can be used to solve problems related
to 2D Cables.
"""

from sympy.core.sympify import sympify
from sympy.core.symbol import Symbol,symbols
from sympy import sin, cos, pi, atan, diff, Piecewise, solve, rad
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.solvers.solveset import linsolve
from sympy.matrices import Matrix
from sympy.plotting import plot

class Cable:
    """
    Cables are structures in engineering that support
    the applied transverse loads through the tensile
    resistance developed in its members.

    Cables are widely used in suspension bridges, tension
    leg offshore platforms, transmission lines, and find
    use in several other engineering applications.

    Examples
    ========
    A cable is supported at (0, 10) and (10, 10). Two point loads
    acting vertically downwards act on the cable, one with magnitude 3 kN
    and acting 2 meters from the left support and 3 meters below it, while
    the other with magnitude 2 kN is 6 meters from the left support and
    6 meters below it.

    >>> from sympy.physics.continuum_mechanics.cable import Cable
    >>> c = Cable(('A', 0, 10), ('B', 10, 10))
    >>> c.apply_load(-1, ('P', 2, 7, 3, 270))
    >>> c.apply_load(-1, ('Q', 6, 4, 2, 270))
    >>> c.loads
    {'distributed': {}, 'point_load': {'P': [3, 270], 'Q': [2, 270]}}
    >>> c.loads_position
    {'P': [2, 7], 'Q': [6, 4]}
    """
    def __init__(self, support_1, support_2):
        """
        Initializes the class.

        Parameters
        ==========

        support_1 and support_2 are tuples of the form
        (label, x, y), where

        label : String or symbol
            The label of the support

        x : Sympifyable
            The x coordinate of the position of the support

        y : Sympifyable
            The y coordinate of the position of the support
        """
        self._left_support = []
        self._right_support = []
        self._supports = {}
        self._support_labels = []
        self._loads = {"distributed": {}, "point_load": {}}
        self._loads_position = {}
        self._length = 0
        self._reaction_loads = {}
        self._tension = {}
        self._lowest_x_global = sympify(0)
        self._lowest_y_global = sympify(0)
        self._cable_eqn = None
        self._tension_func = None
        if support_1[0] == support_2[0]:
            raise ValueError("Supports can not have the same label")

        elif support_1[1] == support_2[1]:
            raise ValueError("Supports can not be at the same location")

        x1 = sympify(support_1[1])
        y1 = sympify(support_1[2])
        self._supports[support_1[0]] = [x1, y1]

        x2 = sympify(support_2[1])
        y2 = sympify(support_2[2])
        self._supports[support_2[0]] = [x2, y2]

        if support_1[1] < support_2[1]:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x2)
            self._right_support.append(y2)
            self._support_labels.append(support_1[0])
            self._support_labels.append(support_2[0])

        else:
            self._left_support.append(x2)
            self._left_support.append(y2)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.append(support_2[0])
            self._support_labels.append(support_1[0])

        for i in self._support_labels:
            self._reaction_loads[Symbol("R_"+ i +"_x")] = 0
            self._reaction_loads[Symbol("R_"+ i +"_y")] = 0

    @property
    def supports(self):
        """
        Returns the supports of the cable along with their
        positions.
        """
        return self._supports

    @property
    def left_support(self):
        """
        Returns the position of the left support.
        """
        return self._left_support

    @property
    def right_support(self):
        """
        Returns the position of the right support.
        """
        return self._right_support

    @property
    def loads(self):
        """
        Returns the magnitude and direction of the loads
        acting on the cable.
        """
        return self._loads

    @property
    def loads_position(self):
        """
        Returns the position of the point loads acting on the
        cable.
        """
        return self._loads_position

    @property
    def length(self):
        """
        Returns the length of the cable.
        """
        return self._length

    @property
    def reaction_loads(self):
        """
        Returns the reaction forces at the supports, which are
        initialized to 0.
        """
        return self._reaction_loads

    @property
    def tension(self):
        """
        Returns the tension developed in the cable due to the loads
        applied.
        """
        return self._tension

    def tension_at(self, x):
        """
        Returns the tension at a given value of x developed due to
        distributed load.
        """
        if 'distributed' not in self._tension.keys():
            raise ValueError("No distributed load added or solve method not called")

        if x > self._right_support[0] or x < self._left_support[0]:
            raise ValueError("The value of x should be between the two supports")

        A = self._tension['distributed']
        X = Symbol('X')

        return A.subs({X:(x-self._lowest_x_global)})

    def apply_length(self, length):
        """
        This method specifies the length of the cable

        Parameters
        ==========

        length : Sympifyable
            The length of the cable

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_length(20)
        >>> c.length
        20
        """
        dist = ((self._left_support[0] - self._right_support[0])**2
                - (self._left_support[1] - self._right_support[1])**2)**(1/2)

        if length < dist:
            raise ValueError("length should not be less than the distance between the supports")

        self._length = length

    def change_support(self, label, new_support):
        """
        This method changes the mentioned support with a new support.

        Parameters
        ==========
        label: String or symbol
            The label of the support to be changed

        new_support: Tuple of the form (new_label, x, y)
            new_label: String or symbol
                The label of the new support

            x: Sympifyable
                The x-coordinate of the position of the new support.

            y: Sympifyable
                The y-coordinate of the position of the new support.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.supports
        {'A': [0, 10], 'B': [10, 10]}
        >>> c.change_support('B', ('C', 5, 6))
        >>> c.supports
        {'A': [0, 10], 'C': [5, 6]}
        """
        if label not in self._supports:
            raise ValueError("No support exists with the given label")

        i = self._support_labels.index(label)
        rem_label = self._support_labels[(i+1)%2]
        x1 = self._supports[rem_label][0]
        y1 = self._supports[rem_label][1]

        x = sympify(new_support[1])
        y = sympify(new_support[2])

        for l in self._loads_position:
            if l[0] >= max(x, x1) or l[0] <= min(x, x1):
                raise ValueError("The change in support will throw an existing load out of range")

        self._supports.pop(label)
        self._left_support.clear()
        self._right_support.clear()
        self._reaction_loads.clear()
        self._support_labels.remove(label)

        self._supports[new_support[0]] = [x, y]

        if x1 < x:
            self._left_support.append(x1)
            self._left_support.append(y1)
            self._right_support.append(x)
            self._right_support.append(y)
            self._support_labels.append(new_support[0])

        else:
            self._left_support.append(x)
            self._left_support.append(y)
            self._right_support.append(x1)
            self._right_support.append(y1)
            self._support_labels.insert(0, new_support[0])

        for i in self._support_labels:
            self._reaction_loads[Symbol("R_"+ i +"_x")] = 0
            self._reaction_loads[Symbol("R_"+ i +"_y")] = 0

    def apply_load(self, order, load):
        """
        This method adds load to the cable.

        Parameters
        ==========

        order : Integer
            The order of the applied load.

                - For point loads, order = -1
                - For distributed load, order = 0

        load : tuple

            * For point loads, load is of the form (label, x, y, magnitude, direction), where:

            label : String or symbol
                The label of the load

            x : Sympifyable
                The x coordinate of the position of the load

            y : Sympifyable
                The y coordinate of the position of the load

            magnitude : Sympifyable
                The magnitude of the load. It must always be positive

            direction : Sympifyable
                The angle, in degrees, that the load vector makes with the horizontal
                in the counter-clockwise direction. It takes the values 0 to 360,
                inclusive.


            * For uniformly distributed load, load is of the form (label, magnitude)

            label : String or symbol
                The label of the load

            magnitude : Sympifyable
                The magnitude of the load. It must always be positive

        Examples
        ========

        For a point load of magnitude 12 units inclined at 30 degrees with the horizontal:

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.loads_position
        {'Z': [5, 5]}


        For a uniformly distributed load of magnitude 9 units:

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(0, ('X', 9))
        >>> c.loads
        {'distributed': {'X': 9}, 'point_load': {}}
        """
        if order == -1:
            if len(self._loads["distributed"]) != 0:
                raise ValueError("Distributed load already exists")

            label = load[0]
            if label in self._loads["point_load"]:
                raise ValueError("Label already exists")

            x = sympify(load[1])
            y = sympify(load[2])

            if x > self._right_support[0] or x < self._left_support[0]:
                raise ValueError("The load should be positioned between the supports")

            magnitude = sympify(load[3])
            direction = sympify(load[4])

            self._loads["point_load"][label] = [magnitude, direction]
            self._loads_position[label] = [x, y]

        elif order == 0:
            if len(self._loads_position) != 0:
                raise ValueError("Point load(s) already exist")

            label = load[0]
            if label in self._loads["distributed"]:
                raise ValueError("Label already exists")

            magnitude = sympify(load[1])

            self._loads["distributed"][label] = magnitude

        else:
            raise ValueError("Order should be either -1 or 0")

    def remove_loads(self, *args):
        """
        This methods removes the specified loads.

        Parameters
        ==========
        This input takes multiple label(s) as input
        label(s): String or symbol
            The label(s) of the loads to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(('A', 0, 10), ('B', 10, 10))
        >>> c.apply_load(-1, ('Z', 5, 5, 12, 30))
        >>> c.loads
        {'distributed': {}, 'point_load': {'Z': [12, 30]}}
        >>> c.remove_loads('Z')
        >>> c.loads
        {'distributed': {}, 'point_load': {}}
        """
        for i in args:
            if len(self._loads_position) == 0:
                if i not in self._loads['distributed']:
                    raise ValueError("Error removing load " + i + ": no such load exists")

                else:
                    self._loads['disrtibuted'].pop(i)

            else:
                if i not in self._loads['point_load']:
                    raise ValueError("Error removing load " + i + ": no such load exists")

                else:
                    self._loads['point_load'].pop(i)
                    self._loads_position.pop(i)

    def solve(self, *args):
        """
        This method solves for the reaction forces at the supports, the tension developed in
        the cable, and updates the length of the cable.

        Parameters
        ==========
        This method requires no input when solving for point loads
        For distributed load, the x and y coordinates of the lowest point of the cable are
        required as

        x: Sympifyable
            The x coordinate of the lowest point

        y: Sympifyable
            The y coordinate of the lowest point

        Examples
        ========
        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> c.tension
        {A_Z: 8.91403453669861, X_B: 19*sqrt(13)/10, Z_X: 4.79150773600774}
        >>> c.reaction_loads
        {R_A_x: -5.25547445255474, R_A_y: 7.2, R_B_x: 5.25547445255474, R_B_y: 3.8}
        >>> c.length
        5.7560958484519 + 2*sqrt(13)

        For distributed load,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58)
        >>> c.tension
        {'distributed': 36465.0*sqrt(0.00054335718671383*X**2 + 1)}
        >>> c.tension_at(0)
        61717.4130533677
        >>> c.reaction_loads
        {R_A_x: 36465.0, R_A_y: -49793.0, R_B_x: 44399.9537590861, R_B_y: 42868.2071025955}
        """

        if len(self._loads_position) != 0:
            sorted_position = sorted(self._loads_position.items(), key = lambda item : item[1][0])

            sorted_position.append(self._support_labels[1])
            sorted_position.insert(0, self._support_labels[0])

            self._tension.clear()
            moment_sum_from_left_support = 0
            moment_sum_from_right_support = 0
            F_x = 0
            F_y = 0
            self._length = 0
            tension_func = []
            x = symbols('x')
            for i in range(1, len(sorted_position)-1):
                if i == 1:
                    self._length+=sqrt((self._left_support[0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._left_support[1] - self._loads_position[sorted_position[i][0]][1])**2)

                else:
                    self._length+=sqrt((self._loads_position[sorted_position[i-1][0]][0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._loads_position[sorted_position[i-1][0]][1] - self._loads_position[sorted_position[i][0]][1])**2)

                if i == len(sorted_position)-2:
                    self._length+=sqrt((self._right_support[0] - self._loads_position[sorted_position[i][0]][0])**2 + (self._right_support[1] - self._loads_position[sorted_position[i][0]][1])**2)

                moment_sum_from_left_support += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._left_support[1] - self._loads_position[sorted_position[i][0]][1])
                moment_sum_from_left_support += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._left_support[0] - self._loads_position[sorted_position[i][0]][0])

                F_x += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180)
                F_y += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180)

                label = Symbol(sorted_position[i][0]+"_"+sorted_position[i+1][0])
                y2 = self._loads_position[sorted_position[i][0]][1]
                x2 = self._loads_position[sorted_position[i][0]][0]
                y1 = 0
                x1 = 0

                if i == len(sorted_position)-2:
                    x1 = self._right_support[0]
                    y1 = self._right_support[1]

                else:
                    x1 = self._loads_position[sorted_position[i+1][0]][0]
                    y1 = self._loads_position[sorted_position[i+1][0]][1]

                angle_with_horizontal = atan((y1 - y2)/(x1 - x2))

                tension = -(moment_sum_from_left_support)/(abs(self._left_support[1] - self._loads_position[sorted_position[i][0]][1])*cos(angle_with_horizontal) + abs(self._left_support[0] - self._loads_position[sorted_position[i][0]][0])*sin(angle_with_horizontal))
                self._tension[label] = tension
                tension_func.append((tension, x<=x1))
                moment_sum_from_right_support += self._loads['point_load'][sorted_position[i][0]][0] * cos(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._right_support[1] - self._loads_position[sorted_position[i][0]][1])
                moment_sum_from_right_support += self._loads['point_load'][sorted_position[i][0]][0] * sin(pi * self._loads['point_load'][sorted_position[i][0]][1] / 180) * abs(self._right_support[0] - self._loads_position[sorted_position[i][0]][0])

            label = Symbol(sorted_position[0][0]+"_"+sorted_position[1][0])
            y2 = self._loads_position[sorted_position[1][0]][1]
            x2 = self._loads_position[sorted_position[1][0]][0]
            x1 = self._left_support[0]
            y1 = self._left_support[1]

            angle_with_horizontal = -atan((y2 - y1)/(x2 - x1))
            tension = -(moment_sum_from_right_support)/(abs(self._right_support[1] - self._loads_position[sorted_position[1][0]][1])*cos(angle_with_horizontal) + abs(self._right_support[0] - self._loads_position[sorted_position[1][0]][0])*sin(angle_with_horizontal))
            self._tension[label] = tension

            tension_func.insert(0,(tension, x<=x2))
            self._tension_func = Piecewise(*tension_func)
            angle_with_horizontal = pi/2 - angle_with_horizontal
            label = self._support_labels[0]
            self._reaction_loads[Symbol("R_"+label+"_x")] = -sin(angle_with_horizontal) * tension
            F_x += -sin(angle_with_horizontal) * tension
            self._reaction_loads[Symbol("R_"+label+"_y")] = cos(angle_with_horizontal) * tension
            F_y += cos(angle_with_horizontal) * tension

            label = self._support_labels[1]
            self._reaction_loads[Symbol("R_"+label+"_x")] = -F_x
            self._reaction_loads[Symbol("R_"+label+"_y")] = -F_y

        elif len(self._loads['distributed']) != 0 :

            if len(args) == 0:
                raise ValueError("Provide the lowest point of the cable")

            lowest_x = sympify(args[0])
            self._lowest_x_global = lowest_x

            a = Symbol('a', positive=True)
            c = Symbol('c')
            # augmented matrix form of linsolve

            M = Matrix(
                [[(self._left_support[0]-lowest_x)**2, 1, self._left_support[1]],
                [(self._right_support[0]-lowest_x)**2, 1, self._right_support[1]],
                ])

            coefficient_solution = list(linsolve(M, (a, c)))
            if len(coefficient_solution) ==0 or coefficient_solution[0][0]== 0:
                raise ValueError("The lowest point is inconsistent with the supports")

            A = coefficient_solution[0][0]
            C = coefficient_solution[0][1] + coefficient_solution[0][0]*lowest_x**2
            B = -2*coefficient_solution[0][0]*lowest_x
            self._lowest_y_global = coefficient_solution[0][1]
            lowest_y = self._lowest_y_global

            # y = A*x**2 + B*x + C
            # shifting origin to lowest point
            X = Symbol('X')
            Y = Symbol('Y')
            Y = A*(X + lowest_x)**2 + B*(X + lowest_x) + C - lowest_y

            temp_list = list(self._loads['distributed'].values())
            applied_force = temp_list[0]

            horizontal_force_constant = (applied_force * (self._right_support[0] - lowest_x)**2) / (2 * (self._right_support[1] - lowest_y))

            self._tension.clear()
            tangent_slope_to_curve = diff(Y, X)
            self._tension['distributed'] = horizontal_force_constant / (cos(atan(tangent_slope_to_curve)))

            label = self._support_labels[0]
            self._reaction_loads[Symbol("R_"+label+"_x")] = self.tension_at(self._left_support[0]) * cos(atan(tangent_slope_to_curve.subs(X, self._left_support[0] - lowest_x)))
            self._reaction_loads[Symbol("R_"+label+"_y")] = self.tension_at(self._left_support[0]) * sin(atan(tangent_slope_to_curve.subs(X, self._left_support[0] - lowest_x)))

            label = self._support_labels[1]
            self._reaction_loads[Symbol("R_"+label+"_x")] = self.tension_at(self._left_support[0]) * cos(atan(tangent_slope_to_curve.subs(X, self._right_support[0] - lowest_x)))
            self._reaction_loads[Symbol("R_"+label+"_y")] = self.tension_at(self._left_support[0]) * sin(atan(tangent_slope_to_curve.subs(X, self._right_support[0] - lowest_x)))

    def draw(self):
        """
        This method is used to obtain a plot for the specified cable with its supports,
        shape and loads.

        Examples
        ========

        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> p = c.draw()
        >>> p  # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: Piecewise((10 - 1.37*x, x <= 2), (8.52 - 0.63*x, x <= 4), (2*x/3 + 10/3, x <= 10)) for x over (0.0, 10.0)
        ...
        >>> p.show()

        For uniformly distributed loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58)
        >>> p = c.draw()
        >>> p # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: 0.0116550116550117*(x - 58.58)**2 + 0.00447086247086247 for x over (0.0, 100.0)
        [1]: cartesian line: -7.49552913752915 for x over (0.0, 100.0)
        ...
        >>> p.show()
        """
        x = Symbol("x")
        annotations = []
        support_rectangles = self._draw_supports()

        xy_min = min(self._left_support[0],self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1],self._left_support[1]))
        max_diff = xy_max - xy_min
        if len(self._loads_position) != 0:
            self._cable_eqn = self._draw_cable(-1)
            annotations += self._draw_loads(-1)

        elif len(self._loads['distributed']) != 0 :
            self._cable_eqn = self._draw_cable(0)
            annotations += self._draw_loads(0)

        if not self._cable_eqn:
            raise ValueError("solve method not called and/or values provided for loads and supports not adequate")

        cab_plot = plot(*self._cable_eqn,(x,self._left_support[0],self._right_support[0]),
                        xlim=(xy_min-0.5*max_diff,xy_max+0.5*max_diff),
                        ylim=(xy_min-0.5*max_diff,xy_max+0.5*max_diff),
                        rectangles=support_rectangles,show= False,annotations=annotations, axis=False)

        return cab_plot

    def _draw_supports(self):
        member_rectangles = []
        xy_min = min(self._left_support[0],self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1],self._left_support[1]))
        max_diff = xy_max - xy_min

        supp_width = 0.075*max_diff

        member_rectangles.append(
            {
                'xy': (self._left_support[0]-supp_width,self._left_support[1]),
                'width': supp_width,
                'height':supp_width,
                'color':'brown',
                'fill': False
            }
        )

        member_rectangles.append(
            {
                'xy': (self._right_support[0],self._right_support[1]),
                'width': supp_width,
                'height':supp_width,
                'color':'brown',
                'fill': False
            }
        )

        return member_rectangles

    def _draw_cable(self,order):
        xy_min = min(self._left_support[0],self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1],self._left_support[1]))
        max_diff = xy_max - xy_min
        if order == -1 :
            x,y = symbols('x y')
            line_func = []
            sorted_position = sorted(self._loads_position.items(), key = lambda item : item[1][0])

            for i in range(len(sorted_position)):
                if(i==0):
                    y = ((sorted_position[i][1][1] - self._left_support[1])*(x-self._left_support[0]))/(sorted_position[i][1][0]- self._left_support[0]) + self._left_support[1]
                else:
                    y = ((sorted_position[i][1][1] - sorted_position[i-1][1][1] )*(x-sorted_position[i-1][1][0]))/(sorted_position[i][1][0]- sorted_position[i-1][1][0]) + sorted_position[i-1][1][1]
                line_func.append((y,x<=sorted_position[i][1][0]))

            y = ((sorted_position[len(sorted_position)-1][1][1] - self._right_support[1])*(x-self._right_support[0]))/(sorted_position[i][1][0]- self._right_support[0]) + self._right_support[1]
            line_func.append((y,x<=self._right_support[0]))
            return [Piecewise(*line_func)]

        elif order == 0:
            x0 = self._lowest_x_global
            diff_force_height = max_diff*0.075

            a,c,x,y = symbols('a c x y')
            parabola_eqn = a*(x-x0)**2 + c - y

            points = [(self._left_support[0],self._left_support[1]),(self._right_support[0],self._right_support[1])]
            equations = []
            for px, py in points:
                equations.append(parabola_eqn.subs({x: px, y: py}))
            solution = solve(equations, (a, c))
            parabola_eqn = solution[a]*(x-x0)**2 + solution[c]
            return [parabola_eqn, self._lowest_y_global - diff_force_height]

    def _draw_loads(self,order):
        xy_min = min(self._left_support[0],self._lowest_y_global)
        xy_max = max(self._right_support[0], max(self._right_support[1],self._left_support[1]))
        max_diff = xy_max - xy_min
        if(order==-1):
            arrow_length = max_diff*0.1
            force_arrows = []
            for key in self._loads['point_load']:
                force_arrows.append(
                    {
                        'text': '',
                        'xy':(self._loads_position[key][0]+arrow_length*cos(rad(self._loads['point_load'][key][1])),\
                              self._loads_position[key][1] + arrow_length*sin(rad(self._loads['point_load'][key][1]))),
                        'xytext': (self._loads_position[key][0],self._loads_position[key][1]),
                        'arrowprops': {'width': 1, 'headlength':3, 'headwidth':3 , 'facecolor': 'black', }
                    }
                )
                mag = self._loads['point_load'][key][0]
                force_arrows.append(
                    {
                        'text':f'{mag}N',
                        'xy': (self._loads_position[key][0]+arrow_length*1.6*cos(rad(self._loads['point_load'][key][1])),\
                               self._loads_position[key][1] + arrow_length*1.6*sin(rad(self._loads['point_load'][key][1]))),
                    }
                )
            return force_arrows

        elif (order == 0):
            x = symbols('x')
            force_arrows = []
            x_val = [self._left_support[0] + ((self._right_support[0]-self._left_support[0])/10)*i for i in range(1,10)]
            for i in x_val:
                force_arrows.append(
                    {
                        'text':'',
                        'xytext':(
                            i,
                            self._cable_eqn[0].subs(x,i)
                        ),
                        'xy':(
                            i,
                            self._cable_eqn[1].subs(x,i)
                        ),
                        'arrowprops':{'width':1, 'headlength':3.5, 'headwidth':3.5, 'facecolor':'black'}
                    }
                )
            mag = 0
            for key in self._loads['distributed']:
                mag += self._loads['distributed'][key]

            force_arrows.append(
                {
                    'text':f'{mag} N/m',
                    'xy':((self._left_support[0]+self._right_support[0])/2,self._lowest_y_global - max_diff*0.15)
                }
            )
            return force_arrows

    def plot_tension(self):
        """
        Returns the diagram/plot of the tension generated in the cable at various points.

        Examples
        ========

        For point loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c = Cable(("A", 0, 10), ("B", 10, 10))
        >>> c.apply_load(-1, ('Z', 2, 7.26, 3, 270))
        >>> c.apply_load(-1, ('X', 4, 6, 8, 270))
        >>> c.solve()
        >>> p = c.plot_tension()
        >>> p
        Plot object containing:
        [0]: cartesian line: Piecewise((8.91403453669861, x <= 2), (4.79150773600774, x <= 4), (19*sqrt(13)/10, x <= 10)) for x over (0.0, 10.0)
        >>> p.show()

        For uniformly distributed loads,

        >>> from sympy.physics.continuum_mechanics.cable import Cable
        >>> c=Cable(("A", 0, 40),("B", 100, 20))
        >>> c.apply_load(0, ("X", 850))
        >>> c.solve(58.58)
        >>> p = c.plot_tension()
        >>> p
        Plot object containing:
        [0]: cartesian line: 36465.0*sqrt(0.00054335718671383*X**2 + 1) for X over (0.0, 100.0)
        >>> p.show()

        """
        if len(self._loads_position) != 0:
            x = symbols('x')
            tension_plot  = plot(self._tension_func, (x,self._left_support[0],self._right_support[0]), show=False)
        else:
            X = symbols('X')
            tension_plot  = plot(self._tension['distributed'], (X,self._left_support[0],self._right_support[0]), show=False)
        return tension_plot
<|endoftext|>"""
This module can be used to solve probelsm related to 2D parabolic arches
"""
from sympy.core.sympify import sympify
from sympy.core.symbol import Symbol,symbols
from sympy import diff, sqrt, cos , sin, atan, rad, Min
from sympy.core.relational import Eq
from sympy.solvers.solvers import solve
from sympy.functions import Piecewise
from sympy.plotting import plot
from sympy import limit
from sympy.utilities.decorator import doctest_depends_on
from sympy.external.importtools import import_module

numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})

class Arch:
    """
    This class is used to solve problems related to a three hinged arch(determinate) structure.\n
    An arch is a curved vertical structure spanning an open space underneath it.\n
    Arches can be used to reduce the bending moments in long-span structures.\n

    Arches are used in structural engineering(over windows, door and even bridges)\n
    because they can support a very large mass placed on top of them.

    Example
    ========
    >>> from sympy.physics.continuum_mechanics.arch import Arch
    >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
    >>> a.get_shape_eqn
    5 - (x - 5)**2/5

    >>> from sympy.physics.continuum_mechanics.arch import Arch
    >>> a = Arch((0,0),(10,1),crown_x=6)
    >>> a.get_shape_eqn
    9/5 - (x - 6)**2/20
    """
    def __init__(self,left_support,right_support,**kwargs):
        self._shape_eqn = None
        self._left_support  = (sympify(left_support[0]),sympify(left_support[1]))
        self._right_support  = (sympify(right_support[0]),sympify(right_support[1]))
        self._crown_x = None
        self._crown_y = None
        if 'crown_x' in kwargs:
            self._crown_x = sympify(kwargs['crown_x'])
        if 'crown_y' in kwargs:
            self._crown_y = sympify(kwargs['crown_y'])
        self._shape_eqn = self.get_shape_eqn
        self._conc_loads = {}
        self._distributed_loads = {}
        self._loads = {'concentrated': self._conc_loads, 'distributed':self._distributed_loads}
        self._loads_applied = {}
        self._supports = {'left':'hinge', 'right':'hinge'}
        self._member = None
        self._member_force = None
        self._reaction_force = {Symbol('R_A_x'):0, Symbol('R_A_y'):0, Symbol('R_B_x'):0, Symbol('R_B_y'):0}
        self._points_disc_x = set()
        self._points_disc_y = set()
        self._moment_x  = {}
        self._moment_y = {}
        self._load_x = {}
        self._load_y = {}
        self._moment_x_func = Piecewise((0,True))
        self._moment_y_func = Piecewise((0,True))
        self._load_x_func = Piecewise((0,True))
        self._load_y_func = Piecewise((0,True))
        self._bending_moment = None
        self._shear_force = None
        self._axial_force = None
        # self._crown = (sympify(crown[0]),sympify(crown[1]))

    @property
    def get_shape_eqn(self):
        "returns the equation of the shape of arch developed"
        if self._shape_eqn:
            return self._shape_eqn

        x,y,c = symbols('x y c')
        a = Symbol('a',positive=False)
        if self._crown_x and self._crown_y:
            x0  = self._crown_x
            y0 = self._crown_y
            parabola_eqn = a*(x-x0)**2 + y0 - y
            eq1 = parabola_eqn.subs({x:self._left_support[0], y:self._left_support[1]})
            solution = solve((eq1),(a))
            parabola_eqn = solution[0]*(x-x0)**2 + y0
            if(parabola_eqn.subs({x:self._right_support[0]}) != self._right_support[1]):
                raise ValueError("provided coordinates of crown and supports are not consistent with parabolic arch")

        elif self._crown_x:
            x0  = self._crown_x
            parabola_eqn = a*(x-x0)**2 + c - y
            eq1 = parabola_eqn.subs({x:self._left_support[0], y:self._left_support[1]})
            eq2 = parabola_eqn.subs({x:self._right_support[0], y:self._right_support[1]})
            solution = solve((eq1,eq2),(a,c))
            if len(solution) <2 or solution[a] == 0:
                raise ValueError("parabolic arch cannot be constructed with the provided coordinates, try providing crown_y")
            parabola_eqn = solution[a]*(x-x0)**2+ solution[c]
            self._crown_y = solution[c]

        else:
            raise KeyError("please provide crown_x to construct arch")

        return parabola_eqn

    @property
    def get_loads(self):
        """
        return the position of the applied load and angle (for concentrated loads)
        """
        return self._loads

    @property
    def supports(self):
        """
        Returns the type of support
        """
        return self._supports

    @property
    def left_support(self):
        """
        Returns the position of the left support.
        """
        return self._left_support

    @property
    def right_support(self):
        """
        Returns the position of the right support.
        """
        return self._right_support

    @property
    def reaction_force(self):
        """
        return the reaction forces generated
        """
        return self._reaction_force

    def apply_load(self,order,label,start,mag,end=None,angle=None):
        """
        This method adds load to the Arch.

        Parameters
        ==========

            order : Integer
                Order of the applied load.

                    - For point/concentrated loads, order = -1
                    - For distributed load, order = 0

            label : String or Symbol
                The label of the load
                - should not use 'A' or 'B' as it is used for supports.

            start : Float

                    - For concentrated/point loads, start is the x coordinate
                    - For distributed loads, start is the starting position of distributed load

            mag : Sympifyable
                Magnitude of the applied load. Must be positive

            end : Float
                Required for distributed loads

                    - For concentrated/point load , end is None(may not be given)
                    - For distributed loads, end is the end position of distributed load

            angle: Sympifyable
                The angle in degrees, the load vector makes with the horizontal
                in the counter-clockwise direction.

        Examples
        ========
        For applying distributed load

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)

        For applying point/concentrated_loads

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(-1,'C',start=2,mag=15,angle=45)

        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')
        # y0 = Symbol('y0')
        order= sympify(order)
        mag = sympify(mag)
        angle = sympify(angle)

        if label in self._loads_applied:
            raise ValueError("load with the given label already exists")

        if label in ['A','B']:
            raise ValueError("cannot use the given label, reserved for supports")

        if order == 0:
            if end is None or end<start:
                raise KeyError("provide end greater than start")

            self._distributed_loads[label] = {'start':start, 'end':end, 'f_y': mag}
            self._points_disc_y.add(start)

            if start in self._moment_y:
                self._moment_y[start] -= mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
                self._load_y[start] += mag*(Min(end,x)-start)
            else:
                self._moment_y[start] = -mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
                self._load_y[start] = mag*(Min(end,x)-start)

            self._loads_applied[label] = 'distributed'

        if order == -1:

            if angle is None:
                raise TypeError("please provide direction of force")
            height = self._shape_eqn.subs({'x':start})

            self._conc_loads[label] = {'x':start, 'y':height, 'f_x':mag*cos(rad(angle)), 'f_y': mag*sin(rad(angle)), 'mag':mag, 'angle':angle}
            self._points_disc_x.add(start)
            self._points_disc_y.add(start)

            if start in self._moment_x:
                self._moment_x[start] += self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
                self._load_x[start] += self._conc_loads[label]['f_x']
            else:
                self._moment_x[start] = self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
                self._load_x[start] = self._conc_loads[label]['f_x']

            if start in self._moment_y:
                self._moment_y[start] -= self._conc_loads[label]['f_y']*(x0-start)
                self._load_y[start] += self._conc_loads[label]['f_y']
            else:
                self._moment_y[start] = -self._conc_loads[label]['f_y']*(x0-start)
                self._load_y[start] = self._conc_loads[label]['f_y']

            self._loads_applied[label] = 'concentrated'


    def remove_load(self,label):
        """
        This methods removes the load applied to the arch

        Parameters
        ==========

        label : String or Symbol
            The label of the applied load

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)
        >>> a.remove_load('C')
        removed load C: {'start': 3, 'end': 5, 'f_y': -10}
        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')

        if label in self._distributed_loads :

            self._loads_applied.pop(label)
            start = self._distributed_loads[label]['start']
            end = self._distributed_loads[label]['end']
            mag  = self._distributed_loads[label]['f_y']
            self._points_disc_y.remove(start)
            self._load_y[start] -= mag*(Min(x,end)-start)
            self._moment_y[start] += mag*(Min(x,end)-start)*(x0-(start+(Min(x,end)))/2)
            val = self._distributed_loads.pop(label)
            print(f"removed load {label}: {val}")

        elif label in self._conc_loads :

            self._loads_applied.pop(label)
            start = self._conc_loads[label]['x']
            self._points_disc_x.remove(start)
            self._points_disc_y.remove(start)
            self._moment_y[start] += self._conc_loads[label]['f_y']*(x0-start)
            self._moment_x[start] -= self._conc_loads[label]['f_x']*(y-self._conc_loads[label]['y'])
            self._load_x[start] -= self._conc_loads[label]['f_x']
            self._load_y[start] -= self._conc_loads[label]['f_y']
            val = self._conc_loads.pop(label)
            print(f"removed load {label}: {val}")

        else :
            raise ValueError("label not found")

    def change_support_position(self, left_support=None, right_support=None):
        """
        Change position of supports.
        If not provided , defaults to the old value.
        Parameters
        ==========

            left_support: tuple (x, y)
                x: float
                    x-coordinate value of the left_support

                y: float
                    y-coordinate value of the left_support

            right_support: tuple (x, y)
                x: float
                    x-coordinate value of the right_support

                y: float
                    y-coordinate value of the right_support
        """
        if left_support is not None:
            self._left_support = (left_support[0],left_support[1])

        if right_support is not None:
            self._right_support = (right_support[0],right_support[1])

        self._shape_eqn = None
        self._shape_eqn = self.get_shape_eqn

    def change_crown_position(self,crown_x=None,crown_y=None):
        """
        Change the position of the crown/hinge of the arch

        Parameters
        ==========

            crown_x: Float
                The x coordinate of the position of the hinge
                - if not provided, defaults to old value

            crown_y: Float
                The y coordinate of the position of the hinge
                - if not provided defaults to None
        """
        self._crown_x = crown_x
        self._crown_y = crown_y
        self._shape_eqn = None
        self._shape_eqn = self.get_shape_eqn

    def change_support_type(self,left_support=None,right_support=None):
        """
        Add the type for support at each end.
        Can use roller or hinge support at each end.

        Parameters
        ==========

            left_support, right_support : string
                Type of support at respective end

                    - For roller support , left_support/right_support = "roller"
                    - For hinged support, left_support/right_support = "hinge"
                    - defaults to hinge if value not provided

        Examples
        ========

        For applying roller support at right end

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.change_support_type(right_support="roller")

        """
        support_types = ['roller','hinge']
        if left_support:
            if left_support not in support_types:
                raise ValueError("supports must only be roller or hinge")

            self._supports['left'] = left_support

        if right_support:
            if right_support not in support_types:
                raise ValueError("supports must only be roller or hinge")

            self._supports['right'] = right_support

    def add_member(self,y):
        """
        This method adds a member/rod at a particular height y.
        A rod is used for stability of the structure in case of a roller support.
        """
        if y>self._crown_y or y<min(self._left_support[1],  self._right_support[1]):
            raise ValueError(f"position of support must be between y={min(self._left_support[1],  self._right_support[1])} and y={self._crown_y}")
        x = Symbol('x')
        a = diff(self._shape_eqn,x).subs(x,self._crown_x+1)/2
        x_diff = sqrt((y - self._crown_y)/a)
        x1 = self._crown_x + x_diff
        x2 = self._crown_x - x_diff
        self._member = (x1,x2,y)

    def shear_force_at(self, pos = None, **kwargs):
        """
        return the shear at some x-coordinates
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._shear_force
        else:
            x = Symbol('x')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._shear_force,x,pos,dir=dir)
            return self._shear_force.subs(x,pos)

    def bending_moment_at(self, pos = None, **kwargs):
        """
        return the bending moment at some x-coordinates
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._bending_moment
        else:
            x0 = Symbol('x0')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._bending_moment,x0,pos,dir=dir)
            return self._bending_moment.subs(x0,pos)


    def axial_force_at(self,pos = None, **kwargs):
        """
        return the axial/normal force generated at some x-coordinate
        if no x value provided, returns the formula
        """
        if pos is None:
            return self._axial_force
        else:
            x = Symbol('x')
            if 'dir' in kwargs:
                dir = kwargs['dir']
                return limit(self._axial_force,x,pos,dir=dir)
            return self._axial_force.subs(x,pos)

    def solve(self):
        """
        This method solves for the reaction forces generated at the supports,\n
        and bending moment and generated in the arch and tension produced in the member if used.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(10,0),crown_x=5,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=-10)
        >>> a.solve()
        >>> a.reaction_force
        {R_A_x: 8, R_A_y: 12, R_B_x: -8, R_B_y: 8}

        >>> from sympy import Symbol
        >>> t = Symbol('t')
        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(16,0),crown_x=8,crown_y=5)
        >>> a.apply_load(0,'C',start=3,end=5,mag=t)
        >>> a.solve()
        >>> a.reaction_force
        {R_A_x: -4*t/5, R_A_y: -3*t/2, R_B_x: 4*t/5, R_B_y: -t/2}

        >>> a.bending_moment_at(4)
        -5*t/2
        """
        y = Symbol('y')
        x = Symbol('x')
        x0 = Symbol('x0')

        discontinuity_points_x = sorted(self._points_disc_x)
        discontinuity_points_y = sorted(self._points_disc_y)

        self._moment_x_func = Piecewise((0,True))
        self._moment_y_func = Piecewise((0,True))

        self._load_x_func = Piecewise((0,True))
        self._load_y_func = Piecewise((0,True))

        accumulated_x_moment = 0
        accumulated_y_moment = 0

        accumulated_x_load = 0
        accumulated_y_load = 0

        for point in discontinuity_points_x:
            cond = (x >= point)
            accumulated_x_load += self._load_x[point]
            accumulated_x_moment += self._moment_x[point]
            self._load_x_func = Piecewise((accumulated_x_load,cond),(self._load_x_func,True))
            self._moment_x_func = Piecewise((accumulated_x_moment,cond),(self._moment_x_func,True))

        for point in discontinuity_points_y:
            cond = (x >= point)
            accumulated_y_moment += self._moment_y[point]
            accumulated_y_load += self._load_y[point]
            self._load_y_func = Piecewise((accumulated_y_load,cond),(self._load_y_func,True))
            self._moment_y_func = Piecewise((accumulated_y_moment,cond),(self._moment_y_func,True))

        moment_A = self._moment_y_func.subs(x,self._right_support[0]).subs(x0,self._left_support[0]) +\
                   self._moment_x_func.subs(x,self._right_support[0]).subs(y,self._left_support[1])

        moment_hinge_left = self._moment_y_func.subs(x,self._crown_x).subs(x0,self._crown_x) +\
                            self._moment_x_func.subs(x,self._crown_x).subs(y,self._crown_y)

        moment_hinge_right = self._moment_y_func.subs(x,self._right_support[0]).subs(x0,self._crown_x)- \
                             self._moment_y_func.subs(x,self._crown_x).subs(x0,self._crown_x) +\
                             self._moment_x_func.subs(x,self._right_support[0]).subs(y,self._crown_y) -\
                             self._moment_x_func.subs(x,self._crown_x).subs(y,self._crown_y)

        net_x = self._load_x_func.subs(x,self._right_support[0])
        net_y = self._load_y_func.subs(x,self._right_support[0])

        if (self._supports['left']=='roller' or self._supports['right']=='roller') and not self._member:
            print("member must be added if any of the supports is roller")
            return

        R_A_x, R_A_y, R_B_x, R_B_y, T = symbols('R_A_x R_A_y R_B_x R_B_y T')

        if self._supports['left'] == 'roller' and self._supports['right'] == 'roller':

            if self._member[2]>=max(self._left_support[1],self._right_support[1]):

                if net_x!=0:
                    raise ValueError("net force in x direction not possible under the specified conditions")

                else:
                    eq1 = Eq(R_A_x ,0)
                    eq2 = Eq(R_B_x, 0)
                    eq3 = Eq(R_A_y + R_B_y + net_y,0)

                    eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                             R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)

                    eq5 = Eq(moment_hinge_right + R_B_y*(self._right_support[0]-self._crown_x) +\
                             T*(self._member[2]-self._crown_y),0)
                    solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x, 0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         T*(self._member[2]-self._left_support[1])+moment_A,0)
                eq5 = Eq(T+net_x,0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x, 0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])+\
                         T*(self._member[2]-self._left_support[1])+moment_A,0)
                eq5 = Eq(T-net_x,0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

        elif self._supports['left'] == 'roller':
            if self._member[2]>=max(self._left_support[1], self._right_support[1]):
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x+net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)
                eq5 = Eq(moment_hinge_left + R_A_y*(self._left_support[0]-self._crown_x) -\
                         T*(self._member[2]-self._crown_y),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_A_x ,0)
                eq2 = Eq(R_B_x+ T +net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])-\
                         T*(self._member[2]-self._left_support[0])+moment_A,0)
                eq5 = Eq(moment_hinge_left + R_A_y*(self._left_support[0]-self._crown_x)-\
                         T*(self._member[2]-self._crown_y),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[0]:
                eq1 = Eq(R_A_x,0)
                eq2 = Eq(R_B_x- T +net_x,0)
                eq3 = Eq(R_A_y + R_B_y + net_y,0)
                eq4 = Eq(moment_hinge_left+R_A_y*(self._left_support[0]-self._crown_x),0)
                eq5 = Eq(moment_A+R_B_y*(self._right_support[0]-self._left_support[0])-\
                         R_B_x*(self._right_support[1]-self._left_support[1])+\
                         T*(self._member[2]-self._left_support[1]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

        elif self._supports['right'] == 'roller':
            if self._member[2]>=max(self._left_support[1], self._right_support[1]):
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x)+\
                         T*(self._member[2]-self._crown_y),0)
                eq5 = Eq(moment_A+R_B_y*(self._right_support[0]-self._left_support[0]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._left_support[1]:
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x+T+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x),0)
                eq5 = Eq(moment_A-T*(self._member[2]-self._left_support[1])+\
                         R_B_y*(self._right_support[0]-self._left_support[0]),0)
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))

            elif self._member[2]>=self._right_support[1]:
                eq1 = Eq(R_B_x,0)
                eq2 = Eq(R_A_x-T+net_x,0)
                eq3 = Eq(R_A_y+R_B_y+net_y,0)
                eq4 = Eq(moment_hinge_right+R_B_y*(self._right_support[0]-self._crown_x)+\
                         T*(self._member[2]-self._crown_y),0)
                eq5 = Eq(moment_A+T*(self._member[2]-self._left_support[1])+\
                         R_B_y*(self._right_support[0]-self._left_support[0]))
                solution = solve((eq1,eq2,eq3,eq4,eq5),(R_A_x,R_A_y,R_B_x,R_B_y,T))
        else:
            eq1 = Eq(R_A_x + R_B_x + net_x,0)
            eq2 = Eq(R_A_y + R_B_y + net_y,0)
            eq3 = Eq(R_B_y*(self._right_support[0]-self._left_support[0])-\
                     R_B_x*(self._right_support[1]-self._left_support[1])+moment_A,0)
            eq4 = Eq(moment_hinge_right + R_B_y*(self._right_support[0]-self._crown_x) -\
                     R_B_x*(self._right_support[1]-self._crown_y),0)
            solution = solve((eq1,eq2,eq3,eq4),(R_A_x,R_A_y,R_B_x,R_B_y))

        for symb in self._reaction_force:
            self._reaction_force[symb] = solution[symb]

        self._bending_moment = - (self._moment_x_func.subs(x,x0) + self._moment_y_func.subs(x,x0) -\
                                  solution[R_A_y]*(x0-self._left_support[0]) +\
                                  solution[R_A_x]*(self._shape_eqn.subs({x:x0})-self._left_support[1]))

        angle  = atan(diff(self._shape_eqn,x))

        fx = (self._load_x_func+solution[R_A_x])
        fy = (self._load_y_func+solution[R_A_y])

        axial_force = fx*cos(angle) + fy*sin(angle)
        shear_force = -fx*sin(angle) + fy*cos(angle)

        self._axial_force = axial_force
        self._shear_force = shear_force

    @doctest_depends_on(modules=('numpy',))
    def draw(self):
        """
        This method returns a plot object containing the diagram of the specified arch along with the supports
        and forces applied to the structure.

        Examples
        ========

        >>> from sympy import Symbol
        >>> t = Symbol('t')
        >>> from sympy.physics.continuum_mechanics.arch import Arch
        >>> a = Arch((0,0),(40,0),crown_x=20,crown_y=12)
        >>> a.apply_load(-1,'C',8,150,angle=270)
        >>> a.apply_load(0,'D',start=20,end=40,mag=-4)
        >>> a.apply_load(-1,'E',10,t,angle=300)
        >>> p = a.draw()
        >>> p # doctest: +ELLIPSIS
        Plot object containing:
        [0]: cartesian line: 11.325 - 3*(x - 20)**2/100 for x over (0.0, 40.0)
        [1]: cartesian line: 12 - 3*(x - 20)**2/100 for x over (0.0, 40.0)
        ...
        >>> p.show()

        """
        x = Symbol('x')
        markers = []
        annotations = self._draw_loads()
        rectangles = []
        supports = self._draw_supports()
        markers+=supports

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        lim = max(xmax*1.1-xmin*0.8+1, ymax*1.1-ymin*0.8+1)

        rectangles = self._draw_rectangles()

        filler = self._draw_filler()
        rectangles+=filler

        if self._member is not None:
            if(self._member[2]>=self._right_support[1]):
                markers.append(
                    {
                        'args':[[self._member[1]+0.005*lim],[self._member[2]]],
                        'marker':'o',
                        'markersize': 4,
                        'color': 'white',
                        'markerfacecolor':'none'
                    }
                )

            if(self._member[2]>=self._left_support[1]):
                markers.append(
                    {
                        'args':[[self._member[0]-0.005*lim],[self._member[2]]],
                        'marker':'o',
                        'markersize': 4,
                        'color': 'white',
                        'markerfacecolor':'none'
                    }
                )



        markers.append({
            'args':[[self._crown_x],[self._crown_y-0.005*lim]],
            'marker':'o',
            'markersize': 5,
            'color':'white',
            'markerfacecolor':'none',
        })

        if lim==xmax*1.1-xmin*0.8+1:

            sing_plot = plot(self._shape_eqn-0.015*lim,
                             self._shape_eqn,
                             (x, self._left_support[0], self._right_support[0]),
                             markers=markers,
                             show=False,
                             annotations=annotations,
                             rectangles = rectangles,
                             xlim=(xmin-0.05*lim, xmax*1.1),
                             ylim=(xmin-0.05*lim, xmax*1.1),
                             axis=False,
                             line_color='brown')

        else:
            sing_plot = plot(self._shape_eqn-0.015*lim,
                             self._shape_eqn,
                             (x, self._left_support[0], self._right_support[0]),
                             markers=markers,
                             show=False,
                             annotations=annotations,
                             rectangles = rectangles,
                             xlim=(ymin-0.05*lim, ymax*1.1),
                             ylim=(ymin-0.05*lim, ymax*1.1),
                             axis=False,
                             line_color='brown')

        return sing_plot


    def _draw_supports(self):
        support_markers = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        if self._supports['left']=='roller':
            support_markers.append(
                {
                    'args':[
                        [self._left_support[0]],
                        [self._left_support[1]-0.02*max_diff]
                    ],
                    'marker':'o',
                    'markersize':11,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )
        else:
            support_markers.append(
                {
                    'args':[
                        [self._left_support[0]],
                        [self._left_support[1]-0.007*max_diff]
                    ],
                    'marker':6,
                    'markersize':15,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )

        if self._supports['right']=='roller':
            support_markers.append(
                {
                    'args':[
                        [self._right_support[0]],
                        [self._right_support[1]-0.02*max_diff]
                    ],
                    'marker':'o',
                    'markersize':11,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )
        else:
            support_markers.append(
                {
                    'args':[
                        [self._right_support[0]],
                        [self._right_support[1]-0.007*max_diff]
                    ],
                    'marker':6,
                    'markersize':15,
                    'color':'black',
                    'markerfacecolor':'none'
                }
            )

        support_markers.append(
            {
                'args':[
                    [self._right_support[0]],
                    [self._right_support[1]-0.036*max_diff]
                ],
                'marker':'_',
                'markersize':15,
                'color':'black',
                'markerfacecolor':'none'
            }
        )

        support_markers.append(
            {
                'args':[
                    [self._left_support[0]],
                    [self._left_support[1]-0.036*max_diff]
                ],
                'marker':'_',
                'markersize':15,
                'color':'black',
                'markerfacecolor':'none'
            }
        )

        return support_markers

    def _draw_rectangles(self):
        member = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        if self._member is not None:
            if self._member[2]>= max(self._left_support[1],self._right_support[1]):
                member.append(
                    {
                        'xy':(self._member[0],self._member[2]-0.005*max_diff),
                        'width':self._member[1]-self._member[0],
                        'height': 0.01*max_diff,
                        'angle': 0,
                        'color':'brown',
                    }
                )

            elif self._member[2]>=self._left_support[1]:
                member.append(
                    {
                        'xy':(self._member[0],self._member[2]-0.005*max_diff),
                        'width':self._right_support[0]-self._member[0],
                        'height': 0.01*max_diff,
                        'angle': 0,
                        'color':'brown',
                    }
                )

            else:
                member.append(
                    {
                        'xy':(self._member[1],self._member[2]-0.005*max_diff),
                        'width':abs(self._left_support[0]-self._member[1]),
                        'height': 0.01*max_diff,
                        'angle': 180,
                        'color':'brown',
                    }
                )

        if self._distributed_loads:
            for loads in self._distributed_loads:

                start = self._distributed_loads[loads]['start']
                end = self._distributed_loads[loads]['end']

                member.append(
                    {
                        'xy':(start,self._crown_y+max_diff*0.15),
                        'width': (end-start),
                        'height': max_diff*0.01,
                        'color': 'orange'
                    }
                )


        return member

    def _draw_loads(self):
        load_annotations = []

        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for load in self._conc_loads:
            x = self._conc_loads[load]['x']
            y = self._conc_loads[load]['y']
            angle = self._conc_loads[load]['angle']
            mag = self._conc_loads[load]['mag']
            load_annotations.append(
                {
                    'text':'',
                    'xy':(
                        x+cos(rad(angle))*max_diff*0.08,
                        y+sin(rad(angle))*max_diff*0.08
                    ),
                    'xytext':(x,y),
                    'fontsize':10,
                    'fontweight': 'bold',
                    'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'blue','edgecolor':'blue'}
                }
            )
            load_annotations.append(
                {
                    'text':f'{load}: {mag} N',
                    'fontsize':10,
                    'fontweight': 'bold',
                    'xy': (x+cos(rad(angle))*max_diff*0.12,y+sin(rad(angle))*max_diff*0.12)
                }
            )

        for load in self._distributed_loads:
            start = self._distributed_loads[load]['start']
            end = self._distributed_loads[load]['end']
            mag = self._distributed_loads[load]['f_y']
            x_points = numpy.arange(start,end,(end-start)/(max_diff*0.25))
            x_points = numpy.append(x_points,end)
            for point in x_points:
                if(mag<0):
                    load_annotations.append(
                        {
                            'text':'',
                            'xy':(point,self._crown_y+max_diff*0.05),
                            'xytext': (point,self._crown_y+max_diff*0.15),
                            'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'orange','edgecolor':'orange'}
                        }
                    )
                else:
                    load_annotations.append(
                        {
                            'text':'',
                            'xy':(point,self._crown_y+max_diff*0.2),
                            'xytext': (point,self._crown_y+max_diff*0.15),
                            'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'orange','edgecolor':'orange'}
                        }
                    )
            if(mag<0):
                load_annotations.append(
                    {
                        'text':f'{load}: {abs(mag)} N/m',
                        'fontsize':10,
                        'fontweight': 'bold',
                        'xy':((start+end)/2,self._crown_y+max_diff*0.175)
                    }
                )
            else:
                load_annotations.append(
                    {
                        'text':f'{load}: {abs(mag)} N/m',
                        'fontsize':10,
                        'fontweight': 'bold',
                        'xy':((start+end)/2,self._crown_y+max_diff*0.125)
                    }
                )
        return load_annotations

    def _draw_filler(self):
        x = Symbol('x')
        filler = []
        xmax = self._right_support[0]
        xmin = self._left_support[0]
        ymin = min(self._left_support[1],self._right_support[1])
        ymax = self._crown_y

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        x_points = numpy.arange(self._left_support[0],self._right_support[0],(self._right_support[0]-self._left_support[0])/(max_diff*max_diff))

        for point in x_points:
            filler.append(
                    {
                        'xy':(point,self._shape_eqn.subs(x,point)-max_diff*0.015),
                        'width': (self._right_support[0]-self._left_support[0])/(max_diff*max_diff),
                        'height': max_diff*0.015,
                        'color': 'brown'
                    }
            )

        return filler
<|endoftext|>"""
This module can be used to solve problems related
to 2D Trusses.
"""


from cmath import atan, inf
from sympy.core.add import Add
from sympy.core.evalf import INF
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.external.importtools import import_module
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
import math
from sympy.physics.units.quantities import Quantity
from sympy.plotting import plot
from sympy.utilities.decorator import doctest_depends_on
from sympy import sin, cos


__doctest_requires__ = {('Truss.draw'): ['matplotlib']}


numpy = import_module('numpy', import_kwargs={'fromlist':['arange']})


class Truss:
    """
    A Truss is an assembly of members such as beams,
    connected by nodes, that create a rigid structure.
    In engineering, a truss is a structure that
    consists of two-force members only.

    Trusses are extremely important in engineering applications
    and can be seen in numerous real-world applications like bridges.

    Examples
    ========

    There is a Truss consisting of four nodes and five
    members connecting the nodes. A force P acts
    downward on the node D and there also exist pinned
    and roller joints on the nodes A and B respectively.

    .. image:: truss_example.png

    >>> from sympy.physics.continuum_mechanics.truss import Truss
    >>> t = Truss()
    >>> t.add_node(("node_1", 0, 0), ("node_2", 6, 0), ("node_3", 2, 2), ("node_4", 2, 0))
    >>> t.add_member(("member_1", "node_1", "node_4"), ("member_2", "node_2", "node_4"), ("member_3", "node_1", "node_3"))
    >>> t.add_member(("member_4", "node_2", "node_3"), ("member_5", "node_3", "node_4"))
    >>> t.apply_load(("node_4", 10, 270))
    >>> t.apply_support(("node_1", "pinned"), ("node_2", "roller"))
    """

    def __init__(self):
        """
        Initializes the class
        """
        self._nodes = []
        self._members = {}
        self._loads = {}
        self._supports = {}
        self._node_labels = []
        self._node_positions = []
        self._node_position_x = []
        self._node_position_y = []
        self._nodes_occupied = {}
        self._member_lengths = {}
        self._reaction_loads = {}
        self._internal_forces = {}
        self._node_coordinates = {}

    @property
    def nodes(self):
        """
        Returns the nodes of the truss along with their positions.
        """
        return self._nodes

    @property
    def node_labels(self):
        """
        Returns the node labels of the truss.
        """
        return self._node_labels

    @property
    def node_positions(self):
        """
        Returns the positions of the nodes of the truss.
        """
        return self._node_positions

    @property
    def members(self):
        """
        Returns the members of the truss along with the start and end points.
        """
        return self._members

    @property
    def member_lengths(self):
        """
        Returns the length of each member of the truss.
        """
        return self._member_lengths

    @property
    def supports(self):
        """
        Returns the nodes with provided supports along with the kind of support provided i.e.
        pinned or roller.
        """
        return self._supports

    @property
    def loads(self):
        """
        Returns the loads acting on the truss.
        """
        return self._loads

    @property
    def reaction_loads(self):
        """
        Returns the reaction forces for all supports which are all initialized to 0.
        """
        return self._reaction_loads

    @property
    def internal_forces(self):
        """
        Returns the internal forces for all members which are all initialized to 0.
        """
        return self._internal_forces

    def add_node(self, *args):
        """
        This method adds a node to the truss along with its name/label and its location.
        Multiple nodes can be added at the same time.

        Parameters
        ==========
        The input(s) for this method are tuples of the form (label, x, y).

        label:  String or a Symbol
            The label for a node. It is the only way to identify a particular node.

        x: Sympifyable
            The x-coordinate of the position of the node.

        y: Sympifyable
            The y-coordinate of the position of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0))
        >>> t.nodes
        [('A', 0, 0)]
        >>> t.add_node(('B', 3, 0), ('C', 4, 1))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 4, 1)]
        """

        for i in args:
            label = i[0]
            x = i[1]
            x = sympify(x)
            y=i[2]
            y = sympify(y)
            if label in self._node_coordinates:
                raise ValueError("Node needs to have a unique label")

            elif [x, y] in self._node_coordinates.values():
                raise ValueError("A node already exists at the given position")

            else :
                self._nodes.append((label, x, y))
                self._node_labels.append(label)
                self._node_positions.append((x, y))
                self._node_position_x.append(x)
                self._node_position_y.append(y)
                self._node_coordinates[label] = [x, y]



    def remove_node(self, *args):
        """
        This method removes a node from the truss.
        Multiple nodes can be removed at the same time.

        Parameters
        ==========
        The input(s) for this method are the labels of the nodes to be removed.

        label:  String or Symbol
            The label of the node to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('C', 5, 0)]
        >>> t.remove_node('A', 'C')
        >>> t.nodes
        [('B', 3, 0)]
        """
        for label in args:
            for i in range(len(self.nodes)):
                if self._node_labels[i] == label:
                    x = self._node_position_x[i]
                    y = self._node_position_y[i]

            if label not in self._node_coordinates:
                raise ValueError("No such node exists in the truss")

            else:
                members_duplicate = self._members.copy()
                for member in members_duplicate:
                    if label == self._members[member][0] or label == self._members[member][1]:
                        raise ValueError("The given node already has member attached to it")
                self._nodes.remove((label, x, y))
                self._node_labels.remove(label)
                self._node_positions.remove((x, y))
                self._node_position_x.remove(x)
                self._node_position_y.remove(y)
                if label in self._loads:
                    self._loads.pop(label)
                if label in self._supports:
                    self._supports.pop(label)
                self._node_coordinates.pop(label)



    def add_member(self, *args):
        """
        This method adds a member between any two nodes in the given truss.

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (label, start, end).

        label: String or Symbol
            The label for a member. It is the only way to identify a particular member.

        start: String or Symbol
            The label of the starting point/node of the member.

        end: String or Symbol
            The label of the ending point/node of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'BC': ['B', 'C']}
        """
        for i in args:
            label = i[0]
            start = i[1]
            end = i[2]

            if start not in self._node_coordinates or end not in self._node_coordinates or start==end:
                raise ValueError("The start and end points of the member must be unique nodes")

            elif label in self._members:
                raise ValueError("A member with the same label already exists for the truss")

            elif self._nodes_occupied.get((start, end)):
                raise ValueError("A member already exists between the two nodes")

            else:
                self._members[label] = [start, end]
                self._member_lengths[label] = sqrt((self._node_coordinates[end][0]-self._node_coordinates[start][0])**2 + (self._node_coordinates[end][1]-self._node_coordinates[start][1])**2)
                self._nodes_occupied[start, end] = True
                self._nodes_occupied[end, start] = True
                self._internal_forces[label] = 0

    def remove_member(self, *args):
        """
        This method removes members from the given truss.

        Parameters
        ==========
        labels: String or Symbol
            The label for the member to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('C', 2, 2))
        >>> t.add_member(('AB', 'A', 'B'), ('AC', 'A', 'C'), ('BC', 'B', 'C'))
        >>> t.members
        {'AB': ['A', 'B'], 'AC': ['A', 'C'], 'BC': ['B', 'C']}
        >>> t.remove_member('AC', 'BC')
        >>> t.members
        {'AB': ['A', 'B']}
        """
        for label in args:
            if label not in self._members:
                raise ValueError("No such member exists in the Truss")

            else:
                self._nodes_occupied.pop((self._members[label][0], self._members[label][1]))
                self._nodes_occupied.pop((self._members[label][1], self._members[label][0]))
                self._members.pop(label)
                self._member_lengths.pop(label)
                self._internal_forces.pop(label)

    def change_node_label(self, *args):
        """
        This method changes the label(s) of the specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label).

        label: String or Symbol
            The label of the node for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0)]
        >>> t.change_node_label(('A', 'C'), ('B', 'D'))
        >>> t.nodes
        [('C', 0, 0), ('D', 3, 0)]
        """
        for i in args:
            label = i[0]
            new_label = i[1]
            if label not in self._node_coordinates:
                raise ValueError("No such node exists for the Truss")
            elif new_label in self._node_coordinates:
                raise ValueError("A node with the given label already exists")
            else:
                for node in self._nodes:
                    if node[0] == label:
                        self._nodes[self._nodes.index((label, node[1], node[2]))] = (new_label, node[1], node[2])
                        self._node_labels[self._node_labels.index(node[0])] = new_label
                        self._node_coordinates[new_label] = self._node_coordinates[label]
                        self._node_coordinates.pop(label)
                        if node[0] in self._supports:
                            self._supports[new_label] = self._supports[node[0]]
                            self._supports.pop(node[0])
                        if new_label in self._supports:
                            if self._supports[new_label] == 'pinned':
                                if 'R_'+str(label)+'_x' in self._reaction_loads and 'R_'+str(label)+'_y' in self._reaction_loads:
                                    self._reaction_loads['R_'+str(new_label)+'_x'] = self._reaction_loads['R_'+str(label)+'_x']
                                    self._reaction_loads['R_'+str(new_label)+'_y'] = self._reaction_loads['R_'+str(label)+'_y']
                                    self._reaction_loads.pop('R_'+str(label)+'_x')
                                    self._reaction_loads.pop('R_'+str(label)+'_y')
                                self._loads[new_label] = self._loads[label]
                                for load in self._loads[new_label]:
                                    if load[1] == 90:
                                        load[0] -= Symbol('R_'+str(label)+'_y')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                for load in self._loads[new_label]:
                                    if load[1] == 0:
                                        load[0] -= Symbol('R_'+str(label)+'_x')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_x'), 0)
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_y'), 90)
                                self._loads.pop(label)
                            elif self._supports[new_label] == 'roller':
                                self._loads[new_label] = self._loads[label]
                                for load in self._loads[label]:
                                    if load[1] == 90:
                                        load[0] -= Symbol('R_'+str(label)+'_y')
                                        if load[0] == 0:
                                            self._loads[label].remove(load)
                                        break
                                self.apply_load(new_label, Symbol('R_'+str(new_label)+'_y'), 90)
                                self._loads.pop(label)
                        else:
                            if label in self._loads:
                                self._loads[new_label] = self._loads[label]
                                self._loads.pop(label)
                        for member in self._members:
                            if self._members[member][0] == node[0]:
                                self._members[member][0] = new_label
                                self._nodes_occupied[(new_label, self._members[member][1])] = True
                                self._nodes_occupied[(self._members[member][1], new_label)] = True
                                self._nodes_occupied.pop((label, self._members[member][1]))
                                self._nodes_occupied.pop((self._members[member][1], label))
                            elif self._members[member][1] == node[0]:
                                self._members[member][1] = new_label
                                self._nodes_occupied[(self._members[member][0], new_label)] = True
                                self._nodes_occupied[(new_label, self._members[member][0])] = True
                                self._nodes_occupied.pop((self._members[member][0], label))
                                self._nodes_occupied.pop((label, self._members[member][0]))

    def change_member_label(self, *args):
        """
        This method changes the label(s) of the specified member(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (label, new_label)

        label: String or Symbol
            The label of the member for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0), ('D', 5, 0))
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.change_node_label(('A', 'C'))
        >>> t.nodes
        [('C', 0, 0), ('B', 3, 0), ('D', 5, 0)]
        >>> t.add_member(('BC', 'B', 'C'), ('BD', 'B', 'D'))
        >>> t.members
        {'BC': ['B', 'C'], 'BD': ['B', 'D']}
        >>> t.change_member_label(('BC', 'BC_new'), ('BD', 'BD_new'))
        >>> t.members
        {'BC_new': ['B', 'C'], 'BD_new': ['B', 'D']}
        """
        for i in args:
            label = i[0]
            new_label = i[1]
            if label not in self._members:
                raise ValueError("No such member exists for the Truss")
            else:
                members_duplicate = list(self._members).copy()
                for member in members_duplicate:
                    if member == label:
                        self._members[new_label] = [self._members[member][0], self._members[member][1]]
                        self._members.pop(label)
                        self._member_lengths[new_label] = self._member_lengths[label]
                        self._member_lengths.pop(label)
                        self._internal_forces[new_label] = self._internal_forces[label]
                        self._internal_forces.pop(label)

    def apply_load(self, *args):
        """
        This method applies external load(s) at the specified node(s).

        Parameters
        ==========
        The input(s) of the method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied.

        magnitude: Sympifyable
            Magnitude of the load applied. It must always be positive and any changes in
            the direction of the load are not reflected here.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        """
        for i in args:
            location = i[0]
            magnitude = i[1]
            direction = i[2]
            magnitude = sympify(magnitude)
            direction = sympify(direction)

            if location not in self._node_coordinates:
                raise ValueError("Load must be applied at a known node")

            else:
                if location in self._loads:
                    self._loads[location].append([magnitude, direction])
                else:
                    self._loads[location] = [[magnitude, direction]]

    def remove_load(self, *args):
        """
        This method removes already
        present external load(s) at specified node(s).

        Parameters
        ==========
        The input(s) of this method are tuple(s) of the form (location, magnitude, direction).

        location: String or Symbol
            Label of the Node at which load is applied and is to be removed.

        magnitude: Sympifyable
            Magnitude of the load applied.

        direction: Sympifyable
            The angle, in degrees, that the load vector makes with the horizontal
            in the counter-clockwise direction. It takes the values 0 to 360,
            inclusive.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> from sympy import symbols
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> P = symbols('P')
        >>> t.apply_load(('A', P, 90), ('A', P/2, 45), ('A', P/4, 90))
        >>> t.loads
        {'A': [[P, 90], [P/2, 45], [P/4, 90]]}
        >>> t.remove_load(('A', P/4, 90), ('A', P/2, 45))
        >>> t.loads
        {'A': [[P, 90]]}
        """
        for i in args:
            location = i[0]
            magnitude = i[1]
            direction = i[2]
            magnitude = sympify(magnitude)
            direction = sympify(direction)

            if location not in self._node_coordinates:
                raise ValueError("Load must be removed from a known node")

            else:
                if [magnitude, direction] not in self._loads[location]:
                    raise ValueError("No load of this magnitude and direction has been applied at this node")
                else:
                    self._loads[location].remove([magnitude, direction])
            if self._loads[location] == []:
                self._loads.pop(location)

    def apply_support(self, *args):
        """
        This method adds a pinned or roller support at specified node(s).

        Parameters
        ==========
        The input(s) of this method are of the form (location, type).

        location: String or Symbol
            Label of the Node at which support is added.

        type: String
            Type of the support being provided at the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        """
        for i in args:
            location = i[0]
            type = i[1]
            if location not in self._node_coordinates:
                raise ValueError("Support must be added on a known node")

            else:
                if location not in self._supports:
                    if type == 'pinned':
                        self.apply_load((location, Symbol('R_'+str(location)+'_x'), 0))
                        self.apply_load((location, Symbol('R_'+str(location)+'_y'), 90))
                    elif type == 'roller':
                        self.apply_load((location, Symbol('R_'+str(location)+'_y'), 90))
                elif self._supports[location] == 'pinned':
                    if type == 'roller':
                        self.remove_load((location, Symbol('R_'+str(location)+'_x'), 0))
                elif self._supports[location] == 'roller':
                    if type == 'pinned':
                        self.apply_load((location, Symbol('R_'+str(location)+'_x'), 0))
                self._supports[location] = type

    def remove_support(self, *args):
        """
        This method removes support from specified node(s.)

        Parameters
        ==========

        locations: String or Symbol
            Label of the Node(s) at which support is to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(('A', 0, 0), ('B', 3, 0))
        >>> t.apply_support(('A', 'pinned'), ('B', 'roller'))
        >>> t.supports
        {'A': 'pinned', 'B': 'roller'}
        >>> t.remove_support('A','B')
        >>> t.supports
        {}
        """
        for location in args:

            if location not in self._node_coordinates:
                raise ValueError("No such node exists in the Truss")

            elif location not in self._supports:
                raise ValueError("No support has been added to the given node")

            else:
                if self._supports[location] == 'pinned':
                    self.remove_load((location, Symbol('R_'+str(location)+'_x'), 0))
                    self.remove_load((location, Symbol('R_'+str(location)+'_y'), 90))
                elif self._supports[location] == 'roller':
                    self.remove_load((location, Symbol('R_'+str(location)+'_y'), 90))
                self._supports.pop(location)

    def solve(self):
        """
        This method solves for all reaction forces of all supports and all internal forces
        of all the members in the truss, provided the Truss is solvable.

        A Truss is solvable if the following condition is met,

        2n >= r + m

        Where n is the number of nodes, r is the number of reaction forces, where each pinned
        support has 2 reaction forces and each roller has 1, and m is the number of members.

        The given condition is derived from the fact that a system of equations is solvable
        only when the number of variables is lesser than or equal to the number of equations.
        Equilibrium Equations in x and y directions give two equations per node giving 2n number
        equations. However, the truss needs to be stable as well and may be unstable if 2n > r + m.
        The number of variables is simply the sum of the number of reaction forces and member
        forces.

        .. note::
           The sign convention for the internal forces present in a member revolves around whether each
           force is compressive or tensile. While forming equations for each node, internal force due
           to a member on the node is assumed to be away from the node i.e. each force is assumed to
           be compressive by default. Hence, a positive value for an internal force implies the
           presence of compressive force in the member and a negative value implies a tensile force.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node(("node_1", 0, 0), ("node_2", 6, 0), ("node_3", 2, 2), ("node_4", 2, 0))
        >>> t.add_member(("member_1", "node_1", "node_4"), ("member_2", "node_2", "node_4"), ("member_3", "node_1", "node_3"))
        >>> t.add_member(("member_4", "node_2", "node_3"), ("member_5", "node_3", "node_4"))
        >>> t.apply_load(("node_4", 10, 270))
        >>> t.apply_support(("node_1", "pinned"), ("node_2", "roller"))
        >>> t.solve()
        >>> t.reaction_loads
        {'R_node_1_x': 0, 'R_node_1_y': 20/3, 'R_node_2_y': 10/3}
        >>> t.internal_forces
        {'member_1': 20/3, 'member_2': 20/3, 'member_3': -20*sqrt(2)/3, 'member_4': -10*sqrt(5)/3, 'member_5': 10}
        """
        count_reaction_loads = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    count_reaction_loads += 2
                elif self._supports[node[0]]=='roller':
                    count_reaction_loads += 1
        if 2*len(self._nodes) != len(self._members) + count_reaction_loads:
            raise ValueError("The given truss cannot be solved")
        coefficients_matrix = [[0 for i in range(2*len(self._nodes))] for j in range(2*len(self._nodes))]
        load_matrix = zeros(2*len(self.nodes), 1)
        load_matrix_row = 0
        for node in self._nodes:
            if node[0] in self._loads:
                for load in self._loads[node[0]]:
                    if load[0]!=Symbol('R_'+str(node[0])+'_x') and load[0]!=Symbol('R_'+str(node[0])+'_y'):
                        load_matrix[load_matrix_row] -= load[0]*cos(pi*load[1]/180)
                        load_matrix[load_matrix_row + 1] -= load[0]*sin(pi*load[1]/180)
            load_matrix_row += 2
        cols = 0
        row = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    coefficients_matrix[row][cols] += 1
                    coefficients_matrix[row+1][cols+1] += 1
                    cols += 2
                elif self._supports[node[0]]=='roller':
                    coefficients_matrix[row+1][cols] += 1
                    cols += 1
            row += 2
        for member in self._members:
            start = self._members[member][0]
            end = self._members[member][1]
            length = sqrt((self._node_coordinates[start][0]-self._node_coordinates[end][0])**2 + (self._node_coordinates[start][1]-self._node_coordinates[end][1])**2)
            start_index = self._node_labels.index(start)
            end_index = self._node_labels.index(end)
            horizontal_component_start = (self._node_coordinates[end][0]-self._node_coordinates[start][0])/length
            vertical_component_start = (self._node_coordinates[end][1]-self._node_coordinates[start][1])/length
            horizontal_component_end = (self._node_coordinates[start][0]-self._node_coordinates[end][0])/length
            vertical_component_end = (self._node_coordinates[start][1]-self._node_coordinates[end][1])/length
            coefficients_matrix[start_index*2][cols] += horizontal_component_start
            coefficients_matrix[start_index*2+1][cols] += vertical_component_start
            coefficients_matrix[end_index*2][cols] += horizontal_component_end
            coefficients_matrix[end_index*2+1][cols] += vertical_component_end
            cols += 1
        forces_matrix = (Matrix(coefficients_matrix)**-1)*load_matrix
        self._reaction_loads = {}
        i = 0
        min_load = inf
        for node in self._nodes:
            if node[0] in self._loads:
                for load in self._loads[node[0]]:
                    if type(load[0]) not in [Symbol, Mul, Add]:
                        min_load = min(min_load, load[0])
        for j in range(len(forces_matrix)):
            if type(forces_matrix[j]) not in [Symbol, Mul, Add]:
                if abs(forces_matrix[j]/min_load) <1E-10:
                    forces_matrix[j] = 0
        for node in self._nodes:
            if node[0] in self._supports:
                if self._supports[node[0]]=='pinned':
                    self._reaction_loads['R_'+str(node[0])+'_x'] = forces_matrix[i]
                    self._reaction_loads['R_'+str(node[0])+'_y'] = forces_matrix[i+1]
                    i += 2
                elif self._supports[node[0]]=='roller':
                    self._reaction_loads['R_'+str(node[0])+'_y'] = forces_matrix[i]
                    i += 1
        for member in self._members:
            self._internal_forces[member] = forces_matrix[i]
            i += 1
        return

    @doctest_depends_on(modules=('numpy',))
    def draw(self, subs_dict=None):
        """
        Returns a plot object of the Truss with all its nodes, members,
        supports and loads.

        .. note::
            The user must be careful while entering load values in their
            directions. The draw function assumes a sign convention that
            is used for plotting loads.

            Given a right-handed coordinate system with XYZ coordinates,
            the supports are assumed to be such that the reaction forces of a
            pinned support is in the +X and +Y direction while those of a
            roller support is in the +Y direction. For the load, the range
            of angles, one can input goes all the way to 360 degrees which, in the
            the plot is the angle that the load vector makes with the positive x-axis in the anticlockwise direction.

            For example, for a 90-degree angle, the load will be a vertically
            directed along +Y while a 270-degree angle denotes a vertical
            load as well but along -Y.

        Examples
        ========

        .. plot::
            :context: close-figs
            :format: doctest
            :include-source: True

            >>> from sympy.physics.continuum_mechanics.truss import Truss
            >>> import math
            >>> t = Truss()
            >>> t.add_node(("A", -4, 0), ("B", 0, 0), ("C", 4, 0), ("D", 8, 0))
            >>> t.add_node(("E", 6, 2/math.sqrt(3)))
            >>> t.add_node(("F", 2, 2*math.sqrt(3)))
            >>> t.add_node(("G", -2, 2/math.sqrt(3)))
            >>> t.add_member(("AB","A","B"), ("BC","B","C"), ("CD","C","D"))
            >>> t.add_member(("AG","A","G"), ("GB","G","B"), ("GF","G","F"))
            >>> t.add_member(("BF","B","F"), ("FC","F","C"), ("CE","C","E"))
            >>> t.add_member(("FE","F","E"), ("DE","D","E"))
            >>> t.apply_support(("A","pinned"), ("D","roller"))
            >>> t.apply_load(("G", 3, 90), ("E", 3, 90), ("F", 2, 90))
            >>> p = t.draw()
            >>> p  # doctest: +ELLIPSIS
            Plot object containing:
            [0]: cartesian line: 1 for x over (1.0, 1.0)
            ...
            >>> p.show()
        """
        if not numpy:
            raise ImportError("To use this function numpy module is required")

        x = Symbol('x')

        markers = []
        annotations = []
        rectangles = []

        node_markers = self._draw_nodes(subs_dict)
        markers += node_markers

        member_rectangles = self._draw_members()
        rectangles += member_rectangles

        support_markers = self._draw_supports()
        markers += support_markers

        load_annotations = self._draw_loads()
        annotations += load_annotations

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        lim = max(xmax*1.1-xmin*0.8+1, ymax*1.1-ymin*0.8+1)

        if lim==xmax*1.1-xmin*0.8+1:
            sing_plot = plot(1, (x, 1, 1), markers=markers, show=False, annotations=annotations, xlim=(xmin-0.05*lim, xmax*1.1), ylim=(xmin-0.05*lim, xmax*1.1), axis=False, rectangles=rectangles)

        else:
            sing_plot = plot(1, (x, 1, 1), markers=markers, show=False, annotations=annotations, xlim=(ymin-0.05*lim, ymax*1.1), ylim=(ymin-0.05*lim, ymax*1.1), axis=False, rectangles=rectangles)

        return sing_plot


    def _draw_nodes(self, subs_dict):
        node_markers = []

        for node in self._node_coordinates:
            if (type(self._node_coordinates[node][0]) in (Symbol, Quantity)):
                if self._node_coordinates[node][0] in subs_dict:
                    self._node_coordinates[node][0] = subs_dict[self._node_coordinates[node][0]]
                else:
                    raise ValueError("provided substituted dictionary is not adequate")
            elif (type(self._node_coordinates[node][0]) == Mul):
                objects = self._node_coordinates[node][0].as_coeff_Mul()
                for object in objects:
                    if type(object) in (Symbol, Quantity):
                        if subs_dict==None or object not in subs_dict:
                            raise ValueError("provided substituted dictionary is not adequate")
                        else:
                            self._node_coordinates[node][0] /= object
                            self._node_coordinates[node][0] *= subs_dict[object]

            if (type(self._node_coordinates[node][1]) in (Symbol, Quantity)):
                if self._node_coordinates[node][1] in subs_dict:
                    self._node_coordinates[node][1] = subs_dict[self._node_coordinates[node][1]]
                else:
                    raise ValueError("provided substituted dictionary is not adequate")
            elif (type(self._node_coordinates[node][1]) == Mul):
                objects = self._node_coordinates[node][1].as_coeff_Mul()
                for object in objects:
                    if type(object) in (Symbol, Quantity):
                        if subs_dict==None or object not in subs_dict:
                            raise ValueError("provided substituted dictionary is not adequate")
                        else:
                            self._node_coordinates[node][1] /= object
                            self._node_coordinates[node][1] *= subs_dict[object]

        for node in self._node_coordinates:
            node_markers.append(
                {
                    'args':[[self._node_coordinates[node][0]], [self._node_coordinates[node][1]]],
                    'marker':'o',
                    'markersize':5,
                    'color':'black'
                }
            )
        return node_markers

    def _draw_members(self):

        member_rectangles = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for member in self._members:
            x1 = self._node_coordinates[self._members[member][0]][0]
            y1 = self._node_coordinates[self._members[member][0]][1]
            x2 = self._node_coordinates[self._members[member][1]][0]
            y2 = self._node_coordinates[self._members[member][1]][1]
            if x2!=x1 and y2!=y1:
                if x2>x1:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff*cos(pi/4+atan((y2-y1)/(x2-x1)))/2, y1-0.005*max_diff*sin(pi/4+atan((y2-y1)/(x2-x1)))/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/math.sqrt(2),
                            'height':0.005*max_diff,
                            'angle':180*atan((y2-y1)/(x2-x1))/pi,
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x2-0.005*max_diff*cos(pi/4+atan((y2-y1)/(x2-x1)))/2, y2-0.005*max_diff*sin(pi/4+atan((y2-y1)/(x2-x1)))/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/math.sqrt(2),
                            'height':0.005*max_diff,
                            'angle':180*atan((y2-y1)/(x2-x1))/pi,
                            'color':'brown'
                        }
                    )
            elif y2==y1:
                if x2>x1:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2),
                            'height':0.005*max_diff,
                            'angle':90*(1-math.copysign(1, x2-x1)),
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2),
                            'height':-0.005*max_diff,
                            'angle':90*(1-math.copysign(1, x2-x1)),
                            'color':'brown'
                        }
                    )
            else:
                if y1<y2:
                    member_rectangles.append(
                        {
                            'xy':(x1-0.005*max_diff/2, y1-0.005*max_diff/2),
                            'width':sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/2,
                            'height':0.005*max_diff,
                            'angle':90*math.copysign(1, y2-y1),
                            'color':'brown'
                        }
                    )
                else:
                    member_rectangles.append(
                        {
                            'xy':(x2-0.005*max_diff/2, y2-0.005*max_diff/2),
                            'width':-(sqrt((x1-x2)**2+(y1-y2)**2)+0.005*max_diff/2),
                            'height':0.005*max_diff,
                            'angle':90*math.copysign(1, y2-y1),
                            'color':'brown'
                        }
                    )

        return member_rectangles

    def _draw_supports(self):
        support_markers = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])
        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin
        else:
            max_diff = 1.1*ymax-0.8*ymin

        for node in self._supports:
            if self._supports[node]=='pinned':
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]]
                        ],
                        'marker':6,
                        'markersize':15,
                        'color':'black',
                        'markerfacecolor':'none'
                    }
                )
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.035*max_diff]
                        ],
                        'marker':'_',
                        'markersize':14,
                        'color':'black'
                    }
                )

            elif self._supports[node]=='roller':
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.02*max_diff]
                        ],
                        'marker':'o',
                        'markersize':11,
                        'color':'black',
                        'markerfacecolor':'none'
                    }
                )
                support_markers.append(
                    {
                        'args':[
                            [self._node_coordinates[node][0]],
                            [self._node_coordinates[node][1]-0.0375*max_diff]
                        ],
                        'marker':'_',
                        'markersize':14,
                        'color':'black'
                    }
                )
        return support_markers

    def _draw_loads(self):
        load_annotations = []

        xmax = -INF
        xmin = INF
        ymax = -INF
        ymin = INF

        for node in self._node_coordinates:
            xmax = max(xmax, self._node_coordinates[node][0])
            xmin = min(xmin, self._node_coordinates[node][0])
            ymax = max(ymax, self._node_coordinates[node][1])
            ymin = min(ymin, self._node_coordinates[node][1])

        if abs(1.1*xmax-0.8*xmin)>abs(1.1*ymax-0.8*ymin):
            max_diff = 1.1*xmax-0.8*xmin+5
        else:
            max_diff = 1.1*ymax-0.8*ymin+5

        for node in self._loads:
            for load in self._loads[node]:
                if load[0] in [Symbol('R_'+str(node)+'_x'), Symbol('R_'+str(node)+'_y')]:
                    continue
                x = self._node_coordinates[node][0]
                y = self._node_coordinates[node][1]
                load_annotations.append(
                    {
                        'text':'',
                        'xy':(
                            x-math.cos(pi*load[1]/180)*(max_diff/100),
                            y-math.sin(pi*load[1]/180)*(max_diff/100)
                        ),
                        'xytext':(
                            x-(max_diff/100+abs(xmax-xmin)+abs(ymax-ymin))*math.cos(pi*load[1]/180)/20,
                            y-(max_diff/100+abs(xmax-xmin)+abs(ymax-ymin))*math.sin(pi*load[1]/180)/20
                        ),
                        'arrowprops':{'width':1.5, 'headlength':5, 'headwidth':5, 'facecolor':'black'}
                    }
                )
        return load_annotations
<|endoftext|>"""
Gaussian optics.

The module implements:

- Ray transfer matrices for geometrical and gaussian optics.

  See RayTransferMatrix, GeometricRay and BeamParameter

- Conjugation relations for geometrical and gaussian optics.

  See geometric_conj*, gauss_conj and conjugate_gauss_beams

The conventions for the distances are as follows:

focal distance
    positive for convergent lenses
object distance
    positive for real objects
image distance
    positive for real images
"""

__all__ = [
    'RayTransferMatrix',
    'FreeSpace',
    'FlatRefraction',
    'CurvedRefraction',
    'FlatMirror',
    'CurvedMirror',
    'ThinLens',
    'GeometricRay',
    'BeamParameter',
    'waist2rayleigh',
    'rayleigh2waist',
    'geometric_conj_ab',
    'geometric_conj_af',
    'geometric_conj_bf',
    'gaussian_conj',
    'conjugate_gauss_beams',
]


from sympy.core.expr import Expr
from sympy.core.numbers import (I, pi)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.polys.rationaltools import together
from sympy.utilities.misc import filldedent

###
# A, B, C, D matrices
###


class RayTransferMatrix(MutableDenseMatrix):
    """
    Base class for a Ray Transfer Matrix.

    It should be used if there is not already a more specific subclass mentioned
    in See Also.

    Parameters
    ==========

    parameters :
        A, B, C and D or 2x2 matrix (Matrix(2, 2, [A, B, C, D]))

    Examples
    ========

    >>> from sympy.physics.optics import RayTransferMatrix, ThinLens
    >>> from sympy import Symbol, Matrix

    >>> mat = RayTransferMatrix(1, 2, 3, 4)
    >>> mat
    Matrix([
    [1, 2],
    [3, 4]])

    >>> RayTransferMatrix(Matrix([[1, 2], [3, 4]]))
    Matrix([
    [1, 2],
    [3, 4]])

    >>> mat.A
    1

    >>> f = Symbol('f')
    >>> lens = ThinLens(f)
    >>> lens
    Matrix([
    [   1, 0],
    [-1/f, 1]])

    >>> lens.C
    -1/f

    See Also
    ========

    GeometricRay, BeamParameter,
    FreeSpace, FlatRefraction, CurvedRefraction,
    FlatMirror, CurvedMirror, ThinLens

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    """

    def __new__(cls, *args):

        if len(args) == 4:
            temp = ((args[0], args[1]), (args[2], args[3]))
        elif len(args) == 1 \
            and isinstance(args[0], Matrix) \
                and args[0].shape == (2, 2):
            temp = args[0]
        else:
            raise ValueError(filldedent('''
                Expecting 2x2 Matrix or the 4 elements of
                the Matrix but got %s''' % str(args)))
        return Matrix.__new__(cls, temp)

    def __mul__(self, other):
        if isinstance(other, RayTransferMatrix):
            return RayTransferMatrix(Matrix(self)*Matrix(other))
        elif isinstance(other, GeometricRay):
            return GeometricRay(Matrix(self)*Matrix(other))
        elif isinstance(other, BeamParameter):
            temp = Matrix(self)*Matrix(((other.q,), (1,)))
            q = (temp[0]/temp[1]).expand(complex=True)
            return BeamParameter(other.wavelen,
                                 together(re(q)),
                                 z_r=together(im(q)))
        else:
            return Matrix.__mul__(self, other)

    @property
    def A(self):
        """
        The A parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.A
        1
        """
        return self[0, 0]

    @property
    def B(self):
        """
        The B parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.B
        2
        """
        return self[0, 1]

    @property
    def C(self):
        """
        The C parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.C
        3
        """
        return self[1, 0]

    @property
    def D(self):
        """
        The D parameter of the Matrix.

        Examples
        ========

        >>> from sympy.physics.optics import RayTransferMatrix
        >>> mat = RayTransferMatrix(1, 2, 3, 4)
        >>> mat.D
        4
        """
        return self[1, 1]


class FreeSpace(RayTransferMatrix):
    """
    Ray Transfer Matrix for free space.

    Parameters
    ==========

    distance

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FreeSpace
    >>> from sympy import symbols
    >>> d = symbols('d')
    >>> FreeSpace(d)
    Matrix([
    [1, d],
    [0, 1]])
    """
    def __new__(cls, d):
        return RayTransferMatrix.__new__(cls, 1, d, 0, 1)


class FlatRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction.

    Parameters
    ==========

    n1 :
        Refractive index of one medium.
    n2 :
        Refractive index of other medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatRefraction
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1 n2')
    >>> FlatRefraction(n1, n2)
    Matrix([
    [1,     0],
    [0, n1/n2]])
    """
    def __new__(cls, n1, n2):
        n1, n2 = map(sympify, (n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, 0, n1/n2)


class CurvedRefraction(RayTransferMatrix):
    """
    Ray Transfer Matrix for refraction on curved interface.

    Parameters
    ==========

    R :
        Radius of curvature (positive for concave).
    n1 :
        Refractive index of one medium.
    n2 :
        Refractive index of other medium.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedRefraction
    >>> from sympy import symbols
    >>> R, n1, n2 = symbols('R n1 n2')
    >>> CurvedRefraction(R, n1, n2)
    Matrix([
    [               1,     0],
    [(n1 - n2)/(R*n2), n1/n2]])
    """
    def __new__(cls, R, n1, n2):
        R, n1, n2 = map(sympify, (R, n1, n2))
        return RayTransferMatrix.__new__(cls, 1, 0, (n1 - n2)/R/n2, n1/n2)


class FlatMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import FlatMirror
    >>> FlatMirror()
    Matrix([
    [1, 0],
    [0, 1]])
    """
    def __new__(cls):
        return RayTransferMatrix.__new__(cls, 1, 0, 0, 1)


class CurvedMirror(RayTransferMatrix):
    """
    Ray Transfer Matrix for reflection from curved surface.

    Parameters
    ==========

    R : radius of curvature (positive for concave)

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import CurvedMirror
    >>> from sympy import symbols
    >>> R = symbols('R')
    >>> CurvedMirror(R)
    Matrix([
    [   1, 0],
    [-2/R, 1]])
    """
    def __new__(cls, R):
        R = sympify(R)
        return RayTransferMatrix.__new__(cls, 1, 0, -2/R, 1)


class ThinLens(RayTransferMatrix):
    """
    Ray Transfer Matrix for a thin lens.

    Parameters
    ==========

    f :
        The focal distance.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import ThinLens
    >>> from sympy import symbols
    >>> f = symbols('f')
    >>> ThinLens(f)
    Matrix([
    [   1, 0],
    [-1/f, 1]])
    """
    def __new__(cls, f):
        f = sympify(f)
        return RayTransferMatrix.__new__(cls, 1, 0, -1/f, 1)


###
# Representation for geometric ray
###

class GeometricRay(MutableDenseMatrix):
    """
    Representation for a geometric ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    h : height, and
    angle : angle, or
    matrix : a 2x1 matrix (Matrix(2, 1, [height, angle]))

    Examples
    ========

    >>> from sympy.physics.optics import GeometricRay, FreeSpace
    >>> from sympy import symbols, Matrix
    >>> d, h, angle = symbols('d, h, angle')

    >>> GeometricRay(h, angle)
    Matrix([
    [    h],
    [angle]])

    >>> FreeSpace(d)*GeometricRay(h, angle)
    Matrix([
    [angle*d + h],
    [      angle]])

    >>> GeometricRay( Matrix( ((h,), (angle,)) ) )
    Matrix([
    [    h],
    [angle]])

    See Also
    ========

    RayTransferMatrix

    """

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], Matrix) \
                and args[0].shape == (2, 1):
            temp = args[0]
        elif len(args) == 2:
            temp = ((args[0],), (args[1],))
        else:
            raise ValueError(filldedent('''
                Expecting 2x1 Matrix or the 2 elements of
                the Matrix but got %s''' % str(args)))
        return Matrix.__new__(cls, temp)

    @property
    def height(self):
        """
        The distance from the optical axis.

        Examples
        ========

        >>> from sympy.physics.optics import GeometricRay
        >>> from sympy import symbols
        >>> h, angle = symbols('h, angle')
        >>> gRay = GeometricRay(h, angle)
        >>> gRay.height
        h
        """
        return self[0]

    @property
    def angle(self):
        """
        The angle with the optical axis.

        Examples
        ========

        >>> from sympy.physics.optics import GeometricRay
        >>> from sympy import symbols
        >>> h, angle = symbols('h, angle')
        >>> gRay = GeometricRay(h, angle)
        >>> gRay.angle
        angle
        """
        return self[1]


###
# Representation for gauss beam
###

class BeamParameter(Expr):
    """
    Representation for a gaussian ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    wavelen : the wavelength,
    z : the distance to waist, and
    w : the waist, or
    z_r : the rayleigh range.
    n : the refractive index of medium.

    Examples
    ========

    >>> from sympy.physics.optics import BeamParameter
    >>> p = BeamParameter(530e-9, 1, w=1e-3)
    >>> p.q
    1 + 1.88679245283019*I*pi

    >>> p.q.n()
    1.0 + 5.92753330865999*I
    >>> p.w_0.n()
    0.00100000000000000
    >>> p.z_r.n()
    5.92753330865999

    >>> from sympy.physics.optics import FreeSpace
    >>> fs = FreeSpace(10)
    >>> p1 = fs*p
    >>> p.w.n()
    0.00101413072159615
    >>> p1.w.n()
    0.00210803120913829

    See Also
    ========

    RayTransferMatrix

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_beam_parameter
    .. [2] https://en.wikipedia.org/wiki/Gaussian_beam
    """
    #TODO A class Complex may be implemented. The BeamParameter may
    # subclass it. See:
    # https://groups.google.com/d/topic/sympy/7XkU07NRBEs/discussion

    def __new__(cls, wavelen, z, z_r=None, w=None, n=1):
        wavelen = sympify(wavelen)
        z = sympify(z)
        n = sympify(n)

        if z_r is not None and w is None:
            z_r = sympify(z_r)
        elif w is not None and z_r is None:
            z_r = waist2rayleigh(sympify(w), wavelen, n)
        elif z_r is None and w is None:
            raise ValueError('Must specify one of w and z_r.')

        return Expr.__new__(cls, wavelen, z, z_r, n)

    @property
    def wavelen(self):
        return self.args[0]

    @property
    def z(self):
        return self.args[1]

    @property
    def z_r(self):
        return self.args[2]

    @property
    def n(self):
        return self.args[3]

    @property
    def q(self):
        """
        The complex parameter representing the beam.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.q
        1 + 1.88679245283019*I*pi
        """
        return self.z + I*self.z_r

    @property
    def radius(self):
        """
        The radius of curvature of the phase front.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.radius
        1 + 3.55998576005696*pi**2
        """
        return self.z*(1 + (self.z_r/self.z)**2)

    @property
    def w(self):
        """
        The radius of the beam w(z), at any position z along the beam.
        The beam radius at `1/e^2` intensity (axial value).

        See Also
        ========

        w_0 :
            The minimal radius of beam.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.w
        0.001*sqrt(0.2809/pi**2 + 1)
        """
        return self.w_0*sqrt(1 + (self.z/self.z_r)**2)

    @property
    def w_0(self):
        """
         The minimal radius of beam at `1/e^2` intensity (peak value).

        See Also
        ========

        w : the beam radius at `1/e^2` intensity (axial value).

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.w_0
        0.00100000000000000
        """
        return sqrt(self.z_r/(pi*self.n)*self.wavelen)

    @property
    def divergence(self):
        """
        Half of the total angular spread.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.divergence
        0.00053/pi
        """
        return self.wavelen/pi/self.w_0

    @property
    def gouy(self):
        """
        The Gouy phase.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.gouy
        atan(0.53/pi)
        """
        return atan2(self.z, self.z_r)

    @property
    def waist_approximation_limit(self):
        """
        The minimal waist for which the gauss beam approximation is valid.

        Explanation
        ===========

        The gauss beam is a solution to the paraxial equation. For curvatures
        that are too great it is not a valid approximation.

        Examples
        ========

        >>> from sympy.physics.optics import BeamParameter
        >>> p = BeamParameter(530e-9, 1, w=1e-3)
        >>> p.waist_approximation_limit
        1.06e-6/pi
        """
        return 2*self.wavelen/pi


###
# Utilities
###

def waist2rayleigh(w, wavelen, n=1):
    """
    Calculate the rayleigh range from the waist of a gaussian beam.

    See Also
    ========

    rayleigh2waist, BeamParameter

    Examples
    ========

    >>> from sympy.physics.optics import waist2rayleigh
    >>> from sympy import symbols
    >>> w, wavelen = symbols('w wavelen')
    >>> waist2rayleigh(w, wavelen)
    pi*w**2/wavelen
    """
    w, wavelen = map(sympify, (w, wavelen))
    return w**2*n*pi/wavelen


def rayleigh2waist(z_r, wavelen):
    """Calculate the waist from the rayleigh range of a gaussian beam.

    See Also
    ========

    waist2rayleigh, BeamParameter

    Examples
    ========

    >>> from sympy.physics.optics import rayleigh2waist
    >>> from sympy import symbols
    >>> z_r, wavelen = symbols('z_r wavelen')
    >>> rayleigh2waist(z_r, wavelen)
    sqrt(wavelen*z_r)/sqrt(pi)
    """
    z_r, wavelen = map(sympify, (z_r, wavelen))
    return sqrt(z_r/pi*wavelen)


def geometric_conj_ab(a, b):
    """
    Conjugation relation for geometrical beams under paraxial conditions.

    Explanation
    ===========

    Takes the distances to the optical element and returns the needed
    focal distance.

    See Also
    ========

    geometric_conj_af, geometric_conj_bf

    Examples
    ========

    >>> from sympy.physics.optics import geometric_conj_ab
    >>> from sympy import symbols
    >>> a, b = symbols('a b')
    >>> geometric_conj_ab(a, b)
    a*b/(a + b)
    """
    a, b = map(sympify, (a, b))
    if a.is_infinite or b.is_infinite:
        return a if b.is_infinite else b
    else:
        return a*b/(a + b)


def geometric_conj_af(a, f):
    """
    Conjugation relation for geometrical beams under paraxial conditions.

    Explanation
    ===========

    Takes the object distance (for geometric_conj_af) or the image distance
    (for geometric_conj_bf) to the optical element and the focal distance.
    Then it returns the other distance needed for conjugation.

    See Also
    ========

    geometric_conj_ab

    Examples
    ========

    >>> from sympy.physics.optics.gaussopt import geometric_conj_af, geometric_conj_bf
    >>> from sympy import symbols
    >>> a, b, f = symbols('a b f')
    >>> geometric_conj_af(a, f)
    a*f/(a - f)
    >>> geometric_conj_bf(b, f)
    b*f/(b - f)
    """
    a, f = map(sympify, (a, f))
    return -geometric_conj_ab(a, -f)

geometric_conj_bf = geometric_conj_af


def gaussian_conj(s_in, z_r_in, f):
    """
    Conjugation relation for gaussian beams.

    Parameters
    ==========

    s_in :
        The distance to optical element from the waist.
    z_r_in :
        The rayleigh range of the incident beam.
    f :
        The focal length of the optical element.

    Returns
    =======

    a tuple containing (s_out, z_r_out, m)
    s_out :
        The distance between the new waist and the optical element.
    z_r_out :
        The rayleigh range of the emergent beam.
    m :
        The ration between the new and the old waists.

    Examples
    ========

    >>> from sympy.physics.optics import gaussian_conj
    >>> from sympy import symbols
    >>> s_in, z_r_in, f = symbols('s_in z_r_in f')

    >>> gaussian_conj(s_in, z_r_in, f)[0]
    1/(-1/(s_in + z_r_in**2/(-f + s_in)) + 1/f)

    >>> gaussian_conj(s_in, z_r_in, f)[1]
    z_r_in/(1 - s_in**2/f**2 + z_r_in**2/f**2)

    >>> gaussian_conj(s_in, z_r_in, f)[2]
    1/sqrt(1 - s_in**2/f**2 + z_r_in**2/f**2)
    """
    s_in, z_r_in, f = map(sympify, (s_in, z_r_in, f))
    s_out = 1 / ( -1/(s_in + z_r_in**2/(s_in - f)) + 1/f )
    m = 1/sqrt((1 - (s_in/f)**2) + (z_r_in/f)**2)
    z_r_out = z_r_in / ((1 - (s_in/f)**2) + (z_r_in/f)**2)
    return (s_out, z_r_out, m)


def conjugate_gauss_beams(wavelen, waist_in, waist_out, **kwargs):
    """
    Find the optical setup conjugating the object/image waists.

    Parameters
    ==========

    wavelen :
        The wavelength of the beam.
    waist_in and waist_out :
        The waists to be conjugated.
    f :
        The focal distance of the element used in the conjugation.

    Returns
    =======

    a tuple containing (s_in, s_out, f)
    s_in :
        The distance before the optical element.
    s_out :
        The distance after the optical element.
    f :
        The focal distance of the optical element.

    Examples
    ========

    >>> from sympy.physics.optics import conjugate_gauss_beams
    >>> from sympy import symbols, factor
    >>> l, w_i, w_o, f = symbols('l w_i w_o f')

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[0]
    f*(1 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))

    >>> factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1])
    f*w_o**2*(w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 -
              pi**2*w_i**4/(f**2*l**2)))/w_i**2

    >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[2]
    f
    """
    #TODO add the other possible arguments
    wavelen, waist_in, waist_out = map(sympify, (wavelen, waist_in, waist_out))
    m = waist_out / waist_in
    z = waist2rayleigh(waist_in, wavelen)
    if len(kwargs) != 1:
        raise ValueError("The function expects only one named argument")
    elif 'dist' in kwargs:
        raise NotImplementedError(filldedent('''
            Currently only focal length is supported as a parameter'''))
    elif 'f' in kwargs:
        f = sympify(kwargs['f'])
        s_in = f * (1 - sqrt(1/m**2 - z**2/f**2))
        s_out = gaussian_conj(s_in, z, f)[0]
    elif 's_in' in kwargs:
        raise NotImplementedError(filldedent('''
            Currently only focal length is supported as a parameter'''))
    else:
        raise ValueError(filldedent('''
            The functions expects the focal length as a named argument'''))
    return (s_in, s_out, f)

#TODO
#def plot_beam():
#    """Plot the beam radius as it propagates in space."""
#    pass

#TODO
#def plot_beam_conjugation():
#    """
#    Plot the intersection of two beams.
#
#    Represents the conjugation relation.
#
#    See Also
#    ========
#
#    conjugate_gauss_beams
#    """
#    pass
<|endoftext|>"""
**Contains**

* refraction_angle
* fresnel_coefficients
* deviation
* brewster_angle
* critical_angle
* lens_makers_formula
* mirror_formula
* lens_formula
* hyperfocal_distance
* transverse_magnification
"""

__all__ = ['refraction_angle',
           'deviation',
           'fresnel_coefficients',
           'brewster_angle',
           'critical_angle',
           'lens_makers_formula',
           'mirror_formula',
           'lens_formula',
           'hyperfocal_distance',
           'transverse_magnification'
           ]

from sympy.core.numbers import (Float, I, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2, cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import cancel
from sympy.series.limits import Limit
from sympy.geometry.line import Ray3D
from sympy.geometry.util import intersection
from sympy.geometry.plane import Plane
from sympy.utilities.iterables import is_sequence
from .medium import Medium


def refractive_index_of_medium(medium):
    """
    Helper function that returns refractive index, given a medium
    """
    if isinstance(medium, Medium):
        n = medium.refractive_index
    else:
        n = sympify(medium)
    return n


def refraction_angle(incident, medium1, medium2, normal=None, plane=None):
    """
    This function calculates transmitted vector after refraction at planar
    surface. ``medium1`` and ``medium2`` can be ``Medium`` or any sympifiable object.
    If ``incident`` is a number then treated as angle of incidence (in radians)
    in which case refraction angle is returned.

    If ``incident`` is an object of `Ray3D`, `normal` also has to be an instance
    of `Ray3D` in order to get the output as a `Ray3D`. Please note that if
    plane of separation is not provided and normal is an instance of `Ray3D`,
    ``normal`` will be assumed to be intersecting incident ray at the plane of
    separation. This will not be the case when `normal` is a `Matrix` or
    any other sequence.
    If ``incident`` is an instance of `Ray3D` and `plane` has not been provided
    and ``normal`` is not `Ray3D`, output will be a `Matrix`.

    Parameters
    ==========

    incident : Matrix, Ray3D, sequence or a number
        Incident vector or angle of incidence
    medium1 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 1 or its refractive index
    medium2 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 2 or its refractive index
    normal : Matrix, Ray3D, or sequence
        Normal vector
    plane : Plane
        Plane of separation of the two media.

    Returns
    =======

    Returns an angle of refraction or a refracted ray depending on inputs.

    Examples
    ========

    >>> from sympy.physics.optics import refraction_angle
    >>> from sympy.geometry import Point3D, Ray3D, Plane
    >>> from sympy.matrices import Matrix
    >>> from sympy import symbols, pi
    >>> n = Matrix([0, 0, 1])
    >>> P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    >>> r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    >>> refraction_angle(r1, 1, 1, n)
    Matrix([
    [ 1],
    [ 1],
    [-1]])
    >>> refraction_angle(r1, 1, 1, plane=P)
    Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))

    With different index of refraction of the two media

    >>> n1, n2 = symbols('n1, n2')
    >>> refraction_angle(r1, n1, n2, n)
    Matrix([
    [                                n1/n2],
    [                                n1/n2],
    [-sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)]])
    >>> refraction_angle(r1, n1, n2, plane=P)
    Ray3D(Point3D(0, 0, 0), Point3D(n1/n2, n1/n2, -sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)))
    >>> round(refraction_angle(pi/6, 1.2, 1.5), 5)
    0.41152
    """

    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    # check if an incidence angle was supplied instead of a ray
    try:
        angle_of_incidence = float(incident)
    except TypeError:
        angle_of_incidence = None

    try:
        critical_angle_ = critical_angle(medium1, medium2)
    except (ValueError, TypeError):
        critical_angle_ = None

    if angle_of_incidence is not None:
        if normal is not None or plane is not None:
            raise ValueError('Normal/plane not allowed if incident is an angle')

        if not 0.0 <= angle_of_incidence < pi*0.5:
            raise ValueError('Angle of incidence not in range [0:pi/2)')

        if critical_angle_ and angle_of_incidence > critical_angle_:
            raise ValueError('Ray undergoes total internal reflection')
        return asin(n1*sin(angle_of_incidence)/n2)

    # Treat the incident as ray below
    # A flag to check whether to return Ray3D or not
    return_ray = False

    if plane is not None and normal is not None:
        raise ValueError("Either plane or normal is acceptable.")

    if not isinstance(incident, Matrix):
        if is_sequence(incident):
            _incident = Matrix(incident)
        elif isinstance(incident, Ray3D):
            _incident = Matrix(incident.direction_ratio)
        else:
            raise TypeError(
                "incident should be a Matrix, Ray3D, or sequence")
    else:
        _incident = incident

    # If plane is provided, get direction ratios of the normal
    # to the plane from the plane else go with `normal` param.
    if plane is not None:
        if not isinstance(plane, Plane):
            raise TypeError("plane should be an instance of geometry.plane.Plane")
        # If we have the plane, we can get the intersection
        # point of incident ray and the plane and thus return
        # an instance of Ray3D.
        if isinstance(incident, Ray3D):
            return_ray = True
            intersection_pt = plane.intersection(incident)[0]
        _normal = Matrix(plane.normal_vector)
    else:
        if not isinstance(normal, Matrix):
            if is_sequence(normal):
                _normal = Matrix(normal)
            elif isinstance(normal, Ray3D):
                _normal = Matrix(normal.direction_ratio)
                if isinstance(incident, Ray3D):
                    intersection_pt = intersection(incident, normal)
                    if len(intersection_pt) == 0:
                        raise ValueError(
                            "Normal isn't concurrent with the incident ray.")
                    else:
                        return_ray = True
                        intersection_pt = intersection_pt[0]
            else:
                raise TypeError(
                    "Normal should be a Matrix, Ray3D, or sequence")
        else:
            _normal = normal

    eta = n1/n2  # Relative index of refraction
    # Calculating magnitude of the vectors
    mag_incident = sqrt(sum(i**2 for i in _incident))
    mag_normal = sqrt(sum(i**2 for i in _normal))
    # Converting vectors to unit vectors by dividing
    # them with their magnitudes
    _incident /= mag_incident
    _normal /= mag_normal
    c1 = -_incident.dot(_normal)  # cos(angle_of_incidence)
    cs2 = 1 - eta**2*(1 - c1**2)  # cos(angle_of_refraction)**2
    if cs2.is_negative:  # This is the case of total internal reflection(TIR).
        return S.Zero
    drs = eta*_incident + (eta*c1 - sqrt(cs2))*_normal
    # Multiplying unit vector by its magnitude
    drs = drs*mag_incident
    if not return_ray:
        return drs
    else:
        return Ray3D(intersection_pt, direction_ratio=drs)


def fresnel_coefficients(angle_of_incidence, medium1, medium2):
    """
    This function uses Fresnel equations to calculate reflection and
    transmission coefficients. Those are obtained for both polarisations
    when the electric field vector is in the plane of incidence (labelled 'p')
    and when the electric field vector is perpendicular to the plane of
    incidence (labelled 's'). There are four real coefficients unless the
    incident ray reflects in total internal in which case there are two complex
    ones. Angle of incidence is the angle between the incident ray and the
    surface normal. ``medium1`` and ``medium2`` can be ``Medium`` or any
    sympifiable object.

    Parameters
    ==========

    angle_of_incidence : sympifiable

    medium1 : Medium or sympifiable
        Medium 1 or its refractive index

    medium2 : Medium or sympifiable
        Medium 2 or its refractive index

    Returns
    =======

    Returns a list with four real Fresnel coefficients:
    [reflection p (TM), reflection s (TE),
    transmission p (TM), transmission s (TE)]
    If the ray is undergoes total internal reflection then returns a
    list of two complex Fresnel coefficients:
    [reflection p (TM), reflection s (TE)]

    Examples
    ========

    >>> from sympy.physics.optics import fresnel_coefficients
    >>> fresnel_coefficients(0.3, 1, 2)
    [0.317843553417859, -0.348645229818821,
            0.658921776708929, 0.651354770181179]
    >>> fresnel_coefficients(0.6, 2, 1)
    [-0.235625382192159 - 0.971843958291041*I,
             0.816477005968898 - 0.577377951366403*I]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fresnel_equations
    """
    if not 0 <= 2*angle_of_incidence < pi:
        raise ValueError('Angle of incidence not in range [0:pi/2)')

    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    angle_of_refraction = asin(n1*sin(angle_of_incidence)/n2)
    try:
        angle_of_total_internal_reflection_onset = critical_angle(n1, n2)
    except ValueError:
        angle_of_total_internal_reflection_onset = None

    if angle_of_total_internal_reflection_onset is None or\
    angle_of_total_internal_reflection_onset > angle_of_incidence:
        R_s = -sin(angle_of_incidence - angle_of_refraction)\
                /sin(angle_of_incidence + angle_of_refraction)
        R_p = tan(angle_of_incidence - angle_of_refraction)\
                /tan(angle_of_incidence + angle_of_refraction)
        T_s = 2*sin(angle_of_refraction)*cos(angle_of_incidence)\
                /sin(angle_of_incidence + angle_of_refraction)
        T_p = 2*sin(angle_of_refraction)*cos(angle_of_incidence)\
                /(sin(angle_of_incidence + angle_of_refraction)\
                *cos(angle_of_incidence - angle_of_refraction))
        return [R_p, R_s, T_p, T_s]
    else:
        n = n2/n1
        R_s = cancel((cos(angle_of_incidence)-\
                I*sqrt(sin(angle_of_incidence)**2 - n**2))\
                /(cos(angle_of_incidence)+\
                I*sqrt(sin(angle_of_incidence)**2 - n**2)))
        R_p = cancel((n**2*cos(angle_of_incidence)-\
                I*sqrt(sin(angle_of_incidence)**2 - n**2))\
                /(n**2*cos(angle_of_incidence)+\
                I*sqrt(sin(angle_of_incidence)**2 - n**2)))
        return [R_p, R_s]


def deviation(incident, medium1, medium2, normal=None, plane=None):
    """
    This function calculates the angle of deviation of a ray
    due to refraction at planar surface.

    Parameters
    ==========

    incident : Matrix, Ray3D, sequence or float
        Incident vector or angle of incidence
    medium1 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 1 or its refractive index
    medium2 : sympy.physics.optics.medium.Medium or sympifiable
        Medium 2 or its refractive index
    normal : Matrix, Ray3D, or sequence
        Normal vector
    plane : Plane
        Plane of separation of the two media.

    Returns angular deviation between incident and refracted rays

    Examples
    ========

    >>> from sympy.physics.optics import deviation
    >>> from sympy.geometry import Point3D, Ray3D, Plane
    >>> from sympy.matrices import Matrix
    >>> from sympy import symbols
    >>> n1, n2 = symbols('n1, n2')
    >>> n = Matrix([0, 0, 1])
    >>> P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    >>> r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    >>> deviation(r1, 1, 1, n)
    0
    >>> deviation(r1, n1, n2, plane=P)
    -acos(-sqrt(-2*n1**2/(3*n2**2) + 1)) + acos(-sqrt(3)/3)
    >>> round(deviation(0.1, 1.2, 1.5), 5)
    -0.02005
    """
    refracted = refraction_angle(incident,
                                 medium1,
                                 medium2,
                                 normal=normal,
                                 plane=plane)
    try:
        angle_of_incidence = Float(incident)
    except TypeError:
        angle_of_incidence = None

    if angle_of_incidence is not None:
        return float(refracted) - angle_of_incidence

    if refracted != 0:
        if isinstance(refracted, Ray3D):
            refracted = Matrix(refracted.direction_ratio)

        if not isinstance(incident, Matrix):
            if is_sequence(incident):
                _incident = Matrix(incident)
            elif isinstance(incident, Ray3D):
                _incident = Matrix(incident.direction_ratio)
            else:
                raise TypeError(
                    "incident should be a Matrix, Ray3D, or sequence")
        else:
            _incident = incident

        if plane is None:
            if not isinstance(normal, Matrix):
                if is_sequence(normal):
                    _normal = Matrix(normal)
                elif isinstance(normal, Ray3D):
                    _normal = Matrix(normal.direction_ratio)
                else:
                    raise TypeError(
                        "normal should be a Matrix, Ray3D, or sequence")
            else:
                _normal = normal
        else:
            _normal = Matrix(plane.normal_vector)

        mag_incident = sqrt(sum(i**2 for i in _incident))
        mag_normal = sqrt(sum(i**2 for i in _normal))
        mag_refracted = sqrt(sum(i**2 for i in refracted))
        _incident /= mag_incident
        _normal /= mag_normal
        refracted /= mag_refracted
        i = acos(_incident.dot(_normal))
        r = acos(refracted.dot(_normal))
        return i - r


def brewster_angle(medium1, medium2):
    """
    This function calculates the Brewster's angle of incidence to Medium 2 from
    Medium 1 in radians.

    Parameters
    ==========

    medium 1 : Medium or sympifiable
        Refractive index of Medium 1
    medium 2 : Medium or sympifiable
        Refractive index of Medium 1

    Examples
    ========

    >>> from sympy.physics.optics import brewster_angle
    >>> brewster_angle(1, 1.33)
    0.926093295503462

    """

    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    return atan2(n2, n1)

def critical_angle(medium1, medium2):
    """
    This function calculates the critical angle of incidence (marking the onset
    of total internal) to Medium 2 from Medium 1 in radians.

    Parameters
    ==========

    medium 1 : Medium or sympifiable
        Refractive index of Medium 1.
    medium 2 : Medium or sympifiable
        Refractive index of Medium 1.

    Examples
    ========

    >>> from sympy.physics.optics import critical_angle
    >>> critical_angle(1.33, 1)
    0.850908514477849

    """

    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)

    if n2 > n1:
        raise ValueError('Total internal reflection impossible for n1 < n2')
    else:
        return asin(n2/n1)



def lens_makers_formula(n_lens, n_surr, r1, r2, d=0):
    """
    This function calculates focal length of a lens.
    It follows cartesian sign convention.

    Parameters
    ==========

    n_lens : Medium or sympifiable
        Index of refraction of lens.
    n_surr : Medium or sympifiable
        Index of reflection of surrounding.
    r1 : sympifiable
        Radius of curvature of first surface.
    r2 : sympifiable
        Radius of curvature of second surface.
    d : sympifiable, optional
        Thickness of lens, default value is 0.

    Examples
    ========

    >>> from sympy.physics.optics import lens_makers_formula
    >>> from sympy import S
    >>> lens_makers_formula(1.33, 1, 10, -10)
    15.1515151515151
    >>> lens_makers_formula(1.2, 1, 10, S.Infinity)
    50.0000000000000
    >>> lens_makers_formula(1.33, 1, 10, -10, d=1)
    15.3418463277618

    """

    if isinstance(n_lens, Medium):
        n_lens = n_lens.refractive_index
    else:
        n_lens = sympify(n_lens)
    if isinstance(n_surr, Medium):
        n_surr = n_surr.refractive_index
    else:
        n_surr = sympify(n_surr)
    d = sympify(d)

    focal_length = 1/((n_lens - n_surr) / n_surr*(1/r1 - 1/r2 + (((n_lens - n_surr) * d) / (n_lens * r1 * r2))))

    if focal_length == zoo:
        return S.Infinity
    return focal_length


def mirror_formula(focal_length=None, u=None, v=None):
    """
    This function provides one of the three parameters
    when two of them are supplied.
    This is valid only for paraxial rays.

    Parameters
    ==========

    focal_length : sympifiable
        Focal length of the mirror.
    u : sympifiable
        Distance of object from the pole on
        the principal axis.
    v : sympifiable
        Distance of the image from the pole
        on the principal axis.

    Examples
    ========

    >>> from sympy.physics.optics import mirror_formula
    >>> from sympy.abc import f, u, v
    >>> mirror_formula(focal_length=f, u=u)
    f*u/(-f + u)
    >>> mirror_formula(focal_length=f, v=v)
    f*v/(-f + v)
    >>> mirror_formula(u=u, v=v)
    u*v/(u + v)

    """
    if focal_length and u and v:
        raise ValueError("Please provide only two parameters")

    focal_length = sympify(focal_length)
    u = sympify(u)
    v = sympify(v)
    if u is oo:
        _u = Symbol('u')
    if v is oo:
        _v = Symbol('v')
    if focal_length is oo:
        _f = Symbol('f')
    if focal_length is None:
        if u is oo and v is oo:
            return Limit(Limit(_v*_u/(_v + _u), _u, oo), _v, oo).doit()
        if u is oo:
            return Limit(v*_u/(v + _u), _u, oo).doit()
        if v is oo:
            return Limit(_v*u/(_v + u), _v, oo).doit()
        return v*u/(v + u)
    if u is None:
        if v is oo and focal_length is oo:
            return Limit(Limit(_v*_f/(_v - _f), _v, oo), _f, oo).doit()
        if v is oo:
            return Limit(_v*focal_length/(_v - focal_length), _v, oo).doit()
        if focal_length is oo:
            return Limit(v*_f/(v - _f), _f, oo).doit()
        return v*focal_length/(v - focal_length)
    if v is None:
        if u is oo and focal_length is oo:
            return Limit(Limit(_u*_f/(_u - _f), _u, oo), _f, oo).doit()
        if u is oo:
            return Limit(_u*focal_length/(_u - focal_length), _u, oo).doit()
        if focal_length is oo:
            return Limit(u*_f/(u - _f), _f, oo).doit()
        return u*focal_length/(u - focal_length)


def lens_formula(focal_length=None, u=None, v=None):
    """
    This function provides one of the three parameters
    when two of them are supplied.
    This is valid only for paraxial rays.

    Parameters
    ==========

    focal_length : sympifiable
        Focal length of the mirror.
    u : sympifiable
        Distance of object from the optical center on
        the principal axis.
    v : sympifiable
        Distance of the image from the optical center
        on the principal axis.

    Examples
    ========

    >>> from sympy.physics.optics import lens_formula
    >>> from sympy.abc import f, u, v
    >>> lens_formula(focal_length=f, u=u)
    f*u/(f + u)
    >>> lens_formula(focal_length=f, v=v)
    f*v/(f - v)
    >>> lens_formula(u=u, v=v)
    u*v/(u - v)

    """
    if focal_length and u and v:
        raise ValueError("Please provide only two parameters")

    focal_length = sympify(focal_length)
    u = sympify(u)
    v = sympify(v)
    if u is oo:
        _u = Symbol('u')
    if v is oo:
        _v = Symbol('v')
    if focal_length is oo:
        _f = Symbol('f')
    if focal_length is None:
        if u is oo and v is oo:
            return Limit(Limit(_v*_u/(_u - _v), _u, oo), _v, oo).doit()
        if u is oo:
            return Limit(v*_u/(_u - v), _u, oo).doit()
        if v is oo:
            return Limit(_v*u/(u - _v), _v, oo).doit()
        return v*u/(u - v)
    if u is None:
        if v is oo and focal_length is oo:
            return Limit(Limit(_v*_f/(_f - _v), _v, oo), _f, oo).doit()
        if v is oo:
            return Limit(_v*focal_length/(focal_length - _v), _v, oo).doit()
        if focal_length is oo:
            return Limit(v*_f/(_f - v), _f, oo).doit()
        return v*focal_length/(focal_length - v)
    if v is None:
        if u is oo and focal_length is oo:
            return Limit(Limit(_u*_f/(_u + _f), _u, oo), _f, oo).doit()
        if u is oo:
            return Limit(_u*focal_length/(_u + focal_length), _u, oo).doit()
        if focal_length is oo:
            return Limit(u*_f/(u + _f), _f, oo).doit()
        return u*focal_length/(u + focal_length)

def hyperfocal_distance(f, N, c):
    """

    Parameters
    ==========

    f: sympifiable
        Focal length of a given lens.

    N: sympifiable
