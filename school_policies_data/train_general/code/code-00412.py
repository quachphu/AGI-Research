    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of both the body and the
    flag relative to the floor are found:

    >>> tube.frame.dcm(floor.frame)
    Matrix([
    [ cos(q0_C1(t)), sin(q0_C1(t)), 0],
    [-sin(q0_C1(t)), cos(q0_C1(t)), 0],
    [             0,             0, 1]])
    >>> flag.frame.dcm(floor.frame)
    Matrix([
    [cos(q0_C1(t))*cos(q0_C2(t)), sin(q0_C1(t))*cos(q0_C2(t)), -sin(q0_C2(t))],
    [             -sin(q0_C1(t)),               cos(q0_C1(t)),              0],
    [sin(q0_C2(t))*cos(q0_C1(t)), sin(q0_C1(t))*sin(q0_C2(t)),  cos(q0_C2(t))]])

    The position of the flag's center of mass is found with:

    >>> flag.masscenter.pos_from(floor.masscenter)
    q1_C1(t)*floor_frame.z + (r + q1_C2(t))*tube_frame.y + w*flag_frame.y - l*flag_frame.z

    The angular velocities of the two tubes can be computed with respect to the
    floor.

    >>> tube.frame.ang_vel_in(floor.frame)
    u0_C1(t)*floor_frame.z
    >>> flag.frame.ang_vel_in(floor.frame)
    u0_C1(t)*floor_frame.z + u0_C2(t)*tube_frame.y

    Finally, the linear velocities of the two tube centers of mass can be
    computed with respect to the floor, while expressed in the tube's frame.

    >>> tube.masscenter.vel(floor.frame).to_matrix(tube.frame)
    Matrix([
    [       0],
    [       0],
    [u1_C1(t)]])
    >>> flag.masscenter.vel(floor.frame).to_matrix(tube.frame).simplify()
    Matrix([
    [-l*u0_C2(t)*cos(q0_C2(t)) - r*u0_C1(t) - w*u0_C1(t) - q1_C2(t)*u0_C1(t)],
    [                    -l*u0_C1(t)*sin(q0_C2(t)) + Derivative(q1_C2(t), t)],
    [                                    l*u0_C2(t)*sin(q0_C2(t)) + u1_C1(t)]])

    """

    def __init__(self, name, parent, child, rotation_coordinate=None,
                 translation_coordinate=None, rotation_speed=None,
                 translation_speed=None, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None,
                 joint_axis=None):
        self._joint_axis = joint_axis
        coordinates = (rotation_coordinate, translation_coordinate)
        speeds = (rotation_speed, translation_speed)
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'CylindricalJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis about and along which the rotation and translation occurs."""
        return self._joint_axis

    @property
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    def translation_coordinate(self):
        """Generalized coordinate corresponding to the translation distance."""
        return self.coordinates[1]

    @property
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    def translation_speed(self):
        """Generalized speed corresponding to the translation velocity."""
        return self.speeds[1]

    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 2, 'q')

    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, 2, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, self.rotation_coordinate)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(
            self.parent_interframe,
            self.rotation_speed * self.joint_axis.normalize())

    def _set_linear_velocity(self):
        self.child_point.set_pos(
            self.parent_point,
            self.translation_coordinate * self.joint_axis.normalize())
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child_point.set_vel(
            self._parent_frame,
            self.translation_speed * self.joint_axis.normalize())
        self.child.masscenter.v2pt_theory(self.child_point, self._parent_frame,
                                          self.child_interframe)


class PlanarJoint(Joint):
    """Planar Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/PlanarJoint.svg

    Explanation
    ===========

    A planar joint is defined such that the child body translates over a fixed
    plane of the parent body as well as rotate about the rotation axis, which
    is perpendicular to that plane. The origin of this plane is the
    ``parent_point`` and the plane is spanned by two nonparallel planar vectors.
    The location of the ``child_point`` is based on the planar vectors
    ($\\vec{v}_1$, $\\vec{v}_2$) and generalized coordinates ($q_1$, $q_2$),
    i.e. $\\vec{r} = q_1 \\hat{v}_1 + q_2 \\hat{v}_2$. The direction cosine
    matrix between the ``child_interframe`` and ``parent_interframe`` is formed
    using a simple rotation ($q_0$) about the rotation axis.

    In order to simplify the definition of the ``PlanarJoint``, the
    ``rotation_axis`` and ``planar_vectors`` are set to be the unit vectors of
    the ``parent_interframe`` according to the table below. This ensures that
    you can only define these vectors by creating a separate frame and supplying
    that as the interframe. If you however would only like to supply the normals
    of the plane with respect to the parent and child bodies, then you can also
    supply those to the ``parent_interframe`` and ``child_interframe``
    arguments. An example of both of these cases is in the examples section
    below and the page on the joints framework provides a more detailed
    explanation of the intermediate frames.

    .. list-table::

        * - ``rotation_axis``
          - ``parent_interframe.x``
        * - ``planar_vectors[0]``
          - ``parent_interframe.y``
        * - ``planar_vectors[1]``
          - ``parent_interframe.z``

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
    planar_coordinates : iterable of dynamicsymbols, optional
        Two generalized coordinates used for the planar translation. The default
        value is ``dynamicsymbols(f'q1_{joint.name} q2_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    planar_speeds : dynamicsymbols, optional
        Two generalized speeds used for the planar translation velocity. The
        default value is ``dynamicsymbols(f'u1_{joint.name} u2_{joint.name}')``.
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
    planar_coordinates : Matrix
        Two generalized coordinates used for the planar translation.
    rotation_speed : dynamicsymbol
        Generalized speed corresponding to the angular velocity.
    planar_speeds : Matrix
        Two generalized speeds used for the planar translation velocity.
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
    rotation_axis : Vector
        The axis about which the rotation occurs.
    planar_vectors : list
        The vectors that describe the planar translation directions.

    Examples
    =========

    A single planar joint is created between two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, PlanarJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PlanarJoint('PC', parent, child)
    >>> joint
    PlanarJoint: PC  parent: P  child: C
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
    >>> joint.rotation_axis
    P_frame.x
    >>> joint.planar_vectors
    [P_frame.y, P_frame.z]
    >>> joint.rotation_coordinate
    q0_PC(t)
    >>> joint.planar_coordinates
    Matrix([
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.rotation_speed
    u0_PC(t)
    >>> joint.planar_speeds
    Matrix([
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u0_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.y + q2_PC(t)*P_frame.z
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.y + u2_PC(t)*P_frame.z

    To further demonstrate the use of the planar joint, the kinematics of a
    block sliding on a slope, can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import PlanarJoint, RigidBody, ReferenceFrame
    >>> a, d, h = symbols('a d h')

    First create bodies to represent the slope and the block.

    >>> ground = RigidBody('G')
    >>> block = RigidBody('B')

    To define the slope you can either define the plane by specifying the
    ``planar_vectors`` or/and the ``rotation_axis``. However it is advisable to
    create a rotated intermediate frame, so that the ``parent_vectors`` and
    ``rotation_axis`` will be the unit vectors of this intermediate frame.

    >>> slope = ReferenceFrame('A')
    >>> slope.orient_axis(ground.frame, ground.y, a)

    The planar joint can be created using these bodies and intermediate frame.
    We can specify the origin of the slope to be ``d`` above the slope's center
    of mass and the block's center of mass to be a distance ``h`` above the
    slope's surface. Note that we can specify the normal of the plane using the
    rotation axis argument.

    >>> joint = PlanarJoint('PC', ground, block, parent_point=d * ground.x,
    ...                     child_point=-h * block.x, parent_interframe=slope)

    Once the joint is established the kinematics of the bodies can be accessed.
    First the ``rotation_axis``, which is normal to the plane and the
    ``plane_vectors``, can be found.

    >>> joint.rotation_axis
    A.x
    >>> joint.planar_vectors
    [A.y, A.z]

    The direction cosine matrix of the block with respect to the ground can be
    found with:

    >>> block.frame.dcm(ground.frame)
    Matrix([
    [              cos(a),              0,              -sin(a)],
    [sin(a)*sin(q0_PC(t)),  cos(q0_PC(t)), sin(q0_PC(t))*cos(a)],
    [sin(a)*cos(q0_PC(t)), -sin(q0_PC(t)), cos(a)*cos(q0_PC(t))]])

    The angular velocity of the block can be computed with respect to the
    ground.

    >>> block.frame.ang_vel_in(ground.frame)
    u0_PC(t)*A.x

    The position of the block's center of mass can be found with:

    >>> block.masscenter.pos_from(ground.masscenter)
    d*G_frame.x + h*B_frame.x + q1_PC(t)*A.y + q2_PC(t)*A.z

    Finally, the linear velocity of the block's center of mass can be
    computed with respect to the ground.

    >>> block.masscenter.vel(ground.frame)
    u1_PC(t)*A.y + u2_PC(t)*A.z

    In some cases it could be your preference to only define the normals of the
    plane with respect to both bodies. This can most easily be done by supplying
    vectors to the ``interframe`` arguments. What will happen in this case is
    that an interframe will be created with its ``x`` axis aligned with the
    provided vector. For a further explanation of how this is done see the notes
    of the ``Joint`` class. In the code below, the above example (with the block
    on the slope) is recreated by supplying vectors to the interframe arguments.
    Note that the previously described option is however more computationally
    efficient, because the algorithm now has to compute the rotation angle
    between the provided vector and the 'x' axis.

    >>> from sympy import symbols, cos, sin
    >>> from sympy.physics.mechanics import PlanarJoint, RigidBody
    >>> a, d, h = symbols('a d h')
    >>> ground = RigidBody('G')
    >>> block = RigidBody('B')
    >>> joint = PlanarJoint(
    ...     'PC', ground, block, parent_point=d * ground.x,
    ...     child_point=-h * block.x, child_interframe=block.x,
    ...     parent_interframe=cos(a) * ground.x + sin(a) * ground.z)
    >>> block.frame.dcm(ground.frame).simplify()
    Matrix([
    [               cos(a),              0,               sin(a)],
    [-sin(a)*sin(q0_PC(t)),  cos(q0_PC(t)), sin(q0_PC(t))*cos(a)],
    [-sin(a)*cos(q0_PC(t)), -sin(q0_PC(t)), cos(a)*cos(q0_PC(t))]])

    """

    def __init__(self, name, parent, child, rotation_coordinate=None,
                 planar_coordinates=None, rotation_speed=None,
                 planar_speeds=None, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None):
        # A ready to merge implementation of setting the planar_vectors and
        # rotation_axis was added and removed in PR #24046
        coordinates = (rotation_coordinate, planar_coordinates)
        speeds = (rotation_speed, planar_speeds)
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'PlanarJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    def planar_coordinates(self):
        """Two generalized coordinates used for the planar translation."""
        return self.coordinates[1:, 0]

    @property
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    def planar_speeds(self):
        """Two generalized speeds used for the planar translation velocity."""
        return self.speeds[1:, 0]

    @property
    def rotation_axis(self):
        """The axis about which the rotation occurs."""
        return self.parent_interframe.x

    @property
    def planar_vectors(self):
        """The vectors that describe the planar translation directions."""
        return [self.parent_interframe.y, self.parent_interframe.z]

    def _generate_coordinates(self, coordinates):
        rotation_speed = self._fill_coordinate_list(coordinates[0], 1, 'q',
                                                    number_single=True)
        planar_speeds = self._fill_coordinate_list(coordinates[1], 2, 'q', 1)
        return rotation_speed.col_join(planar_speeds)

    def _generate_speeds(self, speeds):
        rotation_speed = self._fill_coordinate_list(speeds[0], 1, 'u',
                                                    number_single=True)
        planar_speeds = self._fill_coordinate_list(speeds[1], 2, 'u', 1)
        return rotation_speed.col_join(planar_speeds)

    def _orient_frames(self):
        self.child_interframe.orient_axis(
            self.parent_interframe, self.rotation_axis,
            self.rotation_coordinate)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(
            self.parent_interframe,
            self.rotation_speed * self.rotation_axis)

    def _set_linear_velocity(self):
        self.child_point.set_pos(
            self.parent_point,
            self.planar_coordinates[0] * self.planar_vectors[0] +
            self.planar_coordinates[1] * self.planar_vectors[1])
        self.parent_point.set_vel(self.parent_interframe, 0)
        self.child_point.set_vel(self.child_interframe, 0)
        self.child_point.set_vel(
            self._parent_frame, self.planar_speeds[0] * self.planar_vectors[0] +
            self.planar_speeds[1] * self.planar_vectors[1])
        self.child.masscenter.v2pt_theory(self.child_point, self._parent_frame,
                                          self._child_frame)


class SphericalJoint(Joint):
    """Spherical (Ball-and-Socket) Joint.

    .. image:: SphericalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A spherical joint is defined such that the child body is free to rotate in
    any direction, without allowing a translation of the ``child_point``. As can
    also be seen in the image, the ``parent_point`` and ``child_point`` are
    fixed on top of each other, i.e. the ``joint_point``. This rotation is
    defined using the :func:`parent_interframe.orient(child_interframe,
    rot_type, amounts, rot_order)
    <sympy.physics.vector.frame.ReferenceFrame.orient>` method. The default
    rotation consists of three relative rotations, i.e. body-fixed rotations.
    Based on the direction cosine matrix following from these rotations, the
    angular velocity is computed based on the generalized coordinates and
    generalized speeds.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
    coordinates: iterable of dynamicsymbols, optional
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
    rot_type : str, optional
        The method used to generate the direction cosine matrix. Supported
        methods are:

        - ``'Body'``: three successive rotations about new intermediate axes,
          also called "Euler and Tait-Bryan angles"
        - ``'Space'``: three successive rotations about the parent frames' unit
          vectors

        The default method is ``'Body'``.
    amounts :
        Expressions defining the rotation angles or direction cosine matrix.
        These must match the ``rot_type``. See examples below for details. The
        input types are:

        - ``'Body'``: 3-tuple of expressions, symbols, or functions
        - ``'Space'``: 3-tuple of expressions, symbols, or functions

        The default amounts are the given ``coordinates``.
    rot_order : str or int, optional
        If applicable, the order of the successive of rotations. The string
        ``'123'`` and integer ``123`` are equivalent, for example. Required for
        ``'Body'`` and ``'Space'``. The default value is ``123``.

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

    A single spherical joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, SphericalJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = SphericalJoint('PC', parent, child)
    >>> joint
    SphericalJoint: PC  parent: P  child: C
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
    >>> joint.parent_interframe
    P_frame
    >>> joint.child_interframe
    C_frame
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame).to_matrix(child.frame)
    Matrix([
    [ u0_PC(t)*cos(q1_PC(t))*cos(q2_PC(t)) + u1_PC(t)*sin(q2_PC(t))],
    [-u0_PC(t)*sin(q2_PC(t))*cos(q1_PC(t)) + u1_PC(t)*cos(q2_PC(t))],
    [                             u0_PC(t)*sin(q1_PC(t)) + u2_PC(t)]])
    >>> child.frame.x.to_matrix(parent.frame)
    Matrix([
    [                                            cos(q1_PC(t))*cos(q2_PC(t))],
    [sin(q0_PC(t))*sin(q1_PC(t))*cos(q2_PC(t)) + sin(q2_PC(t))*cos(q0_PC(t))],
    [sin(q0_PC(t))*sin(q2_PC(t)) - sin(q1_PC(t))*cos(q0_PC(t))*cos(q2_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the spherical joint, the kinematics of a
    spherical joint with a ZXZ rotation can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import RigidBody, SphericalJoint
    >>> l1 = symbols('l1')

    First create bodies to represent the fixed floor and a pendulum bob.

    >>> floor = RigidBody('F')
    >>> bob = RigidBody('B')

    The joint will connect the bob to the floor, with the joint located at a
    distance of ``l1`` from the child's center of mass and the rotation set to a
    body-fixed ZXZ rotation.

    >>> joint = SphericalJoint('S', floor, bob, child_point=l1 * bob.y,
    ...                        rot_type='body', rot_order='ZXZ')

    Now that the joint is established, the kinematics of the connected body can
    be accessed.

    The position of the bob's masscenter is found with:

    >>> bob.masscenter.pos_from(floor.masscenter)
    - l1*B_frame.y

    The angular velocities of the pendulum link can be computed with respect to
    the floor.

    >>> bob.frame.ang_vel_in(floor.frame).to_matrix(
    ...     floor.frame).simplify()
    Matrix([
    [u1_S(t)*cos(q0_S(t)) + u2_S(t)*sin(q0_S(t))*sin(q1_S(t))],
    [u1_S(t)*sin(q0_S(t)) - u2_S(t)*sin(q1_S(t))*cos(q0_S(t))],
    [                          u0_S(t) + u2_S(t)*cos(q1_S(t))]])

    Finally, the linear velocity of the bob's center of mass can be computed.

    >>> bob.masscenter.vel(floor.frame).to_matrix(bob.frame)
    Matrix([
    [                           l1*(u0_S(t)*cos(q1_S(t)) + u2_S(t))],
    [                                                             0],
    [-l1*(u0_S(t)*sin(q1_S(t))*sin(q2_S(t)) + u1_S(t)*cos(q2_S(t)))]])

    """
    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, rot_type='BODY', amounts=None,
                 rot_order=123):
        self._rot_type = rot_type
        self._amounts = amounts
        self._rot_order = rot_order
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'SphericalJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 3, 'q')

    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, len(self.coordinates), 'u')

    def _orient_frames(self):
        supported_rot_types = ('BODY', 'SPACE')
        if self._rot_type.upper() not in supported_rot_types:
            raise NotImplementedError(
                f'Rotation type "{self._rot_type}" is not implemented. '
                f'Implemented rotation types are: {supported_rot_types}')
        amounts = self.coordinates if self._amounts is None else self._amounts
        self.child_interframe.orient(self.parent_interframe, self._rot_type,
                                     amounts, self._rot_order)

    def _set_angular_velocity(self):
        t = dynamicsymbols._t
        vel = self.child_interframe.ang_vel_in(self.parent_interframe).xreplace(
            {q.diff(t): u for q, u in zip(self.coordinates, self.speeds)}
        )
        self.child_interframe.set_ang_vel(self.parent_interframe, vel)

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.v2pt_theory(self.parent_point, self._parent_frame,
                                          self._child_frame)


class WeldJoint(Joint):
    """Weld Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/WeldJoint.svg

    Explanation
    ===========

    A weld joint is defined such that there is no relative motion between the
    child and parent bodies. The direction cosine matrix between the attachment
    frame (``parent_interframe`` and ``child_interframe``) is the identity
    matrix and the attachment points (``parent_point`` and ``child_point``) are
    coincident. The page on the joints framework gives a more detailed
    explanation of the intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody
        The parent body of joint.
    child : Particle or RigidBody
        The child body of joint.
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

    A single weld joint is created from two bodies and has the following basic
    attributes:

    >>> from sympy.physics.mechanics import RigidBody, WeldJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = WeldJoint('PC', parent, child)
    >>> joint
    WeldJoint: PC  parent: P  child: C
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
    >>> joint.coordinates
    Matrix(0, 0, [])
    >>> joint.speeds
    Matrix(0, 0, [])
    >>> child.frame.ang_vel_in(parent.frame)
    0
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the weld joint, two relatively-fixed
    bodies rotated by a quarter turn about the Y axis can be created as follows:

    >>> from sympy import symbols, pi
    >>> from sympy.physics.mechanics import ReferenceFrame, RigidBody, WeldJoint
    >>> l1, l2 = symbols('l1 l2')

    First create the bodies to represent the parent and rotated child body.

    >>> parent = RigidBody('P')
    >>> child = RigidBody('C')

    Next the intermediate frame specifying the fixed rotation with respect to
    the parent can be created.

    >>> rotated_frame = ReferenceFrame('Pr')
    >>> rotated_frame.orient_axis(parent.frame, parent.y, pi / 2)

    The weld between the parent body and child body is located at a distance
    ``l1`` from the parent's center of mass in the X direction and ``l2`` from
    the child's center of mass in the child's negative X direction.

    >>> weld = WeldJoint('weld', parent, child, parent_point=l1 * parent.x,
    ...                  child_point=-l2 * child.x,
    ...                  parent_interframe=rotated_frame)

    Now that the joint has been established, the kinematics of the bodies can be
    accessed. The direction cosine matrix of the child body with respect to the
    parent can be found:

    >>> child.frame.dcm(parent.frame)
    Matrix([
    [0, 0, -1],
    [0, 1,  0],
    [1, 0,  0]])

    As can also been seen from the direction cosine matrix, the parent X axis is
    aligned with the child's Z axis:
    >>> parent.x == child.z
    True

    The position of the child's center of mass with respect to the parent's
    center of mass can be found with:

    >>> child.masscenter.pos_from(parent.masscenter)
    l1*P_frame.x + l2*C_frame.x

    The angular velocity of the child with respect to the parent is 0 as one
    would expect.

    >>> child.frame.ang_vel_in(parent.frame)
    0

    """

    def __init__(self, name, parent, child, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None):
        super().__init__(name, parent, child, [], [], parent_point,
                         child_point, parent_interframe=parent_interframe,
                         child_interframe=child_interframe)
        self._kdes = Matrix(1, 0, []).T  # Removes stackability problems #10770

    def __str__(self):
        return (f'WeldJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    def _generate_coordinates(self, coordinate):
        return Matrix()

    def _generate_speeds(self, speed):
        return Matrix()

    def _orient_frames(self):
        self.child_interframe.orient_axis(self.parent_interframe,
                                          self.parent_interframe.x, 0)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, 0)

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.set_vel(self._parent_frame, 0)
<|endoftext|>__all__ = [
    'vector',

    'CoordinateSym', 'ReferenceFrame', 'Dyadic', 'Vector', 'Point', 'cross',
    'dot', 'express', 'time_derivative', 'outer', 'kinematic_equations',
    'get_motion_params', 'partial_velocity', 'dynamicsymbols', 'vprint',
    'vsstrrepr', 'vsprint', 'vpprint', 'vlatex', 'init_vprinting', 'curl',
    'divergence', 'gradient', 'is_conservative', 'is_solenoidal',
    'scalar_potential', 'scalar_potential_difference',

    'KanesMethod',

    'RigidBody',

    'linear_momentum', 'angular_momentum', 'kinetic_energy', 'potential_energy',
    'Lagrangian', 'mechanics_printing', 'mprint', 'msprint', 'mpprint',
    'mlatex', 'msubs', 'find_dynamicsymbols',

    'inertia', 'inertia_of_point_mass', 'Inertia',

    'Force', 'Torque',

    'Particle',

    'LagrangesMethod',

    'Linearizer',

    'Body',

    'SymbolicSystem', 'System',

    'PinJoint', 'PrismaticJoint', 'CylindricalJoint', 'PlanarJoint',
    'SphericalJoint', 'WeldJoint',

    'JointsMethod',

    'WrappingCylinder', 'WrappingGeometryBase', 'WrappingSphere',

    'PathwayBase', 'LinearPathway', 'ObstacleSetPathway', 'WrappingPathway',

    'ActuatorBase', 'ForceActuator', 'LinearDamper', 'LinearSpring',
    'TorqueActuator', 'DuffingSpring', 'CoulombKineticFriction',
]

from sympy.physics import vector

from sympy.physics.vector import (CoordinateSym, ReferenceFrame, Dyadic, Vector, Point,
        cross, dot, express, time_derivative, outer, kinematic_equations,
        get_motion_params, partial_velocity, dynamicsymbols, vprint,
        vsstrrepr, vsprint, vpprint, vlatex, init_vprinting, curl, divergence,
        gradient, is_conservative, is_solenoidal, scalar_potential,
        scalar_potential_difference)

from .kane import KanesMethod

from .rigidbody import RigidBody

from .functions import (linear_momentum, angular_momentum, kinetic_energy,
                        potential_energy, Lagrangian, mechanics_printing,
                        mprint, msprint, mpprint, mlatex, msubs,
                        find_dynamicsymbols)

from .inertia import inertia, inertia_of_point_mass, Inertia

from .loads import Force, Torque

from .particle import Particle

from .lagrange import LagrangesMethod

from .linearize import Linearizer

from .body import Body

from .system import SymbolicSystem, System

from .jointsmethod import JointsMethod

from .joint import (PinJoint, PrismaticJoint, CylindricalJoint, PlanarJoint,
                    SphericalJoint, WeldJoint)

from .wrapping_geometry import (WrappingCylinder, WrappingGeometryBase,
                                WrappingSphere)

from .pathway import (PathwayBase, LinearPathway, ObstacleSetPathway,
                      WrappingPathway)

from .actuator import (ActuatorBase, ForceActuator, LinearDamper, LinearSpring,
                       TorqueActuator, DuffingSpring, CoulombKineticFriction)
<|endoftext|>from abc import ABC
from collections import namedtuple
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.vector import Vector, ReferenceFrame, Point

__all__ = ['LoadBase', 'Force', 'Torque']


class LoadBase(ABC, namedtuple('LoadBase', ['location', 'vector'])):
    """Abstract base class for the various loading types."""

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


class Force(LoadBase):
    """Force acting upon a point.

    Explanation
    ===========

    A force is a vector that is bound to a line of action. This class stores
    both a point, which lies on the line of action, and the vector. A tuple can
    also be used, with the location as the first entry and the vector as second
    entry.

    Examples
    ========

    A force of magnitude 2 along N.x acting on a point Po can be created as
    follows:

    >>> from sympy.physics.mechanics import Point, ReferenceFrame, Force
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Force(Po, 2 * N.x)
    (Po, 2*N.x)

    If a body is supplied, then the center of mass of that body is used.

    >>> from sympy.physics.mechanics import Particle
    >>> P = Particle('P', point=Po)
    >>> Force(P, 2 * N.x)
    (Po, 2*N.x)

    """

    def __new__(cls, point, force):
        if isinstance(point, BodyBase):
            point = point.masscenter
        if not isinstance(point, Point):
            raise TypeError('Force location should be a Point.')
        if not isinstance(force, Vector):
            raise TypeError('Force vector should be a Vector.')
        return super().__new__(cls, point, force)

    def __repr__(self):
        return (f'{self.__class__.__name__}(point={self.point}, '
                f'force={self.force})')

    @property
    def point(self):
        return self.location

    @property
    def force(self):
        return self.vector


class Torque(LoadBase):
    """Torque acting upon a frame.

    Explanation
    ===========

    A torque is a free vector that is acting on a reference frame, which is
    associated with a rigid body. This class stores both the frame and the
    vector. A tuple can also be used, with the location as the first item and
    the vector as second item.

    Examples
    ========

    A torque of magnitude 2 about N.x acting on a frame N can be created as
    follows:

    >>> from sympy.physics.mechanics import ReferenceFrame, Torque
    >>> N = ReferenceFrame('N')
    >>> Torque(N, 2 * N.x)
    (N, 2*N.x)

    If a body is supplied, then the frame fixed to that body is used.

    >>> from sympy.physics.mechanics import RigidBody
    >>> rb = RigidBody('rb', frame=N)
    >>> Torque(rb, 2 * N.x)
    (N, 2*N.x)

    """

    def __new__(cls, frame, torque):
        if isinstance(frame, BodyBase):
            frame = frame.frame
        if not isinstance(frame, ReferenceFrame):
            raise TypeError('Torque location should be a ReferenceFrame.')
        if not isinstance(torque, Vector):
            raise TypeError('Torque vector should be a Vector.')
        return super().__new__(cls, frame, torque)

    def __repr__(self):
        return (f'{self.__class__.__name__}(frame={self.frame}, '
                f'torque={self.torque})')

    @property
    def frame(self):
        return self.location

    @property
    def torque(self):
        return self.vector


def gravity(acceleration, *bodies):
    """
    Returns a list of gravity forces given the acceleration
    due to gravity and any number of particles or rigidbodies.

    Example
    =======

    >>> from sympy.physics.mechanics import ReferenceFrame, Particle, RigidBody
    >>> from sympy.physics.mechanics.loads import gravity
    >>> from sympy import symbols
    >>> N = ReferenceFrame('N')
    >>> g = symbols('g')
    >>> P = Particle('P')
    >>> B = RigidBody('B')
    >>> gravity(g*N.y, P, B)
    [(P_masscenter, P_mass*g*N.y),
     (B_masscenter, B_mass*g*N.y)]

    """

    gravity_force = []
    for body in bodies:
        if not isinstance(body, BodyBase):
            raise TypeError(f'{type(body)} is not a body type')
        gravity_force.append(Force(body.masscenter, body.mass * acceleration))
    return gravity_force


def _parse_load(load):
    """Helper function to parse loads and convert tuples to load objects."""
    if isinstance(load, LoadBase):
        return load
    elif isinstance(load, tuple):
        if len(load) != 2:
            raise ValueError(f'Load {load} should have a length of 2.')
        if isinstance(load[0], Point):
            return Force(load[0], load[1])
        elif isinstance(load[0], ReferenceFrame):
            return Torque(load[0], load[1])
        else:
            raise ValueError(f'Load not recognized. The load location {load[0]}'
                             f' should either be a Point or a ReferenceFrame.')
    raise TypeError(f'Load type {type(load)} not recognized as a load. It '
                    f'should be a Force, Torque or tuple.')
<|endoftext|>"""Implementations of actuators for linked force and torque application."""

from abc import ABC, abstractmethod

from sympy import S, sympify, exp, sign
from sympy.physics.mechanics.joint import PinJoint
from sympy.physics.mechanics.loads import Torque
from sympy.physics.mechanics.pathway import PathwayBase
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.vector import ReferenceFrame, Vector


__all__ = [
    'ActuatorBase',
    'ForceActuator',
    'LinearDamper',
    'LinearSpring',
    'TorqueActuator',
    'DuffingSpring',
    'CoulombKineticFriction',
]


class ActuatorBase(ABC):
    """Abstract base class for all actuator classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom actuator types through subclassing.

    """

    def __init__(self):
        """Initializer for ``ActuatorBase``."""
        pass

    @abstractmethod
    def to_loads(self):
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
        """Default representation of an actuator."""
        return f'{self.__class__.__name__}()'


class ForceActuator(ActuatorBase):
    """Force-producing actuator.

    Explanation
    ===========

    A ``ForceActuator`` is an actuator that produces a (expansile) force along
    its length.

    A force actuator uses a pathway instance to determine the direction and
    number of forces that it applies to a system. Consider the simplest case
    where a ``LinearPathway`` instance is used. This pathway is made up of two
    points that can move relative to each other, and results in a pair of equal
    and opposite forces acting on the endpoints. If the positive time-varying
    Euclidean distance between the two points is defined, then the "extension
    velocity" is the time derivative of this distance. The extension velocity
    is positive when the two points are moving away from each other and
    negative when moving closer to each other. The direction for the force
    acting on either point is determined by constructing a unit vector directed
    from the other point to this point. This establishes a sign convention such
    that a positive force magnitude tends to push the points apart, this is the
    meaning of "expansile" in this context. The following diagram shows the
    positive force sense and the distance between the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct an actuator, an expression (or symbol) must be supplied to
    represent the force it can produce, alongside a pathway specifying its line
    of action. Let's also create a global reference frame and spatially fix one
    of the points in it while setting the other to be positioned such that it
    can freely move in the frame's x direction specified by the coordinate
    ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ForceActuator, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> force = symbols('F')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> actuator = ForceActuator(force, linear_pathway)
    >>> actuator
    ForceActuator(F, LinearPathway(pA, pB))

    Parameters
    ==========

    force : Expr
        The scalar expression defining the (expansile) force that the actuator
        produces.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

    """

    def __init__(self, force, pathway):
        """Initializer for ``ForceActuator``.

        Parameters
        ==========

        force : Expr
            The scalar expression defining the (expansile) force that the
            actuator produces.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

        """
        self.force = force
        self.pathway = pathway

    @property
    def force(self):
        """The magnitude of the force produced by the actuator."""
        return self._force

    @force.setter
    def force(self, force):
        if hasattr(self, '_force'):
            msg = (
                f'Can\'t set attribute `force` to {repr(force)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._force = sympify(force, strict=True)

    @property
    def pathway(self):
        """The ``Pathway`` defining the actuator's line of action."""
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        if hasattr(self, '_pathway'):
            msg = (
                f'Can\'t set attribute `pathway` to {repr(pathway)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(pathway, PathwayBase):
            msg = (
                f'Value {repr(pathway)} passed to `pathway` was of type '
                f'{type(pathway)}, must be {PathwayBase}.'
            )
            raise TypeError(msg)
        self._pathway = pathway

    def to_loads(self):
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

        The below example shows how to generate the loads produced by a force
        actuator that follows a linear pathway. In this example we'll assume
        that the force actuator is being used to model a simple linear spring.
        First, create a linear pathway between two points separated by the
        coordinate ``q`` in the ``x`` direction of the global frame ``N``.

        >>> from sympy.physics.mechanics import (LinearPathway, Point,
        ...     ReferenceFrame)
        >>> from sympy.physics.vector import dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> N = ReferenceFrame('N')
        >>> pA, pB = Point('pA'), Point('pB')
        >>> pB.set_pos(pA, q*N.x)
        >>> pathway = LinearPathway(pA, pB)

        Now create a symbol ``k`` to describe the spring's stiffness and
        instantiate a force actuator that produces a (contractile) force
        proportional to both the spring's stiffness and the pathway's length.
        Note that actuator classes use the sign convention that expansile
        forces are positive, so for a spring to produce a contractile force the
        spring force needs to be calculated as the negative for the stiffness
        multiplied by the length.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import ForceActuator
        >>> stiffness = symbols('k')
        >>> spring_force = -stiffness*pathway.length
        >>> spring = ForceActuator(spring_force, pathway)

        The forces produced by the spring can be generated in the list of loads
        form that ``KanesMethod`` (and other equations of motion methods)
        requires by calling the ``to_loads`` method.

        >>> spring.to_loads()
        [(pA, k*q(t)*N.x), (pB, - k*q(t)*N.x)]

        A simple linear damper can be modeled in a similar way. Create another
        symbol ``c`` to describe the dampers damping coefficient. This time
        instantiate a force actuator that produces a force proportional to both
        the damper's damping coefficient and the pathway's extension velocity.
        Note that the damping force is negative as it acts in the opposite
        direction to which the damper is changing in length.

        >>> damping_coefficient = symbols('c')
        >>> damping_force = -damping_coefficient*pathway.extension_velocity
        >>> damper = ForceActuator(damping_force, pathway)

        Again, the forces produces by the damper can be generated by calling
        the ``to_loads`` method.

        >>> damper.to_loads()
        [(pA, c*Derivative(q(t), t)*N.x), (pB, - c*Derivative(q(t), t)*N.x)]

        """
        return self.pathway.to_loads(self.force)

    def __repr__(self):
        """Representation of a ``ForceActuator``."""
        return f'{self.__class__.__name__}({self.force}, {self.pathway})'


class LinearSpring(ForceActuator):
    """A spring with its spring force as a linear function of its length.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearSpring`` refers to the fact that
    the spring force is a linear function of the springs length. I.e. for a
    linear spring with stiffness ``k``, distance between its ends of ``x``, and
    an equilibrium length of ``0``, the spring force will be ``-k*x``, which is
    a linear function in ``x``. To create a spring that follows a linear, or
    straight, pathway between its two ends, a ``LinearPathway`` instance needs
    to be passed to the ``pathway`` parameter.

    A ``LinearSpring`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear spring is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the spring away from one another.
    Because springs produces a contractile force and acts to pull the two ends
    together towards the equilibrium length when stretched, the scalar portion
    of the forces on the endpoint are negative in order to flip the sign of the
    forces on the endpoints when converted into vector quantities. The
    following diagram shows the positive force sense and the distance between
    the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear spring, an expression (or symbol) must be supplied to
    represent the stiffness (spring constant) of the spring, alongside a
    pathway specifying its line of action. Let's also create a global reference
    frame and spatially fix one of the points in it while setting the other to
    be positioned such that it can freely move in the frame's x direction
    specified by the coordinate ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, LinearSpring,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> stiffness = symbols('k')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> spring = LinearSpring(stiffness, linear_pathway)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB))

    This spring will produce a force that is proportional to both its stiffness
    and the pathway's length. Note that this force is negative as SymPy's sign
    convention for actuators is that negative forces are contractile.

    >>> spring.force
    -k*sqrt(q(t)**2)

    To create a linear spring with a non-zero equilibrium length, an expression
    (or symbol) can be passed to the ``equilibrium_length`` parameter on
    construction on a ``LinearSpring`` instance. Let's create a symbol ``l``
    to denote a non-zero equilibrium length and create another linear spring.

    >>> l = symbols('l')
    >>> spring = LinearSpring(stiffness, linear_pathway, equilibrium_length=l)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB), equilibrium_length=l)

    The spring force of this new spring is again proportional to both its
    stiffness and the pathway's length. However, the spring will not produce
    any force when ``q(t)`` equals ``l``. Note that the force will become
    expansile when ``q(t)`` is less than ``l``, as expected.

    >>> spring.force
    -k*(-l + sqrt(q(t)**2))

    Parameters
    ==========

    stiffness : Expr
        The spring constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium, i.e. it produces no
        force. The default value is 0, i.e. the spring force is a linear
        function of the pathway's length with no constant offset.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearSpring``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    def __init__(self, stiffness, pathway, equilibrium_length=S.Zero):
        """Initializer for ``LinearSpring``.

        Parameters
        ==========

        stiffness : Expr
            The spring constant.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
        equilibrium_length : Expr, optional
            The length at which the spring is in equilibrium, i.e. it produces
            no force. The default value is 0, i.e. the spring force is a linear
            function of the pathway's length with no constant offset.

        """
        self.stiffness = stiffness
        self.pathway = pathway
        self.equilibrium_length = equilibrium_length

    @property
    def force(self):
        """The spring force produced by the linear spring."""
        return -self.stiffness*(self.pathway.length - self.equilibrium_length)

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    @property
    def stiffness(self):
        """The spring constant for the linear spring."""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, stiffness):
        if hasattr(self, '_stiffness'):
            msg = (
                f'Can\'t set attribute `stiffness` to {repr(stiffness)} as it '
                f'is immutable.'
            )
            raise AttributeError(msg)
        self._stiffness = sympify(stiffness, strict=True)

    @property
    def equilibrium_length(self):
        """The length of the spring at which it produces no force."""
        return self._equilibrium_length

    @equilibrium_length.setter
    def equilibrium_length(self, equilibrium_length):
        if hasattr(self, '_equilibrium_length'):
            msg = (
                f'Can\'t set attribute `equilibrium_length` to '
                f'{repr(equilibrium_length)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    def __repr__(self):
        """Representation of a ``LinearSpring``."""
        string = f'{self.__class__.__name__}({self.stiffness}, {self.pathway}'
        if self.equilibrium_length == S.Zero:
            string += ')'
        else:
            string += f', equilibrium_length={self.equilibrium_length})'
        return string


class LinearDamper(ForceActuator):
    """A damper whose force is a linear function of its extension velocity.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearDamper`` refers to the fact that
    the damping force is a linear function of the damper's rate of change in
    its length. I.e. for a linear damper with damping ``c`` and extension
    velocity ``v``, the damping force will be ``-c*v``, which is a linear
    function in ``v``. To create a damper that follows a linear, or straight,
    pathway between its two ends, a ``LinearPathway`` instance needs to be
    passed to the ``pathway`` parameter.

    A ``LinearDamper`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear damper is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the damper away from one another.
    Because dampers produce a force that opposes the direction of change in
    length, when extension velocity is positive the scalar portions of the
    forces applied at the two endpoints are negative in order to flip the sign
    of the forces on the endpoints wen converted into vector quantities. When
    extension velocity is negative (i.e. when the damper is shortening), the
    scalar portions of the fofces applied are also negative so that the signs
    cancel producing forces on the endpoints that are in the same direction as
    the positive sign convention for the forces at the endpoints of the pathway
    (i.e. they act to push the endpoints away from one another). The following
    diagram shows the positive force sense and the distance between the
    points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear damper, an expression (or symbol) must be supplied to
    represent the damping coefficient of the damper (we'll use the symbol
    ``c``), alongside a pathway specifying its line of action. Let's also
    create a global reference frame and spatially fix one of the points in it
    while setting the other to be positioned such that it can freely move in
    the frame's x direction specified by the coordinate ``q``. The velocity
    that the two points move away from one another can be specified by the
    coordinate ``u`` where ``u`` is the first time derivative of ``q``
    (i.e., ``u = Derivative(q(t), t)``).

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearDamper, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> damping = symbols('c')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> pB.vel(N)
    Derivative(q(t), t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> damper = LinearDamper(damping, linear_pathway)
    >>> damper
    LinearDamper(c, LinearPathway(pA, pB))

    This damper will produce a force that is proportional to both its damping
    coefficient and the pathway's extension length. Note that this force is
    negative as SymPy's sign convention for actuators is that negative forces
    are contractile and the damping force of the damper will oppose the
    direction of length change.

    >>> damper.force
    -c*sqrt(q(t)**2)*Derivative(q(t), t)/q(t)

    Parameters
    ==========

    damping : Expr
        The damping constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearDamper``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    def __init__(self, damping, pathway):
        """Initializer for ``LinearDamper``.

        Parameters
        ==========

        damping : Expr
            The damping constant.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

        """
        self.damping = damping
        self.pathway = pathway

    @property
    def force(self):
        """The damping force produced by the linear damper."""
        return -self.damping*self.pathway.extension_velocity

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    @property
    def damping(self):
        """The damping constant for the linear damper."""
        return self._damping

    @damping.setter
    def damping(self, damping):
        if hasattr(self, '_damping'):
            msg = (
                f'Can\'t set attribute `damping` to {repr(damping)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._damping = sympify(damping, strict=True)

    def __repr__(self):
        """Representation of a ``LinearDamper``."""
        return f'{self.__class__.__name__}({self.damping}, {self.pathway})'


class TorqueActuator(ActuatorBase):
    """Torque-producing actuator.

    Explanation
    ===========

    A ``TorqueActuator`` is an actuator that produces a pair of equal and
    opposite torques on a pair of bodies.

    Examples
    ========

    To construct a torque actuator, an expression (or symbol) must be supplied
    to represent the torque it can produce, alongside a vector specifying the
    axis about which the torque will act, and a pair of frames on which the
    torque will act.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ReferenceFrame, RigidBody,
    ...     TorqueActuator)
    >>> N = ReferenceFrame('N')
    >>> A = ReferenceFrame('A')
    >>> torque = symbols('T')
    >>> axis = N.z
    >>> parent = RigidBody('parent', frame=N)
    >>> child = RigidBody('child', frame=A)
    >>> bodies = (child, parent)
    >>> actuator = TorqueActuator(torque, axis, *bodies)
    >>> actuator
    TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

    Note that because torques actually act on frames, not bodies,
    ``TorqueActuator`` will extract the frame associated with a ``RigidBody``
    when one is passed instead of a ``ReferenceFrame``.

    Parameters
    ==========

    torque : Expr
        The scalar expression defining the torque that the actuator produces.
    axis : Vector
        The axis about which the actuator applies torques.
    target_frame : ReferenceFrame | RigidBody
        The primary frame on which the actuator will apply the torque.
    reaction_frame : ReferenceFrame | RigidBody | None
        The secondary frame on which the actuator will apply the torque. Note
        that the (equal and opposite) reaction torque is applied to this frame.

    """

    def __init__(self, torque, axis, target_frame, reaction_frame=None):
        """Initializer for ``TorqueActuator``.

        Parameters
        ==========

        torque : Expr
            The scalar expression defining the torque that the actuator
            produces.
        axis : Vector
            The axis about which the actuator applies torques.
        target_frame : ReferenceFrame | RigidBody
            The primary frame on which the actuator will apply the torque.
        reaction_frame : ReferenceFrame | RigidBody | None
           The secondary frame on which the actuator will apply the torque.
           Note that the (equal and opposite) reaction torque is applied to
           this frame.

        """
        self.torque = torque
        self.axis = axis
        self.target_frame = target_frame
        self.reaction_frame = reaction_frame

    @classmethod
    def at_pin_joint(cls, torque, pin_joint):
        """Alternate constructor to instantiate from a ``PinJoint`` instance.

        Examples
        ========

        To create a pin joint the ``PinJoint`` class requires a name, parent
        body, and child body to be passed to its constructor. It is also
        possible to control the joint axis using the ``joint_axis`` keyword
        argument. In this example let's use the parent body's reference frame's
        z-axis as the joint axis.

        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,
        ...     RigidBody, TorqueActuator)
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> parent = RigidBody('parent', frame=N)
        >>> child = RigidBody('child', frame=A)
        >>> pin_joint = PinJoint(
        ...     'pin',
        ...     parent,
        ...     child,
        ...     joint_axis=N.z,
        ... )

        Let's also create a symbol ``T`` that will represent the torque applied
        by the torque actuator.

        >>> from sympy import symbols
        >>> torque = symbols('T')

        To create the torque actuator from the ``torque`` and ``pin_joint``
        variables previously instantiated, these can be passed to the alternate
        constructor class method ``at_pin_joint`` of the ``TorqueActuator``
        class. It should be noted that a positive torque will cause a positive
        displacement of the joint coordinate or that the torque is applied on
        the child body with a reaction torque on the parent.

        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)
        >>> actuator
        TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

        Parameters
        ==========

        torque : Expr
            The scalar expression defining the torque that the actuator
            produces.
        pin_joint : PinJoint
            The pin joint, and by association the parent and child bodies, on
            which the torque actuator will act. The pair of bodies acted upon
            by the torque actuator are the parent and child bodies of the pin
            joint, with the child acting as the reaction body. The pin joint's
            axis is used as the axis about which the torque actuator will apply
            its torque.

        """
        if not isinstance(pin_joint, PinJoint):
            msg = (
                f'Value {repr(pin_joint)} passed to `pin_joint` was of type '
                f'{type(pin_joint)}, must be {PinJoint}.'
            )
            raise TypeError(msg)
        return cls(
            torque,
            pin_joint.joint_axis,
            pin_joint.child_interframe,
            pin_joint.parent_interframe,
        )

    @property
    def torque(self):
        """The magnitude of the torque produced by the actuator."""
        return self._torque

    @torque.setter
    def torque(self, torque):
        if hasattr(self, '_torque'):
            msg = (
                f'Can\'t set attribute `torque` to {repr(torque)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._torque = sympify(torque, strict=True)

    @property
    def axis(self):
        """The axis about which the torque acts."""
        return self._axis

    @axis.setter
    def axis(self, axis):
        if hasattr(self, '_axis'):
            msg = (
                f'Can\'t set attribute `axis` to {repr(axis)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(axis, Vector):
            msg = (
                f'Value {repr(axis)} passed to `axis` was of type '
                f'{type(axis)}, must be {Vector}.'
            )
            raise TypeError(msg)
        self._axis = axis

    @property
    def target_frame(self):
        """The primary reference frames on which the torque will act."""
        return self._target_frame

    @target_frame.setter
    def target_frame(self, target_frame):
        if hasattr(self, '_target_frame'):
            msg = (
                f'Can\'t set attribute `target_frame` to {repr(target_frame)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if isinstance(target_frame, RigidBody):
            target_frame = target_frame.frame
        elif not isinstance(target_frame, ReferenceFrame):
            msg = (
                f'Value {repr(target_frame)} passed to `target_frame` was of '
                f'type {type(target_frame)}, must be {ReferenceFrame}.'
            )
            raise TypeError(msg)
        self._target_frame = target_frame

    @property
    def reaction_frame(self):
        """The primary reference frames on which the torque will act."""
        return self._reaction_frame

    @reaction_frame.setter
    def reaction_frame(self, reaction_frame):
        if hasattr(self, '_reaction_frame'):
            msg = (
                f'Can\'t set attribute `reaction_frame` to '
                f'{repr(reaction_frame)} as it is immutable.'
            )
            raise AttributeError(msg)
        if isinstance(reaction_frame, RigidBody):
            reaction_frame = reaction_frame.frame
        elif (
            not isinstance(reaction_frame, ReferenceFrame)
            and reaction_frame is not None
        ):
            msg = (
                f'Value {repr(reaction_frame)} passed to `reaction_frame` was '
                f'of type {type(reaction_frame)}, must be {ReferenceFrame}.'
            )
            raise TypeError(msg)
        self._reaction_frame = reaction_frame

    def to_loads(self):
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

        The below example shows how to generate the loads produced by a torque
        actuator that acts on a pair of bodies attached by a pin joint.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,
        ...     RigidBody, TorqueActuator)
        >>> torque = symbols('T')
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> parent = RigidBody('parent', frame=N)
        >>> child = RigidBody('child', frame=A)
        >>> pin_joint = PinJoint(
        ...     'pin',
        ...     parent,
        ...     child,
        ...     joint_axis=N.z,
        ... )
        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)

        The forces produces by the damper can be generated by calling the
        ``to_loads`` method.

        >>> actuator.to_loads()
        [(A, T*N.z), (N, - T*N.z)]

        Alternatively, if a torque actuator is created without a reaction frame
        then the loads returned by the ``to_loads`` method will contain just
        the single load acting on the target frame.

        >>> actuator = TorqueActuator(torque, N.z, N)
        >>> actuator.to_loads()
        [(N, T*N.z)]

        """
        loads = [
            Torque(self.target_frame, self.torque*self.axis),
        ]
        if self.reaction_frame is not None:
            loads.append(Torque(self.reaction_frame, -self.torque*self.axis))
        return loads

    def __repr__(self):
        """Representation of a ``TorqueActuator``."""
        string = (
            f'{self.__class__.__name__}({self.torque}, axis={self.axis}, '
            f'target_frame={self.target_frame}'
        )
        if self.reaction_frame is not None:
            string += f', reaction_frame={self.reaction_frame})'
        else:
            string += ')'
        return string


class DuffingSpring(ForceActuator):
    """A nonlinear spring based on the Duffing equation.

    Explanation
    ===========

    Here, ``DuffingSpring`` represents the force exerted by a nonlinear spring based on the Duffing equation:
    F = -beta*x-alpha*x**3, where x is the displacement from the equilibrium position, beta is the linear spring constant,
    and alpha is the coefficient for the nonlinear cubic term.

    Parameters
    ==========

    linear_stiffness : Expr
        The linear stiffness coefficient (beta).
    nonlinear_stiffness : Expr
        The nonlinear stiffness coefficient (alpha).
    pathway : PathwayBase
        The pathway that the actuator follows.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium (x).
    """

    def __init__(self, linear_stiffness, nonlinear_stiffness, pathway, equilibrium_length=S.Zero):
        self.linear_stiffness = sympify(linear_stiffness, strict=True)
        self.nonlinear_stiffness = sympify(nonlinear_stiffness, strict=True)
        self.equilibrium_length = sympify(equilibrium_length, strict=True)

        if not isinstance(pathway, PathwayBase):
            raise TypeError("pathway must be an instance of PathwayBase.")
        self._pathway = pathway

    @property
    def linear_stiffness(self):
        return self._linear_stiffness

    @linear_stiffness.setter
    def linear_stiffness(self, linear_stiffness):
        if hasattr(self, '_linear_stiffness'):
            msg = (
                f'Can\'t set attribute `linear_stiffness` to '
                f'{repr(linear_stiffness)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._linear_stiffness = sympify(linear_stiffness, strict=True)

    @property
    def nonlinear_stiffness(self):
        return self._nonlinear_stiffness

    @nonlinear_stiffness.setter
    def nonlinear_stiffness(self, nonlinear_stiffness):
        if hasattr(self, '_nonlinear_stiffness'):
            msg = (
                f'Can\'t set attribute `nonlinear_stiffness` to '
                f'{repr(nonlinear_stiffness)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._nonlinear_stiffness = sympify(nonlinear_stiffness, strict=True)

    @property
    def pathway(self):
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        if hasattr(self, '_pathway'):
            msg = (
                f'Can\'t set attribute `pathway` to {repr(pathway)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(pathway, PathwayBase):
            msg = (
                f'Value {repr(pathway)} passed to `pathway` was of type '
                f'{type(pathway)}, must be {PathwayBase}.'
            )
            raise TypeError(msg)
        self._pathway = pathway

    @property
    def equilibrium_length(self):
        return self._equilibrium_length

    @equilibrium_length.setter
    def equilibrium_length(self, equilibrium_length):
        if hasattr(self, '_equilibrium_length'):
            msg = (
                f'Can\'t set attribute `equilibrium_length` to '
                f'{repr(equilibrium_length)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    @property
    def force(self):
        """The force produced by the Duffing spring."""
        displacement = self.pathway.length - self.equilibrium_length
        return -self.linear_stiffness * displacement - self.nonlinear_stiffness * displacement**3

    @force.setter
    def force(self, force):
        if hasattr(self, '_force'):
            msg = (
                f'Can\'t set attribute `force` to {repr(force)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._force = sympify(force, strict=True)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self.linear_stiffness}, {self.nonlinear_stiffness}, {self.pathway}, "
                f"equilibrium_length={self.equilibrium_length})")

class CoulombKineticFriction(ForceActuator):
    r"""Coulomb kinetic friction with Stribeck and viscous effects.

    Explanation
    ===========

    This represents a Coulomb kinetic friction with the Stribeck and viscous effect,
    described by the function:

    .. math::
        F = (\mu_k f_n + (\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2}) \text{sign}(v) + \sigma  v

    where :math:`\mu_k` is the coefficient of kinetic friction, :math:`\mu_s` is the
    coefficient of static friction, :math:`f_n` is the normal force, :math:`v` is the
    relative velocity, :math:`v_s` is the Stribeck friction coefficient, and
    :math:`\sigma` is the viscous friction constant.

    The default friction force is :math:`F = \mu_k f_n`.
    When specified, the actuator includes:

    - Stribeck effect: :math:`(\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2}`
    - Viscous effect: :math:`\sigma v`

    Notes
    =====

    The actuator makes the following assumptions:

    - The actuator assumes relative motion is non-zero.
    - The normal force is assumed to be a non-negative scalar.
    - The resultant friction force is opposite to the velocity direction.
    - Each point in the pathway is fixed within separate objects that are sliding relative to each other. In other words, these two points are fixed in the mutually sliding objects.

    This actuator has been tested for straightforward motions, like a block sliding
    on a surface.

    The friction force is defined to always oppose the direction of relative velocity :math:`v`.
    Specifically:

    - The default Coulomb friction force :math:`\mu_k f_n \text{sign}(v)` is opposite to :math:`v`.
    - The Stribeck effect :math:`(\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2} \text{sign}(v)` is also opposite to :math:`v`.
    - The viscous friction term :math:`\sigma v` is opposite to :math:`v`.

    Examples
    ========

    The below example shows how to generate the loads produced by a Coulomb kinetic
    friction actuator in a mass-spring system with friction.

    >>> import sympy as sm
    >>> from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
    ...     LinearPathway, CoulombKineticFriction, LinearSpring, KanesMethod, Particle)

    >>> x, v = dynamicsymbols('x, v', real=True)
    >>> m, g, k, mu_k, mu_s, v_s, sigma = sm.symbols('m, g, k, mu_k, mu_s, v_s, sigma')

    >>> N = ReferenceFrame('N')
    >>> O, P = Point('O'), Point('P')
    >>> O.set_vel(N, 0)
    >>> P.set_pos(O, x*N.x)

    >>> pathway = LinearPathway(O, P)
    >>> friction = CoulombKineticFriction(mu_k, m*g, pathway, v_s=v_s, sigma=sigma, mu_s=mu_k)
    >>> spring = LinearSpring(k, pathway)
    >>> block = Particle('block', point=P, mass=m)

    >>> kane = KanesMethod(N, (x,), (v,), kd_eqs=(x.diff() - v,))
    >>> friction.to_loads()
        [(O, (g*m*mu_k*sign(sign(x(t))*Derivative(x(t), t)) + sigma*sign(x(t))*Derivative(x(t), t))*x(t)/Abs(x(t))*N.x), (P, (-g*m*mu_k*sign(sign(x(t))*Derivative(x(t), t)) - sigma*sign(x(t))*Derivative(x(t), t))*x(t)/Abs(x(t))*N.x)]
    >>> loads = friction.to_loads() + spring.to_loads()
    >>> fr, frstar = kane.kanes_equations([block], loads)
    >>> eom = fr + frstar
    >>> eom
        Matrix([[-k*x(t) - m*Derivative(v(t), t) + (-g*m*mu_k*sign(v(t)*sign(x(t))) - sigma*v(t)*sign(x(t)))*x(t)/Abs(x(t))]])

    Parameters
    ==========

    f_n : sympifiable
        The normal force between the surfaces. It should always be a non-negative scalar.
    mu_k : sympifiable
        The coefficient of kinetic friction.
    pathway : PathwayBase
        The pathway that the actuator follows.
    v_s : sympifiable, optional
        The Stribeck friction coefficient.
    sigma : sympifiable, optional
        The viscous friction coefficient.
    mu_s : sympifiable, optional
        The coefficient of static friction. Defaults to mu_k, meaning the Stribeck effect evaluates to 0 by default.

    References
    ==========

    .. [Moore2022] https://moorepants.github.io/learn-multibody-dynamics/loads.html#friction.
    .. [Flores2023] Paulo Flores, Jorge Ambrosio, Hamid M. Lankarani,
            "Contact-impact events with friction in multibody dynamics: Back to basics",
            Mechanism and Machine Theory, vol. 184, 2023. https://doi.org/10.1016/j.mechmachtheory.2023.105305.
    .. [Rogner2017] I. Rogner, "Friction modelling for robotic applications with planar motion",
            Chalmers University of Technology, Department of Electrical Engineering, 2017.

    """

    def __init__(self, mu_k, f_n, pathway, *, v_s=None, sigma=None, mu_s=None):
        self._mu_k = sympify(mu_k, strict=True) if mu_k is not None else 1
        self._mu_s = sympify(mu_s, strict=True) if mu_s is not None else self._mu_k
        self._f_n = sympify(f_n, strict=True)
        self._sigma = sympify(sigma, strict=True) if sigma is not None else 0
        self._v_s = sympify(v_s, strict=True) if v_s is not None or v_s == 0 else 0.01
        self.pathway = pathway

    @property
    def mu_k(self):
        """The coefficient of kinetic friction."""
        return self._mu_k

    @property
    def mu_s(self):
        """The coefficient of static friction."""
        return self._mu_s

    @property
    def f_n(self):
        """The normal force between the surfaces."""
        return self._f_n

    @property
    def sigma(self):
        """The viscous friction coefficient."""
        return self._sigma

    @property
    def v_s(self):
        """The Stribeck friction coefficient."""
        return self._v_s

    @property
    def force(self):
        v = self.pathway.extension_velocity
        f_c = self.mu_k * self.f_n
        f_max = self.mu_s * self.f_n
        stribeck_term = (f_max - f_c) * exp(-(v / self.v_s)**2) if self.v_s is not None else 0
        viscous_term = self.sigma * v if self.sigma is not None else 0
        return (f_c + stribeck_term) * -sign(v) - viscous_term

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    def __repr__(self):
        return (f'{self.__class__.__name__}({self._mu_k}, {self._mu_s} '
                f'{self._f_n}, {self.pathway}, {self._v_s}, '
                f'{self._sigma})')
<|endoftext|>from sympy import zeros, Matrix, diff, eye, linear_eq_to_matrix
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
                                  partial_velocity)
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (msubs, find_dynamicsymbols,
                                               _f_list_parser,
                                               _validate_coordinates,
                                               _parse_linear_solver)
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable


__all__ = ['KanesMethod']


class KanesMethod(_Methods):
    r"""Kane's method object.

    Explanation
    ===========

    This object is used to do the "book-keeping" as you go through and form
    equations of motion in the way Kane presents in:
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    The attributes are for equations in the form [M] udot = forcing.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Particle and RigidBody objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    auxiliary_eqs : Matrix
        If applicable, the set of auxiliary Kane's
        equations used to solve for non-contributing
        forces.
    mass_matrix : Matrix
        The system's dynamics mass matrix: [k_d; k_dnh]
    forcing : Matrix
        The system's dynamics forcing vector: -[f_d; f_dnh]
    mass_matrix_kin : Matrix
        The "mass matrix" for kinematic differential equations: k_kqdot
    forcing_kin : Matrix
        The forcing vector for kinematic differential equations: -(k_ku*u + f_k)
    mass_matrix_full : Matrix
        The "mass matrix" for the u's and q's with dynamics and kinematics
    forcing_full : Matrix
        The "forcing vector" for the u's and q's with dynamics and kinematics

    Parameters
    ==========

    frame : ReferenceFrame
        The inertial reference frame for the system.
    q_ind : iterable of dynamicsymbols
        Independent generalized coordinates.
    u_ind : iterable of dynamicsymbols
        Independent generalized speeds.
    kd_eqs : iterable of Expr, optional
        Kinematic differential equations, which linearly relate the generalized
        speeds to the time-derivatives of the generalized coordinates.
    q_dependent : iterable of dynamicsymbols, optional
        Dependent generalized coordinates.
    configuration_constraints : iterable of Expr, optional
        Constraints on the system's configuration, i.e. holonomic constraints.
    u_dependent : iterable of dynamicsymbols, optional
        Dependent generalized speeds.
    velocity_constraints : iterable of Expr, optional
        Constraints on the system's velocity, i.e. the combination of the
        nonholonomic constraints and the time-derivative of the holonomic
        constraints.
    acceleration_constraints : iterable of Expr, optional
        Constraints on the system's acceleration, by default these are the
        time-derivative of the velocity constraints.
    u_auxiliary : iterable of dynamicsymbols, optional
        Auxiliary generalized speeds.
    bodies : iterable of Particle and/or RigidBody, optional
        The particles and rigid bodies in the system.
    forcelist : iterable of tuple[Point | ReferenceFrame, Vector], optional
        Forces and torques applied on the system.
    explicit_kinematics : bool
        Boolean whether the mass matrices and forcing vectors should use the
        explicit form (default) or implicit form for kinematics.
        See the notes for more details.
    kd_eqs_solver : str, callable
        Method used to solve the kinematic differential equations. If a string
        is supplied, it should be a valid method that can be used with the
        :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
        supplied, it should have the format ``f(A, rhs)``, where it solves the
        equations and returns the solution. The default utilizes LU solve. See
        the notes for more information.
    constraint_solver : str, callable
        Method used to solve the velocity constraints. If a string is
        supplied, it should be a valid method that can be used with the
        :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
        supplied, it should have the format ``f(A, rhs)``, where it solves the
        equations and returns the solution. The default utilizes LU solve. See
        the notes for more information.

    Notes
    =====

    The mass matrices and forcing vectors related to kinematic equations
    are given in the explicit form by default. In other words, the kinematic
    mass matrix is $\mathbf{k_{k\dot{q}}} = \mathbf{I}$.
    In order to get the implicit form of those matrices/vectors, you can set the
    ``explicit_kinematics`` attribute to ``False``. So $\mathbf{k_{k\dot{q}}}$
    is not necessarily an identity matrix. This can provide more compact
    equations for non-simple kinematics.

    Two linear solvers can be supplied to ``KanesMethod``: one for solving the
    kinematic differential equations and one to solve the velocity constraints.
    Both of these sets of equations can be expressed as a linear system ``Ax = rhs``,
    which have to be solved in order to obtain the equations of motion.

    The default solver ``'LU'``, which stands for LU solve, results relatively low
    number of operations. The weakness of this method is that it can result in zero
    division errors.

    If zero divisions are encountered, a possible solver which may solve the problem
    is ``"CRAMER"``. This method uses Cramer's rule to solve the system. This method
    is slower and results in more operations than the default solver. However it only
    uses a single division by default per entry of the solution.

    While a valid list of solvers can be found at
    :meth:`sympy.matrices.matrixbase.MatrixBase.solve`, it is also possible to supply a
    `callable`. This way it is possible to use a different solver routine. If the
    kinematic differential equations are not too complex it can be worth it to simplify
    the solution by using ``lambda A, b: simplify(Matrix.LUsolve(A, b))``. Another
    option solver one may use is :func:`sympy.solvers.solveset.linsolve`. This can be
    done using `lambda A, b: tuple(linsolve((A, b)))[0]`, where we select the first
    solution as our system should have only one unique solution.

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized speeds and coordinates and their
    derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.mechanics import Point, Particle, KanesMethod
        >>> q, u = dynamicsymbols('q u')
        >>> qd, ud = dynamicsymbols('q u', 1)
        >>> m, c, k = symbols('m c k')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, u * N.x)

    Next we need to arrange/store information in the way that KanesMethod
    requires. The kinematic differential equations should be an iterable of
    expressions. A list of forces/torques must be constructed, where each entry
    in the list is a (Point, Vector) or (ReferenceFrame, Vector) tuple, where
    the Vectors represent the Force or Torque.
    Next a particle needs to be created, and it needs to have a point and mass
    assigned to it.
    Finally, a list of all bodies and particles needs to be created.

        >>> kd = [qd - u]
        >>> FL = [(P, (-k * q - c * u) * N.x)]
        >>> pa = Particle('pa', P, m)
        >>> BL = [pa]

    Finally we can generate the equations of motion.
    First we create the KanesMethod object and supply an inertial frame,
    coordinates, generalized speeds, and the kinematic differential equations.
    Additional quantities such as configuration and motion constraints,
    dependent coordinates and speeds, and auxiliary speeds are also supplied
    here (see the online documentation).
    Next we form FR* and FR to complete: Fr + Fr* = 0.
    We have the equations of motion at this point.
    It makes sense to rearrange them though, so we calculate the mass matrix and
    the forcing terms, for E.o.M. in the form: [MM] udot = forcing, where MM is
    the mass matrix, udot is a vector of the time derivatives of the
    generalized speeds, and forcing is a vector representing "forcing" terms.

        >>> KM = KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=kd)
        >>> (fr, frstar) = KM.kanes_equations(BL, FL)
        >>> MM = KM.mass_matrix
        >>> forcing = KM.forcing
        >>> rhs = MM.inv() * forcing
        >>> rhs
        Matrix([[(-c*u(t) - k*q(t))/m]])
        >>> KM.linearize(A_and_B=True)[0]
        Matrix([
        [   0,    1],
        [-k/m, -c/m]])

    Please look at the documentation pages for more information on how to
    perform linearization and how to deal with dependent coordinates & speeds,
    and how do deal with bringing non-contributing forces into evidence.

    """

    def __init__(self, frame, q_ind, u_ind, kd_eqs=None, q_dependent=None,
                 configuration_constraints=None, u_dependent=None,
                 velocity_constraints=None, acceleration_constraints=None,
                 u_auxiliary=None, bodies=None, forcelist=None,
                 explicit_kinematics=True, kd_eqs_solver='LU',
                 constraint_solver='LU'):

        """Please read the online documentation. """
        if not q_ind:
            q_ind = [dynamicsymbols('dummy_q')]
            kd_eqs = [dynamicsymbols('dummy_kd')]

        if not isinstance(frame, ReferenceFrame):
            raise TypeError('An inertial ReferenceFrame must be supplied')
        self._inertial = frame

        self._fr = None
        self._frstar = None

        self._forcelist = forcelist
        self._bodylist = bodies

        self.explicit_kinematics = explicit_kinematics
        self._constraint_solver = constraint_solver
        self._initialize_vectors(q_ind, q_dependent, u_ind, u_dependent,
                u_auxiliary)
        _validate_coordinates(self.q, self.u)
        self._initialize_kindiffeq_matrices(kd_eqs, kd_eqs_solver)
        self._initialize_constraint_matrices(
            configuration_constraints, velocity_constraints,
            acceleration_constraints, constraint_solver)

    def _initialize_vectors(self, q_ind, q_dep, u_ind, u_dep, u_aux):
        """Initialize the coordinate and speed vectors."""

        none_handler = lambda x: Matrix(x) if x else Matrix()

        # Initialize generalized coordinates
        q_dep = none_handler(q_dep)
        if not iterable(q_ind):
            raise TypeError('Generalized coordinates must be an iterable.')
        if not iterable(q_dep):
            raise TypeError('Dependent coordinates must be an iterable.')
        q_ind = Matrix(q_ind)
        self._qdep = q_dep
        self._q = Matrix([q_ind, q_dep])
        self._qdot = self.q.diff(dynamicsymbols._t)

        # Initialize generalized speeds
        u_dep = none_handler(u_dep)
        if not iterable(u_ind):
            raise TypeError('Generalized speeds must be an iterable.')
        if not iterable(u_dep):
            raise TypeError('Dependent speeds must be an iterable.')
        u_ind = Matrix(u_ind)
        self._udep = u_dep
        self._u = Matrix([u_ind, u_dep])
        self._udot = self.u.diff(dynamicsymbols._t)
        self._uaux = none_handler(u_aux)

    def _initialize_constraint_matrices(self, config, vel, acc, linear_solver='LU'):
        """Initializes constraint matrices."""
        linear_solver = _parse_linear_solver(linear_solver)
        # Define vector dimensions
        o = len(self.u)
        m = len(self._udep)
        p = o - m
        none_handler = lambda x: Matrix(x) if x else Matrix()

        # Initialize configuration constraints
        config = none_handler(config)
        if len(self._qdep) != len(config):
            raise ValueError('There must be an equal number of dependent '
                             'coordinates and configuration constraints.')
        self._f_h = none_handler(config)

        # Initialize velocity and acceleration constraints
        vel = none_handler(vel)
        acc = none_handler(acc)
        if len(vel) != m:
            raise ValueError('There must be an equal number of dependent '
                             'speeds and velocity constraints.')
        if acc and (len(acc) != m):
            raise ValueError('There must be an equal number of dependent '
                             'speeds and acceleration constraints.')
        if vel:

            # When calling kanes_equations, another class instance will be
            # created if auxiliary u's are present. In this case, the
            # computation of kinetic differential equation matrices will be
            # skipped as this was computed during the original KanesMethod
            # object, and the qd_u_map will not be available.
            if self._qdot_u_map is not None:
                vel = msubs(vel, self._qdot_u_map)
            self._k_nh, f_nh_neg = linear_eq_to_matrix(vel, self.u[:])
            self._f_nh = -f_nh_neg

            # If no acceleration constraints given, calculate them.
            if not acc:
                _f_dnh = (self._k_nh.diff(dynamicsymbols._t) * self.u +
                    self._f_nh.diff(dynamicsymbols._t))
                if self._qdot_u_map is not None:
                    _f_dnh = msubs(_f_dnh, self._qdot_u_map)
                self._f_dnh = _f_dnh
                self._k_dnh = self._k_nh
            else:
                if self._qdot_u_map is not None:
                    acc = msubs(acc, self._qdot_u_map)

                self._k_dnh, f_dnh_neg = linear_eq_to_matrix(acc, self._udot[:])
                self._f_dnh = -f_dnh_neg
            # Form of non-holonomic constraints is B*u + C = 0.
            # We partition B into independent and dependent columns:
            # Ars is then -B_dep.inv() * B_ind, and it relates dependent speeds
            # to independent speeds as: udep = Ars*uind, neglecting the C term.
            B_ind = self._k_nh[:, :p]
            B_dep = self._k_nh[:, p:o]
            self._Ars = -linear_solver(B_dep, B_ind)
        else:
            self._f_nh = Matrix()
            self._k_nh = Matrix()
            self._f_dnh = Matrix()
            self._k_dnh = Matrix()
            self._Ars = Matrix()

    def _initialize_kindiffeq_matrices(self, kdeqs, linear_solver='LU'):
        """Initialize the kinematic differential equation matrices.

        Parameters
        ==========
        kdeqs : sequence of sympy expressions
            Kinematic differential equations in the form of f(u,q',q,t) where
            f() = 0. The equations have to be linear in the time-derivatives of
            the generalized coordinates and in the generalized speeds.

        """
        linear_solver = _parse_linear_solver(linear_solver)
        if kdeqs:
            if len(self.q) != len(kdeqs):
                raise ValueError('There must be an equal number of kinematic '
                                 'differential equations and coordinates.')

            u = self.u
            qdot = self._qdot

            kdeqs = Matrix(kdeqs)

            u_zero = dict.fromkeys(u, 0)
            uaux_zero = dict.fromkeys(self._uaux, 0)
            qdot_zero = dict.fromkeys(qdot, 0)

            # Extract the linear coefficient matrices as per the following
            # equation:
            #
            # k_ku(q,t)*u(t) + k_kqdot(q,t)*q'(t) + f_k(q,t) = 0
            #
            k_ku = kdeqs.jacobian(u)
            k_kqdot = kdeqs.jacobian(qdot)
            f_k = kdeqs.xreplace(u_zero).xreplace(qdot_zero)

            # The kinematic differential equations should be linear in both q'
            # and u so check for u and q' in the components.
            dy_syms = find_dynamicsymbols(k_ku.row_join(k_kqdot).row_join(f_k))
            nonlin_vars = [vari for vari in u[:] + qdot[:] if vari in dy_syms]
            if nonlin_vars:
                msg = ('The provided kinematic differential equations are '
                       'nonlinear in {}. They must be linear in the '
                       'generalized speeds and derivatives of the generalized '
                       'coordinates.')
                raise ValueError(msg.format(nonlin_vars))

            self._f_k_implicit = f_k.xreplace(uaux_zero)
            self._k_ku_implicit = k_ku.xreplace(uaux_zero)
            self._k_kqdot_implicit = k_kqdot

            # Solve for q'(t) such that the coefficient matrices are now in
            # this form:
            #
            # k_kqdot^-1*k_ku*u(t) + I*q'(t) + k_kqdot^-1*f_k = 0
            #
            # NOTE : Solving the kinematic differential equations here is not
            # necessary and prevents the equations from being provided in fully
            # implicit form.
            f_k_explicit = linear_solver(k_kqdot, f_k)
            k_ku_explicit = linear_solver(k_kqdot, k_ku)
            self._qdot_u_map = dict(zip(qdot, -(k_ku_explicit*u + f_k_explicit)))

            self._f_k = f_k_explicit.xreplace(uaux_zero)
            self._k_ku = k_ku_explicit.xreplace(uaux_zero)
            self._k_kqdot = eye(len(qdot))

        else:
            self._qdot_u_map = None
            self._f_k_implicit = self._f_k = Matrix()
            self._k_ku_implicit = self._k_ku = Matrix()
            self._k_kqdot_implicit = self._k_kqdot = Matrix()

    def _form_fr(self, fl):
        """Form the generalized active force."""
        if fl is not None and (len(fl) == 0 or not iterable(fl)):
            raise ValueError('Force pairs must be supplied in an '
                'non-empty iterable or None.')

        N = self._inertial
        # pull out relevant velocities for constructing partial velocities
        vel_list, f_list = _f_list_parser(fl, N)
        vel_list = [msubs(i, self._qdot_u_map) for i in vel_list]
        f_list = [msubs(i, self._qdot_u_map) for i in f_list]

        # Fill Fr with dot product of partial velocities and forces
        o = len(self.u)
        b = len(f_list)
        FR = zeros(o, 1)
        partials = partial_velocity(vel_list, self.u, N)
        for i in range(o):
            FR[i] = sum(partials[j][i].dot(f_list[j]) for j in range(b))

        # In case there are dependent speeds
        if self._udep:
            p = o - len(self._udep)
            FRtilde = FR[:p, 0]
            FRold = FR[p:o, 0]
            FRtilde += self._Ars.T * FRold
            FR = FRtilde

        self._forcelist = fl
        self._fr = FR
        return FR

    def _form_frstar(self, bl):
        """Form the generalized inertia force."""

        if not iterable(bl):
            raise TypeError('Bodies must be supplied in an iterable.')

        t = dynamicsymbols._t
        N = self._inertial
        # Dicts setting things to zero
        udot_zero = dict.fromkeys(self._udot, 0)
        uaux_zero = dict.fromkeys(self._uaux, 0)
        uauxdot = [diff(i, t) for i in self._uaux]
        uauxdot_zero = dict.fromkeys(uauxdot, 0)
        # Dictionary of q' and q'' to u and u'
        q_ddot_u_map = {k.diff(t): v.diff(t).xreplace(
            self._qdot_u_map) for (k, v) in self._qdot_u_map.items()}
        q_ddot_u_map.update(self._qdot_u_map)

        # Fill up the list of partials: format is a list with num elements
        # equal to number of entries in body list. Each of these elements is a
        # list - either of length 1 for the translational components of
        # particles or of length 2 for the translational and rotational
        # components of rigid bodies. The inner most list is the list of
        # partial velocities.
        def get_partial_velocity(body):
            if isinstance(body, RigidBody):
                vlist = [body.masscenter.vel(N), body.frame.ang_vel_in(N)]
            elif isinstance(body, Particle):
                vlist = [body.point.vel(N),]
            else:
                raise TypeError('The body list may only contain either '
                                'RigidBody or Particle as list elements.')
            v = [msubs(vel, self._qdot_u_map) for vel in vlist]
            return partial_velocity(v, self.u, N)
        partials = [get_partial_velocity(body) for body in bl]

        # Compute fr_star in two components:
        # fr_star = -(MM*u' + nonMM)
        o = len(self.u)
        MM = zeros(o, o)
        nonMM = zeros(o, 1)
        zero_uaux = lambda expr: msubs(expr, uaux_zero)
        zero_udot_uaux = lambda expr: msubs(msubs(expr, udot_zero), uaux_zero)
        for i, body in enumerate(bl):
            if isinstance(body, RigidBody):
                M = zero_uaux(body.mass)
                I = zero_uaux(body.central_inertia)
                vel = zero_uaux(body.masscenter.vel(N))
                omega = zero_uaux(body.frame.ang_vel_in(N))
                acc = zero_udot_uaux(body.masscenter.acc(N))
                inertial_force = (M.diff(t) * vel + M * acc)
                inertial_torque = zero_uaux((I.dt(body.frame).dot(omega)) +
                    msubs(I.dot(body.frame.ang_acc_in(N)), udot_zero) +
                    (omega.cross(I.dot(omega))))
                for j in range(o):
                    tmp_vel = zero_uaux(partials[i][0][j])
                    tmp_ang = zero_uaux(I.dot(partials[i][1][j]))
                    for k in range(o):
                        # translational
                        MM[j, k] += M*tmp_vel.dot(partials[i][0][k])
                        # rotational
                        MM[j, k] += tmp_ang.dot(partials[i][1][k])
                    nonMM[j] += inertial_force.dot(partials[i][0][j])
                    nonMM[j] += inertial_torque.dot(partials[i][1][j])
            else:
                M = zero_uaux(body.mass)
                vel = zero_uaux(body.point.vel(N))
                acc = zero_udot_uaux(body.point.acc(N))
                inertial_force = (M.diff(t) * vel + M * acc)
                for j in range(o):
                    temp = zero_uaux(partials[i][0][j])
                    for k in range(o):
                        MM[j, k] += M*temp.dot(partials[i][0][k])
                    nonMM[j] += inertial_force.dot(partials[i][0][j])
        # Compose fr_star out of MM and nonMM
        MM = zero_uaux(msubs(MM, q_ddot_u_map))
        nonMM = msubs(msubs(nonMM, q_ddot_u_map),
                udot_zero, uauxdot_zero, uaux_zero)
        fr_star = -(MM * msubs(Matrix(self._udot), uauxdot_zero) + nonMM)

        # If there are dependent speeds, we need to find fr_star_tilde
        if self._udep:
            p = o - len(self._udep)
            fr_star_ind = fr_star[:p, 0]
            fr_star_dep = fr_star[p:o, 0]
            fr_star = fr_star_ind + (self._Ars.T * fr_star_dep)
            # Apply the same to MM
            MMi = MM[:p, :]
            MMd = MM[p:o, :]
            MM = MMi + (self._Ars.T * MMd)
            # Apply the same to nonMM
            nonMM = nonMM[:p, :] + (self._Ars.T * nonMM[p:o, :])

        self._bodylist = bl
        self._frstar = fr_star
        self._k_d = MM
        self._f_d = -(self._fr - nonMM)
        return fr_star

    def to_linearizer(self, linear_solver='LU'):
        """Returns an instance of the Linearizer class, initiated from the
        data in the KanesMethod class. This may be more desirable than using
        the linearize class method, as the Linearizer object will allow more
        efficient recalculation (i.e. about varying operating points).

        Parameters
        ==========
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.

        Returns
        =======
        Linearizer
            An instantiated
            :class:`sympy.physics.mechanics.linearize.Linearizer`.

        """

        if (self._fr is None) or (self._frstar is None):
            raise ValueError('Need to compute Fr, Fr* first.')

        # Get required equation components. The Kane's method class breaks
        # these into pieces. Need to reassemble
        f_c = self._f_h
        if self._f_nh and self._k_nh:
            f_v = self._f_nh + self._k_nh*Matrix(self.u)
        else:
            f_v = Matrix()
        if self._f_dnh and self._k_dnh:
            f_a = self._f_dnh + self._k_dnh*Matrix(self._udot)
        else:
            f_a = Matrix()
        # Dicts to sub to zero, for splitting up expressions
        u_zero = dict.fromkeys(self.u, 0)
        ud_zero = dict.fromkeys(self._udot, 0)
        qd_zero = dict.fromkeys(self._qdot, 0)
        qd_u_zero = dict.fromkeys(Matrix([self._qdot, self.u]), 0)
        # Break the kinematic differential eqs apart into f_0 and f_1
        f_0 = msubs(self._f_k, u_zero) + self._k_kqdot*Matrix(self._qdot)
        f_1 = msubs(self._f_k, qd_zero) + self._k_ku*Matrix(self.u)
        # Break the dynamic differential eqs into f_2 and f_3
        f_2 = msubs(self._frstar, qd_u_zero)
        f_3 = msubs(self._frstar, ud_zero) + self._fr
        f_4 = zeros(len(f_2), 1)

        # Get the required vector components
        q = self.q
        u = self.u
        if self._qdep:
            q_i = q[:-len(self._qdep)]
        else:
            q_i = q
        q_d = self._qdep
        if self._udep:
            u_i = u[:-len(self._udep)]
        else:
            u_i = u
        u_d = self._udep

        # Form dictionary to set auxiliary speeds & their derivatives to 0.
        uaux = self._uaux
        uauxdot = uaux.diff(dynamicsymbols._t)
        uaux_zero = dict.fromkeys(Matrix([uaux, uauxdot]), 0)

        # Checking for dynamic symbols outside the dynamic differential
        # equations; throws error if there is.
        sym_list = set(Matrix([q, self._qdot, u, self._udot, uaux, uauxdot]))
        if any(find_dynamicsymbols(i, sym_list) for i in [self._k_kqdot,
                self._k_ku, self._f_k, self._k_dnh, self._f_dnh, self._k_d]):
            raise ValueError('Cannot have dynamicsymbols outside dynamic \
                             forcing vector.')

        # Find all other dynamic symbols, forming the forcing vector r.
        # Sort r to make it canonical.
        r = list(find_dynamicsymbols(msubs(self._f_d, uaux_zero), sym_list))
        r.sort(key=default_sort_key)

        # Check for any derivatives of variables in r that are also found in r.
        for i in r:
            if diff(i, dynamicsymbols._t) in r:
                raise ValueError('Cannot have derivatives of specified \
                                 quantities when linearizing forcing terms.')
        return Linearizer(f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i,
                q_d, u_i, u_d, r, linear_solver=linear_solver)

    # TODO : Remove `new_method` after 1.1 has been released.
    def linearize(self, *, new_method=None, linear_solver='LU', **kwargs):
        """ Linearize the equations of motion about a symbolic operating point.

        Parameters
        ==========
        new_method
            Deprecated, does nothing and will be removed.
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.
        **kwargs
            Extra keyword arguments are passed to
            :meth:`sympy.physics.mechanics.linearize.Linearizer.linearize`.

        Explanation
        ===========

        If kwarg A_and_B is False (default), returns M, A, B, r for the
        linearized form, M*[q', u']^T = A*[q_ind, u_ind]^T + B*r.

        If kwarg A_and_B is True, returns A, B, r for the linearized form
        dx = A*x + B*r, where x = [q_ind, u_ind]^T. Note that this is
        computationally intensive if there are many symbolic parameters. For
        this reason, it may be more desirable to use the default A_and_B=False,
        returning M, A, and B. Values may then be substituted in to these
        matrices, and the state space form found as
        A = P.T*M.inv()*A, B = P.T*M.inv()*B, where P = Linearizer.perm_mat.

        In both cases, r is found as all dynamicsymbols in the equations of
        motion that are not part of q, u, q', or u'. They are sorted in
        canonical form.

        The operating points may be also entered using the ``op_point`` kwarg.
        This takes a dictionary of {symbol: value}, or a an iterable of such
        dictionaries. The values may be numeric or symbolic. The more values
        you can specify beforehand, the faster this computation will run.

        For more documentation, please see the ``Linearizer`` class.

        """

        linearizer = self.to_linearizer(linear_solver=linear_solver)
        result = linearizer.linearize(**kwargs)
        return result + (linearizer.r,)

    def kanes_equations(self, bodies=None, loads=None):
        """ Method to form Kane's equations, Fr + Fr* = 0.

        Explanation
        ===========

        Returns (Fr, Fr*). In the case where auxiliary generalized speeds are
        present (say, s auxiliary speeds, o generalized speeds, and m motion
        constraints) the length of the returned vectors will be o - m + s in
        length. The first o - m equations will be the constrained Kane's
        equations, then the s auxiliary Kane's equations. These auxiliary
        equations can be accessed with the auxiliary_eqs property.

        Parameters
        ==========

        bodies : iterable
            An iterable of all RigidBody's and Particle's in the system.
            A system must have at least one body.
        loads : iterable
            Takes in an iterable of (Particle, Vector) or (ReferenceFrame, Vector)
            tuples which represent the force at a point or torque on a frame.
            Must be either a non-empty iterable of tuples or None which corresponds
            to a system with no constraints.
        """
        if bodies is None:
            bodies = self.bodies
        if  loads is None and self._forcelist is not None:
            loads = self._forcelist
        if loads == []:
            loads = None
        if not self._k_kqdot:
            raise AttributeError('Create an instance of KanesMethod with '
                    'kinematic differential equations to use this method.')
        fr = self._form_fr(loads)
        frstar = self._form_frstar(bodies)
        if self._uaux:
            if not self._udep:
                km = KanesMethod(self._inertial, self.q, self._uaux,
                             u_auxiliary=self._uaux, constraint_solver=self._constraint_solver)
            else:
                km = KanesMethod(self._inertial, self.q, self._uaux,
                        u_auxiliary=self._uaux, u_dependent=self._udep,
                        velocity_constraints=(self._k_nh * self.u +
                        self._f_nh),
                        acceleration_constraints=(self._k_dnh * self._udot +
                        self._f_dnh),
                        constraint_solver=self._constraint_solver
                        )
            km._qdot_u_map = self._qdot_u_map
            self._km = km
            fraux = km._form_fr(loads)
            frstaraux = km._form_frstar(bodies)
            self._aux_eq = fraux + frstaraux
            self._fr = fr.col_join(fraux)
            self._frstar = frstar.col_join(frstaraux)
        return (self._fr, self._frstar)

    def _form_eoms(self):
        fr, frstar = self.kanes_equations(self.bodylist, self.forcelist)
        return fr + frstar

    def rhs(self, inv_method=None):
        """Returns the system's equations of motion in first order form. The
        output is the right hand side of::

           x' = |q'| =: f(q, u, r, p, t)
                |u'|

        The right hand side is what is needed by most numerical ODE
        integrators.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        """
        rhs = zeros(len(self.q) + len(self.u), 1)
        kdes = self.kindiffdict()
        for i, q_i in enumerate(self.q):
            rhs[i] = kdes[q_i.diff()]

        if inv_method is None:
            rhs[len(self.q):, 0] = self.mass_matrix.LUsolve(self.forcing)
        else:
            rhs[len(self.q):, 0] = (self.mass_matrix.inv(inv_method,
                                                         try_block_diag=True) *
                                    self.forcing)

        return rhs

    def kindiffdict(self):
        """Returns a dictionary mapping q' to u."""
        if not self._qdot_u_map:
            raise AttributeError('Create an instance of KanesMethod with '
                    'kinematic differential equations to use this method.')
        return self._qdot_u_map

    @property
    def auxiliary_eqs(self):
        """A matrix containing the auxiliary equations."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        if not self._uaux:
            raise ValueError('No auxiliary speeds have been declared.')
        return self._aux_eq

    @property
    def mass_matrix_kin(self):
        r"""The kinematic "mass matrix" $\mathbf{k_{k\dot{q}}}$ of the system."""
        return self._k_kqdot if self.explicit_kinematics else self._k_kqdot_implicit

    @property
    def forcing_kin(self):
        """The kinematic "forcing vector" of the system."""
        if self.explicit_kinematics:
            return -(self._k_ku * Matrix(self.u) + self._f_k)
        else:
            return -(self._k_ku_implicit * Matrix(self.u) + self._f_k_implicit)

    @property
    def mass_matrix(self):
        """The mass matrix of the system."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        return Matrix([self._k_d, self._k_dnh])

    @property
    def forcing(self):
        """The forcing vector of the system."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        return -Matrix([self._f_d, self._f_dnh])

    @property
    def mass_matrix_full(self):
        """The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        o, n = len(self.u), len(self.q)
        return (self.mass_matrix_kin.row_join(zeros(n, o))).col_join(
            zeros(o, n).row_join(self.mass_matrix))

    @property
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        return Matrix([self.forcing_kin, self.forcing])

    @property
    def q(self):
        return self._q

    @property
    def u(self):
        return self._u

    @property
    def bodylist(self):
        return self._bodylist

    @property
    def forcelist(self):
        return self._forcelist

    @property
    def bodies(self):
        return self._bodylist

    @property
    def loads(self):
        return self._forcelist
<|endoftext|>__all__ = ['Linearizer']

from sympy import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs, _parse_linear_solver

from collections import namedtuple
from collections.abc import Iterable


class Linearizer:
    """This object holds the general model form for a dynamic system. This
    model is used for computing the linearized form of the system, while
    properly dealing with constraints leading to  dependent coordinates and
    speeds. The notation and method is described in [1]_.

    Attributes
    ==========

    f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : Matrix
        Matrices holding the general system form.
    q, u, r : Matrix
        Matrices holding the generalized coordinates, speeds, and
        input vectors.
    q_i, u_i : Matrix
        Matrices of the independent generalized coordinates and speeds.
    q_d, u_d : Matrix
        Matrices of the dependent generalized coordinates and speeds.
    perm_mat : Matrix
        Permutation matrix such that [q_ind, u_ind]^T = perm_mat*[q, u]^T

    References
    ==========

    .. [1] D. L. Peterson, G. Gede, and M. Hubbard, "Symbolic linearization of
           equations of motion of constrained multibody systems," Multibody
           Syst Dyn, vol. 33, no. 2, pp. 143-161, Feb. 2015, doi:
           10.1007/s11044-014-9436-5.

    """

    def __init__(self, f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i=None,
                 q_d=None, u_i=None, u_d=None, r=None, lams=None,
                 linear_solver='LU'):
        """
        Parameters
        ==========

        f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : array_like
            System of equations holding the general system form.
            Supply empty array or Matrix if the parameter
            does not exist.
        q : array_like
            The generalized coordinates.
        u : array_like
            The generalized speeds
        q_i, u_i : array_like, optional
            The independent generalized coordinates and speeds.
        q_d, u_d : array_like, optional
            The dependent generalized coordinates and speeds.
        r : array_like, optional
            The input variables.
        lams : array_like, optional
            The lagrange multipliers
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.

        """
        self.linear_solver = _parse_linear_solver(linear_solver)

        # Generalized equation form
        self.f_0 = Matrix(f_0)
        self.f_1 = Matrix(f_1)
        self.f_2 = Matrix(f_2)
        self.f_3 = Matrix(f_3)
        self.f_4 = Matrix(f_4)
        self.f_c = Matrix(f_c)
        self.f_v = Matrix(f_v)
        self.f_a = Matrix(f_a)

        # Generalized equation variables
        self.q = Matrix(q)
        self.u = Matrix(u)
        none_handler = lambda x: Matrix(x) if x else Matrix()
        self.q_i = none_handler(q_i)
        self.q_d = none_handler(q_d)
        self.u_i = none_handler(u_i)
        self.u_d = none_handler(u_d)
        self.r = none_handler(r)
        self.lams = none_handler(lams)

        # Derivatives of generalized equation variables
        self._qd = self.q.diff(dynamicsymbols._t)
        self._ud = self.u.diff(dynamicsymbols._t)
        # If the user doesn't actually use generalized variables, and the
        # qd and u vectors have any intersecting variables, this can cause
        # problems. We'll fix this with some hackery, and Dummy variables
        dup_vars = set(self._qd).intersection(self.u)
        self._qd_dup = Matrix([var if var not in dup_vars else Dummy() for var
                               in self._qd])

        # Derive dimension terms
        l = len(self.f_c)
        m = len(self.f_v)
        n = len(self.q)
        o = len(self.u)
        s = len(self.r)
        k = len(self.lams)
        dims = namedtuple('dims', ['l', 'm', 'n', 'o', 's', 'k'])
        self._dims = dims(l, m, n, o, s, k)

        self._Pq = None
        self._Pqi = None
        self._Pqd = None
        self._Pu = None
        self._Pui = None
        self._Pud = None
        self._C_0 = None
        self._C_1 = None
        self._C_2 = None
        self.perm_mat = None

        self._setup_done = False

    def _setup(self):
        # Calculations here only need to be run once. They are moved out of
        # the __init__ method to increase the speed of Linearizer creation.
        self._form_permutation_matrices()
        self._form_block_matrices()
        self._form_coefficient_matrices()
        self._setup_done = True

    def _form_permutation_matrices(self):
        """Form the permutation matrices Pq and Pu."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Compute permutation matrices
        if n != 0:
            self._Pq = permutation_matrix(self.q, Matrix([self.q_i, self.q_d]))
            if l > 0:
                self._Pqi = self._Pq[:, :-l]
                self._Pqd = self._Pq[:, -l:]
            else:
                self._Pqi = self._Pq
                self._Pqd = Matrix()
        if o != 0:
            self._Pu = permutation_matrix(self.u, Matrix([self.u_i, self.u_d]))
            if m > 0:
                self._Pui = self._Pu[:, :-m]
                self._Pud = self._Pu[:, -m:]
            else:
                self._Pui = self._Pu
                self._Pud = Matrix()
        # Compute combination permutation matrix for computing A and B
        P_col1 = Matrix([self._Pqi, zeros(o + k, n - l)])
        P_col2 = Matrix([zeros(n, o - m), self._Pui, zeros(k, o - m)])
        if P_col1:
            if P_col2:
                self.perm_mat = P_col1.row_join(P_col2)
            else:
                self.perm_mat = P_col1
        else:
            self.perm_mat = P_col2

    def _form_coefficient_matrices(self):
        """Form the coefficient matrices C_0, C_1, and C_2."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Build up the coefficient matrices C_0, C_1, and C_2
        # If there are configuration constraints (l > 0), form C_0 as normal.
        # If not, C_0 is I_(nxn). Note that this works even if n=0
        if l > 0:
            f_c_jac_q = self.f_c.jacobian(self.q)
            self._C_0 = (eye(n) - self._Pqd *
                         self.linear_solver(f_c_jac_q*self._Pqd,
                                            f_c_jac_q))*self._Pqi
        else:
            self._C_0 = eye(n)
        # If there are motion constraints (m > 0), form C_1 and C_2 as normal.
        # If not, C_1 is 0, and C_2 is I_(oxo). Note that this works even if
        # o = 0.
        if m > 0:
            f_v_jac_u = self.f_v.jacobian(self.u)
            temp = f_v_jac_u * self._Pud
            if n != 0:
                f_v_jac_q = self.f_v.jacobian(self.q)
                self._C_1 = -self._Pud * self.linear_solver(temp, f_v_jac_q)
            else:
                self._C_1 = zeros(o, n)
            self._C_2 = (eye(o) - self._Pud *
                         self.linear_solver(temp, f_v_jac_u))*self._Pui
        else:
            self._C_1 = zeros(o, n)
            self._C_2 = eye(o)

    def _form_block_matrices(self):
        """Form the block matrices for composing M, A, and B."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Block Matrix Definitions. These are only defined if under certain
        # conditions. If undefined, an empty matrix is used instead
        if n != 0:
            self._M_qq = self.f_0.jacobian(self._qd)
            self._A_qq = -(self.f_0 + self.f_1).jacobian(self.q)
        else:
            self._M_qq = Matrix()
            self._A_qq = Matrix()
        if n != 0 and m != 0:
            self._M_uqc = self.f_a.jacobian(self._qd_dup)
            self._A_uqc = -self.f_a.jacobian(self.q)
        else:
            self._M_uqc = Matrix()
            self._A_uqc = Matrix()
        if n != 0 and o - m + k != 0:
            self._M_uqd = self.f_3.jacobian(self._qd_dup)
            self._A_uqd = -(self.f_2 + self.f_3 + self.f_4).jacobian(self.q)
        else:
            self._M_uqd = Matrix()
            self._A_uqd = Matrix()
        if o != 0 and m != 0:
            self._M_uuc = self.f_a.jacobian(self._ud)
            self._A_uuc = -self.f_a.jacobian(self.u)
        else:
            self._M_uuc = Matrix()
            self._A_uuc = Matrix()
        if o != 0 and o - m + k != 0:
            self._M_uud = self.f_2.jacobian(self._ud)
            self._A_uud = -(self.f_2 + self.f_3).jacobian(self.u)
        else:
            self._M_uud = Matrix()
            self._A_uud = Matrix()
        if o != 0 and n != 0:
            self._A_qu = -self.f_1.jacobian(self.u)
        else:
            self._A_qu = Matrix()
        if k != 0 and o - m + k != 0:
            self._M_uld = self.f_4.jacobian(self.lams)
        else:
            self._M_uld = Matrix()
        if s != 0 and o - m + k != 0:
            self._B_u = -self.f_3.jacobian(self.r)
        else:
            self._B_u = Matrix()

    def linearize(self, op_point=None, A_and_B=False, simplify=False):
        """Linearize the system about the operating point. Note that
        q_op, u_op, qd_op, ud_op must satisfy the equations of motion.
        These may be either symbolic or numeric.

        Parameters
        ==========
        op_point : dict or iterable of dicts, optional
            Dictionary or iterable of dictionaries containing the operating
            point conditions for all or a subset of the generalized
            coordinates, generalized speeds, and time derivatives of the
            generalized speeds. These will be substituted into the linearized
            system before the linearization is complete. Leave set to ``None``
            if you want the operating point to be an arbitrary set of symbols.
            Note that any reduction in symbols (whether substituted for numbers
            or expressions with a common parameter) will result in faster
            runtime.
        A_and_B : bool, optional
            If A_and_B=False (default), (M, A, B) is returned and of
            A_and_B=True, (A, B) is returned. See below.
        simplify : bool, optional
            Determines if returned values are simplified before return.
            For large expressions this may be time consuming. Default is False.

        Returns
        =======
        M, A, B : Matrices, ``A_and_B=False``
            Matrices from the implicit form:
                ``[M]*[q', u']^T = [A]*[q_ind, u_ind]^T + [B]*r``
        A, B : Matrices, ``A_and_B=True``
            Matrices from the explicit form:
                ``[q_ind', u_ind']^T = [A]*[q_ind, u_ind]^T + [B]*r``

        Notes
        =====

        Note that the process of solving with A_and_B=True is computationally
        intensive if there are many symbolic parameters. For this reason, it
        may be more desirable to use the default A_and_B=False, returning M, A,
        and B. More values may then be substituted in to these matrices later
        on. The state space form can then be found as A = P.T*M.LUsolve(A), B =
        P.T*M.LUsolve(B), where P = Linearizer.perm_mat.

        """

        # Run the setup if needed:
        if not self._setup_done:
            self._setup()

        # Compose dict of operating conditions
        if isinstance(op_point, dict):
            op_point_dict = op_point
        elif isinstance(op_point, Iterable):
            op_point_dict = {}
            for op in op_point:
                op_point_dict.update(op)
        else:
            op_point_dict = {}

        # Extract dimension variables
        l, m, n, o, s, k = self._dims

        # Rename terms to shorten expressions
        M_qq = self._M_qq
        M_uqc = self._M_uqc
        M_uqd = self._M_uqd
        M_uuc = self._M_uuc
        M_uud = self._M_uud
        M_uld = self._M_uld
        A_qq = self._A_qq
        A_uqc = self._A_uqc
        A_uqd = self._A_uqd
        A_qu = self._A_qu
        A_uuc = self._A_uuc
        A_uud = self._A_uud
        B_u = self._B_u
        C_0 = self._C_0
        C_1 = self._C_1
        C_2 = self._C_2

        # Build up Mass Matrix
        #     |M_qq    0_nxo   0_nxk|
        # M = |M_uqc   M_uuc   0_mxk|
        #     |M_uqd   M_uud   M_uld|
        if o != 0:
            col2 = Matrix([zeros(n, o), M_uuc, M_uud])
        if k != 0:
            col3 = Matrix([zeros(n + m, k), M_uld])
        if n != 0:
            col1 = Matrix([M_qq, M_uqc, M_uqd])
            if o != 0 and k != 0:
                M = col1.row_join(col2).row_join(col3)
            elif o != 0:
                M = col1.row_join(col2)
            else:
                M = col1
        elif k != 0:
            M = col2.row_join(col3)
        else:
            M = col2
        M_eq = msubs(M, op_point_dict)

        # Build up state coefficient matrix A
        #     |(A_qq + A_qu*C_1)*C_0       A_qu*C_2|
        # A = |(A_uqc + A_uuc*C_1)*C_0    A_uuc*C_2|
        #     |(A_uqd + A_uud*C_1)*C_0    A_uud*C_2|
        # Col 1 is only defined if n != 0
        if n != 0:
            r1c1 = A_qq
            if o != 0:
                r1c1 += (A_qu * C_1)
            r1c1 = r1c1 * C_0
            if m != 0:
                r2c1 = A_uqc
                if o != 0:
                    r2c1 += (A_uuc * C_1)
                r2c1 = r2c1 * C_0
            else:
                r2c1 = Matrix()
            if o - m + k != 0:
                r3c1 = A_uqd
                if o != 0:
                    r3c1 += (A_uud * C_1)
                r3c1 = r3c1 * C_0
            else:
                r3c1 = Matrix()
            col1 = Matrix([r1c1, r2c1, r3c1])
        else:
            col1 = Matrix()
        # Col 2 is only defined if o != 0
        if o != 0:
            if n != 0:
                r1c2 = A_qu * C_2
            else:
                r1c2 = Matrix()
            if m != 0:
                r2c2 = A_uuc * C_2
            else:
                r2c2 = Matrix()
            if o - m + k != 0:
                r3c2 = A_uud * C_2
            else:
                r3c2 = Matrix()
            col2 = Matrix([r1c2, r2c2, r3c2])
        else:
            col2 = Matrix()
        if col1:
            if col2:
                Amat = col1.row_join(col2)
            else:
                Amat = col1
        else:
            Amat = col2
        Amat_eq = msubs(Amat, op_point_dict)

        # Build up the B matrix if there are forcing variables
        #     |0_(n + m)xs|
        # B = |B_u        |
        if s != 0 and o - m + k != 0:
            Bmat = zeros(n + m, s).col_join(B_u)
            Bmat_eq = msubs(Bmat, op_point_dict)
        else:
            Bmat_eq = Matrix()

        # kwarg A_and_B indicates to return  A, B for forming the equation
        # dx = [A]x + [B]r, where x = [q_indnd, u_indnd]^T,
        if A_and_B:
            A_cont = self.perm_mat.T * self.linear_solver(M_eq, Amat_eq)
            if Bmat_eq:
                B_cont = self.perm_mat.T * self.linear_solver(M_eq, Bmat_eq)
            else:
                # Bmat = Matrix([]), so no need to sub
                B_cont = Bmat_eq
            if simplify:
                A_cont.simplify()
                B_cont.simplify()
            return A_cont, B_cont
        # Otherwise return M, A, B for forming the equation
        # [M]dx = [A]x + [B]r, where x = [q, u]^T
        else:
            if simplify:
                M_eq.simplify()
                Amat_eq.simplify()
                Bmat_eq.simplify()
            return M_eq, Amat_eq, Bmat_eq


def permutation_matrix(orig_vec, per_vec):
    """Compute the permutation matrix to change order of
    orig_vec into order of per_vec.

    Parameters
    ==========

    orig_vec : array_like
        Symbols in original ordering.
    per_vec : array_like
        Symbols in new ordering.

    Returns
    =======

    p_matrix : Matrix
        Permutation matrix such that orig_vec == (p_matrix * per_vec).
    """
    if not isinstance(orig_vec, (list, tuple)):
        orig_vec = flatten(orig_vec)
    if not isinstance(per_vec, (list, tuple)):
        per_vec = flatten(per_vec)
    if set(orig_vec) != set(per_vec):
        raise ValueError("orig_vec and per_vec must be the same length, "
                         "and contain the same symbols.")
    ind_list = [orig_vec.index(i) for i in per_vec]
    p_matrix = zeros(len(orig_vec))
    for i, j in enumerate(ind_list):
        p_matrix[i, j] = 1
    return p_matrix
<|endoftext|>from abc import ABC, abstractmethod

class _Methods(ABC):
    """Abstract Base Class for all methods."""

    @abstractmethod
    def q(self):
        pass

    @abstractmethod
    def u(self):
        pass

    @abstractmethod
    def bodies(self):
        pass

    @abstractmethod
    def loads(self):
        pass

    @abstractmethod
    def mass_matrix(self):
        pass

    @abstractmethod
    def forcing(self):
        pass

    @abstractmethod
    def mass_matrix_full(self):
        pass

    @abstractmethod
    def forcing_full(self):
        pass

    def _form_eoms(self):
        raise NotImplementedError("Subclasses must implement this.")
<|endoftext|>from sympy import diff, zeros, Matrix, eye, sympify
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import dynamicsymbols, ReferenceFrame
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.functions import (
    find_dynamicsymbols, msubs, _f_list_parser, _validate_coordinates)
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable

__all__ = ['LagrangesMethod']


class LagrangesMethod(_Methods):
    """Lagrange's method object.

    Explanation
    ===========

    This object generates the equations of motion in a two step procedure. The
    first step involves the initialization of LagrangesMethod by supplying the
    Lagrangian and the generalized coordinates, at the bare minimum. If there
    are any constraint equations, they can be supplied as keyword arguments.
    The Lagrange multipliers are automatically generated and are equal in
    number to the constraint equations. Similarly any non-conservative forces
    can be supplied in an iterable (as described below and also shown in the
    example) along with a ReferenceFrame. This is also discussed further in the
    __init__ method.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    bodies : iterable
        Iterable containing the rigid bodies and particles of the system.
    mass_matrix : Matrix
        The system's mass matrix
    forcing : Matrix
        The system's forcing vector
    mass_matrix_full : Matrix
        The "mass matrix" for the qdot's, qdoubledot's, and the
        lagrange multipliers (lam)
    forcing_full : Matrix
        The forcing vector for the qdot's, qdoubledot's and
        lagrange multipliers (lam)

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized coordinates and their derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy.physics.mechanics import LagrangesMethod, Lagrangian
        >>> from sympy.physics.mechanics import ReferenceFrame, Particle, Point
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy import symbols
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, qd * N.x)

    We need to then prepare the information as required by LagrangesMethod to
    generate equations of motion.
    First we create the Particle, which has a point attached to it.
    Following this the lagrangian is created from the kinetic and potential
    energies.
    Then, an iterable of nonconservative forces/torques must be constructed,
    where each item is a (Point, Vector) or (ReferenceFrame, Vector) tuple,
    with the Vectors representing the nonconservative forces or torques.

        >>> Pa = Particle('Pa', P, m)
        >>> Pa.potential_energy = k * q**2 / 2.0
        >>> L = Lagrangian(N, Pa)
        >>> fl = [(P, -b * qd * N.x)]

    Finally we can generate the equations of motion.
    First we create the LagrangesMethod object. To do this one must supply
    the Lagrangian, and the generalized coordinates. The constraint equations,
    the forcelist, and the inertial frame may also be provided, if relevant.
    Next we generate Lagrange's equations of motion, such that:
    Lagrange's equations of motion = 0.
    We have the equations of motion at this point.

        >>> l = LagrangesMethod(L, [q], forcelist = fl, frame = N)
        >>> print(l.form_lagranges_equations())
        Matrix([[b*Derivative(q(t), t) + 1.0*k*q(t) + m*Derivative(q(t), (t, 2))]])

    We can also solve for the states using the 'rhs' method.

        >>> print(l.rhs())
        Matrix([[Derivative(q(t), t)], [(-b*Derivative(q(t), t) - 1.0*k*q(t))/m]])

    Please refer to the docstrings on each method for more details.
    """

    def __init__(self, Lagrangian, qs, forcelist=None, bodies=None, frame=None,
                 hol_coneqs=None, nonhol_coneqs=None):
        """Supply the following for the initialization of LagrangesMethod.

        Lagrangian : Sympifyable

        qs : array_like
            The generalized coordinates

        hol_coneqs : array_like, optional
            The holonomic constraint equations

        nonhol_coneqs : array_like, optional
            The nonholonomic constraint equations

        forcelist : iterable, optional
            Takes an iterable of (Point, Vector) or (ReferenceFrame, Vector)
            tuples which represent the force at a point or torque on a frame.
            This feature is primarily to account for the nonconservative forces
            and/or moments.

        bodies : iterable, optional
            Takes an iterable containing the rigid bodies and particles of the
            system.

        frame : ReferenceFrame, optional
            Supply the inertial frame. This is used to determine the
            generalized forces due to non-conservative forces.
        """

        self._L = Matrix([sympify(Lagrangian)])
        self.eom = None
        self._m_cd = Matrix()           # Mass Matrix of differentiated coneqs
        self._m_d = Matrix()            # Mass Matrix of dynamic equations
        self._f_cd = Matrix()           # Forcing part of the diff coneqs
        self._f_d = Matrix()            # Forcing part of the dynamic equations
        self.lam_coeffs = Matrix()      # The coeffecients of the multipliers

        forcelist = forcelist if forcelist else []
        if not iterable(forcelist):
            raise TypeError('Force pairs must be supplied in an iterable.')
        self._forcelist = forcelist
        if frame and not isinstance(frame, ReferenceFrame):
            raise TypeError('frame must be a valid ReferenceFrame')
        self._bodies = bodies
        self.inertial = frame

        self.lam_vec = Matrix()

        self._term1 = Matrix()
        self._term2 = Matrix()
        self._term3 = Matrix()
        self._term4 = Matrix()

        # Creating the qs, qdots and qdoubledots
        if not iterable(qs):
            raise TypeError('Generalized coordinates must be an iterable')
        self._q = Matrix(qs)
        self._qdots = self.q.diff(dynamicsymbols._t)
        self._qdoubledots = self._qdots.diff(dynamicsymbols._t)
        _validate_coordinates(self.q)

        mat_build = lambda x: Matrix(x) if x else Matrix()
        hol_coneqs = mat_build(hol_coneqs)
        nonhol_coneqs = mat_build(nonhol_coneqs)
        self.coneqs = Matrix([hol_coneqs.diff(dynamicsymbols._t),
                nonhol_coneqs])
        self._hol_coneqs = hol_coneqs

    def form_lagranges_equations(self):
        """Method to form Lagrange's equations of motion.

        Returns a vector of equations of motion using Lagrange's equations of
        the second kind.
        """

        qds = self._qdots
        qdd_zero = dict.fromkeys(self._qdoubledots, 0)
        n = len(self.q)

        # Internally we represent the EOM as four terms:
        # EOM = term1 - term2 - term3 - term4 = 0

        # First term
        self._term1 = self._L.jacobian(qds)
        self._term1 = self._term1.diff(dynamicsymbols._t).T

        # Second term
        self._term2 = self._L.jacobian(self.q).T

        # Third term
        if self.coneqs:
            coneqs = self.coneqs
            m = len(coneqs)
            # Creating the multipliers
            self.lam_vec = Matrix(dynamicsymbols('lam1:' + str(m + 1)))
            self.lam_coeffs = -coneqs.jacobian(qds)
            self._term3 = self.lam_coeffs.T * self.lam_vec
            # Extracting the coeffecients of the qdds from the diff coneqs
            diffconeqs = coneqs.diff(dynamicsymbols._t)
            self._m_cd = diffconeqs.jacobian(self._qdoubledots)
            # The remaining terms i.e. the 'forcing' terms in diff coneqs
            self._f_cd = -diffconeqs.subs(qdd_zero)
        else:
            self._term3 = zeros(n, 1)

        # Fourth term
        if self.forcelist:
            N = self.inertial
            self._term4 = zeros(n, 1)
            for i, qd in enumerate(qds):
                flist = zip(*_f_list_parser(self.forcelist, N))
                self._term4[i] = sum(v.diff(qd, N).dot(f) for (v, f) in flist)
        else:
            self._term4 = zeros(n, 1)

        # Form the dynamic mass and forcing matrices
        without_lam = self._term1 - self._term2 - self._term4
        self._m_d = without_lam.jacobian(self._qdoubledots)
        self._f_d = -without_lam.subs(qdd_zero)

        # Form the EOM
        self.eom = without_lam - self._term3
        return self.eom

    def _form_eoms(self):
        return self.form_lagranges_equations()

    @property
    def mass_matrix(self):
        """Returns the mass matrix, which is augmented by the Lagrange
        multipliers, if necessary.

        Explanation
        ===========

        If the system is described by 'n' generalized coordinates and there are
        no constraint equations then an n X n matrix is returned.

        If there are 'n' generalized coordinates and 'm' constraint equations
        have been supplied during initialization then an n X (n+m) matrix is
        returned. The (n + m - 1)th and (n + m)th columns contain the
        coefficients of the Lagrange multipliers.
        """

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        if self.coneqs:
            return (self._m_d).row_join(self.lam_coeffs.T)
        else:
            return self._m_d

    @property
    def mass_matrix_full(self):
        """Augments the coefficients of qdots to the mass_matrix."""

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        n = len(self.q)
        m = len(self.coneqs)
        row1 = eye(n).row_join(zeros(n, n + m))
        row2 = zeros(n, n).row_join(self.mass_matrix)
        if self.coneqs:
            row3 = zeros(m, n).row_join(self._m_cd).row_join(zeros(m, m))
            return row1.col_join(row2).col_join(row3)
        else:
            return row1.col_join(row2)

    @property
    def forcing(self):
        """Returns the forcing vector from 'lagranges_equations' method."""

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        return self._f_d

    @property
    def forcing_full(self):
        """Augments qdots to the forcing vector above."""

        if self.eom is None:
            raise ValueError('Need to compute the equations of motion first')
        if self.coneqs:
            return self._qdots.col_join(self.forcing).col_join(self._f_cd)
        else:
            return self._qdots.col_join(self.forcing)

    def to_linearizer(self, q_ind=None, qd_ind=None, q_dep=None, qd_dep=None,
                      linear_solver='LU'):
        """Returns an instance of the Linearizer class, initiated from the data
        in the LagrangesMethod class. This may be more desirable than using the
        linearize class method, as the Linearizer object will allow more
        efficient recalculation (i.e. about varying operating points).

        Parameters
        ==========

        q_ind, qd_ind : array_like, optional
            The independent generalized coordinates and speeds.
        q_dep, qd_dep : array_like, optional
            The dependent generalized coordinates and speeds.
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.

        Returns
        =======
        Linearizer
            An instantiated
            :class:`sympy.physics.mechanics.linearize.Linearizer`.

        """

        # Compose vectors
        t = dynamicsymbols._t
        q = self.q
        u = self._qdots
        ud = u.diff(t)
        # Get vector of lagrange multipliers
        lams = self.lam_vec

        mat_build = lambda x: Matrix(x) if x else Matrix()
        q_i = mat_build(q_ind)
        q_d = mat_build(q_dep)
        u_i = mat_build(qd_ind)
        u_d = mat_build(qd_dep)

        # Compose general form equations
        f_c = self._hol_coneqs
        f_v = self.coneqs
        f_a = f_v.diff(t)
        f_0 = u
        f_1 = -u
        f_2 = self._term1
        f_3 = -(self._term2 + self._term4)
        f_4 = -self._term3

        # Check that there are an appropriate number of independent and
        # dependent coordinates
        if len(q_d) != len(f_c) or len(u_d) != len(f_v):
            raise ValueError(("Must supply {:} dependent coordinates, and " +
                    "{:} dependent speeds").format(len(f_c), len(f_v)))
        if set(Matrix([q_i, q_d])) != set(q):
            raise ValueError("Must partition q into q_ind and q_dep, with " +
                    "no extra or missing symbols.")
        if set(Matrix([u_i, u_d])) != set(u):
            raise ValueError("Must partition qd into qd_ind and qd_dep, " +
                    "with no extra or missing symbols.")

        # Find all other dynamic symbols, forming the forcing vector r.
        # Sort r to make it canonical.
        insyms = set(Matrix([q, u, ud, lams]))
        r = list(find_dynamicsymbols(f_3, insyms))
        r.sort(key=default_sort_key)
        # Check for any derivatives of variables in r that are also found in r.
        for i in r:
            if diff(i, dynamicsymbols._t) in r:
                raise ValueError('Cannot have derivatives of specified \
                                 quantities when linearizing forcing terms.')

        return Linearizer(f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i,
                          q_d, u_i, u_d, r, lams, linear_solver=linear_solver)

    def linearize(self, q_ind=None, qd_ind=None, q_dep=None, qd_dep=None,
                  linear_solver='LU', **kwargs):
        """Linearize the equations of motion about a symbolic operating point.

        Parameters
        ==========
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.
        **kwargs
            Extra keyword arguments are passed to
            :meth:`sympy.physics.mechanics.linearize.Linearizer.linearize`.

        Explanation
        ===========

        If kwarg A_and_B is False (default), returns M, A, B, r for the
        linearized form, M*[q', u']^T = A*[q_ind, u_ind]^T + B*r.

        If kwarg A_and_B is True, returns A, B, r for the linearized form
        dx = A*x + B*r, where x = [q_ind, u_ind]^T. Note that this is
        computationally intensive if there are many symbolic parameters. For
        this reason, it may be more desirable to use the default A_and_B=False,
        returning M, A, and B. Values may then be substituted in to these
        matrices, and the state space form found as
        A = P.T*M.inv()*A, B = P.T*M.inv()*B, where P = Linearizer.perm_mat.

        In both cases, r is found as all dynamicsymbols in the equations of
        motion that are not part of q, u, q', or u'. They are sorted in
        canonical form.

        The operating points may be also entered using the ``op_point`` kwarg.
        This takes a dictionary of {symbol: value}, or a an iterable of such
        dictionaries. The values may be numeric or symbolic. The more values
        you can specify beforehand, the faster this computation will run.

        For more documentation, please see the ``Linearizer`` class."""

        linearizer = self.to_linearizer(q_ind, qd_ind, q_dep, qd_dep,
                                        linear_solver=linear_solver)
        result = linearizer.linearize(**kwargs)
        return result + (linearizer.r,)

    def solve_multipliers(self, op_point=None, sol_type='dict'):
        """Solves for the values of the lagrange multipliers symbolically at
        the specified operating point.

        Parameters
        ==========

        op_point : dict or iterable of dicts, optional
            Point at which to solve at. The operating point is specified as
            a dictionary or iterable of dictionaries of {symbol: value}. The
            value may be numeric or symbolic itself.

        sol_type : str, optional
            Solution return type. Valid options are:
            - 'dict': A dict of {symbol : value} (default)
            - 'Matrix': An ordered column matrix of the solution
        """

        # Determine number of multipliers
        k = len(self.lam_vec)
        if k == 0:
            raise ValueError("System has no lagrange multipliers to solve for.")
        # Compose dict of operating conditions
        if isinstance(op_point, dict):
            op_point_dict = op_point
        elif iterable(op_point):
            op_point_dict = {}
            for op in op_point:
                op_point_dict.update(op)
        elif op_point is None:
            op_point_dict = {}
        else:
            raise TypeError("op_point must be either a dictionary or an "
                            "iterable of dictionaries.")
        # Compose the system to be solved
        mass_matrix = self.mass_matrix.col_join(-self.lam_coeffs.row_join(
                zeros(k, k)))
        force_matrix = self.forcing.col_join(self._f_cd)
        # Sub in the operating point
        mass_matrix = msubs(mass_matrix, op_point_dict)
        force_matrix = msubs(force_matrix, op_point_dict)
        # Solve for the multipliers
        sol_list = mass_matrix.LUsolve(-force_matrix)[-k:]
        if sol_type == 'dict':
            return dict(zip(self.lam_vec, sol_list))
        elif sol_type == 'Matrix':
            return Matrix(sol_list)
        else:
            raise ValueError("Unknown sol_type {:}.".format(sol_type))

    def rhs(self, inv_method=None, **kwargs):
        """Returns equations that can be solved numerically.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`
        """

        if inv_method is None:
            self._rhs = self.mass_matrix_full.LUsolve(self.forcing_full)
        else:
            self._rhs = (self.mass_matrix_full.inv(inv_method,
                         try_block_diag=True) * self.forcing_full)
        return self._rhs

    @property
    def q(self):
        return self._q

    @property
    def u(self):
        return self._qdots

    @property
    def bodies(self):
        return self._bodies

    @property
    def forcelist(self):
        return self._forcelist

    @property
    def loads(self):
        return self._forcelist
<|endoftext|>from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
                                    RigidBody, Particle)
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.method import _Methods
from sympy import Matrix
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['JointsMethod']


class JointsMethod(_Methods):
    """Method for formulating the equations of motion using a set of interconnected bodies with joints.

    .. deprecated:: 1.13
        The JointsMethod class is deprecated. Its functionality has been
        replaced by the new :class:`~.System` class.

    Parameters
    ==========

    newtonion : Body or ReferenceFrame
        The newtonion(inertial) frame.
    *joints : Joint
        The joints in the system

    Attributes
    ==========

    q, u : iterable
        Iterable of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Body objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    mass_matrix : Matrix, shape(n, n)
        The system's mass matrix
    forcing : Matrix, shape(n, 1)
        The system's forcing vector
    mass_matrix_full : Matrix, shape(2*n, 2*n)
        The "mass matrix" for the u's and q's
    forcing_full : Matrix, shape(2*n, 1)
        The "forcing vector" for the u's and q's
    method : KanesMethod or Lagrange's method
        Method's object.
    kdes : iterable
        Iterable of kde in they system.

    Examples
    ========

    As Body and JointsMethod have been deprecated, the following examples are
    for illustrative purposes only. The functionality of Body is fully captured
    by :class:`~.RigidBody` and :class:`~.Particle` and the functionality of
    JointsMethod is fully captured by :class:`~.System`. To ignore the
    deprecation warning we can use the ignore_warnings context manager.

    >>> from sympy.utilities.exceptions import ignore_warnings

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Body, JointsMethod, PrismaticJoint
    >>> from sympy.physics.vector import dynamicsymbols
    >>> c, k = symbols('c k')
    >>> x, v = dynamicsymbols('x v')
    >>> with ignore_warnings(DeprecationWarning):
    ...     wall = Body('W')
    ...     body = Body('B')
    >>> J = PrismaticJoint('J', wall, body, coordinates=x, speeds=v)
    >>> wall.apply_force(c*v*wall.x, reaction_body=body)
    >>> wall.apply_force(k*x*wall.x, reaction_body=body)
    >>> with ignore_warnings(DeprecationWarning):
    ...     method = JointsMethod(wall, J)
    >>> method.form_eoms()
    Matrix([[-B_mass*Derivative(v(t), t) - c*v(t) - k*x(t)]])
    >>> M = method.mass_matrix_full
    >>> F = method.forcing_full
    >>> rhs = M.LUsolve(F)
    >>> rhs
    Matrix([
    [                     v(t)],
    [(-c*v(t) - k*x(t))/B_mass]])

    Notes
    =====

    ``JointsMethod`` currently only works with systems that do not have any
    configuration or motion constraints.

    """

    def __init__(self, newtonion, *joints):
        sympy_deprecation_warning(
            """
            The JointsMethod class is deprecated.
            Its functionality has been replaced by the new System class.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="deprecated-mechanics-jointsmethod"
        )
        if isinstance(newtonion, BodyBase):
            self.frame = newtonion.frame
        else:
            self.frame = newtonion

        self._joints = joints
        self._bodies = self._generate_bodylist()
        self._loads = self._generate_loadlist()
        self._q = self._generate_q()
        self._u = self._generate_u()
        self._kdes = self._generate_kdes()

        self._method = None

    @property
    def bodies(self):
        """List of bodies in they system."""
        return self._bodies

    @property
    def loads(self):
        """List of loads on the system."""
        return self._loads

    @property
    def q(self):
        """List of the generalized coordinates."""
        return self._q

    @property
    def u(self):
        """List of the generalized speeds."""
        return self._u

    @property
    def kdes(self):
        """List of the generalized coordinates."""
        return self._kdes

    @property
    def forcing_full(self):
        """The "forcing vector" for the u's and q's."""
        return self.method.forcing_full

    @property
    def mass_matrix_full(self):
        """The "mass matrix" for the u's and q's."""
        return self.method.mass_matrix_full

    @property
    def mass_matrix(self):
        """The system's mass matrix."""
        return self.method.mass_matrix

    @property
    def forcing(self):
        """The system's forcing vector."""
        return self.method.forcing

    @property
    def method(self):
        """Object of method used to form equations of systems."""
        return self._method

    def _generate_bodylist(self):
        bodies = []
        for joint in self._joints:
            if joint.child not in bodies:
                bodies.append(joint.child)
            if joint.parent not in bodies:
                bodies.append(joint.parent)
        return bodies

    def _generate_loadlist(self):
        load_list = []
        for body in self.bodies:
            if isinstance(body, Body):
                load_list.extend(body.loads)
        return load_list

    def _generate_q(self):
        q_ind = []
        for joint in self._joints:
            for coordinate in joint.coordinates:
                if coordinate in q_ind:
                    raise ValueError('Coordinates of joints should be unique.')
                q_ind.append(coordinate)
        return Matrix(q_ind)

    def _generate_u(self):
        u_ind = []
        for joint in self._joints:
            for speed in joint.speeds:
                if speed in u_ind:
                    raise ValueError('Speeds of joints should be unique.')
                u_ind.append(speed)
        return Matrix(u_ind)

    def _generate_kdes(self):
        kd_ind = Matrix(1, 0, []).T
        for joint in self._joints:
            kd_ind = kd_ind.col_join(joint.kdes)
        return kd_ind

    def _convert_bodies(self):
        # Convert `Body` to `Particle` and `RigidBody`
        bodylist = []
        for body in self.bodies:
            if not isinstance(body, Body):
                bodylist.append(body)
                continue
            if body.is_rigidbody:
                rb = RigidBody(body.name, body.masscenter, body.frame, body.mass,
                    (body.central_inertia, body.masscenter))
                rb.potential_energy = body.potential_energy
                bodylist.append(rb)
            else:
                part = Particle(body.name, body.masscenter, body.mass)
                part.potential_energy = body.potential_energy
                bodylist.append(part)
        return bodylist

    def form_eoms(self, method=KanesMethod):
        """Method to form system's equation of motions.

        Parameters
        ==========

        method : Class
            Class name of method.

        Returns
        ========

        Matrix
            Vector of equations of motions.

        Examples
        ========

        As Body and JointsMethod have been deprecated, the following examples
        are for illustrative purposes only. The functionality of Body is fully
        captured by :class:`~.RigidBody` and :class:`~.Particle` and the
        functionality of JointsMethod is fully captured by :class:`~.System`. To
        ignore the deprecation warning we can use the ignore_warnings context
        manager.

        >>> from sympy.utilities.exceptions import ignore_warnings

        This is a simple example for a one degree of freedom translational
        spring-mass-damper.

        >>> from sympy import S, symbols
        >>> from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols, Body
        >>> from sympy.physics.mechanics import PrismaticJoint, JointsMethod
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> with ignore_warnings(DeprecationWarning):
        ...     wall = Body('W')
        ...     part = Body('P', mass=m)
        >>> part.potential_energy = k * q**2 / S(2)
        >>> J = PrismaticJoint('J', wall, part, coordinates=q, speeds=qd)
        >>> wall.apply_force(b * qd * wall.x, reaction_body=part)
        >>> with ignore_warnings(DeprecationWarning):
        ...     method = JointsMethod(wall, J)
        >>> method.form_eoms(LagrangesMethod)
        Matrix([[b*Derivative(q(t), t) + k*q(t) + m*Derivative(q(t), (t, 2))]])

        We can also solve for the states using the 'rhs' method.

        >>> method.rhs()
        Matrix([
        [                Derivative(q(t), t)],
        [(-b*Derivative(q(t), t) - k*q(t))/m]])

        """

        bodylist = self._convert_bodies()
        if issubclass(method, LagrangesMethod): #LagrangesMethod or similar
            L = Lagrangian(self.frame, *bodylist)
            self._method = method(L, self.q, self.loads, bodylist, self.frame)
        else: #KanesMethod or similar
            self._method = method(self.frame, q_ind=self.q, u_ind=self.u, kd_eqs=self.kdes,
                                    forcelist=self.loads, bodies=bodylist)
        soln = self.method._form_eoms()
        return soln

    def rhs(self, inv_method=None):
        """Returns equations that can be solved numerically.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        Returns
        ========

        Matrix
            Numerically solvable equations.

        See Also
        ========

        sympy.physics.mechanics.kane.KanesMethod.rhs:
            KanesMethod's rhs function.
        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:
            LagrangesMethod's rhs function.

        """

        return self.method.rhs(inv_method=inv_method)
<|endoftext|>from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
                                  Point, dynamicsymbols)
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
                                           init_vprinting)
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy import Matrix, Mul, Derivative, sin, cos, tan, S
from sympy.core.function import AppliedUndef
from sympy.physics.mechanics.inertia import (inertia as _inertia,
    inertia_of_point_mass as _inertia_of_point_mass)
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['linear_momentum',
           'angular_momentum',
           'kinetic_energy',
           'potential_energy',
           'Lagrangian',
           'mechanics_printing',
           'mprint',
           'msprint',
           'mpprint',
           'mlatex',
           'msubs',
           'find_dynamicsymbols']

# These are functions that we've moved and renamed during extracting the
# basic vector calculus code from the mechanics packages.

mprint = vprint
msprint = vsprint
mpprint = vpprint
mlatex = vlatex


def mechanics_printing(**kwargs):
    """
    Initializes time derivative printing for all SymPy objects in
    mechanics module.
    """

    init_vprinting(**kwargs)

mechanics_printing.__doc__ = init_vprinting.__doc__


def inertia(frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0):
    sympy_deprecation_warning(
        """
        The inertia function has been moved.
        Import it from "sympy.physics.mechanics".
        """,
        deprecated_since_version="1.13",
        active_deprecations_target="moved-mechanics-functions"
    )
    return _inertia(frame, ixx, iyy, izz, ixy, iyz, izx)


def inertia_of_point_mass(mass, pos_vec, frame):
    sympy_deprecation_warning(
        """
        The inertia_of_point_mass function has been moved.
        Import it from "sympy.physics.mechanics".
        """,
        deprecated_since_version="1.13",
        active_deprecations_target="moved-mechanics-functions"
    )
    return _inertia_of_point_mass(mass, pos_vec, frame)


def linear_momentum(frame, *body):
    """Linear momentum of the system.

    Explanation
    ===========

    This function returns the linear momentum of a system of Particle's and/or
    RigidBody's. The linear momentum of a system is equal to the vector sum of
    the linear momentum of its constituents. Consider a system, S, comprised of
    a rigid body, A, and a particle, P. The linear momentum of the system, L,
    is equal to the vector sum of the linear momentum of the particle, L1, and
    the linear momentum of the rigid body, L2, i.e.

    L = L1 + L2

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which linear momentum is desired.
    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose linear momentum is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, linear_momentum
    >>> N = ReferenceFrame('N')
    >>> P = Point('P')
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = Point('Ac')
    >>> Ac.set_vel(N, 25 * N.y)
    >>> I = outer(N.x, N.x)
    >>> A = RigidBody('A', Ac, N, 20, (I, Ac))
    >>> linear_momentum(N, A, Pa)
    10*N.x + 500*N.y

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please specify a valid ReferenceFrame')
    else:
        linear_momentum_sys = Vector(0)
        for e in body:
            if isinstance(e, (RigidBody, Particle)):
                linear_momentum_sys += e.linear_momentum(frame)
            else:
                raise TypeError('*body must have only Particle or RigidBody')
    return linear_momentum_sys


def angular_momentum(point, frame, *body):
    """Angular momentum of a system.

    Explanation
    ===========

    This function returns the angular momentum of a system of Particle's and/or
    RigidBody's. The angular momentum of such a system is equal to the vector
    sum of the angular momentum of its constituents. Consider a system, S,
    comprised of a rigid body, A, and a particle, P. The angular momentum of
    the system, H, is equal to the vector sum of the angular momentum of the
    particle, H1, and the angular momentum of the rigid body, H2, i.e.

    H = H1 + H2

    Parameters
    ==========

    point : Point
        The point about which angular momentum of the system is desired.
    frame : ReferenceFrame
        The frame in which angular momentum is desired.
    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose angular momentum is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, angular_momentum
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> angular_momentum(O, N, Pa, A)
    10*N.z

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please enter a valid ReferenceFrame')
    if not isinstance(point, Point):
        raise TypeError('Please specify a valid Point')
    else:
        angular_momentum_sys = Vector(0)
        for e in body:
            if isinstance(e, (RigidBody, Particle)):
                angular_momentum_sys += e.angular_momentum(point, frame)
            else:
                raise TypeError('*body must have only Particle or RigidBody')
    return angular_momentum_sys


def kinetic_energy(frame, *body):
    """Kinetic energy of a multibody system.

    Explanation
    ===========

    This function returns the kinetic energy of a system of Particle's and/or
    RigidBody's. The kinetic energy of such a system is equal to the sum of
    the kinetic energies of its constituents. Consider a system, S, comprising
    a rigid body, A, and a particle, P. The kinetic energy of the system, T,
    is equal to the vector sum of the kinetic energy of the particle, T1, and
    the kinetic energy of the rigid body, T2, i.e.

    T = T1 + T2

    Kinetic energy is a scalar.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which the velocity or angular velocity of the body is
        defined.
    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose kinetic energy is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, kinetic_energy
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> kinetic_energy(N, Pa, A)
    350

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please enter a valid ReferenceFrame')
    ke_sys = S.Zero
    for e in body:
        if isinstance(e, (RigidBody, Particle)):
            ke_sys += e.kinetic_energy(frame)
        else:
            raise TypeError('*body must have only Particle or RigidBody')
    return ke_sys


def potential_energy(*body):
    """Potential energy of a multibody system.

    Explanation
    ===========

    This function returns the potential energy of a system of Particle's and/or
    RigidBody's. The potential energy of such a system is equal to the sum of
    the potential energy of its constituents. Consider a system, S, comprising
    a rigid body, A, and a particle, P. The potential energy of the system, V,
    is equal to the vector sum of the potential energy of the particle, V1, and
    the potential energy of the rigid body, V2, i.e.

    V = V1 + V2

    Potential energy is a scalar.

    Parameters
    ==========

    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose potential energy is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, potential_energy
    >>> from sympy import symbols
    >>> M, m, g, h = symbols('M m g h')
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> Pa = Particle('Pa', P, m)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> a = ReferenceFrame('a')
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, M, (I, Ac))
    >>> Pa.potential_energy = m * g * h
    >>> A.potential_energy = M * g * h
    >>> potential_energy(Pa, A)
    M*g*h + g*h*m

    """

    pe_sys = S.Zero
    for e in body:
        if isinstance(e, (RigidBody, Particle)):
            pe_sys += e.potential_energy
        else:
            raise TypeError('*body must have only Particle or RigidBody')
    return pe_sys


def gravity(acceleration, *bodies):
    from sympy.physics.mechanics.loads import gravity as _gravity
    sympy_deprecation_warning(
        """
        The gravity function has been moved.
        Import it from "sympy.physics.mechanics.loads".
        """,
        deprecated_since_version="1.13",
        active_deprecations_target="moved-mechanics-functions"
    )
    return _gravity(acceleration, *bodies)


def center_of_mass(point, *bodies):
    """
    Returns the position vector from the given point to the center of mass
    of the given bodies(particles or rigidbodies).

    Example
    =======

    >>> from sympy import symbols, S
    >>> from sympy.physics.vector import Point
    >>> from sympy.physics.mechanics import Particle, ReferenceFrame, RigidBody, outer
    >>> from sympy.physics.mechanics.functions import center_of_mass
    >>> a = ReferenceFrame('a')
    >>> m = symbols('m', real=True)
    >>> p1 = Particle('p1', Point('p1_pt'), S(1))
    >>> p2 = Particle('p2', Point('p2_pt'), S(2))
    >>> p3 = Particle('p3', Point('p3_pt'), S(3))
    >>> p4 = Particle('p4', Point('p4_pt'), m)
    >>> b_f = ReferenceFrame('b_f')
    >>> b_cm = Point('b_cm')
    >>> mb = symbols('mb')
    >>> b = RigidBody('b', b_cm, b_f, mb, (outer(b_f.x, b_f.x), b_cm))
    >>> p2.point.set_pos(p1.point, a.x)
    >>> p3.point.set_pos(p1.point, a.x + a.y)
    >>> p4.point.set_pos(p1.point, a.y)
    >>> b.masscenter.set_pos(p1.point, a.y + a.z)
    >>> point_o=Point('o')
    >>> point_o.set_pos(p1.point, center_of_mass(p1.point, p1, p2, p3, p4, b))
    >>> expr = 5/(m + mb + 6)*a.x + (m + mb + 3)/(m + mb + 6)*a.y + mb/(m + mb + 6)*a.z
    >>> point_o.pos_from(p1.point)
    5/(m + mb + 6)*a.x + (m + mb + 3)/(m + mb + 6)*a.y + mb/(m + mb + 6)*a.z

    """
    if not bodies:
        raise TypeError("No bodies(instances of Particle or Rigidbody) were passed.")

    total_mass = 0
    vec = Vector(0)
    for i in bodies:
        total_mass += i.mass

        masscenter = getattr(i, 'masscenter', None)
        if masscenter is None:
            masscenter = i.point
        vec += i.mass*masscenter.pos_from(point)

    return vec/total_mass


def Lagrangian(frame, *body):
    """Lagrangian of a multibody system.

    Explanation
    ===========

    This function returns the Lagrangian of a system of Particle's and/or
    RigidBody's. The Lagrangian of such a system is equal to the difference
    between the kinetic energies and potential energies of its constituents. If
    T and V are the kinetic and potential energies of a system then it's
    Lagrangian, L, is defined as

    L = T - V

    The Lagrangian is a scalar.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which the velocity or angular velocity of the body is
        defined to determine the kinetic energy.

    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose Lagrangian is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, Lagrangian
    >>> from sympy import symbols
    >>> M, m, g, h = symbols('M m g h')
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> Pa.potential_energy = m * g * h
    >>> A.potential_energy = M * g * h
    >>> Lagrangian(N, Pa, A)
    -M*g*h - g*h*m + 350

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please supply a valid ReferenceFrame')
    for e in body:
        if not isinstance(e, (RigidBody, Particle)):
            raise TypeError('*body must have only Particle or RigidBody')
    return kinetic_energy(frame, *body) - potential_energy(*body)


def find_dynamicsymbols(expression, exclude=None, reference_frame=None):
    """Find all dynamicsymbols in expression.

    Explanation
    ===========

    If the optional ``exclude`` kwarg is used, only dynamicsymbols
    not in the iterable ``exclude`` are returned.
    If we intend to apply this function on a vector, the optional
    ``reference_frame`` is also used to inform about the corresponding frame
    with respect to which the dynamic symbols of the given vector is to be
    determined.

    Parameters
    ==========

    expression : SymPy expression

    exclude : iterable of dynamicsymbols, optional

    reference_frame : ReferenceFrame, optional
        The frame with respect to which the dynamic symbols of the
        given vector is to be determined.

    Examples
    ========

    >>> from sympy.physics.mechanics import dynamicsymbols, find_dynamicsymbols
    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> x, y = dynamicsymbols('x, y')
    >>> expr = x + x.diff()*y
    >>> find_dynamicsymbols(expr)
    {x(t), y(t), Derivative(x(t), t)}
    >>> find_dynamicsymbols(expr, exclude=[x, y])
    {Derivative(x(t), t)}
    >>> a, b, c = dynamicsymbols('a, b, c')
    >>> A = ReferenceFrame('A')
    >>> v = a * A.x + b * A.y + c * A.z
    >>> find_dynamicsymbols(v, reference_frame=A)
    {a(t), b(t), c(t)}

    """
    t_set = {dynamicsymbols._t}
    if exclude:
        if iterable(exclude):
            exclude_set = set(exclude)
        else:
            raise TypeError("exclude kwarg must be iterable")
    else:
        exclude_set = set()
    if isinstance(expression, Vector):
        if reference_frame is None:
            raise ValueError("You must provide reference_frame when passing a "
                             "vector expression, got %s." % reference_frame)
        else:
            expression = expression.to_matrix(reference_frame)
    return {i for i in expression.atoms(AppliedUndef, Derivative) if
            i.free_symbols == t_set} - exclude_set


def msubs(expr, *sub_dicts, smart=False, **kwargs):
    """A custom subs for use on expressions derived in physics.mechanics.

    Traverses the expression tree once, performing the subs found in sub_dicts.
    Terms inside ``Derivative`` expressions are ignored:

    Examples
    ========

    >>> from sympy.physics.mechanics import dynamicsymbols, msubs
    >>> x = dynamicsymbols('x')
    >>> msubs(x.diff() + x, {x: 1})
    Derivative(x(t), t) + 1

    Note that sub_dicts can be a single dictionary, or several dictionaries:

    >>> x, y, z = dynamicsymbols('x, y, z')
    >>> sub1 = {x: 1, y: 2}
    >>> sub2 = {z: 3, x.diff(): 4}
    >>> msubs(x.diff() + x + y + z, sub1, sub2)
    10

    If smart=True (default False), also checks for conditions that may result
    in ``nan``, but if simplified would yield a valid expression. For example:

    >>> from sympy import sin, tan
    >>> (sin(x)/tan(x)).subs(x, 0)
    nan
    >>> msubs(sin(x)/tan(x), {x: 0}, smart=True)
    1

    It does this by first replacing all ``tan`` with ``sin/cos``. Then each
    node is traversed. If the node is a fraction, subs is first evaluated on
    the denominator. If this results in 0, simplification of the entire
    fraction is attempted. Using this selective simplification, only
    subexpressions that result in 1/0 are targeted, resulting in faster
    performance.

    """

    sub_dict = dict_merge(*sub_dicts)
    if smart:
        func = _smart_subs
    elif hasattr(expr, 'msubs'):
        return expr.msubs(sub_dict)
    else:
        func = lambda expr, sub_dict: _crawl(expr, _sub_func, sub_dict)
    if isinstance(expr, (Matrix, Vector, Dyadic)):
        return expr.applyfunc(lambda x: func(x, sub_dict))
    else:
        return func(expr, sub_dict)


def _crawl(expr, func, *args, **kwargs):
    """Crawl the expression tree, and apply func to every node."""
    val = func(expr, *args, **kwargs)
    if val is not None:
        return val
    new_args = (_crawl(arg, func, *args, **kwargs) for arg in expr.args)
    return expr.func(*new_args)


def _sub_func(expr, sub_dict):
    """Perform direct matching substitution, ignoring derivatives."""
    if expr in sub_dict:
        return sub_dict[expr]
    elif not expr.args or expr.is_Derivative:
        return expr


def _tan_repl_func(expr):
    """Replace tan with sin/cos."""
    if isinstance(expr, tan):
        return sin(*expr.args) / cos(*expr.args)
    elif not expr.args or expr.is_Derivative:
        return expr


def _smart_subs(expr, sub_dict):
    """Performs subs, checking for conditions that may result in `nan` or
    `oo`, and attempts to simplify them out.

    The expression tree is traversed twice, and the following steps are
    performed on each expression node:
    - First traverse:
        Replace all `tan` with `sin/cos`.
    - Second traverse:
        If node is a fraction, check if the denominator evaluates to 0.
        If so, attempt to simplify it out. Then if node is in sub_dict,
        sub in the corresponding value.

    """
    expr = _crawl(expr, _tan_repl_func)

    def _recurser(expr, sub_dict):
        # Decompose the expression into num, den
        num, den = _fraction_decomp(expr)
        if den != 1:
            # If there is a non trivial denominator, we need to handle it
            denom_subbed = _recurser(den, sub_dict)
            if denom_subbed.evalf() == 0:
                # If denom is 0 after this, attempt to simplify the bad expr
                expr = simplify(expr)
            else:
                # Expression won't result in nan, find numerator
                num_subbed = _recurser(num, sub_dict)
                return num_subbed / denom_subbed
        # We have to crawl the tree manually, because `expr` may have been
        # modified in the simplify step. First, perform subs as normal:
        val = _sub_func(expr, sub_dict)
        if val is not None:
            return val
        new_args = (_recurser(arg, sub_dict) for arg in expr.args)
        return expr.func(*new_args)
    return _recurser(expr, sub_dict)


def _fraction_decomp(expr):
    """Return num, den such that expr = num/den."""
    if not isinstance(expr, Mul):
        return expr, 1
    num = []
    den = []
    for a in expr.args:
        if a.is_Pow and a.args[1] < 0:
            den.append(1 / a)
        else:
            num.append(a)
    if not den:
        return expr, 1
    num = Mul(*num)
    den = Mul(*den)
    return num, den


def _f_list_parser(fl, ref_frame):
    """Parses the provided forcelist composed of items
    of the form (obj, force).
    Returns a tuple containing:
        vel_list: The velocity (ang_vel for Frames, vel for Points) in
                the provided reference frame.
        f_list: The forces.

    Used internally in the KanesMethod and LagrangesMethod classes.

    """
    def flist_iter():
        for pair in fl:
            obj, force = pair
            if isinstance(obj, ReferenceFrame):
                yield obj.ang_vel_in(ref_frame), force
            elif isinstance(obj, Point):
                yield obj.vel(ref_frame), force
            else:
                raise TypeError('First entry in each forcelist pair must '
                                'be a point or frame.')

    if not fl:
        vel_list, f_list = (), ()
    else:
        unzip = lambda l: list(zip(*l)) if l[0] else [(), ()]
        vel_list, f_list = unzip(list(flist_iter()))
    return vel_list, f_list


def _validate_coordinates(coordinates=None, speeds=None, check_duplicates=True,
                          is_dynamicsymbols=True, u_auxiliary=None):
    """Validate the generalized coordinates and generalized speeds.

    Parameters
    ==========
    coordinates : iterable, optional
        Generalized coordinates to be validated.
    speeds : iterable, optional
        Generalized speeds to be validated.
    check_duplicates : bool, optional
        Checks if there are duplicates in the generalized coordinates and
        generalized speeds. If so it will raise a ValueError. The default is
        True.
    is_dynamicsymbols : iterable, optional
        Checks if all the generalized coordinates and generalized speeds are
        dynamicsymbols. If any is not a dynamicsymbol, a ValueError will be
        raised. The default is True.
    u_auxiliary : iterable, optional
        Auxiliary generalized speeds to be validated.

    """
    t_set = {dynamicsymbols._t}
    # Convert input to iterables
    if coordinates is None:
        coordinates = []
    elif not iterable(coordinates):
        coordinates = [coordinates]
    if speeds is None:
        speeds = []
    elif not iterable(speeds):
        speeds = [speeds]
    if u_auxiliary is None:
        u_auxiliary = []
    elif not iterable(u_auxiliary):
        u_auxiliary = [u_auxiliary]

    msgs = []
    if check_duplicates:  # Check for duplicates
        seen = set()
        coord_duplicates = {x for x in coordinates if x in seen or seen.add(x)}
        seen = set()
        speed_duplicates = {x for x in speeds if x in seen or seen.add(x)}
        seen = set()
        aux_duplicates = {x for x in u_auxiliary if x in seen or seen.add(x)}
        overlap_coords = set(coordinates).intersection(speeds)
        overlap_aux = set(coordinates).union(speeds).intersection(u_auxiliary)
        if coord_duplicates:
            msgs.append(f'The generalized coordinates {coord_duplicates} are '
                        f'duplicated, all generalized coordinates should be '
                        f'unique.')
        if speed_duplicates:
            msgs.append(f'The generalized speeds {speed_duplicates} are '
                        f'duplicated, all generalized speeds should be unique.')
        if aux_duplicates:
            msgs.append(f'The auxiliary speeds {aux_duplicates} are duplicated,'
                        f' all auxiliary speeds should be unique.')
        if overlap_coords:
            msgs.append(f'{overlap_coords} are defined as both generalized '
                        f'coordinates and generalized speeds.')
        if overlap_aux:
            msgs.append(f'The auxiliary speeds {overlap_aux} are also defined '
                        f'as generalized coordinates or generalized speeds.')
    if is_dynamicsymbols:  # Check whether all coordinates are dynamicsymbols
        for coordinate in coordinates:
            if not (isinstance(coordinate, (AppliedUndef, Derivative)) and
                    coordinate.free_symbols == t_set):
                msgs.append(f'Generalized coordinate "{coordinate}" is not a '
                            f'dynamicsymbol.')
        for speed in speeds:
            if not (isinstance(speed, (AppliedUndef, Derivative)) and
                    speed.free_symbols == t_set):
                msgs.append(
                    f'Generalized speed "{speed}" is not a dynamicsymbol.')
        for aux in u_auxiliary:
            if not (isinstance(aux, (AppliedUndef, Derivative)) and
                    aux.free_symbols == t_set):
                msgs.append(
                    f'Auxiliary speed "{aux}" is not a dynamicsymbol.')
    if msgs:
        raise ValueError('\n'.join(msgs))


def _parse_linear_solver(linear_solver):
    """Helper function to retrieve a specified linear solver."""
    if callable(linear_solver):
        return linear_solver
    return lambda A, b: Matrix.solve(A, b, method=linear_solver)
<|endoftext|>from collections import deque
from sympy.core.random import randint

from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger

__all__ = [
    # Public interfaces
    'generate_gate_rules',
    'generate_equivalent_ids',
    'GateIdentity',
    'bfs_identity_search',
    'random_identity_search',

    # "Private" functions
    'is_scalar_sparse_matrix',
    'is_scalar_nonsparse_matrix',
    'is_degenerate',
    'is_reducible',
]

np = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def is_scalar_sparse_matrix(circuit, nqubits, identity_only, eps=1e-11):
    """Checks if a given scipy.sparse matrix is a scalar matrix.

    A scalar matrix is such that B = bI, where B is the scalar
