    matrix, b is some scalar multiple, and I is the identity
    matrix.  A scalar matrix would have only the element b along
    it's main diagonal and zeroes elsewhere.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        The tolerance value for zeroing out elements in the matrix.
        Values in the range [-eps, +eps] will be changed to a zero.
    """

    if not np or not scipy:
        pass

    matrix = represent(Mul(*circuit), nqubits=nqubits,
                       format='scipy.sparse')

    # In some cases, represent returns a 1D scalar value in place
    # of a multi-dimensional scalar matrix
    if (isinstance(matrix, int)):
        return matrix == 1 if identity_only else True

    # If represent returns a matrix, check if the matrix is diagonal
    # and if every item along the diagonal is the same
    else:
        # Due to floating pointing operations, must zero out
        # elements that are "very" small in the dense matrix
        # See parameter for default value.

        # Get the ndarray version of the dense matrix
        dense_matrix = matrix.todense().getA()
        # Since complex values can't be compared, must split
        # the matrix into real and imaginary components
        # Find the real values in between -eps and eps
        bool_real = np.logical_and(dense_matrix.real > -eps,
                                   dense_matrix.real < eps)
        # Find the imaginary values between -eps and eps
        bool_imag = np.logical_and(dense_matrix.imag > -eps,
                                   dense_matrix.imag < eps)
        # Replaces values between -eps and eps with 0
        corrected_real = np.where(bool_real, 0.0, dense_matrix.real)
        corrected_imag = np.where(bool_imag, 0.0, dense_matrix.imag)
        # Convert the matrix with real values into imaginary values
        corrected_imag = corrected_imag * complex(1j)
        # Recombine the real and imaginary components
        corrected_dense = corrected_real + corrected_imag

        # Check if it's diagonal
        row_indices = corrected_dense.nonzero()[0]
        col_indices = corrected_dense.nonzero()[1]
        # Check if the rows indices and columns indices are the same
        # If they match, then matrix only contains elements along diagonal
        bool_indices = row_indices == col_indices
        is_diagonal = bool_indices.all()

        first_element = corrected_dense[0][0]
        # If the first element is a zero, then can't rescale matrix
        # and definitely not diagonal
        if (first_element == 0.0 + 0.0j):
            return False

        # The dimensions of the dense matrix should still
        # be 2^nqubits if there are elements all along the
        # the main diagonal
        trace_of_corrected = (corrected_dense/first_element).trace()
        expected_trace = pow(2, nqubits)
        has_correct_trace = trace_of_corrected == expected_trace

        # If only looking for identity matrices
        # first element must be a 1
        real_is_one = abs(first_element.real - 1.0) < eps
        imag_is_zero = abs(first_element.imag) < eps
        is_one = real_is_one and imag_is_zero
        is_identity = is_one if identity_only else True
        return bool(is_diagonal and has_correct_trace and is_identity)


def is_scalar_nonsparse_matrix(circuit, nqubits, identity_only, eps=None):
    """Checks if a given circuit, in matrix form, is equivalent to
    a scalar value.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        This argument is ignored. It is just for signature compatibility with
        is_scalar_sparse_matrix.

    Note: Used in situations when is_scalar_sparse_matrix has bugs
    """

    matrix = represent(Mul(*circuit), nqubits=nqubits)

    # In some cases, represent returns a 1D scalar value in place
    # of a multi-dimensional scalar matrix
    if (isinstance(matrix, Number)):
        return matrix == 1 if identity_only else True

    # If represent returns a matrix, check if the matrix is diagonal
    # and if every item along the diagonal is the same
    else:
        # Added up the diagonal elements
        matrix_trace = matrix.trace()
        # Divide the trace by the first element in the matrix
        # if matrix is not required to be the identity matrix
        adjusted_matrix_trace = (matrix_trace/matrix[0]
                                 if not identity_only
                                 else matrix_trace)

        is_identity = equal_valued(matrix[0], 1) if identity_only else True

        has_correct_trace = adjusted_matrix_trace == pow(2, nqubits)

        # The matrix is scalar if it's diagonal and the adjusted trace
        # value is equal to 2^nqubits
        return bool(
            matrix.is_diagonal() and has_correct_trace and is_identity)

if np and scipy:
    is_scalar_matrix = is_scalar_sparse_matrix
else:
    is_scalar_matrix = is_scalar_nonsparse_matrix


def _get_min_qubits(a_gate):
    if isinstance(a_gate, Pow):
        return a_gate.base.min_qubits
    else:
        return a_gate.min_qubits


def ll_op(left, right):
    """Perform a LL operation.

    A LL operation multiplies both left and right circuits
    with the dagger of the left circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a LL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LL operation:

    >>> from sympy.physics.quantum.identitysearch import ll_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> ll_op((x, y, z), ())
    ((Y(0), Z(0)), (X(0),))

    >>> ll_op((y, z), (x,))
    ((Z(0),), (Y(0), X(0)))
    """

    if (len(left) > 0):
        ll_gate = left[0]
        ll_gate_is_unitary = is_scalar_matrix(
            (Dagger(ll_gate), ll_gate), _get_min_qubits(ll_gate), True)

    if (len(left) > 0 and ll_gate_is_unitary):
        # Get the new left side w/o the leftmost gate
        new_left = left[1:len(left)]
        # Add the leftmost gate to the left position on the right side
        new_right = (Dagger(ll_gate),) + right
        # Return the new gate rule
        return (new_left, new_right)

    return None


def lr_op(left, right):
    """Perform a LR operation.

    A LR operation multiplies both left and right circuits
    with the dagger of the left circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a LR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LR operation:

    >>> from sympy.physics.quantum.identitysearch import lr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> lr_op((x, y, z), ())
    ((X(0), Y(0)), (Z(0),))

    >>> lr_op((x, y), (z,))
    ((X(0),), (Z(0), Y(0)))
    """

    if (len(left) > 0):
        lr_gate = left[len(left) - 1]
        lr_gate_is_unitary = is_scalar_matrix(
            (Dagger(lr_gate), lr_gate), _get_min_qubits(lr_gate), True)

    if (len(left) > 0 and lr_gate_is_unitary):
        # Get the new left side w/o the rightmost gate
        new_left = left[0:len(left) - 1]
        # Add the rightmost gate to the right position on the right side
        new_right = right + (Dagger(lr_gate),)
        # Return the new gate rule
        return (new_left, new_right)

    return None


def rl_op(left, right):
    """Perform a RL operation.

    A RL operation multiplies both left and right circuits
    with the dagger of the right circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a RL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RL operation:

    >>> from sympy.physics.quantum.identitysearch import rl_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rl_op((x,), (y, z))
    ((Y(0), X(0)), (Z(0),))

    >>> rl_op((x, y), (z,))
    ((Z(0), X(0), Y(0)), ())
    """

    if (len(right) > 0):
        rl_gate = right[0]
        rl_gate_is_unitary = is_scalar_matrix(
            (Dagger(rl_gate), rl_gate), _get_min_qubits(rl_gate), True)

    if (len(right) > 0 and rl_gate_is_unitary):
        # Get the new right side w/o the leftmost gate
        new_right = right[1:len(right)]
        # Add the leftmost gate to the left position on the left side
        new_left = (Dagger(rl_gate),) + left
        # Return the new gate rule
        return (new_left, new_right)

    return None


def rr_op(left, right):
    """Perform a RR operation.

    A RR operation multiplies both left and right circuits
    with the dagger of the right circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a RR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RR operation:

    >>> from sympy.physics.quantum.identitysearch import rr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rr_op((x, y), (z,))
    ((X(0), Y(0), Z(0)), ())

    >>> rr_op((x,), (y, z))
    ((X(0), Z(0)), (Y(0),))
    """

    if (len(right) > 0):
        rr_gate = right[len(right) - 1]
        rr_gate_is_unitary = is_scalar_matrix(
            (Dagger(rr_gate), rr_gate), _get_min_qubits(rr_gate), True)

    if (len(right) > 0 and rr_gate_is_unitary):
        # Get the new right side w/o the rightmost gate
        new_right = right[0:len(right) - 1]
        # Add the rightmost gate to the right position on the right side
        new_left = left + (Dagger(rr_gate),)
        # Return the new gate rule
        return (new_left, new_right)

    return None


def generate_gate_rules(gate_seq, return_as_muls=False):
    """Returns a set of gate rules.  Each gate rules is represented
    as a 2-tuple of tuples or Muls.  An empty tuple represents an arbitrary
    scalar value.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules.

    A gate rule is an expression such as ABC = D or AB = CD, where
    A, B, C, and D are gates.  Each value on either side of the
    equal sign represents a circuit.  The four operations allow
    one to find a set of equivalent circuits from a gate identity.
    The letters denoting the operation tell the user what
    activities to perform on each expression.  The first letter
    indicates which side of the equal sign to focus on.  The
    second letter indicates which gate to focus on given the
    side.  Once this information is determined, the inverse
    of the gate is multiplied on both circuits to create a new
    gate rule.

    For example, given the identity, ABCD = 1, a LL operation
    means look at the left value and multiply both left sides by the
    inverse of the leftmost gate A.  If A is Hermitian, the inverse
    of A is still A.  The resulting new rule is BCD = A.

    The following is a summary of the four operations.  Assume
    that in the examples, all gates are Hermitian.

        LL : left circuit, left multiply
             ABCD = E -> AABCD = AE -> BCD = AE
        LR : left circuit, right multiply
             ABCD = E -> ABCDD = ED -> ABC = ED
        RL : right circuit, left multiply
             ABC = ED -> EABC = EED -> EABC = D
        RR : right circuit, right multiply
             AB = CD -> ABD = CDD -> ABD = C

    The number of gate rules generated is n*(n+1), where n
    is the number of gates in the sequence (unproven).

    Parameters
    ==========

    gate_seq : Gate tuple, Mul, or Number
        A variable length tuple or Mul of Gates whose product is equal to
        a scalar matrix
    return_as_muls : bool
        True to return a set of Muls; False to return a set of tuples

    Examples
    ========

    Find the gate rules of the current circuit using tuples:

    >>> from sympy.physics.quantum.identitysearch import generate_gate_rules
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> generate_gate_rules((x, x))
    {((X(0),), (X(0),)), ((X(0), X(0)), ())}

    >>> generate_gate_rules((x, y, z))
    {((), (X(0), Z(0), Y(0))), ((), (Y(0), X(0), Z(0))),
     ((), (Z(0), Y(0), X(0))), ((X(0),), (Z(0), Y(0))),
     ((Y(0),), (X(0), Z(0))), ((Z(0),), (Y(0), X(0))),
     ((X(0), Y(0)), (Z(0),)), ((Y(0), Z(0)), (X(0),)),
     ((Z(0), X(0)), (Y(0),)), ((X(0), Y(0), Z(0)), ()),
     ((Y(0), Z(0), X(0)), ()), ((Z(0), X(0), Y(0)), ())}

    Find the gate rules of the current circuit using Muls:

    >>> generate_gate_rules(x*x, return_as_muls=True)
    {(1, 1)}

    >>> generate_gate_rules(x*y*z, return_as_muls=True)
    {(1, X(0)*Z(0)*Y(0)), (1, Y(0)*X(0)*Z(0)),
     (1, Z(0)*Y(0)*X(0)), (X(0)*Y(0), Z(0)),
     (Y(0)*Z(0), X(0)), (Z(0)*X(0), Y(0)),
     (X(0)*Y(0)*Z(0), 1), (Y(0)*Z(0)*X(0), 1),
     (Z(0)*X(0)*Y(0), 1), (X(0), Z(0)*Y(0)),
     (Y(0), X(0)*Z(0)), (Z(0), Y(0)*X(0))}
    """

    if isinstance(gate_seq, Number):
        if return_as_muls:
            return {(S.One, S.One)}
        else:
            return {((), ())}

    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args

    # Each item in queue is a 3-tuple:
    #     i)   first item is the left side of an equality
    #    ii)   second item is the right side of an equality
    #   iii)   third item is the number of operations performed
    # The argument, gate_seq, will start on the left side, and
    # the right side will be empty, implying the presence of an
    # identity.
    queue = deque()
    # A set of gate rules
    rules = set()
    # Maximum number of operations to perform
    max_ops = len(gate_seq)

    def process_new_rule(new_rule, ops):
        if new_rule is not None:
            new_left, new_right = new_rule

            if new_rule not in rules and (new_right, new_left) not in rules:
                rules.add(new_rule)
            # If haven't reached the max limit on operations
            if ops + 1 < max_ops:
                queue.append(new_rule + (ops + 1,))

    queue.append((gate_seq, (), 0))
    rules.add((gate_seq, ()))

    while len(queue) > 0:
        left, right, ops = queue.popleft()

        # Do a LL
        new_rule = ll_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a LR
        new_rule = lr_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a RL
        new_rule = rl_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a RR
        new_rule = rr_op(left, right)
        process_new_rule(new_rule, ops)

    if return_as_muls:
        # Convert each rule as tuples into a rule as muls
        mul_rules = set()
        for rule in rules:
            left, right = rule
            mul_rules.add((Mul(*left), Mul(*right)))

        rules = mul_rules

    return rules


def generate_equivalent_ids(gate_seq, return_as_muls=False):
    """Returns a set of equivalent gate identities.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules and, subsequently, to locate equivalent
    gate identities.

    Note that all equivalent identities are reachable in n operations
    from the starting gate identity, where n is the number of gates
    in the sequence.

    The max number of gate identities is 2n, where n is the number
    of gates in the sequence (unproven).

    Parameters
    ==========

    gate_seq : Gate tuple, Mul, or Number
        A variable length tuple or Mul of Gates whose product is equal to
        a scalar matrix.
    return_as_muls: bool
        True to return as Muls; False to return as tuples

    Examples
    ========

    Find equivalent gate identities from the current circuit with tuples:

    >>> from sympy.physics.quantum.identitysearch import generate_equivalent_ids
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> generate_equivalent_ids((x, x))
    {(X(0), X(0))}

    >>> generate_equivalent_ids((x, y, z))
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}

    Find equivalent gate identities from the current circuit with Muls:

    >>> generate_equivalent_ids(x*x, return_as_muls=True)
    {1}

    >>> generate_equivalent_ids(x*y*z, return_as_muls=True)
    {X(0)*Y(0)*Z(0), X(0)*Z(0)*Y(0), Y(0)*X(0)*Z(0),
     Y(0)*Z(0)*X(0), Z(0)*X(0)*Y(0), Z(0)*Y(0)*X(0)}
    """

    if isinstance(gate_seq, Number):
        return {S.One}
    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args

    # Filter through the gate rules and keep the rules
    # with an empty tuple either on the left or right side

    # A set of equivalent gate identities
    eq_ids = set()

    gate_rules = generate_gate_rules(gate_seq)
    for rule in gate_rules:
        l, r = rule
        if l == ():
            eq_ids.add(r)
        elif r == ():
            eq_ids.add(l)

    if return_as_muls:
        convert_to_mul = lambda id_seq: Mul(*id_seq)
        eq_ids = set(map(convert_to_mul, eq_ids))

    return eq_ids


class GateIdentity(Basic):
    """Wrapper class for circuits that reduce to a scalar value.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    Parameters
    ==========

    args : Gate tuple
        A variable length tuple of Gates that form an identity.

    Examples
    ========

    Create a GateIdentity and look at its attributes:

    >>> from sympy.physics.quantum.identitysearch import GateIdentity
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> an_identity.circuit
    X(0)*Y(0)*Z(0)

    >>> an_identity.equivalent_ids
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}
    """

    def __new__(cls, *args):
        # args should be a tuple - a variable length argument list
        obj = Basic.__new__(cls, *args)
        obj._circuit = Mul(*args)
        obj._rules = generate_gate_rules(args)
        obj._eq_ids = generate_equivalent_ids(args)

        return obj

    @property
    def circuit(self):
        return self._circuit

    @property
    def gate_rules(self):
        return self._rules

    @property
    def equivalent_ids(self):
        return self._eq_ids

    @property
    def sequence(self):
        return self.args

    def __str__(self):
        """Returns the string of gates in a tuple."""
        return str(self.circuit)


def is_degenerate(identity_set, gate_identity):
    """Checks if a gate identity is a permutation of another identity.

    Parameters
    ==========

    identity_set : set
        A Python set with GateIdentity objects.
    gate_identity : GateIdentity
        The GateIdentity to check for existence in the set.

    Examples
    ========

    Check if the identity is a permutation of another identity:

    >>> from sympy.physics.quantum.identitysearch import (
    ...     GateIdentity, is_degenerate)
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> id_set = {an_identity}
    >>> another_id = (y, z, x)
    >>> is_degenerate(id_set, another_id)
    True

    >>> another_id = (x, x)
    >>> is_degenerate(id_set, another_id)
    False
    """

    # For now, just iteratively go through the set and check if the current
    # gate_identity is a permutation of an identity in the set
    for an_id in identity_set:
        if (gate_identity in an_id.equivalent_ids):
            return True
    return False


def is_reducible(circuit, nqubits, begin, end):
    """Determines if a circuit is reducible by checking
    if its subcircuits are scalar values.

    Parameters
    ==========

    circuit : Gate tuple
        A tuple of Gates representing a circuit.  The circuit to check
        if a gate identity is contained in a subcircuit.
    nqubits : int
        The number of qubits the circuit operates on.
    begin : int
        The leftmost gate in the circuit to include in a subcircuit.
    end : int
        The rightmost gate in the circuit to include in a subcircuit.

    Examples
    ========

    Check if the circuit can be reduced:

    >>> from sympy.physics.quantum.identitysearch import is_reducible
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> is_reducible((x, y, z), 1, 0, 3)
    True

    Check if an interval in the circuit can be reduced:

    >>> is_reducible((x, y, z), 1, 1, 3)
    False

    >>> is_reducible((x, y, y), 1, 1, 3)
    True
    """

    current_circuit = ()
    # Start from the gate at "end" and go down to almost the gate at "begin"
    for ndx in reversed(range(begin, end)):
        next_gate = circuit[ndx]
        current_circuit = (next_gate,) + current_circuit

        # If a circuit as a matrix is equivalent to a scalar value
        if (is_scalar_matrix(current_circuit, nqubits, False)):
            return True

    return False


def bfs_identity_search(gate_list, nqubits, max_depth=None,
       identity_only=False):
    """Constructs a set of gate identities from the list of possible gates.

    Performs a breadth first search over the space of gate identities.
    This allows the finding of the shortest gate identities first.

    Parameters
    ==========

    gate_list : list, Gate
        A list of Gates from which to search for gate identities.
    nqubits : int
        The number of qubits the quantum circuit operates on.
    max_depth : int
        The longest quantum circuit to construct from gate_list.
    identity_only : bool
        True to search for gate identities that reduce to identity;
        False to search for gate identities that reduce to a scalar.

    Examples
    ========

    Find a list of gate identities:

    >>> from sympy.physics.quantum.identitysearch import bfs_identity_search
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> bfs_identity_search([x], 1, max_depth=2)
    {GateIdentity(X(0), X(0))}

    >>> bfs_identity_search([x, y, z], 1)
    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),
     GateIdentity(Z(0), Z(0)), GateIdentity(X(0), Y(0), Z(0))}

    Find a list of identities that only equal to 1:

    >>> bfs_identity_search([x, y, z], 1, identity_only=True)
    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),
     GateIdentity(Z(0), Z(0))}
    """

    if max_depth is None or max_depth <= 0:
        max_depth = len(gate_list)

    id_only = identity_only

    # Start with an empty sequence (implicitly contains an IdentityGate)
    queue = deque([()])

    # Create an empty set of gate identities
    ids = set()

    # Begin searching for gate identities in given space.
    while (len(queue) > 0):
        current_circuit = queue.popleft()

        for next_gate in gate_list:
            new_circuit = current_circuit + (next_gate,)

            # Determines if a (strict) subcircuit is a scalar matrix
            circuit_reducible = is_reducible(new_circuit, nqubits,
                                             1, len(new_circuit))

            # In many cases when the matrix is a scalar value,
            # the evaluated matrix will actually be an integer
            if (is_scalar_matrix(new_circuit, nqubits, id_only) and
                not is_degenerate(ids, new_circuit) and
                    not circuit_reducible):
                ids.add(GateIdentity(*new_circuit))

            elif (len(new_circuit) < max_depth and
                  not circuit_reducible):
                queue.append(new_circuit)

    return ids


def random_identity_search(gate_list, numgates, nqubits):
    """Randomly selects numgates from gate_list and checks if it is
    a gate identity.

    If the circuit is a gate identity, the circuit is returned;
    Otherwise, None is returned.
    """

    gate_size = len(gate_list)
    circuit = ()

    for i in range(numgates):
        next_gate = gate_list[randint(0, gate_size - 1)]
        circuit = circuit + (next_gate,)

    is_scalar = is_scalar_matrix(circuit, nqubits, False)

    return circuit if is_scalar else None
<|endoftext|>"""Transforms that are always applied to quantum expressions.

This module uses the kind and _constructor_postprocessor_mapping APIs
to transform different combinations of Operators, Bras, and Kets into
Inner/Outer/TensorProducts. These transformations are registered
with the postprocessing API of core classes like `Mul` and `Pow` and
are always applied to any expression involving Bras, Kets, and
Operators. This API replaces the custom `__mul__` and `__pow__`
methods of the quantum classes, which were found to be inconsistent.

THIS IS EXPERIMENTAL.
"""
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.multipledispatch.dispatcher import (
    Dispatcher, ambiguity_register_error_ignore_dup
)
from sympy.utilities.misc import debug

from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.kind import KetKind, BraKind, OperatorKind
from sympy.physics.quantum.operator import (
    OuterProduct, IdentityOperator, Operator
)
from sympy.physics.quantum.state import BraBase, KetBase, StateBase
from sympy.physics.quantum.tensorproduct import TensorProduct


#-----------------------------------------------------------------------------
# Multipledispatch based transformed for Mul and Pow
#-----------------------------------------------------------------------------

_transform_state_pair = Dispatcher('_transform_state_pair')
"""Transform a pair of expression in a Mul to their canonical form.

All functions that are registered with this dispatcher need to take
two inputs and return either tuple of transformed outputs, or None if no
transform is applied. The output tuple is inserted into the right place
of the ``Mul`` that is being put into canonical form. It works something like
the following:

``Mul(a, b, c, d, e, f) -> Mul(*(_transform_state_pair(a, b) + (c, d, e, f))))``

The transforms here are always applied when quantum objects are multiplied.

THIS IS EXPERIMENTAL.

However, users of ``sympy.physics.quantum`` can import this dispatcher and
register their own transforms to control the canonical form of products
of quantum expressions.
"""

@_transform_state_pair.register(Expr, Expr)
def _transform_expr(a, b):
    """Default transformer that does nothing for base types."""
    return None


# The identity times anything is the anything.
_transform_state_pair.add(
    (IdentityOperator, Expr),
    lambda x, y: (y,),
    on_ambiguity=ambiguity_register_error_ignore_dup
)
_transform_state_pair.add(
    (Expr, IdentityOperator),
    lambda x, y: (x,),
    on_ambiguity=ambiguity_register_error_ignore_dup
)
_transform_state_pair.add(
    (IdentityOperator, IdentityOperator),
    lambda x, y: S.One,
    on_ambiguity=ambiguity_register_error_ignore_dup
)

@_transform_state_pair.register(BraBase, KetBase)
def _transform_bra_ket(a, b):
    """Transform a bra*ket -> InnerProduct(bra, ket)."""
    return (InnerProduct(a, b),)

@_transform_state_pair.register(KetBase, BraBase)
def _transform_ket_bra(a, b):
    """Transform a keT*bra -> OuterProduct(ket, bra)."""
    return (OuterProduct(a, b),)

@_transform_state_pair.register(KetBase, KetBase)
def _transform_ket_ket(a, b):
    """Raise a TypeError if a user tries to multiply two kets.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    raise TypeError(
        'Multiplication of two kets is not allowed. Use TensorProduct instead.'
    )

@_transform_state_pair.register(BraBase, BraBase)
def _transform_bra_bra(a, b):
    """Raise a TypeError if a user tries to multiply two bras.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    raise TypeError(
        'Multiplication of two bras is not allowed. Use TensorProduct instead.'
    )

@_transform_state_pair.register(OuterProduct, KetBase)
def _transform_op_ket(a, b):
    return (InnerProduct(a.bra, b), a.ket)

@_transform_state_pair.register(BraBase, OuterProduct)
def _transform_bra_op(a, b):
    return (InnerProduct(a, b.ket), b.bra)

@_transform_state_pair.register(TensorProduct, KetBase)
def _transform_tp_ket(a, b):
    """Raise a TypeError if a user tries to multiply TensorProduct(*kets)*ket.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if a.kind == KetKind:
        raise TypeError(
            'Multiplication of TensorProduct(*kets)*ket is invalid.'
        )

@_transform_state_pair.register(KetBase, TensorProduct)
def _transform_ket_tp(a, b):
    """Raise a TypeError if a user tries to multiply ket*TensorProduct(*kets).

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if b.kind == KetKind:
        raise TypeError(
            'Multiplication of ket*TensorProduct(*kets) is invalid.'
        )

@_transform_state_pair.register(TensorProduct, BraBase)
def _transform_tp_bra(a, b):
    """Raise a TypeError if a user tries to multiply TensorProduct(*bras)*bra.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if a.kind == BraKind:
        raise TypeError(
            'Multiplication of TensorProduct(*bras)*bra is invalid.'
        )

@_transform_state_pair.register(BraBase, TensorProduct)
def _transform_bra_tp(a, b):
    """Raise a TypeError if a user tries to multiply bra*TensorProduct(*bras).

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if b.kind == BraKind:
        raise TypeError(
            'Multiplication of bra*TensorProduct(*bras) is invalid.'
        )

@_transform_state_pair.register(TensorProduct, TensorProduct)
def _transform_tp_tp(a, b):
    """Combine a product of tensor products if their number of args matches."""
    debug('_transform_tp_tp', a, b)
    if len(a.args) == len(b.args):
        if a.kind == BraKind and b.kind == KetKind:
            return tuple([InnerProduct(i, j) for (i, j) in zip(a.args, b.args)])
        else:
            return (TensorProduct(*(i*j for (i, j) in zip(a.args, b.args))), )

@_transform_state_pair.register(OuterProduct, OuterProduct)
def _transform_op_op(a, b):
    """Extract an inner produt from a product of outer products."""
    return (InnerProduct(a.bra, b.ket), OuterProduct(a.ket, b.bra))


#-----------------------------------------------------------------------------
# Postprocessing transforms for Mul and Pow
#-----------------------------------------------------------------------------


def _postprocess_state_mul(expr):
    """Transform a ``Mul`` of quantum expressions into canonical form.

    This function is registered ``_constructor_postprocessor_mapping`` as a
    transformer for ``Mul``. This means that every time a quantum expression
    is multiplied, this function will be called to transform it into canonical
    form as defined by the binary functions registered with
    ``_transform_state_pair``.

    The algorithm of this function is as follows. It walks the args
    of the input ``Mul`` from left to right and calls ``_transform_state_pair``
    on every overlapping pair of args. Each time ``_transform_state_pair``
    is called it can return a tuple of items or None. If None, the pair isn't
    transformed. If a tuple, then the last element of the tuple goes back into
    the args to be transformed again and the others are extended onto the result
    args list.

    The algorithm can be visualized in the following table:

    step   result                                 args
    ============================================================================
    #0     []                                     [a, b, c, d, e, f]
    #1     []                                     [T(a,b), c, d, e, f]
    #2     [T(a,b)[:-1]]                          [T(a,b)[-1], c, d, e, f]
    #3     [T(a,b)[:-1]]                          [T(T(a,b)[-1], c), d, e, f]
    #4     [T(a,b)[:-1], T(T(a,b)[-1], c)[:-1]]   [T(T(T(a,b)[-1], c)[-1], d), e, f]
    #5     ...

    One limitation of the current implementation is that we assume that only the
    last item of the transformed tuple goes back into the args to be transformed
    again. These seems to handle the cases needed for Mul. However, we may need
    to extend the algorithm to have the entire tuple go back into the args for
    further transformation.
    """
    args = list(expr.args)
    result = []

    # Continue as long as we have at least 2 elements
    while len(args) > 1:
        # Get first two elements
        first = args.pop(0)
        second = args[0]  # Look at second element without popping yet

        transformed = _transform_state_pair(first, second)

        if transformed is None:
            # If transform returns None, append first element
            result.append(first)
        else:
            # This item was transformed, pop and discard
            args.pop(0)
            # The last item goes back to be transformed again
            args.insert(0, transformed[-1])
            # All other items go directly into the result
            result.extend(transformed[:-1])

    # Append any remaining element
    if args:
        result.append(args[0])

    return Mul._from_args(result, is_commutative=False)


def _postprocess_state_pow(expr):
    """Handle bras and kets raised to powers.

    Under ``*`` multiplication this is invalid. Users should use a
    TensorProduct instead.
    """
    base, exp = expr.as_base_exp()
    if base.kind == KetKind or base.kind == BraKind:
        raise TypeError(
            'A bra or ket to a power is invalid, use TensorProduct instead.'
        )


def _postprocess_tp_pow(expr):
    """Handle TensorProduct(*operators)**(positive integer).

    This handles a tensor product of operators, to an integer power.
    The power here is interpreted as regular multiplication, not
    tensor product exponentiation. The form of exponentiation performed
    here leaves the space and dimension of the object the same.

    This operation does not make sense for tensor product's of states.
    """
    base, exp = expr.as_base_exp()
    debug('_postprocess_tp_pow: ', base, exp, expr.args)
    if isinstance(base, TensorProduct) and exp.is_integer and exp.is_positive and base.kind == OperatorKind:
        new_args = [a**exp for a in base.args]
        return TensorProduct(*new_args)


#-----------------------------------------------------------------------------
# Register the transformers with Basic._constructor_postprocessor_mapping
#-----------------------------------------------------------------------------


Basic._constructor_postprocessor_mapping[StateBase] = {
    "Mul": [_postprocess_state_mul],
    "Pow": [_postprocess_state_pow]
}

Basic._constructor_postprocessor_mapping[TensorProduct] = {
    "Mul": [_postprocess_state_mul],
    "Pow": [_postprocess_tp_pow]
}

Basic._constructor_postprocessor_mapping[Operator] = {
    "Mul": [_postprocess_state_mul]
}
<|endoftext|>"""Logic for representing operators in state in various bases.

TODO:

* Get represent working with continuous hilbert spaces.
* Document default basis functionality.
"""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.integrals.integrals import integrate
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.matrixutils import flatten_scalar
from sympy.physics.quantum.state import KetBase, BraBase, StateBase
from sympy.physics.quantum.operator import Operator, OuterProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.operatorset import operators_to_state, state_to_operators


__all__ = [
    'represent',
    'rep_innerproduct',
    'rep_expectation',
    'integrate_result',
    'get_basis',
    'enumerate_states'
]

#-----------------------------------------------------------------------------
# Represent
#-----------------------------------------------------------------------------


def _sympy_to_scalar(e):
    """Convert from a SymPy scalar to a Python scalar."""
    if isinstance(e, Expr):
        if e.is_Integer:
            return int(e)
        elif e.is_Float:
            return float(e)
        elif e.is_Rational:
            return float(e)
        elif e.is_Number or e.is_NumberSymbol or e == I:
            return complex(e)
    raise TypeError('Expected number, got: %r' % e)


def represent(expr, **options):
    """Represent the quantum expression in the given basis.

    In quantum mechanics abstract states and operators can be represented in
    various basis sets. Under this operation the follow transforms happen:

    * Ket -> column vector or function
    * Bra -> row vector of function
    * Operator -> matrix or differential operator

    This function is the top-level interface for this action.

    This function walks the SymPy expression tree looking for ``QExpr``
    instances that have a ``_represent`` method. This method is then called
    and the object is replaced by the representation returned by this method.
    By default, the ``_represent`` method will dispatch to other methods
    that handle the representation logic for a particular basis set. The
    naming convention for these methods is the following::

        def _represent_FooBasis(self, e, basis, **options)

    This function will have the logic for representing instances of its class
    in the basis set having a class named ``FooBasis``.

    Parameters
    ==========

    expr  : Expr
        The expression to represent.
    basis : Operator, basis set
        An object that contains the information about the basis set. If an
        operator is used, the basis is assumed to be the orthonormal
        eigenvectors of that operator. In general though, the basis argument
        can be any object that contains the basis set information.
    options : dict
        Key/value pairs of options that are passed to the underlying method
        that finds the representation. These options can be used to
        control how the representation is done. For example, this is where
        the size of the basis set would be set.

    Returns
    =======

    e : Expr
        The SymPy expression of the represented quantum expression.

    Examples
    ========

    Here we subclass ``Operator`` and ``Ket`` to create the z-spin operator
    and its spin 1/2 up eigenstate. By defining the ``_represent_SzOp``
    method, the ket can be represented in the z-spin basis.

    >>> from sympy.physics.quantum import Operator, represent, Ket
    >>> from sympy import Matrix

    >>> class SzUpKet(Ket):
    ...     def _represent_SzOp(self, basis, **options):
    ...         return Matrix([1,0])
    ...
    >>> class SzOp(Operator):
    ...     pass
    ...
    >>> sz = SzOp('Sz')
    >>> up = SzUpKet('up')
    >>> represent(up, basis=sz)
    Matrix([
    [1],
    [0]])

    Here we see an example of representations in a continuous
    basis. We see that the result of representing various combinations
    of cartesian position operators and kets give us continuous
    expressions involving DiracDelta functions.

    >>> from sympy.physics.quantum.cartesian import XOp, XKet, XBra
    >>> X = XOp()
    >>> x = XKet()
    >>> y = XBra('y')
    >>> represent(X*x)
    x*DiracDelta(x - x_2)
    """

    format = options.get('format', 'sympy')
    if format == 'numpy':
        import numpy as np
    if isinstance(expr, QExpr) and not isinstance(expr, OuterProduct):
        options['replace_none'] = False
        temp_basis = get_basis(expr, **options)
        if temp_basis is not None:
            options['basis'] = temp_basis
        try:
            return expr._represent(**options)
        except NotImplementedError as strerr:
            #If no _represent_FOO method exists, map to the
            #appropriate basis state and try
            #the other methods of representation
            options['replace_none'] = True

            if isinstance(expr, (KetBase, BraBase)):
                try:
                    return rep_innerproduct(expr, **options)
                except NotImplementedError:
                    raise NotImplementedError(strerr)
            elif isinstance(expr, Operator):
                try:
                    return rep_expectation(expr, **options)
                except NotImplementedError:
                    raise NotImplementedError(strerr)
            else:
                raise NotImplementedError(strerr)
    elif isinstance(expr, Add):
        result = represent(expr.args[0], **options)
        for args in expr.args[1:]:
            # scipy.sparse doesn't support += so we use plain = here.
            result = result + represent(args, **options)
        return result
    elif isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        if format in ('numpy', 'scipy.sparse'):
            exp = _sympy_to_scalar(exp)
        base = represent(base, **options)
        # scipy.sparse doesn't support negative exponents
        # and warns when inverting a matrix in csr format.
        if format == 'scipy.sparse' and exp < 0:
            from scipy.sparse.linalg import inv
            exp = - exp
            base = inv(base.tocsc()).tocsr()
        if format == 'numpy':
            return np.linalg.matrix_power(base, exp)
        return base ** exp
    elif isinstance(expr, TensorProduct):
        new_args = [represent(arg, **options) for arg in expr.args]
        return TensorProduct(*new_args)
    elif isinstance(expr, Dagger):
        return Dagger(represent(expr.args[0], **options))
    elif isinstance(expr, Commutator):
        A = expr.args[0]
        B = expr.args[1]
        return represent(Mul(A, B) - Mul(B, A), **options)
    elif isinstance(expr, AntiCommutator):
        A = expr.args[0]
        B = expr.args[1]
        return represent(Mul(A, B) + Mul(B, A), **options)
    elif not isinstance(expr, (Mul, OuterProduct, InnerProduct)):
        # We have removed special handling of inner products that used to be
        # required (before automatic transforms).
        # For numpy and scipy.sparse, we can only handle numerical prefactors.
        if format in ('numpy', 'scipy.sparse'):
            return _sympy_to_scalar(expr)
        return expr

    if not isinstance(expr, (Mul, OuterProduct, InnerProduct)):
        raise TypeError('Mul expected, got: %r' % expr)

    if "index" in options:
        options["index"] += 1
    else:
        options["index"] = 1

    if "unities" not in options:
        options["unities"] = []

    result = represent(expr.args[-1], **options)
    last_arg = expr.args[-1]

    for arg in reversed(expr.args[:-1]):
        if isinstance(last_arg, Operator):
            options["index"] += 1
            options["unities"].append(options["index"])
        elif isinstance(last_arg, BraBase) and isinstance(arg, KetBase):
            options["index"] += 1
        elif isinstance(last_arg, KetBase) and isinstance(arg, Operator):
            options["unities"].append(options["index"])
        elif isinstance(last_arg, KetBase) and isinstance(arg, BraBase):
            options["unities"].append(options["index"])

        next_arg = represent(arg, **options)
        if format == 'numpy' and isinstance(next_arg, np.ndarray):
            # Must use np.matmult to "matrix multiply" two np.ndarray
            result = np.matmul(next_arg, result)
        else:
            result = next_arg*result
        last_arg = arg

    # All three matrix formats create 1 by 1 matrices when inner products of
    # vectors are taken. In these cases, we simply return a scalar.
    result = flatten_scalar(result)

    result = integrate_result(expr, result, **options)

    return result


def rep_innerproduct(expr, **options):
    """
    Returns an innerproduct like representation (e.g. ``<x'|x>``) for the
    given state.

    Attempts to calculate inner product with a bra from the specified
    basis. Should only be passed an instance of KetBase or BraBase

    Parameters
    ==========

    expr : KetBase or BraBase
        The expression to be represented

    Examples
    ========

    >>> from sympy.physics.quantum.represent import rep_innerproduct
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> rep_innerproduct(XKet())
    DiracDelta(x - x_1)
    >>> rep_innerproduct(XKet(), basis=PxOp())
    sqrt(2)*exp(-I*px_1*x/hbar)/(2*sqrt(hbar)*sqrt(pi))
    >>> rep_innerproduct(PxKet(), basis=XOp())
    sqrt(2)*exp(I*px*x_1/hbar)/(2*sqrt(hbar)*sqrt(pi))

    """

    if not isinstance(expr, (KetBase, BraBase)):
        raise TypeError("expr passed is not a Bra or Ket")

    basis = get_basis(expr, **options)

    if not isinstance(basis, StateBase):
        raise NotImplementedError("Can't form this representation!")

    if "index" not in options:
        options["index"] = 1

    basis_kets = enumerate_states(basis, options["index"], 2)

    if isinstance(expr, BraBase):
        bra = expr
        ket = (basis_kets[1] if basis_kets[0].dual == expr else basis_kets[0])
    else:
        bra = (basis_kets[1].dual if basis_kets[0]
               == expr else basis_kets[0].dual)
        ket = expr

    prod = InnerProduct(bra, ket)
    result = prod.doit()

    format = options.get('format', 'sympy')
    result = expr._format_represent(result, format)
    return result


def rep_expectation(expr, **options):
    """
    Returns an ``<x'|A|x>`` type representation for the given operator.

    Parameters
    ==========

    expr : Operator
        Operator to be represented in the specified basis

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XOp, PxOp, PxKet
    >>> from sympy.physics.quantum.represent import rep_expectation
    >>> rep_expectation(XOp())
    x_1*DiracDelta(x_1 - x_2)
    >>> rep_expectation(XOp(), basis=PxOp())
    <px_2|*X*|px_1>
    >>> rep_expectation(XOp(), basis=PxKet())
    <px_2|*X*|px_1>

    """

    if "index" not in options:
        options["index"] = 1

    if not isinstance(expr, Operator):
        raise TypeError("The passed expression is not an operator")

    basis_state = get_basis(expr, **options)

    if basis_state is None or not isinstance(basis_state, StateBase):
        raise NotImplementedError("Could not get basis kets for this operator")

    basis_kets = enumerate_states(basis_state, options["index"], 2)

    bra = basis_kets[1].dual
    ket = basis_kets[0]

    result = qapply(bra*expr*ket)
    return result


def integrate_result(orig_expr, result, **options):
    """
    Returns the result of integrating over any unities ``(|x><x|)`` in
    the given expression. Intended for integrating over the result of
    representations in continuous bases.

    This function integrates over any unities that may have been
    inserted into the quantum expression and returns the result.
    It uses the interval of the Hilbert space of the basis state
    passed to it in order to figure out the limits of integration.
    The unities option must be
    specified for this to work.

    Note: This is mostly used internally by represent(). Examples are
    given merely to show the use cases.

    Parameters
    ==========

    orig_expr : quantum expression
        The original expression which was to be represented

    result: Expr
        The resulting representation that we wish to integrate over

    Examples
    ========

    >>> from sympy import symbols, DiracDelta
    >>> from sympy.physics.quantum.represent import integrate_result
    >>> from sympy.physics.quantum.cartesian import XOp, XKet
    >>> x_ket = XKet()
    >>> X_op = XOp()
    >>> x, x_1, x_2 = symbols('x, x_1, x_2')
    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2))
    x*DiracDelta(x - x_1)*DiracDelta(x_1 - x_2)
    >>> integrate_result(X_op*x_ket, x*DiracDelta(x-x_1)*DiracDelta(x_1-x_2),
    ...     unities=[1])
    x*DiracDelta(x - x_2)

    """
    if not isinstance(result, Expr):
        return result

    options['replace_none'] = True
    if "basis" not in options:
        arg = orig_expr.args[-1]
        options["basis"] = get_basis(arg, **options)
    elif not isinstance(options["basis"], StateBase):
        options["basis"] = get_basis(orig_expr, **options)

    basis = options.pop("basis", None)

    if basis is None:
        return result

    unities = options.pop("unities", [])

    if len(unities) == 0:
        return result

    kets = enumerate_states(basis, unities)
    coords = [k.label[0] for k in kets]

    for coord in coords:
        if coord in result.free_symbols:
            #TODO: Add support for sets of operators
            basis_op = state_to_operators(basis)
            start = basis_op.hilbert_space.interval.start
            end = basis_op.hilbert_space.interval.end
            result = integrate(result, (coord, start, end))

    return result


def get_basis(expr, *, basis=None, replace_none=True, **options):
    """
    Returns a basis state instance corresponding to the basis specified in
    options=s. If no basis is specified, the function tries to form a default
    basis state of the given expression.

    There are three behaviors:

    1. The basis specified in options is already an instance of StateBase. If
       this is the case, it is simply returned. If the class is specified but
       not an instance, a default instance is returned.

    2. The basis specified is an operator or set of operators. If this
       is the case, the operator_to_state mapping method is used.

    3. No basis is specified. If expr is a state, then a default instance of
       its class is returned.  If expr is an operator, then it is mapped to the
       corresponding state.  If it is neither, then we cannot obtain the basis
       state.

    If the basis cannot be mapped, then it is not changed.

    This will be called from within represent, and represent will
    only pass QExpr's.

    TODO (?): Support for Muls and other types of expressions?

    Parameters
    ==========

    expr : Operator or StateBase
        Expression whose basis is sought

    Examples
    ========

    >>> from sympy.physics.quantum.represent import get_basis
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> x = XKet()
    >>> X = XOp()
    >>> get_basis(x)
    |x>
    >>> get_basis(X)
    |x>
    >>> get_basis(x, basis=PxOp())
    |px>
    >>> get_basis(x, basis=PxKet)
    |px>

    """

    if basis is None and not replace_none:
        return None

    if basis is None:
        if isinstance(expr, KetBase):
            return _make_default(expr.__class__)
        elif isinstance(expr, BraBase):
            return _make_default(expr.dual_class())
        elif isinstance(expr, Operator):
            state_inst = operators_to_state(expr)
            return (state_inst if state_inst is not None else None)
        else:
            return None
    elif (isinstance(basis, Operator) or
          (not isinstance(basis, StateBase) and issubclass(basis, Operator))):
        state = operators_to_state(basis)
        if state is None:
            return None
        elif isinstance(state, StateBase):
            return state
        else:
            return _make_default(state)
    elif isinstance(basis, StateBase):
        return basis
    elif issubclass(basis, StateBase):
        return _make_default(basis)
    else:
        return None


def _make_default(expr):
    # XXX: Catching TypeError like this is a bad way of distinguishing
    # instances from classes. The logic using this function should be
    # rewritten somehow.
    try:
        expr = expr()
    except TypeError:
        return expr

    return expr


def enumerate_states(*args, **options):
    """
    Returns instances of the given state with dummy indices appended

    Operates in two different modes:

    1. Two arguments are passed to it. The first is the base state which is to
       be indexed, and the second argument is a list of indices to append.

    2. Three arguments are passed. The first is again the base state to be
       indexed. The second is the start index for counting.  The final argument
       is the number of kets you wish to receive.

    Tries to call state._enumerate_state. If this fails, returns an empty list

    Parameters
    ==========

    args : list
        See list of operation modes above for explanation

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XBra, XKet
    >>> from sympy.physics.quantum.represent import enumerate_states
    >>> test = XKet('foo')
    >>> enumerate_states(test, 1, 3)
    [|foo_1>, |foo_2>, |foo_3>]
    >>> test2 = XBra('bar')
    >>> enumerate_states(test2, [4, 5, 10])
    [<bar_4|, <bar_5|, <bar_10|]

    """

    state = args[0]

    if len(args) not in (2, 3):
        raise NotImplementedError("Wrong number of arguments!")

    if not isinstance(state, StateBase):
        raise TypeError("First argument is not a state!")

    if len(args) == 3:
        num_states = args[2]
        options['start_index'] = args[1]
    else:
        num_states = len(args[1])
        options['index_list'] = args[1]

    try:
        ret = state._enumerate_state(num_states, **options)
    except NotImplementedError:
        ret = []

    return ret
<|endoftext|>"""

qasm.py - Functions to parse a set of qasm commands into a SymPy Circuit.

Examples taken from Chuang's page: https://web.archive.org/web/20220120121541/https://www.media.mit.edu/quanta/qasm2circ/

The code returns a circuit and an associated list of labels.

>>> from sympy.physics.quantum.qasm import Qasm
>>> q = Qasm('qubit q0', 'qubit q1', 'h q0', 'cnot q0,q1')
>>> q.get_circuit()
CNOT(1,0)*H(1)

>>> q = Qasm('qubit q0', 'qubit q1', 'cnot q0,q1', 'cnot q1,q0', 'cnot q0,q1')
>>> q.get_circuit()
CNOT(1,0)*CNOT(0,1)*CNOT(1,0)
"""

__all__ = [
    'Qasm',
    ]

from math import prod

from sympy.physics.quantum.gate import H, CNOT, X, Z, CGate, CGateS, SWAP, S, T,CPHASE
from sympy.physics.quantum.circuitplot import Mz

def read_qasm(lines):
    return Qasm(*lines.splitlines())

def read_qasm_file(filename):
    return Qasm(*open(filename).readlines())

def flip_index(i, n):
    """Reorder qubit indices from largest to smallest.

    >>> from sympy.physics.quantum.qasm import flip_index
    >>> flip_index(0, 2)
    1
    >>> flip_index(1, 2)
    0
    """
    return n-i-1

def trim(line):
    """Remove everything following comment # characters in line.

    >>> from sympy.physics.quantum.qasm import trim
    >>> trim('nothing happens here')
    'nothing happens here'
    >>> trim('something #happens here')
    'something '
    """
    if '#' not in line:
        return line
    return line.split('#')[0]

def get_index(target, labels):
    """Get qubit labels from the rest of the line,and return indices

    >>> from sympy.physics.quantum.qasm import get_index
    >>> get_index('q0', ['q0', 'q1'])
    1
    >>> get_index('q1', ['q0', 'q1'])
    0
    """
    nq = len(labels)
    return flip_index(labels.index(target), nq)

def get_indices(targets, labels):
    return [get_index(t, labels) for t in targets]

def nonblank(args):
    for line in args:
        line = trim(line)
        if line.isspace():
            continue
        yield line
    return

def fullsplit(line):
    words = line.split()
    rest = ' '.join(words[1:])
    return fixcommand(words[0]), [s.strip() for s in rest.split(',')]

def fixcommand(c):
    """Fix Qasm command names.

    Remove all of forbidden characters from command c, and
    replace 'def' with 'qdef'.
    """
    forbidden_characters = ['-']
    c = c.lower()
    for char in forbidden_characters:
        c = c.replace(char, '')
    if c == 'def':
        return 'qdef'
    return c

def stripquotes(s):
    """Replace explicit quotes in a string.

    >>> from sympy.physics.quantum.qasm import stripquotes
    >>> stripquotes("'S'") == 'S'
    True
    >>> stripquotes('"S"') == 'S'
    True
    >>> stripquotes('S') == 'S'
    True
    """
    s = s.replace('"', '') # Remove second set of quotes?
    s = s.replace("'", '')
    return s

class Qasm:
    """Class to form objects from Qasm lines

    >>> from sympy.physics.quantum.qasm import Qasm
    >>> q = Qasm('qubit q0', 'qubit q1', 'h q0', 'cnot q0,q1')
    >>> q.get_circuit()
    CNOT(1,0)*H(1)
    >>> q = Qasm('qubit q0', 'qubit q1', 'cnot q0,q1', 'cnot q1,q0', 'cnot q0,q1')
    >>> q.get_circuit()
    CNOT(1,0)*CNOT(0,1)*CNOT(1,0)
    """
    def __init__(self, *args, **kwargs):
        self.defs = {}
        self.circuit = []
        self.labels = []
        self.inits = {}
        self.add(*args)
        self.kwargs = kwargs

    def add(self, *lines):
        for line in nonblank(lines):
            command, rest = fullsplit(line)
            if self.defs.get(command): #defs come first, since you can override built-in
                function = self.defs.get(command)
                indices = self.indices(rest)
                if len(indices) == 1:
                    self.circuit.append(function(indices[0]))
                else:
                    self.circuit.append(function(indices[:-1], indices[-1]))
            elif hasattr(self, command):
                function = getattr(self, command)
                function(*rest)
            else:
                print("Function %s not defined. Skipping" % command)

    def get_circuit(self):
        return prod(reversed(self.circuit))

    def get_labels(self):
        return list(reversed(self.labels))

    def plot(self):
        from sympy.physics.quantum.circuitplot import CircuitPlot
        circuit, labels = self.get_circuit(), self.get_labels()
        CircuitPlot(circuit, len(labels), labels=labels, inits=self.inits)

    def qubit(self, arg, init=None):
        self.labels.append(arg)
        if init: self.inits[arg] = init

    def indices(self, args):
        return get_indices(args, self.labels)

    def index(self, arg):
        return get_index(arg, self.labels)

    def nop(self, *args):
        pass

    def x(self, arg):
        self.circuit.append(X(self.index(arg)))

    def z(self, arg):
        self.circuit.append(Z(self.index(arg)))

    def h(self, arg):
        self.circuit.append(H(self.index(arg)))

    def s(self, arg):
        self.circuit.append(S(self.index(arg)))

    def t(self, arg):
        self.circuit.append(T(self.index(arg)))

    def measure(self, arg):
        self.circuit.append(Mz(self.index(arg)))

    def cnot(self, a1, a2):
        self.circuit.append(CNOT(*self.indices([a1, a2])))

    def swap(self, a1, a2):
        self.circuit.append(SWAP(*self.indices([a1, a2])))

    def cphase(self, a1, a2):
        self.circuit.append(CPHASE(*self.indices([a1, a2])))

    def toffoli(self, a1, a2, a3):
        i1, i2, i3 = self.indices([a1, a2, a3])
        self.circuit.append(CGateS((i1, i2), X(i3)))

    def cx(self, a1, a2):
        fi, fj = self.indices([a1, a2])
        self.circuit.append(CGate(fi, X(fj)))

    def cz(self, a1, a2):
        fi, fj = self.indices([a1, a2])
        self.circuit.append(CGate(fi, Z(fj)))

    def defbox(self, *args):
        print("defbox not supported yet. Skipping: ", args)

    def qdef(self, name, ncontrols, symbol):
        from sympy.physics.quantum.circuitplot import CreateOneQubitGate, CreateCGate
        ncontrols = int(ncontrols)
        command = fixcommand(name)
        symbol = stripquotes(symbol)
        if ncontrols > 0:
            self.defs[command] = CreateCGate(symbol)
        else:
            self.defs[command] = CreateOneQubitGate(symbol)
<|endoftext|>"""Shor's algorithm and helper functions.

Todo:

* Get the CMod gate working again using the new Gate API.
* Fix everything.
* Update docstrings and reformat.
"""

import math
import random

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.intfunc import igcd
from sympy.ntheory import continued_fraction_periodic as continued_fraction
from sympy.utilities.iterables import variations

from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import Qubit, measure_partial_oneshot
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qft import QFT
from sympy.physics.quantum.qexpr import QuantumError


class OrderFindingException(QuantumError):
    pass


class CMod(Gate):
    """A controlled mod gate.

    This is black box controlled Mod function for use by shor's algorithm.
    TODO: implement a decompose property that returns how to do this in terms
    of elementary gates
    """

    @classmethod
    def _eval_args(cls, args):
        # t = args[0]
        # a = args[1]
        # N = args[2]
        raise NotImplementedError('The CMod gate has not been completed.')

    @property
    def t(self):
        """Size of 1/2 input register.  First 1/2 holds output."""
        return self.label[0]

    @property
    def a(self):
        """Base of the controlled mod function."""
        return self.label[1]

    @property
    def N(self):
        """N is the type of modular arithmetic we are doing."""
        return self.label[2]

    def _apply_operator_Qubit(self, qubits, **options):
        """
            This directly calculates the controlled mod of the second half of
            the register and puts it in the second
            This will look pretty when we get Tensor Symbolically working
        """
        n = 1
        k = 0
        # Determine the value stored in high memory.
        for i in range(self.t):
            k += n*qubits[self.t + i]
            n *= 2

        # The value to go in low memory will be out.
        out = int(self.a**k % self.N)

        # Create array for new qbit-ket which will have high memory unaffected
        outarray = list(qubits.args[0][:self.t])

        # Place out in low memory
        for i in reversed(range(self.t)):
            outarray.append((out >> i) & 1)

        return Qubit(*outarray)


def shor(N):
    """This function implements Shor's factoring algorithm on the Integer N

    The algorithm starts by picking a random number (a) and seeing if it is
    coprime with N. If it is not, then the gcd of the two numbers is a factor
    and we are done. Otherwise, it begins the period_finding subroutine which
    finds the period of a in modulo N arithmetic. This period, if even, can
    be used to calculate factors by taking a**(r/2)-1 and a**(r/2)+1.
    These values are returned.
    """
    a = random.randrange(N - 2) + 2
    if igcd(N, a) != 1:
        return igcd(N, a)
    r = period_find(a, N)
    if r % 2 == 1:
        shor(N)
    answer = (igcd(a**(r/2) - 1, N), igcd(a**(r/2) + 1, N))
    return answer


def getr(x, y, N):
    fraction = continued_fraction(x, y)
    # Now convert into r
    total = ratioize(fraction, N)
    return total


def ratioize(list, N):
    if list[0] > N:
        return S.Zero
    if len(list) == 1:
        return list[0]
    return list[0] + ratioize(list[1:], N)


def period_find(a, N):
    """Finds the period of a in modulo N arithmetic

    This is quantum part of Shor's algorithm. It takes two registers,
    puts first in superposition of states with Hadamards so: ``|k>|0>``
    with k being all possible choices. It then does a controlled mod and
    a QFT to determine the order of a.
    """
    epsilon = .5
    # picks out t's such that maintains accuracy within epsilon
    t = int(2*math.ceil(log(N, 2)))
    # make the first half of register be 0's |000...000>
    start = [0 for x in range(t)]
    # Put second half into superposition of states so we have |1>x|0> + |2>x|0> + ... |k>x>|0> + ... + |2**n-1>x|0>
    factor = 1/sqrt(2**t)
    qubits = 0
    for arr in variations(range(2), t, repetition=True):
        qbitArray = list(arr) + start
        qubits = qubits + Qubit(*qbitArray)
    circuit = (factor*qubits).expand()
    # Controlled second half of register so that we have:
    # |1>x|a**1 %N> + |2>x|a**2 %N> + ... + |k>x|a**k %N >+ ... + |2**n-1=k>x|a**k % n>
    circuit = CMod(t, a, N)*circuit
    # will measure first half of register giving one of the a**k%N's

    circuit = qapply(circuit)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i)
    # Now apply Inverse Quantum Fourier Transform on the second half of the register

    circuit = qapply(QFT(t, t*2).decompose()*circuit, floatingPoint=True)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i + t)
    if isinstance(circuit, Qubit):
        register = circuit
    elif isinstance(circuit, Mul):
        register = circuit.args[-1]
    else:
        register = circuit.args[-1].args[-1]

    n = 1
    answer = 0
    for i in range(len(register)/2):
        answer += n*register[i + t]
        n = n << 1
    if answer == 0:
        raise OrderFindingException(
            "Order finder returned 0. Happens with chance %f" % epsilon)
    #turn answer into r using continued fractions
    g = getr(answer, 2**t, N)
    return g
<|endoftext|>"""The commutator: [A,B] = A*B - B*A."""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.kind import KindDispatcher
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.kind import _OperatorKind, OperatorKind


__all__ = [
    'Commutator'
]

#-----------------------------------------------------------------------------
# Commutator
#-----------------------------------------------------------------------------


class Commutator(Expr):
    """The standard commutator, in an unevaluated state.

    Explanation
    ===========

    Evaluating a commutator is defined [1]_ as: ``[A, B] = A*B - B*A``. This
    class returns the commutator in an unevaluated form. To evaluate the
    commutator, use the ``.doit()`` method.

    Canonical ordering of a commutator is ``[A, B]`` for ``A < B``. The
    arguments of the commutator are put into canonical order using ``__cmp__``.
    If ``B < A``, then ``[B, A]`` is returned as ``-[A, B]``.

    Parameters
    ==========

    A : Expr
        The first argument of the commutator [A,B].
    B : Expr
        The second argument of the commutator [A,B].

    Examples
    ========

    >>> from sympy.physics.quantum import Commutator, Dagger, Operator
    >>> from sympy.abc import x, y
    >>> A = Operator('A')
    >>> B = Operator('B')
    >>> C = Operator('C')

    Create a commutator and use ``.doit()`` to evaluate it:

    >>> comm = Commutator(A, B)
    >>> comm
    [A,B]
    >>> comm.doit()
    A*B - B*A

    The commutator orders it arguments in canonical order:

    >>> comm = Commutator(B, A); comm
    -[A,B]

    Commutative constants are factored out:

    >>> Commutator(3*x*A, x*y*B)
    3*x**2*y*[A,B]

    Using ``.expand(commutator=True)``, the standard commutator expansion rules
    can be applied:

    >>> Commutator(A+B, C).expand(commutator=True)
    [A,C] + [B,C]
    >>> Commutator(A, B+C).expand(commutator=True)
    [A,B] + [A,C]
    >>> Commutator(A*B, C).expand(commutator=True)
    [A,C]*B + A*[B,C]
    >>> Commutator(A, B*C).expand(commutator=True)
    [A,B]*C + B*[A,C]

    Adjoint operations applied to the commutator are properly applied to the
    arguments:

    >>> Dagger(Commutator(A, B))
    -[Dagger(A),Dagger(B)]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Commutator
    """
    is_commutative = False

    _kind_dispatcher = KindDispatcher("Commutator_kind_dispatcher", commutative=True)

    @property
    def kind(self):
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    def __new__(cls, A, B):
        r = cls.eval(A, B)
        if r is not None:
            return r
        obj = Expr.__new__(cls, A, B)
        return obj

    @classmethod
    def eval(cls, a, b):
        if not (a and b):
            return S.Zero
        if a == b:
            return S.Zero
        if a.is_commutative or b.is_commutative:
            return S.Zero

        # [xA,yB]  ->  xy*[A,B]
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = ca + cb
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))

        # Canonical ordering of arguments
        # The Commutator [A, B] is in canonical form if A < B.
        if a.compare(b) == 1:
            return S.NegativeOne*cls(b, a)

    def _expand_pow(self, A, B, sign):
        exp = A.exp
        if not exp.is_integer or not exp.is_constant() or abs(exp) <= 1:
            # nothing to do
            return self
        base = A.base
        if exp.is_negative:
            base = A.base**-1
            exp = -exp
        comm = Commutator(base, B).expand(commutator=True)

        result = base**(exp - 1) * comm
        for i in range(1, exp):
            result += base**(exp - 1 - i) * comm * base**i
        return sign*result.expand()

    def _eval_expand_commutator(self, **hints):
        A = self.args[0]
        B = self.args[1]

        if isinstance(A, Add):
            # [A + B, C]  ->  [A, C] + [B, C]
            sargs = []
            for term in A.args:
                comm = Commutator(term, B)
                if isinstance(comm, Commutator):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            return Add(*sargs)
        elif isinstance(B, Add):
            # [A, B + C]  ->  [A, B] + [A, C]
            sargs = []
            for term in B.args:
                comm = Commutator(A, term)
                if isinstance(comm, Commutator):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            return Add(*sargs)
        elif isinstance(A, Mul):
            # [A*B, C] -> A*[B, C] + [A, C]*B
            a = A.args[0]
            b = Mul(*A.args[1:])
            c = B
            comm1 = Commutator(b, c)
            comm2 = Commutator(a, c)
            if isinstance(comm1, Commutator):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, Commutator):
                comm2 = comm2._eval_expand_commutator()
            first = Mul(a, comm1)
            second = Mul(comm2, b)
            return Add(first, second)
        elif isinstance(B, Mul):
            # [A, B*C] -> [A, B]*C + B*[A, C]
            a = A
            b = B.args[0]
            c = Mul(*B.args[1:])
            comm1 = Commutator(a, b)
            comm2 = Commutator(a, c)
            if isinstance(comm1, Commutator):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, Commutator):
                comm2 = comm2._eval_expand_commutator()
            first = Mul(comm1, c)
            second = Mul(b, comm2)
            return Add(first, second)
        elif isinstance(A, Pow):
            # [A**n, C] -> A**(n - 1)*[A, C] + A**(n - 2)*[A, C]*A + ... + [A, C]*A**(n-1)
            return self._expand_pow(A, B, 1)
        elif isinstance(B, Pow):
            # [A, C**n] -> C**(n - 1)*[C, A] + C**(n - 2)*[C, A]*C + ... + [C, A]*C**(n-1)
            return self._expand_pow(B, A, -1)

        # No changes, so return self
        return self

    def doit(self, **hints):
        """ Evaluate commutator """
        # Keep the import of Operator here to avoid problems with
        # circular imports.
        from sympy.physics.quantum.operator import Operator
        A = self.args[0]
        B = self.args[1]
        if isinstance(A, Operator) and isinstance(B, Operator):
            try:
                comm = A._eval_commutator(B, **hints)
            except NotImplementedError:
                try:
                    comm = -1*B._eval_commutator(A, **hints)
                except NotImplementedError:
                    comm = None
            if comm is not None:
                return comm.doit(**hints)
        return (A*B - B*A).doit(**hints)

    def _eval_adjoint(self):
        return Commutator(Dagger(self.args[1]), Dagger(self.args[0]))

    def _sympyrepr(self, printer, *args):
        return "%s(%s,%s)" % (
            self.__class__.__name__, printer._print(
                self.args[0]), printer._print(self.args[1])
        )

    def _sympystr(self, printer, *args):
        return "[%s,%s]" % (
            printer._print(self.args[0]), printer._print(self.args[1]))

    def _pretty(self, printer, *args):
        pform = printer._print(self.args[0], *args)
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(printer._print(self.args[1], *args)))
        pform = prettyForm(*pform.parens(left='[', right=']'))
        return pform

    def _latex(self, printer, *args):
        return "\\left[%s,%s\\right]" % tuple([
            printer._print(arg, *args) for arg in self.args])


@Commutator._kind_dispatcher.register(_OperatorKind, _OperatorKind)
def find_op_kind(e1, e2):
    """Find the kind of an anticommutator of two OperatorKinds."""
    return OperatorKind
<|endoftext|>"""Dirac notation for states."""

from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import oo, equal_valued
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.printing.pretty.stringpict import stringPict
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.physics.quantum.kind import KetKind, BraKind


__all__ = [
    'KetBase',
    'BraBase',
    'StateBase',
    'State',
    'Ket',
    'Bra',
    'TimeDepState',
    'TimeDepBra',
    'TimeDepKet',
    'OrthogonalKet',
    'OrthogonalBra',
    'OrthogonalState',
    'Wavefunction'
]


#-----------------------------------------------------------------------------
# States, bras and kets.
#-----------------------------------------------------------------------------

# ASCII brackets
_lbracket = "<"
_rbracket = ">"
_straight_bracket = "|"


# Unicode brackets
# MATHEMATICAL ANGLE BRACKETS
_lbracket_ucode = "\N{MATHEMATICAL LEFT ANGLE BRACKET}"
_rbracket_ucode = "\N{MATHEMATICAL RIGHT ANGLE BRACKET}"
# LIGHT VERTICAL BAR
_straight_bracket_ucode = "\N{LIGHT VERTICAL BAR}"

# Other options for unicode printing of <, > and | for Dirac notation.

# LEFT-POINTING ANGLE BRACKET
# _lbracket = "\u2329"
# _rbracket = "\u232A"

# LEFT ANGLE BRACKET
# _lbracket = "\u3008"
# _rbracket = "\u3009"

# VERTICAL LINE
# _straight_bracket = "\u007C"


class StateBase(QExpr):
    """Abstract base class for general abstract states in quantum mechanics.

    All other state classes defined will need to inherit from this class. It
    carries the basic structure for all other states such as dual, _eval_adjoint
    and label.

    This is an abstract base class and you should not instantiate it directly,
    instead use State.
    """

    @classmethod
    def _operators_to_state(self, ops, **options):
        """ Returns the eigenstate instance for the passed operators.

        This method should be overridden in subclasses. It will handle being
        passed either an Operator instance or set of Operator instances. It
        should return the corresponding state INSTANCE or simply raise a
        NotImplementedError. See cartesian.py for an example.
        """

        raise NotImplementedError("Cannot map operators to states in this class. Method not implemented!")

    def _state_to_operators(self, op_classes, **options):
        """ Returns the operators which this state instance is an eigenstate
        of.

        This method should be overridden in subclasses. It will be called on
        state instances and be passed the operator classes that we wish to make
        into instances. The state instance will then transform the classes
        appropriately, or raise a NotImplementedError if it cannot return
        operator instances. See cartesian.py for examples,
        """

        raise NotImplementedError(
            "Cannot map this state to operators. Method not implemented!")

    @property
    def operators(self):
        """Return the operator(s) that this state is an eigenstate of"""
        from .operatorset import state_to_operators  # import internally to avoid circular import errors
        return state_to_operators(self)

    def _enumerate_state(self, num_states, **options):
        raise NotImplementedError("Cannot enumerate this state!")

    def _represent_default_basis(self, **options):
        return self._represent(basis=self.operators)

    def _apply_operator(self, op, **options):
        return None

    #-------------------------------------------------------------------------
    # Dagger/dual
    #-------------------------------------------------------------------------

    @property
    def dual(self):
        """Return the dual state of this one."""
        return self.dual_class()._new_rawargs(self.hilbert_space, *self.args)

    @classmethod
    def dual_class(self):
        """Return the class used to construct the dual."""
        raise NotImplementedError(
            'dual_class must be implemented in a subclass'
        )

    def _eval_adjoint(self):
        """Compute the dagger of this state using the dual."""
        return self.dual

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _pretty_brackets(self, height, use_unicode=True):
        # Return pretty printed brackets for the state
        # Ideally, this could be done by pform.parens but it does not support the angled < and >

        # Setup for unicode vs ascii
        if use_unicode:
            lbracket, rbracket = getattr(self, 'lbracket_ucode', ""), getattr(self, 'rbracket_ucode', "")
            slash, bslash, vert = '\N{BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT}', \
                                  '\N{BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT}', \
                                  '\N{BOX DRAWINGS LIGHT VERTICAL}'
        else:
            lbracket, rbracket = getattr(self, 'lbracket', ""), getattr(self, 'rbracket', "")
            slash, bslash, vert = '/', '\\', '|'

        # If height is 1, just return brackets
        if height == 1:
            return stringPict(lbracket), stringPict(rbracket)
        # Make height even
        height += (height % 2)

        brackets = []
        for bracket in lbracket, rbracket:
            # Create left bracket
            if bracket in {_lbracket, _lbracket_ucode}:
                bracket_args = [ ' ' * (height//2 - i - 1) +
                                 slash for i in range(height // 2)]
                bracket_args.extend(
                    [' ' * i + bslash for i in range(height // 2)])
            # Create right bracket
            elif bracket in {_rbracket, _rbracket_ucode}:
                bracket_args = [ ' ' * i + bslash for i in range(height // 2)]
                bracket_args.extend([ ' ' * (
                    height//2 - i - 1) + slash for i in range(height // 2)])
            # Create straight bracket
            elif bracket in {_straight_bracket, _straight_bracket_ucode}:
                bracket_args = [vert] * height
            else:
                raise ValueError(bracket)
            brackets.append(
                stringPict('\n'.join(bracket_args), baseline=height//2))
        return brackets

    def _sympystr(self, printer, *args):
        contents = self._print_contents(printer, *args)
        return '%s%s%s' % (getattr(self, 'lbracket', ""), contents, getattr(self, 'rbracket', ""))

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        # Get brackets
        pform = self._print_contents_pretty(printer, *args)
        lbracket, rbracket = self._pretty_brackets(
            pform.height(), printer._use_unicode)
        # Put together state
        pform = prettyForm(*pform.left(lbracket))
        pform = prettyForm(*pform.right(rbracket))
        return pform

    def _latex(self, printer, *args):
        contents = self._print_contents_latex(printer, *args)
        # The extra {} brackets are needed to get matplotlib's latex
        # rendered to render this properly.
        return '{%s%s%s}' % (getattr(self, 'lbracket_latex', ""), contents, getattr(self, 'rbracket_latex', ""))


class KetBase(StateBase):
    """Base class for Kets.

    This class defines the dual property and the brackets for printing. This is
    an abstract base class and you should not instantiate it directly, instead
    use Ket.
    """

    kind = KetKind

    lbracket = _straight_bracket
    rbracket = _rbracket
    lbracket_ucode = _straight_bracket_ucode
    rbracket_ucode = _rbracket_ucode
    lbracket_latex = r'\left|'
    rbracket_latex = r'\right\rangle '

    @classmethod
    def default_args(self):
        return ("psi",)

    @classmethod
    def dual_class(self):
        return BraBase

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------

    def _eval_innerproduct(self, bra, **hints):
        """Evaluate the inner product between this ket and a bra.

        This is called to compute <bra|ket>, where the ket is ``self``.

        This method will dispatch to sub-methods having the format::

            ``def _eval_innerproduct_BraClass(self, **hints):``

        Subclasses should define these methods (one for each BraClass) to
        teach the ket how to take inner products with bras.
        """
        return dispatch_method(self, '_eval_innerproduct', bra, **hints)

    def _apply_from_right_to(self, op, **options):
        """Apply an Operator to this Ket as Operator*Ket

        This method will dispatch to methods having the format::

            ``def _apply_from_right_to_OperatorName(op, **options):``

        Subclasses should define these methods (one for each OperatorName) to
        teach the Ket how to implement OperatorName*Ket

        Parameters
        ==========

        op : Operator
            The Operator that is acting on the Ket as op*Ket
        options : dict
            A dict of key/value pairs that control how the operator is applied
            to the Ket.
        """
        return dispatch_method(self, '_apply_from_right_to', op, **options)


class BraBase(StateBase):
    """Base class for Bras.

    This class defines the dual property and the brackets for printing. This
    is an abstract base class and you should not instantiate it directly,
    instead use Bra.
    """

    kind = BraKind

    lbracket = _lbracket
    rbracket = _straight_bracket
    lbracket_ucode = _lbracket_ucode
    rbracket_ucode = _straight_bracket_ucode
    lbracket_latex = r'\left\langle '
    rbracket_latex = r'\right|'

    @classmethod
    def _operators_to_state(self, ops, **options):
        state = self.dual_class()._operators_to_state(ops, **options)
        return state.dual

    def _state_to_operators(self, op_classes, **options):
        return self.dual._state_to_operators(op_classes, **options)

    def _enumerate_state(self, num_states, **options):
        dual_states = self.dual._enumerate_state(num_states, **options)
        return [x.dual for x in dual_states]

    @classmethod
    def default_args(self):
        return self.dual_class().default_args()

    @classmethod
    def dual_class(self):
        return KetBase

    def _represent(self, **options):
        """A default represent that uses the Ket's version."""
        from sympy.physics.quantum.dagger import Dagger
        return Dagger(self.dual._represent(**options))


class State(StateBase):
    """General abstract quantum state used as a base class for Ket and Bra."""
    pass


class Ket(State, KetBase):
    """A general time-independent Ket in quantum mechanics.

    Inherits from State and KetBase. This class should be used as the base
    class for all physical, time-independent Kets in a system. This class
    and its subclasses will be the main classes that users will use for
    expressing Kets in Dirac notation [1]_.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        ket. This will usually be its symbol or its quantum numbers. For
        time-dependent state, this will include the time.

    Examples
    ========

    Create a simple Ket and looking at its properties::

        >>> from sympy.physics.quantum import Ket
        >>> from sympy import symbols, I
        >>> k = Ket('psi')
        >>> k
        |psi>
        >>> k.hilbert_space
        H
        >>> k.is_commutative
        False
        >>> k.label
        (psi,)

    Ket's know about their associated bra::

        >>> k.dual
        <psi|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.state.Bra'>

    Take a linear combination of two kets::

        >>> k0 = Ket(0)
        >>> k1 = Ket(1)
        >>> 2*I*k0 - 4*k1
        2*I*|0> - 4*|1>

    Compound labels are passed as tuples::

        >>> n, m = symbols('n,m')
        >>> k = Ket(n,m)
        >>> k
        |nm>

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bra-ket_notation
    """

    @classmethod
    def dual_class(self):
        return Bra


class Bra(State, BraBase):
    """A general time-independent Bra in quantum mechanics.

    Inherits from State and BraBase. A Bra is the dual of a Ket [1]_. This
    class and its subclasses will be the main classes that users will use for
    expressing Bras in Dirac notation.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        ket. This will usually be its symbol or its quantum numbers. For
        time-dependent state, this will include the time.

    Examples
    ========

    Create a simple Bra and look at its properties::

        >>> from sympy.physics.quantum import Bra
        >>> from sympy import symbols, I
        >>> b = Bra('psi')
        >>> b
        <psi|
        >>> b.hilbert_space
        H
        >>> b.is_commutative
        False

    Bra's know about their dual Ket's::

        >>> b.dual
        |psi>
        >>> b.dual_class()
        <class 'sympy.physics.quantum.state.Ket'>

    Like Kets, Bras can have compound labels and be manipulated in a similar
    manner::

        >>> n, m = symbols('n,m')
        >>> b = Bra(n,m) - I*Bra(m,n)
        >>> b
        -I*<mn| + <nm|

    Symbols in a Bra can be substituted using ``.subs``::

        >>> b.subs(n,m)
        <mm| - I*<mm|

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bra-ket_notation
    """

    @classmethod
    def dual_class(self):
        return Ket

#-----------------------------------------------------------------------------
# Time dependent states, bras and kets.
#-----------------------------------------------------------------------------


class TimeDepState(StateBase):
    """Base class for a general time-dependent quantum state.

    This class is used as a base class for any time-dependent state. The main
    difference between this class and the time-independent state is that this
    class takes a second argument that is the time in addition to the usual
    label argument.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.
    """

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def default_args(self):
        return ("psi", "t")

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def label(self):
        """The label of the state."""
        return self.args[:-1]

    @property
    def time(self):
        """The time of the state."""
        return self.args[-1]

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _print_time(self, printer, *args):
        return printer._print(self.time, *args)

    _print_time_repr = _print_time
    _print_time_latex = _print_time

    def _print_time_pretty(self, printer, *args):
        pform = printer._print(self.time, *args)
        return pform

    def _print_contents(self, printer, *args):
        label = self._print_label(printer, *args)
        time = self._print_time(printer, *args)
        return '%s;%s' % (label, time)

    def _print_label_repr(self, printer, *args):
        label = self._print_sequence(self.label, ',', printer, *args)
        time = self._print_time_repr(printer, *args)
        return '%s,%s' % (label, time)

    def _print_contents_pretty(self, printer, *args):
        label = self._print_label_pretty(printer, *args)
        time = self._print_time_pretty(printer, *args)
        return printer._print_seq((label, time), delimiter=';')

    def _print_contents_latex(self, printer, *args):
        label = self._print_sequence(
            self.label, self._label_separator, printer, *args)
        time = self._print_time_latex(printer, *args)
        return '%s;%s' % (label, time)


class TimeDepKet(TimeDepState, KetBase):
    """General time-dependent Ket in quantum mechanics.

    This inherits from ``TimeDepState`` and ``KetBase`` and is the main class
    that should be used for Kets that vary with time. Its dual is a
    ``TimeDepBra``.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.

    Examples
    ========

    Create a TimeDepKet and look at its attributes::

        >>> from sympy.physics.quantum import TimeDepKet
        >>> k = TimeDepKet('psi', 't')
        >>> k
        |psi;t>
        >>> k.time
        t
        >>> k.label
        (psi,)
        >>> k.hilbert_space
        H

    TimeDepKets know about their dual bra::

        >>> k.dual
        <psi;t|
        >>> k.dual_class()
        <class 'sympy.physics.quantum.state.TimeDepBra'>
    """

    @classmethod
    def dual_class(self):
        return TimeDepBra


class TimeDepBra(TimeDepState, BraBase):
    """General time-dependent Bra in quantum mechanics.

    This inherits from TimeDepState and BraBase and is the main class that
    should be used for Bras that vary with time. Its dual is a TimeDepBra.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the ket. This
        will usually be its symbol or its quantum numbers. For time-dependent
        state, this will include the time as the final argument.

    Examples
    ========

        >>> from sympy.physics.quantum import TimeDepBra
        >>> b = TimeDepBra('psi', 't')
        >>> b
        <psi;t|
        >>> b.time
        t
        >>> b.label
        (psi,)
        >>> b.hilbert_space
        H
        >>> b.dual
        |psi;t>
    """

    @classmethod
    def dual_class(self):
        return TimeDepKet


class OrthogonalState(State):
    """General abstract quantum state used as a base class for Ket and Bra."""
    pass

class OrthogonalKet(OrthogonalState, KetBase):
    """Orthogonal Ket in quantum mechanics.

    The inner product of two states with different labels will give zero,
    states with the same label will give one.

        >>> from sympy.physics.quantum import OrthogonalBra, OrthogonalKet
        >>> from sympy.abc import m, n
        >>> (OrthogonalBra(n)*OrthogonalKet(n)).doit()
        1
        >>> (OrthogonalBra(n)*OrthogonalKet(n+1)).doit()
        0
        >>> (OrthogonalBra(n)*OrthogonalKet(m)).doit()
        <n|m>
    """

    @classmethod
    def dual_class(self):
        return OrthogonalBra

    def _eval_innerproduct(self, bra, **hints):

        if len(self.args) != len(bra.args):
            raise ValueError('Cannot multiply a ket that has a different number of labels.')

        for arg, bra_arg in zip(self.args, bra.args):
            diff = arg - bra_arg
            diff = diff.expand()

            is_zero = diff.is_zero

            if is_zero is False:
                return S.Zero # i.e. Integer(0)

            if is_zero is None:
                return None

        return S.One # i.e. Integer(1)


class OrthogonalBra(OrthogonalState, BraBase):
    """Orthogonal Bra in quantum mechanics.
    """

    @classmethod
    def dual_class(self):
        return OrthogonalKet


class Wavefunction(Function):
    """Class for representations in continuous bases

    This class takes an expression and coordinates in its constructor. It can
    be used to easily calculate normalizations and probabilities.

    Parameters
    ==========

    expr : Expr
           The expression representing the functional form of the w.f.

    coords : Symbol or tuple
           The coordinates to be integrated over, and their bounds

    Examples
    ========

    Particle in a box, specifying bounds in the more primitive way of using
    Piecewise:

        >>> from sympy import Symbol, Piecewise, pi, N
        >>> from sympy.functions import sqrt, sin
        >>> from sympy.physics.quantum.state import Wavefunction
        >>> x = Symbol('x', real=True)
        >>> n = 1
        >>> L = 1
        >>> g = Piecewise((0, x < 0), (0, x > L), (sqrt(2//L)*sin(n*pi*x/L), True))
        >>> f = Wavefunction(g, x)
        >>> f.norm
        1
        >>> f.is_normalized
        True
        >>> p = f.prob()
        >>> p(0)
        0
        >>> p(L)
        0
        >>> p(0.5)
        2
        >>> p(0.85*L)
        2*sin(0.85*pi)**2
        >>> N(p(0.85*L))
        0.412214747707527

    Additionally, you can specify the bounds of the function and the indices in
    a more compact way:

        >>> from sympy import symbols, pi, diff
        >>> from sympy.functions import sqrt, sin
        >>> from sympy.physics.quantum.state import Wavefunction
        >>> x, L = symbols('x,L', positive=True)
        >>> n = symbols('n', integer=True, positive=True)
        >>> g = sqrt(2/L)*sin(n*pi*x/L)
        >>> f = Wavefunction(g, (x, 0, L))
        >>> f.norm
        1
        >>> f(L+1)
        0
        >>> f(L-1)
        sqrt(2)*sin(pi*n*(L - 1)/L)/sqrt(L)
        >>> f(-1)
        0
        >>> f(0.85)
        sqrt(2)*sin(0.85*pi*n/L)/sqrt(L)
        >>> f(0.85, n=1, L=1)
        sqrt(2)*sin(0.85*pi)
        >>> f.is_commutative
        False

    All arguments are automatically sympified, so you can define the variables
    as strings rather than symbols:

        >>> expr = x**2
        >>> f = Wavefunction(expr, 'x')
        >>> type(f.variables[0])
        <class 'sympy.core.symbol.Symbol'>

    Derivatives of Wavefunctions will return Wavefunctions:

        >>> diff(f, x)
        Wavefunction(2*x, x)

    """

    #Any passed tuples for coordinates and their bounds need to be
    #converted to Tuples before Function's constructor is called, to
    #avoid errors from calling is_Float in the constructor
    def __new__(cls, *args, **options):
        new_args = [None for i in args]
        ct = 0
        for arg in args:
            if isinstance(arg, tuple):
                new_args[ct] = Tuple(*arg)
            else:
                new_args[ct] = arg
            ct += 1

        return super().__new__(cls, *new_args, **options)

    def __call__(self, *args, **options):
        var = self.variables

        if len(args) != len(var):
            raise NotImplementedError(
                "Incorrect number of arguments to function!")

        ct = 0
        #If the passed value is outside the specified bounds, return 0
        for v in var:
            lower, upper = self.limits[v]

            #Do the comparison to limits only if the passed symbol is actually
            #a symbol present in the limits;
            #Had problems with a comparison of x > L
            if isinstance(args[ct], Expr) and \
                not (lower in args[ct].free_symbols
                     or upper in args[ct].free_symbols):
                continue

            if (args[ct] < lower) == True or (args[ct] > upper) == True:
                return S.Zero

            ct += 1

        expr = self.expr

        #Allows user to make a call like f(2, 4, m=1, n=1)
        for symbol in list(expr.free_symbols):
            if str(symbol) in options.keys():
                val = options[str(symbol)]
                expr = expr.subs(symbol, val)

        return expr.subs(zip(var, args))

    def _eval_derivative(self, symbol):
        expr = self.expr
        deriv = expr._eval_derivative(symbol)

        return Wavefunction(deriv, *self.args[1:])

    def _eval_conjugate(self):
        return Wavefunction(conjugate(self.expr), *self.args[1:])

    def _eval_transpose(self):
        return self

    @property
    def is_commutative(self):
        """
        Override Function's is_commutative so that order is preserved in
        represented expressions
        """
        return False

    @classmethod
    def eval(self, *args):
        return None

    @property
    def variables(self):
        """
        Return the coordinates which the wavefunction depends on

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x,y = symbols('x,y')
            >>> f = Wavefunction(x*y, x, y)
            >>> f.variables
            (x, y)
            >>> g = Wavefunction(x*y, x)
            >>> g.variables
            (x,)

        """
        var = [g[0] if isinstance(g, Tuple) else g for g in self._args[1:]]
        return tuple(var)

    @property
    def limits(self):
        """
        Return the limits of the coordinates which the w.f. depends on If no
        limits are specified, defaults to ``(-oo, oo)``.

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x, y = symbols('x, y')
            >>> f = Wavefunction(x**2, (x, 0, 1))
            >>> f.limits
            {x: (0, 1)}
            >>> f = Wavefunction(x**2, x)
            >>> f.limits
            {x: (-oo, oo)}
            >>> f = Wavefunction(x**2 + y**2, x, (y, -1, 2))
            >>> f.limits
            {x: (-oo, oo), y: (-1, 2)}

        """
        limits = [(g[1], g[2]) if isinstance(g, Tuple) else (-oo, oo)
                  for g in self._args[1:]]
        return dict(zip(self.variables, tuple(limits)))

    @property
    def expr(self):
        """
        Return the expression which is the functional form of the Wavefunction

        Examples
        ========

            >>> from sympy.physics.quantum.state import Wavefunction
            >>> from sympy import symbols
            >>> x, y = symbols('x, y')
            >>> f = Wavefunction(x**2, x)
            >>> f.expr
            x**2

        """
        return self._args[0]

    @property
    def is_normalized(self):
        """
        Returns true if the Wavefunction is properly normalized

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sqrt, sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sqrt(2/L)*sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.is_normalized
            True

        """

        return equal_valued(self.norm, 1)

    @property  # type: ignore
    @cacheit
    def norm(self):
        """
        Return the normalization of the specified functional form.

        This function integrates over the coordinates of the Wavefunction, with
        the bounds specified.

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sqrt, sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sqrt(2/L)*sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.norm
            1
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.norm
            sqrt(2)*sqrt(L)/2

        """

        exp = self.expr*conjugate(self.expr)
        var = self.variables
        limits = self.limits

        for v in var:
            curr_limits = limits[v]
            exp = integrate(exp, (v, curr_limits[0], curr_limits[1]))

        return sqrt(exp)

    def normalize(self):
        """
        Return a normalized version of the Wavefunction

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x = symbols('x', real=True)
            >>> L = symbols('L', positive=True)
            >>> n = symbols('n', integer=True, positive=True)
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.normalize()
            Wavefunction(sqrt(2)*sin(pi*n*x/L)/sqrt(L), (x, 0, L))

        """
        const = self.norm

        if const is oo:
            raise NotImplementedError("The function is not normalizable!")
        else:
            return Wavefunction((const)**(-1)*self.expr, *self.args[1:])

    def prob(self):
        r"""
        Return the absolute magnitude of the w.f., `|\psi(x)|^2`

        Examples
        ========

            >>> from sympy import symbols, pi
            >>> from sympy.functions import sin
            >>> from sympy.physics.quantum.state import Wavefunction
            >>> x, L = symbols('x,L', real=True)
            >>> n = symbols('n', integer=True)
            >>> g = sin(n*pi*x/L)
            >>> f = Wavefunction(g, (x, 0, L))
            >>> f.prob()
            Wavefunction(sin(pi*n*x/L)**2, x)

        """

        return Wavefunction(self.expr*conjugate(self.expr), *self.variables)
<|endoftext|>"""Kinds for Operators, Bras, and Kets.

This module defines kinds for operators, bras, and kets. These are useful
in various places in ``sympy.physics.quantum`` as you often want to know
what the kind is of a compound expression. For example, if you multiply
an operator, bra, or ket by a number, you get back another operator, bra,
or ket - even though if you did an ``isinstance`` check you would find that
you have a ``Mul`` instead. The kind system is meant to give you a quick
way of determining how a compound expression behaves in terms of lower
level kinds.

The resolution calculation of kinds for compound expressions can be found
either in container classes or in functions that are registered with
kind dispatchers.
"""

from sympy.core.mul import Mul
from sympy.core.kind import Kind, _NumberKind


__all__ = [
    '_KetKind',
    'KetKind',
    '_BraKind',
    'BraKind',
    '_OperatorKind',
    'OperatorKind',
]


class _KetKind(Kind):
    """A kind for quantum kets."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "KetKind"

# Create an instance as many situations need this.
KetKind = _KetKind()


class _BraKind(Kind):
    """A kind for quantum bras."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "BraKind"

# Create an instance as many situations need this.
BraKind = _BraKind()


from sympy.core.kind import Kind

class _OperatorKind(Kind):
    """A kind for quantum operators."""

    def __new__(cls):
        obj = super().__new__(cls)
        return obj

    def __repr__(self):
        return "OperatorKind"

# Create an instance as many situations need this.
OperatorKind = _OperatorKind()


#-----------------------------------------------------------------------------
# Kind resolution.
#-----------------------------------------------------------------------------

# Note: We can't currently add kind dispatchers for the following combinations
#       as the Mul._kind_dispatcher is set to commutative and will also
#       register the opposite order, which isn't correct for these pairs:
#
# 1. (_OperatorKind, _KetKind)
# 2. (_BraKind, _OperatorKind)
# 3. (_BraKind, _KetKind)


@Mul._kind_dispatcher.register(_NumberKind, _KetKind)
def _mul_number_ket_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*KetKind -> KetKind."""
    return KetKind


@Mul._kind_dispatcher.register(_NumberKind, _BraKind)
def _mul_number_bra_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*BraKind -> BraKind."""
    return BraKind


@Mul._kind_dispatcher.register(_NumberKind, _OperatorKind)
def _mul_operator_kind(lhs, rhs):
    """Perform the kind calculation of NumberKind*OperatorKind -> OperatorKind."""
    return OperatorKind
<|endoftext|>from itertools import product

from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import HermitianOperator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import numpy_ndarray, scipy_sparse_matrix, to_numpy
from sympy.physics.quantum.trace import Tr


class Density(HermitianOperator):
    """Density operator for representing mixed states.

    TODO: Density operator support for Qubits

    Parameters
    ==========

    values : tuples/lists
    Each tuple/list should be of form (state, prob) or [state,prob]

    Examples
    ========

    Create a density operator with 2 states represented by Kets.

    >>> from sympy.physics.quantum.state import Ket
    >>> from sympy.physics.quantum.density import Density
    >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
    >>> d
    Density((|0>, 0.5),(|1>, 0.5))

    """
    @classmethod
    def _eval_args(cls, args):
        # call this to qsympify the args
        args = super()._eval_args(args)

        for arg in args:
            # Check if arg is a tuple
            if not (isinstance(arg, Tuple) and len(arg) == 2):
                raise ValueError("Each argument should be of form [state,prob]"
                                 " or ( state, prob )")

        return args

    def states(self):
        """Return list of all states.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()
        (|0>, |1>)

        """
        return Tuple(*[arg[0] for arg in self.args])

    def probs(self):
        """Return list of all probabilities.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()
        (0.5, 0.5)

        """
        return Tuple(*[arg[1] for arg in self.args])

    def get_state(self, index):
        """Return specific state by index.

        Parameters
        ==========

        index : index of state to be returned

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.states()[1]
        |1>

        """
        state = self.args[index][0]
        return state

    def get_prob(self, index):
        """Return probability of specific state by index.

        Parameters
        ===========

        index : index of states whose probability is returned.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.probs()[1]
        0.500000000000000

        """
        prob = self.args[index][1]
        return prob

    def apply_op(self, op):
        """op will operate on each individual state.

        Parameters
        ==========

        op : Operator

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.apply_op(A)
        Density((A*|0>, 0.5),(A*|1>, 0.5))

        """
        new_args = [(op*state, prob) for (state, prob) in self.args]
        return Density(*new_args)

    def doit(self, **hints):
        """Expand the density operator into an outer product format.

        Examples
        ========

        >>> from sympy.physics.quantum.state import Ket
        >>> from sympy.physics.quantum.density import Density
        >>> from sympy.physics.quantum.operator import Operator
        >>> A = Operator('A')
        >>> d = Density([Ket(0), 0.5], [Ket(1),0.5])
        >>> d.doit()
        0.5*|0><0| + 0.5*|1><1|

        """

        terms = []
        for (state, prob) in self.args:
            state = state.expand()  # needed to break up (a+b)*c
            if (isinstance(state, Add)):
                for arg in product(state.args, repeat=2):
                    terms.append(prob*self._generate_outer_prod(arg[0],
                                                                arg[1]))
            else:
                terms.append(prob*self._generate_outer_prod(state, state))

        return Add(*terms)

    def _generate_outer_prod(self, arg1, arg2):
        c_part1, nc_part1 = arg1.args_cnc()
        c_part2, nc_part2 = arg2.args_cnc()

        if (len(nc_part1) == 0 or len(nc_part2) == 0):
            raise ValueError('Atleast one-pair of'
                             ' Non-commutative instance required'
                             ' for outer product.')

        # We were able to remove some tensor product simplifications that
        # used to be here as those transformations are not automatically
        # applied by transforms.py.
        op = Mul(*nc_part1)*Dagger(Mul(*nc_part2))

        return Mul(*c_part1)*Mul(*c_part2) * op

    def _represent(self, **options):
        return represent(self.doit(), **options)

    def _print_operator_name_latex(self, printer, *args):
        return r'\rho'

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm('\N{GREEK SMALL LETTER RHO}')

    def _eval_trace(self, **kwargs):
        indices = kwargs.get('indices', [])
        return Tr(self.doit(), indices).doit()

    def entropy(self):
        """ Compute the entropy of a density matrix.

        Refer to density.entropy() method  for examples.
        """
        return entropy(self)


def entropy(density):
    """Compute the entropy of a matrix/density object.

    This computes -Tr(density*ln(density)) using the eigenvalue decomposition
    of density, which is given as either a Density instance or a matrix
    (numpy.ndarray, sympy.Matrix or scipy.sparse).

    Parameters
    ==========

    density : density matrix of type Density, SymPy matrix,
    scipy.sparse or numpy.ndarray

    Examples
    ========

    >>> from sympy.physics.quantum.density import Density, entropy
    >>> from sympy.physics.quantum.spin import JzKet
    >>> from sympy import S
    >>> up = JzKet(S(1)/2,S(1)/2)
    >>> down = JzKet(S(1)/2,-S(1)/2)
    >>> d = Density((up,S(1)/2),(down,S(1)/2))
    >>> entropy(d)
    log(2)/2

    """
    if isinstance(density, Density):
        density = represent(density)  # represent in Matrix

    if isinstance(density, scipy_sparse_matrix):
        density = to_numpy(density)

    if isinstance(density, Matrix):
        eigvals = density.eigenvals().keys()
        return expand(-sum(e*log(e) for e in eigvals))
    elif isinstance(density, numpy_ndarray):
        import numpy as np
        eigvals = np.linalg.eigvals(density)
        return -np.sum(eigvals*np.log(eigvals))
    else:
        raise ValueError(
            "numpy.ndarray, scipy.sparse or SymPy matrix expected")


def fidelity(state1, state2):
    """ Computes the fidelity [1]_ between two quantum states

    The arguments provided to this function should be a square matrix or a
    Density object. If it is a square matrix, it is assumed to be diagonalizable.

    Parameters
    ==========

    state1, state2 : a density matrix or Matrix


    Examples
    ========

    >>> from sympy import S, sqrt
    >>> from sympy.physics.quantum.dagger import Dagger
    >>> from sympy.physics.quantum.spin import JzKet
    >>> from sympy.physics.quantum.density import fidelity
    >>> from sympy.physics.quantum.represent import represent
    >>>
    >>> up = JzKet(S(1)/2,S(1)/2)
    >>> down = JzKet(S(1)/2,-S(1)/2)
    >>> amp = 1/sqrt(2)
    >>> updown = (amp*up) + (amp*down)
    >>>
    >>> # represent turns Kets into matrices
    >>> up_dm = represent(up*Dagger(up))
    >>> down_dm = represent(down*Dagger(down))
    >>> updown_dm = represent(updown*Dagger(updown))
    >>>
    >>> fidelity(up_dm, up_dm)
    1
    >>> fidelity(up_dm, down_dm) #orthogonal states
    0
    >>> fidelity(up_dm, updown_dm).evalf().round(3)
    0.707

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fidelity_of_quantum_states

    """
    state1 = represent(state1) if isinstance(state1, Density) else state1
    state2 = represent(state2) if isinstance(state2, Density) else state2

    if not isinstance(state1, Matrix) or not isinstance(state2, Matrix):
        raise ValueError("state1 and state2 must be of type Density or Matrix "
                         "received type=%s for state1 and type=%s for state2" %
                         (type(state1), type(state2)))

    if state1.shape != state2.shape and state1.is_square:
        raise ValueError("The dimensions of both args should be equal and the "
                         "matrix obtained should be a square matrix")

    sqrt_state1 = state1**S.Half
    return Tr((sqrt_state1*state2*sqrt_state1)**S.Half).doit()
<|endoftext|>"""Quantum mechanical angular momentum."""

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import int_valued
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol

from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
                                            UnitaryOperator)
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply


__all__ = [
    'm_values',
    'Jplus',
    'Jminus',
    'Jx',
    'Jy',
    'Jz',
    'J2',
    'Rotation',
    'WignerD',
    'JxKet',
    'JxBra',
    'JyKet',
    'JyBra',
    'JzKet',
    'JzBra',
    'JzOp',
    'J2Op',
    'JxKetCoupled',
    'JxBraCoupled',
    'JyKetCoupled',
    'JyBraCoupled',
    'JzKetCoupled',
    'JzBraCoupled',
    'couple',
    'uncouple'
]


def m_values(j):
    j = sympify(j)
    size = 2*j + 1
    if not size.is_Integer or not size > 0:
        raise ValueError(
            'Only integer or half-integer values allowed for j, got: : %r' % j
        )
    return size, [j - i for i in range(int(2*j + 1))]


#-----------------------------------------------------------------------------
# Spin Operators
#-----------------------------------------------------------------------------


class SpinOpBase:
    """Base class for spin operators."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        # We consider all j values so our space is infinite.
        return ComplexSpace(S.Infinity)

    @property
    def name(self):
        return self.args[0]

    def _print_contents(self, printer, *args):
        return '%s%s' % (self.name, self._coord)

    def _print_contents_pretty(self, printer, *args):
        a = stringPict(str(self.name))
        b = stringPict(self._coord)
        return self._print_subscript_pretty(a, b)

    def _print_contents_latex(self, printer, *args):
        return r'%s_%s' % ((self.name, self._coord))

    def _represent_base(self, basis, **options):
        j = options.get('j', S.Half)
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                result[p, q] = me
        return result

    def _apply_op(self, ket, orig_basis, **options):
        state = ket.rewrite(self.basis)
        # If the state has only one term
        if isinstance(state, State):
            ret = (hbar*state.m)*state
        # state is a linear combination of states
        elif isinstance(state, Sum):
            ret = self._apply_operator_Sum(state, **options)
        else:
            ret = qapply(self*state)
        if ret == self*state:
            raise NotImplementedError
        return ret.rewrite(orig_basis)

    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_TensorProduct(self, tp, **options):
        # Uncoupling operator is only easily found for coordinate basis spin operators
        # TODO: add methods for uncoupling operators
        if not isinstance(self, (JxOp, JyOp, JzOp)):
            raise NotImplementedError
        result = []
        for n in range(len(tp.args)):
            arg = []
            arg.extend(tp.args[:n])
            arg.append(self._apply_operator(tp.args[n]))
            arg.extend(tp.args[n + 1:])
            result.append(tp.__class__(*arg))
        return Add(*result).expand()

    # TODO: move this to qapply_Mul
    def _apply_operator_Sum(self, s, **options):
        new_func = qapply(self*s.function)
        if new_func == self*s.function:
            raise NotImplementedError
        return Sum(new_func, *s.limits)

    def _eval_trace(self, **options):
        #TODO: use options to use different j values
        #For now eval at default basis

        # is it efficient to represent each time
        # to do a trace?
        return self._represent_default_basis().trace()


class JplusOp(SpinOpBase, Operator):
    """The J+ operator."""

    _coord = '+'

    basis = 'Jz'

    def _eval_commutator_JminusOp(self, other):
        return 2*hbar*JzOp(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*JzKet(j, m + S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*JzKetCoupled(j, m + S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*sqrt(j*(j + S.One) - mp*(mp + S.One))
        result *= KroneckerDelta(m, mp + 1)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) + I*JyOp(args[0])


class JminusOp(SpinOpBase, Operator):
    """The J- operator."""

    _coord = '-'

    basis = 'Jz'

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*JzKet(j, m - S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*JzKetCoupled(j, m - S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*sqrt(j*(j + S.One) - mp*(mp - S.One))
        result *= KroneckerDelta(m, mp - 1)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) - I*JyOp(args[0])


class JxOp(SpinOpBase, HermitianOperator):
    """The Jx operator."""

    _coord = 'x'

    basis = 'Jx'

    def _eval_commutator_JyOp(self, other):
        return I*hbar*JzOp(self.name)

    def _eval_commutator_JzOp(self, other):
        return -I*hbar*JyOp(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKet(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKet(ket, **options)
        return (jp + jm)/Integer(2)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        return (jp + jm)/Integer(2)

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        jp = JplusOp(self.name)._represent_JzOp(basis, **options)
        jm = JminusOp(self.name)._represent_JzOp(basis, **options)
        return (jp + jm)/Integer(2)

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        return (JplusOp(args[0]) + JminusOp(args[0]))/2


class JyOp(SpinOpBase, HermitianOperator):
    """The Jy operator."""

    _coord = 'y'

    basis = 'Jy'

    def _eval_commutator_JzOp(self, other):
        return I*hbar*JxOp(self.name)

    def _eval_commutator_JxOp(self, other):
        return -I*hbar*J2Op(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKet(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKet(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        jp = JplusOp(self.name)._represent_JzOp(basis, **options)
        jm = JminusOp(self.name)._represent_JzOp(basis, **options)
        return (jp - jm)/(Integer(2)*I)

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        return (JplusOp(args[0]) - JminusOp(args[0]))/(2*I)


class JzOp(SpinOpBase, HermitianOperator):
    """The Jz operator."""

    _coord = 'z'

    basis = 'Jz'

    def _eval_commutator_JxOp(self, other):
        return I*hbar*JyOp(self.name)

    def _eval_commutator_JyOp(self, other):
        return -I*hbar*JxOp(self.name)

    def _eval_commutator_JplusOp(self, other):
        return hbar*JplusOp(self.name)

    def _eval_commutator_JminusOp(self, other):
        return -hbar*JminusOp(self.name)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*mp
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)


class J2Op(SpinOpBase, HermitianOperator):
    """The J^2 operator."""

    _coord = '2'

    def _eval_commutator_JxOp(self, other):
        return S.Zero

    def _eval_commutator_JyOp(self, other):
        return S.Zero

    def _eval_commutator_JzOp(self, other):
        return S.Zero

    def _eval_commutator_JplusOp(self, other):
        return S.Zero

    def _eval_commutator_JminusOp(self, other):
        return S.Zero

    def _apply_operator_JxKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JxKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JyKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JyKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def matrix_element(self, j, m, jp, mp):
        result = (hbar**2)*j*(j + 1)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _print_contents_pretty(self, printer, *args):
        a = prettyForm(str(self.name))
        b = prettyForm('2')
        return a**b

    def _print_contents_latex(self, printer, *args):
        return r'%s^2' % str(self.name)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0])**2 + JyOp(args[0])**2 + JzOp(args[0])**2

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        a = args[0]
        return JzOp(a)**2 + \
            S.Half*(JplusOp(a)*JminusOp(a) + JminusOp(a)*JplusOp(a))


class Rotation(UnitaryOperator):
    """Wigner D operator in terms of Euler angles.

    Defines the rotation operator in terms of the Euler angles defined by
    the z-y-z convention for a passive transformation. That is the coordinate
    axes are rotated first about the z-axis, giving the new x'-y'-z' axes. Then
    this new coordinate system is rotated about the new y'-axis, giving new
    x''-y''-z'' axes. Then this new coordinate system is rotated about the
    z''-axis. Conventions follow those laid out in [1]_.

    Parameters
    ==========

    alpha : Number, Symbol
        First Euler Angle
    beta : Number, Symbol
        Second Euler angle
    gamma : Number, Symbol
        Third Euler angle

    Examples
    ========

    A simple example rotation operator:

        >>> from sympy import pi
        >>> from sympy.physics.quantum.spin import Rotation
        >>> Rotation(pi, 0, pi/2)
        R(pi,0,pi/2)

    With symbolic Euler angles and calculating the inverse rotation operator:

        >>> from sympy import symbols
        >>> a, b, c = symbols('a b c')
        >>> Rotation(a, b, c)
        R(a,b,c)
        >>> Rotation(a, b, c).inverse()
        R(-c,-b,-a)

    See Also
    ========

    WignerD: Symbolic Wigner-D function
    D: Wigner-D function
    d: Wigner small-d function

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    @classmethod
    def _eval_args(cls, args):
        args = QExpr._eval_args(args)
        if len(args) != 3:
            raise ValueError('3 Euler angles required, got: %r' % args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, label):
        # We consider all j values so our space is infinite.
        return ComplexSpace(S.Infinity)

    @property
    def alpha(self):
        return self.label[0]

    @property
    def beta(self):
        return self.label[1]

    @property
    def gamma(self):
        return self.label[2]

    def _print_operator_name(self, printer, *args):
        return 'R'

    def _print_operator_name_pretty(self, printer, *args):
        if printer._use_unicode:
            return prettyForm('\N{SCRIPT CAPITAL R}' + ' ')
        else:
            return prettyForm("R ")

    def _print_operator_name_latex(self, printer, *args):
        return r'\mathcal{R}'

    def _eval_inverse(self):
        return Rotation(-self.gamma, -self.beta, -self.alpha)

    @classmethod
    def D(cls, j, m, mp, alpha, beta, gamma):
        """Wigner D-function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        alpha : Number, Symbol
            First Euler angle of rotation
        beta : Number, Symbol
            Second Euler angle of rotation
        gamma : Number, Symbol
            Third Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> alpha, beta, gamma = symbols('alpha beta gamma')
            >>> Rotation.D(1, 1, 0,pi, pi/2,-pi)
            WignerD(1, 1, 0, pi, pi/2, -pi)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        return WignerD(j, m, mp, alpha, beta, gamma)

    @classmethod
    def d(cls, j, m, mp, beta):
        """Wigner small-d function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters with the alpha and gamma angles
        given as 0.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        beta : Number, Symbol
            Second Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> beta = symbols('beta')
            >>> Rotation.d(1, 1, 0, pi/2)
            WignerD(1, 1, 0, 0, pi/2, 0)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        return WignerD(j, m, mp, 0, beta, 0)

    def matrix_element(self, j, m, jp, mp):
        result = self.__class__.D(
            jp, m, mp, self.alpha, self.beta, self.gamma
        )
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_base(self, basis, **options):
        j = sympify(options.get('j', S.Half))
        # TODO: move evaluation up to represent function/implement elsewhere
        evaluate = sympify(options.get('doit'))
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                if evaluate:
                    result[p, q] = me.doit()
                else:
                    result[p, q] = me
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _apply_operator_uncoupled(self, state, ket, *, dummy=True, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z*state(j, mp))
            return Add(*s)
        else:
            if dummy:
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g)*state(j, mp), (mp, -j, j))

    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_operator_uncoupled(JxKet, ket, **options)

    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_operator_uncoupled(JyKet, ket, **options)

    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_operator_uncoupled(JzKet, ket, **options)

    def _apply_operator_coupled(self, state, ket, *, dummy=True, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z*state(j, mp, jn, coupling))
            return Add(*s)
        else:
            if dummy:
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g)*state(
                j, mp, jn, coupling), (mp, -j, j))

    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JxKetCoupled, ket, **options)

    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JyKetCoupled, ket, **options)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JzKetCoupled, ket, **options)

class WignerD(Expr):
    r"""Wigner-D function

    The Wigner D-function gives the matrix elements of the rotation
    operator in the jm-representation. For the Euler angles `\alpha`,
    `\beta`, `\gamma`, the D-function is defined such that:

    .. math ::
        <j,m| \mathcal{R}(\alpha, \beta, \gamma ) |j',m'> = \delta_{jj'} D(j, m, m', \alpha, \beta, \gamma)

    Where the rotation operator is as defined by the Rotation class [1]_.

    The Wigner D-function defined in this way gives:

    .. math ::
        D(j, m, m', \alpha, \beta, \gamma) = e^{-i m \alpha} d(j, m, m', \beta) e^{-i m' \gamma}

    Where d is the Wigner small-d function, which is given by Rotation.d.

    The Wigner small-d function gives the component of the Wigner
    D-function that is determined by the second Euler angle. That is the
    Wigner D-function is:

    .. math ::
        D(j, m, m', \alpha, \beta, \gamma) = e^{-i m \alpha} d(j, m, m', \beta) e^{-i m' \gamma}

    Where d is the small-d function. The Wigner D-function is given by
    Rotation.D.

    Note that to evaluate the D-function, the j, m and mp parameters must
    be integer or half integer numbers.

    Parameters
    ==========

    j : Number
        Total angular momentum
    m : Number
        Eigenvalue of angular momentum along axis after rotation
    mp : Number
        Eigenvalue of angular momentum along rotated axis
    alpha : Number, Symbol
        First Euler angle of rotation
    beta : Number, Symbol
        Second Euler angle of rotation
    gamma : Number, Symbol
        Third Euler angle of rotation

    Examples
    ========

    Evaluate the Wigner-D matrix elements of a simple rotation:

        >>> from sympy.physics.quantum.spin import Rotation
        >>> from sympy import pi
        >>> rot = Rotation.D(1, 1, 0, pi, pi/2, 0)
        >>> rot
        WignerD(1, 1, 0, pi, pi/2, 0)
        >>> rot.doit()
        sqrt(2)/2

    Evaluate the Wigner-d matrix elements of a simple rotation

        >>> rot = Rotation.d(1, 1, 0, pi/2)
        >>> rot
        WignerD(1, 1, 0, 0, pi/2, 0)
        >>> rot.doit()
        -sqrt(2)/2

    See Also
    ========

    Rotation: Rotation operator

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    is_commutative = True

    def __new__(cls, *args, **hints):
        if not len(args) == 6:
            raise ValueError('6 parameters expected, got %s' % args)
        args = sympify(args)
        evaluate = hints.get('evaluate', False)
        if evaluate:
            return Expr.__new__(cls, *args)._eval_wignerd()
        return Expr.__new__(cls, *args)

    @property
    def j(self):
        return self.args[0]

    @property
    def m(self):
        return self.args[1]

    @property
    def mp(self):
        return self.args[2]

    @property
    def alpha(self):
        return self.args[3]

    @property
    def beta(self):
        return self.args[4]

    @property
    def gamma(self):
        return self.args[5]

    def _latex(self, printer, *args):
        if self.alpha == 0 and self.gamma == 0:
            return r'd^{%s}_{%s,%s}\left(%s\right)' % \
                (
                    printer._print(self.j), printer._print(
                        self.m), printer._print(self.mp),
                    printer._print(self.beta) )
        return r'D^{%s}_{%s,%s}\left(%s,%s,%s\right)' % \
            (
                printer._print(
                    self.j), printer._print(self.m), printer._print(self.mp),
                printer._print(self.alpha), printer._print(self.beta), printer._print(self.gamma) )

    def _pretty(self, printer, *args):
        top = printer._print(self.j)

        bot = printer._print(self.m)
        bot = prettyForm(*bot.right(','))
        bot = prettyForm(*bot.right(printer._print(self.mp)))

        pad = max(top.width(), bot.width())
        top = prettyForm(*top.left(' '))
        bot = prettyForm(*bot.left(' '))
        if pad > top.width():
            top = prettyForm(*top.right(' '*(pad - top.width())))
        if pad > bot.width():
            bot = prettyForm(*bot.right(' '*(pad - bot.width())))
        if self.alpha == 0 and self.gamma == 0:
            args = printer._print(self.beta)
            s = stringPict('d' + ' '*pad)
        else:
            args = printer._print(self.alpha)
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.beta)))
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.gamma)))

            s = stringPict('D' + ' '*pad)

        args = prettyForm(*args.parens())
        s = prettyForm(*s.above(top))
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.right(args))
        return s

    def doit(self, **hints):
        hints['evaluate'] = True
        return WignerD(*self.args, **hints)

    def _eval_wignerd(self):
        j = self.j
        m = self.m
        mp = self.mp
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        if alpha == 0 and beta == 0 and gamma == 0:
            return KroneckerDelta(m, mp)
        if not j.is_number:
            raise ValueError(
                'j parameter must be numerical to evaluate, got %s' % j)
        r = 0
        if beta == pi/2:
            # Varshalovich Equation (5), Section 4.16, page 113, setting
            # alpha=gamma=0.
            for k in range(2*j + 1):
                if k > j + mp or k > j - m or k < mp - m:
                    continue
                r += (S.NegativeOne)**k*binomial(j + mp, k)*binomial(j - mp, k + m - mp)
            r *= (S.NegativeOne)**(m - mp) / 2**j*sqrt(factorial(j + m) *
                    factorial(j - m) / (factorial(j + mp)*factorial(j - mp)))
        else:
            # Varshalovich Equation(5), Section 4.7.2, page 87, where we set
            # beta1=beta2=pi/2, and we get alpha=gamma=pi/2 and beta=phi+pi,
            # then we use the Eq. (1), Section 4.4. page 79, to simplify:
            # d(j, m, mp, beta+pi) = (-1)**(j-mp)*d(j, m, -mp, beta)
            # This happens to be almost the same as in Eq.(10), Section 4.16,
            # except that we need to substitute -mp for mp.
            size, mvals = m_values(j)
            for mpp in mvals:
                r += Rotation.d(j, m, mpp, pi/2).doit()*(cos(-mpp*beta) + I*sin(-mpp*beta))*\
                    Rotation.d(j, mpp, -mp, pi/2).doit()
            # Empirical normalization factor so results match Varshalovich
            # Tables 4.3-4.12
            # Note that this exact normalization does not follow from the
            # above equations
            r = r*I**(2*j - m - mp)*(-1)**(2*m)
            # Finally, simplify the whole expression
            r = simplify(r)
        r *= exp(-I*m*alpha)*exp(-I*mp*gamma)
        return r


Jx = JxOp('J')
Jy = JyOp('J')
Jz = JzOp('J')
J2 = J2Op('J')
Jplus = JplusOp('J')
Jminus = JminusOp('J')


#-----------------------------------------------------------------------------
# Spin States
#-----------------------------------------------------------------------------


class SpinState(State):
    """Base class for angular momentum states."""

    _label_separator = ','

    def __new__(cls, j, m):
        j = sympify(j)
        m = sympify(m)
        if j.is_number:
            if 2*j != int(2*j):
                raise ValueError(
                    'j must be integer or half-integer, got: %s' % j)
            if j < 0:
                raise ValueError('j must be >= 0, got: %s' % j)
        if m.is_number:
            if 2*m != int(2*m):
                raise ValueError(
                    'm must be integer or half-integer, got: %s' % m)
        if j.is_number and m.is_number:
            if abs(m) > j:
                raise ValueError('Allowed values for m are -j <= m <= j, got j, m: %s, %s' % (j, m))
            if int(j - m) != j - m:
                raise ValueError('Both j and m must be integer or half-integer, got j, m: %s, %s' % (j, m))
        return State.__new__(cls, j, m)

    @property
    def j(self):
        return self.label[0]

    @property
    def m(self):
        return self.label[1]

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2*label[0] + 1)

    def _represent_base(self, **options):
        j = self.j
        m = self.m
        alpha = sympify(options.get('alpha', 0))
        beta = sympify(options.get('beta', 0))
        gamma = sympify(options.get('gamma', 0))
        size, mvals = m_values(j)
        result = zeros(size, 1)
        # breaks finding angles on L930
        for p, mval in enumerate(mvals):
            if m.is_number:
                result[p, 0] = Rotation.D(
                    self.j, mval, self.m, alpha, beta, gamma).doit()
            else:
                result[p, 0] = Rotation.D(self.j, mval,
                                          self.m, alpha, beta, gamma)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBra, **options)
        return self._rewrite_basis(Jx, JxKet, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBra, **options)
        return self._rewrite_basis(Jy, JyKet, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBra, **options)
        return self._rewrite_basis(Jz, JzKet, **options)

    def _rewrite_basis(self, basis, evect, **options):
        from sympy.physics.quantum.represent import represent
        j = self.j
        args = self.args[2:]
        if j.is_number:
            if isinstance(self, CoupledSpinState):
                if j == int(j):
                    start = j**2
                else:
                    start = (2*j - 1)*(2*j + 1)/4
            else:
                start = 0
            vect = represent(self, basis=basis, **options)
            result = Add(
                *[vect[start + i]*evect(j, j - i, *args) for i in range(2*j + 1)])
            if isinstance(self, CoupledSpinState) and options.get('coupled') is False:
                return uncouple(result)
            return result
        else:
            i = 0
            mi = symbols('mi')
            # make sure not to introduce a symbol already in the state
            while self.subs(mi, 0) != self:
                i += 1
                mi = symbols('mi%d' % i)
                break
            # TODO: better way to get angles of rotation
            if isinstance(self, CoupledSpinState):
                test_args = (0, mi, (0, 0))
            else:
                test_args = (0, mi)
            if isinstance(self, Ket):
                angles = represent(
                    self.__class__(*test_args), basis=basis)[0].args[3:6]
            else:
                angles = represent(self.__class__(
                    *test_args), basis=basis)[0].args[0].args[3:6]
            if angles == (0, 0, 0):
                return self
            else:
                state = evect(j, mi, *args)
                lt = Rotation.D(j, mi, self.m, *angles)
                return Sum(lt*state, (mi, -j, j))

    def _eval_innerproduct_JxBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JxOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_JyBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JyOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_JzBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JzOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_trace(self, bra, **hints):

        # One way to implement this method is to assume the basis set k is
        # passed.
        # Then we can apply the discrete form of Trace formula here
        # Tr(|i><j| ) = \Sum_k <k|i><j|k>
        #then we do qapply() on each each inner product and sum over them.

        # OR

        # Inner product of |i><j| = Trace(Outer Product).
        # we could just use this unless there are cases when this is not true

        return (bra*self).doit()


class JxKet(SpinState, Ket):
    """Eigenket of Jx.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JxBra

    @classmethod
    def coupled_class(self):
        return JxKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JxOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(**options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(beta=pi/2, **options)


class JxBra(SpinState, Bra):
    """Eigenbra of Jx.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JxKet

    @classmethod
    def coupled_class(self):
        return JxBraCoupled


class JyKet(SpinState, Ket):
    """Eigenket of Jy.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JyBra

    @classmethod
    def coupled_class(self):
        return JyKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JyOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(**options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), beta=-pi/2, gamma=pi/2, **options)


class JyBra(SpinState, Bra):
    """Eigenbra of Jy.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JyKet

    @classmethod
    def coupled_class(self):
        return JyBraCoupled


class JzKet(SpinState, Ket):
    """Eigenket of Jz.

    Spin state which is an eigenstate of the Jz operator. Uncoupled states,
    that is states representing the interaction of multiple separate spin
    states, are defined as a tensor product of states.

    Parameters
    ==========

    j : Number, Symbol
        Total spin angular momentum
    m : Number, Symbol
        Eigenvalue of the Jz spin operator

    Examples
    ========

    *Normal States:*

    Defining simple spin states, both numerical and symbolic:

        >>> from sympy.physics.quantum.spin import JzKet, JxKet
        >>> from sympy import symbols
        >>> JzKet(1, 0)
        |1,0>
        >>> j, m = symbols('j m')
        >>> JzKet(j, m)
        |j,m>

    Rewriting the JzKet in terms of eigenkets of the Jx operator:
    Note: that the resulting eigenstates are JxKet's

        >>> JzKet(1,1).rewrite("Jx")
        |1,-1>/2 - sqrt(2)*|1,0>/2 + |1,1>/2

    Get the vector representation of a state in terms of the basis elements
    of the Jx operator:

        >>> from sympy.physics.quantum.represent import represent
        >>> from sympy.physics.quantum.spin import Jx, Jz
        >>> represent(JzKet(1,-1), basis=Jx)
        Matrix([
        [      1/2],
        [sqrt(2)/2],
        [      1/2]])

    Apply innerproducts between states:

        >>> from sympy.physics.quantum.innerproduct import InnerProduct
        >>> from sympy.physics.quantum.spin import JxBra
        >>> i = InnerProduct(JxBra(1,1), JzKet(1,1))
        >>> i
        <1,1|1,1>
        >>> i.doit()
        1/2

    *Uncoupled States:*

    Define an uncoupled state as a TensorProduct between two Jz eigenkets:

        >>> from sympy.physics.quantum.tensorproduct import TensorProduct
        >>> j1,m1,j2,m2 = symbols('j1 m1 j2 m2')
        >>> TensorProduct(JzKet(1,0), JzKet(1,1))
        |1,0>x|1,1>
        >>> TensorProduct(JzKet(j1,m1), JzKet(j2,m2))
        |j1,m1>x|j2,m2>

    A TensorProduct can be rewritten, in which case the eigenstates that make
    up the tensor product is rewritten to the new basis:

        >>> TensorProduct(JzKet(1,1),JxKet(1,1)).rewrite('Jz')
        |1,1>x|1,-1>/2 + sqrt(2)*|1,1>x|1,0>/2 + |1,1>x|1,1>/2

    The represent method for TensorProduct's gives the vector representation of
    the state. Note that the state in the product basis is the equivalent of the
    tensor product of the vector representation of the component eigenstates:

        >>> represent(TensorProduct(JzKet(1,0),JzKet(1,1)))
        Matrix([
        [0],
        [0],
        [0],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0]])
        >>> represent(TensorProduct(JzKet(1,1),JxKet(1,1)), basis=Jz)
        Matrix([
        [      1/2],
        [sqrt(2)/2],
        [      1/2],
        [        0],
        [        0],
        [        0],
        [        0],
        [        0],
        [        0]])

    See Also
    ========

    JzKetCoupled: Coupled eigenstates
    sympy.physics.quantum.tensorproduct.TensorProduct: Used to specify uncoupled states
    uncouple: Uncouples states given coupling parameters
    couple: Couples uncoupled states

    """

    @classmethod
    def dual_class(self):
        return JzBra

    @classmethod
    def coupled_class(self):
        return JzKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(beta=pi*Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(**options)


class JzBra(SpinState, Bra):
    """Eigenbra of Jz.

    See the JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JzKet

    @classmethod
    def coupled_class(self):
        return JzBraCoupled


# Method used primarily to create coupled_n and coupled_jn by __new__ in
# CoupledSpinState
# This same method is also used by the uncouple method, and is separated from
# the CoupledSpinState class to maintain consistency in defining coupling
def _build_coupled(jcoupling, length):
    n_list = [ [n + 1] for n in range(length) ]
    coupled_jn = []
    coupled_n = []
    for n1, n2, j_new in jcoupling:
        coupled_jn.append(j_new)
        coupled_n.append( (n_list[n1 - 1], n_list[n2 - 1]) )
        n_sort = sorted(n_list[n1 - 1] + n_list[n2 - 1])
        n_list[n_sort[0] - 1] = n_sort
    return coupled_n, coupled_jn


class CoupledSpinState(SpinState):
    """Base class for coupled angular momentum states."""

    def __new__(cls, j, m, jn, *jcoupling):
        # Check j and m values using SpinState
        SpinState(j, m)
        # Build and check coupling scheme from arguments
        if len(jcoupling) == 0:
            # Use default coupling scheme
            jcoupling = []
            for n in range(2, len(jn)):
                jcoupling.append( (1, n, Add(*[jn[i] for i in range(n)])) )
            jcoupling.append( (1, len(jn), j) )
        elif len(jcoupling) == 1:
            # Use specified coupling scheme
            jcoupling = jcoupling[0]
        else:
            raise TypeError("CoupledSpinState only takes 3 or 4 arguments, got: %s" % (len(jcoupling) + 3) )
        # Check arguments have correct form
        if not isinstance(jn, (list, tuple, Tuple)):
            raise TypeError('jn must be Tuple, list or tuple, got %s' %
                            jn.__class__.__name__)
        if not isinstance(jcoupling, (list, tuple, Tuple)):
            raise TypeError('jcoupling must be Tuple, list or tuple, got %s' %
                            jcoupling.__class__.__name__)
        if not all(isinstance(term, (list, tuple, Tuple)) for term in jcoupling):
            raise TypeError(
                'All elements of jcoupling must be list, tuple or Tuple')
        if not len(jn) - 1 == len(jcoupling):
            raise ValueError('jcoupling must have length of %d, got %d' %
                             (len(jn) - 1, len(jcoupling)))
        if not all(len(x) == 3 for x in jcoupling):
            raise ValueError('All elements of jcoupling must have length 3')
        # Build sympified args
        j = sympify(j)
        m = sympify(m)
        jn = Tuple( *[sympify(ji) for ji in jn] )
        jcoupling = Tuple( *[Tuple(sympify(
            n1), sympify(n2), sympify(ji)) for (n1, n2, ji) in jcoupling] )
        # Check values in coupling scheme give physical state
        if any(2*ji != int(2*ji) for ji in jn if ji.is_number):
            raise ValueError('All elements of jn must be integer or half-integer, got: %s' % jn)
        if any(n1 != int(n1) or n2 != int(n2) for (n1, n2, _) in jcoupling):
            raise ValueError('Indices in jcoupling must be integers')
        if any(n1 < 1 or n2 < 1 or n1 > len(jn) or n2 > len(jn) for (n1, n2, _) in jcoupling):
            raise ValueError('Indices must be between 1 and the number of coupled spin spaces')
        if any(2*ji != int(2*ji) for (_, _, ji) in jcoupling if ji.is_number):
            raise ValueError('All coupled j values in coupling scheme must be integer or half-integer')
        coupled_n, coupled_jn = _build_coupled(jcoupling, len(jn))
        jvals = list(jn)
        for n, (n1, n2) in enumerate(coupled_n):
            j1 = jvals[min(n1) - 1]
            j2 = jvals[min(n2) - 1]
            j3 = coupled_jn[n]
            if sympify(j1).is_number and sympify(j2).is_number and sympify(j3).is_number:
                if j1 + j2 < j3:
                    raise ValueError('All couplings must have j1+j2 >= j3, '
                        'in coupling number %d got j1,j2,j3: %d,%d,%d' % (n + 1, j1, j2, j3))
                if abs(j1 - j2) > j3:
                    raise ValueError("All couplings must have |j1+j2| <= j3, "
                        "in coupling number %d got j1,j2,j3: %d,%d,%d" % (n + 1, j1, j2, j3))
                if int_valued(j1 + j2):
                    pass
            jvals[min(n1 + n2) - 1] = j3
        if len(jcoupling) > 0 and jcoupling[-1][2] != j:
            raise ValueError('Last j value coupled together must be the final j of the state')
        # Return state
        return State.__new__(cls, j, m, jn, jcoupling)

    def _print_label(self, printer, *args):
        label = [printer._print(self.j), printer._print(self.m)]
        for i, ji in enumerate(self.jn, start=1):
            label.append('j%d=%s' % (
                i, printer._print(ji)
            ))
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            label.append('j(%s)=%s' % (
                ','.join(str(i) for i in sorted(n1 + n2)), printer._print(jn)
            ))
        return ','.join(label)

    def _print_label_pretty(self, printer, *args):
        label = [self.j, self.m]
        for i, ji in enumerate(self.jn, start=1):
            symb = 'j%d' % i
            symb = pretty_symbol(symb)
            symb = prettyForm(symb + '=')
            item = prettyForm(*symb.right(printer._print(ji)))
            label.append(item)
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join(pretty_symbol("j%d" % i)[-1] for i in sorted(n1 + n2))
            symb = prettyForm('j' + n + '=')
            item = prettyForm(*symb.right(printer._print(jn)))
            label.append(item)
        return self._print_sequence_pretty(
            label, self._label_separator, printer, *args
        )

    def _print_label_latex(self, printer, *args):
        label = [
            printer._print(self.j, *args),
            printer._print(self.m, *args)
        ]
        for i, ji in enumerate(self.jn, start=1):
            label.append('j_{%d}=%s' % (i, printer._print(ji, *args)) )
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join(str(i) for i in sorted(n1 + n2))
            label.append('j_{%s}=%s' % (n, printer._print(jn, *args)) )
        return self._label_separator.join(label)

    @property
    def jn(self):
        return self.label[2]

    @property
    def coupling(self):
        return self.label[3]

    @property
    def coupled_jn(self):
        return _build_coupled(self.label[3], len(self.label[2]))[1]

    @property
    def coupled_n(self):
        return _build_coupled(self.label[3], len(self.label[2]))[0]

    @classmethod
    def _eval_hilbert_space(cls, label):
        j = Add(*label[2])
        if j.is_number:
            return DirectSumHilbertSpace(*[ ComplexSpace(x) for x in range(int(2*j + 1), 0, -2) ])
        else:
            # TODO: Need hilbert space fix, see issue 5732
            # Desired behavior:
            #ji = symbols('ji')
            #ret = Sum(ComplexSpace(2*ji + 1), (ji, 0, j))
            # Temporary fix:
            return ComplexSpace(2*j + 1)

    def _represent_coupled_base(self, **options):
        evect = self.uncoupled_class()
        if not self.j.is_number:
            raise ValueError(
                'State must not have symbolic j value to represent')
        if not self.hilbert_space.dimension.is_number:
            raise ValueError(
                'State must not have symbolic j values to represent')
        result = zeros(self.hilbert_space.dimension, 1)
        if self.j == int(self.j):
            start = self.j**2
        else:
            start = (2*self.j - 1)*(1 + 2*self.j)/4
        result[start:start + 2*self.j + 1, 0] = evect(
            self.j, self.m)._represent_base(**options)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBraCoupled, **options)
        return self._rewrite_basis(Jx, JxKetCoupled, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBraCoupled, **options)
        return self._rewrite_basis(Jy, JyKetCoupled, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBraCoupled, **options)
        return self._rewrite_basis(Jz, JzKetCoupled, **options)


class JxKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JxBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JxKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(**options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(beta=pi/2, **options)


class JxBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JxKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JxBra


class JyKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jy.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JyBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JyKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(**options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), beta=-pi/2, gamma=pi/2, **options)


class JyBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jy.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JyKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JyBra


class JzKetCoupled(CoupledSpinState, Ket):
    r"""Coupled eigenket of Jz

    Spin state that is an eigenket of Jz which represents the coupling of
    separate spin spaces.

    The arguments for creating instances of JzKetCoupled are ``j``, ``m``,
    ``jn`` and an optional ``jcoupling`` argument. The ``j`` and ``m`` options
    are the total angular momentum quantum numbers, as used for normal states
    (e.g. JzKet).

    The other required parameter in ``jn``, which is a tuple defining the `j_n`
    angular momentum quantum numbers of the product spaces. So for example, if
    a state represented the coupling of the product basis state
    `\left|j_1,m_1\right\rangle\times\left|j_2,m_2\right\rangle`, the ``jn``
    for this state would be ``(j1,j2)``.

    The final option is ``jcoupling``, which is used to define how the spaces
    specified by ``jn`` are coupled, which includes both the order these spaces
    are coupled together and the quantum numbers that arise from these
    couplings. The ``jcoupling`` parameter itself is a list of lists, such that
    each of the sublists defines a single coupling between the spin spaces. If
    there are N coupled angular momentum spaces, that is ``jn`` has N elements,
    then there must be N-1 sublists. Each of these sublists making up the
    ``jcoupling`` parameter have length 3. The first two elements are the
    indices of the product spaces that are considered to be coupled together.
    For example, if we want to couple `j_1` and `j_4`, the indices would be 1
    and 4. If a state has already been coupled, it is referenced by the
    smallest index that is coupled, so if `j_2` and `j_4` has already been
    coupled to some `j_{24}`, then this value can be coupled by referencing it
    with index 2. The final element of the sublist is the quantum number of the
    coupled state. So putting everything together, into a valid sublist for
    ``jcoupling``, if `j_1` and `j_2` are coupled to an angular momentum space
    with quantum number `j_{12}` with the value ``j12``, the sublist would be
    ``(1,2,j12)``, N-1 of these sublists are used in the list for
    ``jcoupling``.

    Note the ``jcoupling`` parameter is optional, if it is not specified, the
    default coupling is taken. This default value is to coupled the spaces in
    order and take the quantum number of the coupling to be the maximum value.
    For example, if the spin spaces are `j_1`, `j_2`, `j_3`, `j_4`, then the
    default coupling couples `j_1` and `j_2` to `j_{12}=j_1+j_2`, then,
    `j_{12}` and `j_3` are coupled to `j_{123}=j_{12}+j_3`, and finally
    `j_{123}` and `j_4` to `j=j_{123}+j_4`. The jcoupling value that would
    correspond to this is:

        ``((1,2,j1+j2),(1,3,j1+j2+j3))``

    Parameters
    ==========

    args : tuple
        The arguments that must be passed are ``j``, ``m``, ``jn``, and
        ``jcoupling``. The ``j`` value is the total angular momentum. The ``m``
        value is the eigenvalue of the Jz spin operator. The ``jn`` list are
        the j values of argular momentum spaces coupled together. The
        ``jcoupling`` parameter is an optional parameter defining how the spaces
        are coupled together. See the above description for how these coupling
        parameters are defined.

    Examples
    ========

    Defining simple spin states, both numerical and symbolic:

        >>> from sympy.physics.quantum.spin import JzKetCoupled
        >>> from sympy import symbols
        >>> JzKetCoupled(1, 0, (1, 1))
        |1,0,j1=1,j2=1>
        >>> j, m, j1, j2 = symbols('j m j1 j2')
        >>> JzKetCoupled(j, m, (j1, j2))
        |j,m,j1=j1,j2=j2>

    Defining coupled spin states for more than 2 coupled spaces with various
    coupling parameters:

        >>> JzKetCoupled(2, 1, (1, 1, 1))
        |2,1,j1=1,j2=1,j3=1,j(1,2)=2>
        >>> JzKetCoupled(2, 1, (1, 1, 1), ((1,2,2),(1,3,2)) )
        |2,1,j1=1,j2=1,j3=1,j(1,2)=2>
        >>> JzKetCoupled(2, 1, (1, 1, 1), ((2,3,1),(1,2,2)) )
        |2,1,j1=1,j2=1,j3=1,j(2,3)=1>

    Rewriting the JzKetCoupled in terms of eigenkets of the Jx operator:
    Note: that the resulting eigenstates are JxKetCoupled

        >>> JzKetCoupled(1,1,(1,1)).rewrite("Jx")
        |1,-1,j1=1,j2=1>/2 - sqrt(2)*|1,0,j1=1,j2=1>/2 + |1,1,j1=1,j2=1>/2

    The rewrite method can be used to convert a coupled state to an uncoupled
    state. This is done by passing coupled=False to the rewrite function:

        >>> JzKetCoupled(1, 0, (1, 1)).rewrite('Jz', coupled=False)
        -sqrt(2)*|1,-1>x|1,1>/2 + sqrt(2)*|1,1>x|1,-1>/2

    Get the vector representation of a state in terms of the basis elements
    of the Jx operator:

        >>> from sympy.physics.quantum.represent import represent
        >>> from sympy.physics.quantum.spin import Jx
        >>> from sympy import S
        >>> represent(JzKetCoupled(1,-1,(S(1)/2,S(1)/2)), basis=Jx)
        Matrix([
        [        0],
        [      1/2],
        [sqrt(2)/2],
        [      1/2]])

    See Also
    ========

    JzKet: Normal spin eigenstates
    uncouple: Uncoupling of coupling spin states
    couple: Coupling of uncoupled spin states

    """

    @classmethod
    def dual_class(self):
        return JzBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JzKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(beta=pi*Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(**options)


class JzBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jz.

    See the JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JzKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JzBra

#-----------------------------------------------------------------------------
# Coupling/uncoupling
#-----------------------------------------------------------------------------


def couple(expr, jcoupling_list=None):
    """ Couple a tensor product of spin states

    This function can be used to couple an uncoupled tensor product of spin
    states. All of the eigenstates to be coupled must be of the same class. It
    will return a linear combination of eigenstates that are subclasses of
    CoupledSpinState determined by Clebsch-Gordan angular momentum coupling
    coefficients.

    Parameters
    ==========

    expr : Expr
        An expression involving TensorProducts of spin states to be coupled.
        Each state must be a subclass of SpinState and they all must be the
        same class.

    jcoupling_list : list or tuple
        Elements of this list are sub-lists of length 2 specifying the order of
        the coupling of the spin spaces. The length of this must be N-1, where N
        is the number of states in the tensor product to be coupled. The
        elements of this sublist are the same as the first two elements of each
        sublist in the ``jcoupling`` parameter defined for JzKetCoupled. If this
        parameter is not specified, the default value is taken, which couples
        the first and second product basis spaces, then couples this new coupled
        space to the third product space, etc

    Examples
    ========

    Couple a tensor product of numerical states for two spaces:

        >>> from sympy.physics.quantum.spin import JzKet, couple
        >>> from sympy.physics.quantum.tensorproduct import TensorProduct
        >>> couple(TensorProduct(JzKet(1,0), JzKet(1,1)))
        -sqrt(2)*|1,1,j1=1,j2=1>/2 + sqrt(2)*|2,1,j1=1,j2=1>/2


    Numerical coupling of three spaces using the default coupling method, i.e.
    first and second spaces couple, then this couples to the third space:

        >>> couple(TensorProduct(JzKet(1,1), JzKet(1,1), JzKet(1,0)))
        sqrt(6)*|2,2,j1=1,j2=1,j3=1,j(1,2)=2>/3 + sqrt(3)*|3,2,j1=1,j2=1,j3=1,j(1,2)=2>/3

    Perform this same coupling, but we define the coupling to first couple
    the first and third spaces:

        >>> couple(TensorProduct(JzKet(1,1), JzKet(1,1), JzKet(1,0)), ((1,3),(1,2)) )
        sqrt(2)*|2,2,j1=1,j2=1,j3=1,j(1,3)=1>/2 - sqrt(6)*|2,2,j1=1,j2=1,j3=1,j(1,3)=2>/6 + sqrt(3)*|3,2,j1=1,j2=1,j3=1,j(1,3)=2>/3

    Couple a tensor product of symbolic states:

        >>> from sympy import symbols
        >>> j1,m1,j2,m2 = symbols('j1 m1 j2 m2')
        >>> couple(TensorProduct(JzKet(j1,m1), JzKet(j2,m2)))
        Sum(CG(j1, m1, j2, m2, j, m1 + m2)*|j,m1 + m2,j1=j1,j2=j2>, (j, m1 + m2, j1 + j2))

    """
    a = expr.atoms(TensorProduct)
    for tp in a:
        # Allow other tensor products to be in expression
        if not all(isinstance(state, SpinState) for state in tp.args):
            continue
        # If tensor product has all spin states, raise error for invalid tensor product state
        if not all(state.__class__ is tp.args[0].__class__ for state in tp.args):
            raise TypeError('All states must be the same basis')
        expr = expr.subs(tp, _couple(tp, jcoupling_list))
    return expr


def _couple(tp, jcoupling_list):
    states = tp.args
    coupled_evect = states[0].coupled_class()

    # Define default coupling if none is specified
    if jcoupling_list is None:
        jcoupling_list = []
        for n in range(1, len(states)):
            jcoupling_list.append( (1, n + 1) )

    # Check jcoupling_list valid
    if not len(jcoupling_list) == len(states) - 1:
        raise TypeError('jcoupling_list must be length %d, got %d' %
                        (len(states) - 1, len(jcoupling_list)))
    if not all( len(coupling) == 2 for coupling in jcoupling_list):
        raise ValueError('Each coupling must define 2 spaces')
    if any(n1 == n2 for n1, n2 in jcoupling_list):
        raise ValueError('Spin spaces cannot couple to themselves')
    if all(sympify(n1).is_number and sympify(n2).is_number for n1, n2 in jcoupling_list):
        j_test = [0]*len(states)
        for n1, n2 in jcoupling_list:
            if j_test[n1 - 1] == -1 or j_test[n2 - 1] == -1:
                raise ValueError('Spaces coupling j_n\'s are referenced by smallest n value')
            j_test[max(n1, n2) - 1] = -1

    # j values of states to be coupled together
    jn = [state.j for state in states]
    mn = [state.m for state in states]

    # Create coupling_list, which defines all the couplings between all
    # the spaces from jcoupling_list
    coupling_list = []
    n_list = [ [i + 1] for i in range(len(states)) ]
    for j_coupling in jcoupling_list:
        # Least n for all j_n which is coupled as first and second spaces
        n1, n2 = j_coupling
        # List of all n's coupled in first and second spaces
        j1_n = list(n_list[n1 - 1])
        j2_n = list(n_list[n2 - 1])
        coupling_list.append( (j1_n, j2_n) )
        # Set new j_n to be coupling of all j_n in both first and second spaces
        n_list[ min(n1, n2) - 1 ] = sorted(j1_n + j2_n)

    if all(state.j.is_number and state.m.is_number for state in states):
        # Numerical coupling
        # Iterate over difference between maximum possible j value of each coupling and the actual value
        diff_max = [ Add( *[ jn[n - 1] - mn[n - 1] for n in coupling[0] +
                         coupling[1] ] ) for coupling in coupling_list ]
        result = []
        for diff in range(diff_max[-1] + 1):
            # Determine available configurations
            n = len(coupling_list)
            tot = binomial(diff + n - 1, diff)

            for config_num in range(tot):
                diff_list = _confignum_to_difflist(config_num, diff, n)

                # Skip the configuration if non-physical
                # This is a lazy check for physical states given the loose restrictions of diff_max
                if any(d > m for d, m in zip(diff_list, diff_max)):
                    continue

                # Determine term
                cg_terms = []
                coupled_j = list(jn)
                jcoupling = []
                for (j1_n, j2_n), coupling_diff in zip(coupling_list, diff_list):
                    j1 = coupled_j[ min(j1_n) - 1 ]
                    j2 = coupled_j[ min(j2_n) - 1 ]
                    j3 = j1 + j2 - coupling_diff
                    coupled_j[ min(j1_n + j2_n) - 1 ] = j3
                    m1 = Add( *[ mn[x - 1] for x in j1_n] )
                    m2 = Add( *[ mn[x - 1] for x in j2_n] )
                    m3 = m1 + m2
                    cg_terms.append( (j1, m1, j2, m2, j3, m3) )
                    jcoupling.append( (min(j1_n), min(j2_n), j3) )
                # Better checks that state is physical
                if any(abs(term[5]) > term[4] for term in cg_terms):
                    continue
                if any(term[0] + term[2] < term[4] for term in cg_terms):
                    continue
                if any(abs(term[0] - term[2]) > term[4] for term in cg_terms):
                    continue
                coeff = Mul( *[ CG(*term).doit() for term in cg_terms] )
                state = coupled_evect(j3, m3, jn, jcoupling)
                result.append(coeff*state)
        return Add(*result)
    else:
        # Symbolic coupling
        cg_terms = []
        jcoupling = []
        sum_terms = []
        coupled_j = list(jn)
        for j1_n, j2_n in coupling_list:
            j1 = coupled_j[ min(j1_n) - 1 ]
            j2 = coupled_j[ min(j2_n) - 1 ]
            if len(j1_n + j2_n) == len(states):
                j3 = symbols('j')
            else:
                j3_name = 'j' + ''.join(["%s" % n for n in j1_n + j2_n])
                j3 = symbols(j3_name)
            coupled_j[ min(j1_n + j2_n) - 1 ] = j3
            m1 = Add( *[ mn[x - 1] for x in j1_n] )
            m2 = Add( *[ mn[x - 1] for x in j2_n] )
            m3 = m1 + m2
            cg_terms.append( (j1, m1, j2, m2, j3, m3) )
            jcoupling.append( (min(j1_n), min(j2_n), j3) )
            sum_terms.append((j3, m3, j1 + j2))
        coeff = Mul( *[ CG(*term) for term in cg_terms] )
        state = coupled_evect(j3, m3, jn, jcoupling)
        return Sum(coeff*state, *sum_terms)


def uncouple(expr, jn=None, jcoupling_list=None):
    """ Uncouple a coupled spin state

    Gives the uncoupled representation of a coupled spin state. Arguments must
    be either a spin state that is a subclass of CoupledSpinState or a spin
    state that is a subclass of SpinState and an array giving the j values
    of the spaces that are to be coupled

    Parameters
    ==========

    expr : Expr
        The expression containing states that are to be coupled. If the states
        are a subclass of SpinState, the ``jn`` and ``jcoupling`` parameters
        must be defined. If the states are a subclass of CoupledSpinState,
        ``jn`` and ``jcoupling`` will be taken from the state.

    jn : list or tuple
        The list of the j-values that are coupled. If state is a
        CoupledSpinState, this parameter is ignored. This must be defined if
        state is not a subclass of CoupledSpinState. The syntax of this
        parameter is the same as the ``jn`` parameter of JzKetCoupled.

    jcoupling_list : list or tuple
        The list defining how the j-values are coupled together. If state is a
        CoupledSpinState, this parameter is ignored. This must be defined if
        state is not a subclass of CoupledSpinState. The syntax of this
        parameter is the same as the ``jcoupling`` parameter of JzKetCoupled.

    Examples
    ========

    Uncouple a numerical state using a CoupledSpinState state:

        >>> from sympy.physics.quantum.spin import JzKetCoupled, uncouple
        >>> from sympy import S
        >>> uncouple(JzKetCoupled(1, 0, (S(1)/2, S(1)/2)))
        sqrt(2)*|1/2,-1/2>x|1/2,1/2>/2 + sqrt(2)*|1/2,1/2>x|1/2,-1/2>/2

    Perform the same calculation using a SpinState state:

        >>> from sympy.physics.quantum.spin import JzKet
        >>> uncouple(JzKet(1, 0), (S(1)/2, S(1)/2))
        sqrt(2)*|1/2,-1/2>x|1/2,1/2>/2 + sqrt(2)*|1/2,1/2>x|1/2,-1/2>/2

    Uncouple a numerical state of three coupled spaces using a CoupledSpinState state:

        >>> uncouple(JzKetCoupled(1, 1, (1, 1, 1), ((1,3,1),(1,2,1)) ))
        |1,-1>x|1,1>x|1,1>/2 - |1,0>x|1,0>x|1,1>/2 + |1,1>x|1,0>x|1,0>/2 - |1,1>x|1,1>x|1,-1>/2

    Perform the same calculation using a SpinState state:

        >>> uncouple(JzKet(1, 1), (1, 1, 1), ((1,3,1),(1,2,1)) )
        |1,-1>x|1,1>x|1,1>/2 - |1,0>x|1,0>x|1,1>/2 + |1,1>x|1,0>x|1,0>/2 - |1,1>x|1,1>x|1,-1>/2

    Uncouple a symbolic state using a CoupledSpinState state:

        >>> from sympy import symbols
        >>> j,m,j1,j2 = symbols('j m j1 j2')
        >>> uncouple(JzKetCoupled(j, m, (j1, j2)))
        Sum(CG(j1, m1, j2, m2, j, m)*|j1,m1>x|j2,m2>, (m1, -j1, j1), (m2, -j2, j2))

    Perform the same calculation using a SpinState state

        >>> uncouple(JzKet(j, m), (j1, j2))
        Sum(CG(j1, m1, j2, m2, j, m)*|j1,m1>x|j2,m2>, (m1, -j1, j1), (m2, -j2, j2))

    """
    a = expr.atoms(SpinState)
    for state in a:
        expr = expr.subs(state, _uncouple(state, jn, jcoupling_list))
    return expr


def _uncouple(state, jn, jcoupling_list):
    if isinstance(state, CoupledSpinState):
        jn = state.jn
        coupled_n = state.coupled_n
        coupled_jn = state.coupled_jn
        evect = state.uncoupled_class()
    elif isinstance(state, SpinState):
        if jn is None:
            raise ValueError("Must specify j-values for coupled state")
        if not isinstance(jn, (list, tuple)):
            raise TypeError("jn must be list or tuple")
        if jcoupling_list is None:
            # Use default
            jcoupling_list = []
            for i in range(1, len(jn)):
                jcoupling_list.append(
                    (1, 1 + i, Add(*[jn[j] for j in range(i + 1)])) )
        if not isinstance(jcoupling_list, (list, tuple)):
            raise TypeError("jcoupling must be a list or tuple")
        if not len(jcoupling_list) == len(jn) - 1:
            raise ValueError("Must specify 2 fewer coupling terms than the number of j values")
        coupled_n, coupled_jn = _build_coupled(jcoupling_list, len(jn))
        evect = state.__class__
    else:
        raise TypeError("state must be a spin state")
    j = state.j
    m = state.m
    coupling_list = []
    j_list = list(jn)

    # Create coupling, which defines all the couplings between all the spaces
    for j3, (n1, n2) in zip(coupled_jn, coupled_n):
        # j's which are coupled as first and second spaces
        j1 = j_list[n1[0] - 1]
        j2 = j_list[n2[0] - 1]
        # Build coupling list
        coupling_list.append( (n1, n2, j1, j2, j3) )
        # Set new value in j_list
        j_list[min(n1 + n2) - 1] = j3

    if j.is_number and m.is_number:
        diff_max = [ 2*x for x in jn ]
        diff = Add(*jn) - m

        n = len(jn)
        tot = binomial(diff + n - 1, diff)

        result = []
        for config_num in range(tot):
            diff_list = _confignum_to_difflist(config_num, diff, n)
            if any(d > p for d, p in zip(diff_list, diff_max)):
                continue

            cg_terms = []
            for coupling in coupling_list:
                j1_n, j2_n, j1, j2, j3 = coupling
                m1 = Add( *[ jn[x - 1] - diff_list[x - 1] for x in j1_n ] )
                m2 = Add( *[ jn[x - 1] - diff_list[x - 1] for x in j2_n ] )
                m3 = m1 + m2
                cg_terms.append( (j1, m1, j2, m2, j3, m3) )
            coeff = Mul( *[ CG(*term).doit() for term in cg_terms ] )
            state = TensorProduct(
                *[ evect(j, j - d) for j, d in zip(jn, diff_list) ] )
            result.append(coeff*state)
        return Add(*result)
    else:
        # Symbolic coupling
        m_str = "m1:%d" % (len(jn) + 1)
        mvals = symbols(m_str)
        cg_terms = [(j1, Add(*[mvals[n - 1] for n in j1_n]),
                     j2, Add(*[mvals[n - 1] for n in j2_n]),
                     j3, Add(*[mvals[n - 1] for n in j1_n + j2_n])) for j1_n, j2_n, j1, j2, j3 in coupling_list[:-1] ]
        cg_terms.append(*[(j1, Add(*[mvals[n - 1] for n in j1_n]),
                           j2, Add(*[mvals[n - 1] for n in j2_n]),
                           j, m) for j1_n, j2_n, j1, j2, j3 in [coupling_list[-1]] ])
        cg_coeff = Mul(*[CG(*cg_term) for cg_term in cg_terms])
        sum_terms = [ (m, -j, j) for j, m in zip(jn, mvals) ]
        state = TensorProduct( *[ evect(j, m) for j, m in zip(jn, mvals) ] )
        return Sum(cg_coeff*state, *sum_terms)


def _confignum_to_difflist(config_num, diff, list_len):
    # Determines configuration of diffs into list_len number of slots
    diff_list = []
    for n in range(list_len):
        prev_diff = diff
        # Number of spots after current one
        rem_spots = list_len - n - 1
        # Number of configurations of distributing diff among the remaining spots
        rem_configs = binomial(diff + rem_spots - 1, diff)
        while config_num >= rem_configs:
            config_num -= rem_configs
            diff -= 1
            rem_configs = binomial(diff + rem_spots - 1, diff)
        diff_list.append(prev_diff - diff)
    return diff_list
<|endoftext|>""" A module for mapping operators to their corresponding eigenstates
and vice versa

It contains a global dictionary with eigenstate-operator pairings.
If a new state-operator pair is created, this dictionary should be
updated as well.

It also contains functions operators_to_state and state_to_operators
for mapping between the two. These can handle both classes and
instances of operators and states. See the individual function
descriptions for details.

TODO List:
- Update the dictionary with a complete list of state-operator pairs
"""

from sympy.physics.quantum.cartesian import (XOp, YOp, ZOp, XKet, PxOp, PxKet,
                                             PositionKet3D)
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import StateBase, BraBase, Ket
from sympy.physics.quantum.spin import (JxOp, JyOp, JzOp, J2Op, JxKet, JyKet,
                                        JzKet)

__all__ = [
    'operators_to_state',
    'state_to_operators'
]

#state_mapping stores the mappings between states and their associated
#operators or tuples of operators. This should be updated when new
#classes are written! Entries are of the form PxKet : PxOp or
#something like 3DKet : (ROp, ThetaOp, PhiOp)

#frozenset is used so that the reverse mapping can be made
#(regular sets are not hashable because they are mutable
state_mapping = { JxKet: frozenset((J2Op, JxOp)),
                  JyKet: frozenset((J2Op, JyOp)),
                  JzKet: frozenset((J2Op, JzOp)),
                  Ket: Operator,
                  PositionKet3D: frozenset((XOp, YOp, ZOp)),
                  PxKet: PxOp,
                  XKet: XOp }

op_mapping = {v: k for k, v in state_mapping.items()}


def operators_to_state(operators, **options):
    """ Returns the eigenstate of the given operator or set of operators

    A global function for mapping operator classes to their associated
    states. It takes either an Operator or a set of operators and
    returns the state associated with these.

    This function can handle both instances of a given operator or
    just the class itself (i.e. both XOp() and XOp)

    There are multiple use cases to consider:

    1) A class or set of classes is passed: First, we try to
    instantiate default instances for these operators. If this fails,
    then the class is simply returned. If we succeed in instantiating
    default instances, then we try to call state._operators_to_state
    on the operator instances. If this fails, the class is returned.
    Otherwise, the instance returned by _operators_to_state is returned.

    2) An instance or set of instances is passed: In this case,
    state._operators_to_state is called on the instances passed. If
    this fails, a state class is returned. If the method returns an
    instance, that instance is returned.

    In both cases, if the operator class or set does not exist in the
    state_mapping dictionary, None is returned.

    Parameters
    ==========

    arg: Operator or set
         The class or instance of the operator or set of operators
         to be mapped to a state

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XOp, PxOp
    >>> from sympy.physics.quantum.operatorset import operators_to_state
    >>> from sympy.physics.quantum.operator import Operator
    >>> operators_to_state(XOp)
    |x>
    >>> operators_to_state(XOp())
    |x>
    >>> operators_to_state(PxOp)
    |px>
    >>> operators_to_state(PxOp())
    |px>
    >>> operators_to_state(Operator)
    |psi>
    >>> operators_to_state(Operator())
    |psi>
    """

    if not (isinstance(operators, (Operator, set)) or issubclass(operators, Operator)):
        raise NotImplementedError("Argument is not an Operator or a set!")

    if isinstance(operators, set):
        for s in operators:
            if not (isinstance(s, Operator)
                   or issubclass(s, Operator)):
                raise NotImplementedError("Set is not all Operators!")

        ops = frozenset(operators)

        if ops in op_mapping:  # ops is a list of classes in this case
            #Try to get an object from default instances of the
            #operators...if this fails, return the class
            try:
                op_instances = [op() for op in ops]
                ret = _get_state(op_mapping[ops], set(op_instances), **options)
            except NotImplementedError:
                ret = op_mapping[ops]

            return ret
        else:
            tmp = [type(o) for o in ops]
            classes = frozenset(tmp)

            if classes in op_mapping:
                ret = _get_state(op_mapping[classes], ops, **options)
            else:
                ret = None

            return ret
    else:
        if operators in op_mapping:
            try:
                op_instance = operators()
                ret = _get_state(op_mapping[operators], op_instance, **options)
            except NotImplementedError:
                ret = op_mapping[operators]

            return ret
        elif type(operators) in op_mapping:
            return _get_state(op_mapping[type(operators)], operators, **options)
        else:
            return None


def state_to_operators(state, **options):
    """ Returns the operator or set of operators corresponding to the
    given eigenstate

    A global function for mapping state classes to their associated
    operators or sets of operators. It takes either a state class
    or instance.

    This function can handle both instances of a given state or just
    the class itself (i.e. both XKet() and XKet)

    There are multiple use cases to consider:

    1) A state class is passed: In this case, we first try
    instantiating a default instance of the class. If this succeeds,
    then we try to call state._state_to_operators on that instance.
    If the creation of the default instance or if the calling of
    _state_to_operators fails, then either an operator class or set of
    operator classes is returned. Otherwise, the appropriate
    operator instances are returned.

    2) A state instance is returned: Here, state._state_to_operators
    is called for the instance. If this fails, then a class or set of
    operator classes is returned. Otherwise, the instances are returned.

    In either case, if the state's class does not exist in
    state_mapping, None is returned.

    Parameters
    ==========

    arg: StateBase class or instance (or subclasses)
         The class or instance of the state to be mapped to an
         operator or set of operators

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XKet, PxKet, XBra, PxBra
    >>> from sympy.physics.quantum.operatorset import state_to_operators
    >>> from sympy.physics.quantum.state import Ket, Bra
    >>> state_to_operators(XKet)
    X
    >>> state_to_operators(XKet())
    X
    >>> state_to_operators(PxKet)
    Px
    >>> state_to_operators(PxKet())
    Px
    >>> state_to_operators(PxBra)
    Px
    >>> state_to_operators(XBra)
    X
    >>> state_to_operators(Ket)
    O
    >>> state_to_operators(Bra)
    O
    """

    if not (isinstance(state, StateBase) or issubclass(state, StateBase)):
        raise NotImplementedError("Argument is not a state!")

    if state in state_mapping:  # state is a class
        state_inst = _make_default(state)
        try:
            ret = _get_ops(state_inst,
                           _make_set(state_mapping[state]), **options)
        except (NotImplementedError, TypeError):
            ret = state_mapping[state]
    elif type(state) in state_mapping:
        ret = _get_ops(state,
                       _make_set(state_mapping[type(state)]), **options)
    elif isinstance(state, BraBase) and state.dual_class() in state_mapping:
        ret = _get_ops(state,
                       _make_set(state_mapping[state.dual_class()]))
    elif issubclass(state, BraBase) and state.dual_class() in state_mapping:
        state_inst = _make_default(state)
        try:
            ret = _get_ops(state_inst,
                           _make_set(state_mapping[state.dual_class()]))
        except (NotImplementedError, TypeError):
            ret = state_mapping[state.dual_class()]
    else:
        ret = None

    return _make_set(ret)


def _make_default(expr):
