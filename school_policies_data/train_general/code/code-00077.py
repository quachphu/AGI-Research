        return None

    # Add contraction to possible choices
    return [sort, positions, new_input_sets]


def _update_other_results(results, best):
    """Update the positions and provisional input_sets of ``results``
    based on performing the contraction result ``best``. Remove any
    involving the tensors contracted.

    Parameters
    ----------
    results : list
        List of contraction results produced by
        ``_parse_possible_contraction``.
    best : list
        The best contraction of ``results`` i.e. the one that
        will be performed.

    Returns
    -------
    mod_results : list
        The list of modified results, updated with outcome of
        ``best`` contraction.
    """

    best_con = best[1]
    bx, by = best_con
    mod_results = []

    for cost, (x, y), con_sets in results:

        # Ignore results involving tensors just contracted
        if x in best_con or y in best_con:
            continue

        # Update the input_sets
        del con_sets[by - int(by > x) - int(by > y)]
        del con_sets[bx - int(bx > x) - int(bx > y)]
        con_sets.insert(-1, best[2][-1])

        # Update the position indices
        mod_con = x - int(x > bx) - int(x > by), y - int(y > bx) - int(y > by)
        mod_results.append((cost, mod_con, con_sets))

    return mod_results

def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
    """
    Finds the path by contracting the best pair until the input list is
    exhausted. The best pair is found by minimizing the tuple
    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
    matrix multiplication or inner product operations, then Hadamard like
    operations, and finally outer operations. Outer products are limited by
    ``memory_limit``. This algorithm scales cubically with respect to the
    number of elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The greedy contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set()
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _greedy_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """

    # Handle trivial cases that leaked through
    if len(input_sets) == 1:
        return [(0,)]
    elif len(input_sets) == 2:
        return [(0, 1)]

    # Build up a naive cost
    contract = _find_contraction(
        range(len(input_sets)), input_sets, output_set
    )
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    naive_cost = _flop_count(
        idx_contract, idx_removed, len(input_sets), idx_dict
    )

    # Initially iterate over all pairs
    comb_iter = itertools.combinations(range(len(input_sets)), 2)
    known_contractions = []

    path_cost = 0
    path = []

    for iteration in range(len(input_sets) - 1):

        # Iterate over all pairs on the first step, only previously
        # found pairs on subsequent steps
        for positions in comb_iter:

            # Always initially ignore outer products
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue

            result = _parse_possible_contraction(
                positions, input_sets, output_set, idx_dict,
                memory_limit, path_cost, naive_cost
            )
            if result is not None:
                known_contractions.append(result)

        # If we do not have a inner contraction, rescan pairs
        # including outer products
        if len(known_contractions) == 0:

            # Then check the outer products
            for positions in itertools.combinations(
                range(len(input_sets)), 2
            ):
                result = _parse_possible_contraction(
                    positions, input_sets, output_set, idx_dict,
                    memory_limit, path_cost, naive_cost
                )
                if result is not None:
                    known_contractions.append(result)

            # If we still did not find any remaining contractions,
            # default back to einsum like behavior
            if len(known_contractions) == 0:
                path.append(tuple(range(len(input_sets))))
                break

        # Sort based on first index
        best = min(known_contractions, key=lambda x: x[0])

        # Now propagate as many unused contractions as possible
        # to the next iteration
        known_contractions = _update_other_results(known_contractions, best)

        # Next iteration only compute contractions with the new tensor
        # All other contractions have been accounted for
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))

        # Update path and total cost
        path.append(best[1])
        path_cost += best[0][1]

    return path


def _can_dot(inputs, result, idx_removed):
    """
    Checks if we can use BLAS (np.tensordot) call and its beneficial to do so.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation


    Returns
    -------
    type : bool
        Returns true if BLAS should and can be used, else False

    Notes
    -----
    If the operations is BLAS level 1 or 2 and is not already aligned
    we default back to einsum as the memory movement to copy is more
    costly than the operation itself.


    Examples
    --------

    # Standard GEMM operation
    >>> _can_dot(['ij', 'jk'], 'ik', set('j'))
    True

    # Can use the standard BLAS, but requires odd data movement
    >>> _can_dot(['ijj', 'jk'], 'ik', set('j'))
    False

    # DDOT where the memory is not aligned
    >>> _can_dot(['ijk', 'ikj'], '', set('ijk'))
    False

    """

    # All `dot` calls remove indices
    if len(idx_removed) == 0:
        return False

    # BLAS can only handle two operands
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        # can't deal with repeated indices on same input or more than 2 total
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # can't do implicit summation or dimension collapse e.g.
        #     "ab,bc->c" (implicitly sum over 'a')
        #     "ab,ca->ca" (take diagonal of 'a')
        if nl + nr - 1 == int(c in result):
            return False

    # Build a few temporaries
    set_left = set(input_left)
    set_right = set(input_right)
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed
    rs = len(idx_removed)

    # At this point we are a DOT, GEMV, or GEMM operation

    # Handle inner products

    # DDOT with aligned data
    if input_left == input_right:
        return True

    # DDOT without aligned data (better to use einsum)
    if set_left == set_right:
        return False

    # Handle the 4 possible (aligned) GEMV or GEMM cases

    # GEMM or GEMV no transpose
    if input_left[-rs:] == input_right[:rs]:
        return True

    # GEMM or GEMV transpose both
    if input_left[:rs] == input_right[-rs:]:
        return True

    # GEMM or GEMV transpose right
    if input_left[-rs:] == input_right[-rs:]:
        return True

    # GEMM or GEMV transpose left
    if input_left[:rs] == input_right[:rs]:
        return True

    # Einsum is faster than GEMV if we have to copy data
    if not keep_left or not keep_right:
        return False

    # We are a matrix-matrix product, but we need to copy data
    return True


def _parse_einsum_input(operands):
    """
    A reproduction of einsum c side einsum parsing in python.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> np.random.seed(123)
    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b]) # may vary

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b]) # may vary
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [asanyarray(v) for v in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asanyarray(v) for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain "
                            "either int or Ellipsis"
                        ) from e
                    subscripts += einsum_symbols[s]
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain "
                            "either int or Ellipsis"
                        ) from e
                    subscripts += einsum_symbols[s]
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) -
                                         set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError("Output character %s appeared more than once in "
                             "the output." % char)
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input"
                             % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")

    return (input_subscripts, output_subscript, operands)


def _einsum_path_dispatcher(*operands, optimize=None, einsum_call=None):
    # NOTE: technically, we should only dispatch on array-like arguments, not
    # subscripts (given as strings). But separating operands into
    # arrays/subscripts is a little tricky/slow (given einsum's two supported
    # signatures), so as a practical shortcut we dispatch on everything.
    # Strings will be ignored for dispatching since they don't define
    # __array_function__.
    return operands


@array_function_dispatch(_einsum_path_dispatcher, module='numpy')
def einsum_path(*operands, optimize='greedy', einsum_call=False):
    """
    einsum_path(subscripts, *operands, optimize='greedy')

    Evaluates the lowest cost contraction order for an einsum expression by
    considering the creation of intermediate arrays.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    optimize : {bool, list, tuple, 'greedy', 'optimal'}
        Choose the type of path. If a tuple is provided, the second argument is
        assumed to be the maximum intermediate size created. If only a single
        argument is provided the largest input or output array size is used
        as a maximum intermediate size.

        * if a list is given that starts with ``einsum_path``, uses this as the
          contraction path
        * if False no optimization is taken
        * if True defaults to the 'greedy' algorithm
        * 'optimal' An algorithm that combinatorially explores all possible
          ways of contracting the listed tensors and chooses the least costly
          path. Scales exponentially with the number of terms in the
          contraction.
        * 'greedy' An algorithm that chooses the best pair contraction
          at each step. Effectively, this algorithm searches the largest inner,
          Hadamard, and then outer products at each step. Scales cubically with
          the number of terms in the contraction. Equivalent to the 'optimal'
          path for most contractions.

        Default is 'greedy'.

    Returns
    -------
    path : list of tuples
        A list representation of the einsum path.
    string_repr : str
        A printable representation of the einsum path.

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the
    end of the contraction list. This list can then be iterated over until all
    intermediate contractions are complete.

    See Also
    --------
    einsum, linalg.multi_dot

    Examples
    --------

    We can begin with a chain dot example. In this case, it is optimal to
    contract the ``b`` and ``c`` tensors first as represented by the first
    element of the path ``(1, 2)``. The resulting tensor is added to the end
    of the contraction and the remaining contraction ``(0, 1)`` is then
    completed.

    >>> np.random.seed(123)
    >>> a = np.random.rand(2, 2)
    >>> b = np.random.rand(2, 5)
    >>> c = np.random.rand(5, 2)
    >>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
    >>> print(path_info[0])
    ['einsum_path', (1, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ij,jk,kl->il # may vary
             Naive scaling:  4
         Optimized scaling:  3
          Naive FLOP count:  1.600e+02
      Optimized FLOP count:  5.600e+01
       Theoretical speedup:  2.857
      Largest intermediate:  4.000e+00 elements
    -------------------------------------------------------------------------
    scaling                  current                                remaining
    -------------------------------------------------------------------------
       3                   kl,jk->jl                                ij,jl->il
       3                   jl,ij->il                                   il->il


    A more complex index transformation example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,
    ...                            optimize='greedy')

    >>> print(path_info[0])
    ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ea,fb,abcd,gc,hd->efgh # may vary
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  8.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1000.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------
    scaling                  current                                remaining
    --------------------------------------------------------------------------
       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5               bcde,fb->cdef                         gc,hd,cdef->efgh
       5               cdef,gc->defg                            hd,defg->efgh
       5               defg,hd->efgh                               efgh->efgh
    """

    # Figure out what the path really is
    path_type = optimize
    if path_type is True:
        path_type = 'greedy'
    if path_type is None:
        path_type = False

    explicit_einsum_path = False
    memory_limit = None

    # No optimization or a named path algorithm
    if (path_type is False) or isinstance(path_type, str):
        pass

    # Given an explicit path
    elif len(path_type) and (path_type[0] == 'einsum_path'):
        explicit_einsum_path = True

    # Path tuple with memory limit
    elif ((len(path_type) == 2) and isinstance(path_type[0], str) and
            isinstance(path_type[1], (int, float))):
        memory_limit = int(path_type[1])
        path_type = path_type[0]

    else:
        raise TypeError("Did not understand the path: %s" % str(path_type))

    # Hidden option, only einsum should call this
    einsum_call_arg = einsum_call

    # Python side parsing
    input_subscripts, output_subscript, operands = (
        _parse_einsum_input(operands)
    )

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    broadcast_indices = [[] for x in range(len(input_list))]
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "
                             "correct number of indices for operand %d."
                             % (input_subscripts[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = sh[cnum]

            # Build out broadcast indices
            if dim == 1:
                broadcast_indices[tnum].append(char)

            if char in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError("Size of label '%s' for operand %d (%d) "
                                     "does not match previous terms (%d)."
                                     % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim

    # Convert broadcast inds to sets
    broadcast_indices = [set(x) for x in broadcast_indices]

    # Compute size of each input array plus the output array
    size_list = [_compute_size_by_dict(term, dimension_dict)
                 for term in input_list + [output_subscript]]
    max_size = max(size_list)

    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit

    # Compute naive cost
    # This isn't quite right, need to look into exactly how einsum does this
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = _flop_count(
        indices, inner_product, len(input_list), dimension_dict
    )

    # Compute the path
    if explicit_einsum_path:
        path = path_type[1:]
    elif (
        (path_type is False)
        or (len(input_list) in [1, 2])
        or (indices == output_set)
    ):
        # Nothing to be optimized, leave it to einsum
        path = [tuple(range(len(input_list)))]
    elif path_type == "greedy":
        path = _greedy_path(
            input_sets, output_set, dimension_dict, memory_arg
        )
    elif path_type == "optimal":
        path = _optimal_path(
            input_sets, output_set, dimension_dict, memory_arg
        )
    else:
        raise KeyError("Path name %s not found", path_type)

    cost_list, scale_list, size_list, contraction_list = [], [], [], []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        cost = _flop_count(
            idx_contract, idx_removed, len(contract_inds), dimension_dict
        )
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(_compute_size_by_dict(out_inds, dimension_dict))

        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)

        new_bcast_inds = bcast - idx_removed

        # If we're broadcasting, nix blas
        if not len(idx_removed & bcast):
            do_blas = _can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        contraction = (
            contract_inds, idx_removed, einsum_str, input_list[:], do_blas
        )
        contraction_list.append(contraction)

    opt_cost = sum(cost_list) + 1

    if len(input_list) != 1:
        # Explicit "einsum_path" is usually trusted, but we detect this kind of
        # mistake in order to prevent from returning an intermediate value.
        raise RuntimeError(
            "Invalid einsum_path is specified: {} more operands has to be "
            "contracted.".format(len(input_list) - 1))

    if einsum_call_arg:
        return (operands, contraction_list)

    # Return the path along with a nice string representation
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "current", "remaining")

    speedup = naive_cost / opt_cost
    max_i = max(size_list)

    path_print = "  Complete contraction:  %s\n" % overall_contraction
    path_print += "         Naive scaling:  %d\n" % len(indices)
    path_print += "     Optimized scaling:  %d\n" % max(scale_list)
    path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
    path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
    path_print += "   Theoretical speedup:  %3.3f\n" % speedup
    path_print += "  Largest intermediate:  %.3e elements\n" % max_i
    path_print += "-" * 74 + "\n"
    path_print += "%6s %24s %40s\n" % header
    path_print += "-" * 74

    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        remaining_str = ",".join(remaining) + "->" + output_subscript
        path_run = (scale_list[n], einsum_str, remaining_str)
        path_print += "\n%4d    %24s %40s" % path_run

    path = ['einsum_path'] + path
    return (path, path_print)


def _einsum_dispatcher(*operands, out=None, optimize=None, **kwargs):
    # Arguably we dispatch on more arguments than we really should; see note in
    # _einsum_path_dispatcher for why.
    yield from operands
    yield out


# Rewrite einsum to handle different cases
@array_function_dispatch(_einsum_dispatcher, module='numpy')
def einsum(*operands, out=None, optimize=False, **kwargs):
    """
    einsum(subscripts, *operands, out=None, dtype=None, order='K',
           casting='safe', optimize=False)

    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    See the notes and examples for clarification.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of array_like
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    dtype : {data-type, None}, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the output. 'C' means it should
        be C contiguous. 'F' means it should be Fortran contiguous,
        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
        'K' means it should be as close to the layout as the inputs as
        is possible, including arbitrarily permuted axes.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.  Setting this to
        'unsafe' is not recommended, as it can adversely affect accumulations.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : {False, True, 'greedy', 'optimal'}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False and True will default to the 'greedy' algorithm.
        Also accepts an explicit contraction list from the ``np.einsum_path``
        function. See ``np.einsum_path`` for more details. Defaults to False.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    See Also
    --------
    einsum_path, dot, inner, outer, tensordot, linalg.multi_dot
    einsum:
        Similar verbose interface is provided by the
        `einops <https://github.com/arogozhnikov/einops>`_ package to cover
        additional operations: transpose, reshape/flatten, repeat/tile,
        squeeze/unsqueeze and reductions.
        The `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_
        optimizes contraction order for einsum-like expressions
        in backend-agnostic manner.

    Notes
    -----
    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:

    * Trace of an array, :py:func:`numpy.trace`.
    * Return a diagonal, :py:func:`numpy.diag`.
    * Array axis summations, :py:func:`numpy.sum`.
    * Transpositions and permutations, :py:func:`numpy.transpose`.
    * Matrix multiplication and dot product, :py:func:`numpy.matmul`
        :py:func:`numpy.dot`.
    * Vector inner and outer products, :py:func:`numpy.inner`
        :py:func:`numpy.outer`.
    * Broadcasting, element-wise and scalar multiplication,
        :py:func:`numpy.multiply`.
    * Tensor contractions, :py:func:`numpy.tensordot`.
    * Chained array operations, in efficient calculation order,
        :py:func:`numpy.einsum_path`.

    The subscripts string is a comma-separated list of subscript labels,
    where each label refers to a dimension of the corresponding operand.
    Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
    is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label
    appears only once, it is not summed, so ``np.einsum('i', a)``
    produces a view of ``a`` with no changes. A further example
    ``np.einsum('ij,jk', a, b)`` describes traditional matrix multiplication
    and is equivalent to :py:func:`np.matmul(a,b) <numpy.matmul>`.
    Repeated subscript labels in one operand take the diagonal.
    For example, ``np.einsum('ii', a)`` is equivalent to
    :py:func:`np.trace(a) <numpy.trace>`.

    In *implicit mode*, the chosen subscripts are important
    since the axes of the output are reordered alphabetically.  This
    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
    ``np.einsum('ji', a)`` takes its transpose. Additionally,
    ``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
    ``np.einsum('ij,jh', a, b)`` returns the transpose of the
    multiplication since subscript 'h' precedes subscript 'i'.

    In *explicit mode* the output can be directly controlled by
    specifying output subscript labels.  This requires the
    identifier '->' as well as the list of output subscript labels.
    This feature increases the flexibility of the function since
    summing can be disabled or forced when required. The call
    ``np.einsum('i->', a)`` is like :py:func:`np.sum(a) <numpy.sum>`
    if ``a`` is a 1-D array, and ``np.einsum('ii->i', a)``
    is like :py:func:`np.diag(a) <numpy.diag>` if ``a`` is a square 2-D array.
    The difference is that `einsum` does not allow broadcasting by default.
    Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
    order of the output subscript labels and therefore returns matrix
    multiplication, unlike the example above in implicit mode.

    To enable and control broadcasting, use an ellipsis.  Default
    NumPy-style broadcasting is done by adding an ellipsis
    to the left of each term, like ``np.einsum('...ii->...i', a)``.
    ``np.einsum('...i->...', a)`` is like
    :py:func:`np.sum(a, axis=-1) <numpy.sum>` for array ``a`` of any shape.
    To take the trace along the first and last axes,
    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, one can do
    ``np.einsum('ij...,jk...->ik...', a, b)``.

    When there is only one operand, no axes are summed, and no output
    parameter is provided, a view into the operand is returned instead
    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
    produces a view (changed in version 1.10.0).

    `einsum` also provides an alternative way to provide the subscripts and
    operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
    If the output shape is not provided in this format `einsum` will be
    calculated in implicit mode, otherwise it will be performed explicitly.
    The examples below have corresponding `einsum` calls with the two
    parameter methods.

    Views returned from einsum are now writeable whenever the input array
    is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
    have the same effect as :py:func:`np.swapaxes(a, 0, 2) <numpy.swapaxes>`
    and ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
    of a 2D array.

    Added the ``optimize`` argument which will optimize the contraction order
    of an einsum expression. For a contraction with three or more operands
    this can greatly increase the computational efficiency at the cost of
    a larger memory footprint during computation.

    Typically a 'greedy' algorithm is applied which empirical tests have shown
    returns the optimal path in the majority of cases. In some cases 'optimal'
    will return the superlative path through a more expensive, exhaustive
    search. For iterative calculations it may be advisable to calculate
    the optimal path once and reuse that path by supplying it as an argument.
    An example is given below.

    See :py:func:`numpy.einsum_path` for more details.

    Examples
    --------
    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)

    Trace of a matrix:

    >>> np.einsum('ii', a)
    60
    >>> np.einsum(a, [0,0])
    60
    >>> np.trace(a)
    60

    Extract the diagonal (requires explicit form):

    >>> np.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0,0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis (requires explicit form):

    >>> np.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0,1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])

    For higher dimensional arrays summing a single axis can be done
    with ellipsis:

    >>> np.einsum('...j->...', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])

    Compute a matrix transpose, or reorder any number of axes:

    >>> np.einsum('ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum('ij->ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1,0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])

    Vector inner products:

    >>> np.einsum('i,i', b, b)
    30
    >>> np.einsum(b, [0], b, [0])
    30
    >>> np.inner(b,b)
    30

    Matrix vector multiplication:

    >>> np.einsum('ij,j', a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0,1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum('...j,j', a, b)
    array([ 30,  80, 130, 180, 230])

    Broadcasting and scalar multiplication:

    >>> np.einsum('..., ...', 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(',ij', 3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.multiply(3, c)
    array([[ 0,  3,  6],
           [ 9, 12, 15]])

    Vector outer product:

    >>> np.einsum('i,j', np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.einsum(np.arange(2)+1, [0], b, [1])
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.outer(np.arange(2)+1, b)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])

    Tensor contraction:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> np.einsum('ijk,jil->kl', a, b)
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    >>> np.tensordot(a,b, axes=([1,0],[0,1]))
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    Writeable returned arrays (since version 1.10.0):

    >>> a = np.zeros((3, 3))
    >>> np.einsum('ii->i', a)[:] = 1
    >>> a
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Example of ellipsis use:

    >>> a = np.arange(6).reshape((3,2))
    >>> b = np.arange(12).reshape((4,3))
    >>> np.einsum('ki,jk->ij', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('ki,...k->i...', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('k...,jk', a, b)
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])

    Chained array operations. For more complicated contractions, speed ups
    might be achieved by repeatedly computing a 'greedy' path or pre-computing
    the 'optimal' path and repeatedly applying it, using an `einsum_path`
    insertion (since version 1.12.0). Performance improvements can be
    particularly significant with larger arrays:

    >>> a = np.ones(64).reshape(2,4,8)

    Basic `einsum`: ~1520ms  (benchmarked on 3.1GHz Intel i5.)

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)

    Sub-optimal `einsum` (due to repeated path calculation time): ~330ms

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,
    ...         optimize='optimal')

    Greedy `einsum` (faster optimal path approximation): ~160ms

    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='greedy')

    Optimal `einsum` (best usage pattern in some use cases): ~110ms

    >>> path = np.einsum_path('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,
    ...     optimize='optimal')[0]
    >>> for iteration in range(500):
    ...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)

    """
    # Special handling if out is specified
    specified_out = out is not None

    # If no optimization, run pure einsum
    if optimize is False:
        if specified_out:
            kwargs['out'] = out
        return c_einsum(*operands, **kwargs)

    # Check the kwargs to avoid a more cryptic error later, without having to
    # repeat default values here
    valid_einsum_kwargs = ['dtype', 'order', 'casting']
    unknown_kwargs = [k for (k, v) in kwargs.items() if
                      k not in valid_einsum_kwargs]
    if len(unknown_kwargs):
        raise TypeError("Did not understand the following kwargs: %s"
                        % unknown_kwargs)

    # Build the contraction list and operand
    operands, contraction_list = einsum_path(*operands, optimize=optimize,
                                             einsum_call=True)

    # Handle order kwarg for output array, c_einsum allows mixed case
    output_order = kwargs.pop('order', 'K')
    if output_order.upper() == 'A':
        if all(arr.flags.f_contiguous for arr in operands):
            output_order = 'F'
        else:
            output_order = 'C'

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        tmp_operands = [operands.pop(x) for x in inds]

        # Do we need to deal with the output?
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        # Call tensordot if still possible
        if blas:
            # Checks have already been handled
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, "")

            # Find indices to contract over
            left_pos, right_pos = [], []
            for s in sorted(idx_rm):
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            # Contract!
            new_view = tensordot(
                *tmp_operands, axes=(tuple(left_pos), tuple(right_pos))
            )

            # Build a new view if needed
            if (tensor_result != results_index) or handle_out:
                if handle_out:
                    kwargs["out"] = out
                new_view = c_einsum(
                    tensor_result + '->' + results_index, new_view, **kwargs
                )

        # Call einsum
        else:
            # If out was specified
            if handle_out:
                kwargs["out"] = out

            # Do the contraction
            new_view = c_einsum(einsum_str, *tmp_operands, **kwargs)

        # Append new items and dereference what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out
    else:
        return asanyarray(operands[0], order=output_order)
<|endoftext|>"""Array printing function

$Id: arrayprint.py,v 1.9 2005/09/13 13:58:44 teoliphant Exp $

"""
__all__ = ["array2string", "array_str", "array_repr",
           "set_printoptions", "get_printoptions", "printoptions",
           "format_float_positional", "format_float_scientific"]
__docformat__ = 'restructuredtext'

#
# Written by Konrad Hinsen <hinsenk@ere.umontreal.ca>
# last revision: 1996-3-13
# modified by Jim Hugunin 1997-3-3 for repr's and str's (and other details)
# and by Perry Greenfield 2000-4-1 for numarray
# and by Travis Oliphant  2005-8-22 for numpy


# Note: Both scalartypes.c.src and arrayprint.py implement strs for numpy
# scalars but for different purposes. scalartypes.c.src has str/reprs for when
# the scalar is printed on its own, while arrayprint.py has strs for when
# scalars are printed inside an ndarray. Only the latter strs are currently
# user-customizable.

import functools
import numbers
import sys
try:
    from _thread import get_ident
except ImportError:
    from _dummy_thread import get_ident

import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
                         datetime_as_string, datetime_data, ndarray)
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float64, complex128,
                           flexible)
from .overrides import array_function_dispatch, set_module
from .printoptions import format_options
import operator
import warnings
import contextlib


def _make_options_dict(precision=None, threshold=None, edgeitems=None,
                       linewidth=None, suppress=None, nanstr=None, infstr=None,
                       sign=None, formatter=None, floatmode=None, legacy=None,
                       override_repr=None):
    """
    Make a dictionary out of the non-None arguments, plus conversion of
    *legacy* and sanity checks.
    """

    options = {k: v for k, v in list(locals().items()) if v is not None}

    if suppress is not None:
        options['suppress'] = bool(suppress)

    modes = ['fixed', 'unique', 'maxprec', 'maxprec_equal']
    if floatmode not in modes + [None]:
        raise ValueError("floatmode option must be one of " +
                         ", ".join('"{}"'.format(m) for m in modes))

    if sign not in [None, '-', '+', ' ']:
        raise ValueError("sign option must be one of ' ', '+', or '-'")

    if legacy is False:
        options['legacy'] = sys.maxsize
    elif legacy == False:  # noqa: E712
        warnings.warn(
            f"Passing `legacy={legacy!r}` is deprecated.",
            FutureWarning, stacklevel=3
        )
        options['legacy'] = sys.maxsize
    elif legacy == '1.13':
        options['legacy'] = 113
    elif legacy == '1.21':
        options['legacy'] = 121
    elif legacy == '1.25':
        options['legacy'] = 125
    elif legacy == '2.1':
        options['legacy'] = 201
    elif legacy is None:
        pass  # OK, do nothing.
    else:
        warnings.warn(
            "legacy printing option can currently only be '1.13', '1.21', "
            "'1.25', '2.1, or `False`", stacklevel=3)

    if threshold is not None:
        # forbid the bad threshold arg suggested by stack overflow, gh-12351
        if not isinstance(threshold, numbers.Number):
            raise TypeError("threshold must be numeric")
        if np.isnan(threshold):
            raise ValueError("threshold must be non-NAN, try "
                             "sys.maxsize for untruncated representation")

    if precision is not None:
        # forbid the bad precision arg as suggested by issue #18254
        try:
            options['precision'] = operator.index(precision)
        except TypeError as e:
            raise TypeError('precision must be an integer') from e

    return options


@set_module('numpy')
def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None, nanstr=None,
                     infstr=None, formatter=None, sign=None, floatmode=None,
                     *, legacy=None, override_repr=None):
    """
    Set printing options.

    These options determine the way floating point numbers, arrays and
    other NumPy objects are displayed.

    Parameters
    ----------
    precision : int or None, optional
        Number of digits of precision for floating point output (default 8).
        May be None if `floatmode` is not `fixed`, to print as many digits as
        necessary to uniquely specify the value.
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr (default 1000).
        To always use the full repr without summarization, pass `sys.maxsize`.
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension (default 3).
    linewidth : int, optional
        The number of characters per line for the purpose of inserting
        line breaks (default 75).
    suppress : bool, optional
        If True, always print floating point numbers using fixed point
        notation, in which case numbers equal to zero in the current precision
        will print as zero.  If False, then scientific notation is used when
        absolute value of the smallest number is < 1e-4 or the ratio of the
        maximum absolute value to the minimum is > 1e3. The default is False.
    nanstr : str, optional
        String representation of floating point not-a-number (default nan).
    infstr : str, optional
        String representation of floating point infinity (default inf).
    sign : string, either '-', '+', or ' ', optional
        Controls printing of the sign of floating-point types. If '+', always
        print the sign of positive values. If ' ', always prints a space
        (whitespace character) in the sign position of positive values.  If
        '-', omit the sign character of positive values. (default '-')

        .. versionchanged:: 2.0
             The sign parameter can now be an integer type, previously
             types were floating-point types.

    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are:

        - 'bool'
        - 'int'
        - 'timedelta' : a `numpy.timedelta64`
        - 'datetime' : a `numpy.datetime64`
        - 'float'
        - 'longfloat' : 128-bit floats
        - 'complexfloat'
        - 'longcomplexfloat' : composed of two 128-bit floats
        - 'numpystr' : types `numpy.bytes_` and `numpy.str_`
        - 'object' : `np.object_` arrays

        Other keys that can be used to set a group of types at once are:

        - 'all' : sets all types
        - 'int_kind' : sets 'int'
        - 'float_kind' : sets 'float' and 'longfloat'
        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
        - 'str_kind' : sets 'numpystr'
    floatmode : str, optional
        Controls the interpretation of the `precision` option for
        floating-point types. Can take the following values
        (default maxprec_equal):

        * 'fixed': Always print exactly `precision` fractional digits,
                even if this would print more or fewer digits than
                necessary to specify the value uniquely.
        * 'unique': Print the minimum number of fractional digits necessary
                to represent each value uniquely. Different elements may
                have a different number of digits. The value of the
                `precision` option is ignored.
        * 'maxprec': Print at most `precision` fractional digits, but if
                an element can be uniquely represented with fewer digits
                only print it with that many.
        * 'maxprec_equal': Print at most `precision` fractional digits,
                but if every element in the array can be uniquely
                represented with an equal number of fewer digits, use that
                many digits for all elements.
    legacy : string or `False`, optional
        If set to the string ``'1.13'`` enables 1.13 legacy printing mode. This
        approximates numpy 1.13 print output by including a space in the sign
        position of floats and different behavior for 0d arrays. This also
        enables 1.21 legacy printing mode (described below).

        If set to the string ``'1.21'`` enables 1.21 legacy printing mode. This
        approximates numpy 1.21 print output of complex structured dtypes
        by not inserting spaces after commas that separate fields and after
        colons.

        If set to ``'1.25'`` approximates printing of 1.25 which mainly means
        that numeric scalars are printed without their type information, e.g.
        as ``3.0`` rather than ``np.float64(3.0)``.

        If set to ``'2.1'``, shape information is not given when arrays are
        summarized (i.e., multiple elements replaced with ``...``).

        If set to `False`, disables legacy mode.

        Unrecognized strings will be ignored with a warning for forward
        compatibility.

        .. versionchanged:: 1.22.0
        .. versionchanged:: 2.2

    override_repr: callable, optional
        If set a passed function will be used for generating arrays' repr.
        Other options will be ignored.

    See Also
    --------
    get_printoptions, printoptions, array2string

    Notes
    -----
    `formatter` is always reset with a call to `set_printoptions`.

    Use `printoptions` as a context manager to set the values temporarily.

    Examples
    --------
    Floating point precision can be set:

    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> np.array([1.123456789])
    [1.1235]

    Long arrays can be summarised:

    >>> np.set_printoptions(threshold=5)
    >>> np.arange(10)
    array([0, 1, 2, ..., 7, 8, 9], shape=(10,))

    Small results can be suppressed:

    >>> eps = np.finfo(float).eps
    >>> x = np.arange(4.)
    >>> x**2 - (x + eps)**2
    array([-4.9304e-32, -4.4409e-16,  0.0000e+00,  0.0000e+00])
    >>> np.set_printoptions(suppress=True)
    >>> x**2 - (x + eps)**2
    array([-0., -0.,  0.,  0.])

    A custom formatter can be used to display array elements as desired:

    >>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})
    >>> x = np.arange(3)
    >>> x
    array([int: 0, int: -1, int: -2])
    >>> np.set_printoptions()  # formatter gets reset
    >>> x
    array([0, 1, 2])

    To put back the default options, you can use:

    >>> np.set_printoptions(edgeitems=3, infstr='inf',
    ... linewidth=75, nanstr='nan', precision=8,
    ... suppress=False, threshold=1000, formatter=None)

    Also to temporarily override options, use `printoptions`
    as a context manager:

    >>> with np.printoptions(precision=2, suppress=True, threshold=5):
    ...     np.linspace(0, 10, 10)
    array([ 0.  ,  1.11,  2.22, ...,  7.78,  8.89, 10.  ], shape=(10,))

    """
    _set_printoptions(precision, threshold, edgeitems, linewidth, suppress,
                      nanstr, infstr, formatter, sign, floatmode,
                      legacy=legacy, override_repr=override_repr)


def _set_printoptions(precision=None, threshold=None, edgeitems=None,
                      linewidth=None, suppress=None, nanstr=None,
                      infstr=None, formatter=None, sign=None, floatmode=None,
                      *, legacy=None, override_repr=None):
    new_opt = _make_options_dict(precision, threshold, edgeitems, linewidth,
                                 suppress, nanstr, infstr, sign, formatter,
                                 floatmode, legacy)
    # formatter and override_repr are always reset
    new_opt['formatter'] = formatter
    new_opt['override_repr'] = override_repr

    updated_opt = format_options.get() | new_opt
    updated_opt.update(new_opt)

    if updated_opt['legacy'] == 113:
        updated_opt['sign'] = '-'

    return format_options.set(updated_opt)


@set_module('numpy')
def get_printoptions():
    """
    Return the current print options.

    Returns
    -------
    print_opts : dict
        Dictionary of current print options with keys

        - precision : int
        - threshold : int
        - edgeitems : int
        - linewidth : int
        - suppress : bool
        - nanstr : str
        - infstr : str
        - sign : str
        - formatter : dict of callables
        - floatmode : str
        - legacy : str or False

        For a full description of these options, see `set_printoptions`.

    See Also
    --------
    set_printoptions, printoptions

    Examples
    --------
    >>> import numpy as np

    >>> np.get_printoptions()
    {'edgeitems': 3, 'threshold': 1000, ..., 'override_repr': None}

    >>> np.get_printoptions()['linewidth']
    75
    >>> np.set_printoptions(linewidth=100)
    >>> np.get_printoptions()['linewidth']
    100

    """
    opts = format_options.get().copy()
    opts['legacy'] = {
        113: '1.13', 121: '1.21', 125: '1.25', sys.maxsize: False,
    }[opts['legacy']]
    return opts


def _get_legacy_print_mode():
    """Return the legacy print mode as an int."""
    return format_options.get()['legacy']


@set_module('numpy')
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Context manager for setting print options.

    Set print options for the scope of the `with` block, and restore the old
    options at the end. See `set_printoptions` for the full description of
    available options.

    Examples
    --------
    >>> import numpy as np

    >>> from numpy.testing import assert_equal
    >>> with np.printoptions(precision=2):
    ...     np.array([2.0]) / 3
    array([0.67])

    The `as`-clause of the `with`-statement gives the current print options:

    >>> with np.printoptions(precision=2) as opts:
    ...      assert_equal(opts, np.get_printoptions())

    See Also
    --------
    set_printoptions, get_printoptions

    """
    token = _set_printoptions(*args, **kwargs)

    try:
        yield get_printoptions()
    finally:
        format_options.reset(token)


def _leading_trailing(a, edgeitems, index=()):
    """
    Keep only the N-D corners (leading and trailing edges) of an array.

    Should be passed a base-class ndarray, since it makes no guarantees about
    preserving subclasses.
    """
    axis = len(index)
    if axis == a.ndim:
        return a[index]

    if a.shape[axis] > 2*edgeitems:
        return concatenate((
            _leading_trailing(a, edgeitems, index + np.index_exp[:edgeitems]),
            _leading_trailing(a, edgeitems, index + np.index_exp[-edgeitems:])
        ), axis=axis)
    else:
        return _leading_trailing(a, edgeitems, index + np.index_exp[:])


def _object_format(o):
    """ Object arrays containing lists should be printed unambiguously """
    if type(o) is list:
        fmt = 'list({!r})'
    else:
        fmt = '{!r}'
    return fmt.format(o)

def repr_format(x):
    if isinstance(x, (np.str_, np.bytes_)):
        return repr(x.item())
    return repr(x)

def str_format(x):
    if isinstance(x, (np.str_, np.bytes_)):
        return str(x.item())
    return str(x)

def _get_formatdict(data, *, precision, floatmode, suppress, sign, legacy,
                    formatter, **kwargs):
    # note: extra arguments in kwargs are ignored

    # wrapped in lambdas to avoid taking a code path
    # with the wrong type of data
    formatdict = {
        'bool': lambda: BoolFormat(data),
        'int': lambda: IntegerFormat(data, sign),
        'float': lambda: FloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),
        'longfloat': lambda: FloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),
        'complexfloat': lambda: ComplexFloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),
        'longcomplexfloat': lambda: ComplexFloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),
        'datetime': lambda: DatetimeFormat(data, legacy=legacy),
        'timedelta': lambda: TimedeltaFormat(data),
        'object': lambda: _object_format,
        'void': lambda: str_format,
        'numpystr': lambda: repr_format}

    # we need to wrap values in `formatter` in a lambda, so that the interface
    # is the same as the above values.
    def indirect(x):
        return lambda: x

    if formatter is not None:
        fkeys = [k for k in formatter.keys() if formatter[k] is not None]
        if 'all' in fkeys:
            for key in formatdict.keys():
                formatdict[key] = indirect(formatter['all'])
        if 'int_kind' in fkeys:
            for key in ['int']:
                formatdict[key] = indirect(formatter['int_kind'])
        if 'float_kind' in fkeys:
            for key in ['float', 'longfloat']:
                formatdict[key] = indirect(formatter['float_kind'])
        if 'complex_kind' in fkeys:
            for key in ['complexfloat', 'longcomplexfloat']:
                formatdict[key] = indirect(formatter['complex_kind'])
        if 'str_kind' in fkeys:
            formatdict['numpystr'] = indirect(formatter['str_kind'])
        for key in formatdict.keys():
            if key in fkeys:
                formatdict[key] = indirect(formatter[key])

    return formatdict

def _get_format_function(data, **options):
    """
    find the right formatting function for the dtype_
    """
    dtype_ = data.dtype
    dtypeobj = dtype_.type
    formatdict = _get_formatdict(data, **options)
    if dtypeobj is None:
        return formatdict["numpystr"]()
    elif issubclass(dtypeobj, _nt.bool):
        return formatdict['bool']()
    elif issubclass(dtypeobj, _nt.integer):
        if issubclass(dtypeobj, _nt.timedelta64):
            return formatdict['timedelta']()
        else:
            return formatdict['int']()
    elif issubclass(dtypeobj, _nt.floating):
        if issubclass(dtypeobj, _nt.longdouble):
            return formatdict['longfloat']()
        else:
            return formatdict['float']()
    elif issubclass(dtypeobj, _nt.complexfloating):
        if issubclass(dtypeobj, _nt.clongdouble):
            return formatdict['longcomplexfloat']()
        else:
            return formatdict['complexfloat']()
    elif issubclass(dtypeobj, (_nt.str_, _nt.bytes_)):
        return formatdict['numpystr']()
    elif issubclass(dtypeobj, _nt.datetime64):
        return formatdict['datetime']()
    elif issubclass(dtypeobj, _nt.object_):
        return formatdict['object']()
    elif issubclass(dtypeobj, _nt.void):
        if dtype_.names is not None:
            return StructuredVoidFormat.from_data(data, **options)
        else:
            return formatdict['void']()
    else:
        return formatdict['numpystr']()


def _recursive_guard(fillvalue='...'):
    """
    Like the python 3.2 reprlib.recursive_repr, but forwards *args and **kwargs

    Decorates a function such that if it calls itself with the same first
    argument, it returns `fillvalue` instead of recursing.

    Largely copied from reprlib.recursive_repr
    """

    def decorating_function(f):
        repr_running = set()

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                return f(self, *args, **kwargs)
            finally:
                repr_running.discard(key)

        return wrapper

    return decorating_function


# gracefully handle recursive calls, when object arrays contain themselves
@_recursive_guard()
def _array2string(a, options, separator=' ', prefix=""):
    # The formatter __init__s in _get_format_function cannot deal with
    # subclasses yet, and we also need to avoid recursion issues in
    # _formatArray with subclasses which return 0d arrays in place of scalars
    data = asarray(a)
    if a.shape == ():
        a = data

    if a.size > options['threshold']:
        summary_insert = "..."
        data = _leading_trailing(data, options['edgeitems'])
    else:
        summary_insert = ""

    # find the right formatting function for the array
    format_function = _get_format_function(data, **options)

    # skip over "["
    next_line_prefix = " "
    # skip over array(
    next_line_prefix += " "*len(prefix)

    lst = _formatArray(a, format_function, options['linewidth'],
                       next_line_prefix, separator, options['edgeitems'],
                       summary_insert, options['legacy'])
    return lst


def _array2string_dispatcher(
        a, max_line_width=None, precision=None,
        suppress_small=None, separator=None, prefix=None,
        style=None, formatter=None, threshold=None,
        edgeitems=None, sign=None, floatmode=None, suffix=None,
        *, legacy=None):
    return (a,)


@array_function_dispatch(_array2string_dispatcher, module='numpy')
def array2string(a, max_line_width=None, precision=None,
                 suppress_small=None, separator=' ', prefix="",
                 style=np._NoValue, formatter=None, threshold=None,
                 edgeitems=None, sign=None, floatmode=None, suffix="",
                 *, legacy=None):
    """
    Return a string representation of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int or None, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.
    separator : str, optional
        Inserted between elements.
    prefix : str, optional
    suffix : str, optional
        The length of the prefix and suffix strings are used to respectively
        align and wrap the output. An array is typically printed as::

          prefix + array2string(a) + suffix

        The output is left-padded by the length of the prefix string, and
        wrapping is forced at the column ``max_line_width - len(suffix)``.
        It should be noted that the content of prefix and suffix strings are
        not included in the output.
    style : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.14.0
    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are:

        - 'bool'
        - 'int'
        - 'timedelta' : a `numpy.timedelta64`
        - 'datetime' : a `numpy.datetime64`
        - 'float'
        - 'longfloat' : 128-bit floats
        - 'complexfloat'
        - 'longcomplexfloat' : composed of two 128-bit floats
        - 'void' : type `numpy.void`
        - 'numpystr' : types `numpy.bytes_` and `numpy.str_`

        Other keys that can be used to set a group of types at once are:

        - 'all' : sets all types
        - 'int_kind' : sets 'int'
        - 'float_kind' : sets 'float' and 'longfloat'
        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
        - 'str_kind' : sets 'numpystr'
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr.
        Defaults to ``numpy.get_printoptions()['threshold']``.
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension.
        Defaults to ``numpy.get_printoptions()['edgeitems']``.
    sign : string, either '-', '+', or ' ', optional
        Controls printing of the sign of floating-point types. If '+', always
        print the sign of positive values. If ' ', always prints a space
        (whitespace character) in the sign position of positive values.  If
        '-', omit the sign character of positive values.
        Defaults to ``numpy.get_printoptions()['sign']``.

        .. versionchanged:: 2.0
             The sign parameter can now be an integer type, previously
             types were floating-point types.

    floatmode : str, optional
        Controls the interpretation of the `precision` option for
        floating-point types.
        Defaults to ``numpy.get_printoptions()['floatmode']``.
        Can take the following values:

        - 'fixed': Always print exactly `precision` fractional digits,
          even if this would print more or fewer digits than
          necessary to specify the value uniquely.
        - 'unique': Print the minimum number of fractional digits necessary
          to represent each value uniquely. Different elements may
          have a different number of digits.  The value of the
          `precision` option is ignored.
        - 'maxprec': Print at most `precision` fractional digits, but if
          an element can be uniquely represented with fewer digits
          only print it with that many.
        - 'maxprec_equal': Print at most `precision` fractional digits,
          but if every element in the array can be uniquely
          represented with an equal number of fewer digits, use that
          many digits for all elements.
    legacy : string or `False`, optional
        If set to the string ``'1.13'`` enables 1.13 legacy printing mode. This
        approximates numpy 1.13 print output by including a space in the sign
        position of floats and different behavior for 0d arrays. If set to
        `False`, disables legacy mode. Unrecognized strings will be ignored
        with a warning for forward compatibility.

    Returns
    -------
    array_str : str
        String representation of the array.

    Raises
    ------
    TypeError
        if a callable in `formatter` does not return a string.

    See Also
    --------
    array_str, array_repr, set_printoptions, get_printoptions

    Notes
    -----
    If a formatter is specified for a certain type, the `precision` keyword is
    ignored for that type.

    This is a very flexible function; `array_repr` and `array_str` are using
    `array2string` internally so keywords with the same name should work
    identically in all three functions.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1e-16,1,2,3])
    >>> np.array2string(x, precision=2, separator=',',
    ...                       suppress_small=True)
    '[0.,1.,2.,3.]'

    >>> x  = np.arange(3.)
    >>> np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
    '[0.00 1.00 2.00]'

    >>> x  = np.arange(3)
    >>> np.array2string(x, formatter={'int':lambda x: hex(x)})
    '[0x0 0x1 0x2]'

    """

    overrides = _make_options_dict(precision, threshold, edgeitems,
                                   max_line_width, suppress_small, None, None,
                                   sign, formatter, floatmode, legacy)
    options = format_options.get().copy()
    options.update(overrides)

    if options['legacy'] <= 113:
        if style is np._NoValue:
            style = repr

        if a.shape == () and a.dtype.names is None:
            return style(a.item())
    elif style is not np._NoValue:
        # Deprecation 11-9-2017  v1.14
        warnings.warn("'style' argument is deprecated and no longer functional"
                      " except in 1.13 'legacy' mode",
                      DeprecationWarning, stacklevel=2)

    if options['legacy'] > 113:
        options['linewidth'] -= len(suffix)

    # treat as a null array if any of shape elements == 0
    if a.size == 0:
        return "[]"

    return _array2string(a, options, separator, prefix)


def _extendLine(s, line, word, line_width, next_line_prefix, legacy):
    needs_wrap = len(line) + len(word) > line_width
    if legacy > 113:
        # don't wrap lines if it won't help
        if len(line) <= len(next_line_prefix):
            needs_wrap = False

    if needs_wrap:
        s += line.rstrip() + "\n"
        line = next_line_prefix
    line += word
    return s, line


def _extendLine_pretty(s, line, word, line_width, next_line_prefix, legacy):
    """
    Extends line with nicely formatted (possibly multi-line) string ``word``.
    """
    words = word.splitlines()
    if len(words) == 1 or legacy <= 113:
        return _extendLine(s, line, word, line_width, next_line_prefix, legacy)

    max_word_length = max(len(word) for word in words)
    if (len(line) + max_word_length > line_width and
            len(line) > len(next_line_prefix)):
        s += line.rstrip() + '\n'
        line = next_line_prefix + words[0]
        indent = next_line_prefix
    else:
        indent = len(line)*' '
        line += words[0]

    for word in words[1::]:
        s += line.rstrip() + '\n'
        line = indent + word

    suffix_length = max_word_length - len(words[-1])
    line += suffix_length*' '

    return s, line

def _formatArray(a, format_function, line_width, next_line_prefix,
                 separator, edge_items, summary_insert, legacy):
    """formatArray is designed for two modes of operation:

    1. Full output

    2. Summarized output

    """
    def recurser(index, hanging_indent, curr_width):
        """
        By using this local function, we don't need to recurse with all the
        arguments. Since this function is not created recursively, the cost is
        not significant
        """
        axis = len(index)
        axes_left = a.ndim - axis

        if axes_left == 0:
            return format_function(a[index])

        # when recursing, add a space to align with the [ added, and reduce the
        # length of the line by 1
        next_hanging_indent = hanging_indent + ' '
        if legacy <= 113:
            next_width = curr_width
        else:
            next_width = curr_width - len(']')

        a_len = a.shape[axis]
        show_summary = summary_insert and 2*edge_items < a_len
        if show_summary:
            leading_items = edge_items
            trailing_items = edge_items
        else:
            leading_items = 0
            trailing_items = a_len

        # stringify the array with the hanging indent on the first line too
        s = ''

        # last axis (rows) - wrap elements if they would not fit on one line
        if axes_left == 1:
            # the length up until the beginning of the separator / bracket
            if legacy <= 113:
                elem_width = curr_width - len(separator.rstrip())
            else:
                elem_width = curr_width - max(
                    len(separator.rstrip()), len(']')
                )

            line = hanging_indent
            for i in range(leading_items):
                word = recurser(index + (i,), next_hanging_indent, next_width)
                s, line = _extendLine_pretty(
                    s, line, word, elem_width, hanging_indent, legacy)
                line += separator

            if show_summary:
                s, line = _extendLine(
                    s, line, summary_insert, elem_width, hanging_indent, legacy
                )
                if legacy <= 113:
                    line += ", "
                else:
                    line += separator

            for i in range(trailing_items, 1, -1):
                word = recurser(index + (-i,), next_hanging_indent, next_width)
                s, line = _extendLine_pretty(
                    s, line, word, elem_width, hanging_indent, legacy)
                line += separator

            if legacy <= 113:
                # width of the separator is not considered on 1.13
                elem_width = curr_width
            word = recurser(index + (-1,), next_hanging_indent, next_width)
            s, line = _extendLine_pretty(
                s, line, word, elem_width, hanging_indent, legacy)

            s += line

        # other axes - insert newlines between rows
        else:
            s = ''
            line_sep = separator.rstrip() + '\n'*(axes_left - 1)

            for i in range(leading_items):
                nested = recurser(
                    index + (i,), next_hanging_indent, next_width
                )
                s += hanging_indent + nested + line_sep

            if show_summary:
                if legacy <= 113:
                    # trailing space, fixed nbr of newlines,
                    # and fixed separator
                    s += hanging_indent + summary_insert + ", \n"
                else:
                    s += hanging_indent + summary_insert + line_sep

            for i in range(trailing_items, 1, -1):
                nested = recurser(index + (-i,), next_hanging_indent,
                                  next_width)
                s += hanging_indent + nested + line_sep

            nested = recurser(index + (-1,), next_hanging_indent, next_width)
            s += hanging_indent + nested

        # remove the hanging indent, and wrap in []
        s = '[' + s[len(hanging_indent):] + ']'
        return s

    try:
        # invoke the recursive part with an initial index and prefix
        return recurser(index=(),
                        hanging_indent=next_line_prefix,
                        curr_width=line_width)
    finally:
        # recursive closures have a cyclic reference to themselves, which
        # requires gc to collect (gh-10620). To avoid this problem, for
        # performance and PyPy friendliness, we break the cycle:
        recurser = None

def _none_or_positive_arg(x, name):
    if x is None:
        return -1
    if x < 0:
        raise ValueError("{} must be >= 0".format(name))
    return x

class FloatingFormat:
    """ Formatter for subtypes of np.floating """
    def __init__(self, data, precision, floatmode, suppress_small, sign=False,
                 *, legacy=None):
        # for backcompatibility, accept bools
        if isinstance(sign, bool):
            sign = '+' if sign else '-'

        self._legacy = legacy
        if self._legacy <= 113:
            # when not 0d, legacy does not support '-'
            if data.shape != () and sign == '-':
                sign = ' '

        self.floatmode = floatmode
        if floatmode == 'unique':
            self.precision = None
        else:
            self.precision = precision

        self.precision = _none_or_positive_arg(self.precision, 'precision')

        self.suppress_small = suppress_small
        self.sign = sign
        self.exp_format = False
        self.large_exponent = False
        self.fillFormat(data)

    def fillFormat(self, data):
        # only the finite values are used to compute the number of digits
        finite_vals = data[isfinite(data)]

        # choose exponential mode based on the non-zero finite values:
        abs_non_zero = absolute(finite_vals[finite_vals != 0])
        if len(abs_non_zero) != 0:
            max_val = np.max(abs_non_zero)
            min_val = np.min(abs_non_zero)
            with errstate(over='ignore'):  # division can overflow
                if max_val >= 1.e8 or (not self.suppress_small and
                        (min_val < 0.0001 or max_val/min_val > 1000.)):
                    self.exp_format = True

        # do a first pass of printing all the numbers, to determine sizes
        if len(finite_vals) == 0:
            self.pad_left = 0
            self.pad_right = 0
            self.trim = '.'
            self.exp_size = -1
            self.unique = True
            self.min_digits = None
        elif self.exp_format:
            trim, unique = '.', True
            if self.floatmode == 'fixed' or self._legacy <= 113:
                trim, unique = 'k', False
            strs = (dragon4_scientific(x, precision=self.precision,
                               unique=unique, trim=trim, sign=self.sign == '+')
                    for x in finite_vals)
            frac_strs, _, exp_strs = zip(*(s.partition('e') for s in strs))
            int_part, frac_part = zip(*(s.split('.') for s in frac_strs))
            self.exp_size = max(len(s) for s in exp_strs) - 1

            self.trim = 'k'
            self.precision = max(len(s) for s in frac_part)
            self.min_digits = self.precision
            self.unique = unique

            # for back-compat with np 1.13, use 2 spaces & sign and full prec
            if self._legacy <= 113:
                self.pad_left = 3
            else:
                # this should be only 1 or 2. Can be calculated from sign.
                self.pad_left = max(len(s) for s in int_part)
            # pad_right is only needed for nan length calculation
            self.pad_right = self.exp_size + 2 + self.precision
        else:
            trim, unique = '.', True
            if self.floatmode == 'fixed':
                trim, unique = 'k', False
            strs = (dragon4_positional(x, precision=self.precision,
                                       fractional=True,
                                       unique=unique, trim=trim,
                                       sign=self.sign == '+')
                    for x in finite_vals)
            int_part, frac_part = zip(*(s.split('.') for s in strs))
            if self._legacy <= 113:
                self.pad_left = 1 + max(len(s.lstrip('-+')) for s in int_part)
            else:
                self.pad_left = max(len(s) for s in int_part)
            self.pad_right = max(len(s) for s in frac_part)
            self.exp_size = -1
            self.unique = unique

            if self.floatmode in ['fixed', 'maxprec_equal']:
                self.precision = self.min_digits = self.pad_right
                self.trim = 'k'
            else:
                self.trim = '.'
                self.min_digits = 0

        if self._legacy > 113:
            # account for sign = ' ' by adding one to pad_left
            if self.sign == ' ' and not any(np.signbit(finite_vals)):
                self.pad_left += 1

        # if there are non-finite values, may need to increase pad_left
        if data.size != finite_vals.size:
            neginf = self.sign != '-' or any(data[isinf(data)] < 0)
            offset = self.pad_right + 1  # +1 for decimal pt
            current_options = format_options.get()
            self.pad_left = max(
                self.pad_left, len(current_options['nanstr']) - offset,
                len(current_options['infstr']) + neginf - offset
            )

    def __call__(self, x):
        if not np.isfinite(x):
            with errstate(invalid='ignore'):
                current_options = format_options.get()
                if np.isnan(x):
                    sign = '+' if self.sign == '+' else ''
                    ret = sign + current_options['nanstr']
                else:  # isinf
                    sign = '-' if x < 0 else '+' if self.sign == '+' else ''
                    ret = sign + current_options['infstr']
                return ' '*(
                    self.pad_left + self.pad_right + 1 - len(ret)
                ) + ret

        if self.exp_format:
            return dragon4_scientific(x,
                                      precision=self.precision,
                                      min_digits=self.min_digits,
                                      unique=self.unique,
                                      trim=self.trim,
                                      sign=self.sign == '+',
                                      pad_left=self.pad_left,
                                      exp_digits=self.exp_size)
        else:
            return dragon4_positional(x,
                                      precision=self.precision,
                                      min_digits=self.min_digits,
                                      unique=self.unique,
                                      fractional=True,
                                      trim=self.trim,
                                      sign=self.sign == '+',
                                      pad_left=self.pad_left,
                                      pad_right=self.pad_right)


@set_module('numpy')
def format_float_scientific(x, precision=None, unique=True, trim='k',
                            sign=False, pad_left=None, exp_digits=None,
                            min_digits=None):
    """
    Format a floating-point scalar as a decimal string in scientific notation.

    Provides control over rounding, trimming and padding. Uses and assumes
    IEEE unbiased rounding. Uses the "Dragon4" algorithm.

    Parameters
    ----------
    x : python float or numpy floating scalar
        Value to format.
    precision : non-negative integer or None, optional
        Maximum number of digits to print. May be None if `unique` is
        `True`, but must be an integer if unique is `False`.
    unique : boolean, optional
        If `True`, use a digit-generation strategy which gives the shortest
        representation which uniquely identifies the floating-point number from
        other values of the same type, by judicious rounding. If `precision`
        is given fewer digits than necessary can be printed. If `min_digits`
        is given more can be printed, in which cases the last digit is rounded
        with unbiased rounding.
        If `False`, digits are generated as if printing an infinite-precision
        value and stopping after `precision` digits, rounding the remaining
        value with unbiased rounding
    trim : one of 'k', '.', '0', '-', optional
        Controls post-processing trimming of trailing digits, as follows:

        * 'k' : keep trailing zeros, keep decimal point (no trimming)
        * '.' : trim all trailing zeros, leave decimal point
        * '0' : trim all but the zero before the decimal point. Insert the
          zero if it is missing.
        * '-' : trim trailing zeros and any trailing decimal point
    sign : boolean, optional
        Whether to show the sign for positive values.
    pad_left : non-negative integer, optional
        Pad the left side of the string with whitespace until at least that
        many characters are to the left of the decimal point.
    exp_digits : non-negative integer, optional
        Pad the exponent with zeros until it contains at least this
        many digits. If omitted, the exponent will be at least 2 digits.
    min_digits : non-negative integer or None, optional
        Minimum number of digits to print. This only has an effect for
        `unique=True`. In that case more digits than necessary to uniquely
        identify the value may be printed and rounded unbiased.

        .. versionadded:: 1.21.0

    Returns
    -------
    rep : string
        The string representation of the floating point value

    See Also
    --------
    format_float_positional

    Examples
    --------
    >>> import numpy as np
    >>> np.format_float_scientific(np.float32(np.pi))
    '3.1415927e+00'
    >>> s = np.float32(1.23e24)
    >>> np.format_float_scientific(s, unique=False, precision=15)
    '1.230000071797338e+24'
    >>> np.format_float_scientific(s, exp_digits=4)
    '1.23e+0024'
    """
    precision = _none_or_positive_arg(precision, 'precision')
    pad_left = _none_or_positive_arg(pad_left, 'pad_left')
    exp_digits = _none_or_positive_arg(exp_digits, 'exp_digits')
    min_digits = _none_or_positive_arg(min_digits, 'min_digits')
    if min_digits > 0 and precision > 0 and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    return dragon4_scientific(x, precision=precision, unique=unique,
                              trim=trim, sign=sign, pad_left=pad_left,
                              exp_digits=exp_digits, min_digits=min_digits)


@set_module('numpy')
def format_float_positional(x, precision=None, unique=True,
                            fractional=True, trim='k', sign=False,
                            pad_left=None, pad_right=None, min_digits=None):
    """
    Format a floating-point scalar as a decimal string in positional notation.

    Provides control over rounding, trimming and padding. Uses and assumes
    IEEE unbiased rounding. Uses the "Dragon4" algorithm.

    Parameters
    ----------
    x : python float or numpy floating scalar
        Value to format.
    precision : non-negative integer or None, optional
        Maximum number of digits to print. May be None if `unique` is
        `True`, but must be an integer if unique is `False`.
    unique : boolean, optional
        If `True`, use a digit-generation strategy which gives the shortest
        representation which uniquely identifies the floating-point number from
        other values of the same type, by judicious rounding. If `precision`
        is given fewer digits than necessary can be printed, or if `min_digits`
        is given more can be printed, in which cases the last digit is rounded
        with unbiased rounding.
        If `False`, digits are generated as if printing an infinite-precision
        value and stopping after `precision` digits, rounding the remaining
        value with unbiased rounding
    fractional : boolean, optional
        If `True`, the cutoffs of `precision` and `min_digits` refer to the
        total number of digits after the decimal point, including leading
        zeros.
        If `False`, `precision` and `min_digits` refer to the total number of
        significant digits, before or after the decimal point, ignoring leading
        zeros.
    trim : one of 'k', '.', '0', '-', optional
        Controls post-processing trimming of trailing digits, as follows:

        * 'k' : keep trailing zeros, keep decimal point (no trimming)
        * '.' : trim all trailing zeros, leave decimal point
        * '0' : trim all but the zero before the decimal point. Insert the
          zero if it is missing.
        * '-' : trim trailing zeros and any trailing decimal point
    sign : boolean, optional
        Whether to show the sign for positive values.
    pad_left : non-negative integer, optional
        Pad the left side of the string with whitespace until at least that
        many characters are to the left of the decimal point.
    pad_right : non-negative integer, optional
        Pad the right side of the string with whitespace until at least that
        many characters are to the right of the decimal point.
    min_digits : non-negative integer or None, optional
        Minimum number of digits to print. Only has an effect if `unique=True`
        in which case additional digits past those necessary to uniquely
        identify the value may be printed, rounding the last additional digit.

        .. versionadded:: 1.21.0

    Returns
    -------
    rep : string
        The string representation of the floating point value

    See Also
    --------
    format_float_scientific

    Examples
    --------
    >>> import numpy as np
    >>> np.format_float_positional(np.float32(np.pi))
    '3.1415927'
    >>> np.format_float_positional(np.float16(np.pi))
    '3.14'
    >>> np.format_float_positional(np.float16(0.3))
    '0.3'
    >>> np.format_float_positional(np.float16(0.3), unique=False, precision=10)
    '0.3000488281'
    """
    precision = _none_or_positive_arg(precision, 'precision')
    pad_left = _none_or_positive_arg(pad_left, 'pad_left')
    pad_right = _none_or_positive_arg(pad_right, 'pad_right')
    min_digits = _none_or_positive_arg(min_digits, 'min_digits')
    if not fractional and precision == 0:
        raise ValueError("precision must be greater than 0 if "
                         "fractional=False")
    if min_digits > 0 and precision > 0 and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    return dragon4_positional(x, precision=precision, unique=unique,
                              fractional=fractional, trim=trim,
                              sign=sign, pad_left=pad_left,
                              pad_right=pad_right, min_digits=min_digits)

class IntegerFormat:
    def __init__(self, data, sign='-'):
        if data.size > 0:
            data_max = np.max(data)
            data_min = np.min(data)
            data_max_str_len = len(str(data_max))
            if sign == ' ' and data_min < 0:
                sign = '-'
            if data_max >= 0 and sign in "+ ":
                data_max_str_len += 1
            max_str_len = max(data_max_str_len,
                              len(str(data_min)))
        else:
            max_str_len = 0
        self.format = f'{{:{sign}{max_str_len}d}}'

    def __call__(self, x):
        return self.format.format(x)

class BoolFormat:
    def __init__(self, data, **kwargs):
        # add an extra space so " True" and "False" have the same length and
        # array elements align nicely when printed, except in 0d arrays
        self.truestr = ' True' if data.shape != () else 'True'

    def __call__(self, x):
        return self.truestr if x else "False"


class ComplexFloatingFormat:
    """ Formatter for subtypes of np.complexfloating """
    def __init__(self, x, precision, floatmode, suppress_small,
                 sign=False, *, legacy=None):
        # for backcompatibility, accept bools
        if isinstance(sign, bool):
            sign = '+' if sign else '-'

        floatmode_real = floatmode_imag = floatmode
        if legacy <= 113:
            floatmode_real = 'maxprec_equal'
            floatmode_imag = 'maxprec'

        self.real_format = FloatingFormat(
            x.real, precision, floatmode_real, suppress_small,
            sign=sign, legacy=legacy
        )
        self.imag_format = FloatingFormat(
            x.imag, precision, floatmode_imag, suppress_small,
            sign='+', legacy=legacy
        )

    def __call__(self, x):
        r = self.real_format(x.real)
        i = self.imag_format(x.imag)

        # add the 'j' before the terminal whitespace in i
        sp = len(i.rstrip())
        i = i[:sp] + 'j' + i[sp:]

        return r + i


class _TimelikeFormat:
    def __init__(self, data):
        non_nat = data[~isnat(data)]
        if len(non_nat) > 0:
            # Max str length of non-NaT elements
            max_str_len = max(len(self._format_non_nat(np.max(non_nat))),
                              len(self._format_non_nat(np.min(non_nat))))
        else:
            max_str_len = 0
        if len(non_nat) < data.size:
            # data contains a NaT
            max_str_len = max(max_str_len, 5)
        self._format = '%{}s'.format(max_str_len)
        self._nat = "'NaT'".rjust(max_str_len)

    def _format_non_nat(self, x):
        # override in subclass
        raise NotImplementedError

    def __call__(self, x):
        if isnat(x):
            return self._nat
        else:
            return self._format % self._format_non_nat(x)


class DatetimeFormat(_TimelikeFormat):
    def __init__(self, x, unit=None, timezone=None, casting='same_kind',
                 legacy=False):
        # Get the unit from the dtype
        if unit is None:
            if x.dtype.kind == 'M':
                unit = datetime_data(x.dtype)[0]
            else:
                unit = 's'

        if timezone is None:
            timezone = 'naive'
        self.timezone = timezone
        self.unit = unit
        self.casting = casting
        self.legacy = legacy

        # must be called after the above are configured
        super().__init__(x)

    def __call__(self, x):
        if self.legacy <= 113:
            return self._format_non_nat(x)
        return super().__call__(x)

    def _format_non_nat(self, x):
        return "'%s'" % datetime_as_string(x,
                                    unit=self.unit,
                                    timezone=self.timezone,
                                    casting=self.casting)


class TimedeltaFormat(_TimelikeFormat):
    def _format_non_nat(self, x):
        return str(x.astype('i8'))


class SubArrayFormat:
    def __init__(self, format_function, **options):
        self.format_function = format_function
        self.threshold = options['threshold']
        self.edge_items = options['edgeitems']

    def __call__(self, a):
        self.summary_insert = "..." if a.size > self.threshold else ""
        return self.format_array(a)

    def format_array(self, a):
        if np.ndim(a) == 0:
            return self.format_function(a)

        if self.summary_insert and a.shape[0] > 2*self.edge_items:
            formatted = (
                [self.format_array(a_) for a_ in a[:self.edge_items]]
                + [self.summary_insert]
                + [self.format_array(a_) for a_ in a[-self.edge_items:]]
            )
        else:
            formatted = [self.format_array(a_) for a_ in a]

        return "[" + ", ".join(formatted) + "]"


class StructuredVoidFormat:
    """
    Formatter for structured np.void objects.

    This does not work on structured alias types like
    np.dtype(('i4', 'i2,i2')), as alias scalars lose their field information,
    and the implementation relies upon np.void.__getitem__.
    """
    def __init__(self, format_functions):
        self.format_functions = format_functions

    @classmethod
    def from_data(cls, data, **options):
        """
        This is a second way to initialize StructuredVoidFormat,
        using the raw data as input. Added to avoid changing
        the signature of __init__.
        """
        format_functions = []
        for field_name in data.dtype.names:
            format_function = _get_format_function(data[field_name], **options)
            if data.dtype[field_name].shape != ():
                format_function = SubArrayFormat(format_function, **options)
            format_functions.append(format_function)
        return cls(format_functions)

    def __call__(self, x):
        str_fields = [
            format_function(field)
            for field, format_function in zip(x, self.format_functions)
        ]
        if len(str_fields) == 1:
            return "({},)".format(str_fields[0])
        else:
            return "({})".format(", ".join(str_fields))


def _void_scalar_to_string(x, is_repr=True):
    """
    Implements the repr for structured-void scalars. It is called from the
    scalartypes.c.src code, and is placed here because it uses the elementwise
    formatters defined above.
    """
    options = format_options.get().copy()

    if options["legacy"] <= 125:
        return StructuredVoidFormat.from_data(array(x), **options)(x)

    if options.get('formatter') is None:
        options['formatter'] = {}
    options['formatter'].setdefault('float_kind', str)
    val_repr = StructuredVoidFormat.from_data(array(x), **options)(x)
    if not is_repr:
        return val_repr
    cls = type(x)
    cls_fqn = cls.__module__.replace("numpy", "np") + "." + cls.__name__
    void_dtype = np.dtype((np.void, x.dtype))
    return f"{cls_fqn}({val_repr}, dtype={void_dtype!s})"


_typelessdata = [int_, float64, complex128, _nt.bool]


def dtype_is_implied(dtype):
    """
    Determine if the given dtype is implied by the representation
    of its values.

    Parameters
    ----------
    dtype : dtype
        Data type

    Returns
    -------
    implied : bool
        True if the dtype is implied by the representation of its values.

    Examples
    --------
    >>> import numpy as np
    >>> np._core.arrayprint.dtype_is_implied(int)
    True
    >>> np.array([1, 2, 3], int)
    array([1, 2, 3])
    >>> np._core.arrayprint.dtype_is_implied(np.int8)
    False
    >>> np.array([1, 2, 3], np.int8)
    array([1, 2, 3], dtype=int8)
    """
    dtype = np.dtype(dtype)
    if format_options.get()['legacy'] <= 113 and dtype.type == np.bool:
        return False

    # not just void types can be structured, and names are not part of the repr
    if dtype.names is not None:
        return False

    # should care about endianness *unless size is 1* (e.g., int8, bool)
    if not dtype.isnative:
        return False

    return dtype.type in _typelessdata


def dtype_short_repr(dtype):
    """
    Convert a dtype to a short form which evaluates to the same dtype.

    The intent is roughly that the following holds

    >>> from numpy import *
    >>> dt = np.int64([1, 2]).dtype
    >>> assert eval(dtype_short_repr(dt)) == dt
    """
    if type(dtype).__repr__ != np.dtype.__repr__:
        # TODO: Custom repr for user DTypes, logic should likely move.
        return repr(dtype)
    if dtype.names is not None:
        # structured dtypes give a list or tuple repr
        return str(dtype)
    elif issubclass(dtype.type, flexible):
        # handle these separately so they don't give garbage like str256
        return "'%s'" % str(dtype)

    typename = dtype.name
    if not dtype.isnative:
        # deal with cases like dtype('<u2') that are identical to an
        # established dtype (in this case uint16)
        # except that they have a different endianness.
        return "'%s'" % str(dtype)
    # quote typenames which can't be represented as python variable names
    if typename and not (typename[0].isalpha() and typename.isalnum()):
        typename = repr(typename)
    return typename


def _array_repr_implementation(
        arr, max_line_width=None, precision=None, suppress_small=None,
        array2string=array2string):
    """Internal version of array_repr() that allows overriding array2string."""
    current_options = format_options.get()
    override_repr = current_options["override_repr"]
    if override_repr is not None:
        return override_repr(arr)

    if max_line_width is None:
        max_line_width = current_options['linewidth']

    if type(arr) is not ndarray:
        class_name = type(arr).__name__
    else:
        class_name = "array"

    prefix = class_name + "("
    if (current_options['legacy'] <= 113 and
            arr.shape == () and not arr.dtype.names):
        lst = repr(arr.item())
    else:
        lst = array2string(arr, max_line_width, precision, suppress_small,
                           ', ', prefix, suffix=")")

    # Add dtype and shape information if these cannot be inferred from
    # the array string.
    extras = []
    if (arr.size == 0 and arr.shape != (0,)
            or current_options['legacy'] > 210
            and arr.size > current_options['threshold']):
        extras.append(f"shape={arr.shape}")
    if not dtype_is_implied(arr.dtype) or arr.size == 0:
        extras.append(f"dtype={dtype_short_repr(arr.dtype)}")

    if not extras:
        return prefix + lst + ")"

    arr_str = prefix + lst + ","
    extra_str = ", ".join(extras) + ")"
    # compute whether we should put extras on a new line: Do so if adding the
    # extras would extend the last line past max_line_width.
    # Note: This line gives the correct result even when rfind returns -1.
    last_line_len = len(arr_str) - (arr_str.rfind('\n') + 1)
    spacer = " "
    if current_options['legacy'] <= 113:
        if issubclass(arr.dtype.type, flexible):
            spacer = '\n' + ' '*len(prefix)
    elif last_line_len + len(extra_str) + 1 > max_line_width:
        spacer = '\n' + ' '*len(prefix)

    return arr_str + spacer + extra_str


def _array_repr_dispatcher(
        arr, max_line_width=None, precision=None, suppress_small=None):
    return (arr,)


@array_function_dispatch(_array_repr_dispatcher, module='numpy')
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """
    Return the string representation of an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.

    Returns
    -------
    string : str
      The string representation of an array.

    See Also
    --------
    array_str, array2string, set_printoptions

    Examples
    --------
    >>> import numpy as np
    >>> np.array_repr(np.array([1,2]))
    'array([1, 2])'
    >>> np.array_repr(np.ma.array([0.]))
    'MaskedArray([0.])'
    >>> np.array_repr(np.array([], np.int32))
    'array([], dtype=int32)'

    >>> x = np.array([1e-6, 4e-7, 2, 3])
    >>> np.array_repr(x, precision=6, suppress_small=True)
    'array([0.000001,  0.      ,  2.      ,  3.      ])'

    """
    return _array_repr_implementation(
        arr, max_line_width, precision, suppress_small)


@_recursive_guard()
def _guarded_repr_or_str(v):
    if isinstance(v, bytes):
        return repr(v)
    return str(v)


def _array_str_implementation(
        a, max_line_width=None, precision=None, suppress_small=None,
        array2string=array2string):
    """Internal version of array_str() that allows overriding array2string."""
    if (format_options.get()['legacy'] <= 113 and
            a.shape == () and not a.dtype.names):
        return str(a.item())

    # the str of 0d arrays is a special case: It should appear like a scalar,
    # so floats are not truncated by `precision`, and strings are not wrapped
    # in quotes. So we return the str of the scalar value.
    if a.shape == ():
        # obtain a scalar and call str on it, avoiding problems for subclasses
        # for which indexing with () returns a 0d instead of a scalar by using
        # ndarray's getindex. Also guard against recursive 0d object arrays.
        return _guarded_repr_or_str(np.ndarray.__getitem__(a, ()))

    return array2string(a, max_line_width, precision, suppress_small, ' ', "")


def _array_str_dispatcher(
        a, max_line_width=None, precision=None, suppress_small=None):
    return (a,)


@array_function_dispatch(_array_str_dispatcher, module='numpy')
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """
    Return a string representation of the data in an array.

    The data in the array is returned as a single string.  This function is
    similar to `array_repr`, the difference being that `array_repr` also
    returns information on the kind of array and its data type.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.

    See Also
    --------
    array2string, array_repr, set_printoptions

    Examples
    --------
    >>> import numpy as np
    >>> np.array_str(np.arange(3))
    '[0 1 2]'

    """
    return _array_str_implementation(
        a, max_line_width, precision, suppress_small)


# needed if __array_function__ is disabled
_array2string_impl = getattr(array2string, '__wrapped__', array2string)
_default_array_str = functools.partial(_array_str_implementation,
                                       array2string=_array2string_impl)
_default_array_repr = functools.partial(_array_repr_implementation,
                                        array2string=_array2string_impl)
<|endoftext|>"""
Due to compatibility, numpy has a very large number of different naming
conventions for the scalar types (those subclassing from `numpy.generic`).
This file produces a convoluted set of dictionaries mapping names to types,
and sometimes other mappings too.

.. data:: allTypes
    A dictionary of names to types that will be exposed as attributes through
    ``np._core.numerictypes.*``

.. data:: sctypeDict
    Similar to `allTypes`, but maps a broader set of aliases to their types.

.. data:: sctypes
    A dictionary keyed by a "type group" string, providing a list of types
    under that group.

"""

import numpy._core.multiarray as ma
from numpy._core.multiarray import typeinfo, dtype

######################################
# Building `sctypeDict` and `allTypes`
######################################

sctypeDict = {}
allTypes = {}
c_names_dict = {}

_abstract_type_names = {
    "generic", "integer", "inexact", "floating", "number",
    "flexible", "character", "complexfloating", "unsignedinteger",
    "signedinteger"
}

for _abstract_type_name in _abstract_type_names:
    allTypes[_abstract_type_name] = getattr(ma, _abstract_type_name)

for k, v in typeinfo.items():
    if k.startswith("NPY_") and v not in c_names_dict:
        c_names_dict[k[4:]] = v
    else:
        concrete_type = v.type
        allTypes[k] = concrete_type
        sctypeDict[k] = concrete_type

_aliases = {
    "double": "float64",
    "cdouble": "complex128",
    "single": "float32",
    "csingle": "complex64",
    "half": "float16",
    "bool_": "bool",
    # Default integer:
    "int_": "intp",
    "uint": "uintp",
}

for k, v in _aliases.items():
    sctypeDict[k] = allTypes[v]
    allTypes[k] = allTypes[v]

# extra aliases are added only to `sctypeDict`
# to support dtype name access, such as`np.dtype("float")`
_extra_aliases = {
    "float": "float64",
    "complex": "complex128",
    "object": "object_",
    "bytes": "bytes_",
    "a": "bytes_",
    "int": "int_",
    "str": "str_",
    "unicode": "str_",
}

for k, v in _extra_aliases.items():
    sctypeDict[k] = allTypes[v]

# include extended precision sized aliases
for is_complex, full_name in [(False, "longdouble"), (True, "clongdouble")]:
    longdouble_type: type = allTypes[full_name]

    bits: int = dtype(longdouble_type).itemsize * 8
    base_name: str = "complex" if is_complex else "float"
    extended_prec_name: str = f"{base_name}{bits}"
    if extended_prec_name not in allTypes:
        sctypeDict[extended_prec_name] = longdouble_type
        allTypes[extended_prec_name] = longdouble_type


####################
# Building `sctypes`
####################

sctypes = {"int": set(), "uint": set(), "float": set(),
           "complex": set(), "others": set()}

for type_info in typeinfo.values():
    if type_info.kind in ["M", "m"]:  # exclude timedelta and datetime
        continue

    concrete_type = type_info.type

    # find proper group for each concrete type
    for type_group, abstract_type in [
        ("int", ma.signedinteger), ("uint", ma.unsignedinteger),
        ("float", ma.floating), ("complex", ma.complexfloating),
        ("others", ma.generic)
    ]:
        if issubclass(concrete_type, abstract_type):
            sctypes[type_group].add(concrete_type)
            break

# sort sctype groups by bitsize
for sctype_key in sctypes.keys():
    sctype_list = list(sctypes[sctype_key])
    sctype_list.sort(key=lambda x: dtype(x).itemsize)
    sctypes[sctype_key] = sctype_list
<|endoftext|>"""
This module contains a set of functions for vectorized string
operations.
"""

import sys
import numpy as np
from numpy import (
    equal, not_equal, less, less_equal, greater, greater_equal,
    add, multiply as _multiply_ufunc,
)
from numpy._core.multiarray import _vec_string
from numpy._core.overrides import set_module
from numpy._core.umath import (
    isalpha,
    isdigit,
    isspace,
    isalnum,
    islower,
    isupper,
    istitle,
    isdecimal,
    isnumeric,
    str_len,
    find as _find_ufunc,
    rfind as _rfind_ufunc,
    index as _index_ufunc,
    rindex as _rindex_ufunc,
    count as _count_ufunc,
    startswith as _startswith_ufunc,
    endswith as _endswith_ufunc,
    _lstrip_whitespace,
    _lstrip_chars,
    _rstrip_whitespace,
    _rstrip_chars,
    _strip_whitespace,
    _strip_chars,
    _replace,
    _expandtabs_length,
    _expandtabs,
    _center,
    _ljust,
    _rjust,
    _zfill,
    _partition,
    _partition_index,
    _rpartition,
    _rpartition_index,
)


def _override___module__():
    for ufunc in [
        isalnum, isalpha, isdecimal, isdigit, islower, isnumeric, isspace,
        istitle, isupper, str_len,
    ]:
        ufunc.__module__ = "numpy.strings"
        ufunc.__qualname__ = ufunc.__name__


_override___module__()


__all__ = [
    # UFuncs
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "add", "multiply", "isalpha", "isdigit", "isspace", "isalnum", "islower",
    "isupper", "istitle", "isdecimal", "isnumeric", "str_len", "find",
    "rfind", "index", "rindex", "count", "startswith", "endswith", "lstrip",
    "rstrip", "strip", "replace", "expandtabs", "center", "ljust", "rjust",
    "zfill", "partition", "rpartition",

    # _vec_string - Will gradually become ufuncs as well
    "upper", "lower", "swapcase", "capitalize", "title",

    # _vec_string - Will probably not become ufuncs
    "mod", "decode", "encode", "translate",

    # Removed from namespace until behavior has been crystallized
    # "join", "split", "rsplit", "splitlines",
]


MAX = np.iinfo(np.int64).max


def _get_num_chars(a):
    """
    Helper function that returns the number of characters per field in
    a string or unicode array.  This is to abstract out the fact that
    for a unicode array this is itemsize / 4.
    """
    if issubclass(a.dtype.type, np.str_):
        return a.itemsize // 4
    return a.itemsize


def _to_bytes_or_str_array(result, output_dtype_like):
    """
    Helper function to cast a result back into an array
    with the appropriate dtype if an object array must be used
    as an intermediary.
    """
    output_dtype_like = np.asarray(output_dtype_like)
    if result.size == 0:
        # Calling asarray & tolist in an empty array would result
        # in losing shape information
        return result.astype(output_dtype_like.dtype)
    ret = np.asarray(result.tolist())
    if isinstance(output_dtype_like.dtype, np.dtypes.StringDType):
        return ret.astype(type(output_dtype_like.dtype))
    return ret.astype(type(output_dtype_like.dtype)(_get_num_chars(ret)))


def _clean_args(*args):
    """
    Helper function for delegating arguments to Python string
    functions.

    Many of the Python string operations that have optional arguments
    do not use 'None' to indicate a default value.  In these cases,
    we need to remove all None arguments, and those following them.
    """
    newargs = []
    for chk in args:
        if chk is None:
            break
        newargs.append(chk)
    return newargs


@set_module("numpy.strings")
def multiply(a, i):
    """
    Return (a * i), that is string multiple concatenation,
    element-wise.

    Values in ``i`` of less than 0 are treated as 0 (which yields an
    empty string).

    Parameters
    ----------
    a : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype

    i : array_like, with any integer dtype

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["a", "b", "c"])
    >>> np.strings.multiply(a, 3)
    array(['aaa', 'bbb', 'ccc'], dtype='<U3')
    >>> i = np.array([1, 2, 3])
    >>> np.strings.multiply(a, i)
    array(['a', 'bb', 'ccc'], dtype='<U3')
    >>> np.strings.multiply(np.array(['a']), i)
    array(['a', 'aa', 'aaa'], dtype='<U3')
    >>> a = np.array(['a', 'b', 'c', 'd', 'e', 'f']).reshape((2, 3))
    >>> np.strings.multiply(a, 3)
    array([['aaa', 'bbb', 'ccc'],
           ['ddd', 'eee', 'fff']], dtype='<U3')
    >>> np.strings.multiply(a, i)
    array([['a', 'bb', 'ccc'],
           ['d', 'ee', 'fff']], dtype='<U3')

    """
    a = np.asanyarray(a)

    i = np.asanyarray(i)
    if not np.issubdtype(i.dtype, np.integer):
        raise TypeError(f"unsupported type {i.dtype} for operand 'i'")
    i = np.maximum(i, 0)

    # delegate to stringdtype loops that also do overflow checking
    if a.dtype.char == "T":
        return a * i

    a_len = str_len(a)

    # Ensure we can do a_len * i without overflow.
    if np.any(a_len > sys.maxsize / np.maximum(i, 1)):
        raise MemoryError("repeated string is too long")

    buffersizes = a_len * i
    out_dtype = f"{a.dtype.char}{buffersizes.max()}"
    out = np.empty_like(a, shape=buffersizes.shape, dtype=out_dtype)
    return _multiply_ufunc(a, i, out=out)


@set_module("numpy.strings")
def mod(a, values):
    """
    Return (a % i), that is pre-Python 2.6 string formatting
    (interpolation), element-wise for a pair of array_likes of str
    or unicode.

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    values : array_like of values
       These values will be element-wise interpolated into the string.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["NumPy is a %s library"])
    >>> np.strings.mod(a, values=["Python"])
    array(['NumPy is a Python library'], dtype='<U25')

    >>> a = np.array([b'%d bytes', b'%d bits'])
    >>> values = np.array([8, 64])
    >>> np.strings.mod(a, values)
    array([b'8 bytes', b'64 bits'], dtype='|S7')

    """
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, '__mod__', (values,)), a)


@set_module("numpy.strings")
def find(a, sub, start=0, end=None):
    """
    For each element, return the lowest index in the string where
    substring ``sub`` is found, such that ``sub`` is contained in the
    range [``start``, ``end``).

    Parameters
    ----------
    a : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype

    sub : array_like, with `np.bytes_` or `np.str_` dtype
        The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.find

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["NumPy is a Python library"])
    >>> np.strings.find(a, "Python")
    array([11])

    """
    end = end if end is not None else MAX
    return _find_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def rfind(a, sub, start=0, end=None):
    """
    For each element, return the highest index in the string where
    substring ``sub`` is found, such that ``sub`` is contained in the
    range [``start``, ``end``).

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sub : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.rfind

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["Computer Science"])
    >>> np.strings.rfind(a, "Science", start=0, end=None)
    array([9])
    >>> np.strings.rfind(a, "Science", start=0, end=8)
    array([-1])
    >>> b = np.array(["Computer Science", "Science"])
    >>> np.strings.rfind(b, "Science", start=0, end=None)
    array([9, 0])

    """
    end = end if end is not None else MAX
    return _rfind_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def index(a, sub, start=0, end=None):
    """
    Like `find`, but raises :exc:`ValueError` when the substring is not found.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sub : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    start, end : array_like, with any integer dtype, optional

    Returns
    -------
    out : ndarray
        Output array of ints.

    See Also
    --------
    find, str.index

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["Computer Science"])
    >>> np.strings.index(a, "Science", start=0, end=None)
    array([9])

    """
    end = end if end is not None else MAX
    return _index_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def rindex(a, sub, start=0, end=None):
    """
    Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is
    not found.

    Parameters
    ----------
    a : array-like, with `np.bytes_` or `np.str_` dtype

    sub : array-like, with `np.bytes_` or `np.str_` dtype

    start, end : array-like, with any integer dtype, optional

    Returns
    -------
    out : ndarray
        Output array of ints.

    See Also
    --------
    rfind, str.rindex

    Examples
    --------
    >>> a = np.array(["Computer Science"])
    >>> np.strings.rindex(a, "Science", start=0, end=None)
    array([9])

    """
    end = end if end is not None else MAX
    return _rindex_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def count(a, sub, start=0, end=None):
    """
    Returns an array with the number of non-overlapping occurrences of
    substring ``sub`` in the range [``start``, ``end``).

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sub : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
       The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.count

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.count(c, 'A')
    array([3, 1, 1])
    >>> np.strings.count(c, 'aA')
    array([3, 1, 0])
    >>> np.strings.count(c, 'A', start=1, end=4)
    array([2, 1, 1])
    >>> np.strings.count(c, 'A', start=1, end=3)
    array([1, 0, 0])

    """
    end = end if end is not None else MAX
    return _count_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def startswith(a, prefix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in ``a`` starts with ``prefix``, otherwise `False`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    prefix : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    start, end : array_like, with any integer dtype
        With ``start``, test beginning at that position. With ``end``,
        stop comparing at that position.

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.startswith

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array(['foo', 'bar'])
    >>> s
    array(['foo', 'bar'], dtype='<U3')
    >>> np.strings.startswith(s, 'fo')
    array([True,  False])
    >>> np.strings.startswith(s, 'o', start=1, end=2)
    array([True,  False])

    """
    end = end if end is not None else MAX
    return _startswith_ufunc(a, prefix, start, end)


@set_module("numpy.strings")
def endswith(a, suffix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in ``a`` ends with ``suffix``, otherwise `False`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    suffix : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    start, end : array_like, with any integer dtype
        With ``start``, test beginning at that position. With ``end``,
        stop comparing at that position.

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.endswith

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array(['foo', 'bar'])
    >>> s
    array(['foo', 'bar'], dtype='<U3')
    >>> np.strings.endswith(s, 'ar')
    array([False,  True])
    >>> np.strings.endswith(s, 'a', start=1, end=2)
    array([False,  True])

    """
    end = end if end is not None else MAX
    return _endswith_ufunc(a, suffix, start, end)


@set_module("numpy.strings")
def decode(a, encoding=None, errors=None):
    r"""
    Calls :meth:`bytes.decode` element-wise.

    The set of available codecs comes from the Python standard library,
    and may be extended at runtime.  For more information, see the
    :mod:`codecs` module.

    Parameters
    ----------
    a : array_like, with ``bytes_`` dtype

    encoding : str, optional
       The name of an encoding

    errors : str, optional
       Specifies how to handle encoding errors

    Returns
    -------
    out : ndarray

    See Also
    --------
    :py:meth:`bytes.decode`

    Notes
    -----
    The type of the result will depend on the encoding specified.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
    ...               b'\x81\x82\xc2\xc1\xc2\x82\x81'])
    >>> c
    array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
           b'\x81\x82\xc2\xc1\xc2\x82\x81'], dtype='|S7')
    >>> np.strings.decode(c, encoding='cp037')
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')

    """
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, 'decode', _clean_args(encoding, errors)),
        np.str_(''))


@set_module("numpy.strings")
def encode(a, encoding=None, errors=None):
    """
    Calls :meth:`str.encode` element-wise.

    The set of available codecs comes from the Python standard library,
    and may be extended at runtime. For more information, see the
    :mod:`codecs` module.

    Parameters
    ----------
    a : array_like, with ``StringDType`` or ``str_`` dtype

    encoding : str, optional
       The name of an encoding

    errors : str, optional
       Specifies how to handle encoding errors

    Returns
    -------
    out : ndarray

    See Also
    --------
    str.encode

    Notes
    -----
    The type of the result will depend on the encoding specified.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.encode(a, encoding='cp037')
    array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
       b'\x81\x82\xc2\xc1\xc2\x82\x81'], dtype='|S7')

    """
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, 'encode', _clean_args(encoding, errors)),
        np.bytes_(b''))


@set_module("numpy.strings")
def expandtabs(a, tabsize=8):
    """
    Return a copy of each string element where all tab characters are
    replaced by one or more spaces.

    Calls :meth:`str.expandtabs` element-wise.

    Return a copy of each string element where all tab characters are
    replaced by one or more spaces, depending on the current column
    and the given `tabsize`. The column number is reset to zero after
    each newline occurring in the string. This doesn't understand other
    non-printing characters or escape sequences.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    tabsize : int, optional
        Replace tabs with `tabsize` number of spaces.  If not given defaults
        to 8 spaces.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type

    See Also
    --------
    str.expandtabs

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['\t\tHello\tworld'])
    >>> np.strings.expandtabs(a, tabsize=4)  # doctest: +SKIP
    array(['        Hello   world'], dtype='<U21')  # doctest: +SKIP

    """
    a = np.asanyarray(a)
    tabsize = np.asanyarray(tabsize)

    if a.dtype.char == "T":
        return _expandtabs(a, tabsize)

    buffersizes = _expandtabs_length(a, tabsize)
    out_dtype = f"{a.dtype.char}{buffersizes.max()}"
    out = np.empty_like(a, shape=buffersizes.shape, dtype=out_dtype)
    return _expandtabs(a, tabsize, out=out)


@set_module("numpy.strings")
def center(a, width, fillchar=' '):
    """
    Return a copy of `a` with its elements centered in a string of
    length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        The length of the resulting strings, unless ``width < str_len(a)``.
    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Optional padding character to use (default is space).

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.center

    Notes
    -----
    While it is possible for ``a`` and ``fillchar`` to have different dtypes,
    passing a non-ASCII character in ``fillchar`` when ``a`` is of dtype "S"
    is not allowed, and a ``ValueError`` is raised.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b']); c
    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')
    >>> np.strings.center(c, width=9)
    array(['   a1b2  ', '   1b2a  ', '   b2a1  ', '   2a1b  '], dtype='<U9')
    >>> np.strings.center(c, width=9, fillchar='*')
    array(['***a1b2**', '***1b2a**', '***b2a1**', '***2a1b**'], dtype='<U9')
    >>> np.strings.center(c, width=1)
    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')

    """
    width = np.asanyarray(width)

    if not np.issubdtype(width.dtype, np.integer):
        raise TypeError(f"unsupported type {width.dtype} for operand 'width'")

    a = np.asanyarray(a)
    fillchar = np.asanyarray(fillchar)

    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")

    if np.result_type(a, fillchar).char == "T":
        return _center(a, width, fillchar)

    fillchar = fillchar.astype(a.dtype, copy=False)
    width = np.maximum(str_len(a), width)
    out_dtype = f"{a.dtype.char}{width.max()}"
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
    out = np.empty_like(a, shape=shape, dtype=out_dtype)

    return _center(a, width, fillchar, out=out)


@set_module("numpy.strings")
def ljust(a, width, fillchar=' '):
    """
    Return an array with the elements of `a` left-justified in a
    string of length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        The length of the resulting strings, unless ``width < str_len(a)``.
    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Optional character to use for padding (default is space).

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.ljust

    Notes
    -----
    While it is possible for ``a`` and ``fillchar`` to have different dtypes,
    passing a non-ASCII character in ``fillchar`` when ``a`` is of dtype "S"
    is not allowed, and a ``ValueError`` is raised.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.ljust(c, width=3)
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.ljust(c, width=9)
    array(['aAaAaA   ', '  aA     ', 'abBABba  '], dtype='<U9')

    """
    width = np.asanyarray(width)
    if not np.issubdtype(width.dtype, np.integer):
        raise TypeError(f"unsupported type {width.dtype} for operand 'width'")

    a = np.asanyarray(a)
    fillchar = np.asanyarray(fillchar)

    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")

    if np.result_type(a, fillchar).char == "T":
        return _ljust(a, width, fillchar)

    fillchar = fillchar.astype(a.dtype, copy=False)
    width = np.maximum(str_len(a), width)
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
    out_dtype = f"{a.dtype.char}{width.max()}"
    out = np.empty_like(a, shape=shape, dtype=out_dtype)

    return _ljust(a, width, fillchar, out=out)


@set_module("numpy.strings")
def rjust(a, width, fillchar=' '):
    """
    Return an array with the elements of `a` right-justified in a
    string of length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        The length of the resulting strings, unless ``width < str_len(a)``.
    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Optional padding character to use (default is space).

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.rjust

    Notes
    -----
    While it is possible for ``a`` and ``fillchar`` to have different dtypes,
    passing a non-ASCII character in ``fillchar`` when ``a`` is of dtype "S"
    is not allowed, and a ``ValueError`` is raised.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.rjust(a, width=3)
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.rjust(a, width=9)
    array(['   aAaAaA', '     aA  ', '  abBABba'], dtype='<U9')

    """
    width = np.asanyarray(width)
    if not np.issubdtype(width.dtype, np.integer):
        raise TypeError(f"unsupported type {width.dtype} for operand 'width'")

    a = np.asanyarray(a)
    fillchar = np.asanyarray(fillchar)

    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")

    if np.result_type(a, fillchar).char == "T":
        return _rjust(a, width, fillchar)

    fillchar = fillchar.astype(a.dtype, copy=False)
    width = np.maximum(str_len(a), width)
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
    out_dtype = f"{a.dtype.char}{width.max()}"
    out = np.empty_like(a, shape=shape, dtype=out_dtype)

    return _rjust(a, width, fillchar, out=out)


@set_module("numpy.strings")
def zfill(a, width):
    """
    Return the numeric string left-filled with zeros. A leading
    sign prefix (``+``/``-``) is handled by inserting the padding
    after the sign character rather than before.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        Width of string to left-fill elements in `a`.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type

    See Also
    --------
    str.zfill

    Examples
    --------
    >>> import numpy as np
    >>> np.strings.zfill(['1', '-1', '+1'], 3)
    array(['001', '-01', '+01'], dtype='<U3')

    """
    width = np.asanyarray(width)
    if not np.issubdtype(width.dtype, np.integer):
        raise TypeError(f"unsupported type {width.dtype} for operand 'width'")

    a = np.asanyarray(a)

    if a.dtype.char == "T":
        return _zfill(a, width)

    width = np.maximum(str_len(a), width)
    shape = np.broadcast_shapes(a.shape, width.shape)
    out_dtype = f"{a.dtype.char}{width.max()}"
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    return _zfill(a, width, out=out)


@set_module("numpy.strings")
def lstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading characters
    removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.lstrip

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    # The 'a' variable is unstripped from c[1] because of leading whitespace.
    >>> np.strings.lstrip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABba'], dtype='<U7')
    >>> np.strings.lstrip(c, 'A') # leaves c unchanged
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> (np.strings.lstrip(c, ' ') == np.strings.lstrip(c, '')).all()
    np.False_
    >>> (np.strings.lstrip(c, ' ') == np.strings.lstrip(c)).all()
    np.True_

    """
    if chars is None:
        return _lstrip_whitespace(a)
    return _lstrip_chars(a, chars)


@set_module("numpy.strings")
def rstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the trailing characters
    removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.rstrip

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', 'abBABba'])
    >>> c
    array(['aAaAaA', 'abBABba'], dtype='<U7')
    >>> np.strings.rstrip(c, 'a')
    array(['aAaAaA', 'abBABb'], dtype='<U7')
    >>> np.strings.rstrip(c, 'A')
    array(['aAaAa', 'abBABba'], dtype='<U7')

    """
    if chars is None:
        return _rstrip_whitespace(a)
    return _rstrip_chars(a, chars)


@set_module("numpy.strings")
def strip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading and
    trailing characters removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.strip

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.strip(c)
    array(['aAaAaA', 'aA', 'abBABba'], dtype='<U7')
    # 'a' unstripped from c[1] because of leading whitespace.
    >>> np.strings.strip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABb'], dtype='<U7')
    # 'A' unstripped from c[1] because of trailing whitespace.
    >>> np.strings.strip(c, 'A')
    array(['aAaAa', '  aA  ', 'abBABba'], dtype='<U7')

    """
    if chars is None:
        return _strip_whitespace(a)
    return _strip_chars(a, chars)


@set_module("numpy.strings")
def upper(a):
    """
    Return an array with the elements converted to uppercase.

    Calls :meth:`str.upper` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.upper

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['a1b c', '1bca', 'bca1']); c
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')
    >>> np.strings.upper(c)
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'upper')


@set_module("numpy.strings")
def lower(a):
    """
    Return an array with the elements converted to lowercase.

    Call :meth:`str.lower` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.lower

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['A1B C', '1BCA', 'BCA1']); c
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')
    >>> np.strings.lower(c)
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'lower')


@set_module("numpy.strings")
def swapcase(a):
    """
    Return element-wise a copy of the string with
    uppercase characters converted to lowercase and vice versa.

    Calls :meth:`str.swapcase` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.swapcase

    Examples
    --------
    >>> import numpy as np
    >>> c=np.array(['a1B c','1b Ca','b Ca1','cA1b'],'S5'); c
    array(['a1B c', '1b Ca', 'b Ca1', 'cA1b'],
        dtype='|S5')
    >>> np.strings.swapcase(c)
    array(['A1b C', '1B cA', 'B cA1', 'Ca1B'],
        dtype='|S5')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'swapcase')


@set_module("numpy.strings")
def capitalize(a):
    """
    Return a copy of ``a`` with only the first character of each element
    capitalized.

    Calls :meth:`str.capitalize` element-wise.

    For byte strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array of strings to capitalize.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.capitalize

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'],'S4'); c
    array(['a1b2', '1b2a', 'b2a1', '2a1b'],
        dtype='|S4')
    >>> np.strings.capitalize(c)
    array(['A1b2', '1b2a', 'B2a1', '2a1b'],
        dtype='|S4')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'capitalize')


@set_module("numpy.strings")
def title(a):
    """
    Return element-wise title cased version of string or unicode.

    Title case words start with uppercase characters, all remaining cased
    characters are lowercase.

    Calls :meth:`str.title` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.title

    Examples
    --------
    >>> import numpy as np
    >>> c=np.array(['a1b c','1b ca','b ca1','ca1b'],'S5'); c
    array(['a1b c', '1b ca', 'b ca1', 'ca1b'],
        dtype='|S5')
    >>> np.strings.title(c)
    array(['A1B C', '1B Ca', 'B Ca1', 'Ca1B'],
        dtype='|S5')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'title')


@set_module("numpy.strings")
def replace(a, old, new, count=-1):
    """
    For each element in ``a``, return a copy of the string with
    occurrences of substring ``old`` replaced by ``new``.

    Parameters
    ----------
    a : array_like, with ``bytes_`` or ``str_`` dtype

    old, new : array_like, with ``bytes_`` or ``str_`` dtype

    count : array_like, with ``int_`` dtype
        If the optional argument ``count`` is given, only the first
        ``count`` occurrences are replaced.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.replace

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["That is a mango", "Monkeys eat mangos"])
    >>> np.strings.replace(a, 'mango', 'banana')
    array(['That is a banana', 'Monkeys eat bananas'], dtype='<U19')

    >>> a = np.array(["The dish is fresh", "This is it"])
    >>> np.strings.replace(a, 'is', 'was')
    array(['The dwash was fresh', 'Thwas was it'], dtype='<U19')

    """
    count = np.asanyarray(count)
    if not np.issubdtype(count.dtype, np.integer):
        raise TypeError(f"unsupported type {count.dtype} for operand 'count'")

    arr = np.asanyarray(a)
    old_dtype = getattr(old, 'dtype', None)
    old = np.asanyarray(old)
    new_dtype = getattr(new, 'dtype', None)
    new = np.asanyarray(new)

    if np.result_type(arr, old, new).char == "T":
        return _replace(arr, old, new, count)

    a_dt = arr.dtype
    old = old.astype(old_dtype if old_dtype else a_dt, copy=False)
    new = new.astype(new_dtype if new_dtype else a_dt, copy=False)
    max_int64 = np.iinfo(np.int64).max
    counts = _count_ufunc(arr, old, 0, max_int64)
    counts = np.where(count < 0, counts, np.minimum(counts, count))
    buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))
    out_dtype = f"{arr.dtype.char}{buffersizes.max()}"
    out = np.empty_like(arr, shape=buffersizes.shape, dtype=out_dtype)

    return _replace(arr, old, new, counts, out=out)


def _join(sep, seq):
    """
    Return a string which is the concatenation of the strings in the
    sequence `seq`.

    Calls :meth:`str.join` element-wise.

    Parameters
    ----------
    sep : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    seq : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.join

    Examples
    --------
    >>> import numpy as np
    >>> np.strings.join('-', 'osd')  # doctest: +SKIP
    array('o-s-d', dtype='<U5')  # doctest: +SKIP

    >>> np.strings.join(['-', '.'], ['ghc', 'osd'])  # doctest: +SKIP
    array(['g-h-c', 'o.s.d'], dtype='<U5')  # doctest: +SKIP

    """
    return _to_bytes_or_str_array(
        _vec_string(sep, np.object_, 'join', (seq,)), seq)


def _split(a, sep=None, maxsplit=None):
    """
    For each element in `a`, return a list of the words in the
    string, using `sep` as the delimiter string.

    Calls :meth:`str.split` element-wise.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sep : str or unicode, optional
       If `sep` is not specified or None, any whitespace string is a
       separator.

    maxsplit : int, optional
        If `maxsplit` is given, at most `maxsplit` splits are done.

    Returns
    -------
    out : ndarray
        Array of list objects

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array("Numpy is nice!")
    >>> np.strings.split(x, " ")  # doctest: +SKIP
    array(list(['Numpy', 'is', 'nice!']), dtype=object)  # doctest: +SKIP

    >>> np.strings.split(x, " ", 1)  # doctest: +SKIP
    array(list(['Numpy', 'is nice!']), dtype=object)  # doctest: +SKIP

    See Also
    --------
    str.split, rsplit

    """
    # This will return an array of lists of different sizes, so we
    # leave it as an object array
    return _vec_string(
        a, np.object_, 'split', [sep] + _clean_args(maxsplit))


def _rsplit(a, sep=None, maxsplit=None):
    """
    For each element in `a`, return a list of the words in the
    string, using `sep` as the delimiter string.

    Calls :meth:`str.rsplit` element-wise.

    Except for splitting from the right, `rsplit`
    behaves like `split`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sep : str or unicode, optional
        If `sep` is not specified or None, any whitespace string
        is a separator.
    maxsplit : int, optional
        If `maxsplit` is given, at most `maxsplit` splits are done,
        the rightmost ones.

    Returns
    -------
    out : ndarray
        Array of list objects

    See Also
    --------
    str.rsplit, split

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', 'abBABba'])
    >>> np.strings.rsplit(a, 'A')  # doctest: +SKIP
    array([list(['a', 'a', 'a', '']),  # doctest: +SKIP
           list(['abB', 'Bba'])], dtype=object)  # doctest: +SKIP

    """
    # This will return an array of lists of different sizes, so we
    # leave it as an object array
    return _vec_string(
        a, np.object_, 'rsplit', [sep] + _clean_args(maxsplit))


def _splitlines(a, keepends=None):
    """
    For each element in `a`, return a list of the lines in the
    element, breaking at line boundaries.

    Calls :meth:`str.splitlines` element-wise.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    keepends : bool, optional
        Line breaks are not included in the resulting list unless
        keepends is given and true.

    Returns
    -------
    out : ndarray
        Array of list objects

    See Also
    --------
    str.splitlines

    Examples
    --------
    >>> np.char.splitlines("first line\\nsecond line")
    array(list(['first line', 'second line']), dtype=object)
    >>> a = np.array(["first\\nsecond", "third\\nfourth"])
    >>> np.char.splitlines(a)
    array([list(['first', 'second']), list(['third', 'fourth'])], dtype=object)

    """
    return _vec_string(
        a, np.object_, 'splitlines', _clean_args(keepends))


@set_module("numpy.strings")
def partition(a, sep):
    """
    Partition each element in ``a`` around ``sep``.

    For each element in ``a``, split the element at the first
    occurrence of ``sep``, and return a 3-tuple containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, the first item of
    the tuple will contain the whole string, and the second and third
    ones will be the empty string.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Separator to split each string element in ``a``.

    Returns
    -------
    out : 3-tuple:
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part before the separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part after the separator

    See Also
    --------
    str.partition

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array(["Numpy is nice!"])
    >>> np.strings.partition(x, " ")
    (array(['Numpy'], dtype='<U5'),
     array([' '], dtype='<U1'),
     array(['is nice!'], dtype='<U8'))

    """
    a = np.asanyarray(a)
    sep = np.asanyarray(sep)

    if np.result_type(a, sep).char == "T":
        return _partition(a, sep)

    sep = sep.astype(a.dtype, copy=False)
    pos = _find_ufunc(a, sep, 0, MAX)
    a_len = str_len(a)
    sep_len = str_len(sep)

    not_found = pos < 0
    buffersizes1 = np.where(not_found, a_len, pos)
    buffersizes3 = np.where(not_found, 0, a_len - pos - sep_len)

    out_dtype = ",".join([f"{a.dtype.char}{n}" for n in (
        buffersizes1.max(),
        1 if np.all(not_found) else sep_len.max(),
        buffersizes3.max(),
    )])
    shape = np.broadcast_shapes(a.shape, sep.shape)
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    return _partition_index(a, sep, pos, out=(out["f0"], out["f1"], out["f2"]))


@set_module("numpy.strings")
def rpartition(a, sep):
    """
    Partition (split) each element around the right-most separator.

    For each element in ``a``, split the element at the last
    occurrence of ``sep``, and return a 3-tuple containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, the third item of
    the tuple will contain the whole string, and the first and second
    ones will be the empty string.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Separator to split each string element in ``a``.

    Returns
    -------
    out : 3-tuple:
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part before the separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part after the separator

    See Also
    --------
    str.rpartition

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.rpartition(a, 'A')
    (array(['aAaAa', '  a', 'abB'], dtype='<U5'),
     array(['A', 'A', 'A'], dtype='<U1'),
     array(['', '  ', 'Bba'], dtype='<U3'))

    """
    a = np.asanyarray(a)
    sep = np.asanyarray(sep)

    if np.result_type(a, sep).char == "T":
        return _rpartition(a, sep)

    sep = sep.astype(a.dtype, copy=False)
    pos = _rfind_ufunc(a, sep, 0, MAX)
    a_len = str_len(a)
    sep_len = str_len(sep)

    not_found = pos < 0
    buffersizes1 = np.where(not_found, 0, pos)
    buffersizes3 = np.where(not_found, a_len, a_len - pos - sep_len)

    out_dtype = ",".join([f"{a.dtype.char}{n}" for n in (
        buffersizes1.max(),
        1 if np.all(not_found) else sep_len.max(),
        buffersizes3.max(),
    )])
    shape = np.broadcast_shapes(a.shape, sep.shape)
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    return _rpartition_index(
        a, sep, pos, out=(out["f0"], out["f1"], out["f2"]))


@set_module("numpy.strings")
def translate(a, table, deletechars=None):
    """
    For each element in `a`, return a copy of the string where all
    characters occurring in the optional argument `deletechars` are
    removed, and the remaining characters have been mapped through the
    given translation table.

    Calls :meth:`str.translate` element-wise.

    Parameters
    ----------
    a : array-like, with `np.bytes_` or `np.str_` dtype

    table : str of length 256

    deletechars : str

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.translate

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['a1b c', '1bca', 'bca1'])
    >>> table = a[0].maketrans('abc', '123')
    >>> deletechars = ' '
    >>> np.char.translate(a, table, deletechars)
    array(['112 3', '1231', '2311'], dtype='<U5')

    """
    a_arr = np.asarray(a)
    if issubclass(a_arr.dtype.type, np.str_):
        return _vec_string(
            a_arr, a_arr.dtype, 'translate', (table,))
    else:
        return _vec_string(
            a_arr,
            a_arr.dtype,
            'translate',
            [table] + _clean_args(deletechars)
        )
<|endoftext|>from contextlib import nullcontext
import operator
import numpy as np
from .._utils import set_module
from .numeric import uint8, ndarray, dtype

__all__ = ['memmap']

dtypedescr = dtype
valid_filemodes = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+", "w+"]

mode_equivalents = {
    "readonly":"r",
    "copyonwrite":"c",
    "readwrite":"r+",
    "write":"w+"
    }


@set_module('numpy')
class memmap(ndarray):
    """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  NumPy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.

    This subclass of ndarray has some unpleasant interactions with
    some operations, because it doesn't quite fit properly as a subclass.
    An alternative to using this subclass is to create the ``mmap``
    object yourself, then create an ndarray with ndarray.__new__ directly,
    passing the object created in its 'buffer=' parameter.

    This class may at some point be turned into a factory function
    which returns a view into an mmap buffer.

    Flush the memmap instance to write the changes to the file. Currently there
    is no API to close the underlying ``mmap``. It is tricky to ensure the
    resource is actually closed, since it may be shared between different
    memmap instances.


    Parameters
    ----------
    filename : str, file-like object, or pathlib.Path instance
        The file name or file object to be used as the array data buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is `uint8`.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:

        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        |      | If ``mode == 'w+'`` then `shape` must also be specified.    |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+

        Default is 'r+'.
    offset : int, optional
        In the file, array data starts at this offset. Since `offset` is
        measured in bytes, it should normally be a multiple of the byte-size
        of `dtype`. When ``mode != 'r'``, even positive offsets beyond end of
        file are valid; The file will be extended to accommodate the
        additional data. By default, ``memmap`` will start at the beginning of
        the file, even if ``filename`` is a file pointer ``fp`` and
        ``fp.tell() != 0``.
    shape : int or sequence of ints, optional
        The desired shape of the array. If ``mode == 'r'`` and the number
        of remaining bytes after `offset` is not a multiple of the byte-size
        of `dtype`, you must specify `shape`. By default, the returned array
        will be 1-D with the number of elements determined by file size
        and data-type.

        .. versionchanged:: 2.0
         The shape parameter can now be any integer sequence type, previously
         types were limited to tuple and int.

    order : {'C', 'F'}, optional
        Specify the order of the ndarray memory layout:
        :term:`row-major`, C-style or :term:`column-major`,
        Fortran-style.  This only has an effect if the shape is
        greater than 1-D.  The default order is 'C'.

    Attributes
    ----------
    filename : str or pathlib.Path instance
        Path to the mapped file.
    offset : int
        Offset position in the file.
    mode : str
        File mode.

    Methods
    -------
    flush
        Flush any changes in memory to file on disk.
        When you delete a memmap object, flush is called first to write
        changes to disk.


    See also
    --------
    lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

    Notes
    -----
    The memmap object can be used anywhere an ndarray is accepted.
    Given a memmap ``fp``, ``isinstance(fp, numpy.ndarray)`` returns
    ``True``.

    Memory-mapped files cannot be larger than 2GB on 32-bit systems.

    When a memmap causes a file to be created or extended beyond its
    current size in the filesystem, the contents of the new part are
    unspecified. On systems with POSIX filesystem semantics, the extended
    part will be filled with zero bytes.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(12, dtype='float32')
    >>> data.resize((3,4))

    This example uses a temporary file so that doctest doesn't write
    files to your directory. You would use a 'normal' filename.

    >>> from tempfile import mkdtemp
    >>> import os.path as path
    >>> filename = path.join(mkdtemp(), 'newfile.dat')

    Create a memmap with dtype and shape that matches our data:

    >>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
    >>> fp
    memmap([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]], dtype=float32)

    Write data to memmap array:

    >>> fp[:] = data[:]
    >>> fp
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)

    >>> fp.filename == path.abspath(filename)
    True

    Flushes memory changes to disk in order to read them back

    >>> fp.flush()

    Load the memmap and verify data was stored:

    >>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> newfp
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)

    Read-only memmap:

    >>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> fpr.flags.writeable
    False

    Copy-on-write memmap:

    >>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
    >>> fpc.flags.writeable
    True

    It's possible to assign to copy-on-write array, but values are only
    written into the memory copy of the array, and not written to disk:

    >>> fpc
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    >>> fpc[0,:] = 0
    >>> fpc
    memmap([[  0.,   0.,   0.,   0.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)

    File on disk is unchanged:

    >>> fpr
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)

    Offset into a memmap:

    >>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
    >>> fpo
    memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)

    """

    __array_priority__ = -100.0

    def __new__(subtype, filename, dtype=uint8, mode='r+', offset=0,
                shape=None, order='C'):
        # Import here to minimize 'import numpy' overhead
        import mmap
        import os.path
        try:
            mode = mode_equivalents[mode]
        except KeyError as e:
            if mode not in valid_filemodes:
                raise ValueError(
                    "mode must be one of {!r} (got {!r})"
                    .format(valid_filemodes + list(mode_equivalents.keys()), mode)
                ) from None

        if mode == 'w+' and shape is None:
            raise ValueError("shape must be given if mode == 'w+'")

        if hasattr(filename, 'read'):
            f_ctx = nullcontext(filename)
        else:
            f_ctx = open(
                os.fspath(filename),
                ('r' if mode == 'c' else mode)+'b'
            )

        with f_ctx as fid:
            fid.seek(0, 2)
            flen = fid.tell()
            descr = dtypedescr(dtype)
            _dbytes = descr.itemsize

            if shape is None:
                bytes = flen - offset
                if bytes % _dbytes:
                    raise ValueError("Size of available data is not a "
                            "multiple of the data-type size.")
                size = bytes // _dbytes
                shape = (size,)
            else:
                if type(shape) not in (tuple, list):
                    try:
                        shape = [operator.index(shape)]
                    except TypeError:
                        pass
                shape = tuple(shape)
                size = np.intp(1)  # avoid default choice of np.int_, which might overflow
                for k in shape:
                    size *= k

            bytes = int(offset + size*_dbytes)

            if mode in ('w+', 'r+'):
                # gh-27723
                # if bytes == 0, we write out 1 byte to allow empty memmap.
                bytes = max(bytes, 1)
                if flen < bytes:
                    fid.seek(bytes - 1, 0)
                    fid.write(b'\0')
                    fid.flush()

            if mode == 'c':
                acc = mmap.ACCESS_COPY
            elif mode == 'r':
                acc = mmap.ACCESS_READ
            else:
                acc = mmap.ACCESS_WRITE

            start = offset - offset % mmap.ALLOCATIONGRANULARITY
            bytes -= start
            # bytes == 0 is problematic as in mmap length=0 maps the full file.
            # See PR gh-27723 for a more detailed explanation.
            if bytes == 0 and start > 0:
                bytes += mmap.ALLOCATIONGRANULARITY
                start -= mmap.ALLOCATIONGRANULARITY
            array_offset = offset - start
            mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)

            self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
                                   offset=array_offset, order=order)
            self._mmap = mm
            self.offset = offset
            self.mode = mode

            if isinstance(filename, os.PathLike):
                # special case - if we were constructed with a pathlib.path,
                # then filename is a path object, not a string
                self.filename = filename.resolve()
            elif hasattr(fid, "name") and isinstance(fid.name, str):
                # py3 returns int for TemporaryFile().name
                self.filename = os.path.abspath(fid.name)
            # same as memmap copies (e.g. memmap + 1)
            else:
                self.filename = None

        return self

    def __array_finalize__(self, obj):
        if hasattr(obj, '_mmap') and np.may_share_memory(self, obj):
            self._mmap = obj._mmap
            self.filename = obj.filename
            self.offset = obj.offset
            self.mode = obj.mode
        else:
            self._mmap = None
            self.filename = None
            self.offset = None
            self.mode = None

    def flush(self):
        """
        Write any changes in the array to the file on disk.

        For further information, see `memmap`.

        Parameters
        ----------
        None

        See Also
        --------
        memmap

        """
        if self.base is not None and hasattr(self.base, 'flush'):
            self.base.flush()

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        arr = super().__array_wrap__(arr, context)

        # Return a memmap if a memmap was given as the output of the
        # ufunc. Leave the arr class unchanged if self is not a memmap
        # to keep original memmap subclasses behavior
        if self is arr or type(self) is not memmap:
            return arr

        # Return scalar instead of 0d memmap, e.g. for np.sum with
        # axis=None (note that subclasses will not reach here)
        if return_scalar:
            return arr[()]

        # Return ndarray otherwise
        return arr.view(np.ndarray)

    def __getitem__(self, index):
        res = super().__getitem__(index)
        if type(res) is memmap and res._mmap is None:
            return res.view(type=ndarray)
        return res
<|endoftext|>"""
This module contains a set of functions for vectorized string
operations and methods.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `object_`, `bytes_` or `str_`, and use the free functions
   in the `numpy.char` module for fast vectorized string operations.

Some methods will only be available if the corresponding string method is
available in your version of Python.

The preferred alias for `defchararray` is `numpy.char`.

"""
import functools

import numpy as np
from .._utils import set_module
from .numerictypes import bytes_, str_, character
from .numeric import ndarray, array as narray, asarray as asnarray
from numpy._core.multiarray import compare_chararrays
from numpy._core import overrides
from numpy.strings import *
from numpy.strings import (
    multiply as strings_multiply,
    partition as strings_partition,
    rpartition as strings_rpartition,
)
from numpy._core.strings import (
    _split as split,
    _rsplit as rsplit,
    _splitlines as splitlines,
    _join as join,
)

__all__ = [
    'equal', 'not_equal', 'greater_equal', 'less_equal',
    'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize',
    'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs',
    'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
    'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
    'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
    'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
    'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal',
    'array', 'asarray', 'compare_chararrays', 'chararray'
    ]


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.char')


def _binary_op_dispatcher(x1, x2):
    return (x1, x2)


@array_function_dispatch(_binary_op_dispatcher)
def equal(x1, x2):
    """
    Return (x1 == x2) element-wise.

    Unlike `numpy.equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    Examples
    --------
    >>> import numpy as np
    >>> y = "aa "
    >>> x = "aa"
    >>> np.char.equal(x, y)
    array(True)

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less
    """
    return compare_chararrays(x1, x2, '==', True)


@array_function_dispatch(_binary_op_dispatcher)
def not_equal(x1, x2):
    """
    Return (x1 != x2) element-wise.

    Unlike `numpy.not_equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, greater_equal, less_equal, greater, less

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.not_equal(x1, 'b')
    array([ True, False,  True])

    """
    return compare_chararrays(x1, x2, '!=', True)


@array_function_dispatch(_binary_op_dispatcher)
def greater_equal(x1, x2):
    """
    Return (x1 >= x2) element-wise.

    Unlike `numpy.greater_equal`, this comparison is performed by
    first stripping whitespace characters from the end of the string.
    This behavior is provided for backward-compatibility with
    numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, less_equal, greater, less

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.greater_equal(x1, 'b')
    array([False,  True,  True])

    """
    return compare_chararrays(x1, x2, '>=', True)


@array_function_dispatch(_binary_op_dispatcher)
def less_equal(x1, x2):
    """
    Return (x1 <= x2) element-wise.

    Unlike `numpy.less_equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, greater, less

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.less_equal(x1, 'b')
    array([ True,  True, False])

    """
    return compare_chararrays(x1, x2, '<=', True)


@array_function_dispatch(_binary_op_dispatcher)
def greater(x1, x2):
    """
    Return (x1 > x2) element-wise.

    Unlike `numpy.greater`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, less_equal, less

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.greater(x1, 'b')
    array([False, False,  True])

    """
    return compare_chararrays(x1, x2, '>', True)


@array_function_dispatch(_binary_op_dispatcher)
def less(x1, x2):
    """
    Return (x1 < x2) element-wise.

    Unlike `numpy.greater`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, less_equal, greater

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.less(x1, 'b')
    array([True, False, False])

    """
    return compare_chararrays(x1, x2, '<', True)


@set_module("numpy.char")
def multiply(a, i):
    """
    Return (a * i), that is string multiple concatenation,
    element-wise.

    Values in ``i`` of less than 0 are treated as 0 (which yields an
    empty string).

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    i : array_like, with any integer dtype

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input types

    Notes
    -----
    This is a thin wrapper around np.strings.multiply that raises
    `ValueError` when ``i`` is not an integer. It only
    exists for backwards-compatibility.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["a", "b", "c"])
    >>> np.strings.multiply(a, 3)
    array(['aaa', 'bbb', 'ccc'], dtype='<U3')
    >>> i = np.array([1, 2, 3])
    >>> np.strings.multiply(a, i)
    array(['a', 'bb', 'ccc'], dtype='<U3')
    >>> np.strings.multiply(np.array(['a']), i)
    array(['a', 'aa', 'aaa'], dtype='<U3')
    >>> a = np.array(['a', 'b', 'c', 'd', 'e', 'f']).reshape((2, 3))
    >>> np.strings.multiply(a, 3)
    array([['aaa', 'bbb', 'ccc'],
           ['ddd', 'eee', 'fff']], dtype='<U3')
    >>> np.strings.multiply(a, i)
    array([['a', 'bb', 'ccc'],
           ['d', 'ee', 'fff']], dtype='<U3')

    """
    try:
        return strings_multiply(a, i)
    except TypeError:
        raise ValueError("Can only multiply by integers")


@set_module("numpy.char")
def partition(a, sep):
    """
    Partition each element in `a` around `sep`.

    Calls :meth:`str.partition` element-wise.

    For each element in `a`, split the element as the first
    occurrence of `sep`, and return 3 strings containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, return 3 strings
    containing the string itself, followed by two empty strings.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : {str, unicode}
        Separator to split each string element in `a`.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types. The output array will have an extra
        dimension with 3 elements per input element.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array(["Numpy is nice!"])
    >>> np.char.partition(x, " ")
    array([['Numpy', ' ', 'is nice!']], dtype='<U8')

    See Also
    --------
    str.partition

    """
    return np.stack(strings_partition(a, sep), axis=-1)


@set_module("numpy.char")
def rpartition(a, sep):
    """
    Partition (split) each element around the right-most separator.

    Calls :meth:`str.rpartition` element-wise.

    For each element in `a`, split the element as the last
    occurrence of `sep`, and return 3 strings containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, return 3 strings
    containing the string itself, followed by two empty strings.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : str or unicode
        Right-most separator to split each element in array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types. The output array will have an extra
        dimension with 3 elements per input element.

    See Also
    --------
    str.rpartition

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.char.rpartition(a, 'A')
    array([['aAaAa', 'A', ''],
       ['  a', 'A', '  '],
       ['abB', 'A', 'Bba']], dtype='<U5')

    """
    return np.stack(strings_rpartition(a, sep), axis=-1)


@set_module("numpy.char")
class chararray(ndarray):
    """
    chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
              strides=None, order=None)

    Provides a convenient view on arrays of string and unicode values.

    .. note::
       The `chararray` class exists for backwards compatibility with
       Numarray, it is not recommended for new development. Starting from numpy
       1.4, if one needs arrays of strings, it is recommended to use arrays of
       `dtype` `~numpy.object_`, `~numpy.bytes_` or `~numpy.str_`, and use
       the free functions in the `numpy.char` module for fast vectorized
       string operations.

    Versus a NumPy array of dtype `~numpy.bytes_` or `~numpy.str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)

    chararrays should be created using `numpy.char.array` or
    `numpy.char.asarray`, rather than this constructor directly.

    This constructor creates the array, using `buffer` (with `offset`
    and `strides`) if it is not ``None``. If `buffer` is ``None``, then
    constructs a new array with `strides` in "C order", unless both
    ``len(shape) >= 2`` and ``order='F'``, in which case `strides`
    is in "Fortran order".

    Methods
    -------
    astype
    argsort
    copy
    count
    decode
    dump
    dumps
    encode
    endswith
    expandtabs
    fill
    find
    flatten
    getfield
    index
    isalnum
    isalpha
    isdecimal
    isdigit
    islower
    isnumeric
    isspace
    istitle
    isupper
    item
    join
    ljust
    lower
    lstrip
    nonzero
    put
    ravel
    repeat
    replace
    reshape
    resize
    rfind
    rindex
    rjust
    rsplit
    rstrip
    searchsorted
    setfield
    setflags
    sort
    split
    splitlines
    squeeze
    startswith
    strip
    swapaxes
    swapcase
    take
    title
    tofile
    tolist
    tostring
    translate
    transpose
    upper
    view
    zfill

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    itemsize : int, optional
        Length of each array element, in number of characters. Default is 1.
    unicode : bool, optional
        Are the array elements of type unicode (True) or string (False).
        Default is False.
    buffer : object exposing the buffer interface or str, optional
        Memory address of the start of the array data.  Default is None,
        in which case a new array is created.
    offset : int, optional
        Fixed stride displacement from the beginning of an axis?
        Default is 0. Needs to be >=0.
    strides : array_like of ints, optional
        Strides for the array (see `~numpy.ndarray.strides` for
        full description). Default is None.
    order : {'C', 'F'}, optional
        The order in which the array data is stored in memory: 'C' ->
        "row major" order (the default), 'F' -> "column major"
        (Fortran) order.

    Examples
    --------
    >>> import numpy as np
    >>> charar = np.char.chararray((3, 3))
    >>> charar[:] = 'a'
    >>> charar
    chararray([[b'a', b'a', b'a'],
               [b'a', b'a', b'a'],
               [b'a', b'a', b'a']], dtype='|S1')

    >>> charar = np.char.chararray(charar.shape, itemsize=5)
    >>> charar[:] = 'abc'
    >>> charar
    chararray([[b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc']], dtype='|S5')

    """
    def __new__(subtype, shape, itemsize=1, unicode=False, buffer=None,
                offset=0, strides=None, order='C'):
        if unicode:
            dtype = str_
        else:
            dtype = bytes_

        # force itemsize to be a Python int, since using NumPy integer
        # types results in itemsize.itemsize being used as the size of
        # strings in the new array.
        itemsize = int(itemsize)

        if isinstance(buffer, str):
            # unicode objects do not have the buffer interface
            filler = buffer
            buffer = None
        else:
            filler = None

        if buffer is None:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   order=order)
        else:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   buffer=buffer,
                                   offset=offset, strides=strides,
                                   order=order)
        if filler is not None:
            self[...] = filler

        return self

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        # When calling a ufunc (and some other functions), we return a
        # chararray if the ufunc output is a string-like array,
        # or an ndarray otherwise
        if arr.dtype.char in "SUbc":
            return arr.view(type(self))
        return arr

    def __array_finalize__(self, obj):
        # The b is a special case because it is used for reconstructing.
        if self.dtype.char not in 'VSUbc':
            raise ValueError("Can only create a chararray from string data.")

    def __getitem__(self, obj):
        val = ndarray.__getitem__(self, obj)
        if isinstance(val, character):
            return val.rstrip()
        return val

    # IMPLEMENTATION NOTE: Most of the methods of this class are
    # direct delegations to the free functions in this module.
    # However, those that return an array of strings should instead
    # return a chararray, so some extra wrapping is required.

    def __eq__(self, other):
        """
        Return (self == other) element-wise.

        See Also
        --------
        equal
        """
        return equal(self, other)

    def __ne__(self, other):
        """
        Return (self != other) element-wise.

        See Also
        --------
        not_equal
        """
        return not_equal(self, other)

    def __ge__(self, other):
        """
        Return (self >= other) element-wise.

        See Also
        --------
        greater_equal
        """
        return greater_equal(self, other)

    def __le__(self, other):
        """
        Return (self <= other) element-wise.

        See Also
        --------
        less_equal
        """
        return less_equal(self, other)

    def __gt__(self, other):
        """
        Return (self > other) element-wise.

        See Also
        --------
        greater
        """
        return greater(self, other)

    def __lt__(self, other):
        """
        Return (self < other) element-wise.

        See Also
        --------
        less
        """
        return less(self, other)

    def __add__(self, other):
        """
        Return (self + other), that is string concatenation,
        element-wise for a pair of array_likes of str or unicode.

        See Also
        --------
        add
        """
        return add(self, other)

    def __radd__(self, other):
        """
        Return (other + self), that is string concatenation,
        element-wise for a pair of array_likes of `bytes_` or `str_`.

        See Also
        --------
        add
        """
        return add(other, self)

    def __mul__(self, i):
        """
        Return (self * i), that is string multiple concatenation,
        element-wise.

        See Also
        --------
        multiply
        """
        return asarray(multiply(self, i))

    def __rmul__(self, i):
        """
        Return (self * i), that is string multiple concatenation,
        element-wise.

        See Also
        --------
        multiply
        """
        return asarray(multiply(self, i))

    def __mod__(self, i):
        """
        Return (self % i), that is pre-Python 2.6 string formatting
        (interpolation), element-wise for a pair of array_likes of `bytes_`
        or `str_`.

        See Also
        --------
        mod
        """
        return asarray(mod(self, i))

    def __rmod__(self, other):
        return NotImplemented

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Return the indices that sort the array lexicographically.

        For full documentation see `numpy.argsort`, for which this method is
        in fact merely a "thin wrapper."

        Examples
        --------
        >>> c = np.array(['a1b c', '1b ca', 'b ca1', 'Ca1b'], 'S5')
        >>> c = c.view(np.char.chararray); c
        chararray(['a1b c', '1b ca', 'b ca1', 'Ca1b'],
              dtype='|S5')
        >>> c[c.argsort()]
        chararray(['1b ca', 'Ca1b', 'a1b c', 'b ca1'],
              dtype='|S5')

        """
        return self.__array__().argsort(axis, kind, order)
    argsort.__doc__ = ndarray.argsort.__doc__

    def capitalize(self):
        """
        Return a copy of `self` with only the first character of each element
        capitalized.

        See Also
        --------
        char.capitalize

        """
        return asarray(capitalize(self))

    def center(self, width, fillchar=' '):
        """
        Return a copy of `self` with its elements centered in a
        string of length `width`.

        See Also
        --------
        center
        """
        return asarray(center(self, width, fillchar))

    def count(self, sub, start=0, end=None):
        """
        Returns an array with the number of non-overlapping occurrences of
        substring `sub` in the range [`start`, `end`].

        See Also
        --------
        char.count

        """
        return count(self, sub, start, end)

    def decode(self, encoding=None, errors=None):
        """
        Calls ``bytes.decode`` element-wise.

        See Also
        --------
        char.decode

        """
        return decode(self, encoding, errors)

    def encode(self, encoding=None, errors=None):
        """
        Calls :meth:`str.encode` element-wise.

        See Also
        --------
        char.encode

        """
        return encode(self, encoding, errors)

    def endswith(self, suffix, start=0, end=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` ends with `suffix`, otherwise `False`.

        See Also
        --------
        char.endswith

        """
        return endswith(self, suffix, start, end)

    def expandtabs(self, tabsize=8):
        """
        Return a copy of each string element where all tab characters are
        replaced by one or more spaces.

        See Also
        --------
        char.expandtabs

        """
        return asarray(expandtabs(self, tabsize))

    def find(self, sub, start=0, end=None):
        """
        For each element, return the lowest index in the string where
        substring `sub` is found.

        See Also
        --------
        char.find

        """
        return find(self, sub, start, end)

    def index(self, sub, start=0, end=None):
        """
        Like `find`, but raises :exc:`ValueError` when the substring is not
        found.

        See Also
        --------
        char.index

        """
        return index(self, sub, start, end)

    def isalnum(self):
        """
        Returns true for each element if all characters in the string
        are alphanumeric and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isalnum

        """
        return isalnum(self)

    def isalpha(self):
        """
        Returns true for each element if all characters in the string
        are alphabetic and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isalpha

        """
        return isalpha(self)

    def isdigit(self):
        """
        Returns true for each element if all characters in the string are
        digits and there is at least one character, false otherwise.

        See Also
        --------
        char.isdigit

        """
        return isdigit(self)

    def islower(self):
        """
        Returns true for each element if all cased characters in the
        string are lowercase and there is at least one cased character,
        false otherwise.

        See Also
        --------
        char.islower

        """
        return islower(self)

    def isspace(self):
        """
        Returns true for each element if there are only whitespace
        characters in the string and there is at least one character,
        false otherwise.

        See Also
        --------
        char.isspace

        """
        return isspace(self)

    def istitle(self):
        """
        Returns true for each element if the element is a titlecased
        string and there is at least one character, false otherwise.

        See Also
        --------
        char.istitle

        """
        return istitle(self)

    def isupper(self):
        """
        Returns true for each element if all cased characters in the
        string are uppercase and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isupper

        """
        return isupper(self)

    def join(self, seq):
        """
        Return a string which is the concatenation of the strings in the
        sequence `seq`.

        See Also
        --------
        char.join

        """
        return join(self, seq)

    def ljust(self, width, fillchar=' '):
        """
        Return an array with the elements of `self` left-justified in a
        string of length `width`.

        See Also
        --------
        char.ljust

        """
        return asarray(ljust(self, width, fillchar))

    def lower(self):
        """
        Return an array with the elements of `self` converted to
        lowercase.

        See Also
        --------
        char.lower

        """
        return asarray(lower(self))

    def lstrip(self, chars=None):
        """
        For each element in `self`, return a copy with the leading characters
        removed.

        See Also
        --------
        char.lstrip

        """
        return lstrip(self, chars)

    def partition(self, sep):
        """
        Partition each element in `self` around `sep`.

        See Also
        --------
        partition
        """
        return asarray(partition(self, sep))

    def replace(self, old, new, count=None):
        """
        For each element in `self`, return a copy of the string with all
        occurrences of substring `old` replaced by `new`.

        See Also
        --------
        char.replace

        """
        return replace(self, old, new, count if count is not None else -1)

    def rfind(self, sub, start=0, end=None):
        """
        For each element in `self`, return the highest index in the string
        where substring `sub` is found, such that `sub` is contained
        within [`start`, `end`].

        See Also
        --------
        char.rfind

        """
        return rfind(self, sub, start, end)

    def rindex(self, sub, start=0, end=None):
        """
        Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is
        not found.

        See Also
        --------
        char.rindex

        """
        return rindex(self, sub, start, end)

    def rjust(self, width, fillchar=' '):
        """
        Return an array with the elements of `self`
        right-justified in a string of length `width`.

        See Also
        --------
        char.rjust

        """
        return asarray(rjust(self, width, fillchar))

    def rpartition(self, sep):
        """
        Partition each element in `self` around `sep`.

        See Also
        --------
        rpartition
        """
        return asarray(rpartition(self, sep))

    def rsplit(self, sep=None, maxsplit=None):
        """
        For each element in `self`, return a list of the words in
        the string, using `sep` as the delimiter string.

        See Also
        --------
        char.rsplit

        """
        return rsplit(self, sep, maxsplit)

    def rstrip(self, chars=None):
        """
        For each element in `self`, return a copy with the trailing
        characters removed.

        See Also
        --------
        char.rstrip

        """
        return rstrip(self, chars)

    def split(self, sep=None, maxsplit=None):
        """
        For each element in `self`, return a list of the words in the
        string, using `sep` as the delimiter string.

        See Also
        --------
        char.split

        """
        return split(self, sep, maxsplit)

    def splitlines(self, keepends=None):
        """
        For each element in `self`, return a list of the lines in the
        element, breaking at line boundaries.

        See Also
        --------
        char.splitlines

        """
        return splitlines(self, keepends)

    def startswith(self, prefix, start=0, end=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` starts with `prefix`, otherwise `False`.

        See Also
        --------
        char.startswith

        """
        return startswith(self, prefix, start, end)

    def strip(self, chars=None):
        """
        For each element in `self`, return a copy with the leading and
        trailing characters removed.

        See Also
        --------
        char.strip

        """
        return strip(self, chars)

    def swapcase(self):
        """
        For each element in `self`, return a copy of the string with
        uppercase characters converted to lowercase and vice versa.

        See Also
        --------
        char.swapcase

        """
        return asarray(swapcase(self))

    def title(self):
        """
