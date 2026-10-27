    The default option bezout(p, q, x, method='bz') returns Bezout's
    symmetric matrix of p and q, of dimensions (mx) x (mx). The
    determinant of this matrix is equal to the determinant of sylvester2,
    Sylvester's matrix of 1853, whose dimensions are (2*mx) x (2*mx);
    however the subresultants of these two matrices may differ.

    The other option, bezout(p, q, x, 'prs'), is of interest to us
    in this module because it returns a matrix equivalent to sylvester2.
    In this case all subresultants of the two matrices are identical.

    Both the subresultant polynomial remainder sequence (prs) and
    the modified subresultant prs of p and q can be computed by
    evaluating determinants of appropriately selected submatrices of
    bezout(p, q, x, 'prs') --- one determinant per coefficient of the
    remainder polynomials.

    The matrices bezout(p, q, x, 'bz') and bezout(p, q, x, 'prs')
    are related by the formula

    bezout(p, q, x, 'prs') =
    backward_eye(deg(p)) * bezout(p, q, x, 'bz') * backward_eye(deg(p)),

    where backward_eye() is the backward identity function.

    References
    ==========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """
    # obtain degrees of polys
    m, n = degree( Poly(p, x), x), degree( Poly(q, x), x)

    # Special cases:
    # A:: case m = n < 0 (i.e. both polys are 0)
    if m == n and n < 0:
        return Matrix([])

    # B:: case m = n = 0  (i.e. both polys are constants)
    if m == n and n == 0:
        return Matrix([])

    # C:: m == 0 and n < 0 or m < 0 and n == 0
    # (i.e. one poly is constant and the other is 0)
    if m == 0 and n < 0:
        return Matrix([])
    elif m < 0 and n == 0:
        return Matrix([])

    # D:: m >= 1 and n < 0 or m < 0 and n >=1
    # (i.e. one poly is of degree >=1 and the other is 0)
    if m >= 1 and n < 0:
        return Matrix([0])
    elif m < 0 and n >= 1:
        return Matrix([0])

    y = var('y')

    # expr is 0 when x = y
    expr = p * q.subs({x:y}) - p.subs({x:y}) * q

    # hence expr is exactly divisible by x - y
    poly = Poly( quo(expr, x-y), x, y)

    # form Bezout matrix and store them in B as indicated to get
    # the LC coefficient of each poly either in the first position
    # of each row (method='prs') or in the last (method='bz').
    mx = max(m, n)
    B = zeros(mx)
    for i in range(mx):
        for j in range(mx):
            if method == 'prs':
                B[mx - 1 - i, mx - 1 - j] = poly.nth(i, j)
            else:
                B[i, j] = poly.nth(i, j)
    return B

def backward_eye(n):
    '''
    Returns the backward identity matrix of dimensions n x n.

    Needed to "turn" the Bezout matrices
    so that the leading coefficients are first.
    See docstring of the function bezout(p, q, x, method='bz').
    '''
    M = eye(n)  # identity matrix of order n

    for i in range(int(M.rows / 2)):
        M.row_swap(0 + i, M.rows - 1 - i)

    return M

def subresultants_bezout(p, q, x):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant polynomial remainder sequence
    of p, q by evaluating determinants of appropriately selected
    submatrices of bezout(p, q, x, 'prs'). The dimensions of the
    latter are deg(p) x deg(p).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of bezout(p, q, x, 'prs').

    bezout(p, q, x, 'prs) is used instead of sylvester(p, q, x, 1),
    Sylvester's matrix of 1840, because the dimensions of the latter
    are (deg(p) + deg(q)) x (deg(p) + deg(q)).

    If the subresultant prs is complete, then the output coincides
    with the Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    f, g = p, q
    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # subresultant list
    F = LC(f, x)**(degF - degG)

    # form the bezout matrix
    B = bezout(f, g, x, 'prs')

    # pick appropriate submatrices of B
    # and form subresultant polys
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        k, coeff_L = j - 1, []
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1

        # apply Theorem 2.1 in the paper by Toca & Vega 2004
        # to get correct signs
        SR_L.append(int((-1)**(j*(j-1)/2)) * (Poly(coeff_L, x) / F).as_expr())
        j = j + 1

    return process_matrix_output(SR_L, x)

def modified_subresultants_bezout(p, q, x):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the modified subresultant polynomial remainder sequence
    of p, q by evaluating determinants of appropriately selected
    submatrices of bezout(p, q, x, 'prs'). The dimensions of the
    latter are deg(p) x deg(p).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of bezout(p, q, x, 'prs').

    bezout(p, q, x, 'prs') is used instead of sylvester(p, q, x, 2),
    Sylvester's matrix of 1853, because the dimensions of the latter
    are 2*deg(p) x 2*deg(p).

    If the modified subresultant prs is complete, and LC( p ) > 0, the output
    coincides with the (generalized) Sturm's sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    2. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.


    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    f, g = p, q
    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # subresultant list

    # form the bezout matrix
    B = bezout(f, g, x, 'prs')

    # pick appropriate submatrices of B
    # and form subresultant polys
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        k, coeff_L = j - 1, []
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1

        ## Theorem 2.1 in the paper by Toca & Vega 2004 is _not needed_
        ## in this case since
        ## the bezout matrix is equivalent to sylvester2
        SR_L.append(( Poly(coeff_L, x)).as_expr())
        j = j + 1

    return process_matrix_output(SR_L, x)

def sturm_pg(p, q, x, method=0):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Z[x] or Q[x].
    If q = diff(p, x, 1) it is the usual Sturm sequence.

    A. If method == 0, default, the remainder coefficients of the sequence
       are (in absolute value) ``modified'' subresultants, which for non-monic
       polynomials are greater than the coefficients of the corresponding
       subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).

    B. If method == 1, the remainder coefficients of the sequence are (in
       absolute value) subresultants, which for non-monic polynomials are
       smaller than the coefficients of the corresponding ``modified''
       subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).

    If the Sturm sequence is complete, method=0 and LC( p ) > 0, the coefficients
    of the polynomials in the sequence are ``modified'' subresultants.
    That is, they are  determinants of appropriately selected submatrices of
    sylvester2, Sylvester's matrix of 1853. In this case the Sturm sequence
    coincides with the ``modified'' subresultant prs, of the polynomials
    p, q.

    If the Sturm sequence is incomplete and method=0 then the signs of the
    coefficients of the polynomials in the sequence may differ from the signs
    of the coefficients of the corresponding polynomials in the ``modified''
    subresultant prs; however, the absolute values are the same.

    To compute the coefficients, no determinant evaluation takes place. Instead,
    polynomial divisions in Q[x] are performed, using the function rem(p, q, x);
    the coefficients of the remainders computed this way become (``modified'')
    subresultants with the help of the Pell-Gordon Theorem of 1917.
    See also the function euclid_pg(p, q, x).

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # make sure LC(p) > 0
    flag = 0
    if  LC(p,x) < 0:
        flag = 1
        p = -p
        q = -q

    # initialize
    lcf = LC(p, x)**(d0 - d1)               # lcf * subr = modified subr
    a0, a1 = p, q                           # the input polys
    sturm_seq = [a0, a1]                    # the output list
    del0 = d0 - d1                          # degree difference
    rho1 =  LC(a1, x)                       # leading coeff of a1
    exp_deg = d1 - 1                        # expected degree of a2
    a2 = - rem(a0, a1, domain=QQ)           # first remainder
    rho2 =  LC(a2,x)                        # leading coeff of a2
    d2 =  degree(a2, x)                     # actual degree of a2
    deg_diff_new = exp_deg - d2             # expected - actual degree
    del1 = d1 - d2                          # degree difference

    # mul_fac is the factor by which a2 is multiplied to
    # get integer coefficients
    mul_fac_old = rho1**(del0 + del1 - deg_diff_new)

    # append accordingly
    if method == 0:
        sturm_seq.append( simplify(lcf * a2 *  Abs(mul_fac_old)))
    else:
        sturm_seq.append( simplify( a2 *  Abs(mul_fac_old)))

    # main loop
    deg_diff_old = deg_diff_new
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # update polys and degrees
        del0 = del1                          # update degree difference
        exp_deg = d1 - 1                     # new expected degree
        a2 = - rem(a0, a1, domain=QQ)        # new remainder
        rho3 =  LC(a2, x)                    # leading coeff of a2
        d2 =  degree(a2, x)                  # actual degree of a2
        deg_diff_new = exp_deg - d2          # expected - actual degree
        del1 = d1 - d2                       # degree difference

        # take into consideration the power
        # rho1**deg_diff_old that was "left out"
        expo_old = deg_diff_old               # rho1 raised to this power
        expo_new = del0 + del1 - deg_diff_new # rho2 raised to this power

        # update variables and append
        mul_fac_new = rho2**(expo_new) * rho1**(expo_old) * mul_fac_old
        deg_diff_old, mul_fac_old = deg_diff_new, mul_fac_new
        rho1, rho2 = rho2, rho3
        if method == 0:
            sturm_seq.append( simplify(lcf * a2 *  Abs(mul_fac_old)))
        else:
            sturm_seq.append( simplify( a2 *  Abs(mul_fac_old)))

    if flag:          # change the sign of the sequence
        sturm_seq = [-i for i in sturm_seq]

    # gcd is of degree > 0 ?
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)

    return sturm_seq

def sturm_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Q[x].
    Polynomial divisions in Q[x] are performed, using the function rem(p, q, x).

    The coefficients of the polynomials in the Sturm sequence can be uniquely
    determined from the corresponding coefficients of the polynomials found
    either in:

        (a) the ``modified'' subresultant prs, (references 1, 2)

    or in

        (b) the subresultant prs (reference 3).

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2 Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # make sure LC(p) > 0
    flag = 0
    if  LC(p,x) < 0:
        flag = 1
        p = -p
        q = -q

    # initialize
    a0, a1 = p, q                           # the input polys
    sturm_seq = [a0, a1]                    # the output list
    a2 = -rem(a0, a1, domain=QQ)            # first remainder
    d2 =  degree(a2, x)                     # degree of a2
    sturm_seq.append( a2 )

    # main loop
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # update polys and degrees
        a2 = -rem(a0, a1, domain=QQ)         # new remainder
        d2 =  degree(a2, x)                  # actual degree of a2
        sturm_seq.append( a2 )

    if flag:          # change the sign of the sequence
        sturm_seq = [-i for i in sturm_seq]

    # gcd is of degree > 0 ?
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)

    return sturm_seq

def sturm_amv(p, q, x, method=0):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Z[x] or Q[x].
    If q = diff(p, x, 1) it is the usual Sturm sequence.

    A. If method == 0, default, the remainder coefficients of the
       sequence are (in absolute value) ``modified'' subresultants, which
       for non-monic polynomials are greater than the coefficients of the
       corresponding subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).

    B. If method == 1, the remainder coefficients of the sequence are (in
       absolute value) subresultants, which for non-monic polynomials are
       smaller than the coefficients of the corresponding ``modified''
       subresultants by the factor Abs( LC(p)**( deg(p)- deg(q)) ).

    If the Sturm sequence is complete, method=0 and LC( p ) > 0, then the
    coefficients of the polynomials in the sequence are ``modified'' subresultants.
    That is, they are  determinants of appropriately selected submatrices of
    sylvester2, Sylvester's matrix of 1853. In this case the Sturm sequence
    coincides with the ``modified'' subresultant prs, of the polynomials
    p, q.

    If the Sturm sequence is incomplete and method=0 then the signs of the
    coefficients of the polynomials in the sequence may differ from the signs
    of the coefficients of the corresponding polynomials in the ``modified''
    subresultant prs; however, the absolute values are the same.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, we first compute the euclidean sequence  of p and q using
    euclid_amv(p, q, x) and then: (a) change the signs of the remainders in the
    Euclidean sequence according to the pattern "-, -, +, +, -, -, +, +,..."
    (see Lemma 1 in the 1st reference or Theorem 3 in the 2nd reference)
    and (b) if method=0, assuming deg(p) > deg(q), we multiply the remainder
    coefficients of the Euclidean sequence times the factor
    Abs( LC(p)**( deg(p)- deg(q)) ) to make them modified subresultants.
    See also the function sturm_pg(p, q, x).

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.'' Serdica
    Journal of Computing 9(2) (2015), 123-138.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    Remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # compute the euclidean sequence
    prs = euclid_amv(p, q, x)

    # defensive
    if prs == [] or len(prs) == 2:
        return prs

    # the coefficients in prs are subresultants and hence are smaller
    # than the corresponding subresultants by the factor
    # Abs( LC(prs[0])**( deg(prs[0]) - deg(prs[1])) ); Theorem 2, 2nd reference.
    lcf = Abs( LC(prs[0])**( degree(prs[0], x) - degree(prs[1], x) ) )

    # the signs of the first two polys in the sequence stay the same
    sturm_seq = [prs[0], prs[1]]

    # change the signs according to "-, -, +, +, -, -, +, +,..."
    # and multiply times lcf if needed
    flag = 0
    m = len(prs)
    i = 2
    while i <= m-1:
        if  flag == 0:
            sturm_seq.append( - prs[i] )
            i = i + 1
            if i == m:
                break
            sturm_seq.append( - prs[i] )
            i = i + 1
            flag = 1
        elif flag == 1:
            sturm_seq.append( prs[i] )
            i = i + 1
            if i == m:
                break
            sturm_seq.append( prs[i] )
            i = i + 1
            flag = 0

    # subresultants or modified subresultants?
    if method == 0 and lcf > 1:
        aux_seq = [sturm_seq[0], sturm_seq[1]]
        for i in range(2, m):
            aux_seq.append(simplify(sturm_seq[i] * lcf ))
        sturm_seq = aux_seq

    return sturm_seq

def euclid_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the Euclidean sequence of p and q in Z[x] or Q[x].

    If the Euclidean sequence is complete the coefficients of the polynomials
    in the sequence are subresultants. That is, they are  determinants of
    appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.
    In this case the Euclidean sequence coincides with the subresultant prs
    of the polynomials p, q.

    If the Euclidean sequence is incomplete the signs of the coefficients of the
    polynomials in the sequence may differ from the signs of the coefficients of
    the corresponding polynomials in the subresultant prs; however, the absolute
    values are the same.

    To compute the Euclidean sequence, no determinant evaluation takes place.
    We first compute the (generalized) Sturm sequence  of p and q using
    sturm_pg(p, q, x, 1), in which case the coefficients are (in absolute value)
    equal to subresultants. Then we change the signs of the remainders in the
    Sturm sequence according to the pattern "-, -, +, +, -, -, +, +,..." ;
    see Lemma 1 in the 1st reference or Theorem 3 in the 2nd reference as well as
    the function sturm_pg(p, q, x).

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.'' Serdica
    Journal of Computing 9(2) (2015), 123-138.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    Remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.
    """
    # compute the sturmian sequence using the Pell-Gordon (or AMV) theorem
    # with the coefficients in the prs being (in absolute value) subresultants
    prs = sturm_pg(p, q, x, 1)  ## any other method would do

    # defensive
    if prs == [] or len(prs) == 2:
        return prs

    # the signs of the first two polys in the sequence stay the same
    euclid_seq = [prs[0], prs[1]]

    # change the signs according to "-, -, +, +, -, -, +, +,..."
    flag = 0
    m = len(prs)
    i = 2
    while i <= m-1:
        if  flag == 0:
            euclid_seq.append(- prs[i] )
            i = i + 1
            if i == m:
                break
            euclid_seq.append(- prs[i] )
            i = i + 1
            flag = 1
        elif flag == 1:
            euclid_seq.append(prs[i] )
            i = i + 1
            if i == m:
                break
            euclid_seq.append(prs[i] )
            i = i + 1
            flag = 0

    return euclid_seq

def euclid_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the Euclidean sequence of p and q in Q[x].
    Polynomial divisions in Q[x] are performed, using the function rem(p, q, x).

    The coefficients of the polynomials in the Euclidean sequence can be uniquely
    determined from the corresponding coefficients of the polynomials found
    either in:

        (a) the ``modified'' subresultant polynomial remainder sequence,
    (references 1, 2)

    or in

        (b) the subresultant polynomial remainder sequence (references 3).

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # make sure LC(p) > 0
    flag = 0
    if  LC(p,x) < 0:
        flag = 1
        p = -p
        q = -q

    # initialize
    a0, a1 = p, q                           # the input polys
    euclid_seq = [a0, a1]                   # the output list
    a2 = rem(a0, a1, domain=QQ)             # first remainder
    d2 =  degree(a2, x)                     # degree of a2
    euclid_seq.append( a2 )

    # main loop
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # update polys and degrees
        a2 = rem(a0, a1, domain=QQ)          # new remainder
        d2 =  degree(a2, x)                  # actual degree of a2
        euclid_seq.append( a2 )

    if flag:          # change the sign of the sequence
        euclid_seq = [-i for i in euclid_seq]

    # gcd is of degree > 0 ?
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:
        euclid_seq.pop(m - 1)

    return euclid_seq

def euclid_amv(f, g, x):
    """
    f, g are polynomials in Z[x] or Q[x]. It is assumed
    that degree(f, x) >= degree(g, x).

    Computes the Euclidean sequence of p and q in Z[x] or Q[x].

    If the Euclidean sequence is complete the coefficients of the polynomials
    in the sequence are subresultants. That is, they are  determinants of
    appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.
    In this case the Euclidean sequence coincides with the subresultant prs,
    of the polynomials p, q.

    If the Euclidean sequence is incomplete the signs of the coefficients of the
    polynomials in the sequence may differ from the signs of the coefficients of
    the corresponding polynomials in the subresultant prs; however, the absolute
    values are the same.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Z[x] or Q[x] are performed, using
    the function rem_z(f, g, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Collins-Brown-Traub formula for coefficient reduction.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    # make sure proper degrees
    d0 =  degree(f, x)
    d1 =  degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        d0, d1 = d1, d0
        f, g = g, f
    if d0 > 0 and d1 == 0:
        return [f, g]

    # initialize
    a0 = f
    a1 = g
    euclid_seq = [a0, a1]
    deg_dif_p1, c = degree(a0, x) - degree(a1, x) + 1, -1

    # compute the first polynomial of the prs
    i = 1
    a2 = rem_z(a0, a1, x) / Abs( (-1)**deg_dif_p1 )     # first remainder
    euclid_seq.append( a2 )
    d2 =  degree(a2, x)                              # actual degree of a2

    # main loop
    while d2 >= 1:
        a0, a1, d0, d1 = a1, a2, d1, d2       # update polys and degrees
        i += 1
        sigma0 = -LC(a0)
        c = (sigma0**(deg_dif_p1 - 1)) / (c**(deg_dif_p1 - 2))
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs( (c**(deg_dif_p1 - 1)) * sigma0 )
        euclid_seq.append( a2 )
        d2 =  degree(a2, x)                   # actual degree of a2

    # gcd is of degree > 0 ?
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:
        euclid_seq.pop(m - 1)

    return euclid_seq

def modified_subresultants_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the ``modified'' subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    ``modified'' subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester2, Sylvester's matrix of 1853.

    To compute the coefficients, no determinant evaluation takes place. Instead,
    polynomial divisions in Q[x] are performed, using the function rem(p, q, x);
    the coefficients of the remainders computed this way become ``modified''
    subresultants with the help of the Pell-Gordon Theorem of 1917.

    If the ``modified'' subresultant prs is complete, and LC( p ) > 0, it coincides
    with the (generalized) Sturm sequence of the polynomials p, q.

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p,x)
    d1 =  degree(q,x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # initialize
    k =  var('k')                               # index in summation formula
    u_list = []                                 # of elements (-1)**u_i
    subres_l = [p, q]                           # mod. subr. prs output list
    a0, a1 = p, q                               # the input polys
    del0 = d0 - d1                              # degree difference
    degdif = del0                               # save it
    rho_1 = LC(a0)                              # lead. coeff (a0)

    # Initialize Pell-Gordon variables
    rho_list_minus_1 =  sign( LC(a0, x))      # sign of LC(a0)
    rho1 =  LC(a1, x)                         # leading coeff of a1
    rho_list = [ sign(rho1)]                  # of signs
    p_list = [del0]                           # of degree differences
    u =  summation(k, (k, 1, p_list[0]))      # value of u
    u_list.append(u)                          # of u values
    v = sum(p_list)                           # v value

    # first remainder
    exp_deg = d1 - 1                          # expected degree of a2
    a2 = - rem(a0, a1, domain=QQ)             # first remainder
    rho2 =  LC(a2, x)                         # leading coeff of a2
    d2 =  degree(a2, x)                       # actual degree of a2
    deg_diff_new = exp_deg - d2               # expected - actual degree
    del1 = d1 - d2                            # degree difference

    # mul_fac is the factor by which a2 is multiplied to
    # get integer coefficients
    mul_fac_old = rho1**(del0 + del1 - deg_diff_new)

    # update Pell-Gordon variables
    p_list.append(1 + deg_diff_new)              # deg_diff_new is 0 for complete seq

    # apply Pell-Gordon formula (7) in second reference
    num = 1                                     # numerator of fraction
    for u in u_list:
        num *= (-1)**u
    num = num * (-1)**v

    # denominator depends on complete / incomplete seq
    if deg_diff_new == 0:                        # complete seq
        den = 1
        for k in range(len(rho_list)):
            den *= rho_list[k]**(p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
    else:                                       # incomplete seq
        den = 1
        for k in range(len(rho_list)-1):
            den *= rho_list[k]**(p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
        expo = (p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new)
        den = den * rho_list[len(rho_list) - 1]**expo

    # the sign of the determinant depends on sg(num / den)
    if  sign(num / den) > 0:
        subres_l.append( simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )
    else:
        subres_l.append(- simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )

    # update Pell-Gordon variables
    k = var('k')
    rho_list.append( sign(rho2))
    u =  summation(k, (k, 1, p_list[len(p_list) - 1]))
    u_list.append(u)
    v = sum(p_list)
    deg_diff_old=deg_diff_new

    # main loop
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # update polys and degrees
        del0 = del1                          # update degree difference
        exp_deg = d1 - 1                     # new expected degree
        a2 = - rem(a0, a1, domain=QQ)        # new remainder
        rho3 =  LC(a2, x)                    # leading coeff of a2
        d2 =  degree(a2, x)                  # actual degree of a2
        deg_diff_new = exp_deg - d2          # expected - actual degree
        del1 = d1 - d2                       # degree difference

        # take into consideration the power
        # rho1**deg_diff_old that was "left out"
        expo_old = deg_diff_old               # rho1 raised to this power
        expo_new = del0 + del1 - deg_diff_new # rho2 raised to this power

        mul_fac_new = rho2**(expo_new) * rho1**(expo_old) * mul_fac_old

        # update variables
        deg_diff_old, mul_fac_old = deg_diff_new, mul_fac_new
        rho1, rho2 = rho2, rho3

        # update Pell-Gordon variables
        p_list.append(1 + deg_diff_new)       # deg_diff_new is 0 for complete seq

        # apply Pell-Gordon formula (7) in second reference
        num = 1                              # numerator
        for u in u_list:
            num *= (-1)**u
        num = num * (-1)**v

        # denominator depends on complete / incomplete seq
        if deg_diff_new == 0:                 # complete seq
            den = 1
            for k in range(len(rho_list)):
                den *= rho_list[k]**(p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
        else:                                # incomplete seq
            den = 1
            for k in range(len(rho_list)-1):
                den *= rho_list[k]**(p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
            expo = (p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new)
            den = den * rho_list[len(rho_list) - 1]**expo

        # the sign of the determinant depends on sg(num / den)
        if  sign(num / den) > 0:
            subres_l.append( simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )
        else:
            subres_l.append(- simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )

        # update Pell-Gordon variables
        k = var('k')
        rho_list.append( sign(rho2))
        u =  summation(k, (k, 1, p_list[len(p_list) - 1]))
        u_list.append(u)
        v = sum(p_list)

    # gcd is of degree > 0 ?
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)

    # LC( p ) < 0
    m = len(subres_l)   # list may be shorter now due to deg(gcd ) > 0
    if LC( p ) < 0:
        aux_seq = [subres_l[0], subres_l[1]]
        for i in range(2, m):
            aux_seq.append(simplify(subres_l[i] * (-1) ))
        subres_l = aux_seq

    return  subres_l

def subresultants_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x], from
    the modified subresultant prs of p and q.

    The coefficients of the polynomials in these two sequences differ only
    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in
    Theorem 2 of the reference.

    The coefficients of the polynomials in the output sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials."
    Serdica Journal of Computing 9(2) (2015), 123-138.

    """
    # compute the modified subresultant prs
    lst = modified_subresultants_pg(p,q,x)  ## any other method would do

    # defensive
    if lst == [] or len(lst) == 2:
        return lst

    # the coefficients in lst are modified subresultants and, hence, are
    # greater than those of the corresponding subresultants by the factor
    # LC(lst[0])**( deg(lst[0]) - deg(lst[1])); see Theorem 2 in reference.
    lcf = LC(lst[0])**( degree(lst[0], x) - degree(lst[1], x) )

    # Initialize the subresultant prs list
    subr_seq = [lst[0], lst[1]]

    # compute the degree sequences m_i and j_i of Theorem 2 in reference.
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m-1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]

    # compute the AMV factors of Theorem 2 in reference.
    fact = [(-1)**( j*(j-1)/S(2) ) for j in j_seq]

    # shortened list without the first two polys
    lst_s = lst[2:]

    # poly lst_s[k] is multiplied times fact[k], divided by lcf
    # and appended to the subresultant prs list
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append(-lst_s[k] / lcf)
        else:
            subr_seq.append(lst_s[k] / lcf)

    return subr_seq

def subresultants_amv_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Q[x] are performed, using the
    function rem(p, q, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Akritas-Malaschonok-Vigklas Theorem of 2015.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p, q]

    # initialize
    i, s = 0, 0                              # counters for remainders & odd elements
    p_odd_index_sum = 0                      # contains the sum of p_1, p_3, etc
    subres_l = [p, q]                        # subresultant prs output list
    a0, a1 = p, q                            # the input polys
    sigma1 =  LC(a1, x)                      # leading coeff of a1
    p0 = d0 - d1                             # degree difference
    if p0 % 2 == 1:
        s += 1
    phi = floor( (s + 1) / 2 )
    mul_fac = 1
    d2 = d1

    # main loop
    while d2 > 0:
        i += 1
        a2 = rem(a0, a1, domain= QQ)          # new remainder
        if i == 1:
            sigma2 =  LC(a2, x)
        else:
            sigma3 =  LC(a2, x)
            sigma1, sigma2 = sigma2, sigma3
        d2 =  degree(a2, x)
        p1 = d1 - d2
        psi = i + phi + p_odd_index_sum

        # new mul_fac
        mul_fac = sigma1**(p0 + 1) * mul_fac

        ## compute the sign of the first fraction in formula (9) of the paper
        # numerator
        num = (-1)**psi
        # denominator
        den = sign(mul_fac)

        # the sign of the determinant depends on sign( num / den ) != 0
        if  sign(num / den) > 0:
            subres_l.append( simplify(expand(a2* Abs(mul_fac))))
        else:
            subres_l.append(- simplify(expand(a2* Abs(mul_fac))))

        ## bring into mul_fac the missing power of sigma if there was a degree gap
        if p1 - 1 > 0:
            mul_fac = mul_fac * sigma1**(p1 - 1)

       # update AMV variables
        a0, a1, d0, d1 = a1, a2, d1, d2
        p0 = p1
        if p0 % 2 ==1:
            s += 1
        phi = floor( (s + 1) / 2 )
        if i%2 == 1:
            p_odd_index_sum += p0             # p_i has odd index

    # gcd is of degree > 0 ?
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)

    return  subres_l

def compute_sign(base, expo):
    '''
    base != 0 and expo >= 0 are integers;

    returns the sign of base**expo without
    evaluating the power itself!
    '''
    sb = sign(base)
    if sb == 1:
        return 1
    pe = expo % 2
    if pe == 0:
        return -sb
    else:
        return sb

def rem_z(p, q, x):
    '''
    Intended mainly for p, q polynomials in Z[x] so that,
    on dividing p by q, the remainder will also be in Z[x]. (However,
    it also works fine for polynomials in Q[x].) It is assumed
    that degree(p, x) >= degree(q, x).

    It premultiplies p by the _absolute_ value of the leading coefficient
    of q, raised to the power deg(p) - deg(q) + 1 and then performs
    polynomial division in Q[x], using the function rem(p, q, x).

    By contrast the function prem(p, q, x) does _not_ use the absolute
    value of the leading coefficient of q.
    This results not only in ``messing up the signs'' of the Euclidean and
    Sturmian prs's as mentioned in the second reference,
    but also in violation of the main results of the first and third
    references --- Theorem 4 and Theorem 1 respectively. Theorems 4 and 1
    establish a one-to-one correspondence between the Euclidean and the
    Sturmian prs of p, q, on one hand, and the subresultant prs of p, q,
    on the other.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.''
    Serdica Journal of Computing, 9(2) (2015), 123-138.

    2. https://planetMath.org/sturmstheorem

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result on
    the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    '''
    if (p.as_poly().is_univariate and q.as_poly().is_univariate and
            p.as_poly().gens == q.as_poly().gens):
        delta = (degree(p, x) - degree(q, x) + 1)
        return rem(Abs(LC(q, x))**delta  *  p, q, x)
    else:
        return prem(p, q, x)

def quo_z(p, q, x):
    """
    Intended mainly for p, q polynomials in Z[x] so that,
    on dividing p by q, the quotient will also be in Z[x]. (However,
    it also works fine for polynomials in Q[x].) It is assumed
    that degree(p, x) >= degree(q, x).

    It premultiplies p by the _absolute_ value of the leading coefficient
    of q, raised to the power deg(p) - deg(q) + 1 and then performs
    polynomial division in Q[x], using the function quo(p, q, x).

    By contrast the function pquo(p, q, x) does _not_ use the absolute
    value of the leading coefficient of q.

    See also function rem_z(p, q, x) for additional comments and references.

    """
    if (p.as_poly().is_univariate and q.as_poly().is_univariate and
            p.as_poly().gens == q.as_poly().gens):
        delta = (degree(p, x) - degree(q, x) + 1)
        return quo(Abs(LC(q, x))**delta  *  p, q, x)
    else:
        return pquo(p, q, x)

def subresultants_amv(f, g, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(f, x) >= degree(g, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Z[x] or Q[x] are performed, using
    the function rem_z(p, q, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Akritas-Malaschonok-Vigklas Theorem of 2015 and the Collins-Brown-
    Traub formula for coefficient reduction.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    # make sure proper degrees
    d0 =  degree(f, x)
    d1 =  degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        d0, d1 = d1, d0
        f, g = g, f
    if d0 > 0 and d1 == 0:
        return [f, g]

    # initialize
    a0 = f
    a1 = g
    subres_l = [a0, a1]
    deg_dif_p1, c = degree(a0, x) - degree(a1, x) + 1, -1

    # initialize AMV variables
    sigma1 =  LC(a1, x)                      # leading coeff of a1
    i, s = 0, 0                              # counters for remainders & odd elements
    p_odd_index_sum = 0                      # contains the sum of p_1, p_3, etc
    p0 = deg_dif_p1 - 1
    if p0 % 2 == 1:
        s += 1
    phi = floor( (s + 1) / 2 )

    # compute the first polynomial of the prs
    i += 1
    a2 = rem_z(a0, a1, x) / Abs( (-1)**deg_dif_p1 )     # first remainder
    sigma2 =  LC(a2, x)                       # leading coeff of a2
    d2 =  degree(a2, x)                       # actual degree of a2
    p1 = d1 - d2                              # degree difference

    # sgn_den is the factor, the denominator 1st fraction of (9),
    # by which a2 is multiplied to get integer coefficients
    sgn_den = compute_sign( sigma1, p0 + 1 )

    ## compute sign of the 1st fraction in formula (9) of the paper
    # numerator
    psi = i + phi + p_odd_index_sum
    num = (-1)**psi
    # denominator
    den = sgn_den

    # the sign of the determinant depends on sign(num / den) != 0
    if  sign(num / den) > 0:
        subres_l.append( a2 )
    else:
        subres_l.append( -a2 )

    # update AMV variable
    if p1 % 2 == 1:
        s += 1

    # bring in the missing power of sigma if there was gap
    if p1 - 1 > 0:
        sgn_den = sgn_den * compute_sign( sigma1, p1 - 1 )

    # main loop
    while d2 >= 1:
        phi = floor( (s + 1) / 2 )
        if i%2 == 1:
            p_odd_index_sum += p1             # p_i has odd index
        a0, a1, d0, d1 = a1, a2, d1, d2       # update polys and degrees
        p0 = p1                               # update degree difference
        i += 1
        sigma0 = -LC(a0)
        c = (sigma0**(deg_dif_p1 - 1)) / (c**(deg_dif_p1 - 2))
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs( (c**(deg_dif_p1 - 1)) * sigma0 )
        sigma3 =  LC(a2, x)                   # leading coeff of a2
        d2 =  degree(a2, x)                   # actual degree of a2
        p1 = d1 - d2                          # degree difference
        psi = i + phi + p_odd_index_sum

        # update variables
        sigma1, sigma2 = sigma2, sigma3

        # new sgn_den
        sgn_den = compute_sign( sigma1, p0 + 1 ) * sgn_den

        # compute the sign of the first fraction in formula (9) of the paper
        # numerator
        num = (-1)**psi
        # denominator
        den = sgn_den

        # the sign of the determinant depends on sign( num / den ) != 0
        if  sign(num / den) > 0:
            subres_l.append( a2 )
        else:
            subres_l.append( -a2 )

        # update AMV variable
        if p1 % 2 ==1:
            s += 1

        # bring in the missing power of sigma if there was gap
        if p1 - 1 > 0:
            sgn_den = sgn_den * compute_sign( sigma1, p1 - 1 )

    # gcd is of degree > 0 ?
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)

    return subres_l

def modified_subresultants_amv(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the modified subresultant prs of p and q in Z[x] or Q[x],
    from the subresultant prs of p and q.
    The coefficients of the polynomials in the two sequences differ only
    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in
    Theorem 2 of the reference.

    The coefficients of the polynomials in the output sequence are
    modified subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester2, Sylvester's matrix of 1853.

    If the modified subresultant prs is complete, and LC( p ) > 0, it coincides
    with the (generalized) Sturm's sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials."
    Serdica Journal of Computing, Serdica Journal of Computing, 9(2) (2015), 123-138.

    """
    # compute the subresultant prs
    lst = subresultants_amv(p,q,x)     ## any other method would do

    # defensive
    if lst == [] or len(lst) == 2:
        return lst

    # the coefficients in lst are subresultants and, hence, smaller than those
    # of the corresponding modified subresultants by the factor
    # LC(lst[0])**( deg(lst[0]) - deg(lst[1])); see Theorem 2.
    lcf = LC(lst[0])**( degree(lst[0], x) - degree(lst[1], x) )

    # Initialize the modified subresultant prs list
    subr_seq = [lst[0], lst[1]]

    # compute the degree sequences m_i and j_i of Theorem 2
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m-1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]

    # compute the AMV factors of Theorem 2
    fact = [(-1)**( j*(j-1)/S(2) ) for j in j_seq]

    # shortened list without the first two polys
    lst_s = lst[2:]

    # poly lst_s[k] is multiplied times fact[k] and times lcf
    # and appended to the subresultant prs list
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append( simplify(-lst_s[k] * lcf) )
        else:
            subr_seq.append( simplify(lst_s[k] * lcf) )

    return subr_seq

def correct_sign(deg_f, deg_g, s1, rdel, cdel):
    """
    Used in various subresultant prs algorithms.

    Evaluates the determinant, (a.k.a. subresultant) of a properly selected
    submatrix of s1, Sylvester's matrix of 1840, to get the correct sign
    and value of the leading coefficient of a given polynomial remainder.

    deg_f, deg_g are the degrees of the original polynomials p, q for which the
    matrix s1 = sylvester(p, q, x, 1) was constructed.

    rdel denotes the expected degree of the remainder; it is the number of
    rows to be deleted from each group of rows in s1 as described in the
    reference below.

    cdel denotes the expected degree minus the actual degree of the remainder;
    it is the number of columns to be deleted --- starting with the last column
    forming the square matrix --- from the matrix resulting after the row deletions.

    References
    ==========
    Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    M = s1[:, :]   # copy of matrix s1

    # eliminate rdel rows from the first deg_g rows
    for i in range(M.rows - deg_f - 1, M.rows - deg_f - rdel - 1, -1):
        M.row_del(i)

    # eliminate rdel rows from the last deg_f rows
    for i in range(M.rows - 1, M.rows - rdel - 1, -1):
        M.row_del(i)

    # eliminate cdel columns
    for i in range(cdel):
        M.col_del(M.rows - 1)

    # define submatrix
    Md = M[:, 0: M.rows]

    return Md.det()

def subresultants_rem(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients polynomial divisions in Q[x] are
    performed, using the function rem(p, q, x). The coefficients
    of the remainders computed this way become subresultants by evaluating
    one subresultant per remainder --- that of the leading coefficient.
    This way we obtain the correct sign and value of the leading coefficient
    of the remainder and we easily ``force'' the rest of the coefficients
    to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # initialize
    s1 = sylvester(f, g, x, 1)
    sr_list = [f, g]      # subresultant list

    # main loop
    while deg_g > 0:
        r = rem(p, q, x)
        d = degree(r, x)
        if d < 0:
            return sr_list

        # make coefficients subresultants evaluating ONE determinant
        exp_deg = deg_g - 1          # expected degree
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        r = simplify((r / LC(r, x)) * sign_value)

        # append poly with subresultant coeffs
        sr_list.append(r)

        # update degrees and polys
        deg_f, deg_g = deg_g, d
        p, q = q, r

    # gcd is of degree > 0 ?
    m = len(sr_list)
    if sr_list[m - 1] == nan or sr_list[m - 1] == 0:
        sr_list.pop(m - 1)

    return sr_list

def pivot(M, i, j):
    '''
    M is a matrix, and M[i, j] specifies the pivot element.

    All elements below M[i, j], in the j-th column, will
    be zeroed, if they are not already 0, according to
    Dodgson-Bareiss' integer preserving transformations.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    '''
    ma = M[:, :] # copy of matrix M
    rs = ma.rows # No. of rows
    cs = ma.cols # No. of cols
    for r in range(i+1, rs):
        if ma[r, j] != 0:
            for c in range(j + 1, cs):
                ma[r, c] = ma[i, j] * ma[r, c] - ma[i, c] * ma[r, j]
            ma[r, j] = 0
    return ma

def rotate_r(L, k):
    '''
    Rotates right by k. L is a row of a matrix or a list.

    '''
    ll = list(L)
    if ll == []:
        return []
    for i in range(k):
        el = ll.pop(len(ll) - 1)
        ll.insert(0, el)
    return ll if isinstance(L, list) else Matrix([ll])

def rotate_l(L, k):
    '''
    Rotates left by k. L is a row of a matrix or a list.

    '''
    ll = list(L)
    if ll == []:
        return []
    for i in range(k):
        el = ll.pop(0)
        ll.insert(len(ll) - 1, el)
    return ll if isinstance(L, list) else Matrix([ll])

def row2poly(row, deg, x):
    '''
    Converts the row of a matrix to a poly of degree deg and variable x.
    Some entries at the beginning and/or at the end of the row may be zero.

    '''
    k = 0
    poly = []
    leng = len(row)

    # find the beginning of the poly ; i.e. the first
    # non-zero element of the row
    while row[k] == 0:
        k = k + 1

    # append the next deg + 1 elements to poly
    for j in range( deg + 1):
        if k + j <= leng:
            poly.append(row[k + j])

    return Poly(poly, x)

def create_ma(deg_f, deg_g, row1, row2, col_num):
    '''
    Creates a ``small'' matrix M to be triangularized.

    deg_f, deg_g are the degrees of the divident and of the
    divisor polynomials respectively, deg_g > deg_f.

    The coefficients of the divident poly are the elements
    in row2 and those of the divisor poly are the elements
    in row1.

    col_num defines the number of columns of the matrix M.

    '''
    if deg_g - deg_f >= 1:
        print('Reverse degrees')
        return

    m = zeros(deg_f - deg_g + 2, col_num)

    for i in range(deg_f - deg_g + 1):
        m[i, :] = rotate_r(row1, i)
    m[deg_f - deg_g + 1, :] = row2

    return m

def find_degree(M, deg_f):
    '''
    Finds the degree of the poly corresponding (after triangularization)
    to the _last_ row of the ``small'' matrix M, created by create_ma().

    deg_f is the degree of the divident poly.
    If _last_ row is all 0's returns None.

    '''
    j = deg_f
    for i in range(0, M.cols):
        if M[M.rows - 1, i] == 0:
            j = j - 1
        else:
            return max(j, 0)

def final_touches(s2, r, deg_g):
    """
    s2 is sylvester2, r is the row pointer in s2,
    deg_g is the degree of the poly last inserted in s2.

    After a gcd of degree > 0 has been found with Van Vleck's
    method, and was inserted into s2, if its last term is not
    in the last column of s2, then it is inserted as many
    times as needed, rotated right by one each time, until
    the condition is met.

    """
    R = s2.row(r-1)

    # find the first non zero term
    for i in range(s2.cols):
        if R[0,i] == 0:
            continue
        else:
            break

    # missing rows until last term is in last column
    mr = s2.cols - (i + deg_g + 1)

    # insert them by replacing the existing entries in the row
    i = 0
    while mr != 0 and r + i < s2.rows :
        s2[r + i, : ] = rotate_r(R, i + 1)
        i += 1
        mr -= 1

    return s2

def subresultants_vv(p, q, x, method = 0):
    """
    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p, q by triangularizing,
    in Z[x] or in Q[x], all the smaller matrices encountered in the
    process of triangularizing sylvester2, Sylvester's matrix of 1853;
    see references 1 and 2 for Van Vleck's method. With each remainder,
    sylvester2 gets updated and is prepared to be printed if requested.

    If sylvester2 has small dimensions and you want to see the final,
    triangularized matrix use this version with method=1; otherwise,
    use either this version with method=0 (default) or the faster version,
    subresultants_vv_2(p, q, x), where sylvester2 is used implicitly.

    Sylvester's matrix sylvester1  is also used to compute one
    subresultant per remainder; namely, that of the leading
    coefficient, in order to obtain the correct sign and to
    force the remainder coefficients to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    If the final, triangularized matrix s2 is printed, then:
        (a) if deg(p) - deg(q) > 1 or deg( gcd(p, q) ) > 0, several
            of the last rows in s2 will remain unprocessed;
        (b) if deg(p) - deg(q) == 0, p will not appear in the final matrix.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    3. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # initialize
    s1 = sylvester(f, g, x, 1)
    s2 = sylvester(f, g, x, 2)
    sr_list = [f, g]
    col_num = 2 * n         # columns in s2

    # make two rows (row0, row1) of poly coefficients
    row0 = Poly(f, x, domain = QQ).all_coeffs()
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)
    row0 = Matrix([row0])
    row1 = Poly(g,x, domain = QQ).all_coeffs()
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)
    row1 = Matrix([row1])

    # row pointer for deg_f - deg_g == 1; may be reset below
    r = 2

    # modify first rows of s2 matrix depending on poly degrees
    if deg_f - deg_g > 1:
        r = 1
        # replacing the existing entries in the rows of s2,
        # insert row0 (deg_f - deg_g - 1) times, rotated each time
        for i in range(deg_f - deg_g - 1):
            s2[r + i, : ] = rotate_r(row0, i + 1)
        r = r + deg_f - deg_g - 1
        # insert row1 (deg_f - deg_g) times, rotated each time
        for i in range(deg_f - deg_g):
            s2[r + i, : ] = rotate_r(row1, r + i)
        r = r + deg_f - deg_g

    if deg_f - deg_g == 0:
        r = 0

    # main loop
    while deg_g > 0:
        # create a small matrix M, and triangularize it;
        M = create_ma(deg_f, deg_g, row1, row0, col_num)
        # will need only the first and last rows of M
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]

        # treat last row of M as poly; find its degree
        d = find_degree(M, deg_f)
        if d is None:
            break
        exp_deg = deg_g - 1

        # evaluate one determinant & make coefficients subresultants
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        temp2 = LC(poly, x)
        poly = simplify((poly / temp2) * sign_value)

        # update s2 by inserting first row of M as needed
        row0 = M[0, :]
        for i in range(deg_g - d):
            s2[r + i, :] = rotate_r(row0, r + i)
        r = r + deg_g - d

        # update s2 by inserting last row of M as needed
        row1 = rotate_l(M[M.rows - 1, :], deg_f - d)
        row1 = (row1 / temp2) * sign_value
        for i in range(deg_g - d):
            s2[r + i, :] = rotate_r(row1, r + i)
        r = r + deg_g - d

        # update degrees
        deg_f, deg_g = deg_g, d

        # append poly with subresultant coeffs
        sr_list.append(poly)

    # final touches to print the s2 matrix
    if method != 0 and s2.rows > 2:
        s2 = final_touches(s2, r, deg_g)
        pprint(s2)
    elif method != 0 and s2.rows == 2:
        s2[1, :] = rotate_r(s2.row(1), 1)
        pprint(s2)

    return sr_list

def subresultants_vv_2(p, q, x):
    """
    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p, q by triangularizing,
    in Z[x] or in Q[x], all the smaller matrices encountered in the
    process of triangularizing sylvester2, Sylvester's matrix of 1853;
    see references 1 and 2 for Van Vleck's method.

    If the sylvester2 matrix has big dimensions use this version,
    where sylvester2 is used implicitly. If you want to see the final,
    triangularized matrix sylvester2, then use the first version,
    subresultants_vv(p, q, x, 1).

    sylvester1, Sylvester's matrix of 1840, is also used to compute
    one subresultant per remainder; namely, that of the leading
    coefficient, in order to obtain the correct sign and to
    ``force'' the remainder coefficients to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    3. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # initialize
    s1 = sylvester(f, g, x, 1)
    sr_list = [f, g]      # subresultant list
    col_num = 2 * n       # columns in sylvester2

    # make two rows (row0, row1) of poly coefficients
    row0 = Poly(f, x, domain = QQ).all_coeffs()
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)
    row0 = Matrix([row0])
    row1 = Poly(g,x, domain = QQ).all_coeffs()
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)
    row1 = Matrix([row1])

    # main loop
    while deg_g > 0:
        # create a small matrix M, and triangularize it
        M = create_ma(deg_f, deg_g, row1, row0, col_num)
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]

        # treat last row of M as poly; find its degree
        d = find_degree(M, deg_f)
        if d is None:
            return sr_list
        exp_deg = deg_g - 1

        # evaluate one determinant & make coefficients subresultants
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        poly = simplify((poly / LC(poly, x)) * sign_value)

        # append poly with subresultant coeffs
        sr_list.append(poly)

        # update degrees and rows
        deg_f, deg_g = deg_g, d
        row0 = row1
        row1 = Poly(poly, x, domain = QQ).all_coeffs()
        leng1 = len(row1)
        for i in range(col_num - leng1):
            row1.append(0)
        row1 = Matrix([row1])

    return sr_list
<|endoftext|>"""Configuration utilities for polynomial manipulation algorithms. """


from contextlib import contextmanager

_default_config = {
    'USE_COLLINS_RESULTANT':      False,
    'USE_SIMPLIFY_GCD':           True,
    'USE_HEU_GCD':                True,

    'USE_IRREDUCIBLE_IN_FACTOR':  False,
    'USE_CYCLOTOMIC_FACTOR':      True,

    'EEZ_RESTART_IF_NEEDED':      True,
    'EEZ_NUMBER_OF_CONFIGS':      3,
    'EEZ_NUMBER_OF_TRIES':        5,
    'EEZ_MODULUS_STEP':           2,

    'GF_IRRED_METHOD':            'rabin',
    'GF_FACTOR_METHOD':           'zassenhaus',

    'GROEBNER':                   'buchberger',
}

_current_config = {}

@contextmanager
def using(**kwargs):
    for k, v in kwargs.items():
        setup(k, v)

    yield

    for k in kwargs.keys():
        setup(k)

def setup(key, value=None):
    """Assign a value to (or reset) a configuration item. """
    key = key.upper()

    if value is not None:
        _current_config[key] = value
    else:
        _current_config[key] = _default_config[key]


def query(key):
    """Ask for a value of the given configuration item. """
    return _current_config.get(key.upper(), None)


def configure():
    """Initialized configuration of polys module. """
    from os import getenv

    for key, default in _default_config.items():
        value = getenv('SYMPY_' + key)

        if value is not None:
            try:
                _current_config[key] = eval(value)
            except NameError:
                _current_config[key] = value
        else:
            _current_config[key] = default

configure()
<|endoftext|>"""Computations with ideals of polynomial rings."""

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable


class Ideal(IntegerPowerable):
    """
    Abstract base class for ideals.

    Do not instantiate - use explicit constructors in the ring class instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> QQ.old_poly_ring(x).ideal(x+1)
    <x + 1>

    Attributes

    - ring - the ring this ideal belongs to

    Non-implemented methods:

    - _contains_elem
    - _contains_ideal
    - _quotient
    - _intersect
    - _union
    - _product
    - is_whole_ring
    - is_zero
    - is_prime, is_maximal, is_primary, is_radical
    - is_principal
    - height, depth
    - radical

    Methods that likely should be overridden in subclasses:

    - reduce_element
    """

    def _contains_elem(self, x):
        """Implementation of element containment."""
        raise NotImplementedError

    def _contains_ideal(self, I):
        """Implementation of ideal containment."""
        raise NotImplementedError

    def _quotient(self, J):
        """Implementation of ideal quotient."""
        raise NotImplementedError

    def _intersect(self, J):
        """Implementation of ideal intersection."""
        raise NotImplementedError

    def is_whole_ring(self):
        """Return True if ``self`` is the whole ring."""
        raise NotImplementedError

    def is_zero(self):
        """Return True if ``self`` is the zero ideal."""
        raise NotImplementedError

    def _equals(self, J):
        """Implementation of ideal equality."""
        return self._contains_ideal(J) and J._contains_ideal(self)

    def is_prime(self):
        """Return True if ``self`` is a prime ideal."""
        raise NotImplementedError

    def is_maximal(self):
        """Return True if ``self`` is a maximal ideal."""
        raise NotImplementedError

    def is_radical(self):
        """Return True if ``self`` is a radical ideal."""
        raise NotImplementedError

    def is_primary(self):
        """Return True if ``self`` is a primary ideal."""
        raise NotImplementedError

    def is_principal(self):
        """Return True if ``self`` is a principal ideal."""
        raise NotImplementedError

    def radical(self):
        """Compute the radical of ``self``."""
        raise NotImplementedError

    def depth(self):
        """Compute the depth of ``self``."""
        raise NotImplementedError

    def height(self):
        """Compute the height of ``self``."""
        raise NotImplementedError

    # TODO more

    # non-implemented methods end here

    def __init__(self, ring):
        self.ring = ring

    def _check_ideal(self, J):
        """Helper to check ``J`` is an ideal of our ring."""
        if not isinstance(J, Ideal) or J.ring != self.ring:
            raise ValueError(
                'J must be an ideal of %s, got %s' % (self.ring, J))

    def contains(self, elem):
        """
        Return True if ``elem`` is an element of this ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x+1, x-1).contains(3)
        True
        >>> QQ.old_poly_ring(x).ideal(x**2, x**3).contains(x)
        False
        """
        return self._contains_elem(self.ring.convert(elem))

    def subset(self, other):
        """
        Returns True if ``other`` is is a subset of ``self``.

        Here ``other`` may be an ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x+1)
        >>> I.subset([x**2 - 1, x**2 + 2*x + 1])
        True
        >>> I.subset([x**2 + 1, x + 1])
        False
        >>> I.subset(QQ.old_poly_ring(x).ideal(x**2 - 1))
        True
        """
        if isinstance(other, Ideal):
            return self._contains_ideal(other)
        return all(self._contains_elem(x) for x in other)

    def quotient(self, J, **opts):
        r"""
        Compute the ideal quotient of ``self`` by ``J``.

        That is, if ``self`` is the ideal `I`, compute the set
        `I : J = \{x \in R | xJ \subset I \}`.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> R = QQ.old_poly_ring(x, y)
        >>> R.ideal(x*y).quotient(R.ideal(x))
        <y>
        """
        self._check_ideal(J)
        return self._quotient(J, **opts)

    def intersect(self, J):
        """
        Compute the intersection of self with ideal J.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> R = QQ.old_poly_ring(x, y)
        >>> R.ideal(x).intersect(R.ideal(y))
        <x*y>
        """
        self._check_ideal(J)
        return self._intersect(J)

    def saturate(self, J):
        r"""
        Compute the ideal saturation of ``self`` by ``J``.

        That is, if ``self`` is the ideal `I`, compute the set
        `I : J^\infty = \{x \in R | xJ^n \subset I \text{ for some } n\}`.
        """
        raise NotImplementedError
        # Note this can be implemented using repeated quotient

    def union(self, J):
        """
        Compute the ideal generated by the union of ``self`` and ``J``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x**2 - 1).union(QQ.old_poly_ring(x).ideal((x+1)**2)) == QQ.old_poly_ring(x).ideal(x+1)
        True
        """
        self._check_ideal(J)
        return self._union(J)

    def product(self, J):
        r"""
        Compute the ideal product of ``self`` and ``J``.

        That is, compute the ideal generated by products `xy`, for `x` an element
        of ``self`` and `y \in J`.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x, y).ideal(x).product(QQ.old_poly_ring(x, y).ideal(y))
        <x*y>
        """
        self._check_ideal(J)
        return self._product(J)

    def reduce_element(self, x):
        """
        Reduce the element ``x`` of our ring modulo the ideal ``self``.

        Here "reduce" has no specific meaning: it could return a unique normal
        form, simplify the expression a bit, or just do nothing.
        """
        return x

    def __add__(self, e):
        if not isinstance(e, Ideal):
            R = self.ring.quotient_ring(self)
            if isinstance(e, R.dtype):
                return e
            if isinstance(e, R.ring.dtype):
                return R(e)
            return R.convert(e)
        self._check_ideal(e)
        return self.union(e)

    __radd__ = __add__

    def __mul__(self, e):
        if not isinstance(e, Ideal):
            try:
                e = self.ring.ideal(e)
            except CoercionFailed:
                return NotImplemented
        self._check_ideal(e)
        return self.product(e)

    __rmul__ = __mul__

    def _zeroth_power(self):
        return self.ring.ideal(1)

    def _first_power(self):
        # Raising to any power but 1 returns a new instance. So we mult by 1
        # here so that the first power is no exception.
        return self * 1

    def __eq__(self, e):
        if not isinstance(e, Ideal) or e.ring != self.ring:
            return False
        return self._equals(e)

    def __ne__(self, e):
        return not (self == e)


class ModuleImplementedIdeal(Ideal):
    """
    Ideal implementation relying on the modules code.

    Attributes:

    - _module - the underlying module
    """

    def __init__(self, ring, module):
        Ideal.__init__(self, ring)
        self._module = module

    def _contains_elem(self, x):
        return self._module.contains([x])

    def _contains_ideal(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.is_submodule(J._module)

    def _intersect(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.intersect(J._module))

    def _quotient(self, J, **opts):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self._module.module_quotient(J._module, **opts)

    def _union(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.union(J._module))

    @property
    def gens(self):
        """
        Return generators for ``self``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> list(QQ.old_poly_ring(x, y).ideal(x, y, x**2 + y).gens)
        [DMP_Python([[1], []], QQ), DMP_Python([[1, 0]], QQ), DMP_Python([[1], [], [1, 0]], QQ)]
        """
        return (x[0] for x in self._module.gens)

    def is_zero(self):
        """
        Return True if ``self`` is the zero ideal.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).ideal(x).is_zero()
        False
        >>> QQ.old_poly_ring(x).ideal().is_zero()
        True
        """
        return self._module.is_zero()

    def is_whole_ring(self):
        """
        Return True if ``self`` is the whole ring, i.e. one generator is a unit.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ, ilex
        >>> QQ.old_poly_ring(x).ideal(x).is_whole_ring()
        False
        >>> QQ.old_poly_ring(x).ideal(3).is_whole_ring()
        True
        >>> QQ.old_poly_ring(x, order=ilex).ideal(2 + x).is_whole_ring()
        True
        """
        return self._module.is_full_module()

    def __repr__(self):
        from sympy.printing.str import sstr
        gens = [self.ring.to_sympy(x) for [x] in self._module.gens]
        return '<' + ','.join(sstr(g) for g in gens) + '>'

    # NOTE this is the only method using the fact that the module is a SubModule
    def _product(self, J):
        if not isinstance(J, ModuleImplementedIdeal):
            raise NotImplementedError
        return self.__class__(self.ring, self._module.submodule(
            *[[x*y] for [x] in self._module.gens for [y] in J._module.gens]))

    def in_terms_of_generators(self, e):
        """
        Express ``e`` in terms of the generators of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2 + 1, x)
        >>> I.in_terms_of_generators(1)  # doctest: +SKIP
        [DMP_Python([1], QQ), DMP_Python([-1, 0], QQ)]
        """
        return self._module.in_terms_of_generators([e])

    def reduce_element(self, x, **options):
        return self._module.reduce_element([x], **options)[0]
<|endoftext|>"""
Computations with homomorphisms of modules and rings.

This module implements classes for representing homomorphisms of rings and
their modules. Instead of instantiating the classes directly, you should use
the function ``homomorphism(from, to, matrix)`` to create homomorphism objects.
"""


from sympy.polys.agca.modules import (Module, FreeModule, QuotientModule,
    SubModule, SubQuotientModule)
from sympy.polys.polyerrors import CoercionFailed

# The main computational task for module homomorphisms is kernels.
# For this reason, the concrete classes are organised by domain module type.


class ModuleHomomorphism:
    """
    Abstract base class for module homomoprhisms. Do not instantiate.

    Instead, use the ``homomorphism`` function:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])

    Attributes:

    - ring - the ring over which we are considering modules
    - domain - the domain module
    - codomain - the codomain module
    - _ker - cached kernel
    - _img - cached image

    Non-implemented methods:

    - _kernel
    - _image
    - _restrict_domain
    - _restrict_codomain
    - _quotient_domain
    - _quotient_codomain
    - _apply
    - _mul_scalar
    - _compose
    - _add
    """

    def __init__(self, domain, codomain):
        if not isinstance(domain, Module):
            raise TypeError('Source must be a module, got %s' % domain)
        if not isinstance(codomain, Module):
            raise TypeError('Target must be a module, got %s' % codomain)
        if domain.ring != codomain.ring:
            raise ValueError('Source and codomain must be over same ring, '
                             'got %s != %s' % (domain, codomain))
        self.domain = domain
        self.codomain = codomain
        self.ring = domain.ring
        self._ker = None
        self._img = None

    def kernel(self):
        r"""
        Compute the kernel of ``self``.

        That is, if ``self`` is the homomorphism `\phi: M \to N`, then compute
        `ker(\phi) = \{x \in M | \phi(x) = 0\}`.  This is a submodule of `M`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> homomorphism(F, F, [[1, 0], [x, 0]]).kernel()
        <[x, -1]>
        """
        if self._ker is None:
            self._ker = self._kernel()
        return self._ker

    def image(self):
        r"""
        Compute the image of ``self``.

        That is, if ``self`` is the homomorphism `\phi: M \to N`, then compute
        `im(\phi) = \{\phi(x) | x \in M \}`.  This is a submodule of `N`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> homomorphism(F, F, [[1, 0], [x, 0]]).image() == F.submodule([1, 0])
        True
        """
        if self._img is None:
            self._img = self._image()
        return self._img

    def _kernel(self):
        """Compute the kernel of ``self``."""
        raise NotImplementedError

    def _image(self):
        """Compute the image of ``self``."""
        raise NotImplementedError

    def _restrict_domain(self, sm):
        """Implementation of domain restriction."""
        raise NotImplementedError

    def _restrict_codomain(self, sm):
        """Implementation of codomain restriction."""
        raise NotImplementedError

    def _quotient_domain(self, sm):
        """Implementation of domain quotient."""
        raise NotImplementedError

    def _quotient_codomain(self, sm):
        """Implementation of codomain quotient."""
        raise NotImplementedError

    def restrict_domain(self, sm):
        """
        Return ``self``, with the domain restricted to ``sm``.

        Here ``sm`` has to be a submodule of ``self.domain``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.restrict_domain(F.submodule([1, 0]))
        Matrix([
        [1, x], : <[1, 0]> -> QQ[x]**2
        [0, 0]])

        This is the same as just composing on the right with the submodule
        inclusion:

        >>> h * F.submodule([1, 0]).inclusion_hom()
        Matrix([
        [1, x], : <[1, 0]> -> QQ[x]**2
        [0, 0]])
        """
        if not self.domain.is_submodule(sm):
            raise ValueError('sm must be a submodule of %s, got %s'
                             % (self.domain, sm))
        if sm == self.domain:
            return self
        return self._restrict_domain(sm)

    def restrict_codomain(self, sm):
        """
        Return ``self``, with codomain restricted to to ``sm``.

        Here ``sm`` has to be a submodule of ``self.codomain`` containing the
        image.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.restrict_codomain(F.submodule([1, 0]))
        Matrix([
        [1, x], : QQ[x]**2 -> <[1, 0]>
        [0, 0]])
        """
        if not sm.is_submodule(self.image()):
            raise ValueError('the image %s must contain sm, got %s'
                             % (self.image(), sm))
        if sm == self.codomain:
            return self
        return self._restrict_codomain(sm)

    def quotient_domain(self, sm):
        """
        Return ``self`` with domain replaced by ``domain/sm``.

        Here ``sm`` must be a submodule of ``self.kernel()``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.quotient_domain(F.submodule([-x, 1]))
        Matrix([
        [1, x], : QQ[x]**2/<[-x, 1]> -> QQ[x]**2
        [0, 0]])
        """
        if not self.kernel().is_submodule(sm):
            raise ValueError('kernel %s must contain sm, got %s' %
                             (self.kernel(), sm))
        if sm.is_zero():
            return self
        return self._quotient_domain(sm)

    def quotient_codomain(self, sm):
        """
        Return ``self`` with codomain replaced by ``codomain/sm``.

        Here ``sm`` must be a submodule of ``self.codomain``.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2
        [0, 0]])
        >>> h.quotient_codomain(F.submodule([1, 1]))
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2/<[1, 1]>
        [0, 0]])

        This is the same as composing with the quotient map on the left:

        >>> (F/[(1, 1)]).quotient_hom() * h
        Matrix([
        [1, x], : QQ[x]**2 -> QQ[x]**2/<[1, 1]>
        [0, 0]])
        """
        if not self.codomain.is_submodule(sm):
            raise ValueError('sm must be a submodule of codomain %s, got %s'
                             % (self.codomain, sm))
        if sm.is_zero():
            return self
        return self._quotient_codomain(sm)

    def _apply(self, elem):
        """Apply ``self`` to ``elem``."""
        raise NotImplementedError

    def __call__(self, elem):
        return self.codomain.convert(self._apply(self.domain.convert(elem)))

    def _compose(self, oth):
        """
        Compose ``self`` with ``oth``, that is, return the homomorphism
        obtained by first applying then ``self``, then ``oth``.

        (This method is private since in this syntax, it is non-obvious which
        homomorphism is executed first.)
        """
        raise NotImplementedError

    def _mul_scalar(self, c):
        """Scalar multiplication. ``c`` is guaranteed in self.ring."""
        raise NotImplementedError

    def _add(self, oth):
        """
        Homomorphism addition.
        ``oth`` is guaranteed to be a homomorphism with same domain/codomain.
        """
        raise NotImplementedError

    def _check_hom(self, oth):
        """Helper to check that oth is a homomorphism with same domain/codomain."""
        if not isinstance(oth, ModuleHomomorphism):
            return False
        return oth.domain == self.domain and oth.codomain == self.codomain

    def __mul__(self, oth):
        if isinstance(oth, ModuleHomomorphism) and self.domain == oth.codomain:
            return oth._compose(self)
        try:
            return self._mul_scalar(self.ring.convert(oth))
        except CoercionFailed:
            return NotImplemented

    # NOTE: _compose will never be called from rmul
    __rmul__ = __mul__

    def __truediv__(self, oth):
        try:
            return self._mul_scalar(1/self.ring.convert(oth))
        except CoercionFailed:
            return NotImplemented

    def __add__(self, oth):
        if self._check_hom(oth):
            return self._add(oth)
        return NotImplemented

    def __sub__(self, oth):
        if self._check_hom(oth):
            return self._add(oth._mul_scalar(self.ring.convert(-1)))
        return NotImplemented

    def is_injective(self):
        """
        Return True if ``self`` is injective.

        That is, check if the elements of the domain are mapped to the same
        codomain element.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_injective()
        False
        >>> h.quotient_domain(h.kernel()).is_injective()
        True
        """
        return self.kernel().is_zero()

    def is_surjective(self):
        """
        Return True if ``self`` is surjective.

        That is, check if every element of the codomain has at least one
        preimage.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_surjective()
        False
        >>> h.restrict_codomain(h.image()).is_surjective()
        True
        """
        return self.image() == self.codomain

    def is_isomorphism(self):
        """
        Return True if ``self`` is an isomorphism.

        That is, check if every element of the codomain has precisely one
        preimage. Equivalently, ``self`` is both injective and surjective.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h = h.restrict_codomain(h.image())
        >>> h.is_isomorphism()
        False
        >>> h.quotient_domain(h.kernel()).is_isomorphism()
        True
        """
        return self.is_injective() and self.is_surjective()

    def is_zero(self):
        """
        Return True if ``self`` is a zero morphism.

        That is, check if every element of the domain is mapped to zero
        under self.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x
        >>> from sympy.polys.agca import homomorphism

        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> h = homomorphism(F, F, [[1, 0], [x, 0]])
        >>> h.is_zero()
        False
        >>> h.restrict_domain(F.submodule()).is_zero()
        True
        >>> h.quotient_codomain(h.image()).is_zero()
        True
        """
        return self.image().is_zero()

    def __eq__(self, oth):
        try:
            return (self - oth).is_zero()
        except TypeError:
            return False

    def __ne__(self, oth):
        return not (self == oth)


class MatrixHomomorphism(ModuleHomomorphism):
    r"""
    Helper class for all homomoprhisms which are expressed via a matrix.

    That is, for such homomorphisms ``domain`` is contained in a module
    generated by finitely many elements `e_1, \ldots, e_n`, so that the
    homomorphism is determined uniquely by its action on the `e_i`. It
    can thus be represented as a vector of elements of the codomain module,
    or potentially a supermodule of the codomain module
    (and hence conventionally as a matrix, if there is a similar interpretation
    for elements of the codomain module).

    Note that this class does *not* assume that the `e_i` freely generate a
    submodule, nor that ``domain`` is even all of this submodule. It exists
    only to unify the interface.

    Do not instantiate.

    Attributes:

    - matrix - the list of images determining the homomorphism.
    NOTE: the elements of matrix belong to either self.codomain or
          self.codomain.container

    Still non-implemented methods:

    - kernel
    - _apply
    """

    def __init__(self, domain, codomain, matrix):
        ModuleHomomorphism.__init__(self, domain, codomain)
        if len(matrix) != domain.rank:
            raise ValueError('Need to provide %s elements, got %s'
                             % (domain.rank, len(matrix)))

        converter = self.codomain.convert
        if isinstance(self.codomain, (SubModule, SubQuotientModule)):
            converter = self.codomain.container.convert
        self.matrix = tuple(converter(x) for x in matrix)

    def _sympy_matrix(self):
        """Helper function which returns a SymPy matrix ``self.matrix``."""
        from sympy.matrices import Matrix
        c = lambda x: x
        if isinstance(self.codomain, (QuotientModule, SubQuotientModule)):
            c = lambda x: x.data
        return Matrix([[self.ring.to_sympy(y) for y in c(x)] for x in self.matrix]).T

    def __repr__(self):
        lines = repr(self._sympy_matrix()).split('\n')
        t = " : %s -> %s" % (self.domain, self.codomain)
        s = ' '*len(t)
        n = len(lines)
        for i in range(n // 2):
            lines[i] += s
        lines[n // 2] += t
        for i in range(n//2 + 1, n):
            lines[i] += s
        return '\n'.join(lines)

    def _restrict_domain(self, sm):
        """Implementation of domain restriction."""
        return SubModuleHomomorphism(sm, self.codomain, self.matrix)

    def _restrict_codomain(self, sm):
        """Implementation of codomain restriction."""
        return self.__class__(self.domain, sm, self.matrix)

    def _quotient_domain(self, sm):
        """Implementation of domain quotient."""
        return self.__class__(self.domain/sm, self.codomain, self.matrix)

    def _quotient_codomain(self, sm):
        """Implementation of codomain quotient."""
        Q = self.codomain/sm
        converter = Q.convert
        if isinstance(self.codomain, SubModule):
            converter = Q.container.convert
        return self.__class__(self.domain, self.codomain/sm,
            [converter(x) for x in self.matrix])

    def _add(self, oth):
        return self.__class__(self.domain, self.codomain,
                              [x + y for x, y in zip(self.matrix, oth.matrix)])

    def _mul_scalar(self, c):
        return self.__class__(self.domain, self.codomain, [c*x for x in self.matrix])

    def _compose(self, oth):
        return self.__class__(self.domain, oth.codomain, [oth(x) for x in self.matrix])


class FreeModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphisms with domain a free module or a quotient
    thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> homomorphism(F, F, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : QQ[x]**2 -> QQ[x]**2
    [0, 1]])
    """

    def _apply(self, elem):
        if isinstance(self.domain, QuotientModule):
            elem = elem.data
        return sum(x * e for x, e in zip(elem, self.matrix))

    def _image(self):
        return self.codomain.submodule(*self.matrix)

    def _kernel(self):
        # The domain is either a free module or a quotient thereof.
        # It does not matter if it is a quotient, because that won't increase
        # the kernel.
        # Our generators {e_i} are sent to the matrix entries {b_i}.
        # The kernel is essentially the syzygy module of these {b_i}.
        syz = self.image().syzygy_module()
        return self.domain.submodule(*syz.gens)


class SubModuleHomomorphism(MatrixHomomorphism):
    """
    Concrete class for homomorphism with domain a submodule of a free module
    or a quotient thereof.

    Do not instantiate; the constructor does not check that your data is well
    defined. Use the ``homomorphism`` function instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> M = QQ.old_poly_ring(x).free_module(2)*x
    >>> homomorphism(M, M, [[1, 0], [0, 1]])
    Matrix([
    [1, 0], : <[x, 0], [0, x]> -> <[x, 0], [0, x]>
    [0, 1]])
    """

    def _apply(self, elem):
        if isinstance(self.domain, SubQuotientModule):
            elem = elem.data
        return sum(x * e for x, e in zip(elem, self.matrix))

    def _image(self):
        return self.codomain.submodule(*[self(x) for x in self.domain.gens])

    def _kernel(self):
        syz = self.image().syzygy_module()
        return self.domain.submodule(
            *[sum(xi*gi for xi, gi in zip(s, self.domain.gens))
              for s in syz.gens])


def homomorphism(domain, codomain, matrix):
    r"""
    Create a homomorphism object.

    This function tries to build a homomorphism from ``domain`` to ``codomain``
    via the matrix ``matrix``.

    Examples
    ========

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.agca import homomorphism

    >>> R = QQ.old_poly_ring(x)
    >>> T = R.free_module(2)

    If ``domain`` is a free module generated by `e_1, \ldots, e_n`, then
    ``matrix`` should be an n-element iterable `(b_1, \ldots, b_n)` where
    the `b_i` are elements of ``codomain``. The constructed homomorphism is the
    unique homomorphism sending `e_i` to `b_i`.

    >>> F = R.free_module(2)
    >>> h = homomorphism(F, T, [[1, x], [x**2, 0]])
    >>> h
    Matrix([
    [1, x**2], : QQ[x]**2 -> QQ[x]**2
    [x,    0]])
    >>> h([1, 0])
    [1, x]
    >>> h([0, 1])
    [x**2, 0]
    >>> h([1, 1])
    [x**2 + 1, x]

    If ``domain`` is a submodule of a free module, them ``matrix`` determines
    a homomoprhism from the containing free module to ``codomain``, and the
    homomorphism returned is obtained by restriction to ``domain``.

    >>> S = F.submodule([1, 0], [0, x])
    >>> homomorphism(S, T, [[1, x], [x**2, 0]])
    Matrix([
    [1, x**2], : <[1, 0], [0, x]> -> QQ[x]**2
    [x,    0]])

    If ``domain`` is a (sub)quotient `N/K`, then ``matrix`` determines a
    homomorphism from `N` to ``codomain``. If the kernel contains `K`, this
    homomorphism descends to ``domain`` and is returned; otherwise an exception
    is raised.

    >>> homomorphism(S/[(1, 0)], T, [0, [x**2, 0]])
    Matrix([
    [0, x**2], : <[1, 0] + <[1, 0]>, [0, x] + <[1, 0]>, [1, 0] + <[1, 0]>> -> QQ[x]**2
    [0,    0]])
    >>> homomorphism(S/[(0, x)], T, [0, [x**2, 0]])
    Traceback (most recent call last):
    ...
    ValueError: kernel <[1, 0], [0, 0]> must contain sm, got <[0,x]>

    """
    def freepres(module):
        """
        Return a tuple ``(F, S, Q, c)`` where ``F`` is a free module, ``S`` is a
        submodule of ``F``, and ``Q`` a submodule of ``S``, such that
        ``module = S/Q``, and ``c`` is a conversion function.
        """
        if isinstance(module, FreeModule):
            return module, module, module.submodule(), lambda x: module.convert(x)
        if isinstance(module, QuotientModule):
            return (module.base, module.base, module.killed_module,
                    lambda x: module.convert(x).data)
        if isinstance(module, SubQuotientModule):
            return (module.base.container, module.base, module.killed_module,
                    lambda x: module.container.convert(x).data)
        # an ordinary submodule
        return (module.container, module, module.submodule(),
                lambda x: module.container.convert(x))

    SF, SS, SQ, _ = freepres(domain)
    TF, TS, TQ, c = freepres(codomain)
    # NOTE this is probably a bit inefficient (redundant checks)
    return FreeModuleHomomorphism(SF, TF, [c(x) for x in matrix]
         ).restrict_domain(SS).restrict_codomain(TS
         ).quotient_codomain(TQ).quotient_domain(SQ)
<|endoftext|>"""
Computations with modules over polynomial rings.

This module implements various classes that encapsulate groebner basis
computations for modules. Most of them should not be instantiated by hand.
Instead, use the constructing routines on objects you already have.

For example, to construct a free module over ``QQ[x, y]``, call
``QQ[x, y].free_module(rank)`` instead of the ``FreeModule`` constructor.
In fact ``FreeModule`` is an abstract base class that should not be
instantiated, the ``free_module`` method instead returns the implementing class
``FreeModulePolyRing``.

In general, the abstract base classes implement most functionality in terms of
a few non-implemented methods. The concrete base classes supply only these
non-implemented methods. They may also supply new implementations of the
convenience methods, for example if there are faster algorithms available.
"""


from copy import copy
from functools import reduce

from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable

# TODO
# - module saturation
# - module quotient/intersection for quotient rings
# - free resoltutions / syzygies
# - finding small/minimal generating sets
# - ...

##########################################################################
## Abstract base classes #################################################
##########################################################################


class Module:
    """
    Abstract base class for modules.

    Do not instantiate - use ring explicit constructors instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> QQ.old_poly_ring(x).free_module(2)
    QQ[x]**2

    Attributes:

    - dtype - type of elements
    - ring - containing ring

    Non-implemented methods:

    - submodule
    - quotient_module
    - is_zero
    - is_submodule
    - multiply_ideal

    The method convert likely needs to be changed in subclasses.
    """

    def __init__(self, ring):
        self.ring = ring

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into internal representation of this module.

        If ``M`` is not None, it should be a module containing it.
        """
        if not isinstance(elem, self.dtype):
            raise CoercionFailed
        return elem

    def submodule(self, *gens):
        """Generate a submodule."""
        raise NotImplementedError

    def quotient_module(self, other):
        """Generate a quotient module."""
        raise NotImplementedError

    def __truediv__(self, e):
        if not isinstance(e, Module):
            e = self.submodule(*e)
        return self.quotient_module(e)

    def contains(self, elem):
        """Return True if ``elem`` is an element of this module."""
        try:
            self.convert(elem)
            return True
        except CoercionFailed:
            return False

    def __contains__(self, elem):
        return self.contains(elem)

    def subset(self, other):
        """
        Returns True if ``other`` is is a subset of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.subset([(1, x), (x, 2)])
        True
        >>> F.subset([(1/x, x), (x, 2)])
        False
        """
        return all(self.contains(x) for x in other)

    def __eq__(self, other):
        return self.is_submodule(other) and other.is_submodule(self)

    def __ne__(self, other):
        return not (self == other)

    def is_zero(self):
        """Returns True if ``self`` is a zero module."""
        raise NotImplementedError

    def is_submodule(self, other):
        """Returns True if ``other`` is a submodule of ``self``."""
        raise NotImplementedError

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.
        """
        raise NotImplementedError

    def __mul__(self, e):
        if not isinstance(e, Ideal):
            try:
                e = self.ring.ideal(e)
            except (CoercionFailed, NotImplementedError):
                return NotImplemented
        return self.multiply_ideal(e)

    __rmul__ = __mul__

    def identity_hom(self):
        """Return the identity homomorphism on ``self``."""
        raise NotImplementedError


class ModuleElement:
    """
    Base class for module element wrappers.

    Use this class to wrap primitive data types as module elements. It stores
    a reference to the containing module, and implements all the arithmetic
    operators.

    Attributes:

    - module - containing module
    - data - internal data

    Methods that likely need change in subclasses:

    - add
    - mul
    - div
    - eq
    """

    def __init__(self, module, data):
        self.module = module
        self.data = data

    def add(self, d1, d2):
        """Add data ``d1`` and ``d2``."""
        return d1 + d2

    def mul(self, m, d):
        """Multiply module data ``m`` by coefficient d."""
        return m * d

    def div(self, m, d):
        """Divide module data ``m`` by coefficient d."""
        return m / d

    def eq(self, d1, d2):
        """Return true if d1 and d2 represent the same element."""
        return d1 == d2

    def __add__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.add(self.data, om.data))

    __radd__ = __add__

    def __neg__(self):
        return self.__class__(self.module, self.mul(self.data,
                       self.module.ring.convert(-1)))

    def __sub__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return NotImplemented
        return self.__add__(-om)

    def __rsub__(self, om):
        return (-self).__add__(om)

    def __mul__(self, o):
        if not isinstance(o, self.module.ring.dtype):
            try:
                o = self.module.ring.convert(o)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.mul(self.data, o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if not isinstance(o, self.module.ring.dtype):
            try:
                o = self.module.ring.convert(o)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.div(self.data, o))

    def __eq__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return False
        return self.eq(self.data, om.data)

    def __ne__(self, om):
        return not self == om

##########################################################################
## Free Modules ##########################################################
##########################################################################


class FreeModuleElement(ModuleElement):
    """Element of a free module. Data stored as a tuple."""

    def add(self, d1, d2):
        return tuple(x + y for x, y in zip(d1, d2))

    def mul(self, d, p):
        return tuple(x * p for x in d)

    def div(self, d, p):
        return tuple(x / p for x in d)

    def __repr__(self):
        from sympy.printing.str import sstr
        data = self.data
        if any(isinstance(x, DMP) for x in data):
            data = [self.module.ring.to_sympy(x) for x in data]
        return '[' + ', '.join(sstr(x) for x in data) + ']'

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, idx):
        return self.data[idx]


class FreeModule(Module):
    """
    Abstract base class for free modules.

    Additional attributes:

    - rank - rank of the free module

    Non-implemented methods:

    - submodule
    """

    dtype = FreeModuleElement

    def __init__(self, ring, rank):
        Module.__init__(self, ring)
        self.rank = rank

    def __repr__(self):
        return repr(self.ring) + "**" + repr(self.rank)

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> F.is_submodule(F)
        True
        >>> F.is_submodule(M)
        True
        >>> M.is_submodule(F)
        False
        """
        if isinstance(other, SubModule):
            return other.container == self
        if isinstance(other, FreeModule):
            return other.ring == self.ring and other.rank == self.rank
        return False

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.convert([1, 0])
        [1, 0]
        """
        if isinstance(elem, FreeModuleElement):
            if elem.module is self:
                return elem
            if elem.module.rank != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self,
                     tuple(self.ring.convert(x, elem.module.ring) for x in elem.data))
        elif iterable(elem):
            tpl = tuple(self.ring.convert(x) for x in elem)
            if len(tpl) != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self, tpl)
        elif _aresame(elem, 0):
            return FreeModuleElement(self, (self.ring.convert(0),)*self.rank)
        else:
            raise CoercionFailed

    def is_zero(self):
        """
        Returns True if ``self`` is a zero module.

        (If, as this implementation assumes, the coefficient ring is not the
        zero ring, then this is equivalent to the rank being zero.)

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(0).is_zero()
        True
        >>> QQ.old_poly_ring(x).free_module(1).is_zero()
        False
        """
        return self.rank == 0

    def basis(self):
        """
        Return a set of basis elements.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(3).basis()
        ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        """
        from sympy.matrices import eye
        M = eye(self.rank)
        return tuple(self.convert(M.row(i)) for i in range(self.rank))

    def quotient_module(self, submodule):
        """
        Return a quotient module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2)
        >>> M.quotient_module(M.submodule([1, x], [x, 2]))
        QQ[x]**2/<[1, x], [x, 2]>

        Or more conicisely, using the overloaded division operator:

        >>> QQ.old_poly_ring(x).free_module(2) / [[1, x], [x, 2]]
        QQ[x]**2/<[1, x], [x, 2]>
        """
        return QuotientModule(self.ring, self, submodule)

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x)
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.multiply_ideal(I)
        <[x, 0], [0, x]>
        """
        return self.submodule(*self.basis()).multiply_ideal(other)

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2
        [0, 1]])
        """
        from sympy.polys.agca.homomorphisms import homomorphism
        return homomorphism(self, self, self.basis())


class FreeModulePolyRing(FreeModule):
    """
    Free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of the ring instead:

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(3)
    >>> F
    QQ[x]**3
    >>> F.contains([x, 1, 0])
    True
    >>> F.contains([1/x, 0, 1])
    False
    """

    def __init__(self, ring, rank):
        from sympy.polys.domains.old_polynomialring import PolynomialRingBase
        FreeModule.__init__(self, ring, rank)
        if not isinstance(ring, PolynomialRingBase):
            raise NotImplementedError('This implementation only works over '
                                      + 'polynomial rings, got %s' % ring)
        if not isinstance(ring.dom, Field):
            raise NotImplementedError('Ground domain must be a field, '
                                      + 'got %s' % ring.dom)

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x, y).free_module(2).submodule([x, x + y])
        >>> M
        <[x, x + y]>
        >>> M.contains([2*x, 2*x + 2*y])
        True
        >>> M.contains([x, y])
        False
        """
        return SubModulePolyRing(gens, self, **opts)


class FreeModuleQuotientRing(FreeModule):
    """
    Free module over a quotient ring.

    Do not instantiate this, use the constructor method of the ring instead:

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(3)
    >>> F
    (QQ[x]/<x**2 + 1>)**3

    Attributes

    - quot - the quotient module `R^n / IR^n`, where `R/I` is our ring
    """

    def __init__(self, ring, rank):
        from sympy.polys.domains.quotientring import QuotientRing
        FreeModule.__init__(self, ring, rank)
        if not isinstance(ring, QuotientRing):
            raise NotImplementedError('This implementation only works over '
                             + 'quotient rings, got %s' % ring)
        F = self.ring.ring.free_module(self.rank)
        self.quot = F / (self.ring.base_ideal*F)

    def __repr__(self):
        return "(" + repr(self.ring) + ")" + "**" + repr(self.rank)

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x, y)/[x**2 - y**2]).free_module(2).submodule([x, x + y])
        >>> M
        <[x + <x**2 - y**2>, x + y + <x**2 - y**2>]>
        >>> M.contains([y**2, x**2 + x*y])
        True
        >>> M.contains([x, y])
        False
        """
        return SubModuleQuotientRing(gens, self, **opts)

    def lift(self, elem):
        """
        Lift the element ``elem`` of self to the module self.quot.

        Note that self.quot is the same set as self, just as an R-module
        and not as an R/I-module, so this makes sense.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        >>> e = F.convert([1, 0])
        >>> e
        [1 + <x**2 + 1>, 0 + <x**2 + 1>]
        >>> L = F.quot
        >>> l = F.lift(e)
        >>> l
        [1, 0] + <[x**2 + 1, 0], [0, x**2 + 1]>
        >>> L.contains(l)
        True
        """
        return self.quot.convert([x.data for x in elem])

    def unlift(self, elem):
        """
        Push down an element of self.quot to self.

        This undoes ``lift``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        >>> e = F.convert([1, 0])
        >>> l = F.lift(e)
        >>> e == l
        False
        >>> e == F.unlift(l)
        True
        """
        return self.convert(elem.data)

##########################################################################
## Submodules and subquotients ###########################################
##########################################################################


class SubModule(Module):
    """
    Base class for submodules.

    Attributes:

    - container - containing module
    - gens - generators (subset of containing module)
    - rank - rank of containing module

    Non-implemented methods:

    - _contains
    - _syzygies
    - _in_terms_of_generators
    - _intersect
    - _module_quotient

    Methods that likely need change in subclasses:

    - reduce_element
    """

    def __init__(self, gens, container):
        Module.__init__(self, container.ring)
        self.gens = tuple(container.convert(x) for x in gens)
        self.container = container
        self.rank = container.rank
        self.ring = container.ring
        self.dtype = container.dtype

    def __repr__(self):
        return "<" + ", ".join(repr(x) for x in self.gens) + ">"

    def _contains(self, other):
        """Implementation of containment.
           Other is guaranteed to be FreeModuleElement."""
        raise NotImplementedError

    def _syzygies(self):
        """Implementation of syzygy computation wrt self generators."""
        raise NotImplementedError

    def _in_terms_of_generators(self, e):
        """Implementation of expression in terms of generators."""
        raise NotImplementedError

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal represantition.

        Mostly called implicitly.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([1, x])
        >>> M.convert([2, 2*x])
        [2, 2*x]
        """
        if isinstance(elem, self.container.dtype) and elem.module is self:
            return elem
        r = copy(self.container.convert(elem, M))
        r.module = self
        if not self._contains(r):
            raise CoercionFailed
        return r

    def _intersect(self, other):
        """Implementation of intersection.
           Other is guaranteed to be a submodule of same free module."""
        raise NotImplementedError

    def _module_quotient(self, other):
        """Implementation of quotient.
           Other is guaranteed to be a submodule of same free module."""
        raise NotImplementedError

    def intersect(self, other, **options):
        """
        Returns the intersection of ``self`` with submodule ``other``.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x, y).free_module(2)
        >>> F.submodule([x, x]).intersect(F.submodule([y, y]))
        <[x*y, x*y]>

        Some implementation allow further options to be passed. Currently, to
        only one implemented is ``relations=True``, in which case the function
        will return a triple ``(res, rela, relb)``, where ``res`` is the
        intersection module, and ``rela`` and ``relb`` are lists of coefficient
        vectors, expressing the generators of ``res`` in terms of the
        generators of ``self`` (``rela``) and ``other`` (``relb``).

        >>> F.submodule([x, x]).intersect(F.submodule([y, y]), relations=True)
        (<[x*y, x*y]>, [(DMP_Python([[1, 0]], QQ),)], [(DMP_Python([[1], []], QQ),)])

        The above result says: the intersection module is generated by the
        single element `(-xy, -xy) = -y (x, x) = -x (y, y)`, where
        `(x, x)` and `(y, y)` respectively are the unique generators of
        the two modules being intersected.
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self._intersect(other, **options)

    def module_quotient(self, other, **options):
        r"""
        Returns the module quotient of ``self`` by submodule ``other``.

        That is, if ``self`` is the module `M` and ``other`` is `N`, then
        return the ideal `\{f \in R | fN \subset M\}`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> F = QQ.old_poly_ring(x, y).free_module(2)
        >>> S = F.submodule([x*y, x*y])
        >>> T = F.submodule([x, x])
        >>> S.module_quotient(T)
        <y>

        Some implementations allow further options to be passed. Currently, the
        only one implemented is ``relations=True``, which may only be passed
        if ``other`` is principal. In this case the function
        will return a pair ``(res, rel)`` where ``res`` is the ideal, and
        ``rel`` is a list of coefficient vectors, expressing the generators of
        the ideal, multiplied by the generator of ``other`` in terms of
        generators of ``self``.

        >>> S.module_quotient(T, relations=True)
        (<y>, [[DMP_Python([[1]], QQ)]])

        This means that the quotient ideal is generated by the single element
        `y`, and that `y (x, x) = 1 (xy, xy)`, `(x, x)` and `(xy, xy)` being
        the generators of `T` and `S`, respectively.
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self._module_quotient(other, **options)

    def union(self, other):
        """
        Returns the module generated by the union of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(1)
        >>> M = F.submodule([x**2 + x]) # <x(x+1)>
        >>> N = F.submodule([x**2 - 1]) # <(x-1)(x+1)>
        >>> M.union(N) == F.submodule([x+1])
        True
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self.__class__(self.gens + other.gens, self.container)

    def is_zero(self):
        """
        Return True if ``self`` is a zero module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_zero()
        False
        >>> F.submodule([0, 0]).is_zero()
        True
        """
        return all(x == 0 for x in self.gens)

    def submodule(self, *gens):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([x, 1])
        >>> M.submodule([x**2, x])
        <[x**2, x]>
        """
        if not self.subset(gens):
            raise ValueError('%s not a subset of %s' % (gens, self))
        return self.__class__(gens, self.container)

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        return all(self.contains(x) for x in self.container.basis())

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> N = M.submodule([2*x, x**2])
        >>> M.is_submodule(M)
        True
        >>> M.is_submodule(N)
        True
        >>> N.is_submodule(M)
        False
        """
        if isinstance(other, SubModule):
            return self.container == other.container and \
                all(self.contains(x) for x in other.gens)
        if isinstance(other, (FreeModule, QuotientModule)):
            return self.container == other and self.is_full_module()
        return False

    def syzygy_module(self, **opts):
        r"""
        Compute the syzygy module of the generators of ``self``.

        Suppose `M` is generated by `f_1, \ldots, f_n` over the ring
        `R`. Consider the homomorphism `\phi: R^n \to M`, given by
        sending `(r_1, \ldots, r_n) \to r_1 f_1 + \cdots + r_n f_n`.
        The syzygy module is defined to be the kernel of `\phi`.

        Examples
        ========

        The syzygy module is zero iff the generators generate freely a free
        submodule:

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([1, 0], [1, 1]).syzygy_module().is_zero()
        True

        A slightly more interesting example:

        >>> M = QQ.old_poly_ring(x, y).free_module(2).submodule([x, 2*x], [y, 2*y])
        >>> S = QQ.old_poly_ring(x, y).free_module(2).submodule([y, -x])
        >>> M.syzygy_module() == S
        True
        """
        F = self.ring.free_module(len(self.gens))
        # NOTE we filter out zero syzygies. This is for convenience of the
        # _syzygies function and not meant to replace any real "generating set
        # reduction" algorithm
        return F.submodule(*[x for x in self._syzygies() if F.convert(x) != 0],
                           **opts)

    def in_terms_of_generators(self, e):
        """
        Express element ``e`` of ``self`` in terms of the generators.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([1, 0], [1, 1])
        >>> M.in_terms_of_generators([x, x**2])  # doctest: +SKIP
        [DMP_Python([-1, 1, 0], QQ), DMP_Python([1, 0, 0], QQ)]
        """
        try:
            e = self.convert(e)
        except CoercionFailed:
            raise ValueError('%s is not an element of %s' % (e, self))
        return self._in_terms_of_generators(e)

    def reduce_element(self, x):
        """
        Reduce the element ``x`` of our ring modulo the ideal ``self``.

        Here "reduce" has no specific meaning, it could return a unique normal
        form, simplify the expression a bit, or just do nothing.
        """
        return x

    def quotient_module(self, other, **opts):
        """
        Return a quotient module.

        This is the same as taking a submodule of a quotient of the containing
        module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> S1 = F.submodule([x, 1])
        >>> S2 = F.submodule([x**2, x])
        >>> S1.quotient_module(S2)
        <[x, 1] + <[x**2, x]>>

        Or more coincisely, using the overloaded division operator:

        >>> F.submodule([x, 1]) / [(x**2, x)]
        <[x, 1] + <[x**2, x]>>
        """
        if not self.is_submodule(other):
            raise ValueError('%s not a submodule of %s' % (other, self))
        return SubQuotientModule(self.gens,
                self.container.quotient_module(other), **opts)

    def __add__(self, oth):
        return self.container.quotient_module(self).convert(oth)

    __radd__ = __add__

    def multiply_ideal(self, I):
        """
        Multiply ``self`` by the ideal ``I``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2)
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([1, 1])
        >>> I*M
        <[x**2, x**2]>
        """
        return self.submodule(*[x*g for [x] in I._module.gens for g in self.gens])

    def inclusion_hom(self):
        """
        Return a homomorphism representing the inclusion map of ``self``.

        That is, the natural map from ``self`` to ``self.container``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([x, x]).inclusion_hom()
        Matrix([
        [1, 0], : <[x, x]> -> QQ[x]**2
        [0, 1]])
        """
        return self.container.identity_hom().restrict_domain(self)

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([x, x]).identity_hom()
        Matrix([
        [1, 0], : <[x, x]> -> <[x, x]>
        [0, 1]])
        """
        return self.container.identity_hom().restrict_domain(
            self).restrict_codomain(self)


class SubQuotientModule(SubModule):
    """
    Submodule of a quotient module.

    Equivalently, quotient module of a submodule.

    Do not instantiate this, instead use the submodule or quotient_module
    constructing methods:

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> S = F.submodule([1, 0], [1, x])
    >>> Q = F/[(1, 0)]
    >>> S/[(1, 0)] == Q.submodule([5, x])
    True

    Attributes:

    - base - base module we are quotient of
    - killed_module - submodule used to form the quotient
    """
    def __init__(self, gens, container, **opts):
        SubModule.__init__(self, gens, container)
        self.killed_module = self.container.killed_module
        # XXX it is important for some code below that the generators of base
        #     are in this particular order!
        self.base = self.container.base.submodule(
            *[x.data for x in self.gens], **opts).union(self.killed_module)

    def _contains(self, elem):
        return self.base.contains(elem.data)

    def _syzygies(self):
        # let N = self.killed_module be generated by e_1, ..., e_r
        # let F = self.base be generated by f_1, ..., f_s and e_1, ..., e_r
        # Then self = F/N.
        # Let phi: R**s --> self be the evident surjection.
        # Similarly psi: R**(s + r) --> F.
        # We need to find generators for ker(phi). Let chi: R**s --> F be the
        # evident lift of phi. For X in R**s, phi(X) = 0 iff chi(X) is
        # contained in N, iff there exists Y in R**r such that
        # psi(X, Y) = 0.
        # Hence if alpha: R**(s + r) --> R**s is the projection map, then
        # ker(phi) = alpha ker(psi).
        return [X[:len(self.gens)] for X in self.base._syzygies()]

    def _in_terms_of_generators(self, e):
        return self.base._in_terms_of_generators(e.data)[:len(self.gens)]

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        return self.base.is_full_module()

    def quotient_hom(self):
        """
        Return the quotient homomorphism to self.

        That is, return the natural map from ``self.base`` to ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x).free_module(2) / [(1, x)]).submodule([1, 0])
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : <[1, 0], [1, x]> -> <[1, 0] + <[1, x]>, [1, x] + <[1, x]>>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(self.killed_module)


_subs0 = lambda x: x[0]
_subs1 = lambda x: x[1:]


class ModuleOrder(ProductOrder):
    """A product monomial order with a zeroth term as module index."""

    def __init__(self, o1, o2, TOP):
        if TOP:
            ProductOrder.__init__(self, (o2, _subs1), (o1, _subs0))
        else:
            ProductOrder.__init__(self, (o1, _subs0), (o2, _subs1))


class SubModulePolyRing(SubModule):
    """
    Submodule of a free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of FreeModule instead:

    >>> from sympy.abc import x, y
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x, y).free_module(2)
    >>> F.submodule([x, y], [1, 0])
    <[x, y], [1, 0]>

    Attributes:

    - order - monomial order used
    """

    #self._gb - cached groebner basis
    #self._gbe - cached groebner basis relations

    def __init__(self, gens, container, order="lex", TOP=True):
        SubModule.__init__(self, gens, container)
        if not isinstance(container, FreeModulePolyRing):
            raise NotImplementedError('This implementation is for submodules of '
                             + 'FreeModulePolyRing, got %s' % container)
        self.order = ModuleOrder(monomial_key(order), self.ring.order, TOP)
        self._gb = None
        self._gbe = None

    def __eq__(self, other):
        if isinstance(other, SubModulePolyRing) and self.order != other.order:
            return False
        return SubModule.__eq__(self, other)

    def _groebner(self, extended=False):
        """Returns a standard basis in sdm form."""
        from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_mora
        if self._gbe is None and extended:
            gb, gbe = sdm_groebner(
                [self.ring._vector_to_sdm(x, self.order) for x in self.gens],
                sdm_nf_mora, self.order, self.ring.dom, extended=True)
            self._gb, self._gbe = tuple(gb), tuple(gbe)
        if self._gb is None:
            self._gb = tuple(sdm_groebner(
                             [self.ring._vector_to_sdm(x, self.order) for x in self.gens],
               sdm_nf_mora, self.order, self.ring.dom))
        if extended:
            return self._gb, self._gbe
        else:
            return self._gb

    def _groebner_vec(self, extended=False):
        """Returns a standard basis in element form."""
        if not extended:
            return [FreeModuleElement(self,
                        tuple(self.ring._sdm_to_vector(x, self.rank)))
                    for x in self._groebner()]
        gb, gbe = self._groebner(extended=True)
        return ([self.convert(self.ring._sdm_to_vector(x, self.rank))
                 for x in gb],
                [self.ring._sdm_to_vector(x, len(self.gens)) for x in gbe])

    def _contains(self, x):
        from sympy.polys.distributedmodules import sdm_zero, sdm_nf_mora
        return sdm_nf_mora(self.ring._vector_to_sdm(x, self.order),
                           self._groebner(), self.order, self.ring.dom) == \
            sdm_zero()

    def _syzygies(self):
        """Compute syzygies. See [SCA, algorithm 2.5.4]."""
        # NOTE if self.gens is a standard basis, this can be done more
        #      efficiently using Schreyer's theorem

        # First bullet point
        k = len(self.gens)
        r = self.rank
        zero = self.ring.convert(0)
        one = self.ring.convert(1)
        Rkr = self.ring.free_module(r + k)
        newgens = []
        for j, f in enumerate(self.gens):
            m = [0]*(r + k)
            for i, v in enumerate(f):
                m[i] = v
            for i in range(k):
                m[r + i] = one if j == i else zero
            m = FreeModuleElement(Rkr, tuple(m))
            newgens.append(m)
        # Note: we need *descending* order on module index, and TOP=False to
        #       get an elimination order
        F = Rkr.submodule(*newgens, order='ilex', TOP=False)

        # Second bullet point: standard basis of F
        G = F._groebner_vec()

        # Third bullet point: G0 = G intersect the new k components
        G0 = [x[r:] for x in G if all(y == zero for y in x[:r])]

        # Fourth and fifth bullet points: we are done
        return G0

    def _in_terms_of_generators(self, e):
        """Expression in terms of generators. See [SCA, 2.8.1]."""
        # NOTE: if gens is a standard basis, this can be done more efficiently
        M = self.ring.free_module(self.rank).submodule(*((e,) + self.gens))
        S = M.syzygy_module(
            order="ilex", TOP=False)  # We want decreasing order!
        G = S._groebner_vec()
        # This list cannot not be empty since e is an element
        e = [x for x in G if self.ring.is_unit(x[0])][0]
        return [-x/e[0] for x in e[1:]]

    def reduce_element(self, x, NF=None):
        """
        Reduce the element ``x`` of our container modulo ``self``.

        This applies the normal form ``NF`` to ``x``. If ``NF`` is passed
        as none, the default Mora normal form is used (which is not unique!).
        """
        from sympy.polys.distributedmodules import sdm_nf_mora
        if NF is None:
            NF = sdm_nf_mora
        return self.container.convert(self.ring._sdm_to_vector(NF(
            self.ring._vector_to_sdm(x, self.order), self._groebner(),
            self.order, self.ring.dom),
            self.rank))

    def _intersect(self, other, relations=False):
        # See: [SCA, section 2.8.2]
        fi = self.gens
        hi = other.gens
        r = self.rank
        ci = [[0]*(2*r) for _ in range(r)]
        for k in range(r):
            ci[k][k] = 1
            ci[k][r + k] = 1
        di = [list(f) + [0]*r for f in fi]
        ei = [[0]*r + list(h) for h in hi]
        syz = self.ring.free_module(2*r).submodule(*(ci + di + ei))._syzygies()
        nonzero = [x for x in syz if any(y != self.ring.zero for y in x[:r])]
        res = self.container.submodule(*([-y for y in x[:r]] for x in nonzero))
        reln1 = [x[r:r + len(fi)] for x in nonzero]
        reln2 = [x[r + len(fi):] for x in nonzero]
        if relations:
            return res, reln1, reln2
        return res

    def _module_quotient(self, other, relations=False):
        # See: [SCA, section 2.8.4]
        if relations and len(other.gens) != 1:
            raise NotImplementedError
        if len(other.gens) == 0:
            return self.ring.ideal(1)
        elif len(other.gens) == 1:
            # We do some trickery. Let f be the (vector!) generating ``other``
            # and f1, .., fn be the (vectors) generating self.
            # Consider the submodule of R^{r+1} generated by (f, 1) and
            # {(fi, 0) | i}. Then the intersection with the last module
            # component yields the quotient.
            g1 = list(other.gens[0]) + [1]
            gi = [list(x) + [0] for x in self.gens]
            # NOTE: We *need* to use an elimination order
            M = self.ring.free_module(self.rank + 1).submodule(*([g1] + gi),
                                            order='ilex', TOP=False)
            if not relations:
                return self.ring.ideal(*[x[-1] for x in M._groebner_vec() if
                                         all(y == self.ring.zero for y in x[:-1])])
            else:
                G, R = M._groebner_vec(extended=True)
                indices = [i for i, x in enumerate(G) if
                           all(y == self.ring.zero for y in x[:-1])]
                return (self.ring.ideal(*[G[i][-1] for i in indices]),
                        [[-x for x in R[i][1:]] for i in indices])
        # For more generators, we use I : <h1, .., hn> = intersection of
        #                                    {I : <hi> | i}
        # TODO this can be done more efficiently
        return reduce(lambda x, y: x.intersect(y),
            (self._module_quotient(self.container.submodule(x)) for x in other.gens))


class SubModuleQuotientRing(SubModule):
    """
    Class for submodules of free modules over quotient rings.

    Do not instantiate this. Instead use the submodule methods.

    >>> from sympy.abc import x, y
    >>> from sympy import QQ
    >>> M = (QQ.old_poly_ring(x, y)/[x**2 - y**2]).free_module(2).submodule([x, x + y])
    >>> M
    <[x + <x**2 - y**2>, x + y + <x**2 - y**2>]>
    >>> M.contains([y**2, x**2 + x*y])
    True
    >>> M.contains([x, y])
    False

    Attributes:

    - quot - the subquotient of `R^n/IR^n` generated by lifts of our generators
    """

    def __init__(self, gens, container):
        SubModule.__init__(self, gens, container)
        self.quot = self.container.quot.submodule(
            *[self.container.lift(x) for x in self.gens])

    def _contains(self, elem):
        return self.quot._contains(self.container.lift(elem))

    def _syzygies(self):
        return [tuple(self.ring.convert(y, self.quot.ring) for y in x)
                for x in self.quot._syzygies()]

    def _in_terms_of_generators(self, elem):
        return [self.ring.convert(x, self.quot.ring) for x in
            self.quot._in_terms_of_generators(self.container.lift(elem))]

##########################################################################
## Quotient Modules ######################################################
##########################################################################


class QuotientModuleElement(ModuleElement):
    """Element of a quotient module."""

    def eq(self, d1, d2):
        """Equality comparison."""
        return self.module.killed_module.contains(d1 - d2)

    def __repr__(self):
        return repr(self.data) + " + " + repr(self.module.killed_module)


class QuotientModule(Module):
    """
    Class for quotient modules.

    Do not instantiate this directly. For subquotients, see the
    SubQuotientModule class.

    Attributes:

    - base - the base module we are a quotient of
    - killed_module - the submodule used to form the quotient
    - rank of the base
    """

    dtype = QuotientModuleElement

    def __init__(self, ring, base, submodule):
        Module.__init__(self, ring)
        if not base.is_submodule(submodule):
            raise ValueError('%s is not a submodule of %s' % (submodule, base))
        self.base = base
        self.killed_module = submodule
        self.rank = base.rank

    def __repr__(self):
        return repr(self.base) + "/" + repr(self.killed_module)

    def is_zero(self):
        """
        Return True if ``self`` is a zero module.

        This happens if and only if the base module is the same as the
        submodule being killed.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> (F/[(1, 0)]).is_zero()
        False
        >>> (F/[(1, 0), (0, 1)]).is_zero()
        True
        """
        return self.base == self.killed_module

    def is_submodule(self, other):
        """
        Return True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> Q = QQ.old_poly_ring(x).free_module(2) / [(x, x)]
        >>> S = Q.submodule([1, 0])
        >>> Q.is_submodule(S)
        True
        >>> S.is_submodule(Q)
        False
        """
        if isinstance(other, QuotientModule):
            return self.killed_module == other.killed_module and \
                self.base.is_submodule(other.base)
        if isinstance(other, SubQuotientModule):
            return other.container == self
        return False

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        This is the same as taking a quotient of a submodule of the base
        module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> Q = QQ.old_poly_ring(x).free_module(2) / [(x, x)]
        >>> Q.submodule([x, 0])
        <[x, 0] + <[x, x]>>
        """
        return SubQuotientModule(gens, self, **opts)

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> F.convert([1, 0])
        [1, 0] + <[1, 2], [1, x]>
        """
        if isinstance(elem, QuotientModuleElement):
            if elem.module is self:
                return elem
            if self.killed_module.is_submodule(elem.module.killed_module):
                return QuotientModuleElement(self, self.base.convert(elem.data))
            raise CoercionFailed
        return QuotientModuleElement(self, self.base.convert(elem))

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> M.identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2/<[1, 2], [1, x]> -> QQ[x]**2/<[1, 2], [1, x]>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(
            self.killed_module).quotient_domain(self.killed_module)

    def quotient_hom(self):
        """
        Return the quotient homomorphism to ``self``.

        That is, return a homomorphism representing the natural map from
        ``self.base`` to ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2/<[1, 2], [1, x]>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(
            self.killed_module)
<|endoftext|>"""Finite extensions of ring domains."""

from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import (CoercionFailed, NotInvertible,
        GeneratorsError)
from sympy.polys.polytools import Poly
from sympy.printing.defaults import DefaultPrinting


class ExtensionElement(DomainElement, DefaultPrinting):
    """
    Element of a finite extension.

    A class of univariate polynomials modulo the ``modulus``
    of the extension ``ext``. It is represented by the
    unique polynomial ``rep`` of lowest degree. Both
    ``rep`` and the representation ``mod`` of ``modulus``
    are of class DMP.

    """
    __slots__ = ('rep', 'ext')

    def __init__(self, rep, ext):
        self.rep = rep
        self.ext = ext

    def parent(f):
        return f.ext

    def as_expr(f):
        return f.ext.to_sympy(f)

    def __bool__(f):
        return bool(f.rep)

    def __pos__(f):
        return f

    def __neg__(f):
        return ExtElem(-f.rep, f.ext)

    def _get_rep(f, g):
        if isinstance(g, ExtElem):
            if g.ext == f.ext:
                return g.rep
            else:
                return None
        else:
            try:
                g = f.ext.convert(g)
                return g.rep
            except CoercionFailed:
                return None

    def __add__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep + rep, f.ext)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(f.rep - rep, f.ext)
        else:
            return NotImplemented

    def __rsub__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem(rep - f.rep, f.ext)
        else:
            return NotImplemented

    def __mul__(f, g):
        rep = f._get_rep(g)
        if rep is not None:
            return ExtElem((f.rep * rep) % f.ext.mod, f.ext)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def _divcheck(f):
        """Raise if division is not implemented for this divisor"""
        if not f:
            raise NotInvertible('Zero divisor')
        elif f.ext.is_Field:
            return True
        elif f.rep.is_ground and f.ext.domain.is_unit(f.rep.LC()):
            return True
        else:
            # Some cases like (2*x + 2)/2 over ZZ will fail here. It is
            # unclear how to implement division in general if the ground
            # domain is not a field so for now it was decided to restrict the
            # implementation to division by invertible constants.
            msg = (f"Can not invert {f} in {f.ext}. "
                    "Only division by invertible constants is implemented.")
            raise NotImplementedError(msg)

    def inverse(f):
        """Multiplicative inverse.

        Raises
        ======

        NotInvertible
            If the element is a zero divisor.

        """
        f._divcheck()

        if f.ext.is_Field:
            invrep = f.rep.invert(f.ext.mod)
        else:
            R = f.ext.ring
            invrep = R.exquo(R.one, f.rep)

        return ExtElem(invrep, f.ext)

    def __truediv__(f, g):
        rep = f._get_rep(g)
        if rep is None:
            return NotImplemented
        g = ExtElem(rep, f.ext)

        try:
            ginv = g.inverse()
        except NotInvertible:
            raise ZeroDivisionError(f"{f} / {g}")

        return f * ginv

    __floordiv__ = __truediv__

    def __rtruediv__(f, g):
        try:
            g = f.ext.convert(g)
        except CoercionFailed:
            return NotImplemented
        return g / f

    __rfloordiv__ = __rtruediv__

    def __mod__(f, g):
        rep = f._get_rep(g)
        if rep is None:
            return NotImplemented
        g = ExtElem(rep, f.ext)

        try:
            g._divcheck()
        except NotInvertible:
            raise ZeroDivisionError(f"{f} % {g}")

        # Division where defined is always exact so there is no remainder
        return f.ext.zero

    def __rmod__(f, g):
        try:
            g = f.ext.convert(g)
        except CoercionFailed:
            return NotImplemented
        return g % f

    def __pow__(f, n):
        if not isinstance(n, int):
            raise TypeError("exponent of type 'int' expected")
        if n < 0:
            try:
                f, n = f.inverse(), -n
            except NotImplementedError:
                raise ValueError("negative powers are not defined")

        b = f.rep
        m = f.ext.mod
        r = f.ext.one.rep
        while n > 0:
            if n % 2:
                r = (r*b) % m
            b = (b*b) % m
            n //= 2

        return ExtElem(r, f.ext)

    def __eq__(f, g):
        if isinstance(g, ExtElem):
            return f.rep == g.rep and f.ext == g.ext
        else:
            return NotImplemented

    def __ne__(f, g):
        return not f == g

    def __hash__(f):
        return hash((f.rep, f.ext))

    def __str__(f):
        from sympy.printing.str import sstr
        return sstr(f.as_expr())

    __repr__ = __str__

    @property
    def is_ground(f):
        return f.rep.is_ground

    def to_ground(f):
        [c] = f.rep.to_list()
        return c

ExtElem = ExtensionElement


class MonogenicFiniteExtension(Domain):
    r"""
    Finite extension generated by an integral element.

    The generator is defined by a monic univariate
    polynomial derived from the argument ``mod``.

    A shorter alias is ``FiniteExtension``.

    Examples
    ========

    Quadratic integer ring $\mathbb{Z}[\sqrt2]$:

    >>> from sympy import Symbol, Poly
    >>> from sympy.polys.agca.extensions import FiniteExtension
    >>> x = Symbol('x')
    >>> R = FiniteExtension(Poly(x**2 - 2)); R
    ZZ[x]/(x**2 - 2)
    >>> R.rank
    2
    >>> R(1 + x)*(3 - 2*x)
    x - 1

    Finite field $GF(5^3)$ defined by the primitive
    polynomial $x^3 + x^2 + 2$ (over $\mathbb{Z}_5$).

    >>> F = FiniteExtension(Poly(x**3 + x**2 + 2, modulus=5)); F
    GF(5)[x]/(x**3 + x**2 + 2)
    >>> F.basis
    (1, x, x**2)
    >>> F(x + 3)/(x**2 + 2)
    -2*x**2 + x + 2

    Function field of an elliptic curve:

    >>> t = Symbol('t')
    >>> FiniteExtension(Poly(t**2 - x**3 - x + 1, t, field=True))
    ZZ(x)[t]/(t**2 - x**3 - x + 1)

    """
    is_FiniteExtension = True

    dtype = ExtensionElement

    def __init__(self, mod):
        if not (isinstance(mod, Poly) and mod.is_univariate):
            raise TypeError("modulus must be a univariate Poly")

        # Using auto=True (default) potentially changes the ground domain to a
        # field whereas auto=False raises if division is not exact.  We'll let
        # the caller decide whether or not they want to put the ground domain
        # over a field. In most uses mod is already monic.
        mod = mod.monic(auto=False)

        self.rank = mod.degree()
        self.modulus = mod
        self.mod = mod.rep  # DMP representation

        self.domain = dom = mod.domain
        self.ring = dom.old_poly_ring(*mod.gens)

        self.zero = self.convert(self.ring.zero)
        self.one = self.convert(self.ring.one)

        gen = self.ring.gens[0]
        self.symbol = self.ring.symbols[0]
        self.generator = self.convert(gen)
        self.basis = tuple(self.convert(gen**i) for i in range(self.rank))

        # XXX: It might be necessary to check mod.is_irreducible here
        self.is_Field = self.domain.is_Field

    def new(self, arg):
        rep = self.ring.convert(arg)
        return ExtElem(rep % self.mod, self)

    def __eq__(self, other):
        if not isinstance(other, FiniteExtension):
            return False
        return self.modulus == other.modulus

    def __hash__(self):
        return hash((self.__class__.__name__, self.modulus))

    def __str__(self):
        return "%s/(%s)" % (self.ring, self.modulus.as_expr())

    __repr__ = __str__

    @property
    def has_CharacteristicZero(self):
        return self.domain.has_CharacteristicZero

    def characteristic(self):
        return self.domain.characteristic()

    def convert(self, f, base=None):
        rep = self.ring.convert(f, base)
        return ExtElem(rep % self.mod, self)

    def convert_from(self, f, base):
        rep = self.ring.convert(f, base)
        return ExtElem(rep % self.mod, self)

    def to_sympy(self, f):
        return self.ring.to_sympy(f.rep)

    def from_sympy(self, f):
        return self.convert(f)

    def set_domain(self, K):
        mod = self.modulus.set_domain(K)
        return self.__class__(mod)

    def drop(self, *symbols):
        if self.symbol in symbols:
            raise GeneratorsError('Can not drop generator from FiniteExtension')
        K = self.domain.drop(*symbols)
        return self.set_domain(K)

    def quo(self, f, g):
        return self.exquo(f, g)

    def exquo(self, f, g):
        rep = self.ring.exquo(f.rep, g.rep)
        return ExtElem(rep % self.mod, self)

    def is_negative(self, a):
        return False

    def is_unit(self, a):
        if self.is_Field:
            return bool(a)
        elif a.is_ground:
            return self.domain.is_unit(a.to_ground())

FiniteExtension = MonogenicFiniteExtension
<|endoftext|>"""

Module for the DomainMatrix class.

A DomainMatrix represents a matrix with elements that are in a particular
Domain. Each DomainMatrix internally wraps a DDM which is used for the
lower-level operations. The idea is that the DomainMatrix class provides the
convenience routines for converting between Expr and the poly domains as well
as unifying matrices with different domains.

"""
from __future__ import annotations
from collections import Counter
from functools import reduce

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from sympy.core.sympify import _sympify

from ..domains import Domain

from ..constructor import construct_domain

from .exceptions import (
    DMFormatError,
    DMBadInputError,
    DMShapeError,
    DMDomainError,
    DMNotAField,
    DMNonSquareMatrixError,
    DMNonInvertibleMatrixError
)

from .domainscalar import DomainScalar

from sympy.polys.domains import ZZ, EXRAW, QQ

from sympy.polys.densearith import dup_mul
from sympy.polys.densebasic import dup_convert
from sympy.polys.densetools import (
    dup_mul_ground,
    dup_quo_ground,
    dup_content,
    dup_clear_denoms,
    dup_primitive,
    dup_transform,
)
from sympy.polys.factortools import dup_factor_list
from sympy.polys.polyutils import _sort_factors

from .ddm import DDM

from .sdm import SDM

from .dfm import DFM

from .rref import _dm_rref, _dm_rref_den


if GROUND_TYPES != 'flint':
    __doctest_skip__ = ['DomainMatrix.to_dfm', 'DomainMatrix.to_dfm_or_ddm']
else:
    __doctest_skip__ = ['DomainMatrix.from_list']


def DM(rows, domain):
    """Convenient alias for DomainMatrix.from_list

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DM
    >>> DM([[1, 2], [3, 4]], ZZ)
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DomainMatrix.from_list
    """
    return DomainMatrix.from_list(rows, domain)


class DomainMatrix:
    r"""
    Associate Matrix with :py:class:`~.Domain`

    Explanation
    ===========

    DomainMatrix uses :py:class:`~.Domain` for its internal representation
    which makes it faster than the SymPy Matrix class (currently) for many
    common operations, but this advantage makes it not entirely compatible
    with Matrix. DomainMatrix are analogous to numpy arrays with "dtype".
    In the DomainMatrix, each element has a domain such as :ref:`ZZ`
    or  :ref:`QQ(a)`.


    Examples
    ========

    Creating a DomainMatrix from the existing Matrix class:

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices import DomainMatrix
    >>> Matrix1 = Matrix([
    ...    [1, 2],
    ...    [3, 4]])
    >>> A = DomainMatrix.from_Matrix(Matrix1)
    >>> A
    DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    Directly forming a DomainMatrix:

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> A = DomainMatrix([
    ...    [ZZ(1), ZZ(2)],
    ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> A
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DDM
    SDM
    Domain
    Poly

    """
    rep: SDM | DDM | DFM
    shape: tuple[int, int]
    domain: Domain

    def __new__(cls, rows, shape, domain, *, fmt=None):
        """
        Creates a :py:class:`~.DomainMatrix`.

        Parameters
        ==========

        rows : Represents elements of DomainMatrix as list of lists
        shape : Represents dimension of DomainMatrix
        domain : Represents :py:class:`~.Domain` of DomainMatrix

        Raises
        ======

        TypeError
            If any of rows, shape and domain are not provided

        """
        if isinstance(rows, (DDM, SDM, DFM)):
            raise TypeError("Use from_rep to initialise from SDM/DDM")
        elif isinstance(rows, list):
            rep = DDM(rows, shape, domain)
        elif isinstance(rows, dict):
            rep = SDM(rows, shape, domain)
        else:
            msg = "Input should be list-of-lists or dict-of-dicts"
            raise TypeError(msg)

        if fmt is not None:
            if fmt == 'sparse':
                rep = rep.to_sdm()
            elif fmt == 'dense':
                rep = rep.to_ddm()
            else:
                raise ValueError("fmt should be 'sparse' or 'dense'")

        # Use python-flint for dense matrices if possible
        if rep.fmt == 'dense' and DFM._supports_domain(domain):
            rep = rep.to_dfm()

        return cls.from_rep(rep)

    def __reduce__(self):
        rep = self.rep
        if rep.fmt == 'dense':
            arg = self.to_list()
        elif rep.fmt == 'sparse':
            arg = dict(rep)
        else:
            raise RuntimeError # pragma: no cover
        args = (arg, rep.shape, rep.domain)
        return (self.__class__, args)

    def __getitem__(self, key):
        i, j = key
        m, n = self.shape
        if not (isinstance(i, slice) or isinstance(j, slice)):
            return DomainScalar(self.rep.getitem(i, j), self.domain)

        if not isinstance(i, slice):
            if not -m <= i < m:
                raise IndexError("Row index out of range")
            i = i % m
            i = slice(i, i+1)
        if not isinstance(j, slice):
            if not -n <= j < n:
                raise IndexError("Column index out of range")
            j = j % n
            j = slice(j, j+1)

        return self.from_rep(self.rep.extract_slice(i, j))

    def getitem_sympy(self, i, j):
        return self.domain.to_sympy(self.rep.getitem(i, j))

    def extract(self, rowslist, colslist):
        return self.from_rep(self.rep.extract(rowslist, colslist))

    def __setitem__(self, key, value):
        i, j = key
        if not self.domain.of_type(value):
            raise TypeError
        if isinstance(i, int) and isinstance(j, int):
            self.rep.setitem(i, j, value)
        else:
            raise NotImplementedError

    @classmethod
    def from_rep(cls, rep):
        """Create a new DomainMatrix efficiently from DDM/SDM.

        Examples
        ========

        Create a :py:class:`~.DomainMatrix` with an dense internal
        representation as :py:class:`~.DDM`:

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.matrices.ddm import DDM
        >>> drep = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> dM = DomainMatrix.from_rep(drep)
        >>> dM
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

        Create a :py:class:`~.DomainMatrix` with a sparse internal
        representation as :py:class:`~.SDM`:

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import ZZ
        >>> drep = SDM({0:{1:ZZ(1)},1:{0:ZZ(2)}}, (2, 2), ZZ)
        >>> dM = DomainMatrix.from_rep(drep)
        >>> dM
        DomainMatrix({0: {1: 1}, 1: {0: 2}}, (2, 2), ZZ)

        Parameters
        ==========

        rep: SDM or DDM
            The internal sparse or dense representation of the matrix.

        Returns
        =======

        DomainMatrix
            A :py:class:`~.DomainMatrix` wrapping *rep*.

        Notes
        =====

        This takes ownership of rep as its internal representation. If rep is
        being mutated elsewhere then a copy should be provided to
        ``from_rep``. Only minimal verification or checking is done on *rep*
        as this is supposed to be an efficient internal routine.

        """
        if not (isinstance(rep, (DDM, SDM)) or (DFM is not None and isinstance(rep, DFM))):
            raise TypeError("rep should be of type DDM or SDM")
        self = super().__new__(cls)
        self.rep = rep
        self.shape = rep.shape
        self.domain = rep.domain
        return self

    @classmethod
    @doctest_depends_on(ground_types=['python', 'gmpy'])
    def from_list(cls, rows, domain):
        r"""
        Convert a list of lists into a DomainMatrix

        Parameters
        ==========

        rows: list of lists
            Each element of the inner lists should be either the single arg,
            or tuple of args, that would be passed to the domain constructor
            in order to form an element of the domain. See examples.

        Returns
        =======

        DomainMatrix containing elements defined in rows

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import FF, QQ, ZZ
        >>> A = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], ZZ)
        >>> A
        DomainMatrix([[1, 0, 1], [0, 0, 1]], (2, 3), ZZ)
        >>> B = DomainMatrix.from_list([[1, 0, 1], [0, 0, 1]], FF(7))
        >>> B
        DomainMatrix([[1 mod 7, 0 mod 7, 1 mod 7], [0 mod 7, 0 mod 7, 1 mod 7]], (2, 3), GF(7))
        >>> C = DomainMatrix.from_list([[(1, 2), (3, 1)], [(1, 4), (5, 1)]], QQ)
        >>> C
        DomainMatrix([[1/2, 3], [1/4, 5]], (2, 2), QQ)

        See Also
        ========

        from_list_sympy

        """
        nrows = len(rows)
        ncols = 0 if not nrows else len(rows[0])
        conv = lambda e: domain(*e) if isinstance(e, tuple) else domain(e)
        domain_rows = [[conv(e) for e in row] for row in rows]
        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_list_sympy(cls, nrows, ncols, rows, **kwargs):
        r"""
        Convert a list of lists of Expr into a DomainMatrix using construct_domain

        Parameters
        ==========

        nrows: number of rows
        ncols: number of columns
        rows: list of lists

        Returns
        =======

        DomainMatrix containing elements of rows

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x, y, z
        >>> A = DomainMatrix.from_list_sympy(1, 3, [[x, y, z]])
        >>> A
        DomainMatrix([[x, y, z]], (1, 3), ZZ[x,y,z])

        See Also
        ========

        sympy.polys.constructor.construct_domain, from_dict_sympy

        """
        assert len(rows) == nrows
        assert all(len(row) == ncols for row in rows)

        items_sympy = [_sympify(item) for row in rows for item in row]

        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        domain_rows = [[items_domain[ncols*r + c] for c in range(ncols)] for r in range(nrows)]

        return DomainMatrix(domain_rows, (nrows, ncols), domain)

    @classmethod
    def from_dict_sympy(cls, nrows, ncols, elemsdict, **kwargs):
        """

        Parameters
        ==========

        nrows: number of rows
        ncols: number of cols
        elemsdict: dict of dicts containing non-zero elements of the DomainMatrix

        Returns
        =======

        DomainMatrix containing elements of elemsdict

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy.abc import x,y,z
        >>> elemsdict = {0: {0:x}, 1:{1: y}, 2: {2: z}}
        >>> A = DomainMatrix.from_dict_sympy(3, 3, elemsdict)
        >>> A
        DomainMatrix({0: {0: x}, 1: {1: y}, 2: {2: z}}, (3, 3), ZZ[x,y,z])

        See Also
        ========

        from_list_sympy

        """
        if not all(0 <= r < nrows for r in elemsdict):
            raise DMBadInputError("Row out of range")
        if not all(0 <= c < ncols for row in elemsdict.values() for c in row):
            raise DMBadInputError("Column out of range")

        items_sympy = [_sympify(item) for row in elemsdict.values() for item in row.values()]
        domain, items_domain = cls.get_domain(items_sympy, **kwargs)

        idx = 0
        items_dict = {}
        for i, row in elemsdict.items():
            items_dict[i] = {}
            for j in row:
                items_dict[i][j] = items_domain[idx]
                idx += 1

        return DomainMatrix(items_dict, (nrows, ncols), domain)

    @classmethod
    def from_Matrix(cls, M, fmt='sparse',**kwargs):
        r"""
        Convert Matrix to DomainMatrix

        Parameters
        ==========

        M: Matrix

        Returns
        =======

        Returns DomainMatrix with identical elements as M

        Examples
        ========

        >>> from sympy import Matrix
        >>> from sympy.polys.matrices import DomainMatrix
        >>> M = Matrix([
        ...    [1.0, 3.4],
        ...    [2.4, 1]])
        >>> A = DomainMatrix.from_Matrix(M)
        >>> A
        DomainMatrix({0: {0: 1.0, 1: 3.4}, 1: {0: 2.4, 1: 1.0}}, (2, 2), RR)

        We can keep internal representation as ddm using fmt='dense'
        >>> from sympy import Matrix, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix.from_Matrix(Matrix([[QQ(1, 2), QQ(3, 4)], [QQ(0, 1), QQ(0, 1)]]), fmt='dense')
        >>> A.rep
        [[1/2, 3/4], [0, 0]]

        See Also
        ========

        Matrix

        """
        if fmt == 'dense':
            return cls.from_list_sympy(*M.shape, M.tolist(), **kwargs)

        return cls.from_dict_sympy(*M.shape, M.todod(), **kwargs)

    @classmethod
    def get_domain(cls, items_sympy, **kwargs):
        K, items_K = construct_domain(items_sympy, **kwargs)
        return K, items_K

    def choose_domain(self, **opts):
        """Convert to a domain found by :func:`~.construct_domain`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[1, 2], [3, 4]], ZZ)
        >>> M
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> M.choose_domain(field=True)
        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)

        >>> from sympy.abc import x
        >>> M = DM([[1, x], [x**2, x**3]], ZZ[x])
        >>> M.choose_domain(field=True).domain
        ZZ(x)

        Keyword arguments are passed to :func:`~.construct_domain`.

        See Also
        ========

        construct_domain
        convert_to
        """
        elements, data = self.to_sympy().to_flat_nz()
        dom, elements_dom = construct_domain(elements, **opts)
        return self.from_flat_nz(elements_dom, data, dom)

    def copy(self):
        return self.from_rep(self.rep.copy())

    def convert_to(self, K):
        r"""
        Change the domain of DomainMatrix to desired domain or field

        Parameters
        ==========

        K : Represents the desired domain or field.
            Alternatively, ``None`` may be passed, in which case this method
            just returns a copy of this DomainMatrix.

        Returns
        =======

        DomainMatrix
            DomainMatrix with the desired domain or field

        Examples
        ========

        >>> from sympy import ZZ, ZZ_I
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.convert_to(ZZ_I)
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ_I)

        """
        if K == self.domain:
            return self.copy()

        rep = self.rep

        # The DFM, DDM and SDM types do not do any implicit conversions so we
        # manage switching between DDM and DFM here.
        if rep.is_DFM and not DFM._supports_domain(K):
            rep_K = rep.to_ddm().convert_to(K)
        elif rep.is_DDM and DFM._supports_domain(K):
            rep_K = rep.convert_to(K).to_dfm()
        else:
            rep_K = rep.convert_to(K)

        return self.from_rep(rep_K)

    def to_sympy(self):
        return self.convert_to(EXRAW)

    def to_field(self):
        r"""
        Returns a DomainMatrix with the appropriate field

        Returns
        =======

        DomainMatrix
            DomainMatrix with the appropriate field

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.to_field()
        DomainMatrix([[1, 2], [3, 4]], (2, 2), QQ)

        """
        K = self.domain.get_field()
        return self.convert_to(K)

    def to_sparse(self):
        """
        Return a sparse DomainMatrix representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> A.rep
        [[1, 0], [0, 2]]
        >>> B = A.to_sparse()
        >>> B.rep
        {0: {0: 1}, 1: {1: 2}}
        """
        if self.rep.fmt == 'sparse':
            return self

        return self.from_rep(self.rep.to_sdm())

    def to_dense(self):
        """
        Return a dense DomainMatrix representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)
        >>> A.rep
        {0: {0: 1}, 1: {1: 2}}
        >>> B = A.to_dense()
        >>> B.rep
        [[1, 0], [0, 2]]

        """
        rep = self.rep

        if rep.fmt == 'dense':
            return self

        return self.from_rep(rep.to_dfm_or_ddm())

    def to_ddm(self):
        """
        Return a :class:`~.DDM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix({0: {0: 1}, 1: {1: 2}}, (2, 2), QQ)
        >>> ddm = A.to_ddm()
        >>> ddm
        [[1, 0], [0, 2]]
        >>> type(ddm)
        <class 'sympy.polys.matrices.ddm.DDM'>

        See Also
        ========

        to_sdm
        to_dense
        sympy.polys.matrices.ddm.DDM.to_sdm
        """
        return self.rep.to_ddm()

    def to_sdm(self):
        """
        Return a :class:`~.SDM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> sdm = A.to_sdm()
        >>> sdm
        {0: {0: 1}, 1: {1: 2}}
        >>> type(sdm)
        <class 'sympy.polys.matrices.sdm.SDM'>

        See Also
        ========

        to_ddm
        to_sparse
        sympy.polys.matrices.sdm.SDM.to_ddm
        """
        return self.rep.to_sdm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm(self):
        """
        Return a :class:`~.DFM` representation of *self*.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> dfm = A.to_dfm()
        >>> dfm
        [[1, 0], [0, 2]]
        >>> type(dfm)
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm
        to_dense
        DFM
        """
        return self.rep.to_dfm()

    @doctest_depends_on(ground_types=['flint'])
    def to_dfm_or_ddm(self):
        """
        Return a :class:`~.DFM` or :class:`~.DDM` representation of *self*.

        Explanation
        ===========

        The :class:`~.DFM` representation can only be used if the ground types
        are ``flint`` and the ground domain is supported by ``python-flint``.
        This method will return a :class:`~.DFM` representation if possible,
        but will return a :class:`~.DDM` representation otherwise.

        Examples
        ========

        >>> from sympy.polys.matrices import DomainMatrix
        >>> from sympy import QQ
        >>> A = DomainMatrix([[1, 0],[0, 2]], (2, 2), QQ)
        >>> dfm = A.to_dfm_or_ddm()
        >>> dfm
        [[1, 0], [0, 2]]
        >>> type(dfm)  # Depends on the ground domain and ground types
        <class 'sympy.polys.matrices._dfm.DFM'>

        See Also
        ========

        to_ddm: Always return a :class:`~.DDM` representation.
        to_dfm: Returns a :class:`~.DFM` representation or raise an error.
        to_dense: Convert internally to a :class:`~.DFM` or :class:`~.DDM`
        DFM: The :class:`~.DFM` dense FLINT matrix representation.
        DDM: The Python :class:`~.DDM` dense domain matrix representation.
        """
        return self.rep.to_dfm_or_ddm()

    @classmethod
    def _unify_domain(cls, *matrices):
        """Convert matrices to a common domain"""
        domains = {matrix.domain for matrix in matrices}
        if len(domains) == 1:
            return matrices
        domain = reduce(lambda x, y: x.unify(y), domains)
        return tuple(matrix.convert_to(domain) for matrix in matrices)

    @classmethod
    def _unify_fmt(cls, *matrices, fmt=None):
        """Convert matrices to the same format.

        If all matrices have the same format, then return unmodified.
        Otherwise convert both to the preferred format given as *fmt* which
        should be 'dense' or 'sparse'.
        """
        formats = {matrix.rep.fmt for matrix in matrices}
        if len(formats) == 1:
            return matrices
        if fmt == 'sparse':
            return tuple(matrix.to_sparse() for matrix in matrices)
        elif fmt == 'dense':
            return tuple(matrix.to_dense() for matrix in matrices)
        else:
            raise ValueError("fmt should be 'sparse' or 'dense'")

    def unify(self, *others, fmt=None):
        """
        Unifies the domains and the format of self and other
        matrices.

        Parameters
        ==========

        others : DomainMatrix

        fmt: string 'dense', 'sparse' or `None` (default)
            The preferred format to convert to if self and other are not
            already in the same format. If `None` or not specified then no
            conversion if performed.

        Returns
        =======

        Tuple[DomainMatrix]
            Matrices with unified domain and format

        Examples
        ========

        Unify the domain of DomainMatrix that have different domains:

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
        >>> B = DomainMatrix([[QQ(1, 2), QQ(2)]], (1, 2), QQ)
        >>> Aq, Bq = A.unify(B)
        >>> Aq
        DomainMatrix([[1, 2]], (1, 2), QQ)
        >>> Bq
        DomainMatrix([[1/2, 2]], (1, 2), QQ)

        Unify the format (dense or sparse):

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
        >>> B = DomainMatrix({0:{0: ZZ(1)}}, (2, 2), ZZ)
        >>> B.rep
        {0: {0: 1}}

        >>> A2, B2 = A.unify(B, fmt='dense')
        >>> B2.rep
        [[1, 0], [0, 0]]

        See Also
        ========

        convert_to, to_dense, to_sparse

        """
        matrices = (self,) + others
        matrices = DomainMatrix._unify_domain(*matrices)
        if fmt is not None:
            matrices = DomainMatrix._unify_fmt(*matrices, fmt=fmt)
        return matrices

    def to_Matrix(self):
        r"""
        Convert DomainMatrix to Matrix

        Returns
        =======

        Matrix
            MutableDenseMatrix for the DomainMatrix

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.to_Matrix()
        Matrix([
            [1, 2],
            [3, 4]])

        See Also
        ========

        from_Matrix

        """
        from sympy.matrices.dense import MutableDenseMatrix

        # XXX: If the internal representation of RepMatrix changes then this
        # might need to be changed also.
        if self.domain in (ZZ, QQ, EXRAW):
            if self.rep.fmt == "sparse":
                rep = self.copy()
            else:
                rep = self.to_sparse()
        else:
            rep = self.convert_to(EXRAW).to_sparse()

        return MutableDenseMatrix._fromrep(rep)

    def to_list(self):
        """
        Convert :class:`DomainMatrix` to list of lists.

        See Also
        ========

        from_list
        to_list_flat
        to_flat_nz
        to_dok
        """
        return self.rep.to_list()

    def to_list_flat(self):
        """
        Convert :class:`DomainMatrix` to flat list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> A.to_list_flat()
        [1, 2, 3, 4]

        See Also
        ========

        from_list_flat
        to_list
        to_flat_nz
        to_dok
        """
        return self.rep.to_list_flat()

    @classmethod
    def from_list_flat(cls, elements, shape, domain):
        """
        Create :class:`DomainMatrix` from flat list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> element_list = [ZZ(1), ZZ(2), ZZ(3), ZZ(4)]
        >>> A = DomainMatrix.from_list_flat(element_list, (2, 2), ZZ)
        >>> A
        DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)
        >>> A == A.from_list_flat(A.to_list_flat(), A.shape, A.domain)
        True

        See Also
        ========

        to_list_flat
        """
        ddm = DDM.from_list_flat(elements, shape, domain)
        return cls.from_rep(ddm.to_dfm_or_ddm())

    def to_flat_nz(self):
        """
        Convert :class:`DomainMatrix` to list of nonzero elements and data.

        Explanation
        ===========

        Returns a tuple ``(elements, data)`` where ``elements`` is a list of
        elements of the matrix with zeros possibly excluded. The matrix can be
        reconstructed by passing these to :meth:`from_flat_nz`. The idea is to
        be able to modify a flat list of the elements and then create a new
        matrix of the same shape with the modified elements in the same
        positions.

        The format of ``data`` differs depending on whether the underlying
        representation is dense or sparse but either way it represents the
        positions of the elements in the list in a way that
        :meth:`from_flat_nz` can use to reconstruct the matrix. The
        :meth:`from_flat_nz` method should be called on the same
        :class:`DomainMatrix` that was used to call :meth:`to_flat_nz`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> elements, data = A.to_flat_nz()
        >>> elements
        [1, 2, 3, 4]
        >>> A == A.from_flat_nz(elements, data, A.domain)
        True

        Create a matrix with the elements doubled:

        >>> elements_doubled = [2*x for x in elements]
        >>> A2 = A.from_flat_nz(elements_doubled, data, A.domain)
        >>> A2 == 2*A
        True

        See Also
        ========

        from_flat_nz
        """
        return self.rep.to_flat_nz()

    def from_flat_nz(self, elements, data, domain):
        """
        Reconstruct :class:`DomainMatrix` after calling :meth:`to_flat_nz`.

        See :meth:`to_flat_nz` for explanation.

        See Also
        ========

        to_flat_nz
        """
        rep = self.rep.from_flat_nz(elements, data, domain)
        return self.from_rep(rep)

    def to_dod(self):
        """
        Convert :class:`DomainMatrix` to dictionary of dictionaries (dod) format.

        Explanation
        ===========

        Returns a dictionary of dictionaries representing the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[ZZ(1), ZZ(2), ZZ(0)], [ZZ(3), ZZ(0), ZZ(4)]], ZZ)
        >>> A.to_dod()
        {0: {0: 1, 1: 2}, 1: {0: 3, 2: 4}}
        >>> A.to_sparse() == A.from_dod(A.to_dod(), A.shape, A.domain)
        True
        >>> A == A.from_dod_like(A.to_dod())
        True

        See Also
        ========

        from_dod
        from_dod_like
        to_dok
        to_list
        to_list_flat
        to_flat_nz
        sympy.matrices.matrixbase.MatrixBase.todod
        """
        return self.rep.to_dod()

    @classmethod
    def from_dod(cls, dod, shape, domain):
        """
        Create sparse :class:`DomainMatrix` from dict of dict (dod) format.

        See :meth:`to_dod` for explanation.

        See Also
        ========

        to_dod
        from_dod_like
        """
        return cls.from_rep(SDM.from_dod(dod, shape, domain))

    def from_dod_like(self, dod, domain=None):
        """
        Create :class:`DomainMatrix` like ``self`` from dict of dict (dod) format.

        See :meth:`to_dod` for explanation.

        See Also
        ========

        to_dod
        from_dod
        """
        if domain is None:
            domain = self.domain
        return self.from_rep(self.rep.from_dod(dod, self.shape, domain))

    def to_dok(self):
        """
        Convert :class:`DomainMatrix` to dictionary of keys (dok) format.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(0)],
        ...    [ZZ(0), ZZ(4)]], (2, 2), ZZ)
        >>> A.to_dok()
        {(0, 0): 1, (1, 1): 4}

        The matrix can be reconstructed by calling :meth:`from_dok` although
        the reconstructed matrix will always be in sparse format:

        >>> A.to_sparse() == A.from_dok(A.to_dok(), A.shape, A.domain)
        True

        See Also
        ========

        from_dok
        to_list
        to_list_flat
        to_flat_nz
        """
        return self.rep.to_dok()

    @classmethod
    def from_dok(cls, dok, shape, domain):
        """
        Create :class:`DomainMatrix` from dictionary of keys (dok) format.

        See :meth:`to_dok` for explanation.

        See Also
        ========

        to_dok
        """
        return cls.from_rep(SDM.from_dok(dok, shape, domain))

    def iter_values(self):
        """
        Iterate over nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> list(A.iter_values())
        [1, 3, 4]

        See Also
        ========

        iter_items
        to_list_flat
        sympy.matrices.matrixbase.MatrixBase.iter_values
        """
        return self.rep.iter_values()

    def iter_items(self):
        """
        Iterate over indices and values of nonzero elements of the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([[ZZ(1), ZZ(0)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> list(A.iter_items())
        [((0, 0), 1), ((1, 0), 3), ((1, 1), 4)]

        See Also
        ========

        iter_values
        to_dok
        sympy.matrices.matrixbase.MatrixBase.iter_items
        """
        return self.rep.iter_items()

    def nnz(self):
        """
        Number of nonzero elements in the matrix.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> A = DM([[1, 0], [0, 4]], ZZ)
        >>> A.nnz()
        2
        """
        return self.rep.nnz()

    def __repr__(self):
        return 'DomainMatrix(%s, %r, %r)' % (str(self.rep), self.shape, self.domain)

    def transpose(self):
        """Matrix transpose of ``self``"""
        return self.from_rep(self.rep.transpose())

    def flat(self):
        rows, cols = self.shape
        return [self[i,j].element for i in range(rows) for j in range(cols)]

    @property
    def is_zero_matrix(self):
        return self.rep.is_zero_matrix()

    @property
    def is_upper(self):
        """
        Says whether this matrix is upper-triangular. True can be returned
        even if the matrix is not square.
        """
        return self.rep.is_upper()

    @property
    def is_lower(self):
        """
        Says whether this matrix is lower-triangular. True can be returned
        even if the matrix is not square.
        """
        return self.rep.is_lower()

    @property
    def is_diagonal(self):
        """
        True if the matrix is diagonal.

        Can return true for non-square matrices. A matrix is diagonal if
        ``M[i,j] == 0`` whenever ``i != j``.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[ZZ(1), ZZ(0)], [ZZ(0), ZZ(1)]], ZZ)
        >>> M.is_diagonal
        True

        See Also
        ========

        is_upper
        is_lower
        is_square
        diagonal
        """
        return self.rep.is_diagonal()

    def diagonal(self):
        """
        Get the diagonal entries of the matrix as a list.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DM
        >>> M = DM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], ZZ)
        >>> M.diagonal()
        [1, 4]

        See Also
        ========

        is_diagonal
        diag
        """
        return self.rep.diagonal()

    @property
    def is_square(self):
        """
        True if the matrix is square.
        """
        return self.shape[0] == self.shape[1]

    def rank(self):
        rref, pivots = self.rref()
        return len(pivots)

    def hstack(A, *B):
        r"""Horizontally stack the given matrices.

        Parameters
        ==========

        B: DomainMatrix
            Matrices to stack horizontally.

        Returns
        =======

        DomainMatrix
            DomainMatrix by stacking horizontally.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.hstack(B)
        DomainMatrix([[1, 2, 5, 6], [3, 4, 7, 8]], (2, 4), ZZ)

        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.hstack(B, C)
        DomainMatrix([[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]], (2, 6), ZZ)

        See Also
        ========

        unify
        """
        A, *B = A.unify(*B, fmt=A.rep.fmt)
        return DomainMatrix.from_rep(A.rep.hstack(*(Bk.rep for Bk in B)))

    def vstack(A, *B):
        r"""Vertically stack the given matrices.

        Parameters
        ==========

        B: DomainMatrix
            Matrices to stack vertically.

        Returns
        =======

        DomainMatrix
            DomainMatrix by stacking vertically.

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix

        >>> A = DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([[ZZ(5), ZZ(6)], [ZZ(7), ZZ(8)]], (2, 2), ZZ)
        >>> A.vstack(B)
        DomainMatrix([[1, 2], [3, 4], [5, 6], [7, 8]], (4, 2), ZZ)

        >>> C = DomainMatrix([[ZZ(9), ZZ(10)], [ZZ(11), ZZ(12)]], (2, 2), ZZ)
        >>> A.vstack(B, C)
        DomainMatrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], (6, 2), ZZ)

        See Also
        ========

        unify
        """
        A, *B = A.unify(*B, fmt='dense')
        return DomainMatrix.from_rep(A.rep.vstack(*(Bk.rep for Bk in B)))

    def applyfunc(self, func, domain=None):
        if domain is None:
            domain = self.domain
        return self.from_rep(self.rep.applyfunc(func, domain))

    def __add__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        A, B = A.unify(B, fmt='dense')
        return A.add(B)

    def __sub__(A, B):
        if not isinstance(B, DomainMatrix):
            return NotImplemented
        A, B = A.unify(B, fmt='dense')
        return A.sub(B)

    def __neg__(A):
        return A.neg()

    def __mul__(A, B):
        """A * B"""
        if isinstance(B, DomainMatrix):
            A, B = A.unify(B, fmt='dense')
            return A.matmul(B)
        elif B in A.domain:
            return A.scalarmul(B)
        elif isinstance(B, DomainScalar):
            A, B = A.unify(B)
            return A.scalarmul(B.element)
        else:
            return NotImplemented

    def __rmul__(A, B):
        if B in A.domain:
            return A.rscalarmul(B)
        elif isinstance(B, DomainScalar):
            A, B = A.unify(B)
            return A.rscalarmul(B.element)
        else:
            return NotImplemented

    def __pow__(A, n):
        """A ** n"""
        if not isinstance(n, int):
            return NotImplemented
        return A.pow(n)

    def _check(a, op, b, ashape, bshape):
        if a.domain != b.domain:
            msg = "Domain mismatch: %s %s %s" % (a.domain, op, b.domain)
            raise DMDomainError(msg)
        if ashape != bshape:
            msg = "Shape mismatch: %s %s %s" % (a.shape, op, b.shape)
            raise DMShapeError(msg)
        if a.rep.fmt != b.rep.fmt:
            msg = "Format mismatch: %s %s %s" % (a.rep.fmt, op, b.rep.fmt)
            raise DMFormatError(msg)
        if type(a.rep) != type(b.rep):
            msg = "Type mismatch: %s %s %s" % (type(a.rep), op, type(b.rep))
            raise DMFormatError(msg)

    def add(A, B):
        r"""
        Adds two DomainMatrix matrices of the same Domain

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to add

        Returns
        =======

        DomainMatrix
            DomainMatrix after Addition

        Raises
        ======

        DMShapeError
            If the dimensions of the two DomainMatrix are not equal

        ValueError
            If the domain of the two DomainMatrix are not same

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(4), ZZ(3)],
        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)

        >>> A.add(B)
        DomainMatrix([[5, 5], [5, 5]], (2, 2), ZZ)

        See Also
        ========

        sub, matmul

        """
        A._check('+', B, A.shape, B.shape)
        return A.from_rep(A.rep.add(B.rep))


    def sub(A, B):
        r"""
        Subtracts two DomainMatrix matrices of the same Domain

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to subtract

        Returns
        =======

        DomainMatrix
            DomainMatrix after Subtraction

        Raises
        ======

        DMShapeError
            If the dimensions of the two DomainMatrix are not equal

        ValueError
            If the domain of the two DomainMatrix are not same

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
        >>> B = DomainMatrix([
        ...    [ZZ(4), ZZ(3)],
        ...    [ZZ(2), ZZ(1)]], (2, 2), ZZ)

        >>> A.sub(B)
        DomainMatrix([[-3, -1], [1, 3]], (2, 2), ZZ)

        See Also
        ========

        add, matmul

        """
        A._check('-', B, A.shape, B.shape)
        return A.from_rep(A.rep.sub(B.rep))

    def neg(A):
        r"""
        Returns the negative of DomainMatrix

        Parameters
        ==========

        A : Represents a DomainMatrix

        Returns
        =======

        DomainMatrix
            DomainMatrix after Negation

        Examples
        ========

        >>> from sympy import ZZ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [ZZ(1), ZZ(2)],
        ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)

        >>> A.neg()
        DomainMatrix([[-1, -2], [-3, -4]], (2, 2), ZZ)

        """
        return A.from_rep(A.rep.neg())

    def mul(A, b):
        r"""
        Performs term by term multiplication for the second DomainMatrix
        w.r.t first DomainMatrix. Returns a DomainMatrix whose rows are
        list of DomainMatrix matrices created after term by term multiplication.

        Parameters
        ==========

        A, B: DomainMatrix
            matrices to multiply term-wise

        Returns
