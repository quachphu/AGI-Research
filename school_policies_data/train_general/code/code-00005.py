                    non-zero. column-orthogonal: U' U = B
                    here B is a n*n matrix whose first min(m,n) diagonal
                    elements are 1 and all other elements are zero.
      S : n*n       diagonal matrix, only the diagonal elements are stored in
                    the array S. only the first min(m,n) diagonal elements are non-zero.
      V : n*n       orthogonal: V V' = V' V = 1

    parameters:
      A        (input/output) On input, A contains a real matrix of shape m*n.
               On output, if calc_u is true A contains the column-orthogonal
               matrix U; otherwise A is simply used as workspace and thus destroyed.

      V        (input/output) if false, the matrix V is not calculated. otherwise
               V must be a matrix of shape n*n.

      calc_u   (input) If true, the matrix U is calculated and replaces A.
               if false, U is not calculated and A is simply destroyed

    return value:
      S        an array of length n containing the singular values of A sorted by
               decreasing magnitude. only the first min(m,n) elements are non-zero.

    This routine is a python translation of the fortran routine svd.f in the
    software library EISPACK (see netlib.org) which itself is based on the
    algol procedure svd described in:
      - num. math. 14, 403-420(1970) by golub and reinsch.
      - wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).

    """

    m, n = A.rows, A.cols

    S = ctx.zeros(n, 1)

    # work is a temporary array of size n
    work = ctx.zeros(n, 1)

    g = scale = anorm = 0
    maxits = 3 * ctx.dps

    for i in xrange(n):     # householder reduction to bidiagonal form
        work[i] = scale*g
        g = s = scale = 0
        if i < m:
            for k in xrange(i, m):
                scale += ctx.fabs(A[k,i])
            if scale != 0:
                for k in xrange(i, m):
                    A[k,i] /= scale
                    s += A[k,i] * A[k,i]
                f = A[i,i]
                g = -ctx.sqrt(s)
                if f < 0:
                    g = -g
                h = f * g - s
                A[i,i] = f - g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i, m):
                        s += A[k,i] * A[k,j]
                    f = s / h
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for k in xrange(i,m):
                    A[k,i] *= scale

        S[i] = scale * g
        g = s = scale = 0

        if i < m and i != n - 1:
            for k in xrange(i+1, n):
                scale += ctx.fabs(A[i,k])
            if scale:
                for k in xrange(i+1, n):
                    A[i,k] /= scale
                    s += A[i,k] * A[i,k]
                f = A[i,i+1]
                g = -ctx.sqrt(s)
                if f < 0:
                    g = -g
                h = f * g - s
                A[i,i+1] = f - g

                for k in xrange(i+1, n):
                    work[k] = A[i,k] / h

                for j in xrange(i+1, m):
                    s = 0
                    for k in xrange(i+1, n):
                        s += A[j,k] * A[i,k]
                    for k in xrange(i+1, n):
                        A[j,k] += s * work[k]

                for k in xrange(i+1, n):
                    A[i,k] *= scale

        anorm = max(anorm, ctx.fabs(S[i]) + ctx.fabs(work[i]))

    if not isinstance(V, bool):
        for i in xrange(n-2, -1, -1):     # accumulation of right hand transformations
            V[i+1,i+1] = 1

            if work[i+1] != 0:
                for j in xrange(i+1, n):
                    V[i,j] = (A[i,j] / A[i,i+1]) / work[i+1]
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, n):
                        s += A[i,k] * V[j,k]
                    for k in xrange(i+1, n):
                        V[j,k] += s * V[i,k]

            for j in xrange(i+1, n):
                V[j,i] = V[i,j] = 0

        V[0,0] = 1

    if m<n : minnm = m
    else   : minnm = n

    if calc_u:
        for i in xrange(minnm-1, -1, -1): # accumulation of left hand transformations
            g = S[i]
            for j in xrange(i+1, n):
                A[i,j] = 0
            if g != 0:
                g = 1 / g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, m):
                        s += A[k,i] * A[k,j]
                    f = (s / A[i,i]) * g
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for j in xrange(i, m):
                    A[j,i] *= g
            else:
                for j in xrange(i, m):
                    A[j,i] = 0
            A[i,i] += 1

    for k in xrange(n - 1, -1, -1):
        # diagonalization of the bidiagonal form:
        #   loop over singular values, and over allowed itations

        its = 0
        while 1:
            its += 1
            flag = True

            for l in xrange(k, -1, -1):
                nm = l-1

                if ctx.fabs(work[l]) + anorm == anorm:
                    flag = False
                    break

                if ctx.fabs(S[nm]) + anorm == anorm:
                    break

            if flag:
                c = 0
                s = 1
                for i in xrange(l, k + 1):
                    f = s * work[i]
                    work[i] *= c
                    if ctx.fabs(f) + anorm == anorm:
                        break
                    g = S[i]
                    h = ctx.hypot(f, g)
                    S[i] = h
                    h = 1 / h
                    c = g * h
                    s = - f * h

                    if calc_u:
                        for j in xrange(m):
                            y = A[j,nm]
                            z = A[j,i]
                            A[j,nm] = y * c + z * s
                            A[j,i]  = z * c - y * s

            z = S[k]

            if l == k:               # convergence
                if z < 0:            # singular value is made nonnegative
                    S[k] = -z
                    if not isinstance(V, bool):
                        for j in xrange(n):
                            V[k,j] = -V[k,j]
                break

            if its >= maxits:
                raise RuntimeError("svd: no convergence to an eigenvalue after %d iterations" % its)

            x = S[l]         # shift from bottom 2 by 2 minor
            nm = k-1
            y = S[nm]
            g = work[nm]
            h = work[k]
            f = ((y - z) * (y + z) + (g - h) * (g + h))/(2 * h * y)
            g = ctx.hypot(f, 1)
            if f >= 0: f = ((x - z) * (x + z) + h * ((y / (f + g)) - h)) / x
            else:      f = ((x - z) * (x + z) + h * ((y / (f - g)) - h)) / x

            c = s = 1         # next qt transformation

            for j in xrange(l, nm + 1):
                g = work[j+1]
                y = S[j+1]
                h = s * g
                g = c * g
                z = ctx.hypot(f, h)
                work[j] = z
                c = f / z
                s = h / z
                f = x * c + g * s
                g = g * c - x * s
                h = y * s
                y *= c
                if not isinstance(V, bool):
                    for jj in xrange(n):
                        x = V[j  ,jj]
                        z = V[j+1,jj]
                        V[j    ,jj]= x * c + z * s
                        V[j+1  ,jj]= z * c - x * s
                z = ctx.hypot(f, h)
                S[j] = z
                if z != 0:            # rotation can be arbitray if z=0
                    z = 1 / z
                    c = f * z
                    s = h * z
                f = c * g + s * y
                x = c * y - s * g

                if calc_u:
                    for jj in xrange(m):
                        y = A[jj,j  ]
                        z = A[jj,j+1]
                        A[jj,j    ] = y * c + z * s
                        A[jj,j+1  ] = z * c - y * s

            work[l] = 0
            work[k] = f
            S[k] = x

    ##########################

    # Sort singular values into decreasing order (bubble-sort)

    for i in xrange(n):
        imax = i
        s = ctx.fabs(S[i])         # s is the current maximal element

        for j in xrange(i + 1, n):
            c = ctx.fabs(S[j])
            if c > s:
                s = c
                imax = j

        if imax != i:
            # swap singular values

            z = S[i]
            S[i] = S[imax]
            S[imax] = z

            if calc_u:
                for j in xrange(m):
                    z = A[j,i]
                    A[j,i] = A[j,imax]
                    A[j,imax] = z

            if not isinstance(V, bool):
                for j in xrange(n):
                    z = V[i,j]
                    V[i,j] = V[imax,j]
                    V[imax,j] = z

    return S

#######################

def svd_c_raw(ctx, A, V = False, calc_u = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two unitary matrices U and V are calculated such that

                    A = U S V

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    The diagonal elements of S are the singular values of A, i.e. the
    squareroots of the eigenvalues of A' A or A A'. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation). Householder bidiagonalization
    and a variant of the QR algorithm is used.

    overview of the matrices :

      A : m*n       A gets replaced by U
      U : m*n       U replaces A. If n>m then only the first m*m block of U is
                    non-zero. column-unitary: U' U = B
                    here B is a n*n matrix whose first min(m,n) diagonal
                    elements are 1 and all other elements are zero.
      S : n*n       diagonal matrix, only the diagonal elements are stored in
                    the array S. only the first min(m,n) diagonal elements are non-zero.
      V : n*n       unitary: V V' = V' V = 1

    parameters:
      A        (input/output) On input, A contains a complex matrix of shape m*n.
               On output, if calc_u is true A contains the column-unitary
               matrix U; otherwise A is simply used as workspace and thus destroyed.

      V        (input/output) if false, the matrix V is not calculated. otherwise
               V must be a matrix of shape n*n.

      calc_u   (input) If true, the matrix U is calculated and replaces A.
               if false, U is not calculated and A is simply destroyed

    return value:
      S        an array of length n containing the singular values of A sorted by
               decreasing magnitude. only the first min(m,n) elements are non-zero.

    This routine is a python translation of the fortran routine svd.f in the
    software library EISPACK (see netlib.org) which itself is based on the
    algol procedure svd described in:
      - num. math. 14, 403-420(1970) by golub and reinsch.
      - wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).

    """

    m, n = A.rows, A.cols

    S = ctx.zeros(n, 1)

    # work is a temporary array of size n
    work  = ctx.zeros(n, 1)
    lbeta = ctx.zeros(n, 1)
    rbeta = ctx.zeros(n, 1)
    dwork = ctx.zeros(n, 1)

    g = scale = anorm = 0
    maxits = 3 * ctx.dps

    for i in xrange(n):         # householder reduction to bidiagonal form
        dwork[i] = scale * g    # dwork are the side-diagonal elements
        g = s = scale = 0
        if i < m:
            for k in xrange(i, m):
                scale += ctx.fabs(ctx.re(A[k,i])) + ctx.fabs(ctx.im(A[k,i]))
            if scale != 0:
                for k in xrange(i, m):
                    A[k,i] /= scale
                    ar = ctx.re(A[k,i])
                    ai = ctx.im(A[k,i])
                    s += ar * ar + ai * ai
                f = A[i,i]
                g = -ctx.sqrt(s)
                if ctx.re(f) < 0:
                    beta = -g - ctx.conj(f)
                    g = -g
                else:
                    beta = -g + ctx.conj(f)
                beta /= ctx.conj(beta)
                beta += 1
                h = 2 * (ctx.re(f) * g - s)
                A[i,i] = f - g
                beta /= h
                lbeta[i] = (beta / scale) / scale
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i, m):
                        s += ctx.conj(A[k,i]) * A[k,j]
                    f = beta * s
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for k in xrange(i, m):
                    A[k,i] *= scale

        S[i] = scale * g     # S are the diagonal elements
        g = s = scale = 0

        if i < m and i != n - 1:
            for k in xrange(i+1, n):
                scale += ctx.fabs(ctx.re(A[i,k])) + ctx.fabs(ctx.im(A[i,k]))
            if scale:
                for k in xrange(i+1, n):
                    A[i,k] /= scale
                    ar = ctx.re(A[i,k])
                    ai = ctx.im(A[i,k])
                    s += ar * ar + ai * ai
                f = A[i,i+1]
                g = -ctx.sqrt(s)
                if ctx.re(f) < 0:
                    beta = -g - ctx.conj(f)
                    g = -g
                else:
                    beta = -g + ctx.conj(f)

                beta /= ctx.conj(beta)
                beta += 1

                h = 2 * (ctx.re(f) * g - s)
                A[i,i+1] = f - g

                beta /= h
                rbeta[i] = (beta / scale) / scale

                for k in xrange(i+1, n):
                    work[k] = A[i, k]

                for j in xrange(i+1, m):
                    s = 0
                    for k in xrange(i+1, n):
                        s += ctx.conj(A[i,k]) * A[j,k]
                    f = s * beta
                    for k in xrange(i+1,n):
                        A[j,k] += f * work[k]

                for k in xrange(i+1, n):
                    A[i,k] *= scale

        anorm = max(anorm,ctx.fabs(S[i]) + ctx.fabs(dwork[i]))

    if not isinstance(V, bool):
        for i in xrange(n-2, -1, -1):     # accumulation of right hand transformations
            V[i+1,i+1] = 1

            if dwork[i+1] != 0:
                f = ctx.conj(rbeta[i])
                for j in xrange(i+1, n):
                    V[i,j] = A[i,j] * f
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, n):
                        s += ctx.conj(A[i,k]) * V[j,k]
                    for k in xrange(i+1, n):
                        V[j,k] += s * V[i,k]

            for j in xrange(i+1,n):
                V[j,i] = V[i,j] = 0

        V[0,0] = 1

    if m < n : minnm = m
    else     : minnm = n

    if calc_u:
        for i in xrange(minnm-1, -1, -1): # accumulation of left hand transformations
            g = S[i]
            for j in xrange(i+1, n):
                A[i,j] = 0
            if g != 0:
                g = 1 / g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, m):
                        s += ctx.conj(A[k,i]) * A[k,j]
                    f = s * ctx.conj(lbeta[i])
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for j in xrange(i, m):
                    A[j,i] *= g
            else:
                for j in xrange(i, m):
                    A[j,i] = 0
            A[i,i] += 1

    for k in xrange(n-1, -1, -1):
        # diagonalization of the bidiagonal form:
        #   loop over singular values, and over allowed itations

        its = 0
        while 1:
            its += 1
            flag = True

            for l in xrange(k, -1, -1):
                nm = l - 1

                if ctx.fabs(dwork[l]) + anorm == anorm:
                    flag = False
                    break

                if ctx.fabs(S[nm]) + anorm == anorm:
                    break

            if flag:
                c = 0
                s = 1
                for i in xrange(l, k+1):
                    f = s * dwork[i]
                    dwork[i] *= c
                    if ctx.fabs(f) + anorm == anorm:
                        break
                    g = S[i]
                    h = ctx.hypot(f, g)
                    S[i] = h
                    h = 1 / h
                    c = g * h
                    s = -f * h

                    if calc_u:
                        for j in xrange(m):
                            y = A[j,nm]
                            z = A[j,i]
                            A[j,nm]= y * c + z * s
                            A[j,i] = z * c - y * s

            z = S[k]

            if l == k:         # convergence
                if z < 0:    # singular value is made nonnegative
                    S[k] = -z
                    if not isinstance(V, bool):
                        for j in xrange(n):
                            V[k,j] = -V[k,j]
                break

            if its >= maxits:
                raise RuntimeError("svd: no convergence to an eigenvalue after %d iterations" % its)

            x = S[l]         # shift from bottom 2 by 2 minor
            nm = k-1
            y = S[nm]
            g = dwork[nm]
            h = dwork[k]
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
            g = ctx.hypot(f, 1)
            if f >=0: f = (( x - z) *( x + z) + h *((y / (f + g)) - h)) / x
            else:     f = (( x - z) *( x + z) + h *((y / (f - g)) - h)) / x

            c = s = 1         # next qt transformation

            for j in xrange(l, nm + 1):
                g = dwork[j+1]
                y = S[j+1]
                h = s * g
                g = c * g
                z = ctx.hypot(f, h)
                dwork[j] = z
                c = f / z
                s = h / z
                f = x * c + g * s
                g = g * c - x * s
                h = y * s
                y *= c
                if not isinstance(V, bool):
                    for jj in xrange(n):
                        x = V[j  ,jj]
                        z = V[j+1,jj]
                        V[j    ,jj]= x * c + z * s
                        V[j+1,jj  ]= z * c - x * s
                z = ctx.hypot(f, h)
                S[j] = z
                if z != 0:            # rotation can be arbitray if z=0
                    z = 1 / z
                    c = f * z
                    s = h * z
                f = c * g + s * y
                x = c * y - s * g
                if calc_u:
                    for jj in xrange(m):
                        y = A[jj,j  ]
                        z = A[jj,j+1]
                        A[jj,j    ]= y * c + z * s
                        A[jj,j+1  ]= z * c - y * s

            dwork[l] = 0
            dwork[k] = f
            S[k] = x

    ##########################

    # Sort singular values into decreasing order (bubble-sort)

    for i in xrange(n):
        imax = i
        s = ctx.fabs(S[i])         # s is the current maximal element

        for j in xrange(i + 1, n):
            c = ctx.fabs(S[j])
            if c > s:
                s = c
                imax = j

        if imax != i:
            # swap singular values

            z = S[i]
            S[i] = S[imax]
            S[imax] = z

            if calc_u:
                for j in xrange(m):
                    z = A[j,i]
                    A[j,i] = A[j,imax]
                    A[j,imax] = z

            if not isinstance(V, bool):
                for j in xrange(n):
                    z = V[i,j]
                    V[i,j] = V[imax,j]
                    V[imax,j] = z

    return S

##################################################################################################

@defun
def svd_r(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal matrices U and V are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the transpose. The diagonal elements of S are the singular
    values of A, i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a real matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an orthogonal matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an orthogonal matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    examples:

       >>> from mpmath import mp
       >>> A = mp.matrix([[2, -2, -1], [3, 4, -2], [-2, -2, 0]])
       >>> S = mp.svd_r(A, compute_uv = False)
       >>> print(S)
       [6.0]
       [3.0]
       [1.0]

       >>> U, S, V = mp.svd_r(A)
       >>> print(mp.chop(A - U * mp.diag(S) * V))
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]


    see also: svd, svd_c
    """

    m, n = A.rows, A.cols

    if not compute_uv:
        if not overwrite_a:
            A = A.copy()
        S = svd_r_raw(ctx, A, V = False, calc_u = False)
        S = S[:min(m,n)]
        return S

    if full_matrices and n < m:
        V = ctx.zeros(m, m)
        A0 = ctx.zeros(m, m)
        A0[:,:n] = A
        S = svd_r_raw(ctx, A0, V, calc_u = True)

        S = S[:n]
        V = V[:n,:n]

        return (A0, S, V)
    else:
        if not overwrite_a:
            A = A.copy()
        V = ctx.zeros(n, n)
        S = svd_r_raw(ctx, A, V, calc_u = True)

        if n > m:
            if full_matrices == False:
                V = V[:m,:]

            S = S[:m]
            A = A[:,:m]

        return (A, S, V)

##############################

@defun
def svd_c(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two unitary matrices U and V are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the hermitian transpose (i.e. transposition and complex
    conjugation). The diagonal elements of S are the singular values of A,
    i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a complex matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an unitary matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an unitary matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[-2j, -1-3j, -2+2j], [2-2j, -1-3j, 1], [-3+1j,-2j,0]])
      >>> S = mp.svd_c(A, compute_uv = False)
      >>> print(mp.chop(S - mp.matrix([mp.sqrt(34), mp.sqrt(15), mp.sqrt(6)])))
      [0.0]
      [0.0]
      [0.0]

      >>> U, S, V = mp.svd_c(A)
      >>> print(mp.chop(A - U * mp.diag(S) * V))
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]

    see also: svd, svd_r
    """

    m, n = A.rows, A.cols

    if not compute_uv:
        if not overwrite_a:
            A = A.copy()
        S = svd_c_raw(ctx, A, V = False, calc_u = False)
        S = S[:min(m,n)]
        return S

    if full_matrices and n < m:
        V = ctx.zeros(m, m)
        A0 = ctx.zeros(m, m)
        A0[:,:n] = A
        S = svd_c_raw(ctx, A0, V, calc_u = True)

        S = S[:n]
        V = V[:n,:n]

        return (A0, S, V)
    else:
        if not overwrite_a:
            A = A.copy()
        V = ctx.zeros(n, n)
        S = svd_c_raw(ctx, A, V, calc_u = True)

        if n > m:
            if full_matrices == False:
                V = V[:m,:]

            S = S[:m]
            A = A[:,:m]

        return (A, S, V)

@defun
def svd(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    "svd" is a unified interface for "svd_r" and "svd_c". Depending on
    whether A is real or complex the appropriate function is called.

    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal (A real) or unitary (A complex) matrices U and V
    are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the hermitian transpose (i.e. transposition and complex
    conjugation). The diagonal elements of S are the singular values of A,
    i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a real or complex matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an orthogonal or unitary matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an orthogonal or unitary matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    examples:

       >>> from mpmath import mp
       >>> A = mp.matrix([[2, -2, -1], [3, 4, -2], [-2, -2, 0]])
       >>> S = mp.svd(A, compute_uv = False)
       >>> print(S)
       [6.0]
       [3.0]
       [1.0]

       >>> U, S, V = mp.svd(A)
       >>> print(mp.chop(A - U * mp.diag(S) * V))
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]

    see also: svd_r, svd_c
    """

    iscomplex = any(type(x) is ctx.mpc for x in A)

    if iscomplex:
        return ctx.svd_c(A, full_matrices = full_matrices, compute_uv = compute_uv, overwrite_a = overwrite_a)
    else:
        return ctx.svd_r(A, full_matrices = full_matrices, compute_uv = compute_uv, overwrite_a = overwrite_a)
<|endoftext|>#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################
#     module for the eigenvalue problem
#       Copyright 2013 Timo Hartmann (thartmann15 at gmail.com)
#
# todo:
#  - implement balancing
#  - agressive early deflation
#
##################################################################################################

"""
The eigenvalue problem
----------------------

This file contains routines for the eigenvalue problem.

high level routines:

  hessenberg : reduction of a real or complex square matrix to upper Hessenberg form
  schur : reduction of a real or complex square matrix to upper Schur form
  eig : eigenvalues and eigenvectors of a real or complex square matrix

low level routines:

  hessenberg_reduce_0 : reduction of a real or complex square matrix to upper Hessenberg form
  hessenberg_reduce_1 : auxiliary routine to hessenberg_reduce_0
  qr_step : a single implicitly shifted QR step for an upper Hessenberg matrix
  hessenberg_qr : Schur decomposition of an upper Hessenberg matrix
  eig_tr_r : right eigenvectors of an upper triangular matrix
  eig_tr_l : left  eigenvectors of an upper triangular matrix
"""

from ..libmp.backend import xrange

class Eigen(object):
    pass

def defun(f):
    setattr(Eigen, f.__name__, f)
    return f

def hessenberg_reduce_0(ctx, A, T):
    """
    This routine computes the (upper) Hessenberg decomposition of a square matrix A.
    Given A, an unitary matrix Q is calculated such that

               Q' A Q = H              and             Q' Q = Q Q' = 1

    where H is an upper Hessenberg matrix, meaning that it only contains zeros
    below the first subdiagonal. Here ' denotes the hermitian transpose (i.e.
    transposition and conjugation).

    parameters:
      A         (input/output) On input, A contains the square matrix A of
                dimension (n,n). On output, A contains a compressed representation
                of Q and H.
      T         (output) An array of length n containing the first elements of
                the Householder reflectors.
    """

    # internally we work with householder reflections from the right.
    # let u be a row vector (i.e. u[i]=A[i,:i]). then
    # Q is build up by reflectors of the type (1-v'v) where v is a suitable
    # modification of u. these reflectors are applyed to A from the right.
    # because we work with reflectors from the right we have to start with
    # the bottom row of A and work then upwards (this corresponds to
    # some kind of RQ decomposition).
    # the first part of the vectors v (i.e. A[i,:(i-1)]) are stored as row vectors
    # in the lower left part of A (excluding the diagonal and subdiagonal).
    # the last entry of v is stored in T.
    # the upper right part of A (including diagonal and subdiagonal) becomes H.


    n = A.rows
    if n <= 2: return

    for i in xrange(n-1, 1, -1):

        # scale the vector

        scale = 0
        for k in xrange(0, i):
            scale += abs(ctx.re(A[i,k])) + abs(ctx.im(A[i,k]))

        scale_inv = 0
        if scale != 0:
            scale_inv = 1 / scale

        if scale == 0 or ctx.isinf(scale_inv):
            # sadly there are floating point numbers not equal to zero whose reciprocal is infinity
            T[i] = 0
            A[i,i-1] = 0
            continue

        # calculate parameters for housholder transformation

        H = 0
        for k in xrange(0, i):
            A[i,k] *= scale_inv
            rr = ctx.re(A[i,k])
            ii = ctx.im(A[i,k])
            H += rr * rr + ii * ii

        F = A[i,i-1]
        f = abs(F)
        G = ctx.sqrt(H)
        A[i,i-1] = - G * scale

        if f == 0:
            T[i] = G
        else:
            ff = F / f
            T[i] = F + G * ff
            A[i,i-1] *= ff

        H += G * f
        H = 1 / ctx.sqrt(H)

        T[i] *= H
        for k in xrange(0, i - 1):
            A[i,k] *= H

        for j in xrange(0, i):
            # apply housholder transformation (from right)

            G = ctx.conj(T[i]) * A[j,i-1]
            for k in xrange(0, i-1):
                G += ctx.conj(A[i,k]) * A[j,k]

            A[j,i-1] -= G * T[i]
            for k in xrange(0, i-1):
                A[j,k] -= G * A[i,k]

        for j in xrange(0, n):
            # apply housholder transformation (from left)

            G = T[i] * A[i-1,j]
            for k in xrange(0, i-1):
                G += A[i,k] * A[k,j]

            A[i-1,j] -= G * ctx.conj(T[i])
            for k in xrange(0, i-1):
                A[k,j] -= G * ctx.conj(A[i,k])



def hessenberg_reduce_1(ctx, A, T):
    """
    This routine forms the unitary matrix Q described in hessenberg_reduce_0.

    parameters:
      A    (input/output) On input, A is the same matrix as delivered by
           hessenberg_reduce_0. On output, A is set to Q.

      T    (input) On input, T is the same array as delivered by hessenberg_reduce_0.
    """

    n = A.rows

    if n == 1:
        A[0,0] = 1
        return

    A[0,0] = A[1,1] = 1
    A[0,1] = A[1,0] = 0

    for i in xrange(2, n):
        if T[i] != 0:

            for j in xrange(0, i):
                G = T[i] * A[i-1,j]
                for k in xrange(0, i-1):
                    G += A[i,k] * A[k,j]

                A[i-1,j] -= G * ctx.conj(T[i])
                for k in xrange(0, i-1):
                    A[k,j] -= G * ctx.conj(A[i,k])

        A[i,i] = 1
        for j in xrange(0, i):
            A[j,i] = A[i,j] = 0



@defun
def hessenberg(ctx, A, overwrite_a = False):
    """
    This routine computes the Hessenberg decomposition of a square matrix A.
    Given A, an unitary matrix Q is determined such that

          Q' A Q = H                and               Q' Q = Q Q' = 1

    where H is an upper right Hessenberg matrix. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation).

    input:
      A            : a real or complex square matrix
      overwrite_a  : if true, allows modification of A which may improve
                     performance. if false, A is not modified.

    output:
      Q : an unitary matrix
      H : an upper right Hessenberg matrix

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> Q, H = mp.hessenberg(A)
      >>> mp.nprint(H, 3) # doctest:+SKIP
      [  3.15  2.23  4.44]
      [-0.769  4.85  3.05]
      [   0.0  3.61   7.0]
      >>> print(mp.chop(A - Q * H * Q.transpose_conj()))
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]

    return value:   (Q, H)
    """

    n = A.rows

    if n == 1:
        return (ctx.matrix([[1]]), A)

    if not overwrite_a:
        A = A.copy()

    T = ctx.matrix(n, 1)

    hessenberg_reduce_0(ctx, A, T)
    Q = A.copy()
    hessenberg_reduce_1(ctx, Q, T)

    for x in xrange(n):
        for y in xrange(x+2, n):
            A[y,x] = 0

    return Q, A


###########################################################################


def qr_step(ctx, n0, n1, A, Q, shift):
    """
    This subroutine executes a single implicitly shifted QR step applied to an
    upper Hessenberg matrix A. Given A and shift as input, first an QR
    decomposition is calculated:

      Q R = A - shift * 1 .

    The output is then following matrix:

      R Q + shift * 1

    parameters:
      n0, n1    (input) Two integers which specify the submatrix A[n0:n1,n0:n1]
                on which this subroutine operators. The subdiagonal elements
                to the left and below this submatrix must be deflated (i.e. zero).
                following restriction is imposed: n1>=n0+2
      A         (input/output) On input, A is an upper Hessenberg matrix.
                On output, A is replaced by "R Q + shift * 1"
      Q         (input/output) The parameter Q is multiplied by the unitary matrix
                Q arising from the QR decomposition. Q can also be false, in which
                case the unitary matrix Q is not computated.
      shift     (input) a complex number specifying the shift. idealy close to an
                eigenvalue of the bottemmost part of the submatrix A[n0:n1,n0:n1].

    references:
      Stoer, Bulirsch - Introduction to Numerical Analysis.
      Kresser : Numerical Methods for General and Structured Eigenvalue Problems
    """

    # implicitly shifted and bulge chasing is explained at p.398/399 in "Stoer, Bulirsch - Introduction to Numerical Analysis"
    # for bulge chasing see also "Watkins - The Matrix Eigenvalue Problem" sec.4.5,p.173

    # the Givens rotation we used is determined as follows: let c,s be two complex
    # numbers. then we have following relation:
    #
    #     v = sqrt(|c|^2 + |s|^2)
    #
    #     1/v [ c~  s~]  [c] = [v]
    #         [-s   c ]  [s]   [0]
    #
    # the matrix on the left is our Givens rotation.

    n = A.rows

    # first step

    # calculate givens rotation
    c = A[n0  ,n0] - shift
    s = A[n0+1,n0]

    v = ctx.hypot(ctx.hypot(ctx.re(c), ctx.im(c)), ctx.hypot(ctx.re(s), ctx.im(s)))

    if v == 0:
        v = 1
        c = 1
        s = 0
    else:
        c /= v
        s /= v

    cc = ctx.conj(c)
    cs = ctx.conj(s)

    for k in xrange(n0, n):
        # apply givens rotation from the left
        x = A[n0  ,k]
        y = A[n0+1,k]
        A[n0  ,k] = cc * x + cs * y
        A[n0+1,k] = c * y - s * x

    for k in xrange(min(n1, n0+3)):
        # apply givens rotation from the right
        x = A[k,n0  ]
        y = A[k,n0+1]
        A[k,n0  ] = c * x + s * y
        A[k,n0+1] = cc * y - cs * x

    if not isinstance(Q, bool):
        for k in xrange(n):
            # eigenvectors
            x = Q[k,n0  ]
            y = Q[k,n0+1]
            Q[k,n0  ] = c * x + s * y
            Q[k,n0+1] = cc * y - cs * x

    # chase the bulge

    for j in xrange(n0, n1 - 2):
        # calculate givens rotation

        c = A[j+1,j]
        s = A[j+2,j]

        v = ctx.hypot(ctx.hypot(ctx.re(c), ctx.im(c)), ctx.hypot(ctx.re(s), ctx.im(s)))

        if v == 0:
            A[j+1,j] = 0
            v = 1
            c = 1
            s = 0
        else:
            A[j+1,j] = v
            c /= v
            s /= v

        A[j+2,j] = 0

        cc = ctx.conj(c)
        cs = ctx.conj(s)

        for k in xrange(j+1, n):
            # apply givens rotation from the left
            x = A[j+1,k]
            y = A[j+2,k]
            A[j+1,k] = cc * x + cs * y
            A[j+2,k] = c * y - s * x

        for k in xrange(0, min(n1, j+4)):
            # apply givens rotation from the right
            x = A[k,j+1]
            y = A[k,j+2]
            A[k,j+1] = c * x + s * y
            A[k,j+2] = cc * y - cs * x

        if not isinstance(Q, bool):
            for k in xrange(0, n):
                # eigenvectors
                x = Q[k,j+1]
                y = Q[k,j+2]
                Q[k,j+1] = c * x + s * y
                Q[k,j+2] = cc * y - cs * x



def hessenberg_qr(ctx, A, Q):
    """
    This routine computes the Schur decomposition of an upper Hessenberg matrix A.
    Given A, an unitary matrix Q is determined such that

          Q' A Q = R                   and                  Q' Q = Q Q' = 1

    where R is an upper right triangular matrix. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation).

    parameters:
      A         (input/output) On input, A contains an upper Hessenberg matrix.
                On output, A is replace by the upper right triangluar matrix R.

      Q         (input/output) The parameter Q is multiplied by the unitary
                matrix Q arising from the Schur decomposition. Q can also be
                false, in which case the unitary matrix Q is not computated.
    """

    n = A.rows

    norm = 0
    for x in xrange(n):
        for y in xrange(min(x+2, n)):
            norm += ctx.re(A[y,x]) ** 2 + ctx.im(A[y,x]) ** 2
    norm = ctx.sqrt(norm) / n

    if norm == 0:
        return

    n0 = 0
    n1 = n

    eps = ctx.eps / (100 * n)
    maxits = ctx.dps * 4

    its = totalits = 0

    while 1:
        # kressner p.32 algo 3
        # the active submatrix is A[n0:n1,n0:n1]

        k = n0

        while k + 1 < n1:
            s = abs(ctx.re(A[k,k])) + abs(ctx.im(A[k,k])) + abs(ctx.re(A[k+1,k+1])) + abs(ctx.im(A[k+1,k+1]))
            if s < eps * norm:
                s = norm
            if abs(A[k+1,k]) < eps * s:
                break
            k += 1

        if k + 1 < n1:
            # deflation found at position (k+1, k)

            A[k+1,k] = 0
            n0 = k + 1

            its = 0

            if n0 + 1 >= n1:
                # block of size at most two has converged
                n0 = 0
                n1 = k + 1
                if n1 < 2:
                    # QR algorithm has converged
                    return
        else:
            if (its % 30) == 10:
                # exceptional shift
                shift = A[n1-1,n1-2]
            elif (its % 30) == 20:
                # exceptional shift
                shift = abs(A[n1-1,n1-2])
            elif (its % 30) == 29:
                # exceptional shift
                shift = norm
            else:
                #    A = [ a b ]       det(x-A)=x*x-x*tr(A)+det(A)
                #        [ c d ]
                #
                # eigenvalues bad:   (tr(A)+sqrt((tr(A))**2-4*det(A)))/2
                #     bad because of cancellation if |c| is small and |a-d| is small, too.
                #
                # eigenvalues good:     (a+d+sqrt((a-d)**2+4*b*c))/2

                t =  A[n1-2,n1-2] + A[n1-1,n1-1]
                s = (A[n1-1,n1-1] - A[n1-2,n1-2]) ** 2 + 4 * A[n1-1,n1-2] * A[n1-2,n1-1]
                if ctx.re(s) > 0:
                    s = ctx.sqrt(s)
                else:
                    s = ctx.sqrt(-s) * 1j
                a = (t + s) / 2
                b = (t - s) / 2
                if abs(A[n1-1,n1-1] - a) > abs(A[n1-1,n1-1] - b):
                    shift = b
                else:
                    shift = a

            its += 1
            totalits += 1

            qr_step(ctx, n0, n1, A, Q, shift)

            if its > maxits:
                raise RuntimeError("qr: failed to converge after %d steps" % its)


@defun
def schur(ctx, A, overwrite_a = False):
    """
    This routine computes the Schur decomposition of a square matrix A.
    Given A, an unitary matrix Q is determined such that

          Q' A Q = R                and               Q' Q = Q Q' = 1

    where R is an upper right triangular matrix. Here ' denotes the
    hermitian transpose (i.e. transposition and conjugation).

    input:
      A            : a real or complex square matrix
      overwrite_a  : if true, allows modification of A which may improve
                     performance. if false, A is not modified.

    output:
      Q : an unitary matrix
      R : an upper right triangular matrix

    return value:   (Q, R)

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> Q, R = mp.schur(A)
      >>> mp.nprint(R, 3) # doctest:+SKIP
      [2.0  0.417  -2.53]
      [0.0    4.0  -4.74]
      [0.0    0.0    9.0]
      >>> print(mp.chop(A - Q * R * Q.transpose_conj()))
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]

    warning: The Schur decomposition is not unique.
    """

    n = A.rows

    if n == 1:
        return (ctx.matrix([[1]]), A)

    if not overwrite_a:
        A = A.copy()

    T = ctx.matrix(n, 1)

    hessenberg_reduce_0(ctx, A, T)
    Q = A.copy()
    hessenberg_reduce_1(ctx, Q, T)

    for x in xrange(n):
        for y in xrange(x + 2, n):
            A[y,x] = 0

    hessenberg_qr(ctx, A, Q)

    return Q, A


def eig_tr_r(ctx, A):
    """
    This routine calculates the right eigenvectors of an upper right triangular matrix.

    input:
      A      an upper right triangular matrix

    output:
      ER     a matrix whose columns form the right eigenvectors of A

    return value: ER
    """

    # this subroutine is inspired by the lapack routines ctrevc.f,clatrs.f

    n = A.rows

    ER = ctx.eye(n)

    eps = ctx.eps

    unfl = ctx.ldexp(ctx.one, -ctx.prec * 30)
    # since mpmath effectively has no limits on the exponent, we simply scale doubles up
    # original double has prec*20

    smlnum = unfl * (n / eps)
    simin = 1 / ctx.sqrt(eps)

    rmax = 1

    for i in xrange(1, n):
        s = A[i,i]

        smin = max(eps * abs(s), smlnum)

        for j in xrange(i - 1, -1, -1):

            r = 0
            for k in xrange(j + 1, i + 1):
                r += A[j,k] * ER[k,i]

            t = A[j,j] - s
            if abs(t) < smin:
                t = smin

            r = -r / t
            ER[j,i] = r

            rmax = max(rmax, abs(r))
            if rmax > simin:
                for k in xrange(j, i+1):
                    ER[k,i] /= rmax
                rmax = 1

        if rmax != 1:
            for k in xrange(0, i + 1):
                ER[k,i] /= rmax

    return ER

def eig_tr_l(ctx, A):
    """
    This routine calculates the left eigenvectors of an upper right triangular matrix.

    input:
      A      an upper right triangular matrix

    output:
      EL     a matrix whose rows form the left eigenvectors of A

    return value:  EL
    """

    n = A.rows

    EL = ctx.eye(n)

    eps = ctx.eps

    unfl = ctx.ldexp(ctx.one, -ctx.prec * 30)
    # since mpmath effectively has no limits on the exponent, we simply scale doubles up
    # original double has prec*20

    smlnum = unfl * (n / eps)
    simin = 1 / ctx.sqrt(eps)

    rmax = 1

    for i in xrange(0, n - 1):
        s = A[i,i]

        smin = max(eps * abs(s), smlnum)

        for j in xrange(i + 1, n):

            r = 0
            for k in xrange(i, j):
                r += EL[i,k] * A[k,j]

            t = A[j,j] - s
            if abs(t) < smin:
                t = smin

            r = -r / t
            EL[i,j] = r

            rmax = max(rmax, abs(r))
            if rmax > simin:
                for k in xrange(i, j + 1):
                    EL[i,k] /= rmax
                rmax = 1

        if rmax != 1:
            for k in xrange(i, n):
                EL[i,k] /= rmax

    return EL

@defun
def eig(ctx, A, left = False, right = True, overwrite_a = False):
    """
    This routine computes the eigenvalues and optionally the left and right
    eigenvectors of a square matrix A. Given A, a vector E and matrices ER
    and EL are calculated such that

                        A ER[:,i] =         E[i] ER[:,i]
                EL[i,:] A         = EL[i,:] E[i]

    E contains the eigenvalues of A. The columns of ER contain the right eigenvectors
    of A whereas the rows of EL contain the left eigenvectors.


    input:
      A           : a real or complex square matrix of shape (n, n)
      left        : if true, the left eigenvectors are calculated.
      right       : if true, the right eigenvectors are calculated.
      overwrite_a : if true, allows modification of A which may improve
                    performance. if false, A is not modified.

    output:
      E    : a list of length n containing the eigenvalues of A.
      ER   : a matrix whose columns contain the right eigenvectors of A.
      EL   : a matrix whose rows contain the left eigenvectors of A.

    return values:
       E            if left and right are both false.
      (E, ER)       if right is true and left is false.
      (E, EL)       if left is true and right is false.
      (E, EL, ER)   if left and right are true.


    examples:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> E, ER = mp.eig(A)
      >>> print(mp.chop(A * ER[:,0] - E[0] * ER[:,0]))
      [0.0]
      [0.0]
      [0.0]

      >>> E, EL, ER = mp.eig(A,left = True, right = True)
      >>> E, EL, ER = mp.eig_sort(E, EL, ER)
      >>> mp.nprint(E)
      [2.0, 4.0, 9.0]
      >>> print(mp.chop(A * ER[:,0] - E[0] * ER[:,0]))
      [0.0]
      [0.0]
      [0.0]
      >>> print(mp.chop( EL[0,:] * A - EL[0,:] * E[0]))
      [0.0  0.0  0.0]

    warning:
     - If there are multiple eigenvalues, the eigenvectors do not necessarily
       span the whole vectorspace, i.e. ER and EL may have not full rank.
       Furthermore in that case the eigenvectors are numerical ill-conditioned.
     - In the general case the eigenvalues have no natural order.

    see also:
      - eigh (or eigsy, eighe) for the symmetric eigenvalue problem.
      - eig_sort for sorting of eigenvalues and eigenvectors
    """

    n = A.rows

    if n == 1:
        if left and (not right):
            return ([A[0]], ctx.matrix([[1]]))

        if right and (not left):
            return ([A[0]], ctx.matrix([[1]]))

        return ([A[0]], ctx.matrix([[1]]), ctx.matrix([[1]]))

    if not overwrite_a:
        A = A.copy()

    T = ctx.zeros(n, 1)

    hessenberg_reduce_0(ctx, A, T)

    if left or right:
        Q = A.copy()
        hessenberg_reduce_1(ctx, Q, T)
    else:
        Q = False

    for x in xrange(n):
        for y in xrange(x + 2, n):
            A[y,x] = 0

    hessenberg_qr(ctx, A, Q)

    E = [0 for i in xrange(n)]
    for i in xrange(n):
        E[i] = A[i,i]

    if not (left or right):
        return E

    if left:
        EL = eig_tr_l(ctx, A)
        EL = EL * Q.transpose_conj()

    if right:
        ER = eig_tr_r(ctx, A)
        ER = Q * ER

    if left and (not right):
        return (E, EL)

    if right and (not left):
        return (E, ER)

    return (E, EL, ER)

@defun
def eig_sort(ctx, E, EL = False, ER = False, f = "real"):
    """
    This routine sorts the eigenvalues and eigenvectors delivered by ``eig``.

    parameters:
      E  : the eigenvalues as delivered by eig
      EL : the left  eigenvectors as delivered by eig, or false
      ER : the right eigenvectors as delivered by eig, or false
      f  : either a string ("real" sort by increasing real part, "imag" sort by
           increasing imag part, "abs" sort by absolute value) or a function
           mapping complexs to the reals, i.e. ``f = lambda x: -mp.re(x) ``
           would sort the eigenvalues by decreasing real part.

    return values:
       E            if EL and ER are both false.
      (E, ER)       if ER is not false and left is false.
      (E, EL)       if EL is not false and right is false.
      (E, EL, ER)   if EL and ER are not false.

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> E, EL, ER = mp.eig(A,left = True, right = True)
      >>> E, EL, ER = mp.eig_sort(E, EL, ER)
      >>> mp.nprint(E)
      [2.0, 4.0, 9.0]
      >>> E, EL, ER = mp.eig_sort(E, EL, ER,f = lambda x: -mp.re(x))
      >>> mp.nprint(E)
      [9.0, 4.0, 2.0]
      >>> print(mp.chop(A * ER[:,0] - E[0] * ER[:,0]))
      [0.0]
      [0.0]
      [0.0]
      >>> print(mp.chop( EL[0,:] * A - EL[0,:] * E[0]))
      [0.0  0.0  0.0]
    """

    if isinstance(f, str):
        if f == "real":
            f = ctx.re
        elif f == "imag":
            f = ctx.im
        elif f == "abs":
            f = abs
        else:
            raise RuntimeError("unknown function %s" % f)

    n = len(E)

    # Sort eigenvalues (bubble-sort)

    for i in xrange(n):
        imax = i
        s = f(E[i])         # s is the current maximal element

        for j in xrange(i + 1, n):
            c = f(E[j])
            if c < s:
                s = c
                imax = j

        if imax != i:
            # swap eigenvalues

            z = E[i]
            E[i] = E[imax]
            E[imax] = z

            if not isinstance(EL, bool):
                for j in xrange(n):
                    z = EL[i,j]
                    EL[i,j] = EL[imax,j]
                    EL[imax,j] = z

            if not isinstance(ER, bool):
                for j in xrange(n):
                    z = ER[j,i]
                    ER[j,i] = ER[j,imax]
                    ER[j,imax] = z

    if isinstance(EL, bool) and isinstance(ER, bool):
        return E

    if isinstance(EL, bool) and not(isinstance(ER, bool)):
        return (E, ER)

    if isinstance(ER, bool) and not(isinstance(EL, bool)):
        return (E, EL)

    return (E, EL, ER)
<|endoftext|>from ..libmp.backend import xrange
import warnings

# TODO: interpret list as vectors (for multiplication)

rowsep = '\n'
colsep = '  '

class _matrix(object):
    """
    Numerical matrix.

    Specify the dimensions or the data as a nested list.
    Elements default to zero.
    Use a flat list to create a column vector easily.

    The datatype of the context (mpf for mp, mpi for iv, and float for fp) is used to store the data.

    Creating matrices
    -----------------

    Matrices in mpmath are implemented using dictionaries. Only non-zero values
    are stored, so it is cheap to represent sparse matrices.

    The most basic way to create one is to use the ``matrix`` class directly.
    You can create an empty matrix specifying the dimensions:

        >>> from mpmath import *
        >>> mp.dps = 15
        >>> matrix(2)
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])
        >>> matrix(2, 3)
        matrix(
        [['0.0', '0.0', '0.0'],
         ['0.0', '0.0', '0.0']])

    Calling ``matrix`` with one dimension will create a square matrix.

    To access the dimensions of a matrix, use the ``rows`` or ``cols`` keyword:

        >>> A = matrix(3, 2)
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0'],
         ['0.0', '0.0']])
        >>> A.rows
        3
        >>> A.cols
        2

    You can also change the dimension of an existing matrix. This will set the
    new elements to 0. If the new dimension is smaller than before, the
    concerning elements are discarded:

        >>> A.rows = 2
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])

    Internally ``mpmathify`` is used every time an element is set. This
    is done using the syntax A[row,column], counting from 0:

        >>> A = matrix(2)
        >>> A[1,1] = 1 + 1j
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', mpc(real='1.0', imag='1.0')]])

    A more comfortable way to create a matrix lets you use nested lists:

        >>> matrix([[1, 2], [3, 4]])
        matrix(
        [['1.0', '2.0'],
         ['3.0', '4.0']])

    Convenient advanced functions are available for creating various standard
    matrices, see ``zeros``, ``ones``, ``diag``, ``eye``, ``randmatrix`` and
    ``hilbert``.

    Vectors
    .......

    Vectors may also be represented by the ``matrix`` class (with rows = 1 or cols = 1).
    For vectors there are some things which make life easier. A column vector can
    be created using a flat list, a row vectors using an almost flat nested list::

        >>> matrix([1, 2, 3])
        matrix(
        [['1.0'],
         ['2.0'],
         ['3.0']])
        >>> matrix([[1, 2, 3]])
        matrix(
        [['1.0', '2.0', '3.0']])

    Optionally vectors can be accessed like lists, using only a single index::

        >>> x = matrix([1, 2, 3])
        >>> x[1]
        mpf('2.0')
        >>> x[1,0]
        mpf('2.0')

    Other
    .....

    Like you probably expected, matrices can be printed::

        >>> print randmatrix(3) # doctest:+SKIP
        [ 0.782963853573023  0.802057689719883  0.427895717335467]
        [0.0541876859348597  0.708243266653103  0.615134039977379]
        [ 0.856151514955773  0.544759264818486  0.686210904770947]

    Use ``nstr`` or ``nprint`` to specify the number of digits to print::

        >>> nprint(randmatrix(5), 3) # doctest:+SKIP
        [2.07e-1  1.66e-1  5.06e-1  1.89e-1  8.29e-1]
        [6.62e-1  6.55e-1  4.47e-1  4.82e-1  2.06e-2]
        [4.33e-1  7.75e-1  6.93e-2  2.86e-1  5.71e-1]
        [1.01e-1  2.53e-1  6.13e-1  3.32e-1  2.59e-1]
        [1.56e-1  7.27e-2  6.05e-1  6.67e-2  2.79e-1]

    As matrices are mutable, you will need to copy them sometimes::

        >>> A = matrix(2)
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])
        >>> B = A.copy()
        >>> B[0,0] = 1
        >>> B
        matrix(
        [['1.0', '0.0'],
         ['0.0', '0.0']])
        >>> A
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])

    Finally, it is possible to convert a matrix to a nested list. This is very useful,
    as most Python libraries involving matrices or arrays (namely NumPy or SymPy)
    support this format::

        >>> B.tolist()
        [[mpf('1.0'), mpf('0.0')], [mpf('0.0'), mpf('0.0')]]


    Matrix operations
    -----------------

    You can add and subtract matrices of compatible dimensions::

        >>> A = matrix([[1, 2], [3, 4]])
        >>> B = matrix([[-2, 4], [5, 9]])
        >>> A + B
        matrix(
        [['-1.0', '6.0'],
         ['8.0', '13.0']])
        >>> A - B
        matrix(
        [['3.0', '-2.0'],
         ['-2.0', '-5.0']])
        >>> A + ones(3) # doctest:+ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: incompatible dimensions for addition

    It is possible to multiply or add matrices and scalars. In the latter case the
    operation will be done element-wise::

        >>> A * 2
        matrix(
        [['2.0', '4.0'],
         ['6.0', '8.0']])
        >>> A / 4
        matrix(
        [['0.25', '0.5'],
         ['0.75', '1.0']])
        >>> A - 1
        matrix(
        [['0.0', '1.0'],
         ['2.0', '3.0']])

    Of course you can perform matrix multiplication, if the dimensions are
    compatible, using ``@`` (for Python >= 3.5) or ``*``. For clarity, ``@`` is
    recommended (`PEP 465 <https://www.python.org/dev/peps/pep-0465/>`), because
    the meaning of ``*`` is different in many other Python libraries such as NumPy.

        >>> A @ B # doctest:+SKIP
        matrix(
        [['8.0', '22.0'],
         ['14.0', '48.0']])
        >>> A * B # same as A @ B
        matrix(
        [['8.0', '22.0'],
         ['14.0', '48.0']])
        >>> matrix([[1, 2, 3]]) * matrix([[-6], [7], [-2]])
        matrix(
        [['2.0']])

    ..
        COMMENT: TODO: the above "doctest:+SKIP" may be removed as soon as we
        have dropped support for Python 3.5 and below.

    You can raise powers of square matrices::

        >>> A**2
        matrix(
        [['7.0', '10.0'],
         ['15.0', '22.0']])

    Negative powers will calculate the inverse::

        >>> A**-1
        matrix(
        [['-2.0', '1.0'],
         ['1.5', '-0.5']])
        >>> A * A**-1
        matrix(
        [['1.0', '1.0842021724855e-19'],
         ['-2.16840434497101e-19', '1.0']])



    Matrix transposition is straightforward::

        >>> A = ones(2, 3)
        >>> A
        matrix(
        [['1.0', '1.0', '1.0'],
         ['1.0', '1.0', '1.0']])
        >>> A.T
        matrix(
        [['1.0', '1.0'],
         ['1.0', '1.0'],
         ['1.0', '1.0']])

    Norms
    .....

    Sometimes you need to know how "large" a matrix or vector is. Due to their
    multidimensional nature it's not possible to compare them, but there are
    several functions to map a matrix or a vector to a positive real number, the
    so called norms.

    For vectors the p-norm is intended, usually the 1-, the 2- and the oo-norm are
    used.

        >>> x = matrix([-10, 2, 100])
        >>> norm(x, 1)
        mpf('112.0')
        >>> norm(x, 2)
        mpf('100.5186549850325')
        >>> norm(x, inf)
        mpf('100.0')

    Please note that the 2-norm is the most used one, though it is more expensive
    to calculate than the 1- or oo-norm.

    It is possible to generalize some vector norms to matrix norm::

        >>> A = matrix([[1, -1000], [100, 50]])
        >>> mnorm(A, 1)
        mpf('1050.0')
        >>> mnorm(A, inf)
        mpf('1001.0')
        >>> mnorm(A, 'F')
        mpf('1006.2310867787777')

    The last norm (the "Frobenius-norm") is an approximation for the 2-norm, which
    is hard to calculate and not available. The Frobenius-norm lacks some
    mathematical properties you might expect from a norm.
    """

    def __init__(self, *args, **kwargs):
        self.__data = {}
        # LU decompostion cache, this is useful when solving the same system
        # multiple times, when calculating the inverse and when calculating the
        # determinant
        self._LU = None
        if "force_type" in kwargs:
            warnings.warn("The force_type argument was removed, it did not work"
                " properly anyway. If you want to force floating-point or"
                " interval computations, use the respective methods from `fp`"
                " or `mp` instead, e.g., `fp.matrix()` or `iv.matrix()`."
                " If you want to truncate values to integer, use .apply(int) instead.")
        if isinstance(args[0], (list, tuple)):
            if isinstance(args[0][0], (list, tuple)):
                # interpret nested list as matrix
                A = args[0]
                self.__rows = len(A)
                self.__cols = len(A[0])
                for i, row in enumerate(A):
                    for j, a in enumerate(row):
                        # note: this will call __setitem__ which will call self.ctx.convert() to convert the datatype.
                        self[i, j] = a
            else:
                # interpret list as row vector
                v = args[0]
                self.__rows = len(v)
                self.__cols = 1
                for i, e in enumerate(v):
                    self[i, 0] = e
        elif isinstance(args[0], int):
            # create empty matrix of given dimensions
            if len(args) == 1:
                self.__rows = self.__cols = args[0]
            else:
                if not isinstance(args[1], int):
                    raise TypeError("expected int")
                self.__rows = args[0]
                self.__cols = args[1]
        elif isinstance(args[0], _matrix):
            A = args[0]
            self.__rows = A._matrix__rows
            self.__cols = A._matrix__cols
            for i in xrange(A.__rows):
                for j in xrange(A.__cols):
                    self[i, j] = A[i, j]
        elif hasattr(args[0], 'tolist'):
            A = self.ctx.matrix(args[0].tolist())
            self.__data = A._matrix__data
            self.__rows = A._matrix__rows
            self.__cols = A._matrix__cols
        else:
            raise TypeError('could not interpret given arguments')

    def apply(self, f):
        """
        Return a copy of self with the function `f` applied elementwise.
        """
        new = self.ctx.matrix(self.__rows, self.__cols)
        for i in xrange(self.__rows):
            for j in xrange(self.__cols):
                new[i,j] = f(self[i,j])
        return new

    def __nstr__(self, n=None, **kwargs):
        # Build table of string representations of the elements
        res = []
        # Track per-column max lengths for pretty alignment
        maxlen = [0] * self.cols
        for i in range(self.rows):
            res.append([])
            for j in range(self.cols):
                if n:
                    string = self.ctx.nstr(self[i,j], n, **kwargs)
                else:
                    string = str(self[i,j])
                res[-1].append(string)
                maxlen[j] = max(len(string), maxlen[j])
        # Patch strings together
        for i, row in enumerate(res):
            for j, elem in enumerate(row):
                # Pad each element up to maxlen so the columns line up
                row[j] = elem.rjust(maxlen[j])
            res[i] = "[" + colsep.join(row) + "]"
        return rowsep.join(res)

    def __str__(self):
        return self.__nstr__()

    def _toliststr(self, avoid_type=False):
        """
        Create a list string from a matrix.

        If avoid_type: avoid multiple 'mpf's.
        """
        # XXX: should be something like self.ctx._types
        typ = self.ctx.mpf
        s = '['
        for i in xrange(self.__rows):
            s += '['
            for j in xrange(self.__cols):
                if not avoid_type or not isinstance(self[i,j], typ):
                    a = repr(self[i,j])
                else:
                    a = "'" + str(self[i,j]) + "'"
                s += a + ', '
            s = s[:-2]
            s += '],\n '
        s = s[:-3]
        s += ']'
        return s

    def tolist(self):
        """
        Convert the matrix to a nested list.
        """
        return [[self[i,j] for j in range(self.__cols)] for i in range(self.__rows)]

    def __repr__(self):
        if self.ctx.pretty:
            return self.__str__()
        s = 'matrix(\n'
        s += self._toliststr(avoid_type=True) + ')'
        return s

    def __get_element(self, key):
        '''
        Fast extraction of the i,j element from the matrix
            This function is for private use only because is unsafe:
                1. Does not check on the value of key it expects key to be a integer tuple (i,j)
                2. Does not check bounds
        '''
        if key in self.__data:
            return self.__data[key]
        else:
            return self.ctx.zero

    def __set_element(self, key, value):
        '''
        Fast assignment of the i,j element in the matrix
            This function is unsafe:
                1. Does not check on the value of key it expects key to be a integer tuple (i,j)
                2. Does not check bounds
                3. Does not check the value type
                4. Does not reset the LU cache
        '''
        if value: # only store non-zeros
            self.__data[key] = value
        elif key in self.__data:
            del self.__data[key]


    def __getitem__(self, key):
        '''
            Getitem function for mp matrix class with slice index enabled
            it allows the following assingments
            scalar to a slice of the matrix
         B = A[:,2:6]
        '''
        # Convert vector to matrix indexing
        if isinstance(key, int) or isinstance(key,slice):
            # only sufficent for vectors
            if self.__rows == 1:
                key = (0, key)
            elif self.__cols == 1:
                key = (key, 0)
            else:
                raise IndexError('insufficient indices for matrix')

        if isinstance(key[0],slice) or isinstance(key[1],slice):

            #Rows
            if isinstance(key[0],slice):
                #Check bounds
                if (key[0].start is None or key[0].start >= 0) and \
                    (key[0].stop is None or key[0].stop <= self.__rows+1):
                    # Generate indices
                    rows = xrange(*key[0].indices(self.__rows))
                else:
                    raise IndexError('Row index out of bounds')
            else:
                # Single row
                rows = [key[0]]

            # Columns
            if isinstance(key[1],slice):
                # Check bounds
                if (key[1].start is None or key[1].start >= 0) and \
                    (key[1].stop is None or key[1].stop <= self.__cols+1):
                    # Generate indices
                    columns = xrange(*key[1].indices(self.__cols))
                else:
                    raise IndexError('Column index out of bounds')

            else:
                # Single column
                columns = [key[1]]

            # Create matrix slice
            m = self.ctx.matrix(len(rows),len(columns))

            # Assign elements to the output matrix
            for i,x in enumerate(rows):
                for j,y in enumerate(columns):
                    m.__set_element((i,j),self.__get_element((x,y)))

            return m

        else:
            # single element extraction
            if key[0] >= self.__rows or key[1] >= self.__cols:
                raise IndexError('matrix index out of range')
            if key in self.__data:
                return self.__data[key]
            else:
                return self.ctx.zero

    def __setitem__(self, key, value):
        # setitem function for mp matrix class with slice index enabled
        # it allows the following assingments
        #  scalar to a slice of the matrix
        # A[:,2:6] = 2.5
        #  submatrix to matrix (the value matrix should be the same size as the slice size)
        # A[3,:] = B   where A is n x m  and B is n x 1
        # Convert vector to matrix indexing
        if isinstance(key, int) or isinstance(key,slice):
            # only sufficent for vectors
            if self.__rows == 1:
                key = (0, key)
            elif self.__cols == 1:
                key = (key, 0)
            else:
                raise IndexError('insufficient indices for matrix')
        # Slice indexing
        if isinstance(key[0],slice) or isinstance(key[1],slice):
            # Rows
            if isinstance(key[0],slice):
                # Check bounds
                if (key[0].start is None or key[0].start >= 0) and \
                    (key[0].stop is None or key[0].stop <= self.__rows+1):
                    # generate row indices
                    rows = xrange(*key[0].indices(self.__rows))
                else:
                    raise IndexError('Row index out of bounds')
            else:
                # Single row
                rows = [key[0]]
            # Columns
            if isinstance(key[1],slice):
                # Check bounds
                if (key[1].start is None or key[1].start >= 0) and \
                    (key[1].stop is None or key[1].stop <= self.__cols+1):
                    # Generate column indices
                    columns = xrange(*key[1].indices(self.__cols))
                else:
                    raise IndexError('Column index out of bounds')
            else:
                # Single column
                columns = [key[1]]
            # Assign slice with a scalar
            if isinstance(value,self.ctx.matrix):
                # Assign elements to matrix if input and output dimensions match
                if len(rows) == value.rows and len(columns) == value.cols:
                    for i,x in enumerate(rows):
                        for j,y in enumerate(columns):
                            self.__set_element((x,y), value.__get_element((i,j)))
                else:
                    raise ValueError('Dimensions do not match')
            else:
                # Assign slice with scalars
                value = self.ctx.convert(value)
                for i in rows:
                    for j in columns:
                        self.__set_element((i,j), value)
        else:
            # Single element assingment
            # Check bounds
            if key[0] >= self.__rows or key[1] >= self.__cols:
                raise IndexError('matrix index out of range')
            # Convert and store value
            value = self.ctx.convert(value)
            if value: # only store non-zeros
                self.__data[key] = value
            elif key in self.__data:
                del self.__data[key]

        if self._LU:
            self._LU = None
        return

    def __iter__(self):
        for i in xrange(self.__rows):
            for j in xrange(self.__cols):
                yield self[i,j]

    def __mul__(self, other):
        if isinstance(other, self.ctx.matrix):
            # dot multiplication
            if self.__cols != other.__rows:
                raise ValueError('dimensions not compatible for multiplication')
            new = self.ctx.matrix(self.__rows, other.__cols)
            self_zero = self.ctx.zero
            self_get = self.__data.get
            other_zero = other.ctx.zero
            other_get = other.__data.get
            for i in xrange(self.__rows):
                for j in xrange(other.__cols):
                    new[i, j] = self.ctx.fdot((self_get((i,k), self_zero), other_get((k,j), other_zero))
                                     for k in xrange(other.__rows))
            return new
        else:
            # try scalar multiplication
            new = self.ctx.matrix(self.__rows, self.__cols)
            for i in xrange(self.__rows):
                for j in xrange(self.__cols):
                    new[i, j] = other * self[i, j]
            return new

    def __matmul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        # assume other is scalar and thus commutative
        if isinstance(other, self.ctx.matrix):
            raise TypeError("other should not be type of ctx.matrix")
        return self.__mul__(other)

    def __pow__(self, other):
        # avoid cyclic import problems
        #from linalg import inverse
        if not isinstance(other, int):
            raise ValueError('only integer exponents are supported')
        if not self.__rows == self.__cols:
            raise ValueError('only powers of square matrices are defined')
        n = other
        if n == 0:
            return self.ctx.eye(self.__rows)
        if n < 0:
            n = -n
            neg = True
        else:
            neg = False
        i = n
        y = 1
        z = self.copy()
        while i != 0:
            if i % 2 == 1:
                y = y * z
            z = z*z
            i = i // 2
        if neg:
            y = self.ctx.inverse(y)
        return y

    def __div__(self, other):
        # assume other is scalar and do element-wise divison
        assert not isinstance(other, self.ctx.matrix)
        new = self.ctx.matrix(self.__rows, self.__cols)
        for i in xrange(self.__rows):
            for j in xrange(self.__cols):
                new[i,j] = self[i,j] / other
        return new

    __truediv__ = __div__

    def __add__(self, other):
        if isinstance(other, self.ctx.matrix):
            if not (self.__rows == other.__rows and self.__cols == other.__cols):
                raise ValueError('incompatible dimensions for addition')
            new = self.ctx.matrix(self.__rows, self.__cols)
            for i in xrange(self.__rows):
                for j in xrange(self.__cols):
                    new[i,j] = self[i,j] + other[i,j]
            return new
        else:
            # assume other is scalar and add element-wise
            new = self.ctx.matrix(self.__rows, self.__cols)
            for i in xrange(self.__rows):
                for j in xrange(self.__cols):
                    new[i,j] += self[i,j] + other
            return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, self.ctx.matrix) and not (self.__rows == other.__rows
                                              and self.__cols == other.__cols):
            raise ValueError('incompatible dimensions for subtraction')
        return self.__add__(other * (-1))

    def __pos__(self):
        """
        +M returns a copy of M, rounded to current working precision.
        """
        return (+1) * self

    def __neg__(self):
        return (-1) * self

    def __rsub__(self, other):
        return -self + other

    def __eq__(self, other):
        return self.__rows == other.__rows and self.__cols == other.__cols \
               and self.__data == other.__data

    def __len__(self):
        if self.rows == 1:
            return self.cols
        elif self.cols == 1:
            return self.rows
        else:
            return self.rows # do it like numpy

    def __getrows(self):
        return self.__rows

    def __setrows(self, value):
        for key in self.__data.copy():
            if key[0] >= value:
                del self.__data[key]
        self.__rows = value

    rows = property(__getrows, __setrows, doc='number of rows')

    def __getcols(self):
        return self.__cols

    def __setcols(self, value):
        for key in self.__data.copy():
            if key[1] >= value:
                del self.__data[key]
        self.__cols = value

    cols = property(__getcols, __setcols, doc='number of columns')

    def transpose(self):
        new = self.ctx.matrix(self.__cols, self.__rows)
        for i in xrange(self.__rows):
            for j in xrange(self.__cols):
                new[j,i] = self[i,j]
        return new

    T = property(transpose)

    def conjugate(self):
        return self.apply(self.ctx.conj)

    def transpose_conj(self):
        return self.conjugate().transpose()

    H = property(transpose_conj)

    def copy(self):
        new = self.ctx.matrix(self.__rows, self.__cols)
        new.__data = self.__data.copy()
        return new

    __copy__ = copy

    def column(self, n):
        m = self.ctx.matrix(self.rows, 1)
        for i in range(self.rows):
            m[i] = self[i,n]
        return m

class MatrixMethods(object):

    def __init__(ctx):
        # XXX: subclass
        ctx.matrix = type('matrix', (_matrix,), {})
        ctx.matrix.ctx = ctx
        ctx.matrix.convert = ctx.convert

    def eye(ctx, n, **kwargs):
        """
        Create square identity matrix n x n.
        """
        A = ctx.matrix(n, **kwargs)
        for i in xrange(n):
            A[i,i] = 1
        return A

    def diag(ctx, diagonal, **kwargs):
        """
        Create square diagonal matrix using given list.

        Example:
        >>> from mpmath import diag, mp
        >>> mp.pretty = False
        >>> diag([1, 2, 3])
        matrix(
        [['1.0', '0.0', '0.0'],
         ['0.0', '2.0', '0.0'],
         ['0.0', '0.0', '3.0']])
        """
        A = ctx.matrix(len(diagonal), **kwargs)
        for i in xrange(len(diagonal)):
            A[i,i] = diagonal[i]
        return A

    def zeros(ctx, *args, **kwargs):
        """
        Create matrix m x n filled with zeros.
        One given dimension will create square matrix n x n.

        Example:
        >>> from mpmath import zeros, mp
        >>> mp.pretty = False
        >>> zeros(2)
        matrix(
        [['0.0', '0.0'],
         ['0.0', '0.0']])
        """
        if len(args) == 1:
            m = n = args[0]
        elif len(args) == 2:
            m = args[0]
            n = args[1]
        else:
            raise TypeError('zeros expected at most 2 arguments, got %i' % len(args))
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i,j] = 0
        return A

    def ones(ctx, *args, **kwargs):
        """
        Create matrix m x n filled with ones.
        One given dimension will create square matrix n x n.

        Example:
        >>> from mpmath import ones, mp
        >>> mp.pretty = False
        >>> ones(2)
        matrix(
        [['1.0', '1.0'],
         ['1.0', '1.0']])
        """
        if len(args) == 1:
            m = n = args[0]
        elif len(args) == 2:
            m = args[0]
            n = args[1]
        else:
            raise TypeError('ones expected at most 2 arguments, got %i' % len(args))
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i,j] = 1
        return A

    def hilbert(ctx, m, n=None):
        """
        Create (pseudo) hilbert matrix m x n.
        One given dimension will create hilbert matrix n x n.

        The matrix is very ill-conditioned and symmetric, positive definite if
        square.
        """
        if n is None:
            n = m
        A = ctx.matrix(m, n)
        for i in xrange(m):
            for j in xrange(n):
                A[i,j] = ctx.one / (i + j + 1)
        return A

    def randmatrix(ctx, m, n=None, min=0, max=1, **kwargs):
        """
        Create a random m x n matrix.

        All values are >= min and <max.
        n defaults to m.

        Example:
        >>> from mpmath import randmatrix
        >>> randmatrix(2) # doctest:+SKIP
        matrix(
        [['0.53491598236191806', '0.57195669543302752'],
         ['0.85589992269513615', '0.82444367501382143']])
        """
        if not n:
            n = m
        A = ctx.matrix(m, n, **kwargs)
        for i in xrange(m):
            for j in xrange(n):
                A[i,j] = ctx.rand() * (max - min) + min
        return A

    def swap_row(ctx, A, i, j):
        """
        Swap row i with row j.
        """
        if i == j:
            return
        if isinstance(A, ctx.matrix):
            for k in xrange(A.cols):
                A[i,k], A[j,k] = A[j,k], A[i,k]
        elif isinstance(A, list):
            A[i], A[j] = A[j], A[i]
        else:
            raise TypeError('could not interpret type')

    def extend(ctx, A, b):
        """
        Extend matrix A with column b and return result.
        """
        if not isinstance(A, ctx.matrix):
            raise TypeError("A should be a type of ctx.matrix")
        if A.rows != len(b):
            raise ValueError("Value should be equal to len(b)")
        A = A.copy()
        A.cols += 1
        for i in xrange(A.rows):
            A[i, A.cols-1] = b[i]
        return A

    def norm(ctx, x, p=2):
        r"""
        Gives the entrywise `p`-norm of an iterable *x*, i.e. the vector norm
        `\left(\sum_k |x_k|^p\right)^{1/p}`, for any given `1 \le p \le \infty`.

        Special cases:

        If *x* is not iterable, this just returns ``absmax(x)``.

        ``p=1`` gives the sum of absolute values.

        ``p=2`` is the standard Euclidean vector norm.

        ``p=inf`` gives the magnitude of the largest element.

        For *x* a matrix, ``p=2`` is the Frobenius norm.
        For operator matrix norms, use :func:`~mpmath.mnorm` instead.

        You can use the string 'inf' as well as float('inf') or mpf('inf')
        to specify the infinity norm.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> x = matrix([-10, 2, 100])
            >>> norm(x, 1)
            mpf('112.0')
            >>> norm(x, 2)
            mpf('100.5186549850325')
            >>> norm(x, inf)
            mpf('100.0')

        """
        try:
            iter(x)
        except TypeError:
            return ctx.absmax(x)
        if type(p) is not int:
            p = ctx.convert(p)
        if p == ctx.inf:
            return max(ctx.absmax(i) for i in x)
        elif p == 1:
            return ctx.fsum(x, absolute=1)
        elif p == 2:
            return ctx.sqrt(ctx.fsum(x, absolute=1, squared=1))
        elif p > 1:
            return ctx.nthroot(ctx.fsum(abs(i)**p for i in x), p)
        else:
            raise ValueError('p has to be >= 1')

    def mnorm(ctx, A, p=1):
        r"""
        Gives the matrix (operator) `p`-norm of A. Currently ``p=1`` and ``p=inf``
        are supported:

        ``p=1`` gives the 1-norm (maximal column sum)

        ``p=inf`` gives the `\infty`-norm (maximal row sum).
        You can use the string 'inf' as well as float('inf') or mpf('inf')

        ``p=2`` (not implemented) for a square matrix is the usual spectral
        matrix norm, i.e. the largest singular value.

        ``p='f'`` (or 'F', 'fro', 'Frobenius, 'frobenius') gives the
        Frobenius norm, which is the elementwise 2-norm. The Frobenius norm is an
        approximation of the spectral norm and satisfies

        .. math ::

            \frac{1}{\sqrt{\mathrm{rank}(A)}} \|A\|_F \le \|A\|_2 \le \|A\|_F

        The Frobenius norm lacks some mathematical properties that might
        be expected of a norm.

        For general elementwise `p`-norms, use :func:`~mpmath.norm` instead.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> A = matrix([[1, -1000], [100, 50]])
            >>> mnorm(A, 1)
            mpf('1050.0')
            >>> mnorm(A, inf)
            mpf('1001.0')
            >>> mnorm(A, 'F')
            mpf('1006.2310867787777')

        """
        A = ctx.matrix(A)
        if type(p) is not int:
            if type(p) is str and 'frobenius'.startswith(p.lower()):
                return ctx.norm(A, 2)
            p = ctx.convert(p)
        m, n = A.rows, A.cols
        if p == 1:
            return max(ctx.fsum((A[i,j] for i in xrange(m)), absolute=1) for j in xrange(n))
        elif p == ctx.inf:
            return max(ctx.fsum((A[i,j] for j in xrange(n)), absolute=1) for i in xrange(m))
        else:
            raise NotImplementedError("matrix p-norm for arbitrary p")

if __name__ == '__main__':
    import doctest
    doctest.testmod()
<|endoftext|>from .functions import defun, defun_wrapped

def _hermite_param(ctx, n, z, parabolic_cylinder):
    """
    Combined calculation of the Hermite polynomial H_n(z) (and its
    generalization to complex n) and the parabolic cylinder
    function D.
    """
    n, ntyp = ctx._convert_param(n)
    z = ctx.convert(z)
    q = -ctx.mpq_1_2
    # For re(z) > 0, 2F0 -- http://functions.wolfram.com/
    #     HypergeometricFunctions/HermiteHGeneral/06/02/0009/
    # Otherwise, there is a reflection formula
    # 2F0 + http://functions.wolfram.com/HypergeometricFunctions/
    #           HermiteHGeneral/16/01/01/0006/
    #
    # TODO:
    # An alternative would be to use
    # http://functions.wolfram.com/HypergeometricFunctions/
    #     HermiteHGeneral/06/02/0006/
    #
    # Also, the 1F1 expansion
    # http://functions.wolfram.com/HypergeometricFunctions/
    #     HermiteHGeneral/26/01/02/0001/
    # should probably be used for tiny z
    if not z:
        T1 = [2, ctx.pi], [n, 0.5], [], [q*(n-1)], [], [], 0
        if parabolic_cylinder:
            T1[1][0] += q*n
        return T1,
    can_use_2f0 = ctx.isnpint(-n) or ctx.re(z) > 0 or \
        (ctx.re(z) == 0 and ctx.im(z) > 0)
    expprec = ctx.prec*4 + 20
    if parabolic_cylinder:
        u = ctx.fmul(ctx.fmul(z,z,prec=expprec), -0.25, exact=True)
        w = ctx.fmul(z, ctx.sqrt(0.5,prec=expprec), prec=expprec)
    else:
        w = z
    w2 = ctx.fmul(w, w, prec=expprec)
    rw2 = ctx.fdiv(1, w2, prec=expprec)
    nrw2 = ctx.fneg(rw2, exact=True)
    nw = ctx.fneg(w, exact=True)
    if can_use_2f0:
        T1 = [2, w], [n, n], [], [], [q*n, q*(n-1)], [], nrw2
        terms = [T1]
    else:
        T1 = [2, nw], [n, n], [], [], [q*n, q*(n-1)], [], nrw2
        T2 = [2, ctx.pi, nw], [n+2, 0.5, 1], [], [q*n], [q*(n-1)], [1-q], w2
        terms = [T1,T2]
    # Multiply by prefactor for D_n
    if parabolic_cylinder:
        expu = ctx.exp(u)
        for i in range(len(terms)):
            terms[i][1][0] += q*n
            terms[i][0].append(expu)
            terms[i][1].append(1)
    return tuple(terms)

@defun
def hermite(ctx, n, z, **kwargs):
    return ctx.hypercomb(lambda: _hermite_param(ctx, n, z, 0), [], **kwargs)

@defun
def pcfd(ctx, n, z, **kwargs):
    r"""
    Gives the parabolic cylinder function in Whittaker's notation
    `D_n(z) = U(-n-1/2, z)` (see :func:`~mpmath.pcfu`).
    It solves the differential equation

    .. math ::

        y'' + \left(n + \frac{1}{2} - \frac{1}{4} z^2\right) y = 0.

    and can be represented in terms of Hermite polynomials
    (see :func:`~mpmath.hermite`) as

    .. math ::

        D_n(z) = 2^{-n/2} e^{-z^2/4} H_n\left(\frac{z}{\sqrt{2}}\right).

    **Plots**

    .. literalinclude :: /plots/pcfd.py
    .. image :: /plots/pcfd.png

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> pcfd(0,0); pcfd(1,0); pcfd(2,0); pcfd(3,0)
        1.0
        0.0
        -1.0
        0.0
        >>> pcfd(4,0); pcfd(-3,0)
        3.0
        0.6266570686577501256039413
        >>> pcfd('1/2', 2+3j)
        (-5.363331161232920734849056 - 3.858877821790010714163487j)
        >>> pcfd(2, -10)
        1.374906442631438038871515e-9

    Verifying the differential equation::

        >>> n = mpf(2.5)
        >>> y = lambda z: pcfd(n,z)
        >>> z = 1.75
        >>> chop(diff(y,z,2) + (n+0.5-0.25*z**2)*y(z))
        0.0

    Rational Taylor series expansion when `n` is an integer::

        >>> taylor(lambda z: pcfd(5,z), 0, 7)
        [0.0, 15.0, 0.0, -13.75, 0.0, 3.96875, 0.0, -0.6015625]

    """
    return ctx.hypercomb(lambda: _hermite_param(ctx, n, z, 1), [], **kwargs)

@defun
def pcfu(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `U(a,z)`, which may be
    defined for `\Re(z) > 0` in terms of the confluent
    U-function (see :func:`~mpmath.hyperu`) by

    .. math ::

        U(a,z) = 2^{-\frac{1}{4}-\frac{a}{2}} e^{-\frac{1}{4} z^2}
            U\left(\frac{a}{2}+\frac{1}{4},
            \frac{1}{2}, \frac{1}{2}z^2\right)

    or, for arbitrary `z`,

    .. math ::

        e^{-\frac{1}{4}z^2} U(a,z) =
            U(a,0) \,_1F_1\left(-\tfrac{a}{2}+\tfrac{1}{4};
            \tfrac{1}{2}; -\tfrac{1}{2}z^2\right) +
            U'(a,0) z \,_1F_1\left(-\tfrac{a}{2}+\tfrac{3}{4};
            \tfrac{3}{2}; -\tfrac{1}{2}z^2\right).

    **Examples**

    Connection to other functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> z = mpf(3)
        >>> pcfu(0.5,z)
        0.03210358129311151450551963
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(z/sqrt(2))
        0.03210358129311151450551963
        >>> pcfu(0.5,-z)
        23.75012332835297233711255
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(-z/sqrt(2))
        23.75012332835297233711255
        >>> pcfu(0.5,-z)
        23.75012332835297233711255
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(-z/sqrt(2))
        23.75012332835297233711255

    """
    n, _ = ctx._convert_param(a)
    return ctx.pcfd(-n-ctx.mpq_1_2, z)

@defun
def pcfv(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `V(a,z)`, which can be
    represented in terms of :func:`~mpmath.pcfu` as

    .. math ::

        V(a,z) = \frac{\Gamma(a+\tfrac{1}{2}) (U(a,-z)-\sin(\pi a) U(a,z)}{\pi}.

    **Examples**

    Wronskian relation between `U` and `V`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a, z = 2, 3
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> sqrt(2/pi)
        0.7978845608028653558798921
        >>> a, z = 2.5, 3
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> a, z = 0.25, -1
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> a, z = 2+1j, 2+3j
        >>> chop(pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z))
        0.7978845608028653558798921

    """
    n, ntype = ctx._convert_param(a)
    z = ctx.convert(z)
    q = ctx.mpq_1_2
    r = ctx.mpq_1_4
    if ntype == 'Q' and ctx.isint(n*2):
        # Faster for half-integers
        def h():
            jz = ctx.fmul(z, -1j, exact=True)
            T1terms = _hermite_param(ctx, -n-q, z, 1)
            T2terms = _hermite_param(ctx, n-q, jz, 1)
            for T in T1terms:
                T[0].append(1j)
                T[1].append(1)
                T[3].append(q-n)
            u = ctx.expjpi((q*n-r)) * ctx.sqrt(2/ctx.pi)
            for T in T2terms:
                T[0].append(u)
                T[1].append(1)
            return T1terms + T2terms
        v = ctx.hypercomb(h, [], **kwargs)
        if ctx._is_real_type(n) and ctx._is_real_type(z):
            v = ctx._re(v)
        return v
    else:
        def h(n):
            w = ctx.square_exp_arg(z, -0.25)
            u = ctx.square_exp_arg(z, 0.5)
            e = ctx.exp(w)
            l = [ctx.pi, q, ctx.exp(w)]
            Y1 = l, [-q, n*q+r, 1], [r-q*n], [], [q*n+r], [q], u
            Y2 = l + [z], [-q, n*q-r, 1, 1], [1-r-q*n], [], [q*n+1-r], [1+q], u
            c, s = ctx.cospi_sinpi(r+q*n)
            Y1[0].append(s)
            Y2[0].append(c)
            for Y in (Y1, Y2):
                Y[1].append(1)
                Y[3].append(q-n)
            return Y1, Y2
        return ctx.hypercomb(h, [n], **kwargs)


@defun
def pcfw(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `W(a,z)` defined in (DLMF 12.14).

    **Examples**

    Value at the origin::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a = mpf(0.25)
        >>> pcfw(a,0)
        0.9722833245718180765617104
        >>> power(2,-0.75)*sqrt(abs(gamma(0.25+0.5j*a)/gamma(0.75+0.5j*a)))
        0.9722833245718180765617104
        >>> diff(pcfw,(a,0),(0,1))
        -0.5142533944210078966003624
        >>> -power(2,-0.25)*sqrt(abs(gamma(0.75+0.5j*a)/gamma(0.25+0.5j*a)))
        -0.5142533944210078966003624

    """
    n, _ = ctx._convert_param(a)
    z = ctx.convert(z)
    def terms():
        phi2 = ctx.arg(ctx.gamma(0.5 + ctx.j*n))
        phi2 = (ctx.loggamma(0.5+ctx.j*n) - ctx.loggamma(0.5-ctx.j*n))/2j
        rho = ctx.pi/8 + 0.5*phi2
        # XXX: cancellation computing k
        k = ctx.sqrt(1 + ctx.exp(2*ctx.pi*n)) - ctx.exp(ctx.pi*n)
        C = ctx.sqrt(k/2) * ctx.exp(0.25*ctx.pi*n)
        yield C * ctx.expj(rho) * ctx.pcfu(ctx.j*n, z*ctx.expjpi(-0.25))
        yield C * ctx.expj(-rho) * ctx.pcfu(-ctx.j*n, z*ctx.expjpi(0.25))
    v = ctx.sum_accurately(terms)
    if ctx._is_real_type(n) and ctx._is_real_type(z):
        v = ctx._re(v)
    return v

"""
Even/odd PCFs. Useful?

@defun
def pcfy1(ctx, a, z, **kwargs):
    a, _ = ctx._convert_param(n)
    z = ctx.convert(z)
    def h():
        w = ctx.square_exp_arg(z)
        w1 = ctx.fmul(w, -0.25, exact=True)
        w2 = ctx.fmul(w, 0.5, exact=True)
        e = ctx.exp(w1)
        return [e], [1], [], [], [ctx.mpq_1_2*a+ctx.mpq_1_4], [ctx.mpq_1_2], w2
    return ctx.hypercomb(h, [], **kwargs)

@defun
def pcfy2(ctx, a, z, **kwargs):
    a, _ = ctx._convert_param(n)
    z = ctx.convert(z)
    def h():
        w = ctx.square_exp_arg(z)
        w1 = ctx.fmul(w, -0.25, exact=True)
        w2 = ctx.fmul(w, 0.5, exact=True)
        e = ctx.exp(w1)
        return [e, z], [1, 1], [], [], [ctx.mpq_1_2*a+ctx.mpq_3_4], \
            [ctx.mpq_3_2], w2
    return ctx.hypercomb(h, [], **kwargs)
"""

@defun_wrapped
def gegenbauer(ctx, n, a, z, **kwargs):
    # Special cases: a+0.5, a*2 poles
    if ctx.isnpint(a):
        return 0*(z+n)
    if ctx.isnpint(a+0.5):
        # TODO: something else is required here
        # E.g.: gegenbauer(-2, -0.5, 3) == -12
        if ctx.isnpint(n+1):
            raise NotImplementedError("Gegenbauer function with two limits")
        def h(a):
            a2 = 2*a
            T = [], [], [n+a2], [n+1, a2], [-n, n+a2], [a+0.5], 0.5*(1-z)
            return [T]
        return ctx.hypercomb(h, [a], **kwargs)
    def h(n):
        a2 = 2*a
        T = [], [], [n+a2], [n+1, a2], [-n, n+a2], [a+0.5], 0.5*(1-z)
        return [T]
    return ctx.hypercomb(h, [n], **kwargs)

@defun_wrapped
def jacobi(ctx, n, a, b, x, **kwargs):
    if not ctx.isnpint(a):
        def h(n):
            return (([], [], [a+n+1], [n+1, a+1], [-n, a+b+n+1], [a+1], (1-x)*0.5),)
        return ctx.hypercomb(h, [n], **kwargs)
    if not ctx.isint(b):
        def h(n, a):
            return (([], [], [-b], [n+1, -b-n], [-n, a+b+n+1], [b+1], (x+1)*0.5),)
        return ctx.hypercomb(h, [n, a], **kwargs)
    # XXX: determine appropriate limit
    return ctx.binomial(n+a,n) * ctx.hyp2f1(-n,1+n+a+b,a+1,(1-x)/2, **kwargs)

@defun_wrapped
def laguerre(ctx, n, a, z, **kwargs):
    # XXX: limits, poles
    #if ctx.isnpint(n):
    #    return 0*(a+z)
    def h(a):
        return (([], [], [a+n+1], [a+1, n+1], [-n], [a+1], z),)
    return ctx.hypercomb(h, [a], **kwargs)

@defun_wrapped
def legendre(ctx, n, x, **kwargs):
    if ctx.isint(n):
        n = int(n)
        # Accuracy near zeros
        if (n + (n < 0)) & 1:
            if not x:
                return x
            mag = ctx.mag(x)
            if mag < -2*ctx.prec-10:
                return x
            if mag < -5:
                ctx.prec += -mag
    return ctx.hyp2f1(-n,n+1,1,(1-x)/2, **kwargs)

@defun
def legenp(ctx, n, m, z, type=2, **kwargs):
    # Legendre function, 1st kind
    n = ctx.convert(n)
    m = ctx.convert(m)
    # Faster
    if not m:
        return ctx.legendre(n, z, **kwargs)
    # TODO: correct evaluation at singularities
    if type == 2:
        def h(n,m):
            g = m*0.5
            T = [1+z, 1-z], [g, -g], [], [1-m], [-n, n+1], [1-m], 0.5*(1-z)
            return (T,)
        return ctx.hypercomb(h, [n,m], **kwargs)
    if type == 3:
        def h(n,m):
            g = m*0.5
            T = [z+1, z-1], [g, -g], [], [1-m], [-n, n+1], [1-m], 0.5*(1-z)
            return (T,)
        return ctx.hypercomb(h, [n,m], **kwargs)
    raise ValueError("requires type=2 or type=3")

@defun
def legenq(ctx, n, m, z, type=2, **kwargs):
    # Legendre function, 2nd kind
    n = ctx.convert(n)
    m = ctx.convert(m)
    z = ctx.convert(z)
    if z in (1, -1):
        #if ctx.isint(m):
        #    return ctx.nan
        #return ctx.inf  # unsigned
        return ctx.nan
    if type == 2:
        def h(n, m):
            cos, sin = ctx.cospi_sinpi(m)
            s = 2 * sin / ctx.pi
            c = cos
            a = 1+z
            b = 1-z
            u = m/2
            w = (1-z)/2
            T1 = [s, c, a, b], [-1, 1, u, -u], [], [1-m], \
                [-n, n+1], [1-m], w
            T2 = [-s, a, b], [-1, -u, u], [n+m+1], [n-m+1, m+1], \
                [-n, n+1], [m+1], w
            return T1, T2
        return ctx.hypercomb(h, [n, m], **kwargs)
    if type == 3:
        # The following is faster when there only is a single series
        # Note: not valid for -1 < z < 0 (?)
        if abs(z) > 1:
            def h(n, m):
                T1 = [ctx.expjpi(m), 2, ctx.pi, z, z-1, z+1], \
                     [1, -n-1, 0.5, -n-m-1, 0.5*m, 0.5*m], \
                     [n+m+1], [n+1.5], \
                     [0.5*(2+n+m), 0.5*(1+n+m)], [n+1.5], z**(-2)
                return [T1]
            return ctx.hypercomb(h, [n, m], **kwargs)
        else:
            # not valid for 1 < z < inf ?
            def h(n, m):
                s = 2 * ctx.sinpi(m) / ctx.pi
                c = ctx.expjpi(m)
                a = 1+z
                b = z-1
                u = m/2
                w = (1-z)/2
                T1 = [s, c, a, b], [-1, 1, u, -u], [], [1-m], \
                    [-n, n+1], [1-m], w
                T2 = [-s, c, a, b], [-1, 1, -u, u], [n+m+1], [n-m+1, m+1], \
                    [-n, n+1], [m+1], w
                return T1, T2
            return ctx.hypercomb(h, [n, m], **kwargs)
    raise ValueError("requires type=2 or type=3")

@defun_wrapped
def chebyt(ctx, n, x, **kwargs):
    if (not x) and ctx.isint(n) and int(ctx._re(n)) % 2 == 1:
        return x * 0
    return ctx.hyp2f1(-n,n,(1,2),(1-x)/2, **kwargs)

@defun_wrapped
def chebyu(ctx, n, x, **kwargs):
    if (not x) and ctx.isint(n) and int(ctx._re(n)) % 2 == 1:
        return x * 0
    return (n+1) * ctx.hyp2f1(-n, n+2, (3,2), (1-x)/2, **kwargs)

@defun
def spherharm(ctx, l, m, theta, phi, **kwargs):
    l = ctx.convert(l)
    m = ctx.convert(m)
    theta = ctx.convert(theta)
    phi = ctx.convert(phi)
    l_isint = ctx.isint(l)
    l_natural = l_isint and l >= 0
    m_isint = ctx.isint(m)
    if l_isint and l < 0 and m_isint:
        return ctx.spherharm(-(l+1), m, theta, phi, **kwargs)
    if theta == 0 and m_isint and m < 0:
        return ctx.zero * 1j
    if l_natural and m_isint:
        if abs(m) > l:
            return ctx.zero * 1j
        # http://functions.wolfram.com/Polynomials/
        #     SphericalHarmonicY/26/01/02/0004/
        def h(l,m):
            absm = abs(m)
            C = [-1, ctx.expj(m*phi),
                 (2*l+1)*ctx.fac(l+absm)/ctx.pi/ctx.fac(l-absm),
                 ctx.sin(theta)**2,
                 ctx.fac(absm), 2]
            P = [0.5*m*(ctx.sign(m)+1), 1, 0.5, 0.5*absm, -1, -absm-1]
            return ((C, P, [], [], [absm-l, l+absm+1], [absm+1],
                ctx.sin(0.5*theta)**2),)
    else:
        # http://functions.wolfram.com/HypergeometricFunctions/
        #     SphericalHarmonicYGeneral/26/01/02/0001/
        def h(l,m):
            if ctx.isnpint(l-m+1) or ctx.isnpint(l+m+1) or ctx.isnpint(1-m):
                return (([0], [-1], [], [], [], [], 0),)
            cos, sin = ctx.cos_sin(0.5*theta)
            C = [0.5*ctx.expj(m*phi), (2*l+1)/ctx.pi,
                 ctx.gamma(l-m+1), ctx.gamma(l+m+1),
                 cos**2, sin**2]
            P = [1, 0.5, 0.5, -0.5, 0.5*m, -0.5*m]
            return ((C, P, [], [1-m], [-l,l+1], [1-m], sin**2),)
    return ctx.hypercomb(h, [l,m], **kwargs)
<|endoftext|>from ..libmp.backend import xrange
from .functions import defun, defun_wrapped

@defun
def gammaprod(ctx, a, b, _infsign=False):
    a = [ctx.convert(x) for x in a]
    b = [ctx.convert(x) for x in b]
    poles_num = []
    poles_den = []
    regular_num = []
    regular_den = []
    for x in a: [regular_num, poles_num][ctx.isnpint(x)].append(x)
    for x in b: [regular_den, poles_den][ctx.isnpint(x)].append(x)
    # One more pole in numerator or denominator gives 0 or inf
    if len(poles_num) < len(poles_den): return ctx.zero
    if len(poles_num) > len(poles_den):
        # Get correct sign of infinity for x+h, h -> 0 from above
        # XXX: hack, this should be done properly
        if _infsign:
            a = [x and x*(1+ctx.eps) or x+ctx.eps for x in poles_num]
            b = [x and x*(1+ctx.eps) or x+ctx.eps for x in poles_den]
            return ctx.sign(ctx.gammaprod(a+regular_num,b+regular_den)) * ctx.inf
        else:
            return ctx.inf
    # All poles cancel
    # lim G(i)/G(j) = (-1)**(i+j) * gamma(1-j) / gamma(1-i)
    p = ctx.one
    orig = ctx.prec
    try:
        ctx.prec = orig + 15
        while poles_num:
            i = poles_num.pop()
            j = poles_den.pop()
            p *= (-1)**(i+j) * ctx.gamma(1-j) / ctx.gamma(1-i)
        for x in regular_num: p *= ctx.gamma(x)
        for x in regular_den: p /= ctx.gamma(x)
    finally:
        ctx.prec = orig
    return +p

@defun
def beta(ctx, x, y):
    x = ctx.convert(x)
    y = ctx.convert(y)
    if ctx.isinf(y):
        x, y = y, x
    if ctx.isinf(x):
        if x == ctx.inf and not ctx._im(y):
            if y == ctx.ninf:
                return ctx.nan
            if y > 0:
                return ctx.zero
            if ctx.isint(y):
                return ctx.nan
            if y < 0:
                return ctx.sign(ctx.gamma(y)) * ctx.inf
        return ctx.nan
    xy = ctx.fadd(x, y, prec=2*ctx.prec)
    return ctx.gammaprod([x, y], [xy])

@defun
def binomial(ctx, n, k):
    n1 = ctx.fadd(n, 1, prec=2*ctx.prec)
    k1 = ctx.fadd(k, 1, prec=2*ctx.prec)
    nk1 = ctx.fsub(n1, k, prec=2*ctx.prec)
    return ctx.gammaprod([n1], [k1, nk1])

@defun
def rf(ctx, x, n):
    xn = ctx.fadd(x, n, prec=2*ctx.prec)
    return ctx.gammaprod([xn], [x])

@defun
def ff(ctx, x, n):
    x1 = ctx.fadd(x, 1, prec=2*ctx.prec)
    xn1 = ctx.fadd(ctx.fsub(x, n, prec=2*ctx.prec), 1, prec=2*ctx.prec)
    return ctx.gammaprod([x1], [xn1])

@defun_wrapped
def fac2(ctx, x):
    if ctx.isinf(x):
        if x == ctx.inf:
            return x
        return ctx.nan
    return 2**(x/2)*(ctx.pi/2)**((ctx.cospi(x)-1)/4)*ctx.gamma(x/2+1)

@defun_wrapped
def barnesg(ctx, z):
    if ctx.isinf(z):
        if z == ctx.inf:
            return z
        return ctx.nan
    if ctx.isnan(z):
        return z
    if (not ctx._im(z)) and ctx._re(z) <= 0 and ctx.isint(ctx._re(z)):
        return z*0
    # Account for size (would not be needed if computing log(G))
    if abs(z) > 5:
        ctx.dps += 2*ctx.log(abs(z),2)
    # Reflection formula
    if ctx.re(z) < -ctx.dps:
        w = 1-z
        pi2 = 2*ctx.pi
        u = ctx.expjpi(2*w)
        v = ctx.j*ctx.pi/12 - ctx.j*ctx.pi*w**2/2 + w*ctx.ln(1-u) - \
            ctx.j*ctx.polylog(2, u)/pi2
        v = ctx.barnesg(2-z)*ctx.exp(v)/pi2**w
        if ctx._is_real_type(z):
            v = ctx._re(v)
        return v
    # Estimate terms for asymptotic expansion
    # TODO: fixme, obviously
    N = ctx.dps // 2 + 5
    G = 1
    while abs(z) < N or ctx.re(z) < 1:
        G /= ctx.gamma(z)
        z += 1
    z -= 1
    s = ctx.mpf(1)/12
    s -= ctx.log(ctx.glaisher)
    s += z*ctx.log(2*ctx.pi)/2
    s += (z**2/2-ctx.mpf(1)/12)*ctx.log(z)
    s -= 3*z**2/4
    z2k = z2 = z**2
    for k in xrange(1, N+1):
        t = ctx.bernoulli(2*k+2) / (4*k*(k+1)*z2k)
        if abs(t) < ctx.eps:
            #print k, N      # check how many terms were needed
            break
        z2k *= z2
        s += t
    #if k == N:
    #    print "warning: series for barnesg failed to converge", ctx.dps
    return G*ctx.exp(s)

@defun
def superfac(ctx, z):
    return ctx.barnesg(z+2)

@defun_wrapped
def hyperfac(ctx, z):
    # XXX: estimate needed extra bits accurately
    if z == ctx.inf:
        return z
    if abs(z) > 5:
        extra = 4*int(ctx.log(abs(z),2))
    else:
        extra = 0
    ctx.prec += extra
    if not ctx._im(z) and ctx._re(z) < 0 and ctx.isint(ctx._re(z)):
        n = int(ctx.re(z))
        h = ctx.hyperfac(-n-1)
        if ((n+1)//2) & 1:
            h = -h
        if ctx._is_complex_type(z):
            return h + 0j
        return h
    zp1 = z+1
    # Wrong branch cut
    #v = ctx.gamma(zp1)**z
    #ctx.prec -= extra
    #return v / ctx.barnesg(zp1)
    v = ctx.exp(z*ctx.loggamma(zp1))
    ctx.prec -= extra
    return v / ctx.barnesg(zp1)

'''
@defun
def psi0(ctx, z):
    """Shortcut for psi(0,z) (the digamma function)"""
    return ctx.psi(0, z)

@defun
def psi1(ctx, z):
    """Shortcut for psi(1,z) (the trigamma function)"""
    return ctx.psi(1, z)

@defun
def psi2(ctx, z):
    """Shortcut for psi(2,z) (the tetragamma function)"""
    return ctx.psi(2, z)

@defun
def psi3(ctx, z):
    """Shortcut for psi(3,z) (the pentagamma function)"""
    return ctx.psi(3, z)
'''
<|endoftext|>r"""
Elliptic functions historically comprise the elliptic integrals
and their inverses, and originate from the problem of computing the
arc length of an ellipse. From a more modern point of view,
an elliptic function is defined as a doubly periodic function, i.e.
a function which satisfies

.. math ::

    f(z + 2 \omega_1) = f(z + 2 \omega_2) = f(z)

for some half-periods `\omega_1, \omega_2` with
`\mathrm{Im}[\omega_1 / \omega_2] > 0`. The canonical elliptic
functions are the Jacobi elliptic functions. More broadly, this section
includes  quasi-doubly periodic functions (such as the Jacobi theta
functions) and other functions useful in the study of elliptic functions.

Many different conventions for the arguments of
elliptic functions are in use. It is even standard to use
different parameterizations for different functions in the same
text or software (and mpmath is no exception).
The usual parameters are the elliptic nome `q`, which usually
must satisfy `|q| < 1`; the elliptic parameter `m` (an arbitrary
complex number); the elliptic modulus `k` (an arbitrary complex
number); and the half-period ratio `\tau`, which usually must
satisfy `\mathrm{Im}[\tau] > 0`.
These quantities can be expressed in terms of each other
using the following relations:

.. math ::

    m = k^2

.. math ::

    \tau = i \frac{K(1-m)}{K(m)}

.. math ::

    q = e^{i \pi \tau}

.. math ::

    k = \frac{\vartheta_2^2(q)}{\vartheta_3^2(q)}

In addition, an alternative definition is used for the nome in
number theory, which we here denote by q-bar:

.. math ::

    \bar{q} = q^2 = e^{2 i \pi \tau}

For convenience, mpmath provides functions to convert
between the various parameters (:func:`~mpmath.qfrom`, :func:`~mpmath.mfrom`,
:func:`~mpmath.kfrom`, :func:`~mpmath.taufrom`, :func:`~mpmath.qbarfrom`).

**References**

1. [AbramowitzStegun]_

2. [WhittakerWatson]_

"""

from .functions import defun, defun_wrapped

@defun_wrapped
def eta(ctx, tau):
    r"""
    Returns the Dedekind eta function of tau in the upper half-plane.

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> eta(1j); gamma(0.25) / (2*pi**0.75)
        (0.7682254223260566590025942 + 0.0j)
        0.7682254223260566590025942
        >>> tau = sqrt(2) + sqrt(5)*1j
        >>> eta(-1/tau); sqrt(-1j*tau) * eta(tau)
        (0.9022859908439376463573294 + 0.07985093673948098408048575j)
        (0.9022859908439376463573295 + 0.07985093673948098408048575j)
        >>> eta(tau+1); exp(pi*1j/12) * eta(tau)
        (0.4493066139717553786223114 + 0.3290014793877986663915939j)
        (0.4493066139717553786223114 + 0.3290014793877986663915939j)
        >>> f = lambda z: diff(eta, z) / eta(z)
        >>> chop(36*diff(f,tau)**2 - 24*diff(f,tau,2)*f(tau) + diff(f,tau,3))
        0.0

    """
    if ctx.im(tau) <= 0.0:
        raise ValueError("eta is only defined in the upper half-plane")
    q = ctx.expjpi(tau/12)
    return q * ctx.qp(q**24)

def nome(ctx, m):
    m = ctx.convert(m)
    if not m:
        return m
    if m == ctx.one:
        return m
    if ctx.isnan(m):
        return m
    if ctx.isinf(m):
        if m == ctx.ninf:
            return type(m)(-1)
        else:
            return ctx.mpc(-1)
    a = ctx.ellipk(ctx.one-m)
    b = ctx.ellipk(m)
    v = ctx.exp(-ctx.pi*a/b)
    if not ctx._im(m) and ctx._re(m) < 1:
        if ctx._is_real_type(m):
            return v.real
        else:
            return v.real + 0j
    elif m == 2:
        v = ctx.mpc(0, v.imag)
    return v

@defun_wrapped
def qfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the elliptic nome `q`, given any of `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qfrom(q=0.25)
        0.25
        >>> qfrom(m=mfrom(q=0.25))
        0.25
        >>> qfrom(k=kfrom(q=0.25))
        0.25
        >>> qfrom(tau=taufrom(q=0.25))
        (0.25 + 0.0j)
        >>> qfrom(qbar=qbarfrom(q=0.25))
        0.25

    """
    if q is not None:
        return ctx.convert(q)
    if m is not None:
        return nome(ctx, m)
    if k is not None:
        return nome(ctx, ctx.convert(k)**2)
    if tau is not None:
        return ctx.expjpi(tau)
    if qbar is not None:
        return ctx.sqrt(qbar)

@defun_wrapped
def qbarfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the number-theoretic nome `\bar q`, given any of
    `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qbarfrom(qbar=0.25)
        0.25
        >>> qbarfrom(q=qfrom(qbar=0.25))
        0.25
        >>> qbarfrom(m=extraprec(20)(mfrom)(qbar=0.25))  # ill-conditioned
        0.25
        >>> qbarfrom(k=extraprec(20)(kfrom)(qbar=0.25))  # ill-conditioned
        0.25
        >>> qbarfrom(tau=taufrom(qbar=0.25))
        (0.25 + 0.0j)

    """
    if qbar is not None:
        return ctx.convert(qbar)
    if q is not None:
        return ctx.convert(q) ** 2
    if m is not None:
        return nome(ctx, m) ** 2
    if k is not None:
        return nome(ctx, ctx.convert(k)**2) ** 2
    if tau is not None:
        return ctx.expjpi(2*tau)

@defun_wrapped
def taufrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the elliptic half-period ratio `\tau`, given any of
    `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> taufrom(tau=0.5j)
        (0.0 + 0.5j)
        >>> taufrom(q=qfrom(tau=0.5j))
        (0.0 + 0.5j)
        >>> taufrom(m=mfrom(tau=0.5j))
        (0.0 + 0.5j)
        >>> taufrom(k=kfrom(tau=0.5j))
        (0.0 + 0.5j)
        >>> taufrom(qbar=qbarfrom(tau=0.5j))
        (0.0 + 0.5j)

    """
    if tau is not None:
        return ctx.convert(tau)
    if m is not None:
        m = ctx.convert(m)
        return ctx.j*ctx.ellipk(1-m)/ctx.ellipk(m)
    if k is not None:
        k = ctx.convert(k)
        return ctx.j*ctx.ellipk(1-k**2)/ctx.ellipk(k**2)
    if q is not None:
        return ctx.log(q) / (ctx.pi*ctx.j)
    if qbar is not None:
        qbar = ctx.convert(qbar)
        return ctx.log(qbar) / (2*ctx.pi*ctx.j)

@defun_wrapped
def kfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the elliptic modulus `k`, given any of
    `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> kfrom(k=0.25)
        0.25
        >>> kfrom(m=mfrom(k=0.25))
        0.25
        >>> kfrom(q=qfrom(k=0.25))
        0.25
        >>> kfrom(tau=taufrom(k=0.25))
        (0.25 + 0.0j)
        >>> kfrom(qbar=qbarfrom(k=0.25))
        0.25

    As `q \to 1` and `q \to -1`, `k` rapidly approaches
    `1` and `i \infty` respectively::

        >>> kfrom(q=0.75)
        0.9999999999999899166471767
        >>> kfrom(q=-0.75)
        (0.0 + 7041781.096692038332790615j)
        >>> kfrom(q=1)
        1
        >>> kfrom(q=-1)
        (0.0 + +infj)
    """
    if k is not None:
        return ctx.convert(k)
    if m is not None:
        return ctx.sqrt(m)
    if tau is not None:
        q = ctx.expjpi(tau)
    if qbar is not None:
        q = ctx.sqrt(qbar)
    if q == 1:
        return q
    if q == -1:
        return ctx.mpc(0,'inf')
    return (ctx.jtheta(2,0,q)/ctx.jtheta(3,0,q))**2

@defun_wrapped
def mfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    r"""
    Returns the elliptic parameter `m`, given any of
    `q, m, k, \tau, \bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> mfrom(m=0.25)
        0.25
        >>> mfrom(q=qfrom(m=0.25))
        0.25
        >>> mfrom(k=kfrom(m=0.25))
        0.25
        >>> mfrom(tau=taufrom(m=0.25))
        (0.25 + 0.0j)
        >>> mfrom(qbar=qbarfrom(m=0.25))
        0.25

    As `q \to 1` and `q \to -1`, `m` rapidly approaches
    `1` and `-\infty` respectively::

        >>> mfrom(q=0.75)
        0.9999999999999798332943533
        >>> mfrom(q=-0.75)
        -49586681013729.32611558353
        >>> mfrom(q=1)
        1.0
        >>> mfrom(q=-1)
        -inf

    The inverse nome as a function of `q` has an integer
    Taylor series expansion::

        >>> taylor(lambda q: mfrom(q), 0, 7)
        [0.0, 16.0, -128.0, 704.0, -3072.0, 11488.0, -38400.0, 117632.0]

    """
    if m is not None:
        return m
    if k is not None:
        return k**2
    if tau is not None:
        q = ctx.expjpi(tau)
    if qbar is not None:
        q = ctx.sqrt(qbar)
    if q == 1:
        return ctx.convert(q)
    if q == -1:
        return q*ctx.inf
    v = (ctx.jtheta(2,0,q)/ctx.jtheta(3,0,q))**4
    if ctx._is_real_type(q) and q < 0:
        v = v.real
    return v

jacobi_spec = {
  'sn' : ([3],[2],[1],[4], 'sin', 'tanh'),
  'cn' : ([4],[2],[2],[4], 'cos', 'sech'),
  'dn' : ([4],[3],[3],[4], '1', 'sech'),
  'ns' : ([2],[3],[4],[1], 'csc', 'coth'),
  'nc' : ([2],[4],[4],[2], 'sec', 'cosh'),
  'nd' : ([3],[4],[4],[3], '1', 'cosh'),
  'sc' : ([3],[4],[1],[2], 'tan', 'sinh'),
  'sd' : ([3,3],[2,4],[1],[3], 'sin', 'sinh'),
  'cd' : ([3],[2],[2],[3], 'cos', '1'),
  'cs' : ([4],[3],[2],[1], 'cot', 'csch'),
  'dc' : ([2],[3],[3],[2], 'sec', '1'),
  'ds' : ([2,4],[3,3],[3],[1], 'csc', 'csch'),
  'cc' : None,
  'ss' : None,
  'nn' : None,
  'dd' : None
}

@defun
def ellipfun(ctx, kind, u=None, m=None, q=None, k=None, tau=None):
    try:
        S = jacobi_spec[kind]
    except KeyError:
        raise ValueError("First argument must be a two-character string "
            "containing 's', 'c', 'd' or 'n', e.g.: 'sn'")
    if u is None:
        def f(*args, **kwargs):
            return ctx.ellipfun(kind, *args, **kwargs)
        f.__name__ = kind
        return f
    prec = ctx.prec
    try:
        ctx.prec += 10
        u = ctx.convert(u)
        q = ctx.qfrom(m=m, q=q, k=k, tau=tau)
        if S is None:
            v = ctx.one + 0*q*u
        elif q == ctx.zero:
            if S[4] == '1': v = ctx.one
            else:           v = getattr(ctx, S[4])(u)
            v += 0*q*u
        elif q == ctx.one:
            if S[5] == '1': v = ctx.one
            else:           v = getattr(ctx, S[5])(u)
            v += 0*q*u
        else:
            t = u / ctx.jtheta(3, 0, q)**2
            v = ctx.one
            for a in S[0]: v *= ctx.jtheta(a, 0, q)
            for b in S[1]: v /= ctx.jtheta(b, 0, q)
            for c in S[2]: v *= ctx.jtheta(c, t, q)
            for d in S[3]: v /= ctx.jtheta(d, t, q)
    finally:
        ctx.prec = prec
    return +v

@defun_wrapped
def kleinj(ctx, tau=None, **kwargs):
    r"""
    Evaluates the Klein j-invariant, which is a modular function defined for
    `\tau` in the upper half-plane as

    .. math ::

        J(\tau) = \frac{g_2^3(\tau)}{g_2^3(\tau) - 27 g_3^2(\tau)}

    where `g_2` and `g_3` are the modular invariants of the Weierstrass
    elliptic function,

    .. math ::

        g_2(\tau) = 60 \sum_{(m,n) \in \mathbb{Z}^2 \setminus (0,0)} (m \tau+n)^{-4}

        g_3(\tau) = 140 \sum_{(m,n) \in \mathbb{Z}^2 \setminus (0,0)} (m \tau+n)^{-6}.

    An alternative, common notation is that of the j-function
    `j(\tau) = 1728 J(\tau)`.

    **Plots**

    .. literalinclude :: /plots/kleinj.py
    .. image :: /plots/kleinj.png
    .. literalinclude :: /plots/kleinj2.py
    .. image :: /plots/kleinj2.png

    **Examples**

    Verifying the functional equation `J(\tau) = J(\tau+1) = J(-\tau^{-1})`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> tau = 0.625+0.75*j
        >>> tau = 0.625+0.75*j
        >>> kleinj(tau)
        (-0.1507492166511182267125242 + 0.07595948379084571927228948j)
        >>> kleinj(tau+1)
        (-0.1507492166511182267125242 + 0.07595948379084571927228948j)
        >>> kleinj(-1/tau)
        (-0.1507492166511182267125242 + 0.07595948379084571927228946j)

    The j-function has a famous Laurent series expansion in terms of the nome
    `\bar{q}`, `j(\tau) = \bar{q}^{-1} + 744 + 196884\bar{q} + \ldots`::

        >>> mp.dps = 15
        >>> taylor(lambda q: 1728*q*kleinj(qbar=q), 0, 5, singular=True)
        [1.0, 744.0, 196884.0, 21493760.0, 864299970.0, 20245856256.0]

    The j-function admits exact evaluation at special algebraic points
    related to the Heegner numbers 1, 2, 3, 7, 11, 19, 43, 67, 163::

        >>> @extraprec(10)
        ... def h(n):
        ...     v = (1+sqrt(n)*j)
        ...     if n > 2:
        ...         v *= 0.5
        ...     return v
        ...
        >>> mp.dps = 25
        >>> for n in [1,2,3,7,11,19,43,67,163]:
        ...     n, chop(1728*kleinj(h(n)))
        ...
        (1, 1728.0)
        (2, 8000.0)
        (3, 0.0)
        (7, -3375.0)
        (11, -32768.0)
        (19, -884736.0)
        (43, -884736000.0)
        (67, -147197952000.0)
        (163, -262537412640768000.0)

    Also at other special points, the j-function assumes explicit
    algebraic values, e.g.::

        >>> chop(1728*kleinj(j*sqrt(5)))
        1264538.909475140509320227
        >>> identify(cbrt(_))      # note: not simplified
        '((100+sqrt(13520))/2)'
        >>> (50+26*sqrt(5))**3
        1264538.909475140509320227

    """
    q = ctx.qfrom(tau=tau, **kwargs)
    t2 = ctx.jtheta(2,0,q)
    t3 = ctx.jtheta(3,0,q)
    t4 = ctx.jtheta(4,0,q)
    P = (t2**8 + t3**8 + t4**8)**3
    Q = 54*(t2*t3*t4)**8
    return P/Q


def RF_calc(ctx, x, y, z, r):
    if y == z: return RC_calc(ctx, x, y, r)
    if x == z: return RC_calc(ctx, y, x, r)
    if x == y: return RC_calc(ctx, z, x, r)
    if not (ctx.isnormal(x) and ctx.isnormal(y) and ctx.isnormal(z)):
        if ctx.isnan(x) or ctx.isnan(y) or ctx.isnan(z):
            return x*y*z
        if ctx.isinf(x) or ctx.isinf(y) or ctx.isinf(z):
            return ctx.zero
    xm,ym,zm = x,y,z
    A0 = Am = (x+y+z)/3
    Q = ctx.root(3*r, -6) * max(abs(A0-x),abs(A0-y),abs(A0-z))
    g = ctx.mpf(0.25)
    pow4 = ctx.one
    while 1:
        xs = ctx.sqrt(xm)
        ys = ctx.sqrt(ym)
        zs = ctx.sqrt(zm)
        lm = xs*ys + xs*zs + ys*zs
        Am1 = (Am+lm)*g
        xm, ym, zm = (xm+lm)*g, (ym+lm)*g, (zm+lm)*g
        if pow4 * Q < abs(Am):
            break
        Am = Am1
        pow4 *= g
    t = pow4/Am
    X = (A0-x)*t
    Y = (A0-y)*t
    Z = -X-Y
    E2 = X*Y-Z**2
    E3 = X*Y*Z
    return ctx.power(Am,-0.5) * (9240-924*E2+385*E2**2+660*E3-630*E2*E3)/9240

def RC_calc(ctx, x, y, r, pv=True):
    if not (ctx.isnormal(x) and ctx.isnormal(y)):
        if ctx.isinf(x) or ctx.isinf(y):
            return 1/(x*y)
        if y == 0:
            return ctx.inf
        if x == 0:
            return ctx.pi / ctx.sqrt(y) / 2
        raise ValueError
    # Cauchy principal value
    if pv and ctx._im(y) == 0 and ctx._re(y) < 0:
        return ctx.sqrt(x/(x-y)) * RC_calc(ctx, x-y, -y, r)
    if x == y:
        return 1/ctx.sqrt(x)
    extraprec = 2*max(0,-ctx.mag(x-y)+ctx.mag(x))
    ctx.prec += extraprec
    if ctx._is_real_type(x) and ctx._is_real_type(y):
        x = ctx._re(x)
        y = ctx._re(y)
        a = ctx.sqrt(x/y)
        if x < y:
            b = ctx.sqrt(y-x)
            v = ctx.acos(a)/b
        else:
            b = ctx.sqrt(x-y)
            v = ctx.acosh(a)/b
    else:
        sx = ctx.sqrt(x)
        sy = ctx.sqrt(y)
        v = ctx.acos(sx/sy)/(ctx.sqrt((1-x/y))*sy)
    ctx.prec -= extraprec
    return v

def RJ_calc(ctx, x, y, z, p, r, integration):
    """
    With integration == 0, computes RJ only using Carlson's algorithm
    (may be wrong for some values).
    With integration == 1, uses an initial integration to make sure
    Carlson's algorithm is correct.
    With integration == 2, uses only integration.
    """
    if not (ctx.isnormal(x) and ctx.isnormal(y) and \
        ctx.isnormal(z) and ctx.isnormal(p)):
        if ctx.isnan(x) or ctx.isnan(y) or ctx.isnan(z) or ctx.isnan(p):
            return x*y*z
        if ctx.isinf(x) or ctx.isinf(y) or ctx.isinf(z) or ctx.isinf(p):
            return ctx.zero
    if not p:
        return ctx.inf
    if (not x) + (not y) + (not z) > 1:
        return ctx.inf
    # Check conditions and fall back on integration for argument
    # reduction if needed. The following conditions might be needlessly
    # restrictive.
    initial_integral = ctx.zero
    if integration >= 1:
        ok = (x.real >= 0 and y.real >= 0 and z.real >= 0 and p.real > 0)
        if not ok:
            if x == p or y == p or z == p:
                ok = True
        if not ok:
            if p.imag != 0 or p.real >= 0:
                if (x.imag == 0 and x.real >= 0 and ctx.conj(y) == z):
                    ok = True
                if (y.imag == 0 and y.real >= 0 and ctx.conj(x) == z):
                    ok = True
                if (z.imag == 0 and z.real >= 0 and ctx.conj(x) == y):
                    ok = True
        if not ok or (integration == 2):
            N = ctx.ceil(-min(x.real, y.real, z.real, p.real)) + 1
            # Integrate around any singularities
            if all((t.imag >= 0 or t.real > 0) for t in [x, y, z, p]):
                margin = ctx.j
            elif all((t.imag < 0 or t.real > 0) for t in [x, y, z, p]):
                margin = -ctx.j
            else:
                margin = 1
                # Go through the upper half-plane, but low enough that any
                # parameter starting in the lower plane doesn't cross the
                # branch cut
                for t in [x, y, z, p]:
                    if t.imag >= 0 or t.real > 0:
                        continue
                    margin = min(margin, abs(t.imag) * 0.5)
                margin *= ctx.j
            N += margin
            F = lambda t: 1/(ctx.sqrt(t+x)*ctx.sqrt(t+y)*ctx.sqrt(t+z)*(t+p))
            if integration == 2:
                return 1.5 * ctx.quadsubdiv(F, [0, N, ctx.inf])
            initial_integral = 1.5 * ctx.quadsubdiv(F, [0, N])
            x += N; y += N; z += N; p += N
    xm,ym,zm,pm = x,y,z,p
    A0 = Am = (x + y + z + 2*p)/5
    delta = (p-x)*(p-y)*(p-z)
    Q = ctx.root(0.25*r, -6) * max(abs(A0-x),abs(A0-y),abs(A0-z),abs(A0-p))
    g = ctx.mpf(0.25)
    pow4 = ctx.one
    S = 0
    while 1:
        sx = ctx.sqrt(xm)
        sy = ctx.sqrt(ym)
        sz = ctx.sqrt(zm)
        sp = ctx.sqrt(pm)
        lm = sx*sy + sx*sz + sy*sz
        Am1 = (Am+lm)*g
        xm = (xm+lm)*g; ym = (ym+lm)*g; zm = (zm+lm)*g; pm = (pm+lm)*g
        dm = (sp+sx) * (sp+sy) * (sp+sz)
        em = delta * pow4**3 / dm**2
        if pow4 * Q < abs(Am):
            break
        T = RC_calc(ctx, ctx.one, ctx.one+em, r) * pow4 / dm
        S += T
        pow4 *= g
        Am = Am1
    t = pow4 / Am
    X = (A0-x)*t
    Y = (A0-y)*t
    Z = (A0-z)*t
    P = (-X-Y-Z)/2
    E2 = X*Y + X*Z + Y*Z - 3*P**2
    E3 = X*Y*Z + 2*E2*P + 4*P**3
    E4 = (2*X*Y*Z + E2*P + 3*P**3)*P
    E5 = X*Y*Z*P**2
    P = 24024 - 5148*E2 + 2457*E2**2 + 4004*E3 - 4158*E2*E3 - 3276*E4 + 2772*E5
    Q = 24024
    v1 = pow4 * ctx.power(Am, -1.5) * P/Q
    v2 = 6*S
    return initial_integral + v1 + v2

@defun
def elliprf(ctx, x, y, z):
    r"""
    Evaluates the Carlson symmetric elliptic integral of the first kind

    .. math ::

        R_F(x,y,z) = \frac{1}{2}
            \int_0^{\infty} \frac{dt}{\sqrt{(t+x)(t+y)(t+z)}}

    which is defined for `x,y,z \notin (-\infty,0)`, and with
    at most one of `x,y,z` being zero.

    For real `x,y,z \ge 0`, the principal square root is taken in the integrand.
    For complex `x,y,z`, the principal square root is taken as `t \to \infty`
    and as `t \to 0` non-principal branches are chosen as necessary so as to
    make the integrand continuous.

    **Examples**

    Some basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprf(0,1,1); pi/2
        1.570796326794896619231322
        1.570796326794896619231322
        >>> elliprf(0,1,inf)
        0.0
        >>> elliprf(1,1,1)
        1.0
        >>> elliprf(2,2,2)**2
        0.5
        >>> elliprf(1,0,0); elliprf(0,0,1); elliprf(0,1,0); elliprf(0,0,0)
        +inf
        +inf
        +inf
        +inf

    Representing complete elliptic integrals in terms of `R_F`::

        >>> m = mpf(0.75)
        >>> ellipk(m); elliprf(0,1-m,1)
        2.156515647499643235438675
        2.156515647499643235438675
        >>> ellipe(m); elliprf(0,1-m,1)-m*elliprd(0,1-m,1)/3
        1.211056027568459524803563
        1.211056027568459524803563

    Some symmetries and argument transformations::

        >>> x,y,z = 2,3,4
        >>> elliprf(x,y,z); elliprf(y,x,z); elliprf(z,y,x)
        0.5840828416771517066928492
        0.5840828416771517066928492
        0.5840828416771517066928492
        >>> k = mpf(100000)
        >>> elliprf(k*x,k*y,k*z); k**(-0.5) * elliprf(x,y,z)
        0.001847032121923321253219284
        0.001847032121923321253219284
        >>> l = sqrt(x*y) + sqrt(y*z) + sqrt(z*x)
        >>> elliprf(x,y,z); 2*elliprf(x+l,y+l,z+l)
        0.5840828416771517066928492
        0.5840828416771517066928492
        >>> elliprf((x+l)/4,(y+l)/4,(z+l)/4)
        0.5840828416771517066928492

    Comparing with numerical integration::

        >>> x,y,z = 2,3,4
        >>> elliprf(x,y,z)
        0.5840828416771517066928492
        >>> f = lambda t: 0.5*((t+x)*(t+y)*(t+z))**(-0.5)
        >>> q = extradps(25)(quad)
        >>> q(f, [0,inf])
        0.5840828416771517066928492

    With the following arguments, the square root in the integrand becomes
    discontinuous at `t = 1/2` if the principal branch is used. To obtain
    the right value, `-\sqrt{r}` must be taken instead of `\sqrt{r}`
    on `t \in (0, 1/2)`::

        >>> x,y,z = j-1,j,0
        >>> elliprf(x,y,z)
        (0.7961258658423391329305694 - 1.213856669836495986430094j)
        >>> -q(f, [0,0.5]) + q(f, [0.5,inf])
        (0.7961258658423391329305694 - 1.213856669836495986430094j)

    The so-called *first lemniscate constant*, a transcendental number::

        >>> elliprf(0,1,2)
        1.31102877714605990523242
        >>> extradps(25)(quad)(lambda t: 1/sqrt(1-t**4), [0,1])
        1.31102877714605990523242
        >>> gamma('1/4')**2/(4*sqrt(2*pi))
        1.31102877714605990523242

    **References**

    1. [Carlson]_
    2. [DLMF]_ Chapter 19. Elliptic Integrals

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2**10
        v = RF_calc(ctx, x, y, z, tol)
    finally:
        ctx.prec = prec
    return +v

@defun
def elliprc(ctx, x, y, pv=True):
    r"""
    Evaluates the degenerate Carlson symmetric elliptic integral
    of the first kind

    .. math ::

        R_C(x,y) = R_F(x,y,y) =
            \frac{1}{2} \int_0^{\infty} \frac{dt}{(t+y) \sqrt{(t+x)}}.

    If `y \in (-\infty,0)`, either a value defined by continuity,
    or with *pv=True* the Cauchy principal value, can be computed.

    If `x \ge 0, y > 0`, the value can be expressed in terms of
    elementary functions as

    .. math ::

        R_C(x,y) =
        \begin{cases}
          \dfrac{1}{\sqrt{y-x}}
            \cos^{-1}\left(\sqrt{\dfrac{x}{y}}\right),   & x < y \\
          \dfrac{1}{\sqrt{y}},                          & x = y \\
          \dfrac{1}{\sqrt{x-y}}
            \cosh^{-1}\left(\sqrt{\dfrac{x}{y}}\right),  & x > y \\
        \end{cases}.

    **Examples**

    Some special values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprc(1,2)*4; elliprc(0,1)*2; +pi
        3.141592653589793238462643
        3.141592653589793238462643
        3.141592653589793238462643
        >>> elliprc(1,0)
        +inf
        >>> elliprc(5,5)**2
        0.2
        >>> elliprc(1,inf); elliprc(inf,1); elliprc(inf,inf)
        0.0
        0.0
        0.0

    Comparing with the elementary closed-form solution::

        >>> elliprc('1/3', '1/5'); sqrt(7.5)*acosh(sqrt('5/3'))
        2.041630778983498390751238
        2.041630778983498390751238
        >>> elliprc('1/5', '1/3'); sqrt(7.5)*acos(sqrt('3/5'))
        1.875180765206547065111085
        1.875180765206547065111085

    Comparing with numerical integration::

        >>> q = extradps(25)(quad)
        >>> elliprc(2, -3, pv=True)
        0.3333969101113672670749334
        >>> elliprc(2, -3, pv=False)
        (0.3333969101113672670749334 + 0.7024814731040726393156375j)
        >>> 0.5*q(lambda t: 1/(sqrt(t+2)*(t-3)), [0,3-j,6,inf])
        (0.3333969101113672670749334 + 0.7024814731040726393156375j)

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2**10
        v = RC_calc(ctx, x, y, tol, pv)
    finally:
        ctx.prec = prec
    return +v

@defun
def elliprj(ctx, x, y, z, p, integration=1):
    r"""
    Evaluates the Carlson symmetric elliptic integral of the third kind

    .. math ::

        R_J(x,y,z,p) = \frac{3}{2}
            \int_0^{\infty} \frac{dt}{(t+p)\sqrt{(t+x)(t+y)(t+z)}}.

    Like :func:`~mpmath.elliprf`, the branch of the square root in the integrand
    is defined so as to be continuous along the path of integration for
    complex values of the arguments.

    **Examples**

    Some values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprj(1,1,1,1)
        1.0
        >>> elliprj(2,2,2,2); 1/(2*sqrt(2))
        0.3535533905932737622004222
        0.3535533905932737622004222
        >>> elliprj(0,1,2,2)
        1.067937989667395702268688
        >>> 3*(2*gamma('5/4')**2-pi**2/gamma('1/4')**2)/(sqrt(2*pi))
        1.067937989667395702268688
        >>> elliprj(0,1,1,2); 3*pi*(2-sqrt(2))/4
        1.380226776765915172432054
        1.380226776765915172432054
        >>> elliprj(1,3,2,0); elliprj(0,1,1,0); elliprj(0,0,0,0)
        +inf
        +inf
        +inf
        >>> elliprj(1,inf,1,0); elliprj(1,1,1,inf)
        0.0
        0.0
        >>> chop(elliprj(1+j, 1-j, 1, 1))
        0.8505007163686739432927844

    Scale transformation::

        >>> x,y,z,p = 2,3,4,5
        >>> k = mpf(100000)
        >>> elliprj(k*x,k*y,k*z,k*p); k**(-1.5)*elliprj(x,y,z,p)
        4.521291677592745527851168e-9
        4.521291677592745527851168e-9

    Comparing with numerical integration::

        >>> elliprj(1,2,3,4)
        0.2398480997495677621758617
        >>> f = lambda t: 1/((t+4)*sqrt((t+1)*(t+2)*(t+3)))
        >>> 1.5*quad(f, [0,inf])
        0.2398480997495677621758617
        >>> elliprj(1,2+1j,3,4-2j)
        (0.216888906014633498739952 + 0.04081912627366673332369512j)
        >>> f = lambda t: 1/((t+4-2j)*sqrt((t+1)*(t+2+1j)*(t+3)))
        >>> 1.5*quad(f, [0,inf])
        (0.216888906014633498739952 + 0.04081912627366673332369511j)

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    p = ctx.convert(p)
    prec = ctx.prec
    try:
        ctx.prec += 20
        tol = ctx.eps * 2**10
        v = RJ_calc(ctx, x, y, z, p, tol, integration)
    finally:
        ctx.prec = prec
    return +v

@defun
def elliprd(ctx, x, y, z):
    r"""
    Evaluates the degenerate Carlson symmetric elliptic integral
    of the third kind or Carlson elliptic integral of the
    second kind `R_D(x,y,z) = R_J(x,y,z,z)`.

    See :func:`~mpmath.elliprj` for additional information.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprd(1,2,3)
        0.2904602810289906442326534
        >>> elliprj(1,2,3,3)
        0.2904602810289906442326534

    The so-called *second lemniscate constant*, a transcendental number::

        >>> elliprd(0,2,1)/3
        0.5990701173677961037199612
        >>> extradps(25)(quad)(lambda t: t**2/sqrt(1-t**4), [0,1])
        0.5990701173677961037199612
        >>> gamma('3/4')**2/sqrt(2*pi)
        0.5990701173677961037199612

    """
    return ctx.elliprj(x,y,z,z)

@defun
def elliprg(ctx, x, y, z):
    r"""
    Evaluates the Carlson completely symmetric elliptic integral
    of the second kind

    .. math ::

        R_G(x,y,z) = \frac{1}{4} \int_0^{\infty}
            \frac{t}{\sqrt{(t+x)(t+y)(t+z)}}
            \left( \frac{x}{t+x} + \frac{y}{t+y} + \frac{z}{t+z}\right) dt.

    **Examples**

    Evaluation for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> elliprg(0,1,1)*4; +pi
        3.141592653589793238462643
        3.141592653589793238462643
        >>> elliprg(0,0.5,1)
        0.6753219405238377512600874
        >>> chop(elliprg(1+j, 1-j, 2))
        1.172431327676416604532822

    A double integral that can be evaluated in terms of `R_G`::

        >>> x,y,z = 2,3,4
        >>> def f(t,u):
        ...     st = fp.sin(t); ct = fp.cos(t)
        ...     su = fp.sin(u); cu = fp.cos(u)
        ...     return (x*(st*cu)**2 + y*(st*su)**2 + z*ct**2)**0.5 * st
        ...
        >>> nprint(mpf(fp.quad(f, [0,fp.pi], [0,2*fp.pi])/(4*fp.pi)), 13)
        1.725503028069
        >>> nprint(elliprg(x,y,z), 13)
        1.725503028069

    """
    x = ctx.convert(x)
    y = ctx.convert(y)
    z = ctx.convert(z)
    zeros = (not x) + (not y) + (not z)
    if zeros == 3:
        return (x+y+z)*0
    if zeros == 2:
        if x: return 0.5*ctx.sqrt(x)
        if y: return 0.5*ctx.sqrt(y)
        return 0.5*ctx.sqrt(z)
    if zeros == 1:
        if not z:
            x, z = z, x
    def terms():
        T1 = 0.5*z*ctx.elliprf(x,y,z)
        T2 = -0.5*(x-z)*(y-z)*ctx.elliprd(x,y,z)/3
        T3 = 0.5*ctx.sqrt(x)*ctx.sqrt(y)/ctx.sqrt(z)
        return T1,T2,T3
    return ctx.sum_accurately(terms)


@defun_wrapped
def ellipf(ctx, phi, m):
    r"""
    Evaluates the Legendre incomplete elliptic integral of the first kind

     .. math ::

        F(\phi,m) = \int_0^{\phi} \frac{dt}{\sqrt{1-m \sin^2 t}}

    or equivalently

    .. math ::

        F(\phi,m) = \int_0^{\sin \phi}
        \frac{dt}{\left(\sqrt{1-t^2}\right)\left(\sqrt{1-mt^2}\right)}.

    The function reduces to a complete elliptic integral of the first kind
    (see :func:`~mpmath.ellipk`) when `\phi = \frac{\pi}{2}`; that is,

    .. math ::

        F\left(\frac{\pi}{2}, m\right) = K(m).

    In the defining integral, it is assumed that the principal branch
    of the square root is taken and that the path of integration avoids
    crossing any branch cuts. Outside `-\pi/2 \le \Re(\phi) \le \pi/2`,
    the function extends quasi-periodically as

    .. math ::

        F(\phi + n \pi, m) = 2 n K(m) + F(\phi,m), n \in \mathbb{Z}.

    **Plots**

    .. literalinclude :: /plots/ellipf.py
    .. image :: /plots/ellipf.png

    **Examples**

    Basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> ellipf(0,1)
        0.0
        >>> ellipf(0,0)
        0.0
        >>> ellipf(1,0); ellipf(2+3j,0)
        1.0
        (2.0 + 3.0j)
        >>> ellipf(1,1); log(sec(1)+tan(1))
        1.226191170883517070813061
        1.226191170883517070813061
        >>> ellipf(pi/2, -0.5); ellipk(-0.5)
        1.415737208425956198892166
        1.415737208425956198892166
        >>> ellipf(pi/2+eps, 1); ellipf(-pi/2-eps, 1)
        +inf
        +inf
        >>> ellipf(1.5, 1)
        3.340677542798311003320813

    Comparing with numerical integration::

        >>> z,m = 0.5, 1.25
        >>> ellipf(z,m)
        0.5287219202206327872978255
        >>> quad(lambda t: (1-m*sin(t)**2)**(-0.5), [0,z])
        0.5287219202206327872978255

    The arguments may be complex numbers::

        >>> ellipf(3j, 0.5)
        (0.0 + 1.713602407841590234804143j)
        >>> ellipf(3+4j, 5-6j)
        (1.269131241950351323305741 - 0.3561052815014558335412538j)
        >>> z,m = 2+3j, 1.25
        >>> k = 1011
        >>> ellipf(z+pi*k,m); ellipf(z,m) + 2*k*ellipk(m)
        (4086.184383622179764082821 - 3003.003538923749396546871j)
        (4086.184383622179764082821 - 3003.003538923749396546871j)

    For `|\Re(z)| < \pi/2`, the function can be expressed as a
    hypergeometric series of two variables
    (see :func:`~mpmath.appellf1`)::

        >>> z,m = 0.5, 0.25
        >>> ellipf(z,m)
        0.5050887275786480788831083
        >>> sin(z)*appellf1(0.5,0.5,0.5,1.5,sin(z)**2,m*sin(z)**2)
        0.5050887275786480788831083

    """
    z = phi
    if not (ctx.isnormal(z) and ctx.isnormal(m)):
        if m == 0:
            return z + m
        if z == 0:
            return z * m
        if m == ctx.inf or m == ctx.ninf: return z/m
        raise ValueError
    x = z.real
    ctx.prec += max(0, ctx.mag(x))
    pi = +ctx.pi
    away = abs(x) > pi/2
    if m == 1:
        if away:
            return ctx.inf
    if away:
        d = ctx.nint(x/pi)
        z = z-pi*d
        P = 2*d*ctx.ellipk(m)
    else:
        P = 0
    c, s = ctx.cos_sin(z)
    return s * ctx.elliprf(c**2, 1-m*s**2, 1) + P

@defun_wrapped
def ellipe(ctx, *args):
    r"""
    Called with a single argument `m`, evaluates the Legendre complete
    elliptic integral of the second kind, `E(m)`, defined by

        .. math :: E(m) = \int_0^{\pi/2} \sqrt{1-m \sin^2 t} \, dt \,=\,
            \frac{\pi}{2}
            \,_2F_1\left(\frac{1}{2}, -\frac{1}{2}, 1, m\right).

    Called with two arguments `\phi, m`, evaluates the incomplete elliptic
    integral of the second kind

     .. math ::

        E(\phi,m) = \int_0^{\phi} \sqrt{1-m \sin^2 t} \, dt =
                    \int_0^{\sin z}
                    \frac{\sqrt{1-mt^2}}{\sqrt{1-t^2}} \, dt.

    The incomplete integral reduces to a complete integral when
    `\phi = \frac{\pi}{2}`; that is,

    .. math ::

        E\left(\frac{\pi}{2}, m\right) = E(m).

    In the defining integral, it is assumed that the principal branch
    of the square root is taken and that the path of integration avoids
    crossing any branch cuts. Outside `-\pi/2 \le \Re(z) \le \pi/2`,
    the function extends quasi-periodically as

    .. math ::

        E(\phi + n \pi, m) = 2 n E(m) + E(\phi,m), n \in \mathbb{Z}.

    **Plots**

    .. literalinclude :: /plots/ellipe.py
    .. image :: /plots/ellipe.png

    **Examples for the complete integral**

    Basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> ellipe(0)
        1.570796326794896619231322
        >>> ellipe(1)
        1.0
        >>> ellipe(-1)
        1.910098894513856008952381
        >>> ellipe(2)
        (0.5990701173677961037199612 + 0.5990701173677961037199612j)
        >>> ellipe(inf)
        (0.0 + +infj)
        >>> ellipe(-inf)
        +inf

    Verifying the defining integral and hypergeometric
    representation::

        >>> ellipe(0.5)
        1.350643881047675502520175
        >>> quad(lambda t: sqrt(1-0.5*sin(t)**2), [0, pi/2])
        1.350643881047675502520175
        >>> pi/2*hyp2f1(0.5,-0.5,1,0.5)
        1.350643881047675502520175

    Evaluation is supported for arbitrary complex `m`::

        >>> ellipe(0.5+0.25j)
        (1.360868682163129682716687 - 0.1238733442561786843557315j)
        >>> ellipe(3+4j)
        (1.499553520933346954333612 - 1.577879007912758274533309j)

    A definite integral::

        >>> quad(ellipe, [0,1])
        1.333333333333333333333333

    **Examples for the incomplete integral**

    Basic values and limits::

        >>> ellipe(0,1)
        0.0
        >>> ellipe(0,0)
        0.0
        >>> ellipe(1,0)
        1.0
        >>> ellipe(2+3j,0)
        (2.0 + 3.0j)
        >>> ellipe(1,1); sin(1)
        0.8414709848078965066525023
        0.8414709848078965066525023
        >>> ellipe(pi/2, -0.5); ellipe(-0.5)
        1.751771275694817862026502
        1.751771275694817862026502
        >>> ellipe(pi/2, 1); ellipe(-pi/2, 1)
        1.0
        -1.0
        >>> ellipe(1.5, 1)
        0.9974949866040544309417234

    Comparing with numerical integration::

        >>> z,m = 0.5, 1.25
        >>> ellipe(z,m)
        0.4740152182652628394264449
        >>> quad(lambda t: sqrt(1-m*sin(t)**2), [0,z])
        0.4740152182652628394264449

    The arguments may be complex numbers::

        >>> ellipe(3j, 0.5)
        (0.0 + 7.551991234890371873502105j)
        >>> ellipe(3+4j, 5-6j)
        (24.15299022574220502424466 + 75.2503670480325997418156j)
        >>> k = 35
        >>> z,m = 2+3j, 1.25
        >>> ellipe(z+pi*k,m); ellipe(z,m) + 2*k*ellipe(m)
        (48.30138799412005235090766 + 17.47255216721987688224357j)
        (48.30138799412005235090766 + 17.47255216721987688224357j)

    For `|\Re(z)| < \pi/2`, the function can be expressed as a
    hypergeometric series of two variables
    (see :func:`~mpmath.appellf1`)::

        >>> z,m = 0.5, 0.25
        >>> ellipe(z,m)
        0.4950017030164151928870375
        >>> sin(z)*appellf1(0.5,0.5,-0.5,1.5,sin(z)**2,m*sin(z)**2)
        0.4950017030164151928870376

    """
    if len(args) == 1:
        return ctx._ellipe(args[0])
    else:
        phi, m = args
    z = phi
    if not (ctx.isnormal(z) and ctx.isnormal(m)):
        if m == 0:
            return z + m
        if z == 0:
            return z * m
        if m == ctx.inf or m == ctx.ninf:
            return ctx.inf
        raise ValueError
    x = z.real
    ctx.prec += max(0, ctx.mag(x))
    pi = +ctx.pi
    away = abs(x) > pi/2
    if away:
        d = ctx.nint(x/pi)
        z = z-pi*d
        P = 2*d*ctx.ellipe(m)
    else:
        P = 0
    def terms():
        c, s = ctx.cos_sin(z)
        x = c**2
        y = 1-m*s**2
        RF = ctx.elliprf(x, y, 1)
        RD = ctx.elliprd(x, y, 1)
        return s*RF, -m*s**3*RD/3
    return ctx.sum_accurately(terms) + P

@defun_wrapped
def ellippi(ctx, *args):
    r"""
    Called with three arguments `n, \phi, m`, evaluates the Legendre
    incomplete elliptic integral of the third kind

    .. math ::

        \Pi(n; \phi, m) = \int_0^{\phi}
            \frac{dt}{(1-n \sin^2 t) \sqrt{1-m \sin^2 t}} =
            \int_0^{\sin \phi}
            \frac{dt}{(1-nt^2) \sqrt{1-t^2} \sqrt{1-mt^2}}.

    Called with two arguments `n, m`, evaluates the complete
    elliptic integral of the third kind
    `\Pi(n,m) = \Pi(n; \frac{\pi}{2},m)`.

    In the defining integral, it is assumed that the principal branch
    of the square root is taken and that the path of integration avoids
    crossing any branch cuts. Outside `-\pi/2 \le \Re(\phi) \le \pi/2`,
    the function extends quasi-periodically as

    .. math ::

        \Pi(n,\phi+k\pi,m) = 2k\Pi(n,m) + \Pi(n,\phi,m), k \in \mathbb{Z}.

    **Plots**

    .. literalinclude :: /plots/ellippi.py
    .. image :: /plots/ellippi.png

    **Examples for the complete integral**

    Some basic values and limits::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> ellippi(0,-5); ellipk(-5)
        0.9555039270640439337379334
        0.9555039270640439337379334
        >>> ellippi(inf,2)
        0.0
        >>> ellippi(2,inf)
        0.0
        >>> abs(ellippi(1,5))
        +inf
        >>> abs(ellippi(0.25,1))
        +inf

    Evaluation in terms of simpler functions::

        >>> ellippi(0.25,0.25); ellipe(0.25)/(1-0.25)
        1.956616279119236207279727
        1.956616279119236207279727
        >>> ellippi(3,0); pi/(2*sqrt(-2))
        (0.0 - 1.11072073453959156175397j)
        (0.0 - 1.11072073453959156175397j)
        >>> ellippi(-3,0); pi/(2*sqrt(4))
        0.7853981633974483096156609
        0.7853981633974483096156609

    **Examples for the incomplete integral**

    Basic values and limits::

        >>> ellippi(0.25,-0.5); ellippi(0.25,pi/2,-0.5)
        1.622944760954741603710555
        1.622944760954741603710555
        >>> ellippi(1,0,1)
        0.0
        >>> ellippi(inf,0,1)
        0.0
        >>> ellippi(0,0.25,0.5); ellipf(0.25,0.5)
        0.2513040086544925794134591
        0.2513040086544925794134591
        >>> ellippi(1,1,1); (log(sec(1)+tan(1))+sec(1)*tan(1))/2
        2.054332933256248668692452
        2.054332933256248668692452
        >>> ellippi(0.25, 53*pi/2, 0.75); 53*ellippi(0.25,0.75)
        135.240868757890840755058
        135.240868757890840755058
        >>> ellippi(0.5,pi/4,0.5); 2*ellipe(pi/4,0.5)-1/sqrt(3)
        0.9190227391656969903987269
        0.9190227391656969903987269

    Complex arguments are supported::

        >>> ellippi(0.5, 5+6j-2*pi, -7-8j)
        (-0.3612856620076747660410167 + 0.5217735339984807829755815j)

    Some degenerate cases::

        >>> ellippi(1,1)
        +inf
        >>> ellippi(1,0)
        +inf
        >>> ellippi(1,2,0)
        +inf
        >>> ellippi(1,2,1)
        +inf
        >>> ellippi(1,0,1)
        0.0

    """
    if len(args) == 2:
        n, m = args
        complete = True
        z = phi = ctx.pi/2
    else:
        n, phi, m = args
        complete = False
        z = phi
    if not (ctx.isnormal(n) and ctx.isnormal(z) and ctx.isnormal(m)):
        if ctx.isnan(n) or ctx.isnan(z) or ctx.isnan(m):
            raise ValueError
        if complete:
            if m == 0:
                if n == 1:
                    return ctx.inf
                return ctx.pi/(2*ctx.sqrt(1-n))
            if n == 0: return ctx.ellipk(m)
            if ctx.isinf(n) or ctx.isinf(m): return ctx.zero
        else:
            if z == 0: return z
            if ctx.isinf(n): return ctx.zero
            if ctx.isinf(m): return ctx.zero
        if ctx.isinf(n) or ctx.isinf(z) or ctx.isinf(m):
            raise ValueError
    if complete:
        if m == 1:
            if n == 1:
                return ctx.inf
            return -ctx.inf/ctx.sign(n-1)
        away = False
    else:
        x = z.real
        ctx.prec += max(0, ctx.mag(x))
        pi = +ctx.pi
        away = abs(x) > pi/2
    if away:
        d = ctx.nint(x/pi)
        z = z-pi*d
        P = 2*d*ctx.ellippi(n,m)
        if ctx.isinf(P):
            return ctx.inf
    else:
        P = 0
    def terms():
        if complete:
            c, s = ctx.zero, ctx.one
        else:
            c, s = ctx.cos_sin(z)
        x = c**2
        y = 1-m*s**2
        RF = ctx.elliprf(x, y, 1)
        RJ = ctx.elliprj(x, y, 1, 1-n*s**2)
        return s*RF, n*s**3*RJ/3
    return ctx.sum_accurately(terms) + P
<|endoftext|>from .functions import defun, defun_wrapped

@defun
def qp(ctx, a, q=None, n=None, **kwargs):
    r"""
    Evaluates the q-Pochhammer symbol (or q-rising factorial)

    .. math ::

        (a; q)_n = \prod_{k=0}^{n-1} (1-a q^k)

    where `n = \infty` is permitted if `|q| < 1`. Called with two arguments,
    ``qp(a,q)`` computes `(a;q)_{\infty}`; with a single argument, ``qp(q)``
    computes `(q;q)_{\infty}`. The special case

    .. math ::

        \phi(q) = (q; q)_{\infty} = \prod_{k=1}^{\infty} (1-q^k) =
            \sum_{k=-\infty}^{\infty} (-1)^k q^{(3k^2-k)/2}

    is also known as the Euler function, or (up to a factor `q^{-1/24}`)
    the Dedekind eta function.

    **Examples**

    If `n` is a positive integer, the function amounts to a finite product::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qp(2,3,5)
        -725305.0
        >>> fprod(1-2*3**k for k in range(5))
        -725305.0
        >>> qp(2,3,0)
        1.0

    Complex arguments are allowed::

        >>> qp(2-1j, 0.75j)
        (0.4628842231660149089976379 + 4.481821753552703090628793j)

    The regular Pochhammer symbol `(a)_n` is obtained in the
    following limit as `q \to 1`::

        >>> a, n = 4, 7
        >>> limit(lambda q: qp(q**a,q,n) / (1-q)**n, 1)
        604800.0
        >>> rf(a,n)
        604800.0

    The Taylor series of the reciprocal Euler function gives
    the partition function `P(n)`, i.e. the number of ways of writing
    `n` as a sum of positive integers::

        >>> taylor(lambda q: 1/qp(q), 0, 10)
        [1.0, 1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 15.0, 22.0, 30.0, 42.0]

    Special values include::

        >>> qp(0)
        1.0
        >>> findroot(diffun(qp), -0.4)   # location of maximum
        -0.4112484791779547734440257
        >>> qp(_)
        1.228348867038575112586878

    The q-Pochhammer symbol is related to the Jacobi theta functions.
    For example, the following identity holds::

        >>> q = mpf(0.5)    # arbitrary
        >>> qp(q)
        0.2887880950866024212788997
        >>> root(3,-2)*root(q,-24)*jtheta(2,pi/6,root(q,6))
        0.2887880950866024212788997

    """
    a = ctx.convert(a)
    if n is None:
        n = ctx.inf
    else:
        n = ctx.convert(n)
    if n < 0:
        raise ValueError("n cannot be negative")
    if q is None:
        q = a
    else:
        q = ctx.convert(q)
    if n == 0:
        return ctx.one + 0*(a+q)
    infinite = (n == ctx.inf)
    same = (a == q)
    if infinite:
        if abs(q) >= 1:
            if same and (q == -1 or q == 1):
                return ctx.zero * q
            raise ValueError("q-function only defined for |q| < 1")
        elif q == 0:
            return ctx.one - a
    maxterms = kwargs.get('maxterms', 50*ctx.prec)
    if infinite and same:
        # Euler's pentagonal theorem
        def terms():
            t = 1
            yield t
            k = 1
            x1 = q
            x2 = q**2
            while 1:
                yield (-1)**k * x1
                yield (-1)**k * x2
                x1 *= q**(3*k+1)
                x2 *= q**(3*k+2)
                k += 1
                if k > maxterms:
                    raise ctx.NoConvergence
        return ctx.sum_accurately(terms)
    # return ctx.nprod(lambda k: 1-a*q**k, [0,n-1])
    def factors():
        k = 0
        r = ctx.one
        while 1:
            yield 1 - a*r
            r *= q
            k += 1
            if k >= n:
                return
            if k > maxterms:
                raise ctx.NoConvergence
    return ctx.mul_accurately(factors)

@defun_wrapped
def qgamma(ctx, z, q, **kwargs):
    r"""
    Evaluates the q-gamma function

    .. math ::

        \Gamma_q(z) = \frac{(q; q)_{\infty}}{(q^z; q)_{\infty}} (1-q)^{1-z}.


    **Examples**

    Evaluation for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qgamma(4,0.75)
        4.046875
        >>> qgamma(6,6)
        121226245.0
        >>> qgamma(3+4j, 0.5j)
        (0.1663082382255199834630088 + 0.01952474576025952984418217j)

    The q-gamma function satisfies a functional equation similar
    to that of the ordinary gamma function::

        >>> q = mpf(0.25)
        >>> z = mpf(2.5)
        >>> qgamma(z+1,q)
        1.428277424823760954685912
        >>> (1-q**z)/(1-q)*qgamma(z,q)
        1.428277424823760954685912

    """
    if abs(q) > 1:
        return ctx.qgamma(z,1/q)*q**((z-2)*(z-1)*0.5)
    return ctx.qp(q, q, None, **kwargs) / \
        ctx.qp(q**z, q, None, **kwargs) * (1-q)**(1-z)

@defun_wrapped
def qfac(ctx, z, q, **kwargs):
    r"""
    Evaluates the q-factorial,

    .. math ::

        [n]_q! = (1+q)(1+q+q^2)\cdots(1+q+\cdots+q^{n-1})

    or more generally

    .. math ::

        [z]_q! = \frac{(q;q)_z}{(1-q)^z}.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qfac(0,0)
        1.0
        >>> qfac(4,3)
        2080.0
        >>> qfac(5,6)
        121226245.0
        >>> qfac(1+1j, 2+1j)
        (0.4370556551322672478613695 + 0.2609739839216039203708921j)

    """
    if ctx.isint(z) and ctx._re(z) > 0:
        n = int(ctx._re(z))
        return ctx.qp(q, q, n, **kwargs) / (1-q)**n
    return ctx.qgamma(z+1, q, **kwargs)

@defun
def qhyper(ctx, a_s, b_s, q, z, **kwargs):
    r"""
    Evaluates the basic hypergeometric series or hypergeometric q-series

    .. math ::

        \,_r\phi_s \left[\begin{matrix}
            a_1 & a_2 & \ldots & a_r \\
            b_1 & b_2 & \ldots & b_s
        \end{matrix} ; q,z \right] =
        \sum_{n=0}^\infty
        \frac{(a_1;q)_n, \ldots, (a_r;q)_n}
             {(b_1;q)_n, \ldots, (b_s;q)_n}
        \left((-1)^n q^{n\choose 2}\right)^{1+s-r}
        \frac{z^n}{(q;q)_n}

    where `(a;q)_n` denotes the q-Pochhammer symbol (see :func:`~mpmath.qp`).

    **Examples**

    Evaluation works for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qhyper([0.5], [2.25], 0.25, 4)
        -0.1975849091263356009534385
        >>> qhyper([0.5], [2.25], 0.25-0.25j, 4)
        (2.806330244925716649839237 + 3.568997623337943121769938j)
        >>> qhyper([1+j], [2,3+0.5j], 0.25, 3+4j)
        (9.112885171773400017270226 - 1.272756997166375050700388j)

    Comparing with a summation of the defining series, using
    :func:`~mpmath.nsum`::

        >>> b, q, z = 3, 0.25, 0.5
        >>> qhyper([], [b], q, z)
        0.6221136748254495583228324
        >>> nsum(lambda n: z**n / qp(q,q,n)/qp(b,q,n) * q**(n*(n-1)), [0,inf])
        0.6221136748254495583228324

    """
    #a_s = [ctx._convert_param(a)[0] for a in a_s]
    #b_s = [ctx._convert_param(b)[0] for b in b_s]
    #q = ctx._convert_param(q)[0]
    a_s = [ctx.convert(a) for a in a_s]
    b_s = [ctx.convert(b) for b in b_s]
    q = ctx.convert(q)
    z = ctx.convert(z)
    r = len(a_s)
    s = len(b_s)
    d = 1+s-r
    maxterms = kwargs.get('maxterms', 50*ctx.prec)
    def terms():
        t = ctx.one
        yield t
        qk = 1
        k = 0
        x = 1
        while 1:
            for a in a_s:
                p = 1 - a*qk
                t *= p
            for b in b_s:
                p = 1 - b*qk
                if not p:
                    raise ValueError
                t /= p
            t *= z
            x *= (-1)**d * qk ** d
            qk *= q
            t /= (1 - qk)
            k += 1
            yield t * x
            if k > maxterms:
                raise ctx.NoConvergence
    return ctx.sum_accurately(terms)
<|endoftext|>from .functions import defun, defun_wrapped

@defun
def _jacobi_theta2(ctx, z, q):
    extra1 = 10
    extra2 = 20
    # the loops below break when the fixed precision quantities
    # a and b go to zero;
    # right shifting small negative numbers by wp one obtains -1, not zero,
    # so the condition a**2 + b**2 > MIN is used to break the loops.
    MIN = 2
    if z == ctx.zero:
        if (not ctx._im(q)):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            s = x2
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                s += a
            s = (1 << (wp+1)) + (s << 1)
            s = ctx.ldexp(s, -wp)
        else:
            wp = ctx.prec + extra1
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp-1)
            are = bre = x2re
            aim = bim = x2im
            sre = (1<<wp) + are
            sim = aim
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                sre += are
                sim += aim
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
    else:
        if (not ctx._im(q)) and (not ctx._im(z)):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
            cn = c1 = ctx.to_fixed(c1, wp)
            sn = s1 = ctx.to_fixed(s1, wp)
            c2 = (c1*c1 - s1*s1) >> wp
            s2 = (c1 * s1) >> (wp - 1)
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            s = c1 + ((a * cn) >> wp)
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
                s += (a * cn) >> wp
            s = (s << 1)
            s = ctx.ldexp(s, -wp)
            s *= ctx.nthroot(q, 4)
            return s
        # case z real, q complex
        elif not ctx._im(z):
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = x2re
            aim = bim = x2im
            c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
            cn = c1 = ctx.to_fixed(c1, wp)
            sn = s1 = ctx.to_fixed(s1, wp)
            c2 = (c1*c1 - s1*s1) >> wp
            s2 = (c1 * s1) >> (wp - 1)
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            sre = c1 + ((are * cn) >> wp)
            sim = ((aim * cn) >> wp)
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
                sre += ((are * cn) >> wp)
                sim += ((aim * cn) >> wp)
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
        #case z complex, q real
        elif not ctx._im(q):
            wp = ctx.prec + extra2
            x = ctx.to_fixed(ctx._re(q), wp)
            x2 = (x*x) >> wp
            a = b = x2
            prec0 = ctx.prec
            ctx.prec = wp
            c1, s1 = ctx.cos_sin(z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            #c2 = (c1*c1 - s1*s1) >> wp
            c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
            c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
            #s2 = (c1 * s1) >> (wp - 1)
            s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
            s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
            #cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            sre = c1re + ((a * cnre) >> wp)
            sim = c1im + ((a * cnim) >> wp)
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
                t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
                t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
                t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += ((a * cnre) >> wp)
                sim += ((a * cnim) >> wp)
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
        # case z and q complex
        else:
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = x2re
            aim = bim = x2im
            prec0 = ctx.prec
            ctx.prec = wp
            # cos(z), sin(z) with z complex
            c1, s1 = ctx.cos_sin(z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
            c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
            s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
            s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            n = 1
            termre = c1re
            termim = c1im
            sre = c1re + ((are * cnre - aim * cnim) >> wp)
            sim = c1im + ((are * cnim + aim * cnre) >> wp)
            n = 3
            termre = ((are * cnre - aim * cnim) >> wp)
            termim = ((are * cnim + aim * cnre) >> wp)
            sre = c1re + ((are * cnre - aim * cnim) >> wp)
            sim = c1im + ((are * cnim + aim * cnre) >> wp)
            n = 5
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                #cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
                t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
                t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
                t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                termre = ((are * cnre - aim * cnim) >> wp)
                termim = ((aim * cnre + are * cnim) >> wp)
                sre += ((are * cnre - aim * cnim) >> wp)
                sim += ((aim * cnre + are * cnim) >> wp)
                n += 2
            sre = (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
    s *= ctx.nthroot(q, 4)
    return s

@defun
def _djacobi_theta2(ctx, z, q, nd):
    MIN = 2
    extra1 = 10
    extra2 = 20
    if (not ctx._im(q)) and (not ctx._im(z)):
        wp = ctx.prec + extra1
        x = ctx.to_fixed(ctx._re(q), wp)
        x2 = (x*x) >> wp
        a = b = x2
        c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
        cn = c1 = ctx.to_fixed(c1, wp)
        sn = s1 = ctx.to_fixed(s1, wp)
        c2 = (c1*c1 - s1*s1) >> wp
        s2 = (c1 * s1) >> (wp - 1)
        cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        if (nd&1):
            s = s1 + ((a * sn * 3**nd) >> wp)
        else:
            s = c1 + ((a * cn * 3**nd) >> wp)
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
            if nd&1:
                s += (a * sn * (2*n+1)**nd) >> wp
            else:
                s += (a * cn * (2*n+1)**nd) >> wp
            n += 1
        s = -(s << 1)
        s = ctx.ldexp(s, -wp)
        # case z real, q complex
    elif not ctx._im(z):
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = x2re
        aim = bim = x2im
        c1, s1 = ctx.cos_sin(ctx._re(z), prec=wp)
        cn = c1 = ctx.to_fixed(c1, wp)
        sn = s1 = ctx.to_fixed(s1, wp)
        c2 = (c1*c1 - s1*s1) >> wp
        s2 = (c1 * s1) >> (wp - 1)
        cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        if (nd&1):
            sre = s1 + ((are * sn * 3**nd) >> wp)
            sim = ((aim * sn * 3**nd) >> wp)
        else:
            sre = c1 + ((are * cn * 3**nd) >> wp)
            sim = ((aim * cn * 3**nd) >> wp)
        n = 5
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp

            if (nd&1):
                sre += ((are * sn * n**nd) >> wp)
                sim += ((aim * sn * n**nd) >> wp)
            else:
                sre += ((are * cn * n**nd) >> wp)
                sim += ((aim * cn * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    #case z complex, q real
    elif not ctx._im(q):
        wp = ctx.prec + extra2
        x = ctx.to_fixed(ctx._re(q), wp)
        x2 = (x*x) >> wp
        a = b = x2
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        #c2 = (c1*c1 - s1*s1) >> wp
        c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
        c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
        #s2 = (c1 * s1) >> (wp - 1)
        s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
        s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
        #cn, sn = (cn*c2 - sn*s2) >> wp, (sn*c2 + cn*s2) >> wp
        t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
        t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
        t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
        t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
        cnre = t1
        cnim = t2
        snre = t3
        snim = t4
        if (nd&1):
            sre = s1re + ((a * snre * 3**nd) >> wp)
            sim = s1im + ((a * snim * 3**nd) >> wp)
        else:
            sre = c1re + ((a * cnre * 3**nd) >> wp)
            sim = c1im + ((a * cnim * 3**nd) >> wp)
        n = 5
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += ((a * snre * n**nd) >> wp)
                sim += ((a * snim * n**nd) >> wp)
            else:
                sre += ((a * cnre * n**nd) >> wp)
                sim += ((a * cnim * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    # case z and q complex
    else:
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = x2re
        aim = bim = x2im
        prec0 = ctx.prec
        ctx.prec = wp
        # cos(2*z), sin(2*z) with z complex
        c1, s1 = ctx.cos_sin(z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        c2re = (c1re*c1re - c1im*c1im - s1re*s1re + s1im*s1im) >> wp
        c2im = (c1re*c1im - s1re*s1im) >> (wp - 1)
        s2re = (c1re*s1re - c1im*s1im) >> (wp - 1)
        s2im = (c1re*s1im + c1im*s1re) >> (wp - 1)
        t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
        t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
        t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
        t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
        cnre = t1
        cnim = t2
        snre = t3
        snim = t4
        if (nd&1):
            sre = s1re + (((are * snre - aim * snim) * 3**nd) >> wp)
            sim = s1im + (((are * snim + aim * snre)* 3**nd) >> wp)
        else:
            sre = c1re + (((are * cnre - aim * cnim) * 3**nd) >> wp)
            sim = c1im + (((are * cnim + aim * cnre)* 3**nd) >> wp)
        n = 5
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            #cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            t1 = (cnre*c2re - cnim*c2im - snre*s2re + snim*s2im) >> wp
            t2 = (cnre*c2im + cnim*c2re - snre*s2im - snim*s2re) >> wp
            t3 = (snre*c2re - snim*c2im + cnre*s2re - cnim*s2im) >> wp
            t4 = (snre*c2im + snim*c2re + cnre*s2im + cnim*s2re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += (((are * snre - aim * snim) * n**nd) >> wp)
                sim += (((aim * snre + are * snim) * n**nd) >> wp)
            else:
                sre += (((are * cnre - aim * cnim) * n**nd) >> wp)
                sim += (((aim * cnre + are * cnim) * n**nd) >> wp)
            n += 2
        sre = -(sre << 1)
        sim = -(sim << 1)
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    s *= ctx.nthroot(q, 4)
    if (nd&1):
        return (-1)**(nd//2) * s
    else:
        return (-1)**(1 + nd//2) * s

@defun
def _jacobi_theta3(ctx, z, q):
    extra1 = 10
    extra2 = 20
    MIN = 2
    if z == ctx.zero:
        if not ctx._im(q):
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            s = x
            a = b = x
            x2 = (x*x) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                s += a
            s = (1 << wp) + (s << 1)
            s = ctx.ldexp(s, -wp)
            return s
        else:
            wp = ctx.prec + extra1
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            sre = are = bre = xre
            sim = aim = bim = xim
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                sre += are
                sim += aim
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
    else:
        if (not ctx._im(q)) and (not ctx._im(z)):
            s = 0
            wp = ctx.prec + extra1
            x = ctx.to_fixed(ctx._re(q), wp)
            a = b = x
            x2 = (x*x) >> wp
            c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
            c1 = ctx.to_fixed(c1, wp)
            s1 = ctx.to_fixed(s1, wp)
            cn = c1
            sn = s1
            s += (a * cn) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                s += (a * cn) >> wp
            s = (1 << wp) + (s << 1)
            s = ctx.ldexp(s, -wp)
            return s
        # case z real, q complex
        elif not ctx._im(z):
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = xre
            aim = bim = xim
            c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
            c1 = ctx.to_fixed(c1, wp)
            s1 = ctx.to_fixed(s1, wp)
            cn = c1
            sn = s1
            sre = (are * cn) >> wp
            sim = (aim * cn) >> wp
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
                sre += (are * cn) >> wp
                sim += (aim * cn) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
        #case z complex, q real
        elif not ctx._im(q):
            wp = ctx.prec + extra2
            x = ctx.to_fixed(ctx._re(q), wp)
            a = b = x
            x2 = (x*x) >> wp
            prec0 = ctx.prec
            ctx.prec = wp
            c1, s1 = ctx.cos_sin(2*z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            sre = (a * cnre) >> wp
            sim = (a * cnim) >> wp
            while abs(a) > MIN:
                b = (b*x2) >> wp
                a = (a*b) >> wp
                t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
                t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
                t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
                t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += (a * cnre) >> wp
                sim += (a * cnim) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s
        # case z and q complex
        else:
            wp = ctx.prec + extra2
            xre = ctx.to_fixed(ctx._re(q), wp)
            xim = ctx.to_fixed(ctx._im(q), wp)
            x2re = (xre*xre - xim*xim) >> wp
            x2im = (xre*xim) >> (wp - 1)
            are = bre = xre
            aim = bim = xim
            prec0 = ctx.prec
            ctx.prec = wp
            # cos(2*z), sin(2*z) with z complex
            c1, s1 = ctx.cos_sin(2*z)
            ctx.prec = prec0
            cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
            cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
            snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
            snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
            sre = (are * cnre - aim * cnim) >> wp
            sim = (aim * cnre + are * cnim) >> wp
            while are**2 + aim**2 > MIN:
                bre, bim = (bre * x2re - bim * x2im) >> wp, \
                           (bre * x2im + bim * x2re) >> wp
                are, aim = (are * bre - aim * bim) >> wp,   \
                           (are * bim + aim * bre) >> wp
                t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
                t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
                t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
                t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
                cnre = t1
                cnim = t2
                snre = t3
                snim = t4
                sre += (are * cnre - aim * cnim) >> wp
                sim += (aim * cnre + are * cnim) >> wp
            sre = (1 << wp) + (sre << 1)
            sim = (sim << 1)
            sre = ctx.ldexp(sre, -wp)
            sim = ctx.ldexp(sim, -wp)
            s = ctx.mpc(sre, sim)
            return s

@defun
def _djacobi_theta3(ctx, z, q, nd):
    """nd=1,2,3 order of the derivative with respect to z"""
    MIN = 2
    extra1 = 10
    extra2 = 20
    if (not ctx._im(q)) and (not ctx._im(z)):
        s = 0
        wp = ctx.prec + extra1
        x = ctx.to_fixed(ctx._re(q), wp)
        a = b = x
        x2 = (x*x) >> wp
        c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
        c1 = ctx.to_fixed(c1, wp)
        s1 = ctx.to_fixed(s1, wp)
        cn = c1
        sn = s1
        if (nd&1):
            s += (a * sn) >> wp
        else:
            s += (a * cn) >> wp
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            if nd&1:
                s += (a * sn * n**nd) >> wp
            else:
                s += (a * cn * n**nd) >> wp
            n += 1
        s = -(s << (nd+1))
        s = ctx.ldexp(s, -wp)
    # case z real, q complex
    elif not ctx._im(z):
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = xre
        aim = bim = xim
        c1, s1 = ctx.cos_sin(ctx._re(z)*2, prec=wp)
        c1 = ctx.to_fixed(c1, wp)
        s1 = ctx.to_fixed(s1, wp)
        cn = c1
        sn = s1
        if (nd&1):
            sre = (are * sn) >> wp
            sim = (aim * sn) >> wp
        else:
            sre = (are * cn) >> wp
            sim = (aim * cn) >> wp
        n = 2
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            cn, sn = (cn*c1 - sn*s1) >> wp, (sn*c1 + cn*s1) >> wp
            if nd&1:
                sre += (are * sn * n**nd) >> wp
                sim += (aim * sn * n**nd) >> wp
            else:
                sre += (are * cn * n**nd) >> wp
                sim += (aim * cn * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    #case z complex, q real
    elif not ctx._im(q):
        wp = ctx.prec + extra2
        x = ctx.to_fixed(ctx._re(q), wp)
        a = b = x
        x2 = (x*x) >> wp
        prec0 = ctx.prec
        ctx.prec = wp
        c1, s1 = ctx.cos_sin(2*z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        if (nd&1):
            sre = (a * snre) >> wp
            sim = (a * snim) >> wp
        else:
            sre = (a * cnre) >> wp
            sim = (a * cnim) >> wp
        n = 2
        while abs(a) > MIN:
            b = (b*x2) >> wp
            a = (a*b) >> wp
            t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
            t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
            t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
            t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if (nd&1):
                sre += (a * snre * n**nd) >> wp
                sim += (a * snim * n**nd) >> wp
            else:
                sre += (a * cnre * n**nd) >> wp
                sim += (a * cnim * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    # case z and q complex
    else:
        wp = ctx.prec + extra2
        xre = ctx.to_fixed(ctx._re(q), wp)
        xim = ctx.to_fixed(ctx._im(q), wp)
        x2re = (xre*xre - xim*xim) >> wp
        x2im = (xre*xim) >> (wp - 1)
        are = bre = xre
        aim = bim = xim
        prec0 = ctx.prec
        ctx.prec = wp
        # cos(2*z), sin(2*z) with z complex
        c1, s1 = ctx.cos_sin(2*z)
        ctx.prec = prec0
        cnre = c1re = ctx.to_fixed(ctx._re(c1), wp)
        cnim = c1im = ctx.to_fixed(ctx._im(c1), wp)
        snre = s1re = ctx.to_fixed(ctx._re(s1), wp)
        snim = s1im = ctx.to_fixed(ctx._im(s1), wp)
        if (nd&1):
            sre = (are * snre - aim * snim) >> wp
            sim = (aim * snre + are * snim) >> wp
        else:
            sre = (are * cnre - aim * cnim) >> wp
            sim = (aim * cnre + are * cnim) >> wp
        n = 2
        while are**2 + aim**2 > MIN:
            bre, bim = (bre * x2re - bim * x2im) >> wp, \
                       (bre * x2im + bim * x2re) >> wp
            are, aim = (are * bre - aim * bim) >> wp,   \
                       (are * bim + aim * bre) >> wp
            t1 = (cnre*c1re - cnim*c1im - snre*s1re + snim*s1im) >> wp
            t2 = (cnre*c1im + cnim*c1re - snre*s1im - snim*s1re) >> wp
            t3 = (snre*c1re - snim*c1im + cnre*s1re - cnim*s1im) >> wp
            t4 = (snre*c1im + snim*c1re + cnre*s1im + cnim*s1re) >> wp
            cnre = t1
            cnim = t2
            snre = t3
            snim = t4
            if(nd&1):
                sre += ((are * snre - aim * snim) * n**nd) >> wp
                sim += ((aim * snre + are * snim) * n**nd) >> wp
            else:
                sre += ((are * cnre - aim * cnim) * n**nd) >> wp
                sim += ((aim * cnre + are * cnim) * n**nd) >> wp
            n += 1
        sre = -(sre << (nd+1))
        sim = -(sim << (nd+1))
        sre = ctx.ldexp(sre, -wp)
        sim = ctx.ldexp(sim, -wp)
        s = ctx.mpc(sre, sim)
    if (nd&1):
        return (-1)**(nd//2) * s
    else:
        return (-1)**(1 + nd//2) * s

@defun
def _jacobi_theta2a(ctx, z, q):
    """
    case ctx._im(z) != 0
    theta(2, z, q) =
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n=-inf, inf)
    max term for minimum (2*n+1)*log(q).real - 2* ctx._im(z)
    n0 = int(ctx._im(z)/log(q).real - 1/2)
    theta(2, z, q) =
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n=n0, inf) +
    q**1/4 * Sum(q**(n*n + n) * exp(j*(2*n + 1)*z), n, n0-1, -inf)
    """
    n = n0 = int(ctx._im(z)/ctx._re(ctx.log(q)) - 1/2)
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj((2*n+1)*z)
    a = q**(n*n + n)
    # leading term
    term = a * e
    s = term
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    s = s * ctx.nthroot(q, 4)
    return s

@defun
def _jacobi_theta3a(ctx, z, q):
    """
    case ctx._im(z) != 0
    theta3(z, q) = Sum(q**(n*n) * exp(j*2*n*z), n, -inf, inf)
    max term for n*abs(log(q).real) + ctx._im(z) ~= 0
    n0 = int(- ctx._im(z)/abs(log(q).real))
    """
    n = n0 = int(-ctx._im(z)/abs(ctx._re(ctx.log(q))))
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj(2*n*z)
    s = term = q**(n*n) * e
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = q**(n*n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = q**(n*n) * e
        if abs(term) < eps1:
            break
        s += term
    return s

@defun
def _djacobi_theta2a(ctx, z, q, nd):
    """
    case ctx._im(z) != 0
    dtheta(2, z, q, nd) =
    j* q**1/4 * Sum(q**(n*n + n) * (2*n+1)*exp(j*(2*n + 1)*z), n=-inf, inf)
    max term for (2*n0+1)*log(q).real - 2* ctx._im(z) ~= 0
    n0 = int(ctx._im(z)/log(q).real - 1/2)
    """
    n = n0 = int(ctx._im(z)/ctx._re(ctx.log(q)) - 1/2)
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj((2*n + 1)*z)
    a = q**(n*n + n)
    # leading term
    term = (2*n+1)**nd * a * e
    s = term
    eps1 = ctx.eps*abs(term)
    while 1:
        n += 1
        e = e * e2
        term = (2*n+1)**nd * q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        term = (2*n+1)**nd * q**(n*n + n) * e
        if abs(term) < eps1:
            break
        s += term
    return ctx.j**nd * s * ctx.nthroot(q, 4)

@defun
def _djacobi_theta3a(ctx, z, q, nd):
    """
    case ctx._im(z) != 0
    djtheta3(z, q, nd) = (2*j)**nd *
      Sum(q**(n*n) * n**nd * exp(j*2*n*z), n, -inf, inf)
    max term for minimum n*abs(log(q).real) + ctx._im(z)
    """
    n = n0 = int(-ctx._im(z)/abs(ctx._re(ctx.log(q))))
    e2 = ctx.expj(2*z)
    e = e0 = ctx.expj(2*n*z)
    a = q**(n*n) * e
    s = term = n**nd * a
    if n != 0:
        eps1 = ctx.eps*abs(term)
    else:
        eps1 = ctx.eps*abs(a)
    while 1:
        n += 1
        e = e * e2
        a = q**(n*n) * e
        term = n**nd * a
        if n != 0:
            aterm = abs(term)
        else:
            aterm = abs(a)
        if aterm < eps1:
            break
        s += term
    e = e0
    e2 = ctx.expj(-2*z)
    n = n0
    while 1:
        n -= 1
        e = e * e2
        a = q**(n*n) * e
        term = n**nd * a
        if n != 0:
            aterm = abs(term)
        else:
            aterm = abs(a)
        if aterm < eps1:
            break
        s += term
    return (2*ctx.j)**nd * s

@defun
def jtheta(ctx, n, z, q, derivative=0):
    if derivative:
        return ctx._djtheta(n, z, q, derivative)

    z = ctx.convert(z)
    q = ctx.convert(q)

    # Implementation note
    # If ctx._im(z) is close to zero, _jacobi_theta2 and _jacobi_theta3
    # are used,
    # which compute the series starting from n=0 using fixed precision
    # numbers;
    # otherwise  _jacobi_theta2a and _jacobi_theta3a are used, which compute
    # the series starting from n=n0, which is the largest term.

    # TODO: write _jacobi_theta2a and _jacobi_theta3a using fixed-point

    if abs(q) > ctx.THETA_Q_LIM:
        raise ValueError('abs(q) > THETA_Q_LIM = %f' % ctx.THETA_Q_LIM)

    extra = 10
    if z:
        M = ctx.mag(z)
        if M > 5 or (n == 1 and M < -5):
            extra += 2*abs(M)
    cz = 0.5
    extra2 = 50
    prec0 = ctx.prec
    try:
        ctx.prec += extra
        if n == 1:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta2(z - ctx.pi/2, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta2a(z - ctx.pi/2, q)
            else:
                res = ctx._jacobi_theta2(z - ctx.pi/2, q)
        elif n == 2:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta2(z, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta2a(z, q)
            else:
                res = ctx._jacobi_theta2(z, q)
        elif n == 3:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta3(z, q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta3a(z, q)
            else:
                res = ctx._jacobi_theta3(z, q)
        elif n == 4:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._jacobi_theta3(z, -q)
                else:
                    ctx.dps += 10
                    res = ctx._jacobi_theta3a(z, -q)
            else:
                res = ctx._jacobi_theta3(z, -q)
        else:
            raise ValueError
    finally:
        ctx.prec = prec0
    return res

@defun
def _djtheta(ctx, n, z, q, derivative=1):
    z = ctx.convert(z)
    q = ctx.convert(q)
    nd = int(derivative)

    if abs(q) > ctx.THETA_Q_LIM:
        raise ValueError('abs(q) > THETA_Q_LIM = %f' % ctx.THETA_Q_LIM)
    extra = 10 + ctx.prec * nd // 10
    if z:
        M = ctx.mag(z)
        if M > 5 or (n != 1 and M < -5):
            extra += 2*abs(M)
    cz = 0.5
    extra2 = 50
    prec0 = ctx.prec
    try:
        ctx.prec += extra
        if n == 1:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta2(z - ctx.pi/2, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta2a(z - ctx.pi/2, q, nd)
            else:
                res = ctx._djacobi_theta2(z - ctx.pi/2, q, nd)
        elif n == 2:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta2(z, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta2a(z, q, nd)
            else:
                res = ctx._djacobi_theta2(z, q, nd)
        elif n == 3:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta3(z, q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta3a(z, q, nd)
            else:
                res = ctx._djacobi_theta3(z, q, nd)
        elif n == 4:
            if ctx._im(z):
                if abs(ctx._im(z)) < cz * abs(ctx._re(ctx.log(q))):
                    ctx.dps += extra2
                    res = ctx._djacobi_theta3(z, -q, nd)
                else:
                    ctx.dps += 10
                    res = ctx._djacobi_theta3a(z, -q, nd)
            else:
                res = ctx._djacobi_theta3(z, -q, nd)
        else:
            raise ValueError
    finally:
        ctx.prec = prec0
    return +res
<|endoftext|>from .functions import defun, defun_wrapped

@defun_wrapped
def _erf_complex(ctx, z):
    z2 = ctx.square_exp_arg(z, -1)
    #z2 = -z**2
    v = (2/ctx.sqrt(ctx.pi))*z * ctx.hyp1f1((1,2),(3,2), z2)
    if not ctx._re(z):
        v = ctx._im(v)*ctx.j
    return v

@defun_wrapped
def _erfc_complex(ctx, z):
    if ctx.re(z) > 2:
        z2 = ctx.square_exp_arg(z)
        nz2 = ctx.fneg(z2, exact=True)
        v = ctx.exp(nz2)/ctx.sqrt(ctx.pi) * ctx.hyperu((1,2),(1,2), z2)
    else:
        v = 1 - ctx._erf_complex(z)
    if not ctx._re(z):
        v = 1+ctx._im(v)*ctx.j
    return v

@defun
def erf(ctx, z):
    z = ctx.convert(z)
    if ctx._is_real_type(z):
        try:
            return ctx._erf(z)
        except NotImplementedError:
            pass
    if ctx._is_complex_type(z) and not z.imag:
        try:
            return type(z)(ctx._erf(z.real))
        except NotImplementedError:
            pass
    return ctx._erf_complex(z)

@defun
def erfc(ctx, z):
    z = ctx.convert(z)
    if ctx._is_real_type(z):
        try:
            return ctx._erfc(z)
        except NotImplementedError:
            pass
    if ctx._is_complex_type(z) and not z.imag:
        try:
            return type(z)(ctx._erfc(z.real))
        except NotImplementedError:
            pass
    return ctx._erfc_complex(z)

@defun
def square_exp_arg(ctx, z, mult=1, reciprocal=False):
    prec = ctx.prec*4+20
    if reciprocal:
        z2 = ctx.fmul(z, z, prec=prec)
        z2 = ctx.fdiv(ctx.one, z2, prec=prec)
    else:
        z2 = ctx.fmul(z, z, prec=prec)
    if mult != 1:
        z2 = ctx.fmul(z2, mult, exact=True)
    return z2

@defun_wrapped
def erfi(ctx, z):
    if not z:
        return z
    z2 = ctx.square_exp_arg(z)
    v = (2/ctx.sqrt(ctx.pi)*z) * ctx.hyp1f1((1,2), (3,2), z2)
    if not ctx._re(z):
        v = ctx._im(v)*ctx.j
    return v

@defun_wrapped
def erfinv(ctx, x):
    xre = ctx._re(x)
    if (xre != x) or (xre < -1) or (xre > 1):
        return ctx.bad_domain("erfinv(x) is defined only for -1 <= x <= 1")
    x = xre
    #if ctx.isnan(x): return x
    if not x: return x
    if x == 1: return ctx.inf
    if x == -1: return ctx.ninf
    if abs(x) < 0.9:
        a = 0.53728*x**3 + 0.813198*x
    else:
        # An asymptotic formula
        u = ctx.ln(2/ctx.pi/(abs(x)-1)**2)
        a = ctx.sign(x) * ctx.sqrt(u - ctx.ln(u))/ctx.sqrt(2)
    ctx.prec += 10
    return ctx.findroot(lambda t: ctx.erf(t)-x, a)

@defun_wrapped
def npdf(ctx, x, mu=0, sigma=1):
    sigma = ctx.convert(sigma)
    return ctx.exp(-(x-mu)**2/(2*sigma**2)) / (sigma*ctx.sqrt(2*ctx.pi))

@defun_wrapped
def ncdf(ctx, x, mu=0, sigma=1):
    a = (x-mu)/(sigma*ctx.sqrt(2))
    if a < 0:
        return ctx.erfc(-a)/2
    else:
        return (1+ctx.erf(a))/2

@defun_wrapped
def betainc(ctx, a, b, x1=0, x2=1, regularized=False):
    if x1 == x2:
        v = 0
    elif not x1:
        if x1 == 0 and x2 == 1:
            v = ctx.beta(a, b)
        else:
            v = x2**a * ctx.hyp2f1(a, 1-b, a+1, x2) / a
    else:
        m, d = ctx.nint_distance(a)
        if m <= 0:
            if d < -ctx.prec:
                h = +ctx.eps
                ctx.prec *= 2
                a += h
            elif d < -4:
                ctx.prec -= d
        s1 = x2**a * ctx.hyp2f1(a,1-b,a+1,x2)
        s2 = x1**a * ctx.hyp2f1(a,1-b,a+1,x1)
        v = (s1 - s2) / a
    if regularized:
        v /= ctx.beta(a,b)
    return v

@defun
def gammainc(ctx, z, a=0, b=None, regularized=False):
    regularized = bool(regularized)
    z = ctx.convert(z)
    if a is None:
        a = ctx.zero
        lower_modified = False
    else:
        a = ctx.convert(a)
        lower_modified = a != ctx.zero
    if b is None:
        b = ctx.inf
        upper_modified = False
    else:
        b = ctx.convert(b)
        upper_modified = b != ctx.inf
    # Complete gamma function
    if not (upper_modified or lower_modified):
        if regularized:
            if ctx.re(z) < 0:
                return ctx.inf
            elif ctx.re(z) > 0:
                return ctx.one
            else:
                return ctx.nan
        return ctx.gamma(z)
    if a == b:
        return ctx.zero
    # Standardize
    if ctx.re(a) > ctx.re(b):
        return -ctx.gammainc(z, b, a, regularized)
    # Generalized gamma
    if upper_modified and lower_modified:
        return +ctx._gamma3(z, a, b, regularized)
    # Upper gamma
    elif lower_modified:
        return ctx._upper_gamma(z, a, regularized)
    # Lower gamma
    elif upper_modified:
        return ctx._lower_gamma(z, b, regularized)

@defun
def _lower_gamma(ctx, z, b, regularized=False):
    # Pole
    if ctx.isnpint(z):
        return type(z)(ctx.inf)
    G = [z] * regularized
    negb = ctx.fneg(b, exact=True)
    def h(z):
        T1 = [ctx.exp(negb), b, z], [1, z, -1], [], G, [1], [1+z], b
        return (T1,)
    return ctx.hypercomb(h, [z])

@defun
def _upper_gamma(ctx, z, a, regularized=False):
    # Fast integer case, when available
    if ctx.isint(z):
        try:
            if regularized:
                # Gamma pole
                if ctx.isnpint(z):
                    return type(z)(ctx.zero)
                orig = ctx.prec
                try:
                    ctx.prec += 10
                    return ctx._gamma_upper_int(z, a) / ctx.gamma(z)
                finally:
