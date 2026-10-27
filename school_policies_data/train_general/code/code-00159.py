        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = None [not used]
        #       Minv_matvec = None [not used]
        #
        #  mode = 2:
        #    Solve the general eigenvalue problem:
        #      A*x = lambda*M*x
        #       A - symmetric
        #       M - symmetric positive definite
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = left multiplication by M
        #       Minv_matvec = left multiplication by M^-1
        #
        #  mode = 3:
        #    Solve the general eigenvalue problem in shift-invert mode:
        #      A*x = lambda*M*x
        #       A - symmetric
        #       M - symmetric positive semi-definite
        #    Arguments should be
        #       matvec      = None [not used]
        #       M_matvec    = left multiplication by M
        #                     or None, if M is the identity
        #       Minv_matvec = left multiplication by [A-sigma*M]^-1
        #
        #  mode = 4:
        #    Solve the general eigenvalue problem in Buckling mode:
        #      A*x = lambda*AG*x
        #       A  - symmetric positive semi-definite
        #       AG - symmetric indefinite
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = None [not used]
        #       Minv_matvec = left multiplication by [A-sigma*AG]^-1
        #
        #  mode = 5:
        #    Solve the general eigenvalue problem in Cayley-transformed mode:
        #      A*x = lambda*M*x
        #       A - symmetric
        #       M - symmetric positive semi-definite
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = left multiplication by M
        #                     or None, if M is the identity
        #       Minv_matvec = left multiplication by [A-sigma*M]^-1
        if mode == 1:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=1")
            if M_matvec is not None:
                raise ValueError("M_matvec cannot be specified for mode=1")
            if Minv_matvec is not None:
                raise ValueError("Minv_matvec cannot be specified for mode=1")

            self.OP = matvec
            self.B = lambda x: x
            self.bmat = 'I'
        elif mode == 2:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=2")
            if M_matvec is None:
                raise ValueError("M_matvec must be specified for mode=2")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=2")

            self.OP = lambda x: Minv_matvec(matvec(x))
            self.OPa = Minv_matvec
            self.OPb = matvec
            self.B = M_matvec
            self.bmat = 'G'
        elif mode == 3:
            if matvec is not None:
                raise ValueError("matvec must not be specified for mode=3")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=3")

            if M_matvec is None:
                self.OP = Minv_matvec
                self.OPa = Minv_matvec
                self.B = lambda x: x
                self.bmat = 'I'
            else:
                self.OP = lambda x: Minv_matvec(M_matvec(x))
                self.OPa = Minv_matvec
                self.B = M_matvec
                self.bmat = 'G'
        elif mode == 4:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=4")
            if M_matvec is not None:
                raise ValueError("M_matvec must not be specified for mode=4")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=4")
            self.OPa = Minv_matvec
            self.OP = lambda x: self.OPa(matvec(x))
            self.B = matvec
            self.bmat = 'G'
        elif mode == 5:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=5")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=5")

            self.OPa = Minv_matvec
            self.A_matvec = matvec

            if M_matvec is None:
                self.OP = lambda x: Minv_matvec(matvec(x) + sigma * x)
                self.B = lambda x: x
                self.bmat = 'I'
            else:
                self.OP = lambda x: Minv_matvec(matvec(x)
                                                + sigma * M_matvec(x))
                self.B = M_matvec
                self.bmat = 'G'
        else:
            raise ValueError("mode=%i not implemented" % mode)

        if which not in _SEUPD_WHICH:
            raise ValueError(f"which must be one of {' '.join(_SEUPD_WHICH)}")
        if k >= n:
            raise ValueError("k must be less than ndim(A), k=%d" % k)

        _ArpackParams.__init__(self, n, k, tp, mode, sigma,
                               ncv, v0, maxiter, which, tol)

        if self.ncv > n or self.ncv <= k:
            raise ValueError(f"ncv must be k<ncv<=n, ncv={self.ncv}")

        # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
        self.workd = _aligned_zeros(3 * n, self.tp)
        self.workl = _aligned_zeros(self.ncv * (self.ncv + 8), self.tp)

        ltr = _type_conv[self.tp]
        if ltr not in ["s", "d"]:
            raise ValueError("Input matrix is not real-valued.")

        self._arpack_solver = _arpack.__dict__[ltr + 'saupd']
        self._arpack_extract = _arpack.__dict__[ltr + 'seupd']

        self.iterate_infodict = _SAUPD_ERRORS[ltr]
        self.extract_infodict = _SEUPD_ERRORS[ltr]

        self.ipntr = np.zeros(11, arpack_int)

    def iterate(self):
        self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info = \
            self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                self.tol, self.resid, self.v, self.iparam,
                                self.ipntr, self.workd, self.workl, self.info)

        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        if self.ido == -1:
            # initialization
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            # compute y = Op*x
            if self.mode == 1:
                self.workd[yslice] = self.OP(self.workd[xslice])
            elif self.mode == 2:
                self.workd[xslice] = self.OPb(self.workd[xslice])
                self.workd[yslice] = self.OPa(self.workd[xslice])
            elif self.mode == 5:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                Ax = self.A_matvec(self.workd[xslice])
                self.workd[yslice] = self.OPa(Ax + (self.sigma *
                                                    self.workd[Bxslice]))
            else:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            raise ValueError("ARPACK requested user shifts.  Assure ISHIFT==0")
        else:
            self.converged = True

            if self.info == 0:
                pass
            elif self.info == 1:
                self._raise_no_convergence()
            else:
                raise ArpackError(self.info, infodict=self.iterate_infodict)

    def extract(self, return_eigenvectors):
        rvec = return_eigenvectors
        ierr = 0
        howmny = 'A'  # return all eigenvectors
        sselect = np.zeros(self.ncv, 'int')  # unused
        d, z, ierr = self._arpack_extract(rvec, howmny, sselect, self.sigma,
                                          self.bmat, self.which, self.k,
                                          self.tol, self.resid, self.v,
                                          self.iparam[0:7], self.ipntr,
                                          self.workd[0:2 * self.n],
                                          self.workl, ierr)
        if ierr != 0:
            raise ArpackError(ierr, infodict=self.extract_infodict)
        k_ok = self.iparam[4]
        d = d[:k_ok]
        z = z[:, :k_ok]

        if return_eigenvectors:
            return d, z
        else:
            return d


class _UnsymmetricArpackParams(_ArpackParams):
    def __init__(self, n, k, tp, matvec, mode=1, M_matvec=None,
                 Minv_matvec=None, sigma=None,
                 ncv=None, v0=None, maxiter=None, which="LM", tol=0):
        # The following modes are supported:
        #  mode = 1:
        #    Solve the standard eigenvalue problem:
        #      A*x = lambda*x
        #       A - square matrix
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = None [not used]
        #       Minv_matvec = None [not used]
        #
        #  mode = 2:
        #    Solve the generalized eigenvalue problem:
        #      A*x = lambda*M*x
        #       A - square matrix
        #       M - symmetric, positive semi-definite
        #    Arguments should be
        #       matvec      = left multiplication by A
        #       M_matvec    = left multiplication by M
        #       Minv_matvec = left multiplication by M^-1
        #
        #  mode = 3,4:
        #    Solve the general eigenvalue problem in shift-invert mode:
        #      A*x = lambda*M*x
        #       A - square matrix
        #       M - symmetric, positive semi-definite
        #    Arguments should be
        #       matvec      = None [not used]
        #       M_matvec    = left multiplication by M
        #                     or None, if M is the identity
        #       Minv_matvec = left multiplication by [A-sigma*M]^-1
        #    if A is real and mode==3, use the real part of Minv_matvec
        #    if A is real and mode==4, use the imag part of Minv_matvec
        #    if A is complex and mode==3,
        #       use real and imag parts of Minv_matvec
        if mode == 1:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=1")
            if M_matvec is not None:
                raise ValueError("M_matvec cannot be specified for mode=1")
            if Minv_matvec is not None:
                raise ValueError("Minv_matvec cannot be specified for mode=1")

            self.OP = matvec
            self.B = lambda x: x
            self.bmat = 'I'
        elif mode == 2:
            if matvec is None:
                raise ValueError("matvec must be specified for mode=2")
            if M_matvec is None:
                raise ValueError("M_matvec must be specified for mode=2")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified for mode=2")

            self.OP = lambda x: Minv_matvec(matvec(x))
            self.OPa = Minv_matvec
            self.OPb = matvec
            self.B = M_matvec
            self.bmat = 'G'
        elif mode in (3, 4):
            if matvec is None:
                raise ValueError("matvec must be specified "
                                 "for mode in (3,4)")
            if Minv_matvec is None:
                raise ValueError("Minv_matvec must be specified "
                                 "for mode in (3,4)")

            self.matvec = matvec
            if tp in 'DF':  # complex type
                if mode == 3:
                    self.OPa = Minv_matvec
                else:
                    raise ValueError("mode=4 invalid for complex A")
            else:  # real type
                if mode == 3:
                    self.OPa = lambda x: np.real(Minv_matvec(x))
                else:
                    self.OPa = lambda x: np.imag(Minv_matvec(x))
            if M_matvec is None:
                self.B = lambda x: x
                self.bmat = 'I'
                self.OP = self.OPa
            else:
                self.B = M_matvec
                self.bmat = 'G'
                self.OP = lambda x: self.OPa(M_matvec(x))
        else:
            raise ValueError("mode=%i not implemented" % mode)

        if which not in _NEUPD_WHICH:
            raise ValueError(f"Parameter which must be one of {' '.join(_NEUPD_WHICH)}")
        if k >= n - 1:
            raise ValueError("k must be less than ndim(A)-1, k=%d" % k)

        _ArpackParams.__init__(self, n, k, tp, mode, sigma,
                               ncv, v0, maxiter, which, tol)

        if self.ncv > n or self.ncv <= k + 1:
            raise ValueError(f"ncv must be k+1<ncv<=n, ncv={self.ncv}")

        # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
        self.workd = _aligned_zeros(3 * n, self.tp)
        self.workl = _aligned_zeros(3 * self.ncv * (self.ncv + 2), self.tp)

        ltr = _type_conv[self.tp]
        self._arpack_solver = _arpack.__dict__[ltr + 'naupd']
        self._arpack_extract = _arpack.__dict__[ltr + 'neupd']

        self.iterate_infodict = _NAUPD_ERRORS[ltr]
        self.extract_infodict = _NEUPD_ERRORS[ltr]

        self.ipntr = np.zeros(14, arpack_int)

        if self.tp in 'FD':
            # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
            self.rwork = _aligned_zeros(self.ncv, self.tp.lower())
        else:
            self.rwork = None

    def iterate(self):
        if self.tp in 'fd':
            results = self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                          self.tol, self.resid, self.v, self.iparam,
                                          self.ipntr, self.workd, self.workl, self.info)
            self.ido, self.tol, self.resid, self.v, \
                self.iparam, self.ipntr, self.info = results

        else:
            results = self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                          self.tol, self.resid, self.v, self.iparam,
                                          self.ipntr, self.workd, self.workl,
                                          self.rwork, self.info)
            self.ido, self.tol, self.resid, self.v, \
                self.iparam, self.ipntr, self.info = results


        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        if self.ido == -1:
            # initialization
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            # compute y = Op*x
            if self.mode in (1, 2):
                self.workd[yslice] = self.OP(self.workd[xslice])
            else:
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            raise ValueError("ARPACK requested user shifts.  Assure ISHIFT==0")
        else:
            self.converged = True

            if self.info == 0:
                pass
            elif self.info == 1:
                self._raise_no_convergence()
            else:
                raise ArpackError(self.info, infodict=self.iterate_infodict)

    def extract(self, return_eigenvectors):
        k, n = self.k, self.n

        ierr = 0
        howmny = 'A'  # return all eigenvectors
        sselect = np.zeros(self.ncv, 'int')  # unused
        sigmar = np.real(self.sigma)
        sigmai = np.imag(self.sigma)
        workev = np.zeros(3 * self.ncv, self.tp)

        if self.tp in 'fd':
            dr = np.zeros(k + 1, self.tp)
            di = np.zeros(k + 1, self.tp)
            zr = np.zeros((n, k + 1), self.tp)
            dr, di, zr, ierr = \
                self._arpack_extract(return_eigenvectors,
                       howmny, sselect, sigmar, sigmai, workev,
                       self.bmat, self.which, k, self.tol, self.resid,
                       self.v, self.iparam, self.ipntr,
                       self.workd, self.workl, self.info)
            if ierr != 0:
                raise ArpackError(ierr, infodict=self.extract_infodict)
            nreturned = self.iparam[4]  # number of good eigenvalues returned

            # Build complex eigenvalues from real and imaginary parts
            d = dr + 1.0j * di

            # Arrange the eigenvectors: complex eigenvectors are stored as
            # real,imaginary in consecutive columns
            z = zr.astype(self.tp.upper())

            # The ARPACK nonsymmetric real and double interface (s,d)naupd
            # return eigenvalues and eigenvectors in real (float,double)
            # arrays.

            # Efficiency: this should check that return_eigenvectors == True
            #  before going through this construction.
            if sigmai == 0:
                i = 0
                while i <= k:
                    # check if complex
                    if abs(d[i].imag) != 0:
                        # this is a complex conjugate pair with eigenvalues
                        # in consecutive columns
                        if i < k:
                            z[:, i] = zr[:, i] + 1.0j * zr[:, i + 1]
                            z[:, i + 1] = z[:, i].conjugate()
                            i += 1
                        else:
                            #last eigenvalue is complex: the imaginary part of
                            # the eigenvector has not been returned
                            #this can only happen if nreturned > k, so we'll
                            # throw out this case.
                            nreturned -= 1
                    i += 1

            else:
                # real matrix, mode 3 or 4, imag(sigma) is nonzero:
                # see remark 3 in <s,d>neupd.f
                # Build complex eigenvalues from real and imaginary parts
                i = 0
                while i <= k:
                    if abs(d[i].imag) == 0:
                        d[i] = np.dot(zr[:, i], self.matvec(zr[:, i]))
                    else:
                        if i < k:
                            z[:, i] = zr[:, i] + 1.0j * zr[:, i + 1]
                            z[:, i + 1] = z[:, i].conjugate()
                            d[i] = ((np.dot(zr[:, i],
                                            self.matvec(zr[:, i]))
                                     + np.dot(zr[:, i + 1],
                                              self.matvec(zr[:, i + 1])))
                                    + 1j * (np.dot(zr[:, i],
                                                   self.matvec(zr[:, i + 1]))
                                            - np.dot(zr[:, i + 1],
                                                     self.matvec(zr[:, i]))))
                            d[i + 1] = d[i].conj()
                            i += 1
                        else:
                            #last eigenvalue is complex: the imaginary part of
                            # the eigenvector has not been returned
                            #this can only happen if nreturned > k, so we'll
                            # throw out this case.
                            nreturned -= 1
                    i += 1

            # Now we have k+1 possible eigenvalues and eigenvectors
            # Return the ones specified by the keyword "which"

            if nreturned <= k:
                # we got less or equal as many eigenvalues we wanted
                d = d[:nreturned]
                z = z[:, :nreturned]
            else:
                # we got one extra eigenvalue (likely a cc pair, but which?)
                if self.mode in (1, 2):
                    rd = d
                elif self.mode in (3, 4):
                    rd = 1 / (d - self.sigma)

                if self.which in ['LR', 'SR']:
                    ind = np.argsort(rd.real)
                elif self.which in ['LI', 'SI']:
                    # for LI,SI ARPACK returns largest,smallest
                    # abs(imaginary) (complex pairs come together)
                    ind = np.argsort(abs(rd.imag))
                else:
                    ind = np.argsort(abs(rd))

                if self.which in ['LR', 'LM', 'LI']:
                    ind = ind[-k:][::-1]
                elif self.which in ['SR', 'SM', 'SI']:
                    ind = ind[:k]

                d = d[ind]
                z = z[:, ind]
        else:
            # complex is so much simpler...
            d, z, ierr =\
                    self._arpack_extract(return_eigenvectors,
                           howmny, sselect, self.sigma, workev,
                           self.bmat, self.which, k, self.tol, self.resid,
                           self.v, self.iparam, self.ipntr,
                           self.workd, self.workl, self.rwork, ierr)

            if ierr != 0:
                raise ArpackError(ierr, infodict=self.extract_infodict)

            k_ok = self.iparam[4]
            d = d[:k_ok]
            z = z[:, :k_ok]

        if return_eigenvectors:
            return d, z
        else:
            return d

class SpLuInv(LinearOperator):
    """
    SpLuInv:
       helper class to repeatedly solve M*x=b
       using a sparse LU-decomposition of M
    """

    def __init__(self, M):
        self.M_lu = splu(M)
        self.shape = M.shape
        self.dtype = M.dtype
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)

    def _matvec(self, x):
        # careful here: splu.solve will throw away imaginary
        # part of x if M is real
        x = np.asarray(x)
        if self.isreal and np.issubdtype(x.dtype, np.complexfloating):
            return (self.M_lu.solve(np.real(x).astype(self.dtype))
                    + 1j * self.M_lu.solve(np.imag(x).astype(self.dtype)))
        else:
            return self.M_lu.solve(x.astype(self.dtype))


class LuInv(LinearOperator):
    """
    LuInv:
       helper class to repeatedly solve M*x=b
       using an LU-decomposition of M
    """

    def __init__(self, M):
        self.M_lu = lu_factor(M)
        self.shape = M.shape
        self.dtype = M.dtype

    def _matvec(self, x):
        return lu_solve(self.M_lu, x)


def gmres_loose(A, b, tol):
    """
    gmres with looser termination condition.
    """
    b = np.asarray(b)
    min_tol = 1000 * np.sqrt(b.size) * np.finfo(b.dtype).eps
    return gmres(A, b, rtol=max(tol, min_tol), atol=0)


class IterInv(LinearOperator):
    """
    IterInv:
       helper class to repeatedly solve M*x=b
       using an iterative method.
    """

    def __init__(self, M, ifunc=gmres_loose, tol=0):
        self.M = M
        if hasattr(M, 'dtype'):
            self.dtype = M.dtype
        else:
            x = np.zeros(M.shape[1])
            self.dtype = (M * x).dtype
        self.shape = M.shape

        if tol <= 0:
            # when tol=0, ARPACK uses machine tolerance as calculated
            # by LAPACK's _LAMCH function.  We should match this
            tol = 2 * np.finfo(self.dtype).eps
        self.ifunc = ifunc
        self.tol = tol

    def _matvec(self, x):
        b, info = self.ifunc(self.M, x, tol=self.tol)
        if info != 0:
            raise ValueError("Error in inverting M: function "
                             "%s did not converge (info = %i)."
                             % (self.ifunc.__name__, info))
        return b


class IterOpInv(LinearOperator):
    """
    IterOpInv:
       helper class to repeatedly solve [A-sigma*M]*x = b
       using an iterative method
    """

    def __init__(self, A, M, sigma, ifunc=gmres_loose, tol=0):
        self.A = A
        self.M = M
        self.sigma = sigma

        def mult_func(x):
            return A.matvec(x) - sigma * M.matvec(x)

        def mult_func_M_None(x):
            return A.matvec(x) - sigma * x

        x = np.zeros(A.shape[1])
        if M is None:
            dtype = mult_func_M_None(x).dtype
            self.OP = LinearOperator(self.A.shape,
                                     mult_func_M_None,
                                     dtype=dtype)
        else:
            dtype = mult_func(x).dtype
            self.OP = LinearOperator(self.A.shape,
                                     mult_func,
                                     dtype=dtype)
        self.shape = A.shape

        if tol <= 0:
            # when tol=0, ARPACK uses machine tolerance as calculated
            # by LAPACK's _LAMCH function.  We should match this
            tol = 2 * np.finfo(self.OP.dtype).eps
        self.ifunc = ifunc
        self.tol = tol

    def _matvec(self, x):
        b, info = self.ifunc(self.OP, x, tol=self.tol)
        if info != 0:
            raise ValueError("Error in inverting [A-sigma*M]: function "
                             "%s did not converge (info = %i)."
                             % (self.ifunc.__name__, info))
        return b

    @property
    def dtype(self):
        return self.OP.dtype


def _fast_spmatrix_to_csc(A, hermitian=False):
    """Convert sparse matrix to CSC (by transposing, if possible)"""
    if (A.format == "csr" and hermitian
            and not np.issubdtype(A.dtype, np.complexfloating)):
        return A.T
    elif is_pydata_spmatrix(A):
        # No need to convert
        return A
    else:
        return A.tocsc()


def get_inv_matvec(M, hermitian=False, tol=0):
    if isdense(M):
        return LuInv(M).matvec
    elif issparse(M) or is_pydata_spmatrix(M):
        M = _fast_spmatrix_to_csc(M, hermitian=hermitian)
        return SpLuInv(M).matvec
    else:
        return IterInv(M, tol=tol).matvec


def get_OPinv_matvec(A, M, sigma, hermitian=False, tol=0):
    if sigma == 0:
        return get_inv_matvec(A, hermitian=hermitian, tol=tol)

    if M is None:
        #M is the identity matrix
        if isdense(A):
            if (np.issubdtype(A.dtype, np.complexfloating)
                    or np.imag(sigma) == 0):
                A = np.copy(A)
            else:
                A = A + 0j
            A.flat[::A.shape[1] + 1] -= sigma
            return LuInv(A).matvec
        elif issparse(A) or is_pydata_spmatrix(A):
            A = A - sigma * eye(A.shape[0])
            A = _fast_spmatrix_to_csc(A, hermitian=hermitian)
            return SpLuInv(A).matvec
        else:
            return IterOpInv(aslinearoperator(A),
                             M, sigma, tol=tol).matvec
    else:
        if ((not isdense(A) and not issparse(A) and not is_pydata_spmatrix(A)) or
                (not isdense(M) and not issparse(M) and not is_pydata_spmatrix(A))):
            return IterOpInv(aslinearoperator(A),
                             aslinearoperator(M),
                             sigma, tol=tol).matvec
        elif isdense(A) or isdense(M):
            return LuInv(A - sigma * M).matvec
        else:
            OP = A - sigma * M
            OP = _fast_spmatrix_to_csc(OP, hermitian=hermitian)
            return SpLuInv(OP).matvec


# ARPACK is not threadsafe or reentrant (SAVE variables), so we need a
# lock and a re-entering check.
_ARPACK_LOCK = ReentrancyLock("Nested calls to eigs/eighs not allowed: "
                              "ARPACK is not re-entrant")


def eigs(A, k=6, M=None, sigma=None, which='LM', v0=None,
         ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
         Minv=None, OPinv=None, OPpart=None):
    """
    Find k eigenvalues and eigenvectors of the square matrix A.

    Solves ``A @ x[i] = w[i] * x[i]``, the standard eigenvalue problem
    for w[i] eigenvalues with corresponding eigenvectors x[i].

    If M is specified, solves ``A @ x[i] = w[i] * M @ x[i]``, the
    generalized eigenvalue problem for w[i] eigenvalues
    with corresponding eigenvectors x[i]

    Parameters
    ----------
    A : ndarray, sparse matrix or LinearOperator
        An array, sparse matrix, or LinearOperator representing
        the operation ``A @ x``, where A is a real or complex square matrix.
    k : int, optional
        The number of eigenvalues and eigenvectors desired.
        `k` must be smaller than N-1. It is not possible to compute all
        eigenvectors of a matrix.
    M : ndarray, sparse matrix or LinearOperator, optional
        An array, sparse matrix, or LinearOperator representing
        the operation M@x for the generalized eigenvalue problem

            A @ x = w * M @ x.

        M must represent a real symmetric matrix if A is real, and must
        represent a complex Hermitian matrix if A is complex. For best
        results, the data type of M should be the same as that of A.
        Additionally:

            If `sigma` is None, M is positive definite

            If sigma is specified, M is positive semi-definite

        If sigma is None, eigs requires an operator to compute the solution
        of the linear equation ``M @ x = b``.  This is done internally via a
        (sparse) LU decomposition for an explicit matrix M, or via an
        iterative solver for a general linear operator.  Alternatively,
        the user can supply the matrix or operator Minv, which gives
        ``x = Minv @ b = M^-1 @ b``.
    sigma : real or complex, optional
        Find eigenvalues near sigma using shift-invert mode.  This requires
        an operator to compute the solution of the linear system
        ``[A - sigma * M] @ x = b``, where M is the identity matrix if
        unspecified. This is computed internally via a (sparse) LU
        decomposition for explicit matrices A & M, or via an iterative
        solver if either A or M is a general linear operator.
        Alternatively, the user can supply the matrix or operator OPinv,
        which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.
        For a real matrix A, shift-invert can either be done in imaginary
        mode or real mode, specified by the parameter OPpart ('r' or 'i').
        Note that when sigma is specified, the keyword 'which' (below)
        refers to the shifted eigenvalues ``w'[i]`` where:

            If A is real and OPpart == 'r' (default),
              ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``.

            If A is real and OPpart == 'i',
              ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``.

            If A is complex, ``w'[i] = 1/(w[i]-sigma)``.

    v0 : ndarray, optional
        Starting vector for iteration.
        Default: random
    ncv : int, optional
        The number of Lanczos vectors generated
        `ncv` must be greater than `k`; it is recommended that ``ncv > 2*k``.
        Default: ``min(n, max(2*k + 1, 20))``
    which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
        Which `k` eigenvectors and eigenvalues to find:

            'LM' : largest magnitude

            'SM' : smallest magnitude

            'LR' : largest real part

            'SR' : smallest real part

            'LI' : largest imaginary part

            'SI' : smallest imaginary part

        When sigma != None, 'which' refers to the shifted eigenvalues w'[i]
        (see discussion in 'sigma', above).  ARPACK is generally better
        at finding large values than small values.  If small eigenvalues are
        desired, consider using shift-invert mode for better performance.
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed
        Default: ``n*10``
    tol : float, optional
        Relative accuracy for eigenvalues (stopping criterion)
        The default value of 0 implies machine precision.
    return_eigenvectors : bool, optional
        Return eigenvectors (True) in addition to eigenvalues
    Minv : ndarray, sparse matrix or LinearOperator, optional
        See notes in M, above.
    OPinv : ndarray, sparse matrix or LinearOperator, optional
        See notes in sigma, above.
    OPpart : {'r' or 'i'}, optional
        See notes in sigma, above

    Returns
    -------
    w : ndarray
        Array of k eigenvalues.
    v : ndarray
        An array of `k` eigenvectors.
        ``v[:, i]`` is the eigenvector corresponding to the eigenvalue w[i].

    Raises
    ------
    ArpackNoConvergence
        When the requested convergence is not obtained.
        The currently converged eigenvalues and eigenvectors can be found
        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
        object.

    See Also
    --------
    eigsh : eigenvalues and eigenvectors for symmetric matrix A
    svds : singular value decomposition for a matrix A

    Notes
    -----
    This function is a wrapper to the ARPACK [1]_ SNEUPD, DNEUPD, CNEUPD,
    ZNEUPD, functions which use the Implicitly Restarted Arnoldi Method to
    find the eigenvalues and eigenvectors [2]_.

    References
    ----------
    .. [1] ARPACK Software, https://github.com/opencollab/arpack-ng
    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    Examples
    --------
    Find 6 eigenvectors of the identity matrix:

    >>> import numpy as np
    >>> from scipy.sparse.linalg import eigs
    >>> id = np.eye(13)
    >>> vals, vecs = eigs(id, k=6)
    >>> vals
    array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
    >>> vecs.shape
    (13, 6)

    """
    A = convert_pydata_sparse_to_scipy(A)
    M = convert_pydata_sparse_to_scipy(M)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix (shape={A.shape})')
    if M is not None:
        if M.shape != A.shape:
            raise ValueError(f'wrong M dimensions {M.shape}, should be {A.shape}')
        if np.dtype(M.dtype).char.lower() != np.dtype(A.dtype).char.lower():
            warnings.warn('M does not have the same type precision as A. '
                          'This may adversely affect ARPACK convergence',
                          stacklevel=2)

    n = A.shape[0]

    if k <= 0:
        raise ValueError("k=%d must be greater than 0." % k)

    if k >= n - 1:
        warnings.warn("k >= N - 1 for N * N square matrix. "
                      "Attempting to use scipy.linalg.eig instead.",
                      RuntimeWarning, stacklevel=2)

        if issparse(A):
            raise TypeError("Cannot use scipy.linalg.eig for sparse A with "
                            "k >= N - 1. Use scipy.linalg.eig(A.toarray()) or"
                            " reduce k.")
        if isinstance(A, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eig for LinearOperator "
                            "A with k >= N - 1.")
        if isinstance(M, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eig for LinearOperator "
                            "M with k >= N - 1.")

        return eig(A, b=M, right=return_eigenvectors)

    if sigma is None:
        matvec = aslinearoperator(A).matvec

        if OPinv is not None:
            raise ValueError("OPinv should not be specified "
                             "with sigma = None.")
        if OPpart is not None:
            raise ValueError("OPpart should not be specified with "
                             "sigma = None or complex A")

        if M is None:
            #standard eigenvalue problem
            mode = 1
            M_matvec = None
            Minv_matvec = None
            if Minv is not None:
                raise ValueError("Minv should not be "
                                 "specified with M = None.")
        else:
            #general eigenvalue problem
            mode = 2
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                Minv = aslinearoperator(Minv)
                Minv_matvec = Minv.matvec
            M_matvec = aslinearoperator(M).matvec
    else:
        #sigma is not None: shift-invert mode
        if np.issubdtype(A.dtype, np.complexfloating):
            if OPpart is not None:
                raise ValueError("OPpart should not be specified "
                                 "with sigma=None or complex A")
            mode = 3
        elif OPpart is None or OPpart.lower() == 'r':
            mode = 3
        elif OPpart.lower() == 'i':
            if np.imag(sigma) == 0:
                raise ValueError("OPpart cannot be 'i' if sigma is real")
            mode = 4
        else:
            raise ValueError("OPpart must be one of ('r','i')")

        matvec = aslinearoperator(A).matvec
        if Minv is not None:
            raise ValueError("Minv should not be specified when sigma is")
        if OPinv is None:
            Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                           hermitian=False, tol=tol)
        else:
            OPinv = aslinearoperator(OPinv)
            Minv_matvec = OPinv.matvec
        if M is None:
            M_matvec = None
        else:
            M_matvec = aslinearoperator(M).matvec

    params = _UnsymmetricArpackParams(n, k, A.dtype.char, matvec, mode,
                                      M_matvec, Minv_matvec, sigma,
                                      ncv, v0, maxiter, which, tol)

    with _ARPACK_LOCK:
        while not params.converged:
            params.iterate()

        return params.extract(return_eigenvectors)


def eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None,
          ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
          Minv=None, OPinv=None, mode='normal'):
    """
    Find k eigenvalues and eigenvectors of the real symmetric square matrix
    or complex Hermitian matrix A.

    Solves ``A @ x[i] = w[i] * x[i]``, the standard eigenvalue problem for
    w[i] eigenvalues with corresponding eigenvectors x[i].

    If M is specified, solves ``A @ x[i] = w[i] * M @ x[i]``, the
    generalized eigenvalue problem for w[i] eigenvalues
    with corresponding eigenvectors x[i].

    Note that there is no specialized routine for the case when A is a complex
    Hermitian matrix. In this case, ``eigsh()`` will call ``eigs()`` and return the
    real parts of the eigenvalues thus obtained.

    Parameters
    ----------
    A : ndarray, sparse matrix or LinearOperator
        A square operator representing the operation ``A @ x``, where ``A`` is
        real symmetric or complex Hermitian. For buckling mode (see below)
        ``A`` must additionally be positive-definite.
    k : int, optional
        The number of eigenvalues and eigenvectors desired.
        `k` must be smaller than N. It is not possible to compute all
        eigenvectors of a matrix.

    Returns
    -------
    w : array
        Array of k eigenvalues.
    v : array
        An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
        the eigenvector corresponding to the eigenvalue ``w[i]``.

    Other Parameters
    ----------------
    M : An N x N matrix, array, sparse matrix, or linear operator representing
        the operation ``M @ x`` for the generalized eigenvalue problem

            A @ x = w * M @ x.

        M must represent a real symmetric matrix if A is real, and must
        represent a complex Hermitian matrix if A is complex. For best
        results, the data type of M should be the same as that of A.
        Additionally:

            If sigma is None, M is symmetric positive definite.

            If sigma is specified, M is symmetric positive semi-definite.

            In buckling mode, M is symmetric indefinite.

        If sigma is None, eigsh requires an operator to compute the solution
        of the linear equation ``M @ x = b``. This is done internally via a
        (sparse) LU decomposition for an explicit matrix M, or via an
        iterative solver for a general linear operator.  Alternatively,
        the user can supply the matrix or operator Minv, which gives
        ``x = Minv @ b = M^-1 @ b``.
    sigma : real
        Find eigenvalues near sigma using shift-invert mode.  This requires
        an operator to compute the solution of the linear system
        ``[A - sigma * M] x = b``, where M is the identity matrix if
        unspecified.  This is computed internally via a (sparse) LU
        decomposition for explicit matrices A & M, or via an iterative
        solver if either A or M is a general linear operator.
        Alternatively, the user can supply the matrix or operator OPinv,
        which gives ``x = OPinv @ b = [A - sigma * M]^-1 @ b``.
        Note that when sigma is specified, the keyword 'which' refers to
        the shifted eigenvalues ``w'[i]`` where:

            if mode == 'normal', ``w'[i] = 1 / (w[i] - sigma)``.

            if mode == 'cayley', ``w'[i] = (w[i] + sigma) / (w[i] - sigma)``.

            if mode == 'buckling', ``w'[i] = w[i] / (w[i] - sigma)``.

        (see further discussion in 'mode' below)
    v0 : ndarray, optional
        Starting vector for iteration.
        Default: random
    ncv : int, optional
        The number of Lanczos vectors generated ncv must be greater than k and
        smaller than n; it is recommended that ``ncv > 2*k``.
        Default: ``min(n, max(2*k + 1, 20))``
    which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
        If A is a complex Hermitian matrix, 'BE' is invalid.
        Which `k` eigenvectors and eigenvalues to find:

            'LM' : Largest (in magnitude) eigenvalues.

            'SM' : Smallest (in magnitude) eigenvalues.

            'LA' : Largest (algebraic) eigenvalues.

            'SA' : Smallest (algebraic) eigenvalues.

            'BE' : Half (k/2) from each end of the spectrum.

        When k is odd, return one more (k/2+1) from the high end.
        When sigma != None, 'which' refers to the shifted eigenvalues ``w'[i]``
        (see discussion in 'sigma', above).  ARPACK is generally better
        at finding large values than small values.  If small eigenvalues are
        desired, consider using shift-invert mode for better performance.
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed.
        Default: ``n*10``
    tol : float
        Relative accuracy for eigenvalues (stopping criterion).
        The default value of 0 implies machine precision.
    Minv : N x N matrix, array, sparse matrix, or LinearOperator
        See notes in M, above.
    OPinv : N x N matrix, array, sparse matrix, or LinearOperator
        See notes in sigma, above.
    return_eigenvectors : bool
        Return eigenvectors (True) in addition to eigenvalues.
        This value determines the order in which eigenvalues are sorted.
        The sort order is also dependent on the `which` variable.

            For which = 'LM' or 'SA':
                If `return_eigenvectors` is True, eigenvalues are sorted by
                algebraic value.

                If `return_eigenvectors` is False, eigenvalues are sorted by
                absolute value.

            For which = 'BE' or 'LA':
                eigenvalues are always sorted by algebraic value.

            For which = 'SM':
                If `return_eigenvectors` is True, eigenvalues are sorted by
                algebraic value.

                If `return_eigenvectors` is False, eigenvalues are sorted by
                decreasing absolute value.

    mode : string ['normal' | 'buckling' | 'cayley']
        Specify strategy to use for shift-invert mode.  This argument applies
        only for real-valued A and sigma != None.  For shift-invert mode,
        ARPACK internally solves the eigenvalue problem
        ``OP @ x'[i] = w'[i] * B @ x'[i]``
        and transforms the resulting Ritz vectors x'[i] and Ritz values w'[i]
        into the desired eigenvectors and eigenvalues of the problem
        ``A @ x[i] = w[i] * M @ x[i]``.
        The modes are as follows:

            'normal' :
                OP = [A - sigma * M]^-1 @ M,
                B = M,
                w'[i] = 1 / (w[i] - sigma)

            'buckling' :
                OP = [A - sigma * M]^-1 @ A,
                B = A,
                w'[i] = w[i] / (w[i] - sigma)

            'cayley' :
                OP = [A - sigma * M]^-1 @ [A + sigma * M],
                B = M,
                w'[i] = (w[i] + sigma) / (w[i] - sigma)

        The choice of mode will affect which eigenvalues are selected by
        the keyword 'which', and can also impact the stability of
        convergence (see [2] for a discussion).

    Raises
    ------
    ArpackNoConvergence
        When the requested convergence is not obtained.

        The currently converged eigenvalues and eigenvectors can be found
        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
        object.

    See Also
    --------
    eigs : eigenvalues and eigenvectors for a general (nonsymmetric) matrix A
    svds : singular value decomposition for a matrix A

    Notes
    -----
    This function is a wrapper to the ARPACK [1]_ SSEUPD and DSEUPD
    functions which use the Implicitly Restarted Lanczos Method to
    find the eigenvalues and eigenvectors [2]_.

    References
    ----------
    .. [1] ARPACK Software, https://github.com/opencollab/arpack-ng
    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import eigsh
    >>> identity = np.eye(13)
    >>> eigenvalues, eigenvectors = eigsh(identity, k=6)
    >>> eigenvalues
    array([1., 1., 1., 1., 1., 1.])
    >>> eigenvectors.shape
    (13, 6)

    """
    # complex Hermitian matrices should be solved with eigs
    if np.issubdtype(A.dtype, np.complexfloating):
        if mode != 'normal':
            raise ValueError(f"mode={mode} cannot be used with complex matrix A")
        if which == 'BE':
            raise ValueError("which='BE' cannot be used with complex matrix A")
        elif which == 'LA':
            which = 'LR'
        elif which == 'SA':
            which = 'SR'
        ret = eigs(A, k, M=M, sigma=sigma, which=which, v0=v0,
                   ncv=ncv, maxiter=maxiter, tol=tol,
                   return_eigenvectors=return_eigenvectors, Minv=Minv,
                   OPinv=OPinv)

        if return_eigenvectors:
            return ret[0].real, ret[1]
        else:
            return ret.real

    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix (shape={A.shape})')
    if M is not None:
        if M.shape != A.shape:
            raise ValueError(f'wrong M dimensions {M.shape}, should be {A.shape}')
        if np.dtype(M.dtype).char.lower() != np.dtype(A.dtype).char.lower():
            warnings.warn('M does not have the same type precision as A. '
                          'This may adversely affect ARPACK convergence',
                          stacklevel=2)

    n = A.shape[0]

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k >= n:
        warnings.warn("k >= N for N * N square matrix. "
                      "Attempting to use scipy.linalg.eigh instead.",
                      RuntimeWarning, stacklevel=2)

        if issparse(A):
            raise TypeError("Cannot use scipy.linalg.eigh for sparse A with "
                            "k >= N. Use scipy.linalg.eigh(A.toarray()) or"
                            " reduce k.")
        if isinstance(A, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eigh for LinearOperator "
                            "A with k >= N.")
        if isinstance(M, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eigh for LinearOperator "
                            "M with k >= N.")

        return eigh(A, b=M, eigvals_only=not return_eigenvectors)

    if sigma is None:
        A = aslinearoperator(A)
        matvec = A.matvec

        if OPinv is not None:
            raise ValueError("OPinv should not be specified "
                             "with sigma = None.")
        if M is None:
            #standard eigenvalue problem
            mode = 1
            M_matvec = None
            Minv_matvec = None
            if Minv is not None:
                raise ValueError("Minv should not be "
                                 "specified with M = None.")
        else:
            #general eigenvalue problem
            mode = 2
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                Minv = aslinearoperator(Minv)
                Minv_matvec = Minv.matvec
            M_matvec = aslinearoperator(M).matvec
    else:
        # sigma is not None: shift-invert mode
        if Minv is not None:
            raise ValueError("Minv should not be specified when sigma is")

        # normal mode
        if mode == 'normal':
            mode = 3
            matvec = None
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                OPinv = aslinearoperator(OPinv)
                Minv_matvec = OPinv.matvec
            if M is None:
                M_matvec = None
            else:
                M = aslinearoperator(M)
                M_matvec = M.matvec

        # buckling mode
        elif mode == 'buckling':
            mode = 4
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                Minv_matvec = aslinearoperator(OPinv).matvec
            matvec = aslinearoperator(A).matvec
            M_matvec = None

        # cayley-transform mode
        elif mode == 'cayley':
            mode = 5
            matvec = aslinearoperator(A).matvec
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                Minv_matvec = aslinearoperator(OPinv).matvec
            if M is None:
                M_matvec = None
            else:
                M_matvec = aslinearoperator(M).matvec

        # unrecognized mode
        else:
            raise ValueError(f"unrecognized mode '{mode}'")

    params = _SymmetricArpackParams(n, k, A.dtype.char, matvec, mode,
                                    M_matvec, Minv_matvec, sigma,
                                    ncv, v0, maxiter, which, tol)

    with _ARPACK_LOCK:
        while not params.converged:
            params.iterate()

        return params.extract(return_eigenvectors)
<|endoftext|>"""Sparse Equations and Least Squares.

The original Fortran code was written by C. C. Paige and M. A. Saunders as
described in

C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
equations and sparse least squares, TOMS 8(1), 43--71 (1982).

C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
equations and least-squares problems, TOMS 8(2), 195--209 (1982).

It is licensed under the following BSD license:

Copyright (c) 2006, Systems Optimization Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of Stanford University nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The Fortran code was translated to Python for use in CVXOPT by Jeffery
Kline with contributions by Mridul Aanjaneya and Bob Myhill.

Adapted for SciPy by Stefan van der Walt.

"""

__all__ = ['lsqr']

import numpy as np
from math import sqrt
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse._sputils import convert_pydata_sparse_to_scipy

eps = np.finfo(np.float64).eps


def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

    """
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r


def lsqr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         iter_lim=None, show=False, calc_var=False, x0=None):
    """Find the least-squares solution to a large, sparse, linear system
    of equations.

    The function solves ``Ax = b``  or  ``min ||Ax - b||^2`` or
    ``min ||Ax - b||^2 + d^2 ||x - x0||^2``.

    The matrix A may be square or rectangular (over-determined or
    under-determined), and may have any rank.

    ::

      1. Unsymmetric equations --    solve  Ax = b

      2. Linear least squares  --    solve  Ax = b
                                     in the least-squares sense

      3. Damped least squares  --    solve  (   A    )*x = (    b    )
                                            ( damp*I )     ( damp*x0 )
                                     in the least-squares sense

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        Representation of an m-by-n matrix.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    damp : float
        Damping coefficient. Default is 0.
    atol, btol : float, optional
        Stopping tolerances. `lsqr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, `lsqr` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, `lsqr` terminates when ``norm(A^H r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (default),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final ``x`` will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of ``A`` and ``b`` respectively.  For example, if the entries
        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        Another stopping tolerance.  lsqr terminates if an estimate of
        ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =
        b``, `conlim` could be as large as 1.0e+12 (say).  For
        least-squares problems, conlim should be less than 1.0e+8.
        Maximum precision can be obtained by setting ``atol = btol =
        conlim = zero``, but the number of iterations may then be
        excessive. Default is 1e8.
    iter_lim : int, optional
        Explicit limitation on number of iterations (for safety).
    show : bool, optional
        Display an iteration log. Default is False.
    calc_var : bool, optional
        Whether to estimate diagonals of ``(A'A + damp^2*I)^{-1}``.
    x0 : array_like, shape (n,), optional
        Initial guess of x, if None zeros are used. Default is None.

        .. versionadded:: 1.0.0

    Returns
    -------
    x : ndarray of float
        The final solution.
    istop : int
        Gives the reason for termination.
        1 means x is an approximate solution to Ax = b.
        2 means x approximately solves the least-squares problem.
    itn : int
        Iteration number upon termination.
    r1norm : float
        ``norm(r)``, where ``r = b - Ax``.
    r2norm : float
        ``sqrt( norm(r)^2  +  damp^2 * norm(x - x0)^2 )``.  Equal to `r1norm`
        if ``damp == 0``.
    anorm : float
        Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.
    acond : float
        Estimate of ``cond(Abar)``.
    arnorm : float
        Estimate of ``norm(A'@r - damp^2*(x - x0))``.
    xnorm : float
        ``norm(x)``
    var : ndarray of float
        If ``calc_var`` is True, estimates all diagonals of
        ``(A'A)^{-1}`` (if ``damp == 0``) or more generally ``(A'A +
        damp^2*I)^{-1}``.  This is well defined if A has full column
        rank or ``damp > 0``.  (Not sure what var means if ``rank(A)
        < n`` and ``damp = 0.``)

    Notes
    -----
    LSQR uses an iterative method to approximate the solution.  The
    number of iterations required to reach a certain accuracy depends
    strongly on the scaling of the problem.  Poor scaling of the rows
    or columns of A should therefore be avoided where possible.

    For example, in problem 1 the solution is unaltered by
    row-scaling.  If a row of A is very small or large compared to
    the other rows of A, the corresponding row of ( A  b ) should be
    scaled up or down.

    In problems 1 and 2, the solution x is easily recovered
    following column-scaling.  Unless better information is known,
    the nonzero columns of A should be scaled so that they all have
    the same Euclidean norm (e.g., 1.0).

    In problem 3, there is no freedom to re-scale if damp is
    nonzero.  However, the value of damp should be assigned only
    after attention has been paid to the scaling of A.

    The parameter damp is intended to help regularize
    ill-conditioned systems, by preventing the true solution from
    being very large.  Another aid to regularization is provided by
    the parameter acond, which may be used to terminate iterations
    before the computed solution becomes very large.

    If some initial estimate ``x0`` is known and if ``damp == 0``,
    one could proceed as follows:

      1. Compute a residual vector ``r0 = b - A@x0``.
      2. Use LSQR to solve the system  ``A@dx = r0``.
      3. Add the correction dx to obtain a final solution ``x = x0 + dx``.

    This requires that ``x0`` be available before and after the call
    to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
    to solve A@x = b and k2 iterations to solve A@dx = r0.
    If x0 is "good", norm(r0) will be smaller than norm(b).
    If the same stopping tolerances atol and btol are used for each
    system, k1 and k2 will be similar, but the final solution x0 + dx
    should be more accurate.  The only way to reduce the total work
    is to use a larger stopping tolerance for the second system.
    If some value btol is suitable for A@x = b, the larger value
    btol*norm(b)/norm(r0)  should be suitable for A@dx = r0.

    Preconditioning is another way to reduce the number of iterations.
    If it is possible to solve a related system ``M@x = b``
    efficiently, where M approximates A in some helpful way (e.g. M -
    A has low rank or its elements are small relative to those of A),
    LSQR may converge more rapidly on the system ``A@M(inverse)@z =
    b``, after which x can be recovered by solving M@x = z.

    If A is symmetric, LSQR should not be used!

    Alternatives are the symmetric conjugate-gradient method (cg)
    and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
    applies to any symmetric A and will converge more rapidly than
    LSQR.  If A is positive definite, there are other implementations
    of symmetric cg that require slightly less work per iteration than
    SYMMLQ (but will take the same number of iterations).

    References
    ----------
    .. [1] C. C. Paige and M. A. Saunders (1982a).
           "LSQR: An algorithm for sparse linear equations and
           sparse least squares", ACM TOMS 8(1), 43-71.
    .. [2] C. C. Paige and M. A. Saunders (1982b).
           "Algorithm 583.  LSQR: Sparse linear equations and least
           squares problems", ACM TOMS 8(2), 195-209.
    .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
           systems using LSQR and CRAIG", BIT 35, 588-604.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import lsqr
    >>> A = csc_array([[1., 0.], [1., 1.], [0., 1.]], dtype=float)

    The first example has the trivial solution ``[0, 0]``

    >>> b = np.array([0., 0., 0.], dtype=float)
    >>> x, istop, itn, normr = lsqr(A, b)[:4]
    >>> istop
    0
    >>> x
    array([ 0.,  0.])

    The stopping code ``istop=0`` returned indicates that a vector of zeros was
    found as a solution. The returned solution `x` indeed contains
    ``[0., 0.]``. The next example has a non-trivial solution:

    >>> b = np.array([1., 0., -1.], dtype=float)
    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]
    >>> istop
    1
    >>> x
    array([ 1., -1.])
    >>> itn
    1
    >>> r1norm
    4.440892098500627e-16

    As indicated by ``istop=1``, `lsqr` found a solution obeying the tolerance
    limits. The given solution ``[1., -1.]`` obviously solves the equation. The
    remaining return values include information about the number of iterations
    (`itn=1`) and the remaining difference of left and right side of the solved
    equation.
    The final example demonstrates the behavior in the case where there is no
    solution for the equation:

    >>> b = np.array([1., 0.01, -1.], dtype=float)
    >>> x, istop, itn, r1norm = lsqr(A, b)[:4]
    >>> istop
    2
    >>> x
    array([ 1.00333333, -0.99666667])
    >>> A.dot(x)-b
    array([ 0.00333333, -0.00333333,  0.00333333])
    >>> r1norm
    0.005773502691896255

    `istop` indicates that the system is inconsistent and thus `x` is rather an
    approximate solution to the corresponding least-squares problem. `r1norm`
    contains the norm of the minimal residual that was found.
    """
    A = convert_pydata_sparse_to_scipy(A)
    A = aslinearoperator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    m, n = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n)

    msg = ('The exact solution is  x = 0                              ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = f'The matrix A has {m} rows and {n} columns'
        str2 = f'damp = {damp:20.14e}   calc_var = {calc_var:8g}'
        str3 = f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}'
        str4 = f'btol = {btol:8.2e}               iter_lim = {iter_lim:8g}'
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b - A@x,  alfa*v = A'@u.
    u = b
    bnorm = np.linalg.norm(b)

    if x0 is None:
        x = np.zeros(n)
        beta = bnorm.copy()
    else:
        x = np.asarray(x0)
        u = u - A.matvec(x)
        beta = np.linalg.norm(u)

    if beta > 0:
        u = (1/beta) * u
        v = A.rmatvec(u)
        alfa = np.linalg.norm(v)
    else:
        v = x.copy()
        alfa = 0

    if alfa > 0:
        v = (1/alfa) * v
    w = v.copy()

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        if show:
            print(msg[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'

    if show:
        print(' ')
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(str1, str2, str3)

    # Main iteration loop.
    while itn < iter_lim:
        itn = itn + 1
        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alfa, v. These satisfy the relations
        #     beta*u  =  a@v   -  alfa*u,
        #     alfa*v  =  A'@u  -  beta*v.
        u = A.matvec(v) - alfa * u
        beta = np.linalg.norm(u)

        if beta > 0:
            u = (1/beta) * u
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + dampsq)
            v = A.rmatvec(u) - beta * v
            alfa = np.linalg.norm(v)
            if alfa > 0:
                v = (1 / alfa) * v

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        if damp > 0:
            rhobar1 = sqrt(rhobar**2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            # cs1 = 1 and sn1 = 0
            rhobar1 = rhobar
            psi = 0.

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w

        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + np.linalg.norm(dk)**2

        if calc_var:
            var = var + dk**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x - x0||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x - x0||^2).
        # Although there is cancellation, it might be accurate enough.
        if damp > 0:
            r1sq = rnorm**2 - dampsq * xxnorm
            r1norm = sqrt(abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if show:
            # See if it is time to print something.
            prnt = False
            if n <= 40:
                prnt = True
            if itn <= 10:
                prnt = True
            if itn >= iter_lim-10:
                prnt = True
            # if itn%10 == 0: prnt = True
            if test3 <= 2*ctol:
                prnt = True
            if test2 <= 10*atol:
                prnt = True
            if test1 <= 10*rtol:
                prnt = True
            if istop != 0:
                prnt = True

            if prnt:
                str1 = f'{itn:6g} {x[0]:12.5e}'
                str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
                str3 = f'  {test1:8.1e} {test2:8.1e}'
                str4 = f' {anorm:8.1e} {acond:8.1e}'
                print(str1, str2, str3, str4)

        if istop != 0:
            break

    # End of iteration loop.
    # Print the stopping condition.
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        str1 = f'istop ={istop:8g}   r1norm ={r1norm:8.1e}'
        str2 = f'anorm ={anorm:8.1e}   arnorm ={arnorm:8.1e}'
        str3 = f'itn   ={itn:8g}   r2norm ={r2norm:8.1e}'
        str4 = f'acond ={acond:8.1e}   xnorm  ={xnorm:8.1e}'
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
<|endoftext|># Copyright (C) 2009, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as SciPy.

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import get_blas_funcs
from .iterative import _get_atol_rtol
from .utils import make_system

from ._gcrotmk import _fgmres

__all__ = ['lgmres']


def lgmres(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=1000, M=None, callback=None,
           inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True,
           prepend_outer_v=False):
    """
    Solve a matrix equation using the LGMRES algorithm.

    The LGMRES algorithm [1]_ [2]_ is designed to avoid some problems
    in the convergence in restarted GMRES, and often converges in fewer
    iterations.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``rtol=1e-5``, the default for ``atol`` is ``0.0``.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse array, ndarray, LinearOperator}, optional
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    inner_m : int, optional
        Number of inner GMRES iterations per each outer iteration.
    outer_k : int, optional
        Number of vectors to carry between inner GMRES iterations.
        According to [1]_, good values are in the range of 1...3.
        However, note that if you want to use the additional vectors to
        accelerate solving multiple similar problems, larger values may
        be beneficial.
    outer_v : list of tuples, optional
        List containing tuples ``(v, Av)`` of vectors and corresponding
        matrix-vector products, used to augment the Krylov subspace, and
        carried between inner GMRES iterations. The element ``Av`` can
        be `None` if the matrix-vector product should be re-evaluated.
        This parameter is modified in-place by `lgmres`, and can be used
        to pass "guess" vectors in and out of the algorithm when solving
        similar problems.
    store_outer_Av : bool, optional
        Whether LGMRES should store also A@v in addition to vectors `v`
        in the `outer_v` list. Default is True.
    prepend_outer_v : bool, optional
        Whether to put outer_v augmentation vectors before Krylov iterates.
        In standard LGMRES, prepend_outer_v=False.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : int
        Provides convergence information:

            - 0  : successful exit
            - >0 : convergence to tolerance not achieved, number of iterations
            - <0 : illegal input or breakdown

    Notes
    -----
    The LGMRES algorithm [1]_ [2]_ is designed to avoid the
    slowing of convergence in restarted GMRES, due to alternating
    residual vectors. Typically, it often outperforms GMRES(m) of
    comparable memory requirements by some measure, or at least is not
    much worse.

    Another advantage in this algorithm is that you can supply it with
    'guess' vectors in the `outer_v` argument that augment the Krylov
    subspace. If the solution lies close to the span of these vectors,
    the algorithm converges faster. This can be useful if several very
    similar matrices need to be inverted one after another, such as in
    Newton-Krylov iteration where the Jacobian matrix often changes
    little in the nonlinear steps.

    References
    ----------
    .. [1] A.H. Baker and E.R. Jessup and T. Manteuffel, "A Technique for
             Accelerating the Convergence of Restarted GMRES", SIAM J. Matrix
             Anal. Appl. 26, 962 (2005).
    .. [2] A.H. Baker, "On Improving the Performance of the Linear Solver
             restarted GMRES", PhD thesis, University of Colorado (2003).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import lgmres
    >>> A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = lgmres(A, b, atol=1e-5)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A,M,x,b,postprocess = make_system(A,M,x0,b)

    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    matvec = A.matvec
    psolve = M.matvec

    if outer_v is None:
        outer_v = []

    axpy, dot, scal = None, None, None
    nrm2 = get_blas_funcs('nrm2', [b])

    b_norm = nrm2(b)

    # we call this to get the right atol/rtol and raise errors as necessary
    atol, rtol = _get_atol_rtol('lgmres', b_norm, atol, rtol)

    if b_norm == 0:
        x = b
        return (postprocess(x), 0)

    ptol_max_factor = 1.0

    for k_outer in range(maxiter):
        r_outer = matvec(x) - b

        # -- callback
        if callback is not None:
            callback(x)

        # -- determine input type routines
        if axpy is None:
            if np.iscomplexobj(r_outer) and not np.iscomplexobj(x):
                x = x.astype(r_outer.dtype)
            axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'],
                                                   (x, r_outer))

        # -- check stopping condition
        r_norm = nrm2(r_outer)
        if r_norm <= max(atol, rtol * b_norm):
            break

        # -- inner LGMRES iteration
        v0 = -psolve(r_outer)
        inner_res_0 = nrm2(v0)

        if inner_res_0 == 0:
            rnorm = nrm2(r_outer)
            raise RuntimeError("Preconditioner returned a zero vector; "
                               f"|v| ~ {rnorm:.1g}, |M v| = 0")

        v0 = scal(1.0/inner_res_0, v0)

        ptol = min(ptol_max_factor, max(atol, rtol*b_norm)/r_norm)

        try:
            Q, R, B, vs, zs, y, pres = _fgmres(matvec,
                                               v0,
                                               inner_m,
                                               lpsolve=psolve,
                                               atol=ptol,
                                               outer_v=outer_v,
                                               prepend_outer_v=prepend_outer_v)
            y *= inner_res_0
            if not np.isfinite(y).all():
                # Overflow etc. in computation. There's no way to
                # recover from this, so we have to bail out.
                raise LinAlgError()
        except LinAlgError:
            # Floating point over/underflow, non-finite result from
            # matmul etc. -- report failure.
            return postprocess(x), k_outer + 1

        # Inner loop tolerance control
        if pres > ptol:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)
        else:
            ptol_max_factor = max(1e-16, 0.25 * ptol_max_factor)

        # -- GMRES terminated: eval solution
        dx = zs[0]*y[0]
        for w, yc in zip(zs[1:], y[1:]):
            dx = axpy(w, dx, dx.shape[0], yc)  # dx += w*yc

        # -- Store LGMRES augmentation vectors
        nx = nrm2(dx)
        if nx > 0:
            if store_outer_Av:
                q = Q.dot(R.dot(y))
                ax = vs[0]*q[0]
                for v, qc in zip(vs[1:], q[1:]):
                    ax = axpy(v, ax, ax.shape[0], qc)
                outer_v.append((dx/nx, ax/nx))
            else:
                outer_v.append((dx/nx, None))

        # -- Retain only a finite number of augmentation vectors
        while len(outer_v) > outer_k:
            del outer_v[0]

        # -- Apply step
        x += dx
    else:
        # didn't converge ...
        return postprocess(x), maxiter

    return postprocess(x), 0
<|endoftext|># Copyright (C) 2015, Pauli Virtanen <pav@iki.fi>
# Distributed under the same license as SciPy.

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import (get_blas_funcs, qr, solve, svd, qr_insert, lstsq)
from .iterative import _get_atol_rtol
from scipy.sparse.linalg._isolve.utils import make_system


__all__ = ['gcrotmk']


def _fgmres(matvec, v0, m, atol, lpsolve=None, rpsolve=None, cs=(), outer_v=(),
            prepend_outer_v=False):
    """
    FGMRES Arnoldi process, with optional projection or augmentation

    Parameters
    ----------
    matvec : callable
        Operation A*x
    v0 : ndarray
        Initial vector, normalized to nrm2(v0) == 1
    m : int
        Number of GMRES rounds
    atol : float
        Absolute tolerance for early exit
    lpsolve : callable
        Left preconditioner L
    rpsolve : callable
        Right preconditioner R
    cs : list of (ndarray, ndarray)
        Columns of matrices C and U in GCROT
    outer_v : list of ndarrays
        Augmentation vectors in LGMRES
    prepend_outer_v : bool, optional
        Whether augmentation vectors come before or after
        Krylov iterates

    Raises
    ------
    LinAlgError
        If nans encountered

    Returns
    -------
    Q, R : ndarray
        QR decomposition of the upper Hessenberg H=QR
    B : ndarray
        Projections corresponding to matrix C
    vs : list of ndarray
        Columns of matrix V
    zs : list of ndarray
        Columns of matrix Z
    y : ndarray
        Solution to ||H y - e_1||_2 = min!
    res : float
        The final (preconditioned) residual norm

    """

    if lpsolve is None:
        def lpsolve(x):
            return x
    if rpsolve is None:
        def rpsolve(x):
            return x

    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (v0,))

    vs = [v0]
    zs = []
    y = None
    res = np.nan

    m = m + len(outer_v)

    # Orthogonal projection coefficients
    B = np.zeros((len(cs), m), dtype=v0.dtype)

    # H is stored in QR factorized form
    Q = np.ones((1, 1), dtype=v0.dtype)
    R = np.zeros((1, 0), dtype=v0.dtype)

    eps = np.finfo(v0.dtype).eps

    breakdown = False

    # FGMRES Arnoldi process
    for j in range(m):
        # L A Z = C B + V H

        if prepend_outer_v and j < len(outer_v):
            z, w = outer_v[j]
        elif prepend_outer_v and j == len(outer_v):
            z = rpsolve(v0)
            w = None
        elif not prepend_outer_v and j >= m - len(outer_v):
            z, w = outer_v[j - (m - len(outer_v))]
        else:
            z = rpsolve(vs[-1])
            w = None

        if w is None:
            w = lpsolve(matvec(z))
        else:
            # w is clobbered below
            w = w.copy()

        w_norm = nrm2(w)

        # GCROT projection: L A -> (1 - C C^H) L A
        # i.e. orthogonalize against C
        for i, c in enumerate(cs):
            alpha = dot(c, w)
            B[i,j] = alpha
            w = axpy(c, w, c.shape[0], -alpha)  # w -= alpha*c

        # Orthogonalize against V
        hcur = np.zeros(j+2, dtype=Q.dtype)
        for i, v in enumerate(vs):
            alpha = dot(v, w)
            hcur[i] = alpha
            w = axpy(v, w, v.shape[0], -alpha)  # w -= alpha*v
        hcur[i+1] = nrm2(w)

        with np.errstate(over='ignore', divide='ignore'):
            # Careful with denormals
            alpha = 1/hcur[-1]

        if np.isfinite(alpha):
            w = scal(alpha, w)

        if not (hcur[-1] > eps * w_norm):
            # w essentially in the span of previous vectors,
            # or we have nans. Bail out after updating the QR
            # solution.
            breakdown = True

        vs.append(w)
        zs.append(z)

        # Arnoldi LSQ problem

        # Add new column to H=Q@R, padding other columns with zeros
        Q2 = np.zeros((j+2, j+2), dtype=Q.dtype, order='F')
        Q2[:j+1,:j+1] = Q
        Q2[j+1,j+1] = 1

        R2 = np.zeros((j+2, j), dtype=R.dtype, order='F')
        R2[:j+1,:] = R

        Q, R = qr_insert(Q2, R2, hcur, j, which='col',
                         overwrite_qru=True, check_finite=False)

        # Transformed least squares problem
        # || Q R y - inner_res_0 * e_1 ||_2 = min!
        # Since R = [R'; 0], solution is y = inner_res_0 (R')^{-1} (Q^H)[:j,0]

        # Residual is immediately known
        res = abs(Q[0,-1])

        # Check for termination
        if res < atol or breakdown:
            break

    if not np.isfinite(R[j,j]):
        # nans encountered, bail out
        raise LinAlgError()

    # -- Get the LSQ problem solution

    # The problem is triangular, but the condition number may be
    # bad (or in case of breakdown the last diagonal entry may be
    # zero), so use lstsq instead of trtrs.
    y, _, _, _, = lstsq(R[:j+1,:j+1], Q[0,:j+1].conj())

    B = B[:,:j+1]

    return Q, R, B, vs, zs, y, res


def gcrotmk(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=1000, M=None, callback=None,
            m=20, k=None, CU=None, discard_C=False, truncate='oldest'):
    """
    Solve a matrix equation using flexible GCROT(m,k) algorithm.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        `LinearOperator`.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``rtol=1e-5`` and ``atol=0.0``.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved. The
        default is ``1000``.
    M : {sparse array, ndarray, LinearOperator}, optional
        Preconditioner for `A`.  The preconditioner should approximate the
        inverse of `A`. gcrotmk is a 'flexible' algorithm and the preconditioner
        can vary from iteration to iteration. Effective preconditioning
        dramatically improves the rate of convergence, which implies that
        fewer iterations are needed to reach a given error tolerance.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as ``callback(xk)``, where ``xk`` is the current solution vector.
    m : int, optional
        Number of inner FGMRES iterations per each outer iteration.
        Default: 20
    k : int, optional
        Number of vectors to carry between inner FGMRES iterations.
        According to [2]_, good values are around `m`.
        Default: `m`
    CU : list of tuples, optional
        List of tuples ``(c, u)`` which contain the columns of the matrices
        C and U in the GCROT(m,k) algorithm. For details, see [2]_.
        The list given and vectors contained in it are modified in-place.
        If not given, start from empty matrices. The ``c`` elements in the
        tuples can be ``None``, in which case the vectors are recomputed
        via ``c = A u`` on start and orthogonalized as described in [3]_.
    discard_C : bool, optional
        Discard the C-vectors at the end. Useful if recycling Krylov subspaces
        for different linear systems.
    truncate : {'oldest', 'smallest'}, optional
        Truncation scheme to use. Drop: oldest vectors, or vectors with
        smallest singular values using the scheme discussed in [1,2].
        See [2]_ for detailed comparison.
        Default: 'oldest'

    Returns
    -------
    x : ndarray
        The solution found.
    info : int
        Provides convergence information:

        * 0  : successful exit
        * >0 : convergence to tolerance not achieved, number of iterations

    References
    ----------
    .. [1] E. de Sturler, ''Truncation strategies for optimal Krylov subspace
           methods'', SIAM J. Numer. Anal. 36, 864 (1999).
    .. [2] J.E. Hicken and D.W. Zingg, ''A simplified and flexible variant
           of GCROT for solving nonsymmetric linear systems'',
           SIAM J. Sci. Comput. 32, 172 (2010).
    .. [3] M.L. Parks, E. de Sturler, G. Mackey, D.D. Johnson, S. Maiti,
           ''Recycling Krylov subspaces for sequences of linear systems'',
           SIAM J. Sci. Comput. 28, 1651 (2006).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import gcrotmk
    >>> R = np.random.randn(5, 5)
    >>> A = csc_array(R)
    >>> b = np.random.randn(5)
    >>> x, exit_code = gcrotmk(A, b, atol=1e-5)
    >>> print(exit_code)
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    A,M,x,b,postprocess = make_system(A,M,x0,b)

    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    if truncate not in ('oldest', 'smallest'):
        raise ValueError(f"Invalid value for 'truncate': {truncate!r}")

    matvec = A.matvec
    psolve = M.matvec

    if CU is None:
        CU = []

    if k is None:
        k = m

    axpy, dot, scal = None, None, None

    if x0 is None:
        r = b.copy()
    else:
        r = b - matvec(x)

    axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'], (x, r))

    b_norm = nrm2(b)

    # we call this to get the right atol/rtol and raise errors as necessary
    atol, rtol = _get_atol_rtol('gcrotmk', b_norm, atol, rtol)

    if b_norm == 0:
        x = b
        return (postprocess(x), 0)

    if discard_C:
        CU[:] = [(None, u) for c, u in CU]

    # Reorthogonalize old vectors
    if CU:
        # Sort already existing vectors to the front
        CU.sort(key=lambda cu: cu[0] is not None)

        # Fill-in missing ones
        C = np.empty((A.shape[0], len(CU)), dtype=r.dtype, order='F')
        us = []
        j = 0
        while CU:
            # More memory-efficient: throw away old vectors as we go
            c, u = CU.pop(0)
            if c is None:
                c = matvec(u)
            C[:,j] = c
            j += 1
            us.append(u)

        # Orthogonalize
        Q, R, P = qr(C, overwrite_a=True, mode='economic', pivoting=True)
        del C

        # C := Q
        cs = list(Q.T)

        # U := U P R^-1,  back-substitution
        new_us = []
        for j in range(len(cs)):
            u = us[P[j]]
            for i in range(j):
                u = axpy(us[P[i]], u, u.shape[0], -R[i,j])
            if abs(R[j,j]) < 1e-12 * abs(R[0,0]):
                # discard rest of the vectors
                break
            u = scal(1.0/R[j,j], u)
            new_us.append(u)

        # Form the new CU lists
        CU[:] = list(zip(cs, new_us))[::-1]

    if CU:
        axpy, dot = get_blas_funcs(['axpy', 'dot'], (r,))

        # Solve first the projection operation with respect to the CU
        # vectors. This corresponds to modifying the initial guess to
        # be
        #
        #     x' = x + U y
        #     y = argmin_y || b - A (x + U y) ||^2
        #
        # The solution is y = C^H (b - A x)
        for c, u in CU:
            yc = dot(c, r)
            x = axpy(u, x, x.shape[0], yc)
            r = axpy(c, r, r.shape[0], -yc)

    # GCROT main iteration
    for j_outer in range(maxiter):
        # -- callback
        if callback is not None:
            callback(x)

        beta = nrm2(r)

        # -- check stopping condition
        beta_tol = max(atol, rtol * b_norm)

        if beta <= beta_tol and (j_outer > 0 or CU):
            # recompute residual to avoid rounding error
            r = b - matvec(x)
            beta = nrm2(r)

        if beta <= beta_tol:
            j_outer = -1
            break

        ml = m + max(k - len(CU), 0)

        cs = [c for c, u in CU]

        try:
            Q, R, B, vs, zs, y, pres = _fgmres(matvec,
                                               r/beta,
                                               ml,
                                               rpsolve=psolve,
                                               atol=max(atol, rtol*b_norm)/beta,
                                               cs=cs)
            y *= beta
        except LinAlgError:
            # Floating point over/underflow, non-finite result from
            # matmul etc. -- report failure.
            break

        #
        # At this point,
        #
        #     [A U, A Z] = [C, V] G;   G =  [ I  B ]
        #                                   [ 0  H ]
        #
        # where [C, V] has orthonormal columns, and r = beta v_0. Moreover,
        #
        #     || b - A (x + Z y + U q) ||_2 = || r - C B y - V H y - C q ||_2 = min!
        #
        # from which y = argmin_y || beta e_1 - H y ||_2, and q = -B y
        #

        #
        # GCROT(m,k) update
        #

        # Define new outer vectors

        # ux := (Z - U B) y
        ux = zs[0]*y[0]
        for z, yc in zip(zs[1:], y[1:]):
            ux = axpy(z, ux, ux.shape[0], yc)  # ux += z*yc
        by = B.dot(y)
        for cu, byc in zip(CU, by):
            c, u = cu
            ux = axpy(u, ux, ux.shape[0], -byc)  # ux -= u*byc

        # cx := V H y
        with np.errstate(invalid="ignore"):
            hy = Q.dot(R.dot(y))
        cx = vs[0] * hy[0]
        for v, hyc in zip(vs[1:], hy[1:]):
            cx = axpy(v, cx, cx.shape[0], hyc)  # cx += v*hyc

        # Normalize cx, maintaining cx = A ux
        # This new cx is orthogonal to the previous C, by construction
        try:
            alpha = 1/nrm2(cx)
            if not np.isfinite(alpha):
                raise FloatingPointError()
        except (FloatingPointError, ZeroDivisionError):
            # Cannot update, so skip it
            continue

        cx = scal(alpha, cx)
        ux = scal(alpha, ux)

        # Update residual and solution
        gamma = dot(cx, r)
        r = axpy(cx, r, r.shape[0], -gamma)  # r -= gamma*cx
        x = axpy(ux, x, x.shape[0], gamma)  # x += gamma*ux

        # Truncate CU
        if truncate == 'oldest':
            while len(CU) >= k and CU:
                del CU[0]
        elif truncate == 'smallest':
            if len(CU) >= k and CU:
                # cf. [1,2]
                D = solve(R[:-1,:].T, B.T).T
                W, sigma, V = svd(D)

                # C := C W[:,:k-1],  U := U W[:,:k-1]
                new_CU = []
                for j, w in enumerate(W[:,:k-1].T):
                    c, u = CU[0]
                    c = c * w[0]
                    u = u * w[0]
                    for cup, wp in zip(CU[1:], w[1:]):
                        cp, up = cup
                        c = axpy(cp, c, c.shape[0], wp)
                        u = axpy(up, u, u.shape[0], wp)

                    # Reorthogonalize at the same time; not necessary
                    # in exact arithmetic, but floating point error
                    # tends to accumulate here
                    for cp, up in new_CU:
                        alpha = dot(cp, c)
                        c = axpy(cp, c, c.shape[0], -alpha)
                        u = axpy(up, u, u.shape[0], -alpha)
                    alpha = nrm2(c)
                    c = scal(1.0/alpha, c)
                    u = scal(1.0/alpha, u)

                    new_CU.append((c, u))
                CU[:] = new_CU

        # Add new vector to CU
        CU.append((cx, ux))

    # Include the solution vector to the span
    CU.append((None, x.copy()))
    if discard_C:
        CU[:] = [(None, uz) for cz, uz in CU]

    return postprocess(x), j_outer + 1
<|endoftext|>__docformat__ = "restructuredtext en"

__all__ = []


from numpy import asanyarray, asarray, array, zeros

from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator, \
     IdentityOperator

_coerce_rules = {('f','f'):'f', ('f','d'):'d', ('f','F'):'F',
                 ('f','D'):'D', ('d','f'):'d', ('d','d'):'d',
                 ('d','F'):'D', ('d','D'):'D', ('F','f'):'F',
                 ('F','d'):'D', ('F','F'):'F', ('F','D'):'D',
                 ('D','f'):'D', ('D','d'):'D', ('D','F'):'D',
                 ('D','D'):'D'}


def coerce(x,y):
    if x not in 'fdFD':
        x = 'd'
    if y not in 'fdFD':
        y = 'd'
    return _coerce_rules[x,y]


def id(x):
    return x


def make_system(A, M, x0, b):
    """Make a linear system Ax=b

    Parameters
    ----------
    A : LinearOperator
        sparse or dense matrix (or any valid input to aslinearoperator)
    M : {LinearOperator, Nones}
        preconditioner
        sparse or dense matrix (or any valid input to aslinearoperator)
    x0 : {array_like, str, None}
        initial guess to iterative method.
        ``x0 = 'Mb'`` means using the nonzero initial guess ``M @ b``.
        Default is `None`, which means using the zero initial guess.
    b : array_like
        right hand side

    Returns
    -------
    (A, M, x, b, postprocess)
        A : LinearOperator
            matrix of the linear system
        M : LinearOperator
            preconditioner
        x : rank 1 ndarray
            initial guess
        b : rank 1 ndarray
            right hand side
        postprocess : function
            converts the solution vector to the appropriate
            type and dimensions (e.g. (N,1) matrix)

    """
    A_ = A
    A = aslinearoperator(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix, but got shape={(A.shape,)}')

    N = A.shape[0]

    b = asanyarray(b)

    if not (b.shape == (N,1) or b.shape == (N,)):
        raise ValueError(f'shapes of A {A.shape} and b {b.shape} are '
                         'incompatible')

    if b.dtype.char not in 'fdFD':
        b = b.astype('d')  # upcast non-FP types to double

    def postprocess(x):
        return x

    if hasattr(A,'dtype'):
        xtype = A.dtype.char
    else:
        xtype = A.matvec(b).dtype.char
    xtype = coerce(xtype, b.dtype.char)

    b = asarray(b,dtype=xtype)  # make b the same type as x
    b = b.ravel()

    # process preconditioner
    if M is None:
        if hasattr(A_,'psolve'):
            psolve = A_.psolve
        else:
            psolve = id
        if hasattr(A_,'rpsolve'):
            rpsolve = A_.rpsolve
        else:
            rpsolve = id
        if psolve is id and rpsolve is id:
            M = IdentityOperator(shape=A.shape, dtype=A.dtype)
        else:
            M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve,
                               dtype=A.dtype)
    else:
        M = aslinearoperator(M)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')

    # set initial guess
    if x0 is None:
        x = zeros(N, dtype=xtype)
    elif isinstance(x0, str):
        if x0 == 'Mb':  # use nonzero initial guess ``M @ b``
            bCopy = b.copy()
            x = M.matvec(bCopy)
    else:
        x = array(x0, dtype=xtype)
        if not (x.shape == (N, 1) or x.shape == (N,)):
            raise ValueError(f'shapes of A {A.shape} and '
                             f'x0 {x.shape} are incompatible')
        x = x.ravel()

    return A, M, x, b, postprocess
<|endoftext|>"Iterative Solvers for Sparse Linear Systems"

#from info import __doc__
from .iterative import *
from .minres import minres
from .lgmres import lgmres
from .lsqr import lsqr
from .lsmr import lsmr
from ._gcrotmk import gcrotmk
from .tfqmr import tfqmr

__all__ = [
    'bicg', 'bicgstab', 'cg', 'cgs', 'gcrotmk', 'gmres',
    'lgmres', 'lsmr', 'lsqr',
    'minres', 'qmr', 'tfqmr'
]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>from numpy import inner, zeros, inf, finfo
from numpy.linalg import norm
from math import sqrt

from .utils import make_system

__all__ = ['minres']


def minres(A, b, x0=None, *, rtol=1e-5, shift=0.0, maxiter=None,
           M=None, callback=None, show=False, check=False):
    """
    Use MINimum RESidual iteration to solve Ax=b

    MINRES minimizes norm(Ax - b) for a real symmetric matrix A.  Unlike
    the Conjugate Gradient method, A can be indefinite or singular.

    If shift != 0 then the method solves (A - shift*I)x = b

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real symmetric N-by-N matrix of the linear system
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : illegal input or breakdown

    Other Parameters
    ----------------
    x0 : ndarray
        Starting guess for the solution.
    shift : float
        Value to apply to the system ``(A - shift * I)x = b``. Default is 0.
    rtol : float
        Tolerance to achieve. The algorithm terminates when the relative
        residual is below ``rtol``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse array, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    show : bool
        If ``True``, print out a summary and metrics related to the solution
        during iterations. Default is ``False``.
    check : bool
        If ``True``, run additional input validation to check that `A` and
        `M` (if specified) are symmetric. Default is ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import minres
    >>> A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> A = A + A.T
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = minres(A, b)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    References
    ----------
    Solution of sparse indefinite systems of linear equations,
        C. C. Paige and M. A. Saunders (1975),
        SIAM J. Numer. Anal. 12(4), pp. 617-629.
        https://web.stanford.edu/group/SOL/software/minres/

    This file is a translation of the following MATLAB implementation:
        https://web.stanford.edu/group/SOL/software/minres/minres-matlab.zip

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    matvec = A.matvec
    psolve = M.matvec

    first = 'Enter minres.   '
    last = 'Exit  minres.   '

    n = A.shape[0]

    if maxiter is None:
        maxiter = 5 * n

    msg = [' beta2 = 0.  If M = I, b and x are eigenvectors    ',   # -1
            ' beta1 = 0.  The exact solution is x0          ',   # 0
            ' A solution to Ax = b was found, given rtol        ',   # 1
            ' A least-squares solution was found, given rtol    ',   # 2
            ' Reasonable accuracy achieved, given eps           ',   # 3
            ' x has converged to an eigenvector                 ',   # 4
            ' acond has exceeded 0.1/eps                        ',   # 5
            ' The iteration limit was reached                   ',   # 6
            ' A  does not define a symmetric matrix             ',   # 7
            ' M  does not define a symmetric matrix             ',   # 8
            ' M  does not define a pos-def preconditioner       ']   # 9

    if show:
        print(first + 'Solution of symmetric Ax = b')
        print(first + f'n      =  {n:3g}     shift  =  {shift:23.14e}')
        print(first + f'itnlim =  {maxiter:3g}     rtol   =  {rtol:11.2e}')
        print()

    istop = 0
    itn = 0
    Anorm = 0
    Acond = 0
    rnorm = 0
    ynorm = 0

    xtype = x.dtype

    eps = finfo(xtype).eps

    # Set up y and v for the first Lanczos vector v1.
    # y  =  beta1 P' v1,  where  P = C**(-1).
    # v is really P' v1.

    if x0 is None:
        r1 = b.copy()
    else:
        r1 = b - A@x
    y = psolve(r1)

    beta1 = inner(r1, y)

    if beta1 < 0:
        raise ValueError('indefinite preconditioner')
    elif beta1 == 0:
        return (postprocess(x), 0)

    bnorm = norm(b)
    if bnorm == 0:
        x = b
        return (postprocess(x), 0)

    beta1 = sqrt(beta1)

    if check:
        # are these too strict?

        # see if A is symmetric
        w = matvec(y)
        r2 = matvec(w)
        s = inner(w,w)
        t = inner(y,r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0/3.0)
        if z > epsa:
            raise ValueError('non-symmetric matrix')

        # see if M is symmetric
        r2 = psolve(y)
        s = inner(y,y)
        t = inner(r1,r2)
        z = abs(s - t)
        epsa = (s + eps) * eps**(1.0/3.0)
        if z > epsa:
            raise ValueError('non-symmetric preconditioner')

    # Initialize other quantities
    oldb = 0
    beta = beta1
    dbar = 0
    epsln = 0
    qrnorm = beta1
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = 0
    gmax = 0
    gmin = finfo(xtype).max
    cs = -1
    sn = 0
    w = zeros(n, dtype=xtype)
    w2 = zeros(n, dtype=xtype)
    r2 = r1

    if show:
        print()
        print()
        print('   Itn     x(1)     Compatible    LS       norm(A)  cond(A) gbar/|A|')

    while itn < maxiter:
        itn += 1

        s = 1.0/beta
        v = s*y

        y = matvec(v)
        y = y - shift * v

        if itn >= 2:
            y = y - (beta/oldb)*r1

        alfa = inner(v,y)
        y = y - (alfa/beta)*r2
        r1 = r2
        r2 = y
        y = psolve(r2)
        oldb = beta
        beta = inner(r2,y)
        if beta < 0:
            raise ValueError('non-symmetric matrix')
        beta = sqrt(beta)
        tnorm2 += alfa**2 + oldb**2 + beta**2

        if itn == 1:
            if beta/beta1 <= 10*eps:
                istop = -1  # Terminate later

        # Apply previous rotation Qk-1 to get
        #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
        #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

        oldeps = epsln
        delta = cs * dbar + sn * alfa   # delta1 = 0         deltak
        gbar = sn * dbar - cs * alfa   # gbar 1 = alfa1     gbar k
        epsln = sn * beta     # epsln2 = 0         epslnk+1
        dbar = - cs * beta   # dbar 2 = beta2     dbar k+1
        root = norm([gbar, dbar])
        Arnorm = phibar * root

        # Compute the next plane rotation Qk

        gamma = norm([gbar, beta])       # gammak
        gamma = max(gamma, eps)
        cs = gbar / gamma             # ck
        sn = beta / gamma             # sk
        phi = cs * phibar              # phik
        phibar = sn * phibar              # phibark+1

        # Update  x.

        denom = 1.0/gamma
        w1 = w2
        w2 = w
        w = (v - oldeps*w1 - delta*w2) * denom
        x = x + phi*w

        # Go round again.

        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta*z
        rhs2 = - epsln*z

        # Estimate various norms and test for convergence.

        Anorm = sqrt(tnorm2)
        ynorm = norm(x)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * rtol
        diag = gbar

        if diag == 0:
            diag = epsa

        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = inf
        else:
            test1 = rnorm / (Anorm*ynorm)    # ||r||  / (||A|| ||x||)
        if Anorm == 0:
            test2 = inf
        else:
            test2 = root / Anorm            # ||Ar|| / (||A|| ||r||)

        # Estimate  cond(A).
        # In this version we look at the diagonals of  R  in the
        # factorization of the lower Hessenberg matrix,  Q @ H = R,
        # where H is the tridiagonal matrix from Lanczos with one
        # extra row, beta(k+1) e_k^T.

        Acond = gmax/gmin

        # See if any of the stopping criteria are satisfied.
        # In rare cases, istop is already -1 from above (Abar = const*I).

        if istop == 0:
            t1 = 1 + test1      # These tests work if rtol < eps
            t2 = 1 + test2
            if t2 <= 1:
                istop = 2
            if t1 <= 1:
                istop = 1

            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1/eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            # if rnorm <= epsx   : istop = 2
            # if rnorm <= epsr   : istop = 1
            if test2 <= rtol:
                istop = 2
            if test1 <= rtol:
                istop = 1

        # See if it is time to print something.

        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 10:
            prnt = True
        if itn >= maxiter-10:
            prnt = True
        if itn % 10 == 0:
            prnt = True
        if qrnorm <= 10*epsx:
            prnt = True
        if qrnorm <= 10*epsr:
            prnt = True
        if Acond <= 1e-2/eps:
            prnt = True
        if istop != 0:
            prnt = True

        if show and prnt:
            str1 = f'{itn:6g} {x[0]:12.5e} {test1:10.3e}'
            str2 = f' {test2:10.3e}'
            str3 = f' {Anorm:8.1e} {Acond:8.1e} {gbar/Anorm:8.1e}'

            print(str1 + str2 + str3)

            if itn % 10 == 0:
                print()

        if callback is not None:
            callback(x)

        if istop != 0:
            break  # TODO check this

    if show:
        print()
        print(last + f' istop   =  {istop:3g}               itn   ={itn:5g}')
        print(last + f' Anorm   =  {Anorm:12.4e}      Acond =  {Acond:12.4e}')
        print(last + f' rnorm   =  {rnorm:12.4e}      ynorm =  {ynorm:12.4e}')
        print(last + f' Arnorm  =  {Arnorm:12.4e}')
        print(last + msg[istop+1])

    if istop == 6:
        info = maxiter
    else:
        info = 0

    return (postprocess(x),info)
<|endoftext|>import numpy as np
from .iterative import _get_atol_rtol
from .utils import make_system


__all__ = ['tfqmr']


def tfqmr(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None,
          callback=None, show=False):
    """
    Use Transpose-Free Quasi-Minimal Residual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        `scipy.sparse.linalg.LinearOperator`.
    b : {ndarray}
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : {ndarray}
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``rtol=1e-5``, the default for ``atol`` is ``0.0``.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
        Default is ``min(10000, ndofs * 10)``, where ``ndofs = A.shape[0]``.
    M : {sparse array, ndarray, LinearOperator}
        Inverse of the preconditioner of A.  M should approximate the
        inverse of A and be easy to solve for (see Notes).  Effective
        preconditioning dramatically improves the rate of convergence,
        which implies that fewer iterations are needed to reach a given
        error tolerance.  By default, no preconditioner is used.
    callback : function, optional
        User-supplied function to call after each iteration.  It is called
        as ``callback(xk)``, where ``xk`` is the current solution vector.
    show : bool, optional
        Specify ``show = True`` to show the convergence, ``show = False`` is
        to close the output of the convergence.
        Default is `False`.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : int
        Provides convergence information:

            - 0  : successful exit
            - >0 : convergence to tolerance not achieved, number of iterations
            - <0 : illegal input or breakdown

    Notes
    -----
    The Transpose-Free QMR algorithm is derived from the CGS algorithm.
    However, unlike CGS, the convergence curves for the TFQMR method is
    smoothed by computing a quasi minimization of the residual norm. The
    implementation supports left preconditioner, and the "residual norm"
    to compute in convergence criterion is actually an upper bound on the
    actual residual norm ``||b - Axk||``.

    References
    ----------
    .. [1] R. W. Freund, A Transpose-Free Quasi-Minimal Residual Algorithm for
           Non-Hermitian Linear Systems, SIAM J. Sci. Comput., 14(2), 470-482,
           1993.
    .. [2] Y. Saad, Iterative Methods for Sparse Linear Systems, 2nd edition,
           SIAM, Philadelphia, 2003.
    .. [3] C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations,
           number 16 in Frontiers in Applied Mathematics, SIAM, Philadelphia,
           1995.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import tfqmr
    >>> A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = tfqmr(A, b, atol=0.0)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """

    # Check data type
    dtype = A.dtype
    if np.issubdtype(dtype, np.int64):
        dtype = float
        A = A.astype(dtype)
    if np.issubdtype(b.dtype, np.int64):
        b = b.astype(dtype)

    A, M, x, b, postprocess = make_system(A, M, x0, b)

    # Check if the R.H.S is a zero vector
    if np.linalg.norm(b) == 0.:
        x = b.copy()
        return (postprocess(x), 0)

    ndofs = A.shape[0]
    if maxiter is None:
        maxiter = min(10000, ndofs * 10)

    if x0 is None:
        r = b.copy()
    else:
        r = b - A.matvec(x)
    u = r
    w = r.copy()
    # Take rstar as b - Ax0, that is rstar := r = b - Ax0 mathematically
    rstar = r
    v = M.matvec(A.matvec(r))
    uhat = v
    d = theta = eta = 0.
    # at this point we know rstar == r, so rho is always real
    rho = np.inner(rstar.conjugate(), r).real
    rhoLast = rho
    r0norm = np.sqrt(rho)
    tau = r0norm
    if r0norm == 0:
        return (postprocess(x), 0)

    # we call this to get the right atol and raise errors as necessary
    atol, _ = _get_atol_rtol('tfqmr', r0norm, atol, rtol)

    for iter in range(maxiter):
        even = iter % 2 == 0
        if (even):
            vtrstar = np.inner(rstar.conjugate(), v)
            # Check breakdown
            if vtrstar == 0.:
                return (postprocess(x), -1)
            alpha = rho / vtrstar
            uNext = u - alpha * v  # [1]-(5.6)
        w -= alpha * uhat  # [1]-(5.8)
        d = u + (theta**2 / alpha) * eta * d  # [1]-(5.5)
        # [1]-(5.2)
        theta = np.linalg.norm(w) / tau
        c = np.sqrt(1. / (1 + theta**2))
        tau *= theta * c
        # Calculate step and direction [1]-(5.4)
        eta = (c**2) * alpha
        z = M.matvec(d)
        x += eta * z

        if callback is not None:
            callback(x)

        # Convergence criterion
        if tau * np.sqrt(iter+1) < atol:
            if (show):
                print("TFQMR: Linear solve converged due to reach TOL "
                      f"iterations {iter+1}")
            return (postprocess(x), 0)

        if (not even):
            # [1]-(5.7)
            rho = np.inner(rstar.conjugate(), w)
            beta = rho / rhoLast
            u = w + beta * u
            v = beta * uhat + (beta**2) * v
            uhat = M.matvec(A.matvec(u))
            v += uhat
        else:
            uhat = M.matvec(A.matvec(uNext))
            u = uNext
            rhoLast = rho

    if (show):
        print("TFQMR: Linear solve not converged due to reach MAXIT "
              f"iterations {iter+1}")
    return (postprocess(x), maxiter)
<|endoftext|>"""
Copyright (C) 2010 David Fong and Michael Saunders

LSMR uses an iterative method.

07 Jun 2010: Documentation updated
03 Jun 2010: First release version in Python

David Chin-lung Fong            clfong@stanford.edu
Institute for Computational and Mathematical Engineering
Stanford University

Michael Saunders                saunders@stanford.edu
Systems Optimization Laboratory
Dept of MS&E, Stanford University.

"""

__all__ = ['lsmr']

from numpy import zeros, inf, atleast_1d, result_type
from numpy.linalg import norm
from math import sqrt
from scipy.sparse.linalg._interface import aslinearoperator

from scipy.sparse.linalg._isolve.lsqr import _sym_ortho


def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None, show=False, x0=None):
    """Iterative solver for least-squares problems.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    ``A`` is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        Matrix A in the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^H x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : array_like, shape (m,)
        Vector ``b`` in the linear system.
    damp : float
        Damping factor for regularized least-squares. `lsmr` solves
        the regularized least-squares problem::

         min ||(b) - (  A   )x||
             ||(0)   (damp*I) ||_2

        where damp is a scalar.  If damp is None or 0, the system
        is solved without regularization. Default is 0.
    atol, btol : float, optional
        Stopping tolerances. `lsmr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, `lsmr` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, `lsmr` terminates when ``norm(A^H r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (default),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final ``x`` will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of ``A`` and ``b`` respectively.  For example, if the entries
        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        `lsmr` terminates if an estimate of ``cond(A)`` exceeds
        `conlim`.  For compatible systems ``Ax = b``, conlim could be
        as large as 1.0e+12 (say).  For least-squares problems,
        `conlim` should be less than 1.0e+8. If `conlim` is None, the
        default value is 1e+8.  Maximum precision can be obtained by
        setting ``atol = btol = conlim = 0``, but the number of
        iterations may then be excessive. Default is 1e8.
    maxiter : int, optional
        `lsmr` terminates if the number of iterations reaches
        `maxiter`.  The default is ``maxiter = min(m, n)``.  For
        ill-conditioned systems, a larger value of `maxiter` may be
        needed. Default is False.
    show : bool, optional
        Print iterations logs if ``show=True``. Default is False.
    x0 : array_like, shape (n,), optional
        Initial guess of ``x``, if None zeros are used. Default is None.

        .. versionadded:: 1.0.0

    Returns
    -------
    x : ndarray of float
        Least-square solution returned.
    istop : int
        istop gives the reason for stopping::

          istop   = 0 means x=0 is a solution.  If x0 was given, then x=x0 is a
                      solution.
                  = 1 means x is an approximate solution to A@x = B,
                      according to atol and btol.
                  = 2 means x approximately solves the least-squares problem
                      according to atol.
                  = 3 means COND(A) seems to be greater than CONLIM.
                  = 4 is the same as 1 with atol = btol = eps (machine
                      precision)
                  = 5 is the same as 2 with atol = eps.
                  = 6 is the same as 3 with CONLIM = 1/eps.
                  = 7 means ITN reached maxiter before the other stopping
                      conditions were satisfied.

    itn : int
        Number of iterations used.
    normr : float
        ``norm(b-Ax)``
    normar : float
        ``norm(A^H (b - Ax))``
    norma : float
        ``norm(A)``
    conda : float
        Condition number of A.
    normx : float
        ``norm(x)``

    Notes
    -----

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           :arxiv:`1006.0758`
    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import lsmr
    >>> A = csc_array([[1., 0.], [1., 1.], [0., 1.]], dtype=float)

    The first example has the trivial solution ``[0, 0]``

    >>> b = np.array([0., 0., 0.], dtype=float)
    >>> x, istop, itn, normr = lsmr(A, b)[:4]
    >>> istop
    0
    >>> x
    array([0., 0.])

    The stopping code ``istop=0`` returned indicates that a vector of zeros was
    found as a solution. The returned solution `x` indeed contains
    ``[0., 0.]``. The next example has a non-trivial solution:

    >>> b = np.array([1., 0., -1.], dtype=float)
    >>> x, istop, itn, normr = lsmr(A, b)[:4]
    >>> istop
    1
    >>> x
    array([ 1., -1.])
    >>> itn
    1
    >>> normr
    4.440892098500627e-16

    As indicated by ``istop=1``, `lsmr` found a solution obeying the tolerance
    limits. The given solution ``[1., -1.]`` obviously solves the equation. The
    remaining return values include information about the number of iterations
    (`itn=1`) and the remaining difference of left and right side of the solved
    equation.
    The final example demonstrates the behavior in the case where there is no
    solution for the equation:

    >>> b = np.array([1., 0.01, -1.], dtype=float)
    >>> x, istop, itn, normr = lsmr(A, b)[:4]
    >>> istop
    2
    >>> x
    array([ 1.00333333, -0.99666667])
    >>> A.dot(x)-b
    array([ 0.00333333, -0.00333333,  0.00333333])
    >>> normr
    0.005773502691896255

    `istop` indicates that the system is inconsistent and thus `x` is rather an
    approximate solution to the corresponding least-squares problem. `normr`
    contains the minimal distance that was found.
    """

    A = aslinearoperator(A)
    b = atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    msg = ('The exact solution is x = 0, or x = x0, if x0 was given  ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    hdg1 = '   itn      x(1)       norm r    norm Ar'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20   # print frequency (for repeating the heading)
    pcount = 0   # print counter

    m, n = A.shape

    # stores the num of singular values
    minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim

    if x0 is None:
        dtype = result_type(A, b, float)
    else:
        dtype = result_type(A, b, x0, float)

    if show:
        print(' ')
        print('LSMR            Least-squares solution of  Ax = b\n')
        print(f'The matrix A has {m} rows and {n} columns')
        print(f'damp = {damp:20.14e}\n')
        print(f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}\n')
        print(f'btol = {btol:8.2e}             maxiter = {maxiter:8g}\n')

    u = b
    normb = norm(b)
    if x0 is None:
        x = zeros(n, dtype)
        beta = normb.copy()
    else:
        x = atleast_1d(x0.copy())
        u = u - A.matvec(x)
        beta = norm(u)

    if beta > 0:
        u = (1 / beta) * u
        v = A.rmatvec(u)
        alpha = norm(v)
    else:
        v = zeros(n, dtype)
        alpha = 0

    if alpha > 0:
        v = (1 / alpha) * v

    # Initialize variables for 1st iteration.

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = zeros(n, dtype)

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        if show:
            print(msg[0])
        return x, istop, itn, normr, normar, normA, condA, normx

    if normb == 0:
        x[()] = 0
        return x, istop, itn, normr, normar, normA, condA, normx

    if show:
        print(' ')
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {normr:10.3e} {normar:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(''.join([str1, str2, str3]))

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  A@v   -  alpha*u,
        #        alpha*v  =  A'@u  -  beta*v.

        u *= -alpha
        u += A.matvec(v)
        beta = norm(u)

        if beta > 0:
            u *= (1 / beta)
            v *= -beta
            v += A.rmatvec(u)
            alpha = norm(v)
            if alpha > 0:
                v *= (1 / alpha)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.

        hbar *= - (thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        h *= - (thetanew / rho)
        h += v

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.

        if show:
            if (n <= 40) or (itn <= 10) or (itn >= maxiter - 10) or \
               (itn % 10 == 0) or (test3 <= 1.1 * ctol) or \
               (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or \
               (istop != 0):

                if pcount >= pfreq:
                    pcount = 0
                    print(' ')
                    print(hdg1, hdg2)
                pcount = pcount + 1
                str1 = f'{itn:6g} {x[0]:12.5e}'
                str2 = f' {normr:10.3e} {normar:10.3e}'
                str3 = f'  {test1:8.1e} {test2:8.1e}'
                str4 = f' {normA:8.1e} {condA:8.1e}'
                print(''.join([str1, str2, str3, str4]))

        if istop > 0:
            break

    # Print the stopping condition.

    if show:
        print(' ')
        print('LSMR finished')
        print(msg[istop])
        print(f'istop ={istop:8g}    normr ={normr:8.1e}')
        print(f'    normA ={normA:8.1e}    normAr ={normar:8.1e}')
        print(f'itn   ={itn:8g}    condA ={condA:8.1e}')
        print(f'    normx ={normx:8.1e}')
        print(str1, str2)
        print(str3, str4)

    return x, istop, itn, normr, normar, normA, condA, normx
<|endoftext|>import warnings
import numpy as np
from scipy.sparse.linalg._interface import LinearOperator
from .utils import make_system
from scipy.linalg import get_lapack_funcs

__all__ = ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'qmr']


def _get_atol_rtol(name, b_norm, atol=0., rtol=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    if atol == 'legacy' or atol is None or atol < 0:
        msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
               "if set, `atol` must be a real, non-negative number.")
        raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol


def bicg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
    """Use BIConjugate Gradient iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse array, ndarray, LinearOperator}
        Preconditioner for `A`. It should approximate the
        inverse of `A` (see Notes). Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as ``callback(xk)``, where ``xk`` is the current solution vector.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Notes
    -----
    The preconditioner `M` should be a matrix such that ``M @ A`` has a smaller
    condition number than `A`, see [1]_ .

    References
    ----------
    .. [1] "Preconditioner", Wikipedia, 
           https://en.wikipedia.org/wiki/Preconditioner
    .. [2] "Biconjugate gradient method", Wikipedia, 
           https://en.wikipedia.org/wiki/Biconjugate_gradient_method

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import bicg
    >>> A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1.]])
    >>> b = np.array([2., 4., -1.])
    >>> x, exitCode = bicg(A, b, atol=1e-5)
    >>> print(exitCode)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('bicg', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)
    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    if maxiter is None:
        maxiter = n*10

    matvec, rmatvec = A.matvec, A.rmatvec
    psolve, rpsolve = M.matvec, M.rmatvec

    rhotol = np.finfo(x.dtype.char).eps**2

    # Dummy values to initialize vars, silence linter warnings
    rho_prev, p, ptilde = None, None, None

    r = b - matvec(x) if x.any() else b.copy()
    rtilde = r.copy()

    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), 0

        z = psolve(r)
        ztilde = rpsolve(rtilde)
        # order matters in this dot product
        rho_cur = dotprod(rtilde, z)

        if np.abs(rho_cur) < rhotol:  # Breakdown case
            return postprocess, -10

        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
            ptilde *= beta.conj()
            ptilde += ztilde
        else:  # First spin
            p = z.copy()
            ptilde = ztilde.copy()

        q = matvec(p)
        qtilde = rmatvec(ptilde)
        rv = dotprod(ptilde, q)

        if rv == 0:
            return postprocess(x), -11

        alpha = rho_cur / rv
        x += alpha*p
        r -= alpha*q
        rtilde -= alpha.conj()*qtilde
        rho_prev = rho_cur

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter


def bicgstab(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None,
             callback=None):
    """Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse array, ndarray, LinearOperator}
        Preconditioner for `A`. It should approximate the
        inverse of `A` (see Notes). Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as ``callback(xk)``, where ``xk`` is the current solution vector.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Notes
    -----
    The preconditioner `M` should be a matrix such that ``M @ A`` has a smaller
    condition number than `A`, see [1]_ .

    References
    ----------
    .. [1] "Preconditioner", Wikipedia, 
           https://en.wikipedia.org/wiki/Preconditioner
    .. [2] "Biconjugate gradient stabilized method", 
           Wikipedia, https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import bicgstab
    >>> R = np.array([[4, 2, 0, 1],
    ...               [3, 0, 0, 2],
    ...               [0, 1, 1, 1],
    ...               [0, 2, 1, 0]])
    >>> A = csc_array(R)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = bicgstab(A, b, atol=1e-5)
    >>> print(exit_code)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('bicgstab', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec

    # These values make no sense but coming from original Fortran code
    # sqrt might have been meant instead.
    rhotol = np.finfo(x.dtype.char).eps**2
    omegatol = rhotol

    # Dummy values to initialize vars, silence linter warnings
    rho_prev, omega, alpha, p, v = None, None, None, None, None

    r = b - matvec(x) if x.any() else b.copy()
    rtilde = r.copy()

    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), 0

        rho = dotprod(rtilde, r)
        if np.abs(rho) < rhotol:  # rho breakdown
            return postprocess(x), -10

        if iteration > 0:
            if np.abs(omega) < omegatol:  # omega breakdown
                return postprocess(x), -11

            beta = (rho / rho_prev) * (alpha / omega)
            p -= omega*v
            p *= beta
            p += r
        else:  # First spin
            s = np.empty_like(r)
            p = r.copy()

        phat = psolve(p)
        v = matvec(phat)
        rv = dotprod(rtilde, v)
        if rv == 0:
            return postprocess(x), -11
        alpha = rho / rv
        r -= alpha*v
        s[:] = r[:]

        if np.linalg.norm(s) < atol:
            x += alpha*phat
            return postprocess(x), 0

        shat = psolve(s)
        t = matvec(shat)
        omega = dotprod(t, s) / dotprod(t, t)
        x += alpha*phat
        x += omega*shat
        r -= omega*t
        rho_prev = rho

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter


def cg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        `A` must represent a hermitian, positive definite matrix.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse array, ndarray, LinearOperator}
        Preconditioner for `A`. `M` must represent a hermitian, positive definite
        matrix. It should approximate the inverse of `A` (see Notes).
        Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as ``callback(xk)``, where ``xk`` is the current solution vector.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations

    Notes
    -----
    The preconditioner `M` should be a matrix such that ``M @ A`` has a smaller
    condition number than `A`, see [2]_.

    References
    ----------
    .. [1] "Conjugate Gradient Method, Wikipedia, 
           https://en.wikipedia.org/wiki/Conjugate_gradient_method
    .. [2] "Preconditioner", 
           Wikipedia, https://en.wikipedia.org/wiki/Preconditioner

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import cg
    >>> P = np.array([[4, 0, 1, 0],
    ...               [0, 5, 0, 0],
    ...               [1, 0, 3, 2],
    ...               [0, 0, 2, 4]])
    >>> A = csc_array(P)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = cg(A, b, atol=1e-5)
    >>> print(exit_code)    # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('cg', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    matvec = A.matvec
    psolve = M.matvec
    r = b - matvec(x) if x.any() else b.copy()

    # Dummy value to initialize var, silences warnings
    rho_prev, p = None, None

    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), 0

        z = psolve(r)
        rho_cur = dotprod(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        q = matvec(p)
        alpha = rho_cur / dotprod(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter


def cgs(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
    """Use Conjugate Gradient Squared iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real-valued N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse array, ndarray, LinearOperator}
        Preconditioner for ``A``. It should approximate the
        inverse of `A` (see Notes). Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as ``callback(xk)``, where ``xk`` is the current solution vector.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Notes
    -----
    The preconditioner `M` should be a matrix such that ``M @ A`` has a smaller
    condition number than `A`, see [1]_.

    References
    ----------
    .. [1] "Preconditioner", Wikipedia, 
           https://en.wikipedia.org/wiki/Preconditioner
    .. [2] "Conjugate gradient squared", Wikipedia,
           https://en.wikipedia.org/wiki/Conjugate_gradient_squared_method

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import cgs
    >>> R = np.array([[4, 2, 0, 1],
    ...               [3, 0, 0, 2],
    ...               [0, 1, 1, 1],
    ...               [0, 2, 1, 0]])
    >>> A = csc_array(R)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = cgs(A, b)
    >>> print(exit_code)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('cgs', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec

    rhotol = np.finfo(x.dtype.char).eps**2

    r = b - matvec(x) if x.any() else b.copy()

    rtilde = r.copy()
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        bnorm = 1

    # Dummy values to initialize vars, silence linter warnings
    rho_prev, p, u, q = None, None, None, None

    for iteration in range(maxiter):
        rnorm = np.linalg.norm(r)
        if rnorm < atol:  # Are we done?
            return postprocess(x), 0

        rho_cur = dotprod(rtilde, r)
        if np.abs(rho_cur) < rhotol:  # Breakdown case
            return postprocess, -10

        if iteration > 0:
            beta = rho_cur / rho_prev

            # u = r + beta * q
            # p = u + beta * (q + beta * p);
            u[:] = r[:]
            u += beta*q

            p *= beta
            p += q
            p *= beta
            p += u

        else:  # First spin
            p = r.copy()
            u = r.copy()
            q = np.empty_like(r)

        phat = psolve(p)
        vhat = matvec(phat)
        rv = dotprod(rtilde, vhat)

        if rv == 0:  # Dot product breakdown
            return postprocess(x), -11

        alpha = rho_cur / rv
        q[:] = u[:]
        q -= alpha*vhat
        uhat = psolve(u + q)
        x += alpha*uhat

        # Due to numerical error build-up the actual residual is computed
        # instead of the following two lines that were in the original
        # FORTRAN templates, still using a single matvec.

        # qhat = matvec(uhat)
        # r -= alpha*qhat
        r = b - matvec(x)

        rho_prev = rho_cur

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter


def gmres(A, b, x0=None, *, rtol=1e-5, atol=0., restart=None, maxiter=None, M=None,
          callback=None, callback_type=None):
    """
    Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution (a vector of zeros by default).
    atol, rtol : float
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    restart : int, optional
        Number of iterations between restarts. Larger values increase
        iteration cost, but may be necessary for convergence.
        If omitted, ``min(20, n)`` is used.
    maxiter : int, optional
        Maximum number of iterations (restart cycles).  Iteration will stop
        after maxiter steps even if the specified tolerance has not been
        achieved. See `callback_type`.
    M : {sparse array, ndarray, LinearOperator}
        Inverse of the preconditioner of `A`.  `M` should approximate the
        inverse of `A` and be easy to solve for (see Notes).  Effective
        preconditioning dramatically improves the rate of convergence,
        which implies that fewer iterations are needed to reach a given
        error tolerance.  By default, no preconditioner is used.
        In this implementation, left preconditioning is used,
        and the preconditioned residual is minimized. However, the final
        convergence is tested with respect to the ``b - A @ x`` residual.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as ``callback(args)``, where ``args`` are selected by `callback_type`.
    callback_type : {'x', 'pr_norm', 'legacy'}, optional
        Callback function argument requested:
          - ``x``: current iterate (ndarray), called on every restart
          - ``pr_norm``: relative (preconditioned) residual norm (float),
            called on every inner iteration
          - ``legacy`` (default): same as ``pr_norm``, but also changes the
            meaning of `maxiter` to count inner iterations instead of restart
            cycles.

        This keyword has no effect if `callback` is not set.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : int
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations

    See Also
    --------
    LinearOperator

    Notes
    -----
    A preconditioner, P, is chosen such that P is close to A but easy to solve
    for. The preconditioner parameter required by this routine is
    ``M = P^-1``. The inverse should preferably not be calculated
    explicitly.  Rather, use the following template to produce M::

      # Construct a linear operator that computes P^-1 @ x.
      import scipy.sparse.linalg as spla
      M_x = lambda x: spla.spsolve(P, x)
      M = spla.LinearOperator((n, n), M_x)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import gmres
    >>> A = csc_array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = gmres(A, b, atol=1e-5)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    if callback is not None and callback_type is None:
        # Warn about 'callback_type' semantic changes.
        # Probably should be removed only in far future, Scipy 2.0 or so.
        msg = ("scipy.sparse.linalg.gmres called without specifying "
               "`callback_type`. The default value will be changed in"
               " a future release. For compatibility, specify a value "
               "for `callback_type` explicitly, e.g., "
               "``gmres(..., callback_type='pr_norm')``, or to retain the "
               "old behavior ``gmres(..., callback_type='legacy')``"
               )
        warnings.warn(msg, category=DeprecationWarning, stacklevel=3)

    if callback_type is None:
        callback_type = 'legacy'

    if callback_type not in ('x', 'pr_norm', 'legacy'):
        raise ValueError(f"Unknown callback_type: {callback_type!r}")

    if callback is None:
        callback_type = None

    A, M, x, b, postprocess = make_system(A, M, x0, b)
    matvec = A.matvec
    psolve = M.matvec
    n = len(b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('gmres', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    eps = np.finfo(x.dtype.char).eps

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    if maxiter is None:
        maxiter = n*10

    if restart is None:
        restart = 20
    restart = min(restart, n)

    Mb_nrm2 = np.linalg.norm(psolve(b))

    # ====================================================
    # =========== Tolerance control from gh-8400 =========
    # ====================================================
    # Tolerance passed to GMRESREVCOM applies to the inner
    # iteration and deals with the left-preconditioned
    # residual.
    ptol_max_factor = 1.
    ptol = Mb_nrm2 * min(ptol_max_factor, atol / bnrm2)
    presid = 0.
    # ====================================================
    lartg = get_lapack_funcs('lartg', dtype=x.dtype)

    # allocate internal variables
    v = np.empty([restart+1, n], dtype=x.dtype)
    h = np.zeros([restart, restart+1], dtype=x.dtype)
    givens = np.zeros([restart, 2], dtype=x.dtype)

    # legacy iteration count
    inner_iter = 0

    for iteration in range(maxiter):
        if iteration == 0:
            r = b - matvec(x) if x.any() else b.copy()
            if np.linalg.norm(r) < atol:  # Are we done?
                return postprocess(x), 0

        v[0, :] = psolve(r)
        tmp = np.linalg.norm(v[0, :])
        v[0, :] *= (1 / tmp)
        # RHS of the Hessenberg problem
        S = np.zeros(restart+1, dtype=x.dtype)
        S[0] = tmp

        breakdown = False
        for col in range(restart):
            av = matvec(v[col, :])
            w = psolve(av)

            # Modified Gram-Schmidt
            h0 = np.linalg.norm(w)
            for k in range(col+1):
                tmp = dotprod(v[k, :], w)
                h[col, k] = tmp
                w -= tmp*v[k, :]

            h1 = np.linalg.norm(w)
            h[col, col + 1] = h1
            v[col + 1, :] = w[:]

            # Exact solution indicator
            if h1 <= eps*h0:
                h[col, col + 1] = 0
                breakdown = True
            else:
                v[col + 1, :] *= (1 / h1)

            # apply past Givens rotations to current h column
            for k in range(col):
                c, s = givens[k, 0], givens[k, 1]
                n0, n1 = h[col, [k, k+1]]
                h[col, [k, k + 1]] = [c*n0 + s*n1, -s.conj()*n0 + c*n1]

            # get and apply current rotation to h and S
            c, s, mag = lartg(h[col, col], h[col, col+1])
            givens[col, :] = [c, s]
            h[col, [col, col+1]] = mag, 0

            # S[col+1] component is always 0
            tmp = -np.conjugate(s)*S[col]
            S[[col, col + 1]] = [c*S[col], tmp]
            presid = np.abs(tmp)
            inner_iter += 1

            if callback_type in ('legacy', 'pr_norm'):
                callback(presid / bnrm2)
            # Legacy behavior
            if callback_type == 'legacy' and inner_iter == maxiter:
                break
            if presid <= ptol or breakdown:
                break

        # Solve h(col, col) upper triangular system and allow pseudo-solve
        # singular cases as in (but without the f2py copies):
        # y = trsv(h[:col+1, :col+1].T, S[:col+1])

        if h[col, col] == 0:
            S[col] = 0

        y = np.zeros([col+1], dtype=x.dtype)
        y[:] = S[:col+1]
        for k in range(col, 0, -1):
            if y[k] != 0:
                y[k] /= h[k, k]
                tmp = y[k]
                y[:k] -= tmp*h[k, :k]
        if y[0] != 0:
            y[0] /= h[0, 0]

        x += y @ v[:col+1, :]

        r = b - matvec(x)
        rnorm = np.linalg.norm(r)

        # Legacy exit
        if callback_type == 'legacy' and inner_iter == maxiter:
            return postprocess(x), 0 if rnorm <= atol else maxiter

        if callback_type == 'x':
            callback(x)

        if rnorm <= atol:
            break
        elif breakdown:
            # Reached breakdown (= exact solution), but the external
            # tolerance check failed. Bail out with failure.
            break
        elif presid <= ptol:
            # Inner loop passed but outer didn't
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)

        ptol = presid * min(ptol_max_factor, atol / rnorm)

    info = 0 if (rnorm <= atol) else maxiter
    return postprocess(x), info


def qmr(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M1=None, M2=None,
        callback=None):
    """Use Quasi-Minimal Residual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The real-valued N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    atol, rtol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M1 : {sparse array, ndarray, LinearOperator}
        Left preconditioner for A.
    M2 : {sparse array, ndarray, LinearOperator}
        Right preconditioner for A. Used together with the left
        preconditioner M1.  The matrix M1@A@M2 should have better
        conditioned than A alone.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    See Also
    --------
    LinearOperator

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> from scipy.sparse.linalg import qmr
    >>> A = csc_array([[3., 2., 0.], [1., -1., 0.], [0., 5., 1.]])
    >>> b = np.array([2., 4., -1.])
    >>> x, exitCode = qmr(A, b, atol=1e-5)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A_ = A
    A, M, x, b, postprocess = make_system(A, None, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('qmr', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    if M1 is None and M2 is None:
        if hasattr(A_, 'psolve'):
            def left_psolve(b):
                return A_.psolve(b, 'left')

            def right_psolve(b):
                return A_.psolve(b, 'right')

            def left_rpsolve(b):
                return A_.rpsolve(b, 'left')

            def right_rpsolve(b):
                return A_.rpsolve(b, 'right')
            M1 = LinearOperator(A.shape,
                                matvec=left_psolve,
                                rmatvec=left_rpsolve)
            M2 = LinearOperator(A.shape,
                                matvec=right_psolve,
                                rmatvec=right_rpsolve)
        else:
            def id(b):
                return b
            M1 = LinearOperator(A.shape, matvec=id, rmatvec=id)
            M2 = LinearOperator(A.shape, matvec=id, rmatvec=id)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    rhotol = np.finfo(x.dtype.char).eps
    betatol = rhotol
    gammatol = rhotol
    deltatol = rhotol
    epsilontol = rhotol
    xitol = rhotol

    r = b - A.matvec(x) if x.any() else b.copy()

    vtilde = r.copy()
    y = M1.matvec(vtilde)
    rho = np.linalg.norm(y)
    wtilde = r.copy()
    z = M2.rmatvec(wtilde)
    xi = np.linalg.norm(z)
    gamma, eta, theta = 1, -1, 0
    v = np.empty_like(vtilde)
    w = np.empty_like(wtilde)

    # Dummy values to initialize vars, silence linter warnings
    epsilon, q, d, p, s = None, None, None, None, None

    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), 0
        if np.abs(rho) < rhotol:  # rho breakdown
            return postprocess(x), -10
        if np.abs(xi) < xitol:  # xi breakdown
            return postprocess(x), -15

        v[:] = vtilde[:]
        v *= (1 / rho)
        y *= (1 / rho)
        w[:] = wtilde[:]
        w *= (1 / xi)
        z *= (1 / xi)
        delta = dotprod(z, y)

        if np.abs(delta) < deltatol:  # delta breakdown
            return postprocess(x), -13

        ytilde = M2.matvec(y)
        ztilde = M1.rmatvec(z)

        if iteration > 0:
            ytilde -= (xi * delta / epsilon) * p
            p[:] = ytilde[:]
            ztilde -= (rho * (delta / epsilon).conj()) * q
            q[:] = ztilde[:]
        else:  # First spin
            p = ytilde.copy()
            q = ztilde.copy()

        ptilde = A.matvec(p)
        epsilon = dotprod(q, ptilde)
        if np.abs(epsilon) < epsilontol:  # epsilon breakdown
            return postprocess(x), -14

        beta = epsilon / delta
        if np.abs(beta) < betatol:  # beta breakdown
            return postprocess(x), -11

        vtilde[:] = ptilde[:]
        vtilde -= beta*v
        y = M1.matvec(vtilde)

        rho_prev = rho
        rho = np.linalg.norm(y)
        wtilde[:] = w[:]
        wtilde *= - beta.conj()
        wtilde += A.rmatvec(q)
        z = M2.rmatvec(wtilde)
        xi = np.linalg.norm(z)
        gamma_prev = gamma
        theta_prev = theta
        theta = rho / (gamma_prev * np.abs(beta))
        gamma = 1 / np.sqrt(1 + theta**2)

        if np.abs(gamma) < gammatol:  # gamma breakdown
            return postprocess(x), -12

        eta *= -(rho_prev / beta) * (gamma / gamma_prev)**2

        if iteration > 0:
            d *= (theta_prev * gamma) ** 2
            d += eta*p
            s *= (theta_prev * gamma) ** 2
            s += eta*ptilde
        else:
            d = p.copy()
            d *= eta
            s = ptilde.copy()
            s *= eta

        x += d
        r -= s

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.fftpack` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq', 'next_fast_len'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="fftpack", module="helper",
                                   private_modules=["_helper"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Real spectrum transforms (DCT, DST, MDCT)
"""

__all__ = ['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn']

from scipy.fft import _pocketfft
from ._helper import _good_shape

_inverse_typemap = {1: 1, 2: 3, 3: 2, 4: 4}


def dctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
    """
    Return multidimensional Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    shape : int or array_like of ints or None, optional
        The shape of the result. If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes along which the DCT is computed.
        The default is over all axes.
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idctn : Inverse multidimensional DCT

    Notes
    -----
    For full details of the DCT types and normalization modes, as well as
    references, see `dct`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
    True

    """
    shape = _good_shape(x, shape, axes)
    return _pocketfft.dctn(x, type, shape, axes, norm, overwrite_x)


def idctn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
    """
    Return multidimensional Discrete Cosine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    shape : int or array_like of ints or None, optional
        The shape of the result.  If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes along which the IDCT is computed.
        The default is over all axes.
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    dctn : multidimensional DCT

    Notes
    -----
    For full details of the IDCT types and normalization modes, as well as
    references, see `idct`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import dctn, idctn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idctn(dctn(y, norm='ortho'), norm='ortho'))
    True

    """
    type = _inverse_typemap[type]
    shape = _good_shape(x, shape, axes)
    return _pocketfft.dctn(x, type, shape, axes, norm, overwrite_x)


def dstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
    """
    Return multidimensional Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    shape : int or array_like of ints or None, optional
        The shape of the result.  If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes along which the DCT is computed.
        The default is over all axes.
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idstn : Inverse multidimensional DST

    Notes
    -----
    For full details of the DST types and normalization modes, as well as
    references, see `dst`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
    True

    """
    shape = _good_shape(x, shape, axes)
    return _pocketfft.dstn(x, type, shape, axes, norm, overwrite_x)


def idstn(x, type=2, shape=None, axes=None, norm=None, overwrite_x=False):
    """
    Return multidimensional Discrete Sine Transform along the specified axes.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    shape : int or array_like of ints or None, optional
        The shape of the result.  If both `shape` and `axes` (see below) are
        None, `shape` is ``x.shape``; if `shape` is None but `axes` is
        not None, then `shape` is ``numpy.take(x.shape, axes, axis=0)``.
        If ``shape[i] > x.shape[i]``, the ith dimension is padded with zeros.
        If ``shape[i] < x.shape[i]``, the ith dimension is truncated to
        length ``shape[i]``.
        If any element of `shape` is -1, the size of the corresponding
        dimension of `x` is used.
    axes : int or array_like of ints or None, optional
        Axes along which the IDST is computed.
        The default is over all axes.
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    dstn : multidimensional DST

    Notes
    -----
    For full details of the IDST types and normalization modes, as well as
    references, see `idst`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.fftpack import dstn, idstn
    >>> rng = np.random.default_rng()
    >>> y = rng.standard_normal((16, 16))
    >>> np.allclose(y, idstn(dstn(y, norm='ortho'), norm='ortho'))
    True

    """
    type = _inverse_typemap[type]
    shape = _good_shape(x, shape, axes)
    return _pocketfft.dstn(x, type, shape, axes, norm, overwrite_x)


def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    r"""
    Return the Discrete Cosine Transform of arbitrary type sequence x.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the dct is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    y : ndarray of real
        The transformed input array.

    See Also
    --------
    idct : Inverse DCT

    Notes
    -----
    For a single dimension array ``x``, ``dct(x, norm='ortho')`` is equal to
    MATLAB ``dct(x)``.

    There are, theoretically, 8 types of the DCT, only the first 4 types are
    implemented in scipy. 'The' DCT generally refers to DCT type 2, and 'the'
    Inverse DCT generally refers to DCT type 3.

    **Type I**

    There are several definitions of the DCT-I; we use the following
    (for ``norm=None``)

    .. math::

       y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
       \frac{\pi k n}{N-1} \right)

    If ``norm='ortho'``, ``x[0]`` and ``x[N-1]`` are multiplied by a scaling
    factor of :math:`\sqrt{2}`, and ``y[k]`` is multiplied by a scaling factor
    ``f``

    .. math::

        f = \begin{cases}
         \frac{1}{2}\sqrt{\frac{1}{N-1}} & \text{if }k=0\text{ or }N-1, \\
         \frac{1}{2}\sqrt{\frac{2}{N-1}} & \text{otherwise} \end{cases}

    .. versionadded:: 1.2.0
       Orthonormalization in DCT-I.

    .. note::
       The DCT-I is only supported for input size > 1.

    **Type II**

    There are several definitions of the DCT-II; we use the following
    (for ``norm=None``)

    .. math::

       y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)

    If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

    .. math::
       f = \begin{cases}
       \sqrt{\frac{1}{4N}} & \text{if }k=0, \\
       \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

    which makes the corresponding matrix of coefficients orthonormal
    (``O @ O.T = np.eye(N)``).

    **Type III**

    There are several definitions, we use the following (for ``norm=None``)

    .. math::

       y_k = x_0 + 2 \sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi(2k+1)n}{2N}\right)

    or, for ``norm='ortho'``

    .. math::

       y_k = \frac{x_0}{\sqrt{N}} + \sqrt{\frac{2}{N}} \sum_{n=1}^{N-1} x_n
       \cos\left(\frac{\pi(2k+1)n}{2N}\right)

    The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
    to a factor ``2N``. The orthonormalized DCT-III is exactly the inverse of
    the orthonormalized DCT-II.

    **Type IV**

    There are several definitions of the DCT-IV; we use the following
    (for ``norm=None``)

    .. math::

       y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi(2k+1)(2n+1)}{4N} \right)

    If ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

    .. math::

        f = \frac{1}{\sqrt{2N}}

    .. versionadded:: 1.2.0
       Support for DCT-IV.

    References
    ----------
    .. [1] 'A Fast Cosine Transform in One and Two Dimensions', by J.
           Makhoul, `IEEE Transactions on acoustics, speech and signal
           processing` vol. 28(1), pp. 27-34,
           :doi:`10.1109/TASSP.1980.1163351` (1980).
    .. [2] Wikipedia, "Discrete cosine transform",
           https://en.wikipedia.org/wiki/Discrete_cosine_transform

    Examples
    --------
    The Type 1 DCT is equivalent to the FFT (though faster) for real,
    even-symmetrical inputs. The output is also real and even-symmetrical.
    Half of the FFT input is used to generate half of the FFT output:

    >>> from scipy.fftpack import fft, dct
    >>> import numpy as np
    >>> fft(np.array([4., 3., 5., 10., 5., 3.])).real
    array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])
    >>> dct(np.array([4., 3., 5., 10.]), 1)
    array([ 30.,  -8.,   6.,  -2.])

    """
    return _pocketfft.dct(x, type, n, axis, norm, overwrite_x)


def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    """
    Return the Inverse Discrete Cosine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DCT (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idct is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    idct : ndarray of real
        The transformed input array.

    See Also
    --------
    dct : Forward DCT

    Notes
    -----
    For a single dimension array `x`, ``idct(x, norm='ortho')`` is equal to
    MATLAB ``idct(x)``.

    'The' IDCT is the IDCT of type 2, which is the same as DCT of type 3.

    IDCT of type 1 is the DCT of type 1, IDCT of type 2 is the DCT of type
    3, and IDCT of type 3 is the DCT of type 2. IDCT of type 4 is the DCT
    of type 4. For the definition of these types, see `dct`.

    Examples
    --------
    The Type 1 DCT is equivalent to the DFT for real, even-symmetrical
    inputs. The output is also real and even-symmetrical. Half of the IFFT
    input is used to generate half of the IFFT output:

    >>> from scipy.fftpack import ifft, idct
    >>> import numpy as np
    >>> ifft(np.array([ 30.,  -8.,   6.,  -2.,   6.,  -8.])).real
    array([  4.,   3.,   5.,  10.,   5.,   3.])
    >>> idct(np.array([ 30.,  -8.,   6.,  -2.]), 1) / 6
    array([  4.,   3.,   5.,  10.])

    """
    type = _inverse_typemap[type]
    return _pocketfft.dct(x, type, n, axis, norm, overwrite_x)


def dst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    r"""
    Return the Discrete Sine Transform of arbitrary type sequence x.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the dst is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    dst : ndarray of reals
        The transformed input array.

    See Also
    --------
    idst : Inverse DST

    Notes
    -----
    For a single dimension array ``x``.

    There are, theoretically, 8 types of the DST for different combinations of
    even/odd boundary conditions and boundary off sets [1]_, only the first
    4 types are implemented in scipy.

    **Type I**

    There are several definitions of the DST-I; we use the following
    for ``norm=None``. DST-I assumes the input is odd around `n=-1` and `n=N`.

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(n+1)}{N+1}\right)

    Note that the DST-I is only supported for input size > 1.
    The (unnormalized) DST-I is its own inverse, up to a factor ``2(N+1)``.
    The orthonormalized DST-I is exactly its own inverse.

    **Type II**

    There are several definitions of the DST-II; we use the following for
    ``norm=None``. DST-II assumes the input is odd around `n=-1/2` and
    `n=N-1/2`; the output is odd around :math:`k=-1` and even around `k=N-1`

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(k+1)(2n+1)}{2N}\right)

    if ``norm='ortho'``, ``y[k]`` is multiplied by a scaling factor ``f``

    .. math::

        f = \begin{cases}
        \sqrt{\frac{1}{4N}} & \text{if }k = 0, \\
        \sqrt{\frac{1}{2N}} & \text{otherwise} \end{cases}

    **Type III**

    There are several definitions of the DST-III, we use the following (for
    ``norm=None``). DST-III assumes the input is odd around `n=-1` and even
    around `n=N-1`

    .. math::

        y_k = (-1)^k x_{N-1} + 2 \sum_{n=0}^{N-2} x_n \sin\left(
        \frac{\pi(2k+1)(n+1)}{2N}\right)

    The (unnormalized) DST-III is the inverse of the (unnormalized) DST-II, up
    to a factor ``2N``. The orthonormalized DST-III is exactly the inverse of the
    orthonormalized DST-II.

    .. versionadded:: 0.11.0

    **Type IV**

    There are several definitions of the DST-IV, we use the following (for
    ``norm=None``). DST-IV assumes the input is odd around `n=-0.5` and even
    around `n=N-0.5`

    .. math::

        y_k = 2 \sum_{n=0}^{N-1} x_n \sin\left(\frac{\pi(2k+1)(2n+1)}{4N}\right)

    The (unnormalized) DST-IV is its own inverse, up to a factor ``2N``. The
    orthonormalized DST-IV is exactly its own inverse.

    .. versionadded:: 1.2.0
       Support for DST-IV.

    References
    ----------
    .. [1] Wikipedia, "Discrete sine transform",
           https://en.wikipedia.org/wiki/Discrete_sine_transform

    """
    return _pocketfft.dst(x, type, n, axis, norm, overwrite_x)


def idst(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False):
    """
    Return the Inverse Discrete Sine Transform of an arbitrary type sequence.

    Parameters
    ----------
    x : array_like
        The input array.
    type : {1, 2, 3, 4}, optional
        Type of the DST (see Notes). Default type is 2.
    n : int, optional
        Length of the transform.  If ``n < x.shape[axis]``, `x` is
        truncated. If ``n > x.shape[axis]``, `x` is zero-padded. The
        default results in ``n = x.shape[axis]``.
    axis : int, optional
        Axis along which the idst is computed; the default is over the
        last axis (i.e., ``axis=-1``).
    norm : {None, 'ortho'}, optional
        Normalization mode (see Notes). Default is None.
    overwrite_x : bool, optional
        If True, the contents of `x` can be destroyed; the default is False.

    Returns
    -------
    idst : ndarray of real
        The transformed input array.

    See Also
    --------
    dst : Forward DST

    Notes
    -----
    'The' IDST is the IDST of type 2, which is the same as DST of type 3.

    IDST of type 1 is the DST of type 1, IDST of type 2 is the DST of type
    3, and IDST of type 3 is the DST of type 2. For the definition of these
    types, see `dst`.

    .. versionadded:: 0.11.0

    """
    type = _inverse_typemap[type]
    return _pocketfft.dst(x, type, n, axis, norm, overwrite_x)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.fftpack` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn', 'idstn'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="fftpack", module="realtransforms",
                                   private_modules=["_realtransforms"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.fftpack` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'diff',
    'tilbert', 'itilbert', 'hilbert', 'ihilbert',
    'cs_diff', 'cc_diff', 'sc_diff', 'ss_diff',
    'shift', 'convolve'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="fftpack", module="pseudo_diffs",
                                   private_modules=["_pseudo_diffs"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Differential and pseudo-differential operators.
"""
# Created by Pearu Peterson, September 2002

__all__ = ['diff',
           'tilbert','itilbert','hilbert','ihilbert',
           'cs_diff','cc_diff','sc_diff','ss_diff',
           'shift']

import threading

from numpy import pi, asarray, sin, cos, sinh, cosh, tanh, iscomplexobj
from . import convolve

from scipy.fft._pocketfft.helper import _datacopied


_cache = threading.local()


def diff(x,order=1,period=None, _cache=_cache):
    """
    Return kth derivative (or integral) of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = pow(sqrt(-1)*j*2*pi/period, order) * x_j
      y_0 = 0 if order is not 0.

    Parameters
    ----------
    x : array_like
        Input array.
    order : int, optional
        The order of differentiation. Default order is 1. If order is
        negative, then integration is carried out under the assumption
        that ``x_0 == 0``.
    period : float, optional
        The assumed period of the sequence. Default is ``2*pi``.

    Notes
    -----
    If ``sum(x, axis=0) = 0`` then ``diff(diff(x, k), -k) == x`` (within
    numerical accuracy).

    For odd order and even ``len(x)``, the Nyquist mode is taken zero.

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'diff_cache'):
            _cache.diff_cache = {}
        _cache = _cache.diff_cache

    tmp = asarray(x)
    if order == 0:
        return tmp
    if iscomplexobj(tmp):
        return diff(tmp.real, order, period, _cache)+1j*diff(
            tmp.imag, order, period, _cache)
    if period is not None:
        c = 2*pi/period
    else:
        c = 1.0
    n = len(x)
    omega = _cache.get((n,order,c))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k,order=order,c=c):
            if k:
                return pow(c*k,order)
            return 0
        omega = convolve.init_convolution_kernel(n,kernel,d=order,
                                                 zero_nyquist=1)
        _cache[(n,order,c)] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp,omega,swap_real_imag=order % 2,
                             overwrite_x=overwrite_x)


def tilbert(x, h, period=None, _cache=_cache):
    """
    Return h-Tilbert transform of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

        y_j = sqrt(-1)*coth(j*h*2*pi/period) * x_j
        y_0 = 0

    Parameters
    ----------
    x : array_like
        The input array to transform.
    h : float
        Defines the parameter of the Tilbert transform.
    period : float, optional
        The assumed period of the sequence. Default period is ``2*pi``.

    Returns
    -------
    tilbert : ndarray
        The result of the transform.

    Notes
    -----
    If ``sum(x, axis=0) == 0`` and ``n = len(x)`` is odd, then
    ``tilbert(itilbert(x)) == x``.

    If ``2 * pi * h / period`` is approximately 10 or larger, then
    numerically ``tilbert == hilbert``
    (theoretically oo-Tilbert == Hilbert).

    For even ``len(x)``, the Nyquist mode of ``x`` is taken zero.

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'tilbert_cache'):
            _cache.tilbert_cache = {}
        _cache = _cache.tilbert_cache

    tmp = asarray(x)
    if iscomplexobj(tmp):
        return tilbert(tmp.real, h, period, _cache) + \
               1j * tilbert(tmp.imag, h, period, _cache)

    if period is not None:
        h = h * 2 * pi / period

    n = len(x)
    omega = _cache.get((n, h))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k, h=h):
            if k:
                return 1.0/tanh(h*k)

            return 0

        omega = convolve.init_convolution_kernel(n, kernel, d=1)
        _cache[(n,h)] = omega

    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)


def itilbert(x,h,period=None, _cache=_cache):
    """
    Return inverse h-Tilbert transform of a periodic sequence x.

    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = -sqrt(-1)*tanh(j*h*2*pi/period) * x_j
      y_0 = 0

    For more details, see `tilbert`.

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'itilbert_cache'):
            _cache.itilbert_cache = {}
        _cache = _cache.itilbert_cache

    tmp = asarray(x)
    if iscomplexobj(tmp):
        return itilbert(tmp.real, h, period, _cache) + \
               1j*itilbert(tmp.imag, h, period, _cache)
    if period is not None:
        h = h*2*pi/period
    n = len(x)
    omega = _cache.get((n,h))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k,h=h):
            if k:
                return -tanh(h*k)
            return 0
        omega = convolve.init_convolution_kernel(n,kernel,d=1)
        _cache[(n,h)] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)


def hilbert(x, _cache=_cache):
    """
    Return Hilbert transform of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = sqrt(-1)*sign(j) * x_j
      y_0 = 0

    Parameters
    ----------
    x : array_like
        The input array, should be periodic.
    _cache : dict, optional
        Dictionary that contains the kernel used to do a convolution with.

    Returns
    -------
    y : ndarray
        The transformed input.

    See Also
    --------
    scipy.signal.hilbert : Compute the analytic signal, using the Hilbert
                           transform.

    Notes
    -----
    If ``sum(x, axis=0) == 0`` then ``hilbert(ihilbert(x)) == x``.

    For even len(x), the Nyquist mode of x is taken zero.

    The sign of the returned transform does not have a factor -1 that is more
    often than not found in the definition of the Hilbert transform. Note also
    that `scipy.signal.hilbert` does have an extra -1 factor compared to this
    function.

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'hilbert_cache'):
            _cache.hilbert_cache = {}
        _cache = _cache.hilbert_cache

    tmp = asarray(x)
    if iscomplexobj(tmp):
        return hilbert(tmp.real, _cache) + 1j * hilbert(tmp.imag, _cache)
    n = len(x)
    omega = _cache.get(n)
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k):
            if k > 0:
                return 1.0
            elif k < 0:
                return -1.0
            return 0.0
        omega = convolve.init_convolution_kernel(n,kernel,d=1)
        _cache[n] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)


def ihilbert(x, _cache=_cache):
    """
    Return inverse Hilbert transform of a periodic sequence x.

    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = -sqrt(-1)*sign(j) * x_j
      y_0 = 0

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'ihilbert_cache'):
            _cache.ihilbert_cache = {}
        _cache = _cache.ihilbert_cache
    return -hilbert(x, _cache)


def cs_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-cosh/sinh pseudo-derivative of a periodic sequence.

    If ``x_j`` and ``y_j`` are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = -sqrt(-1)*cosh(j*a*2*pi/period)/sinh(j*b*2*pi/period) * x_j
      y_0 = 0

    Parameters
    ----------
    x : array_like
        The array to take the pseudo-derivative from.
    a, b : float
        Defines the parameters of the cosh/sinh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence. Default period is ``2*pi``.

    Returns
    -------
    cs_diff : ndarray
        Pseudo-derivative of periodic sequence `x`.

    Notes
    -----
    For even len(`x`), the Nyquist mode of `x` is taken as zero.

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'cs_diff_cache'):
            _cache.cs_diff_cache = {}
        _cache = _cache.cs_diff_cache

    tmp = asarray(x)
    if iscomplexobj(tmp):
        return cs_diff(tmp.real, a, b, period, _cache) + \
               1j*cs_diff(tmp.imag, a, b, period, _cache)
    if period is not None:
        a = a*2*pi/period
        b = b*2*pi/period
    n = len(x)
    omega = _cache.get((n,a,b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k,a=a,b=b):
            if k:
                return -cosh(a*k)/sinh(b*k)
            return 0
        omega = convolve.init_convolution_kernel(n,kernel,d=1)
        _cache[(n,a,b)] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)


def sc_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-sinh/cosh pseudo-derivative of a periodic sequence x.

    If x_j and y_j are Fourier coefficients of periodic functions x
    and y, respectively, then::

      y_j = sqrt(-1)*sinh(j*a*2*pi/period)/cosh(j*b*2*pi/period) * x_j
      y_0 = 0

    Parameters
    ----------
    x : array_like
        Input array.
    a,b : float
        Defines the parameters of the sinh/cosh pseudo-differential
        operator.
    period : float, optional
        The period of the sequence x. Default is 2*pi.

    Notes
    -----
    ``sc_diff(cs_diff(x,a,b),b,a) == x``
    For even ``len(x)``, the Nyquist mode of x is taken as zero.

    """
    if isinstance(_cache, threading.local):
        if not hasattr(_cache, 'sc_diff_cache'):
            _cache.sc_diff_cache = {}
        _cache = _cache.sc_diff_cache

    tmp = asarray(x)
    if iscomplexobj(tmp):
        return sc_diff(tmp.real, a, b, period, _cache) + \
               1j * sc_diff(tmp.imag, a, b, period, _cache)
    if period is not None:
        a = a*2*pi/period
        b = b*2*pi/period
    n = len(x)
    omega = _cache.get((n,a,b))
    if omega is None:
        if len(_cache) > 20:
            while _cache:
                _cache.popitem()

        def kernel(k,a=a,b=b):
            if k:
                return sinh(a*k)/cosh(b*k)
            return 0
        omega = convolve.init_convolution_kernel(n,kernel,d=1)
        _cache[(n,a,b)] = omega
    overwrite_x = _datacopied(tmp, x)
    return convolve.convolve(tmp,omega,swap_real_imag=1,overwrite_x=overwrite_x)


def ss_diff(x, a, b, period=None, _cache=_cache):
    """
    Return (a,b)-sinh/sinh pseudo-derivative of a periodic sequence x.

