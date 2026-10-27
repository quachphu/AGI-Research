                    b_vl = self.V[tuple(b_vl)]
                    bc_vc.connect(b_vl)  # Connect aN cross pairs
                    d_bc_vc.connect(b_vl)  # Connect all to centroid

                    yield b_vl
                    b_vu = self.V[tuple(b_vu)]
                    bc_vc.connect(b_vu)  # Connect aN cross pairs
                    d_bc_vc.connect(b_vu)  # Connect all to centroid

                    b_vl_c = self.split_edge(b_vu.x, b_vl.x)
                    bc_vc.connect(b_vl_c)

                    yield b_vu
                    ba_vu = self.V[tuple(ba_vu)]
                    bc_vc.connect(ba_vu)  # Connect aN cross pairs
                    d_bc_vc.connect(ba_vu)  # Connect all to centroid

                    # Split the a + b edge of the initial triangulation:
                    os_v = self.split_edge(vectors[1].x, ba_vu.x)  # o-s
                    ss_v = self.split_edge(b_vl.x, ba_vu.x)  # s-s
                    b_vu_c = self.split_edge(b_vu.x, ba_vu.x)
                    bc_vc.connect(b_vu_c)
                    yield os_v.x  # often equal to vco, but not always
                    yield ss_v.x  # often equal to bc_vu, but not always
                    yield ba_vu
                    # Split remaining to centre, call this centre group
                    # "d = 0.5*a"
                    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    yield d_bc_vc.x
                    d_b_vl = self.split_edge(vectors[1].x, b_vl.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_b_vl)  # Connect dN cross pairs
                    yield d_b_vl.x
                    d_b_vu = self.split_edge(vectors[2].x, b_vu.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_b_vu)  # Connect dN cross pairs
                    yield d_b_vu.x
                    d_ba_vu = self.split_edge(vectors[3].x, ba_vu.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_ba_vu)  # Connect dN cross pairs
                    yield d_ba_vu

                    # comb = [c_vc, vl, vu, a_vl, a_vu,
                    #       bc_vc, b_vl, b_vu, ba_vl, ba_vu]
                    comb = [vl, vu, a_vu,
                            b_vl, b_vu, ba_vu]
                    comb_iter = itertools.combinations(comb, 2)
                    for vecs in comb_iter:
                        self.split_edge(vecs[0].x, vecs[1].x)
                    # Add new list of cross pairs
                    ab_C.append((d_bc_vc, vectors[1], b_vl, a_vu, ba_vu))
                    ab_C.append((d_bc_vc, vl, b_vl, a_vu, ba_vu))  # = prev

                for vectors in ab_Cc:
                    bc_vc = list(vectors[0].x)
                    b_vl = list(vectors[1].x)
                    b_vu = list(vectors[2].x)
                    ba_vl = list(vectors[3].x)
                    ba_vu = list(vectors[4].x)
                    bc_vc[i + 1] = vut[i + 1]
                    b_vl[i + 1] = vut[i + 1]
                    b_vu[i + 1] = vut[i + 1]
                    ba_vl[i + 1] = vut[i + 1]
                    ba_vu[i + 1] = vut[i + 1]
                    bc_vc = self.V[tuple(bc_vc)]
                    bc_vc.connect(vco)  # NOTE: Unneeded?
                    yield bc_vc

                    # Split to centre, call this centre group "d = 0.5*a"
                    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                    d_bc_vc.connect(bc_vc)
                    d_bc_vc.connect(vectors[1])  # Connect all to centroid
                    d_bc_vc.connect(vectors[2])  # Connect all to centroid
                    d_bc_vc.connect(vectors[3])  # Connect all to centroid
                    d_bc_vc.connect(vectors[4])  # Connect all to centroid
                    yield d_bc_vc.x
                    b_vl = self.V[tuple(b_vl)]
                    bc_vc.connect(b_vl)  # Connect aN cross pairs
                    d_bc_vc.connect(b_vl)  # Connect all to centroid
                    yield b_vl
                    b_vu = self.V[tuple(b_vu)]
                    bc_vc.connect(b_vu)  # Connect aN cross pairs
                    d_bc_vc.connect(b_vu)  # Connect all to centroid
                    yield b_vu
                    ba_vl = self.V[tuple(ba_vl)]
                    bc_vc.connect(ba_vl)  # Connect aN cross pairs
                    d_bc_vc.connect(ba_vl)  # Connect all to centroid
                    self.split_edge(b_vu.x, ba_vl.x)
                    yield ba_vl
                    ba_vu = self.V[tuple(ba_vu)]
                    bc_vc.connect(ba_vu)  # Connect aN cross pairs
                    d_bc_vc.connect(ba_vu)  # Connect all to centroid
                    # Split the a + b edge of the initial triangulation:
                    os_v = self.split_edge(vectors[1].x, ba_vu.x)  # o-s
                    ss_v = self.split_edge(b_vl.x, ba_vu.x)  # s-s
                    yield os_v.x  # often equal to vco, but not always
                    yield ss_v.x  # often equal to bc_vu, but not always
                    yield ba_vu
                    # Split remaining to centre, call this centre group
                    # "d = 0.5*a"
                    d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    yield d_bc_vc.x
                    d_b_vl = self.split_edge(vectors[1].x, b_vl.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_b_vl)  # Connect dN cross pairs
                    yield d_b_vl.x
                    d_b_vu = self.split_edge(vectors[2].x, b_vu.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_b_vu)  # Connect dN cross pairs
                    yield d_b_vu.x
                    d_ba_vl = self.split_edge(vectors[3].x, ba_vl.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_ba_vl)  # Connect dN cross pairs
                    yield d_ba_vl
                    d_ba_vu = self.split_edge(vectors[4].x, ba_vu.x)
                    d_bc_vc.connect(vco)  # NOTE: Unneeded?
                    d_bc_vc.connect(d_ba_vu)  # Connect dN cross pairs
                    yield d_ba_vu
                    c_vc, vl, vu, a_vl, a_vu = vectors

                    comb = [vl, vu, a_vl, a_vu,
                            b_vl, b_vu, ba_vl, ba_vu]
                    comb_iter = itertools.combinations(comb, 2)
                    for vecs in comb_iter:
                        self.split_edge(vecs[0].x, vecs[1].x)

                    # Add new list of cross pairs
                    ab_C.append((bc_vc, b_vl, b_vu, ba_vl, ba_vu))
                    ab_C.append((d_bc_vc, d_b_vl, d_b_vu, d_ba_vl, d_ba_vu))
                    ab_C.append((d_bc_vc, vectors[1], b_vl, a_vu, ba_vu))
                    ab_C.append((d_bc_vc, vu, b_vu, a_vl, ba_vl))

                for j, (VL, VC, VU) in enumerate(zip(cCox, cCcx, cCux)):
                    for k, (vl, vc, vu) in enumerate(zip(VL, VC, VU)):
                        # Build aN vertices for each lower-upper C3 group in N:
                        a_vl = list(vl.x)
                        a_vu = list(vu.x)
                        a_vl[i + 1] = vut[i + 1]
                        a_vu[i + 1] = vut[i + 1]
                        a_vl = self.V[tuple(a_vl)]
                        a_vu = self.V[tuple(a_vu)]
                        # Note, build (a + vc) later for consistent yields
                        # Split the a + b edge of the initial triangulation:
                        c_vc = self.split_edge(vl.x, a_vu.x)
                        self.split_edge(vl.x, vu.x)  # Equal to vc
                        # Build cN vertices for each lower-upper C3 group in N:
                        c_vc.connect(vco)
                        c_vc.connect(vc)
                        c_vc.connect(vl)  # Connect c + ac operations
                        c_vc.connect(vu)  # Connect c + ac operations
                        c_vc.connect(a_vl)  # Connect c + ac operations
                        c_vc.connect(a_vu)  # Connect c + ac operations
                        yield c_vc.x
                        c_vl = self.split_edge(vl.x, a_vl.x)
                        c_vl.connect(vco)
                        c_vc.connect(c_vl)  # Connect cN group vertices
                        yield c_vl.x
                        # yield at end of loop:
                        c_vu = self.split_edge(vu.x, a_vu.x)
                        c_vu.connect(vco)
                        # Connect remaining cN group vertices
                        c_vc.connect(c_vu)  # Connect cN group vertices
                        yield c_vu.x

                        a_vc = self.split_edge(a_vl.x, a_vu.x)  # is (a + vc) ?
                        a_vc.connect(vco)
                        a_vc.connect(c_vc)

                        # Storage for connecting c + ac operations:
                        ab_C.append((c_vc, vl, vu, a_vl, a_vu))

                        # Update the containers
                        Cox[i + 1].append(vl)
                        Cox[i + 1].append(vc)
                        Cox[i + 1].append(vu)
                        Ccx[i + 1].append(c_vl)
                        Ccx[i + 1].append(c_vc)
                        Ccx[i + 1].append(c_vu)
                        Cux[i + 1].append(a_vl)
                        Cux[i + 1].append(a_vc)
                        Cux[i + 1].append(a_vu)

                        # Update old containers
                        Cox[j].append(c_vl)  # !
                        Cox[j].append(a_vl)
                        Ccx[j].append(c_vc)  # !
                        Ccx[j].append(a_vc)  # !
                        Cux[j].append(c_vu)  # !
                        Cux[j].append(a_vu)

                        # Yield new points
                        yield a_vc.x

            except IndexError:
                for vectors in ab_Cc:
                    ba_vl = list(vectors[3].x)
                    ba_vu = list(vectors[4].x)
                    ba_vl[i + 1] = vut[i + 1]
                    ba_vu[i + 1] = vut[i + 1]
                    ba_vu = self.V[tuple(ba_vu)]
                    yield ba_vu
                    d_bc_vc = self.split_edge(vectors[1].x, ba_vu.x)  # o-s
                    yield ba_vu
                    d_bc_vc.connect(vectors[1])  # Connect all to centroid
                    d_bc_vc.connect(vectors[2])  # Connect all to centroid
                    d_bc_vc.connect(vectors[3])  # Connect all to centroid
                    d_bc_vc.connect(vectors[4])  # Connect all to centroid
                    yield d_bc_vc.x
                    ba_vl = self.V[tuple(ba_vl)]
                    yield ba_vl
                    d_ba_vl = self.split_edge(vectors[3].x, ba_vl.x)
                    d_ba_vu = self.split_edge(vectors[4].x, ba_vu.x)
                    d_ba_vc = self.split_edge(d_ba_vl.x, d_ba_vu.x)
                    yield d_ba_vl
                    yield d_ba_vu
                    yield d_ba_vc
                    c_vc, vl, vu, a_vl, a_vu = vectors
                    comb = [vl, vu, a_vl, a_vu,
                            ba_vl,
                            ba_vu]
                    comb_iter = itertools.combinations(comb, 2)
                    for vecs in comb_iter:
                        self.split_edge(vecs[0].x, vecs[1].x)

                # Copy lists for iteration
                cCox = Cox[i]
                cCcx = Ccx[i]
                cCux = Cux[i]
                VL, VC, VU = cCox, cCcx, cCux
                for k, (vl, vc, vu) in enumerate(zip(VL, VC, VU)):
                    # Build aN vertices for each lower-upper pair in N:
                    a_vu = list(vu.x)
                    a_vu[i + 1] = vut[i + 1]

                    # Connect vertices in N to corresponding vertices
                    # in aN:
                    a_vu = self.V[tuple(a_vu)]
                    yield a_vl.x
                    # Split the a + b edge of the initial triangulation:
                    c_vc = self.split_edge(vl.x, a_vu.x)
                    self.split_edge(vl.x, vu.x)  # Equal to vc
                    c_vc.connect(vco)
                    c_vc.connect(vc)
                    c_vc.connect(vl)  # Connect c + ac operations
                    c_vc.connect(vu)  # Connect c + ac operations
                    c_vc.connect(a_vu)  # Connect c + ac operations
                    yield (c_vc.x)
                    c_vu = self.split_edge(vu.x,
                                           a_vu.x)  # yield at end of loop
                    c_vu.connect(vco)
                    # Connect remaining cN group vertices
                    c_vc.connect(c_vu)  # Connect cN group vertices
                    yield (c_vu.x)

                    # Update the containers
                    Cox[i + 1].append(vu)
                    Ccx[i + 1].append(c_vu)
                    Cux[i + 1].append(a_vu)

                    # Update old containers
                    s_ab_C.append([c_vc, vl, vu, a_vu])

                    yield a_vu.x

        # Clean class trash
        try:
            del Cox
            del Ccx
            del Cux
            del ab_C
            del ab_Cc
        except UnboundLocalError:
            pass

        try:
            self.triangulated_vectors.remove((tuple(origin_c),
                                              tuple(supremum_c)))
        except ValueError:
            # Turn this into a logging warning?
            pass
        # Add newly triangulated vectors:
        for vs in sup_set:
            self.triangulated_vectors.append((tuple(vco.x), tuple(vs.x)))

        # Extra yield to ensure that the triangulation is completed
        if centroid:
            vcn_set = set()
            c_nn_lists = []
            for vs in sup_set:
                # Build centroid
                c_nn = self.vpool(vco.x, vs.x)
                try:
                    c_nn.remove(vcn_set)
                except KeyError:
                    pass
                c_nn_lists.append(c_nn)

            for c_nn in c_nn_lists:
                try:
                    c_nn.remove(vcn_set)
                except KeyError:
                    pass

            for vs, c_nn in zip(sup_set, c_nn_lists):
                # Build centroid
                vcn = self.split_edge(vco.x, vs.x)
                vcn_set.add(vcn)
                try:  # Shouldn't be needed?
                    c_nn.remove(vcn_set)
                except KeyError:
                    pass
                for vnn in c_nn:
                    vcn.connect(vnn)
                yield vcn.x
        else:
            pass

        yield vut
        return

    def refine_star(self, v):
        """Refine the star domain of a vertex `v`."""
        # Copy lists before iteration
        vnn = copy.copy(v.nn)
        v1nn = []
        d_v0v1_set = set()
        for v1 in vnn:
            v1nn.append(copy.copy(v1.nn))

        for v1, v1nn in zip(vnn, v1nn):
            vnnu = v1nn.intersection(vnn)

            d_v0v1 = self.split_edge(v.x, v1.x)
            for o_d_v0v1 in d_v0v1_set:
                d_v0v1.connect(o_d_v0v1)
            d_v0v1_set.add(d_v0v1)
            for v2 in vnnu:
                d_v1v2 = self.split_edge(v1.x, v2.x)
                d_v0v1.connect(d_v1v2)
        return

    @cache
    def split_edge(self, v1, v2):
        v1 = self.V[v1]
        v2 = self.V[v2]
        # Destroy original edge, if it exists:
        v1.disconnect(v2)
        # Compute vertex on centre of edge:
        try:
            vct = (v2.x_a - v1.x_a) / 2.0 + v1.x_a
        except TypeError:  # Allow for decimal operations
            vct = (v2.x_a - v1.x_a) / decimal.Decimal(2.0) + v1.x_a

        vc = self.V[tuple(vct)]
        # Connect to original 2 vertices to the new centre vertex
        vc.connect(v1)
        vc.connect(v2)
        return vc

    def vpool(self, origin, supremum):
        vot = tuple(origin)
        vst = tuple(supremum)
        # Initiate vertices in case they don't exist
        vo = self.V[vot]
        vs = self.V[vst]

        # Remove origin - supremum disconnect

        # Find the lower/upper bounds of the refinement hyperrectangle
        bl = list(vot)
        bu = list(vst)
        for i, (voi, vsi) in enumerate(zip(vot, vst)):
            if bl[i] > vsi:
                bl[i] = vsi
            if bu[i] < voi:
                bu[i] = voi

        #      NOTE: This is mostly done with sets/lists because we aren't sure
        #            how well the numpy arrays will scale to thousands of
        #             dimensions.
        vn_pool = set()
        vn_pool.update(vo.nn)
        vn_pool.update(vs.nn)
        cvn_pool = copy.copy(vn_pool)
        for vn in cvn_pool:
            for i, xi in enumerate(vn.x):
                if bl[i] <= xi <= bu[i]:
                    pass
                else:
                    try:
                        vn_pool.remove(vn)
                    except KeyError:
                        pass  # NOTE: Not all neighbours are in initial pool
        return vn_pool

    def vf_to_vv(self, vertices, simplices):
        """
        Convert a vertex-face mesh to a vertex-vertex mesh used by this class

        Parameters
        ----------
        vertices : list
            Vertices
        simplices : list
            Simplices
        """
        if self.dim > 1:
            for s in simplices:
                edges = itertools.combinations(s, self.dim)
                for e in edges:
                    self.V[tuple(vertices[e[0]])].connect(
                        self.V[tuple(vertices[e[1]])])
        else:
            for e in simplices:
                self.V[tuple(vertices[e[0]])].connect(
                    self.V[tuple(vertices[e[1]])])
        return

    def connect_vertex_non_symm(self, v_x, near=None):
        """
        Adds a vertex at coords v_x to the complex that is not symmetric to the
        initial triangulation and sub-triangulation.

        If near is specified (for example; a star domain or collections of
        cells known to contain v) then only those simplices containd in near
        will be searched, this greatly speeds up the process.

        If near is not specified this method will search the entire simplicial
        complex structure.

        Parameters
        ----------
        v_x : tuple
            Coordinates of non-symmetric vertex
        near : set or list
            List of vertices, these are points near v to check for
        """
        if near is None:
            star = self.V
        else:
            star = near
        # Create the vertex origin
        if tuple(v_x) in self.V.cache:
            if self.V[v_x] in self.V_non_symm:
                pass
            else:
                return

        self.V[v_x]
        found_nn = False
        S_rows = []
        for v in star:
            S_rows.append(v.x)

        S_rows = np.array(S_rows)
        A = np.array(S_rows) - np.array(v_x)
        # Iterate through all the possible simplices of S_rows
        for s_i in itertools.combinations(range(S_rows.shape[0]),
                                          r=self.dim + 1):
            # Check if connected, else s_i is not a simplex
            valid_simplex = True
            for i in itertools.combinations(s_i, r=2):
                # Every combination of vertices must be connected, we check of
                # the current iteration of all combinations of s_i are
                # connected we break the loop if it is not.
                if ((self.V[tuple(S_rows[i[1]])] not in
                        self.V[tuple(S_rows[i[0]])].nn)
                    and (self.V[tuple(S_rows[i[0]])] not in
                         self.V[tuple(S_rows[i[1]])].nn)):
                    valid_simplex = False
                    break

            S = S_rows[tuple([s_i])]
            if valid_simplex:
                if self.deg_simplex(S, proj=None):
                    valid_simplex = False

            # If s_i is a valid simplex we can test if v_x is inside si
            if valid_simplex:
                # Find the A_j0 value from the precalculated values
                A_j0 = A[tuple([s_i])]
                if self.in_simplex(S, v_x, A_j0):
                    found_nn = True
                    # breaks the main for loop, s_i is the target simplex:
                    break

        # Connect the simplex to point
        if found_nn:
            for i in s_i:
                self.V[v_x].connect(self.V[tuple(S_rows[i])])
        # Attached the simplex to storage for all non-symmetric vertices
        self.V_non_symm.append(self.V[v_x])
        # this bool value indicates a successful connection if True:
        return found_nn

    def in_simplex(self, S, v_x, A_j0=None):
        """Check if a vector v_x is in simplex `S`.

        Parameters
        ----------
        S : array_like
            Array containing simplex entries of vertices as rows
        v_x :
            A candidate vertex
        A_j0 : array, optional,
            Allows for A_j0 to be pre-calculated

        Returns
        -------
        res : boolean
            True if `v_x` is in `S`
        """
        A_11 = np.delete(S, 0, 0) - S[0]

        sign_det_A_11 = np.sign(np.linalg.det(A_11))
        if sign_det_A_11 == 0:
            # NOTE: We keep the variable A_11, but we loop through A_jj
            # ind=
            # while sign_det_A_11 == 0:
            #    A_11 = np.delete(S, ind, 0) - S[ind]
            #    sign_det_A_11 = np.sign(np.linalg.det(A_11))

            sign_det_A_11 = -1  # TODO: Choose another det of j instead?
            # TODO: Unlikely to work in many cases

        if A_j0 is None:
            A_j0 = S - v_x

        for d in range(self.dim + 1):
            det_A_jj = (-1)**d * sign_det_A_11
            # TODO: Note that scipy might be faster to add as an optional
            #       dependency
            sign_det_A_j0 = np.sign(np.linalg.det(np.delete(A_j0, d,
                                                                     0)))
            # TODO: Note if sign_det_A_j0 == then the point is coplanar to the
            #       current simplex facet, so perhaps return True and attach?
            if det_A_jj == sign_det_A_j0:
                continue
            else:
                return False

        return True

    def deg_simplex(self, S, proj=None):
        """Test a simplex S for degeneracy (linear dependence in R^dim).

        Parameters
        ----------
        S : np.array
            Simplex with rows as vertex vectors
        proj : array, optional,
            If the projection S[1:] - S[0] is already
            computed it can be added as an optional argument.
        """
        # Strategy: we test all combination of faces, if any of the
        # determinants are zero then the vectors lie on the same face and is
        # therefore linearly dependent in the space of R^dim
        if proj is None:
            proj = S[1:] - S[0]

        # TODO: Is checking the projection of one vertex against faces of other
        #       vertices sufficient? Or do we need to check more vertices in
        #       dimensions higher than 2?
        # TODO: Literature seems to suggest using proj.T, but why is this
        #       needed?
        if np.linalg.det(proj) == 0.0:  # TODO: Replace with tolerance?
            return True  # Simplex is degenerate
        else:
            return False  # Simplex is not degenerate
<|endoftext|>"""Progress report printers."""

class ReportBase:
    COLUMN_NAMES: list[str] = NotImplemented
    COLUMN_WIDTHS: list[int] = NotImplemented
    ITERATION_FORMATS: list[str] = NotImplemented

    @classmethod
    def print_header(cls):
        fmt = ("|"
               + "|".join([f"{{:^{x}}}" for x in cls.COLUMN_WIDTHS])
               + "|")
        separators = ['-' * x for x in cls.COLUMN_WIDTHS]
        print(fmt.format(*cls.COLUMN_NAMES))
        print(fmt.format(*separators))

    @classmethod
    def print_iteration(cls, *args):
        iteration_format = [f"{{:{x}}}" for x in cls.ITERATION_FORMATS]
        fmt = "|" + "|".join(iteration_format) + "|"
        print(fmt.format(*args))

    @classmethod
    def print_footer(cls):
        print()


class BasicReport(ReportBase):
    COLUMN_NAMES = ["niter", "f evals", "CG iter", "obj func", "tr radius",
                    "opt", "c viol"]
    COLUMN_WIDTHS = [7, 7, 7, 13, 10, 10, 10]
    ITERATION_FORMATS = ["^7", "^7", "^7", "^+13.4e",
                         "^10.2e", "^10.2e", "^10.2e"]


class SQPReport(ReportBase):
    COLUMN_NAMES = ["niter", "f evals", "CG iter", "obj func", "tr radius",
                    "opt", "c viol", "penalty", "CG stop"]
    COLUMN_WIDTHS = [7, 7, 7, 13, 10, 10, 10, 10, 7]
    ITERATION_FORMATS = ["^7", "^7", "^7", "^+13.4e", "^10.2e", "^10.2e",
                         "^10.2e", "^10.2e", "^7"]


class IPReport(ReportBase):
    COLUMN_NAMES = ["niter", "f evals", "CG iter", "obj func", "tr radius",
                    "opt", "c viol", "penalty", "barrier param", "CG stop"]
    COLUMN_WIDTHS = [7, 7, 7, 13, 10, 10, 10, 10, 13, 7]
    ITERATION_FORMATS = ["^7", "^7", "^7", "^+13.4e", "^10.2e", "^10.2e",
                         "^10.2e", "^10.2e", "^13.2e", "^7"]
<|endoftext|>import numpy as np
import scipy.sparse as sps


class CanonicalConstraint:
    """Canonical constraint to use with trust-constr algorithm.

    It represents the set of constraints of the form::

        f_eq(x) = 0
        f_ineq(x) <= 0

    where ``f_eq`` and ``f_ineq`` are evaluated by a single function, see
    below.

    The class is supposed to be instantiated by factory methods, which
    should prepare the parameters listed below.

    Parameters
    ----------
    n_eq, n_ineq : int
        Number of equality and inequality constraints respectively.
    fun : callable
        Function defining the constraints. The signature is
        ``fun(x) -> c_eq, c_ineq``, where ``c_eq`` is ndarray with `n_eq`
        components and ``c_ineq`` is ndarray with `n_ineq` components.
    jac : callable
        Function to evaluate the Jacobian of the constraint. The signature
        is ``jac(x) -> J_eq, J_ineq``, where ``J_eq`` and ``J_ineq`` are
        either ndarray of csr_matrix of shapes (n_eq, n) and (n_ineq, n),
        respectively.
    hess : callable
        Function to evaluate the Hessian of the constraints multiplied
        by Lagrange multipliers, that is
        ``dot(f_eq, v_eq) + dot(f_ineq, v_ineq)``. The signature is
        ``hess(x, v_eq, v_ineq) -> H``, where ``H`` has an implied
        shape (n, n) and provide a matrix-vector product operation
        ``H.dot(p)``.
    keep_feasible : ndarray, shape (n_ineq,)
        Mask indicating which inequality constraints should be kept feasible.
    """
    def __init__(self, n_eq, n_ineq, fun, jac, hess, keep_feasible):
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.keep_feasible = keep_feasible

    @classmethod
    def from_PreparedConstraint(cls, constraint):
        """Create an instance from `PreparedConstrained` object."""
        lb, ub = constraint.bounds
        cfun = constraint.fun
        keep_feasible = constraint.keep_feasible

        if np.all(lb == -np.inf) and np.all(ub == np.inf):
            return cls.empty(cfun.n)

        if np.all(lb == -np.inf) and np.all(ub == np.inf):
            return cls.empty(cfun.n)
        elif np.all(lb == ub):
            return cls._equal_to_canonical(cfun, lb)
        elif np.all(lb == -np.inf):
            return cls._less_to_canonical(cfun, ub, keep_feasible)
        elif np.all(ub == np.inf):
            return cls._greater_to_canonical(cfun, lb, keep_feasible)
        else:
            return cls._interval_to_canonical(cfun, lb, ub, keep_feasible)

    @classmethod
    def empty(cls, n):
        """Create an "empty" instance.

        This "empty" instance is required to allow working with unconstrained
        problems as if they have some constraints.
        """
        empty_fun = np.empty(0)
        empty_jac = np.empty((0, n))
        empty_hess = sps.csr_matrix((n, n))

        def fun(x):
            return empty_fun, empty_fun

        def jac(x):
            return empty_jac, empty_jac

        def hess(x, v_eq, v_ineq):
            return empty_hess

        return cls(0, 0, fun, jac, hess, np.empty(0, dtype=np.bool_))

    @classmethod
    def concatenate(cls, canonical_constraints, sparse_jacobian):
        """Concatenate multiple `CanonicalConstraint` into one.

        `sparse_jacobian` (bool) determines the Jacobian format of the
        concatenated constraint. Note that items in `canonical_constraints`
        must have their Jacobians in the same format.
        """
        def fun(x):
            if canonical_constraints:
                eq_all, ineq_all = zip(
                        *[c.fun(x) for c in canonical_constraints])
            else:
                eq_all, ineq_all = [], []

            return np.hstack(eq_all), np.hstack(ineq_all)

        if sparse_jacobian:
            vstack = sps.vstack
        else:
            vstack = np.vstack

        def jac(x):
            if canonical_constraints:
                eq_all, ineq_all = zip(
                        *[c.jac(x) for c in canonical_constraints])
            else:
                eq_all, ineq_all = [], []

            return vstack(eq_all), vstack(ineq_all)

        def hess(x, v_eq, v_ineq):
            hess_all = []
            index_eq = 0
            index_ineq = 0
            for c in canonical_constraints:
                vc_eq = v_eq[index_eq:index_eq + c.n_eq]
                vc_ineq = v_ineq[index_ineq:index_ineq + c.n_ineq]
                hess_all.append(c.hess(x, vc_eq, vc_ineq))
                index_eq += c.n_eq
                index_ineq += c.n_ineq

            def matvec(p):
                result = np.zeros_like(p, dtype=float)
                for h in hess_all:
                    result += h.dot(p)
                return result

            n = x.shape[0]
            return sps.linalg.LinearOperator((n, n), matvec, dtype=float)

        n_eq = sum(c.n_eq for c in canonical_constraints)
        n_ineq = sum(c.n_ineq for c in canonical_constraints)
        keep_feasible = np.hstack([c.keep_feasible for c in
                                   canonical_constraints])

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _equal_to_canonical(cls, cfun, value):
        empty_fun = np.empty(0)
        n = cfun.n

        n_eq = value.shape[0]
        n_ineq = 0
        keep_feasible = np.empty(0, dtype=bool)

        if cfun.sparse_jacobian:
            empty_jac = sps.csr_matrix((0, n))
        else:
            empty_jac = np.empty((0, n))

        def fun(x):
            return cfun.fun(x) - value, empty_fun

        def jac(x):
            return cfun.jac(x), empty_jac

        def hess(x, v_eq, v_ineq):
            return cfun.hess(x, v_eq)

        empty_fun = np.empty(0)
        n = cfun.n
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_matrix((0, n))
        else:
            empty_jac = np.empty((0, n))

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _less_to_canonical(cls, cfun, ub, keep_feasible):
        empty_fun = np.empty(0)
        n = cfun.n
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_matrix((0, n))
        else:
            empty_jac = np.empty((0, n))

        finite_ub = ub < np.inf
        n_eq = 0
        n_ineq = np.sum(finite_ub)

        if np.all(finite_ub):
            def fun(x):
                return empty_fun, cfun.fun(x) - ub

            def jac(x):
                return empty_jac, cfun.jac(x)

            def hess(x, v_eq, v_ineq):
                return cfun.hess(x, v_ineq)
        else:
            finite_ub = np.nonzero(finite_ub)[0]
            keep_feasible = keep_feasible[finite_ub]
            ub = ub[finite_ub]

            def fun(x):
                return empty_fun, cfun.fun(x)[finite_ub] - ub

            def jac(x):
                return empty_jac, cfun.jac(x)[finite_ub]

            def hess(x, v_eq, v_ineq):
                v = np.zeros(cfun.m)
                v[finite_ub] = v_ineq
                return cfun.hess(x, v)

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _greater_to_canonical(cls, cfun, lb, keep_feasible):
        empty_fun = np.empty(0)
        n = cfun.n
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_matrix((0, n))
        else:
            empty_jac = np.empty((0, n))

        finite_lb = lb > -np.inf
        n_eq = 0
        n_ineq = np.sum(finite_lb)

        if np.all(finite_lb):
            def fun(x):
                return empty_fun, lb - cfun.fun(x)

            def jac(x):
                return empty_jac, -cfun.jac(x)

            def hess(x, v_eq, v_ineq):
                return cfun.hess(x, -v_ineq)
        else:
            finite_lb = np.nonzero(finite_lb)[0]
            keep_feasible = keep_feasible[finite_lb]
            lb = lb[finite_lb]

            def fun(x):
                return empty_fun, lb - cfun.fun(x)[finite_lb]

            def jac(x):
                return empty_jac, -cfun.jac(x)[finite_lb]

            def hess(x, v_eq, v_ineq):
                v = np.zeros(cfun.m)
                v[finite_lb] = -v_ineq
                return cfun.hess(x, v)

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _interval_to_canonical(cls, cfun, lb, ub, keep_feasible):
        lb_inf = lb == -np.inf
        ub_inf = ub == np.inf
        equal = lb == ub
        less = lb_inf & ~ub_inf
        greater = ub_inf & ~lb_inf
        interval = ~equal & ~lb_inf & ~ub_inf

        equal = np.nonzero(equal)[0]
        less = np.nonzero(less)[0]
        greater = np.nonzero(greater)[0]
        interval = np.nonzero(interval)[0]
        n_less = less.shape[0]
        n_greater = greater.shape[0]
        n_interval = interval.shape[0]
        n_ineq = n_less + n_greater + 2 * n_interval
        n_eq = equal.shape[0]

        keep_feasible = np.hstack((keep_feasible[less],
                                   keep_feasible[greater],
                                   keep_feasible[interval],
                                   keep_feasible[interval]))

        def fun(x):
            f = cfun.fun(x)
            eq = f[equal] - lb[equal]
            le = f[less] - ub[less]
            ge = lb[greater] - f[greater]
            il = f[interval] - ub[interval]
            ig = lb[interval] - f[interval]
            return eq, np.hstack((le, ge, il, ig))

        def jac(x):
            J = cfun.jac(x)
            eq = J[equal]
            le = J[less]
            ge = -J[greater]
            il = J[interval]
            ig = -il
            if sps.issparse(J):
                ineq = sps.vstack((le, ge, il, ig))
            else:
                ineq = np.vstack((le, ge, il, ig))
            return eq, ineq

        def hess(x, v_eq, v_ineq):
            n_start = 0
            v_l = v_ineq[n_start:n_start + n_less]
            n_start += n_less
            v_g = v_ineq[n_start:n_start + n_greater]
            n_start += n_greater
            v_il = v_ineq[n_start:n_start + n_interval]
            n_start += n_interval
            v_ig = v_ineq[n_start:n_start + n_interval]

            v = np.zeros_like(lb)
            v[equal] = v_eq
            v[less] = v_l
            v[greater] = -v_g
            v[interval] = v_il - v_ig

            return cfun.hess(x, v)

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)


def initial_constraints_as_canonical(n, prepared_constraints, sparse_jacobian):
    """Convert initial values of the constraints to the canonical format.

    The purpose to avoid one additional call to the constraints at the initial
    point. It takes saved values in `PreparedConstraint`, modifies and
    concatenates them to the canonical constraint format.
    """
    c_eq = []
    c_ineq = []
    J_eq = []
    J_ineq = []

    for c in prepared_constraints:
        f = c.fun.f
        J = c.fun.J
        lb, ub = c.bounds
        if np.all(lb == ub):
            c_eq.append(f - lb)
            J_eq.append(J)
        elif np.all(lb == -np.inf):
            finite_ub = ub < np.inf
            c_ineq.append(f[finite_ub] - ub[finite_ub])
            J_ineq.append(J[finite_ub])
        elif np.all(ub == np.inf):
            finite_lb = lb > -np.inf
            c_ineq.append(lb[finite_lb] - f[finite_lb])
            J_ineq.append(-J[finite_lb])
        else:
            lb_inf = lb == -np.inf
            ub_inf = ub == np.inf
            equal = lb == ub
            less = lb_inf & ~ub_inf
            greater = ub_inf & ~lb_inf
            interval = ~equal & ~lb_inf & ~ub_inf

            c_eq.append(f[equal] - lb[equal])
            c_ineq.append(f[less] - ub[less])
            c_ineq.append(lb[greater] - f[greater])
            c_ineq.append(f[interval] - ub[interval])
            c_ineq.append(lb[interval] - f[interval])

            J_eq.append(J[equal])
            J_ineq.append(J[less])
            J_ineq.append(-J[greater])
            J_ineq.append(J[interval])
            J_ineq.append(-J[interval])

    c_eq = np.hstack(c_eq) if c_eq else np.empty(0)
    c_ineq = np.hstack(c_ineq) if c_ineq else np.empty(0)

    if sparse_jacobian:
        vstack = sps.vstack
        empty = sps.csr_matrix((0, n))
    else:
        vstack = np.vstack
        empty = np.empty((0, n))

    J_eq = vstack(J_eq) if J_eq else empty
    J_ineq = vstack(J_ineq) if J_ineq else empty

    return c_eq, c_ineq, J_eq, J_ineq
<|endoftext|>"""Byrd-Omojokun Trust-Region SQP method."""

from scipy.sparse import eye as speye
from .projections import projections
from .qp_subproblem import modified_dogleg, projected_cg, box_intersections
import numpy as np
from numpy.linalg import norm

__all__ = ['equality_constrained_sqp']


def default_scaling(x):
    n, = np.shape(x)
    return speye(n)


def equality_constrained_sqp(fun_and_constr, grad_and_jac, lagr_hess,
                             x0, fun0, grad0, constr0,
                             jac0, stop_criteria,
                             state,
                             initial_penalty,
                             initial_trust_radius,
                             factorization_method,
                             trust_lb=None,
                             trust_ub=None,
                             scaling=default_scaling):
    """Solve nonlinear equality-constrained problem using trust-region SQP.

    Solve optimization problem:

        minimize fun(x)
        subject to: constr(x) = 0

    using Byrd-Omojokun Trust-Region SQP method described in [1]_. Several
    implementation details are based on [2]_ and [3]_, p. 549.

    References
    ----------
    .. [1] Lalee, Marucha, Jorge Nocedal, and Todd Plantenga. "On the
           implementation of an algorithm for large-scale equality
           constrained optimization." SIAM Journal on
           Optimization 8.3 (1998): 682-706.
    .. [2] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    .. [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
           Second Edition (2006).
    """
    PENALTY_FACTOR = 0.3  # Rho from formula (3.51), reference [2]_, p.891.
    LARGE_REDUCTION_RATIO = 0.9
    INTERMEDIARY_REDUCTION_RATIO = 0.3
    SUFFICIENT_REDUCTION_RATIO = 1e-8  # Eta from reference [2]_, p.892.
    TRUST_ENLARGEMENT_FACTOR_L = 7.0
    TRUST_ENLARGEMENT_FACTOR_S = 2.0
    MAX_TRUST_REDUCTION = 0.5
    MIN_TRUST_REDUCTION = 0.1
    SOC_THRESHOLD = 0.1
    TR_FACTOR = 0.8  # Zeta from formula (3.21), reference [2]_, p.885.
    BOX_FACTOR = 0.5

    n, = np.shape(x0)  # Number of parameters

    # Set default lower and upper bounds.
    if trust_lb is None:
        trust_lb = np.full(n, -np.inf)
    if trust_ub is None:
        trust_ub = np.full(n, np.inf)

    # Initial values
    x = np.copy(x0)
    trust_radius = initial_trust_radius
    penalty = initial_penalty
    # Compute Values
    f = fun0
    c = grad0
    b = constr0
    A = jac0
    S = scaling(x)
    # Get projections
    try:
        Z, LS, Y = projections(A, factorization_method)
    except ValueError as e:
        if str(e) == "expected square matrix":
            # can be the case if there are more equality
            # constraints than independent variables
            raise ValueError(
                "The 'expected square matrix' error can occur if there are"
                " more equality constraints than independent variables."
                " Consider how your constraints are set up, or use"
                " factorization_method='SVDFactorization'."
            ) from e
        else:
            raise e

    # Compute least-square lagrange multipliers
    v = -LS.dot(c)
    # Compute Hessian
    H = lagr_hess(x, v)

    # Update state parameters
    optimality = norm(c + A.T.dot(v), np.inf)
    constr_violation = norm(b, np.inf) if len(b) > 0 else 0
    cg_info = {'niter': 0, 'stop_cond': 0,
               'hits_boundary': False}

    last_iteration_failed = False
    while not stop_criteria(state, x, last_iteration_failed,
                            optimality, constr_violation,
                            trust_radius, penalty, cg_info):
        # Normal Step - `dn`
        # minimize 1/2*||A dn + b||^2
        # subject to:
        # ||dn|| <= TR_FACTOR * trust_radius
        # BOX_FACTOR * lb <= dn <= BOX_FACTOR * ub.
        dn = modified_dogleg(A, Y, b,
                             TR_FACTOR*trust_radius,
                             BOX_FACTOR*trust_lb,
                             BOX_FACTOR*trust_ub)

        # Tangential Step - `dt`
        # Solve the QP problem:
        # minimize 1/2 dt.T H dt + dt.T (H dn + c)
        # subject to:
        # A dt = 0
        # ||dt|| <= sqrt(trust_radius**2 - ||dn||**2)
        # lb - dn <= dt <= ub - dn
        c_t = H.dot(dn) + c
        b_t = np.zeros_like(b)
        trust_radius_t = np.sqrt(trust_radius**2 - np.linalg.norm(dn)**2)
        lb_t = trust_lb - dn
        ub_t = trust_ub - dn
        dt, cg_info = projected_cg(H, c_t, Z, Y, b_t,
                                   trust_radius_t,
                                   lb_t, ub_t)

        # Compute update (normal + tangential steps).
        d = dn + dt

        # Compute second order model: 1/2 d H d + c.T d + f.
        quadratic_model = 1/2*(H.dot(d)).dot(d) + c.T.dot(d)
        # Compute linearized constraint: l = A d + b.
        linearized_constr = A.dot(d)+b
        # Compute new penalty parameter according to formula (3.52),
        # reference [2]_, p.891.
        vpred = norm(b) - norm(linearized_constr)
        # Guarantee `vpred` always positive,
        # regardless of roundoff errors.
        vpred = max(1e-16, vpred)
        previous_penalty = penalty
        if quadratic_model > 0:
            new_penalty = quadratic_model / ((1-PENALTY_FACTOR)*vpred)
            penalty = max(penalty, new_penalty)
        # Compute predicted reduction according to formula (3.52),
        # reference [2]_, p.891.
        predicted_reduction = -quadratic_model + penalty*vpred

        # Compute merit function at current point
        merit_function = f + penalty*norm(b)
        # Evaluate function and constraints at trial point
        x_next = x + S.dot(d)
        f_next, b_next = fun_and_constr(x_next)
        # Compute merit function at trial point
        merit_function_next = f_next + penalty*norm(b_next)
        # Compute actual reduction according to formula (3.54),
        # reference [2]_, p.892.
        actual_reduction = merit_function - merit_function_next
        # Compute reduction ratio
        reduction_ratio = actual_reduction / predicted_reduction

        # Second order correction (SOC), reference [2]_, p.892.
        if reduction_ratio < SUFFICIENT_REDUCTION_RATIO and \
           norm(dn) <= SOC_THRESHOLD * norm(dt):
            # Compute second order correction
            y = -Y.dot(b_next)
            # Make sure increment is inside box constraints
            _, t, intersect = box_intersections(d, y, trust_lb, trust_ub)
            # Compute tentative point
            x_soc = x + S.dot(d + t*y)
            f_soc, b_soc = fun_and_constr(x_soc)
            # Recompute actual reduction
            merit_function_soc = f_soc + penalty*norm(b_soc)
            actual_reduction_soc = merit_function - merit_function_soc
            # Recompute reduction ratio
            reduction_ratio_soc = actual_reduction_soc / predicted_reduction
            if intersect and reduction_ratio_soc >= SUFFICIENT_REDUCTION_RATIO:
                x_next = x_soc
                f_next = f_soc
                b_next = b_soc
                reduction_ratio = reduction_ratio_soc

        # Readjust trust region step, formula (3.55), reference [2]_, p.892.
        if reduction_ratio >= LARGE_REDUCTION_RATIO:
            trust_radius = max(TRUST_ENLARGEMENT_FACTOR_L * norm(d),
                               trust_radius)
        elif reduction_ratio >= INTERMEDIARY_REDUCTION_RATIO:
            trust_radius = max(TRUST_ENLARGEMENT_FACTOR_S * norm(d),
                               trust_radius)
        # Reduce trust region step, according to reference [3]_, p.696.
        elif reduction_ratio < SUFFICIENT_REDUCTION_RATIO:
            trust_reduction = ((1-SUFFICIENT_REDUCTION_RATIO) /
                               (1-reduction_ratio))
            new_trust_radius = trust_reduction * norm(d)
            if new_trust_radius >= MAX_TRUST_REDUCTION * trust_radius:
                trust_radius *= MAX_TRUST_REDUCTION
            elif new_trust_radius >= MIN_TRUST_REDUCTION * trust_radius:
                trust_radius = new_trust_radius
            else:
                trust_radius *= MIN_TRUST_REDUCTION

        # Update iteration
        if reduction_ratio >= SUFFICIENT_REDUCTION_RATIO:
            x = x_next
            f, b = f_next, b_next
            c, A = grad_and_jac(x)
            S = scaling(x)
            # Get projections
            Z, LS, Y = projections(A, factorization_method)
            # Compute least-square lagrange multipliers
            v = -LS.dot(c)
            # Compute Hessian
            H = lagr_hess(x, v)
            # Set Flag
            last_iteration_failed = False
            # Optimality values
            optimality = norm(c + A.T.dot(v), np.inf)
            constr_violation = norm(b, np.inf) if len(b) > 0 else 0
        else:
            penalty = previous_penalty
            last_iteration_failed = True

    return x, state
<|endoftext|>"""Equality-constrained quadratic programming solvers."""

from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm

__all__ = [
    'eqp_kktfact',
    'sphere_intersections',
    'box_intersections',
    'box_sphere_intersections',
    'inside_box_boundaries',
    'modified_dogleg',
    'projected_cg'
]


# For comparison with the projected CG
def eqp_kktfact(H, c, A, b):
    """Solve equality-constrained quadratic programming (EQP) problem.

    Solve ``min 1/2 x.T H x + x.t c`` subject to ``A x + b = 0``
    using direct factorization of the KKT system.

    Parameters
    ----------
    H : sparse matrix, shape (n, n)
        Hessian matrix of the EQP problem.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    A : sparse matrix
        Jacobian matrix of the EQP problem.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the KKT problem.
    lagrange_multipliers : ndarray, shape (m,)
        Lagrange multipliers of the KKT problem.
    """
    n, = np.shape(c)  # Number of parameters
    m, = np.shape(b)  # Number of constraints

    # Karush-Kuhn-Tucker matrix of coefficients.
    # Defined as in Nocedal/Wright "Numerical
    # Optimization" p.452 in Eq. (16.4).
    kkt_matrix = csc_matrix(bmat([[H, A.T], [A, None]]))
    # Vector of coefficients.
    kkt_vec = np.hstack([-c, -b])

    # TODO: Use a symmetric indefinite factorization
    #       to solve the system twice as fast (because
    #       of the symmetry).
    lu = linalg.splu(kkt_matrix)
    kkt_sol = lu.solve(kkt_vec)
    x = kkt_sol[:n]
    lagrange_multipliers = -kkt_sol[n:n+m]

    return x, lagrange_multipliers


def sphere_intersections(z, d, trust_radius,
                         entire_line=False):
    """Find the intersection between segment (or line) and spherical constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the ball
    ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    trust_radius : float
        Ball radius.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the ball
        ``||x|| <= trust_radius``. When ``False``, the function returns the intersection
        between the segment ``x(t) = z + t*d``, ``0 <= t <= 1``, and the ball.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the ball for
        for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line/segment
        and the sphere. On the other hand, when ``False``, there is no
        intersection.
    """
    # Special case when d=0
    if norm(d) == 0:
        return 0, 0, False
    # Check for inf trust_radius
    if np.isinf(trust_radius):
        if entire_line:
            ta = -np.inf
            tb = np.inf
        else:
            ta = 0
            tb = 1
        intersect = True
        return ta, tb, intersect

    a = np.dot(d, d)
    b = 2 * np.dot(z, d)
    c = np.dot(z, z) - trust_radius**2
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        intersect = False
        return 0, 0, intersect
    sqrt_discriminant = np.sqrt(discriminant)

    # The following calculation is mathematically
    # equivalent to:
    # ta = (-b - sqrt_discriminant) / (2*a)
    # tb = (-b + sqrt_discriminant) / (2*a)
    # but produce smaller round off errors.
    # Look at Matrix Computation p.97
    # for a better justification.
    aux = b + copysign(sqrt_discriminant, b)
    ta = -aux / (2*a)
    tb = -2*c / aux
    ta, tb = sorted([ta, tb])

    if entire_line:
        intersect = True
    else:
        # Checks to see if intersection happens
        # within vectors length.
        if tb < 0 or ta > 1:
            intersect = False
            ta = 0
            tb = 0
        else:
            intersect = True
            # Restrict intersection interval
            # between 0 and 1.
            ta = max(0, ta)
            tb = min(1, tb)

    return ta, tb, intersect


def box_intersections(z, d, lb, ub,
                      entire_line=False):
    """Find the intersection between segment (or line) and box constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d`` and the rectangular box
    ``lb <= x <= ub``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the rectangular
        box. When ``False``, the function returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1``, and the rectangular box.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the box for
        for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line (or segment)
        and the rectangular box. On the other hand, when ``False``, there is no
        intersection.
    """
    # Make sure it is a numpy array
    z = np.asarray(z)
    d = np.asarray(d)
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    # Special case when d=0
    if norm(d) == 0:
        return 0, 0, False

    # Get values for which d==0
    zero_d = (d == 0)
    # If the boundaries are not satisfied for some coordinate
    # for which "d" is zero, there is no box-line intersection.
    if (z[zero_d] < lb[zero_d]).any() or (z[zero_d] > ub[zero_d]).any():
        intersect = False
        return 0, 0, intersect
    # Remove values for which d is zero
    not_zero_d = np.logical_not(zero_d)
    z = z[not_zero_d]
    d = d[not_zero_d]
    lb = lb[not_zero_d]
    ub = ub[not_zero_d]

    # Find a series of intervals (t_lb[i], t_ub[i]).
    t_lb = (lb-z) / d
    t_ub = (ub-z) / d
    # Get the intersection of all those intervals.
    ta = max(np.minimum(t_lb, t_ub))
    tb = min(np.maximum(t_lb, t_ub))

    # Check if intersection is feasible
    if ta <= tb:
        intersect = True
    else:
        intersect = False
    # Checks to see if intersection happens within vectors length.
    if not entire_line:
        if tb < 0 or ta > 1:
            intersect = False
            ta = 0
            tb = 0
        else:
            # Restrict intersection interval between 0 and 1.
            ta = max(0, ta)
            tb = min(1, tb)

    return ta, tb, intersect


def box_sphere_intersections(z, d, lb, ub, trust_radius,
                             entire_line=False,
                             extra_info=False):
    """Find the intersection between segment (or line) and box/sphere constraints.

    Find the intersection between the segment (or line) defined by the
    parametric  equation ``x(t) = z + t*d``, the rectangular box
    ``lb <= x <= ub`` and the ball ``||x|| <= trust_radius``.

    Parameters
    ----------
    z : array_like, shape (n,)
        Initial point.
    d : array_like, shape (n,)
        Direction.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``. Used
        to delimit the rectangular box.
    trust_radius : float
        Ball radius.
    entire_line : bool, optional
        When ``True``, the function returns the intersection between the line
        ``x(t) = z + t*d`` (``t`` can assume any value) and the constraints.
        When ``False``, the function returns the intersection between the segment
        ``x(t) = z + t*d``, ``0 <= t <= 1`` and the constraints.
    extra_info : bool, optional
        When ``True``, the function returns ``intersect_sphere`` and ``intersect_box``.

    Returns
    -------
    ta, tb : float
        The line/segment ``x(t) = z + t*d`` is inside the rectangular box and
        inside the ball for ``ta <= t <= tb``.
    intersect : bool
        When ``True``, there is a intersection between the line (or segment)
        and both constraints. On the other hand, when ``False``, there is no
        intersection.
    sphere_info : dict, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercepts the ball. And a boolean value indicating
        whether the sphere is intersected by the line.
    box_info : dict, optional
        Dictionary ``{ta, tb, intersect}`` containing the interval ``[ta, tb]``
        for which the line intercepts the box. And a boolean value indicating
        whether the box is intersected by the line.
    """
    ta_b, tb_b, intersect_b = box_intersections(z, d, lb, ub,
                                                entire_line)
    ta_s, tb_s, intersect_s = sphere_intersections(z, d,
                                                   trust_radius,
                                                   entire_line)
    ta = np.maximum(ta_b, ta_s)
    tb = np.minimum(tb_b, tb_s)
    if intersect_b and intersect_s and ta <= tb:
        intersect = True
    else:
        intersect = False

    if extra_info:
        sphere_info = {'ta': ta_s, 'tb': tb_s, 'intersect': intersect_s}
        box_info = {'ta': ta_b, 'tb': tb_b, 'intersect': intersect_b}
        return ta, tb, intersect, sphere_info, box_info
    else:
        return ta, tb, intersect


def inside_box_boundaries(x, lb, ub):
    """Check if lb <= x <= ub."""
    return (lb <= x).all() and (x <= ub).all()


def reinforce_box_boundaries(x, lb, ub):
    """Return clipped value of x"""
    return np.minimum(np.maximum(x, lb), ub)


def modified_dogleg(A, Y, b, trust_radius, lb, ub):
    """Approximately  minimize ``1/2*|| A x + b ||^2`` inside trust-region.

    Approximately solve the problem of minimizing ``1/2*|| A x + b ||^2``
    subject to ``||x|| < Delta`` and ``lb <= x <= ub`` using a modification
    of the classical dogleg approach.

    Parameters
    ----------
    A : LinearOperator (or sparse matrix or ndarray), shape (m, n)
        Matrix ``A`` in the minimization problem. It should have
        dimension ``(m, n)`` such that ``m < n``.
    Y : LinearOperator (or sparse matrix or ndarray), shape (n, m)
        LinearOperator that apply the projection matrix
        ``Q = A.T inv(A A.T)`` to the vector. The obtained vector
        ``y = Q x`` being the minimum norm solution of ``A y = x``.
    b : array_like, shape (m,)
        Vector ``b``in the minimization problem.
    trust_radius: float
        Trust radius to be considered. Delimits a sphere boundary
        to the problem.
    lb : array_like, shape (n,)
        Lower bounds to each one of the components of ``x``.
        It is expected that ``lb <= 0``, otherwise the algorithm
        may fail. If ``lb[i] = -Inf``, the lower
        bound for the ith component is just ignored.
    ub : array_like, shape (n, )
        Upper bounds to each one of the components of ``x``.
        It is expected that ``ub >= 0``, otherwise the algorithm
        may fail. If ``ub[i] = Inf``, the upper bound for the ith
        component is just ignored.

    Returns
    -------
    x : array_like, shape (n,)
        Solution to the problem.

    Notes
    -----
    Based on implementations described in pp. 885-886 from [1]_.

    References
    ----------
    .. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
           "An interior point algorithm for large-scale nonlinear
           programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
    """
    # Compute minimum norm minimizer of 1/2*|| A x + b ||^2.
    newton_point = -Y.dot(b)
    # Check for interior point
    if inside_box_boundaries(newton_point, lb, ub)  \
       and norm(newton_point) <= trust_radius:
        x = newton_point
        return x

    # Compute gradient vector ``g = A.T b``
    g = A.T.dot(b)
    # Compute Cauchy point
    # `cauchy_point = g.T g / (g.T A.T A g)``.
    A_g = A.dot(g)
    cauchy_point = -np.dot(g, g) / np.dot(A_g, A_g) * g
    # Origin
    origin_point = np.zeros_like(cauchy_point)

    # Check the segment between cauchy_point and newton_point
    # for a possible solution.
    z = cauchy_point
    p = newton_point - cauchy_point
    _, alpha, intersect = box_sphere_intersections(z, p, lb, ub,
                                                   trust_radius)
    if intersect:
        x1 = z + alpha*p
    else:
        # Check the segment between the origin and cauchy_point
        # for a possible solution.
        z = origin_point
        p = cauchy_point
        _, alpha, _ = box_sphere_intersections(z, p, lb, ub,
                                               trust_radius)
        x1 = z + alpha*p

    # Check the segment between origin and newton_point
    # for a possible solution.
    z = origin_point
    p = newton_point
    _, alpha, _ = box_sphere_intersections(z, p, lb, ub,
                                           trust_radius)
    x2 = z + alpha*p

    # Return the best solution among x1 and x2.
    if norm(A.dot(x1) + b) < norm(A.dot(x2) + b):
        return x1
    else:
        return x2


def projected_cg(H, c, Z, Y, b, trust_radius=np.inf,
                 lb=None, ub=None, tol=None,
                 max_iter=None, max_infeasible_iter=None,
                 return_all=False):
    """Solve EQP problem with projected CG method.

    Solve equality-constrained quadratic programming problem
    ``min 1/2 x.T H x + x.t c``  subject to ``A x + b = 0`` and,
    possibly, to trust region constraints ``||x|| < trust_radius``
    and box constraints ``lb <= x <= ub``.

    Parameters
    ----------
    H : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for computing ``H v``.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    Z : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for projecting ``x`` into the null space of A.
    Y : LinearOperator,  sparse matrix, ndarray, shape (n, m)
        Operator that, for a given a vector ``b``, compute smallest
        norm solution of ``A x + b = 0``.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.
    trust_radius : float, optional
        Trust radius to be considered. By default, uses ``trust_radius=inf``,
        which means no trust radius at all.
    lb : array_like, shape (n,), optional
        Lower bounds to each one of the components of ``x``.
        If ``lb[i] = -Inf`` the lower bound for the i-th
        component is just ignored (default).
    ub : array_like, shape (n, ), optional
        Upper bounds to each one of the components of ``x``.
        If ``ub[i] = Inf`` the upper bound for the i-th
        component is just ignored (default).
    tol : float, optional
        Tolerance used to interrupt the algorithm.
    max_iter : int, optional
        Maximum algorithm iterations. Where ``max_inter <= n-m``.
        By default, uses ``max_iter = n-m``.
    max_infeasible_iter : int, optional
        Maximum infeasible (regarding box constraints) iterations the
        algorithm is allowed to take.
        By default, uses ``max_infeasible_iter = n-m``.
    return_all : bool, optional
        When ``true``, return the list of all vectors through the iterations.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the EQP problem.
    info : Dict
        Dictionary containing the following:

            - niter : Number of iterations.
            - stop_cond : Reason for algorithm termination:
                1. Iteration limit was reached;
                2. Reached the trust-region boundary;
                3. Negative curvature detected;
                4. Tolerance was satisfied.
            - allvecs : List containing all intermediary vectors (optional).
            - hits_boundary : True if the proposed step is on the boundary
              of the trust region.

    Notes
    -----
    Implementation of Algorithm 6.2 on [1]_.

    In the absence of spherical and box constraints, for sufficient
    iterations, the method returns a truly optimal result.
    In the presence of those constraints, the value returned is only
    a inexpensive approximation of the optimal value.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
           "On the solution of equality constrained quadratic
            programming problems arising in optimization."
            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    CLOSE_TO_ZERO = 1e-25

    n, = np.shape(c)  # Number of parameters
    m, = np.shape(b)  # Number of constraints

    # Initial Values
    x = Y.dot(-b)
    r = Z.dot(H.dot(x) + c)
    g = Z.dot(r)
    p = -g

    # Store ``x`` value
    if return_all:
        allvecs = [x]
    # Values for the first iteration
    H_p = H.dot(p)
    rt_g = norm(g)**2  # g.T g = r.T Z g = r.T g (ref [1]_ p.1389)

    # If x > trust-region the problem does not have a solution.
    tr_distance = trust_radius - norm(x)
    if tr_distance < 0:
        raise ValueError("Trust region problem does not have a solution.")
    # If x == trust_radius, then x is the solution
    # to the optimization problem, since x is the
    # minimum norm solution to Ax=b.
    elif tr_distance < CLOSE_TO_ZERO:
        info = {'niter': 0, 'stop_cond': 2, 'hits_boundary': True}
        if return_all:
            allvecs.append(x)
            info['allvecs'] = allvecs
        return x, info

    # Set default tolerance
    if tol is None:
        tol = max(min(0.01 * np.sqrt(rt_g), 0.1 * rt_g), CLOSE_TO_ZERO)
    # Set default lower and upper bounds
    if lb is None:
        lb = np.full(n, -np.inf)
    if ub is None:
        ub = np.full(n, np.inf)
    # Set maximum iterations
    if max_iter is None:
        max_iter = n-m
    max_iter = min(max_iter, n-m)
    # Set maximum infeasible iterations
    if max_infeasible_iter is None:
        max_infeasible_iter = n-m

    hits_boundary = False
    stop_cond = 1
    counter = 0
    last_feasible_x = np.zeros_like(x)
    k = 0
    for i in range(max_iter):
        # Stop criteria - Tolerance : r.T g < tol
        if rt_g < tol:
            stop_cond = 4
            break
        k += 1
        # Compute curvature
        pt_H_p = H_p.dot(p)
        # Stop criteria - Negative curvature
        if pt_H_p <= 0:
            if np.isinf(trust_radius):
                raise ValueError("Negative curvature not allowed "
                                 "for unrestricted problems.")
            else:
                # Find intersection with constraints
                _, alpha, intersect = box_sphere_intersections(
                    x, p, lb, ub, trust_radius, entire_line=True)
                # Update solution
                if intersect:
                    x = x + alpha*p
                # Reinforce variables are inside box constraints.
                # This is only necessary because of roundoff errors.
                x = reinforce_box_boundaries(x, lb, ub)
                # Attribute information
                stop_cond = 3
                hits_boundary = True
                break

        # Get next step
        alpha = rt_g / pt_H_p
        x_next = x + alpha*p

        # Stop criteria - Hits boundary
        if np.linalg.norm(x_next) >= trust_radius:
            # Find intersection with box constraints
            _, theta, intersect = box_sphere_intersections(x, alpha*p, lb, ub,
                                                           trust_radius)
            # Update solution
            if intersect:
                x = x + theta*alpha*p
            # Reinforce variables are inside box constraints.
            # This is only necessary because of roundoff errors.
            x = reinforce_box_boundaries(x, lb, ub)
            # Attribute information
            stop_cond = 2
            hits_boundary = True
            break

        # Check if ``x`` is inside the box and start counter if it is not.
        if inside_box_boundaries(x_next, lb, ub):
            counter = 0
        else:
            counter += 1
        # Whenever outside box constraints keep looking for intersections.
        if counter > 0:
            _, theta, intersect = box_sphere_intersections(x, alpha*p, lb, ub,
                                                           trust_radius)
            if intersect:
                last_feasible_x = x + theta*alpha*p
                # Reinforce variables are inside box constraints.
                # This is only necessary because of roundoff errors.
                last_feasible_x = reinforce_box_boundaries(last_feasible_x,
                                                           lb, ub)
                counter = 0
        # Stop after too many infeasible (regarding box constraints) iteration.
        if counter > max_infeasible_iter:
            break
        # Store ``x_next`` value
        if return_all:
            allvecs.append(x_next)

        # Update residual
        r_next = r + alpha*H_p
        # Project residual g+ = Z r+
        g_next = Z.dot(r_next)
        # Compute conjugate direction step d
        rt_g_next = norm(g_next)**2  # g.T g = r.T g (ref [1]_ p.1389)
        beta = rt_g_next / rt_g
        p = - g_next + beta*p
        # Prepare for next iteration
        x = x_next
        g = g_next
        r = g_next
        rt_g = norm(g)**2  # g.T g = r.T Z g = r.T g (ref [1]_ p.1389)
        H_p = H.dot(p)

    if not inside_box_boundaries(x, lb, ub):
        x = last_feasible_x
        hits_boundary = True
    info = {'niter': k, 'stop_cond': stop_cond,
            'hits_boundary': hits_boundary}
    if return_all:
        info['allvecs'] = allvecs
    return x, info
<|endoftext|>"""Trust-region interior point method.

References
----------
.. [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
       "An interior point algorithm for large-scale nonlinear
       programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
.. [2] Byrd, Richard H., Guanghui Liu, and Jorge Nocedal.
       "On the local behavior of an interior point method for
       nonlinear programming." Numerical analysis 1997 (1997): 37-56.
.. [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
       Second Edition (2006).
"""

import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator

__all__ = ['tr_interior_point']


class BarrierSubproblem:
    """
    Barrier optimization problem:
        minimize fun(x) - barrier_parameter*sum(log(s))
        subject to: constr_eq(x)     = 0
                  constr_ineq(x) + s = 0
    """

    def __init__(self, x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq,
                 constr, jac, barrier_parameter, tolerance,
                 enforce_feasibility, global_stop_criteria,
                 xtol, fun0, grad0, constr_ineq0, jac_ineq0, constr_eq0,
                 jac_eq0, finite_diff_bounds):
        # Store parameters
        self.n_vars = n_vars
        self.x0 = x0
        self.s0 = s0
        self.fun = fun
        self.grad = grad
        self.lagr_hess = lagr_hess
        self.constr = constr
        self.jac = jac
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.enforce_feasibility = enforce_feasibility
        self.global_stop_criteria = global_stop_criteria
        self.xtol = xtol
        self.fun0 = self._compute_function(fun0, constr_ineq0, s0)
        self.grad0 = self._compute_gradient(grad0)
        self.constr0 = self._compute_constr(constr_ineq0, constr_eq0, s0)
        self.jac0 = self._compute_jacobian(jac_eq0, jac_ineq0, s0)
        self.terminate = False
        self.lb = finite_diff_bounds[0]
        self.ub = finite_diff_bounds[1]

    def update(self, barrier_parameter, tolerance):
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance

    def get_slack(self, z):
        return z[self.n_vars:self.n_vars+self.n_ineq]

    def get_variables(self, z):
        return z[:self.n_vars]

    def function_and_constraints(self, z):
        """Returns barrier function and constraints at given point.

        For z = [x, s], returns barrier function:
            function(z) = fun(x) - barrier_parameter*sum(log(s))
        and barrier constraints:
            constraints(z) = [   constr_eq(x)     ]
                             [ constr_ineq(x) + s ]

        """
        # Get variables and slack variables
        x = self.get_variables(z)
        s = self.get_slack(z)

        # Compute function and constraints,
        # making sure x is within any strict bounds
        if np.any((x < self.lb) | (x > self.ub)):
            # If x is out of the strict bounds, set f = inf,
            # and just set both equality and inequality
            # constraints to 0 since we can't evaluate
            # them separately.
            f = np.inf
            c_eq = np.full(self.n_eq, 0.)
            c_ineq = np.full(self.n_ineq, 0.)
        else:
            f = self.fun(x)
            c_eq, c_ineq = self.constr(x)

        # Return objective function and constraints
        return (self._compute_function(f, c_ineq, s),
                self._compute_constr(c_ineq, c_eq, s))

    def _compute_function(self, f, c_ineq, s):
        # Use technique from Nocedal and Wright book, ref [3]_, p.576,
        # to guarantee constraints from `enforce_feasibility`
        # stay feasible along iterations.
        s[self.enforce_feasibility] = -c_ineq[self.enforce_feasibility]
        log_s = [np.log(s_i) if s_i > 0 else -np.inf for s_i in s]
        # Compute barrier objective function
        return f - self.barrier_parameter*np.sum(log_s)

    def _compute_constr(self, c_ineq, c_eq, s):
        # Compute barrier constraint
        return np.hstack((c_eq,
                          c_ineq + s))

    def scaling(self, z):
        """Returns scaling vector.
        Given by:
            scaling = [ones(n_vars), s]
        """
        s = self.get_slack(z)
        diag_elements = np.hstack((np.ones(self.n_vars), s))

        # Diagonal matrix
        def matvec(vec):
            return diag_elements*vec
        return LinearOperator((self.n_vars+self.n_ineq,
                               self.n_vars+self.n_ineq),
                              matvec)

    def gradient_and_jacobian(self, z):
        """Returns scaled gradient.

        Return scaled gradient:
            gradient = [             grad(x)             ]
                       [ -barrier_parameter*ones(n_ineq) ]
        and scaled Jacobian matrix:
            jacobian = [  jac_eq(x)  0  ]
                       [ jac_ineq(x) S  ]
        Both of them scaled by the previously defined scaling factor.
        """
        # Get variables and slack variables
        x = self.get_variables(z)
        s = self.get_slack(z)
        # Compute first derivatives
        g = self.grad(x)
        J_eq, J_ineq = self.jac(x)
        # Return gradient and Jacobian
        return (self._compute_gradient(g),
                self._compute_jacobian(J_eq, J_ineq, s))

    def _compute_gradient(self, g):
        return np.hstack((g, -self.barrier_parameter*np.ones(self.n_ineq)))

    def _compute_jacobian(self, J_eq, J_ineq, s):
        if self.n_ineq == 0:
            return J_eq
        else:
            if sps.issparse(J_eq) or sps.issparse(J_ineq):
                # It is expected that J_eq and J_ineq
                # are already `csr_matrix` because of
                # the way ``BoxConstraint``, ``NonlinearConstraint``
                # and ``LinearConstraint`` are defined.
                J_eq = sps.csr_matrix(J_eq)
                J_ineq = sps.csr_matrix(J_ineq)
                return self._assemble_sparse_jacobian(J_eq, J_ineq, s)
            else:
                S = np.diag(s)
                zeros = np.zeros((self.n_eq, self.n_ineq))
                # Convert to matrix
                if sps.issparse(J_ineq):
                    J_ineq = J_ineq.toarray()
                if sps.issparse(J_eq):
                    J_eq = J_eq.toarray()
                # Concatenate matrices
                return np.block([[J_eq, zeros],
                                 [J_ineq, S]])

    def _assemble_sparse_jacobian(self, J_eq, J_ineq, s):
        """Assemble sparse Jacobian given its components.

        Given ``J_eq``, ``J_ineq`` and ``s`` returns:
            jacobian = [ J_eq,     0     ]
                       [ J_ineq, diag(s) ]

        It is equivalent to:
            sps.bmat([[ J_eq,   None    ],
                      [ J_ineq, diag(s) ]], "csr")
        but significantly more efficient for this
        given structure.
        """
        n_vars, n_ineq, n_eq = self.n_vars, self.n_ineq, self.n_eq
        J_aux = sps.vstack([J_eq, J_ineq], "csr")
        indptr, indices, data = J_aux.indptr, J_aux.indices, J_aux.data
        new_indptr = indptr + np.hstack((np.zeros(n_eq, dtype=int),
                                         np.arange(n_ineq+1, dtype=int)))
        size = indices.size+n_ineq
        new_indices = np.empty(size)
        new_data = np.empty(size)
        mask = np.full(size, False, bool)
        mask[new_indptr[-n_ineq:]-1] = True
        new_indices[mask] = n_vars+np.arange(n_ineq)
        new_indices[~mask] = indices
        new_data[mask] = s
        new_data[~mask] = data
        J = sps.csr_matrix((new_data, new_indices, new_indptr),
                           (n_eq + n_ineq, n_vars + n_ineq))
        return J

    def lagrangian_hessian_x(self, z, v):
        """Returns Lagrangian Hessian (in relation to `x`) -> Hx"""
        x = self.get_variables(z)
        # Get lagrange multipliers related to nonlinear equality constraints
        v_eq = v[:self.n_eq]
        # Get lagrange multipliers related to nonlinear ineq. constraints
        v_ineq = v[self.n_eq:self.n_eq+self.n_ineq]
        lagr_hess = self.lagr_hess
        return lagr_hess(x, v_eq, v_ineq)

    def lagrangian_hessian_s(self, z, v):
        """Returns scaled Lagrangian Hessian (in relation to`s`) -> S Hs S"""
        s = self.get_slack(z)
        # Using the primal formulation:
        #     S Hs S = diag(s)*diag(barrier_parameter/s**2)*diag(s).
        # Reference [1]_ p. 882, formula (3.1)
        primal = self.barrier_parameter
        # Using the primal-dual formulation
        #     S Hs S = diag(s)*diag(v/s)*diag(s)
        # Reference [1]_ p. 883, formula (3.11)
        primal_dual = v[-self.n_ineq:]*s
        # Uses the primal-dual formulation for
        # positives values of v_ineq, and primal
        # formulation for the remaining ones.
        return np.where(v[-self.n_ineq:] > 0, primal_dual, primal)

    def lagrangian_hessian(self, z, v):
        """Returns scaled Lagrangian Hessian"""
        # Compute Hessian in relation to x and s
        Hx = self.lagrangian_hessian_x(z, v)
        if self.n_ineq > 0:
            S_Hs_S = self.lagrangian_hessian_s(z, v)

        # The scaled Lagragian Hessian is:
        #     [ Hx    0    ]
        #     [ 0   S Hs S ]
        def matvec(vec):
            vec_x = self.get_variables(vec)
            vec_s = self.get_slack(vec)
            if self.n_ineq > 0:
                return np.hstack((Hx.dot(vec_x), S_Hs_S*vec_s))
            else:
                return Hx.dot(vec_x)
        return LinearOperator((self.n_vars+self.n_ineq,
                               self.n_vars+self.n_ineq),
                              matvec)

    def stop_criteria(self, state, z, last_iteration_failed,
                      optimality, constr_violation,
                      trust_radius, penalty, cg_info):
        """Stop criteria to the barrier problem.
        The criteria here proposed is similar to formula (2.3)
        from [1]_, p.879.
        """
        x = self.get_variables(z)
        if self.global_stop_criteria(state, x,
                                     last_iteration_failed,
                                     trust_radius, penalty,
                                     cg_info,
                                     self.barrier_parameter,
                                     self.tolerance):
            self.terminate = True
            return True
        else:
            g_cond = (optimality < self.tolerance and
                      constr_violation < self.tolerance)
            x_cond = trust_radius < self.xtol
            return g_cond or x_cond


def tr_interior_point(fun, grad, lagr_hess, n_vars, n_ineq, n_eq,
                      constr, jac, x0, fun0, grad0,
                      constr_ineq0, jac_ineq0, constr_eq0,
                      jac_eq0, stop_criteria,
                      enforce_feasibility, xtol, state,
                      initial_barrier_parameter,
                      initial_tolerance,
                      initial_penalty,
                      initial_trust_radius,
                      factorization_method,
                      finite_diff_bounds):
    """Trust-region interior points method.

    Solve problem:
        minimize fun(x)
        subject to: constr_ineq(x) <= 0
                    constr_eq(x) = 0
    using trust-region interior point method described in [1]_.
    """
    # BOUNDARY_PARAMETER controls the decrease on the slack
    # variables. Represents ``tau`` from [1]_ p.885, formula (3.18).
    BOUNDARY_PARAMETER = 0.995
    # BARRIER_DECAY_RATIO controls the decay of the barrier parameter
    # and of the subproblem tolerance. Represents ``theta`` from [1]_ p.879.
    BARRIER_DECAY_RATIO = 0.2
    # TRUST_ENLARGEMENT controls the enlargement on trust radius
    # after each iteration
    TRUST_ENLARGEMENT = 5

    # Default enforce_feasibility
    if enforce_feasibility is None:
        enforce_feasibility = np.zeros(n_ineq, bool)
    # Initial Values
    barrier_parameter = initial_barrier_parameter
    tolerance = initial_tolerance
    trust_radius = initial_trust_radius
    # Define initial value for the slack variables
    s0 = np.maximum(-1.5*constr_ineq0, np.ones(n_ineq))
    # Define barrier subproblem
    subprob = BarrierSubproblem(
        x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac,
        barrier_parameter, tolerance, enforce_feasibility,
        stop_criteria, xtol, fun0, grad0, constr_ineq0, jac_ineq0,
        constr_eq0, jac_eq0, finite_diff_bounds)
    # Define initial parameter for the first iteration.
    z = np.hstack((x0, s0))
    fun0_subprob, constr0_subprob = subprob.fun0, subprob.constr0
    grad0_subprob, jac0_subprob = subprob.grad0, subprob.jac0
    # Define trust region bounds
    trust_lb = np.hstack((np.full(subprob.n_vars, -np.inf),
                          np.full(subprob.n_ineq, -BOUNDARY_PARAMETER)))
    trust_ub = np.full(subprob.n_vars+subprob.n_ineq, np.inf)

    # Solves a sequence of barrier problems
    while True:
        # Solve SQP subproblem
        z, state = equality_constrained_sqp(
            subprob.function_and_constraints,
            subprob.gradient_and_jacobian,
            subprob.lagrangian_hessian,
            z, fun0_subprob, grad0_subprob,
            constr0_subprob, jac0_subprob, subprob.stop_criteria,
            state, initial_penalty, trust_radius,
            factorization_method, trust_lb, trust_ub, subprob.scaling)
        if subprob.terminate:
            break
        # Update parameters
        trust_radius = max(initial_trust_radius,
                           TRUST_ENLARGEMENT*state.tr_radius)
        # TODO: Use more advanced strategies from [2]_
        # to update this parameters.
        barrier_parameter *= BARRIER_DECAY_RATIO
        tolerance *= BARRIER_DECAY_RATIO
        # Update Barrier Problem
        subprob.update(barrier_parameter, tolerance)
        # Compute initial values for next iteration
        fun0_subprob, constr0_subprob = subprob.function_and_constraints(z)
        grad0_subprob, jac0_subprob = subprob.gradient_and_jacobian(z)

    # Get x and s
    x = subprob.get_variables(z)
    return x, state
<|endoftext|>import time
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .._differentiable_functions import VectorFunction
from .._constraints import (
    NonlinearConstraint, LinearConstraint, PreparedConstraint, Bounds, strict_bounds)
from .._hessian_update_strategy import BFGS
from .._optimize import OptimizeResult
from .._differentiable_functions import ScalarFunction
from .equality_constrained_sqp import equality_constrained_sqp
from .canonical_constraint import (CanonicalConstraint,
                                   initial_constraints_as_canonical)
from .tr_interior_point import tr_interior_point
from .report import BasicReport, SQPReport, IPReport


TERMINATION_MESSAGES = {
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`xtol` termination condition is satisfied.",
    3: "`callback` function requested termination.",
    4: "Constraint violation exceeds 'gtol'"
}


class HessianLinearOperator:
    """Build LinearOperator from hessp"""
    def __init__(self, hessp, n):
        self.hessp = hessp
        self.n = n

    def __call__(self, x, *args):
        def matvec(p):
            return self.hessp(x, p, *args)

        return LinearOperator((self.n, self.n), matvec=matvec)


class LagrangianHessian:
    """The Hessian of the Lagrangian as LinearOperator.

    The Lagrangian is computed as the objective function plus all the
    constraints multiplied with some numbers (Lagrange multipliers).
    """
    def __init__(self, n, objective_hess, constraints_hess):
        self.n = n
        self.objective_hess = objective_hess
        self.constraints_hess = constraints_hess

    def __call__(self, x, v_eq, v_ineq=None):
        if v_ineq is None:
            v_ineq = np.empty(0)
        H_objective = self.objective_hess(x)
        H_constraints = self.constraints_hess(x, v_eq, v_ineq)

        def matvec(p):
            return H_objective.dot(p) + H_constraints.dot(p)

        return LinearOperator((self.n, self.n), matvec)


def update_state_sqp(state, x, last_iteration_failed, objective, prepared_constraints,
                     start_time, tr_radius, constr_penalty, cg_info):
    state.nit += 1
    state.nfev = objective.nfev
    state.njev = objective.ngev
    state.nhev = objective.nhev
    state.constr_nfev = [c.fun.nfev if isinstance(c.fun, VectorFunction) else 0
                         for c in prepared_constraints]
    state.constr_njev = [c.fun.njev if isinstance(c.fun, VectorFunction) else 0
                         for c in prepared_constraints]
    state.constr_nhev = [c.fun.nhev if isinstance(c.fun, VectorFunction) else 0
                         for c in prepared_constraints]

    if not last_iteration_failed:
        state.x = x
        state.fun = objective.f
        state.grad = objective.g
        state.v = [c.fun.v for c in prepared_constraints]
        state.constr = [c.fun.f for c in prepared_constraints]
        state.jac = [c.fun.J for c in prepared_constraints]
        # Compute Lagrangian Gradient
        state.lagrangian_grad = np.copy(state.grad)
        for c in prepared_constraints:
            state.lagrangian_grad += c.fun.J.T.dot(c.fun.v)
        state.optimality = np.linalg.norm(state.lagrangian_grad, np.inf)
        # Compute maximum constraint violation
        state.constr_violation = 0
        for i in range(len(prepared_constraints)):
            lb, ub = prepared_constraints[i].bounds
            c = state.constr[i]
            state.constr_violation = np.max([state.constr_violation,
                                             np.max(lb - c),
                                             np.max(c - ub)])

    state.execution_time = time.time() - start_time
    state.tr_radius = tr_radius
    state.constr_penalty = constr_penalty
    state.cg_niter += cg_info["niter"]
    state.cg_stop_cond = cg_info["stop_cond"]

    return state


def update_state_ip(state, x, last_iteration_failed, objective,
                    prepared_constraints, start_time,
                    tr_radius, constr_penalty, cg_info,
                    barrier_parameter, barrier_tolerance):
    state = update_state_sqp(state, x, last_iteration_failed, objective,
                             prepared_constraints, start_time, tr_radius,
                             constr_penalty, cg_info)
    state.barrier_parameter = barrier_parameter
    state.barrier_tolerance = barrier_tolerance
    return state


def _minimize_trustregion_constr(fun, x0, args, grad,
                                 hess, hessp, bounds, constraints,
                                 xtol=1e-8, gtol=1e-8,
                                 barrier_tol=1e-8,
                                 sparse_jacobian=None,
                                 callback=None, maxiter=1000,
                                 verbose=0, finite_diff_rel_step=None,
                                 initial_constr_penalty=1.0, initial_tr_radius=1.0,
                                 initial_barrier_parameter=0.1,
                                 initial_barrier_tolerance=0.1,
                                 factorization_method=None,
                                 disp=False):
    """Minimize a scalar function subject to constraints.

    Parameters
    ----------
    gtol : float, optional
        Tolerance for termination by the norm of the Lagrangian gradient.
        The algorithm will terminate when both the infinity norm (i.e., max
        abs value) of the Lagrangian gradient and the constraint violation
        are smaller than ``gtol``. Default is 1e-8.
    xtol : float, optional
        Tolerance for termination by the change of the independent variable.
        The algorithm will terminate when ``tr_radius < xtol``, where
        ``tr_radius`` is the radius of the trust region used in the algorithm.
        Default is 1e-8.
    barrier_tol : float, optional
        Threshold on the barrier parameter for the algorithm termination.
        When inequality constraints are present, the algorithm will terminate
        only when the barrier parameter is less than `barrier_tol`.
        Default is 1e-8.
    sparse_jacobian : {bool, None}, optional
        Determines how to represent Jacobians of the constraints. If bool,
        then Jacobians of all the constraints will be converted to the
        corresponding format. If None (default), then Jacobians won't be
        converted, but the algorithm can proceed only if they all have the
        same format.
    initial_tr_radius: float, optional
        Initial trust radius. The trust radius gives the maximum distance
        between solution points in consecutive iterations. It reflects the
        trust the algorithm puts in the local approximation of the optimization
        problem. For an accurate local approximation the trust-region should be
        large and for an  approximation valid only close to the current point it
        should be a small one. The trust radius is automatically updated throughout
        the optimization process, with ``initial_tr_radius`` being its initial value.
        Default is 1 (recommended in [1]_, p. 19).
    initial_constr_penalty : float, optional
        Initial constraints penalty parameter. The penalty parameter is used for
        balancing the requirements of decreasing the objective function
        and satisfying the constraints. It is used for defining the merit function:
        ``merit_function(x) = fun(x) + constr_penalty * constr_norm_l2(x)``,
        where ``constr_norm_l2(x)`` is the l2 norm of a vector containing all
        the constraints. The merit function is used for accepting or rejecting
        trial points and ``constr_penalty`` weights the two conflicting goals
        of reducing objective function and constraints. The penalty is automatically
        updated throughout the optimization  process, with
        ``initial_constr_penalty`` being its  initial value. Default is 1
        (recommended in [1]_, p 19).
    initial_barrier_parameter, initial_barrier_tolerance: float, optional
        Initial barrier parameter and initial tolerance for the barrier subproblem.
        Both are used only when inequality constraints are present. For dealing with
        optimization problems ``min_x f(x)`` subject to inequality constraints
        ``c(x) <= 0`` the algorithm introduces slack variables, solving the problem
        ``min_(x,s) f(x) + barrier_parameter*sum(ln(s))`` subject to the equality
        constraints  ``c(x) + s = 0`` instead of the original problem. This subproblem
        is solved for decreasing values of ``barrier_parameter`` and with decreasing
        tolerances for the termination, starting with ``initial_barrier_parameter``
        for the barrier parameter and ``initial_barrier_tolerance`` for the
        barrier tolerance. Default is 0.1 for both values (recommended in [1]_ p. 19).
        Also note that ``barrier_parameter`` and ``barrier_tolerance`` are updated
        with the same prefactor.
    factorization_method : string or None, optional
        Method to factorize the Jacobian of the constraints. Use None (default)
        for the auto selection or one of:

        - 'NormalEquation' (requires scikit-sparse)
        - 'AugmentedSystem'
        - 'QRFactorization'
        - 'SVDFactorization'

        The methods 'NormalEquation' and 'AugmentedSystem' can be used only
        with sparse constraints. The projections required by the algorithm
        will be computed using, respectively, the normal equation  and the
        augmented system approaches explained in [1]_. 'NormalEquation'
        computes the Cholesky factorization of ``A A.T`` and 'AugmentedSystem'
        performs the LU factorization of an augmented system. They usually
        provide similar results. 'AugmentedSystem' is used by default for
        sparse matrices.

        The methods 'QRFactorization' and 'SVDFactorization' can be used
        only with dense constraints. They compute the required projections
        using, respectively, QR and SVD factorizations. The 'SVDFactorization'
        method can cope with Jacobian matrices with deficient row rank and will
        be used whenever other factorization methods fail (which may imply the
        conversion of sparse matrices to a dense format when required).
        By default, 'QRFactorization' is used for dense matrices.
    finite_diff_rel_step : None or array_like, optional
        Relative step size for the finite difference approximation.
    maxiter : int, optional
        Maximum number of algorithm iterations. Default is 1000.
    verbose : {0, 1, 2, 3}, optional
        Level of algorithm's verbosity:

        * 0 (default) : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations.
        * 3 : display progress during iterations (more complete report).

    disp : bool, optional
        If True (default), then `verbose` will be set to 1 if it was 0.

    Returns
    -------
    `OptimizeResult` with the fields documented below. Note the following:

        1. All values corresponding to the constraints are ordered as they
           were passed to the solver. And values corresponding to `bounds`
           constraints are put *after* other constraints.
        2. All numbers of function, Jacobian or Hessian evaluations correspond
           to numbers of actual Python function calls. It means, for example,
           that if a Jacobian is estimated by finite differences, then the
           number of Jacobian evaluations will be zero and the number of
           function evaluations will be incremented by all calls during the
           finite difference estimation.

    x : ndarray, shape (n,)
        Solution found.
    optimality : float
        Infinity norm of the Lagrangian gradient at the solution.
    constr_violation : float
        Maximum constraint violation at the solution.
    fun : float
        Objective function at the solution.
    grad : ndarray, shape (n,)
        Gradient of the objective function at the solution.
    lagrangian_grad : ndarray, shape (n,)
        Gradient of the Lagrangian function at the solution.
    nit : int
        Total number of iterations.
    nfev : integer
        Number of the objective function evaluations.
    njev : integer
        Number of the objective function gradient evaluations.
    nhev : integer
        Number of the objective function Hessian evaluations.
    cg_niter : int
        Total number of the conjugate gradient method iterations.
    method : {'equality_constrained_sqp', 'tr_interior_point'}
        Optimization method used.
    constr : list of ndarray
        List of constraint values at the solution.
    jac : list of {ndarray, sparse matrix}
        List of the Jacobian matrices of the constraints at the solution.
    v : list of ndarray
        List of the Lagrange multipliers for the constraints at the solution.
        For an inequality constraint a positive multiplier means that the upper
        bound is active, a negative multiplier means that the lower bound is
        active and if a multiplier is zero it means the constraint is not
        active.
    constr_nfev : list of int
        Number of constraint evaluations for each of the constraints.
    constr_njev : list of int
        Number of Jacobian matrix evaluations for each of the constraints.
    constr_nhev : list of int
        Number of Hessian evaluations for each of the constraints.
    tr_radius : float
        Radius of the trust region at the last iteration.
    constr_penalty : float
        Penalty parameter at the last iteration, see `initial_constr_penalty`.
    barrier_tolerance : float
        Tolerance for the barrier subproblem at the last iteration.
        Only for problems with inequality constraints.
    barrier_parameter : float
        Barrier parameter at the last iteration. Only for problems
        with inequality constraints.
    execution_time : float
        Total execution time.
    message : str
        Termination message.
    status : {0, 1, 2, 3, 4}
        Termination status:

        * 0 : The maximum number of function evaluations is exceeded.
        * 1 : `gtol` termination condition is satisfied.
        * 2 : `xtol` termination condition is satisfied.
        * 3 : `callback` function requested termination.
        * 4 : Constraint violation exceeds 'gtol'.

        .. versionchanged:: 1.15.0
            If the constraint violation exceeds `gtol`, then ``result.success``
            will now be False.

    cg_stop_cond : int
        Reason for CG subproblem termination at the last iteration:

        * 0 : CG subproblem not evaluated.
        * 1 : Iteration limit was reached.
        * 2 : Reached the trust-region boundary.
        * 3 : Negative curvature detected.
        * 4 : Tolerance was satisfied.

    References
    ----------
    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
           Trust region methods. 2000. Siam. pp. 19.
    """
    x0 = np.atleast_1d(x0).astype(float)
    n_vars = np.size(x0)
    if hess is None:
        if callable(hessp):
            hess = HessianLinearOperator(hessp, n_vars)
        else:
            hess = BFGS()
    if disp and verbose == 0:
        verbose = 1

    if bounds is not None:
        modified_lb = np.nextafter(bounds.lb, -np.inf, where=bounds.lb > -np.inf)
        modified_ub = np.nextafter(bounds.ub, np.inf, where=bounds.ub < np.inf)
        modified_lb = np.where(np.isfinite(bounds.lb), modified_lb, bounds.lb)
        modified_ub = np.where(np.isfinite(bounds.ub), modified_ub, bounds.ub)
        bounds = Bounds(modified_lb, modified_ub, keep_feasible=bounds.keep_feasible)
        finite_diff_bounds = strict_bounds(bounds.lb, bounds.ub,
                                           bounds.keep_feasible, n_vars)
    else:
        finite_diff_bounds = (-np.inf, np.inf)

    # Define Objective Function
    objective = ScalarFunction(fun, x0, args, grad, hess,
                               finite_diff_rel_step, finite_diff_bounds)

    # Put constraints in list format when needed.
    if isinstance(constraints, (NonlinearConstraint | LinearConstraint)):
        constraints = [constraints]

    # Prepare constraints.
    prepared_constraints = [
        PreparedConstraint(c, x0, sparse_jacobian, finite_diff_bounds)
        for c in constraints]

    # Check that all constraints are either sparse or dense.
    n_sparse = sum(c.fun.sparse_jacobian for c in prepared_constraints)
    if 0 < n_sparse < len(prepared_constraints):
        raise ValueError("All constraints must have the same kind of the "
                         "Jacobian --- either all sparse or all dense. "
                         "You can set the sparsity globally by setting "
                         "`sparse_jacobian` to either True of False.")
    if prepared_constraints:
        sparse_jacobian = n_sparse > 0

    if bounds is not None:
        if sparse_jacobian is None:
            sparse_jacobian = True
        prepared_constraints.append(PreparedConstraint(bounds, x0,
                                                       sparse_jacobian))

    # Concatenate initial constraints to the canonical form.
    c_eq0, c_ineq0, J_eq0, J_ineq0 = initial_constraints_as_canonical(
        n_vars, prepared_constraints, sparse_jacobian)

    # Prepare all canonical constraints and concatenate it into one.
    canonical_all = [CanonicalConstraint.from_PreparedConstraint(c)
                     for c in prepared_constraints]

    if len(canonical_all) == 0:
        canonical = CanonicalConstraint.empty(n_vars)
    elif len(canonical_all) == 1:
        canonical = canonical_all[0]
    else:
        canonical = CanonicalConstraint.concatenate(canonical_all,
                                                    sparse_jacobian)

    # Generate the Hessian of the Lagrangian.
    lagrangian_hess = LagrangianHessian(n_vars, objective.hess, canonical.hess)

    # Choose appropriate method
    if canonical.n_ineq == 0:
        method = 'equality_constrained_sqp'
    else:
        method = 'tr_interior_point'

    # Construct OptimizeResult
    state = OptimizeResult(
        nit=0, nfev=0, njev=0, nhev=0,
        cg_niter=0, cg_stop_cond=0,
        fun=objective.f, grad=objective.g,
        lagrangian_grad=np.copy(objective.g),
        constr=[c.fun.f for c in prepared_constraints],
        jac=[c.fun.J for c in prepared_constraints],
        constr_nfev=[0 for c in prepared_constraints],
        constr_njev=[0 for c in prepared_constraints],
        constr_nhev=[0 for c in prepared_constraints],
        v=[c.fun.v for c in prepared_constraints],
        method=method)

    # Start counting
    start_time = time.time()

    # Define stop criteria
    if method == 'equality_constrained_sqp':
        def stop_criteria(state, x, last_iteration_failed,
                          optimality, constr_violation,
                          tr_radius, constr_penalty, cg_info):
            state = update_state_sqp(state, x, last_iteration_failed,
                                     objective, prepared_constraints,
                                     start_time, tr_radius, constr_penalty,
                                     cg_info)
            if verbose == 2:
                BasicReport.print_iteration(state.nit,
                                            state.nfev,
                                            state.cg_niter,
                                            state.fun,
                                            state.tr_radius,
                                            state.optimality,
                                            state.constr_violation)
            elif verbose > 2:
                SQPReport.print_iteration(state.nit,
                                          state.nfev,
                                          state.cg_niter,
                                          state.fun,
                                          state.tr_radius,
                                          state.optimality,
                                          state.constr_violation,
                                          state.constr_penalty,
                                          state.cg_stop_cond)
            state.status = None
            state.niter = state.nit  # Alias for callback (backward-compatibility)
            if callback is not None:
                callback_stop = False
                try:
                    callback_stop = callback(state)
                except StopIteration:
                    callback_stop = True
                if callback_stop:
                    state.status = 3
                    return True
            if state.optimality < gtol and state.constr_violation < gtol:
                state.status = 1
            elif state.tr_radius < xtol:
                state.status = 2
            elif state.nit >= maxiter:
                state.status = 0
            return state.status in (0, 1, 2, 3)
    elif method == 'tr_interior_point':
        def stop_criteria(state, x, last_iteration_failed, tr_radius,
                          constr_penalty, cg_info, barrier_parameter,
                          barrier_tolerance):
            state = update_state_ip(state, x, last_iteration_failed,
                                    objective, prepared_constraints,
                                    start_time, tr_radius, constr_penalty,
                                    cg_info, barrier_parameter, barrier_tolerance)
            if verbose == 2:
                BasicReport.print_iteration(state.nit,
                                            state.nfev,
                                            state.cg_niter,
                                            state.fun,
                                            state.tr_radius,
                                            state.optimality,
                                            state.constr_violation)
            elif verbose > 2:
                IPReport.print_iteration(state.nit,
                                         state.nfev,
                                         state.cg_niter,
                                         state.fun,
                                         state.tr_radius,
                                         state.optimality,
                                         state.constr_violation,
                                         state.constr_penalty,
                                         state.barrier_parameter,
                                         state.cg_stop_cond)
            state.status = None
            state.niter = state.nit  # Alias for callback (backward compatibility)
            if callback is not None:
                callback_stop = False
                try:
                    callback_stop = callback(state)
                except StopIteration:
                    callback_stop = True
                if callback_stop:
                    state.status = 3
                    return True
            if state.optimality < gtol and state.constr_violation < gtol:
                state.status = 1
            elif (state.tr_radius < xtol
                  and state.barrier_parameter < barrier_tol):
                state.status = 2
            elif state.nit >= maxiter:
                state.status = 0
            return state.status in (0, 1, 2, 3)

    if verbose == 2:
        BasicReport.print_header()
    elif verbose > 2:
        if method == 'equality_constrained_sqp':
            SQPReport.print_header()
        elif method == 'tr_interior_point':
            IPReport.print_header()

    # Call inferior function to do the optimization
    if method == 'equality_constrained_sqp':
        def fun_and_constr(x):
            f = objective.fun(x)
            c_eq, _ = canonical.fun(x)
            return f, c_eq

        def grad_and_jac(x):
            g = objective.grad(x)
            J_eq, _ = canonical.jac(x)
            return g, J_eq

        _, result = equality_constrained_sqp(
            fun_and_constr, grad_and_jac, lagrangian_hess,
            x0, objective.f, objective.g,
            c_eq0, J_eq0,
            stop_criteria, state,
            initial_constr_penalty, initial_tr_radius,
            factorization_method)

    elif method == 'tr_interior_point':
        _, result = tr_interior_point(
            objective.fun, objective.grad, lagrangian_hess,
            n_vars, canonical.n_ineq, canonical.n_eq,
            canonical.fun, canonical.jac,
            x0, objective.f, objective.g,
            c_ineq0, J_ineq0, c_eq0, J_eq0,
            stop_criteria,
            canonical.keep_feasible,
            xtol, state, initial_barrier_parameter,
            initial_barrier_tolerance,
            initial_constr_penalty, initial_tr_radius,
            factorization_method, finite_diff_bounds)

    # Status 4 occurs when minimize is successful but constraints are not satisfied.
    if result.status in (1, 2) and state.constr_violation > gtol:
        result.status = 4

    # Status 3 occurs when the callback function requests termination,
    # this is assumed to not be a success.
    result.success = True if result.status in (1, 2) else False
    result.message = TERMINATION_MESSAGES[result.status]

    # Alias (for backward compatibility with 1.1.0)
    result.niter = result.nit

    if verbose == 2:
        BasicReport.print_footer()
    elif verbose > 2:
        if method == 'equality_constrained_sqp':
            SQPReport.print_footer()
        elif method == 'tr_interior_point':
            IPReport.print_footer()
    if verbose >= 1:
        print(result.message)
        print(f"Number of iterations: {result.nit}, "
              f"function evaluations: {result.nfev}, "
              f"CG iterations: {result.cg_niter}, "
              f"optimality: {result.optimality:.2e}, "
              f"constraint violation: {result.constr_violation:.2e}, "
              f"execution time: {result.execution_time:4.2} s.")
    return result
<|endoftext|>"""Basic linear factorizations needed by the solver."""

from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
try:
    from sksparse.cholmod import cholesky_AAt
    sksparse_available = True
except ImportError:
    import warnings
    sksparse_available = False
import numpy as np
from warnings import warn

__all__ = [
    'orthogonality',
    'projections',
]


def orthogonality(A, g):
    """Measure orthogonality between a vector and the null space of a matrix.

    Compute a measure of orthogonality between the null space
    of the (possibly sparse) matrix ``A`` and a given vector ``g``.

    The formula is a simplified (and cheaper) version of formula (3.13)
    from [1]_.
    ``orth =  norm(A g, ord=2)/(norm(A, ord='fro')*norm(g, ord=2))``.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
           "On the solution of equality constrained quadratic
            programming problems arising in optimization."
            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    # Compute vector norms
    norm_g = np.linalg.norm(g)
    # Compute Froebnius norm of the matrix A
    if issparse(A):
        norm_A = scipy.sparse.linalg.norm(A, ord='fro')
    else:
        norm_A = np.linalg.norm(A, ord='fro')

    # Check if norms are zero
    if norm_g == 0 or norm_A == 0:
        return 0

    norm_A_g = np.linalg.norm(A.dot(g))
    # Orthogonality measure
    orth = norm_A_g / (norm_A*norm_g)
    return orth


def normal_equation_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``NormalEquation`` approach.
    """
    # Cholesky factorization
    factor = cholesky_AAt(A)

    # z = x - A.T inv(A A.T) A x
    def null_space(x):
        v = factor(A.dot(x))
        z = x - A.T.dot(v)

        # Iterative refinement to improve roundoff
        # errors described in [2]_, algorithm 5.1.
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # z_next = z - A.T inv(A A.T) A z
            v = factor(A.dot(z))
            z = z - A.T.dot(v)
            k += 1

        return z

    # z = inv(A A.T) A x
    def least_squares(x):
        return factor(A.dot(x))

    # z = A.T inv(A A.T) x
    def row_space(x):
        return A.T.dot(factor(x))

    return null_space, least_squares, row_space


def augmented_system_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A - ``AugmentedSystem``."""
    # Form augmented system
    K = csc_matrix(bmat([[eye(n), A.T], [A, None]]))
    # LU factorization
    # TODO: Use a symmetric indefinite factorization
    #       to solve the system twice as fast (because
    #       of the symmetry).
    try:
        solve = scipy.sparse.linalg.factorized(K)
    except RuntimeError:
        warn("Singular Jacobian matrix. Using dense SVD decomposition to "
             "perform the factorizations.",
             stacklevel=3)
        return svd_factorization_projections(A.toarray(),
                                             m, n, orth_tol,
                                             max_refin, tol)

    # z = x - A.T inv(A A.T) A x
    # is computed solving the extended system:
    # [I A.T] * [ z ] = [x]
    # [A  O ]   [aux]   [0]
    def null_space(x):
        # v = [x]
        #     [0]
        v = np.hstack([x, np.zeros(m)])
        # lu_sol = [ z ]
        #          [aux]
        lu_sol = solve(v)
        z = lu_sol[:n]

        # Iterative refinement to improve roundoff
        # errors described in [2]_, algorithm 5.2.
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # new_v = [x] - [I A.T] * [ z ]
            #         [0]   [A  O ]   [aux]
            new_v = v - K.dot(lu_sol)
            # [I A.T] * [delta  z ] = new_v
            # [A  O ]   [delta aux]
            lu_update = solve(new_v)
            #  [ z ] += [delta  z ]
            #  [aux]    [delta aux]
            lu_sol += lu_update
            z = lu_sol[:n]
            k += 1

        # return z = x - A.T inv(A A.T) A x
        return z

    # z = inv(A A.T) A x
    # is computed solving the extended system:
    # [I A.T] * [aux] = [x]
    # [A  O ]   [ z ]   [0]
    def least_squares(x):
        # v = [x]
        #     [0]
        v = np.hstack([x, np.zeros(m)])
        # lu_sol = [aux]
        #          [ z ]
        lu_sol = solve(v)
        # return z = inv(A A.T) A x
        return lu_sol[n:m+n]

    # z = A.T inv(A A.T) x
    # is computed solving the extended system:
    # [I A.T] * [ z ] = [0]
    # [A  O ]   [aux]   [x]
    def row_space(x):
        # v = [0]
        #     [x]
        v = np.hstack([np.zeros(n), x])
        # lu_sol = [ z ]
        #          [aux]
        lu_sol = solve(v)
        # return z = A.T inv(A A.T) x
        return lu_sol[:n]

    return null_space, least_squares, row_space


def qr_factorization_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``QRFactorization`` approach.
    """
    # QRFactorization
    Q, R, P = scipy.linalg.qr(A.T, pivoting=True, mode='economic')

    if np.linalg.norm(R[-1, :], np.inf) < tol:
        warn('Singular Jacobian matrix. Using SVD decomposition to ' +
             'perform the factorizations.',
             stacklevel=3)
        return svd_factorization_projections(A, m, n,
                                             orth_tol,
                                             max_refin,
                                             tol)

    # z = x - A.T inv(A A.T) A x
    def null_space(x):
        # v = P inv(R) Q.T x
        aux1 = Q.T.dot(x)
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
        v = np.zeros(m)
        v[P] = aux2
        z = x - A.T.dot(v)

        # Iterative refinement to improve roundoff
        # errors described in [2]_, algorithm 5.1.
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # v = P inv(R) Q.T x
            aux1 = Q.T.dot(z)
            aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
            v[P] = aux2
            # z_next = z - A.T v
            z = z - A.T.dot(v)
            k += 1

        return z

    # z = inv(A A.T) A x
    def least_squares(x):
        # z = P inv(R) Q.T x
        aux1 = Q.T.dot(x)
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
        z = np.zeros(m)
        z[P] = aux2
        return z

    # z = A.T inv(A A.T) x
    def row_space(x):
        # z = Q inv(R.T) P.T x
        aux1 = x[P]
        aux2 = scipy.linalg.solve_triangular(R, aux1,
                                             lower=False,
                                             trans='T')
        z = Q.dot(aux2)
        return z

    return null_space, least_squares, row_space


def svd_factorization_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``SVDFactorization`` approach.
    """
    # SVD Factorization
    U, s, Vt = scipy.linalg.svd(A, full_matrices=False)

    # Remove dimensions related with very small singular values
    U = U[:, s > tol]
    Vt = Vt[s > tol, :]
    s = s[s > tol]

    # z = x - A.T inv(A A.T) A x
    def null_space(x):
        # v = U 1/s V.T x = inv(A A.T) A x
        aux1 = Vt.dot(x)
        aux2 = 1/s*aux1
        v = U.dot(aux2)
        z = x - A.T.dot(v)

        # Iterative refinement to improve roundoff
        # errors described in [2]_, algorithm 5.1.
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # v = U 1/s V.T x = inv(A A.T) A x
            aux1 = Vt.dot(z)
            aux2 = 1/s*aux1
            v = U.dot(aux2)
            # z_next = z - A.T v
            z = z - A.T.dot(v)
            k += 1

        return z

    # z = inv(A A.T) A x
    def least_squares(x):
        # z = U 1/s V.T x = inv(A A.T) A x
        aux1 = Vt.dot(x)
        aux2 = 1/s*aux1
        z = U.dot(aux2)
        return z

    # z = A.T inv(A A.T) x
    def row_space(x):
        # z = V 1/s U.T x
        aux1 = U.T.dot(x)
        aux2 = 1/s*aux1
        z = Vt.T.dot(aux2)
        return z

    return null_space, least_squares, row_space


def projections(A, method=None, orth_tol=1e-12, max_refin=3, tol=1e-15):
    """Return three linear operators related with a given matrix A.

    Parameters
    ----------
    A : sparse matrix (or ndarray), shape (m, n)
        Matrix ``A`` used in the projection.
    method : string, optional
        Method used for compute the given linear
        operators. Should be one of:

            - 'NormalEquation': The operators
               will be computed using the
               so-called normal equation approach
               explained in [1]_. In order to do
               so the Cholesky factorization of
               ``(A A.T)`` is computed. Exclusive
               for sparse matrices.
            - 'AugmentedSystem': The operators
               will be computed using the
               so-called augmented system approach
               explained in [1]_. Exclusive
               for sparse matrices.
            - 'QRFactorization': Compute projections
               using QR factorization. Exclusive for
               dense matrices.
            - 'SVDFactorization': Compute projections
               using SVD factorization. Exclusive for
               dense matrices.

    orth_tol : float, optional
        Tolerance for iterative refinements.
    max_refin : int, optional
        Maximum number of iterative refinements.
    tol : float, optional
        Tolerance for singular values.

    Returns
    -------
    Z : LinearOperator, shape (n, n)
        Null-space operator. For a given vector ``x``,
        the null space operator is equivalent to apply
        a projection matrix ``P = I - A.T inv(A A.T) A``
        to the vector. It can be shown that this is
        equivalent to project ``x`` into the null space
        of A.
    LS : LinearOperator, shape (m, n)
        Least-squares operator. For a given vector ``x``,
        the least-squares operator is equivalent to apply a
        pseudoinverse matrix ``pinv(A.T) = inv(A A.T) A``
        to the vector. It can be shown that this vector
        ``pinv(A.T) x`` is the least_square solution to
        ``A.T y = x``.
    Y : LinearOperator, shape (n, m)
        Row-space operator. For a given vector ``x``,
        the row-space operator is equivalent to apply a
        projection matrix ``Q = A.T inv(A A.T)``
        to the vector.  It can be shown that this
        vector ``y = Q x``  the minimum norm solution
        of ``A y = x``.

    Notes
    -----
    Uses iterative refinements described in [1]
    during the computation of ``Z`` in order to
    cope with the possibility of large roundoff errors.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
        "On the solution of equality constrained quadratic
        programming problems arising in optimization."
        SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    m, n = np.shape(A)

    # The factorization of an empty matrix
    # only works for the sparse representation.
    if m*n == 0:
        A = csc_matrix(A)

    # Check Argument
    if issparse(A):
        if method is None:
            method = "AugmentedSystem"
        if method not in ("NormalEquation", "AugmentedSystem"):
            raise ValueError("Method not allowed for sparse matrix.")
        if method == "NormalEquation" and not sksparse_available:
            warnings.warn("Only accepts 'NormalEquation' option when "
                          "scikit-sparse is available. Using "
                          "'AugmentedSystem' option instead.",
                          ImportWarning, stacklevel=3)
            method = 'AugmentedSystem'
    else:
        if method is None:
            method = "QRFactorization"
        if method not in ("QRFactorization", "SVDFactorization"):
            raise ValueError("Method not allowed for dense array.")

    if method == 'NormalEquation':
        null_space, least_squares, row_space \
            = normal_equation_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == 'AugmentedSystem':
        null_space, least_squares, row_space \
            = augmented_system_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == "QRFactorization":
        null_space, least_squares, row_space \
            = qr_factorization_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == "SVDFactorization":
        null_space, least_squares, row_space \
            = svd_factorization_projections(A, m, n, orth_tol, max_refin, tol)

    Z = LinearOperator((n, n), null_space)
    LS = LinearOperator((m, n), least_squares)
    Y = LinearOperator((n, m), row_space)

    return Z, LS, Y
<|endoftext|>"""
Cython optimize root finding API
================================
The underlying C functions for the following root finders can be accessed
directly using Cython:

- `~scipy.optimize.bisect`
- `~scipy.optimize.ridder`
- `~scipy.optimize.brenth`
- `~scipy.optimize.brentq`

The Cython API for the root finding functions is similar except there is no
``disp`` argument. Import the root finding functions using ``cimport`` from
`scipy.optimize.cython_optimize`. ::

    from scipy.optimize.cython_optimize cimport bisect, ridder, brentq, brenth


Callback signature
------------------
The zeros functions in `~scipy.optimize.cython_optimize` expect a callback that
takes a double for the scalar independent variable as the 1st argument and a
user defined ``struct`` with any extra parameters as the 2nd argument. ::

    double (*callback_type)(double, void*) noexcept


Examples
--------
Usage of `~scipy.optimize.cython_optimize` requires Cython to write callbacks
that are compiled into C. For more information on compiling Cython, see the
`Cython Documentation <http://docs.cython.org/en/latest/index.html>`_.

These are the basic steps:

1. Create a Cython ``.pyx`` file, for example: ``myexample.pyx``.
2. Import the desired root finder from `~scipy.optimize.cython_optimize`.
3. Write the callback function, and call the selected root finding function
   passing the callback, any extra arguments, and the other solver
   parameters. ::

       from scipy.optimize.cython_optimize cimport brentq

       # import math from Cython
       from libc cimport math

       myargs = {'C0': 1.0, 'C1': 0.7}  # a dictionary of extra arguments
       XLO, XHI = 0.5, 1.0  # lower and upper search boundaries
       XTOL, RTOL, MITR = 1e-3, 1e-3, 10  # other solver parameters

       # user-defined struct for extra parameters
       ctypedef struct test_params:
           double C0
           double C1


       # user-defined callback
       cdef double f(double x, void *args) noexcept:
           cdef test_params *myargs = <test_params *> args
           return myargs.C0 - math.exp(-(x - myargs.C1))


       # Cython wrapper function
       cdef double brentq_wrapper_example(dict args, double xa, double xb,
                                          double xtol, double rtol, int mitr):
           # Cython automatically casts dictionary to struct
           cdef test_params myargs = args
           return brentq(
               f, xa, xb, <test_params *> &myargs, xtol, rtol, mitr, NULL)


       # Python function
       def brentq_example(args=myargs, xa=XLO, xb=XHI, xtol=XTOL, rtol=RTOL,
                          mitr=MITR):
           '''Calls Cython wrapper from Python.'''
           return brentq_wrapper_example(args, xa, xb, xtol, rtol, mitr)

4. If you want to call your function from Python, create a Cython wrapper, and
   a Python function that calls the wrapper, or use ``cpdef``. Then, in Python,
   you can import and run the example. ::

       from myexample import brentq_example

       x = brentq_example()
       # 0.6999942848231314

5. Create a Cython ``.pxd`` file if you need to export any Cython functions.


Full output
-----------
The  functions in `~scipy.optimize.cython_optimize` can also copy the full
output from the solver to a C ``struct`` that is passed as its last argument.
If you don't want the full output, just pass ``NULL``. The full output
``struct`` must be type ``zeros_full_output``, which is defined in
`scipy.optimize.cython_optimize` with the following fields:

- ``int funcalls``: number of function calls
- ``int iterations``: number of iterations
- ``int error_num``: error number
- ``double root``: root of function

The root is copied by `~scipy.optimize.cython_optimize` to the full output
``struct``. An error number of -1 means a sign error, -2 means a convergence
error, and 0 means the solver converged. Continuing from the previous example::

    from scipy.optimize.cython_optimize cimport zeros_full_output


    # cython brentq solver with full output
    cdef zeros_full_output brentq_full_output_wrapper_example(
            dict args, double xa, double xb, double xtol, double rtol,
            int mitr):
        cdef test_params myargs = args
        cdef zeros_full_output my_full_output
        # use my_full_output instead of NULL
        brentq(f, xa, xb, &myargs, xtol, rtol, mitr, &my_full_output)
        return my_full_output


    # Python function
    def brent_full_output_example(args=myargs, xa=XLO, xb=XHI, xtol=XTOL,
                                  rtol=RTOL, mitr=MITR):
        '''Returns full output'''
        return brentq_full_output_wrapper_example(args, xa, xb, xtol, rtol,
                                                  mitr)

    result = brent_full_output_example()
    # {'error_num': 0,
    #  'funcalls': 6,
    #  'iterations': 5,
    #  'root': 0.6999942848231314}
"""
<|endoftext|>"""The adaptation of Trust Region Reflective algorithm for a linear
least-squares problem."""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult

from .givens_elimination import givens_elimination
from .common import (
    EPS, step_size_to_bound, find_active_constraints, in_bounds,
    make_strictly_feasible, build_quadratic_1d, evaluate_quadratic,
    minimize_quadratic_1d, CL_scaling_vector, reflective_transformation,
    print_header_linear, print_iteration_linear, compute_grad,
    regularized_lsq_operator, right_multiplied_operator)


def regularized_lsq_with_qr(m, n, R, QTb, perm, diag, copy_R=True):
    """Solve regularized least squares using information from QR-decomposition.

    The initial problem is to solve the following system in a least-squares
    sense::

        A x = b
        D x = 0

    where D is diagonal matrix. The method is based on QR decomposition
    of the form A P = Q R, where P is a column permutation matrix, Q is an
    orthogonal matrix and R is an upper triangular matrix.

    Parameters
    ----------
    m, n : int
        Initial shape of A.
    R : ndarray, shape (n, n)
        Upper triangular matrix from QR decomposition of A.
    QTb : ndarray, shape (n,)
        First n components of Q^T b.
    perm : ndarray, shape (n,)
        Array defining column permutation of A, such that ith column of
        P is perm[i]-th column of identity matrix.
    diag : ndarray, shape (n,)
        Array containing diagonal elements of D.

    Returns
    -------
    x : ndarray, shape (n,)
        Found least-squares solution.
    """
    if copy_R:
        R = R.copy()
    v = QTb.copy()

    givens_elimination(R, v, diag[perm])

    abs_diag_R = np.abs(np.diag(R))
    threshold = EPS * max(m, n) * np.max(abs_diag_R)
    nns, = np.nonzero(abs_diag_R > threshold)

    R = R[np.ix_(nns, nns)]
    v = v[nns]

    x = np.zeros(n)
    x[perm[nns]] = solve_triangular(R, v)

    return x


def backtracking(A, g, x, p, theta, p_dot_g, lb, ub):
    """Find an appropriate step size using backtracking line search."""
    alpha = 1
    while True:
        x_new, _ = reflective_transformation(x + alpha * p, lb, ub)
        step = x_new - x
        cost_change = -evaluate_quadratic(A, g, step)
        if cost_change > -0.1 * alpha * p_dot_g:
            break
        alpha *= 0.5

    active = find_active_constraints(x_new, lb, ub)
    if np.any(active != 0):
        x_new, _ = reflective_transformation(x + theta * alpha * p, lb, ub)
        x_new = make_strictly_feasible(x_new, lb, ub, rstep=0)
        step = x_new - x
        cost_change = -evaluate_quadratic(A, g, step)

    return x, step, cost_change


def select_step(x, A_h, g_h, c_h, p, p_h, d, lb, ub, theta):
    """Select the best step according to Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):
        return p

    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict step, such that it hits the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Find the step size along reflected direction.
    r_stride_u, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # Stay interior.
    r_stride_l = (1 - theta) * r_stride_u
    r_stride_u *= theta

    if r_stride_u > 0:
        a, b, c = build_quadratic_1d(A_h, g_h, r_h, s0=p_h, diag=c_h)
        r_stride, r_value = minimize_quadratic_1d(
            a, b, r_stride_l, r_stride_u, c=c)
        r_h = p_h + r_h * r_stride
        r = d * r_h
    else:
        r_value = np.inf

    # Now correct p_h to make it strictly interior.
    p_h *= theta
    p *= theta
    p_value = evaluate_quadratic(A_h, g_h, p_h, diag=c_h)

    ag_h = -g_h
    ag = d * ag_h
    ag_stride_u, _ = step_size_to_bound(x, ag, lb, ub)
    ag_stride_u *= theta
    a, b = build_quadratic_1d(A_h, g_h, ag_h, diag=c_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride_u)
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p
    elif r_value < p_value and r_value < ag_value:
        return r
    else:
        return ag


def trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol,
               max_iter, verbose, *, lsmr_maxiter=None):
    m, n = A.shape
    x, _ = reflective_transformation(x_lsq, lb, ub)
    x = make_strictly_feasible(x, lb, ub, rstep=0.1)

    if lsq_solver == 'exact':
        QT, R, perm = qr(A, mode='economic', pivoting=True)
        QT = QT.T

        if m < n:
            R = np.vstack((R, np.zeros((n - m, n))))

        QTr = np.zeros(n)
        k = min(m, n)
    elif lsq_solver == 'lsmr':
        r_aug = np.zeros(m + n)
        auto_lsmr_tol = False
        if lsmr_tol is None:
            lsmr_tol = 1e-2 * tol
        elif lsmr_tol == 'auto':
            auto_lsmr_tol = True

    r = A.dot(x) - b
    g = compute_grad(A, r)
    cost = 0.5 * np.dot(r, r)
    initial_cost = cost

    termination_status = None
    step_norm = None
    cost_change = None

    if max_iter is None:
        max_iter = 100

    if verbose == 2:
        print_header_linear()

    for iteration in range(max_iter):
        v, dv = CL_scaling_vector(x, g, lb, ub)
        g_scaled = g * v
        g_norm = norm(g_scaled, ord=np.inf)
        if g_norm < tol:
            termination_status = 1

        if verbose == 2:
            print_iteration_linear(iteration, cost, cost_change,
                                   step_norm, g_norm)

        if termination_status is not None:
            break

        diag_h = g * dv
        diag_root_h = diag_h ** 0.5
        d = v ** 0.5
        g_h = d * g

        A_h = right_multiplied_operator(A, d)
        if lsq_solver == 'exact':
            QTr[:k] = QT.dot(r)
            p_h = -regularized_lsq_with_qr(m, n, R * d[perm], QTr, perm,
                                           diag_root_h, copy_R=False)
        elif lsq_solver == 'lsmr':
            lsmr_op = regularized_lsq_operator(A_h, diag_root_h)
            r_aug[:m] = r
            if auto_lsmr_tol:
                eta = 1e-2 * min(0.5, g_norm)
                lsmr_tol = max(EPS, min(0.1, eta * g_norm))
            p_h = -lsmr(lsmr_op, r_aug, maxiter=lsmr_maxiter,
                        atol=lsmr_tol, btol=lsmr_tol)[0]

        p = d * p_h

        p_dot_g = np.dot(p, g)
        if p_dot_g > 0:
            termination_status = -1

        theta = 1 - min(0.005, g_norm)
        step = select_step(x, A_h, g_h, diag_h, p, p_h, d, lb, ub, theta)
        cost_change = -evaluate_quadratic(A, g, step)

        # Perhaps almost never executed, the idea is that `p` is descent
        # direction thus we must find acceptable cost decrease using simple
        # "backtracking", otherwise the algorithm's logic would break.
        if cost_change < 0:
            x, step, cost_change = backtracking(
                A, g, x, p, theta, p_dot_g, lb, ub)
        else:
            x = make_strictly_feasible(x + step, lb, ub, rstep=0)

        step_norm = norm(step)
        r = A.dot(x) - b
        g = compute_grad(A, r)

        if cost_change < tol * cost:
            termination_status = 2

        cost = 0.5 * np.dot(r, r)

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=tol)

    return OptimizeResult(
        x=x, fun=r, cost=cost, optimality=g_norm, active_mask=active_mask,
        nit=iteration + 1, status=termination_status,
        initial_cost=initial_cost)
<|endoftext|>"""Linear least squares with bound constraints on independent variables."""
import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.optimize import OptimizeResult
from scipy.optimize._minimize import Bounds

from .common import in_bounds, compute_grad
from .trf_linear import trf_linear
from .bvls import bvls


def prepare_bounds(bounds, n):
    if len(bounds) != 2:
        raise ValueError("`bounds` must contain 2 elements.")
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)

    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


TERMINATION_MESSAGES = {
    -1: "The algorithm was not able to make progress on the last iteration.",
    0: "The maximum number of iterations is exceeded.",
    1: "The first-order optimality measure is less than `tol`.",
    2: "The relative change of the cost function is less than `tol`.",
    3: "The unconstrained solution is optimal."
}


def lsq_linear(A, b, bounds=(-np.inf, np.inf), method='trf', tol=1e-10,
               lsq_solver=None, lsmr_tol=None, max_iter=None,
               verbose=0, *, lsmr_maxiter=None,):
    r"""Solve a linear least-squares problem with bounds on the variables.

    Given a m-by-n design matrix A and a target vector b with m elements,
    `lsq_linear` solves the following optimization problem::

        minimize 0.5 * ||A x - b||**2
        subject to lb <= x <= ub

    This optimization problem is convex, hence a found minimum (if iterations
    have converged) is guaranteed to be global.

    Parameters
    ----------
    A : array_like, sparse matrix of LinearOperator, shape (m, n)
        Design matrix. Can be `scipy.sparse.linalg.LinearOperator`.
    b : array_like, shape (m,)
        Target vector.
    bounds : 2-tuple of array_like or `Bounds`, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        There are two ways to specify the bounds:

        - Instance of `Bounds` class.
        - 2-tuple of array_like: Each element of the tuple must be either
          an array with the length equal to the number of parameters, or a
          scalar (in which case the bound is taken to be the same for all
          parameters). Use ``np.inf`` with an appropriate sign to disable
          bounds on all or some parameters.

    method : 'trf' or 'bvls', optional
        Method to perform minimization.

        * 'trf' : Trust Region Reflective algorithm adapted for a linear
          least-squares problem. This is an interior-point-like method
          and the required number of iterations is weakly correlated with
          the number of variables.
        * 'bvls' : Bounded-variable least-squares algorithm. This is
          an active set method, which requires the number of iterations
          comparable to the number of variables. Can't be used when `A` is
          sparse or LinearOperator.

        Default is 'trf'.
    tol : float, optional
        Tolerance parameter. The algorithm terminates if a relative change
        of the cost function is less than `tol` on the last iteration.
        Additionally, the first-order optimality measure is considered:

        * ``method='trf'`` terminates if the uniform norm of the gradient,
          scaled to account for the presence of the bounds, is less than
          `tol`.
        * ``method='bvls'`` terminates if Karush-Kuhn-Tucker conditions
          are satisfied within `tol` tolerance.

    lsq_solver : {None, 'exact', 'lsmr'}, optional
        Method of solving unbounded least-squares problems throughout
        iterations:

        * 'exact' : Use dense QR or SVD decomposition approach. Can't be
          used when `A` is sparse or LinearOperator.
        * 'lsmr' : Use `scipy.sparse.linalg.lsmr` iterative procedure
          which requires only matrix-vector product evaluations. Can't
          be used with ``method='bvls'``.

        If None (default), the solver is chosen based on type of `A`.
    lsmr_tol : None, float or 'auto', optional
        Tolerance parameters 'atol' and 'btol' for `scipy.sparse.linalg.lsmr`
        If None (default), it is set to ``1e-2 * tol``. If 'auto', the
        tolerance will be adjusted based on the optimality of the current
        iterate, which can speed up the optimization process, but is not always
        reliable.
    max_iter : None or int, optional
        Maximum number of iterations before termination. If None (default), it
        is set to 100 for ``method='trf'`` or to the number of variables for
        ``method='bvls'`` (not counting iterations for 'bvls' initialization).
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

        * 0 : work silently (default).
        * 1 : display a termination report.
        * 2 : display progress during iterations.

    lsmr_maxiter : None or int, optional
        Maximum number of iterations for the lsmr least squares solver,
        if it is used (by setting ``lsq_solver='lsmr'``). If None (default), it
        uses lsmr's default of ``min(m, n)`` where ``m`` and ``n`` are the
        number of rows and columns of `A`, respectively. Has no effect if
        ``lsq_solver='exact'``.

    Returns
    -------
    OptimizeResult with the following fields defined:
    x : ndarray, shape (n,)
        Solution found.
    cost : float
        Value of the cost function at the solution.
    fun : ndarray, shape (m,)
        Vector of residuals at the solution.
    optimality : float
        First-order optimality measure. The exact meaning depends on `method`,
        refer to the description of `tol` parameter.
    active_mask : ndarray of int, shape (n,)
        Each component shows whether a corresponding constraint is active
        (that is, whether a variable is at the bound):

        *  0 : a constraint is not active.
        * -1 : a lower bound is active.
        *  1 : an upper bound is active.

        Might be somewhat arbitrary for the `trf` method as it generates a
        sequence of strictly feasible iterates and active_mask is determined
        within a tolerance threshold.
    unbounded_sol : tuple
        Unbounded least squares solution tuple returned by the least squares
        solver (set with `lsq_solver` option). If `lsq_solver` is not set or is
        set to ``'exact'``, the tuple contains an ndarray of shape (n,) with
        the unbounded solution, an ndarray with the sum of squared residuals,
        an int with the rank of `A`, and an ndarray with the singular values
        of `A` (see NumPy's ``linalg.lstsq`` for more information). If
        `lsq_solver` is set to ``'lsmr'``, the tuple contains an ndarray of
        shape (n,) with the unbounded solution, an int with the exit code,
        an int with the number of iterations, and five floats with
        various norms and the condition number of `A` (see SciPy's
        ``sparse.linalg.lsmr`` for more information). This output can be
        useful for determining the convergence of the least squares solver,
        particularly the iterative ``'lsmr'`` solver. The unbounded least
        squares problem is to minimize ``0.5 * ||A x - b||**2``.
    nit : int
        Number of iterations. Zero if the unconstrained solution is optimal.
    status : int
        Reason for algorithm termination:

        * -1 : the algorithm was not able to make progress on the last
          iteration.
        *  0 : the maximum number of iterations is exceeded.
        *  1 : the first-order optimality measure is less than `tol`.
        *  2 : the relative change of the cost function is less than `tol`.
        *  3 : the unconstrained solution is optimal.

    message : str
        Verbal description of the termination reason.
    success : bool
        True if one of the convergence criteria is satisfied (`status` > 0).

    See Also
    --------
    nnls : Linear least squares with non-negativity constraint.
    least_squares : Nonlinear least squares with bounds on the variables.

    Notes
    -----
    The algorithm first computes the unconstrained least-squares solution by
    `numpy.linalg.lstsq` or `scipy.sparse.linalg.lsmr` depending on
    `lsq_solver`. This solution is returned as optimal if it lies within the
    bounds.

    Method 'trf' runs the adaptation of the algorithm described in [STIR]_ for
    a linear least-squares problem. The iterations are essentially the same as
    in the nonlinear least-squares algorithm, but as the quadratic function
    model is always accurate, we don't need to track or modify the radius of
    a trust region. The line search (backtracking) is used as a safety net
    when a selected step does not decrease the cost function. Read more
    detailed description of the algorithm in `scipy.optimize.least_squares`.

    Method 'bvls' runs a Python implementation of the algorithm described in
    [BVLS]_. The algorithm maintains active and free sets of variables, on
    each iteration chooses a new variable to move from the active set to the
    free set and then solves the unconstrained least-squares problem on free
    variables. This algorithm is guaranteed to give an accurate solution
    eventually, but may require up to n iterations for a problem with n
    variables. Additionally, an ad-hoc initialization procedure is
    implemented, that determines which variables to set free or active
    initially. It takes some number of iterations before actual BVLS starts,
    but can significantly reduce the number of further iterations.

    References
    ----------
    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
              and Conjugate Gradient Method for Large-Scale Bound-Constrained
              Minimization Problems," SIAM Journal on Scientific Computing,
              Vol. 21, Number 1, pp 1-23, 1999.
    .. [BVLS] P. B. Start and R. L. Parker, "Bounded-Variable Least-Squares:
              an Algorithm and Applications", Computational Statistics, 10,
              129-141, 1995.

    Examples
    --------
    In this example, a problem with a large sparse matrix and bounds on the
    variables is solved.

    >>> import numpy as np
    >>> from scipy.sparse import rand
    >>> from scipy.optimize import lsq_linear
    >>> rng = np.random.default_rng()
    ...
    >>> m = 2000
    >>> n = 1000
    ...
    >>> A = rand(m, n, density=1e-4, random_state=rng)
    >>> b = rng.standard_normal(m)
    ...
    >>> lb = rng.standard_normal(n)
    >>> ub = lb + 1
    ...
    >>> res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)
    The relative change of the cost function is less than `tol`.
    Number of iterations 10, initial cost 1.0070e+03, final cost 9.6602e+02,
    first-order optimality 2.21e-09.        # may vary
    """
    if method not in ['trf', 'bvls']:
        raise ValueError("`method` must be 'trf' or 'bvls'")

    if lsq_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`solver` must be None, 'exact' or 'lsmr'.")

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    if issparse(A):
        A = csr_matrix(A)
    elif not isinstance(A, LinearOperator):
        A = np.atleast_2d(np.asarray(A))

    if method == 'bvls':
        if lsq_solver == 'lsmr':
            raise ValueError("method='bvls' can't be used with "
                             "lsq_solver='lsmr'")

        if not isinstance(A, np.ndarray):
            raise ValueError("method='bvls' can't be used with `A` being "
                             "sparse or LinearOperator.")

    if lsq_solver is None:
        if isinstance(A, np.ndarray):
            lsq_solver = 'exact'
        else:
            lsq_solver = 'lsmr'
    elif lsq_solver == 'exact' and not isinstance(A, np.ndarray):
        raise ValueError("`exact` solver can't be used when `A` is "
                         "sparse or LinearOperator.")

    if len(A.shape) != 2:  # No ndim for LinearOperator.
        raise ValueError("`A` must have at most 2 dimensions.")

    if max_iter is not None and max_iter <= 0:
        raise ValueError("`max_iter` must be None or positive integer.")

    m, n = A.shape

    b = np.atleast_1d(b)
    if b.ndim != 1:
        raise ValueError("`b` must have at most 1 dimension.")

    if b.size != m:
        raise ValueError("Inconsistent shapes between `A` and `b`.")

    if isinstance(bounds, Bounds):
        lb = bounds.lb
        ub = bounds.ub
    else:
        lb, ub = prepare_bounds(bounds, n)

    if lb.shape != (n,) and ub.shape != (n,):
        raise ValueError("Bounds have wrong shape.")

    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")

    if lsmr_maxiter is not None and lsmr_maxiter < 1:
        raise ValueError("`lsmr_maxiter` must be None or positive integer.")

    if not ((isinstance(lsmr_tol, float) and lsmr_tol > 0) or
            lsmr_tol in ('auto', None)):
        raise ValueError("`lsmr_tol` must be None, 'auto', or positive float.")

    if lsq_solver == 'exact':
        unbd_lsq = np.linalg.lstsq(A, b, rcond=-1)
    elif lsq_solver == 'lsmr':
        first_lsmr_tol = lsmr_tol  # tol of first call to lsmr
        if lsmr_tol is None or lsmr_tol == 'auto':
            first_lsmr_tol = 1e-2 * tol  # default if lsmr_tol not defined
        unbd_lsq = lsmr(A, b, maxiter=lsmr_maxiter,
                        atol=first_lsmr_tol, btol=first_lsmr_tol)
    x_lsq = unbd_lsq[0]  # extract the solution from the least squares solver

    if in_bounds(x_lsq, lb, ub):
        r = A @ x_lsq - b
        cost = 0.5 * np.dot(r, r)
        termination_status = 3
        termination_message = TERMINATION_MESSAGES[termination_status]
        g = compute_grad(A, r)
        g_norm = norm(g, ord=np.inf)

        if verbose > 0:
            print(termination_message)
            print(f"Final cost {cost:.4e}, first-order optimality {g_norm:.2e}")

        return OptimizeResult(
            x=x_lsq, fun=r, cost=cost, optimality=g_norm,
            active_mask=np.zeros(n), unbounded_sol=unbd_lsq,
            nit=0, status=termination_status,
            message=termination_message, success=True)

    if method == 'trf':
        res = trf_linear(A, b, x_lsq, lb, ub, tol, lsq_solver, lsmr_tol,
                         max_iter, verbose, lsmr_maxiter=lsmr_maxiter)
    elif method == 'bvls':
        res = bvls(A, b, x_lsq, lb, ub, tol, max_iter, verbose)

    res.unbounded_sol = unbd_lsq
    res.message = TERMINATION_MESSAGES[res.status]
    res.success = res.status > 0

    if verbose > 0:
        print(res.message)
        print(
            f"Number of iterations {res.nit}, initial cost {res.initial_cost:.4e}, "
            f"final cost {res.cost:.4e}, first-order optimality {res.optimality:.2e}."
        )

    del res.initial_cost

    return res
<|endoftext|>"""Functions used by least-squares algorithms."""
from math import copysign

import numpy as np
from numpy.linalg import norm

from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator


EPS = np.finfo(float).eps


# Functions related to a trust-region problem.


def intersect_trust_region(x, s, Delta):
    """Find the intersection of a line with the boundary of a trust region.

    This function solves the quadratic equation with respect to t
    ||(x + s*t)||**2 = Delta**2.

    Returns
    -------
    t_neg, t_pos : tuple of float
        Negative and positive roots.

    Raises
    ------
    ValueError
        If `s` is zero or `x` is not within the trust region.
    """
    a = np.dot(s, s)
    if a == 0:
        raise ValueError("`s` is zero.")

    b = np.dot(x, s)

    c = np.dot(x, x) - Delta**2
    if c > 0:
        raise ValueError("`x` is not within the trust region.")

    d = np.sqrt(b*b - a*c)  # Root from one fourth of the discriminant.

    # Computations below avoid loss of significance, see "Numerical Recipes".
    q = -(b + copysign(d, b))
    t1 = q / a
    t2 = c / q

    if t1 < t2:
        return t1, t2
    else:
        return t2, t1


def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None,
                           rtol=0.01, max_iter=10):
    """Solve a trust-region problem arising in least-squares minimization.

    This function implements a method described by J. J. More [1]_ and used
    in MINPACK, but it relies on a single SVD of Jacobian instead of series
    of Cholesky decompositions. Before running this function, compute:
    ``U, s, VT = svd(J, full_matrices=False)``.

    Parameters
    ----------
    n : int
        Number of variables.
    m : int
        Number of residuals.
    uf : ndarray
        Computed as U.T.dot(f).
    s : ndarray
        Singular values of J.
    V : ndarray
        Transpose of VT.
    Delta : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy ``abs(norm(p) - Delta) < rtol * Delta``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.

    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    alpha : float
        Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
        Sometimes called Levenberg-Marquardt parameter.
    n_iter : int
        Number of iterations made by root-finding procedure. Zero means
        that Gauss-Newton step was selected as the solution.

    References
    ----------
    .. [1] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
           and Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes
           in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    """
    def phi_and_derivative(alpha, suf, s, Delta):
        """Function of which to find zero.

        It is defined as "norm of regularized (by alpha) least-squares
        solution minus `Delta`". Refer to [1]_.
        """
        denom = s**2 + alpha
        p_norm = norm(suf / denom)
        phi = p_norm - Delta
        phi_prime = -np.sum(suf ** 2 / denom**3) / p_norm
        return phi, phi_prime

    suf = s * uf

    # Check if J has full rank and try Gauss-Newton step.
    if m >= n:
        threshold = EPS * m * s[0]
        full_rank = s[-1] > threshold
    else:
        full_rank = False

    if full_rank:
        p = -V.dot(uf / s)
        if norm(p) <= Delta:
            return p, 0.0, 0

    alpha_upper = norm(suf) / Delta

    if full_rank:
        phi, phi_prime = phi_and_derivative(0.0, suf, s, Delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0

    if initial_alpha is None or not full_rank and initial_alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)
    else:
        alpha = initial_alpha

    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)

        phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)

        if phi < 0:
            alpha_upper = alpha

        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta

        if np.abs(phi) < rtol * Delta:
            break

    p = -V.dot(suf / (s**2 + alpha))

    # Make the norm of p equal to Delta, p is changed only slightly during
    # this. It is done to prevent p lie outside the trust region (which can
    # cause problems later).
    p *= Delta / norm(p)

    return p, alpha, it + 1


def solve_trust_region_2d(B, g, Delta):
    """Solve a general trust-region problem in 2 dimensions.

    The problem is reformulated as a 4th order algebraic equation,
    the solution of which is found by numpy.roots.

    Parameters
    ----------
    B : ndarray, shape (2, 2)
        Symmetric matrix, defines a quadratic term of the function.
    g : ndarray, shape (2,)
        Defines a linear term of the function.
    Delta : float
        Radius of a trust region.

    Returns
    -------
    p : ndarray, shape (2,)
        Found solution.
    newton_step : bool
        Whether the returned solution is the Newton step which lies within
        the trust region.
    """
    try:
        R, lower = cho_factor(B)
        p = -cho_solve((R, lower), g)
        if np.dot(p, p) <= Delta**2:
            return p, True
    except LinAlgError:
        pass

    a = B[0, 0] * Delta**2
    b = B[0, 1] * Delta**2
    c = B[1, 1] * Delta**2

    d = g[0] * Delta
    f = g[1] * Delta

    coeffs = np.array(
        [-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = np.roots(coeffs)  # Can handle leading zeros.
    t = np.real(t[np.isreal(t)])

    p = Delta * np.vstack((2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)))
    value = 0.5 * np.sum(p * B.dot(p), axis=0) + np.dot(g, p)
    i = np.argmin(value)
    p = p[:, i]

    return p, False


def update_tr_radius(Delta, actual_reduction, predicted_reduction,
                     step_norm, bound_hit):
    """Update the radius of a trust region based on the cost reduction.

    Returns
    -------
    Delta : float
        New radius.
    ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0

    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        Delta *= 2.0

    return Delta, ratio


# Construction and minimization of quadratic functions.


def build_quadratic_1d(J, g, s, diag=None, s0=None):
    """Parameterize a multivariate quadratic function along a line.

    The resulting univariate quadratic function is given as follows::

        f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) +
               g.T * (s0 + s*t)

    Parameters
    ----------
    J : ndarray, sparse matrix or LinearOperator shape (m, n)
        Jacobian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (n,)
        Direction vector of a line.
    diag : None or ndarray with shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.
    s0 : None or ndarray with shape (n,), optional
        Initial point. If None, assumed to be 0.

    Returns
    -------
    a : float
        Coefficient for t**2.
    b : float
        Coefficient for t.
    c : float
        Free term. Returned only if `s0` is provided.
    """
    v = J.dot(s)
    a = np.dot(v, v)
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5

    b = np.dot(g, s)

    if s0 is not None:
        u = J.dot(s0)
        b += np.dot(u, v)
        c = 0.5 * np.dot(u, u) + np.dot(g, s0)
        if diag is not None:
            b += np.dot(s0 * diag, s)
            c += 0.5 * np.dot(s0 * diag, s0)
        return a, b, c
    else:
        return a, b


def minimize_quadratic_1d(a, b, lb, ub, c=0):
    """Minimize a 1-D quadratic function subject to bounds.

    The free term `c` is 0 by default. Bounds must be finite.

    Returns
    -------
    t : float
        Minimum point.
    y : float
        Minimum value.
    """
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = np.asarray(t)
    y = t * (a * t + b) + c
    min_index = np.argmin(y)
    return t[min_index], y[min_index]


def evaluate_quadratic(J, g, s, diag=None):
    """Compute values of a quadratic function arising in least squares.

    The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.

    Parameters
    ----------
    J : ndarray, sparse matrix or LinearOperator, shape (m, n)
        Jacobian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (k, n) or (n,)
        Array containing steps as rows.
    diag : ndarray, shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.

    Returns
    -------
    values : ndarray with shape (k,) or float
        Values of the function. If `s` was 2-D, then ndarray is
        returned, otherwise, float is returned.
    """
    if s.ndim == 1:
        Js = J.dot(s)
        q = np.dot(Js, Js)
        if diag is not None:
            q += np.dot(s * diag, s)
    else:
        Js = J.dot(s.T)
        q = np.sum(Js**2, axis=0)
        if diag is not None:
            q += np.sum(diag * s**2, axis=1)

    l = np.dot(s, g)

    return 0.5 * q + l


# Utility functions to work with bound constraints.


def in_bounds(x, lb, ub):
    """Check if a point lies within bounds."""
    return np.all((x >= lb) & (x <= ub))


def step_size_to_bound(x, s, lb, ub):
    """Compute a min_step size required to reach a bound.

    The function computes a positive scalar t, such that x + s * t is on
    the bound.

    Returns
    -------
    step : float
        Computed step. Non-negative value.
    hits : ndarray of int with shape of x
        Each element indicates whether a corresponding variable reaches the
        bound:

             *  0 - the bound was not hit.
             * -1 - the lower bound was hit.
             *  1 - the upper bound was hit.
    """
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    with np.errstate(over='ignore'):
        steps[non_zero] = np.maximum((lb - x)[non_zero] / s_non_zero,
                                     (ub - x)[non_zero] / s_non_zero)
    min_step = np.min(steps)
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def find_active_constraints(x, lb, ub, rtol=1e-10):
    """Determine which constraints are active in a given point.

    The threshold is computed using `rtol` and the absolute value of the
    closest bound.

    Returns
    -------
    active : ndarray of int with shape of x
        Each component shows whether the corresponding constraint is active:

             *  0 - a constraint is not active.
             * -1 - a lower bound is active.
             *  1 - a upper bound is active.
    """
    active = np.zeros_like(x, dtype=int)

    if rtol == 0:
        active[x <= lb] = -1
        active[x >= ub] = 1
        return active

    lower_dist = x - lb
    upper_dist = ub - x

    lower_threshold = rtol * np.maximum(1, np.abs(lb))
    upper_threshold = rtol * np.maximum(1, np.abs(ub))

    lower_active = (np.isfinite(lb) &
                    (lower_dist <= np.minimum(upper_dist, lower_threshold)))
    active[lower_active] = -1

    upper_active = (np.isfinite(ub) &
                    (upper_dist <= np.minimum(lower_dist, upper_threshold)))
    active[upper_active] = 1

    return active


def make_strictly_feasible(x, lb, ub, rstep=1e-10):
    """Shift a point to the interior of a feasible region.

    Each element of the returned vector is at least at a relative distance
    `rstep` from the closest bound. If ``rstep=0`` then `np.nextafter` is used.
    """
    x_new = x.copy()

    active = find_active_constraints(x, lb, ub, rstep)
    lower_mask = np.equal(active, -1)
    upper_mask = np.equal(active, 1)

    if rstep == 0:
        x_new[lower_mask] = np.nextafter(lb[lower_mask], ub[lower_mask])
        x_new[upper_mask] = np.nextafter(ub[upper_mask], lb[upper_mask])
    else:
        x_new[lower_mask] = (lb[lower_mask] +
                             rstep * np.maximum(1, np.abs(lb[lower_mask])))
        x_new[upper_mask] = (ub[upper_mask] -
                             rstep * np.maximum(1, np.abs(ub[upper_mask])))

    tight_bounds = (x_new < lb) | (x_new > ub)
    x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])

    return x_new


def CL_scaling_vector(x, g, lb, ub):
    """Compute Coleman-Li scaling vector and its derivatives.

    Components of a vector v are defined as follows::

               | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
        v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
               | 1,           otherwise

    According to this definition v[i] >= 0 for all i. It differs from the
    definition in paper [1]_ (eq. (2.2)), where the absolute value of v is
    used. Both definitions are equivalent down the line.
    Derivatives of v with respect to x take value 1, -1 or 0 depending on a
    case.

    Returns
    -------
    v : ndarray with shape of x
        Scaling vector.
    dv : ndarray with shape of x
        Derivatives of v[i] with respect to x[i], diagonal elements of v's
        Jacobian.

    References
    ----------
    .. [1] M.A. Branch, T.F. Coleman, and Y. Li, "A Subspace, Interior,
           and Conjugate Gradient Method for Large-Scale Bound-Constrained
           Minimization Problems," SIAM Journal on Scientific Computing,
           Vol. 21, Number 1, pp 1-23, 1999.
    """
    v = np.ones_like(x)
    dv = np.zeros_like(x)

    mask = (g < 0) & np.isfinite(ub)
    v[mask] = ub[mask] - x[mask]
    dv[mask] = -1

    mask = (g > 0) & np.isfinite(lb)
    v[mask] = x[mask] - lb[mask]
    dv[mask] = 1

    return v, dv


def reflective_transformation(y, lb, ub):
    """Compute reflective transformation and its gradient."""
    if in_bounds(y, lb, ub):
        return y, np.ones_like(y)

    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    x = y.copy()
    g_negative = np.zeros_like(y, dtype=bool)

    mask = lb_finite & ~ub_finite
    x[mask] = np.maximum(y[mask], 2 * lb[mask] - y[mask])
    g_negative[mask] = y[mask] < lb[mask]

    mask = ~lb_finite & ub_finite
    x[mask] = np.minimum(y[mask], 2 * ub[mask] - y[mask])
    g_negative[mask] = y[mask] > ub[mask]

    mask = lb_finite & ub_finite
    d = ub - lb
    t = np.remainder(y[mask] - lb[mask], 2 * d[mask])
    x[mask] = lb[mask] + np.minimum(t, 2 * d[mask] - t)
    g_negative[mask] = t > d[mask]

    g = np.ones_like(y)
    g[g_negative] = -1

    return x, g


# Functions to display algorithm's progress.


def print_header_nonlinear():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}"
          .format("Iteration", "Total nfev", "Cost", "Cost reduction",
                  "Step norm", "Optimality"))


def print_iteration_nonlinear(iteration, nfev, cost, cost_reduction,
                              step_norm, optimality):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = f"{cost_reduction:^15.2e}"

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = f"{step_norm:^15.2e}"

    print(f"{iteration:^15}{nfev:^15}{cost:^15.4e}{cost_reduction}{step_norm}{optimality:^15.2e}")


def print_header_linear():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}"
          .format("Iteration", "Cost", "Cost reduction", "Step norm",
                  "Optimality"))


def print_iteration_linear(iteration, cost, cost_reduction, step_norm,
                           optimality):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = f"{cost_reduction:^15.2e}"

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = f"{step_norm:^15.2e}"

    print(f"{iteration:^15}{cost:^15.4e}{cost_reduction}{step_norm}{optimality:^15.2e}")


# Simple helper functions.


def compute_grad(J, f):
    """Compute gradient of the least-squares cost function."""
    if isinstance(J, LinearOperator):
        return J.rmatvec(f)
    else:
        return J.T.dot(f)


def compute_jac_scale(J, scale_inv_old=None):
    """Compute variables scale based on the Jacobian matrix."""
    if issparse(J):
        scale_inv = np.asarray(J.power(2).sum(axis=0)).ravel()**0.5
    else:
        scale_inv = np.sum(J**2, axis=0)**0.5

    if scale_inv_old is None:
        scale_inv[scale_inv == 0] = 1
    else:
        scale_inv = np.maximum(scale_inv, scale_inv_old)

    return 1 / scale_inv, scale_inv


def left_multiplied_operator(J, d):
    """Return diag(d) J as LinearOperator."""
    J = aslinearoperator(J)

    def matvec(x):
        return d * J.matvec(x)

    def matmat(X):
        return d[:, np.newaxis] * J.matmat(X)

    def rmatvec(x):
        return J.rmatvec(x.ravel() * d)

    return LinearOperator(J.shape, matvec=matvec, matmat=matmat,
                          rmatvec=rmatvec)


def right_multiplied_operator(J, d):
    """Return J diag(d) as LinearOperator."""
    J = aslinearoperator(J)

    def matvec(x):
        return J.matvec(np.ravel(x) * d)

    def matmat(X):
        return J.matmat(X * d[:, np.newaxis])

    def rmatvec(x):
        return d * J.rmatvec(x)

    return LinearOperator(J.shape, matvec=matvec, matmat=matmat,
                          rmatvec=rmatvec)


def regularized_lsq_operator(J, diag):
    """Return a matrix arising in regularized least squares as LinearOperator.

    The matrix is
        [ J ]
        [ D ]
    where D is diagonal matrix with elements from `diag`.
    """
    J = aslinearoperator(J)
    m, n = J.shape

    def matvec(x):
        return np.hstack((J.matvec(x), diag * x))

    def rmatvec(x):
        x1 = x[:m]
        x2 = x[m:]
        return J.rmatvec(x1) + diag * x2

    return LinearOperator((m + n, n), matvec=matvec, rmatvec=rmatvec)


def right_multiply(J, d, copy=True):
    """Compute J diag(d).

    If `copy` is False, `J` is modified in place (unless being LinearOperator).
    """
    if copy and not isinstance(J, LinearOperator):
        J = J.copy()

    if issparse(J):
        J.data *= d.take(J.indices, mode='clip')  # scikit-learn recipe.
    elif isinstance(J, LinearOperator):
        J = right_multiplied_operator(J, d)
    else:
        J *= d

    return J


def left_multiply(J, d, copy=True):
    """Compute diag(d) J.

    If `copy` is False, `J` is modified in place (unless being LinearOperator).
    """
    if copy and not isinstance(J, LinearOperator):
        J = J.copy()

    if issparse(J):
        J.data *= np.repeat(d, np.diff(J.indptr))  # scikit-learn recipe.
    elif isinstance(J, LinearOperator):
        J = left_multiplied_operator(J, d)
    else:
        J *= d[:, np.newaxis]

    return J


def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    """Check termination condition for nonlinear least squares."""
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)

    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None


def scale_for_robust_loss_function(J, f, rho):
    """Scale Jacobian and residuals for a robust loss function.

    Arrays are modified in place.
    """
    J_scale = rho[1] + 2 * rho[2] * f**2
    J_scale[J_scale < EPS] = EPS
    J_scale **= 0.5

    f *= rho[1] / J_scale

    return left_multiply(J, J_scale, copy=False), f
<|endoftext|>"""Generic interface for least-squares minimization."""
from warnings import warn

import numpy as np
from numpy.linalg import norm

from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import _minpack, OptimizeResult
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.optimize._minimize import Bounds

from .trf import trf
from .dogbox import dogbox
from .common import EPS, in_bounds, make_strictly_feasible


TERMINATION_MESSAGES = {
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied."
}


FROM_MINPACK_TO_COMMON = {
    0: -1,  # Improper input parameters from MINPACK.
    1: 2,
    2: 3,
    3: 4,
    4: 1,
    5: 0
    # There are 6, 7, 8 for too small tolerance parameters,
    # but we guard against it by checking ftol, xtol, gtol beforehand.
}


def call_minpack(fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, diff_step):
    n = x0.size

    if diff_step is None:
        epsfcn = EPS
    else:
        epsfcn = diff_step**2

    # Compute MINPACK's `diag`, which is inverse of our `x_scale` and
    # ``x_scale='jac'`` corresponds to ``diag=None``.
    if isinstance(x_scale, str) and x_scale == 'jac':
        diag = None
    else:
        diag = 1 / x_scale

    full_output = True
    col_deriv = False
    factor = 100.0

    if jac is None:
        if max_nfev is None:
            # n squared to account for Jacobian evaluations.
            max_nfev = 100 * n * (n + 1)
        x, info, status = _minpack._lmdif(
            fun, x0, (), full_output, ftol, xtol, gtol,
            max_nfev, epsfcn, factor, diag)
    else:
        if max_nfev is None:
            max_nfev = 100 * n
        x, info, status = _minpack._lmder(
            fun, jac, x0, (), full_output, col_deriv,
            ftol, xtol, gtol, max_nfev, factor, diag)

    f = info['fvec']

    if callable(jac):
        J = jac(x)
    else:
        J = np.atleast_2d(approx_derivative(fun, x))

    cost = 0.5 * np.dot(f, f)
    g = J.T.dot(f)
    g_norm = norm(g, ord=np.inf)

    nfev = info['nfev']
    njev = info.get('njev', None)

    status = FROM_MINPACK_TO_COMMON[status]
    active_mask = np.zeros_like(x0, dtype=int)

    return OptimizeResult(
        x=x, cost=cost, fun=f, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev, status=status)


def prepare_bounds(bounds, n):
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


def check_tolerance(ftol, xtol, gtol, method):
    def check(tol, name):
        if tol is None:
            tol = 0
        elif tol < EPS:
            warn(f"Setting `{name}` below the machine epsilon ({EPS:.2e}) effectively "
                 f"disables the corresponding termination condition.",
                 stacklevel=3)
        return tol

    ftol = check(ftol, "ftol")
    xtol = check(xtol, "xtol")
    gtol = check(gtol, "gtol")

    if method == "lm" and (ftol < EPS or xtol < EPS or gtol < EPS):
        raise ValueError("All tolerances must be higher than machine epsilon "
                         f"({EPS:.2e}) for method 'lm'.")
    elif ftol < EPS and xtol < EPS and gtol < EPS:
        raise ValueError("At least one of the tolerances must be higher than "
                         f"machine epsilon ({EPS:.2e}).")

    return ftol, xtol, gtol


def check_x_scale(x_scale, x0):
    if isinstance(x_scale, str) and x_scale == 'jac':
        return x_scale

    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False

    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with "
                         "positive numbers.")

    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)

    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    return x_scale


def check_jac_sparsity(jac_sparsity, m, n):
    if jac_sparsity is None:
        return None

    if not issparse(jac_sparsity):
        jac_sparsity = np.atleast_2d(jac_sparsity)

    if jac_sparsity.shape != (m, n):
        raise ValueError("`jac_sparsity` has wrong shape.")

    return jac_sparsity, group_columns(jac_sparsity)


# Loss functions.


def huber(z, rho, cost_only):
    mask = z <= 1
    rho[0, mask] = z[mask]
    rho[0, ~mask] = 2 * z[~mask]**0.5 - 1
    if cost_only:
        return
    rho[1, mask] = 1
    rho[1, ~mask] = z[~mask]**-0.5
    rho[2, mask] = 0
    rho[2, ~mask] = -0.5 * z[~mask]**-1.5


def soft_l1(z, rho, cost_only):
    t = 1 + z
    rho[0] = 2 * (t**0.5 - 1)
    if cost_only:
        return
    rho[1] = t**-0.5
    rho[2] = -0.5 * t**-1.5


def cauchy(z, rho, cost_only):
    rho[0] = np.log1p(z)
    if cost_only:
        return
    t = 1 + z
    rho[1] = 1 / t
    rho[2] = -1 / t**2


def arctan(z, rho, cost_only):
    rho[0] = np.arctan(z)
    if cost_only:
        return
    t = 1 + z**2
    rho[1] = 1 / t
    rho[2] = -2 * z / t**2


IMPLEMENTED_LOSSES = dict(linear=None, huber=huber, soft_l1=soft_l1,
                          cauchy=cauchy, arctan=arctan)


def construct_loss_function(m, loss, f_scale):
    if loss == 'linear':
        return None

    if not callable(loss):
        loss = IMPLEMENTED_LOSSES[loss]
        rho = np.empty((3, m))

        def loss_function(f, cost_only=False):
            z = (f / f_scale) ** 2
            loss(z, rho, cost_only=cost_only)
            if cost_only:
                return 0.5 * f_scale ** 2 * np.sum(rho[0])
            rho[0] *= f_scale ** 2
            rho[2] /= f_scale ** 2
            return rho
    else:
        def loss_function(f, cost_only=False):
            z = (f / f_scale) ** 2
            rho = loss(z)
            if cost_only:
                return 0.5 * f_scale ** 2 * np.sum(rho[0])
            rho[0] *= f_scale ** 2
            rho[2] /= f_scale ** 2
            return rho

    return loss_function


def least_squares(
        fun, x0, jac='2-point', bounds=(-np.inf, np.inf), method='trf',
        ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=1.0, loss='linear',
        f_scale=1.0, diff_step=None, tr_solver=None, tr_options=None,
        jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs=None):
    """Solve a nonlinear least-squares problem with bounds on the variables.

    Given the residuals f(x) (an m-D real function of n real
    variables) and the loss function rho(s) (a scalar function), `least_squares`
    finds a local minimum of the cost function F(x)::

        minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
        subject to lb <= x <= ub

    The purpose of the loss function rho(s) is to reduce the influence of
    outliers on the solution.

    Parameters
    ----------
    fun : callable
        Function which computes the vector of residuals, with the signature
        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
        respect to its first argument. The argument ``x`` passed to this
        function is an ndarray of shape (n,) (never a scalar, even for n=1).
        It must allocate and return a 1-D array_like of shape (m,) or a scalar.
        If the argument ``x`` is complex or the function ``fun`` returns
        complex residuals, it must be wrapped in a real function of real
        arguments, as shown at the end of the Examples section.
    x0 : array_like with shape (n,) or float
        Initial guess on independent variables. If float, it will be treated
        as a 1-D array with one element. When `method` is 'trf', the initial
        guess might be slightly adjusted to lie sufficiently within the given
        `bounds`.
    jac : {'2-point', '3-point', 'cs', callable}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix, where
        element (i, j) is the partial derivative of f[i] with respect to
        x[j]). The keywords select a finite difference scheme for numerical
        estimation. The scheme '3-point' is more accurate, but requires
        twice as many operations as '2-point' (default). The scheme 'cs'
        uses complex steps, and while potentially the most accurate, it is
        applicable only when `fun` correctly handles complex inputs and
        can be analytically continued to the complex plane. Method 'lm'
        always uses the '2-point' scheme. If callable, it is used as
        ``jac(x, *args, **kwargs)`` and should return a good approximation
        (or the exact value) for the Jacobian as an array_like (np.atleast_2d
        is applied), a sparse matrix (csr_matrix preferred for performance) or
        a `scipy.sparse.linalg.LinearOperator`.
    bounds : 2-tuple of array_like or `Bounds`, optional
        There are two ways to specify bounds:

        1. Instance of `Bounds` class
        2. Lower and upper bounds on independent variables. Defaults to no
           bounds. Each array must match the size of `x0` or be a scalar,
           in the latter case a bound will be the same for all variables.
           Use ``np.inf`` with an appropriate sign to disable bounds on all
           or some variables.

    method : {'trf', 'dogbox', 'lm'}, optional
        Algorithm to perform minimization.

        * 'trf' : Trust Region Reflective algorithm, particularly suitable
          for large sparse problems with bounds. Generally robust method.
        * 'dogbox' : dogleg algorithm with rectangular trust regions,
          typical use case is small problems with bounds. Not recommended
          for problems with rank-deficient Jacobian.
        * 'lm' : Levenberg-Marquardt algorithm as implemented in MINPACK.
          Doesn't handle bounds and sparse Jacobians. Usually the most
          efficient method for small unconstrained problems.

        Default is 'trf'. See Notes for more information.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step.

        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8. The exact condition depends on the `method` used:

        * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``.
        * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
          a trust-region radius and ``xs`` is the value of ``x``
          scaled according to `x_scale` parameter (see below).

        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    gtol : float or None, optional
        Tolerance for termination by the norm of the gradient. Default is 1e-8.
        The exact condition depends on a `method` used:

        * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
          ``g_scaled`` is the value of the gradient scaled to account for
          the presence of the bounds [STIR]_.
        * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
          ``g_free`` is the gradient with respect to the variables which
          are not in the optimal state on the boundary.
        * For 'lm' : the maximum absolute value of the cosine of angles
          between columns of the Jacobian and the residual vector is less
          than `gtol`, or the residual vector is zero.

        If None and 'method' is not 'lm', the termination by this condition is
        disabled. If 'method' is 'lm', this tolerance must be higher than
        machine epsilon.
    x_scale : array_like or 'jac', optional
        Characteristic scale of each variable. Setting `x_scale` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust region along jth
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting `x_scale` such that a step of a given size
        along any of the scaled variables has a similar effect on the cost
        function. If set to 'jac', the scale is iteratively updated using the
        inverse norms of the columns of the Jacobian matrix (as described in
        [JJMore]_).
    loss : str or callable, optional
        Determines the loss function. The following keyword values are allowed:

        * 'linear' (default) : ``rho(z) = z``. Gives a standard
          least-squares problem.
        * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
          approximation of l1 (absolute value) loss. Usually a good
          choice for robust least squares.
        * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
          similarly to 'soft_l1'.
        * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
          influence, but may cause difficulties in optimization process.
        * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
          a single residual, has properties similar to 'cauchy'.

        If callable, it must take a 1-D ndarray ``z=f**2`` and return an
        array_like with shape (3, m) where row 0 contains function values,
        row 1 contains first derivatives and row 2 contains second
        derivatives. Method 'lm' supports only 'linear' loss.
    f_scale : float, optional
        Value of soft margin between inlier and outlier residuals, default
        is 1.0. The loss function is evaluated as follows
        ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
        and ``rho`` is determined by `loss` parameter. This parameter has
        no effect with ``loss='linear'``, but for other `loss` values it is
        of crucial importance.
    max_nfev : None or int, optional
        Maximum number of function evaluations before the termination.
        If None (default), the value is chosen automatically:

        * For 'trf' and 'dogbox' : 100 * n.
        * For 'lm' :  100 * n if `jac` is callable and 100 * n * (n + 1)
          otherwise (because 'lm' counts function calls in Jacobian
          estimation).

    diff_step : None or array_like, optional
        Determines the relative step size for the finite difference
        approximation of the Jacobian. The actual step is computed as
        ``x * diff_step``. If None (default), then `diff_step` is taken to be
        a conventional "optimal" power of machine epsilon for the finite
        difference scheme used [NR]_.
    tr_solver : {None, 'exact', 'lsmr'}, optional
        Method for solving trust-region subproblems, relevant only for 'trf'
        and 'dogbox' methods.

        * 'exact' is suitable for not very large problems with dense
          Jacobian matrices. The computational complexity per iteration is
          comparable to a singular value decomposition of the Jacobian
          matrix.
        * 'lsmr' is suitable for problems with sparse and large Jacobian
          matrices. It uses the iterative procedure
          `scipy.sparse.linalg.lsmr` for finding a solution of a linear
          least-squares problem and only requires matrix-vector product
          evaluations.

        If None (default), the solver is chosen based on the type of Jacobian
        returned on the first iteration.
    tr_options : dict, optional
        Keyword options passed to trust-region solver.

        * ``tr_solver='exact'``: `tr_options` are ignored.
        * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
          Additionally,  ``method='trf'`` supports  'regularize' option
          (bool, default is True), which adds a regularization term to the
          normal equation, which improves convergence if the Jacobian is
          rank-deficient [Byrd]_ (eq. 3.4).

    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines the sparsity structure of the Jacobian matrix for finite
        difference estimation, its shape must be (m, n). If the Jacobian has
        only few non-zero elements in *each* row, providing the sparsity
        structure will greatly speed up the computations [Curtis]_. A zero
        entry means that a corresponding element in the Jacobian is identically
        zero. If provided, forces the use of 'lsmr' trust-region solver.
        If None (default), then dense differencing will be used. Has no effect
        for 'lm' method.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

        * 0 (default) : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations (not supported by 'lm'
          method).

    args, kwargs : tuple and dict, optional
        Additional arguments passed to `fun` and `jac`. Both empty by default.
        The calling signature is ``fun(x, *args, **kwargs)`` and the same for
        `jac`.

    Returns
    -------
    result : OptimizeResult
        `OptimizeResult` with the following fields defined:

        x : ndarray, shape (n,)
            Solution found.
        cost : float
            Value of the cost function at the solution.
        fun : ndarray, shape (m,)
            Vector of residuals at the solution.
        jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
            Modified Jacobian matrix at the solution, in the sense that J^T J
            is a Gauss-Newton approximation of the Hessian of the cost function.
            The type is the same as the one used by the algorithm.
        grad : ndarray, shape (m,)
            Gradient of the cost function at the solution.
        optimality : float
            First-order optimality measure. In unconstrained problems, it is
            always the uniform norm of the gradient. In constrained problems,
            it is the quantity which was compared with `gtol` during iterations.
        active_mask : ndarray of int, shape (n,)
            Each component shows whether a corresponding constraint is active
            (that is, whether a variable is at the bound):

            *  0 : a constraint is not active.
            * -1 : a lower bound is active.
            *  1 : an upper bound is active.

            Might be somewhat arbitrary for 'trf' method as it generates a
            sequence of strictly feasible iterates and `active_mask` is
            determined within a tolerance threshold.
        nfev : int
            Number of function evaluations done. Methods 'trf' and 'dogbox' do
            not count function calls for numerical Jacobian approximation, as
            opposed to 'lm' method.
        njev : int or None
            Number of Jacobian evaluations done. If numerical Jacobian
            approximation is used in 'lm' method, it is set to None.
        status : int
            The reason for algorithm termination:

            * -1 : improper input parameters status returned from MINPACK.
            *  0 : the maximum number of function evaluations is exceeded.
            *  1 : `gtol` termination condition is satisfied.
            *  2 : `ftol` termination condition is satisfied.
            *  3 : `xtol` termination condition is satisfied.
            *  4 : Both `ftol` and `xtol` termination conditions are satisfied.

        message : str
            Verbal description of the termination reason.
        success : bool
            True if one of the convergence criteria is satisfied (`status` > 0).

    See Also
    --------
    leastsq : A legacy wrapper for the MINPACK implementation of the
              Levenberg-Marquadt algorithm.
    curve_fit : Least-squares minimization applied to a curve-fitting problem.

    Notes
    -----
    Method 'lm' (Levenberg-Marquardt) calls a wrapper over least-squares
    algorithms implemented in MINPACK (lmder, lmdif). It runs the
    Levenberg-Marquardt algorithm formulated as a trust-region type algorithm.
    The implementation is based on paper [JJMore]_, it is very robust and
    efficient with a lot of smart tricks. It should be your first choice
    for unconstrained problems. Note that it doesn't support bounds. Also,
    it doesn't work when m < n.

    Method 'trf' (Trust Region Reflective) is motivated by the process of
    solving a system of equations, which constitute the first-order optimality
    condition for a bound-constrained minimization problem as formulated in
    [STIR]_. The algorithm iteratively solves trust-region subproblems
    augmented by a special diagonal quadratic term and with trust-region shape
    determined by the distance from the bounds and the direction of the
    gradient. This enhancements help to avoid making steps directly into bounds
    and efficiently explore the whole space of variables. To further improve
    convergence, the algorithm considers search directions reflected from the
    bounds. To obey theoretical requirements, the algorithm keeps iterates
    strictly feasible. With dense Jacobians trust-region subproblems are
    solved by an exact method very similar to the one described in [JJMore]_
    (and implemented in MINPACK). The difference from the MINPACK
    implementation is that a singular value decomposition of a Jacobian
    matrix is done once per iteration, instead of a QR decomposition and series
    of Givens rotation eliminations. For large sparse Jacobians a 2-D subspace
    approach of solving trust-region subproblems is used [STIR]_, [Byrd]_.
    The subspace is spanned by a scaled gradient and an approximate
    Gauss-Newton solution delivered by `scipy.sparse.linalg.lsmr`. When no
    constraints are imposed the algorithm is very similar to MINPACK and has
    generally comparable performance. The algorithm works quite robust in
    unbounded and bounded problems, thus it is chosen as a default algorithm.

    Method 'dogbox' operates in a trust-region framework, but considers
    rectangular trust regions as opposed to conventional ellipsoids [Voglis]_.
    The intersection of a current trust region and initial bounds is again
    rectangular, so on each iteration a quadratic minimization problem subject
    to bound constraints is solved approximately by Powell's dogleg method
    [NumOpt]_. The required Gauss-Newton step can be computed exactly for
    dense Jacobians or approximately by `scipy.sparse.linalg.lsmr` for large
    sparse Jacobians. The algorithm is likely to exhibit slow convergence when
    the rank of Jacobian is less than the number of variables. The algorithm
    often outperforms 'trf' in bounded problems with a small number of
    variables.

    Robust loss functions are implemented as described in [BA]_. The idea
    is to modify a residual vector and a Jacobian matrix on each iteration
    such that computed gradient and Gauss-Newton Hessian approximation match
    the true gradient and Hessian approximation of the cost function. Then
    the algorithm proceeds in a normal way, i.e., robust loss functions are
    implemented as a simple wrapper over standard least-squares algorithms.

    .. versionadded:: 0.17.0

    References
    ----------
    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
              and Conjugate Gradient Method for Large-Scale Bound-Constrained
              Minimization Problems," SIAM Journal on Scientific Computing,
              Vol. 21, Number 1, pp 1-23, 1999.
    .. [NR] William H. Press et. al., "Numerical Recipes. The Art of Scientific
            Computing. 3rd edition", Sec. 5.7.
    .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate
              solution of the trust region problem by minimization over
              two-dimensional subspaces", Math. Programming, 40, pp. 247-263,
              1988.
    .. [Curtis] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
                sparse Jacobian matrices", Journal of the Institute of
                Mathematics and its Applications, 13, pp. 117-120, 1974.
    .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation
                and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
                Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region
                Dogleg Approach for Unconstrained and Bound Constrained
                Nonlinear Optimization", WSEAS International Conference on
                Applied Mathematics, Corfu, Greece, 2004.
    .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,
                2nd edition", Chapter 4.
    .. [BA] B. Triggs et. al., "Bundle Adjustment - A Modern Synthesis",
            Proceedings of the International Workshop on Vision Algorithms:
            Theory and Practice, pp. 298-372, 1999.

    Examples
    --------
    In this example we find a minimum of the Rosenbrock function without bounds
    on independent variables.

    >>> import numpy as np
    >>> def fun_rosenbrock(x):
    ...     return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

    Notice that we only provide the vector of the residuals. The algorithm
    constructs the cost function as a sum of squares of the residuals, which
    gives the Rosenbrock function. The exact minimum is at ``x = [1.0, 1.0]``.

    >>> from scipy.optimize import least_squares
    >>> x0_rosenbrock = np.array([2, 2])
    >>> res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
    >>> res_1.x
    array([ 1.,  1.])
    >>> res_1.cost
    9.8669242910846867e-30
    >>> res_1.optimality
    8.8928864934219529e-14

    We now constrain the variables, in such a way that the previous solution
    becomes infeasible. Specifically, we require that ``x[1] >= 1.5``, and
    ``x[0]`` left unconstrained. To this end, we specify the `bounds` parameter
    to `least_squares` in the form ``bounds=([-np.inf, 1.5], np.inf)``.

    We also provide the analytic Jacobian:

    >>> def jac_rosenbrock(x):
    ...     return np.array([
    ...         [-20 * x[0], 10],
    ...         [-1, 0]])

    Putting this all together, we see that the new solution lies on the bound:

    >>> res_2 = least_squares(fun_rosenbrock, x0_rosenbrock, jac_rosenbrock,
    ...                       bounds=([-np.inf, 1.5], np.inf))
    >>> res_2.x
    array([ 1.22437075,  1.5       ])
    >>> res_2.cost
    0.025213093946805685
    >>> res_2.optimality
    1.5885401433157753e-07

    Now we solve a system of equations (i.e., the cost function should be zero
    at a minimum) for a Broyden tridiagonal vector-valued function of 100000
    variables:

    >>> def fun_broyden(x):
    ...     f = (3 - x) * x + 1
    ...     f[1:] -= x[:-1]
    ...     f[:-1] -= 2 * x[1:]
    ...     return f

    The corresponding Jacobian matrix is sparse. We tell the algorithm to
    estimate it by finite differences and provide the sparsity structure of
    Jacobian to significantly speed up this process.

    >>> from scipy.sparse import lil_matrix
    >>> def sparsity_broyden(n):
    ...     sparsity = lil_matrix((n, n), dtype=int)
    ...     i = np.arange(n)
    ...     sparsity[i, i] = 1
    ...     i = np.arange(1, n)
    ...     sparsity[i, i - 1] = 1
    ...     i = np.arange(n - 1)
    ...     sparsity[i, i + 1] = 1
    ...     return sparsity
    ...
    >>> n = 100000
    >>> x0_broyden = -np.ones(n)
    ...
    >>> res_3 = least_squares(fun_broyden, x0_broyden,
