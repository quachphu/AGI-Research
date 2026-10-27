    yc2 = ctx.mag(68*(L+2)*yA)
    yc4 = ctx.mag(yB1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        yc3 = yc2 - k*yc4+ctx.mag(ctx.fac(k+0.5))/2.
        ywptcoef[k] = ((max(c1,yc3-ctx.mag(yeps4)+1))+10)*1.5
        ywpterm[k] = (max(c1,ctx.mag(L+2)+yc3-ctx.mag(yeps3)+1)+1)+10

    # check of power of pi
    # computing the fortcoef[mu,k,ell]
    xfortcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                xfortcoef[mu,k,ell]=0
    for mu in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xfortcoef[mu,k,ell]=xd[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                xfortcoef[mu,k,ell]=xfortcoef[mu,k,ell]/((2*ctx.j)**ell)

    def trunc_a(t):
        wp = ctx.prec
        ctx.prec = wp + 2
        aa = ctx.sqrt(t/(2*ctx.pi))
        ctx.prec = wp
        return aa

    # computing the tcoef[k,ell]
    xtcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                xtcoef[mu,k,ell]=0
    ctx.prec = max(xwptcoef[0],ywptcoef[0])+3
    aa= trunc_a(t)
    la = -ctx.ln(aa)

    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xtcoef[chi,k,ell] =0
                for mu in range(0, chi+1):
                    tcoefter=ctx.binomial(chi,mu)*ctx.power(la,mu)*xfortcoef[chi-mu,k,ell]
                    xtcoef[chi,k,ell] += tcoefter

    # COMPUTING THE COEFFICIENTS ytcoef[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    # check of power of pi
    # computing the fortcoef[mu,k,ell]
    yfortcoef={}
    for mu in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                yfortcoef[mu,k,ell]=0
    for mu in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                yfortcoef[mu,k,ell]=yd[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                yfortcoef[mu,k,ell]=yfortcoef[mu,k,ell]/((2*ctx.j)**ell)
    # computing the tcoef[k,ell]
    ytcoef={}
    for chi in range(0,der+1):
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                ytcoef[chi,k,ell]=0
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                ytcoef[chi,k,ell] =0
                for mu in range(0, chi+1):
                    tcoefter=ctx.binomial(chi,mu)*ctx.power(la,mu)*yfortcoef[chi-mu,k,ell]
                    ytcoef[chi,k,ell] += tcoefter

    # COMPUTING tv[k,ell]
    # See II Section 3.8
    #
    #  a has a good value
    ctx.prec = max(xwptcoef[0], ywptcoef[0])+2
    av = {}
    av[0] = 1
    av[1] = av[0]/a

    ctx.prec = max(xwptcoef[0],ywptcoef[0])
    for k in range(2,L):
        av[k] = av[k-1] * av[1]

    # Computing the quotients
    xtv = {}
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = xwptcoef[k]
            for ell in range(0,3*k//2+1):
                xtv[chi,k,ell] = xtcoef[chi,k,ell]* av[k]
    # Computing the quotients
    ytv = {}
    for chi in range(0,der+1):
        for k in range(0,L):
            ctx.prec = ywptcoef[k]
            for ell in range(0,3*k//2+1):
                ytv[chi,k,ell] = ytcoef[chi,k,ell]* av[k]

    # COMPUTING THE TERMS xterm[k]
    # See II Section 3.6
    xterm = {}
    for chi in range(0,der+1):
        for n in range(0,L):
            ctx.prec = xwpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += xtv[chi,n,k]
            xterm[chi,n] = te

    # COMPUTING THE TERMS yterm[k]
    # See II Section 3.6
    yterm = {}
    for chi in range(0,der+1):
        for n in range(0,L):
            ctx.prec = ywpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += ytv[chi,n,k]
            yterm[chi,n] = te

    # COMPUTING  rssum
    # See II Section 3.5
    xrssum={}
    ctx.prec=15
    xrsbound = math.sqrt(ctx.pi) * xc /(xb*a)
    ctx.prec=15
    xwprssum = ctx.mag(4.4*((L+3)**2)*xrsbound / xeps2)
    xwprssum = max(xwprssum, ctx.mag(10*(L+1)))
    ctx.prec = xwprssum
    for chi in range(0,der+1):
        xrssum[chi] = 0
        for k in range(1,L+1):
            xrssum[chi] += xterm[chi,L-k]
    yrssum={}
    ctx.prec=15
    yrsbound = math.sqrt(ctx.pi) * yc /(yb*a)
    ctx.prec=15
    ywprssum = ctx.mag(4.4*((L+3)**2)*yrsbound / yeps2)
    ywprssum = max(ywprssum, ctx.mag(10*(L+1)))
    ctx.prec = ywprssum
    for chi in range(0,der+1):
        yrssum[chi] = 0
        for k in range(1,L+1):
            yrssum[chi] += yterm[chi,L-k]

    # COMPUTING S3
    # See II Section 3.19
    ctx.prec = 15
    A2 = 2**(max(ctx.mag(abs(xrssum[0])), ctx.mag(abs(yrssum[0]))))
    eps8 = eps/(3*A2)
    T = t *ctx.ln(t/(2*ctx.pi))
    xwps3 = 5 +  ctx.mag((1+(2/eps8)*ctx.power(a,-xsigma))*T)
    ywps3 = 5 +  ctx.mag((1+(2/eps8)*ctx.power(a,-ysigma))*T)

    ctx.prec = max(xwps3, ywps3)

    tpi = t/(2*ctx.pi)
    arg = (t/2)*ctx.ln(tpi)-(t/2)-ctx.pi/8
    U = ctx.expj(-arg)
    a = trunc_a(t)
    xasigma = ctx.power(a, -xsigma)
    yasigma = ctx.power(a, -ysigma)
    xS3 = ((-1)**(N-1)) * xasigma * U
    yS3 = ((-1)**(N-1)) * yasigma * U

    # COMPUTING S1 the zetasum
    # See II Section 3.18
    ctx.prec = 15
    xwpsum =  4+ ctx.mag((N+ctx.power(N,1-xsigma))*ctx.ln(N) /eps1)
    ywpsum =  4+ ctx.mag((N+ctx.power(N,1-ysigma))*ctx.ln(N) /eps1)
    wpsum = max(xwpsum, ywpsum)

    ctx.prec = wpsum +10
    '''
    # This can be improved
    xS1={}
    yS1={}
    for chi in range(0,der+1):
        xS1[chi] = 0
        yS1[chi] = 0
    for n in range(1,int(N)+1):
        ln = ctx.ln(n)
        xexpn = ctx.exp(-ln*(xsigma+ctx.j*t))
        yexpn = ctx.conj(1/(n*xexpn))
        for chi in range(0,der+1):
            pown = ctx.power(-ln, chi)
            xterm = pown*xexpn
            yterm = pown*yexpn
            xS1[chi] += xterm
            yS1[chi] += yterm
    '''
    xS1, yS1 = ctx._zetasum(s, 1, int(N)-1, range(0,der+1), True)

    # END OF COMPUTATION of xrz, yrz
    #  See II Section 3.1
    ctx.prec = 15
    xabsS1 = abs(xS1[der])
    xabsS2 = abs(xrssum[der] * xS3)
    xwpend = max(6, wpinitial+ctx.mag(6*(3*xabsS1+7*xabsS2) ) )

    ctx.prec = xwpend
    xrz={}
    for chi in range(0,der+1):
        xrz[chi] = xS1[chi]+xrssum[chi]*xS3

    ctx.prec = 15
    yabsS1 = abs(yS1[der])
    yabsS2 = abs(yrssum[der] * yS3)
    ywpend = max(6, wpinitial+ctx.mag(6*(3*yabsS1+7*yabsS2) ) )

    ctx.prec = ywpend
    yrz={}
    for chi in range(0,der+1):
        yrz[chi] = yS1[chi]+yrssum[chi]*yS3
        yrz[chi] = ctx.conj(yrz[chi])
    ctx.prec = wpinitial
    return xrz, yrz

def Rzeta_set(ctx, s, derivatives=[0]):
    r"""
    Computes several derivatives of the auxiliary function of Riemann `R(s)`.

    **Definition**

    The function is defined by

    .. math ::

        \begin{equation}
        {\mathop{\mathcal R }\nolimits}(s)=
        \int_{0\swarrow1}\frac{x^{-s} e^{\pi i x^2}}{e^{\pi i x}-
        e^{-\pi i x}}\,dx
        \end{equation}

    To this function we apply the Riemann-Siegel expansion.
    """
    der = max(derivatives)
    # First we take the value of ctx.prec
    # During the computation we will change ctx.prec, and finally we will
    # restaurate the initial value
    wpinitial = ctx.prec
    # Take the real and imaginary part of s
    t = ctx._im(s)
    sigma = ctx._re(s)
    # Now compute several parameter that appear on the program
    ctx.prec = 15
    a = ctx.sqrt(t/(2*ctx.pi))     #  Careful
    asigma = ctx.power(a, sigma)  #  Careful
    # We need a simple bound A1 < asigma  (see II Section 3.1 and 3.3)
    A1 = ctx.power(2, ctx.mag(asigma)-1)
    # We compute various epsilon's  (see II end of Section 3.1)
    eps = ctx.power(2, -wpinitial)
    eps1 = eps/6.
    eps2 = eps * A1/3.
    # COMPUTING SOME COEFFICIENTS THAT DEPENDS
    #               ON  sigma
    # constant b and c  (see I  Theorem 2 formula (26) )
    # coefficients A and B1  (see I Section 6.1 equation (50))
    # here we not need high precision
    ctx.prec = 15
    if sigma > 0:
        b = 2.
        c = math.pow(9,sigma)/4.44288
        # 4.44288 =(math.sqrt(2)*math.pi)
        A = math.pow(9,sigma)
        B1 = 1
    else:
        b = 2.25158  #  math.sqrt( (3-2* math.log(2))*math.pi )
        c = math.pow(2,-sigma)/4.44288
        A = math.pow(2,-sigma)
        B1 = 1.10789   #  = 2*sqrt(1-log(2))
    #  COMPUTING L THE NUMBER OF TERMS NEEDED IN THE RIEMANN-SIEGEL
    #                         CORRECTION
    #  See II Section 3.2
    ctx.prec = 15
    L = 1
    while 3*c*ctx.gamma(L*0.5) * ctx.power(b*a,-L) >= eps2:
        L = L+1
    L = max(2,L)
    #  The number L has to satify some conditions.
    #  If not RS can not compute Rzeta(s) with the prescribed precision
    #  (see II, Section 3.2 condition (20)  ) and
    #  (II, Section 3.3 condition (22) ). Also we have added
    #  an additional technical  condition in Section 3.17 Proposition 17
    if ((3*L >= 2*a*a/25.) or (3*L+2+sigma<0) or (abs(sigma)> a/2.)):
        #print 'Error Riemann-Siegel can not compute with such precision'
        ctx.prec = wpinitial
        raise NotImplementedError("Riemann-Siegel can not compute with such precision")

    #  INITIALIZATION (CONTINUATION)
    #
    # eps3 is the constant defined on (II, Section 3.5 equation (27) )
    # each term of the RS correction must be computed with error <= eps3
    eps3 =  eps2/(4*L)

    # eps4 is defined on (II Section 3.6  equation (30) )
    # each component of the formula (II Section 3.6 equation (29) )
    # must be computed with error <= eps4
    eps4 = eps3/(3*L)

    # COMPUTING M.  NUMBER OF DERIVATIVES Fp[m] TO COMPUTE
    M = aux_M_Fp(ctx, A, eps4, a, B1, L)
    Fp = {}
    for n in range(M, 3*L-2):
        Fp[n] = 0

    #  But I have not seen an instance of  M != 3*L-3
    #
    #  DETERMINATION OF  J  THE NUMBER OF TERMS NEEDED
    #            IN THE TAYLOR SERIES OF F.
    #  See II Section 3.11 equation (49))
    h1 = eps4/(632*A)
    h2 = ctx.pi*ctx.pi*B1*a *ctx.sqrt(3)*math.e*math.e
    h2 = h1 * ctx.power((h2/M**2),(M-1)/3) / M
    h3 = min(h1,h2)
    J=12
    jvalue = (2*ctx.pi)**J / ctx.gamma(J+1)
    while jvalue > h3:
        J = J+1
        jvalue = (2*ctx.pi)*jvalue/J

    # COMPUTING eps5[m] for 1 <= m <= 21
    #  See II Section 10 equation (43)
    eps5={}
    foreps5 = math.pi*math.pi*B1*a
    for m in range(0,22):
        aux1 = math.pow(foreps5, m/3)/(316.*A)
        aux2 = ctx.gamma(m+1)/ctx.gamma(m/3.0+0.5)
        aux2 = math.sqrt(aux2)
        eps5[m] = aux1*aux2*eps4

    # COMPUTING wpfp
    #  See II Section 3.13 equation (59)
    twenty = min(3*L-3, 21)+1
    aux = 6812*J
    wpfp = ctx.mag(44*J)
    for m in range(0, twenty):
        wpfp = max(wpfp, ctx.mag(aux*ctx.gamma(m+1)/eps5[m]))
    # COMPUTING N AND p
    #  See II Section
    ctx.prec = wpfp + ctx.mag(t) + 20
    a = ctx.sqrt(t/(2*ctx.pi))
    N = ctx.floor(a)
    p = 1-2*(a-N)

    # now we get a rounded version of p to the precision wpfp
    # this possibly is not necessary
    num = ctx.floor(p*(ctx.mpf(2)**wpfp))
    difference = p * (ctx.mpf(2)**wpfp)-num
    if difference < 0.5:
        num = num
    else:
        num = num+1
    p = ctx.convert(num * (ctx.mpf(2)**(-wpfp)))

    # COMPUTING THE COEFFICIENTS c[n] = cc[n]
    # We shall use the notation cc[n], since there is
    # a constant that is called c
    # See II Section 3.14
    # We compute the coefficients and also save then in a
    # cache.  The bulk of the computation is passed to
    # the function  coef()
    #
    #  eps6 is defined in II Section 3.13  equation (58)
    eps6 = ctx.power(2*ctx.pi, J)/(ctx.gamma(J+1)*3*J)

    #  Now we compute the coefficients
    cc={}
    cont={}
    cont, pipowers = coef(ctx, J, eps6)
    cc = cont.copy()   # we need a copy since we have
    Fp={}
    for n in range(M, 3*L-2):
        Fp[n] = 0
    ctx.prec = wpfp
    for m in range(0,M+1):
        sumP = 0
        for k in range(2*J-m-1,-1,-1):
            sumP = (sumP * p) + cc[k]
        Fp[m] = sumP
        # preparation of the new coefficients
        for k in range(0, 2*J-m-1):
            cc[k] = (k+1) * cc[k+1]

    # COMPUTING THE NUMBERS  d[n,k]
    #  See II Section 3.17

    #  First we compute the working precisions wpd[k]
    #   Se II equation (92)
    wpd = {}
    d1 = max(6, ctx.mag(40*L*L))
    d2 = 13+ctx.mag((1+abs(sigma))*A)-ctx.mag(eps4)-1
    const = ctx.ln(8/(ctx.pi*ctx.pi*a*a*B1*B1)) /2
    for n in range(0,L):
        d3 = ctx.mag(ctx.sqrt(ctx.gamma(n-0.5)))-ctx.floor(n*const)+d2
        wpd[n] = max(d3,d1)

    # procedure of II Section 3.17
    ctx.prec = wpd[1]+10
    psigma = 1-(2*sigma)
    d = {}
    d[0,0,-2]=0; d[0,0,-1]=0; d[0,0,0]=1; d[0,0,1]=0
    d[0,-1,-2]=0; d[0,-1,-1]=0; d[0,-1,0]=1; d[0,-1,1]=0
    for n in range(1,L):
        ctx.prec = wpd[n]+10
        for k in range(0,3*n//2+1):
            m = 3*n-2*k
            if (m!=0):
                m1 = ctx.one/m
                c1 = m1/4
                c2 = (psigma*m1)/2
                c3 = -(m+1)
                d[0,n,k] = c3*d[0,n-1,k-2]+c1*d[0,n-1,k]+c2*d[0,n-1,k-1]
            else:
                d[0,n,k]=0
                for r in range(0,k):
                    add = d[0,n,r]*(ctx.one*ctx.fac(2*k-2*r)/ctx.fac(k-r))
                    d[0,n,k] -= ((-1)**(k-r))*add
        d[0,n,-2]=0; d[0,n,-1]=0; d[0,n,3*n//2+1]=0

    for mu in range(-2,der+1):
        for n in range(-2,L):
            for k in range(-3,max(1,3*n//2+2)):
                if ((mu<0)or (n<0) or(k<0)or (k>3*n//2)):
                    d[mu,n,k] = 0

    for mu in range(1,der+1):
        for n in range(0,L):
            ctx.prec = wpd[n]+10
            for k in range(0,3*n//2+1):
                aux=(2*mu-2)*d[mu-2,n-2,k-3]+2*(sigma+n-2)*d[mu-1,n-2,k-3]
                d[mu,n,k] = aux - d[mu-1,n-1,k-1]

    # COMPUTING THE COEFFICIENTS t[k,l]
    #  See II Section 3.9
    #
    # computing the needed wp
    wptcoef = {}
    wpterm = {}
    ctx.prec = 15
    c1 = ctx.mag(40*(L+2))
    c2 = ctx.mag(68*(L+2)*A)
    c4 = ctx.mag(B1*a*math.sqrt(ctx.pi))-1
    for k in range(0,L):
        c3 = c2 - k*c4+ctx.mag(ctx.fac(k+0.5))/2.
        wptcoef[k] = max(c1,c3-ctx.mag(eps4)+1)+1 +10
        wpterm[k] = max(c1,ctx.mag(L+2)+c3-ctx.mag(eps3)+1)+1 +10

    # check of power of pi

    # computing the fortcoef[mu,k,ell]
    fortcoef={}
    for mu in derivatives:
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                fortcoef[mu,k,ell]=0

    for mu in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                fortcoef[mu,k,ell]=d[mu,k,ell]*Fp[3*k-2*ell]/pipowers[2*k-ell]
                fortcoef[mu,k,ell]=fortcoef[mu,k,ell]/((2*ctx.j)**ell)

    def trunc_a(t):
        wp = ctx.prec
        ctx.prec = wp + 2
        aa = ctx.sqrt(t/(2*ctx.pi))
        ctx.prec = wp
        return aa

    # computing the tcoef[chi,k,ell]
    tcoef={}
    for chi in derivatives:
        for k in range(0,L):
            for ell in range(-2,3*k//2+1):
                tcoef[chi,k,ell]=0
    ctx.prec = wptcoef[0]+3
    aa = trunc_a(t)
    la = -ctx.ln(aa)

    for chi in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                tcoef[chi,k,ell] = 0
                for mu in range(0, chi+1):
                    tcoefter = ctx.binomial(chi,mu) * la**mu * \
                        fortcoef[chi-mu,k,ell]
                    tcoef[chi,k,ell] += tcoefter

    # COMPUTING tv[k,ell]
    # See II Section 3.8

    # Computing the powers av[k] = a**(-k)
    ctx.prec = wptcoef[0] + 2

    # a has a good value of a.
    # See II Section 3.6
    av = {}
    av[0] = 1
    av[1] = av[0]/a

    ctx.prec = wptcoef[0]
    for k in range(2,L):
        av[k] = av[k-1] * av[1]

    # Computing the quotients
    tv = {}
    for chi in derivatives:
        for k in range(0,L):
            ctx.prec = wptcoef[k]
            for ell in range(0,3*k//2+1):
                tv[chi,k,ell] = tcoef[chi,k,ell]* av[k]

    # COMPUTING THE TERMS term[k]
    # See II Section 3.6
    term = {}
    for chi in derivatives:
        for n in range(0,L):
            ctx.prec = wpterm[n]
            te = 0
            for k in range(0, 3*n//2+1):
                te += tv[chi,n,k]
            term[chi,n] = te

    # COMPUTING  rssum
    # See II Section 3.5
    rssum={}
    ctx.prec=15
    rsbound = math.sqrt(ctx.pi) * c /(b*a)
    ctx.prec=15
    wprssum = ctx.mag(4.4*((L+3)**2)*rsbound / eps2)
    wprssum = max(wprssum, ctx.mag(10*(L+1)))
    ctx.prec = wprssum
    for chi in derivatives:
        rssum[chi] = 0
        for k in range(1,L+1):
            rssum[chi] += term[chi,L-k]

    # COMPUTING S3
    # See II Section 3.19
    ctx.prec = 15
    A2 = 2**(ctx.mag(rssum[0]))
    eps8 = eps/(3* A2)
    T = t * ctx.ln(t/(2*ctx.pi))
    wps3 = 5 + ctx.mag((1+(2/eps8)*ctx.power(a,-sigma))*T)

    ctx.prec = wps3
    tpi = t/(2*ctx.pi)
    arg = (t/2)*ctx.ln(tpi)-(t/2)-ctx.pi/8
    U = ctx.expj(-arg)
    a = trunc_a(t)
    asigma = ctx.power(a, -sigma)
    S3 = ((-1)**(N-1)) * asigma * U

    # COMPUTING S1 the zetasum
    # See II Section 3.18
    ctx.prec = 15
    wpsum = 4 + ctx.mag((N+ctx.power(N,1-sigma))*ctx.ln(N)/eps1)

    ctx.prec = wpsum + 10
    '''
    # This can be improved
    S1 = {}
    for chi in derivatives:
        S1[chi] = 0
    for n in range(1,int(N)+1):
        ln = ctx.ln(n)
        expn = ctx.exp(-ln*(sigma+ctx.j*t))
        for chi in derivatives:
            term = ctx.power(-ln, chi)*expn
            S1[chi] += term
    '''
    S1 = ctx._zetasum(s, 1, int(N)-1, derivatives)[0]

    # END OF COMPUTATION
    #  See II Section 3.1
    ctx.prec = 15
    absS1 = abs(S1[der])
    absS2 = abs(rssum[der] * S3)
    wpend = max(6, wpinitial + ctx.mag(6*(3*absS1+7*absS2)))
    ctx.prec = wpend
    rz = {}
    for chi in derivatives:
        rz[chi] = S1[chi]+rssum[chi]*S3
    ctx.prec = wpinitial
    return rz


def z_half(ctx,t,der=0):
    r"""
    z_half(t,der=0) Computes Z^(der)(t)
    """
    s=ctx.mpf('0.5')+ctx.j*t
    wpinitial = ctx.prec
    ctx.prec = 15
    tt = t/(2*ctx.pi)
    wptheta = wpinitial +1 + ctx.mag(3*(tt**1.5)*ctx.ln(tt))
    wpz = wpinitial + 1 + ctx.mag(12*tt*ctx.ln(tt))
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t)
    ctx.prec = wpz
    rz = Rzeta_set(ctx,s, range(der+1))
    if der > 0: ps1 = ctx._re(ctx.psi(0,s/2)/2 - ctx.ln(ctx.pi)/2)
    if der > 1: ps2 = ctx._re(ctx.j*ctx.psi(1,s/2)/4)
    if der > 2: ps3 = ctx._re(-ctx.psi(2,s/2)/8)
    if der > 3: ps4 = ctx._re(-ctx.j*ctx.psi(3,s/2)/16)
    exptheta = ctx.expj(theta)
    if der == 0:
        z = 2*exptheta*rz[0]
    if der == 1:
        zf = 2j*exptheta
        z = zf*(ps1*rz[0]+rz[1])
    if der == 2:
        zf = 2 * exptheta
        z = -zf*(2*rz[1]*ps1+rz[0]*ps1**2+rz[2]-ctx.j*rz[0]*ps2)
    if der == 3:
        zf = -2j*exptheta
        z = 3*rz[1]*ps1**2+rz[0]*ps1**3+3*ps1*rz[2]
        z = zf*(z-3j*rz[1]*ps2-3j*rz[0]*ps1*ps2+rz[3]-rz[0]*ps3)
    if der == 4:
        zf = 2*exptheta
        z = 4*rz[1]*ps1**3+rz[0]*ps1**4+6*ps1**2*rz[2]
        z = z-12j*rz[1]*ps1*ps2-6j*rz[0]*ps1**2*ps2-6j*rz[2]*ps2-3*rz[0]*ps2*ps2
        z = z + 4*ps1*rz[3]-4*rz[1]*ps3-4*rz[0]*ps1*ps3+rz[4]+ctx.j*rz[0]*ps4
        z = zf*z
    ctx.prec = wpinitial
    return ctx._re(z)

def zeta_half(ctx, s, k=0):
    """
    zeta_half(s,k=0) Computes zeta^(k)(s) when Re s = 0.5
    """
    wpinitial = ctx.prec
    sigma = ctx._re(s)
    t = ctx._im(s)
    #--- compute wptheta, wpR, wpbasic ---
    ctx.prec = 53
    #  X see II Section 3.21 (109) and (110)
    if sigma > 0:
        X = ctx.sqrt(abs(s))
    else:
        X = (2*ctx.pi)**(sigma-1) * abs(1-s)**(0.5-sigma)
    # M1  see II Section 3.21 (111) and (112)
    if sigma > 0:
        M1 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M1 = 4 * t * X
    # T  see II Section 3.21 (113)
    abst = abs(0.5-s)
    T = 2* abst*math.log(abst)
    # computing wpbasic, wptheta, wpR  see II Section 3.21
    wpbasic = max(6,3+ctx.mag(t))
    wpbasic2 = 2+ctx.mag(2.12*M1+21.2*M1*X+1.3*M1*X*T)+wpinitial+1
    wpbasic = max(wpbasic, wpbasic2)
    wptheta = max(4, 3+ctx.mag(2.7*M1*X)+wpinitial+1)
    wpR = 3+ctx.mag(1.1+2*X)+wpinitial+1
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t-ctx.j*(sigma-ctx.mpf('0.5')))
    if k > 0: ps1 = (ctx._re(ctx.psi(0,s/2)))/2 - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = -(ctx._im(ctx.psi(1,s/2)))/4
    if k > 2: ps3 = -(ctx._re(ctx.psi(2,s/2)))/8
    if k > 3: ps4 = (ctx._im(ctx.psi(3,s/2)))/16
    ctx.prec = wpR
    xrz = Rzeta_set(ctx,s,range(k+1))
    yrz={}
    for chi in range(0,k+1):
        yrz[chi] = ctx.conj(xrz[chi])
    ctx.prec = wpbasic
    exptheta = ctx.expj(-2*theta)
    if k==0:
        zv = xrz[0]+exptheta*yrz[0]
    if k==1:
        zv1 = -yrz[1] - 2*yrz[0]*ps1
        zv = xrz[1] + exptheta*zv1
    if k==2:
        zv1 = 4*yrz[1]*ps1+4*yrz[0]*(ps1**2)+yrz[2]+2j*yrz[0]*ps2
        zv = xrz[2]+exptheta*zv1
    if k==3:
        zv1 = -12*yrz[1]*ps1**2-8*yrz[0]*ps1**3-6*yrz[2]*ps1-6j*yrz[1]*ps2
        zv1 = zv1 - 12j*yrz[0]*ps1*ps2-yrz[3]+2*yrz[0]*ps3
        zv = xrz[3]+exptheta*zv1
    if k == 4:
        zv1 = 32*yrz[1]*ps1**3 +16*yrz[0]*ps1**4+24*yrz[2]*ps1**2
        zv1 = zv1 +48j*yrz[1]*ps1*ps2+48j*yrz[0]*(ps1**2)*ps2
        zv1 = zv1+12j*yrz[2]*ps2-12*yrz[0]*ps2**2+8*yrz[3]*ps1-8*yrz[1]*ps3
        zv1 = zv1-16*yrz[0]*ps1*ps3+yrz[4]-2j*yrz[0]*ps4
        zv = xrz[4]+exptheta*zv1
    ctx.prec = wpinitial
    return zv

def zeta_offline(ctx, s, k=0):
    """
    Computes zeta^(k)(s) off the line
    """
    wpinitial = ctx.prec
    sigma = ctx._re(s)
    t = ctx._im(s)
    #--- compute wptheta, wpR, wpbasic ---
    ctx.prec = 53
    #  X see II Section 3.21 (109) and (110)
    if sigma > 0:
        X = ctx.power(abs(s), 0.5)
    else:
        X = ctx.power(2*ctx.pi, sigma-1)*ctx.power(abs(1-s),0.5-sigma)
    # M1  see II Section 3.21 (111) and (112)
    if (sigma > 0):
        M1 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M1 = 4 * t * X
    # M2  see II Section 3.21 (111) and (112)
    if (1-sigma > 0):
        M2 = 2*ctx.sqrt(t/(2*ctx.pi))
    else:
        M2 = 4*t*ctx.power(2*ctx.pi, -sigma)*ctx.power(abs(s),sigma-0.5)
    # T  see II Section 3.21 (113)
    abst = abs(0.5-s)
    T = 2* abst*math.log(abst)
    # computing wpbasic, wptheta, wpR  see II Section 3.21
    wpbasic = max(6,3+ctx.mag(t))
    wpbasic2 = 2+ctx.mag(2.12*M1+21.2*M2*X+1.3*M2*X*T)+wpinitial+1
    wpbasic = max(wpbasic, wpbasic2)
    wptheta = max(4, 3+ctx.mag(2.7*M2*X)+wpinitial+1)
    wpR = 3+ctx.mag(1.1+2*X)+wpinitial+1
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t-ctx.j*(sigma-ctx.mpf('0.5')))
    s1 = s
    s2 = ctx.conj(1-s1)
    ctx.prec = wpR
    xrz, yrz = Rzeta_simul(ctx, s, k)
    if k > 0: ps1 = (ctx.psi(0,s1/2)+ctx.psi(0,(1-s1)/2))/4 - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = ctx.j*(ctx.psi(1,s1/2)-ctx.psi(1,(1-s1)/2))/8
    if k > 2: ps3 = -(ctx.psi(2,s1/2)+ctx.psi(2,(1-s1)/2))/16
    if k > 3: ps4 = -ctx.j*(ctx.psi(3,s1/2)-ctx.psi(3,(1-s1)/2))/32
    ctx.prec = wpbasic
    exptheta = ctx.expj(-2*theta)
    if k == 0:
        zv = xrz[0]+exptheta*yrz[0]
    if k == 1:
        zv1 = -yrz[1]-2*yrz[0]*ps1
        zv = xrz[1]+exptheta*zv1
    if k == 2:
        zv1 = 4*yrz[1]*ps1+4*yrz[0]*(ps1**2) +yrz[2]+2j*yrz[0]*ps2
        zv = xrz[2]+exptheta*zv1
    if k == 3:
        zv1 = -12*yrz[1]*ps1**2 -8*yrz[0]*ps1**3-6*yrz[2]*ps1-6j*yrz[1]*ps2
        zv1 = zv1 - 12j*yrz[0]*ps1*ps2-yrz[3]+2*yrz[0]*ps3
        zv = xrz[3]+exptheta*zv1
    if k == 4:
        zv1 = 32*yrz[1]*ps1**3 +16*yrz[0]*ps1**4+24*yrz[2]*ps1**2
        zv1 = zv1 +48j*yrz[1]*ps1*ps2+48j*yrz[0]*(ps1**2)*ps2
        zv1 = zv1+12j*yrz[2]*ps2-12*yrz[0]*ps2**2+8*yrz[3]*ps1-8*yrz[1]*ps3
        zv1 = zv1-16*yrz[0]*ps1*ps3+yrz[4]-2j*yrz[0]*ps4
        zv = xrz[4]+exptheta*zv1
    ctx.prec = wpinitial
    return zv

def z_offline(ctx, w, k=0):
    r"""
    Computes Z(w) and its derivatives off the line
    """
    s = ctx.mpf('0.5')+ctx.j*w
    s1 = s
    s2 = ctx.conj(1-s1)
    wpinitial = ctx.prec
    ctx.prec = 35
    #  X see II Section 3.21 (109) and (110)
    # M1  see II Section 3.21 (111) and (112)
    if (ctx._re(s1) >= 0):
        M1 = 2*ctx.sqrt(ctx._im(s1)/(2 * ctx.pi))
        X = ctx.sqrt(abs(s1))
    else:
        X = (2*ctx.pi)**(ctx._re(s1)-1) * abs(1-s1)**(0.5-ctx._re(s1))
        M1 = 4 * ctx._im(s1)*X
    # M2  see II Section 3.21 (111) and (112)
    if (ctx._re(s2) >= 0):
        M2 = 2*ctx.sqrt(ctx._im(s2)/(2 * ctx.pi))
    else:
        M2 = 4 * ctx._im(s2)*(2*ctx.pi)**(ctx._re(s2)-1)*abs(1-s2)**(0.5-ctx._re(s2))
    # T  see II Section 3.21  Prop. 27
    T = 2*abs(ctx.siegeltheta(w))
    # defining some precisions
    # see II Section 3.22 (115), (116), (117)
    aux1 = ctx.sqrt(X)
    aux2 = aux1*(M1+M2)
    aux3 = 3 +wpinitial
    wpbasic = max(6, 3+ctx.mag(T), ctx.mag(aux2*(26+2*T))+aux3)
    wptheta = max(4,ctx.mag(2.04*aux2)+aux3)
    wpR = ctx.mag(4*aux1)+aux3
    # now the computations
    ctx.prec = wptheta
    theta = ctx.siegeltheta(w)
    ctx.prec = wpR
    xrz, yrz = Rzeta_simul(ctx,s,k)
    pta = 0.25 + 0.5j*w
    ptb = 0.25 - 0.5j*w
    if k > 0: ps1 = 0.25*(ctx.psi(0,pta)+ctx.psi(0,ptb)) - ctx.ln(ctx.pi)/2
    if k > 1: ps2 = (1j/8)*(ctx.psi(1,pta)-ctx.psi(1,ptb))
    if k > 2: ps3 = (-1./16)*(ctx.psi(2,pta)+ctx.psi(2,ptb))
    if k > 3: ps4 = (-1j/32)*(ctx.psi(3,pta)-ctx.psi(3,ptb))
    ctx.prec = wpbasic
    exptheta = ctx.expj(theta)
    if k == 0:
        zv = exptheta*xrz[0]+yrz[0]/exptheta
    j = ctx.j
    if k == 1:
        zv = j*exptheta*(xrz[1]+xrz[0]*ps1)-j*(yrz[1]+yrz[0]*ps1)/exptheta
    if k == 2:
        zv = exptheta*(-2*xrz[1]*ps1-xrz[0]*ps1**2-xrz[2]+j*xrz[0]*ps2)
        zv =zv + (-2*yrz[1]*ps1-yrz[0]*ps1**2-yrz[2]-j*yrz[0]*ps2)/exptheta
    if k == 3:
        zv1 = -3*xrz[1]*ps1**2-xrz[0]*ps1**3-3*xrz[2]*ps1+j*3*xrz[1]*ps2
        zv1 = (zv1+ 3j*xrz[0]*ps1*ps2-xrz[3]+xrz[0]*ps3)*j*exptheta
        zv2 = 3*yrz[1]*ps1**2+yrz[0]*ps1**3+3*yrz[2]*ps1+j*3*yrz[1]*ps2
        zv2 = j*(zv2 + 3j*yrz[0]*ps1*ps2+ yrz[3]-yrz[0]*ps3)/exptheta
        zv = zv1+zv2
    if k == 4:
        zv1 = 4*xrz[1]*ps1**3+xrz[0]*ps1**4 + 6*xrz[2]*ps1**2
        zv1 = zv1-12j*xrz[1]*ps1*ps2-6j*xrz[0]*ps1**2*ps2-6j*xrz[2]*ps2
        zv1 = zv1-3*xrz[0]*ps2*ps2+4*xrz[3]*ps1-4*xrz[1]*ps3-4*xrz[0]*ps1*ps3
        zv1 = zv1+xrz[4]+j*xrz[0]*ps4
        zv2 = 4*yrz[1]*ps1**3+yrz[0]*ps1**4 + 6*yrz[2]*ps1**2
        zv2 = zv2+12j*yrz[1]*ps1*ps2+6j*yrz[0]*ps1**2*ps2+6j*yrz[2]*ps2
        zv2 = zv2-3*yrz[0]*ps2*ps2+4*yrz[3]*ps1-4*yrz[1]*ps3-4*yrz[0]*ps1*ps3
        zv2 = zv2+yrz[4]-j*yrz[0]*ps4
        zv = exptheta*zv1+zv2/exptheta
    ctx.prec = wpinitial
    return zv

@defun
def rs_zeta(ctx, s, derivative=0, **kwargs):
    if derivative > 4:
        raise NotImplementedError
    s = ctx.convert(s)
    re = ctx._re(s); im = ctx._im(s)
    if im < 0:
        z = ctx.conj(ctx.rs_zeta(ctx.conj(s), derivative))
        return z
    critical_line = (re == 0.5)
    if critical_line:
        return zeta_half(ctx, s, derivative)
    else:
        return zeta_offline(ctx, s, derivative)

@defun
def rs_z(ctx, w, derivative=0):
    w = ctx.convert(w)
    re = ctx._re(w); im = ctx._im(w)
    if re < 0:
        return rs_z(ctx, -w, derivative)
    critical_line = (im == 0)
    if critical_line :
        return z_half(ctx, w, derivative)
    else:
        return z_offline(ctx, w, derivative)
<|endoftext|>import enum
import functools
import itertools
import reprlib
import sys
import threading
from array import array
from collections.abc import (
    Callable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    ValuesView,
)
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    NoReturn,
    TypeVar,
    cast,
    final,
    overload,
)

from ._abc import MDArg, MultiMapping, MutableMultiMapping, SupportsKeys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

MAXSIZE = sys.maxsize

# Entry hashes are folded non-negative (& MAXSIZE) so HASH_MARK, the
# hash range's high bit, can mark a hash as temporarily invalid: OR it in,
# AND it out with MAXSIZE. A real folded hash never has that bit set.
HASH_MARK = MAXSIZE + 1
# Same as HT_LOG_RESUME_SLOTS_MINSIZE and HT_RESUME_SLOTS_MIN_STEPS in htkeys.h
_LOG_RESUME_SLOTS_MINSIZE = 10
_RESUME_SLOTS_MIN_STEPS = 32


@final
class istr(str):
    """Case insensitive str."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError(f"type '{istr.__module__}.istr' is not an acceptable base type")

    # Implementation note:
    # The class doesn't use __slots__ because slot-based memory model
    # is ~20% slower than classic __dict__-based one.
    # See also: https://github.com/aio-libs/multidict/pull/1386

    __is_istr__ = True
    __istr_identity__: str | None = None


_V = TypeVar("_V")
_T = TypeVar("_T")

_SENTINEL = enum.Enum("_SENTINEL", "sentinel")
sentinel = _SENTINEL.sentinel

_version = itertools.count(1)

_FREE_THREADED = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()

_VERSION_LOCK = threading.Lock()


def _other_lock(arg: object) -> "threading.RLock | None":
    """Return the RLock backing `arg`, if it's a multidict, one of our
    own views over one, or an iterator over one, that we don't already
    own."""
    if isinstance(arg, MultiDictProxy):
        return arg._md._lock
    if isinstance(arg, MultiDict):
        return arg._lock
    if isinstance(arg, (_ItemsView, _KeysView, _ValuesView)):
        return arg._md._lock
    if isinstance(arg, _Iter):
        return arg._lock
    return None


def _iter_lock(md: "MultiDict[Any]") -> "threading.RLock | None":
    """Return the lock a view iterator over `md` should take per step."""
    return md._lock if _FREE_THREADED else None


class _PairLock:
    """Acquire one or two RLocks in a fixed, deadlock-safe order."""

    __slots__ = ("_a", "_b")

    def __init__(self, a: "threading.RLock", b: "threading.RLock | None") -> None:
        if b is None or b is a:
            self._a: threading.RLock | None = a
            self._b: threading.RLock | None = None
        elif id(a) < id(b):
            self._a, self._b = a, b
        else:
            self._a, self._b = b, a

    def __enter__(self) -> None:
        assert self._a is not None
        self._a.acquire()
        if self._b is not None:
            self._b.acquire()

    def __exit__(self, *exc: object) -> None:
        assert self._a is not None
        if self._b is not None:
            self._b.release()
        self._a.release()


_F = TypeVar("_F", bound=Callable[..., Any])


def _locked_always(fn: _F) -> _F:
    """Wrap `fn` to hold `self._lock` for the call, on every build.

    update()/extend()/merge()/clear()/popitem() have their own race that,
    unlike the rest of this module's locking, is not specific to
    free-threading: pure-Python bytecode is not atomic even under the GIL
    (the GIL can be released between any two bytecodes), so enough
    concurrent contention lets two calls interleave mid hash-table insert
    or deletion and corrupt the shared index table. `_locked`/`_locked_pair`
    skip locking on a GIL-enabled build because the race they guard against
    is free-threading-only; this variant cannot.
    """

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        with self._lock:
            return fn(self, *args, **kwargs)

    return cast(_F, wrapper)


def _locked_pair_always(fn: _F) -> _F:
    """Like `_locked_always`, but also pairs in the first argument's lock,
    for the same reason `_locked_pair` does."""

    @functools.wraps(fn)
    def wrapper(self: Any, arg: Any = None, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        with _PairLock(self._lock, _other_lock(arg)):
            return fn(self, arg, *args, **kwargs)

    return cast(_F, wrapper)


def _locked(fn: _F) -> _F:
    """Wrap `fn` to hold `self._lock` for the call, but only when it
    guards against a free-threading-only race; see `_locked_always` for
    the unconditional version a few methods need."""
    if not _FREE_THREADED:
        return fn
    return _locked_always(fn)


def _locked_md(fn: _F) -> _F:
    """Like `_locked`, but for view objects: holds `self._md._lock`."""
    if not _FREE_THREADED:
        return fn

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        with self._md._lock:
            return fn(self, *args, **kwargs)

    return cast(_F, wrapper)


def _locked_pair(fn: _F) -> _F:
    """Wrap `fn` to hold `self._lock` and its first argument's lock, but
    only when it guards against a free-threading-only race; see
    `_locked_pair_always`."""
    if not _FREE_THREADED:
        return fn
    return _locked_pair_always(fn)


def _locked_md_pair(fn: _F) -> _F:
    """Like `_locked_md`, but also pairs in its first argument's lock."""
    if not _FREE_THREADED:
        return fn

    @functools.wraps(fn)
    def wrapper(self: Any, other: Any = None, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        with _PairLock(self._md._lock, _other_lock(other)):
            return fn(self, other, *args, **kwargs)

    return cast(_F, wrapper)


class _Iter(Generic[_T]):
    __slots__ = ("_size", "_iter", "_lock")

    def __init__(
        self,
        size: int,
        iterator: Iterator[_T],
        lock: "threading.RLock | None" = None,
    ):
        self._size = size
        self._iter = iterator
        self._lock = lock

    def __iter__(self) -> Self:
        return self

    @_locked
    def __next__(self) -> _T:
        return next(self._iter)

    def __length_hint__(self) -> int:
        return self._size


class _ViewBase(Generic[_V]):
    def __init__(
        self,
        md: "MultiDict[_V]",
    ):
        self._md = md

    def __len__(self) -> int:
        return len(self._md)


class _ItemsView(_ViewBase[_V], ItemsView[str, _V]):
    @_locked_md
    def __contains__(self, item: object) -> bool:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            return False
        key, value = item
        try:
            identity = self._md._identity(key)
        except TypeError:
            return False
        hash_ = hash(identity) & MAXSIZE
        for slot, idx, e in self._md._keys.iter_hash(hash_):
            if e.identity == identity and value == e.value:
                return True
        return False

    def __iter__(self) -> _Iter[tuple[str, _V]]:
        return _Iter(len(self), self._iter(self._md._version), _iter_lock(self._md))

    def __reversed__(self) -> _Iter[tuple[str, _V]]:
        return _Iter(
            len(self),
            self._iter(self._md._version, reverse=True),
            _iter_lock(self._md),
        )

    def _iter(self, version: int, reverse: bool = False) -> Iterator[tuple[str, _V]]:
        entries = self._md._keys.iter_entries(reverse)
        while True:
            if version != self._md._version:
                raise RuntimeError("Dictionary changed during iteration")
            try:
                e = next(entries)
            except StopIteration:
                return
            yield self._md._key(e.key), e.value

    @reprlib.recursive_repr()
    @_locked_md
    def __repr__(self) -> str:
        lst = []
        for e in self._md._keys.iter_entries():
            lst.append(f"{e.key!r}: {e.value!r}")
        body = ", ".join(lst)
        return f"<{self.__class__.__name__}({body})>"

    def _parse_item(self, arg: tuple[str, _V] | _T) -> tuple[int, str, str, _V] | None:
        if not isinstance(arg, tuple):
            return None
        if len(arg) != 2:
            return None
        try:
            identity = self._md._identity(arg[0])
            return (hash(identity) & MAXSIZE, identity, arg[0], arg[1])
        except TypeError:
            return None

    def _tmp_set(self, it: Iterable[_T]) -> set[tuple[str, _V]]:
        tmp = set()
        for arg in it:
            item = self._parse_item(arg)
            if item is None:
                continue
            else:
                tmp.add((item[1], item[3]))
        return tmp

    @_locked_md_pair
    def __and__(self, other: Iterable[Any]) -> set[tuple[str, _V]]:  # type: ignore[misc]
        ret = set()
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for arg in it:
            item = self._parse_item(arg)
            if item is None:
                continue
            hash_, identity, key, value = item
            matches = []
            last = -1
            for slot, idx, e in self._md._keys.iter_hash(hash_):
                if idx > last and e.identity == identity:
                    matches.append((e.key, e.value))
                    last = idx
            for e_key, e_value in matches:
                if e_value == value:
                    ret.add((e_key, e_value))
        return ret

    @_locked_md_pair
    def __rand__(self, other: Iterable[_T]) -> set[_T]:
        ret = set()
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for arg in it:
            item = self._parse_item(arg)
            if item is None:
                continue
            hash_, identity, key, value = item
            for slot, idx, e in self._md._keys.iter_hash(hash_):
                if e.identity == identity and e.value == value:
                    ret.add(arg)
                    break
        return ret

    @_locked_md_pair
    def __or__(self, other: Iterable[_T]) -> set[tuple[str, _V] | _T]:
        ret: set[tuple[str, _V] | _T] = set(self)
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for arg in it:
            item: tuple[int, str, str, _V] | None = self._parse_item(arg)
            if item is None:
                ret.add(arg)
                continue
            hash_, identity, key, value = item
            for slot, idx, e in self._md._keys.iter_hash(hash_):
                if e.identity == identity and e.value == value:  # pragma: no branch
                    break
            else:
                ret.add(arg)
        return ret

    @_locked_md_pair
    def __ror__(self, other: Iterable[_T]) -> set[tuple[str, _V] | _T]:
        try:
            ret: set[tuple[str, _V] | _T] = set(other)
        except TypeError:
            return NotImplemented
        tmp = self._tmp_set(ret)

        for e in self._md._keys.iter_entries():
            if (e.identity, e.value) not in tmp:
                ret.add((e.key, e.value))
        return ret

    @_locked_md_pair
    def __sub__(self, other: Iterable[_T]) -> set[tuple[str, _V] | _T]:
        ret: set[tuple[str, _V] | _T] = set()
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        tmp = self._tmp_set(it)

        for e in self._md._keys.iter_entries():
            if (e.identity, e.value) not in tmp:
                ret.add((e.key, e.value))

        return ret

    @_locked_md_pair
    def __rsub__(self, other: Iterable[_T]) -> set[_T]:
        ret: set[_T] = set()
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for arg in it:
            item = self._parse_item(arg)
            if item is None:
                ret.add(arg)
                continue

            hash_, identity, key, value = item
            for slot, idx, e in self._md._keys.iter_hash(hash_):
                if e.identity == identity and e.value == value:  # pragma: no branch
                    break
            else:
                ret.add(arg)
        return ret

    @_locked_md_pair
    def __xor__(self, other: Iterable[_T]) -> set[tuple[str, _V] | _T]:
        try:
            rgt = set(other)
        except TypeError:
            return NotImplemented
        ret: set[tuple[str, _V] | _T] = self - rgt
        ret |= rgt - self
        return ret

    __rxor__ = __xor__

    @_locked_md_pair
    def isdisjoint(self, other: Iterable[tuple[str, _V]]) -> bool:
        for arg in other:
            item = self._parse_item(arg)
            if item is None:
                continue

            hash_, identity, key, value = item
            for slot, idx, e in self._md._keys.iter_hash(hash_):
                if e.identity == identity and e.value == value:  # pragma: no branch
                    return False
        return True


class _ValuesView(_ViewBase[_V], ValuesView[_V]):
    @_locked_md
    def __contains__(self, value: object) -> bool:
        for e in self._md._keys.iter_entries():
            if e.value == value:
                return True
        return False

    def __iter__(self) -> _Iter[_V]:
        return _Iter(len(self), self._iter(self._md._version), _iter_lock(self._md))

    def __reversed__(self) -> _Iter[_V]:
        return _Iter(
            len(self),
            self._iter(self._md._version, reverse=True),
            _iter_lock(self._md),
        )

    def _iter(self, version: int, reverse: bool = False) -> Iterator[_V]:
        entries = self._md._keys.iter_entries(reverse)
        while True:
            if version != self._md._version:
                raise RuntimeError("Dictionary changed during iteration")
            try:
                e = next(entries)
            except StopIteration:
                return
            yield e.value

    @reprlib.recursive_repr()
    @_locked_md
    def __repr__(self) -> str:
        lst = []
        for e in self._md._keys.iter_entries():
            lst.append(repr(e.value))
        body = ", ".join(lst)
        return f"<{self.__class__.__name__}({body})>"


class _KeysView(_ViewBase[_V], KeysView[str]):
    @_locked_md
    def __len__(self) -> int:
        return sum(1 for _ in self._md._keys.iter_first_entries())

    @_locked_md
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        identity = self._md._identity(key)
        hash_ = hash(identity) & MAXSIZE
        for slot, idx, e in self._md._keys.iter_hash(hash_):
            if e.identity == identity:  # pragma: no branch
                return True
        return False

    def __iter__(self) -> _Iter[str]:
        return _Iter(len(self._md), self._iter(self._md._version), _iter_lock(self._md))

    def __reversed__(self) -> _Iter[str]:
        return _Iter(
            len(self._md),
            self._iter(self._md._version, reverse=True),
            _iter_lock(self._md),
        )

    def _iter(self, version: int, reverse: bool = False) -> Iterator[str]:
        keys = self._md._keys
        if keys.maybe_dups:
            entries = keys.iter_first_entries(reverse)
        else:
            entries = keys.iter_entries(reverse)
        while True:
            if version != self._md._version:
                raise RuntimeError("Dictionary changed during iteration")
            try:
                e = next(entries)
            except StopIteration:
                return
            yield self._md._key(e.key)

    @reprlib.recursive_repr()
    @_locked_md
    def __repr__(self) -> str:
        lst = []
        for e in self._md._keys.iter_first_entries():
            lst.append(f"{e.key!r}")
        body = ", ".join(lst)
        return f"<{self.__class__.__name__}({body})>"

    @_locked_md_pair
    def __and__(self, other: Iterable[object]) -> set[str]:
        ret = set()
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for key in it:
            if not isinstance(key, str):
                continue
            identity = self._md._identity(key)
            hash_ = hash(identity) & MAXSIZE
            for slot, idx, e in self._md._keys.iter_hash(hash_):
                if e.identity == identity:  # pragma: no branch
                    ret.add(e.key)
                    break
        return ret

    @_locked_md_pair
    def __rand__(self, other: Iterable[_T]) -> set[_T]:
        ret = set()
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for key in it:
            if not isinstance(key, str):
                continue
            if key in self._md:
                ret.add(key)
        return cast(set[_T], ret)

    @_locked_md_pair
    def __or__(self, other: Iterable[_T]) -> set[str | _T]:
        ret: set[str | _T] = set(self)
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for key in it:
            if not isinstance(key, str):
                ret.add(key)
                continue
            if key not in self._md:
                ret.add(key)
        return ret

    @_locked_md_pair
    def __ror__(self, other: Iterable[_T]) -> set[str | _T]:
        try:
            ret: set[str | _T] = set(other)
        except TypeError:
            return NotImplemented

        tmp = set()
        for key in ret:
            if not isinstance(key, str):
                continue
            identity = self._md._identity(key)
            tmp.add(identity)

        for e in self._md._keys.iter_first_entries():
            if e.identity not in tmp:
                ret.add(e.key)
        return ret

    @_locked_md_pair
    def __sub__(self, other: Iterable[object]) -> set[str]:
        ret = set(self)
        try:
            it = iter(other)
        except TypeError:
            return NotImplemented
        for key in it:
            if not isinstance(key, str):
                continue
            identity = self._md._identity(key)
            hash_ = hash(identity) & MAXSIZE
            for slot, idx, e in self._md._keys.iter_hash(hash_):
                if e.identity == identity:  # pragma: no branch
                    ret.discard(e.key)
                    break
        return ret

    @_locked_md_pair
    def __rsub__(self, other: Iterable[_T]) -> set[_T]:
        try:
            ret: set[_T] = set(other)
        except TypeError:
            return NotImplemented
        for key in other:
            if not isinstance(key, str):
                continue
            if key in self._md:
                ret.discard(key)
        return ret

    @_locked_md_pair
    def __xor__(self, other: Iterable[_T]) -> set[str | _T]:
        try:
            rgt = set(other)
        except TypeError:
            return NotImplemented
        ret: set[str | _T] = self - rgt  # type: ignore[assignment]
        ret |= rgt - self
        return ret

    __rxor__ = __xor__

    @_locked_md_pair
    def isdisjoint(self, other: Iterable[object]) -> bool:
        for key in other:
            if not isinstance(key, str):
                continue
            if key in self._md:
                return False
        return True


class _CSMixin:
    _ci: ClassVar[bool] = False

    def _key(self, key: str) -> str:
        return key

    def _identity(self, key: str) -> str:
        if type(key) is str:
            return key
        if isinstance(key, str):
            return str.__str__(key)
        raise TypeError("MultiDict keys should be either str or subclasses of str")


class _CIMixin:
    _ci: ClassVar[bool] = True

    def _key(self, key: str) -> str:
        if type(key) is str:
            return istr(key)
        if type(key) is istr:
            return key
        # istr(key) would call a subclass's own __str__, which may spell a
        # different string than the key
        return istr(str.__str__(key))

    def _identity(self, key: str) -> str:
        if isinstance(key, istr):
            ret = key.__istr_identity__
            if ret is None:
                ret = key.lower()
                key.__istr_identity__ = ret
            return ret
        if isinstance(key, str):
            # never a subclass's own lower()
            return str.lower(key)
        else:
            raise TypeError("MultiDict keys should be either str or subclasses of str")


def estimate_log2_keysize(n: int) -> int:
    # 7 == HT_MINSIZE - 1
    return (((n * 3 + 1) // 2) | 7).bit_length()


@dataclass
class _Entry(Generic[_V]):
    hash: int
    identity: str
    key: str
    value: _V


@dataclass
class _HtKeys(Generic[_V]):
    LOG_MINSIZE: ClassVar[int] = 3
    MINSIZE: ClassVar[int] = 8
    PREALLOCATED_INDICES: ClassVar[dict[int, array]] = {  # type: ignore[type-arg]
        log2_size: array(
            "b" if log2_size < 8 else "h", (-1 for i in range(1 << log2_size))
        )
        for log2_size in range(3, 10)
    }

    log2_size: int
    usable: int

    indices: array  # type: ignore[type-arg] # TODO(PY312): array[int]
    entries: list[_Entry[_V] | None]
    # first slot probed with perturb == 0 -> last slot used after it,
    # created on the first long probe
    resume_slots: dict[int, int] | None = None
    # set once an insert finds an equal hash on its probe chain, so two
    # entries may share an identity; never cleared, only carried
    maybe_dups: bool = False

    @functools.cached_property
    def nslots(self) -> int:
        return 1 << self.log2_size

    @functools.cached_property
    def mask(self) -> int:
        return self.nslots - 1

    if sys.implementation.name != "pypy":

        def __sizeof__(self) -> int:
            return (
                object.__sizeof__(self)
                + sys.getsizeof(self.indices)
                + sys.getsizeof(self.entries)
                + (0 if self.resume_slots is None else sys.getsizeof(self.resume_slots))
            )

    @classmethod
    def new(cls, log2_size: int, entries: list[_Entry[_V] | None]) -> Self:
        size = 1 << log2_size
        usable = (size << 1) // 3
        if log2_size < 10:
            indices = cls.PREALLOCATED_INDICES[log2_size].__copy__()
        elif log2_size < 16:
            indices = array("h", (-1 for i in range(size)))
        elif log2_size < 32:
            indices = array("l", (-1 for i in range(size)))
        else:  # pragma: no cover  # don't test huge multidicts
            indices = array("q", (-1 for i in range(size)))
        ret = cls(
            log2_size=log2_size,
            usable=usable,
            indices=indices,
            entries=entries,
        )
        return ret

    def clone(self) -> "_HtKeys[_V]":
        entries = [
            _Entry(e.hash, e.identity, e.key, e.value) if e is not None else None
            for e in self.entries
        ]

        return _HtKeys(
            log2_size=self.log2_size,
            usable=self.usable,
            indices=self.indices.__copy__(),
            entries=entries,
            maybe_dups=self.maybe_dups,
        )

    def build_indices(self) -> None:
        mask = self.mask
        indices = self.indices
        for idx, e in enumerate(self.entries):
            assert e is not None
            hash_ = e.hash
            i = hash_ & mask
            perturb = hash_
            while indices[i] != -1:
                perturb >>= 5
                i = mask & (i * 5 + perturb + 1)
                if not perturb:
                    i = self._find_empty_slot_resume(i)
                    break
            indices[i] = idx

    def find_empty_slot(self, hash_: int, absent: bool = False) -> int:
        # `absent`: the key was just looked up and is missing
        mask = self.mask
        indices = self.indices
        i = hash_ & mask
        perturb = hash_
        ix = indices[i]
        if ix != -1 and not absent and not self.maybe_dups:
            return self._find_empty_slot_noting_dups(hash_)
        while ix != -1:
            perturb >>= 5
            i = (i * 5 + perturb + 1) & mask
            if not perturb:
                return self._find_empty_slot_resume(i)
            ix = indices[i]
        return i

    def _find_empty_slot_noting_dups(self, hash_: int) -> int:
        # Sets maybe_dups on passing an entry of the same hash, possibly
        # marked by an update() in flight: an earlier entry of the new key's
        # identity would be there.  A probe long enough for the resume slots
        # sets it without looking.
        mask = self.mask
        indices = self.indices
        entries = self.entries
        i = hash_ & mask
        perturb = hash_
        ix = indices[i]
        while ix != -1:
            if ix >= 0 and entries[ix].hash & ~HASH_MARK == hash_:
                self.maybe_dups = True
                return self.find_empty_slot(hash_)
            perturb >>= 5
            i = (i * 5 + perturb + 1) & mask
            if not perturb:
                self.maybe_dups = True
                return self._find_empty_slot_resume(i)
            ix = indices[i]
        return i

    def _find_empty_slot_resume(self, start: int) -> int:
        mask = self.mask
        indices = self.indices
        resume_slots = self.resume_slots
        resume = None if resume_slots is None else resume_slots.get(start)
        # the resume slot is used, start after it
        i = start if resume is None else (resume * 5 + 1) & mask
        steps = 0
        while indices[i] != -1:
            i = (i * 5 + 1) & mask
            steps += 1
        if resume_slots is None:
            if (
                steps < _RESUME_SLOTS_MIN_STEPS
                or self.log2_size < _LOG_RESUME_SLOTS_MINSIZE
            ):
                return i
            resume_slots = self.resume_slots = {}
        resume_slots[start] = i
        return i

    def iter_hash(self, hash_: int) -> Iterator[tuple[int, int, _Entry[_V]]]:
        mask = self.mask
        indices = self.indices
        entries = self.entries
        i = hash_ & mask
        perturb = hash_
        ix = indices[i]
        while ix != -1:
            if ix != -2:
                e = entries[ix]
                if e.hash == hash_:
                    yield i, ix, e
            perturb >>= 5
            i = (i * 5 + perturb + 1) & mask
            ix = indices[i]

    def del_idx(self, hash_: int, idx: int) -> None:
        mask = self.mask
        indices = self.indices
        i = hash_ & mask
        perturb = hash_
        ix = indices[i]
        while ix != idx:
            perturb >>= 5
            i = (i * 5 + perturb + 1) & mask
            ix = indices[i]
        indices[i] = -2

    def iter_entries(self, reverse: bool = False) -> Iterator[_Entry[_V]]:
        entries = reversed(self.entries) if reverse else self.entries
        return filter(None, entries)

    def iter_first_entries(self, reverse: bool = False) -> Iterator[_Entry[_V]]:
        if not self.maybe_dups:
            yield from self.iter_entries(reverse)
        elif not reverse:
            seen = set()
            for e in self.iter_entries():
                if e.identity not in seen:
                    seen.add(e.identity)
                    yield e
        else:
            # Inserts take only empty slots and rebuilds reinsert in index
            # order, so the first entry of an identity on its probe chain is
            # its lowest index, the one getone() returns.
            entries = self.entries
            for idx in range(len(entries) - 1, -1, -1):
                entry = entries[idx]
                if entry is not None and idx == next(
                    ix
                    for _, ix, e2 in self.iter_hash(entry.hash)
                    if e2.identity == entry.identity
                ):
                    yield entry

    def restore_hash(self, hash_: int) -> None:
        mask = self.mask
        indices = self.indices
        entries = self.entries
        i = hash_ & mask
        perturb = hash_
        ix = indices[i]
        while ix != -1:
            if ix != -2:
                entry = entries[ix]
                if entry.hash & HASH_MARK:
                    entry.hash = hash_
            perturb >>= 5
            i = (i * 5 + perturb + 1) & mask
            ix = indices[i]


class MultiDict(_CSMixin, MutableMultiMapping[_V]):
    """Dictionary with the support for duplicate keys."""

    __slots__ = ("_keys", "_used", "_version", "_lock")

    _lock: threading.RLock

    # __new__ runs once per object, never on a later re-init, so the lock
    # identity is stable even across `d.__init__(other)`. The lock itself
    # is always created (not just under free-threading): update()/
    # extend()/merge()/clear()/popitem() need it on every build, since
    # their race isn't free-threading-specific -- see `_locked_always`.
    def __new__(cls, *args: object, **kwargs: object) -> Self:
        self = super().__new__(cls)
        self._lock = threading.RLock()
        return self

    @_locked_pair_always
    def __init__(self, arg: MDArg[_V] = None, /, **kwargs: _V):
        self._incr_version()
        if not kwargs:
            md = None
            if isinstance(arg, MultiDictProxy):
                md = arg._md
            elif isinstance(arg, MultiDict):
                md = arg
            if md is not None and md._ci is self._ci:
                self._from_md(md)
                return

        it = self._parse_args(arg, kwargs)
        log2_size = estimate_log2_keysize(cast(int, next(it)))
        if log2_size > 17:  # pragma: no cover
            # Don't overallocate really huge keys space in init
            log2_size = 17
        # Replacing the keys drops the old pairs, whose finalizers must
        # find the new state complete rather than have it written over.
        self._used = 0
        self._keys: _HtKeys[_V] = _HtKeys.new(log2_size, [])
        self._extend_items(cast(Iterator[_Entry[_V]], it))

    def _from_md(self, md: "MultiDict[_V]") -> None:
        # Copy everything as-is without compacting the new multidict,
        # otherwise it requires reindexing
        keys = md._keys.clone()
        self._used = md._used
        self._keys = keys

    @overload
    def getall(self, key: str) -> list[_V]: ...
    @overload
    def getall(self, key: str, default: _T) -> list[_V] | _T: ...
    @_locked
    def getall(self, key: str, default: _T | _SENTINEL = sentinel) -> list[_V] | _T:
        """Return a list of all values matching the key."""
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        res = []
        # Equal keys are first reached in insertion order; see
        # md_walk_with_hash() in _multilib/walk.h.
        last = -1
        for slot, idx, e in self._keys.iter_hash(hash_):
            if idx > last and e.identity == identity:
                res.append(e.value)
                last = idx
                if not self._keys.maybe_dups:
                    break  # the key's only entry

        if res:
            return res
        if default is not sentinel:
            return default
        raise KeyError(f"Key not found: {key!r}")

    @overload
    def getone(self, key: str) -> _V: ...
    @overload
    def getone(self, key: str, default: _T) -> _V | _T: ...
    @_locked
    def getone(self, key: str, default: _T | _SENTINEL = sentinel) -> _V | _T:
        """Get first value matching the key.

        Raises KeyError if the key is not found and no default is provided.
        """
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        for slot, idx, e in self._keys.iter_hash(hash_):
            if e.identity == identity:  # pragma: no branch
                return e.value
        if default is not sentinel:
            return default
        raise KeyError(f"Key not found: {key!r}")

    # Mapping interface #

    def __getitem__(self, key: str) -> _V:
        return self.getone(key)

    @overload
    def get(self, key: str, /) -> _V | None: ...
    @overload
    def get(self, key: str, /, default: _T) -> _V | _T: ...
    def get(self, key: str, default: _T | None = None) -> _V | _T | None:
        """Get first value matching the key.

        If the key is not found, returns the default (or None if no default is provided)
        """
        return self.getone(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return self._used

    def keys(self) -> KeysView[str]:
        """Return a new view of the dictionary's keys."""
        return _KeysView(self)

    def items(self) -> ItemsView[str, _V]:
        """Return a new view of the dictionary's items as ``(key, value)`` pairs."""
        return _ItemsView(self)

    def values(self) -> _ValuesView[_V]:
        """Return a new view of the dictionary's values."""
        return _ValuesView(self)

    @_locked_pair
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        if isinstance(other, MultiDictProxy):
            return self == other._md
        if isinstance(other, MultiDict):
            lft = self._keys
            rht = other._keys
            if self._used != other._used:
                return False
            for e1, e2 in zip(lft.iter_entries(), rht.iter_entries()):
                if e1.identity != e2.identity or e1.value != e2.value:
                    return False
            return True
        if self._used != len(other):
            return False
        for k, v in self.items():
            nv = other.get(k, sentinel)
            if v != nv:
                return False
        return True

    @_locked
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        for slot, idx, e in self._keys.iter_hash(hash_):
            if e.identity == identity:  # pragma: no branch
                return True
        return False

    @reprlib.recursive_repr()
    @_locked
    def __repr__(self) -> str:
        body = ", ".join(f"{e.key!r}: {e.value!r}" for e in self._keys.iter_entries())
        return f"<{self.__class__.__name__}({body})>"

    if sys.implementation.name != "pypy":

        def __sizeof__(self) -> int:
            return object.__sizeof__(self) + sys.getsizeof(self._keys)

    def __reduce__(self) -> tuple[type[Self], tuple[list[tuple[str, _V]]]]:
        return (self.__class__, (list(self.items()),))

    @_locked
    def to_dict(self) -> dict[str, list[_V]]:
        """Return a dict with lists of all values for each key."""
        result: dict[str, list[_V]] = {}
        # Keyed by identity so a case-insensitive multidict groups every
        # spelling of a key together, and holding the list itself rather than
        # the first key so the value append needs no second lookup.
        seen: dict[str, list[_V]] = {}
        version = self._version
        for e in self._keys.iter_entries():
            values = seen.get(e.identity)
            if values is None:
                values = seen[e.identity] = [e.value]
                result[self._key(e.key)] = values
            else:
                values.append(e.value)
            if self._version != version:
                # Building and hashing a key both run a str subclass's own
                # code, which must not mutate what is being converted.
                raise RuntimeError("Dictionary changed during iteration")
        return result

    @_locked_always
    def add(self, key: str, value: _V) -> None:
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        self._add_with_hash(_Entry(hash_, identity, key, value))
        self._incr_version()

    def copy(self) -> Self:
        """Return a copy of itself."""
        cls = self.__class__
        return cls(self)

    __copy__ = copy

    @_locked_pair_always
    def extend(self, arg: MDArg[_V] = None, /, **kwargs: _V) -> None:
        """Extend current MultiDict with more values.

        This method must be used instead of update.
        """
        it = self._parse_args(arg, kwargs)
        self._reserve(cast(int, next(it)))
        self._extend_items(cast(Iterator[_Entry[_V]], it))

    def _parse_args(
        self,
        arg: MDArg[_V],
        kwargs: Mapping[str, _V],
    ) -> Iterator[int | _Entry[_V]]:
        identity_func = self._identity
        if isinstance(arg, MultiDictProxy):
            if arg._md is self:
                arg = [(e.key, e.value) for e in self._keys.iter_entries()]
        elif arg is self:
            arg = [(e.key, e.value) for e in self._keys.iter_entries()]
        if arg:
            if isinstance(arg, MultiDictProxy):
                arg = arg._md
            if isinstance(arg, MultiDict):
                yield len(arg) + len(kwargs)
                if self._ci is not arg._ci:
                    key_func = arg._key
                    for e in arg._keys.iter_entries():
                        key = key_func(e.key)
                        if key is not e.key:
                            # a fresh istr keeps the source's identity
                            key.__istr_identity__ = e.identity  # type: ignore[attr-defined]
                        identity = identity_func(key)
                        yield _Entry(hash(identity) & MAXSIZE, identity, key, e.value)
                else:
                    for e in arg._keys.iter_entries():
                        yield _Entry(e.hash, e.identity, e.key, e.value)
                if kwargs:
                    for key, value in kwargs.items():
                        identity = identity_func(key)
                        yield _Entry(hash(identity) & MAXSIZE, identity, key, value)
            else:
                if hasattr(arg, "keys"):
                    arg = cast(SupportsKeys[_V], arg)
                    arg = [(k, arg[k]) for k in arg.keys()]
                if kwargs:
                    arg = list(arg)
                    arg.extend(list(kwargs.items()))
                try:
                    yield len(arg)  # type: ignore[arg-type]
                except TypeError:
                    yield 0
                for pos, item in enumerate(arg):
                    if not len(item) == 2:
                        raise ValueError(
                            f"multidict update sequence element #{pos} "
                            f"has length {len(item)}; 2 is required"
                        )
                    try:
                        key = item[0]
                    except Exception as exc:
                        raise ValueError(
                            f"multidict update sequence element #{pos}'s "
                            f"key could not be fetched"
                        ) from exc
                    try:
                        value = item[1]
                    except Exception as exc:
                        raise ValueError(
                            f"multidict update sequence element #{pos}'s "
                            f"value could not be fetched"
                        ) from exc
                    identity = identity_func(key)
                    yield _Entry(hash(identity) & MAXSIZE, identity, key, value)
        else:
            yield len(kwargs)
            for key, value in kwargs.items():
                identity = identity_func(key)
                yield _Entry(hash(identity) & MAXSIZE, identity, key, value)

    def _extend_items(self, items: Iterable[_Entry[_V]]) -> None:
        for e in items:
            self._add_with_hash(e)
        self._incr_version()

    @_locked_always
    def clear(self) -> None:
        """Remove all items from MultiDict."""
        self._used = 0
        self._keys = _HtKeys.new(_HtKeys.LOG_MINSIZE, [])
        self._incr_version()

    # Mapping interface #

    @_locked_always
    def __setitem__(self, key: str, value: _V) -> None:
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        found = False
        # See __delitem__()
        removed: list[object] = []

        for slot, idx, e in self._keys.iter_hash(hash_):
            if e.identity == identity:  # pragma: no branch
                if not found:
                    removed.append((e.key, e.value))
                    e.key = key
                    e.value = value
                    e.hash = hash_ | HASH_MARK
                    found = True
                    self._incr_version()
                elif not (e.hash & HASH_MARK):  # pragma: no branch
                    self._del_at(slot, idx)
                    removed.append(e)
                if not self._keys.maybe_dups:
                    break  # the key's only entry

        if not found:
            self._add_with_hash(_Entry(hash_, identity, key, value), absent=True)
        else:
            self._keys.restore_hash(hash_)

    @_locked_always
    def __delitem__(self, key: str) -> None:
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        # Holds the removed pairs until every match is gone: their
        # finalizers must not run, and add a match, mid-scan.
        removed = []
        for slot, idx, e in self._keys.iter_hash(hash_):
            if e.identity == identity:  # pragma: no branch
                self._del_at(slot, idx)
                removed.append(e)
                if not self._keys.maybe_dups:
                    break  # the key's only entry
        if not removed:
            raise KeyError(key)
        else:
            self._incr_version()

    @overload
    def setdefault(
        self: "MultiDict[_T | None]", key: str, default: None = None
    ) -> _T | None: ...
    @overload
    def setdefault(self, key: str, default: _V) -> _V: ...
    @_locked_always  # type: ignore[misc]
    def setdefault(self, key: str, default: _V | None = None) -> _V | None:
        """Return value for key, set value to default if key is not present."""
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        for slot, idx, e in self._keys.iter_hash(hash_):
            if e.identity == identity:  # pragma: no branch
                return e.value
        self._add_with_hash(
            _Entry(hash_, identity, key, default),  # type: ignore[arg-type]
            absent=True,
        )
        self._incr_version()
        return default

    @overload
    def popone(self, key: str) -> _V: ...
    @overload
    def popone(self, key: str, default: _T) -> _V | _T: ...
    @_locked_always
    def popone(self, key: str, default: _T | _SENTINEL = sentinel) -> _V | _T:
        """Remove the first occurrence of key and return the corresponding value.

        If key is not found, default is returned if given, otherwise
        KeyError is raised.

        """
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        for slot, idx, e in self._keys.iter_hash(hash_):
            if e.identity == identity:  # pragma: no branch
                value = e.value
                self._del_at(slot, idx)
                self._incr_version()
                return value
        if default is sentinel:
            raise KeyError(key)
        else:
            return default

    # Type checking will inherit signature for pop() if we don't confuse it here.
    if not TYPE_CHECKING:
        pop = popone

    @overload
    def popall(self, key: str) -> list[_V]: ...
    @overload
    def popall(self, key: str, default: _T) -> list[_V] | _T: ...
    @_locked_always
    def popall(self, key: str, default: _T | _SENTINEL = sentinel) -> list[_V] | _T:
        """Remove all occurrences of key and return the list of corresponding
        values.

        If key is not found, default is returned if given, otherwise
        KeyError is raised.

        """
        identity = self._identity(key)
        hash_ = hash(identity) & MAXSIZE
        # See __delitem__()
        removed = []
        for slot, idx, e in self._keys.iter_hash(hash_):
            if e.identity == identity:  # pragma: no branch
                removed.append(e)
                self._del_at(slot, idx)
                self._incr_version()
                if not self._keys.maybe_dups:
                    break  # the key's only entry

        if not removed:
            if default is sentinel:
                raise KeyError(key)
            else:
                return default
        else:
            return [e.value for e in removed]

    @_locked_always
    def popitem(self) -> tuple[str, _V]:
        """Remove and return an arbitrary (key, value) pair."""
        if self._used <= 0:
            raise KeyError("empty multidict")

        entries = self._keys.entries
        pos = len(entries) - 1
        entry = entries[pos]
        while entry is None:
            pos -= 1
            entry = entries[pos]

        # Clear the indices slot before truncating entries, so a
        # concurrent unlocked read never sees an index pointing past
        # the end of the (now shorter) entries list.
        self._keys.del_idx(entry.hash, pos)
        del entries[pos:]
        # Not held past here: the pop can run a finalizer that replaces the
        # table, whose entries then go at once, as they do in C.
        del entries
        self._used -= 1
        self._incr_version()

        # istr() runs a str subclass's __str__, which may mutate self.
        return self._key(entry.key), entry.value

    @_locked_pair_always
    def update(self, arg: MDArg[_V] = None, /, **kwargs: _V) -> None:
        """Update the dictionary, overwriting existing keys."""
        it = self._parse_args(arg, kwargs)
        reserve = self._reserve_batch(cast(int, next(it)))
        # The entries this call wrote and those it doomed, by id(): kept
        # out of the table, they survive whatever runs between items. They
        # and `replaced` also hold the dropped pairs until the call ends.
        updated: dict[int, _Entry[_V]] = {}
        deleted: dict[int, tuple[_Entry[_V], _V]] = {}
        replaced: list[tuple[str, _V]] = []
        try:
            self._update_items(
                cast(Iterator[_Entry[_V]], it), updated, deleted, replaced, reserve
            )
        finally:
            self._post_update(deleted)

    def _update_items(
        self,
        items: Iterator[_Entry[_V]],
        updated: dict[int, _Entry[_V]],
        deleted: dict[int, tuple[_Entry[_V], _V]],
        replaced: list[tuple[str, _V]],
        reserve: int,
    ) -> None:
        for entry in items:
            found = False
            identity = entry.identity
            for slot, idx, e in self._keys.iter_hash(entry.hash):
                eid = id(e)
                if e.identity != identity or eid in updated:
                    continue
                if not found:
                    found = True
                    updated[eid] = e
                    if deleted:
                        # one an earlier item doomed keeps its position
                        deleted.pop(eid, None)
                    replaced.append((e.key, e.value))
                    e.key = entry.key
                    e.value = entry.value
                elif eid not in deleted:
                    # with what it holds now: a write since is kept
                    deleted[eid] = (e, e.value)
            if not found:
                if reserve:
                    self._reserve(reserve)
                    reserve = 0
                self._add_with_hash(entry)
                updated[id(entry)] = entry

    def _post_update(self, deleted: dict[int, tuple[_Entry[_V], _V]]) -> None:
        if deleted:
            keys = self._keys
            indices = keys.indices
            entries = keys.entries
            for slot in range(keys.nslots):
                idx = indices[slot]
                if idx < 0:
                    continue
                doomed = deleted.get(id(entries[idx]))
                if doomed is not None:
                    e, value = doomed
                    if e.value is value:
                        entries[idx] = None
                        indices[slot] = -2
                        self._used -= 1
        self._incr_version()

    @_locked_pair_always
    def merge(self, arg: MDArg[_V] = None, /, **kwargs: _V) -> None:
        """Merge into the dictionary, adding non-existing keys."""
        it = self._parse_args(arg, kwargs)
        reserve = self._reserve_batch(cast(int, next(it)))
        try:
            self._merge_items(cast(Iterator[_Entry[_V]], it), reserve)
        finally:
            self._incr_version()

    def _merge_items(self, items: Iterator[_Entry[_V]], reserve: int) -> None:
        # See update()
        added: dict[int, _Entry[_V]] = {}
        for entry in items:
            identity = entry.identity
            for slot, idx, e in self._keys.iter_hash(entry.hash):
                if e.identity == identity and id(e) not in added:
                    break
            else:
                if reserve:
                    self._reserve(reserve)
                    reserve = 0
                self._add_with_hash(entry)
                added[id(entry)] = entry

    if _FREE_THREADED:
        # `_version` is shared by every instance, so it needs its own lock.
        def _incr_version(self) -> None:
            with _VERSION_LOCK:
                v = next(_version)
            self._version = v

    else:
        # One C call, so the GIL cannot switch threads in the middle of it.
        def _incr_version(self) -> None:
            self._version = next(_version)

    def _reserve(self, extra: int) -> None:
        if self._keys.usable < extra:
            self._resize(estimate_log2_keysize(self._used + extra))

    def _reserve_batch(self, extra: int) -> int:
        # update() and merge() often find their keys there already: see
        # md_reserve_batch(). Returns the room their first add makes.
        keys = self._keys
        n = len(keys.entries)
        if n != self._used or n + keys.usable < extra:
            self._reserve(extra)
        return extra

    def _resize(self, log2_newsize: int) -> None:
        oldkeys = self._keys
        newentries = self._used

        if len(oldkeys.entries) == newentries:
            entries = oldkeys.entries
        else:
            entries = [e for e in oldkeys.entries if e is not None]
        newkeys: _HtKeys[_V] = _HtKeys.new(log2_newsize, entries)
        newkeys.maybe_dups = oldkeys.maybe_dups
        newkeys.usable -= newentries
        newkeys.build_indices()
        self._keys = newkeys

    def _add_with_hash(self, entry: _Entry[_V], absent: bool = False) -> None:
        if self._keys.usable <= 0:
            self._resize((self._used * 3 | _HtKeys.MINSIZE - 1).bit_length())
        keys = self._keys
        slot = keys.find_empty_slot(entry.hash, absent)
        keys.entries.append(entry)
        keys.indices[slot] = len(keys.entries) - 1
        self._incr_version()
        self._used += 1
        keys.usable -= 1

    def _del_at(self, slot: int, idx: int) -> None:
        self._keys.entries[idx] = None
        self._keys.indices[slot] = -2
        self._used -= 1


class CIMultiDict(_CIMixin, MultiDict[_V]):
    """Dictionary with the support for duplicate case-insensitive keys."""


class MultiDictProxy(_CSMixin, MultiMapping[_V]):
    """Read-only proxy for MultiDict instance."""

    __slots__ = ("_md",)

    _md: MultiDict[_V]

    def __init__(self, arg: "MultiDict[_V] | MultiDictProxy[_V]"):
        if not isinstance(arg, (MultiDict, MultiDictProxy)):
            raise TypeError(
                f"ctor requires MultiDict or MultiDictProxy instance, not {type(arg)}"
            )
        if isinstance(arg, MultiDictProxy):
            self._md = arg._md
        else:
            self._md = arg

    def __reduce__(self) -> NoReturn:
        raise TypeError(f"can't pickle {self.__class__.__name__} objects")

    @overload
    def getall(self, key: str) -> list[_V]: ...
    @overload
    def getall(self, key: str, default: _T) -> list[_V] | _T: ...
    def getall(self, key: str, default: _T | _SENTINEL = sentinel) -> list[_V] | _T:
        """Return a list of all values matching the key."""
        if default is not sentinel:
            return self._md.getall(key, default)
        else:
            return self._md.getall(key)

    @overload
    def getone(self, key: str) -> _V: ...
    @overload
    def getone(self, key: str, default: _T) -> _V | _T: ...
    def getone(self, key: str, default: _T | _SENTINEL = sentinel) -> _V | _T:
        """Get first value matching the key.

        Raises KeyError if the key is not found and no default is provided.
        """
        if default is not sentinel:
            return self._md.getone(key, default)
        else:
            return self._md.getone(key)

    # Mapping interface #

    def __getitem__(self, key: str) -> _V:
        return self.getone(key)

    @overload
    def get(self, key: str, /) -> _V | None: ...
    @overload
    def get(self, key: str, /, default: _T) -> _V | _T: ...
    def get(self, key: str, default: _T | None = None) -> _V | _T | None:
        """Get first value matching the key.

        If the key is not found, returns the default (or None if no default is provided)
        """
        return self._md.getone(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._md.keys())

    def __len__(self) -> int:
        return len(self._md)

    def keys(self) -> KeysView[str]:
        """Return a new view of the dictionary's keys."""
        return self._md.keys()

    def items(self) -> ItemsView[str, _V]:
        """Return a new view of the dictionary's items as ``(key, value)`` pairs."""
        return self._md.items()

    def values(self) -> _ValuesView[_V]:
        """Return a new view of the dictionary's values."""
        return self._md.values()

    def __eq__(self, other: object) -> bool:
        return self._md == other

    def __contains__(self, key: object) -> bool:
        return key in self._md

    @reprlib.recursive_repr()
    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"<{self.__class__.__name__}({body})>"

    def to_dict(self) -> dict[str, list[_V]]:
        """Return a dict with lists of all values for each key."""
        return self._md.to_dict()

    def copy(self) -> MultiDict[_V]:
        """Return a copy of itself."""
        return MultiDict(self._md)


class CIMultiDictProxy(_CIMixin, MultiDictProxy[_V]):
    """Read-only proxy for CIMultiDict instance."""

    def __init__(self, arg: MultiDict[_V] | MultiDictProxy[_V]):
        if not isinstance(arg, (CIMultiDict, CIMultiDictProxy)):
            raise TypeError(
                "ctor requires CIMultiDict or CIMultiDictProxy instance"
                f", not {type(arg)}"
            )

        super().__init__(arg)

    def copy(self) -> CIMultiDict[_V]:
        """Return a copy of itself."""
        return CIMultiDict(self._md)


def getversion(md: MultiDict[object] | MultiDictProxy[object]) -> int:
    if isinstance(md, MultiDictProxy):
        md = md._md
    elif not isinstance(md, MultiDict):
        raise TypeError("Parameter should be multidict or proxy")
    return md._version
<|endoftext|>import os
import platform

NO_EXTENSIONS = bool(os.environ.get("MULTIDICT_NO_EXTENSIONS"))

PYPY = platform.python_implementation() == "PyPy"

USE_EXTENSIONS = not NO_EXTENSIONS and not PYPY

if USE_EXTENSIONS:
    try:
        from . import _multidict  # type: ignore[attr-defined]  # noqa: F401
    except ImportError:  # pragma: no cover
        # FIXME: Refactor for coverage. See #837.
        USE_EXTENSIONS = False
<|endoftext|>"""
Multidict implementation.

HTTP Headers and URL query string require specific data structure:
multidict. It behaves mostly like a dict but it can have
several values for the same key.
"""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._abc import MultiMapping, MutableMultiMapping
from ._compat import USE_EXTENSIONS

__all__ = (
    "CIMultiDict",
    "CIMultiDictProxy",
    "MultiDict",
    "MultiDictProxy",
    "MultiMapping",
    "MutableMultiMapping",
    "get_include",
    "getversion",
    "istr",
)

__version__ = "7.1.0"


if TYPE_CHECKING or not USE_EXTENSIONS:
    from ._multidict_py import (
        CIMultiDict,
        CIMultiDictProxy,
        MultiDict,
        MultiDictProxy,
        getversion,
        istr,
    )
else:
    from collections.abc import ItemsView, KeysView, ValuesView

    from ._multidict import (
        CIMultiDict,
        CIMultiDictProxy,
        MultiDict,
        MultiDictProxy,
        _ItemsView,
        _KeysView,
        _ValuesView,
        getversion,
        istr,
    )

    MultiMapping.register(MultiDictProxy)
    MutableMultiMapping.register(MultiDict)
    KeysView.register(_KeysView)
    ItemsView.register(_ItemsView)
    ValuesView.register(_ValuesView)


if TYPE_CHECKING:
    upstr = istr
else:

    def __getattr__(name: str) -> Any:
        if name == "upstr":
            warnings.warn(
                "upstr is deprecated, use istr instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return istr
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return [*globals(), "upstr"]


def get_include() -> str:
    """Return the directory containing multidict's C API headers.

    Pass it as an ``-I`` include path when building a C extension
    against ``multidict_capi.h``.
    """
    return str(Path(__file__).absolute().parent)
<|endoftext|>import abc
from collections.abc import Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Protocol, TypeVar, Union, overload

if TYPE_CHECKING:
    from ._multidict_py import istr
else:
    istr = str

_V = TypeVar("_V")
_V_co = TypeVar("_V_co", covariant=True)
_T = TypeVar("_T")


class SupportsKeys(Protocol[_V_co]):
    def keys(self) -> Iterable[str]: ...
    def __getitem__(self, key: str, /) -> _V_co: ...


class SupportsIKeys(Protocol[_V_co]):
    def keys(self) -> Iterable[istr]: ...
    def __getitem__(self, key: istr, /) -> _V_co: ...


MDArg = SupportsKeys[_V] | SupportsIKeys[_V] | Iterable[tuple[str, _V]] | None


class MultiMapping(Mapping[str, _V_co]):
    @overload
    def getall(self, key: str) -> list[_V_co]: ...
    @overload
    def getall(self, key: str, default: _T) -> list[_V_co] | _T: ...
    @abc.abstractmethod
    def getall(self, key: str, default: _T = ...) -> list[_V_co] | _T:
        """Return all values for key."""

    @overload
    def getone(self, key: str) -> _V_co: ...
    @overload
    def getone(self, key: str, default: _T) -> _V_co | _T: ...
    @abc.abstractmethod
    def getone(self, key: str, default: _T = ...) -> _V_co | _T:
        """Return first value for key."""

    def to_dict(self) -> dict[str, list[_V_co]]:
        """Return a dict with lists of all values for each key.

        Deliberately concrete, not abstract: ``MultiMapping`` is public, so
        requiring a new method would stop every existing subclass outside
        this project from being instantiated. Subclasses whose keys compare
        equal under a normalisation the iteration does not apply, such as
        case-insensitive mappings, must override this; the default would
        emit one entry per spelling.
        """
        result: dict[str, list[_V_co]] = {}
        for key in self:
            # Iteration yields a duplicated key once per occurrence, so the
            # membership test is what keeps getall() from being applied twice.
            if key not in result:
                # getall() already returns a fresh list.
                result[key] = self.getall(key)
        return result


class MutableMultiMapping(MultiMapping[_V], MutableMapping[str, _V]):
    @abc.abstractmethod
    def add(self, key: str, value: _V) -> None:
        """Add value to list."""

    @abc.abstractmethod
    def extend(self, arg: MDArg[_V] = None, /, **kwargs: _V) -> None:
        """Add everything from arg and kwargs to the mapping."""

    @abc.abstractmethod
    def merge(self, arg: MDArg[_V] = None, /, **kwargs: _V) -> None:
        """Merge into the mapping, adding non-existing keys."""

    @overload
    def popone(self, key: str) -> _V: ...
    @overload
    def popone(self, key: str, default: _T) -> _V | _T: ...
    @abc.abstractmethod
    def popone(self, key: str, default: _T = ...) -> _V | _T:
        """Remove specified key and return the corresponding value."""

    @overload
    def popall(self, key: str) -> list[_V]: ...
    @overload
    def popall(self, key: str, default: _T) -> list[_V] | _T: ...
    @abc.abstractmethod
    def popall(self, key: str, default: _T = ...) -> list[_V] | _T:
        """Remove all occurrences of key and return the list of corresponding values."""
<|endoftext|># SPDX-License-Identifier: MIT

from attr import (
    NOTHING,
    Attribute,
    AttrsInstance,
    Converter,
    Factory,
    NothingType,
    _make_getattr,
    assoc,
    cmp_using,
    define,
    evolve,
    field,
    fields,
    fields_dict,
    frozen,
    has,
    make_class,
    mutable,
    resolve_types,
    validate,
)
from attr._make import ClassProps
from attr._next_gen import asdict, astuple, inspect

from . import converters, exceptions, filters, setters, validators


__all__ = [
    "NOTHING",
    "Attribute",
    "AttrsInstance",
    "ClassProps",
    "Converter",
    "Factory",
    "NothingType",
    "__author__",
    "__copyright__",
    "__description__",
    "__doc__",
    "__email__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "asdict",
    "assoc",
    "astuple",
    "cmp_using",
    "converters",
    "define",
    "evolve",
    "exceptions",
    "field",
    "fields",
    "fields_dict",
    "filters",
    "frozen",
    "has",
    "inspect",
    "make_class",
    "mutable",
    "resolve_types",
    "setters",
    "validate",
    "validators",
]

__getattr__ = _make_getattr(__name__)
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import warn

from .api import from_bytes
from .constant import CHARDET_CORRESPONDENCE, TOO_SMALL_SEQUENCE

if TYPE_CHECKING:
    from typing import TypedDict

    class ResultDict(TypedDict):
        encoding: str | None
        language: str
        confidence: float | None


def detect(
    byte_str: bytes, should_rename_legacy: bool = False, **kwargs: Any
) -> ResultDict:
    """
    chardet legacy method
    Detect the encoding of the given byte string. It should be mostly backward-compatible.
    Encoding name will match Chardet own writing whenever possible. (Not on encoding name unsupported by it)
    This function is deprecated and should be used to migrate your project easily, consult the documentation for
    further information. Not planned for removal.

    :param byte_str:     The byte sequence to examine.
    :param should_rename_legacy:  Should we rename legacy encodings
                                  to their more modern equivalents?
    """
    if len(kwargs):
        warn(
            f"charset-normalizer disregard arguments '{','.join(list(kwargs.keys()))}' in legacy function detect()"
        )

    if not isinstance(byte_str, (bytearray, bytes)):
        raise TypeError(  # pragma: nocover
            f"Expected object of type bytes or bytearray, got: {type(byte_str)}"
        )

    if isinstance(byte_str, bytearray):
        byte_str = bytes(byte_str)

    r = from_bytes(byte_str).best()

    encoding = r.encoding if r is not None else None
    language = r.language if r is not None and r.language != "Unknown" else ""
    confidence = 1.0 - r.chaos if r is not None else None

    # automatically lower confidence
    # on small bytes samples.
    # https://github.com/jawah/charset_normalizer/issues/391
    if (
        confidence is not None
        and confidence >= 0.9
        and encoding
        not in {
            "utf_8",
            "ascii",
        }
        and not r.bom  # type: ignore[union-attr]
        and len(byte_str) < TOO_SMALL_SEQUENCE
    ):
        confidence -= 0.2

    # Note: CharsetNormalizer does not return 'UTF-8-SIG' as the sig get stripped in the detection/normalization process
    # but chardet does return 'utf-8-sig' and it is a valid codec name.
    if r is not None and encoding == "utf_8" and r.bom:
        encoding += "_sig"

    if not should_rename_legacy and encoding in CHARDET_CORRESPONDENCE:
        encoding = CHARDET_CORRESPONDENCE[encoding]

    return {
        "encoding": encoding,
        "language": language,
        "confidence": confidence,
    }
<|endoftext|>from __future__ import annotations

from functools import lru_cache
from logging import getLogger


from .constant import (
    COMMON_CJK_CHARACTERS,
    COMMON_SAFE_ASCII_CHARACTERS,
    TRACE,
    CompatibleFamillyRange,
    _ACCENTUATED,
    _ARABIC,
    _ARABIC_ISOLATED_FORM,
    _BASIC_LATIN_COMPATIBLE_RANGE_FAMILIES,
    _CJK,
    _COMPATIBLE_RANGE_FAMILIES,
    _COMPATIBLE_WITH_ANY_RANGE_FAMILIES,
    _HANGUL,
    _HALFWIDTH_KATAKANA,
    _HIRAGANA,
    _KATAKANA,
    _LATIN,
    _LIGATURE,
    _RANGE_FAMILIES,
    _SENTENCE_OPEN_PUNCTUATION,
    _SUPERSCRIPT,
    _THAI,
)
from .utils import (
    _character_flags,
    is_emoticon,
    is_punctuation,
    is_separator,
    is_symbol,
    remove_accent,
    unicode_range,
)

# Combined bitmask for CJK/Hangul/Katakana/Hiragana/Thai glyph detection.
_GLYPH_MASK: int = _CJK | _HANGUL | _KATAKANA | _HIRAGANA | _THAI


class CharInfo:
    """Pre-computed character properties shared across all detectors."""

    __slots__ = (
        "character",
        "printable",
        "alpha",
        "upper",
        "lower",
        "space",
        "digit",
        "is_ascii",
        "case_variable",
        "flags",
        "accentuated",
        "latin",
        "is_cjk",
        "is_katakana",
        "is_halfwidth_katakana",
        "is_arabic",
        "is_ligature",
        "is_superscript",
        "is_sentence_open_punctuation",
        "is_glyph",
        "punct",
        "sym",
        "range",
        "sep",
        "emoticon",
        "safe",
        "common_cjk",
        "unaccented",
    )

    character: str
    printable: bool
    alpha: bool
    upper: bool
    lower: bool
    space: bool
    digit: bool
    is_ascii: bool
    case_variable: bool
    flags: int
    accentuated: bool
    latin: bool
    is_cjk: bool
    is_katakana: bool
    is_halfwidth_katakana: bool
    is_arabic: bool
    is_ligature: bool
    is_superscript: bool
    is_sentence_open_punctuation: bool
    is_glyph: bool
    punct: bool
    sym: bool
    range: str | None
    sep: bool
    emoticon: bool
    safe: bool
    common_cjk: bool
    unaccented: str

    def __init__(self, character: str) -> None:
        """Compute all properties for *character* (built once per codepoint,
        every branch assigns every slot)."""
        self.character = character

        # ASCII fast-path: for characters with ord < 128, we can skip
        # _character_flags() entirely and derive most properties from ord.
        o: int = ord(character)
        if o < 128:
            self.is_ascii = True
            self.accentuated = False
            self.unaccented = character
            self.emoticon = False
            self.common_cjk = False
            self.safe = character in COMMON_SAFE_ASCII_CHARACTERS
            self.is_cjk = False
            self.is_katakana = False
            self.is_halfwidth_katakana = False
            self.is_arabic = False
            self.is_ligature = False
            self.is_superscript = False
            self.is_sentence_open_punctuation = False
            self.is_glyph = False
            # ASCII alpha: a-z (97-122) or A-Z (65-90)
            if 65 <= o <= 90:
                # Uppercase ASCII letter
                self.alpha = True
                self.upper = True
                self.lower = False
                self.space = False
                self.digit = False
                self.printable = True
                self.case_variable = True
                self.flags = _LATIN
                self.latin = True
                self.punct = False
                self.sym = False
            elif 97 <= o <= 122:
                # Lowercase ASCII letter
                self.alpha = True
                self.upper = False
                self.lower = True
                self.space = False
                self.digit = False
                self.printable = True
                self.case_variable = True
                self.flags = _LATIN
                self.latin = True
                self.punct = False
                self.sym = False
            elif 48 <= o <= 57:
                # ASCII digit 0-9
                self.alpha = False
                self.upper = False
                self.lower = False
                self.space = False
                self.digit = True
                self.printable = True
                self.case_variable = False
                self.flags = 0
                self.latin = False
                self.punct = False
                self.sym = False
            elif o == 32 or (9 <= o <= 13):
                # Space, tab, newline, etc.
                self.alpha = False
                self.upper = False
                self.lower = False
                self.space = True
                self.digit = False
                self.printable = o == 32
                self.case_variable = False
                self.flags = 0
                self.latin = False
                self.punct = False
                self.sym = False
            else:
                # Other ASCII (punctuation, symbols, control chars)
                self.printable = character.isprintable()
                self.alpha = False
                self.upper = False
                self.lower = False
                self.space = False
                self.digit = False
                self.case_variable = False
                self.flags = 0
                self.latin = False
                self.punct = is_punctuation(character) if self.printable else False
                self.sym = is_symbol(character) if self.printable else False
        else:
            # Non-ASCII path
            self.is_ascii = False
            self.safe = False
            self.printable = character.isprintable()
            self.alpha = character.isalpha()
            self.upper = character.isupper()
            self.lower = character.islower()
            self.space = character.isspace()
            self.digit = character.isdigit()
            self.case_variable = self.lower != self.upper

            # Flag-based classification (single unicodedata.name() call, lru-cached)
            flags: int = _character_flags(character)
            if self.alpha:
                self.emoticon = False
            else:
                self.emoticon = is_emoticon(character)
            self.flags = flags
            self.accentuated = bool(flags & _ACCENTUATED)
            self.latin = bool(flags & _LATIN)
            self.is_cjk = bool(flags & _CJK)
            self.is_katakana = bool(flags & _KATAKANA)
            self.is_halfwidth_katakana = bool(flags & _HALFWIDTH_KATAKANA)
            self.is_arabic = bool(flags & _ARABIC)
            self.is_ligature = bool(flags & _LIGATURE)
            self.is_superscript = bool(flags & _SUPERSCRIPT)
            self.is_sentence_open_punctuation = bool(flags & _SENTENCE_OPEN_PUNCTUATION)
            self.is_glyph = bool(flags & _GLYPH_MASK)

            if self.latin and self.accentuated:
                self.unaccented = remove_accent(character)
            else:
                self.unaccented = character

            self.common_cjk = self.is_cjk and character in COMMON_CJK_CHARACTERS

            # Eagerly compute punct and sym (avoids property dispatch overhead
            # on 300K+ accesses in the hot loop).
            if self.printable:
                self.punct = is_punctuation(character)
                self.sym = is_symbol(character)
            else:
                self.punct = False
                self.sym = False

        self.range = unicode_range(character)
        self.sep = is_separator(character)


# Per-codepoint cache of CharInfo instances
# At most UTF-8 size allocated.
@lru_cache(maxsize=None)
def _char_info(character: str) -> CharInfo:
    """Build (once per codepoint) and cache the CharInfo for *character*."""
    return CharInfo(character)


# ASCII table indexed by codepoint.
_ASCII_CHAR_INFO: list[CharInfo] = [
    CharInfo(chr(_codepoint)) for _codepoint in range(128)
]


class MessDetectorPlugin:
    """
    Base abstract class used for mess detection plugins.
    All detectors MUST extend and implement given methods.
    """

    __slots__ = ()

    def feed_info(self, character: str, info: CharInfo) -> None:
        """
        The main routine to be executed upon character.
        Insert the logic in witch the text would be considered chaotic.
        """
        raise NotImplementedError  # Defensive:

    def reset(self) -> None:  # Defensive:
        """
        Permit to reset the plugin to the initial state.
        """
        raise NotImplementedError

    @property
    def ratio(self) -> float:
        """
        Compute the chaos ratio based on what your feed() has seen.
        Must NOT be lower than 0.; No restriction gt 0.
        """
        raise NotImplementedError  # Defensive:


class TooManySymbolOrPunctuationPlugin(MessDetectorPlugin):
    __slots__ = (
        "_punctuation_count",
        "_symbol_count",
        "_character_count",
        "_last_printable_char",
    )

    def __init__(self) -> None:
        self._punctuation_count: int = 0
        self._symbol_count: int = 0
        self._character_count: int = 0

        self._last_printable_char: str | None = None

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        self._character_count += 1

        if character != self._last_printable_char and not info.safe:
            if info.punct:
                self._punctuation_count += 1
            elif not info.digit and info.sym and not info.emoticon:
                self._symbol_count += 2

        self._last_printable_char = character

    def reset(self) -> None:  # Abstract
        self._punctuation_count = 0
        self._character_count = 0
        self._symbol_count = 0

    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0

        ratio_of_punctuation: float = (
            self._punctuation_count + self._symbol_count
        ) / self._character_count

        return ratio_of_punctuation if ratio_of_punctuation >= 0.3 else 0.0


class TooManyAccentuatedPlugin(MessDetectorPlugin):
    __slots__ = ("_character_count", "_accentuated_count")

    def __init__(self) -> None:
        self._character_count: int = 0
        self._accentuated_count: int = 0

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        self._character_count += 1

        if info.accentuated:
            self._accentuated_count += 1

    def reset(self) -> None:  # Abstract
        self._character_count = 0
        self._accentuated_count = 0

    @property
    def ratio(self) -> float:
        if self._character_count < 8:
            return 0.0

        ratio_of_accentuation: float = self._accentuated_count / self._character_count
        return ratio_of_accentuation if ratio_of_accentuation >= 0.35 else 0.0


class UnprintablePlugin(MessDetectorPlugin):
    __slots__ = ("_unprintable_count", "_character_count", "_has_escape")

    def __init__(self) -> None:
        self._unprintable_count: int = 0
        self._character_count: int = 0
        self._has_escape: bool = False

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        if character == "\x1b":
            self._has_escape = True

        if (
            not info.printable
            and not info.space
            and character != "\x1a"
            and character != "\ufeff"
        ):
            self._unprintable_count += 1
        self._character_count += 1

    def reset(self) -> None:  # Abstract
        self._unprintable_count = 0
        self._has_escape = False

    @property
    def ratio(self) -> float:
        if self._character_count == 0:  # Defensive:
            return 0.0

        if self._has_escape:
            return 1.0

        return (self._unprintable_count * 8) / self._character_count


class SuspiciousDuplicateAccentPlugin(MessDetectorPlugin):
    __slots__ = (
        "_successive_count",
        "_character_count",
        "_last_latin_character",
        "_last_was_accentuated",
    )

    def __init__(self) -> None:
        self._successive_count: int = 0
        self._character_count: int = 0

        self._last_latin_character: CharInfo | None = None
        self._last_was_accentuated: bool = False

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        self._character_count += 1
        if (
            self._last_latin_character is not None
            and info.accentuated
            and self._last_was_accentuated
        ):
            if info.upper and self._last_latin_character.upper:
                self._successive_count += 1
            if info.unaccented == self._last_latin_character.unaccented:
                self._successive_count += 1
        self._last_latin_character = info
        self._last_was_accentuated = info.accentuated

    def reset(self) -> None:  # Abstract
        self._successive_count = 0
        self._character_count = 0
        self._last_latin_character = None
        self._last_was_accentuated = False

    @property
    def ratio(self) -> float:
        if self._character_count == 0:
            return 0.0

        return (self._successive_count * 2) / self._character_count


class SuspiciousRange(MessDetectorPlugin):
    __slots__ = (
        "_suspicious_successive_range_count",
        "_character_count",
        "_last_printable_seen",
        "_last_printable_range",
    )

    def __init__(self) -> None:
        self._suspicious_successive_range_count: int = 0
        self._character_count: int = 0
        self._last_printable_seen: str | None = None
        self._last_printable_range: str | None = None

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        self._character_count += 1

        if info.space or info.punct or info.safe:
            self._last_printable_seen = None
            self._last_printable_range = None
            return

        if self._last_printable_seen is None:
            self._last_printable_seen = character
            self._last_printable_range = info.range
            return

        unicode_range_a: str | None = self._last_printable_range
        unicode_range_b: str | None = info.range

        # Identical non-None ranges can never be suspicious.
        if unicode_range_a != unicode_range_b or unicode_range_a is None:
            if is_suspiciously_successive_range(unicode_range_a, unicode_range_b):
                self._suspicious_successive_range_count += 1

        self._last_printable_seen = character
        self._last_printable_range = unicode_range_b

    def reset(self) -> None:  # Abstract
        self._character_count = 0
        self._suspicious_successive_range_count = 0
        self._last_printable_seen = None
        self._last_printable_range = None

    @property
    def ratio(self) -> float:
        if self._character_count <= 13:
            return 0.0

        ratio_of_suspicious_range_usage: float = (
            self._suspicious_successive_range_count * 2
        ) / self._character_count

        return ratio_of_suspicious_range_usage


class SuperWeirdWordPlugin(MessDetectorPlugin):
    __slots__ = (
        "_word_count",
        "_bad_word_count",
        "_foreign_long_count",
        "_is_current_word_bad",
        "_foreign_long_watch",
        "_character_count",
        "_bad_character_count",
        "_buffer_length",
        "_buffer_last_char",
        "_buffer_last_char_accentuated",
        "_buffer_accent_count",
        "_buffer_glyph_count",
        "_buffer_upper_count",
        "_buffer_first_lower",
        "_buffer_has_non_ascii",
        "_buffer_last_char_ligature",
        "_buffer_has_internal_ligature",
        "_is_current_word_invalid",
        "_invalid_word_count",
    )

    def __init__(self) -> None:
        self._word_count: int = 0
        self._bad_word_count: int = 0
        self._foreign_long_count: int = 0

        self._is_current_word_bad: bool = False
        self._foreign_long_watch: bool = False

        self._character_count: int = 0
        self._bad_character_count: int = 0

        self._buffer_length: int = 0
        self._buffer_last_char: str | None = None
        self._buffer_last_char_accentuated: bool = False
        self._buffer_accent_count: int = 0
        self._buffer_glyph_count: int = 0
        self._buffer_upper_count: int = 0
        self._buffer_first_lower: bool = False
        self._buffer_has_non_ascii: bool = False
        self._buffer_last_char_ligature: bool = False
        self._buffer_has_internal_ligature: bool = False
        self._is_current_word_invalid: bool = False
        self._invalid_word_count: int = 0

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        if info.alpha:
            if self._buffer_last_char_ligature:
                self._buffer_has_internal_ligature = True
            self._buffer_last_char_ligature = info.is_ligature
            if self._buffer_length == 0:
                self._buffer_first_lower = info.lower
            self._buffer_length += 1
            self._buffer_last_char = character

            if info.upper:
                self._buffer_upper_count += 1
            if not info.is_ascii:
                self._buffer_has_non_ascii = True

            self._buffer_last_char_accentuated = info.accentuated

            if info.accentuated:
                self._buffer_accent_count += 1
            if info.is_glyph:
                self._buffer_glyph_count += 1
            elif not self._foreign_long_watch and (not info.latin or info.accentuated):
                self._foreign_long_watch = True
            return
        if not self._buffer_length:
            return
        if info.is_sentence_open_punctuation or (
            info.is_superscript and self._buffer_has_internal_ligature
        ):
            self._is_current_word_bad = True
            self._is_current_word_invalid = True
        if info.space or info.punct or info.sep:
            self._word_count += 1
            buffer_length: int = self._buffer_length

            self._character_count += buffer_length

            if buffer_length >= 4:
                if self._buffer_accent_count / buffer_length >= 0.5:
                    self._is_current_word_bad = True
                elif (
                    self._buffer_last_char_accentuated
                    and self._buffer_last_char.isupper()  # type: ignore[union-attr]
                    and self._buffer_upper_count != buffer_length
                ):
                    self._foreign_long_count += 1
                    self._is_current_word_bad = True
                elif self._buffer_glyph_count == 1:
                    self._is_current_word_bad = True
                    self._foreign_long_count += 1
                elif (
                    self._buffer_has_non_ascii
                    and self._buffer_first_lower
                    and self._buffer_upper_count == buffer_length - 1
                ):
                    # Inverse capitalization detector.
                    # No natural writing produces such words.
                    # see https://github.com/jawah/charset_normalizer/issues/731
                    self._foreign_long_count += 1
                    self._is_current_word_bad = True
            if buffer_length >= 24 and self._foreign_long_watch:
                probable_camel_cased: bool = (
                    self._buffer_upper_count > 0
                    and self._buffer_upper_count / buffer_length <= 0.3
                )

                if not probable_camel_cased:
                    self._foreign_long_count += 1
                    self._is_current_word_bad = True

            if self._is_current_word_bad:
                self._bad_word_count += 1
                self._bad_character_count += buffer_length
                self._is_current_word_bad = False
            if self._is_current_word_invalid:
                self._invalid_word_count += 1
                self._is_current_word_invalid = False

            self._foreign_long_watch = False
            self._buffer_length = 0
            self._buffer_last_char = None
            self._buffer_last_char_accentuated = False
            self._buffer_accent_count = 0
            self._buffer_glyph_count = 0
            self._buffer_upper_count = 0
            self._buffer_first_lower = False
            self._buffer_has_non_ascii = False
            self._buffer_last_char_ligature = False
            self._buffer_has_internal_ligature = False
        elif (
            character not in {"<", ">", "-", "=", "~", "|", "_"}
            and not info.digit
            and info.sym
        ):
            self._is_current_word_bad = True
            self._buffer_length += 1
            self._buffer_last_char = character
            self._buffer_last_char_accentuated = False

    def reset(self) -> None:  # Abstract
        self._buffer_length = 0
        self._buffer_last_char = None
        self._buffer_last_char_accentuated = False
        self._is_current_word_bad = False
        self._foreign_long_watch = False
        self._bad_word_count = 0
        self._word_count = 0
        self._character_count = 0
        self._bad_character_count = 0
        self._foreign_long_count = 0
        self._buffer_accent_count = 0
        self._buffer_glyph_count = 0
        self._buffer_upper_count = 0
        self._buffer_first_lower = False
        self._buffer_has_non_ascii = False
        self._buffer_last_char_ligature = False
        self._buffer_has_internal_ligature = False
        self._is_current_word_invalid = False
        self._invalid_word_count = 0

    @property
    def ratio(self) -> float:
        if self._invalid_word_count:
            return 1.0

        if self._word_count <= 10 and self._foreign_long_count == 0:
            return 0.0

        return self._bad_character_count / self._character_count


class CjkUncommonPlugin(MessDetectorPlugin):
    """
    Detect messy CJK text that probably means nothing.
    """

    __slots__ = ("_character_count", "_uncommon_count")

    def __init__(self) -> None:
        self._character_count: int = 0
        self._uncommon_count: int = 0

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        self._character_count += 1

        if not info.common_cjk:
            self._uncommon_count += 1

    def reset(self) -> None:  # Abstract
        self._character_count = 0
        self._uncommon_count = 0

    @property
    def ratio(self) -> float:
        if self._character_count < 4:
            return 0.0

        uncommon_form_usage: float = (
            2 * self._uncommon_count - self._character_count
        ) / (5 * max(self._character_count, 16))

        # we can be pretty sure it's garbage when uncommon characters are widely
        # used. otherwise it could just be traditional chinese for example.
        return max(0.0, uncommon_form_usage)


class SuspiciousKatakanaPlugin(MessDetectorPlugin):
    """Detect implausible halfwidth Katakana and uncommon CJK combinations."""

    __slots__ = (
        "_katakana_count",
        "_halfwidth_katakana_count",
        "_cjk_count",
        "_uncommon_cjk_count",
    )

    def __init__(self) -> None:
        self._katakana_count: int = 0
        self._halfwidth_katakana_count: int = 0
        self._cjk_count: int = 0
        self._uncommon_cjk_count: int = 0

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        if info.is_katakana:
            self._katakana_count += 1
            if info.is_halfwidth_katakana:
                self._halfwidth_katakana_count += 1
            return

        self._cjk_count += 1
        if not info.common_cjk:
            self._uncommon_cjk_count += 1

    def reset(self) -> None:  # Abstract
        self._katakana_count = 0
        self._halfwidth_katakana_count = 0
        self._cjk_count = 0
        self._uncommon_cjk_count = 0

    @property
    def ratio(self) -> float:
        if (
            self._halfwidth_katakana_count >= 4
            and self._halfwidth_katakana_count == self._katakana_count
            and 3 <= self._cjk_count == self._uncommon_cjk_count
        ):
            return 1.0

        return 0.0


class ArchaicUpperLowerPlugin(MessDetectorPlugin):
    __slots__ = (
        "_buf",
        "_character_count_since_last_sep",
        "_successive_upper_lower_count",
        "_successive_upper_lower_count_final",
        "_character_count",
        "_last_alpha_seen_upper",
        "_last_alpha_seen_lower",
        "_current_ascii_only",
    )

    def __init__(self) -> None:
        self._buf: bool = False

        self._character_count_since_last_sep: int = 0

        self._successive_upper_lower_count: int = 0
        self._successive_upper_lower_count_final: int = 0

        self._character_count: int = 0

        self._last_alpha_seen_upper: bool = False
        self._last_alpha_seen_lower: bool = False
        self._current_ascii_only: bool = True

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        is_concerned: bool = info.alpha and info.case_variable
        chunk_sep: bool = not is_concerned

        if chunk_sep and self._character_count_since_last_sep > 0:
            if (
                self._character_count_since_last_sep <= 64
                and not info.digit
                and not self._current_ascii_only
            ):
                self._successive_upper_lower_count_final += (
                    self._successive_upper_lower_count
                )

            self._successive_upper_lower_count = 0
            self._character_count_since_last_sep = 0
            self._buf = False
            self._character_count += 1
            self._current_ascii_only = True

            return

        if self._current_ascii_only and not info.is_ascii:
            self._current_ascii_only = False

        if self._character_count_since_last_sep > 0:
            if (info.upper and self._last_alpha_seen_lower) or (
                info.lower and self._last_alpha_seen_upper
            ):
                if self._buf:
                    self._successive_upper_lower_count += 2
                    self._buf = False
                else:
                    self._buf = True
            else:
                self._buf = False

        self._character_count += 1
        self._character_count_since_last_sep += 1
        self._last_alpha_seen_upper = info.upper
        self._last_alpha_seen_lower = info.lower

    def reset(self) -> None:  # Abstract
        self._character_count = 0
        self._character_count_since_last_sep = 0
        self._successive_upper_lower_count = 0
        self._successive_upper_lower_count_final = 0
        self._last_alpha_seen_upper = False
        self._last_alpha_seen_lower = False
        self._buf = False
        self._current_ascii_only = True

    @property
    def ratio(self) -> float:
        if self._character_count == 0:  # Defensive:
            return 0.0

        return self._successive_upper_lower_count_final / self._character_count


class ArabicIsolatedFormPlugin(MessDetectorPlugin):
    __slots__ = ("_character_count", "_isolated_form_count")

    def __init__(self) -> None:
        self._character_count: int = 0
        self._isolated_form_count: int = 0

    def reset(self) -> None:  # Abstract
        self._character_count = 0
        self._isolated_form_count = 0

    def feed_info(self, character: str, info: CharInfo) -> None:
        """Optimized feed using pre-computed character info."""
        self._character_count += 1

        if info.flags & _ARABIC_ISOLATED_FORM:
            self._isolated_form_count += 1

    @property
    def ratio(self) -> float:
        if self._character_count < 8:
            return 0.0

        isolated_form_usage: float = self._isolated_form_count / self._character_count

        return isolated_form_usage


@lru_cache(maxsize=None)
def is_suspiciously_successive_range(
    unicode_range_a: str | None, unicode_range_b: str | None
) -> bool:
    """
    Determine if two Unicode ranges seen next to each other can be considered suspicious.
    """
    if unicode_range_a is None or unicode_range_b is None:
        return True

    familly_a: str = _RANGE_FAMILIES[unicode_range_a]
    familly_b: str = _RANGE_FAMILIES[unicode_range_b]

    if familly_a == familly_b:
        return False

    if (
        familly_a in _COMPATIBLE_WITH_ANY_RANGE_FAMILIES
        or familly_b in _COMPATIBLE_WITH_ANY_RANGE_FAMILIES
    ):
        return False

    if CompatibleFamillyRange(familly_a, familly_b) in _COMPATIBLE_RANGE_FAMILIES:
        return False

    # Basic Latin is commonly interspersed with East Asian scripts, but the
    # compatibility must not extend to every range in the Latin family.
    if unicode_range_a == "Basic Latin":
        return familly_b not in _BASIC_LATIN_COMPATIBLE_RANGE_FAMILIES

    if unicode_range_b == "Basic Latin":
        return familly_a not in _BASIC_LATIN_COMPATIBLE_RANGE_FAMILIES

    return True


def mess_ratio(
    decoded_sequence: str, maximum_threshold: float = 0.2, debug: bool = False
) -> float:
    """
    Compute a mess ratio given a decoded bytes sequence. The maximum threshold does stop the computation earlier.
    """

    seq_len: int = len(decoded_sequence)

    if seq_len < 511:
        step: int = 32
    elif seq_len < 1024:
        step = 64
    else:
        step = 128

    # str.isascii() is O(1) (the flag lives in the str header). Seven of the
    # ten detectors provably keep a 0.0 ratio on ASCII-only input and are
    # therefore not fed at all.
    is_pure_ascii: bool = decoded_sequence.isascii()

    # Cached per-codepoint character properties (see CharInfo). ASCII
    # characters resolve through the immutable import-time table; anything
    # else goes through the lru_cache-backed slow path.
    ascii_info = _ASCII_CHAR_INFO
    char_info = _char_info

    mean_mess_ratio: float
    info: CharInfo

    # Create each detector as a named local variable (unrolled from the generic loop).
    # This eliminates per-character iteration over the detector list and
    # per-character eligible() virtual dispatch, while keeping every plugin class
    # intact and fully readable.
    d_sp: TooManySymbolOrPunctuationPlugin = TooManySymbolOrPunctuationPlugin()
    d_ta: TooManyAccentuatedPlugin = TooManyAccentuatedPlugin()
    d_up: UnprintablePlugin = UnprintablePlugin()
    d_sda: SuspiciousDuplicateAccentPlugin = SuspiciousDuplicateAccentPlugin()
    d_sr: SuspiciousRange = SuspiciousRange()
    d_sw: SuperWeirdWordPlugin = SuperWeirdWordPlugin()
    d_cu: CjkUncommonPlugin = CjkUncommonPlugin()
    d_sk: SuspiciousKatakanaPlugin = SuspiciousKatakanaPlugin()
    d_au: ArchaicUpperLowerPlugin = ArchaicUpperLowerPlugin()
    d_ai: ArabicIsolatedFormPlugin = ArabicIsolatedFormPlugin()

    # Local references avoid repeated bound-method creation in the hot loop.
    d_sp_feed = d_sp.feed_info
    d_ta_feed = d_ta.feed_info
    d_up_feed = d_up.feed_info
    d_sda_feed = d_sda.feed_info
    d_sr_feed = d_sr.feed_info
    d_sw_feed = d_sw.feed_info
    d_cu_feed = d_cu.feed_info
    d_sk_feed = d_sk.feed_info
    d_au_feed = d_au.feed_info
    d_ai_feed = d_ai.feed_info

    for block_start in range(0, seq_len, step):
        for character in decoded_sequence[block_start : block_start + step]:
            # Character properties computed once per distinct codepoint
            # (shared across all plugins and all mess_ratio calls).
            # ord() doubles as the ASCII table index.
            codepoint: int = ord(character)
            if codepoint < 128:
                info = ascii_info[codepoint]
            else:
                info = char_info(character)

            # Detectors with eligible() == always True
            d_up_feed(character, info)
            d_sw_feed(character, info)

            if is_pure_ascii:
                # The seven remaining detectors provably stay at 0.0 (see above).
                if info.printable:
                    d_sp_feed(character, info)
                continue

            d_au_feed(character, info)

            # Detectors with eligible() == isprintable
            if info.printable:
                d_sp_feed(character, info)
                d_sr_feed(character, info)

            # Detectors with eligible() == isalpha
            if info.alpha:
                d_ta_feed(character, info)
                # SuspiciousDuplicateAccent: isalpha() and is_latin()
                if info.latin:
                    d_sda_feed(character, info)
                # CjkUncommon and SuspiciousKatakana: is_cjk()
                if info.is_cjk:
                    d_cu_feed(character, info)
                    d_sk_feed(character, info)
                elif info.is_katakana:
                    d_sk_feed(character, info)
                # ArabicIsolatedForm: is_arabic()
                if info.is_arabic:
                    d_ai_feed(character, info)

        mean_mess_ratio = (
            d_sp.ratio
            + d_ta.ratio
            + d_up.ratio
            + d_sda.ratio
            + d_sr.ratio
            + d_sw.ratio
            + d_cu.ratio
            + d_sk.ratio
            + d_au.ratio
            + d_ai.ratio
        )

        if mean_mess_ratio >= maximum_threshold:
            break
    else:
        # Flush last word buffer in SuperWeirdWordPlugin via trailing newline.
        nl_info = ascii_info[10]  # "\n"
        d_sw_feed("\n", nl_info)
        if not is_pure_ascii:
            d_au_feed("\n", nl_info)
        d_up_feed("\n", nl_info)

        mean_mess_ratio = (
            d_sp.ratio
            + d_ta.ratio
            + d_up.ratio
            + d_sda.ratio
            + d_sr.ratio
            + d_sw.ratio
            + d_cu.ratio
            + d_sk.ratio
            + d_au.ratio
            + d_ai.ratio
        )

    if debug:  # Defensive:
        logger = getLogger("charset_normalizer")

        logger.log(
            TRACE,
            "Mess-detector extended-analysis start. "
            f"intermediary_mean_mess_ratio_calc={step} mean_mess_ratio={mean_mess_ratio} "
            f"maximum_threshold={maximum_threshold}",
        )

        if seq_len > 16:
            logger.log(TRACE, f"Starting with: {decoded_sequence[:16]}")
            logger.log(TRACE, f"Ending with: {decoded_sequence[-16::]}")

        for dt in [d_sp, d_ta, d_up, d_sda, d_sr, d_sw, d_cu, d_sk, d_au, d_ai]:
            logger.log(TRACE, f"{dt.__class__}: {dt.ratio}")

    return round(mean_mess_ratio, 3)
<|endoftext|>from __future__ import annotations

import importlib
import logging
import unicodedata
from bisect import bisect_right
from codecs import IncrementalDecoder
from functools import lru_cache
from typing import Generator

from .constant import (
    ENCODING_MARKS,
    IANA_SUPPORTED_SIMILAR,
    RE_POSSIBLE_ENCODING_INDICATION,
    UNICODE_RANGES_COMBINED,
    _SECONDARY_RANGE_NAMES,
    COMMON_CJK_CHARACTERS,
    _LATIN,
    _CJK,
    _HANGUL,
    _KATAKANA,
    _HIRAGANA,
    _HALFWIDTH_KATAKANA,
    _THAI,
    _ARABIC,
    _ARABIC_ISOLATED_FORM,
    _LIGATURE,
    _SUPERSCRIPT,
    _SENTENCE_OPEN_PUNCTUATION,
    _ACCENT_KEYWORDS,
    _ACCENTUATED,
    _KNOWN_MB_DECODERS,
    _KNOWN_MB_CLASSES,
    _IANA_NAMES,
    _MULTIBYTE_SEARCH_RADIUS,
)


def _character_flags(character: str) -> int:
    """Compute all name-based classification flags with a single unicodedata.name() call."""
    try:
        desc: str = unicodedata.name(character)
    except ValueError:
        return 0

    flags: int = 0

    if "LATIN" in desc:
        flags |= _LATIN
    if "CJK" in desc:
        flags |= _CJK
    if "HANGUL" in desc:
        flags |= _HANGUL
    if "KATAKANA" in desc:
        flags |= _KATAKANA
        if "HALFWIDTH" in desc:
            flags |= _HALFWIDTH_KATAKANA
    if "HIRAGANA" in desc:
        flags |= _HIRAGANA
    if "THAI" in desc:
        flags |= _THAI
    if "ARABIC" in desc:
        flags |= _ARABIC
        if "ISOLATED FORM" in desc:
            flags |= _ARABIC_ISOLATED_FORM
    if "LIGATURE" in desc or desc.endswith("LETTER AE"):
        flags |= _LIGATURE
    if "SUPERSCRIPT" in desc:
        flags |= _SUPERSCRIPT
    if desc in {"INVERTED QUESTION MARK", "INVERTED EXCLAMATION MARK"}:
        flags |= _SENTENCE_OPEN_PUNCTUATION

    for kw in _ACCENT_KEYWORDS:
        if kw in desc:
            flags |= _ACCENTUATED
            break

    return flags


def is_accentuated(character: str) -> bool:
    return bool(_character_flags(character) & _ACCENTUATED)


def remove_accent(character: str) -> str:
    decomposed: str = unicodedata.decomposition(character)
    if not decomposed:
        return character

    codes: list[str] = decomposed.split(" ")

    return chr(int(codes[0], 16))


# Pre-built sorted lookup table for O(log n) binary search in unicode_range().
# Each entry is (range_start, range_end_exclusive, range_name).
_UNICODE_RANGES_SORTED: list[tuple[int, int, str]] = sorted(
    (ord_range.start, ord_range.stop, name)
    for name, ord_range in UNICODE_RANGES_COMBINED.items()
)
_UNICODE_RANGE_STARTS: list[int] = [e[0] for e in _UNICODE_RANGES_SORTED]


def unicode_range(character: str) -> str | None:
    """
    Retrieve the Unicode range official name from a single character.
    """
    character_ord: int = ord(character)

    if character_ord < 32:
        return "Control character"
    if character_ord < 128:
        return "Basic Latin"

    # Binary search: find the rightmost range whose start <= character_ord
    idx = bisect_right(_UNICODE_RANGE_STARTS, character_ord) - 1
    if idx >= 0:
        start, stop, name = _UNICODE_RANGES_SORTED[idx]
        if character_ord < stop:
            return name

    return None


def is_latin(character: str) -> bool:
    return bool(_character_flags(character) & _LATIN)


def is_punctuation(character: str) -> bool:
    character_category: str = unicodedata.category(character)

    if "P" in character_category:
        return True

    character_range: str | None = unicode_range(character)

    if character_range is None:
        return False

    return "Punctuation" in character_range


def is_symbol(character: str) -> bool:
    character_category: str = unicodedata.category(character)

    if "S" in character_category or "N" in character_category:
        return True

    character_range: str | None = unicode_range(character)

    if character_range is None:
        return False

    return "Forms" in character_range and character_category != "Lo"


def is_emoticon(character: str) -> bool:
    character_range: str | None = unicode_range(character)

    if character_range is None:
        return False

    return "Emoticons" in character_range or "Pictographs" in character_range


def is_separator(character: str) -> bool:
    if character.isspace() or character in {"｜", "+", "<", ">"}:
        return True

    character_category: str = unicodedata.category(character)

    return "Z" in character_category or character_category in {"Po", "Pd", "Pc"}


def is_case_variable(character: str) -> bool:
    return character.islower() != character.isupper()


def is_cjk(character: str) -> bool:
    return bool(_character_flags(character) & _CJK)


def is_hiragana(character: str) -> bool:
    return bool(_character_flags(character) & _HIRAGANA)


def is_katakana(character: str) -> bool:
    return bool(_character_flags(character) & _KATAKANA)


def is_hangul(character: str) -> bool:
    return bool(_character_flags(character) & _HANGUL)


def is_thai(character: str) -> bool:
    return bool(_character_flags(character) & _THAI)


def is_arabic(character: str) -> bool:
    return bool(_character_flags(character) & _ARABIC)


def is_arabic_isolated_form(character: str) -> bool:
    return bool(_character_flags(character) & _ARABIC_ISOLATED_FORM)


def is_cjk_uncommon(character: str) -> bool:
    return character not in COMMON_CJK_CHARACTERS


def is_unicode_range_secondary(range_name: str) -> bool:
    return range_name in _SECONDARY_RANGE_NAMES


def is_unprintable(character: str) -> bool:
    return (
        not character.isspace()  # includes \n \t \r \v
        and not character.isprintable()
        and character != "\x1a"  # Why? Its the ASCII substitute character.
        and character != "\ufeff"  # bug discovered in Python,
        # Zero Width No-Break Space located in 	Arabic Presentation Forms-B, Unicode 1.1 not acknowledged as space.
    )


def any_specified_encoding(
    sequence: bytes | bytearray, search_zone: int = 8192
) -> str | None:
    """
    Extract using ASCII-only decoder any specified encoding in the first n-bytes.
    """
    if not isinstance(sequence, (bytes, bytearray)):
        raise TypeError

    seq_len: int = len(sequence)

    # Cheap literal pre-filter.
    search_bytes = sequence[: min(seq_len, search_zone)]
    lowered_bytes = search_bytes.lower()
    if b"coding" not in lowered_bytes and b"charset" not in lowered_bytes:
        return None

    decoded_zone: str = search_bytes.decode("ascii", errors="ignore")

    for match in RE_POSSIBLE_ENCODING_INDICATION.finditer(decoded_zone):
        specified_encoding = match.group(1).lower().replace("-", "_")
        encoding_iana = _IANA_NAMES.get(specified_encoding)
        if encoding_iana is not None:
            return encoding_iana

    return None


@lru_cache(maxsize=None)
def is_multi_byte_encoding(name: str) -> bool:
    """
    Verify is a specific encoding is a multi byte one based on it IANA name
    """
    if name in _KNOWN_MB_DECODERS:
        return True

    # Besides the Unicode family above, every multibyte codec shipped with
    # Python is implemented by _multibytecodec through exactly one of the six
    # cjkcodecs providers below. Probing those providers directly (getcodec)
    # classifies a name without importing its "encodings.<name>" module:
    # classifying the whole IANA_SUPPORTED list would otherwise import many
    # modules and dominate "import charset_normalizer" wall time.
    # see https://github.com/jawah/charset_normalizer/issues/742
    for provider in _KNOWN_MB_CLASSES:
        try:
            importlib.import_module(provider).getcodec(name)  # type: ignore[attr-defined]
        except (ImportError, AttributeError, LookupError):  # Defensive: edge cases
            continue
        return True

    return False


def identify_sig_or_bom(sequence: bytes | bytearray) -> tuple[str | None, bytes]:
    """
    Identify and extract SIG/BOM in given sequence.
    """

    for iana_encoding in ENCODING_MARKS:
        marks: bytes | list[bytes] = ENCODING_MARKS[iana_encoding]

        if isinstance(marks, bytes):
            marks = [marks]

        for mark in marks:
            if sequence.startswith(mark):
                return iana_encoding, mark

    return None, b""


def should_strip_sig_or_bom(iana_encoding: str) -> bool:
    return iana_encoding not in {"utf_16", "utf_32"}


def iana_name(cp_name: str, strict: bool = True) -> str:
    """Returns the Python normalized encoding name (Not the IANA official name)."""
    cp_name = cp_name.lower().replace("-", "_")

    encoding_iana = _IANA_NAMES.get(cp_name)
    if encoding_iana is not None:
        return encoding_iana

    if strict:
        raise ValueError(f"Unable to retrieve IANA for '{cp_name}'")

    return cp_name


def cp_similarity(iana_name_a: str, iana_name_b: str) -> float:
    if is_multi_byte_encoding(iana_name_a) or is_multi_byte_encoding(iana_name_b):
        return 0.0

    decoder_a = importlib.import_module(f"encodings.{iana_name_a}").IncrementalDecoder
    decoder_b = importlib.import_module(f"encodings.{iana_name_b}").IncrementalDecoder

    id_a: IncrementalDecoder = decoder_a(errors="ignore")
    id_b: IncrementalDecoder = decoder_b(errors="ignore")

    character_match_count: int = 0

    for i in range(256):
        to_be_decoded: bytes = bytes([i])
        if id_a.decode(to_be_decoded) == id_b.decode(to_be_decoded):
            character_match_count += 1

    return character_match_count / 256


def is_cp_similar(iana_name_a: str, iana_name_b: str) -> bool:
    """
    Determine if two code page are at least 80% similar. IANA_SUPPORTED_SIMILAR dict was generated using
    the function cp_similarity.
    """
    return (
        iana_name_a in IANA_SUPPORTED_SIMILAR
        and iana_name_b in IANA_SUPPORTED_SIMILAR[iana_name_a]
    )


def set_logging_handler(
    name: str = "charset_normalizer",
    level: int = logging.INFO,
    format_string: str = "%(asctime)s | %(levelname)s | %(message)s",
) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(handler)


def cut_sequence_chunks(
    sequences: bytes | bytearray,
    encoding_iana: str,
    offsets: range,
    chunk_size: int,
    bom_or_sig_available: bool,
    strip_sig_or_bom: bool,
    sig_payload: bytes,
    is_multi_byte_decoder: bool,
    decoded_payload: str | None = None,
    deferred_decoding: bool = False,
) -> Generator[str, None, None]:
    # iso2022 codec is stateful, generic mb cuter isn't going to cut it!
    if decoded_payload and encoding_iana.startswith("iso2022_"):
        decoded_length = len(decoded_payload)
        sequence_length = len(sequences)
        for i in offsets:
            decoded_offset = i * decoded_length // sequence_length
            chunk = decoded_payload[decoded_offset : decoded_offset + chunk_size]
            if not chunk:
                break
            yield chunk
    elif decoded_payload and not is_multi_byte_decoder:
        for i in offsets:
            chunk = decoded_payload[i : i + chunk_size]
            if not chunk:
                break
            yield chunk
    elif deferred_decoding:
        # Deferred single-byte probing: the whole payload is not decoded
        # yet. Single-byte codecs are stateless (1 byte == 1 char), hence
        # decode(base)[i:j] == decode(base[i:j]): slicing the raw bytes
        # yields exactly the chunks the branch above would have produced,
        # short trailing chunks included, and raises UnicodeDecodeError on
        # invalid bytes just like the whole-payload decode would.
        base_bytes = (
            sequences if not strip_sig_or_bom else sequences[len(sig_payload) :]
        )
        for i in offsets:
            cut_sequence = base_bytes[i : i + chunk_size]
            if not cut_sequence:
                break
            yield str(cut_sequence, encoding_iana)
    else:
        for i in offsets:
            chunk_end = i + chunk_size
            if chunk_end > len(sequences) + 8:
                continue

            cut_sequence = sequences[i : i + chunk_size]

            if bom_or_sig_available and not strip_sig_or_bom:
                cut_sequence = sig_payload + cut_sequence

            chunk = cut_sequence.decode(
                encoding_iana,
                errors="ignore" if is_multi_byte_decoder else "strict",
            )

            # multi-byte bad cutting detector and adjustment
            # not the cleanest way to perform that fix but clever enough for now.
            if is_multi_byte_decoder and i > 0:
                chunk_partial_size_chk: int = min(chunk_size, 16)
                chunk_prefix = chunk[:chunk_partial_size_chk]
                found_nearby = False

                if decoded_payload:
                    decoded_length = len(decoded_payload)
                    expected_offset = i * decoded_length // len(sequences)
                    search_start = max(0, expected_offset - _MULTIBYTE_SEARCH_RADIUS)
                    search_end = min(
                        decoded_length, expected_offset + _MULTIBYTE_SEARCH_RADIUS
                    )
                    found_nearby = (
                        decoded_payload.find(chunk_prefix, search_start, search_end)
                        >= 0
                    )

                if (
                    decoded_payload
                    and not found_nearby
                    and chunk_prefix not in decoded_payload
                ):
                    for j in range(i, i - 4, -1):
                        cut_sequence = sequences[j:chunk_end]

                        if bom_or_sig_available and not strip_sig_or_bom:
                            cut_sequence = sig_payload + cut_sequence

                        chunk = cut_sequence.decode(encoding_iana, errors="ignore")

                        if chunk[:chunk_partial_size_chk] in decoded_payload:
                            break

            yield chunk
<|endoftext|>from __future__ import annotations

from encodings.aliases import aliases
from re import sub
from typing import Any, Iterator, List, Tuple

from .constant import RE_POSSIBLE_ENCODING_INDICATION, TOO_BIG_SEQUENCE
from .utils import iana_name, is_multi_byte_encoding, unicode_range


class CharsetMatch:
    def __init__(
        self,
        payload: bytes | bytearray,
        guessed_encoding: str,
        mean_mess_ratio: float,
        has_sig_or_bom: bool,
        languages: CoherenceMatches,
        decoded_payload: str | None = None,
        preemptive_declaration: str | None = None,
    ):
        self._payload: bytes | bytearray = payload

        self._encoding: str = guessed_encoding
        self._mean_mess_ratio: float = mean_mess_ratio
        self._languages: CoherenceMatches = languages
        self._has_sig_or_bom: bool = has_sig_or_bom
        self._unicode_ranges: list[str] | None = None

        self._leaves: list[CharsetMatch] = []
        self._mean_coherence_ratio: float = 0.0

        self._output_payload: bytes | None = None
        self._output_encoding: str | None = None

        self._string: str | None = decoded_payload

        self._preemptive_declaration: str | None = preemptive_declaration

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CharsetMatch):
            if isinstance(other, str):
                # Use non-strict iana_name so an operand that is not a known
                # encoding alias compares unequal instead of raising, keeping
                # __eq__ total as the data model requires (mirrors the lookup
                # in CharsetMatches.__getitem__).
                return iana_name(other, False) == self.encoding
            return False
        return self.encoding == other.encoding and self.fingerprint == other.fingerprint

    def __lt__(self, other: object) -> bool:
        """
        Implemented to make sorted available upon CharsetMatches items.
        """
        if not isinstance(other, CharsetMatch):
            raise ValueError

        chaos_difference: float = abs(self.chaos - other.chaos)
        coherence_difference: float = abs(self.coherence - other.coherence)

        # Below 0.5% difference --> Use Coherence
        if chaos_difference < 0.005 and coherence_difference > 0.02:
            return self.coherence > other.coherence
        elif chaos_difference < 0.005 and coherence_difference <= 0.02:
            # When having a difficult decision, use the result that decoded as many multi-byte as possible.
            # preserve RAM usage!
            if len(self._payload) >= TOO_BIG_SEQUENCE:
                return self.chaos < other.chaos
            return self.multi_byte_usage > other.multi_byte_usage

        return self.chaos < other.chaos

    @property
    def multi_byte_usage(self) -> float:
        # Empty payloads are valid (from_bytes(b"") returns a match); avoid /0.
        raw_len = len(self.raw)
        if raw_len == 0:
            return 0.0
        return 1.0 - (len(str(self)) / raw_len)

    def __str__(self) -> str:
        # Lazy Str Loading
        if self._string is None:
            self._string = str(self._payload, self._encoding, "strict")
            # UTF-7 BOM is encoded in modified Base64 whose byte boundary
            # can overlap with the next character, so raw-byte stripping
            # is unreliable. Strip the decoded BOM character instead.
            if (
                self._has_sig_or_bom
                and self._encoding == "utf_7"
                and self._string
                and self._string[0] == "\ufeff"
            ):
                self._string = self._string[1:]
        return self._string

    def __repr__(self) -> str:
        return f"<CharsetMatch '{self.encoding}' fp({self.fingerprint})>"

    def add_submatch(self, other: CharsetMatch) -> None:
        if not isinstance(other, CharsetMatch) or other == self:
            raise ValueError(
                "Unable to add instance <{}> as a submatch of a CharsetMatch".format(
                    other.__class__
                )
            )

        other._string = None  # Unload RAM usage; dirty trick.
        self._leaves.append(other)

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def encoding_aliases(self) -> list[str]:
        """
        Encoding name are known by many name, using this could help when searching for IBM855 when it's listed as CP855.
        """
        also_known_as: list[str] = []
        for u, p in aliases.items():
            if self.encoding == u:
                also_known_as.append(p)
            elif self.encoding == p:
                also_known_as.append(u)
        return also_known_as

    @property
    def bom(self) -> bool:
        return self._has_sig_or_bom

    @property
    def byte_order_mark(self) -> bool:
        return self._has_sig_or_bom

    @property
    def languages(self) -> list[str]:
        """
        Return the complete list of possible languages found in decoded sequence.
        Usually not really useful. Returned list may be empty even if 'language' property return something != 'Unknown'.
        """
        return [e[0] for e in self._languages]

    @property
    def language(self) -> str:
        """
        Most probable language found in decoded sequence. If none were detected or inferred, the property will return
        "Unknown".
        """
        if not self._languages:
            # Trying to infer the language based on the given encoding
            # Its either English or we should not pronounce ourselves in certain cases.
            if "ascii" in self.could_be_from_charset:
                return "English"

            # doing it there to avoid circular import
            from charset_normalizer.cd import encoding_languages, mb_encoding_languages

            languages = (
                mb_encoding_languages(self.encoding)
                if is_multi_byte_encoding(self.encoding)
                else encoding_languages(self.encoding)
            )

            if len(languages) == 0 or "Latin Based" in languages:
                return "Unknown"

            return languages[0]

        return self._languages[0][0]

    @property
    def chaos(self) -> float:
        return self._mean_mess_ratio

    @property
    def coherence(self) -> float:
        if not self._languages:
            return 0.0
        return self._languages[0][1]

    @property
    def percent_chaos(self) -> float:
        return round(self.chaos * 100, ndigits=3)

    @property
    def percent_coherence(self) -> float:
        return round(self.coherence * 100, ndigits=3)

    @property
    def raw(self) -> bytes | bytearray:
        """
        Original untouched bytes.
        """
        return self._payload

    @property
    def submatch(self) -> list[CharsetMatch]:
        return self._leaves

    @property
    def has_submatch(self) -> bool:
        return len(self._leaves) > 0

    @property
    def alphabets(self) -> list[str]:
        if self._unicode_ranges is not None:
            return self._unicode_ranges
        # list detected ranges
        detected_ranges: list[str | None] = [unicode_range(char) for char in str(self)]
        # filter and sort
        self._unicode_ranges = sorted(list({r for r in detected_ranges if r}))
        return self._unicode_ranges

    @property
    def could_be_from_charset(self) -> list[str]:
        """
        The complete list of encoding that output the exact SAME str result and therefore could be the originating
        encoding.
        This list does include the encoding available in property 'encoding'.
        """
        return [self._encoding] + [m.encoding for m in self._leaves]

    def output(self, encoding: str = "utf_8") -> bytes:
        """
        Method to get re-encoded bytes payload using given target encoding. Default to UTF-8.
        Any errors will be simply ignored by the encoder NOT replaced.
        """
        if self._output_encoding is None or self._output_encoding != encoding:
            self._output_encoding = encoding
            decoded_string = str(self)
            if (
                self._preemptive_declaration is not None
                and self._preemptive_declaration.lower()
                not in ["utf-8", "utf8", "utf_8"]
            ):
                patched_header = sub(
                    RE_POSSIBLE_ENCODING_INDICATION,
                    lambda m: m.string[m.span()[0] : m.span()[1]].replace(
                        m.groups()[0],
                        iana_name(self._output_encoding).replace("_", "-"),  # type: ignore[arg-type]
                    ),
                    decoded_string[:8192],
                    count=1,
                )

                decoded_string = patched_header + decoded_string[8192:]

            self._output_payload = decoded_string.encode(encoding, "replace")

        return self._output_payload  # type: ignore

    @property
    def fingerprint(self) -> int:
        """
        Retrieve a hash fingerprint of the decoded payload, used for deduplication.
        """
        return hash(str(self))


class CharsetMatches:
    """
    Container with every CharsetMatch items ordered by default from most probable to the less one.
    Act like a list(iterable) but does not implements all related methods.
    """

    def __init__(self, results: list[CharsetMatch] | None = None):
        self._results: list[CharsetMatch] = sorted(results) if results else []
        self._is_sorted: bool = True

    def _ensure_sorted(self) -> None:
        if not self._is_sorted:
            self._results.sort()
            self._is_sorted = True

    def __iter__(self) -> Iterator[CharsetMatch]:
        self._ensure_sorted()
        yield from self._results

    def __getitem__(self, item: int | str) -> CharsetMatch:
        """
        Retrieve a single item either by its position or encoding name (alias may be used here).
        Raise KeyError upon invalid index or encoding not present in results.
        """
        if isinstance(item, int):
            self._ensure_sorted()
            return self._results[item]
        if isinstance(item, str):
            item = iana_name(item, False)
            for result in self._results:
                if item in result.could_be_from_charset:
                    return result
        raise KeyError

    def __len__(self) -> int:
        return len(self._results)

    def __bool__(self) -> bool:
        return len(self._results) > 0

    def append(self, item: CharsetMatch) -> None:
        """
        Insert a single match. Will be inserted accordingly to preserve sort.
        Can be inserted as a submatch.
        """
        if not isinstance(item, CharsetMatch):
            raise ValueError(
                "Cannot append instance '{}' to CharsetMatches".format(
                    str(item.__class__)
                )
            )
        # We should disable the submatch factoring when the input file is too heavy (conserve RAM usage)
        if len(item.raw) < TOO_BIG_SEQUENCE:
            for match in self._results:
                if match.fingerprint == item.fingerprint and match.chaos == item.chaos:
                    match.add_submatch(item)
                    return
        self._results.append(item)
        self._is_sorted = False

    def best(self) -> CharsetMatch | None:
        """
        Simply return the first match. Strict equivalent to matches[0].
        """
        if not self._results:
            return None
        self._ensure_sorted()
        return self._results[0]

    def first(self) -> CharsetMatch | None:
        """
        Redundant method, call the method best(). Kept for BC reasons.
        """
        return self.best()


CoherenceMatch = Tuple[str, float]
CoherenceMatches = List[CoherenceMatch]


class CliDetectionResult:
    def __init__(
        self,
        path: str,
        encoding: str | None,
        encoding_aliases: list[str],
        alternative_encodings: list[str],
        language: str,
        alphabets: list[str],
        has_sig_or_bom: bool,
        chaos: float,
        coherence: float,
        unicode_path: str | None,
        is_preferred: bool,
    ):
        self.path: str = path
        self.unicode_path: str | None = unicode_path
        self.encoding: str | None = encoding
        self.encoding_aliases: list[str] = encoding_aliases
        self.alternative_encodings: list[str] = alternative_encodings
        self.language: str = language
        self.alphabets: list[str] = alphabets
        self.has_sig_or_bom: bool = has_sig_or_bom
        self.chaos: float = chaos
        self.coherence: float = coherence
        self.is_preferred: bool = is_preferred

    @property
    def __dict__(self) -> dict[str, Any]:  # type: ignore
        return {
            "path": self.path,
            "encoding": self.encoding,
            "encoding_aliases": self.encoding_aliases,
            "alternative_encodings": self.alternative_encodings,
            "language": self.language,
            "alphabets": self.alphabets,
            "has_sig_or_bom": self.has_sig_or_bom,
            "chaos": self.chaos,
            "coherence": self.coherence,
            "unicode_path": self.unicode_path,
            "is_preferred": self.is_preferred,
        }

    def to_json(self) -> str:
        from json import dumps

        return dumps(self.__dict__, ensure_ascii=True, indent=4)
<|endoftext|>"""
Charset-Normalizer
~~~~~~~~~~~~~~
The Real First Universal Charset Detector.
A library that helps you read text from an unknown charset encoding.
Motivated by chardet, This package is trying to resolve the issue by taking a new approach.
All IANA character set names for which the Python core library provides codecs are supported.

Basic usage:
   >>> from charset_normalizer import from_bytes
   >>> results = from_bytes('Bсеки човек има право на образование. Oбразованието!'.encode('utf_8'))
   >>> best_guess = results.best()
   >>> str(best_guess)
   'Bсеки човек има право на образование. Oбразованието!'

Others methods and usages are available - see the full documentation
at <https://github.com/Ousret/charset_normalizer>.
:copyright: (c) Ahmed TAHRI
:license: MIT, see LICENSE for more details.
"""

from __future__ import annotations

import logging

from .api import from_bytes, from_fp, from_path, is_binary
from .legacy import detect
from .models import CharsetMatch, CharsetMatches
from .utils import set_logging_handler
from .version import VERSION, __version__

__all__ = (
    "from_fp",
    "from_path",
    "from_bytes",
    "is_binary",
    "detect",
    "CharsetMatch",
    "CharsetMatches",
    "__version__",
    "VERSION",
    "set_logging_handler",
)

# Attach a NullHandler to the top level logger by default
# https://docs.python.org/3.3/howto/logging.html#configuring-logging-for-a-library

logging.getLogger("charset_normalizer").addHandler(logging.NullHandler())
<|endoftext|>from __future__ import annotations

import logging
from functools import lru_cache
from os import PathLike
from typing import BinaryIO

from .cd import (
    coherence_ratio,
    encoding_languages,
    mb_encoding_languages,
    merge_coherence_ratios,
)
from .constant import (
    IANA_SUPPORTED,
    IANA_SUPPORTED_SIMILAR,
    TOO_BIG_SEQUENCE,
    TOO_SMALL_SEQUENCE,
    TRACE,
)
from .md import mess_ratio
from .models import CharsetMatch, CharsetMatches
from .utils import (
    any_specified_encoding,
    cut_sequence_chunks,
    iana_name,
    identify_sig_or_bom,
    is_multi_byte_encoding,
    should_strip_sig_or_bom,
)

logger = logging.getLogger("charset_normalizer")
explain_handler = logging.StreamHandler()
explain_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)

# Pre-compute a reordered encoding list: multibyte first, then single-byte.
# This allows the mb_definitive_match optimization to fire earlier, skipping
# all single-byte encodings for genuine CJK content. Multibyte codecs
# hard-fail (UnicodeDecodeError) on single-byte data almost instantly, so
# testing them first costs negligible time for non-CJK files.
# Stable sort on a boolean key: multibyte (False) first, IANA order kept
# within each group.
IANA_SUPPORTED_MB_FIRST: list[str] = sorted(
    IANA_SUPPORTED, key=lambda encoding: not is_multi_byte_encoding(encoding)
)


def from_bytes(
    sequences: bytes | bytearray,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.2,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    """
    Given a raw bytes sequence, return the best possibles charset usable to render str objects.
    If there is no results, it is a strong indicator that the source is binary/not text.
    By default, the process will extract 5 blocks of 512o each to assess the mess and coherence of a given sequence.
    And will give up a particular code page after 20% of measured mess. Those criteria are customizable at will.

    The preemptive behavior DOES NOT replace the traditional detection workflow, it prioritize a particular code page
    but never take it for granted. Can improve the performance.

    You may want to focus your attention to some code page or/and not others, use cp_isolation and cp_exclusion for that
    purpose.

    This function will strip the SIG in the payload/sequence every time except on UTF-16, UTF-32.
    By default the library does not setup any handler other than the NullHandler, if you choose to set the 'explain'
    toggle to True it will alter the logger configuration to add a StreamHandler that is suitable for debugging.
    Custom logging format and handler can be set manually.
    """

    if not isinstance(sequences, (bytearray, bytes)):
        raise TypeError(
            "Expected object of type bytes or bytearray, got: {}".format(
                type(sequences)
            )
        )

    if explain:
        previous_logger_level: int = logger.level
        logger.addHandler(explain_handler)
        logger.setLevel(TRACE)

    length: int = len(sequences)

    if length == 0:
        logger.debug("Encoding detection on empty bytes, assuming utf_8 intention.")
        if explain:  # Defensive: ensure exit path clean handler
            logger.removeHandler(explain_handler)
            logger.setLevel(previous_logger_level)
        return CharsetMatches([CharsetMatch(sequences, "utf_8", 0.0, False, [], "")])

    if cp_isolation is not None:
        logger.log(
            TRACE,
            "cp_isolation is set. use this flag for debugging purpose. "
            "limited list of encoding allowed : %s.",
            ", ".join(cp_isolation),
        )
        cp_isolation = [iana_name(cp, False) for cp in cp_isolation]
    else:
        cp_isolation = []

    if cp_exclusion is not None:
        logger.log(
            TRACE,
            "cp_exclusion is set. use this flag for debugging purpose. "
            "limited list of encoding excluded : %s.",
            ", ".join(cp_exclusion),
        )
        cp_exclusion = [iana_name(cp, False) for cp in cp_exclusion]
    else:
        cp_exclusion = []

    if length <= (chunk_size * steps):
        logger.log(
            TRACE,
            "override steps (%i) and chunk_size (%i) as content does not fit (%i byte(s) given) parameters.",
            steps,
            chunk_size,
            length,
        )
        steps = 1
        chunk_size = length

    if steps > 1 and length / steps < chunk_size:
        chunk_size = int(length / steps)

    is_too_small_sequence: bool = len(sequences) < TOO_SMALL_SEQUENCE
    is_too_large_sequence: bool = len(sequences) >= TOO_BIG_SEQUENCE

    if is_too_small_sequence:
        logger.log(
            TRACE,
            "Trying to detect encoding from a tiny portion of (%i) byte(s).",
            length,
        )
    elif is_too_large_sequence:
        logger.log(
            TRACE,
            "Using lazy str decoding because the payload is quite large, (%i) byte(s).",
            length,
        )

    prioritized_encodings: list[str] = []

    specified_encoding: str | None = (
        any_specified_encoding(sequences) if preemptive_behaviour else None
    )

    if specified_encoding is not None:
        prioritized_encodings.append(specified_encoding)
        logger.log(
            TRACE,
            "Detected declarative mark in sequence. Priority +1 given for %s.",
            specified_encoding,
        )

    tested: set[str] = set()
    tested_but_hard_failure: list[str] = []
    tested_but_soft_failure: list[str] = []
    soft_failure_skip: set[str] = set()

    # Avoid unoptimized RSS usage.
    # this cache is mostly interesting for
    # local usage. Garbage collected at the
    # end. Like it should.
    cached_mess_ratio = lru_cache(maxsize=None)(mess_ratio)
    cached_coherence_ratio = lru_cache(maxsize=None)(coherence_ratio)

    # When a definitive result (chaos=0.0 and good coherence) is found after testing
    # the prioritized encodings (ascii, utf_8), we can significantly reduce the remaining
    # work. Encodings that target completely different language families (e.g., Cyrillic
    # when the definitive match is Latin) are skipped entirely.
    # Additionally, for same-family encodings that pass chaos probing, we reuse the
    # definitive match's coherence ratios instead of recomputing them — a major savings
    # since coherence_ratio accounts for ~30% of total time on slow Latin files.
    definitive_match_found: bool = False
    definitive_target_languages: set[str] = set()
    # After the definitive match fires, we cap the number of additional same-family
    # single-byte encodings that pass chaos probing. Once we've accumulated enough
    # good candidates (N), further same-family SB encodings are unlikely to produce
    # a better best() result and just waste mess_ratio + coherence_ratio time.
    # The first encoding to trigger the definitive match is NOT counted (it's already in).
    post_definitive_sb_success_count: int = 0
    POST_DEFINITIVE_SB_CAP: int = 7

    # When a non-UTF multibyte encoding passes chaos probing with significant multibyte
    # content (decoded length < 98% of raw length), skip all remaining single-byte encodings.
    # Rationale: multi-byte decoders (CJK) have strict byte-sequence validation — if they
    # decode without error AND pass chaos probing with substantial multibyte content, the
    # data is genuinely multibyte encoded. Single-byte encodings will always decode (every
    # byte maps to something) but waste time on mess_ratio before failing.
    # The 98% threshold prevents false triggers on files that happen to have a few valid
    # multibyte pairs (e.g., cp424/_ude_1.txt where big5 decodes with 99% ratio).
    mb_definitive_match_found: bool = False

    fallback_ascii: CharsetMatch | None = None
    fallback_u8: CharsetMatch | None = None
    fallback_specified: CharsetMatch | None = None

    results: CharsetMatches = CharsetMatches()

    early_stop_results: CharsetMatches = CharsetMatches()

    sig_encoding, sig_payload = identify_sig_or_bom(sequences)

    if sig_encoding is not None:
        prioritized_encodings.insert(0, sig_encoding)
        logger.log(
            TRACE,
            "Detected a SIG or BOM mark on first %i byte(s). Priority +1 given for %s.",
            len(sig_payload),
            sig_encoding,
        )

    prioritized_encodings.append("ascii")

    if "utf_8" not in prioritized_encodings:
        prioritized_encodings.append("utf_8")

    for encoding_iana in prioritized_encodings + IANA_SUPPORTED_MB_FIRST:
        if cp_isolation and encoding_iana not in cp_isolation:
            continue

        if cp_exclusion and encoding_iana in cp_exclusion:
            continue

        if encoding_iana in tested:
            continue

        tested.add(encoding_iana)

        decoded_payload: str | None = None
        bom_or_sig_available: bool = sig_encoding == encoding_iana
        strip_sig_or_bom: bool = bom_or_sig_available and should_strip_sig_or_bom(
            encoding_iana
        )

        if encoding_iana in {"utf_16", "utf_32"} and not bom_or_sig_available:
            logger.log(
                TRACE,
                "Encoding %s won't be tested as-is because it require a BOM. Will try some sub-encoder LE/BE.",
                encoding_iana,
            )
            continue
        if encoding_iana in {"utf_7"} and not bom_or_sig_available:
            logger.log(
                TRACE,
                "Encoding %s won't be tested as-is because detection is unreliable without BOM/SIG.",
                encoding_iana,
            )
            continue

        # Skip encodings similar to ones that already soft-failed (high mess ratio).
        # Checked BEFORE the expensive decode attempt.
        if encoding_iana in soft_failure_skip:
            logger.log(
                TRACE,
                "%s is deemed too similar to a code page that was already considered unsuited. Continuing!",
                encoding_iana,
            )
            continue

        try:
            is_multi_byte_decoder: bool = is_multi_byte_encoding(encoding_iana)
        except (ModuleNotFoundError, ImportError):  # Defensive:
            logger.log(
                TRACE,
                "Encoding %s does not provide an IncrementalDecoder",
                encoding_iana,
            )
            continue

        # When we've already found a definitive match (chaos=0.0 with good coherence)
        # after testing the prioritized encodings, skip encodings that target
        # completely different language families. This avoids running expensive
        # mess_ratio + coherence_ratio on clearly unrelated candidates (e.g., Cyrillic
        # when the definitive match is Latin-based).
        if definitive_match_found:
            if not is_multi_byte_decoder:
                enc_languages = set(encoding_languages(encoding_iana))
            else:
                enc_languages = set(mb_encoding_languages(encoding_iana))
            if not enc_languages.intersection(definitive_target_languages):
                logger.log(
                    TRACE,
                    "Skipping %s: definitive match already found, this encoding targets different languages (%s vs %s).",
                    encoding_iana,
                    enc_languages,
                    definitive_target_languages,
                )
                continue

        # After the definitive match, cap the number of additional same-family
        # single-byte encodings that pass chaos probing. This avoids testing the
        # tail of rare, low-value same-family encodings (mac_iceland, cp860, etc.)
        # that almost never change best() but each cost ~1-2ms of mess_ratio + coherence.
        if (
            definitive_match_found
            and not is_multi_byte_decoder
            and post_definitive_sb_success_count >= POST_DEFINITIVE_SB_CAP
        ):
            logger.log(
                TRACE,
                "Skipping %s: already accumulated %d same-family results after definitive match (cap=%d).",
                encoding_iana,
                post_definitive_sb_success_count,
                POST_DEFINITIVE_SB_CAP,
            )
            continue

        # When a multibyte encoding with significant multibyte content has already
        # passed chaos probing, skip all single-byte encodings. They will either fail
        # chaos probing (wasting mess_ratio time) or produce inferior results.
        if mb_definitive_match_found and not is_multi_byte_decoder:
            logger.log(
                TRACE,
                "Skipping single-byte %s: multi-byte definitive match already found.",
                encoding_iana,
            )
            continue

        # Single-byte candidates of regular size defer the expensive whole
        # payload decode until after chunk probing: single-byte codecs are
        # stateless (1 byte == 1 char) so decoding chunk slices is provably
        # identical to slicing the decoded payload, and candidates rejected
        # by chaos probing (the common case) never pay the full decode nor
        # the payload hash.
        deferred_decoding: bool = (
            not is_multi_byte_decoder and not is_too_large_sequence
        )

        try:
            if is_too_large_sequence and not is_multi_byte_decoder:
                str(
                    (
                        sequences[: int(50e4)]
                        if not strip_sig_or_bom
                        else sequences[len(sig_payload) : int(50e4)]
                    ),
                    encoding=encoding_iana,
                )
            elif not deferred_decoding:
                # UTF-7 BOM is encoded in modified Base64 whose byte boundary
                # can overlap with the next character. Stripping raw SIG bytes
                # before decoding may leave stray bytes that decode as garbage.
                # Decode the full sequence and remove the leading BOM char instead.
                # see https://github.com/jawah/charset_normalizer/issues/718
                # and https://github.com/jawah/charset_normalizer/issues/716
                if encoding_iana == "utf_7" and bom_or_sig_available:
                    decoded_payload = str(
                        sequences,
                        encoding=encoding_iana,
                    )
                    if decoded_payload and decoded_payload[0] == "\ufeff":
                        decoded_payload = decoded_payload[1:]
                else:
                    decoded_payload = str(
                        (
                            sequences
                            if not strip_sig_or_bom
                            else sequences[len(sig_payload) :]
                        ),
                        encoding=encoding_iana,
                    )
        except (UnicodeDecodeError, LookupError) as e:
            if not isinstance(e, LookupError):
                logger.log(
                    TRACE,
                    "Code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
            tested_but_hard_failure.append(encoding_iana)
            continue

        r_ = range(
            0 if not bom_or_sig_available else len(sig_payload),
            length,
            int(length / steps),
        )

        multi_byte_bonus: bool = (
            is_multi_byte_decoder
            and decoded_payload is not None
            and len(decoded_payload) < length
        )

        if multi_byte_bonus:
            logger.log(
                TRACE,
                "Code page %s is a multi byte encoding table and it appear that at least one character "
                "was encoded using n-bytes.",
                encoding_iana,
            )

        max_chunk_gave_up: int = int(len(r_) / 4)

        max_chunk_gave_up = max(max_chunk_gave_up, 2)
        early_stop_count: int = 0
        lazy_str_hard_failure = False

        md_chunks: list[str] = []
        md_ratios = []

        try:
            for chunk in cut_sequence_chunks(
                sequences,
                encoding_iana,
                r_,
                chunk_size,
                bom_or_sig_available,
                strip_sig_or_bom,
                sig_payload,
                is_multi_byte_decoder,
                decoded_payload,
                deferred_decoding,
            ):
                md_chunks.append(chunk)

                md_ratios.append(
                    cached_mess_ratio(
                        chunk,
                        threshold,
                        explain and 1 <= len(cp_isolation) <= 2,
                    )
                )

                if md_ratios[-1] >= threshold:
                    early_stop_count += 1

                if (early_stop_count >= max_chunk_gave_up) or (
                    bom_or_sig_available and not strip_sig_or_bom
                ):
                    break
        except (
            UnicodeDecodeError,
            LookupError,
        ) as e:  # Lazy str loading may have missed something there
            if deferred_decoding:
                # Deferred single-byte validation failed on a chunk (or the
                # codec is unavailable on this interpreter build): identical
                # outcome and bookkeeping to the eager full-decode failure.
                logger.log(
                    TRACE,
                    "Code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
                tested_but_hard_failure.append(encoding_iana)
                continue
            logger.log(
                TRACE,
                "LazyStr Loading: After MD chunk decode, code page %s does not fit given bytes sequence at ALL. %s",
                encoding_iana,
                str(e),
            )
            early_stop_count = max_chunk_gave_up
            lazy_str_hard_failure = True

        mean_mess_ratio: float = sum(md_ratios) / len(md_ratios) if md_ratios else 0.0

        # We might want to check the sequence again with the whole content,
        # but only if initial MD tests passed.
        if (
            not lazy_str_hard_failure
            and is_too_large_sequence
            and not is_multi_byte_decoder
            and mean_mess_ratio < threshold
            and early_stop_count < max_chunk_gave_up
        ):
            try:
                sequences[int(50e3) :].decode(encoding_iana, errors="strict")
            except UnicodeDecodeError as e:
                logger.log(
                    TRACE,
                    "LazyStr Loading: After final lookup, code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
                tested_but_hard_failure.append(encoding_iana)
                continue

        if mean_mess_ratio >= threshold or early_stop_count >= max_chunk_gave_up:
            tested_but_soft_failure.append(encoding_iana)
            if encoding_iana in IANA_SUPPORTED_SIMILAR:
                soft_failure_skip.update(IANA_SUPPORTED_SIMILAR[encoding_iana])
            logger.log(
                TRACE,
                "%s was excluded because of initial chaos probing. Gave up %i time(s). "
                "Computed mean chaos is %f %%.",
                encoding_iana,
                early_stop_count,
                round(mean_mess_ratio * 100, ndigits=3),
            )
            # Preparing those fallbacks in case we got nothing.
            if (
                enable_fallback
                and encoding_iana
                in ["ascii", "utf_8", specified_encoding, "utf_16", "utf_32"]
                and not lazy_str_hard_failure
            ):
                # Always fully decode payload before.
                # We've missed a UnicodeDecodeError proof
                # while issuing release 3.4.8
                # see https://github.com/jawah/charset_normalizer/issues/771
                if decoded_payload is None:
                    try:
                        decoded_payload = str(
                            (
                                sequences
                                if not strip_sig_or_bom
                                else sequences[len(sig_payload) :]
                            ),
                            encoding=encoding_iana,
                        )
                    except (UnicodeDecodeError, LookupError):
                        logger.log(
                            TRACE,
                            "%s does not decode the whole payload: fallback entry withheld.",
                            encoding_iana,
                        )
                        continue
                    if is_too_large_sequence:
                        # Don't retain huge payload in RAM.
                        decoded_payload = None

                fallback_entry = CharsetMatch(
                    sequences,
                    encoding_iana,
                    threshold,
                    bom_or_sig_available,
                    [],
                    decoded_payload,
                    preemptive_declaration=specified_encoding,
                )
                if encoding_iana == specified_encoding:
                    fallback_specified = fallback_entry
                elif encoding_iana == "ascii":
                    fallback_ascii = fallback_entry
                else:
                    fallback_u8 = fallback_entry
            continue

        if deferred_decoding:
            # The candidate passed chaos probing: perform the whole payload
            # decode (validation + payload reuse) that was deferred earlier.
            try:
                decoded_payload = str(
                    (
                        sequences
                        if not strip_sig_or_bom
                        else sequences[len(sig_payload) :]
                    ),
                    encoding=encoding_iana,
                )
            except (UnicodeDecodeError, LookupError) as e:
                logger.log(
                    TRACE,
                    "Code page %s does not fit given bytes sequence at ALL. %s",
                    encoding_iana,
                    str(e),
                )
                tested_but_hard_failure.append(encoding_iana)
                continue

        logger.log(
            TRACE,
            "%s passed initial chaos probing. Mean measured chaos is %f %%",
            encoding_iana,
            round(mean_mess_ratio * 100, ndigits=3),
        )

        if not is_multi_byte_decoder:
            target_languages: list[str] = encoding_languages(encoding_iana)
        else:
            target_languages = mb_encoding_languages(encoding_iana)

        if target_languages:
            logger.log(
                TRACE,
                "%s should target any language(s) of %s",
                encoding_iana,
                target_languages,
            )

        cd_ratios = []

        # Run coherence detection on all chunks. We previously tried limiting to
        # 1-2 chunks for post-definitive encodings to save time, but this caused
        # coverage regressions by producing unrepresentative coherence scores.
        # The SB cap and language-family skip optimizations provide sufficient
        # speedup without sacrificing coherence accuracy.
        if encoding_iana != "ascii":
            # We shall skip the CD when its about ASCII
            # Most of the time its not relevant to run "language-detection" on it.
            lg_inclusion: str | None = (
                ",".join(target_languages) if target_languages else None
            )

            for chunk in md_chunks:
                chunk_languages = cached_coherence_ratio(
                    chunk,
                    language_threshold,
                    lg_inclusion,
                )

                cd_ratios.append(chunk_languages)

        cd_ratios_merged = merge_coherence_ratios(cd_ratios)

        if cd_ratios_merged:
            logger.log(
                TRACE,
                "We detected language %s using %s",
                cd_ratios_merged,
                encoding_iana,
            )

        current_match = CharsetMatch(
            sequences,
            encoding_iana,
            mean_mess_ratio,
            bom_or_sig_available,
            cd_ratios_merged,
            (
                decoded_payload
                if (
                    not is_too_large_sequence
                    or encoding_iana in [specified_encoding, "ascii", "utf_8"]
                )
                else None
            ),
            preemptive_declaration=specified_encoding,
        )

        results.append(current_match)

        # Count post-definitive same-family SB successes for the early termination cap.
        # Only count low-mess encodings (< 2%) toward the cap. High-mess encodings are
        # marginal results that shouldn't prevent better-quality candidates from being
        # tested. For example, iso8859_4 (mess=0%) should not be skipped just because
        # 7 high-mess Latin encodings (cp1252 at 8%, etc.) were tried first.
        if (
            definitive_match_found
            and not is_multi_byte_decoder
            and mean_mess_ratio < 0.02
        ):
            post_definitive_sb_success_count += 1

        if (
            encoding_iana in [specified_encoding, "ascii", "utf_8"]
            and mean_mess_ratio < 0.1
        ):
            # If md says nothing to worry about, then... stop immediately!
            if mean_mess_ratio == 0.0:
                logger.debug(
                    "Encoding detection: %s is most likely the one.",
                    current_match.encoding,
                )
                if explain:  # Defensive: ensure exit path clean handler
                    logger.removeHandler(explain_handler)
                    logger.setLevel(previous_logger_level)
                return CharsetMatches([current_match])

            early_stop_results.append(current_match)

        if (
            len(early_stop_results)
            and (specified_encoding is None or specified_encoding in tested)
            and "ascii" in tested
            and "utf_8" in tested
        ):
            probable_result = early_stop_results.best()
            assert probable_result is not None
            logger.debug(
                "Encoding detection: %s is most likely the one.",
                probable_result.encoding,
            )
            if explain:  # Defensive: ensure exit path clean handler
                logger.removeHandler(explain_handler)
                logger.setLevel(previous_logger_level)

            return CharsetMatches([probable_result])

        # Once we find a result with good coherence (>= 0.5) after testing the
        # prioritized encodings (ascii, utf_8), activate "definitive mode": skip
        # encodings that target completely different language families. This avoids
        # running expensive mess_ratio + coherence_ratio on clearly unrelated
        # candidates (e.g., Cyrillic encodings when the match is Latin-based).
        # We require coherence >= 0.5 to avoid false positives (e.g., cp1251 decoding
        # Hebrew text with 0.0 chaos but wrong language detection at coherence 0.33).
        if not definitive_match_found and not is_multi_byte_decoder:
            best_coherence = (
                max((v for _, v in cd_ratios_merged), default=0.0)
                if cd_ratios_merged
                else 0.0
            )
            if best_coherence >= 0.5 and "ascii" in tested and "utf_8" in tested:
                definitive_match_found = True
                definitive_target_languages.update(target_languages)
                logger.log(
                    TRACE,
                    "Definitive match found: %s (chaos=%.3f, coherence=%.2f). Encodings targeting different language families will be skipped.",
                    encoding_iana,
                    mean_mess_ratio,
                    best_coherence,
                )

        # When a non-UTF multibyte encoding passes chaos probing with significant
        # multibyte content (decoded < 98% of raw), activate mb_definitive_match.
        # This skips all remaining single-byte encodings which would either soft-fail
        # (running expensive mess_ratio for nothing) or produce inferior results.
        if (
            not mb_definitive_match_found
            and is_multi_byte_decoder
            and multi_byte_bonus
            and decoded_payload is not None
            and len(decoded_payload) < length * 0.98
            and encoding_iana
            not in {
                "utf_8",
                "utf_8_sig",
                "utf_16",
                "utf_16_be",
                "utf_16_le",
                "utf_32",
                "utf_32_be",
                "utf_32_le",
                "utf_7",
            }
            and "ascii" in tested
            and "utf_8" in tested
        ):
            mb_definitive_match_found = True
            logger.log(
                TRACE,
                "Multi-byte definitive match: %s (chaos=%.3f, decoded=%d/%d=%.1f%%). Single-byte encodings will be skipped.",
                encoding_iana,
                mean_mess_ratio,
                len(decoded_payload),
                length,
                len(decoded_payload) / length * 100,
            )

        if encoding_iana == sig_encoding:
            logger.debug(
                "Encoding detection: %s is most likely the one as we detected a BOM or SIG within "
                "the beginning of the sequence.",
                encoding_iana,
            )
            if explain:  # Defensive: ensure exit path clean handler
                logger.removeHandler(explain_handler)
                logger.setLevel(previous_logger_level)
            return CharsetMatches([results[encoding_iana]])

    if len(results) == 0:
        if fallback_u8 or fallback_ascii or fallback_specified:
            logger.log(
                TRACE,
                "Nothing got out of the detection process. Using ASCII/UTF-8/Specified fallback.",
            )

        if fallback_specified:
            logger.debug(
                "Encoding detection: %s will be used as a fallback match",
                fallback_specified.encoding,
            )
            results.append(fallback_specified)
        elif (
            (fallback_u8 and fallback_ascii is None)
            or (
                fallback_u8
                and fallback_ascii
                and fallback_u8.fingerprint != fallback_ascii.fingerprint
            )
            or (fallback_u8 is not None)
        ):
            logger.debug("Encoding detection: utf_8 will be used as a fallback match")
            results.append(fallback_u8)
        elif fallback_ascii:
            logger.debug("Encoding detection: ascii will be used as a fallback match")
            results.append(fallback_ascii)

    if results:
        logger.debug(
            "Encoding detection: Found %s as plausible (best-candidate) for content. With %i alternatives.",
            results.best().encoding,  # type: ignore
            len(results) - 1,
        )
    else:
        logger.debug("Encoding detection: Unable to determine any suitable charset.")

    if explain:
        logger.removeHandler(explain_handler)
        logger.setLevel(previous_logger_level)

    return results


def from_fp(
    fp: BinaryIO,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    """
    Same thing than the function from_bytes but using a file pointer that is already ready.
    Will not close the file pointer.
    """
    return from_bytes(
        fp.read(),
        steps,
        chunk_size,
        threshold,
        cp_isolation,
        cp_exclusion,
        preemptive_behaviour,
        explain,
        language_threshold,
        enable_fallback,
    )


def from_path(
    path: str | bytes | PathLike,  # type: ignore[type-arg]
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    """
    Same thing than the function from_bytes but with one extra step. Opening and reading given file path in binary mode.
    Can raise IOError.
    """
    with open(path, "rb") as fp:
        return from_fp(
            fp,
            steps,
            chunk_size,
            threshold,
            cp_isolation,
            cp_exclusion,
            preemptive_behaviour,
            explain,
            language_threshold,
            enable_fallback,
        )


def is_binary(
    fp_or_path_or_payload: PathLike | str | BinaryIO | bytes,  # type: ignore[type-arg]
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.20,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = False,
) -> bool:
    """
    Detect if the given input (file, bytes, or path) points to a binary file. aka. not a string.
    Based on the same main heuristic algorithms and default kwargs at the sole exception that fallbacks match
    are disabled to be stricter around ASCII-compatible but unlikely to be a string.
    """
    if isinstance(fp_or_path_or_payload, (str, PathLike)):
        guesses = from_path(
            fp_or_path_or_payload,
            steps=steps,
            chunk_size=chunk_size,
            threshold=threshold,
            cp_isolation=cp_isolation,
            cp_exclusion=cp_exclusion,
            preemptive_behaviour=preemptive_behaviour,
            explain=explain,
            language_threshold=language_threshold,
            enable_fallback=enable_fallback,
        )
    elif isinstance(
        fp_or_path_or_payload,
        (
            bytes,
            bytearray,
        ),
    ):
        guesses = from_bytes(
            fp_or_path_or_payload,
            steps=steps,
            chunk_size=chunk_size,
            threshold=threshold,
            cp_isolation=cp_isolation,
            cp_exclusion=cp_exclusion,
            preemptive_behaviour=preemptive_behaviour,
            explain=explain,
            language_threshold=language_threshold,
            enable_fallback=enable_fallback,
        )
    else:
        guesses = from_fp(
            fp_or_path_or_payload,
            steps=steps,
            chunk_size=chunk_size,
            threshold=threshold,
            cp_isolation=cp_isolation,
            cp_exclusion=cp_exclusion,
            preemptive_behaviour=preemptive_behaviour,
            explain=explain,
            language_threshold=language_threshold,
            enable_fallback=enable_fallback,
        )

    return not guesses
<|endoftext|>from __future__ import annotations

import importlib
from codecs import IncrementalDecoder
from functools import lru_cache

from .constant import (
    FREQUENCIES,
    KO_NAMES,
    LANGUAGE_SUPPORTED_COUNT,
    TOO_SMALL_SEQUENCE,
    ZH_NAMES,
    _FREQUENCIES_SET,
    _FREQUENCIES_RANK,
)
from .md import _ASCII_CHAR_INFO, _char_info, is_suspiciously_successive_range
from .models import CoherenceMatches
from .utils import (
    is_multi_byte_encoding,
    is_unicode_range_secondary,
)


def encoding_unicode_range(iana_name: str) -> list[str]:
    """
    Return associated unicode ranges in a single byte code page.
    """
    if is_multi_byte_encoding(iana_name):
        raise OSError(  # Defensive:
            "Function not supported on multi-byte code page"
        )

    decoder = importlib.import_module(f"encodings.{iana_name}").IncrementalDecoder

    p: IncrementalDecoder = decoder(errors="ignore")
    seen_ranges: dict[str, int] = {}
    character_count: int = 0

    for i in range(0x40, 0xFF):
        chunk: str = p.decode(bytes([i]))

        if chunk:
            chunk_codepoint = ord(chunk)
            character_range: str | None = (
                _ASCII_CHAR_INFO[chunk_codepoint].range
                if chunk_codepoint < 128
                else _char_info(chunk).range
            )

            if character_range is None:
                continue

            if not is_unicode_range_secondary(character_range):
                if character_range not in seen_ranges:
                    seen_ranges[character_range] = 0
                seen_ranges[character_range] += 1
                character_count += 1

    return sorted(
        [
            character_range
            for character_range in seen_ranges
            if seen_ranges[character_range] / character_count >= 0.15
        ]
    )


def unicode_range_languages(primary_range: str) -> list[str]:
    """
    Return inferred languages used with a unicode range.
    """
    languages: list[str] = []

    for language, characters in FREQUENCIES.items():
        for character in characters:
            codepoint = ord(character)
            info = (
                _ASCII_CHAR_INFO[codepoint]
                if codepoint < 128
                else _char_info(character)
            )
            if info.range == primary_range:
                languages.append(language)
                break

    return languages


@lru_cache()
def encoding_languages(iana_name: str) -> list[str]:
    """
    Single-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    try:
        unicode_ranges: list[str] = encoding_unicode_range(iana_name)
    except ImportError:  # Defensive: encoding unavailable on this build.
        return []

    primary_range: str | None = None

    for specified_range in unicode_ranges:
        if "Latin" not in specified_range:
            primary_range = specified_range
            break

    if primary_range is None:
        return ["Latin Based"]

    return unicode_range_languages(primary_range)


@lru_cache()
def mb_encoding_languages(iana_name: str) -> list[str]:
    """
    Multi-byte encoding language association. Some code page are heavily linked to particular language(s).
    This function does the correspondence.
    """
    if (
        iana_name.startswith("shift_")
        or iana_name.startswith("iso2022_jp")
        or iana_name.startswith("euc_j")
        or iana_name == "cp932"
    ):
        return ["Japanese"]
    if iana_name.startswith("gb") or iana_name in ZH_NAMES:
        return ["Chinese"]
    if iana_name.startswith("iso2022_kr") or iana_name in KO_NAMES:
        return ["Korean"]

    return []


@lru_cache(maxsize=LANGUAGE_SUPPORTED_COUNT)
def get_target_features(language: str) -> tuple[bool, bool]:
    """
    Determine main aspects from a supported language if it contains accents and if is pure Latin.
    """
    target_have_accents: bool = False
    target_pure_latin: bool = True

    for character in FREQUENCIES[language]:
        codepoint = ord(character)
        info = _ASCII_CHAR_INFO[codepoint] if codepoint < 128 else _char_info(character)
        if not target_have_accents and info.accentuated:
            target_have_accents = True
        if target_pure_latin and not info.latin:
            target_pure_latin = False

    return target_have_accents, target_pure_latin


def alphabet_languages(
    characters: list[str], ignore_non_latin: bool = False
) -> list[str]:
    """
    Return associated languages associated to given characters.
    """
    languages: list[tuple[str, float]] = []

    characters_set: frozenset[str] = frozenset(characters)
    source_have_accents = False
    for character in characters:
        codepoint = ord(character)
        info = _ASCII_CHAR_INFO[codepoint] if codepoint < 128 else _char_info(character)
        if info.accentuated:
            source_have_accents = True
            break

    for language, language_characters in FREQUENCIES.items():
        target_have_accents, target_pure_latin = get_target_features(language)

        if ignore_non_latin and not target_pure_latin:
            continue

        if not target_have_accents and source_have_accents:
            continue

        character_count: int = len(language_characters)

        character_match_count: int = len(_FREQUENCIES_SET[language] & characters_set)

        ratio: float = character_match_count / character_count

        if ratio >= 0.2:
            languages.append((language, ratio))

    languages = sorted(languages, key=lambda x: x[1], reverse=True)

    return [compatible_language[0] for compatible_language in languages]


def characters_popularity_compare(
    language: str, ordered_characters: list[str]
) -> float:
    """
    Determine if a ordered characters list (by occurrence from most appearance to rarest) match a particular language.
    The result is a ratio between 0. (absolutely no correspondence) and 1. (near perfect fit).
    Beware that is function is not strict on the match in order to ease the detection. (Meaning close match is 1.)
    """
    if language not in FREQUENCIES:
        raise ValueError(f"{language} not available")  # Defensive:

    character_approved_count: int = 0
