                            k, ename, edecl['charselector'][k], charselect[k]))
                    else:
                        edecl['charselector'][k] = copy.copy(charselect[k])
            if 'typename' not in edecl:
                edecl['typename'] = typename
            elif typename and (not edecl['typename'] == typename):
                outmess('updatevars: attempt to change the typename of "%s" ("%s") to "%s". Ignoring.\n' % (
                    ename, edecl['typename'], typename))
            if 'attrspec' not in edecl:
                edecl['attrspec'] = copy.copy(attrspec)
            elif attrspec:
                for a in attrspec:
                    if a not in edecl['attrspec']:
                        edecl['attrspec'].append(a)
        else:
            edecl['typespec'] = copy.copy(typespec)
            edecl['kindselector'] = copy.copy(kindselect)
            edecl['charselector'] = copy.copy(charselect)
            edecl['typename'] = typename
            edecl['attrspec'] = copy.copy(attrspec)
        if 'external' in (edecl.get('attrspec') or []) and e in groupcache[groupcounter]['args']:
            if 'externals' not in groupcache[groupcounter]:
                groupcache[groupcounter]['externals'] = []
            groupcache[groupcounter]['externals'].append(e)
        if m.group('after'):
            m1 = lenarraypattern.match(markouterparen(m.group('after')))
            if m1:
                d1 = m1.groupdict()
                for lk in ['len', 'array', 'init']:
                    if d1[lk + '2'] is not None:
                        d1[lk] = d1[lk + '2']
                        del d1[lk + '2']
                for k in list(d1.keys()):
                    if d1[k] is not None:
                        d1[k] = unmarkouterparen(d1[k])
                    else:
                        del d1[k]

                if 'len' in d1 and 'array' in d1:
                    if d1['len'] == '':
                        d1['len'] = d1['array']
                        del d1['array']
                    elif typespec == 'character':
                        if ('charselector' not in edecl) or (not edecl['charselector']):
                            edecl['charselector'] = {}
                        if 'len' in edecl['charselector']:
                            del edecl['charselector']['len']
                        edecl['charselector']['*'] = d1['len']
                        del d1['len']
                    else:
                        d1['array'] = d1['array'] + ',' + d1['len']
                        del d1['len']
                        errmess('updatevars: "%s %s" is mapped to "%s %s(%s)"\n' % (
                            typespec, e, typespec, ename, d1['array']))

                if 'len' in d1:
                    if typespec in ['complex', 'integer', 'logical', 'real']:
                        if ('kindselector' not in edecl) or (not edecl['kindselector']):
                            edecl['kindselector'] = {}
                        edecl['kindselector']['*'] = d1['len']
                        del d1['len']
                    elif typespec == 'character':
                        if ('charselector' not in edecl) or (not edecl['charselector']):
                            edecl['charselector'] = {}
                        if 'len' in edecl['charselector']:
                            del edecl['charselector']['len']
                        edecl['charselector']['*'] = d1['len']
                        del d1['len']

                if 'init' in d1:
                    if '=' in edecl and (not edecl['='] == d1['init']):
                        outmess('updatevars: attempt to change the init expression of "%s" ("%s") to "%s". Ignoring.\n' % (
                            ename, edecl['='], d1['init']))
                    else:
                        edecl['='] = d1['init']

                if 'array' in d1:
                    dm = 'dimension(%s)' % d1['array']
                    if 'attrspec' not in edecl or (not edecl['attrspec']):
                        edecl['attrspec'] = [dm]
                    else:
                        edecl['attrspec'].append(dm)
                        for dm1 in edecl['attrspec']:
                            if dm1[:9] == 'dimension' and dm1 != dm:
                                del edecl['attrspec'][-1]
                                errmess('updatevars:%s: attempt to change %r to %r. Ignoring.\n'
                                        % (ename, dm1, dm))
                                break

            else:
                outmess('updatevars: could not crack entity declaration "%s". Ignoring.\n' % (
                    ename + m.group('after')))
        for k in list(edecl.keys()):
            if not edecl[k]:
                del edecl[k]
        groupcache[groupcounter]['vars'][ename] = edecl
        if 'varnames' in groupcache[groupcounter]:
            groupcache[groupcounter]['varnames'].append(ename)
        last_name = ename
    return last_name


def cracktypespec(typespec, selector):
    kindselect = None
    charselect = None
    typename = None
    if selector:
        if typespec in ['complex', 'integer', 'logical', 'real']:
            kindselect = kindselector.match(selector)
            if not kindselect:
                outmess(
                    'cracktypespec: no kindselector pattern found for %s\n' % (repr(selector)))
                return
            kindselect = kindselect.groupdict()
            kindselect['*'] = kindselect['kind2']
            del kindselect['kind2']
            for k in list(kindselect.keys()):
                if not kindselect[k]:
                    del kindselect[k]
            for k, i in list(kindselect.items()):
                kindselect[k] = rmbadname1(i)
        elif typespec == 'character':
            charselect = charselector.match(selector)
            if not charselect:
                outmess(
                    'cracktypespec: no charselector pattern found for %s\n' % (repr(selector)))
                return
            charselect = charselect.groupdict()
            charselect['*'] = charselect['charlen']
            del charselect['charlen']
            if charselect['lenkind']:
                lenkind = lenkindpattern.match(
                    markoutercomma(charselect['lenkind']))
                lenkind = lenkind.groupdict()
                for lk in ['len', 'kind']:
                    if lenkind[lk + '2']:
                        lenkind[lk] = lenkind[lk + '2']
                    charselect[lk] = lenkind[lk]
                    del lenkind[lk + '2']
                if lenkind['f2py_len'] is not None:
                    # used to specify the length of assumed length strings
                    charselect['f2py_len'] = lenkind['f2py_len']
            del charselect['lenkind']
            for k in list(charselect.keys()):
                if not charselect[k]:
                    del charselect[k]
            for k, i in list(charselect.items()):
                charselect[k] = rmbadname1(i)
        elif typespec == 'type':
            typename = re.match(r'\s*\(\s*(?P<name>\w+)\s*\)', selector, re.I)
            if typename:
                typename = typename.group('name')
            else:
                outmess('cracktypespec: no typename found in %s\n' %
                        (repr(typespec + selector)))
        else:
            outmess('cracktypespec: no selector used for %s\n' %
                    (repr(selector)))
    return kindselect, charselect, typename
######


def setattrspec(decl, attr, force=0):
    if not decl:
        decl = {}
    if not attr:
        return decl
    if 'attrspec' not in decl:
        decl['attrspec'] = [attr]
        return decl
    if force:
        decl['attrspec'].append(attr)
    if attr in decl['attrspec']:
        return decl
    if attr == 'static' and 'automatic' not in decl['attrspec']:
        decl['attrspec'].append(attr)
    elif attr == 'automatic' and 'static' not in decl['attrspec']:
        decl['attrspec'].append(attr)
    elif attr == 'public':
        if 'private' not in decl['attrspec']:
            decl['attrspec'].append(attr)
    elif attr == 'private':
        if 'public' not in decl['attrspec']:
            decl['attrspec'].append(attr)
    else:
        decl['attrspec'].append(attr)
    return decl


def setkindselector(decl, sel, force=0):
    if not decl:
        decl = {}
    if not sel:
        return decl
    if 'kindselector' not in decl:
        decl['kindselector'] = sel
        return decl
    for k in list(sel.keys()):
        if force or k not in decl['kindselector']:
            decl['kindselector'][k] = sel[k]
    return decl


def setcharselector(decl, sel, force=0):
    if not decl:
        decl = {}
    if not sel:
        return decl
    if 'charselector' not in decl:
        decl['charselector'] = sel
        return decl

    for k in list(sel.keys()):
        if force or k not in decl['charselector']:
            decl['charselector'][k] = sel[k]
    return decl


def getblockname(block, unknown='unknown'):
    if 'name' in block:
        return block['name']
    return unknown

# post processing


def setmesstext(block):
    global filepositiontext

    try:
        filepositiontext = 'In: %s:%s\n' % (block['from'], block['name'])
    except Exception:
        pass


def get_usedict(block):
    usedict = {}
    if 'parent_block' in block:
        usedict = get_usedict(block['parent_block'])
    if 'use' in block:
        usedict.update(block['use'])
    return usedict


def get_useparameters(block, param_map=None):
    global f90modulevars

    if param_map is None:
        param_map = {}
    usedict = get_usedict(block)
    if not usedict:
        return param_map
    for usename, mapping in list(usedict.items()):
        usename = usename.lower()
        if usename not in f90modulevars:
            outmess('get_useparameters: no module %s info used by %s\n' %
                    (usename, block.get('name')))
            continue
        mvars = f90modulevars[usename]
        params = get_parameters(mvars)
        if not params:
            continue
        # XXX: apply mapping
        if mapping:
            errmess('get_useparameters: mapping for %s not impl.\n' % (mapping))
        for k, v in list(params.items()):
            if k in param_map:
                outmess('get_useparameters: overriding parameter %s with'
                        ' value from module %s\n' % (repr(k), repr(usename)))
            param_map[k] = v

    return param_map


def postcrack2(block, tab='', param_map=None):
    global f90modulevars

    if not f90modulevars:
        return block
    if isinstance(block, list):
        ret = [postcrack2(g, tab=tab + '\t', param_map=param_map)
               for g in block]
        return ret
    setmesstext(block)
    outmess('%sBlock: %s\n' % (tab, block['name']), 0)

    if param_map is None:
        param_map = get_useparameters(block)

    if param_map is not None and 'vars' in block:
        vars = block['vars']
        for n in list(vars.keys()):
            var = vars[n]
            if 'kindselector' in var:
                kind = var['kindselector']
                if 'kind' in kind:
                    val = kind['kind']
                    if val in param_map:
                        kind['kind'] = param_map[val]
    new_body = [postcrack2(b, tab=tab + '\t', param_map=param_map)
                for b in block['body']]
    block['body'] = new_body

    return block


def postcrack(block, args=None, tab=''):
    """
    TODO:
          function return values
          determine expression types if in argument list
    """
    global usermodules, onlyfunctions

    if isinstance(block, list):
        gret = []
        uret = []
        for g in block:
            setmesstext(g)
            g = postcrack(g, tab=tab + '\t')
            # sort user routines to appear first
            if 'name' in g and '__user__' in g['name']:
                uret.append(g)
            else:
                gret.append(g)
        return uret + gret
    setmesstext(block)
    if not isinstance(block, dict) and 'block' not in block:
        raise Exception('postcrack: Expected block dictionary instead of ' +
                        str(block))
    if 'name' in block and not block['name'] == 'unknown_interface':
        outmess('%sBlock: %s\n' % (tab, block['name']), 0)
    block = analyzeargs(block)
    block = analyzecommon(block)
    block['vars'] = analyzevars(block)
    block['sortvars'] = sortvarnames(block['vars'])
    if block.get('args'):
        args = block['args']
    block['body'] = analyzebody(block, args, tab=tab)

    userisdefined = []
    if 'use' in block:
        useblock = block['use']
        for k in list(useblock.keys()):
            if '__user__' in k:
                userisdefined.append(k)
    else:
        useblock = {}
    name = ''
    if 'name' in block:
        name = block['name']
    # and not userisdefined: # Build a __user__ module
    if block.get('externals'):
        interfaced = []
        if 'interfaced' in block:
            interfaced = block['interfaced']
        mvars = copy.copy(block['vars'])
        if name:
            mname = name + '__user__routines'
        else:
            mname = 'unknown__user__routines'
        if mname in userisdefined:
            i = 1
            while '%s_%i' % (mname, i) in userisdefined:
                i = i + 1
            mname = '%s_%i' % (mname, i)
        interface = {'block': 'interface', 'body': [],
                     'vars': {}, 'name': name + '_user_interface'}
        for e in block['externals']:
            if e in interfaced:
                edef = []
                j = -1
                for b in block['body']:
                    j = j + 1
                    if b['block'] == 'interface':
                        i = -1
                        for bb in b['body']:
                            i = i + 1
                            if 'name' in bb and bb['name'] == e:
                                edef = copy.copy(bb)
                                del b['body'][i]
                                break
                        if edef:
                            if not b['body']:
                                del block['body'][j]
                            del interfaced[interfaced.index(e)]
                            break
                interface['body'].append(edef)
            else:
                if e in mvars and not isexternal(mvars[e]):
                    interface['vars'][e] = mvars[e]
        if interface['vars'] or interface['body']:
            block['interfaced'] = interfaced
            mblock = {'block': 'python module', 'body': [
                interface], 'vars': {}, 'name': mname, 'interfaced': block['externals']}
            useblock[mname] = {}
            usermodules.append(mblock)
    if useblock:
        block['use'] = useblock
    return block


def sortvarnames(vars):
    indep = []
    dep = []
    for v in list(vars.keys()):
        if 'depend' in vars[v] and vars[v]['depend']:
            dep.append(v)
        else:
            indep.append(v)
    n = len(dep)
    i = 0
    while dep:  # XXX: How to catch dependence cycles correctly?
        v = dep[0]
        fl = 0
        for w in dep[1:]:
            if w in vars[v]['depend']:
                fl = 1
                break
        if fl:
            dep = dep[1:] + [v]
            i = i + 1
            if i > n:
                errmess('sortvarnames: failed to compute dependencies because'
                        ' of cyclic dependencies between '
                        + ', '.join(dep) + '\n')
                indep = indep + dep
                break
        else:
            indep.append(v)
            dep = dep[1:]
            n = len(dep)
            i = 0
    return indep


def analyzecommon(block):
    if not hascommon(block):
        return block
    commonvars = []
    for k in list(block['common'].keys()):
        comvars = []
        for e in block['common'][k]:
            m = re.match(
                r'\A\s*\b(?P<name>.*?)\b\s*(\((?P<dims>.*?)\)|)\s*\Z', e, re.I)
            if m:
                dims = []
                if m.group('dims'):
                    dims = [x.strip()
                            for x in markoutercomma(m.group('dims')).split('@,@')]
                n = rmbadname1(m.group('name').strip())
                if n in block['vars']:
                    if 'attrspec' in block['vars'][n]:
                        block['vars'][n]['attrspec'].append(
                            'dimension(%s)' % (','.join(dims)))
                    else:
                        block['vars'][n]['attrspec'] = [
                            'dimension(%s)' % (','.join(dims))]
                else:
                    if dims:
                        block['vars'][n] = {
                            'attrspec': ['dimension(%s)' % (','.join(dims))]}
                    else:
                        block['vars'][n] = {}
                if n not in commonvars:
                    commonvars.append(n)
            else:
                n = e
                errmess(
                    'analyzecommon: failed to extract "<name>[(<dims>)]" from "%s" in common /%s/.\n' % (e, k))
            comvars.append(n)
        block['common'][k] = comvars
    if 'commonvars' not in block:
        block['commonvars'] = commonvars
    else:
        block['commonvars'] = block['commonvars'] + commonvars
    return block


def analyzebody(block, args, tab=''):
    global usermodules, skipfuncs, onlyfuncs, f90modulevars

    setmesstext(block)

    maybe_private = {
        key: value
        for key, value in block['vars'].items()
        if 'attrspec' not in value or 'public' not in value['attrspec']
    }

    body = []
    for b in block['body']:
        b['parent_block'] = block
        if b['block'] in ['function', 'subroutine']:
            if args is not None and b['name'] not in args:
                continue
            else:
                as_ = b['args']
            # Add private members to skipfuncs for gh-23879
            if b['name'] in maybe_private.keys():
                skipfuncs.append(b['name'])
            if b['name'] in skipfuncs:
                continue
            if onlyfuncs and b['name'] not in onlyfuncs:
                continue
            b['saved_interface'] = crack2fortrangen(
                b, '\n' + ' ' * 6, as_interface=True)

        else:
            as_ = args
        b = postcrack(b, as_, tab=tab + '\t')
        if b['block'] in ['interface', 'abstract interface'] and \
           not b['body'] and not b.get('implementedby'):
            if 'f2pyenhancements' not in b:
                continue
        if b['block'].replace(' ', '') == 'pythonmodule':
            usermodules.append(b)
        else:
            if b['block'] == 'module':
                f90modulevars[b['name']] = b['vars']
            body.append(b)
    return body


def buildimplicitrules(block):
    setmesstext(block)
    implicitrules = defaultimplicitrules
    attrrules = {}
    if 'implicit' in block:
        if block['implicit'] is None:
            implicitrules = None
            if verbose > 1:
                outmess(
                    'buildimplicitrules: no implicit rules for routine %s.\n' % repr(block['name']))
        else:
            for k in list(block['implicit'].keys()):
                if block['implicit'][k].get('typespec') not in ['static', 'automatic']:
                    implicitrules[k] = block['implicit'][k]
                else:
                    attrrules[k] = block['implicit'][k]['typespec']
    return implicitrules, attrrules


def myeval(e, g=None, l=None):
    """ Like `eval` but returns only integers and floats """
    r = eval(e, g, l)
    if type(r) in [int, float]:
        return r
    raise ValueError('r=%r' % (r))

getlincoef_re_1 = re.compile(r'\A\b\w+\b\Z', re.I)


def getlincoef(e, xset):  # e = a*x+b ; x in xset
    """
    Obtain ``a`` and ``b`` when ``e == "a*x+b"``, where ``x`` is a symbol in
    xset.

    >>> getlincoef('2*x + 1', {'x'})
    (2, 1, 'x')
    >>> getlincoef('3*x + x*2 + 2 + 1', {'x'})
    (5, 3, 'x')
    >>> getlincoef('0', {'x'})
    (0, 0, None)
    >>> getlincoef('0*x', {'x'})
    (0, 0, 'x')
    >>> getlincoef('x*x', {'x'})
    (None, None, None)

    This can be tricked by sufficiently complex expressions

    >>> getlincoef('(x - 0.5)*(x - 1.5)*(x - 1)*x + 2*x + 3', {'x'})
    (2.0, 3.0, 'x')
    """
    try:
        c = int(myeval(e, {}, {}))
        return 0, c, None
    except Exception:
        pass
    if getlincoef_re_1.match(e):
        return 1, 0, e
    len_e = len(e)
    for x in xset:
        if len(x) > len_e:
            continue
        if re.search(r'\w\s*\([^)]*\b' + x + r'\b', e):
            # skip function calls having x as an argument, e.g max(1, x)
            continue
        re_1 = re.compile(r'(?P<before>.*?)\b' + x + r'\b(?P<after>.*)', re.I)
        m = re_1.match(e)
        if m:
            try:
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (
                        m1.group('before'), 0, m1.group('after'))
                    m1 = re_1.match(ee)
                b = myeval(ee, {}, {})
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (
                        m1.group('before'), 1, m1.group('after'))
                    m1 = re_1.match(ee)
                a = myeval(ee, {}, {}) - b
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (
                        m1.group('before'), 0.5, m1.group('after'))
                    m1 = re_1.match(ee)
                c = myeval(ee, {}, {})
                # computing another point to be sure that expression is linear
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (
                        m1.group('before'), 1.5, m1.group('after'))
                    m1 = re_1.match(ee)
                c2 = myeval(ee, {}, {})
                if (a * 0.5 + b == c and a * 1.5 + b == c2):
                    return a, b, x
            except Exception:
                pass
            break
    return None, None, None


word_pattern = re.compile(r'\b[a-z][\w$]*\b', re.I)


def _get_depend_dict(name, vars, deps):
    if name in vars:
        words = vars[name].get('depend', [])

        if '=' in vars[name] and not isstring(vars[name]):
            for word in word_pattern.findall(vars[name]['=']):
                # The word_pattern may return values that are not
                # only variables, they can be string content for instance
                if word not in words and word in vars and word != name:
                    words.append(word)
        for word in words[:]:
            for w in deps.get(word, []) \
                    or _get_depend_dict(word, vars, deps):
                if w not in words:
                    words.append(w)
    else:
        outmess('_get_depend_dict: no dependence info for %s\n' % (repr(name)))
        words = []
    deps[name] = words
    return words


def _calc_depend_dict(vars):
    names = list(vars.keys())
    depend_dict = {}
    for n in names:
        _get_depend_dict(n, vars, depend_dict)
    return depend_dict


def get_sorted_names(vars):
    depend_dict = _calc_depend_dict(vars)
    names = []
    for name in list(depend_dict.keys()):
        if not depend_dict[name]:
            names.append(name)
            del depend_dict[name]
    while depend_dict:
        for name, lst in list(depend_dict.items()):
            new_lst = [n for n in lst if n in depend_dict]
            if not new_lst:
                names.append(name)
                del depend_dict[name]
            else:
                depend_dict[name] = new_lst
    return [name for name in names if name in vars]


def _kind_func(string):
    # XXX: return something sensible.
    if string[0] in "'\"":
        string = string[1:-1]
    if real16pattern.match(string):
        return 8
    elif real8pattern.match(string):
        return 4
    return 'kind(' + string + ')'


def _selected_int_kind_func(r):
    # XXX: This should be processor dependent
    m = 10 ** r
    if m <= 2 ** 8:
        return 1
    if m <= 2 ** 16:
        return 2
    if m <= 2 ** 32:
        return 4
    if m <= 2 ** 63:
        return 8
    if m <= 2 ** 128:
        return 16
    return -1


def _selected_real_kind_func(p, r=0, radix=0):
    # XXX: This should be processor dependent
    # This is only verified for 0 <= p <= 20, possibly good for p <= 33 and above
    if p < 7:
        return 4
    if p < 16:
        return 8
    machine = platform.machine().lower()
    if machine.startswith(('aarch64', 'alpha', 'arm64', 'loongarch', 'mips', 'power', 'ppc', 'riscv', 's390x', 'sparc')):
        if p <= 33:
            return 16
    else:
        if p < 19:
            return 10
        elif p <= 33:
            return 16
    return -1


def get_parameters(vars, global_params={}):
    params = copy.copy(global_params)
    g_params = copy.copy(global_params)
    for name, func in [('kind', _kind_func),
                       ('selected_int_kind', _selected_int_kind_func),
                       ('selected_real_kind', _selected_real_kind_func), ]:
        if name not in g_params:
            g_params[name] = func
    param_names = []
    for n in get_sorted_names(vars):
        if 'attrspec' in vars[n] and 'parameter' in vars[n]['attrspec']:
            param_names.append(n)
    kind_re = re.compile(r'\bkind\s*\(\s*(?P<value>.*)\s*\)', re.I)
    selected_int_kind_re = re.compile(
        r'\bselected_int_kind\s*\(\s*(?P<value>.*)\s*\)', re.I)
    selected_kind_re = re.compile(
        r'\bselected_(int|real)_kind\s*\(\s*(?P<value>.*)\s*\)', re.I)
    for n in param_names:
        if '=' in vars[n]:
            v = vars[n]['=']
            if islogical(vars[n]):
                v = v.lower()
                for repl in [
                    ('.false.', 'False'),
                    ('.true.', 'True'),
                    # TODO: test .eq., .neq., etc replacements.
                ]:
                    v = v.replace(*repl)

            v = kind_re.sub(r'kind("\1")', v)
            v = selected_int_kind_re.sub(r'selected_int_kind(\1)', v)

            # We need to act according to the data.
            # The easy case is if the data has a kind-specifier,
            # then we may easily remove those specifiers.
            # However, it may be that the user uses other specifiers...(!)
            is_replaced = False

            if 'kindselector' in vars[n]:
                # Remove kind specifier (including those defined
                # by parameters)
                if 'kind' in vars[n]['kindselector']:
                    orig_v_len = len(v)
                    v = v.replace('_' + vars[n]['kindselector']['kind'], '')
                    # Again, this will be true if even a single specifier
                    # has been replaced, see comment above.
                    is_replaced = len(v) < orig_v_len

            if not is_replaced:
                if not selected_kind_re.match(v):
                    v_ = v.split('_')
                    # In case there are additive parameters
                    if len(v_) > 1: 
                        v = ''.join(v_[:-1]).lower().replace(v_[-1].lower(), '')

            # Currently this will not work for complex numbers.
            # There is missing code for extracting a complex number,
            # which may be defined in either of these:
            #  a) (Re, Im)
            #  b) cmplx(Re, Im)
            #  c) dcmplx(Re, Im)
            #  d) cmplx(Re, Im, <prec>)

            if isdouble(vars[n]):
                tt = list(v)
                for m in real16pattern.finditer(v):
                    tt[m.start():m.end()] = list(
                        v[m.start():m.end()].lower().replace('d', 'e'))
                v = ''.join(tt)

            elif iscomplex(vars[n]):
                outmess(f'get_parameters[TODO]: '
                        f'implement evaluation of complex expression {v}\n')

            dimspec = ([s.removeprefix('dimension').strip()
                        for s in vars[n]['attrspec']
                       if s.startswith('dimension')] or [None])[0]

            # Handle _dp for gh-6624
            # Also fixes gh-20460
            if real16pattern.search(v):
                v = 8
            elif real8pattern.search(v):
                v = 4
            try:
                params[n] = param_eval(v, g_params, params, dimspec=dimspec)
            except Exception as msg:
                params[n] = v
                outmess(f'get_parameters: got "{msg}" on {n!r}\n')

            if isstring(vars[n]) and isinstance(params[n], int):
                params[n] = chr(params[n])
            nl = n.lower()
            if nl != n:
                params[nl] = params[n]
        else:
            print(vars[n])
            outmess(f'get_parameters:parameter {n!r} does not have value?!\n')
    return params


def _eval_length(length, params):
    if length in ['(:)', '(*)', '*']:
        return '(*)'
    return _eval_scalar(length, params)


_is_kind_number = re.compile(r'\d+_').match


def _eval_scalar(value, params):
    if _is_kind_number(value):
        value = value.split('_')[0]
    try:
        # TODO: use symbolic from PR #19805
        value = eval(value, {}, params)
        value = (repr if isinstance(value, str) else str)(value)
    except (NameError, SyntaxError, TypeError):
        return value
    except Exception as msg:
        errmess('"%s" in evaluating %r '
                '(available names: %s)\n'
                % (msg, value, list(params.keys())))
    return value


def analyzevars(block):
    """
    Sets correct dimension information for each variable/parameter
    """

    global f90modulevars

    setmesstext(block)
    implicitrules, attrrules = buildimplicitrules(block)
    vars = copy.copy(block['vars'])
    if block['block'] == 'function' and block['name'] not in vars:
        vars[block['name']] = {}
    if '' in block['vars']:
        del vars['']
        if 'attrspec' in block['vars']['']:
            gen = block['vars']['']['attrspec']
            for n in set(vars) | set(b['name'] for b in block['body']):
                for k in ['public', 'private']:
                    if k in gen:
                        vars[n] = setattrspec(vars.get(n, {}), k)
    svars = []
    args = block['args']
    for a in args:
        try:
            vars[a]
            svars.append(a)
        except KeyError:
            pass
    for n in list(vars.keys()):
        if n not in args:
            svars.append(n)

    params = get_parameters(vars, get_useparameters(block))
    # At this point, params are read and interpreted, but
    # the params used to define vars are not yet parsed
    dep_matches = {}
    name_match = re.compile(r'[A-Za-z][\w$]*').match
    for v in list(vars.keys()):
        m = name_match(v)
        if m:
            n = v[m.start():m.end()]
            try:
                dep_matches[n]
            except KeyError:
                dep_matches[n] = re.compile(r'.*\b%s\b' % (v), re.I).match
    for n in svars:
        if n[0] in list(attrrules.keys()):
            vars[n] = setattrspec(vars[n], attrrules[n[0]])
        if 'typespec' not in vars[n]:
            if not('attrspec' in vars[n] and 'external' in vars[n]['attrspec']):
                if implicitrules:
                    ln0 = n[0].lower()
                    for k in list(implicitrules[ln0].keys()):
                        if k == 'typespec' and implicitrules[ln0][k] == 'undefined':
                            continue
                        if k not in vars[n]:
                            vars[n][k] = implicitrules[ln0][k]
                        elif k == 'attrspec':
                            for l in implicitrules[ln0][k]:
                                vars[n] = setattrspec(vars[n], l)
                elif n in block['args']:
                    outmess('analyzevars: typespec of variable %s is not defined in routine %s.\n' % (
                        repr(n), block['name']))
        if 'charselector' in vars[n]:
            if 'len' in vars[n]['charselector']:
                l = vars[n]['charselector']['len']
                try:
                    l = str(eval(l, {}, params))
                except Exception:
                    pass
                vars[n]['charselector']['len'] = l

        if 'kindselector' in vars[n]:
            if 'kind' in vars[n]['kindselector']:
                l = vars[n]['kindselector']['kind']
                try:
                    l = str(eval(l, {}, params))
                except Exception:
                    pass
                vars[n]['kindselector']['kind'] = l

        dimension_exprs = {}
        if 'attrspec' in vars[n]:
            attr = vars[n]['attrspec']
            attr.reverse()
            vars[n]['attrspec'] = []
            dim, intent, depend, check, note = None, None, None, None, None
            for a in attr:
                if a[:9] == 'dimension':
                    dim = (a[9:].strip())[1:-1]
                elif a[:6] == 'intent':
                    intent = (a[6:].strip())[1:-1]
                elif a[:6] == 'depend':
                    depend = (a[6:].strip())[1:-1]
                elif a[:5] == 'check':
                    check = (a[5:].strip())[1:-1]
                elif a[:4] == 'note':
                    note = (a[4:].strip())[1:-1]
                else:
                    vars[n] = setattrspec(vars[n], a)
                if intent:
                    if 'intent' not in vars[n]:
                        vars[n]['intent'] = []
                    for c in [x.strip() for x in markoutercomma(intent).split('@,@')]:
                        # Remove spaces so that 'in out' becomes 'inout'
                        tmp = c.replace(' ', '')
                        if tmp not in vars[n]['intent']:
                            vars[n]['intent'].append(tmp)
                    intent = None
                if note:
                    note = note.replace('\\n\\n', '\n\n')
                    note = note.replace('\\n ', '\n')
                    if 'note' not in vars[n]:
                        vars[n]['note'] = [note]
                    else:
                        vars[n]['note'].append(note)
                    note = None
                if depend is not None:
                    if 'depend' not in vars[n]:
                        vars[n]['depend'] = []
                    for c in rmbadname([x.strip() for x in markoutercomma(depend).split('@,@')]):
                        if c not in vars[n]['depend']:
                            vars[n]['depend'].append(c)
                    depend = None
                if check is not None:
                    if 'check' not in vars[n]:
                        vars[n]['check'] = []
                    for c in [x.strip() for x in markoutercomma(check).split('@,@')]:
                        if c not in vars[n]['check']:
                            vars[n]['check'].append(c)
                    check = None
            if dim and 'dimension' not in vars[n]:
                vars[n]['dimension'] = []
                for d in rmbadname(
                        [x.strip() for x in markoutercomma(dim).split('@,@')]
                ):
                    # d is the expression inside the dimension declaration
                    # Evaluate `d` with respect to params
                    try:
                        # the dimension for this variable depends on a
                        # previously defined parameter
                        d = param_parse(d, params)
                    except (ValueError, IndexError, KeyError):
                        outmess(
                            'analyzevars: could not parse dimension for '
                            f'variable {d!r}\n'
                        )

                    dim_char = ':' if d == ':' else '*'
                    if d == dim_char:
                        dl = [dim_char]
                    else:
                        dl = markoutercomma(d, ':').split('@:@')
                    if len(dl) == 2 and '*' in dl:  # e.g. dimension(5:*)
                        dl = ['*']
                        d = '*'
                    if len(dl) == 1 and dl[0] != dim_char:
                        dl = ['1', dl[0]]
                    if len(dl) == 2:
                        d1, d2 = map(symbolic.Expr.parse, dl)
                        dsize = d2 - d1 + 1
                        d = dsize.tostring(language=symbolic.Language.C)
                        # find variables v that define d as a linear
                        # function, `d == a * v + b`, and store
                        # coefficients a and b for further analysis.
                        solver_and_deps = {}
                        for v in block['vars']:
                            s = symbolic.as_symbol(v)
                            if dsize.contains(s):
                                try:
                                    a, b = dsize.linear_solve(s)

                                    def solve_v(s, a=a, b=b):
                                        return (s - b) / a

                                    all_symbols = set(a.symbols())
                                    all_symbols.update(b.symbols())
                                except RuntimeError as msg:
                                    # d is not a linear function of v,
                                    # however, if v can be determined
                                    # from d using other means,
                                    # implement the corresponding
                                    # solve_v function here.
                                    solve_v = None
                                    all_symbols = set(dsize.symbols())
                                v_deps = set(
                                    s.data for s in all_symbols
                                    if s.data in vars)
                                solver_and_deps[v] = solve_v, list(v_deps)
                        # Note that dsize may contain symbols that are
                        # not defined in block['vars']. Here we assume
                        # these correspond to Fortran/C intrinsic
                        # functions or that are defined by other
                        # means. We'll let the compiler validate the
                        # definiteness of such symbols.
                        dimension_exprs[d] = solver_and_deps
                    vars[n]['dimension'].append(d)

        if 'check' not in vars[n] and 'args' in block and n in block['args']:
            # n is an argument that has no checks defined. Here we
            # generate some consistency checks for n, and when n is an
            # array, generate checks for its dimensions and construct
            # initialization expressions.
            n_deps = vars[n].get('depend', [])
            n_checks = []
            n_is_input = l_or(isintent_in, isintent_inout,
                              isintent_inplace)(vars[n])
            if isarray(vars[n]):  # n is array
                for i, d in enumerate(vars[n]['dimension']):
                    coeffs_and_deps = dimension_exprs.get(d)
                    if coeffs_and_deps is None:
                        # d is `:` or `*` or a constant expression
                        pass
                    elif n_is_input:
                        # n is an input array argument and its shape
                        # may define variables used in dimension
                        # specifications.
                        for v, (solver, deps) in coeffs_and_deps.items():
                            def compute_deps(v, deps):
                                for v1 in coeffs_and_deps.get(v, [None, []])[1]:
                                    if v1 not in deps:
                                        deps.add(v1)
                                        compute_deps(v1, deps)
                            all_deps = set()
                            compute_deps(v, all_deps)
                            if (v in n_deps
                                 or '=' in vars[v]
                                 or 'depend' in vars[v]):
                                # Skip a variable that
                                # - n depends on
                                # - has user-defined initialization expression
                                # - has user-defined dependencies
                                continue
                            if solver is not None and v not in all_deps:
                                # v can be solved from d, hence, we
                                # make it an optional argument with
                                # initialization expression:
                                is_required = False
                                init = solver(symbolic.as_symbol(
                                    f'shape({n}, {i})'))
                                init = init.tostring(
                                    language=symbolic.Language.C)
                                vars[v]['='] = init
                                # n needs to be initialized before v. So,
                                # making v dependent on n and on any
                                # variables in solver or d.
                                vars[v]['depend'] = [n] + deps
                                if 'check' not in vars[v]:
                                    # add check only when no
                                    # user-specified checks exist
                                    vars[v]['check'] = [
                                        f'shape({n}, {i}) == {d}']
                            else:
                                # d is a non-linear function on v,
                                # hence, v must be a required input
                                # argument that n will depend on
                                is_required = True
                                if 'intent' not in vars[v]:
                                    vars[v]['intent'] = []
                                if 'in' not in vars[v]['intent']:
                                    vars[v]['intent'].append('in')
                                # v needs to be initialized before n
                                n_deps.append(v)
                                n_checks.append(
                                    f'shape({n}, {i}) == {d}')
                            v_attr = vars[v].get('attrspec', [])
                            if not ('optional' in v_attr
                                    or 'required' in v_attr):
                                v_attr.append(
                                    'required' if is_required else 'optional')
                            if v_attr:
                                vars[v]['attrspec'] = v_attr
                    if coeffs_and_deps is not None:
                        # extend v dependencies with ones specified in attrspec
                        for v, (solver, deps) in coeffs_and_deps.items():
                            v_deps = vars[v].get('depend', [])
                            for aa in vars[v].get('attrspec', []):
                                if aa.startswith('depend'):
                                    aa = ''.join(aa.split())
                                    v_deps.extend(aa[7:-1].split(','))
                            if v_deps:
                                vars[v]['depend'] = list(set(v_deps))
                            if n not in v_deps:
                                n_deps.append(v)
            elif isstring(vars[n]):
                if 'charselector' in vars[n]:
                    if '*' in vars[n]['charselector']:
                        length = _eval_length(vars[n]['charselector']['*'],
                                              params)
                        vars[n]['charselector']['*'] = length
                    elif 'len' in vars[n]['charselector']:
                        length = _eval_length(vars[n]['charselector']['len'],
                                              params)
                        del vars[n]['charselector']['len']
                        vars[n]['charselector']['*'] = length
            if n_checks:
                vars[n]['check'] = n_checks
            if n_deps:
                vars[n]['depend'] = list(set(n_deps))

        if '=' in vars[n]:
            if 'attrspec' not in vars[n]:
                vars[n]['attrspec'] = []
            if ('optional' not in vars[n]['attrspec']) and \
               ('required' not in vars[n]['attrspec']):
                vars[n]['attrspec'].append('optional')
            if 'depend' not in vars[n]:
                vars[n]['depend'] = []
                for v, m in list(dep_matches.items()):
                    if m(vars[n]['=']):
                        vars[n]['depend'].append(v)
                if not vars[n]['depend']:
                    del vars[n]['depend']
            if isscalar(vars[n]):
                vars[n]['='] = _eval_scalar(vars[n]['='], params)

    for n in list(vars.keys()):
        if n == block['name']:  # n is block name
            if 'note' in vars[n]:
                block['note'] = vars[n]['note']
            if block['block'] == 'function':
                if 'result' in block and block['result'] in vars:
                    vars[n] = appenddecl(vars[n], vars[block['result']])
                if 'prefix' in block:
                    pr = block['prefix']
                    pr1 = pr.replace('pure', '')
                    ispure = (not pr == pr1)
                    pr = pr1.replace('recursive', '')
                    isrec = (not pr == pr1)
                    m = typespattern[0].match(pr)
                    if m:
                        typespec, selector, attr, edecl = cracktypespec0(
                            m.group('this'), m.group('after'))
                        kindselect, charselect, typename = cracktypespec(
                            typespec, selector)
                        vars[n]['typespec'] = typespec
                        try:
                            if block['result']:
                                vars[block['result']]['typespec'] = typespec
                        except Exception:
                            pass
                        if kindselect:
                            if 'kind' in kindselect:
                                try:
                                    kindselect['kind'] = eval(
                                        kindselect['kind'], {}, params)
                                except Exception:
                                    pass
                            vars[n]['kindselector'] = kindselect
                        if charselect:
                            vars[n]['charselector'] = charselect
                        if typename:
                            vars[n]['typename'] = typename
                        if ispure:
                            vars[n] = setattrspec(vars[n], 'pure')
                        if isrec:
                            vars[n] = setattrspec(vars[n], 'recursive')
                    else:
                        outmess(
                            'analyzevars: prefix (%s) were not used\n' % repr(block['prefix']))
    if block['block'] not in ['module', 'pythonmodule', 'python module', 'block data']:
        if 'commonvars' in block:
            neededvars = copy.copy(block['args'] + block['commonvars'])
        else:
            neededvars = copy.copy(block['args'])
        for n in list(vars.keys()):
            if l_or(isintent_callback, isintent_aux)(vars[n]):
                neededvars.append(n)
        if 'entry' in block:
            neededvars.extend(list(block['entry'].keys()))
            for k in list(block['entry'].keys()):
                for n in block['entry'][k]:
                    if n not in neededvars:
                        neededvars.append(n)
        if block['block'] == 'function':
            if 'result' in block:
                neededvars.append(block['result'])
            else:
                neededvars.append(block['name'])
        if block['block'] in ['subroutine', 'function']:
            name = block['name']
            if name in vars and 'intent' in vars[name]:
                block['intent'] = vars[name]['intent']
        if block['block'] == 'type':
            neededvars.extend(list(vars.keys()))
        for n in list(vars.keys()):
            if n not in neededvars:
                del vars[n]
    return vars


analyzeargs_re_1 = re.compile(r'\A[a-z]+[\w$]*\Z', re.I)


def param_eval(v, g_params, params, dimspec=None):
    """
    Creates a dictionary of indices and values for each parameter in a
    parameter array to be evaluated later.

    WARNING: It is not possible to initialize multidimensional array
    parameters e.g. dimension(-3:1, 4, 3:5) at this point. This is because in
    Fortran initialization through array constructor requires the RESHAPE
    intrinsic function. Since the right-hand side of the parameter declaration
    is not executed in f2py, but rather at the compiled c/fortran extension,
    later, it is not possible to execute a reshape of a parameter array.
    One issue remains: if the user wants to access the array parameter from
    python, we should either
    1) allow them to access the parameter array using python standard indexing
       (which is often incompatible with the original fortran indexing)
    2) allow the parameter array to be accessed in python as a dictionary with
       fortran indices as keys
    We are choosing 2 for now.
    """
    if dimspec is None:
        try:
            p = eval(v, g_params, params)
        except Exception as msg:
            p = v
            outmess(f'param_eval: got "{msg}" on {v!r}\n')
        return p

    # This is an array parameter.
    # First, we parse the dimension information
    if len(dimspec) < 2 or dimspec[::len(dimspec)-1] != "()":
        raise ValueError(f'param_eval: dimension {dimspec} can\'t be parsed')
    dimrange = dimspec[1:-1].split(',')
    if len(dimrange) == 1:
        # e.g. dimension(2) or dimension(-1:1)
        dimrange = dimrange[0].split(':')
        # now, dimrange is a list of 1 or 2 elements
        if len(dimrange) == 1:
            bound = param_parse(dimrange[0], params)
            dimrange = range(1, int(bound)+1)
        else:
            lbound = param_parse(dimrange[0], params)
            ubound = param_parse(dimrange[1], params)
            dimrange = range(int(lbound), int(ubound)+1)
    else:
        raise ValueError('param_eval: multidimensional array parameters '
                         f'{dimspec} not supported')

    # Parse parameter value
    v = (v[2:-2] if v.startswith('(/') else v).split(',')
    v_eval = []
    for item in v:
        try:
            item = eval(item, g_params, params)
        except Exception as msg:
            outmess(f'param_eval: got "{msg}" on {item!r}\n')
        v_eval.append(item)

    p = dict(zip(dimrange, v_eval))

    return p


def param_parse(d, params):
    """Recursively parse array dimensions.

    Parses the declaration of an array variable or parameter
    `dimension` keyword, and is called recursively if the
    dimension for this array is a previously defined parameter
    (found in `params`).

    Parameters
    ----------
    d : str
        Fortran expression describing the dimension of an array.
    params : dict
        Previously parsed parameters declared in the Fortran source file.

    Returns
    -------
    out : str
        Parsed dimension expression.

    Examples
    --------

    * If the line being analyzed is

      `integer, parameter, dimension(2) :: pa = (/ 3, 5 /)`

      then `d = 2` and we return immediately, with

    >>> d = '2'
    >>> param_parse(d, params)
    2

    * If the line being analyzed is

      `integer, parameter, dimension(pa) :: pb = (/1, 2, 3/)`

      then `d = 'pa'`; since `pa` is a previously parsed parameter,
      and `pa = 3`, we call `param_parse` recursively, to obtain

    >>> d = 'pa'
    >>> params = {'pa': 3}
    >>> param_parse(d, params)
    3

    * If the line being analyzed is

      `integer, parameter, dimension(pa(1)) :: pb = (/1, 2, 3/)`

      then `d = 'pa(1)'`; since `pa` is a previously parsed parameter,
      and `pa(1) = 3`, we call `param_parse` recursively, to obtain

    >>> d = 'pa(1)'
    >>> params = dict(pa={1: 3, 2: 5})
    >>> param_parse(d, params)
    3
    """
    if "(" in d:
        # this dimension expression is an array
        dname = d[:d.find("(")]
        ddims = d[d.find("(")+1:d.rfind(")")]
        # this dimension expression is also a parameter;
        # parse it recursively
        index = int(param_parse(ddims, params))
        return str(params[dname][index])
    elif d in params:
        return str(params[d])
    else:
        for p in params:
            re_1 = re.compile(
                r'(?P<before>.*?)\b' + p + r'\b(?P<after>.*)', re.I
            )
            m = re_1.match(d)
            while m:
                d = m.group('before') + \
                    str(params[p]) + m.group('after')
                m = re_1.match(d)
        return d


def expr2name(a, block, args=[]):
    orig_a = a
    a_is_expr = not analyzeargs_re_1.match(a)
    if a_is_expr:  # `a` is an expression
        implicitrules, attrrules = buildimplicitrules(block)
        at = determineexprtype(a, block['vars'], implicitrules)
        na = 'e_'
        for c in a:
            c = c.lower()
            if c not in string.ascii_lowercase + string.digits:
                c = '_'
            na = na + c
        if na[-1] == '_':
            na = na + 'e'
        else:
            na = na + '_e'
        a = na
        while a in block['vars'] or a in block['args']:
            a = a + 'r'
    if a in args:
        k = 1
        while a + str(k) in args:
            k = k + 1
        a = a + str(k)
    if a_is_expr:
        block['vars'][a] = at
    else:
        if a not in block['vars']:
            if orig_a in block['vars']:
                block['vars'][a] = block['vars'][orig_a]
            else:
                block['vars'][a] = {}
        if 'externals' in block and orig_a in block['externals'] + block['interfaced']:
            block['vars'][a] = setattrspec(block['vars'][a], 'external')
    return a


def analyzeargs(block):
    setmesstext(block)
    implicitrules, _ = buildimplicitrules(block)
    if 'args' not in block:
        block['args'] = []
    args = []
    for a in block['args']:
        a = expr2name(a, block, args)
        args.append(a)
    block['args'] = args
    if 'entry' in block:
        for k, args1 in list(block['entry'].items()):
            for a in args1:
                if a not in block['vars']:
                    block['vars'][a] = {}

    for b in block['body']:
        if b['name'] in args:
            if 'externals' not in block:
                block['externals'] = []
            if b['name'] not in block['externals']:
                block['externals'].append(b['name'])
    if 'result' in block and block['result'] not in block['vars']:
        block['vars'][block['result']] = {}
    return block

determineexprtype_re_1 = re.compile(r'\A\(.+?,.+?\)\Z', re.I)
determineexprtype_re_2 = re.compile(r'\A[+-]?\d+(_(?P<name>\w+)|)\Z', re.I)
determineexprtype_re_3 = re.compile(
    r'\A[+-]?[\d.]+[-\d+de.]*(_(?P<name>\w+)|)\Z', re.I)
determineexprtype_re_4 = re.compile(r'\A\(.*\)\Z', re.I)
determineexprtype_re_5 = re.compile(r'\A(?P<name>\w+)\s*\(.*?\)\s*\Z', re.I)


def _ensure_exprdict(r):
    if isinstance(r, int):
        return {'typespec': 'integer'}
    if isinstance(r, float):
        return {'typespec': 'real'}
    if isinstance(r, complex):
        return {'typespec': 'complex'}
    if isinstance(r, dict):
        return r
    raise AssertionError(repr(r))


def determineexprtype(expr, vars, rules={}):
    if expr in vars:
        return _ensure_exprdict(vars[expr])
    expr = expr.strip()
    if determineexprtype_re_1.match(expr):
        return {'typespec': 'complex'}
    m = determineexprtype_re_2.match(expr)
    if m:
        if 'name' in m.groupdict() and m.group('name'):
            outmess(
                'determineexprtype: selected kind types not supported (%s)\n' % repr(expr))
        return {'typespec': 'integer'}
    m = determineexprtype_re_3.match(expr)
    if m:
        if 'name' in m.groupdict() and m.group('name'):
            outmess(
                'determineexprtype: selected kind types not supported (%s)\n' % repr(expr))
        return {'typespec': 'real'}
    for op in ['+', '-', '*', '/']:
        for e in [x.strip() for x in markoutercomma(expr, comma=op).split('@' + op + '@')]:
            if e in vars:
                return _ensure_exprdict(vars[e])
    t = {}
    if determineexprtype_re_4.match(expr):  # in parenthesis
        t = determineexprtype(expr[1:-1], vars, rules)
    else:
        m = determineexprtype_re_5.match(expr)
        if m:
            rn = m.group('name')
            t = determineexprtype(m.group('name'), vars, rules)
            if t and 'attrspec' in t:
                del t['attrspec']
            if not t:
                if rn[0] in rules:
                    return _ensure_exprdict(rules[rn[0]])
    if expr[0] in '\'"':
        return {'typespec': 'character', 'charselector': {'*': '*'}}
    if not t:
        outmess(
            'determineexprtype: could not determine expressions (%s) type.\n' % (repr(expr)))
    return t

######


def crack2fortrangen(block, tab='\n', as_interface=False):
    global skipfuncs, onlyfuncs

    setmesstext(block)
    ret = ''
    if isinstance(block, list):
        for g in block:
            if g and g['block'] in ['function', 'subroutine']:
                if g['name'] in skipfuncs:
                    continue
                if onlyfuncs and g['name'] not in onlyfuncs:
                    continue
            ret = ret + crack2fortrangen(g, tab, as_interface=as_interface)
        return ret
    prefix = ''
    name = ''
    args = ''
    blocktype = block['block']
    if blocktype == 'program':
        return ''
    argsl = []
    if 'name' in block:
        name = block['name']
    if 'args' in block:
        vars = block['vars']
        for a in block['args']:
            a = expr2name(a, block, argsl)
            if not isintent_callback(vars[a]):
                argsl.append(a)
        if block['block'] == 'function' or argsl:
            args = '(%s)' % ','.join(argsl)
    f2pyenhancements = ''
    if 'f2pyenhancements' in block:
        for k in list(block['f2pyenhancements'].keys()):
            f2pyenhancements = '%s%s%s %s' % (
                f2pyenhancements, tab + tabchar, k, block['f2pyenhancements'][k])
    intent_lst = block.get('intent', [])[:]
    if blocktype == 'function' and 'callback' in intent_lst:
        intent_lst.remove('callback')
    if intent_lst:
        f2pyenhancements = '%s%sintent(%s) %s' %\
                           (f2pyenhancements, tab + tabchar,
                            ','.join(intent_lst), name)
    use = ''
    if 'use' in block:
        use = use2fortran(block['use'], tab + tabchar)
    common = ''
    if 'common' in block:
        common = common2fortran(block['common'], tab + tabchar)
    if name == 'unknown_interface':
        name = ''
    result = ''
    if 'result' in block:
        result = ' result (%s)' % block['result']
        if block['result'] not in argsl:
            argsl.append(block['result'])
    body = crack2fortrangen(block['body'], tab + tabchar, as_interface=as_interface)
    vars = vars2fortran(
        block, block['vars'], argsl, tab + tabchar, as_interface=as_interface)
    mess = ''
    if 'from' in block and not as_interface:
        mess = '! in %s' % block['from']
    if 'entry' in block:
        entry_stmts = ''
        for k, i in list(block['entry'].items()):
            entry_stmts = '%s%sentry %s(%s)' \
                          % (entry_stmts, tab + tabchar, k, ','.join(i))
        body = body + entry_stmts
    if blocktype == 'block data' and name == '_BLOCK_DATA_':
        name = ''
    ret = '%s%s%s %s%s%s %s%s%s%s%s%s%send %s %s' % (
        tab, prefix, blocktype, name, args, result, mess, f2pyenhancements, use, vars, common, body, tab, blocktype, name)
    return ret


def common2fortran(common, tab=''):
    ret = ''
    for k in list(common.keys()):
        if k == '_BLNK_':
            ret = '%s%scommon %s' % (ret, tab, ','.join(common[k]))
        else:
            ret = '%s%scommon /%s/ %s' % (ret, tab, k, ','.join(common[k]))
    return ret


def use2fortran(use, tab=''):
    ret = ''
    for m in list(use.keys()):
        ret = '%s%suse %s,' % (ret, tab, m)
        if use[m] == {}:
            if ret and ret[-1] == ',':
                ret = ret[:-1]
            continue
        if 'only' in use[m] and use[m]['only']:
            ret = '%s only:' % (ret)
        if 'map' in use[m] and use[m]['map']:
            c = ' '
            for k in list(use[m]['map'].keys()):
                if k == use[m]['map'][k]:
                    ret = '%s%s%s' % (ret, c, k)
                    c = ','
                else:
                    ret = '%s%s%s=>%s' % (ret, c, k, use[m]['map'][k])
                    c = ','
        if ret and ret[-1] == ',':
            ret = ret[:-1]
    return ret


def true_intent_list(var):
    lst = var['intent']
    ret = []
    for intent in lst:
        try:
            f = globals()['isintent_%s' % intent]
        except KeyError:
            pass
        else:
            if f(var):
                ret.append(intent)
    return ret


def vars2fortran(block, vars, args, tab='', as_interface=False):
    setmesstext(block)
    ret = ''
    nout = []
    for a in args:
        if a in block['vars']:
            nout.append(a)
    if 'commonvars' in block:
        for a in block['commonvars']:
            if a in vars:
                if a not in nout:
                    nout.append(a)
            else:
                errmess(
                    'vars2fortran: Confused?!: "%s" is not defined in vars.\n' % a)
    if 'varnames' in block:
        nout.extend(block['varnames'])
    if not as_interface:
        for a in list(vars.keys()):
            if a not in nout:
                nout.append(a)
    for a in nout:
        if 'depend' in vars[a]:
            for d in vars[a]['depend']:
                if d in vars and 'depend' in vars[d] and a in vars[d]['depend']:
                    errmess(
                        'vars2fortran: Warning: cross-dependence between variables "%s" and "%s"\n' % (a, d))
        if 'externals' in block and a in block['externals']:
            if isintent_callback(vars[a]):
                ret = '%s%sintent(callback) %s' % (ret, tab, a)
            ret = '%s%sexternal %s' % (ret, tab, a)
            if isoptional(vars[a]):
                ret = '%s%soptional %s' % (ret, tab, a)
            if a in vars and 'typespec' not in vars[a]:
                continue
            cont = 1
            for b in block['body']:
                if a == b['name'] and b['block'] == 'function':
                    cont = 0
                    break
            if cont:
                continue
        if a not in vars:
            show(vars)
            outmess('vars2fortran: No definition for argument "%s".\n' % a)
            continue
        if a == block['name']:
            if block['block'] != 'function' or block.get('result'):
                # 1) skip declaring a variable that name matches with
                #    subroutine name
                # 2) skip declaring function when its type is
                #    declared via `result` construction
                continue
        if 'typespec' not in vars[a]:
            if 'attrspec' in vars[a] and 'external' in vars[a]['attrspec']:
                if a in args:
                    ret = '%s%sexternal %s' % (ret, tab, a)
                continue
            show(vars[a])
            outmess('vars2fortran: No typespec for argument "%s".\n' % a)
            continue
        vardef = vars[a]['typespec']
        if vardef == 'type' and 'typename' in vars[a]:
            vardef = '%s(%s)' % (vardef, vars[a]['typename'])
        selector = {}
        if 'kindselector' in vars[a]:
            selector = vars[a]['kindselector']
        elif 'charselector' in vars[a]:
            selector = vars[a]['charselector']
        if '*' in selector:
            if selector['*'] in ['*', ':']:
                vardef = '%s*(%s)' % (vardef, selector['*'])
            else:
                vardef = '%s*%s' % (vardef, selector['*'])
        else:
            if 'len' in selector:
                vardef = '%s(len=%s' % (vardef, selector['len'])
                if 'kind' in selector:
                    vardef = '%s,kind=%s)' % (vardef, selector['kind'])
                else:
                    vardef = '%s)' % (vardef)
            elif 'kind' in selector:
                vardef = '%s(kind=%s)' % (vardef, selector['kind'])
        c = ' '
        if 'attrspec' in vars[a]:
            attr = [l for l in vars[a]['attrspec']
                    if l not in ['external']]
            if as_interface and 'intent(in)' in attr and 'intent(out)' in attr:
                # In Fortran, intent(in, out) are conflicting while
                # intent(in, out) can be specified only via
                # `!f2py intent(out) ..`.
                # So, for the Fortran interface, we'll drop
                # intent(out) to resolve the conflict.
                attr.remove('intent(out)')
            if attr:
                vardef = '%s, %s' % (vardef, ','.join(attr))
                c = ','
        if 'dimension' in vars[a]:
            vardef = '%s%sdimension(%s)' % (
                vardef, c, ','.join(vars[a]['dimension']))
            c = ','
        if 'intent' in vars[a]:
            lst = true_intent_list(vars[a])
            if lst:
                vardef = '%s%sintent(%s)' % (vardef, c, ','.join(lst))
            c = ','
        if 'check' in vars[a]:
            vardef = '%s%scheck(%s)' % (vardef, c, ','.join(vars[a]['check']))
            c = ','
        if 'depend' in vars[a]:
            vardef = '%s%sdepend(%s)' % (
                vardef, c, ','.join(vars[a]['depend']))
            c = ','
        if '=' in vars[a]:
            v = vars[a]['=']
            if vars[a]['typespec'] in ['complex', 'double complex']:
                try:
                    v = eval(v)
                    v = '(%s,%s)' % (v.real, v.imag)
                except Exception:
                    pass
            vardef = '%s :: %s=%s' % (vardef, a, v)
        else:
            vardef = '%s :: %s' % (vardef, a)
        ret = '%s%s%s' % (ret, tab, vardef)
    return ret
######


# We expose post_processing_hooks as global variable so that
# user-libraries could register their own hooks to f2py.
post_processing_hooks = []


def crackfortran(files):
    global usermodules, post_processing_hooks

    outmess('Reading fortran codes...\n', 0)
    readfortrancode(files, crackline)
    outmess('Post-processing...\n', 0)
    usermodules = []
    postlist = postcrack(grouplist[0])
    outmess('Applying post-processing hooks...\n', 0)
    for hook in post_processing_hooks:
        outmess(f'  {hook.__name__}\n', 0)
        postlist = traverse(postlist, hook)
    outmess('Post-processing (stage 2)...\n', 0)
    postlist = postcrack2(postlist)
    return usermodules + postlist


def crack2fortran(block):
    global f2py_version

    pyf = crack2fortrangen(block) + '\n'
    header = """!    -*- f90 -*-
! Note: the context of this file is case sensitive.
"""
    footer = """
! This file was auto-generated with f2py (version:%s).
! See:
! https://web.archive.org/web/20140822061353/http://cens.ioc.ee/projects/f2py2e
""" % (f2py_version)
    return header + pyf + footer


def _is_visit_pair(obj):
    return (isinstance(obj, tuple)
            and len(obj) == 2
            and isinstance(obj[0], (int, str)))


def traverse(obj, visit, parents=[], result=None, *args, **kwargs):
    '''Traverse f2py data structure with the following visit function:

    def visit(item, parents, result, *args, **kwargs):
        """

        parents is a list of key-"f2py data structure" pairs from which
        items are taken from.

        result is a f2py data structure that is filled with the
        return value of the visit function.

        item is 2-tuple (index, value) if parents[-1][1] is a list
        item is 2-tuple (key, value) if parents[-1][1] is a dict

        The return value of visit must be None, or of the same kind as
        item, that is, if parents[-1] is a list, the return value must
        be 2-tuple (new_index, new_value), or if parents[-1] is a
        dict, the return value must be 2-tuple (new_key, new_value).

        If new_index or new_value is None, the return value of visit
        is ignored, that is, it will not be added to the result.

        If the return value is None, the content of obj will be
        traversed, otherwise not.
        """
    '''

    if _is_visit_pair(obj):
        if obj[0] == 'parent_block':
            # avoid infinite recursion
            return obj
        new_result = visit(obj, parents, result, *args, **kwargs)
        if new_result is not None:
            assert _is_visit_pair(new_result)
            return new_result
        parent = obj
        result_key, obj = obj
    else:
        parent = (None, obj)
        result_key = None

    if isinstance(obj, list):
        new_result = []
        for index, value in enumerate(obj):
            new_index, new_item = traverse((index, value), visit,
                                           parents=parents + [parent],
                                           result=result, *args, **kwargs)
            if new_index is not None:
                new_result.append(new_item)
    elif isinstance(obj, dict):
        new_result = dict()
        for key, value in obj.items():
            new_key, new_value = traverse((key, value), visit,
                                          parents=parents + [parent],
                                          result=result, *args, **kwargs)
            if new_key is not None:
                new_result[new_key] = new_value
    else:
        new_result = obj

    if result_key is None:
        return new_result
    return result_key, new_result


def character_backward_compatibility_hook(item, parents, result,
                                          *args, **kwargs):
    """Previously, Fortran character was incorrectly treated as
    character*1. This hook fixes the usage of the corresponding
    variables in `check`, `dimension`, `=`, and `callstatement`
    expressions.

    The usage of `char*` in `callprotoargument` expression can be left
    unchanged because C `character` is C typedef of `char`, although,
    new implementations should use `character*` in the corresponding
    expressions.

    See https://github.com/numpy/numpy/pull/19388 for more information.

    """
    parent_key, parent_value = parents[-1]
    key, value = item

    def fix_usage(varname, value):
        value = re.sub(r'[*]\s*\b' + varname + r'\b', varname, value)
        value = re.sub(r'\b' + varname + r'\b\s*[\[]\s*0\s*[\]]',
                       varname, value)
        return value

    if parent_key in ['dimension', 'check']:
        assert parents[-3][0] == 'vars'
        vars_dict = parents[-3][1]
    elif key == '=':
        assert parents[-2][0] == 'vars'
        vars_dict = parents[-2][1]
    else:
        vars_dict = None

    new_value = None
    if vars_dict is not None:
        new_value = value
        for varname, vd in vars_dict.items():
            if ischaracter(vd):
                new_value = fix_usage(varname, new_value)
    elif key == 'callstatement':
        vars_dict = parents[-2][1]['vars']
        new_value = value
        for varname, vd in vars_dict.items():
            if ischaracter(vd):
                # replace all occurrences of `<varname>` with
                # `&<varname>` in argument passing
                new_value = re.sub(
                    r'(?<![&])\b' + varname + r'\b', '&' + varname, new_value)

    if new_value is not None:
        if new_value != value:
            # We report the replacements here so that downstream
            # software could update their source codes
            # accordingly. However, such updates are recommended only
            # when BC with numpy 1.21 or older is not required.
            outmess(f'character_bc_hook[{parent_key}.{key}]:'
                    f' replaced `{value}` -> `{new_value}`\n', 1)
        return (key, new_value)


post_processing_hooks.append(character_backward_compatibility_hook)


if __name__ == "__main__":
    files = []
    funcs = []
    f = 1
    f2 = 0
    f3 = 0
    showblocklist = 0
    for l in sys.argv[1:]:
        if l == '':
            pass
        elif l[0] == ':':
            f = 0
        elif l == '-quiet':
            quiet = 1
            verbose = 0
        elif l == '-verbose':
            verbose = 2
            quiet = 0
        elif l == '-fix':
            if strictf77:
                outmess(
                    'Use option -f90 before -fix if Fortran 90 code is in fix form.\n', 0)
            skipemptyends = 1
            sourcecodeform = 'fix'
        elif l == '-skipemptyends':
            skipemptyends = 1
        elif l == '--ignore-contains':
            ignorecontains = 1
        elif l == '-f77':
            strictf77 = 1
            sourcecodeform = 'fix'
        elif l == '-f90':
            strictf77 = 0
            sourcecodeform = 'free'
            skipemptyends = 1
        elif l == '-h':
            f2 = 1
        elif l == '-show':
            showblocklist = 1
        elif l == '-m':
            f3 = 1
        elif l[0] == '-':
            errmess('Unknown option %s\n' % repr(l))
        elif f2:
            f2 = 0
            pyffilename = l
        elif f3:
            f3 = 0
            f77modulename = l
        elif f:
            try:
                open(l).close()
                files.append(l)
            except OSError as detail:
                errmess(f'OSError: {detail!s}\n')
        else:
            funcs.append(l)
    if not strictf77 and f77modulename and not skipemptyends:
        outmess("""\
  Warning: You have specified module name for non Fortran 77 code that
  should not need one (expect if you are scanning F90 code for non
  module blocks but then you should use flag -skipemptyends and also
  be sure that the files do not contain programs without program
  statement).
""", 0)

    postlist = crackfortran(files)
    if pyffilename:
        outmess('Writing fortran code to file %s\n' % repr(pyffilename), 0)
        pyf = crack2fortran(postlist)
        with open(pyffilename, 'w') as f:
            f.write(pyf)
    if showblocklist:
        show(postlist)
<|endoftext|>"""
Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
from . import __version__
f2py_version = __version__.version

import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map

# The environment provided by auxfuncs.py is needed for some calls to eval.
# As the needed functions cannot be determined by static inspection of the
# code, it is safest to use import * pending a major refactoring of f2py.
from .auxfuncs import *

__all__ = [
    'getctype', 'getstrlength', 'getarrdims', 'getpydocsign',
    'getarrdocsign', 'getinit', 'sign2map', 'routsign2map', 'modsign2map',
    'cb_sign2map', 'cb_routsign2map', 'common_sign2map', 'process_f2cmap_dict'
]


depargs = []
lcb_map = {}
lcb2_map = {}
# forced casting: mainly caused by the fact that Python or Numeric
#                 C/APIs do not support the corresponding C types.
c2py_map = {'double': 'float',
            'float': 'float',                          # forced casting
            'long_double': 'float',                    # forced casting
            'char': 'int',                             # forced casting
            'signed_char': 'int',                      # forced casting
            'unsigned_char': 'int',                    # forced casting
            'short': 'int',                            # forced casting
            'unsigned_short': 'int',                   # forced casting
            'int': 'int',                              # forced casting
            'long': 'int',
            'long_long': 'long',
            'unsigned': 'int',                         # forced casting
            'complex_float': 'complex',                # forced casting
            'complex_double': 'complex',
            'complex_long_double': 'complex',          # forced casting
            'string': 'string',
            'character': 'bytes',
            }

c2capi_map = {'double': 'NPY_DOUBLE',
                'float': 'NPY_FLOAT',
                'long_double': 'NPY_LONGDOUBLE',
                'char': 'NPY_BYTE',
                'unsigned_char': 'NPY_UBYTE',
                'signed_char': 'NPY_BYTE',
                'short': 'NPY_SHORT',
                'unsigned_short': 'NPY_USHORT',
                'int': 'NPY_INT',
                'unsigned': 'NPY_UINT',
                'long': 'NPY_LONG',
                'unsigned_long': 'NPY_ULONG',
                'long_long': 'NPY_LONGLONG',
                'unsigned_long_long': 'NPY_ULONGLONG',
                'complex_float': 'NPY_CFLOAT',
                'complex_double': 'NPY_CDOUBLE',
                'complex_long_double': 'NPY_CDOUBLE',
                'string': 'NPY_STRING',
                'character': 'NPY_STRING'}

c2pycode_map = {'double': 'd',
                'float': 'f',
                'long_double': 'g',
                'char': 'b',
                'unsigned_char': 'B',
                'signed_char': 'b',
                'short': 'h',
                'unsigned_short': 'H',
                'int': 'i',
                'unsigned': 'I',
                'long': 'l',
                'unsigned_long': 'L',
                'long_long': 'q',
                'unsigned_long_long': 'Q',
                'complex_float': 'F',
                'complex_double': 'D',
                'complex_long_double': 'G',
                'string': 'S',
                'character': 'c'}

# https://docs.python.org/3/c-api/arg.html#building-values
c2buildvalue_map = {'double': 'd',
                    'float': 'f',
                    'char': 'b',
                    'signed_char': 'b',
                    'short': 'h',
                    'int': 'i',
                    'long': 'l',
                    'long_long': 'L',
                    'complex_float': 'N',
                    'complex_double': 'N',
                    'complex_long_double': 'N',
                    'string': 'y',
                    'character': 'c'}

f2cmap_all = {'real': {'': 'float', '4': 'float', '8': 'double',
                       '12': 'long_double', '16': 'long_double'},
              'integer': {'': 'int', '1': 'signed_char', '2': 'short',
                          '4': 'int', '8': 'long_long',
                          '-1': 'unsigned_char', '-2': 'unsigned_short',
                          '-4': 'unsigned', '-8': 'unsigned_long_long'},
              'complex': {'': 'complex_float', '8': 'complex_float',
                          '16': 'complex_double', '24': 'complex_long_double',
                          '32': 'complex_long_double'},
              'complexkind': {'': 'complex_float', '4': 'complex_float',
                              '8': 'complex_double', '12': 'complex_long_double',
                              '16': 'complex_long_double'},
              'logical': {'': 'int', '1': 'char', '2': 'short', '4': 'int',
                          '8': 'long_long'},
              'double complex': {'': 'complex_double'},
              'double precision': {'': 'double'},
              'byte': {'': 'char'},
              }

# Add ISO_C handling
c2pycode_map.update(isoc_c2pycode_map)
c2py_map.update(iso_c2py_map)
f2cmap_all, _ = process_f2cmap_dict(f2cmap_all, iso_c_binding_map, c2py_map)
# End ISO_C handling
f2cmap_default = copy.deepcopy(f2cmap_all)

f2cmap_mapped = []

def load_f2cmap_file(f2cmap_file):
    global f2cmap_all, f2cmap_mapped

    f2cmap_all = copy.deepcopy(f2cmap_default)

    if f2cmap_file is None:
        # Default value
        f2cmap_file = '.f2py_f2cmap'
        if not os.path.isfile(f2cmap_file):
            return

    # User defined additions to f2cmap_all.
    # f2cmap_file must contain a dictionary of dictionaries, only. For
    # example, {'real':{'low':'float'}} means that Fortran 'real(low)' is
    # interpreted as C 'float'. This feature is useful for F90/95 users if
    # they use PARAMETERS in type specifications.
    try:
        outmess('Reading f2cmap from {!r} ...\n'.format(f2cmap_file))
        with open(f2cmap_file) as f:
            d = eval(f.read().lower(), {}, {})
        f2cmap_all, f2cmap_mapped = process_f2cmap_dict(f2cmap_all, d, c2py_map, True)
        outmess('Successfully applied user defined f2cmap changes\n')
    except Exception as msg:
        errmess('Failed to apply user defined f2cmap changes: %s. Skipping.\n' % (msg))


cformat_map = {'double': '%g',
               'float': '%g',
               'long_double': '%Lg',
               'char': '%d',
               'signed_char': '%d',
               'unsigned_char': '%hhu',
               'short': '%hd',
               'unsigned_short': '%hu',
               'int': '%d',
               'unsigned': '%u',
               'long': '%ld',
               'unsigned_long': '%lu',
               'long_long': '%ld',
               'complex_float': '(%g,%g)',
               'complex_double': '(%g,%g)',
               'complex_long_double': '(%Lg,%Lg)',
               'string': '\\"%s\\"',
               'character': "'%c'",
               }

# Auxiliary functions


def getctype(var):
    """
    Determines C type
    """
    ctype = 'void'
    if isfunction(var):
        if 'result' in var:
            a = var['result']
        else:
            a = var['name']
        if a in var['vars']:
            return getctype(var['vars'][a])
        else:
            errmess('getctype: function %s has no return value?!\n' % a)
    elif issubroutine(var):
        return ctype
    elif ischaracter_or_characterarray(var):
        return 'character'
    elif isstring_or_stringarray(var):
        return 'string'
    elif 'typespec' in var and var['typespec'].lower() in f2cmap_all:
        typespec = var['typespec'].lower()
        f2cmap = f2cmap_all[typespec]
        ctype = f2cmap['']  # default type
        if 'kindselector' in var:
            if '*' in var['kindselector']:
                try:
                    ctype = f2cmap[var['kindselector']['*']]
                except KeyError:
                    errmess('getctype: "%s %s %s" not supported.\n' %
                            (var['typespec'], '*', var['kindselector']['*']))
            elif 'kind' in var['kindselector']:
                if typespec + 'kind' in f2cmap_all:
                    f2cmap = f2cmap_all[typespec + 'kind']
                try:
                    ctype = f2cmap[var['kindselector']['kind']]
                except KeyError:
                    if typespec in f2cmap_all:
                        f2cmap = f2cmap_all[typespec]
                    try:
                        ctype = f2cmap[str(var['kindselector']['kind'])]
                    except KeyError:
                        errmess('getctype: "%s(kind=%s)" is mapped to C "%s" (to override define dict(%s = dict(%s="<C typespec>")) in %s/.f2py_f2cmap file).\n'
                                % (typespec, var['kindselector']['kind'], ctype,
                                   typespec, var['kindselector']['kind'], os.getcwd()))
    else:
        if not isexternal(var):
            errmess('getctype: No C-type found in "%s", assuming void.\n' % var)
    return ctype


def f2cexpr(expr):
    """Rewrite Fortran expression as f2py supported C expression.

    Due to the lack of a proper expression parser in f2py, this
    function uses a heuristic approach that assumes that Fortran
    arithmetic expressions are valid C arithmetic expressions when
    mapping Fortran function calls to the corresponding C function/CPP
    macros calls.

    """
    # TODO: support Fortran `len` function with optional kind parameter
    expr = re.sub(r'\blen\b', 'f2py_slen', expr)
    return expr


def getstrlength(var):
    if isstringfunction(var):
        if 'result' in var:
            a = var['result']
        else:
            a = var['name']
        if a in var['vars']:
            return getstrlength(var['vars'][a])
        else:
            errmess('getstrlength: function %s has no return value?!\n' % a)
    if not isstring(var):
        errmess(
            'getstrlength: expected a signature of a string but got: %s\n' % (repr(var)))
    len = '1'
    if 'charselector' in var:
        a = var['charselector']
        if '*' in a:
            len = a['*']
        elif 'len' in a:
            len = f2cexpr(a['len'])
    if re.match(r'\(\s*(\*|:)\s*\)', len) or re.match(r'(\*|:)', len):
        if isintent_hide(var):
            errmess('getstrlength:intent(hide): expected a string with defined length but got: %s\n' % (
                repr(var)))
        len = '-1'
    return len


def getarrdims(a, var, verbose=0):
    ret = {}
    if isstring(var) and not isarray(var):
        ret['size'] = getstrlength(var)
        ret['rank'] = '0'
        ret['dims'] = ''
    elif isscalar(var):
        ret['size'] = '1'
        ret['rank'] = '0'
        ret['dims'] = ''
    elif isarray(var):
        dim = copy.copy(var['dimension'])
        ret['size'] = '*'.join(dim)
        try:
            ret['size'] = repr(eval(ret['size']))
        except Exception:
            pass
        ret['dims'] = ','.join(dim)
        ret['rank'] = repr(len(dim))
        ret['rank*[-1]'] = repr(len(dim) * [-1])[1:-1]
        for i in range(len(dim)):  # solve dim for dependencies
            v = []
            if dim[i] in depargs:
                v = [dim[i]]
            else:
                for va in depargs:
                    if re.match(r'.*?\b%s\b.*' % va, dim[i]):
                        v.append(va)
            for va in v:
                if depargs.index(va) > depargs.index(a):
                    dim[i] = '*'
                    break
        ret['setdims'], i = '', -1
        for d in dim:
            i = i + 1
            if d not in ['*', ':', '(*)', '(:)']:
                ret['setdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['setdims'], i, d)
        if ret['setdims']:
            ret['setdims'] = ret['setdims'][:-1]
        ret['cbsetdims'], i = '', -1
        for d in var['dimension']:
            i = i + 1
            if d not in ['*', ':', '(*)', '(:)']:
                ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['cbsetdims'], i, d)
            elif isintent_in(var):
                outmess('getarrdims:warning: assumed shape array, using 0 instead of %r\n'
                        % (d))
                ret['cbsetdims'] = '%s#varname#_Dims[%d]=%s,' % (
                    ret['cbsetdims'], i, 0)
            elif verbose:
                errmess(
                    'getarrdims: If in call-back function: array argument %s must have bounded dimensions: got %s\n' % (repr(a), repr(d)))
        if ret['cbsetdims']:
            ret['cbsetdims'] = ret['cbsetdims'][:-1]
#         if not isintent_c(var):
#             var['dimension'].reverse()
    return ret


def getpydocsign(a, var):
    global lcb_map
    if isfunction(var):
        if 'result' in var:
            af = var['result']
        else:
            af = var['name']
        if af in var['vars']:
            return getpydocsign(af, var['vars'][af])
        else:
            errmess('getctype: function %s has no return value?!\n' % af)
        return '', ''
    sig, sigout = a, a
    opt = ''
    if isintent_in(var):
        opt = 'input'
    elif isintent_inout(var):
        opt = 'in/output'
    out_a = a
    if isintent_out(var):
        for k in var['intent']:
            if k[:4] == 'out=':
                out_a = k[4:]
                break
    init = ''
    ctype = getctype(var)

    if hasinitvalue(var):
        init, showinit = getinit(a, var)
        init = ', optional\\n    Default: %s' % showinit
    if isscalar(var):
        if isintent_inout(var):
            sig = '%s : %s rank-0 array(%s,\'%s\')%s' % (a, opt, c2py_map[ctype],
                                                         c2pycode_map[ctype], init)
        else:
            sig = '%s : %s %s%s' % (a, opt, c2py_map[ctype], init)
        sigout = '%s : %s' % (out_a, c2py_map[ctype])
    elif isstring(var):
        if isintent_inout(var):
            sig = '%s : %s rank-0 array(string(len=%s),\'c\')%s' % (
                a, opt, getstrlength(var), init)
        else:
            sig = '%s : %s string(len=%s)%s' % (
                a, opt, getstrlength(var), init)
        sigout = '%s : string(len=%s)' % (out_a, getstrlength(var))
    elif isarray(var):
        dim = var['dimension']
        rank = repr(len(dim))
        sig = '%s : %s rank-%s array(\'%s\') with bounds (%s)%s' % (a, opt, rank,
                                                                    c2pycode_map[
                                                                        ctype],
                                                                    ','.join(dim), init)
        if a == out_a:
            sigout = '%s : rank-%s array(\'%s\') with bounds (%s)'\
                % (a, rank, c2pycode_map[ctype], ','.join(dim))
        else:
            sigout = '%s : rank-%s array(\'%s\') with bounds (%s) and %s storage'\
                % (out_a, rank, c2pycode_map[ctype], ','.join(dim), a)
    elif isexternal(var):
        ua = ''
        if a in lcb_map and lcb_map[a] in lcb2_map and 'argname' in lcb2_map[lcb_map[a]]:
            ua = lcb2_map[lcb_map[a]]['argname']
            if not ua == a:
                ua = ' => %s' % ua
            else:
                ua = ''
        sig = '%s : call-back function%s' % (a, ua)
        sigout = sig
    else:
        errmess(
            'getpydocsign: Could not resolve docsignature for "%s".\n' % a)
    return sig, sigout


def getarrdocsign(a, var):
    ctype = getctype(var)
    if isstring(var) and (not isarray(var)):
        sig = '%s : rank-0 array(string(len=%s),\'c\')' % (a,
                                                           getstrlength(var))
    elif isscalar(var):
        sig = '%s : rank-0 array(%s,\'%s\')' % (a, c2py_map[ctype],
                                                c2pycode_map[ctype],)
    elif isarray(var):
        dim = var['dimension']
        rank = repr(len(dim))
        sig = '%s : rank-%s array(\'%s\') with bounds (%s)' % (a, rank,
                                                               c2pycode_map[
                                                                   ctype],
                                                               ','.join(dim))
    return sig


def getinit(a, var):
    if isstring(var):
        init, showinit = '""', "''"
    else:
        init, showinit = '', ''
    if hasinitvalue(var):
        init = var['=']
        showinit = init
        if iscomplex(var) or iscomplexarray(var):
            ret = {}

            try:
                v = var["="]
                if ',' in v:
                    ret['init.r'], ret['init.i'] = markoutercomma(
                        v[1:-1]).split('@,@')
                else:
                    v = eval(v, {}, {})
                    ret['init.r'], ret['init.i'] = str(v.real), str(v.imag)
            except Exception:
                raise ValueError(
                    'getinit: expected complex number `(r,i)\' but got `%s\' as initial value of %r.' % (init, a))
            if isarray(var):
                init = '(capi_c.r=%s,capi_c.i=%s,capi_c)' % (
                    ret['init.r'], ret['init.i'])
        elif isstring(var):
            if not init:
                init, showinit = '""', "''"
            if init[0] == "'":
                init = '"%s"' % (init[1:-1].replace('"', '\\"'))
            if init[0] == '"':
                showinit = "'%s'" % (init[1:-1])
    return init, showinit


def get_elsize(var):
    if isstring(var) or isstringarray(var):
        elsize = getstrlength(var)
        # override with user-specified length when available:
        elsize = var['charselector'].get('f2py_len', elsize)
        return elsize
    if ischaracter(var) or ischaracterarray(var):
        return '1'
    # for numerical types, PyArray_New* functions ignore specified
    # elsize, so we just return 1 and let elsize be determined at
    # runtime, see fortranobject.c
    return '1'


def sign2map(a, var):
    """
    varname,ctype,atype
    init,init.r,init.i,pytype
    vardebuginfo,vardebugshowvalue,varshowvalue
    varrformat

    intent
    """
    out_a = a
    if isintent_out(var):
        for k in var['intent']:
            if k[:4] == 'out=':
                out_a = k[4:]
                break
    ret = {'varname': a, 'outvarname': out_a, 'ctype': getctype(var)}
    intent_flags = []
    for f, s in isintent_dict.items():
        if f(var):
            intent_flags.append('F2PY_%s' % s)
    if intent_flags:
        # TODO: Evaluate intent_flags here.
        ret['intent'] = '|'.join(intent_flags)
    else:
        ret['intent'] = 'F2PY_INTENT_IN'
    if isarray(var):
        ret['varrformat'] = 'N'
    elif ret['ctype'] in c2buildvalue_map:
        ret['varrformat'] = c2buildvalue_map[ret['ctype']]
    else:
        ret['varrformat'] = 'O'
    ret['init'], ret['showinit'] = getinit(a, var)
    if hasinitvalue(var) and iscomplex(var) and not isarray(var):
        ret['init.r'], ret['init.i'] = markoutercomma(
            ret['init'][1:-1]).split('@,@')
    if isexternal(var):
        ret['cbnamekey'] = a
        if a in lcb_map:
            ret['cbname'] = lcb_map[a]
            ret['maxnofargs'] = lcb2_map[lcb_map[a]]['maxnofargs']
            ret['nofoptargs'] = lcb2_map[lcb_map[a]]['nofoptargs']
            ret['cbdocstr'] = lcb2_map[lcb_map[a]]['docstr']
            ret['cblatexdocstr'] = lcb2_map[lcb_map[a]]['latexdocstr']
        else:
            ret['cbname'] = a
            errmess('sign2map: Confused: external %s is not in lcb_map%s.\n' % (
                a, list(lcb_map.keys())))
    if isstring(var):
        ret['length'] = getstrlength(var)
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
        dim = copy.copy(var['dimension'])
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    # Debug info
    if debugcapi(var):
        il = [isintent_in, 'input', isintent_out, 'output',
              isintent_inout, 'inoutput', isrequired, 'required',
              isoptional, 'optional', isintent_hide, 'hidden',
              iscomplex, 'complex scalar',
              l_and(isscalar, l_not(iscomplex)), 'scalar',
              isstring, 'string', isarray, 'array',
              iscomplexarray, 'complex array', isstringarray, 'string array',
              iscomplexfunction, 'complex function',
              l_and(isfunction, l_not(iscomplexfunction)), 'function',
              isexternal, 'callback',
              isintent_callback, 'callback',
              isintent_aux, 'auxiliary',
              ]
        rl = []
        for i in range(0, len(il), 2):
            if il[i](var):
                rl.append(il[i + 1])
        if isstring(var):
            rl.append('slen(%s)=%s' % (a, ret['length']))
        if isarray(var):
            ddim = ','.join(
                map(lambda x, y: '%s|%s' % (x, y), var['dimension'], dim))
            rl.append('dims(%s)' % ddim)
        if isexternal(var):
            ret['vardebuginfo'] = 'debug-capi:%s=>%s:%s' % (
                a, ret['cbname'], ','.join(rl))
        else:
            ret['vardebuginfo'] = 'debug-capi:%s %s=%s:%s' % (
                ret['ctype'], a, ret['showinit'], ','.join(rl))
        if isscalar(var):
            if ret['ctype'] in cformat_map:
                ret['vardebugshowvalue'] = 'debug-capi:%s=%s' % (
                    a, cformat_map[ret['ctype']])
        if isstring(var):
            ret['vardebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (
                a, a)
        if isexternal(var):
            ret['vardebugshowvalue'] = 'debug-capi:%s=%%p' % (a)
    if ret['ctype'] in cformat_map:
        ret['varshowvalue'] = '#name#:%s=%s' % (a, cformat_map[ret['ctype']])
        ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
    if isstring(var):
        ret['varshowvalue'] = '#name#:slen(%s)=%%d %s=\\"%%s\\"' % (a, a)
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    if hasnote(var):
        ret['note'] = var['note']
    return ret


def routsign2map(rout):
    """
    name,NAME,begintitle,endtitle
    rname,ctype,rformat
    routdebugshowvalue
    """
    global lcb_map
    name = rout['name']
    fname = getfortranname(rout)
    ret = {'name': name,
           'texname': name.replace('_', '\\_'),
           'name_lower': name.lower(),
           'NAME': name.upper(),
           'begintitle': gentitle(name),
           'endtitle': gentitle('end of %s' % name),
           'fortranname': fname,
           'FORTRANNAME': fname.upper(),
           'callstatement': getcallstatement(rout) or '',
           'usercode': getusercode(rout) or '',
           'usercode1': getusercode1(rout) or '',
           }
    if '_' in fname:
        ret['F_FUNC'] = 'F_FUNC_US'
    else:
        ret['F_FUNC'] = 'F_FUNC'
    if '_' in name:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC_US'
    else:
        ret['F_WRAPPEDFUNC'] = 'F_WRAPPEDFUNC'
    lcb_map = {}
    if 'use' in rout:
        for u in rout['use'].keys():
            if u in cb_rules.cb_map:
                for un in cb_rules.cb_map[u]:
                    ln = un[0]
                    if 'map' in rout['use'][u]:
                        for k in rout['use'][u]['map'].keys():
                            if rout['use'][u]['map'][k] == un[0]:
                                ln = k
                                break
                    lcb_map[ln] = un[1]
    elif rout.get('externals'):
        errmess('routsign2map: Confused: function %s has externals %s but no "use" statement.\n' % (
            ret['name'], repr(rout['externals'])))
    ret['callprotoargument'] = getcallprotoargument(rout, lcb_map) or ''
    if isfunction(rout):
        if 'result' in rout:
            a = rout['result']
        else:
            a = rout['name']
        ret['rname'] = a
        ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
        ret['ctype'] = getctype(rout['vars'][a])
        if hasresultnote(rout):
            ret['resultnote'] = rout['vars'][a]['note']
            rout['vars'][a]['note'] = ['See elsewhere.']
        if ret['ctype'] in c2buildvalue_map:
            ret['rformat'] = c2buildvalue_map[ret['ctype']]
        else:
            ret['rformat'] = 'O'
            errmess('routsign2map: no c2buildvalue key for type %s\n' %
                    (repr(ret['ctype'])))
        if debugcapi(rout):
            if ret['ctype'] in cformat_map:
                ret['routdebugshowvalue'] = 'debug-capi:%s=%s' % (
                    a, cformat_map[ret['ctype']])
            if isstringfunction(rout):
                ret['routdebugshowvalue'] = 'debug-capi:slen(%s)=%%d %s=\\"%%s\\"' % (
                    a, a)
        if isstringfunction(rout):
            ret['rlength'] = getstrlength(rout['vars'][a])
            if ret['rlength'] == '-1':
                errmess('routsign2map: expected explicit specification of the length of the string returned by the fortran function %s; taking 10.\n' % (
                    repr(rout['name'])))
                ret['rlength'] = '10'
    if hasnote(rout):
        ret['note'] = rout['note']
        rout['note'] = ['See elsewhere.']
    return ret


def modsign2map(m):
    """
    modulename
    """
    if ismodule(m):
        ret = {'f90modulename': m['name'],
               'F90MODULENAME': m['name'].upper(),
               'texf90modulename': m['name'].replace('_', '\\_')}
    else:
        ret = {'modulename': m['name'],
               'MODULENAME': m['name'].upper(),
               'texmodulename': m['name'].replace('_', '\\_')}
    ret['restdoc'] = getrestdoc(m) or []
    if hasnote(m):
        ret['note'] = m['note']
    ret['usercode'] = getusercode(m) or ''
    ret['usercode1'] = getusercode1(m) or ''
    if m['body']:
        ret['interface_usercode'] = getusercode(m['body'][0]) or ''
    else:
        ret['interface_usercode'] = ''
    ret['pymethoddef'] = getpymethoddef(m) or ''
    if 'gil_used' in m:
        ret['gil_used'] = m['gil_used']
    if 'coutput' in m:
        ret['coutput'] = m['coutput']
    if 'f2py_wrapper_output' in m:
        ret['f2py_wrapper_output'] = m['f2py_wrapper_output']
    return ret


def cb_sign2map(a, var, index=None):
    ret = {'varname': a}
    ret['varname_i'] = ret['varname']
    ret['ctype'] = getctype(var)
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    if hasnote(var):
        ret['note'] = var['note']
        var['note'] = ['See elsewhere.']
    return ret


def cb_routsign2map(rout, um):
    """
    name,begintitle,endtitle,argname
    ctype,rctype,maxnofargs,nofoptargs,returncptr
    """
    ret = {'name': 'cb_%s_in_%s' % (rout['name'], um),
           'returncptr': ''}
    if isintent_callback(rout):
        if '_' in rout['name']:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        ret['callbackname'] = '%s(%s,%s)' \
                              % (F_FUNC,
                                 rout['name'].lower(),
                                 rout['name'].upper(),
                                 )
        ret['static'] = 'extern'
    else:
        ret['callbackname'] = ret['name']
        ret['static'] = 'static'
    ret['argname'] = rout['name']
    ret['begintitle'] = gentitle(ret['name'])
    ret['endtitle'] = gentitle('end of %s' % ret['name'])
    ret['ctype'] = getctype(rout)
    ret['rctype'] = 'void'
    if ret['ctype'] == 'string':
        ret['rctype'] = 'void'
    else:
        ret['rctype'] = ret['ctype']
    if ret['rctype'] != 'void':
        if iscomplexfunction(rout):
            ret['returncptr'] = """
#ifdef F2PY_CB_RETURNCOMPLEX
return_value=
#endif
"""
        else:
            ret['returncptr'] = 'return_value='
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
    if isstringfunction(rout):
        ret['strlength'] = getstrlength(rout)
    if isfunction(rout):
        if 'result' in rout:
            a = rout['result']
        else:
            a = rout['name']
        if hasnote(rout['vars'][a]):
            ret['note'] = rout['vars'][a]['note']
            rout['vars'][a]['note'] = ['See elsewhere.']
        ret['rname'] = a
        ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
        if iscomplexfunction(rout):
            ret['rctype'] = """
#ifdef F2PY_CB_RETURNCOMPLEX
#ctype#
#else
void
#endif
"""
    else:
        if hasnote(rout):
            ret['note'] = rout['note']
            rout['note'] = ['See elsewhere.']
    nofargs = 0
    nofoptargs = 0
    if 'args' in rout and 'vars' in rout:
        for a in rout['args']:
            var = rout['vars'][a]
            if l_or(isintent_in, isintent_inout)(var):
                nofargs = nofargs + 1
                if isoptional(var):
                    nofoptargs = nofoptargs + 1
    ret['maxnofargs'] = repr(nofargs)
    ret['nofoptargs'] = repr(nofoptargs)
    if hasnote(rout) and isfunction(rout) and 'result' in rout:
        ret['routnote'] = rout['note']
        rout['note'] = ['See elsewhere.']
    return ret


def common_sign2map(a, var):  # obsolete
    ret = {'varname': a, 'ctype': getctype(var)}
    if isstringarray(var):
        ret['ctype'] = 'char'
    if ret['ctype'] in c2capi_map:
        ret['atype'] = c2capi_map[ret['ctype']]
        ret['elsize'] = get_elsize(var)
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = '%s' % (cformat_map[ret['ctype']])
    if isarray(var):
        ret = dictappend(ret, getarrdims(a, var))
    elif isstring(var):
        ret['size'] = getstrlength(var)
        ret['rank'] = '1'
    ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, var)
    if hasnote(var):
        ret['note'] = var['note']
        var['note'] = ['See elsewhere.']
    # for strings this returns 0-rank but actually is 1-rank
    ret['arrdocstr'] = getarrdocsign(a, var)
    return ret
<|endoftext|>#!/usr/bin/env python3
import os
import sys
import tempfile


def run_command(cmd):
    print('Running %r:' % (cmd))
    os.system(cmd)
    print('------')


def run():
    _path = os.getcwd()
    os.chdir(tempfile.gettempdir())
    print('------')
    print('os.name=%r' % (os.name))
    print('------')
    print('sys.platform=%r' % (sys.platform))
    print('------')
    print('sys.version:')
    print(sys.version)
    print('------')
    print('sys.prefix:')
    print(sys.prefix)
    print('------')
    print('sys.path=%r' % (':'.join(sys.path)))
    print('------')

    try:
        import numpy
        has_newnumpy = 1
    except ImportError as e:
        print('Failed to import new numpy:', e)
        has_newnumpy = 0

    try:
        from numpy.f2py import f2py2e
        has_f2py2e = 1
    except ImportError as e:
        print('Failed to import f2py2e:', e)
        has_f2py2e = 0

    try:
        import numpy.distutils
        has_numpy_distutils = 2
    except ImportError:
        try:
            import numpy_distutils
            has_numpy_distutils = 1
        except ImportError as e:
            print('Failed to import numpy_distutils:', e)
            has_numpy_distutils = 0

    if has_newnumpy:
        try:
            print('Found new numpy version %r in %s' %
                  (numpy.__version__, numpy.__file__))
        except Exception as msg:
            print('error:', msg)
            print('------')

    if has_f2py2e:
        try:
            print('Found f2py2e version %r in %s' %
                  (f2py2e.__version__.version, f2py2e.__file__))
        except Exception as msg:
            print('error:', msg)
            print('------')

    if has_numpy_distutils:
        try:
            if has_numpy_distutils == 2:
                print('Found numpy.distutils version %r in %r' % (
                    numpy.distutils.__version__,
                    numpy.distutils.__file__))
            else:
                print('Found numpy_distutils version %r in %r' % (
                    numpy_distutils.numpy_distutils_version.numpy_distutils_version,
                    numpy_distutils.__file__))
            print('------')
        except Exception as msg:
            print('error:', msg)
            print('------')
        try:
            if has_numpy_distutils == 1:
                print(
                    'Importing numpy_distutils.command.build_flib ...', end=' ')
                import numpy_distutils.command.build_flib as build_flib
                print('ok')
                print('------')
                try:
                    print(
                        'Checking availability of supported Fortran compilers:')
                    for compiler_class in build_flib.all_compilers:
                        compiler_class(verbose=1).is_available()
                        print('------')
                except Exception as msg:
                    print('error:', msg)
                    print('------')
        except Exception as msg:
            print(
                'error:', msg, '(ignore it, build_flib is obsolete for numpy.distutils 0.2.2 and up)')
            print('------')
        try:
            if has_numpy_distutils == 2:
                print('Importing numpy.distutils.fcompiler ...', end=' ')
                import numpy.distutils.fcompiler as fcompiler
            else:
                print('Importing numpy_distutils.fcompiler ...', end=' ')
                import numpy_distutils.fcompiler as fcompiler
            print('ok')
            print('------')
            try:
                print('Checking availability of supported Fortran compilers:')
                fcompiler.show_fcompilers()
                print('------')
            except Exception as msg:
                print('error:', msg)
                print('------')
        except Exception as msg:
            print('error:', msg)
            print('------')
        try:
            if has_numpy_distutils == 2:
                print('Importing numpy.distutils.cpuinfo ...', end=' ')
                from numpy.distutils.cpuinfo import cpuinfo
                print('ok')
                print('------')
            else:
                try:
                    print(
                        'Importing numpy_distutils.command.cpuinfo ...', end=' ')
                    from numpy_distutils.command.cpuinfo import cpuinfo
                    print('ok')
                    print('------')
                except Exception as msg:
                    print('error:', msg, '(ignore it)')
                    print('Importing numpy_distutils.cpuinfo ...', end=' ')
                    from numpy_distutils.cpuinfo import cpuinfo
                    print('ok')
                    print('------')
            cpu = cpuinfo()
            print('CPU information:', end=' ')
            for name in dir(cpuinfo):
                if name[0] == '_' and name[1] != '_' and getattr(cpu, name[1:])():
                    print(name[1:], end=' ')
            print('------')
        except Exception as msg:
            print('error:', msg)
            print('------')
    os.chdir(_path)
if __name__ == "__main__":
    run()
<|endoftext|>"""
Build F90 module support for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__version__ = "$Revision: 1.27 $"[10:-1]

f2py_version = 'See `f2py -v`'

import numpy as np

from . import capi_maps
from . import func2subr
from .crackfortran import undo_rmbadname, undo_rmbadname1

# The environment provided by auxfuncs.py is needed for some calls to eval.
# As the needed functions cannot be determined by static inspection of the
# code, it is safest to use import * pending a major refactoring of f2py.
from .auxfuncs import *

options = {}


def findf90modules(m):
    if ismodule(m):
        return [m]
    if not hasbody(m):
        return []
    ret = []
    for b in m['body']:
        if ismodule(b):
            ret.append(b)
        else:
            ret = ret + findf90modules(b)
    return ret

fgetdims1 = """\
      external f2pysetdata
      logical ns
      integer r,i
      integer(%d) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then""" % np.intp().itemsize

fgetdims2 = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))"""

fgetdims2_sa = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
         !s(r) must be equal to len(d(1))
      end if
      flag = 2
      call f2pysetdata(d,allocated(d))"""


def buildhooks(pymod):
    from . import rules
    ret = {'f90modhooks': [], 'initf90modhooks': [], 'body': [],
           'need': ['F_FUNC', 'arrayobject.h'],
           'separatorsfor': {'includes0': '\n', 'includes': '\n'},
           'docs': ['"Fortran 90/95 modules:\\n"'],
           'latexdoc': []}
    fhooks = ['']

    def fadd(line, s=fhooks):
        s[0] = '%s\n      %s' % (s[0], line)
    doc = ['']

    def dadd(line, s=doc):
        s[0] = '%s\n%s' % (s[0], line)

    usenames = getuseblocks(pymod)
    for m in findf90modules(pymod):
        sargs, fargs, efargs, modobjs, notvars, onlyvars = [], [], [], [], [
            m['name']], []
        sargsp = []
        ifargs = []
        mfargs = []
        if hasbody(m):
            for b in m['body']:
                notvars.append(b['name'])
        for n in m['vars'].keys():
            var = m['vars'][n]

            if (n not in notvars and isvariable(var)) and (not l_or(isintent_hide, isprivate)(var)):
                onlyvars.append(n)
                mfargs.append(n)
        outmess('\t\tConstructing F90 module support for "%s"...\n' %
                (m['name']))
        if len(onlyvars) == 0 and len(notvars) == 1 and m['name'] in notvars:
            outmess(f"\t\t\tSkipping {m['name']} since there are no public vars/func in this module...\n")
            continue

        # gh-25186
        if m['name'] in usenames and containscommon(m):
            outmess(f"\t\t\tSkipping {m['name']} since it is in 'use' and contains a common block...\n")
            continue
        if onlyvars:
            outmess('\t\t  Variables: %s\n' % (' '.join(onlyvars)))
        chooks = ['']

        def cadd(line, s=chooks):
            s[0] = '%s\n%s' % (s[0], line)
        ihooks = ['']

        def iadd(line, s=ihooks):
            s[0] = '%s\n%s' % (s[0], line)

        vrd = capi_maps.modsign2map(m)
        cadd('static FortranDataDef f2py_%s_def[] = {' % (m['name']))
        dadd('\\subsection{Fortran 90/95 module \\texttt{%s}}\n' % (m['name']))
        if hasnote(m):
            note = m['note']
            if isinstance(note, list):
                note = '\n'.join(note)
            dadd(note)
        if onlyvars:
            dadd('\\begin{description}')
        for n in onlyvars:
            var = m['vars'][n]
            modobjs.append(n)
            ct = capi_maps.getctype(var)
            at = capi_maps.c2capi_map[ct]
            dm = capi_maps.getarrdims(n, var)
            dms = dm['dims'].replace('*', '-1').strip()
            dms = dms.replace(':', '-1').strip()
            if not dms:
                dms = '-1'
            use_fgetdims2 = fgetdims2
            cadd('\t{"%s",%s,{{%s}},%s, %s},' %
                 (undo_rmbadname1(n), dm['rank'], dms, at,
                  capi_maps.get_elsize(var)))
            dadd('\\item[]{{}\\verb@%s@{}}' %
                 (capi_maps.getarrdocsign(n, var)))
            if hasnote(var):
                note = var['note']
                if isinstance(note, list):
                    note = '\n'.join(note)
                dadd('--- %s' % (note))
            if isallocatable(var):
                fargs.append('f2py_%s_getdims_%s' % (m['name'], n))
                efargs.append(fargs[-1])
                sargs.append(
                    'void (*%s)(int*,npy_intp*,void(*)(char*,npy_intp*),int*)' % (n))
                sargsp.append('void (*)(int*,npy_intp*,void(*)(char*,npy_intp*),int*)')
                iadd('\tf2py_%s_def[i_f2py++].func = %s;' % (m['name'], n))
                fadd('subroutine %s(r,s,f2pysetdata,flag)' % (fargs[-1]))
                fadd('use %s, only: d => %s\n' %
                     (m['name'], undo_rmbadname1(n)))
                fadd('integer flag\n')
                fhooks[0] = fhooks[0] + fgetdims1
                dms = range(1, int(dm['rank']) + 1)
                fadd(' allocate(d(%s))\n' %
                     (','.join(['s(%s)' % i for i in dms])))
                fhooks[0] = fhooks[0] + use_fgetdims2
                fadd('end subroutine %s' % (fargs[-1]))
            else:
                fargs.append(n)
                sargs.append('char *%s' % (n))
                sargsp.append('char*')
                iadd('\tf2py_%s_def[i_f2py++].data = %s;' % (m['name'], n))
        if onlyvars:
            dadd('\\end{description}')
        if hasbody(m):
            for b in m['body']:
                if not isroutine(b):
                    outmess("f90mod_rules.buildhooks:"
                            f" skipping {b['block']} {b['name']}\n")
                    continue
                modobjs.append('%s()' % (b['name']))
                b['modulename'] = m['name']
                api, wrap = rules.buildapi(b)
                if isfunction(b):
                    fhooks[0] = fhooks[0] + wrap
                    fargs.append('f2pywrap_%s_%s' % (m['name'], b['name']))
                    ifargs.append(func2subr.createfuncwrapper(b, signature=1))
                else:
                    if wrap:
                        fhooks[0] = fhooks[0] + wrap
                        fargs.append('f2pywrap_%s_%s' % (m['name'], b['name']))
                        ifargs.append(
                            func2subr.createsubrwrapper(b, signature=1))
                    else:
                        fargs.append(b['name'])
                        mfargs.append(fargs[-1])
                api['externroutines'] = []
                ar = applyrules(api, vrd)
                ar['docs'] = []
                ar['docshort'] = []
                ret = dictappend(ret, ar)
                cadd(('\t{"%s",-1,{{-1}},0,0,NULL,(void *)'
                      'f2py_rout_#modulename#_%s_%s,'
                      'doc_f2py_rout_#modulename#_%s_%s},')
                     % (b['name'], m['name'], b['name'], m['name'], b['name']))
                sargs.append('char *%s' % (b['name']))
                sargsp.append('char *')
                iadd('\tf2py_%s_def[i_f2py++].data = %s;' %
                     (m['name'], b['name']))
        cadd('\t{NULL}\n};\n')
        iadd('}')
        ihooks[0] = 'static void f2py_setup_%s(%s) {\n\tint i_f2py=0;%s' % (
            m['name'], ','.join(sargs), ihooks[0])
        if '_' in m['name']:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        iadd('extern void %s(f2pyinit%s,F2PYINIT%s)(void (*)(%s));'
             % (F_FUNC, m['name'], m['name'].upper(), ','.join(sargsp)))
        iadd('static void f2py_init_%s(void) {' % (m['name']))
        iadd('\t%s(f2pyinit%s,F2PYINIT%s)(f2py_setup_%s);'
             % (F_FUNC, m['name'], m['name'].upper(), m['name']))
        iadd('}\n')
        ret['f90modhooks'] = ret['f90modhooks'] + chooks + ihooks
        ret['initf90modhooks'] = ['\tPyDict_SetItemString(d, "%s", PyFortranObject_New(f2py_%s_def,f2py_init_%s));' % (
            m['name'], m['name'], m['name'])] + ret['initf90modhooks']
        fadd('')
        fadd('subroutine f2pyinit%s(f2pysetupfunc)' % (m['name']))
        if mfargs:
            for a in undo_rmbadname(mfargs):
                fadd('use %s, only : %s' % (m['name'], a))
        if ifargs:
            fadd(' '.join(['interface'] + ifargs))
            fadd('end interface')
        fadd('external f2pysetupfunc')
        if efargs:
            for a in undo_rmbadname(efargs):
                fadd('external %s' % (a))
        fadd('call f2pysetupfunc(%s)' % (','.join(undo_rmbadname(fargs))))
        fadd('end subroutine f2pyinit%s\n' % (m['name']))

        dadd('\n'.join(ret['latexdoc']).replace(
            r'\subsection{', r'\subsubsection{'))

        ret['latexdoc'] = []
        ret['docs'].append('"\t%s --- %s"' % (m['name'],
                                              ','.join(undo_rmbadname(modobjs))))

    ret['routine_defs'] = ''
    ret['doc'] = []
    ret['docshort'] = []
    ret['latexdoc'] = doc[0]
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    return ret, fhooks[0]
<|endoftext|>from ._backend import Backend

from numpy.distutils.core import setup, Extension
from numpy.distutils.system_info import get_info
from numpy.distutils.misc_util import dict_append
from numpy.exceptions import VisibleDeprecationWarning
import os
import sys
import shutil
import warnings


class DistutilsBackend(Backend):
    def __init__(sef, *args, **kwargs):
        warnings.warn(
            "\ndistutils has been deprecated since NumPy 1.26.x\n"
            "Use the Meson backend instead, or generate wrappers"
            " without -c and use a custom build script",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def compile(self):
        num_info = {}
        if num_info:
            self.include_dirs.extend(num_info.get("include_dirs", []))
        ext_args = {
            "name": self.modulename,
            "sources": self.sources,
            "include_dirs": self.include_dirs,
            "library_dirs": self.library_dirs,
            "libraries": self.libraries,
            "define_macros": self.define_macros,
            "undef_macros": self.undef_macros,
            "extra_objects": self.extra_objects,
            "f2py_options": self.f2py_flags,
        }

        if self.sysinfo_flags:
            for n in self.sysinfo_flags:
                i = get_info(n)
                if not i:
                    print(
                        f"No {n!r} resources found"
                        "in system (try `f2py --help-link`)"
                    )
                dict_append(ext_args, **i)

        ext = Extension(**ext_args)

        sys.argv = [sys.argv[0]] + self.setup_flags
        sys.argv.extend(
            [
                "build",
                "--build-temp",
                self.build_dir,
                "--build-base",
                self.build_dir,
                "--build-platlib",
                ".",
                "--disable-optimization",
            ]
        )

        if self.fc_flags:
            sys.argv.extend(["config_fc"] + self.fc_flags)
        if self.flib_flags:
            sys.argv.extend(["build_ext"] + self.flib_flags)

        setup(ext_modules=[ext])

        if self.remove_build_dir and os.path.exists(self.build_dir):
            print(f"Removing build directory {self.build_dir}")
            shutil.rmtree(self.build_dir)
<|endoftext|>def f2py_build_generator(name):
    if name == "meson":
        from ._meson import MesonBackend
        return MesonBackend
    elif name == "distutils":
        from ._distutils import DistutilsBackend
        return DistutilsBackend
    else:
        raise ValueError(f"Unknown backend: {name}")
<|endoftext|>from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    def __init__(
        self,
        modulename,
        sources,
        extra_objects,
        build_dir,
        include_dirs,
        library_dirs,
        libraries,
        define_macros,
        undef_macros,
        f2py_flags,
        sysinfo_flags,
        fc_flags,
        flib_flags,
        setup_flags,
        remove_build_dir,
        extra_dat,
    ):
        self.modulename = modulename
        self.sources = sources
        self.extra_objects = extra_objects
        self.build_dir = build_dir
        self.include_dirs = include_dirs
        self.library_dirs = library_dirs
        self.libraries = libraries
        self.define_macros = define_macros
        self.undef_macros = undef_macros
        self.f2py_flags = f2py_flags
        self.sysinfo_flags = sysinfo_flags
        self.fc_flags = fc_flags
        self.flib_flags = flib_flags
        self.setup_flags = setup_flags
        self.remove_build_dir = remove_build_dir
        self.extra_dat = extra_dat

    @abstractmethod
    def compile(self) -> None:
        """Compile the wrapper."""
        pass
<|endoftext|>from __future__ import annotations

import os
import errno
import shutil
import subprocess
import sys
import re
from pathlib import Path

from ._backend import Backend
from string import Template
from itertools import chain



class MesonTemplate:
    """Template meson build file generation class."""

    def __init__(
        self,
        modulename: str,
        sources: list[Path],
        deps: list[str],
        libraries: list[str],
        library_dirs: list[Path],
        include_dirs: list[Path],
        object_files: list[Path],
        linker_args: list[str],
        fortran_args: list[str],
        build_type: str,
        python_exe: str,
    ):
        self.modulename = modulename
        self.build_template_path = (
            Path(__file__).parent.absolute() / "meson.build.template"
        )
        self.sources = sources
        self.deps = deps
        self.libraries = libraries
        self.library_dirs = library_dirs
        if include_dirs is not None:
            self.include_dirs = include_dirs
        else:
            self.include_dirs = []
        self.substitutions = {}
        self.objects = object_files
        # Convert args to '' wrapped variant for meson
        self.fortran_args = [
            f"'{x}'" if not (x.startswith("'") and x.endswith("'")) else x
            for x in fortran_args
        ]
        self.pipeline = [
            self.initialize_template,
            self.sources_substitution,
            self.deps_substitution,
            self.include_substitution,
            self.libraries_substitution,
            self.fortran_args_substitution,
        ]
        self.build_type = build_type
        self.python_exe = python_exe
        self.indent = " " * 21

    def meson_build_template(self) -> str:
        if not self.build_template_path.is_file():
            raise FileNotFoundError(
                errno.ENOENT,
                "Meson build template"
                f" {self.build_template_path.absolute()}"
                " does not exist.",
            )
        return self.build_template_path.read_text()

    def initialize_template(self) -> None:
        self.substitutions["modulename"] = self.modulename
        self.substitutions["buildtype"] = self.build_type
        self.substitutions["python"] = self.python_exe

    def sources_substitution(self) -> None:
        self.substitutions["source_list"] = ",\n".join(
            [f"{self.indent}'''{source}'''," for source in self.sources]
        )

    def deps_substitution(self) -> None:
        self.substitutions["dep_list"] = f",\n{self.indent}".join(
            [f"{self.indent}dependency('{dep}')," for dep in self.deps]
        )

    def libraries_substitution(self) -> None:
        self.substitutions["lib_dir_declarations"] = "\n".join(
            [
                f"lib_dir_{i} = declare_dependency(link_args : ['''-L{lib_dir}'''])"
                for i, lib_dir in enumerate(self.library_dirs)
            ]
        )

        self.substitutions["lib_declarations"] = "\n".join(
            [
                f"{lib.replace('.','_')} = declare_dependency(link_args : ['-l{lib}'])"
                for lib in self.libraries
            ]
        )

        self.substitutions["lib_list"] = f"\n{self.indent}".join(
            [f"{self.indent}{lib.replace('.','_')}," for lib in self.libraries]
        )
        self.substitutions["lib_dir_list"] = f"\n{self.indent}".join(
            [f"{self.indent}lib_dir_{i}," for i in range(len(self.library_dirs))]
        )

    def include_substitution(self) -> None:
        self.substitutions["inc_list"] = f",\n{self.indent}".join(
            [f"{self.indent}'''{inc}'''," for inc in self.include_dirs]
        )

    def fortran_args_substitution(self) -> None:
        if self.fortran_args:
            self.substitutions["fortran_args"] = (
                f"{self.indent}fortran_args: [{', '.join(list(self.fortran_args))}],"
            )
        else:
            self.substitutions["fortran_args"] = ""

    def generate_meson_build(self):
        for node in self.pipeline:
            node()
        template = Template(self.meson_build_template())
        meson_build = template.substitute(self.substitutions)
        meson_build = re.sub(r",,", ",", meson_build)
        return meson_build


class MesonBackend(Backend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dependencies = self.extra_dat.get("dependencies", [])
        self.meson_build_dir = "bbdir"
        self.build_type = (
            "debug" if any("debug" in flag for flag in self.fc_flags) else "release"
        )
        self.fc_flags = _get_flags(self.fc_flags)

    def _move_exec_to_root(self, build_dir: Path):
        walk_dir = Path(build_dir) / self.meson_build_dir
        path_objects = chain(
            walk_dir.glob(f"{self.modulename}*.so"),
            walk_dir.glob(f"{self.modulename}*.pyd"),
        )
        # Same behavior as distutils
        # https://github.com/numpy/numpy/issues/24874#issuecomment-1835632293
        for path_object in path_objects:
            dest_path = Path.cwd() / path_object.name
            if dest_path.exists():
                dest_path.unlink()
            shutil.copy2(path_object, dest_path)
            os.remove(path_object)

    def write_meson_build(self, build_dir: Path) -> None:
        """Writes the meson build file at specified location"""
        meson_template = MesonTemplate(
            self.modulename,
            self.sources,
            self.dependencies,
            self.libraries,
            self.library_dirs,
            self.include_dirs,
            self.extra_objects,
            self.flib_flags,
            self.fc_flags,
            self.build_type,
            sys.executable,
        )
        src = meson_template.generate_meson_build()
        Path(build_dir).mkdir(parents=True, exist_ok=True)
        meson_build_file = Path(build_dir) / "meson.build"
        meson_build_file.write_text(src)
        return meson_build_file

    def _run_subprocess_command(self, command, cwd):
        subprocess.run(command, cwd=cwd, check=True)

    def run_meson(self, build_dir: Path):
        setup_command = ["meson", "setup", self.meson_build_dir]
        self._run_subprocess_command(setup_command, build_dir)
        compile_command = ["meson", "compile", "-C", self.meson_build_dir]
        self._run_subprocess_command(compile_command, build_dir)

    def compile(self) -> None:
        self.sources = _prepare_sources(self.modulename, self.sources, self.build_dir)
        self.write_meson_build(self.build_dir)
        self.run_meson(self.build_dir)
        self._move_exec_to_root(self.build_dir)


def _prepare_sources(mname, sources, bdir):
    extended_sources = sources.copy()
    Path(bdir).mkdir(parents=True, exist_ok=True)
    # Copy sources
    for source in sources:
        if Path(source).exists() and Path(source).is_file():
            shutil.copy(source, bdir)
    generated_sources = [
        Path(f"{mname}module.c"),
        Path(f"{mname}-f2pywrappers2.f90"),
        Path(f"{mname}-f2pywrappers.f"),
    ]
    bdir = Path(bdir)
    for generated_source in generated_sources:
        if generated_source.exists():
            shutil.copy(generated_source, bdir / generated_source.name)
            extended_sources.append(generated_source.name)
            generated_source.unlink()
    extended_sources = [
        Path(source).name
        for source in extended_sources
        if not Path(source).suffix == ".pyf"
    ]
    return extended_sources


def _get_flags(fc_flags):
    flag_values = []
    flag_pattern = re.compile(r"--f(77|90)flags=(.*)")
    for flag in fc_flags:
        match_result = flag_pattern.match(flag)
        if match_result:
            values = match_result.group(2).strip().split()
            values = [val.strip("'\"") for val in values]
            flag_values.extend(values)
    # Hacky way to preserve order of flags
    unique_flags = list(dict.fromkeys(flag_values))
    return unique_flags
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import _dtype
    from ._utils import _raise_warning
    ret = getattr(_dtype, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._dtype' has no attribute {attr_name}")
    _raise_warning(attr_name, "_dtype")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import einsumfunc
    from ._utils import _raise_warning
    ret = getattr(einsumfunc, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.einsumfunc' has no attribute {attr_name}")
    _raise_warning(attr_name, "einsumfunc")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import defchararray
    from ._utils import _raise_warning
    ret = getattr(defchararray, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.defchararray' has no attribute {attr_name}")
    _raise_warning(attr_name, "defchararray")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import numeric
    from ._utils import _raise_warning

    sentinel = object()
    ret = getattr(numeric, attr_name, sentinel)
    if ret is sentinel:
        raise AttributeError(
            f"module 'numpy.core.numeric' has no attribute {attr_name}")
    _raise_warning(attr_name, "numeric")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import numerictypes
    from ._utils import _raise_warning
    ret = getattr(numerictypes, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.numerictypes' has no attribute {attr_name}")
    _raise_warning(attr_name, "numerictypes")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import overrides
    from ._utils import _raise_warning
    ret = getattr(overrides, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.overrides' has no attribute {attr_name}")
    _raise_warning(attr_name, "overrides")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import records
    from ._utils import _raise_warning
    ret = getattr(records, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.records' has no attribute {attr_name}")
    _raise_warning(attr_name, "records")
    return ret
<|endoftext|>from numpy._core import _internal

# Build a new array from the information in a pickle.
# Note that the name numpy.core._internal._reconstruct is embedded in
# pickles of ndarrays made with NumPy before release 1.0
# so don't remove the name here, or you'll
# break backward compatibility.
def _reconstruct(subtype, shape, dtype):
    from numpy import ndarray
    return ndarray.__new__(subtype, shape, dtype)


# Pybind11 (in versions <= 2.11.1) imports _dtype_from_pep3118 from the
# _internal submodule, therefore it must be importable without a warning.
_dtype_from_pep3118 = _internal._dtype_from_pep3118

def __getattr__(attr_name):
    from numpy._core import _internal
    from ._utils import _raise_warning
    ret = getattr(_internal, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._internal' has no attribute {attr_name}")
    _raise_warning(attr_name, "_internal")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import shape_base
    from ._utils import _raise_warning
    ret = getattr(shape_base, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.shape_base' has no attribute {attr_name}")
    _raise_warning(attr_name, "shape_base")
    return ret
<|endoftext|>"""
The `numpy.core` submodule exists solely for backward compatibility
purposes. The original `core` was renamed to `_core` and made private.
`numpy.core` will be removed in the future.
"""
from numpy import _core
from ._utils import _raise_warning


# We used to use `np.core._ufunc_reconstruct` to unpickle.
# This is unnecessary, but old pickles saved before 1.20 will be using it,
# and there is no reason to break loading them.
def _ufunc_reconstruct(module, name):
    # The `fromlist` kwarg is required to ensure that `mod` points to the
    # inner-most module rather than the parent package when module name is
    # nested. This makes it possible to pickle non-toplevel ufuncs such as
    # scipy.special.expit for instance.
    mod = __import__(module, fromlist=[name])
    return getattr(mod, name)


# force lazy-loading of submodules to ensure a warning is printed

__all__ = ["arrayprint", "defchararray", "_dtype_ctypes", "_dtype",
           "einsumfunc", "fromnumeric", "function_base", "getlimits",
           "_internal", "multiarray", "_multiarray_umath", "numeric",
           "numerictypes", "overrides", "records", "shape_base", "umath"]

def __getattr__(attr_name):
    attr = getattr(_core, attr_name)
    _raise_warning(attr_name)
    return attr
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import getlimits
    from ._utils import _raise_warning
    ret = getattr(getlimits, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.getlimits' has no attribute {attr_name}")
    _raise_warning(attr_name, "getlimits")
    return ret
<|endoftext|>import warnings


def _raise_warning(attr: str, submodule: str | None = None) -> None:
    new_module = "numpy._core"
    old_module = "numpy.core"
    if submodule is not None:
        new_module = f"{new_module}.{submodule}"
        old_module = f"{old_module}.{submodule}"
    warnings.warn(
        f"{old_module} is deprecated and has been renamed to {new_module}. "
        "The numpy._core namespace contains private NumPy internals and its "
        "use is discouraged, as NumPy internals can change without warning in "
        "any release. In practice, most real-world usage of numpy.core is to "
        "access functionality in the public NumPy API. If that is the case, "
        "use the public NumPy API. If not, you are using NumPy internals. "
        "If you would still like to access an internal attribute, "
        f"use {new_module}.{attr}.",
        DeprecationWarning,
        stacklevel=3
    )
<|endoftext|>from numpy._core import multiarray

# these must import without warning or error from numpy.core.multiarray to
# support old pickle files
for item in ["_reconstruct", "scalar"]:
    globals()[item] = getattr(multiarray, item)

# Pybind11 (in versions <= 2.11.1) imports _ARRAY_API from the multiarray
# submodule as a part of NumPy initialization, therefore it must be importable
# without a warning.
_ARRAY_API = multiarray._ARRAY_API

def __getattr__(attr_name):
    from numpy._core import multiarray
    from ._utils import _raise_warning
    ret = getattr(multiarray, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.multiarray' has no attribute {attr_name}")
    _raise_warning(attr_name, "multiarray")
    return ret


del multiarray
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import function_base
    from ._utils import _raise_warning
    ret = getattr(function_base, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.function_base' has no attribute {attr_name}")
    _raise_warning(attr_name, "function_base")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import fromnumeric
    from ._utils import _raise_warning
    ret = getattr(fromnumeric, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.fromnumeric' has no attribute {attr_name}")
    _raise_warning(attr_name, "fromnumeric")
    return ret
<|endoftext|>from numpy._core import _multiarray_umath
from numpy import ufunc

for item in _multiarray_umath.__dir__():
    # ufuncs appear in pickles with a path in numpy.core._multiarray_umath
    # and so must import from this namespace without warning or error
    attr = getattr(_multiarray_umath, item)
    if isinstance(attr, ufunc):
        globals()[item] = attr


def __getattr__(attr_name):
    from numpy._core import _multiarray_umath
    from ._utils import _raise_warning

    if attr_name in {"_ARRAY_API", "_UFUNC_API"}:
        from numpy.version import short_version
        import textwrap
        import traceback
        import sys

        msg = textwrap.dedent(f"""
            A module that was compiled using NumPy 1.x cannot be run in
            NumPy {short_version} as it may crash. To support both 1.x and 2.x
            versions of NumPy, modules must be compiled with NumPy 2.0.
            Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

            If you are a user of the module, the easiest solution will be to
            downgrade to 'numpy<2' or try to upgrade the affected module.
            We expect that some modules will need time to support NumPy 2.

            """)
        tb_msg = "Traceback (most recent call last):"
        for line in traceback.format_stack()[:-1]:
            if "frozen importlib" in line:
                continue
            tb_msg += line

        # Also print the message (with traceback).  This is because old versions
        # of NumPy unfortunately set up the import to replace (and hide) the
        # error.  The traceback shouldn't be needed, but e.g. pytest plugins
        # seem to swallow it and we should be failing anyway...
        sys.stderr.write(msg + tb_msg)
        raise ImportError(msg)

    ret = getattr(_multiarray_umath, attr_name, None)
    if ret is None:
        raise AttributeError(
            "module 'numpy.core._multiarray_umath' has no attribute "
            f"{attr_name}")
    _raise_warning(attr_name, "_multiarray_umath")
    return ret


del _multiarray_umath, ufunc
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import _dtype_ctypes
    from ._utils import _raise_warning
    ret = getattr(_dtype_ctypes, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core._dtype_ctypes' has no attribute {attr_name}")
    _raise_warning(attr_name, "_dtype_ctypes")
    return ret
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import umath
    from ._utils import _raise_warning
    ret = getattr(umath, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.umath' has no attribute {attr_name}")
    _raise_warning(attr_name, "umath")
    return ret
<|endoftext|>from __future__ import annotations

import sys
from collections.abc import Collection, Callable, Sequence
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable, TYPE_CHECKING

import numpy as np
from numpy import (
    ndarray,
    dtype,
    generic,
    unsignedinteger,
    integer,
    floating,
    complexfloating,
    number,
    timedelta64,
    datetime64,
    object_,
    void,
    str_,
    bytes_,
)
from ._nbit_base import _32Bit, _64Bit
from ._nested_sequence import _NestedSequence
from ._shape import _Shape

if TYPE_CHECKING:
    StringDType = np.dtypes.StringDType
else:
    # at runtime outside of type checking importing this from numpy.dtypes
    # would lead to a circular import
    from numpy._core.multiarray import StringDType

_T = TypeVar("_T")
_ScalarType = TypeVar("_ScalarType", bound=generic)
_ScalarType_co = TypeVar("_ScalarType_co", bound=generic, covariant=True)
_DType = TypeVar("_DType", bound=dtype[Any])
_DType_co = TypeVar("_DType_co", covariant=True, bound=dtype[Any])

NDArray: TypeAlias = ndarray[_Shape, dtype[_ScalarType_co]]

# The `_SupportsArray` protocol only cares about the default dtype
# (i.e. `dtype=None` or no `dtype` parameter at all) of the to-be returned
# array.
# Concrete implementations of the protocol are responsible for adding
# any and all remaining overloads
@runtime_checkable
class _SupportsArray(Protocol[_DType_co]):
    def __array__(self) -> ndarray[Any, _DType_co]: ...


@runtime_checkable
class _SupportsArrayFunc(Protocol):
    """A protocol class representing `~class.__array_function__`."""
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Collection[type[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object: ...


# TODO: Wait until mypy supports recursive objects in combination with typevars
_FiniteNestedSequence: TypeAlias = (
    _T
    | Sequence[_T]
    | Sequence[Sequence[_T]]
    | Sequence[Sequence[Sequence[_T]]]
    | Sequence[Sequence[Sequence[Sequence[_T]]]]
)

# A subset of `npt.ArrayLike` that can be parametrized w.r.t. `np.generic`
_ArrayLike: TypeAlias = (
    _SupportsArray[dtype[_ScalarType]]
    | _NestedSequence[_SupportsArray[dtype[_ScalarType]]]
)

# A union representing array-like objects; consists of two typevars:
# One representing types that can be parametrized w.r.t. `np.dtype`
# and another one for the rest
_DualArrayLike: TypeAlias = (
    _SupportsArray[_DType]
    | _NestedSequence[_SupportsArray[_DType]]
    | _T
    | _NestedSequence[_T]
)

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as _Buffer
else:
    @runtime_checkable
    class _Buffer(Protocol):
        def __buffer__(self, flags: int, /) -> memoryview: ...

ArrayLike: TypeAlias = _Buffer | _DualArrayLike[
    dtype[Any],
    bool | int | float | complex | str | bytes,
]

# `ArrayLike<X>_co`: array-like objects that can be coerced into `X`
# given the casting rules `same_kind`
_ArrayLikeBool_co: TypeAlias = _DualArrayLike[
    dtype[np.bool],
    bool,
]
_ArrayLikeUInt_co: TypeAlias = _DualArrayLike[
    dtype[np.bool] | dtype[unsignedinteger[Any]],
    bool,
]
_ArrayLikeInt_co: TypeAlias = _DualArrayLike[
    dtype[np.bool] | dtype[integer[Any]],
    bool | int,
]
_ArrayLikeFloat_co: TypeAlias = _DualArrayLike[
    dtype[np.bool] | dtype[integer[Any]] | dtype[floating[Any]],
    bool | int | float,
]
_ArrayLikeComplex_co: TypeAlias = _DualArrayLike[
    (
        dtype[np.bool]
        | dtype[integer[Any]]
        | dtype[floating[Any]]
        | dtype[complexfloating[Any, Any]]
    ),
    bool | int | float | complex,
]
_ArrayLikeNumber_co: TypeAlias = _DualArrayLike[
    dtype[np.bool] | dtype[number[Any]],
    bool | int | float | complex,
]
_ArrayLikeTD64_co: TypeAlias = _DualArrayLike[
    dtype[np.bool] | dtype[integer[Any]] | dtype[timedelta64],
    bool | int,
]
_ArrayLikeDT64_co: TypeAlias = (
    _SupportsArray[dtype[datetime64]]
    | _NestedSequence[_SupportsArray[dtype[datetime64]]]
)
_ArrayLikeObject_co: TypeAlias = (
    _SupportsArray[dtype[object_]]
    | _NestedSequence[_SupportsArray[dtype[object_]]]
)

_ArrayLikeVoid_co: TypeAlias = (
    _SupportsArray[dtype[void]]
    | _NestedSequence[_SupportsArray[dtype[void]]]
)
_ArrayLikeStr_co: TypeAlias = _DualArrayLike[
    dtype[str_],
    str,
]
_ArrayLikeBytes_co: TypeAlias = _DualArrayLike[
    dtype[bytes_],
    bytes,
]
_ArrayLikeString_co: TypeAlias = _DualArrayLike[
    StringDType,
    str
]
_ArrayLikeAnyString_co: TypeAlias = (
    _ArrayLikeStr_co |
    _ArrayLikeBytes_co |
    _ArrayLikeString_co
)

__Float64_co: TypeAlias = np.floating[_64Bit] | np.float32 | np.float16 | np.integer | np.bool
__Complex128_co: TypeAlias = np.number[_64Bit] | np.number[_32Bit] | np.float16 | np.integer | np.bool
_ArrayLikeFloat64_co: TypeAlias = _DualArrayLike[dtype[__Float64_co], float | int]
_ArrayLikeComplex128_co: TypeAlias = _DualArrayLike[dtype[__Complex128_co], complex | float | int]

# NOTE: This includes `builtins.bool`, but not `numpy.bool`.
_ArrayLikeInt: TypeAlias = _DualArrayLike[
    dtype[integer[Any]],
    int,
]

# Extra ArrayLike type so that pyright can deal with NDArray[Any]
# Used as the first overload, should only match NDArray[Any],
# not any actual types.
# https://github.com/numpy/numpy/pull/22193
if sys.version_info >= (3, 11):
    from typing import Never as _UnknownType
else:
    from typing import NoReturn as _UnknownType


_ArrayLikeUnknown: TypeAlias = _DualArrayLike[
    dtype[_UnknownType],
    _UnknownType,
]
<|endoftext|>"""A module containing the `_NestedSequence` protocol."""

from __future__ import annotations

from typing import (
    Any,
    TypeVar,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["_NestedSequence"]

_T_co = TypeVar("_T_co", covariant=True)


@runtime_checkable
class _NestedSequence(Protocol[_T_co]):
    """A protocol for representing nested sequences.

    Warning
    -------
    `_NestedSequence` currently does not work in combination with typevars,
    *e.g.* ``def func(a: _NestedSequnce[T]) -> T: ...``.

    See Also
    --------
    collections.abc.Sequence
        ABCs for read-only and mutable :term:`sequences`.

    Examples
    --------
    .. code-block:: python

        >>> from __future__ import annotations

        >>> from typing import TYPE_CHECKING
        >>> import numpy as np
        >>> from numpy._typing import _NestedSequence

        >>> def get_dtype(seq: _NestedSequence[float]) -> np.dtype[np.float64]:
        ...     return np.asarray(seq).dtype

        >>> a = get_dtype([1.0])
        >>> b = get_dtype([[1.0]])
        >>> c = get_dtype([[[1.0]]])
        >>> d = get_dtype([[[[1.0]]]])

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
        ...     # note:     b: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
        ...     # note:     c: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
        ...     # note:     d: numpy.dtype[numpy.floating[numpy._typing._64Bit]]

    """

    def __len__(self, /) -> int:
        """Implement ``len(self)``."""
        raise NotImplementedError

    def __getitem__(self, index: int, /) -> _T_co | _NestedSequence[_T_co]:
        """Implement ``self[x]``."""
        raise NotImplementedError

    def __contains__(self, x: object, /) -> bool:
        """Implement ``x in self``."""
        raise NotImplementedError

    def __iter__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]:
        """Implement ``iter(self)``."""
        raise NotImplementedError

    def __reversed__(self, /) -> Iterator[_T_co | _NestedSequence[_T_co]]:
        """Implement ``reversed(self)``."""
        raise NotImplementedError

    def count(self, value: Any, /) -> int:
        """Return the number of occurrences of `value`."""
        raise NotImplementedError

    def index(self, value: Any, /) -> int:
        """Return the first index of `value`."""
        raise NotImplementedError
<|endoftext|>"""Private counterpart of ``numpy.typing``."""

from __future__ import annotations

from ._nested_sequence import (
    _NestedSequence as _NestedSequence,
)
from ._nbit_base import (
    NBitBase as NBitBase,
    _8Bit as _8Bit,
    _16Bit as _16Bit,
    _32Bit as _32Bit,
    _64Bit as _64Bit,
    _80Bit as _80Bit,
    _96Bit as _96Bit,
    _128Bit as _128Bit,
    _256Bit as _256Bit,
)
from ._nbit import (
    _NBitByte as _NBitByte,
    _NBitShort as _NBitShort,
    _NBitIntC as _NBitIntC,
    _NBitIntP as _NBitIntP,
    _NBitInt as _NBitInt,
    _NBitLong as _NBitLong,
    _NBitLongLong as _NBitLongLong,
    _NBitHalf as _NBitHalf,
    _NBitSingle as _NBitSingle,
    _NBitDouble as _NBitDouble,
    _NBitLongDouble as _NBitLongDouble,
)
from ._char_codes import (
    _BoolCodes as _BoolCodes,
    _UInt8Codes as _UInt8Codes,
    _UInt16Codes as _UInt16Codes,
    _UInt32Codes as _UInt32Codes,
    _UInt64Codes as _UInt64Codes,
    _Int8Codes as _Int8Codes,
    _Int16Codes as _Int16Codes,
    _Int32Codes as _Int32Codes,
    _Int64Codes as _Int64Codes,
    _Float16Codes as _Float16Codes,
    _Float32Codes as _Float32Codes,
    _Float64Codes as _Float64Codes,
    _Complex64Codes as _Complex64Codes,
    _Complex128Codes as _Complex128Codes,
    _ByteCodes as _ByteCodes,
    _ShortCodes as _ShortCodes,
    _IntCCodes as _IntCCodes,
    _IntPCodes as _IntPCodes,
    _IntCodes as _IntCodes,
    _LongCodes as _LongCodes,
    _LongLongCodes as _LongLongCodes,
    _UByteCodes as _UByteCodes,
    _UShortCodes as _UShortCodes,
    _UIntCCodes as _UIntCCodes,
    _UIntPCodes as _UIntPCodes,
    _UIntCodes as _UIntCodes,
    _ULongCodes as _ULongCodes,
    _ULongLongCodes as _ULongLongCodes,
    _HalfCodes as _HalfCodes,
    _SingleCodes as _SingleCodes,
    _DoubleCodes as _DoubleCodes,
    _LongDoubleCodes as _LongDoubleCodes,
    _CSingleCodes as _CSingleCodes,
    _CDoubleCodes as _CDoubleCodes,
    _CLongDoubleCodes as _CLongDoubleCodes,
    _DT64Codes as _DT64Codes,
    _TD64Codes as _TD64Codes,
    _StrCodes as _StrCodes,
    _BytesCodes as _BytesCodes,
    _VoidCodes as _VoidCodes,
    _ObjectCodes as _ObjectCodes,
    _StringCodes as _StringCodes,
    _UnsignedIntegerCodes as _UnsignedIntegerCodes,
    _SignedIntegerCodes as _SignedIntegerCodes,
    _IntegerCodes as _IntegerCodes,
    _FloatingCodes as _FloatingCodes,
    _ComplexFloatingCodes as _ComplexFloatingCodes,
    _InexactCodes as _InexactCodes,
    _NumberCodes as _NumberCodes,
    _CharacterCodes as _CharacterCodes,
    _FlexibleCodes as _FlexibleCodes,
    _GenericCodes as _GenericCodes,
)
from ._scalars import (
    _CharLike_co as _CharLike_co,
    _BoolLike_co as _BoolLike_co,
    _UIntLike_co as _UIntLike_co,
    _IntLike_co as _IntLike_co,
    _FloatLike_co as _FloatLike_co,
    _ComplexLike_co as _ComplexLike_co,
    _TD64Like_co as _TD64Like_co,
    _NumberLike_co as _NumberLike_co,
    _ScalarLike_co as _ScalarLike_co,
    _VoidLike_co as _VoidLike_co,
)
from ._shape import (
    _Shape as _Shape,
    _ShapeLike as _ShapeLike,
)
from ._dtype_like import (
    DTypeLike as DTypeLike,
    _DTypeLike as _DTypeLike,
    _SupportsDType as _SupportsDType,
    _VoidDTypeLike as _VoidDTypeLike,
    _DTypeLikeBool as _DTypeLikeBool,
    _DTypeLikeUInt as _DTypeLikeUInt,
    _DTypeLikeInt as _DTypeLikeInt,
    _DTypeLikeFloat as _DTypeLikeFloat,
    _DTypeLikeComplex as _DTypeLikeComplex,
    _DTypeLikeTD64 as _DTypeLikeTD64,
    _DTypeLikeDT64 as _DTypeLikeDT64,
    _DTypeLikeObject as _DTypeLikeObject,
    _DTypeLikeVoid as _DTypeLikeVoid,
    _DTypeLikeStr as _DTypeLikeStr,
    _DTypeLikeBytes as _DTypeLikeBytes,
    _DTypeLikeComplex_co as _DTypeLikeComplex_co,
)
from ._array_like import (
    NDArray as NDArray,
    ArrayLike as ArrayLike,
    _ArrayLike as _ArrayLike,
    _ArrayLikeInt as _ArrayLikeInt,
    _ArrayLikeBool_co as _ArrayLikeBool_co,
    _ArrayLikeUInt_co as _ArrayLikeUInt_co,
    _ArrayLikeInt_co as _ArrayLikeInt_co,
    _ArrayLikeFloat_co as _ArrayLikeFloat_co,
    _ArrayLikeFloat64_co as _ArrayLikeFloat64_co,
    _ArrayLikeComplex_co as _ArrayLikeComplex_co,
    _ArrayLikeComplex128_co as _ArrayLikeComplex128_co,
    _ArrayLikeNumber_co as _ArrayLikeNumber_co,
    _ArrayLikeTD64_co as _ArrayLikeTD64_co,
    _ArrayLikeDT64_co as _ArrayLikeDT64_co,
    _ArrayLikeObject_co as _ArrayLikeObject_co,
    _ArrayLikeVoid_co as _ArrayLikeVoid_co,
    _ArrayLikeStr_co as _ArrayLikeStr_co,
    _ArrayLikeBytes_co as _ArrayLikeBytes_co,
    _ArrayLikeString_co as _ArrayLikeString_co,
    _ArrayLikeAnyString_co as _ArrayLikeAnyString_co,
    _ArrayLikeUnknown as _ArrayLikeUnknown,
    _FiniteNestedSequence as _FiniteNestedSequence,
    _SupportsArray as _SupportsArray,
    _SupportsArrayFunc as _SupportsArrayFunc,
    _UnknownType as _UnknownType,
)

from ._ufunc import (
    _UFunc_Nin1_Nout1 as _UFunc_Nin1_Nout1,
    _UFunc_Nin2_Nout1 as _UFunc_Nin2_Nout1,
    _UFunc_Nin1_Nout2 as _UFunc_Nin1_Nout2,
    _UFunc_Nin2_Nout2 as _UFunc_Nin2_Nout2,
    _GUFunc_Nin2_Nout1 as _GUFunc_Nin2_Nout1,
)
<|endoftext|>"""A module with the precisions of generic `~numpy.number` types."""
from .._utils import set_module
from typing import final


@final  # Disallow the creation of arbitrary `NBitBase` subclasses
@set_module("numpy.typing")
class NBitBase:
    """
    A type representing `numpy.number` precision during static type checking.

    Used exclusively for the purpose static type checking, `NBitBase`
    represents the base of a hierarchical set of subclasses.
    Each subsequent subclass is herein used for representing a lower level
    of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

    .. versionadded:: 1.20

    Examples
    --------
    Below is a typical usage example: `NBitBase` is herein used for annotating
    a function that takes a float and integer of arbitrary precision
    as arguments and returns a new float of whichever precision is largest
    (*e.g.* ``np.float16 + np.int64 -> np.float64``).

    .. code-block:: python

        >>> from __future__ import annotations
        >>> from typing import TypeVar, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> S = TypeVar("S", bound=npt.NBitBase)
        >>> T = TypeVar("T", bound=npt.NBitBase)

        >>> def add(a: np.floating[S], b: np.integer[T]) -> np.floating[S | T]:
        ...     return a + b

        >>> a = np.float16()
        >>> b = np.int64()
        >>> out = add(a, b)

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
        ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
        ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

    """

    def __init_subclass__(cls) -> None:
        allowed_names = {
            "NBitBase", "_256Bit", "_128Bit", "_96Bit", "_80Bit",
            "_64Bit", "_32Bit", "_16Bit", "_8Bit",
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()

@final
@set_module("numpy._typing")
# Silence errors about subclassing a `@final`-decorated class
class _256Bit(NBitBase):  # type: ignore[misc]
    pass

@final
@set_module("numpy._typing")
class _128Bit(_256Bit):  # type: ignore[misc]
    pass

@final
@set_module("numpy._typing")
class _96Bit(_128Bit):  # type: ignore[misc]
    pass

@final
@set_module("numpy._typing")
class _80Bit(_96Bit):  # type: ignore[misc]
    pass

@final
@set_module("numpy._typing")
class _64Bit(_80Bit):  # type: ignore[misc]
    pass

@final
@set_module("numpy._typing")
class _32Bit(_64Bit):  # type: ignore[misc]
    pass

@final
@set_module("numpy._typing")
class _16Bit(_32Bit):  # type: ignore[misc]
    pass

@final
@set_module("numpy._typing")
class _8Bit(_16Bit):  # type: ignore[misc]
    pass
<|endoftext|>from collections.abc import Sequence  # noqa: F811
from typing import (
    Any,
    TypeAlias,
    TypeVar,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import numpy as np

from ._shape import _ShapeLike

from ._char_codes import (
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _LongCodes,
    _LongLongCodes,
    _IntPCodes,
    _IntCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _ULongCodes,
    _ULongLongCodes,
    _UIntPCodes,
    _UIntCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,
)

_SCT = TypeVar("_SCT", bound=np.generic)
_DType_co = TypeVar("_DType_co", covariant=True, bound=np.dtype[Any])

_DTypeLikeNested: TypeAlias = Any  # TODO: wait for support for recursive types


# Mandatory keys
class _DTypeDictBase(TypedDict):
    names: Sequence[str]
    formats: Sequence[_DTypeLikeNested]


# Mandatory + optional keys
class _DTypeDict(_DTypeDictBase, total=False):
    # Only `str` elements are usable as indexing aliases,
    # but `titles` can in principle accept any object
    offsets: Sequence[int]
    titles: Sequence[Any]
    itemsize: int
    aligned: bool


# A protocol for anything with the dtype attribute
@runtime_checkable
class _SupportsDType(Protocol[_DType_co]):
    @property
    def dtype(self) -> _DType_co: ...


# A subset of `npt.DTypeLike` that can be parametrized w.r.t. `np.generic`
_DTypeLike: TypeAlias = (
    np.dtype[_SCT]
    | type[_SCT]
    | _SupportsDType[np.dtype[_SCT]]
)


# Would create a dtype[np.void]
_VoidDTypeLike: TypeAlias = (
    # (flexible_dtype, itemsize)
    tuple[_DTypeLikeNested, int]
    # (fixed_dtype, shape)
    | tuple[_DTypeLikeNested, _ShapeLike]
    # [(field_name, field_dtype, field_shape), ...]
    #
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some
    # examples.
    | list[Any]
    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ...,
    #  'itemsize': ...}
    | _DTypeDict
    # (base_dtype, new_dtype)
    | tuple[_DTypeLikeNested, _DTypeLikeNested]
)

# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DTypeLike: TypeAlias = (
    np.dtype[Any]
    # default data type (float64)
    | None
    # array-scalar types and generic types
    | type[Any]  # NOTE: We're stuck with `type[Any]` due to object dtypes
    # anything with a dtype attribute
    | _SupportsDType[np.dtype[Any]]
    # character codes, type strings or comma-separated fields, e.g., 'float64'
    | str
    | _VoidDTypeLike
)

# NOTE: while it is possible to provide the dtype as a dict of
# dtype-like objects (e.g. `{'field1': ..., 'field2': ..., ...}`),
# this syntax is officially discouraged and
# therefore not included in the type-union defining `DTypeLike`.
#
# See https://github.com/numpy/numpy/issues/16891 for more details.

# Aliases for commonly used dtype-like objects.
# Note that the precision of `np.number` subclasses is ignored herein.
_DTypeLikeBool: TypeAlias = (
    type[bool]
    | type[np.bool]
    | np.dtype[np.bool]
    | _SupportsDType[np.dtype[np.bool]]
    | _BoolCodes
)
_DTypeLikeUInt: TypeAlias = (
    type[np.unsignedinteger[Any]]
    | np.dtype[np.unsignedinteger[Any]]
    | _SupportsDType[np.dtype[np.unsignedinteger[Any]]]
    | _UInt8Codes
    | _UInt16Codes
    | _UInt32Codes
    | _UInt64Codes
    | _UByteCodes
    | _UShortCodes
    | _UIntCCodes
    | _LongCodes
    | _ULongLongCodes
    | _UIntPCodes
    | _UIntCodes
)
_DTypeLikeInt: TypeAlias = (
    type[int]
    | type[np.signedinteger[Any]]
    | np.dtype[np.signedinteger[Any]]
    | _SupportsDType[np.dtype[np.signedinteger[Any]]]
    | _Int8Codes
    | _Int16Codes
    | _Int32Codes
    | _Int64Codes
    | _ByteCodes
    | _ShortCodes
    | _IntCCodes
    | _LongCodes
    | _LongLongCodes
    | _IntPCodes
    | _IntCodes
)
_DTypeLikeFloat: TypeAlias = (
    type[float]
    | type[np.floating[Any]]
    | np.dtype[np.floating[Any]]
    | _SupportsDType[np.dtype[np.floating[Any]]]
    | _Float16Codes
    | _Float32Codes
    | _Float64Codes
    | _HalfCodes
    | _SingleCodes
    | _DoubleCodes
    | _LongDoubleCodes
)
_DTypeLikeComplex: TypeAlias = (
    type[complex]
    | type[np.complexfloating[Any]]
    | np.dtype[np.complexfloating[Any]]
    | _SupportsDType[np.dtype[np.complexfloating[Any]]]
    | _Complex64Codes
    | _Complex128Codes
    | _CSingleCodes
    | _CDoubleCodes
    | _CLongDoubleCodes
)
_DTypeLikeDT64: TypeAlias = (
    type[np.timedelta64]
    | np.dtype[np.timedelta64]
    | _SupportsDType[np.dtype[np.timedelta64]]
    | _TD64Codes
)
_DTypeLikeTD64: TypeAlias = (
    type[np.datetime64]
    | np.dtype[np.datetime64]
    | _SupportsDType[np.dtype[np.datetime64]]
    | _DT64Codes
)
_DTypeLikeStr: TypeAlias = (
    type[str]
    | type[np.str_]
    | np.dtype[np.str_]
    | _SupportsDType[np.dtype[np.str_]]
    | _StrCodes
)
_DTypeLikeBytes: TypeAlias = (
    type[bytes]
    | type[np.bytes_]
    | np.dtype[np.bytes_]
    | _SupportsDType[np.dtype[np.bytes_]]
    | _BytesCodes
)
_DTypeLikeVoid: TypeAlias = (
    type[np.void]
    | np.dtype[np.void]
    | _SupportsDType[np.dtype[np.void]]
    | _VoidCodes
    | _VoidDTypeLike
)
_DTypeLikeObject: TypeAlias = (
    type
    | np.dtype[np.object_]
    | _SupportsDType[np.dtype[np.object_]]
    | _ObjectCodes
)

_DTypeLikeComplex_co: TypeAlias = (
    _DTypeLikeBool
    | _DTypeLikeUInt
    | _DTypeLikeInt
    | _DTypeLikeFloat
    | _DTypeLikeComplex
)
<|endoftext|>"""A module for creating docstrings for sphinx ``data`` domains."""

import re
import textwrap

from ._array_like import NDArray

_docstrings_list = []


def add_newdoc(name: str, value: str, doc: str) -> None:
    """Append ``_docstrings_list`` with a docstring for `name`.

    Parameters
    ----------
    name : str
        The name of the object.
    value : str
        A string-representation of the object.
    doc : str
        The docstring of the object.

    """
    _docstrings_list.append((name, value, doc))


def _parse_docstrings() -> str:
    """Convert all docstrings in ``_docstrings_list`` into a single
    sphinx-legible text block.

    """
    type_list_ret = []
    for name, value, doc in _docstrings_list:
        s = textwrap.dedent(doc).replace("\n", "\n    ")

        # Replace sections by rubrics
        lines = s.split("\n")
        new_lines = []
        indent = ""
        for line in lines:
            m = re.match(r'^(\s+)[-=]+\s*$', line)
            if m and new_lines:
                prev = textwrap.dedent(new_lines.pop())
                if prev == "Examples":
                    indent = ""
                    new_lines.append(f'{m.group(1)}.. rubric:: {prev}')
                else:
                    indent = 4 * " "
                    new_lines.append(f'{m.group(1)}.. admonition:: {prev}')
                new_lines.append("")
            else:
                new_lines.append(f"{indent}{line}")

        s = "\n".join(new_lines)
        s_block = f""".. data:: {name}\n    :value: {value}\n    {s}"""
        type_list_ret.append(s_block)
    return "\n".join(type_list_ret)


add_newdoc('ArrayLike', 'typing.Union[...]',
    """
    A `~typing.Union` representing objects that can be coerced
    into an `~numpy.ndarray`.

    Among others this includes the likes of:

    * Scalars.
    * (Nested) sequences.
    * Objects implementing the `~class.__array__` protocol.

    .. versionadded:: 1.20

    See Also
    --------
    :term:`array_like`:
        Any scalar or sequence that can be interpreted as an ndarray.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> def as_array(a: npt.ArrayLike) -> np.ndarray:
        ...     return np.array(a)

    """)

add_newdoc('DTypeLike', 'typing.Union[...]',
    """
    A `~typing.Union` representing objects that can be coerced
    into a `~numpy.dtype`.

    Among others this includes the likes of:

    * :class:`type` objects.
    * Character codes or the names of :class:`type` objects.
    * Objects with the ``.dtype`` attribute.

    .. versionadded:: 1.20

    See Also
    --------
    :ref:`Specifying and constructing data types <arrays.dtypes.constructing>`
        A comprehensive overview of all objects that can be coerced
        into data types.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> def as_dtype(d: npt.DTypeLike) -> np.dtype:
        ...     return np.dtype(d)

    """)

add_newdoc('NDArray', repr(NDArray),
    """
    A `np.ndarray[tuple[int, ...], np.dtype[+ScalarType]] <numpy.ndarray>`
    type alias :term:`generic <generic type>` w.r.t. its
    `dtype.type <numpy.dtype.type>`.

    Can be used during runtime for typing arrays with a given dtype
    and unspecified shape.

    .. versionadded:: 1.21

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> print(npt.NDArray)
        numpy.ndarray[tuple[int, ...], numpy.dtype[+_ScalarType_co]]

        >>> print(npt.NDArray[np.float64])
        numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.float64]]

        >>> NDArrayInt = npt.NDArray[np.int_]
        >>> a: NDArrayInt = np.arange(10)

        >>> def func(a: npt.ArrayLike) -> npt.NDArray[Any]:
        ...     return np.array(a)

    """)

_docstrings = _parse_docstrings()
<|endoftext|>"""Utility to compare pep440 compatible version strings.

The LooseVersion and StrictVersion classes that distutils provides don't
work; they don't recognize anything like alpha/beta/rc/dev versions.
"""

# Copyright (c) Donald Stufft and individual contributors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import collections
import itertools
import re


__all__ = [
    "parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN",
]


# BEGIN packaging/_structures.py


class Infinity:
    def __repr__(self):
        return "Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not isinstance(other, self.__class__)

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __neg__(self):
        return NegativeInfinity


Infinity = Infinity()


class NegativeInfinity:
    def __repr__(self):
        return "-Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not isinstance(other, self.__class__)

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __neg__(self):
        return Infinity


# BEGIN packaging/version.py


NegativeInfinity = NegativeInfinity()

_Version = collections.namedtuple(
    "_Version",
    ["epoch", "release", "dev", "pre", "post", "local"],
)


def parse(version):
    """
    Parse the given version string and return either a :class:`Version` object
    or a :class:`LegacyVersion` object depending on if the given version is
    a valid PEP 440 version or a legacy version.
    """
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)


class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """


class _BaseVersion:

    def __hash__(self):
        return hash(self._key)

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)

    def _compare(self, other, method):
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return method(self._key, other._key)


class LegacyVersion(_BaseVersion):

    def __init__(self, version):
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)

    def __str__(self):
        return self._version

    def __repr__(self):
        return "<LegacyVersion({0})>".format(repr(str(self)))

    @property
    def public(self):
        return self._version

    @property
    def base_version(self):
        return self._version

    @property
    def local(self):
        return None

    @property
    def is_prerelease(self):
        return False

    @property
    def is_postrelease(self):
        return False


_legacy_version_component_re = re.compile(
    r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE,
)

_legacy_version_replacement_map = {
    "pre": "c", "preview": "c", "-": "final-", "rc": "c", "dev": "@",
}


def _parse_version_parts(s):
    for part in _legacy_version_component_re.split(s):
        part = _legacy_version_replacement_map.get(part, part)

        if not part or part == ".":
            continue

        if part[:1] in "0123456789":
            # pad for numeric comparison
            yield part.zfill(8)
        else:
            yield "*" + part

    # ensure that alpha/beta/candidate are before final
    yield "*final"


def _legacy_cmpkey(version):
    # We hardcode an epoch of -1 here. A PEP 440 version can only have an epoch
    # greater than or equal to 0. This will effectively put the LegacyVersion,
    # which uses the defacto standard originally implemented by setuptools,
    # as before all PEP 440 versions.
    epoch = -1

    # This scheme is taken from pkg_resources.parse_version setuptools prior to
    # its adoption of the packaging library.
    parts = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            # remove "-" before a prerelease tag
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # remove trailing zeros from each series of numeric parts
            while parts and parts[-1] == "00000000":
                parts.pop()

        parts.append(part)
    parts = tuple(parts)

    return epoch, parts


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


class Version(_BaseVersion):

    _regex = re.compile(
        r"^\s*" + VERSION_PATTERN + r"\s*$",
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(self, version):
        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion("Invalid version: '{0}'".format(version))

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(
                match.group("pre_l"),
                match.group("pre_n"),
            ),
            post=_parse_letter_version(
                match.group("post_l"),
                match.group("post_n1") or match.group("post_n2"),
            ),
            dev=_parse_letter_version(
                match.group("dev_l"),
                match.group("dev_n"),
            ),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self):
        return "<Version({0})>".format(repr(str(self)))

    def __str__(self):
        parts = []

        # Epoch
        if self._version.epoch != 0:
            parts.append("{0}!".format(self._version.epoch))

        # Release segment
        parts.append(".".join(str(x) for x in self._version.release))

        # Pre-release
        if self._version.pre is not None:
            parts.append("".join(str(x) for x in self._version.pre))

        # Post-release
        if self._version.post is not None:
            parts.append(".post{0}".format(self._version.post[1]))

        # Development release
        if self._version.dev is not None:
            parts.append(".dev{0}".format(self._version.dev[1]))

        # Local version segment
        if self._version.local is not None:
            parts.append(
                "+{0}".format(".".join(str(x) for x in self._version.local))
            )

        return "".join(parts)

    @property
    def public(self):
        return str(self).split("+", 1)[0]

    @property
    def base_version(self):
        parts = []

        # Epoch
        if self._version.epoch != 0:
            parts.append("{0}!".format(self._version.epoch))

        # Release segment
        parts.append(".".join(str(x) for x in self._version.release))

        return "".join(parts)

    @property
    def local(self):
        version_string = str(self)
        if "+" in version_string:
            return version_string.split("+", 1)[1]

    @property
    def is_prerelease(self):
        return bool(self._version.dev or self._version.pre)

    @property
    def is_postrelease(self):
        return bool(self._version.post)


def _parse_letter_version(letter, number):
    if letter:
        # We assume there is an implicit 0 in a pre-release if there is
        # no numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower-case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    if not letter and number:
        # We assume that if we are given a number but not given a letter,
        # then this is using the implicit post release syntax (e.g., 1.0-1)
        letter = "post"

        return letter, int(number)


_local_version_seperators = re.compile(r"[\._-]")


def _parse_local_version(local):
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_seperators.split(local)
        )


def _cmpkey(epoch, release, pre, post, dev, local):
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non-zero, then take the rest,
    # re-reverse it back into the correct order, and make it a tuple and use
    # that for our sorting key.
    release = tuple(
        reversed(list(
            itertools.dropwhile(
                lambda x: x == 0,
                reversed(release),
            )
        ))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre-segment, but we _only_ want to do this
    # if there is no pre- or a post-segment. If we have one of those, then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        pre = -Infinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        pre = Infinity

    # Versions without a post-segment should sort before those with one.
    if post is None:
        post = -Infinity

    # Versions without a development segment should sort after those with one.
    if dev is None:
        dev = Infinity

    if local is None:
        # Versions without a local segment should sort before those with one.
        local = -Infinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alphanumeric segments sort before numeric segments
        # - Alphanumeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        local = tuple(
            (i, "") if isinstance(i, int) else (-Infinity, i)
            for i in local
        )

    return epoch, release, pre, post, dev, local
<|endoftext|>"""
This is a module for defining private helpers which do not depend on the
rest of NumPy.

Everything in here must be self-contained so that it can be
imported anywhere else without creating circular imports.
If a utility requires the import of NumPy, it probably belongs
in ``numpy._core``.
"""

import functools
import warnings
from ._convertions import asunicode, asbytes


def set_module(module):
    """Private decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module('numpy')
        def example():
            pass

        assert example.__module__ == 'numpy'
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator


def _rename_parameter(old_names, new_names, dep_version=None):
    """
    Generate decorator for backward-compatible keyword renaming.

    Apply the decorator generated by `_rename_parameter` to functions with a
    renamed parameter to maintain backward-compatibility.

    After decoration, the function behaves as follows:
    If only the new parameter is passed into the function, behave as usual.
    If only the old parameter is passed into the function (as a keyword), raise
    a DeprecationWarning if `dep_version` is provided, and behave as usual
    otherwise.
    If both old and new parameters are passed into the function, raise a
    DeprecationWarning if `dep_version` is provided, and raise the appropriate
    TypeError (function got multiple values for argument).

    Parameters
    ----------
    old_names : list of str
        Old names of parameters
    new_name : list of str
        New names of parameters
    dep_version : str, optional
        Version of NumPy in which old parameter was deprecated in the format
        'X.Y.Z'. If supplied, the deprecation message will indicate that
        support for the old parameter will be removed in version 'X.Y+2.Z'

    Notes
    -----
    Untested with functions that accept *args. Probably won't work as written.

    """
    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            __tracebackhide__ = True  # Hide traceback for py.test
            for old_name, new_name in zip(old_names, new_names):
                if old_name in kwargs:
                    if dep_version:
                        end_version = dep_version.split('.')
                        end_version[1] = str(int(end_version[1]) + 2)
                        end_version = '.'.join(end_version)
                        msg = (f"Use of keyword argument `{old_name}` is "
                               f"deprecated and replaced by `{new_name}`. "
                               f"Support for `{old_name}` will be removed "
                               f"in NumPy {end_version}.")
                        warnings.warn(msg, DeprecationWarning, stacklevel=2)
                    if new_name in kwargs:
                        msg = (f"{fun.__name__}() got multiple values for "
                               f"argument now known as `{new_name}`")
                        raise TypeError(msg)
                    kwargs[new_name] = kwargs.pop(old_name)
            return fun(*args, **kwargs)
        return wrapper
    return decorator
<|endoftext|>"""Subset of inspect module from upstream python

We use this instead of upstream because upstream inspect is slow to import, and
significantly contributes to numpy import times. Importing this copy has almost
no overhead.

"""
import types

__all__ = ['getargspec', 'formatargspec']

# ----------------------------------------------------------- type-checking
def ismethod(object):
    """Return true if the object is an instance method.

    Instance method objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this method was defined
        im_class        class object in which this method belongs
        im_func         function object containing implementation of method
        im_self         instance to which this method is bound, or None

    """
    return isinstance(object, types.MethodType)

def isfunction(object):
    """Return true if the object is a user-defined function.

    Function objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this function was defined
        func_code       code object containing compiled function bytecode
        func_defaults   tuple of any default values for arguments
        func_doc        (same as __doc__)
        func_globals    global namespace in which this function was defined
        func_name       (same as __name__)

    """
    return isinstance(object, types.FunctionType)

def iscode(object):
    """Return true if the object is a code object.

    Code objects provide these attributes:
        co_argcount     number of arguments (not including * or ** args)
        co_code         string of raw compiled bytecode
        co_consts       tuple of constants used in the bytecode
        co_filename     name of file in which this code object was created
        co_firstlineno  number of first line in Python source code
        co_flags        bitmap: 1=optimized | 2=newlocals | 4=*arg | 8=**arg
        co_lnotab       encoded mapping of line numbers to bytecode indices
        co_name         name with which this code object was defined
        co_names        tuple of names of local variables
        co_nlocals      number of local variables
        co_stacksize    virtual machine stack space required
        co_varnames     tuple of names of arguments and local variables

    """
    return isinstance(object, types.CodeType)

# ------------------------------------------------ argument list extraction
# These constants are from Python's compile.h.
CO_OPTIMIZED, CO_NEWLOCALS, CO_VARARGS, CO_VARKEYWORDS = 1, 2, 4, 8

def getargs(co):
    """Get information about the arguments accepted by a code object.

    Three things are returned: (args, varargs, varkw), where 'args' is
    a list of argument names (possibly containing nested lists), and
    'varargs' and 'varkw' are the names of the * and ** arguments or None.

    """

    if not iscode(co):
        raise TypeError('arg is not a code object')

    nargs = co.co_argcount
    names = co.co_varnames
    args = list(names[:nargs])

    # The following acrobatics are for anonymous (tuple) arguments.
    # Which we do not need to support, so remove to avoid importing
    # the dis module.
    for i in range(nargs):
        if args[i][:1] in ['', '.']:
            raise TypeError("tuple function arguments are not supported")
    varargs = None
    if co.co_flags & CO_VARARGS:
        varargs = co.co_varnames[nargs]
        nargs = nargs + 1
    varkw = None
    if co.co_flags & CO_VARKEYWORDS:
        varkw = co.co_varnames[nargs]
    return args, varargs, varkw

def getargspec(func):
    """Get the names and default values of a function's arguments.

    A tuple of four things is returned: (args, varargs, varkw, defaults).
    'args' is a list of the argument names (it may contain nested lists).
    'varargs' and 'varkw' are the names of the * and ** arguments or None.
    'defaults' is an n-tuple of the default values of the last n arguments.

    """

    if ismethod(func):
        func = func.__func__
    if not isfunction(func):
        raise TypeError('arg is not a Python function')
    args, varargs, varkw = getargs(func.__code__)
    return args, varargs, varkw, func.__defaults__

def getargvalues(frame):
    """Get information about arguments passed into a particular frame.

    A tuple of four things is returned: (args, varargs, varkw, locals).
    'args' is a list of the argument names (it may contain nested lists).
    'varargs' and 'varkw' are the names of the * and ** arguments or None.
    'locals' is the locals dictionary of the given frame.

    """
    args, varargs, varkw = getargs(frame.f_code)
    return args, varargs, varkw, frame.f_locals

def joinseq(seq):
    if len(seq) == 1:
        return '(' + seq[0] + ',)'
    else:
        return '(' + ', '.join(seq) + ')'

def strseq(object, convert, join=joinseq):
    """Recursively walk a sequence, stringifying each element.

    """
    if type(object) in [list, tuple]:
        return join([strseq(_o, convert, join) for _o in object])
    else:
        return convert(object)

def formatargspec(args, varargs=None, varkw=None, defaults=None,
                  formatarg=str,
                  formatvarargs=lambda name: '*' + name,
                  formatvarkw=lambda name: '**' + name,
                  formatvalue=lambda value: '=' + repr(value),
                  join=joinseq):
    """Format an argument spec from the 4 values returned by getargspec.

    The first four arguments are (args, varargs, varkw, defaults).  The
    other four arguments are the corresponding optional formatting functions
    that are called to turn names and values into strings.  The ninth
    argument is an optional function to format the sequence of arguments.

    """
    specs = []
    if defaults:
        firstdefault = len(args) - len(defaults)
    for i in range(len(args)):
        spec = strseq(args[i], formatarg, join)
        if defaults and i >= firstdefault:
            spec = spec + formatvalue(defaults[i - firstdefault])
        specs.append(spec)
    if varargs is not None:
        specs.append(formatvarargs(varargs))
    if varkw is not None:
        specs.append(formatvarkw(varkw))
    return '(' + ', '.join(specs) + ')'

def formatargvalues(args, varargs, varkw, locals,
                    formatarg=str,
                    formatvarargs=lambda name: '*' + name,
                    formatvarkw=lambda name: '**' + name,
                    formatvalue=lambda value: '=' + repr(value),
                    join=joinseq):
    """Format an argument spec from the 4 values returned by getargvalues.

    The first four arguments are (args, varargs, varkw, locals).  The
    next four arguments are the corresponding optional formatting functions
    that are called to turn names and values into strings.  The ninth
    argument is an optional function to format the sequence of arguments.

    """
    def convert(name, locals=locals,
                formatarg=formatarg, formatvalue=formatvalue):
        return formatarg(name) + formatvalue(locals[name])
    specs = [strseq(arg, convert, join) for arg in args]

    if varargs:
        specs.append(formatvarargs(varargs) + formatvalue(locals[varargs]))
    if varkw:
        specs.append(formatvarkw(varkw) + formatvalue(locals[varkw]))
    return '(' + ', '.join(specs) + ')'
<|endoftext|>"""
A set of methods retained from np.compat module that
are still used across codebase.
"""

__all__ = ["asunicode", "asbytes"]


def asunicode(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')
<|endoftext|>"""Automatically adapted for numpy Sep 19, 2005 by convertcode.py

"""
import functools

__all__ = ['iscomplexobj', 'isrealobj', 'imag', 'iscomplex',
           'isreal', 'nan_to_num', 'real', 'real_if_close',
           'typename', 'mintypecode',
           'common_type']

from .._utils import set_module
import numpy._core.numeric as _nx
from numpy._core.numeric import asarray, asanyarray, isnan, zeros
from numpy._core import overrides, getlimits
from ._ufunclike_impl import isneginf, isposinf


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


_typecodes_by_elsize = 'GDFgdfQqLlIiHhBb?'


@set_module('numpy')
def mintypecode(typechars, typeset='GDFgdf', default='d'):
    """
    Return the character for the minimum-size type to which given types can
    be safely cast.

    The returned type character must represent the smallest size dtype such
    that an array of the returned type can handle the data from an array of
    all types in `typechars` (or if `typechars` is an array, then its
    dtype.char).

    Parameters
    ----------
    typechars : list of str or array_like
        If a list of strings, each string should represent a dtype.
        If array_like, the character representation of the array dtype is used.
    typeset : str or list of str, optional
        The set of characters that the returned character is chosen from.
        The default set is 'GDFgdf'.
    default : str, optional
        The default character, this is returned if none of the characters in
        `typechars` matches a character in `typeset`.

    Returns
    -------
    typechar : str
        The character representing the minimum-size type that was found.

    See Also
    --------
    dtype

    Examples
    --------
    >>> import numpy as np
    >>> np.mintypecode(['d', 'f', 'S'])
    'd'
    >>> x = np.array([1.1, 2-3.j])
    >>> np.mintypecode(x)
    'D'

    >>> np.mintypecode('abceh', default='G')
    'G'

    """
    typecodes = ((isinstance(t, str) and t) or asarray(t).dtype.char
                 for t in typechars)
    intersection = set(t for t in typecodes if t in typeset)
    if not intersection:
        return default
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    return min(intersection, key=_typecodes_by_elsize.index)


def _real_dispatcher(val):
    return (val,)


@array_function_dispatch(_real_dispatcher)
def real(val):
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The real component of the complex argument. If `val` is real, the type
        of `val` is used for the output.  If `val` has complex elements, the
        returned type is float.

    See Also
    --------
    real_if_close, imag, angle

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1+2j, 3+4j, 5+6j])
    >>> a.real
    array([1.,  3.,  5.])
    >>> a.real = 9
    >>> a
    array([9.+2.j,  9.+4.j,  9.+6.j])
    >>> a.real = np.array([9, 8, 7])
    >>> a
    array([9.+2.j,  8.+4.j,  7.+6.j])
    >>> np.real(1 + 1j)
    1.0

    """
    try:
        return val.real
    except AttributeError:
        return asanyarray(val).real


def _imag_dispatcher(val):
    return (val,)


@array_function_dispatch(_imag_dispatcher)
def imag(val):
    """
    Return the imaginary part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The imaginary component of the complex argument. If `val` is real,
        the type of `val` is used for the output.  If `val` has complex
        elements, the returned type is float.

    See Also
    --------
    real, angle, real_if_close

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1+2j, 3+4j, 5+6j])
    >>> a.imag
    array([2.,  4.,  6.])
    >>> a.imag = np.array([8, 10, 12])
    >>> a
    array([1. +8.j,  3.+10.j,  5.+12.j])
    >>> np.imag(1 + 1j)
    1.0

    """
    try:
        return val.imag
    except AttributeError:
        return asanyarray(val).imag


def _is_type_dispatcher(x):
    return (x,)


@array_function_dispatch(_is_type_dispatcher)
def iscomplex(x):
    """
    Returns a bool array, where True if input element is complex.

    What is tested is whether the input has a non-zero imaginary part, not if
    the input type is complex.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray of bools
        Output array.

    See Also
    --------
    isreal
    iscomplexobj : Return True if x is a complex type or an array of complex
                   numbers.

    Examples
    --------
