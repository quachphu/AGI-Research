        return 0
    return get_kind(var) == '32'


def iscomplexarray(var):
    return isarray(var) and \
           var.get('typespec') in ['complex', 'double complex']


def isint1array(var):
    return isarray(var) and var.get('typespec') == 'integer' \
        and get_kind(var) == '1'


def isunsigned_chararray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-1'


def isunsigned_shortarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-2'


def isunsignedarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-4'


def isunsigned_long_longarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '-8'


def issigned_chararray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '1'


def issigned_shortarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '2'


def issigned_array(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '4'


def issigned_long_longarray(var):
    return isarray(var) and var.get('typespec') in ['integer', 'logical']\
        and get_kind(var) == '8'


def isallocatable(var):
    return 'attrspec' in var and 'allocatable' in var['attrspec']


def ismutable(var):
    return not ('dimension' not in var or isstring(var))


def ismoduleroutine(rout):
    return 'modulename' in rout


def ismodule(rout):
    return 'block' in rout and 'module' == rout['block']


def isfunction(rout):
    return 'block' in rout and 'function' == rout['block']


def isfunction_wrap(rout):
    if isintent_c(rout):
        return 0
    return wrapfuncs and isfunction(rout) and (not isexternal(rout))


def issubroutine(rout):
    return 'block' in rout and 'subroutine' == rout['block']


def issubroutine_wrap(rout):
    if isintent_c(rout):
        return 0
    return issubroutine(rout) and hasassumedshape(rout)

def isattr_value(var):
    return 'value' in var.get('attrspec', [])


def hasassumedshape(rout):
    if rout.get('hasassumedshape'):
        return True
    for a in rout['args']:
        for d in rout['vars'].get(a, {}).get('dimension', []):
            if d == ':':
                rout['hasassumedshape'] = True
                return True
    return False


def requiresf90wrapper(rout):
    return ismoduleroutine(rout) or hasassumedshape(rout)


def isroutine(rout):
    return isfunction(rout) or issubroutine(rout)


def islogicalfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islogical(rout['vars'][a])
    return 0


def islong_longfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islong_long(rout['vars'][a])
    return 0


def islong_doublefunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islong_double(rout['vars'][a])
    return 0


def iscomplexfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return iscomplex(rout['vars'][a])
    return 0


def iscomplexfunction_warn(rout):
    if iscomplexfunction(rout):
        outmess("""\
    **************************************************************
        Warning: code with a function returning complex value
        may not work correctly with your Fortran compiler.
        When using GNU gcc/g77 compilers, codes should work
        correctly for callbacks with:
        f2py -c -DF2PY_CB_RETURNCOMPLEX
    **************************************************************\n""")
        return 1
    return 0


def isstringfunction(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return isstring(rout['vars'][a])
    return 0


def hasexternals(rout):
    return 'externals' in rout and rout['externals']


def isthreadsafe(rout):
    return 'f2pyenhancements' in rout and \
           'threadsafe' in rout['f2pyenhancements']


def hasvariables(rout):
    return 'vars' in rout and rout['vars']


def isoptional(var):
    return ('attrspec' in var and 'optional' in var['attrspec'] and
            'required' not in var['attrspec']) and isintent_nothide(var)


def isexternal(var):
    return 'attrspec' in var and 'external' in var['attrspec']


def getdimension(var):
    dimpattern = r"\((.*?)\)"
    if 'attrspec' in var.keys():
        if any('dimension' in s for s in var['attrspec']):
            return [re.findall(dimpattern, v) for v in var['attrspec']][0]


def isrequired(var):
    return not isoptional(var) and isintent_nothide(var)


def iscstyledirective(f2py_line):
    directives = {"callstatement", "callprotoargument", "pymethoddef"}
    return any(directive in f2py_line.lower() for directive in directives)


def isintent_in(var):
    if 'intent' not in var:
        return 1
    if 'hide' in var['intent']:
        return 0
    if 'inplace' in var['intent']:
        return 0
    if 'in' in var['intent']:
        return 1
    if 'out' in var['intent']:
        return 0
    if 'inout' in var['intent']:
        return 0
    if 'outin' in var['intent']:
        return 0
    return 1


def isintent_inout(var):
    return ('intent' in var and ('inout' in var['intent'] or
            'outin' in var['intent']) and 'in' not in var['intent'] and
            'hide' not in var['intent'] and 'inplace' not in var['intent'])


def isintent_out(var):
    return 'out' in var.get('intent', [])


def isintent_hide(var):
    return ('intent' in var and ('hide' in var['intent'] or
            ('out' in var['intent'] and 'in' not in var['intent'] and
                (not l_or(isintent_inout, isintent_inplace)(var)))))


def isintent_nothide(var):
    return not isintent_hide(var)


def isintent_c(var):
    return 'c' in var.get('intent', [])


def isintent_cache(var):
    return 'cache' in var.get('intent', [])


def isintent_copy(var):
    return 'copy' in var.get('intent', [])


def isintent_overwrite(var):
    return 'overwrite' in var.get('intent', [])


def isintent_callback(var):
    return 'callback' in var.get('intent', [])


def isintent_inplace(var):
    return 'inplace' in var.get('intent', [])


def isintent_aux(var):
    return 'aux' in var.get('intent', [])


def isintent_aligned4(var):
    return 'aligned4' in var.get('intent', [])


def isintent_aligned8(var):
    return 'aligned8' in var.get('intent', [])


def isintent_aligned16(var):
    return 'aligned16' in var.get('intent', [])


isintent_dict = {isintent_in: 'INTENT_IN', isintent_inout: 'INTENT_INOUT',
                 isintent_out: 'INTENT_OUT', isintent_hide: 'INTENT_HIDE',
                 isintent_cache: 'INTENT_CACHE',
                 isintent_c: 'INTENT_C', isoptional: 'OPTIONAL',
                 isintent_inplace: 'INTENT_INPLACE',
                 isintent_aligned4: 'INTENT_ALIGNED4',
                 isintent_aligned8: 'INTENT_ALIGNED8',
                 isintent_aligned16: 'INTENT_ALIGNED16',
                 }


def isprivate(var):
    return 'attrspec' in var and 'private' in var['attrspec']


def isvariable(var):
    # heuristic to find public/private declarations of filtered subroutines
    if len(var) == 1 and 'attrspec' in var and \
            var['attrspec'][0] in ('public', 'private'):
        is_var = False
    else:
        is_var = True
    return is_var

def hasinitvalue(var):
    return '=' in var


def hasinitvalueasstring(var):
    if not hasinitvalue(var):
        return 0
    return var['='][0] in ['"', "'"]


def hasnote(var):
    return 'note' in var


def hasresultnote(rout):
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return hasnote(rout['vars'][a])
    return 0


def hascommon(rout):
    return 'common' in rout


def containscommon(rout):
    if hascommon(rout):
        return 1
    if hasbody(rout):
        for b in rout['body']:
            if containscommon(b):
                return 1
    return 0


def containsmodule(block):
    if ismodule(block):
        return 1
    if not hasbody(block):
        return 0
    for b in block['body']:
        if containsmodule(b):
            return 1
    return 0


def hasbody(rout):
    return 'body' in rout


def hascallstatement(rout):
    return getcallstatement(rout) is not None


def istrue(var):
    return 1


def isfalse(var):
    return 0


class F2PYError(Exception):
    pass


class throw_error:

    def __init__(self, mess):
        self.mess = mess

    def __call__(self, var):
        mess = '\n\n  var = %s\n  Message: %s\n' % (var, self.mess)
        raise F2PYError(mess)


def l_and(*f):
    l1, l2 = 'lambda v', []
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % (i))
    return eval('%s:%s' % (l1, ' and '.join(l2)))


def l_or(*f):
    l1, l2 = 'lambda v', []
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % (i))
    return eval('%s:%s' % (l1, ' or '.join(l2)))


def l_not(f):
    return eval('lambda v,f=f:not f(v)')


def isdummyroutine(rout):
    try:
        return rout['f2pyenhancements']['fortranname'] == ''
    except KeyError:
        return 0


def getfortranname(rout):
    try:
        name = rout['f2pyenhancements']['fortranname']
        if name == '':
            raise KeyError
        if not name:
            errmess('Failed to use fortranname from %s\n' %
                    (rout['f2pyenhancements']))
            raise KeyError
    except KeyError:
        name = rout['name']
    return name


def getmultilineblock(rout, blockname, comment=1, counter=0):
    try:
        r = rout['f2pyenhancements'].get(blockname)
    except KeyError:
        return
    if not r:
        return
    if counter > 0 and isinstance(r, str):
        return
    if isinstance(r, list):
        if counter >= len(r):
            return
        r = r[counter]
    if r[:3] == "'''":
        if comment:
            r = '\t/* start ' + blockname + \
                ' multiline (' + repr(counter) + ') */\n' + r[3:]
        else:
            r = r[3:]
        if r[-3:] == "'''":
            if comment:
                r = r[:-3] + '\n\t/* end multiline (' + repr(counter) + ')*/'
            else:
                r = r[:-3]
        else:
            errmess("%s multiline block should end with `'''`: %s\n"
                    % (blockname, repr(r)))
    return r


def getcallstatement(rout):
    return getmultilineblock(rout, 'callstatement')


def getcallprotoargument(rout, cb_map={}):
    r = getmultilineblock(rout, 'callprotoargument', comment=0)
    if r:
        return r
    if hascallstatement(rout):
        outmess(
            'warning: callstatement is defined without callprotoargument\n')
        return
    from .capi_maps import getctype
    arg_types, arg_types2 = [], []
    if l_and(isstringfunction, l_not(isfunction_wrap))(rout):
        arg_types.extend(['char*', 'size_t'])
    for n in rout['args']:
        var = rout['vars'][n]
        if isintent_callback(var):
            continue
        if n in cb_map:
            ctype = cb_map[n] + '_typedef'
        else:
            ctype = getctype(var)
            if l_and(isintent_c, l_or(isscalar, iscomplex))(var):
                pass
            elif isstring(var):
                pass
            else:
                if not isattr_value(var):
                    ctype = ctype + '*'
            if (isstring(var)
                 or isarrayofstrings(var)  # obsolete?
                 or isstringarray(var)):
                arg_types2.append('size_t')
        arg_types.append(ctype)

    proto_args = ','.join(arg_types + arg_types2)
    if not proto_args:
        proto_args = 'void'
    return proto_args


def getusercode(rout):
    return getmultilineblock(rout, 'usercode')


def getusercode1(rout):
    return getmultilineblock(rout, 'usercode', counter=1)


def getpymethoddef(rout):
    return getmultilineblock(rout, 'pymethoddef')


def getargs(rout):
    sortargs, args = [], []
    if 'args' in rout:
        args = rout['args']
        if 'sortvars' in rout:
            for a in rout['sortvars']:
                if a in args:
                    sortargs.append(a)
            for a in args:
                if a not in sortargs:
                    sortargs.append(a)
        else:
            sortargs = rout['args']
    return args, sortargs


def getargs2(rout):
    sortargs, args = [], rout.get('args', [])
    auxvars = [a for a in rout['vars'].keys() if isintent_aux(rout['vars'][a])
               and a not in args]
    args = auxvars + args
    if 'sortvars' in rout:
        for a in rout['sortvars']:
            if a in args:
                sortargs.append(a)
        for a in args:
            if a not in sortargs:
                sortargs.append(a)
    else:
        sortargs = auxvars + rout['args']
    return args, sortargs


def getrestdoc(rout):
    if 'f2pymultilines' not in rout:
        return None
    k = None
    if rout['block'] == 'python module':
        k = rout['block'], rout['name']
    return rout['f2pymultilines'].get(k, None)


def gentitle(name):
    ln = (80 - len(name) - 6) // 2
    return '/*%s %s %s*/' % (ln * '*', name, ln * '*')


def flatlist(lst):
    if isinstance(lst, list):
        return reduce(lambda x, y, f=flatlist: x + f(y), lst, [])
    return [lst]


def stripcomma(s):
    if s and s[-1] == ',':
        return s[:-1]
    return s


def replace(str, d, defaultsep=''):
    if isinstance(d, list):
        return [replace(str, _m, defaultsep) for _m in d]
    if isinstance(str, list):
        return [replace(_m, d, defaultsep) for _m in str]
    for k in 2 * list(d.keys()):
        if k == 'separatorsfor':
            continue
        if 'separatorsfor' in d and k in d['separatorsfor']:
            sep = d['separatorsfor'][k]
        else:
            sep = defaultsep
        if isinstance(d[k], list):
            str = str.replace('#%s#' % (k), sep.join(flatlist(d[k])))
        else:
            str = str.replace('#%s#' % (k), d[k])
    return str


def dictappend(rd, ar):
    if isinstance(ar, list):
        for a in ar:
            rd = dictappend(rd, a)
        return rd
    for k in ar.keys():
        if k[0] == '_':
            continue
        if k in rd:
            if isinstance(rd[k], str):
                rd[k] = [rd[k]]
            if isinstance(rd[k], list):
                if isinstance(ar[k], list):
                    rd[k] = rd[k] + ar[k]
                else:
                    rd[k].append(ar[k])
            elif isinstance(rd[k], dict):
                if isinstance(ar[k], dict):
                    if k == 'separatorsfor':
                        for k1 in ar[k].keys():
                            if k1 not in rd[k]:
                                rd[k][k1] = ar[k][k1]
                    else:
                        rd[k] = dictappend(rd[k], ar[k])
        else:
            rd[k] = ar[k]
    return rd


def applyrules(rules, d, var={}):
    ret = {}
    if isinstance(rules, list):
        for r in rules:
            rr = applyrules(r, d, var)
            ret = dictappend(ret, rr)
            if '_break' in rr:
                break
        return ret
    if '_check' in rules and (not rules['_check'](var)):
        return ret
    if 'need' in rules:
        res = applyrules({'needs': rules['need']}, d, var)
        if 'needs' in res:
            cfuncs.append_needs(res['needs'])

    for k in rules.keys():
        if k == 'separatorsfor':
            ret[k] = rules[k]
            continue
        if isinstance(rules[k], str):
            ret[k] = replace(rules[k], d)
        elif isinstance(rules[k], list):
            ret[k] = []
            for i in rules[k]:
                ar = applyrules({k: i}, d, var)
                if k in ar:
                    ret[k].append(ar[k])
        elif k[0] == '_':
            continue
        elif isinstance(rules[k], dict):
            ret[k] = []
            for k1 in rules[k].keys():
                if isinstance(k1, types.FunctionType) and k1(var):
                    if isinstance(rules[k][k1], list):
                        for i in rules[k][k1]:
                            if isinstance(i, dict):
                                res = applyrules({'supertext': i}, d, var)
                                if 'supertext' in res:
                                    i = res['supertext']
                                else:
                                    i = ''
                            ret[k].append(replace(i, d))
                    else:
                        i = rules[k][k1]
                        if isinstance(i, dict):
                            res = applyrules({'supertext': i}, d)
                            if 'supertext' in res:
                                i = res['supertext']
                            else:
                                i = ''
                        ret[k].append(replace(i, d))
        else:
            errmess('applyrules: ignoring rule %s.\n' % repr(rules[k]))
        if isinstance(ret[k], list):
            if len(ret[k]) == 1:
                ret[k] = ret[k][0]
            if ret[k] == []:
                del ret[k]
    return ret

_f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                     re.I).match
_f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'
                                          r'__user__[\w_]*)', re.I).match

def get_f2py_modulename(source):
    name = None
    with open(source) as f:
        for line in f:
            m = _f2py_module_name_match(line)
            if m:
                if _f2py_user_module_name_match(line): # skip *__user__* names
                    continue
                name = m.group('name')
                break
    return name

def getuseblocks(pymod):
    all_uses = []
    for inner in pymod['body']:
        for modblock in inner['body']:
            if modblock.get('use'):
                all_uses.extend([x for x in modblock.get("use").keys() if "__" not in x])
    return all_uses

def process_f2cmap_dict(f2cmap_all, new_map, c2py_map, verbose = False):
    """
    Update the Fortran-to-C type mapping dictionary with new mappings and
    return a list of successfully mapped C types.

    This function integrates a new mapping dictionary into an existing
    Fortran-to-C type mapping dictionary. It ensures that all keys are in
    lowercase and validates new entries against a given C-to-Python mapping
    dictionary. Redefinitions and invalid entries are reported with a warning.

    Parameters
    ----------
    f2cmap_all : dict
        The existing Fortran-to-C type mapping dictionary that will be updated.
        It should be a dictionary of dictionaries where the main keys represent
        Fortran types and the nested dictionaries map Fortran type specifiers
        to corresponding C types.

    new_map : dict
        A dictionary containing new type mappings to be added to `f2cmap_all`.
        The structure should be similar to `f2cmap_all`, with keys representing
        Fortran types and values being dictionaries of type specifiers and their
        C type equivalents.

    c2py_map : dict
        A dictionary used for validating the C types in `new_map`. It maps C
        types to corresponding Python types and is used to ensure that the C
        types specified in `new_map` are valid.

    verbose : boolean
        A flag used to provide information about the types mapped

    Returns
    -------
    tuple of (dict, list)
        The updated Fortran-to-C type mapping dictionary and a list of
        successfully mapped C types.
    """
    f2cmap_mapped = []

    new_map_lower = {}
    for k, d1 in new_map.items():
        d1_lower = {k1.lower(): v1 for k1, v1 in d1.items()}
        new_map_lower[k.lower()] = d1_lower

    for k, d1 in new_map_lower.items():
        if k not in f2cmap_all:
            f2cmap_all[k] = {}

        for k1, v1 in d1.items():
            if v1 in c2py_map:
                if k1 in f2cmap_all[k]:
                    outmess(
                        "\tWarning: redefinition of {'%s':{'%s':'%s'->'%s'}}\n"
                        % (k, k1, f2cmap_all[k][k1], v1)
                    )
                f2cmap_all[k][k1] = v1
                if verbose:
                    outmess('\tMapping "%s(kind=%s)" to "%s"\n' % (k, k1, v1))
                f2cmap_mapped.append(v1)
            else:
                if verbose:
                    errmess(
                        "\tIgnoring map {'%s':{'%s':'%s'}}: '%s' must be in %s\n"
                        % (k, k1, v1, v1, list(c2py_map.keys()))
                    )

    return f2cmap_all, f2cmap_mapped
<|endoftext|>"""
C declarations, CPP macros, and C functions for f2py2e.
Only required declarations/macros/functions will be used.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
import sys
import copy

from . import __version__

f2py_version = __version__.version


def errmess(s: str) -> None:
    """
    Write an error message to stderr.

    This indirection is needed because sys.stderr might not always be available (see #26862).
    """
    if sys.stderr is not None:
        sys.stderr.write(s)

##################### Definitions ##################

outneeds = {'includes0': [], 'includes': [], 'typedefs': [], 'typedefs_generated': [],
            'userincludes': [],
            'cppmacros': [], 'cfuncs': [], 'callbacks': [], 'f90modhooks': [],
            'commonhooks': []}
needs = {}
includes0 = {'includes0': '/*need_includes0*/'}
includes = {'includes': '/*need_includes*/'}
userincludes = {'userincludes': '/*need_userincludes*/'}
typedefs = {'typedefs': '/*need_typedefs*/'}
typedefs_generated = {'typedefs_generated': '/*need_typedefs_generated*/'}
cppmacros = {'cppmacros': '/*need_cppmacros*/'}
cfuncs = {'cfuncs': '/*need_cfuncs*/'}
callbacks = {'callbacks': '/*need_callbacks*/'}
f90modhooks = {'f90modhooks': '/*need_f90modhooks*/',
               'initf90modhooksstatic': '/*initf90modhooksstatic*/',
               'initf90modhooksdynamic': '/*initf90modhooksdynamic*/',
               }
commonhooks = {'commonhooks': '/*need_commonhooks*/',
               'initcommonhooks': '/*need_initcommonhooks*/',
               }

############ Includes ###################

includes0['math.h'] = '#include <math.h>'
includes0['string.h'] = '#include <string.h>'
includes0['setjmp.h'] = '#include <setjmp.h>'

includes['arrayobject.h'] = '''#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API
#include "arrayobject.h"'''
includes['npy_math.h'] = '#include "numpy/npy_math.h"'

includes['arrayobject.h'] = '#include "fortranobject.h"'
includes['stdarg.h'] = '#include <stdarg.h>'

############# Type definitions ###############

typedefs['unsigned_char'] = 'typedef unsigned char unsigned_char;'
typedefs['unsigned_short'] = 'typedef unsigned short unsigned_short;'
typedefs['unsigned_long'] = 'typedef unsigned long unsigned_long;'
typedefs['signed_char'] = 'typedef signed char signed_char;'
typedefs['long_long'] = """
#if defined(NPY_OS_WIN32)
typedef __int64 long_long;
#else
typedef long long long_long;
typedef unsigned long long unsigned_long_long;
#endif
"""
typedefs['unsigned_long_long'] = """
#if defined(NPY_OS_WIN32)
typedef __uint64 long_long;
#else
typedef unsigned long long unsigned_long_long;
#endif
"""
typedefs['long_double'] = """
#ifndef _LONG_DOUBLE
typedef long double long_double;
#endif
"""
typedefs[
    'complex_long_double'] = 'typedef struct {long double r,i;} complex_long_double;'
typedefs['complex_float'] = 'typedef struct {float r,i;} complex_float;'
typedefs['complex_double'] = 'typedef struct {double r,i;} complex_double;'
typedefs['string'] = """typedef char * string;"""
typedefs['character'] = """typedef char character;"""


############### CPP macros ####################
cppmacros['CFUNCSMESS'] = """
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,\"debug-capi:\"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \\
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
    fprintf(stderr,\"\\n\");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif
"""
cppmacros['F_FUNC'] = """
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif
"""
cppmacros['F_WRAPPEDFUNC'] = """
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F
#else
#define F_WRAPPEDFUNC(f,F) _f2pywrap##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) _F2PYWRAP##F##_
#else
#define F_WRAPPEDFUNC(f,F) _f2pywrap##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F
#else
#define F_WRAPPEDFUNC(f,F) f2pywrap##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_WRAPPEDFUNC(f,F) F2PYWRAP##F##_
#else
#define F_WRAPPEDFUNC(f,F) f2pywrap##f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f##_,F##_)
#else
#define F_WRAPPEDFUNC_US(f,F) F_WRAPPEDFUNC(f,F)
#endif
"""
cppmacros['F_MODFUNC'] = """
#if defined(F90MOD2CCONV1) /*E.g. Compaq Fortran */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f
#else
#define F_MODFUNCNAME(m,f) $ ## m ## $ ## f ## _
#endif
#endif

#if defined(F90MOD2CCONV2) /*E.g. IBM XL Fortran, not tested though */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f
#else
#define F_MODFUNCNAME(m,f)  __ ## m ## _MOD_ ## f ## _
#endif
#endif

#if defined(F90MOD2CCONV3) /*E.g. MIPSPro Compilers */
#if defined(NO_APPEND_FORTRAN)
#define F_MODFUNCNAME(m,f)  f ## .in. ## m
#else
#define F_MODFUNCNAME(m,f)  f ## .in. ## m ## _
#endif
#endif
/*
#if defined(UPPERCASE_FORTRAN)
#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(M,F)
#else
#define F_MODFUNC(m,M,f,F) F_MODFUNCNAME(m,f)
#endif
*/

#define F_MODFUNC(m,f) (*(f2pymodstruct##m##.##f))
"""
cppmacros['SWAPUNSAFE'] = """
#define SWAP(a,b) (size_t)(a) = ((size_t)(a) ^ (size_t)(b));\\
 (size_t)(b) = ((size_t)(a) ^ (size_t)(b));\\
 (size_t)(a) = ((size_t)(a) ^ (size_t)(b))
"""
cppmacros['SWAP'] = """
#define SWAP(a,b,t) {\\
    t *c;\\
    c = a;\\
    a = b;\\
    b = c;}
"""
# cppmacros['ISCONTIGUOUS']='#define ISCONTIGUOUS(m) (PyArray_FLAGS(m) &
# NPY_ARRAY_C_CONTIGUOUS)'
cppmacros['PRINTPYOBJERR'] = """
#define PRINTPYOBJERR(obj)\\
    fprintf(stderr,\"#modulename#.error is related to \");\\
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\\
    fprintf(stderr,\"\\n\");
"""
cppmacros['MINMAX'] = """
#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif
"""
cppmacros['len..'] = """
/* See fortranobject.h for definitions. The macros here are provided for BC. */
#define rank f2py_rank
#define shape f2py_shape
#define fshape f2py_shape
#define len f2py_len
#define flen f2py_flen
#define slen f2py_slen
#define size f2py_size
"""
cppmacros['pyobj_from_char1'] = r"""
#define pyobj_from_char1(v) (PyLong_FromLong(v))
"""
cppmacros['pyobj_from_short1'] = r"""
#define pyobj_from_short1(v) (PyLong_FromLong(v))
"""
needs['pyobj_from_int1'] = ['signed_char']
cppmacros['pyobj_from_int1'] = r"""
#define pyobj_from_int1(v) (PyLong_FromLong(v))
"""
cppmacros['pyobj_from_long1'] = r"""
#define pyobj_from_long1(v) (PyLong_FromLong(v))
"""
needs['pyobj_from_long_long1'] = ['long_long']
cppmacros['pyobj_from_long_long1'] = """
#ifdef HAVE_LONG_LONG
#define pyobj_from_long_long1(v) (PyLong_FromLongLong(v))
#else
#warning HAVE_LONG_LONG is not available. Redefining pyobj_from_long_long.
#define pyobj_from_long_long1(v) (PyLong_FromLong(v))
#endif
"""
needs['pyobj_from_long_double1'] = ['long_double']
cppmacros['pyobj_from_long_double1'] = """
#define pyobj_from_long_double1(v) (PyFloat_FromDouble(v))"""
cppmacros['pyobj_from_double1'] = """
#define pyobj_from_double1(v) (PyFloat_FromDouble(v))"""
cppmacros['pyobj_from_float1'] = """
#define pyobj_from_float1(v) (PyFloat_FromDouble(v))"""
needs['pyobj_from_complex_long_double1'] = ['complex_long_double']
cppmacros['pyobj_from_complex_long_double1'] = """
#define pyobj_from_complex_long_double1(v) (PyComplex_FromDoubles(v.r,v.i))"""
needs['pyobj_from_complex_double1'] = ['complex_double']
cppmacros['pyobj_from_complex_double1'] = """
#define pyobj_from_complex_double1(v) (PyComplex_FromDoubles(v.r,v.i))"""
needs['pyobj_from_complex_float1'] = ['complex_float']
cppmacros['pyobj_from_complex_float1'] = """
#define pyobj_from_complex_float1(v) (PyComplex_FromDoubles(v.r,v.i))"""
needs['pyobj_from_string1'] = ['string']
cppmacros['pyobj_from_string1'] = """
#define pyobj_from_string1(v) (PyUnicode_FromString((char *)v))"""
needs['pyobj_from_string1size'] = ['string']
cppmacros['pyobj_from_string1size'] = """
#define pyobj_from_string1size(v,len) (PyUnicode_FromStringAndSize((char *)v, len))"""
needs['TRYPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']
cppmacros['TRYPYARRAYTEMPLATE'] = """
/* New SciPy */
#define TRYPYARRAYTEMPLATECHAR case NPY_STRING: *(char *)(PyArray_DATA(arr))=*v; break;
#define TRYPYARRAYTEMPLATELONG case NPY_LONG: *(long *)(PyArray_DATA(arr))=*v; break;
#define TRYPYARRAYTEMPLATEOBJECT case NPY_OBJECT: PyArray_SETITEM(arr,PyArray_DATA(arr),pyobj_from_ ## ctype ## 1(*v)); break;

#define TRYPYARRAYTEMPLATE(ctype,typecode) \\
        PyArrayObject *arr = NULL;\\
        if (!obj) return -2;\\
        if (!PyArray_Check(obj)) return -1;\\
        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
        if (PyArray_DESCR(arr)->type==typecode)  {*(ctype *)(PyArray_DATA(arr))=*v; return 1;}\\
        switch (PyArray_TYPE(arr)) {\\
                case NPY_DOUBLE: *(npy_double *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_INT: *(npy_int *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONG: *(npy_long *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_FLOAT: *(npy_float *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CDOUBLE: *(npy_double *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CFLOAT: *(npy_float *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=(*v!=0); break;\\
                case NPY_UBYTE: *(npy_ubyte *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_BYTE: *(npy_byte *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_SHORT: *(npy_short *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=*v; break;\\
                case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_ ## ctype ## 1(*v)); break;\\
        default: return -2;\\
        };\\
        return 1
"""

needs['TRYCOMPLEXPYARRAYTEMPLATE'] = ['PRINTPYOBJERR']
cppmacros['TRYCOMPLEXPYARRAYTEMPLATE'] = """
#define TRYCOMPLEXPYARRAYTEMPLATEOBJECT case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_complex_ ## ctype ## 1((*v))); break;
#define TRYCOMPLEXPYARRAYTEMPLATE(ctype,typecode)\\
        PyArrayObject *arr = NULL;\\
        if (!obj) return -2;\\
        if (!PyArray_Check(obj)) return -1;\\
        if (!(arr=(PyArrayObject *)obj)) {fprintf(stderr,\"TRYCOMPLEXPYARRAYTEMPLATE:\");PRINTPYOBJERR(obj);return 0;}\\
        if (PyArray_DESCR(arr)->type==typecode) {\\
            *(ctype *)(PyArray_DATA(arr))=(*v).r;\\
            *(ctype *)(PyArray_DATA(arr)+sizeof(ctype))=(*v).i;\\
            return 1;\\
        }\\
        switch (PyArray_TYPE(arr)) {\\
                case NPY_CDOUBLE: *(npy_double *)(PyArray_DATA(arr))=(*v).r;\\
                                  *(npy_double *)(PyArray_DATA(arr)+sizeof(npy_double))=(*v).i;\\
                                  break;\\
                case NPY_CFLOAT: *(npy_float *)(PyArray_DATA(arr))=(*v).r;\\
                                 *(npy_float *)(PyArray_DATA(arr)+sizeof(npy_float))=(*v).i;\\
                                 break;\\
                case NPY_DOUBLE: *(npy_double *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_LONG: *(npy_long *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_FLOAT: *(npy_float *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_INT: *(npy_int *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_SHORT: *(npy_short *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_UBYTE: *(npy_ubyte *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_BYTE: *(npy_byte *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_BOOL: *(npy_bool *)(PyArray_DATA(arr))=((*v).r!=0 && (*v).i!=0); break;\\
                case NPY_USHORT: *(npy_ushort *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_UINT: *(npy_uint *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_ULONG: *(npy_ulong *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_LONGLONG: *(npy_longlong *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_ULONGLONG: *(npy_ulonglong *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_LONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r; break;\\
                case NPY_CLONGDOUBLE: *(npy_longdouble *)(PyArray_DATA(arr))=(*v).r;\\
                                      *(npy_longdouble *)(PyArray_DATA(arr)+sizeof(npy_longdouble))=(*v).i;\\
                                      break;\\
                case NPY_OBJECT: PyArray_SETITEM(arr, PyArray_DATA(arr), pyobj_from_complex_ ## ctype ## 1((*v))); break;\\
                default: return -2;\\
        };\\
        return -1;
"""
# cppmacros['NUMFROMARROBJ']="""
# define NUMFROMARROBJ(typenum,ctype) \\
#     if (PyArray_Check(obj)) arr = (PyArrayObject *)obj;\\
#     else arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,typenum,0,0);\\
#     if (arr) {\\
#         if (PyArray_TYPE(arr)==NPY_OBJECT) {\\
#             if (!ctype ## _from_pyobj(v,(PyArray_DESCR(arr)->getitem)(PyArray_DATA(arr)),\"\"))\\
#             goto capi_fail;\\
#         } else {\\
#             (PyArray_DESCR(arr)->cast[typenum])(PyArray_DATA(arr),1,(char*)v,1,1);\\
#         }\\
#         if ((PyObject *)arr != obj) { Py_DECREF(arr); }\\
#         return 1;\\
#     }
# """
# XXX: Note that CNUMFROMARROBJ is identical with NUMFROMARROBJ
# cppmacros['CNUMFROMARROBJ']="""
# define CNUMFROMARROBJ(typenum,ctype) \\
#     if (PyArray_Check(obj)) arr = (PyArrayObject *)obj;\\
#     else arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,typenum,0,0);\\
#     if (arr) {\\
#         if (PyArray_TYPE(arr)==NPY_OBJECT) {\\
#             if (!ctype ## _from_pyobj(v,(PyArray_DESCR(arr)->getitem)(PyArray_DATA(arr)),\"\"))\\
#             goto capi_fail;\\
#         } else {\\
#             (PyArray_DESCR(arr)->cast[typenum])((void *)(PyArray_DATA(arr)),1,(void *)(v),1,1);\\
#         }\\
#         if ((PyObject *)arr != obj) { Py_DECREF(arr); }\\
#         return 1;\\
#     }
# """


needs['GETSTRFROMPYTUPLE'] = ['STRINGCOPYN', 'PRINTPYOBJERR']
cppmacros['GETSTRFROMPYTUPLE'] = """
#define GETSTRFROMPYTUPLE(tuple,index,str,len) {\\
        PyObject *rv_cb_str = PyTuple_GetItem((tuple),(index));\\
        if (rv_cb_str == NULL)\\
            goto capi_fail;\\
        if (PyBytes_Check(rv_cb_str)) {\\
            str[len-1]='\\0';\\
            STRINGCOPYN((str),PyBytes_AS_STRING((PyBytesObject*)rv_cb_str),(len));\\
        } else {\\
            PRINTPYOBJERR(rv_cb_str);\\
            PyErr_SetString(#modulename#_error,\"string object expected\");\\
            goto capi_fail;\\
        }\\
    }
"""
cppmacros['GETSCALARFROMPYTUPLE'] = """
#define GETSCALARFROMPYTUPLE(tuple,index,var,ctype,mess) {\\
        if ((capi_tmp = PyTuple_GetItem((tuple),(index)))==NULL) goto capi_fail;\\
        if (!(ctype ## _from_pyobj((var),capi_tmp,mess)))\\
            goto capi_fail;\\
    }
"""

cppmacros['FAILNULL'] = """\
#define FAILNULL(p) do {                                            \\
    if ((p) == NULL) {                                              \\
        PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \\
        goto capi_fail;                                             \\
    }                                                               \\
} while (0)
"""
needs['MEMCOPY'] = ['string.h', 'FAILNULL']
cppmacros['MEMCOPY'] = """
#define MEMCOPY(to,from,n)\\
    do { FAILNULL(to); FAILNULL(from); (void)memcpy(to,from,n); } while (0)
"""
cppmacros['STRINGMALLOC'] = """
#define STRINGMALLOC(str,len)\\
    if ((str = (string)malloc(len+1)) == NULL) {\\
        PyErr_SetString(PyExc_MemoryError, \"out of memory\");\\
        goto capi_fail;\\
    } else {\\
        (str)[len] = '\\0';\\
    }
"""
cppmacros['STRINGFREE'] = """
#define STRINGFREE(str) do {if (!(str == NULL)) free(str);} while (0)
"""
needs['STRINGPADN'] = ['string.h']
cppmacros['STRINGPADN'] = """
/*
STRINGPADN replaces null values with padding values from the right.

`to` must have size of at least N bytes.

If the `to[N-1]` has null value, then replace it and all the
preceding, nulls with the given padding.

STRINGPADN(to, N, PADDING, NULLVALUE) is an inverse operation.
*/
#define STRINGPADN(to, N, NULLVALUE, PADDING)                   \\
    do {                                                        \\
        int _m = (N);                                           \\
        char *_to = (to);                                       \\
        for (_m -= 1; _m >= 0 && _to[_m] == NULLVALUE; _m--) {  \\
             _to[_m] = PADDING;                                 \\
        }                                                       \\
    } while (0)
"""
needs['STRINGCOPYN'] = ['string.h', 'FAILNULL']
cppmacros['STRINGCOPYN'] = """
/*
STRINGCOPYN copies N bytes.

`to` and `from` buffers must have sizes of at least N bytes.
*/
#define STRINGCOPYN(to,from,N)                                  \\
    do {                                                        \\
        int _m = (N);                                           \\
        char *_to = (to);                                       \\
        char *_from = (from);                                   \\
        FAILNULL(_to); FAILNULL(_from);                         \\
        (void)strncpy(_to, _from, _m);             \\
    } while (0)
"""
needs['STRINGCOPY'] = ['string.h', 'FAILNULL']
cppmacros['STRINGCOPY'] = """
#define STRINGCOPY(to,from)\\
    do { FAILNULL(to); FAILNULL(from); (void)strcpy(to,from); } while (0)
"""
cppmacros['CHECKGENERIC'] = """
#define CHECKGENERIC(check,tcheck,name) \\
    if (!(check)) {\\
        PyErr_SetString(#modulename#_error,\"(\"tcheck\") failed for \"name);\\
        /*goto capi_fail;*/\\
    } else """
cppmacros['CHECKARRAY'] = """
#define CHECKARRAY(check,tcheck,name) \\
    if (!(check)) {\\
        PyErr_SetString(#modulename#_error,\"(\"tcheck\") failed for \"name);\\
        /*goto capi_fail;*/\\
    } else """
cppmacros['CHECKSTRING'] = """
#define CHECKSTRING(check,tcheck,name,show,var)\\
    if (!(check)) {\\
        char errstring[256];\\
        sprintf(errstring, \"%s: \"show, \"(\"tcheck\") failed for \"name, slen(var), var);\\
        PyErr_SetString(#modulename#_error, errstring);\\
        /*goto capi_fail;*/\\
    } else """
cppmacros['CHECKSCALAR'] = """
#define CHECKSCALAR(check,tcheck,name,show,var)\\
    if (!(check)) {\\
        char errstring[256];\\
        sprintf(errstring, \"%s: \"show, \"(\"tcheck\") failed for \"name, var);\\
        PyErr_SetString(#modulename#_error,errstring);\\
        /*goto capi_fail;*/\\
    } else """
# cppmacros['CHECKDIMS']="""
# define CHECKDIMS(dims,rank) \\
#     for (int i=0;i<(rank);i++)\\
#         if (dims[i]<0) {\\
#             fprintf(stderr,\"Unspecified array argument requires a complete dimension specification.\\n\");\\
#             goto capi_fail;\\
#         }
# """
cppmacros[
    'ARRSIZE'] = '#define ARRSIZE(dims,rank) (_PyArray_multiply_list(dims,rank))'
cppmacros['OLDPYNUM'] = """
#ifdef OLDPYNUM
#error You need to install NumPy version 0.13 or higher. See https://scipy.org/install.html
#endif
"""

# Defining the correct value to indicate thread-local storage in C without
# running a compile-time check (which we have no control over in generated
# code used outside of NumPy) is hard. Therefore we support overriding this
# via an external define - the f2py-using package can then use the same
# compile-time checks as we use for `NPY_TLS` when building NumPy (see
# scipy#21860 for an example of that).
#
# __STDC_NO_THREADS__ should not be coupled to the availability of _Thread_local.
# In case we get a bug report, guard it with __STDC_NO_THREADS__ after all.
#
# `thread_local` has become a keyword in C23, but don't try to use that yet
# (too new, doing so while C23 support is preliminary will likely cause more
#  problems than it solves).
#
# Note: do not try to use `threads.h`, its availability is very low
# *and* threads.h isn't actually used where `F2PY_THREAD_LOCAL_DECL` is
# in the generated code. See gh-27718 for more details.
cppmacros["F2PY_THREAD_LOCAL_DECL"] = """
#ifndef F2PY_THREAD_LOCAL_DECL
#if defined(_MSC_VER)
#define F2PY_THREAD_LOCAL_DECL __declspec(thread)
#elif defined(NPY_OS_MINGW)
#define F2PY_THREAD_LOCAL_DECL __thread
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
#define F2PY_THREAD_LOCAL_DECL _Thread_local
#elif defined(__GNUC__) \\
      && (__GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 4)))
#define F2PY_THREAD_LOCAL_DECL __thread
#endif
#endif
"""
################# C functions ###############

cfuncs['calcarrindex'] = """
static int calcarrindex(int *i,PyArrayObject *arr) {
    int k,ii = i[0];
    for (k=1; k < PyArray_NDIM(arr); k++)
        ii += (ii*(PyArray_DIM(arr,k) - 1)+i[k]); /* assuming contiguous arr */
    return ii;
}"""
cfuncs['calcarrindextr'] = """
static int calcarrindextr(int *i,PyArrayObject *arr) {
    int k,ii = i[PyArray_NDIM(arr)-1];
    for (k=1; k < PyArray_NDIM(arr); k++)
        ii += (ii*(PyArray_DIM(arr,PyArray_NDIM(arr)-k-1) - 1)+i[PyArray_NDIM(arr)-k-1]); /* assuming contiguous arr */
    return ii;
}"""
cfuncs['forcomb'] = """
static struct { int nd;npy_intp *d;int *i,*i_tr,tr; } forcombcache;
static int initforcomb(npy_intp *dims,int nd,int tr) {
  int k;
  if (dims==NULL) return 0;
  if (nd<0) return 0;
  forcombcache.nd = nd;
  forcombcache.d = dims;
  forcombcache.tr = tr;
  if ((forcombcache.i = (int *)malloc(sizeof(int)*nd))==NULL) return 0;
  if ((forcombcache.i_tr = (int *)malloc(sizeof(int)*nd))==NULL) return 0;
  for (k=1;k<nd;k++) {
    forcombcache.i[k] = forcombcache.i_tr[nd-k-1] = 0;
  }
  forcombcache.i[0] = forcombcache.i_tr[nd-1] = -1;
  return 1;
}
static int *nextforcomb(void) {
  int j,*i,*i_tr,k;
  int nd=forcombcache.nd;
  if ((i=forcombcache.i) == NULL) return NULL;
  if ((i_tr=forcombcache.i_tr) == NULL) return NULL;
  if (forcombcache.d == NULL) return NULL;
  i[0]++;
  if (i[0]==forcombcache.d[0]) {
    j=1;
    while ((j<nd) && (i[j]==forcombcache.d[j]-1)) j++;
    if (j==nd) {
      free(i);
      free(i_tr);
      return NULL;
    }
    for (k=0;k<j;k++) i[k] = i_tr[nd-k-1] = 0;
    i[j]++;
    i_tr[nd-j-1]++;
  } else
    i_tr[nd-1]++;
  if (forcombcache.tr) return i_tr;
  return i;
}"""
needs['try_pyarr_from_string'] = ['STRINGCOPYN', 'PRINTPYOBJERR', 'string']
cfuncs['try_pyarr_from_string'] = """
/*
  try_pyarr_from_string copies str[:len(obj)] to the data of an `ndarray`.

  If obj is an `ndarray`, it is assumed to be contiguous.

  If the specified len==-1, str must be null-terminated.
*/
static int try_pyarr_from_string(PyObject *obj,
                                 const string str, const int len) {
#ifdef DEBUGCFUNCS
fprintf(stderr, "try_pyarr_from_string(str='%s', len=%d, obj=%p)\\n",
        (char*)str,len, obj);
#endif
    if (!obj) return -2; /* Object missing */
    if (obj == Py_None) return -1; /* None */
    if (!PyArray_Check(obj)) goto capi_fail; /* not an ndarray */
    if (PyArray_Check(obj)) {
        PyArrayObject *arr = (PyArrayObject *)obj;
        assert(ISCONTIGUOUS(arr));
        string buf = PyArray_DATA(arr);
        npy_intp n = len;
        if (n == -1) {
            /* Assuming null-terminated str. */
            n = strlen(str);
        }
        if (n > PyArray_NBYTES(arr)) {
            n = PyArray_NBYTES(arr);
        }
        STRINGCOPYN(buf, str, n);
        return 1;
    }
capi_fail:
    PRINTPYOBJERR(obj);
    PyErr_SetString(#modulename#_error, \"try_pyarr_from_string failed\");
    return 0;
}
"""
needs['string_from_pyobj'] = ['string', 'STRINGMALLOC', 'STRINGCOPYN']
cfuncs['string_from_pyobj'] = """
/*
  Create a new string buffer `str` of at most length `len` from a
  Python string-like object `obj`.

  The string buffer has given size (len) or the size of inistr when len==-1.

  The string buffer is padded with blanks: in Fortran, trailing blanks
  are insignificant contrary to C nulls.
 */
static int
string_from_pyobj(string *str, int *len, const string inistr, PyObject *obj,
                  const char *errmess)
{
    PyObject *tmp = NULL;
    string buf = NULL;
    npy_intp n = -1;
#ifdef DEBUGCFUNCS
fprintf(stderr,\"string_from_pyobj(str='%s',len=%d,inistr='%s',obj=%p)\\n\",
               (char*)str, *len, (char *)inistr, obj);
#endif
    if (obj == Py_None) {
        n = strlen(inistr);
        buf = inistr;
    }
    else if (PyArray_Check(obj)) {
        PyArrayObject *arr = (PyArrayObject *)obj;
        if (!ISCONTIGUOUS(arr)) {
            PyErr_SetString(PyExc_ValueError,
                            \"array object is non-contiguous.\");
            goto capi_fail;
        }
        n = PyArray_NBYTES(arr);
        buf = PyArray_DATA(arr);
        n = strnlen(buf, n);
    }
    else {
        if (PyBytes_Check(obj)) {
            tmp = obj;
            Py_INCREF(tmp);
        }
        else if (PyUnicode_Check(obj)) {
            tmp = PyUnicode_AsASCIIString(obj);
        }
        else {
            PyObject *tmp2;
            tmp2 = PyObject_Str(obj);
            if (tmp2) {
                tmp = PyUnicode_AsASCIIString(tmp2);
                Py_DECREF(tmp2);
            }
            else {
                tmp = NULL;
            }
        }
        if (tmp == NULL) goto capi_fail;
        n = PyBytes_GET_SIZE(tmp);
        buf = PyBytes_AS_STRING(tmp);
    }
    if (*len == -1) {
        /* TODO: change the type of `len` so that we can remove this */
        if (n > NPY_MAX_INT) {
            PyErr_SetString(PyExc_OverflowError,
                            "object too large for a 32-bit int");
            goto capi_fail;
        }
        *len = n;
    }
    else if (*len < n) {
        /* discard the last (len-n) bytes of input buf */
        n = *len;
    }
    if (n < 0 || *len < 0 || buf == NULL) {
        goto capi_fail;
    }
    STRINGMALLOC(*str, *len);  // *str is allocated with size (*len + 1)
    if (n < *len) {
        /*
          Pad fixed-width string with nulls. The caller will replace
          nulls with blanks when the corresponding argument is not
          intent(c).
        */
        memset(*str + n, '\\0', *len - n);
    }
    STRINGCOPYN(*str, buf, n);
    Py_XDECREF(tmp);
    return 1;
capi_fail:
    Py_XDECREF(tmp);
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}
"""

cfuncs['character_from_pyobj'] = """
static int
character_from_pyobj(character* v, PyObject *obj, const char *errmess) {
    if (PyBytes_Check(obj)) {
        /* empty bytes has trailing null, so dereferencing is always safe */
        *v = PyBytes_AS_STRING(obj)[0];
        return 1;
    } else if (PyUnicode_Check(obj)) {
        PyObject* tmp = PyUnicode_AsASCIIString(obj);
        if (tmp != NULL) {
            *v = PyBytes_AS_STRING(tmp)[0];
            Py_DECREF(tmp);
            return 1;
        }
    } else if (PyArray_Check(obj)) {
        PyArrayObject* arr = (PyArrayObject*)obj;
        if (F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr)) {
            *v = PyArray_BYTES(arr)[0];
            return 1;
        } else if (F2PY_IS_UNICODE_ARRAY(arr)) {
            // TODO: update when numpy will support 1-byte and
            // 2-byte unicode dtypes
            PyObject* tmp = PyUnicode_FromKindAndData(
                              PyUnicode_4BYTE_KIND,
                              PyArray_BYTES(arr),
                              (PyArray_NBYTES(arr)>0?1:0));
            if (tmp != NULL) {
                if (character_from_pyobj(v, tmp, errmess)) {
                    Py_DECREF(tmp);
                    return 1;
                }
                Py_DECREF(tmp);
            }
        }
    } else if (PySequence_Check(obj)) {
        PyObject* tmp = PySequence_GetItem(obj,0);
        if (tmp != NULL) {
            if (character_from_pyobj(v, tmp, errmess)) {
                Py_DECREF(tmp);
                return 1;
            }
            Py_DECREF(tmp);
        }
    }
    {
        /* TODO: This error (and most other) error handling needs cleaning. */
        char mess[F2PY_MESSAGE_BUFFER_SIZE];
        strcpy(mess, errmess);
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = PyExc_TypeError;
            Py_INCREF(err);
        }
        else {
            Py_INCREF(err);
            PyErr_Clear();
        }
        sprintf(mess + strlen(mess),
                " -- expected str|bytes|sequence-of-str-or-bytes, got ");
        f2py_describe(obj, mess + strlen(mess));
        PyErr_SetString(err, mess);
        Py_DECREF(err);
    }
    return 0;
}
"""

# TODO: These should be dynamically generated, too many mapped to int things,
# see note in _isocbind.py
needs['char_from_pyobj'] = ['int_from_pyobj']
cfuncs['char_from_pyobj'] = """
static int
char_from_pyobj(char* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (char)i;
        return 1;
    }
    return 0;
}
"""


needs['signed_char_from_pyobj'] = ['int_from_pyobj', 'signed_char']
cfuncs['signed_char_from_pyobj'] = """
static int
signed_char_from_pyobj(signed_char* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (signed_char)i;
        return 1;
    }
    return 0;
}
"""


needs['short_from_pyobj'] = ['int_from_pyobj']
cfuncs['short_from_pyobj'] = """
static int
short_from_pyobj(short* v, PyObject *obj, const char *errmess) {
    int i = 0;
    if (int_from_pyobj(&i, obj, errmess)) {
        *v = (short)i;
        return 1;
    }
    return 0;
}
"""


cfuncs['int_from_pyobj'] = """
static int
int_from_pyobj(int* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;

    if (PyLong_Check(obj)) {
        *v = Npy__PyLong_AsInt(obj);
        return !(*v == -1 && PyErr_Occurred());
    }

    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = Npy__PyLong_AsInt(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,\"real\");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (int_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }

    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}
"""


cfuncs['long_from_pyobj'] = """
static int
long_from_pyobj(long* v, PyObject *obj, const char *errmess) {
    PyObject* tmp = NULL;

    if (PyLong_Check(obj)) {
        *v = PyLong_AsLong(obj);
        return !(*v == -1 && PyErr_Occurred());
    }

    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = PyLong_AsLong(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,\"real\");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (long_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}
"""


needs['long_long_from_pyobj'] = ['long_long']
cfuncs['long_long_from_pyobj'] = """
static int
long_long_from_pyobj(long_long* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;

    if (PyLong_Check(obj)) {
        *v = PyLong_AsLongLong(obj);
        return !(*v == -1 && PyErr_Occurred());
    }

    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = PyLong_AsLongLong(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,\"real\");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (long_long_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = #modulename#_error;
        }
        PyErr_SetString(err,errmess);
    }
    return 0;
}
"""


needs['long_double_from_pyobj'] = ['double_from_pyobj', 'long_double']
cfuncs['long_double_from_pyobj'] = """
static int
long_double_from_pyobj(long_double* v, PyObject *obj, const char *errmess)
{
    double d=0;
    if (PyArray_CheckScalar(obj)){
        if PyArray_IsScalar(obj, LongDouble) {
            PyArray_ScalarAsCtype(obj, v);
            return 1;
        }
        else if (PyArray_Check(obj) && PyArray_TYPE(obj) == NPY_LONGDOUBLE) {
            (*v) = *((npy_longdouble *)PyArray_DATA(obj));
            return 1;
        }
    }
    if (double_from_pyobj(&d, obj, errmess)) {
        *v = (long_double)d;
        return 1;
    }
    return 0;
}
"""


cfuncs['double_from_pyobj'] = """
static int
double_from_pyobj(double* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;
    if (PyFloat_Check(obj)) {
        *v = PyFloat_AsDouble(obj);
        return !(*v == -1.0 && PyErr_Occurred());
    }

    tmp = PyNumber_Float(obj);
    if (tmp) {
        *v = PyFloat_AsDouble(tmp);
        Py_DECREF(tmp);
        return !(*v == -1.0 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,\"real\");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = #modulename#_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}
"""


needs['float_from_pyobj'] = ['double_from_pyobj']
cfuncs['float_from_pyobj'] = """
static int
float_from_pyobj(float* v, PyObject *obj, const char *errmess)
{
    double d=0.0;
    if (double_from_pyobj(&d,obj,errmess)) {
        *v = (float)d;
        return 1;
    }
    return 0;
}
"""


needs['complex_long_double_from_pyobj'] = ['complex_long_double', 'long_double',
                                           'complex_double_from_pyobj', 'npy_math.h']
cfuncs['complex_long_double_from_pyobj'] = """
static int
complex_long_double_from_pyobj(complex_long_double* v, PyObject *obj, const char *errmess)
{
    complex_double cd = {0.0,0.0};
    if (PyArray_CheckScalar(obj)){
        if PyArray_IsScalar(obj, CLongDouble) {
            PyArray_ScalarAsCtype(obj, v);
            return 1;
        }
        else if (PyArray_Check(obj) && PyArray_TYPE(obj)==NPY_CLONGDOUBLE) {
            (*v).r = npy_creall(*(((npy_clongdouble *)PyArray_DATA(obj))));
            (*v).i = npy_cimagl(*(((npy_clongdouble *)PyArray_DATA(obj))));
            return 1;
        }
    }
    if (complex_double_from_pyobj(&cd,obj,errmess)) {
        (*v).r = (long_double)cd.r;
        (*v).i = (long_double)cd.i;
        return 1;
    }
    return 0;
}
"""


needs['complex_double_from_pyobj'] = ['complex_double', 'npy_math.h']
cfuncs['complex_double_from_pyobj'] = """
static int
complex_double_from_pyobj(complex_double* v, PyObject *obj, const char *errmess) {
    Py_complex c;
    if (PyComplex_Check(obj)) {
        c = PyComplex_AsCComplex(obj);
        (*v).r = c.real;
        (*v).i = c.imag;
        return 1;
    }
    if (PyArray_IsScalar(obj, ComplexFloating)) {
        if (PyArray_IsScalar(obj, CFloat)) {
            npy_cfloat new;
            PyArray_ScalarAsCtype(obj, &new);
            (*v).r = (double)npy_crealf(new);
            (*v).i = (double)npy_cimagf(new);
        }
        else if (PyArray_IsScalar(obj, CLongDouble)) {
            npy_clongdouble new;
            PyArray_ScalarAsCtype(obj, &new);
            (*v).r = (double)npy_creall(new);
            (*v).i = (double)npy_cimagl(new);
        }
        else { /* if (PyArray_IsScalar(obj, CDouble)) */
            PyArray_ScalarAsCtype(obj, v);
        }
        return 1;
    }
    if (PyArray_CheckScalar(obj)) { /* 0-dim array or still array scalar */
        PyArrayObject *arr;
        if (PyArray_Check(obj)) {
            arr = (PyArrayObject *)PyArray_Cast((PyArrayObject *)obj, NPY_CDOUBLE);
        }
        else {
            arr = (PyArrayObject *)PyArray_FromScalar(obj, PyArray_DescrFromType(NPY_CDOUBLE));
        }
        if (arr == NULL) {
            return 0;
        }
        (*v).r = npy_creal(*(((npy_cdouble *)PyArray_DATA(arr))));
        (*v).i = npy_cimag(*(((npy_cdouble *)PyArray_DATA(arr))));
        Py_DECREF(arr);
        return 1;
    }
    /* Python does not provide PyNumber_Complex function :-( */
    (*v).i = 0.0;
    if (PyFloat_Check(obj)) {
        (*v).r = PyFloat_AsDouble(obj);
        return !((*v).r == -1.0 && PyErr_Occurred());
    }
    if (PyLong_Check(obj)) {
        (*v).r = PyLong_AsDouble(obj);
        return !((*v).r == -1.0 && PyErr_Occurred());
    }
    if (PySequence_Check(obj) && !(PyBytes_Check(obj) || PyUnicode_Check(obj))) {
        PyObject *tmp = PySequence_GetItem(obj,0);
        if (tmp) {
            if (complex_double_from_pyobj(v,tmp,errmess)) {
                Py_DECREF(tmp);
                return 1;
            }
            Py_DECREF(tmp);
        }
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL)
            err = PyExc_TypeError;
        PyErr_SetString(err,errmess);
    }
    return 0;
}
"""


needs['complex_float_from_pyobj'] = [
    'complex_float', 'complex_double_from_pyobj']
cfuncs['complex_float_from_pyobj'] = """
static int
complex_float_from_pyobj(complex_float* v,PyObject *obj,const char *errmess)
{
    complex_double cd={0.0,0.0};
    if (complex_double_from_pyobj(&cd,obj,errmess)) {
        (*v).r = (float)cd.r;
        (*v).i = (float)cd.i;
        return 1;
    }
    return 0;
}
"""


cfuncs['try_pyarr_from_character'] = """
static int try_pyarr_from_character(PyObject* obj, character* v) {
    PyArrayObject *arr = (PyArrayObject*)obj;
    if (!obj) return -2;
    if (PyArray_Check(obj)) {
        if (F2PY_ARRAY_IS_CHARACTER_COMPATIBLE(arr))  {
            *(character *)(PyArray_DATA(arr)) = *v;
            return 1;
        }
    }
    {
        char mess[F2PY_MESSAGE_BUFFER_SIZE];
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = PyExc_ValueError;
            strcpy(mess, "try_pyarr_from_character failed"
                         " -- expected bytes array-scalar|array, got ");
            f2py_describe(obj, mess + strlen(mess));
            PyErr_SetString(err, mess);
        }
    }
    return 0;
}
"""

needs['try_pyarr_from_char'] = ['pyobj_from_char1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_char'] = 'static int try_pyarr_from_char(PyObject* obj,char* v) {\n    TRYPYARRAYTEMPLATE(char,\'c\');\n}\n'
needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'unsigned_char']
cfuncs[
    'try_pyarr_from_unsigned_char'] = 'static int try_pyarr_from_unsigned_char(PyObject* obj,unsigned_char* v) {\n    TRYPYARRAYTEMPLATE(unsigned_char,\'b\');\n}\n'
needs['try_pyarr_from_signed_char'] = ['TRYPYARRAYTEMPLATE', 'signed_char']
cfuncs[
    'try_pyarr_from_signed_char'] = 'static int try_pyarr_from_signed_char(PyObject* obj,signed_char* v) {\n    TRYPYARRAYTEMPLATE(signed_char,\'1\');\n}\n'
needs['try_pyarr_from_short'] = ['pyobj_from_short1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_short'] = 'static int try_pyarr_from_short(PyObject* obj,short* v) {\n    TRYPYARRAYTEMPLATE(short,\'s\');\n}\n'
needs['try_pyarr_from_int'] = ['pyobj_from_int1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_int'] = 'static int try_pyarr_from_int(PyObject* obj,int* v) {\n    TRYPYARRAYTEMPLATE(int,\'i\');\n}\n'
needs['try_pyarr_from_long'] = ['pyobj_from_long1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_long'] = 'static int try_pyarr_from_long(PyObject* obj,long* v) {\n    TRYPYARRAYTEMPLATE(long,\'l\');\n}\n'
needs['try_pyarr_from_long_long'] = [
    'pyobj_from_long_long1', 'TRYPYARRAYTEMPLATE', 'long_long']
cfuncs[
    'try_pyarr_from_long_long'] = 'static int try_pyarr_from_long_long(PyObject* obj,long_long* v) {\n    TRYPYARRAYTEMPLATE(long_long,\'L\');\n}\n'
needs['try_pyarr_from_float'] = ['pyobj_from_float1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_float'] = 'static int try_pyarr_from_float(PyObject* obj,float* v) {\n    TRYPYARRAYTEMPLATE(float,\'f\');\n}\n'
needs['try_pyarr_from_double'] = ['pyobj_from_double1', 'TRYPYARRAYTEMPLATE']
cfuncs[
    'try_pyarr_from_double'] = 'static int try_pyarr_from_double(PyObject* obj,double* v) {\n    TRYPYARRAYTEMPLATE(double,\'d\');\n}\n'
needs['try_pyarr_from_complex_float'] = [
    'pyobj_from_complex_float1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_float']
cfuncs[
    'try_pyarr_from_complex_float'] = 'static int try_pyarr_from_complex_float(PyObject* obj,complex_float* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(float,\'F\');\n}\n'
needs['try_pyarr_from_complex_double'] = [
    'pyobj_from_complex_double1', 'TRYCOMPLEXPYARRAYTEMPLATE', 'complex_double']
cfuncs[
    'try_pyarr_from_complex_double'] = 'static int try_pyarr_from_complex_double(PyObject* obj,complex_double* v) {\n    TRYCOMPLEXPYARRAYTEMPLATE(double,\'D\');\n}\n'


needs['create_cb_arglist'] = ['CFUNCSMESS', 'PRINTPYOBJERR', 'MINMAX']
# create the list of arguments to be used when calling back to python
cfuncs['create_cb_arglist'] = """
static int
create_cb_arglist(PyObject* fun, PyTupleObject* xa , const int maxnofargs,
                  const int nofoptargs, int *nofargs, PyTupleObject **args,
                  const char *errmess)
{
    PyObject *tmp = NULL;
    PyObject *tmp_fun = NULL;
    Py_ssize_t tot, opt, ext, siz, i, di = 0;
    CFUNCSMESS(\"create_cb_arglist\\n\");
    tot=opt=ext=siz=0;
    /* Get the total number of arguments */
    if (PyFunction_Check(fun)) {
        tmp_fun = fun;
        Py_INCREF(tmp_fun);
    }
    else {
        di = 1;
        if (PyObject_HasAttrString(fun,\"im_func\")) {
            tmp_fun = PyObject_GetAttrString(fun,\"im_func\");
        }
        else if (PyObject_HasAttrString(fun,\"__call__\")) {
            tmp = PyObject_GetAttrString(fun,\"__call__\");
            if (PyObject_HasAttrString(tmp,\"im_func\"))
                tmp_fun = PyObject_GetAttrString(tmp,\"im_func\");
            else {
                tmp_fun = fun; /* built-in function */
                Py_INCREF(tmp_fun);
                tot = maxnofargs;
                if (PyCFunction_Check(fun)) {
                    /* In case the function has a co_argcount (like on PyPy) */
                    di = 0;
                }
                if (xa != NULL)
                    tot += PyTuple_Size((PyObject *)xa);
            }
            Py_XDECREF(tmp);
        }
        else if (PyFortran_Check(fun) || PyFortran_Check1(fun)) {
            tot = maxnofargs;
            if (xa != NULL)
                tot += PyTuple_Size((PyObject *)xa);
            tmp_fun = fun;
            Py_INCREF(tmp_fun);
        }
        else if (F2PyCapsule_Check(fun)) {
            tot = maxnofargs;
            if (xa != NULL)
                ext = PyTuple_Size((PyObject *)xa);
            if(ext>0) {
                fprintf(stderr,\"extra arguments tuple cannot be used with PyCapsule call-back\\n\");
                goto capi_fail;
            }
            tmp_fun = fun;
            Py_INCREF(tmp_fun);
        }
    }

    if (tmp_fun == NULL) {
        fprintf(stderr,
                \"Call-back argument must be function|instance|instance.__call__|f2py-function \"
                \"but got %s.\\n\",
                ((fun == NULL) ? \"NULL\" : Py_TYPE(fun)->tp_name));
        goto capi_fail;
    }

    if (PyObject_HasAttrString(tmp_fun,\"__code__\")) {
        if (PyObject_HasAttrString(tmp = PyObject_GetAttrString(tmp_fun,\"__code__\"),\"co_argcount\")) {
            PyObject *tmp_argcount = PyObject_GetAttrString(tmp,\"co_argcount\");
            Py_DECREF(tmp);
            if (tmp_argcount == NULL) {
                goto capi_fail;
            }
            tot = PyLong_AsSsize_t(tmp_argcount) - di;
            Py_DECREF(tmp_argcount);
        }
    }
    /* Get the number of optional arguments */
    if (PyObject_HasAttrString(tmp_fun,\"__defaults__\")) {
        if (PyTuple_Check(tmp = PyObject_GetAttrString(tmp_fun,\"__defaults__\")))
            opt = PyTuple_Size(tmp);
        Py_XDECREF(tmp);
    }
    /* Get the number of extra arguments */
    if (xa != NULL)
        ext = PyTuple_Size((PyObject *)xa);
    /* Calculate the size of call-backs argument list */
    siz = MIN(maxnofargs+ext,tot);
    *nofargs = MAX(0,siz-ext);

#ifdef DEBUGCFUNCS
    fprintf(stderr,
            \"debug-capi:create_cb_arglist:maxnofargs(-nofoptargs),\"
            \"tot,opt,ext,siz,nofargs = %d(-%d), %zd, %zd, %zd, %zd, %d\\n\",
            maxnofargs, nofoptargs, tot, opt, ext, siz, *nofargs);
#endif

    if (siz < tot-opt) {
        fprintf(stderr,
                \"create_cb_arglist: Failed to build argument list \"
                \"(siz) with enough arguments (tot-opt) required by \"
                \"user-supplied function (siz,tot,opt=%zd, %zd, %zd).\\n\",
                siz, tot, opt);
        goto capi_fail;
    }

    /* Initialize argument list */
    *args = (PyTupleObject *)PyTuple_New(siz);
    for (i=0;i<*nofargs;i++) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM((PyObject *)(*args),i,Py_None);
    }
    if (xa != NULL)
        for (i=(*nofargs);i<siz;i++) {
            tmp = PyTuple_GetItem((PyObject *)xa,i-(*nofargs));
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(*args,i,tmp);
        }
    CFUNCSMESS(\"create_cb_arglist-end\\n\");
    Py_DECREF(tmp_fun);
    return 1;

capi_fail:
    if (PyErr_Occurred() == NULL)
        PyErr_SetString(#modulename#_error, errmess);
    Py_XDECREF(tmp_fun);
    return 0;
}
"""


def buildcfuncs():
    from .capi_maps import c2capi_map
    for k in c2capi_map.keys():
        m = 'pyarr_from_p_%s1' % k
        cppmacros[
            m] = '#define %s(v) (PyArray_SimpleNewFromData(0,NULL,%s,(char *)v))' % (m, c2capi_map[k])
    k = 'string'
    m = 'pyarr_from_p_%s1' % k
    # NPY_CHAR compatibility, NPY_STRING with itemsize 1
    cppmacros[
        m] = '#define %s(v,dims) (PyArray_New(&PyArray_Type, 1, dims, NPY_STRING, NULL, v, 1, NPY_ARRAY_CARRAY, NULL))' % (m)


############ Auxiliary functions for sorting needs ###################

def append_needs(need, flag=1):
    # This function modifies the contents of the global `outneeds` dict.
    if isinstance(need, list):
        for n in need:
            append_needs(n, flag)
    elif isinstance(need, str):
        if not need:
            return
        if need in includes0:
            n = 'includes0'
        elif need in includes:
            n = 'includes'
        elif need in typedefs:
            n = 'typedefs'
        elif need in typedefs_generated:
            n = 'typedefs_generated'
        elif need in cppmacros:
            n = 'cppmacros'
        elif need in cfuncs:
            n = 'cfuncs'
        elif need in callbacks:
            n = 'callbacks'
        elif need in f90modhooks:
            n = 'f90modhooks'
        elif need in commonhooks:
            n = 'commonhooks'
        else:
            errmess('append_needs: unknown need %s\n' % (repr(need)))
            return
        if need in outneeds[n]:
            return
        if flag:
            tmp = {}
            if need in needs:
                for nn in needs[need]:
                    t = append_needs(nn, 0)
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = tmp[nnn] + t[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            for nn in tmp.keys():
                for nnn in tmp[nn]:
                    if nnn not in outneeds[nn]:
                        outneeds[nn] = [nnn] + outneeds[nn]
            outneeds[n].append(need)
        else:
            tmp = {}
            if need in needs:
                for nn in needs[need]:
                    t = append_needs(nn, flag)
                    if isinstance(t, dict):
                        for nnn in t.keys():
                            if nnn in tmp:
                                tmp[nnn] = t[nnn] + tmp[nnn]
                            else:
                                tmp[nnn] = t[nnn]
            if n not in tmp:
                tmp[n] = []
            tmp[n].append(need)
            return tmp
    else:
        errmess('append_needs: expected list or string but got :%s\n' %
                (repr(need)))


def get_needs():
    # This function modifies the contents of the global `outneeds` dict.
    res = {}
    for n in outneeds.keys():
        out = []
        saveout = copy.copy(outneeds[n])
        while len(outneeds[n]) > 0:
            if outneeds[n][0] not in needs:
                out.append(outneeds[n][0])
                del outneeds[n][0]
            else:
                flag = 0
                for k in outneeds[n][1:]:
                    if k in needs[outneeds[n][0]]:
                        flag = 1
                        break
                if flag:
                    outneeds[n] = outneeds[n][1:] + [outneeds[n][0]]
                else:
                    out.append(outneeds[n][0])
                    del outneeds[n][0]
            if saveout and (0 not in map(lambda x, y: x == y, saveout, outneeds[n])) \
                    and outneeds[n] != []:
                print(n, saveout)
                errmess(
                    'get_needs: no progress in sorting needs, probably circular dependence, skipping.\n')
                out = out + saveout
                break
            saveout = copy.copy(outneeds[n])
        if out == []:
            out = [n]
        res[n] = out
    return res
<|endoftext|>"""Fortran/C symbolic expressions

References:
- J3/21-007: Draft Fortran 202x. https://j3-fortran.org/doc/year/21/21-007.pdf

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

# To analyze Fortran expressions to solve dimensions specifications,
# for instances, we implement a minimal symbolic engine for parsing
# expressions into a tree of expression instances. As a first
# instance, we care only about arithmetic expressions involving
# integers and operations like addition (+), subtraction (-),
# multiplication (*), division (Fortran / is Python //, Fortran // is
# concatenate), and exponentiation (**).  In addition, .pyf files may
# contain C expressions that support here is implemented as well.
#
# TODO: support logical constants (Op.BOOLEAN)
# TODO: support logical operators (.AND., ...)
# TODO: support defined operators (.MYOP., ...)
#
__all__ = ['Expr']


import re
import warnings
from enum import Enum
from math import gcd


class Language(Enum):
    """
    Used as Expr.tostring language argument.
    """
    Python = 0
    Fortran = 1
    C = 2


class Op(Enum):
    """
    Used as Expr op attribute.
    """
    INTEGER = 10
    REAL = 12
    COMPLEX = 15
    STRING = 20
    ARRAY = 30
    SYMBOL = 40
    TERNARY = 100
    APPLY = 200
    INDEXING = 210
    CONCAT = 220
    RELATIONAL = 300
    TERMS = 1000
    FACTORS = 2000
    REF = 3000
    DEREF = 3001


class RelOp(Enum):
    """
    Used in Op.RELATIONAL expression to specify the function part.
    """
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6

    @classmethod
    def fromstring(cls, s, language=Language.C):
        if language is Language.Fortran:
            return {'.eq.': RelOp.EQ, '.ne.': RelOp.NE,
                    '.lt.': RelOp.LT, '.le.': RelOp.LE,
                    '.gt.': RelOp.GT, '.ge.': RelOp.GE}[s.lower()]
        return {'==': RelOp.EQ, '!=': RelOp.NE, '<': RelOp.LT,
                '<=': RelOp.LE, '>': RelOp.GT, '>=': RelOp.GE}[s]

    def tostring(self, language=Language.C):
        if language is Language.Fortran:
            return {RelOp.EQ: '.eq.', RelOp.NE: '.ne.',
                    RelOp.LT: '.lt.', RelOp.LE: '.le.',
                    RelOp.GT: '.gt.', RelOp.GE: '.ge.'}[self]
        return {RelOp.EQ: '==', RelOp.NE: '!=',
                RelOp.LT: '<', RelOp.LE: '<=',
                RelOp.GT: '>', RelOp.GE: '>='}[self]


class ArithOp(Enum):
    """
    Used in Op.APPLY expression to specify the function part.
    """
    POS = 1
    NEG = 2
    ADD = 3
    SUB = 4
    MUL = 5
    DIV = 6
    POW = 7


class OpError(Exception):
    pass


class Precedence(Enum):
    """
    Used as Expr.tostring precedence argument.
    """
    ATOM = 0
    POWER = 1
    UNARY = 2
    PRODUCT = 3
    SUM = 4
    LT = 6
    EQ = 7
    LAND = 11
    LOR = 12
    TERNARY = 13
    ASSIGN = 14
    TUPLE = 15
    NONE = 100


integer_types = (int,)
number_types = (int, float)


def _pairs_add(d, k, v):
    # Internal utility method for updating terms and factors data.
    c = d.get(k)
    if c is None:
        d[k] = v
    else:
        c = c + v
        if c:
            d[k] = c
        else:
            del d[k]


class ExprWarning(UserWarning):
    pass


def ewarn(message):
    warnings.warn(message, ExprWarning, stacklevel=2)


class Expr:
    """Represents a Fortran expression as a op-data pair.

    Expr instances are hashable and sortable.
    """

    @staticmethod
    def parse(s, language=Language.C):
        """Parse a Fortran expression to a Expr.
        """
        return fromstring(s, language=language)

    def __init__(self, op, data):
        assert isinstance(op, Op)

        # sanity checks
        if op is Op.INTEGER:
            # data is a 2-tuple of numeric object and a kind value
            # (default is 4)
            assert isinstance(data, tuple) and len(data) == 2
            assert isinstance(data[0], int)
            assert isinstance(data[1], (int, str)), data
        elif op is Op.REAL:
            # data is a 2-tuple of numeric object and a kind value
            # (default is 4)
            assert isinstance(data, tuple) and len(data) == 2
            assert isinstance(data[0], float)
            assert isinstance(data[1], (int, str)), data
        elif op is Op.COMPLEX:
            # data is a 2-tuple of constant expressions
            assert isinstance(data, tuple) and len(data) == 2
        elif op is Op.STRING:
            # data is a 2-tuple of quoted string and a kind value
            # (default is 1)
            assert isinstance(data, tuple) and len(data) == 2
            assert (isinstance(data[0], str)
                    and data[0][::len(data[0])-1] in ('""', "''", '@@'))
            assert isinstance(data[1], (int, str)), data
        elif op is Op.SYMBOL:
            # data is any hashable object
            assert hash(data) is not None
        elif op in (Op.ARRAY, Op.CONCAT):
            # data is a tuple of expressions
            assert isinstance(data, tuple)
            assert all(isinstance(item, Expr) for item in data), data
        elif op in (Op.TERMS, Op.FACTORS):
            # data is {<term|base>:<coeff|exponent>} where dict values
            # are nonzero Python integers
            assert isinstance(data, dict)
        elif op is Op.APPLY:
            # data is (<function>, <operands>, <kwoperands>) where
            # operands are Expr instances
            assert isinstance(data, tuple) and len(data) == 3
            # function is any hashable object
            assert hash(data[0]) is not None
            assert isinstance(data[1], tuple)
            assert isinstance(data[2], dict)
        elif op is Op.INDEXING:
            # data is (<object>, <indices>)
            assert isinstance(data, tuple) and len(data) == 2
            # function is any hashable object
            assert hash(data[0]) is not None
        elif op is Op.TERNARY:
            # data is (<cond>, <expr1>, <expr2>)
            assert isinstance(data, tuple) and len(data) == 3
        elif op in (Op.REF, Op.DEREF):
            # data is Expr instance
            assert isinstance(data, Expr)
        elif op is Op.RELATIONAL:
            # data is (<relop>, <left>, <right>)
            assert isinstance(data, tuple) and len(data) == 3
        else:
            raise NotImplementedError(
                f'unknown op or missing sanity check: {op}')

        self.op = op
        self.data = data

    def __eq__(self, other):
        return (isinstance(other, Expr)
                and self.op is other.op
                and self.data == other.data)

    def __hash__(self):
        if self.op in (Op.TERMS, Op.FACTORS):
            data = tuple(sorted(self.data.items()))
        elif self.op is Op.APPLY:
            data = self.data[:2] + tuple(sorted(self.data[2].items()))
        else:
            data = self.data
        return hash((self.op, data))

    def __lt__(self, other):
        if isinstance(other, Expr):
            if self.op is not other.op:
                return self.op.value < other.op.value
            if self.op in (Op.TERMS, Op.FACTORS):
                return (tuple(sorted(self.data.items()))
                        < tuple(sorted(other.data.items())))
            if self.op is Op.APPLY:
                if self.data[:2] != other.data[:2]:
                    return self.data[:2] < other.data[:2]
                return tuple(sorted(self.data[2].items())) < tuple(
                    sorted(other.data[2].items()))
            return self.data < other.data
        return NotImplemented

    def __le__(self, other): return self == other or self < other

    def __gt__(self, other): return not (self <= other)

    def __ge__(self, other): return not (self < other)

    def __repr__(self):
        return f'{type(self).__name__}({self.op}, {self.data!r})'

    def __str__(self):
        return self.tostring()

    def tostring(self, parent_precedence=Precedence.NONE,
                 language=Language.Fortran):
        """Return a string representation of Expr.
        """
        if self.op in (Op.INTEGER, Op.REAL):
            precedence = (Precedence.SUM if self.data[0] < 0
                          else Precedence.ATOM)
            r = str(self.data[0]) + (f'_{self.data[1]}'
                                     if self.data[1] != 4 else '')
        elif self.op is Op.COMPLEX:
            r = ', '.join(item.tostring(Precedence.TUPLE, language=language)
                          for item in self.data)
            r = '(' + r + ')'
            precedence = Precedence.ATOM
        elif self.op is Op.SYMBOL:
            precedence = Precedence.ATOM
            r = str(self.data)
        elif self.op is Op.STRING:
            r = self.data[0]
            if self.data[1] != 1:
                r = self.data[1] + '_' + r
            precedence = Precedence.ATOM
        elif self.op is Op.ARRAY:
            r = ', '.join(item.tostring(Precedence.TUPLE, language=language)
                          for item in self.data)
            r = '[' + r + ']'
            precedence = Precedence.ATOM
        elif self.op is Op.TERMS:
            terms = []
            for term, coeff in sorted(self.data.items()):
                if coeff < 0:
                    op = ' - '
                    coeff = -coeff
                else:
                    op = ' + '
                if coeff == 1:
                    term = term.tostring(Precedence.SUM, language=language)
                else:
                    if term == as_number(1):
                        term = str(coeff)
                    else:
                        term = f'{coeff} * ' + term.tostring(
                            Precedence.PRODUCT, language=language)
                if terms:
                    terms.append(op)
                elif op == ' - ':
                    terms.append('-')
                terms.append(term)
            r = ''.join(terms) or '0'
            precedence = Precedence.SUM if terms else Precedence.ATOM
        elif self.op is Op.FACTORS:
            factors = []
            tail = []
            for base, exp in sorted(self.data.items()):
                op = ' * '
                if exp == 1:
                    factor = base.tostring(Precedence.PRODUCT,
                                           language=language)
                elif language is Language.C:
                    if exp in range(2, 10):
                        factor = base.tostring(Precedence.PRODUCT,
                                               language=language)
                        factor = ' * '.join([factor] * exp)
                    elif exp in range(-10, 0):
                        factor = base.tostring(Precedence.PRODUCT,
                                               language=language)
                        tail += [factor] * -exp
                        continue
                    else:
                        factor = base.tostring(Precedence.TUPLE,
                                               language=language)
                        factor = f'pow({factor}, {exp})'
                else:
                    factor = base.tostring(Precedence.POWER,
                                           language=language) + f' ** {exp}'
                if factors:
                    factors.append(op)
                factors.append(factor)
            if tail:
                if not factors:
                    factors += ['1']
                factors += ['/', '(', ' * '.join(tail), ')']
            r = ''.join(factors) or '1'
            precedence = Precedence.PRODUCT if factors else Precedence.ATOM
        elif self.op is Op.APPLY:
            name, args, kwargs = self.data
            if name is ArithOp.DIV and language is Language.C:
                numer, denom = [arg.tostring(Precedence.PRODUCT,
                                             language=language)
                                for arg in args]
                r = f'{numer} / {denom}'
                precedence = Precedence.PRODUCT
            else:
                args = [arg.tostring(Precedence.TUPLE, language=language)
                        for arg in args]
                args += [k + '=' + v.tostring(Precedence.NONE)
                         for k, v in kwargs.items()]
                r = f'{name}({", ".join(args)})'
                precedence = Precedence.ATOM
        elif self.op is Op.INDEXING:
            name = self.data[0]
            args = [arg.tostring(Precedence.TUPLE, language=language)
                    for arg in self.data[1:]]
            r = f'{name}[{", ".join(args)}]'
            precedence = Precedence.ATOM
        elif self.op is Op.CONCAT:
            args = [arg.tostring(Precedence.PRODUCT, language=language)
                    for arg in self.data]
            r = " // ".join(args)
            precedence = Precedence.PRODUCT
        elif self.op is Op.TERNARY:
            cond, expr1, expr2 = [a.tostring(Precedence.TUPLE,
                                             language=language)
                                  for a in self.data]
            if language is Language.C:
                r = f'({cond}?{expr1}:{expr2})'
            elif language is Language.Python:
                r = f'({expr1} if {cond} else {expr2})'
            elif language is Language.Fortran:
                r = f'merge({expr1}, {expr2}, {cond})'
            else:
                raise NotImplementedError(
                    f'tostring for {self.op} and {language}')
            precedence = Precedence.ATOM
        elif self.op is Op.REF:
            r = '&' + self.data.tostring(Precedence.UNARY, language=language)
            precedence = Precedence.UNARY
        elif self.op is Op.DEREF:
            r = '*' + self.data.tostring(Precedence.UNARY, language=language)
            precedence = Precedence.UNARY
        elif self.op is Op.RELATIONAL:
            rop, left, right = self.data
            precedence = (Precedence.EQ if rop in (RelOp.EQ, RelOp.NE)
                          else Precedence.LT)
            left = left.tostring(precedence, language=language)
            right = right.tostring(precedence, language=language)
            rop = rop.tostring(language=language)
            r = f'{left} {rop} {right}'
        else:
            raise NotImplementedError(f'tostring for op {self.op}')
        if parent_precedence.value < precedence.value:
            # If parent precedence is higher than operand precedence,
            # operand will be enclosed in parenthesis.
            return '(' + r + ')'
        return r

    def __pos__(self):
        return self

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            if self.op is other.op:
                if self.op in (Op.INTEGER, Op.REAL):
                    return as_number(
                        self.data[0] + other.data[0],
                        max(self.data[1], other.data[1]))
                if self.op is Op.COMPLEX:
                    r1, i1 = self.data
                    r2, i2 = other.data
                    return as_complex(r1 + r2, i1 + i2)
                if self.op is Op.TERMS:
                    r = Expr(self.op, dict(self.data))
                    for k, v in other.data.items():
                        _pairs_add(r.data, k, v)
                    return normalize(r)
            if self.op is Op.COMPLEX and other.op in (Op.INTEGER, Op.REAL):
                return self + as_complex(other)
            elif self.op in (Op.INTEGER, Op.REAL) and other.op is Op.COMPLEX:
                return as_complex(self) + other
            elif self.op is Op.REAL and other.op is Op.INTEGER:
                return self + as_real(other, kind=self.data[1])
            elif self.op is Op.INTEGER and other.op is Op.REAL:
                return as_real(self, kind=other.data[1]) + other
            return as_terms(self) + as_terms(other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, number_types):
            return as_number(other) + self
        return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        if isinstance(other, number_types):
            return as_number(other) - self
        return NotImplemented

    def __mul__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            if self.op is other.op:
                if self.op in (Op.INTEGER, Op.REAL):
                    return as_number(self.data[0] * other.data[0],
                                     max(self.data[1], other.data[1]))
                elif self.op is Op.COMPLEX:
                    r1, i1 = self.data
                    r2, i2 = other.data
                    return as_complex(r1 * r2 - i1 * i2, r1 * i2 + r2 * i1)

                if self.op is Op.FACTORS:
                    r = Expr(self.op, dict(self.data))
                    for k, v in other.data.items():
                        _pairs_add(r.data, k, v)
                    return normalize(r)
                elif self.op is Op.TERMS:
                    r = Expr(self.op, {})
                    for t1, c1 in self.data.items():
                        for t2, c2 in other.data.items():
                            _pairs_add(r.data, t1 * t2, c1 * c2)
                    return normalize(r)

            if self.op is Op.COMPLEX and other.op in (Op.INTEGER, Op.REAL):
                return self * as_complex(other)
            elif other.op is Op.COMPLEX and self.op in (Op.INTEGER, Op.REAL):
                return as_complex(self) * other
            elif self.op is Op.REAL and other.op is Op.INTEGER:
                return self * as_real(other, kind=self.data[1])
            elif self.op is Op.INTEGER and other.op is Op.REAL:
                return as_real(self, kind=other.data[1]) * other

            if self.op is Op.TERMS:
                return self * as_terms(other)
            elif other.op is Op.TERMS:
                return as_terms(self) * other

            return as_factors(self) * as_factors(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, number_types):
            return as_number(other) * self
        return NotImplemented

    def __pow__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            if other.op is Op.INTEGER:
                exponent = other.data[0]
                # TODO: other kind not used
                if exponent == 0:
                    return as_number(1)
                if exponent == 1:
                    return self
                if exponent > 0:
                    if self.op is Op.FACTORS:
                        r = Expr(self.op, {})
                        for k, v in self.data.items():
                            r.data[k] = v * exponent
                        return normalize(r)
                    return self * (self ** (exponent - 1))
                elif exponent != -1:
                    return (self ** (-exponent)) ** -1
                return Expr(Op.FACTORS, {self: exponent})
            return as_apply(ArithOp.POW, self, other)
        return NotImplemented

    def __truediv__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            # Fortran / is different from Python /:
            # - `/` is a truncate operation for integer operands
            return normalize(as_apply(ArithOp.DIV, self, other))
        return NotImplemented

    def __rtruediv__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            return other / self
        return NotImplemented

    def __floordiv__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            # Fortran // is different from Python //:
            # - `//` is a concatenate operation for string operands
            return normalize(Expr(Op.CONCAT, (self, other)))
        return NotImplemented

    def __rfloordiv__(self, other):
        other = as_expr(other)
        if isinstance(other, Expr):
            return other // self
        return NotImplemented

    def __call__(self, *args, **kwargs):
        # In Fortran, parenthesis () are use for both function call as
        # well as indexing operations.
        #
        # TODO: implement a method for deciding when __call__ should
        # return an INDEXING expression.
        return as_apply(self, *map(as_expr, args),
                        **dict((k, as_expr(v)) for k, v in kwargs.items()))

    def __getitem__(self, index):
        # Provided to support C indexing operations that .pyf files
        # may contain.
        index = as_expr(index)
        if not isinstance(index, tuple):
            index = index,
        if len(index) > 1:
            ewarn(f'C-index should be a single expression but got `{index}`')
        return Expr(Op.INDEXING, (self,) + index)

    def substitute(self, symbols_map):
        """Recursively substitute symbols with values in symbols map.

        Symbols map is a dictionary of symbol-expression pairs.
        """
        if self.op is Op.SYMBOL:
            value = symbols_map.get(self)
            if value is None:
                return self
            m = re.match(r'\A(@__f2py_PARENTHESIS_(\w+)_\d+@)\Z', self.data)
            if m:
                # complement to fromstring method
                items, paren = m.groups()
                if paren in ['ROUNDDIV', 'SQUARE']:
                    return as_array(value)
                assert paren == 'ROUND', (paren, value)
            return value
        if self.op in (Op.INTEGER, Op.REAL, Op.STRING):
            return self
        if self.op in (Op.ARRAY, Op.COMPLEX):
            return Expr(self.op, tuple(item.substitute(symbols_map)
                                       for item in self.data))
        if self.op is Op.CONCAT:
            return normalize(Expr(self.op, tuple(item.substitute(symbols_map)
                                                 for item in self.data)))
        if self.op is Op.TERMS:
            r = None
            for term, coeff in self.data.items():
                if r is None:
                    r = term.substitute(symbols_map) * coeff
                else:
                    r += term.substitute(symbols_map) * coeff
            if r is None:
                ewarn('substitute: empty TERMS expression interpreted as'
                      ' int-literal 0')
                return as_number(0)
            return r
        if self.op is Op.FACTORS:
            r = None
            for base, exponent in self.data.items():
                if r is None:
                    r = base.substitute(symbols_map) ** exponent
                else:
                    r *= base.substitute(symbols_map) ** exponent
            if r is None:
                ewarn('substitute: empty FACTORS expression interpreted'
                      ' as int-literal 1')
                return as_number(1)
            return r
        if self.op is Op.APPLY:
            target, args, kwargs = self.data
            if isinstance(target, Expr):
                target = target.substitute(symbols_map)
            args = tuple(a.substitute(symbols_map) for a in args)
            kwargs = dict((k, v.substitute(symbols_map))
                          for k, v in kwargs.items())
            return normalize(Expr(self.op, (target, args, kwargs)))
        if self.op is Op.INDEXING:
            func = self.data[0]
            if isinstance(func, Expr):
                func = func.substitute(symbols_map)
            args = tuple(a.substitute(symbols_map) for a in self.data[1:])
            return normalize(Expr(self.op, (func,) + args))
        if self.op is Op.TERNARY:
            operands = tuple(a.substitute(symbols_map) for a in self.data)
            return normalize(Expr(self.op, operands))
        if self.op in (Op.REF, Op.DEREF):
            return normalize(Expr(self.op, self.data.substitute(symbols_map)))
        if self.op is Op.RELATIONAL:
            rop, left, right = self.data
            left = left.substitute(symbols_map)
            right = right.substitute(symbols_map)
            return normalize(Expr(self.op, (rop, left, right)))
        raise NotImplementedError(f'substitute method for {self.op}: {self!r}')

    def traverse(self, visit, *args, **kwargs):
        """Traverse expression tree with visit function.

        The visit function is applied to an expression with given args
        and kwargs.

        Traverse call returns an expression returned by visit when not
        None, otherwise return a new normalized expression with
        traverse-visit sub-expressions.
        """
        result = visit(self, *args, **kwargs)
        if result is not None:
            return result

        if self.op in (Op.INTEGER, Op.REAL, Op.STRING, Op.SYMBOL):
            return self
        elif self.op in (Op.COMPLEX, Op.ARRAY, Op.CONCAT, Op.TERNARY):
            return normalize(Expr(self.op, tuple(
                item.traverse(visit, *args, **kwargs)
                for item in self.data)))
        elif self.op in (Op.TERMS, Op.FACTORS):
            data = {}
            for k, v in self.data.items():
                k = k.traverse(visit, *args, **kwargs)
                v = (v.traverse(visit, *args, **kwargs)
                     if isinstance(v, Expr) else v)
                if k in data:
                    v = data[k] + v
                data[k] = v
            return normalize(Expr(self.op, data))
        elif self.op is Op.APPLY:
            obj = self.data[0]
            func = (obj.traverse(visit, *args, **kwargs)
                    if isinstance(obj, Expr) else obj)
            operands = tuple(operand.traverse(visit, *args, **kwargs)
                             for operand in self.data[1])
            kwoperands = dict((k, v.traverse(visit, *args, **kwargs))
                              for k, v in self.data[2].items())
            return normalize(Expr(self.op, (func, operands, kwoperands)))
        elif self.op is Op.INDEXING:
            obj = self.data[0]
            obj = (obj.traverse(visit, *args, **kwargs)
                   if isinstance(obj, Expr) else obj)
            indices = tuple(index.traverse(visit, *args, **kwargs)
                            for index in self.data[1:])
            return normalize(Expr(self.op, (obj,) + indices))
        elif self.op in (Op.REF, Op.DEREF):
            return normalize(Expr(self.op,
                                  self.data.traverse(visit, *args, **kwargs)))
        elif self.op is Op.RELATIONAL:
            rop, left, right = self.data
            left = left.traverse(visit, *args, **kwargs)
            right = right.traverse(visit, *args, **kwargs)
            return normalize(Expr(self.op, (rop, left, right)))
        raise NotImplementedError(f'traverse method for {self.op}')

    def contains(self, other):
        """Check if self contains other.
        """
        found = []

        def visit(expr, found=found):
            if found:
                return expr
            elif expr == other:
                found.append(1)
                return expr

        self.traverse(visit)

        return len(found) != 0

    def symbols(self):
        """Return a set of symbols contained in self.
        """
        found = set()

        def visit(expr, found=found):
            if expr.op is Op.SYMBOL:
                found.add(expr)

        self.traverse(visit)

        return found

    def polynomial_atoms(self):
        """Return a set of expressions used as atoms in polynomial self.
        """
        found = set()

        def visit(expr, found=found):
            if expr.op is Op.FACTORS:
                for b in expr.data:
                    b.traverse(visit)
                return expr
            if expr.op in (Op.TERMS, Op.COMPLEX):
                return
            if expr.op is Op.APPLY and isinstance(expr.data[0], ArithOp):
                if expr.data[0] is ArithOp.POW:
                    expr.data[1][0].traverse(visit)
                    return expr
                return
            if expr.op in (Op.INTEGER, Op.REAL):
                return expr

            found.add(expr)

            if expr.op in (Op.INDEXING, Op.APPLY):
                return expr

        self.traverse(visit)

        return found

    def linear_solve(self, symbol):
        """Return a, b such that a * symbol + b == self.

        If self is not linear with respect to symbol, raise RuntimeError.
        """
        b = self.substitute({symbol: as_number(0)})
        ax = self - b
        a = ax.substitute({symbol: as_number(1)})

        zero, _ = as_numer_denom(a * symbol - ax)

        if zero != as_number(0):
            raise RuntimeError(f'not a {symbol}-linear equation:'
                               f' {a} * {symbol} + {b} == {self}')
        return a, b


def normalize(obj):
    """Normalize Expr and apply basic evaluation methods.
    """
    if not isinstance(obj, Expr):
        return obj

    if obj.op is Op.TERMS:
        d = {}
        for t, c in obj.data.items():
            if c == 0:
                continue
            if t.op is Op.COMPLEX and c != 1:
                t = t * c
                c = 1
            if t.op is Op.TERMS:
                for t1, c1 in t.data.items():
                    _pairs_add(d, t1, c1 * c)
            else:
                _pairs_add(d, t, c)
        if len(d) == 0:
            # TODO: determine correct kind
            return as_number(0)
        elif len(d) == 1:
            (t, c), = d.items()
            if c == 1:
                return t
        return Expr(Op.TERMS, d)

    if obj.op is Op.FACTORS:
        coeff = 1
        d = {}
        for b, e in obj.data.items():
            if e == 0:
                continue
            if b.op is Op.TERMS and isinstance(e, integer_types) and e > 1:
                # expand integer powers of sums
                b = b * (b ** (e - 1))
                e = 1

            if b.op in (Op.INTEGER, Op.REAL):
                if e == 1:
                    coeff *= b.data[0]
                elif e > 0:
                    coeff *= b.data[0] ** e
                else:
                    _pairs_add(d, b, e)
            elif b.op is Op.FACTORS:
                if e > 0 and isinstance(e, integer_types):
                    for b1, e1 in b.data.items():
                        _pairs_add(d, b1, e1 * e)
                else:
                    _pairs_add(d, b, e)
            else:
                _pairs_add(d, b, e)
        if len(d) == 0 or coeff == 0:
            # TODO: determine correct kind
            assert isinstance(coeff, number_types)
            return as_number(coeff)
        elif len(d) == 1:
            (b, e), = d.items()
            if e == 1:
                t = b
            else:
                t = Expr(Op.FACTORS, d)
            if coeff == 1:
                return t
            return Expr(Op.TERMS, {t: coeff})
        elif coeff == 1:
            return Expr(Op.FACTORS, d)
        else:
            return Expr(Op.TERMS, {Expr(Op.FACTORS, d): coeff})

    if obj.op is Op.APPLY and obj.data[0] is ArithOp.DIV:
        dividend, divisor = obj.data[1]
        t1, c1 = as_term_coeff(dividend)
        t2, c2 = as_term_coeff(divisor)
        if isinstance(c1, integer_types) and isinstance(c2, integer_types):
            g = gcd(c1, c2)
            c1, c2 = c1//g, c2//g
        else:
            c1, c2 = c1/c2, 1

        if t1.op is Op.APPLY and t1.data[0] is ArithOp.DIV:
            numer = t1.data[1][0] * c1
            denom = t1.data[1][1] * t2 * c2
            return as_apply(ArithOp.DIV, numer, denom)

        if t2.op is Op.APPLY and t2.data[0] is ArithOp.DIV:
            numer = t2.data[1][1] * t1 * c1
            denom = t2.data[1][0] * c2
            return as_apply(ArithOp.DIV, numer, denom)

        d = dict(as_factors(t1).data)
        for b, e in as_factors(t2).data.items():
            _pairs_add(d, b, -e)
        numer, denom = {}, {}
        for b, e in d.items():
            if e > 0:
                numer[b] = e
            else:
                denom[b] = -e
        numer = normalize(Expr(Op.FACTORS, numer)) * c1
        denom = normalize(Expr(Op.FACTORS, denom)) * c2

        if denom.op in (Op.INTEGER, Op.REAL) and denom.data[0] == 1:
            # TODO: denom kind not used
            return numer
        return as_apply(ArithOp.DIV, numer, denom)

    if obj.op is Op.CONCAT:
        lst = [obj.data[0]]
        for s in obj.data[1:]:
            last = lst[-1]
            if (
                    last.op is Op.STRING
                    and s.op is Op.STRING
                    and last.data[0][0] in '"\''
                    and s.data[0][0] == last.data[0][-1]
            ):
                new_last = as_string(last.data[0][:-1] + s.data[0][1:],
                                     max(last.data[1], s.data[1]))
                lst[-1] = new_last
            else:
                lst.append(s)
        if len(lst) == 1:
            return lst[0]
        return Expr(Op.CONCAT, tuple(lst))

    if obj.op is Op.TERNARY:
        cond, expr1, expr2 = map(normalize, obj.data)
        if cond.op is Op.INTEGER:
            return expr1 if cond.data[0] else expr2
        return Expr(Op.TERNARY, (cond, expr1, expr2))

    return obj


def as_expr(obj):
    """Convert non-Expr objects to Expr objects.
    """
    if isinstance(obj, complex):
        return as_complex(obj.real, obj.imag)
    if isinstance(obj, number_types):
        return as_number(obj)
    if isinstance(obj, str):
        # STRING expression holds string with boundary quotes, hence
        # applying repr:
        return as_string(repr(obj))
    if isinstance(obj, tuple):
        return tuple(map(as_expr, obj))
    return obj


def as_symbol(obj):
    """Return object as SYMBOL expression (variable or unparsed expression).
    """
    return Expr(Op.SYMBOL, obj)


def as_number(obj, kind=4):
    """Return object as INTEGER or REAL constant.
    """
    if isinstance(obj, int):
        return Expr(Op.INTEGER, (obj, kind))
    if isinstance(obj, float):
        return Expr(Op.REAL, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op in (Op.INTEGER, Op.REAL):
            return obj
    raise OpError(f'cannot convert {obj} to INTEGER or REAL constant')


def as_integer(obj, kind=4):
    """Return object as INTEGER constant.
    """
    if isinstance(obj, int):
        return Expr(Op.INTEGER, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op is Op.INTEGER:
            return obj
    raise OpError(f'cannot convert {obj} to INTEGER constant')


def as_real(obj, kind=4):
    """Return object as REAL constant.
    """
    if isinstance(obj, int):
        return Expr(Op.REAL, (float(obj), kind))
    if isinstance(obj, float):
        return Expr(Op.REAL, (obj, kind))
    if isinstance(obj, Expr):
        if obj.op is Op.REAL:
            return obj
        elif obj.op is Op.INTEGER:
            return Expr(Op.REAL, (float(obj.data[0]), kind))
    raise OpError(f'cannot convert {obj} to REAL constant')


def as_string(obj, kind=1):
    """Return object as STRING expression (string literal constant).
    """
    return Expr(Op.STRING, (obj, kind))


def as_array(obj):
    """Return object as ARRAY expression (array constant).
    """
    if isinstance(obj, Expr):
        obj = obj,
    return Expr(Op.ARRAY, obj)


def as_complex(real, imag=0):
    """Return object as COMPLEX expression (complex literal constant).
    """
    return Expr(Op.COMPLEX, (as_expr(real), as_expr(imag)))


def as_apply(func, *args, **kwargs):
    """Return object as APPLY expression (function call, constructor, etc.)
    """
    return Expr(Op.APPLY,
                (func, tuple(map(as_expr, args)),
                 dict((k, as_expr(v)) for k, v in kwargs.items())))


def as_ternary(cond, expr1, expr2):
    """Return object as TERNARY expression (cond?expr1:expr2).
    """
    return Expr(Op.TERNARY, (cond, expr1, expr2))


def as_ref(expr):
    """Return object as referencing expression.
    """
    return Expr(Op.REF, expr)


def as_deref(expr):
    """Return object as dereferencing expression.
    """
    return Expr(Op.DEREF, expr)


def as_eq(left, right):
    return Expr(Op.RELATIONAL, (RelOp.EQ, left, right))


def as_ne(left, right):
    return Expr(Op.RELATIONAL, (RelOp.NE, left, right))


def as_lt(left, right):
    return Expr(Op.RELATIONAL, (RelOp.LT, left, right))


def as_le(left, right):
    return Expr(Op.RELATIONAL, (RelOp.LE, left, right))


def as_gt(left, right):
    return Expr(Op.RELATIONAL, (RelOp.GT, left, right))


def as_ge(left, right):
    return Expr(Op.RELATIONAL, (RelOp.GE, left, right))


def as_terms(obj):
    """Return expression as TERMS expression.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op is Op.TERMS:
            return obj
        if obj.op is Op.INTEGER:
            return Expr(Op.TERMS, {as_integer(1, obj.data[1]): obj.data[0]})
        if obj.op is Op.REAL:
            return Expr(Op.TERMS, {as_real(1, obj.data[1]): obj.data[0]})
        return Expr(Op.TERMS, {obj: 1})
    raise OpError(f'cannot convert {type(obj)} to terms Expr')


def as_factors(obj):
    """Return expression as FACTORS expression.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op is Op.FACTORS:
            return obj
        if obj.op is Op.TERMS:
            if len(obj.data) == 1:
                (term, coeff), = obj.data.items()
                if coeff == 1:
                    return Expr(Op.FACTORS, {term: 1})
                return Expr(Op.FACTORS, {term: 1, Expr.number(coeff): 1})
        if (obj.op is Op.APPLY
             and obj.data[0] is ArithOp.DIV
             and not obj.data[2]):
            return Expr(Op.FACTORS, {obj.data[1][0]: 1, obj.data[1][1]: -1})
        return Expr(Op.FACTORS, {obj: 1})
    raise OpError(f'cannot convert {type(obj)} to terms Expr')


def as_term_coeff(obj):
    """Return expression as term-coefficient pair.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op is Op.INTEGER:
            return as_integer(1, obj.data[1]), obj.data[0]
        if obj.op is Op.REAL:
            return as_real(1, obj.data[1]), obj.data[0]
        if obj.op is Op.TERMS:
            if len(obj.data) == 1:
                (term, coeff), = obj.data.items()
                return term, coeff
            # TODO: find common divisor of coefficients
        if obj.op is Op.APPLY and obj.data[0] is ArithOp.DIV:
            t, c = as_term_coeff(obj.data[1][0])
            return as_apply(ArithOp.DIV, t, obj.data[1][1]), c
        return obj, 1
    raise OpError(f'cannot convert {type(obj)} to term and coeff')


def as_numer_denom(obj):
    """Return expression as numer-denom pair.
    """
    if isinstance(obj, Expr):
        obj = normalize(obj)
        if obj.op in (Op.INTEGER, Op.REAL, Op.COMPLEX, Op.SYMBOL,
                      Op.INDEXING, Op.TERNARY):
            return obj, as_number(1)
        elif obj.op is Op.APPLY:
            if obj.data[0] is ArithOp.DIV and not obj.data[2]:
                numers, denoms = map(as_numer_denom, obj.data[1])
                return numers[0] * denoms[1], numers[1] * denoms[0]
            return obj, as_number(1)
        elif obj.op is Op.TERMS:
            numers, denoms = [], []
            for term, coeff in obj.data.items():
                n, d = as_numer_denom(term)
                n = n * coeff
                numers.append(n)
                denoms.append(d)
            numer, denom = as_number(0), as_number(1)
            for i in range(len(numers)):
                n = numers[i]
                for j in range(len(numers)):
                    if i != j:
                        n *= denoms[j]
                numer += n
                denom *= denoms[i]
            if denom.op in (Op.INTEGER, Op.REAL) and denom.data[0] < 0:
                numer, denom = -numer, -denom
            return numer, denom
        elif obj.op is Op.FACTORS:
            numer, denom = as_number(1), as_number(1)
            for b, e in obj.data.items():
                bnumer, bdenom = as_numer_denom(b)
                if e > 0:
                    numer *= bnumer ** e
                    denom *= bdenom ** e
                elif e < 0:
                    numer *= bdenom ** (-e)
                    denom *= bnumer ** (-e)
            return numer, denom
    raise OpError(f'cannot convert {type(obj)} to numer and denom')


def _counter():
    # Used internally to generate unique dummy symbols
    counter = 0
    while True:
        counter += 1
        yield counter


COUNTER = _counter()


def eliminate_quotes(s):
    """Replace quoted substrings of input string.

    Return a new string and a mapping of replacements.
    """
    d = {}

    def repl(m):
        kind, value = m.groups()[:2]
        if kind:
            # remove trailing underscore
            kind = kind[:-1]
        p = {"'": "SINGLE", '"': "DOUBLE"}[value[0]]
        k = f'{kind}@__f2py_QUOTES_{p}_{COUNTER.__next__()}@'
        d[k] = value
        return k

    new_s = re.sub(r'({kind}_|)({single_quoted}|{double_quoted})'.format(
        kind=r'\w[\w\d_]*',
        single_quoted=r"('([^'\\]|(\\.))*')",
        double_quoted=r'("([^"\\]|(\\.))*")'),
        repl, s)

    assert '"' not in new_s
    assert "'" not in new_s

    return new_s, d


def insert_quotes(s, d):
    """Inverse of eliminate_quotes.
    """
    for k, v in d.items():
        kind = k[:k.find('@')]
        if kind:
            kind += '_'
        s = s.replace(k, kind + v)
    return s


def replace_parenthesis(s):
    """Replace substrings of input that are enclosed in parenthesis.

    Return a new string and a mapping of replacements.
    """
    # Find a parenthesis pair that appears first.

    # Fortran deliminator are `(`, `)`, `[`, `]`, `(/', '/)`, `/`.
    # We don't handle `/` deliminator because it is not a part of an
    # expression.
    left, right = None, None
    mn_i = len(s)
    for left_, right_ in (('(/', '/)'),
                          '()',
                          '{}',  # to support C literal structs
                          '[]'):
        i = s.find(left_)
        if i == -1:
            continue
        if i < mn_i:
            mn_i = i
            left, right = left_, right_

    if left is None:
        return s, {}

    i = mn_i
    j = s.find(right, i)

    while s.count(left, i + 1, j) != s.count(right, i + 1, j):
        j = s.find(right, j + 1)
        if j == -1:
            raise ValueError(f'Mismatch of {left+right} parenthesis in {s!r}')

    p = {'(': 'ROUND', '[': 'SQUARE', '{': 'CURLY', '(/': 'ROUNDDIV'}[left]

    k = f'@__f2py_PARENTHESIS_{p}_{COUNTER.__next__()}@'
    v = s[i+len(left):j]
    r, d = replace_parenthesis(s[j+len(right):])
    d[k] = v
    return s[:i] + k + r, d


def _get_parenthesis_kind(s):
    assert s.startswith('@__f2py_PARENTHESIS_'), s
    return s.split('_')[4]


def unreplace_parenthesis(s, d):
    """Inverse of replace_parenthesis.
    """
    for k, v in d.items():
        p = _get_parenthesis_kind(k)
        left = dict(ROUND='(', SQUARE='[', CURLY='{', ROUNDDIV='(/')[p]
        right = dict(ROUND=')', SQUARE=']', CURLY='}', ROUNDDIV='/)')[p]
        s = s.replace(k, left + v + right)
    return s


def fromstring(s, language=Language.C):
    """Create an expression from a string.

    This is a "lazy" parser, that is, only arithmetic operations are
    resolved, non-arithmetic operations are treated as symbols.
    """
    r = _FromStringWorker(language=language).parse(s)
    if isinstance(r, Expr):
        return r
    raise ValueError(f'failed to parse `{s}` to Expr instance: got `{r}`')


class _Pair:
    # Internal class to represent a pair of expressions

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def substitute(self, symbols_map):
        left, right = self.left, self.right
        if isinstance(left, Expr):
            left = left.substitute(symbols_map)
        if isinstance(right, Expr):
            right = right.substitute(symbols_map)
        return _Pair(left, right)

    def __repr__(self):
        return f'{type(self).__name__}({self.left}, {self.right})'


class _FromStringWorker:

    def __init__(self, language=Language.C):
        self.original = None
        self.quotes_map = None
        self.language = language

    def finalize_string(self, s):
        return insert_quotes(s, self.quotes_map)

    def parse(self, inp):
        self.original = inp
        unquoted, self.quotes_map = eliminate_quotes(inp)
        return self.process(unquoted)

    def process(self, s, context='expr'):
        """Parse string within the given context.

        The context may define the result in case of ambiguous
        expressions. For instance, consider expressions `f(x, y)` and
        `(x, y) + (a, b)` where `f` is a function and pair `(x, y)`
        denotes complex number. Specifying context as "args" or
        "expr", the subexpression `(x, y)` will be parse to an
        argument list or to a complex number, respectively.
        """
        if isinstance(s, (list, tuple)):
            return type(s)(self.process(s_, context) for s_ in s)

        assert isinstance(s, str), (type(s), s)

        # replace subexpressions in parenthesis with f2py @-names
        r, raw_symbols_map = replace_parenthesis(s)
        r = r.strip()

        def restore(r):
            # restores subexpressions marked with f2py @-names
            if isinstance(r, (list, tuple)):
                return type(r)(map(restore, r))
            return unreplace_parenthesis(r, raw_symbols_map)

        # comma-separated tuple
        if ',' in r:
            operands = restore(r.split(','))
            if context == 'args':
                return tuple(self.process(operands))
            if context == 'expr':
                if len(operands) == 2:
                    # complex number literal
                    return as_complex(*self.process(operands))
            raise NotImplementedError(
                f'parsing comma-separated list (context={context}): {r}')

        # ternary operation
        m = re.match(r'\A([^?]+)[?]([^:]+)[:](.+)\Z', r)
        if m:
            assert context == 'expr', context
            oper, expr1, expr2 = restore(m.groups())
            oper = self.process(oper)
            expr1 = self.process(expr1)
            expr2 = self.process(expr2)
            return as_ternary(oper, expr1, expr2)

        # relational expression
        if self.language is Language.Fortran:
            m = re.match(
                r'\A(.+)\s*[.](eq|ne|lt|le|gt|ge)[.]\s*(.+)\Z', r, re.I)
        else:
            m = re.match(
                r'\A(.+)\s*([=][=]|[!][=]|[<][=]|[<]|[>][=]|[>])\s*(.+)\Z', r)
        if m:
            left, rop, right = m.groups()
            if self.language is Language.Fortran:
                rop = '.' + rop + '.'
            left, right = self.process(restore((left, right)))
            rop = RelOp.fromstring(rop, language=self.language)
            return Expr(Op.RELATIONAL, (rop, left, right))

        # keyword argument
        m = re.match(r'\A(\w[\w\d_]*)\s*[=](.*)\Z', r)
        if m:
            keyname, value = m.groups()
            value = restore(value)
            return _Pair(keyname, self.process(value))

        # addition/subtraction operations
        operands = re.split(r'((?<!\d[edED])[+-])', r)
        if len(operands) > 1:
            result = self.process(restore(operands[0] or '0'))
            for op, operand in zip(operands[1::2], operands[2::2]):
                operand = self.process(restore(operand))
                op = op.strip()
                if op == '+':
                    result += operand
                else:
                    assert op == '-'
                    result -= operand
            return result

        # string concatenate operation
        if self.language is Language.Fortran and '//' in r:
            operands = restore(r.split('//'))
            return Expr(Op.CONCAT,
                        tuple(self.process(operands)))

        # multiplication/division operations
        operands = re.split(r'(?<=[@\w\d_])\s*([*]|/)',
                            (r if self.language is Language.C
                             else r.replace('**', '@__f2py_DOUBLE_STAR@')))
        if len(operands) > 1:
            operands = restore(operands)
            if self.language is not Language.C:
                operands = [operand.replace('@__f2py_DOUBLE_STAR@', '**')
                            for operand in operands]
            # Expression is an arithmetic product
            result = self.process(operands[0])
            for op, operand in zip(operands[1::2], operands[2::2]):
                operand = self.process(operand)
                op = op.strip()
                if op == '*':
                    result *= operand
                else:
                    assert op == '/'
                    result /= operand
            return result

        # referencing/dereferencing
        if r.startswith(('*', '&')):
            op = {'*': Op.DEREF, '&': Op.REF}[r[0]]
            operand = self.process(restore(r[1:]))
            return Expr(op, operand)

        # exponentiation operations
        if self.language is not Language.C and '**' in r:
            operands = list(reversed(restore(r.split('**'))))
            result = self.process(operands[0])
            for operand in operands[1:]:
                operand = self.process(operand)
                result = operand ** result
            return result

        # int-literal-constant
        m = re.match(r'\A({digit_string})({kind}|)\Z'.format(
            digit_string=r'\d+',
            kind=r'_(\d+|\w[\w\d_]*)'), r)
        if m:
            value, _, kind = m.groups()
            if kind and kind.isdigit():
                kind = int(kind)
            return as_integer(int(value), kind or 4)

        # real-literal-constant
        m = re.match(r'\A({significant}({exponent}|)|\d+{exponent})({kind}|)\Z'
                     .format(
                         significant=r'[.]\d+|\d+[.]\d*',
                         exponent=r'[edED][+-]?\d+',
                         kind=r'_(\d+|\w[\w\d_]*)'), r)
        if m:
            value, _, _, kind = m.groups()
            if kind and kind.isdigit():
                kind = int(kind)
            value = value.lower()
            if 'd' in value:
                return as_real(float(value.replace('d', 'e')), kind or 8)
            return as_real(float(value), kind or 4)

        # string-literal-constant with kind parameter specification
        if r in self.quotes_map:
            kind = r[:r.find('@')]
            return as_string(self.quotes_map[r], kind or 1)

        # array constructor or literal complex constant or
        # parenthesized expression
        if r in raw_symbols_map:
            paren = _get_parenthesis_kind(r)
            items = self.process(restore(raw_symbols_map[r]),
                                 'expr' if paren == 'ROUND' else 'args')
            if paren == 'ROUND':
                if isinstance(items, Expr):
                    return items
            if paren in ['ROUNDDIV', 'SQUARE']:
                # Expression is a array constructor
                if isinstance(items, Expr):
                    items = (items,)
                return as_array(items)

        # function call/indexing
        m = re.match(r'\A(.+)\s*(@__f2py_PARENTHESIS_(ROUND|SQUARE)_\d+@)\Z',
                     r)
        if m:
            target, args, paren = m.groups()
            target = self.process(restore(target))
            args = self.process(restore(args)[1:-1], 'args')
            if not isinstance(args, tuple):
                args = args,
            if paren == 'ROUND':
                kwargs = dict((a.left, a.right) for a in args
                              if isinstance(a, _Pair))
                args = tuple(a for a in args if not isinstance(a, _Pair))
                # Warning: this could also be Fortran indexing operation..
                return as_apply(target, *args, **kwargs)
            else:
                # Expression is a C/Python indexing operation
                # (e.g. used in .pyf files)
                assert paren == 'SQUARE'
                return target[args]

        # Fortran standard conforming identifier
        m = re.match(r'\A\w[\w\d_]*\Z', r)
        if m:
            return as_symbol(r)

        # fall-back to symbol
        r = self.finalize_string(restore(r))
        ewarn(
            f'fromstring: treating {r!r} as symbol (original={self.original})')
        return as_symbol(r)
<|endoftext|>"""
crackfortran --- read fortran (77,90) code and extract declaration information.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.


Usage of crackfortran:
======================
Command line keys: -quiet,-verbose,-fix,-f77,-f90,-show,-h <pyffilename>
                   -m <module name for f77 routines>,--ignore-contains
Functions: crackfortran, crack2fortran
The following Fortran statements/constructions are supported
(or will be if needed):
   block data,byte,call,character,common,complex,contains,data,
   dimension,double complex,double precision,end,external,function,
   implicit,integer,intent,interface,intrinsic,
   logical,module,optional,parameter,private,public,
   program,real,(sequence?),subroutine,type,use,virtual,
   include,pythonmodule
Note: 'virtual' is mapped to 'dimension'.
Note: 'implicit integer (z) static (z)' is 'implicit static (z)' (this is minor bug).
Note: code after 'contains' will be ignored until its scope ends.
Note: 'common' statement is extended: dimensions are moved to variable definitions
Note: f2py directive: <commentchar>f2py<line> is read as <line>
Note: pythonmodule is introduced to represent Python module

Usage:
  `postlist=crackfortran(files)`
  `postlist` contains declaration information read from the list of files `files`.
  `crack2fortran(postlist)` returns a fortran code to be saved to pyf-file

  `postlist` has the following structure:
 *** it is a list of dictionaries containing `blocks':
     B = {'block','body','vars','parent_block'[,'name','prefix','args','result',
          'implicit','externals','interfaced','common','sortvars',
          'commonvars','note']}
     B['block'] = 'interface' | 'function' | 'subroutine' | 'module' |
                  'program' | 'block data' | 'type' | 'pythonmodule' |
                  'abstract interface'
     B['body'] --- list containing `subblocks' with the same structure as `blocks'
     B['parent_block'] --- dictionary of a parent block:
                             C['body'][<index>]['parent_block'] is C
     B['vars'] --- dictionary of variable definitions
     B['sortvars'] --- dictionary of variable definitions sorted by dependence (independent first)
     B['name'] --- name of the block (not if B['block']=='interface')
     B['prefix'] --- prefix string (only if B['block']=='function')
     B['args'] --- list of argument names if B['block']== 'function' | 'subroutine'
     B['result'] --- name of the return value (only if B['block']=='function')
     B['implicit'] --- dictionary {'a':<variable definition>,'b':...} | None
     B['externals'] --- list of variables being external
     B['interfaced'] --- list of variables being external and defined
     B['common'] --- dictionary of common blocks (list of objects)
     B['commonvars'] --- list of variables used in common blocks (dimensions are moved to variable definitions)
     B['from'] --- string showing the 'parents' of the current block
     B['use'] --- dictionary of modules used in current block:
         {<modulename>:{['only':<0|1>],['map':{<local_name1>:<use_name1>,...}]}}
     B['note'] --- list of LaTeX comments on the block
     B['f2pyenhancements'] --- optional dictionary
          {'threadsafe':'','fortranname':<name>,
           'callstatement':<C-expr>|<multi-line block>,
           'callprotoargument':<C-expr-list>,
           'usercode':<multi-line block>|<list of multi-line blocks>,
           'pymethoddef:<multi-line block>'
           }
     B['entry'] --- dictionary {entryname:argslist,..}
     B['varnames'] --- list of variable names given in the order of reading the
                       Fortran code, useful for derived types.
     B['saved_interface'] --- a string of scanned routine signature, defines explicit interface
 *** Variable definition is a dictionary
     D = B['vars'][<variable name>] =
     {'typespec'[,'attrspec','kindselector','charselector','=','typename']}
     D['typespec'] = 'byte' | 'character' | 'complex' | 'double complex' |
                     'double precision' | 'integer' | 'logical' | 'real' | 'type'
     D['attrspec'] --- list of attributes (e.g. 'dimension(<arrayspec>)',
                       'external','intent(in|out|inout|hide|c|callback|cache|aligned4|aligned8|aligned16)',
                       'optional','required', etc)
     K = D['kindselector'] = {['*','kind']} (only if D['typespec'] =
                         'complex' | 'integer' | 'logical' | 'real' )
     C = D['charselector'] = {['*','len','kind','f2py_len']}
                             (only if D['typespec']=='character')
     D['='] --- initialization expression string
     D['typename'] --- name of the type if D['typespec']=='type'
     D['dimension'] --- list of dimension bounds
     D['intent'] --- list of intent specifications
     D['depend'] --- list of variable names on which current variable depends on
     D['check'] --- list of C-expressions; if C-expr returns zero, exception is raised
     D['note'] --- list of LaTeX comments on the variable
 *** Meaning of kind/char selectors (few examples):
     D['typespec>']*K['*']
     D['typespec'](kind=K['kind'])
     character*C['*']
     character(len=C['len'],kind=C['kind'], f2py_len=C['f2py_len'])
     (see also fortran type declaration statement formats below)

Fortran 90 type declaration statement format (F77 is subset of F90)
====================================================================
(Main source: IBM XL Fortran 5.1 Language Reference Manual)
type declaration = <typespec> [[<attrspec>]::] <entitydecl>
<typespec> = byte                          |
             character[<charselector>]     |
             complex[<kindselector>]       |
             double complex                |
             double precision              |
             integer[<kindselector>]       |
             logical[<kindselector>]       |
             real[<kindselector>]          |
             type(<typename>)
<charselector> = * <charlen>               |
             ([len=]<len>[,[kind=]<kind>]) |
             (kind=<kind>[,len=<len>])
<kindselector> = * <intlen>                |
             ([kind=]<kind>)
<attrspec> = comma separated list of attributes.
             Only the following attributes are used in
             building up the interface:
                external
                (parameter --- affects '=' key)
                optional
                intent
             Other attributes are ignored.
<intentspec> = in | out | inout
<arrayspec> = comma separated list of dimension bounds.
<entitydecl> = <name> [[*<charlen>][(<arrayspec>)] | [(<arrayspec>)]*<charlen>]
                      [/<init_expr>/ | =<init_expr>] [,<entitydecl>]

In addition, the following attributes are used: check,depend,note

TODO:
    * Apply 'parameter' attribute (e.g. 'integer parameter :: i=2' 'real x(i)'
                                   -> 'real x(2)')
    The above may be solved by creating appropriate preprocessor program, for example.

"""
import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
try:
    import charset_normalizer
except ImportError:
    charset_normalizer = None

from . import __version__

# The environment provided by auxfuncs.py is needed for some calls to eval.
# As the needed functions cannot be determined by static inspection of the
# code, it is safest to use import * pending a major refactoring of f2py.
from .auxfuncs import *
from . import symbolic

f2py_version = __version__.version

# Global flags:
strictf77 = 1          # Ignore `!' comments unless line[0]=='!'
sourcecodeform = 'fix'  # 'fix','free'
quiet = 0              # Be verbose if 0 (Obsolete: not used any more)
verbose = 1            # Be quiet if 0, extra verbose if > 1.
tabchar = 4 * ' '
pyffilename = ''
f77modulename = ''
skipemptyends = 0      # for old F77 programs without 'program' statement
ignorecontains = 1
dolowercase = 1
debug = []

# Global variables
beginpattern = ''
currentfilename = ''
expectbegin = 1
f90modulevars = {}
filepositiontext = ''
gotnextfile = 1
groupcache = None
groupcounter = 0
grouplist = {groupcounter: []}
groupname = ''
include_paths = []
neededmodule = -1
onlyfuncs = []
previous_context = None
skipblocksuntil = -1
skipfuncs = []
skipfunctions = []
usermodules = []


def reset_global_f2py_vars():
    global groupcounter, grouplist, neededmodule, expectbegin
    global skipblocksuntil, usermodules, f90modulevars, gotnextfile
    global filepositiontext, currentfilename, skipfunctions, skipfuncs
    global onlyfuncs, include_paths, previous_context
    global strictf77, sourcecodeform, quiet, verbose, tabchar, pyffilename
    global f77modulename, skipemptyends, ignorecontains, dolowercase, debug

    # flags
    strictf77 = 1
    sourcecodeform = 'fix'
    quiet = 0
    verbose = 1
    tabchar = 4 * ' '
    pyffilename = ''
    f77modulename = ''
    skipemptyends = 0
    ignorecontains = 1
    dolowercase = 1
    debug = []
    # variables
    groupcounter = 0
    grouplist = {groupcounter: []}
    neededmodule = -1
    expectbegin = 1
    skipblocksuntil = -1
    usermodules = []
    f90modulevars = {}
    gotnextfile = 1
    filepositiontext = ''
    currentfilename = ''
    skipfunctions = []
    skipfuncs = []
    onlyfuncs = []
    include_paths = []
    previous_context = None


def outmess(line, flag=1):
    global filepositiontext

    if not verbose:
        return
    if not quiet:
        if flag:
            sys.stdout.write(filepositiontext)
        sys.stdout.write(line)

re._MAXCACHE = 50
defaultimplicitrules = {}
for c in "abcdefghopqrstuvwxyz$_":
    defaultimplicitrules[c] = {'typespec': 'real'}
for c in "ijklmn":
    defaultimplicitrules[c] = {'typespec': 'integer'}
badnames = {}
invbadnames = {}
for n in ['int', 'double', 'float', 'char', 'short', 'long', 'void', 'case', 'while',
          'return', 'signed', 'unsigned', 'if', 'for', 'typedef', 'sizeof', 'union',
          'struct', 'static', 'register', 'new', 'break', 'do', 'goto', 'switch',
          'continue', 'else', 'inline', 'extern', 'delete', 'const', 'auto',
          'len', 'rank', 'shape', 'index', 'slen', 'size', '_i',
          'max', 'min',
          'flen', 'fshape',
          'string', 'complex_double', 'float_double', 'stdin', 'stderr', 'stdout',
          'type', 'default']:
    badnames[n] = n + '_bn'
    invbadnames[n + '_bn'] = n


def rmbadname1(name):
    if name in badnames:
        errmess('rmbadname1: Replacing "%s" with "%s".\n' %
                (name, badnames[name]))
        return badnames[name]
    return name


def rmbadname(names):
    return [rmbadname1(_m) for _m in names]


def undo_rmbadname1(name):
    if name in invbadnames:
        errmess('undo_rmbadname1: Replacing "%s" with "%s".\n'
                % (name, invbadnames[name]))
        return invbadnames[name]
    return name


def undo_rmbadname(names):
    return [undo_rmbadname1(_m) for _m in names]


_has_f_header = re.compile(r'-\*-\s*fortran\s*-\*-', re.I).search
_has_f90_header = re.compile(r'-\*-\s*f90\s*-\*-', re.I).search
_has_fix_header = re.compile(r'-\*-\s*fix\s*-\*-', re.I).search
_free_f90_start = re.compile(r'[^c*]\s*[^\s\d\t]', re.I).match

# Extensions
COMMON_FREE_EXTENSIONS = ['.f90', '.f95', '.f03', '.f08']
COMMON_FIXED_EXTENSIONS = ['.for', '.ftn', '.f77', '.f']


def openhook(filename, mode):
    """Ensures that filename is opened with correct encoding parameter.

    This function uses charset_normalizer package, when available, for
    determining the encoding of the file to be opened. When charset_normalizer
    is not available, the function detects only UTF encodings, otherwise, ASCII
    encoding is used as fallback.
    """
    # Reads in the entire file. Robust detection of encoding.
    # Correctly handles comments or late stage unicode characters
    # gh-22871
    if charset_normalizer is not None:
        encoding = charset_normalizer.from_path(filename).best().encoding
    else:
        # hint: install charset_normalizer for correct encoding handling
        # No need to read the whole file for trying with startswith
        nbytes = min(32, os.path.getsize(filename))
        with open(filename, 'rb') as fhandle:
            raw = fhandle.read(nbytes)
            if raw.startswith(codecs.BOM_UTF8):
                encoding = 'UTF-8-SIG'
            elif raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
                encoding = 'UTF-32'
            elif raw.startswith((codecs.BOM_LE, codecs.BOM_BE)):
                encoding = 'UTF-16'
            else:
                # Fallback, without charset_normalizer
                encoding = 'ascii'
    return open(filename, mode, encoding=encoding)


def is_free_format(fname):
    """Check if file is in free format Fortran."""
    # f90 allows both fixed and free format, assuming fixed unless
    # signs of free format are detected.
    result = False
    if Path(fname).suffix.lower() in COMMON_FREE_EXTENSIONS:
        result = True
    with openhook(fname, 'r') as fhandle:
        line = fhandle.readline()
        n = 15  # the number of non-comment lines to scan for hints
        if _has_f_header(line):
            n = 0
        elif _has_f90_header(line):
            n = 0
            result = True
        while n > 0 and line:
            if line[0] != '!' and line.strip():
                n -= 1
                if (line[0] != '\t' and _free_f90_start(line[:5])) or line[-2:-1] == '&':
                    result = True
                    break
            line = fhandle.readline()
    return result


# Read fortran (77,90) code
def readfortrancode(ffile, dowithline=show, istop=1):
    """
    Read fortran codes from files and
     1) Get rid of comments, line continuations, and empty lines; lower cases.
     2) Call dowithline(line) on every line.
     3) Recursively call itself when statement \"include '<filename>'\" is met.
    """
    global gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77
    global beginpattern, quiet, verbose, dolowercase, include_paths

    if not istop:
        saveglobals = gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77,\
            beginpattern, quiet, verbose, dolowercase
    if ffile == []:
        return
    localdolowercase = dolowercase
    # cont: set to True when the content of the last line read
    # indicates statement continuation
    cont = False
    finalline = ''
    ll = ''
    includeline = re.compile(
        r'\s*include\s*(\'|")(?P<name>[^\'"]*)(\'|")', re.I)
    cont1 = re.compile(r'(?P<line>.*)&\s*\Z')
    cont2 = re.compile(r'(\s*&|)(?P<line>.*)')
    mline_mark = re.compile(r".*?'''")
    if istop:
        dowithline('', -1)
    ll, l1 = '', ''
    spacedigits = [' '] + [str(_m) for _m in range(10)]
    filepositiontext = ''
    fin = fileinput.FileInput(ffile, openhook=openhook)
    while True:
        try:
            l = fin.readline()
        except UnicodeDecodeError as msg:
            raise Exception(
                f'readfortrancode: reading {fin.filename()}#{fin.lineno()}'
                f' failed with\n{msg}.\nIt is likely that installing charset_normalizer'
                ' package will help f2py determine the input file encoding'
                ' correctly.')
        if not l:
            break
        if fin.isfirstline():
            filepositiontext = ''
            currentfilename = fin.filename()
            gotnextfile = 1
            l1 = l
            strictf77 = 0
            sourcecodeform = 'fix'
            ext = os.path.splitext(currentfilename)[1]
            if Path(currentfilename).suffix.lower() in COMMON_FIXED_EXTENSIONS and \
                    not (_has_f90_header(l) or _has_fix_header(l)):
                strictf77 = 1
            elif is_free_format(currentfilename) and not _has_fix_header(l):
                sourcecodeform = 'free'
            if strictf77:
                beginpattern = beginpattern77
            else:
                beginpattern = beginpattern90
            outmess('\tReading file %s (format:%s%s)\n'
                    % (repr(currentfilename), sourcecodeform,
                       strictf77 and ',strict' or ''))

        l = l.expandtabs().replace('\xa0', ' ')
        # Get rid of newline characters
        while not l == '':
            if l[-1] not in "\n\r\f":
                break
            l = l[:-1]
        # Do not lower for directives, gh-2547, gh-27697, gh-26681
        is_f2py_directive = False
        # Unconditionally remove comments
        (l, rl) = split_by_unquoted(l, '!')
        l += ' '
        if rl[:5].lower() == '!f2py':  # f2py directive
            l, _ = split_by_unquoted(l + 4 * ' ' + rl[5:], '!')
            is_f2py_directive = True
        if l.strip() == '':  # Skip empty line
            if sourcecodeform == 'free':
                # In free form, a statement continues in the next line
                # that is not a comment line [3.3.2.4^1], lines with
                # blanks are comment lines [3.3.2.3^1]. Hence, the
                # line continuation flag must retain its state.
                pass
            else:
                # In fixed form, statement continuation is determined
                # by a non-blank character at the 6-th position. Empty
                # line indicates a start of a new statement
                # [3.3.3.3^1]. Hence, the line continuation flag must
                # be reset.
                cont = False
            continue
        if sourcecodeform == 'fix':
            if l[0] in ['*', 'c', '!', 'C', '#']:
                if l[1:5].lower() == 'f2py':  # f2py directive
                    l = '     ' + l[5:]
                    is_f2py_directive = True
                else:  # Skip comment line
                    cont = False
                    is_f2py_directive = False
                    continue
            elif strictf77:
                if len(l) > 72:
                    l = l[:72]
            if l[0] not in spacedigits:
                raise Exception('readfortrancode: Found non-(space,digit) char '
                                'in the first column.\n\tAre you sure that '
                                'this code is in fix form?\n\tline=%s' % repr(l))

            if (not cont or strictf77) and (len(l) > 5 and not l[5] == ' '):
                # Continuation of a previous line
                ll = ll + l[6:]
                finalline = ''
                origfinalline = ''
            else:
                r = cont1.match(l)
                if r:
                    l = r.group('line') # Continuation follows ..
                if cont:
                    ll = ll + cont2.match(l).group('line')
                    finalline = ''
                    origfinalline = ''
                else:
                    # clean up line beginning from possible digits.
                    l = '     ' + l[5:]
                    # f2py directives are already stripped by this point
                    if localdolowercase:
                        finalline = ll.lower()
                    else:
                        finalline = ll
                    origfinalline = ll
                    ll = l

        elif sourcecodeform == 'free':
            if not cont and ext == '.pyf' and mline_mark.match(l):
                l = l + '\n'
                while True:
                    lc = fin.readline()
                    if not lc:
                        errmess(
                            'Unexpected end of file when reading multiline\n')
                        break
                    l = l + lc
                    if mline_mark.match(lc):
                        break
                l = l.rstrip()
            r = cont1.match(l)
            if r:
                l = r.group('line')  # Continuation follows ..
            if cont:
                ll = ll + cont2.match(l).group('line')
                finalline = ''
                origfinalline = ''
            else:
                if localdolowercase:
                    # only skip lowering for C style constructs
                    # gh-2547, gh-27697, gh-26681, gh-28014
                    finalline = ll.lower() if not (is_f2py_directive and iscstyledirective(ll)) else ll
                else:
                    finalline = ll
                origfinalline = ll
                ll = l
            cont = (r is not None)
        else:
            raise ValueError(
                "Flag sourcecodeform must be either 'fix' or 'free': %s" % repr(sourcecodeform))
        filepositiontext = 'Line #%d in %s:"%s"\n\t' % (
            fin.filelineno() - 1, currentfilename, l1)
        m = includeline.match(origfinalline)
        if m:
            fn = m.group('name')
            if os.path.isfile(fn):
                readfortrancode(fn, dowithline=dowithline, istop=0)
            else:
                include_dirs = [
                    os.path.dirname(currentfilename)] + include_paths
                foundfile = 0
                for inc_dir in include_dirs:
                    fn1 = os.path.join(inc_dir, fn)
                    if os.path.isfile(fn1):
                        foundfile = 1
                        readfortrancode(fn1, dowithline=dowithline, istop=0)
                        break
                if not foundfile:
                    outmess('readfortrancode: could not find include file %s in %s. Ignoring.\n' % (
                        repr(fn), os.pathsep.join(include_dirs)))
        else:
            dowithline(finalline)
        l1 = ll
    # Last line should never have an f2py directive anyway
    if localdolowercase:
        finalline = ll.lower()
    else:
        finalline = ll
    origfinalline = ll
    filepositiontext = 'Line #%d in %s:"%s"\n\t' % (
        fin.filelineno() - 1, currentfilename, l1)
    m = includeline.match(origfinalline)
    if m:
        fn = m.group('name')
        if os.path.isfile(fn):
            readfortrancode(fn, dowithline=dowithline, istop=0)
        else:
            include_dirs = [os.path.dirname(currentfilename)] + include_paths
            foundfile = 0
            for inc_dir in include_dirs:
                fn1 = os.path.join(inc_dir, fn)
                if os.path.isfile(fn1):
                    foundfile = 1
                    readfortrancode(fn1, dowithline=dowithline, istop=0)
                    break
            if not foundfile:
                outmess('readfortrancode: could not find include file %s in %s. Ignoring.\n' % (
                    repr(fn), os.pathsep.join(include_dirs)))
    else:
        dowithline(finalline)
    filepositiontext = ''
    fin.close()
    if istop:
        dowithline('', 1)
    else:
        gotnextfile, filepositiontext, currentfilename, sourcecodeform, strictf77,\
            beginpattern, quiet, verbose, dolowercase = saveglobals

# Crack line
beforethisafter = r'\s*(?P<before>%s(?=\s*(\b(%s)\b)))' + \
    r'\s*(?P<this>(\b(%s)\b))' + \
    r'\s*(?P<after>%s)\s*\Z'
##
fortrantypes = r'character|logical|integer|real|complex|double\s*(precision\s*(complex|)|complex)|type(?=\s*\([\w\s,=(*)]*\))|byte'
typespattern = re.compile(
    beforethisafter % ('', fortrantypes, fortrantypes, '.*'), re.I), 'type'
typespattern4implicit = re.compile(beforethisafter % (
    '', fortrantypes + '|static|automatic|undefined', fortrantypes + '|static|automatic|undefined', '.*'), re.I)
#
functionpattern = re.compile(beforethisafter % (
    r'([a-z]+[\w\s(=*+-/)]*?|)', 'function', 'function', '.*'), re.I), 'begin'
subroutinepattern = re.compile(beforethisafter % (
    r'[a-z\s]*?', 'subroutine', 'subroutine', '.*'), re.I), 'begin'
# modulepattern=re.compile(beforethisafter%('[a-z\s]*?','module','module','.*'),re.I),'begin'
#
groupbegins77 = r'program|block\s*data'
beginpattern77 = re.compile(
    beforethisafter % ('', groupbegins77, groupbegins77, '.*'), re.I), 'begin'
groupbegins90 = groupbegins77 + \
    r'|module(?!\s*procedure)|python\s*module|(abstract|)\s*interface|' + \
    r'type(?!\s*\()'
beginpattern90 = re.compile(
    beforethisafter % ('', groupbegins90, groupbegins90, '.*'), re.I), 'begin'
groupends = (r'end|endprogram|endblockdata|endmodule|endpythonmodule|'
             r'endinterface|endsubroutine|endfunction')
endpattern = re.compile(
    beforethisafter % ('', groupends, groupends, '.*'), re.I), 'end'
# block, the Fortran 2008 construct needs special handling in the rest of the file
endifs = r'end\s*(if|do|where|select|while|forall|associate|' + \
         r'critical|enum|team)'
endifpattern = re.compile(
    beforethisafter % (r'[\w]*?', endifs, endifs, '.*'), re.I), 'endif'
#
moduleprocedures = r'module\s*procedure'
moduleprocedurepattern = re.compile(
    beforethisafter % ('', moduleprocedures, moduleprocedures, '.*'), re.I), \
    'moduleprocedure'
implicitpattern = re.compile(
    beforethisafter % ('', 'implicit', 'implicit', '.*'), re.I), 'implicit'
dimensionpattern = re.compile(beforethisafter % (
    '', 'dimension|virtual', 'dimension|virtual', '.*'), re.I), 'dimension'
externalpattern = re.compile(
    beforethisafter % ('', 'external', 'external', '.*'), re.I), 'external'
optionalpattern = re.compile(
    beforethisafter % ('', 'optional', 'optional', '.*'), re.I), 'optional'
requiredpattern = re.compile(
    beforethisafter % ('', 'required', 'required', '.*'), re.I), 'required'
publicpattern = re.compile(
    beforethisafter % ('', 'public', 'public', '.*'), re.I), 'public'
privatepattern = re.compile(
    beforethisafter % ('', 'private', 'private', '.*'), re.I), 'private'
intrinsicpattern = re.compile(
    beforethisafter % ('', 'intrinsic', 'intrinsic', '.*'), re.I), 'intrinsic'
intentpattern = re.compile(beforethisafter % (
    '', 'intent|depend|note|check', 'intent|depend|note|check', r'\s*\(.*?\).*'), re.I), 'intent'
parameterpattern = re.compile(
    beforethisafter % ('', 'parameter', 'parameter', r'\s*\(.*'), re.I), 'parameter'
datapattern = re.compile(
    beforethisafter % ('', 'data', 'data', '.*'), re.I), 'data'
callpattern = re.compile(
    beforethisafter % ('', 'call', 'call', '.*'), re.I), 'call'
entrypattern = re.compile(
    beforethisafter % ('', 'entry', 'entry', '.*'), re.I), 'entry'
callfunpattern = re.compile(
    beforethisafter % ('', 'callfun', 'callfun', '.*'), re.I), 'callfun'
commonpattern = re.compile(
    beforethisafter % ('', 'common', 'common', '.*'), re.I), 'common'
usepattern = re.compile(
    beforethisafter % ('', 'use', 'use', '.*'), re.I), 'use'
containspattern = re.compile(
    beforethisafter % ('', 'contains', 'contains', ''), re.I), 'contains'
formatpattern = re.compile(
    beforethisafter % ('', 'format', 'format', '.*'), re.I), 'format'
# Non-fortran and f2py-specific statements
f2pyenhancementspattern = re.compile(beforethisafter % ('', 'threadsafe|fortranname|callstatement|callprotoargument|usercode|pymethoddef',
                                                        'threadsafe|fortranname|callstatement|callprotoargument|usercode|pymethoddef', '.*'), re.I | re.S), 'f2pyenhancements'
multilinepattern = re.compile(
    r"\s*(?P<before>''')(?P<this>.*?)(?P<after>''')\s*\Z", re.S), 'multiline'
##

def split_by_unquoted(line, characters):
    """
    Splits the line into (line[:i], line[i:]),
    where i is the index of first occurrence of one of the characters
    not within quotes, or len(line) if no such index exists
    """
    assert not (set('"\'') & set(characters)), "cannot split by unquoted quotes"
    r = re.compile(
        r"\A(?P<before>({single_quoted}|{double_quoted}|{not_quoted})*)"
        r"(?P<after>{char}.*)\Z".format(
            not_quoted="[^\"'{}]".format(re.escape(characters)),
            char="[{}]".format(re.escape(characters)),
            single_quoted=r"('([^'\\]|(\\.))*')",
            double_quoted=r'("([^"\\]|(\\.))*")'))
    m = r.match(line)
    if m:
        d = m.groupdict()
        return (d["before"], d["after"])
    return (line, "")

def _simplifyargs(argsline):
    a = []
    for n in markoutercomma(argsline).split('@,@'):
        for r in '(),':
            n = n.replace(r, '_')
        a.append(n)
    return ','.join(a)

crackline_re_1 = re.compile(r'\s*(?P<result>\b[a-z]+\w*\b)\s*=.*', re.I)
crackline_bind_1 = re.compile(r'\s*(?P<bind>\b[a-z]+\w*\b)\s*=.*', re.I)
crackline_bindlang = re.compile(r'\s*bind\(\s*(?P<lang>[^,]+)\s*,\s*name\s*=\s*"(?P<lang_name>[^"]+)"\s*\)', re.I)

def crackline(line, reset=0):
    """
    reset=-1  --- initialize
    reset=0   --- crack the line
    reset=1   --- final check if mismatch of blocks occurred

    Cracked data is saved in grouplist[0].
    """
    global beginpattern, groupcounter, groupname, groupcache, grouplist
    global filepositiontext, currentfilename, neededmodule, expectbegin
    global skipblocksuntil, skipemptyends, previous_context, gotnextfile

    _, has_semicolon = split_by_unquoted(line, ";")
    if has_semicolon and not (f2pyenhancementspattern[0].match(line) or
                               multilinepattern[0].match(line)):
        # XXX: non-zero reset values need testing
        assert reset == 0, repr(reset)
        # split line on unquoted semicolons
        line, semicolon_line = split_by_unquoted(line, ";")
        while semicolon_line:
            crackline(line, reset)
            line, semicolon_line = split_by_unquoted(semicolon_line[1:], ";")
        crackline(line, reset)
        return
    if reset < 0:
        groupcounter = 0
        groupname = {groupcounter: ''}
        groupcache = {groupcounter: {}}
        grouplist = {groupcounter: []}
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['block'] = ''
        groupcache[groupcounter]['name'] = ''
        neededmodule = -1
        skipblocksuntil = -1
        return
    if reset > 0:
        fl = 0
        if f77modulename and neededmodule == groupcounter:
            fl = 2
        while groupcounter > fl:
            outmess('crackline: groupcounter=%s groupname=%s\n' %
                    (repr(groupcounter), repr(groupname)))
            outmess(
                'crackline: Mismatch of blocks encountered. Trying to fix it by assuming "end" statement.\n')
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1
        if f77modulename and neededmodule == groupcounter:
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1  # end interface
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1  # end module
            neededmodule = -1
        return
    if line == '':
        return
    flag = 0
    for pat in [dimensionpattern, externalpattern, intentpattern, optionalpattern,
                requiredpattern,
                parameterpattern, datapattern, publicpattern, privatepattern,
                intrinsicpattern,
                endifpattern, endpattern,
                formatpattern,
                beginpattern, functionpattern, subroutinepattern,
                implicitpattern, typespattern, commonpattern,
                callpattern, usepattern, containspattern,
                entrypattern,
                f2pyenhancementspattern,
                multilinepattern,
                moduleprocedurepattern
                ]:
        m = pat[0].match(line)
        if m:
            break
        flag = flag + 1
    if not m:
        re_1 = crackline_re_1
        if 0 <= skipblocksuntil <= groupcounter:
            return
        if 'externals' in groupcache[groupcounter]:
            for name in groupcache[groupcounter]['externals']:
                if name in invbadnames:
                    name = invbadnames[name]
                if 'interfaced' in groupcache[groupcounter] and name in groupcache[groupcounter]['interfaced']:
                    continue
                m1 = re.match(
                    r'(?P<before>[^"]*)\b%s\b\s*@\(@(?P<args>[^@]*)@\)@.*\Z' % name, markouterparen(line), re.I)
                if m1:
                    m2 = re_1.match(m1.group('before'))
                    a = _simplifyargs(m1.group('args'))
                    if m2:
                        line = 'callfun %s(%s) result (%s)' % (
                            name, a, m2.group('result'))
                    else:
                        line = 'callfun %s(%s)' % (name, a)
                    m = callfunpattern[0].match(line)
                    if not m:
                        outmess(
                            'crackline: could not resolve function call for line=%s.\n' % repr(line))
                        return
                    analyzeline(m, 'callfun', line)
                    return
        if verbose > 1 or (verbose == 1 and currentfilename.lower().endswith('.pyf')):
            previous_context = None
            outmess('crackline:%d: No pattern for line\n' % (groupcounter))
        return
    elif pat[1] == 'end':
        if 0 <= skipblocksuntil < groupcounter:
            groupcounter = groupcounter - 1
            if skipblocksuntil <= groupcounter:
                return
        if groupcounter <= 0:
            raise Exception('crackline: groupcounter(=%s) is nonpositive. '
                            'Check the blocks.'
                            % (groupcounter))
        m1 = beginpattern[0].match(line)
        if (m1) and (not m1.group('this') == groupname[groupcounter]):
            raise Exception('crackline: End group %s does not match with '
                            'previous Begin group %s\n\t%s' %
                            (repr(m1.group('this')), repr(groupname[groupcounter]),
                             filepositiontext)
                            )
        if skipblocksuntil == groupcounter:
            skipblocksuntil = -1
        grouplist[groupcounter - 1].append(groupcache[groupcounter])
        grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
        del grouplist[groupcounter]
        groupcounter = groupcounter - 1
        if not skipemptyends:
            expectbegin = 1
    elif pat[1] == 'begin':
        if 0 <= skipblocksuntil <= groupcounter:
            groupcounter = groupcounter + 1
            return
        gotnextfile = 0
        analyzeline(m, pat[1], line)
        expectbegin = 0
    elif pat[1] == 'endif':
        pass
    elif pat[1] == 'moduleprocedure':
        analyzeline(m, pat[1], line)
    elif pat[1] == 'contains':
        if ignorecontains:
            return
        if 0 <= skipblocksuntil <= groupcounter:
            return
        skipblocksuntil = groupcounter
    else:
        if 0 <= skipblocksuntil <= groupcounter:
            return
        analyzeline(m, pat[1], line)


def markouterparen(line):
    l = ''
    f = 0
    for c in line:
        if c == '(':
            f = f + 1
            if f == 1:
                l = l + '@(@'
                continue
        elif c == ')':
            f = f - 1
            if f == 0:
                l = l + '@)@'
                continue
        l = l + c
    return l


def markoutercomma(line, comma=','):
    l = ''
    f = 0
    before, after = split_by_unquoted(line, comma + '()')
    l += before
    while after:
        if (after[0] == comma) and (f == 0):
            l += '@' + comma + '@'
        else:
            l += after[0]
            if after[0] == '(':
                f += 1
            elif after[0] == ')':
                f -= 1
        before, after = split_by_unquoted(after[1:], comma + '()')
        l += before
    assert not f, repr((f, line, l))
    return l

def unmarkouterparen(line):
    r = line.replace('@(@', '(').replace('@)@', ')')
    return r


def appenddecl(decl, decl2, force=1):
    if not decl:
        decl = {}
    if not decl2:
        return decl
    if decl is decl2:
        return decl
    for k in list(decl2.keys()):
        if k == 'typespec':
            if force or k not in decl:
                decl[k] = decl2[k]
        elif k == 'attrspec':
            for l in decl2[k]:
                decl = setattrspec(decl, l, force)
        elif k == 'kindselector':
            decl = setkindselector(decl, decl2[k], force)
        elif k == 'charselector':
            decl = setcharselector(decl, decl2[k], force)
        elif k in ['=', 'typename']:
            if force or k not in decl:
                decl[k] = decl2[k]
        elif k == 'note':
            pass
        elif k in ['intent', 'check', 'dimension', 'optional',
                   'required', 'depend']:
            errmess('appenddecl: "%s" not implemented.\n' % k)
        else:
            raise Exception('appenddecl: Unknown variable definition key: ' +
                            str(k))
    return decl

selectpattern = re.compile(
    r'\s*(?P<this>(@\(@.*?@\)@|\*[\d*]+|\*\s*@\(@.*?@\)@|))(?P<after>.*)\Z', re.I)
typedefpattern = re.compile(
    r'(?:,(?P<attributes>[\w(),]+))?(::)?(?P<name>\b[a-z$_][\w$]*\b)'
    r'(?:\((?P<params>[\w,]*)\))?\Z', re.I)
nameargspattern = re.compile(
    r'\s*(?P<name>\b[\w$]+\b)\s*(@\(@\s*(?P<args>[\w\s,]*)\s*@\)@|)\s*((result(\s*@\(@\s*(?P<result>\b[\w$]+\b)\s*@\)@|))|(bind\s*@\(@\s*(?P<bind>(?:(?!@\)@).)*)\s*@\)@))*\s*\Z', re.I)
operatorpattern = re.compile(
    r'\s*(?P<scheme>(operator|assignment))'
    r'@\(@\s*(?P<name>[^)]+)\s*@\)@\s*\Z', re.I)
callnameargspattern = re.compile(
    r'\s*(?P<name>\b[\w$]+\b)\s*@\(@\s*(?P<args>.*)\s*@\)@\s*\Z', re.I)
real16pattern = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\d*\.\d+))[dD]((?:[-+]?\d+)?)')
real8pattern = re.compile(
    r'([-+]?((?:\d+(?:\.\d*)?|\d*\.\d+))[eE]((?:[-+]?\d+)?)|(\d+\.\d*))')

_intentcallbackpattern = re.compile(r'intent\s*\(.*?\bcallback\b', re.I)


def _is_intent_callback(vdecl):
    for a in vdecl.get('attrspec', []):
        if _intentcallbackpattern.match(a):
            return 1
    return 0


def _resolvetypedefpattern(line):
    line = ''.join(line.split())  # removes whitespace
    m1 = typedefpattern.match(line)
    print(line, m1)
    if m1:
        attrs = m1.group('attributes')
        attrs = [a.lower() for a in attrs.split(',')] if attrs else []
        return m1.group('name'), attrs, m1.group('params')
    return None, [], None

def parse_name_for_bind(line):
    pattern = re.compile(r'bind\(\s*(?P<lang>[^,]+)(?:\s*,\s*name\s*=\s*["\'](?P<name>[^"\']+)["\']\s*)?\)', re.I)
    match = pattern.search(line)
    bind_statement = None
    if match:
        bind_statement = match.group(0)
        # Remove the 'bind' construct from the line.
        line = line[:match.start()] + line[match.end():]
    return line, bind_statement

def _resolvenameargspattern(line):
    line, bind_cname = parse_name_for_bind(line)
    line = markouterparen(line)
    m1 = nameargspattern.match(line)
    if m1:
        return m1.group('name'), m1.group('args'), m1.group('result'), bind_cname
    m1 = operatorpattern.match(line)
    if m1:
        name = m1.group('scheme') + '(' + m1.group('name') + ')'
        return name, [], None, None
    m1 = callnameargspattern.match(line)
    if m1:
        return m1.group('name'), m1.group('args'), None, None
    return None, [], None, None


def analyzeline(m, case, line):
    """
    Reads each line in the input file in sequence and updates global vars.

    Effectively reads and collects information from the input file to the
    global variable groupcache, a dictionary containing info about each part
    of the fortran module.

    At the end of analyzeline, information is filtered into the correct dict
    keys, but parameter values and dimensions are not yet interpreted.
    """
    global groupcounter, groupname, groupcache, grouplist, filepositiontext
    global currentfilename, f77modulename, neededinterface, neededmodule
    global expectbegin, gotnextfile, previous_context

    block = m.group('this')
    if case != 'multiline':
        previous_context = None
    if expectbegin and case not in ['begin', 'call', 'callfun', 'type'] \
       and not skipemptyends and groupcounter < 1:
        newname = os.path.basename(currentfilename).split('.')[0]
        outmess(
            'analyzeline: no group yet. Creating program group with name "%s".\n' % newname)
        gotnextfile = 0
        groupcounter = groupcounter + 1
        groupname[groupcounter] = 'program'
        groupcache[groupcounter] = {}
        grouplist[groupcounter] = []
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['block'] = 'program'
        groupcache[groupcounter]['name'] = newname
        groupcache[groupcounter]['from'] = 'fromsky'
        expectbegin = 0
    if case in ['begin', 'call', 'callfun']:
        # Crack line => block,name,args,result
        block = block.lower()
        if re.match(r'block\s*data', block, re.I):
            block = 'block data'
        elif re.match(r'python\s*module', block, re.I):
            block = 'python module'
        elif re.match(r'abstract\s*interface', block, re.I):
            block = 'abstract interface'
        if block == 'type':
            name, attrs, _ = _resolvetypedefpattern(m.group('after'))
            groupcache[groupcounter]['vars'][name] = dict(attrspec = attrs)
            args = []
            result = None
        else:
            name, args, result, bindcline = _resolvenameargspattern(m.group('after'))
        if name is None:
            if block == 'block data':
                name = '_BLOCK_DATA_'
            else:
                name = ''
            if block not in ['interface', 'block data', 'abstract interface']:
                outmess('analyzeline: No name/args pattern found for line.\n')

        previous_context = (block, name, groupcounter)
        if args:
            args = rmbadname([x.strip()
                              for x in markoutercomma(args).split('@,@')])
        else:
            args = []
        if '' in args:
            while '' in args:
                args.remove('')
            outmess(
                'analyzeline: argument list is malformed (missing argument).\n')

        # end of crack line => block,name,args,result
        needmodule = 0
        needinterface = 0

        if case in ['call', 'callfun']:
            needinterface = 1
            if 'args' not in groupcache[groupcounter]:
                return
            if name not in groupcache[groupcounter]['args']:
                return
            for it in grouplist[groupcounter]:
                if it['name'] == name:
                    return
            if name in groupcache[groupcounter]['interfaced']:
                return
            block = {'call': 'subroutine', 'callfun': 'function'}[case]
        if f77modulename and neededmodule == -1 and groupcounter <= 1:
            neededmodule = groupcounter + 2
            needmodule = 1
            if block not in ['interface', 'abstract interface']:
                needinterface = 1
        # Create new block(s)
        groupcounter = groupcounter + 1
        groupcache[groupcounter] = {}
        grouplist[groupcounter] = []
        if needmodule:
            if verbose > 1:
                outmess('analyzeline: Creating module block %s\n' %
                        repr(f77modulename), 0)
            groupname[groupcounter] = 'module'
            groupcache[groupcounter]['block'] = 'python module'
            groupcache[groupcounter]['name'] = f77modulename
            groupcache[groupcounter]['from'] = ''
            groupcache[groupcounter]['body'] = []
            groupcache[groupcounter]['externals'] = []
            groupcache[groupcounter]['interfaced'] = []
            groupcache[groupcounter]['vars'] = {}
            groupcounter = groupcounter + 1
            groupcache[groupcounter] = {}
            grouplist[groupcounter] = []
        if needinterface:
            if verbose > 1:
                outmess('analyzeline: Creating additional interface block (groupcounter=%s).\n' % (
                    groupcounter), 0)
            groupname[groupcounter] = 'interface'
            groupcache[groupcounter]['block'] = 'interface'
            groupcache[groupcounter]['name'] = 'unknown_interface'
            groupcache[groupcounter]['from'] = '%s:%s' % (
                groupcache[groupcounter - 1]['from'], groupcache[groupcounter - 1]['name'])
            groupcache[groupcounter]['body'] = []
            groupcache[groupcounter]['externals'] = []
            groupcache[groupcounter]['interfaced'] = []
            groupcache[groupcounter]['vars'] = {}
            groupcounter = groupcounter + 1
            groupcache[groupcounter] = {}
            grouplist[groupcounter] = []
        groupname[groupcounter] = block
        groupcache[groupcounter]['block'] = block
        if not name:
            name = 'unknown_' + block.replace(' ', '_')
        groupcache[groupcounter]['prefix'] = m.group('before')
        groupcache[groupcounter]['name'] = rmbadname1(name)
        groupcache[groupcounter]['result'] = result
        if groupcounter == 1:
            groupcache[groupcounter]['from'] = currentfilename
        else:
            if f77modulename and groupcounter == 3:
                groupcache[groupcounter]['from'] = '%s:%s' % (
                    groupcache[groupcounter - 1]['from'], currentfilename)
            else:
                groupcache[groupcounter]['from'] = '%s:%s' % (
                    groupcache[groupcounter - 1]['from'], groupcache[groupcounter - 1]['name'])
        for k in list(groupcache[groupcounter].keys()):
            if not groupcache[groupcounter][k]:
                del groupcache[groupcounter][k]

        groupcache[groupcounter]['args'] = args
        groupcache[groupcounter]['body'] = []
        groupcache[groupcounter]['externals'] = []
        groupcache[groupcounter]['interfaced'] = []
        groupcache[groupcounter]['vars'] = {}
        groupcache[groupcounter]['entry'] = {}
        # end of creation
        if block == 'type':
            groupcache[groupcounter]['varnames'] = []

        if case in ['call', 'callfun']:  # set parents variables
            if name not in groupcache[groupcounter - 2]['externals']:
                groupcache[groupcounter - 2]['externals'].append(name)
            groupcache[groupcounter]['vars'] = copy.deepcopy(
                groupcache[groupcounter - 2]['vars'])
            try:
                del groupcache[groupcounter]['vars'][name][
                    groupcache[groupcounter]['vars'][name]['attrspec'].index('external')]
            except Exception:
                pass
        if block in ['function', 'subroutine']:  # set global attributes
            # name is fortran name
            if bindcline:
                bindcdat = re.search(crackline_bindlang, bindcline)
                if bindcdat:
                    groupcache[groupcounter]['bindlang'] = {name : {}}
                    groupcache[groupcounter]['bindlang'][name]["lang"] = bindcdat.group('lang')
                    if bindcdat.group('lang_name'):
                        groupcache[groupcounter]['bindlang'][name]["name"] = bindcdat.group('lang_name')
            try:
                groupcache[groupcounter]['vars'][name] = appenddecl(
                    groupcache[groupcounter]['vars'][name], groupcache[groupcounter - 2]['vars'][''])
            except Exception:
                pass
            if case == 'callfun':  # return type
                if result and result in groupcache[groupcounter]['vars']:
                    if not name == result:
                        groupcache[groupcounter]['vars'][name] = appenddecl(
                            groupcache[groupcounter]['vars'][name], groupcache[groupcounter]['vars'][result])
            # if groupcounter>1: # name is interfaced
            try:
                groupcache[groupcounter - 2]['interfaced'].append(name)
            except Exception:
                pass
        if block == 'function':
            t = typespattern[0].match(m.group('before') + ' ' + name)
            if t:
                typespec, selector, attr, edecl = cracktypespec0(
                    t.group('this'), t.group('after'))
                updatevars(typespec, selector, attr, edecl)

        if case in ['call', 'callfun']:
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1  # end routine
            grouplist[groupcounter - 1].append(groupcache[groupcounter])
            grouplist[groupcounter - 1][-1]['body'] = grouplist[groupcounter]
            del grouplist[groupcounter]
            groupcounter = groupcounter - 1  # end interface

    elif case == 'entry':
        name, args, result, _= _resolvenameargspattern(m.group('after'))
        if name is not None:
            if args:
                args = rmbadname([x.strip()
                                  for x in markoutercomma(args).split('@,@')])
            else:
                args = []
            assert result is None, repr(result)
            groupcache[groupcounter]['entry'][name] = args
            previous_context = ('entry', name, groupcounter)
    elif case == 'type':
        typespec, selector, attr, edecl = cracktypespec0(
            block, m.group('after'))
        last_name = updatevars(typespec, selector, attr, edecl)
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case in ['dimension', 'intent', 'optional', 'required', 'external', 'public', 'private', 'intrinsic']:
        edecl = groupcache[groupcounter]['vars']
        ll = m.group('after').strip()
        i = ll.find('::')
        if i < 0 and case == 'intent':
            i = markouterparen(ll).find('@)@') - 2
            ll = ll[:i + 1] + '::' + ll[i + 1:]
            i = ll.find('::')
            if ll[i:] == '::' and 'args' in groupcache[groupcounter]:
                outmess('All arguments will have attribute %s%s\n' %
                        (m.group('this'), ll[:i]))
                ll = ll + ','.join(groupcache[groupcounter]['args'])
        if i < 0:
            i = 0
            pl = ''
        else:
            pl = ll[:i].strip()
            ll = ll[i + 2:]
        ch = markoutercomma(pl).split('@,@')
        if len(ch) > 1:
            pl = ch[0]
            outmess('analyzeline: cannot handle multiple attributes without type specification. Ignoring %r.\n' % (
                ','.join(ch[1:])))
        last_name = None

        for e in [x.strip() for x in markoutercomma(ll).split('@,@')]:
            m1 = namepattern.match(e)
            if not m1:
                if case in ['public', 'private']:
                    k = ''
                else:
                    print(m.groupdict())
                    outmess('analyzeline: no name pattern found in %s statement for %s. Skipping.\n' % (
                        case, repr(e)))
                    continue
            else:
                k = rmbadname1(m1.group('name'))
            if case in ['public', 'private'] and \
               (k == 'operator' or k == 'assignment'):
                k += m1.group('after')
            if k not in edecl:
                edecl[k] = {}
            if case == 'dimension':
                ap = case + m1.group('after')
            if case == 'intent':
                ap = m.group('this') + pl
                if _intentcallbackpattern.match(ap):
                    if k not in groupcache[groupcounter]['args']:
                        if groupcounter > 1:
                            if '__user__' not in groupcache[groupcounter - 2]['name']:
                                outmess(
                                    'analyzeline: missing __user__ module (could be nothing)\n')
                            # fixes ticket 1693
                            if k != groupcache[groupcounter]['name']:
                                outmess('analyzeline: appending intent(callback) %s'
                                        ' to %s arguments\n' % (k, groupcache[groupcounter]['name']))
                                groupcache[groupcounter]['args'].append(k)
                        else:
                            errmess(
                                'analyzeline: intent(callback) %s is ignored\n' % (k))
                    else:
                        errmess('analyzeline: intent(callback) %s is already'
                                ' in argument list\n' % (k))
            if case in ['optional', 'required', 'public', 'external', 'private', 'intrinsic']:
                ap = case
            if 'attrspec' in edecl[k]:
                edecl[k]['attrspec'].append(ap)
            else:
                edecl[k]['attrspec'] = [ap]
            if case == 'external':
                if groupcache[groupcounter]['block'] == 'program':
                    outmess('analyzeline: ignoring program arguments\n')
                    continue
                if k not in groupcache[groupcounter]['args']:
                    continue
                if 'externals' not in groupcache[groupcounter]:
                    groupcache[groupcounter]['externals'] = []
                groupcache[groupcounter]['externals'].append(k)
            last_name = k
        groupcache[groupcounter]['vars'] = edecl
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'moduleprocedure':
        groupcache[groupcounter]['implementedby'] = \
            [x.strip() for x in m.group('after').split(',')]
    elif case == 'parameter':
        edecl = groupcache[groupcounter]['vars']
        ll = m.group('after').strip()[1:-1]
        last_name = None
        for e in markoutercomma(ll).split('@,@'):
            try:
                k, initexpr = [x.strip() for x in e.split('=')]
            except Exception:
                outmess(
                    'analyzeline: could not extract name,expr in parameter statement "%s" of "%s"\n' % (e, ll))
                continue
            params = get_parameters(edecl)
            k = rmbadname1(k)
            if k not in edecl:
                edecl[k] = {}
            if '=' in edecl[k] and (not edecl[k]['='] == initexpr):
                outmess('analyzeline: Overwriting the value of parameter "%s" ("%s") with "%s".\n' % (
                    k, edecl[k]['='], initexpr))
            t = determineexprtype(initexpr, params)
            if t:
                if t.get('typespec') == 'real':
                    tt = list(initexpr)
                    for m in real16pattern.finditer(initexpr):
                        tt[m.start():m.end()] = list(
                            initexpr[m.start():m.end()].lower().replace('d', 'e'))
                    initexpr = ''.join(tt)
                elif t.get('typespec') == 'complex':
                    initexpr = initexpr[1:].lower().replace('d', 'e').\
                        replace(',', '+1j*(')
            try:
                v = eval(initexpr, {}, params)
            except (SyntaxError, NameError, TypeError) as msg:
                errmess('analyzeline: Failed to evaluate %r. Ignoring: %s\n'
                        % (initexpr, msg))
                continue
            edecl[k]['='] = repr(v)
            if 'attrspec' in edecl[k]:
                edecl[k]['attrspec'].append('parameter')
            else:
                edecl[k]['attrspec'] = ['parameter']
            last_name = k
        groupcache[groupcounter]['vars'] = edecl
        if last_name is not None:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'implicit':
        if m.group('after').strip().lower() == 'none':
            groupcache[groupcounter]['implicit'] = None
        elif m.group('after'):
            if 'implicit' in groupcache[groupcounter]:
                impl = groupcache[groupcounter]['implicit']
            else:
                impl = {}
            if impl is None:
                outmess(
                    'analyzeline: Overwriting earlier "implicit none" statement.\n')
                impl = {}
            for e in markoutercomma(m.group('after')).split('@,@'):
                decl = {}
                m1 = re.match(
                    r'\s*(?P<this>.*?)\s*(\(\s*(?P<after>[a-z-, ]+)\s*\)\s*|)\Z', e, re.I)
                if not m1:
                    outmess(
                        'analyzeline: could not extract info of implicit statement part "%s"\n' % (e))
                    continue
                m2 = typespattern4implicit.match(m1.group('this'))
                if not m2:
                    outmess(
                        'analyzeline: could not extract types pattern of implicit statement part "%s"\n' % (e))
                    continue
                typespec, selector, attr, edecl = cracktypespec0(
                    m2.group('this'), m2.group('after'))
                kindselect, charselect, typename = cracktypespec(
                    typespec, selector)
                decl['typespec'] = typespec
                decl['kindselector'] = kindselect
                decl['charselector'] = charselect
                decl['typename'] = typename
                for k in list(decl.keys()):
                    if not decl[k]:
                        del decl[k]
                for r in markoutercomma(m1.group('after')).split('@,@'):
                    if '-' in r:
                        try:
                            begc, endc = [x.strip() for x in r.split('-')]
                        except Exception:
                            outmess(
                                'analyzeline: expected "<char>-<char>" instead of "%s" in range list of implicit statement\n' % r)
                            continue
                    else:
                        begc = endc = r.strip()
                    if not len(begc) == len(endc) == 1:
                        outmess(
                            'analyzeline: expected "<char>-<char>" instead of "%s" in range list of implicit statement (2)\n' % r)
                        continue
                    for o in range(ord(begc), ord(endc) + 1):
                        impl[chr(o)] = decl
            groupcache[groupcounter]['implicit'] = impl
    elif case == 'data':
        ll = []
        dl = ''
        il = ''
        f = 0
        fc = 1
        inp = 0
        for c in m.group('after'):
            if not inp:
                if c == "'":
                    fc = not fc
                if c == '/' and fc:
                    f = f + 1
                    continue
            if c == '(':
                inp = inp + 1
            elif c == ')':
                inp = inp - 1
            if f == 0:
                dl = dl + c
            elif f == 1:
                il = il + c
            elif f == 2:
                dl = dl.strip()
                if dl.startswith(','):
                    dl = dl[1:].strip()
                ll.append([dl, il])
                dl = c
                il = ''
                f = 0
        if f == 2:
            dl = dl.strip()
            if dl.startswith(','):
                dl = dl[1:].strip()
            ll.append([dl, il])
        vars = groupcache[groupcounter].get('vars', {})
        last_name = None
        for l in ll:
            l[0], l[1] = l[0].strip(), l[1].strip()
            if l[0].startswith(','):
                l[0] = l[0][1:]
            if l[0].startswith('('):
                outmess('analyzeline: implied-DO list "%s" is not supported. Skipping.\n' % l[0])
                continue
            for idx, v in enumerate(rmbadname([x.strip() for x in markoutercomma(l[0]).split('@,@')])):
                if v.startswith('('):
                    outmess('analyzeline: implied-DO list "%s" is not supported. Skipping.\n' % v)
                    # XXX: subsequent init expressions may get wrong values.
                    # Ignoring since data statements are irrelevant for
                    # wrapping.
                    continue
                if '!' in l[1]:
                    # Fixes gh-24746 pyf generation
                    # XXX: This essentially ignores the value for generating the pyf which is fine:
                    # integer dimension(3) :: mytab
                    # common /mycom/ mytab
                    # Since in any case it is initialized in the Fortran code
                    outmess('Comment line in declaration "%s" is not supported. Skipping.\n' % l[1])
                    continue
                vars.setdefault(v, {})
                vtype = vars[v].get('typespec')
                vdim = getdimension(vars[v])
                matches = re.findall(r"\(.*?\)", l[1]) if vtype == 'complex' else l[1].split(',')
                try:
                    new_val = "(/{}/)".format(", ".join(matches)) if vdim else matches[idx]
                except IndexError:
                    # gh-24746
                    # Runs only if above code fails. Fixes the line
                    # DATA IVAR1, IVAR2, IVAR3, IVAR4, EVAR5 /4*0,0.0D0/
                    # by expanding to ['0', '0', '0', '0', '0.0d0']
                    if any("*" in m for m in matches):
                        expanded_list = []
                        for match in matches:
                            if "*" in match:
                                try:
                                    multiplier, value = match.split("*")
                                    expanded_list.extend([value.strip()] * int(multiplier))
                                except ValueError: # if int(multiplier) fails
                                    expanded_list.append(match.strip())
                            else:
                                expanded_list.append(match.strip())
                        matches = expanded_list
                    new_val = "(/{}/)".format(", ".join(matches)) if vdim else matches[idx]
                current_val = vars[v].get('=')
                if current_val and (current_val != new_val):
                    outmess('analyzeline: changing init expression of "%s" ("%s") to "%s"\n' % (v, current_val, new_val))
                vars[v]['='] = new_val
                last_name = v
        groupcache[groupcounter]['vars'] = vars
        if last_name:
            previous_context = ('variable', last_name, groupcounter)
    elif case == 'common':
        line = m.group('after').strip()
        if not line[0] == '/':
            line = '//' + line

        cl = []
        [_, bn, ol] = re.split('/', line, maxsplit=2)
        bn = bn.strip()
        if not bn:
            bn = '_BLNK_'
        cl.append([bn, ol])
        commonkey = {}
        if 'common' in groupcache[groupcounter]:
            commonkey = groupcache[groupcounter]['common']
        for c in cl:
            if c[0] not in commonkey:
                commonkey[c[0]] = []
            for i in [x.strip() for x in markoutercomma(c[1]).split('@,@')]:
                if i:
                    commonkey[c[0]].append(i)
        groupcache[groupcounter]['common'] = commonkey
        previous_context = ('common', bn, groupcounter)
    elif case == 'use':
        m1 = re.match(
            r'\A\s*(?P<name>\b\w+\b)\s*((,(\s*\bonly\b\s*:|(?P<notonly>))\s*(?P<list>.*))|)\s*\Z', m.group('after'), re.I)
        if m1:
            mm = m1.groupdict()
            if 'use' not in groupcache[groupcounter]:
                groupcache[groupcounter]['use'] = {}
            name = m1.group('name')
            groupcache[groupcounter]['use'][name] = {}
            isonly = 0
            if 'list' in mm and mm['list'] is not None:
                if 'notonly' in mm and mm['notonly'] is None:
                    isonly = 1
                groupcache[groupcounter]['use'][name]['only'] = isonly
                ll = [x.strip() for x in mm['list'].split(',')]
                rl = {}
                for l in ll:
                    if '=' in l:
                        m2 = re.match(
                            r'\A\s*(?P<local>\b\w+\b)\s*=\s*>\s*(?P<use>\b\w+\b)\s*\Z', l, re.I)
                        if m2:
                            rl[m2.group('local').strip()] = m2.group(
                                'use').strip()
                        else:
                            outmess(
                                'analyzeline: Not local=>use pattern found in %s\n' % repr(l))
                    else:
                        rl[l] = l
                    groupcache[groupcounter]['use'][name]['map'] = rl
            else:
                pass
        else:
            print(m.groupdict())
            outmess('analyzeline: Could not crack the use statement.\n')
    elif case in ['f2pyenhancements']:
        if 'f2pyenhancements' not in groupcache[groupcounter]:
            groupcache[groupcounter]['f2pyenhancements'] = {}
        d = groupcache[groupcounter]['f2pyenhancements']
        if m.group('this') == 'usercode' and 'usercode' in d:
            if isinstance(d['usercode'], str):
                d['usercode'] = [d['usercode']]
            d['usercode'].append(m.group('after'))
        else:
            d[m.group('this')] = m.group('after')
    elif case == 'multiline':
        if previous_context is None:
            if verbose:
                outmess('analyzeline: No context for multiline block.\n')
            return
        gc = groupcounter
        appendmultiline(groupcache[gc],
                        previous_context[:2],
                        m.group('this'))
    else:
        if verbose > 1:
            print(m.groupdict())
            outmess('analyzeline: No code implemented for line.\n')


def appendmultiline(group, context_name, ml):
    if 'f2pymultilines' not in group:
        group['f2pymultilines'] = {}
    d = group['f2pymultilines']
    if context_name not in d:
        d[context_name] = []
    d[context_name].append(ml)
    return


def cracktypespec0(typespec, ll):
    selector = None
    attr = None
    if re.match(r'double\s*complex', typespec, re.I):
        typespec = 'double complex'
    elif re.match(r'double\s*precision', typespec, re.I):
        typespec = 'double precision'
    else:
        typespec = typespec.strip().lower()
    m1 = selectpattern.match(markouterparen(ll))
    if not m1:
        outmess(
            'cracktypespec0: no kind/char_selector pattern found for line.\n')
        return
    d = m1.groupdict()
    for k in list(d.keys()):
        d[k] = unmarkouterparen(d[k])
    if typespec in ['complex', 'integer', 'logical', 'real', 'character', 'type']:
        selector = d['this']
        ll = d['after']
    i = ll.find('::')
    if i >= 0:
        attr = ll[:i].strip()
        ll = ll[i + 2:]
    return typespec, selector, attr, ll
#####
namepattern = re.compile(r'\s*(?P<name>\b\w+\b)\s*(?P<after>.*)\s*\Z', re.I)
kindselector = re.compile(
    r'\s*(\(\s*(kind\s*=)?\s*(?P<kind>.*)\s*\)|\*\s*(?P<kind2>.*?))\s*\Z', re.I)
charselector = re.compile(
    r'\s*(\((?P<lenkind>.*)\)|\*\s*(?P<charlen>.*))\s*\Z', re.I)
lenkindpattern = re.compile(
    r'\s*(kind\s*=\s*(?P<kind>.*?)\s*(@,@\s*len\s*=\s*(?P<len>.*)|)'
    r'|(len\s*=\s*|)(?P<len2>.*?)\s*(@,@\s*(kind\s*=\s*|)(?P<kind2>.*)'
    r'|(f2py_len\s*=\s*(?P<f2py_len>.*))|))\s*\Z', re.I)
lenarraypattern = re.compile(
    r'\s*(@\(@\s*(?!/)\s*(?P<array>.*?)\s*@\)@\s*\*\s*(?P<len>.*?)|(\*\s*(?P<len2>.*?)|)\s*(@\(@\s*(?!/)\s*(?P<array2>.*?)\s*@\)@|))\s*(=\s*(?P<init>.*?)|(@\(@|)/\s*(?P<init2>.*?)\s*/(@\)@|)|)\s*\Z', re.I)


def removespaces(expr):
    expr = expr.strip()
    if len(expr) <= 1:
        return expr
    expr2 = expr[0]
    for i in range(1, len(expr) - 1):
        if (expr[i] == ' ' and
            ((expr[i + 1] in "()[]{}=+-/* ") or
                (expr[i - 1] in "()[]{}=+-/* "))):
            continue
        expr2 = expr2 + expr[i]
    expr2 = expr2 + expr[-1]
    return expr2


def markinnerspaces(line):
    """
    The function replace all spaces in the input variable line which are 
    surrounded with quotation marks, with the triplet "@_@".

    For instance, for the input "a 'b c'" the function returns "a 'b@_@c'"

    Parameters
    ----------
    line : str

    Returns
    -------
    str

    """  
    fragment = ''
    inside = False
    current_quote = None
    escaped = ''
    for c in line:
        if escaped == '\\' and c in ['\\', '\'', '"']:
            fragment += c
            escaped = c
            continue
        if not inside and c in ['\'', '"']:
            current_quote = c
        if c == current_quote:
            inside = not inside
        elif c == ' ' and inside:
            fragment += '@_@'
            continue
        fragment += c
        escaped = c  # reset to non-backslash
    return fragment


def updatevars(typespec, selector, attrspec, entitydecl):
    """
    Returns last_name, the variable name without special chars, parenthesis
        or dimension specifiers.

    Alters groupcache to add the name, typespec, attrspec (and possibly value)
    of current variable.
    """
    global groupcache, groupcounter

    last_name = None
    kindselect, charselect, typename = cracktypespec(typespec, selector)
    # Clean up outer commas, whitespace and undesired chars from attrspec
    if attrspec:
        attrspec = [x.strip() for x in markoutercomma(attrspec).split('@,@')]
        l = []
        c = re.compile(r'(?P<start>[a-zA-Z]+)')
        for a in attrspec:
            if not a:
                continue
            m = c.match(a)
            if m:
                s = m.group('start').lower()
                a = s + a[len(s):]
            l.append(a)
        attrspec = l
    el = [x.strip() for x in markoutercomma(entitydecl).split('@,@')]
    el1 = []
    for e in el:
        for e1 in [x.strip() for x in markoutercomma(removespaces(markinnerspaces(e)), comma=' ').split('@ @')]:
            if e1:
                el1.append(e1.replace('@_@', ' '))
    for e in el1:
        m = namepattern.match(e)
        if not m:
            outmess(
                'updatevars: no name pattern found for entity=%s. Skipping.\n' % (repr(e)))
            continue
        ename = rmbadname1(m.group('name'))
        edecl = {}
        if ename in groupcache[groupcounter]['vars']:
            edecl = groupcache[groupcounter]['vars'][ename].copy()
            not_has_typespec = 'typespec' not in edecl
            if not_has_typespec:
                edecl['typespec'] = typespec
            elif typespec and (not typespec == edecl['typespec']):
                outmess('updatevars: attempt to change the type of "%s" ("%s") to "%s". Ignoring.\n' % (
                    ename, edecl['typespec'], typespec))
            if 'kindselector' not in edecl:
                edecl['kindselector'] = copy.copy(kindselect)
            elif kindselect:
                for k in list(kindselect.keys()):
                    if k in edecl['kindselector'] and (not kindselect[k] == edecl['kindselector'][k]):
                        outmess('updatevars: attempt to change the kindselector "%s" of "%s" ("%s") to "%s". Ignoring.\n' % (
                            k, ename, edecl['kindselector'][k], kindselect[k]))
                    else:
                        edecl['kindselector'][k] = copy.copy(kindselect[k])
            if 'charselector' not in edecl and charselect:
                if not_has_typespec:
                    edecl['charselector'] = charselect
                else:
                    errmess('updatevars:%s: attempt to change empty charselector to %r. Ignoring.\n'
                            % (ename, charselect))
            elif charselect:
                for k in list(charselect.keys()):
                    if k in edecl['charselector'] and (not charselect[k] == edecl['charselector'][k]):
                        outmess('updatevars: attempt to change the charselector "%s" of "%s" ("%s") to "%s". Ignoring.\n' % (
