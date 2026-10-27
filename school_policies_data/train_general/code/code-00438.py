            """Function to parse a file with C source code

            It takes the filename as an attribute and creates a Clang AST
            Translation Unit parsing the file.
            Then the transformation function is called on the translation unit,
            whose results are collected into a list which is returned by the
            function.

            Parameters
            ==========

            filename : string
                Path to the C file to be parsed

            flags: list
                Arguments to be passed to Clang while parsing the C code

            Returns
            =======

            py_nodes: list
                A list of SymPy AST nodes

            """
            filepath = os.path.abspath(filename)
            self.tu = self.index.parse(
                filepath,
                args=flags,
                options=cin.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            for child in self.tu.cursor.get_children():
                if child.kind == cin.CursorKind.VAR_DECL or child.kind == cin.CursorKind.FUNCTION_DECL:
                    self._py_nodes.append(self.transform(child))
            return self._py_nodes

        def parse_str(self, source, flags):
            """Function to parse a string with C source code

            It takes the source code as an attribute, stores it in a temporary
            file and creates a Clang AST Translation Unit parsing the file.
            Then the transformation function is called on the translation unit,
            whose results are collected into a list which is returned by the
            function.

            Parameters
            ==========

            source : string
                A string containing the C source code to be parsed

            flags: list
                Arguments to be passed to Clang while parsing the C code

            Returns
            =======

            py_nodes: list
                A list of SymPy AST nodes

            """
            file = tempfile.NamedTemporaryFile(mode = 'w+', suffix = '.cpp')
            file.write(source)
            file.flush()
            file.seek(0)
            self.tu = self.index.parse(
                file.name,
                args=flags,
                options=cin.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            file.close()
            for child in self.tu.cursor.get_children():
                if child.kind == cin.CursorKind.VAR_DECL or child.kind == cin.CursorKind.FUNCTION_DECL:
                    self._py_nodes.append(self.transform(child))
            return self._py_nodes

        def transform(self, node):
            """Transformation Function for Clang AST nodes

            It determines the kind of node and calls the respective
            transformation function for that node.

            Raises
            ======

            NotImplementedError : if the transformation for the provided node
            is not implemented

            """
            handler = getattr(self, 'transform_%s' % node.kind.name.lower(), None)

            if handler is None:
                print(
                    "Ignoring node of type %s (%s)" % (
                        node.kind,
                        ' '.join(
                            t.spelling for t in node.get_tokens())
                        ),
                    file=sys.stderr
                )

            return handler(node)

        def transform_var_decl(self, node):
            """Transformation Function for Variable Declaration

            Used to create nodes for variable declarations and assignments with
            values or function call for the respective nodes in the clang AST

            Returns
            =======

            A variable node as Declaration, with the initial value if given

            Raises
            ======

            NotImplementedError : if called for data types not currently
            implemented

            Notes
            =====

            The function currently supports following data types:

            Boolean:
                bool, _Bool

            Integer:
                8-bit: signed char and unsigned char
                16-bit: short, short int, signed short,
                    signed short int, unsigned short, unsigned short int
                32-bit: int, signed int, unsigned int
                64-bit: long, long int, signed long,
                    signed long int, unsigned long, unsigned long int

            Floating point:
                Single Precision: float
                Double Precision: double
                Extended Precision: long double

            """
            if node.type.kind in self._data_types["int"]:
                type = self._data_types["int"][node.type.kind]
            elif node.type.kind in self._data_types["float"]:
                type = self._data_types["float"][node.type.kind]
            elif node.type.kind in self._data_types["bool"]:
                type = self._data_types["bool"][node.type.kind]
            else:
                raise NotImplementedError("Only bool, int "
                    "and float are supported")
            try:
                children = node.get_children()
                child = next(children)

                #ignoring namespace and type details for the variable
                while child.kind == cin.CursorKind.NAMESPACE_REF or child.kind == cin.CursorKind.TYPE_REF:
                    child = next(children)

                val = self.transform(child)

                supported_rhs = [
                    cin.CursorKind.INTEGER_LITERAL,
                    cin.CursorKind.FLOATING_LITERAL,
                    cin.CursorKind.UNEXPOSED_EXPR,
                    cin.CursorKind.BINARY_OPERATOR,
                    cin.CursorKind.PAREN_EXPR,
                    cin.CursorKind.UNARY_OPERATOR,
                    cin.CursorKind.CXX_BOOL_LITERAL_EXPR
                ]

                if child.kind in supported_rhs:
                    if isinstance(val, str):
                        value = Symbol(val)
                    elif isinstance(val, bool):
                        if node.type.kind in self._data_types["int"]:
                            value = Integer(0) if val == False else Integer(1)
                        elif node.type.kind in self._data_types["float"]:
                            value = Float(0.0) if val == False else Float(1.0)
                        elif node.type.kind in self._data_types["bool"]:
                            value = sympify(val)
                    elif isinstance(val, (Integer, int, Float, float)):
                        if node.type.kind in self._data_types["int"]:
                            value = Integer(val)
                        elif node.type.kind in self._data_types["float"]:
                            value = Float(val)
                        elif node.type.kind in self._data_types["bool"]:
                            value = sympify(bool(val))
                    else:
                        value = val

                    return Variable(
                    node.spelling
                    ).as_Declaration(
                        type = type,
                        value = value
                    )

                elif child.kind == cin.CursorKind.CALL_EXPR:
                    return Variable(
                        node.spelling
                        ).as_Declaration(
                            value = val
                        )

                else:
                    raise NotImplementedError("Given "
                        "variable declaration \"{}\" "
                        "is not possible to parse yet!"
                        .format(" ".join(
                            t.spelling for t in node.get_tokens()
                            )
                        ))

            except StopIteration:
                return Variable(
                node.spelling
                ).as_Declaration(
                    type = type
                )

        def transform_function_decl(self, node):
            """Transformation Function For Function Declaration

            Used to create nodes for function declarations and definitions for
            the respective nodes in the clang AST

            Returns
            =======

            function : Codegen AST node
                - FunctionPrototype node if function body is not present
                - FunctionDefinition node if the function body is present


            """

            if node.result_type.kind in self._data_types["int"]:
                ret_type = self._data_types["int"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["float"]:
                ret_type = self._data_types["float"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["bool"]:
                ret_type = self._data_types["bool"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["void"]:
                ret_type = self._data_types["void"][node.result_type.kind]
            else:
                raise NotImplementedError("Only void, bool, int "
                    "and float are supported")
            body = []
            param = []

            # Subsequent nodes will be the parameters for the function.
            for child in node.get_children():
                decl = self.transform(child)
                if child.kind == cin.CursorKind.PARM_DECL:
                    param.append(decl)
                elif child.kind == cin.CursorKind.COMPOUND_STMT:
                    for val in decl:
                        body.append(val)
                else:
                    body.append(decl)

            if body == []:
                function = FunctionPrototype(
                    return_type = ret_type,
                    name = node.spelling,
                    parameters = param
                )
            else:
                function = FunctionDefinition(
                    return_type = ret_type,
                    name = node.spelling,
                    parameters = param,
                    body = body
                )
            return function

        def transform_parm_decl(self, node):
            """Transformation function for Parameter Declaration

            Used to create parameter nodes for the required functions for the
            respective nodes in the clang AST

            Returns
            =======

            param : Codegen AST Node
                Variable node with the value and type of the variable

            Raises
            ======

            ValueError if multiple children encountered in the parameter node

            """
            if node.type.kind in self._data_types["int"]:
                type = self._data_types["int"][node.type.kind]
            elif node.type.kind in self._data_types["float"]:
                type = self._data_types["float"][node.type.kind]
            elif node.type.kind in self._data_types["bool"]:
                type = self._data_types["bool"][node.type.kind]
            else:
                raise NotImplementedError("Only bool, int "
                    "and float are supported")
            try:
                children = node.get_children()
                child = next(children)

                # Any namespace nodes can be ignored
                while child.kind in [cin.CursorKind.NAMESPACE_REF,
                                     cin.CursorKind.TYPE_REF,
                                     cin.CursorKind.TEMPLATE_REF]:
                    child = next(children)

                # If there is a child, it is the default value of the parameter.
                lit = self.transform(child)
                if node.type.kind in self._data_types["int"]:
                    val = Integer(lit)
                elif node.type.kind in self._data_types["float"]:
                    val = Float(lit)
                elif node.type.kind in self._data_types["bool"]:
                    val = sympify(bool(lit))
                else:
                    raise NotImplementedError("Only bool, int "
                        "and float are supported")

                param = Variable(
                    node.spelling
                ).as_Declaration(
                    type = type,
                    value = val
                )
            except StopIteration:
                param = Variable(
                    node.spelling
                ).as_Declaration(
                    type = type
                )

            try:
                self.transform(next(children))
                raise ValueError("Can't handle multiple children on parameter")
            except StopIteration:
                pass

            return param

        def transform_integer_literal(self, node):
            """Transformation function for integer literal

            Used to get the value and type of the given integer literal.

            Returns
            =======

            val : list
                List with two arguments type and Value
                type contains the type of the integer
                value contains the value stored in the variable

            Notes
            =====

            Only Base Integer type supported for now

            """
            try:
                value = next(node.get_tokens()).spelling
            except StopIteration:
                # No tokens
                value = node.literal
            return int(value)

        def transform_floating_literal(self, node):
            """Transformation function for floating literal

            Used to get the value and type of the given floating literal.

            Returns
            =======

            val : list
                List with two arguments type and Value
                type contains the type of float
                value contains the value stored in the variable

            Notes
            =====

            Only Base Float type supported for now

            """
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                # No tokens
                value = node.literal
            return float(value)

        def transform_string_literal(self, node):
            #TODO: No string type in AST
            #type =
            #try:
            #    value = next(node.get_tokens()).spelling
            #except (StopIteration, ValueError):
                # No tokens
            #    value = node.literal
            #val = [type, value]
            #return val
            pass

        def transform_character_literal(self, node):
            """Transformation function for character literal

            Used to get the value of the given character literal.

            Returns
            =======

            val : int
                val contains the ascii value of the character literal

            Notes
            =====

            Only for cases where character is assigned to a integer value,
            since character literal is not in SymPy AST

            """
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                # No tokens
                value = node.literal
            return ord(str(value[1]))

        def transform_cxx_bool_literal_expr(self, node):
            """Transformation function for boolean literal

            Used to get the value of the given boolean literal.

            Returns
            =======

            value : bool
                value contains the boolean value of the variable

            """
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                value = node.literal
            return True if value == 'true' else False

        def transform_unexposed_decl(self,node):
            """Transformation function for unexposed declarations"""
            pass

        def transform_unexposed_expr(self, node):
            """Transformation function for unexposed expression

            Unexposed expressions are used to wrap float, double literals and
            expressions

            Returns
            =======

            expr : Codegen AST Node
                the result from the wrapped expression

            None : NoneType
                No children are found for the node

            Raises
            ======

            ValueError if the expression contains multiple children

            """
            # Ignore unexposed nodes; pass whatever is the first
            # (and should be only) child unaltered.
            try:
                children = node.get_children()
                expr = self.transform(next(children))
            except StopIteration:
                return None

            try:
                next(children)
                raise ValueError("Unexposed expression has > 1 children.")
            except StopIteration:
                pass

            return expr

        def transform_decl_ref_expr(self, node):
            """Returns the name of the declaration reference"""
            return node.spelling

        def transform_call_expr(self, node):
            """Transformation function for a call expression

            Used to create function call nodes for the function calls present
            in the C code

            Returns
            =======

            FunctionCall : Codegen AST Node
                FunctionCall node with parameters if any parameters are present

            """
            param = []
            children = node.get_children()
            child = next(children)

            while child.kind == cin.CursorKind.NAMESPACE_REF:
                child = next(children)
            while child.kind == cin.CursorKind.TYPE_REF:
                child = next(children)

            first_child = self.transform(child)
            try:
                for child in children:
                    arg = self.transform(child)
                    if child.kind == cin.CursorKind.INTEGER_LITERAL:
                        param.append(Integer(arg))
                    elif child.kind == cin.CursorKind.FLOATING_LITERAL:
                        param.append(Float(arg))
                    else:
                        param.append(arg)
                return FunctionCall(first_child, param)

            except StopIteration:
                return FunctionCall(first_child)

        def transform_return_stmt(self, node):
            """Returns the Return Node for a return statement"""
            return Return(next(node.get_children()).spelling)

        def transform_compound_stmt(self, node):
            """Transformation function for compound statements

            Returns
            =======

            expr : list
                list of Nodes for the expressions present in the statement

            None : NoneType
                if the compound statement is empty

            """
            expr = []
            children = node.get_children()

            for child in children:
                expr.append(self.transform(child))
            return expr

        def transform_decl_stmt(self, node):
            """Transformation function for declaration statements

            These statements are used to wrap different kinds of declararions
            like variable or function declaration
            The function calls the transformer function for the child of the
            given node

            Returns
            =======

            statement : Codegen AST Node
                contains the node returned by the children node for the type of
                declaration

            Raises
            ======

            ValueError if multiple children present

            """
            try:
                children = node.get_children()
                statement = self.transform(next(children))
            except StopIteration:
                pass

            try:
                self.transform(next(children))
                raise ValueError("Don't know how to handle multiple statements")
            except StopIteration:
                pass

            return statement

        def transform_paren_expr(self, node):
            """Transformation function for Parenthesized expressions

            Returns the result from its children nodes

            """
            return self.transform(next(node.get_children()))

        def transform_compound_assignment_operator(self, node):
            """Transformation function for handling shorthand operators

            Returns
            =======

            augmented_assignment_expression: Codegen AST node
                    shorthand assignment expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If the shorthand operator for bitwise operators
                (~=, ^=, &=, |=, <<=, >>=) is encountered

            """
            return self.transform_binary_operator(node)

        def transform_unary_operator(self, node):
            """Transformation function for handling unary operators

            Returns
            =======

            unary_expression: Codegen AST node
                    simplified unary expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If dereferencing operator(*), address operator(&) or
                bitwise NOT operator(~) is encountered

            """
            # supported operators list
            operators_list = ['+', '-', '++', '--', '!']
            tokens = list(node.get_tokens())

            # it can be either pre increment/decrement or any other operator from the list
            if tokens[0].spelling in operators_list:
                child = self.transform(next(node.get_children()))
                # (decl_ref) e.g.; int a = ++b; or simply ++b;
                if isinstance(child, str):
                    if tokens[0].spelling == '+':
                        return Symbol(child)
                    if tokens[0].spelling == '-':
                        return Mul(Symbol(child), -1)
                    if tokens[0].spelling == '++':
                        return PreIncrement(Symbol(child))
                    if tokens[0].spelling == '--':
                        return PreDecrement(Symbol(child))
                    if tokens[0].spelling == '!':
                        return Not(Symbol(child))
                # e.g.; int a = -1; or int b = -(1 + 2);
                else:
                    if tokens[0].spelling == '+':
                        return child
                    if tokens[0].spelling == '-':
                        return Mul(child, -1)
                    if tokens[0].spelling == '!':
                        return Not(sympify(bool(child)))

            # it can be either post increment/decrement
            # since variable name is obtained in token[0].spelling
            elif tokens[1].spelling in ['++', '--']:
                child = self.transform(next(node.get_children()))
                if tokens[1].spelling == '++':
                    return PostIncrement(Symbol(child))
                if tokens[1].spelling == '--':
                    return PostDecrement(Symbol(child))
            else:
                raise NotImplementedError("Dereferencing operator, "
                    "Address operator and bitwise NOT operator "
                    "have not been implemented yet!")

        def transform_binary_operator(self, node):
            """Transformation function for handling binary operators

            Returns
            =======

            binary_expression: Codegen AST node
                    simplified binary expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If a bitwise operator or
                unary operator(which is a child of any binary
                operator in Clang AST) is encountered

            """
            # get all the tokens of assignment
            # and store it in the tokens list
            tokens = list(node.get_tokens())

            # supported operators list
            operators_list = ['+', '-', '*', '/', '%','=',
            '>', '>=', '<', '<=', '==', '!=', '&&', '||', '+=', '-=',
            '*=', '/=', '%=']

            # this stack will contain variable content
            # and type of variable in the rhs
            combined_variables_stack = []

            # this stack will contain operators
            # to be processed in the rhs
            operators_stack = []

            # iterate through every token
            for token in tokens:
                # token is either '(', ')' or
                # any of the supported operators from the operator list
                if token.kind == cin.TokenKind.PUNCTUATION:

                    # push '(' to the operators stack
                    if token.spelling == '(':
                        operators_stack.append('(')

                    elif token.spelling == ')':
                        # keep adding the expression to the
                        # combined variables stack unless
                        # '(' is found
                        while (operators_stack
                            and operators_stack[-1] != '('):
                            if len(combined_variables_stack) < 2:
                                raise NotImplementedError(
                                    "Unary operators as a part of "
                                    "binary operators is not "
                                    "supported yet!")
                            rhs = combined_variables_stack.pop()
                            lhs = combined_variables_stack.pop()
                            operator = operators_stack.pop()
                            combined_variables_stack.append(
                                self.perform_operation(
                                lhs, rhs, operator))

                        # pop '('
                        operators_stack.pop()

                    # token is an operator (supported)
                    elif token.spelling in operators_list:
                        while (operators_stack
                            and self.priority_of(token.spelling)
                            <= self.priority_of(
                            operators_stack[-1])):
                            if len(combined_variables_stack) < 2:
                                raise NotImplementedError(
                                    "Unary operators as a part of "
                                    "binary operators is not "
                                    "supported yet!")
                            rhs = combined_variables_stack.pop()
                            lhs = combined_variables_stack.pop()
                            operator = operators_stack.pop()
                            combined_variables_stack.append(
                                self.perform_operation(
                                lhs, rhs, operator))

                        # push current operator
                        operators_stack.append(token.spelling)

                    # token is a bitwise operator
                    elif token.spelling in ['&', '|', '^', '<<', '>>']:
                        raise NotImplementedError(
                            "Bitwise operator has not been "
                            "implemented yet!")

                    # token is a shorthand bitwise operator
                    elif token.spelling in ['&=', '|=', '^=', '<<=',
                    '>>=']:
                        raise NotImplementedError(
                            "Shorthand bitwise operator has not been "
                            "implemented yet!")
                    else:
                        raise NotImplementedError(
                            "Given token {} is not implemented yet!"
                            .format(token.spelling))

                # token is an identifier(variable)
                elif token.kind == cin.TokenKind.IDENTIFIER:
                    combined_variables_stack.append(
                        [token.spelling, 'identifier'])

                # token is a literal
                elif token.kind == cin.TokenKind.LITERAL:
                    combined_variables_stack.append(
                        [token.spelling, 'literal'])

                # token is a keyword, either true or false
                elif (token.kind == cin.TokenKind.KEYWORD
                    and token.spelling in ['true', 'false']):
                    combined_variables_stack.append(
                        [token.spelling, 'boolean'])
                else:
                    raise NotImplementedError(
                        "Given token {} is not implemented yet!"
                        .format(token.spelling))

            # process remaining operators
            while operators_stack:
                if len(combined_variables_stack) < 2:
                    raise NotImplementedError(
                        "Unary operators as a part of "
                        "binary operators is not "
                        "supported yet!")
                rhs = combined_variables_stack.pop()
                lhs = combined_variables_stack.pop()
                operator = operators_stack.pop()
                combined_variables_stack.append(
                    self.perform_operation(lhs, rhs, operator))

            return combined_variables_stack[-1][0]

        def priority_of(self, op):
            """To get the priority of given operator"""
            if op in ['=', '+=', '-=', '*=', '/=', '%=']:
                return 1
            if op in ['&&', '||']:
                return 2
            if op in ['<', '<=', '>', '>=', '==', '!=']:
                return 3
            if op in ['+', '-']:
                return 4
            if op in ['*', '/', '%']:
                return 5
            return 0

        def perform_operation(self, lhs, rhs, op):
            """Performs operation supported by the SymPy core

            Returns
            =======

            combined_variable: list
                contains variable content and type of variable

            """
            lhs_value = self.get_expr_for_operand(lhs)
            rhs_value = self.get_expr_for_operand(rhs)
            if op == '+':
                return [Add(lhs_value, rhs_value), 'expr']
            if op == '-':
                return [Add(lhs_value, -rhs_value), 'expr']
            if op == '*':
                return [Mul(lhs_value, rhs_value), 'expr']
            if op == '/':
                return [Mul(lhs_value, Pow(rhs_value, Integer(-1))), 'expr']
            if op == '%':
                return [Mod(lhs_value, rhs_value), 'expr']
            if op in ['<', '<=', '>', '>=', '==', '!=']:
                return [Rel(lhs_value, rhs_value, op), 'expr']
            if op == '&&':
                return [And(as_Boolean(lhs_value), as_Boolean(rhs_value)), 'expr']
            if op == '||':
                return [Or(as_Boolean(lhs_value), as_Boolean(rhs_value)), 'expr']
            if op == '=':
                return [Assignment(Variable(lhs_value), rhs_value), 'expr']
            if op in ['+=', '-=', '*=', '/=', '%=']:
                return [aug_assign(Variable(lhs_value), op[0], rhs_value), 'expr']

        def get_expr_for_operand(self, combined_variable):
            """Gives out SymPy Codegen AST node

            AST node returned is corresponding to
            combined variable passed.Combined variable contains
            variable content and type of variable

            """
            if combined_variable[1] == 'identifier':
                return Symbol(combined_variable[0])
            if combined_variable[1] == 'literal':
                if '.' in combined_variable[0]:
                    return Float(float(combined_variable[0]))
                else:
                    return Integer(int(combined_variable[0]))
            if combined_variable[1] == 'expr':
                return combined_variable[0]
            if combined_variable[1] == 'boolean':
                    return true if combined_variable[0] == 'true' else false

        def transform_null_stmt(self, node):
            """Handles Null Statement and returns None"""
            return none

        def transform_while_stmt(self, node):
            """Transformation function for handling while statement

            Returns
            =======

            while statement : Codegen AST Node
                contains the while statement node having condition and
                statement block

            """
            children = node.get_children()

            condition = self.transform(next(children))
            statements = self.transform(next(children))

            if isinstance(statements, list):
                statement_block = CodeBlock(*statements)
            else:
                statement_block = CodeBlock(statements)

            return While(condition, statement_block)



else:
    class CCodeConverter():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Module not Installed")


def parse_c(source):
    """Function for converting a C source code

    The function reads the source code present in the given file and parses it
    to give out SymPy Expressions

    Returns
    =======

    src : list
        List of Python expression strings

    """
    converter = CCodeConverter()
    if os.path.exists(source):
        src = converter.parse(source, flags = [])
    else:
        src = converter.parse_str(source, flags = [])
    return src
<|endoftext|>import collections
import warnings

from sympy.external import import_module

autolevparser = import_module('sympy.parsing.autolev._antlr.autolevparser',
                              import_kwargs={'fromlist': ['AutolevParser']})
autolevlexer = import_module('sympy.parsing.autolev._antlr.autolevlexer',
                             import_kwargs={'fromlist': ['AutolevLexer']})
autolevlistener = import_module('sympy.parsing.autolev._antlr.autolevlistener',
                                import_kwargs={'fromlist': ['AutolevListener']})

AutolevParser = getattr(autolevparser, 'AutolevParser', None)
AutolevLexer = getattr(autolevlexer, 'AutolevLexer', None)
AutolevListener = getattr(autolevlistener, 'AutolevListener', None)


def strfunc(z):
    if z == 0:
        return ""
    elif z == 1:
        return "_d"
    else:
        return "_" + "d" * z

def declare_phy_entities(self, ctx, phy_type, i, j=None):
    if phy_type in ("frame", "newtonian"):
        declare_frames(self, ctx, i, j)
    elif phy_type == "particle":
        declare_particles(self, ctx, i, j)
    elif phy_type == "point":
        declare_points(self, ctx, i, j)
    elif phy_type == "bodies":
        declare_bodies(self, ctx, i, j)

def declare_frames(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()
    name2 = "frame_" + name1
    if self.getValue(ctx.parentCtx.varType()) == "newtonian":
        self.newtonian = name2

    self.symbol_table2.update({name1: name2})

    self.symbol_table.update({name1 + "1>": name2 + ".x"})
    self.symbol_table.update({name1 + "2>": name2 + ".y"})
    self.symbol_table.update({name1 + "3>": name2 + ".z"})

    self.type2.update({name1: "frame"})
    self.write(name2 + " = " + "_me.ReferenceFrame('" + name1 + "')\n")

def declare_points(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "point_" + name1

    self.symbol_table2.update({name1: name2})
    self.type2.update({name1: "point"})
    self.write(name2 + " = " + "_me.Point('" + name1 + "')\n")

def declare_particles(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "particle_" + name1

    self.symbol_table2.update({name1: name2})
    self.type2.update({name1: "particle"})
    self.bodies.update({name1: name2})
    self.write(name2 + " = " + "_me.Particle('" + name1 + "', " + "_me.Point('" +
                name1 + "_pt" + "'), " + "_sm.Symbol('m'))\n")

def declare_bodies(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "body_" + name1
    self.bodies.update({name1: name2})
    masscenter = name2 + "_cm"
    refFrame = name2 + "_f"

    self.symbol_table2.update({name1: name2})
    self.symbol_table2.update({name1 + "o": masscenter})
    self.symbol_table.update({name1 + "1>": refFrame+".x"})
    self.symbol_table.update({name1 + "2>": refFrame+".y"})
    self.symbol_table.update({name1 + "3>": refFrame+".z"})

    self.type2.update({name1: "bodies"})
    self.type2.update({name1+"o": "point"})

    self.write(masscenter + " = " + "_me.Point('" + name1 + "_cm" + "')\n")
    if self.newtonian:
        self.write(masscenter + ".set_vel(" + self.newtonian + ", " + "0)\n")
    self.write(refFrame + " = " + "_me.ReferenceFrame('" + name1 + "_f" + "')\n")
    # We set a dummy mass and inertia here.
    # They will be reset using the setters later in the code anyway.
    self.write(name2 + " = " + "_me.RigidBody('" + name1 + "', " + masscenter + ", " +
                refFrame + ", " + "_sm.symbols('m'), (_me.outer(" + refFrame +
                ".x," + refFrame + ".x)," + masscenter + "))\n")

def inertia_func(self, v1, v2, l, frame):

    if self.type2[v1] == "particle":
        l.append("_me.inertia_of_point_mass(" + self.bodies[v1] + ".mass, " + self.bodies[v1] +
                 ".point.pos_from(" + self.symbol_table2[v2] + "), " + frame + ")")

    elif self.type2[v1] == "bodies":
        # Inertia has been defined about center of mass.
        if self.inertia_point[v1] == v1 + "o":
            # Asking point is cm as well
            if v2 == self.inertia_point[v1]:
                l.append(self.symbol_table2[v1] + ".inertia[0]")

            # Asking point is not cm
            else:
                l.append(self.bodies[v1] + ".inertia[0]" + " + " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[v2] +
                         "), " + frame + ")")

        # Inertia has been defined about another point
        else:
            # Asking point is the defined point
            if v2 == self.inertia_point[v1]:
                l.append(self.symbol_table2[v1] + ".inertia[0]")
            # Asking point is cm
            elif v2 == v1 + "o":
                l.append(self.bodies[v1] + ".inertia[0]" + " - " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[self.inertia_point[v1]] +
                         "), " + frame + ")")
            # Asking point is some other point
            else:
                l.append(self.bodies[v1] + ".inertia[0]" + " - " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[self.inertia_point[v1]] +
                         "), " + frame + ")" + " + " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[v2] +
                         "), " + frame + ")")


def processConstants(self, ctx):
    # Process constant declarations of the type: Constants F = 3, g = 9.81
    name = ctx.ID().getText().lower()
    if "=" in ctx.getText():
        self.symbol_table.update({name: name})
        # self.inputs.update({self.symbol_table[name]: self.getValue(ctx.getChild(2))})
        self.write(self.symbol_table[name] + " = " + "_sm.S(" + self.getValue(ctx.getChild(2)) + ")\n")
        self.type.update({name: "constants"})
        return

    # Constants declarations of the type: Constants A, B
    else:
        if "{" not in ctx.getText():
            self.symbol_table[name] = name
            self.type[name] = "constants"

    # Process constant declarations of the type: Constants C+, D-
    if ctx.getChildCount() == 2:
        # This is set for declaring nonpositive=True and nonnegative=True
        if ctx.getChild(1).getText() == "+":
            self.sign[name] = "+"
        elif ctx.getChild(1).getText() == "-":
            self.sign[name] = "-"
    else:
        if "{" not in ctx.getText():
            self.sign[name] = "o"

    # Process constant declarations of the type: Constants K{4}, a{1:2, 1:2}, b{1:2}
    if "{" in ctx.getText():
        if ":" in ctx.getText():
            num1 = int(ctx.INT(0).getText())
            num2 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1

        if ":" in ctx.getText():
            if "," in ctx.getText():
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
                for i in range(num1, num2):
                    for j in range(num3, num4):
                        self.symbol_table[name + str(i) + str(j)] = name + str(i) + str(j)
                        self.type[name + str(i) + str(j)] = "constants"
                        self.var_list.append(name + str(i) + str(j))
                        self.sign[name + str(i) + str(j)] = "o"
            else:
                for i in range(num1, num2):
                    self.symbol_table[name + str(i)] = name + str(i)
                    self.type[name + str(i)] = "constants"
                    self.var_list.append(name + str(i))
                    self.sign[name + str(i)] = "o"

        elif "," in ctx.getText():
            for i in range(1, int(ctx.INT(0).getText()) + 1):
                for j in range(1, int(ctx.INT(1).getText()) + 1):
                    self.symbol_table[name] = name + str(i) + str(j)
                    self.type[name + str(i) + str(j)] = "constants"
                    self.var_list.append(name + str(i) + str(j))
                    self.sign[name + str(i) + str(j)] = "o"

        else:
            for i in range(num1, num2):
                self.symbol_table[name + str(i)] = name + str(i)
                self.type[name + str(i)] = "constants"
                self.var_list.append(name + str(i))
                self.sign[name + str(i)] = "o"

    if "{" not in ctx.getText():
        self.var_list.append(name)


def writeConstants(self, ctx):
    l1 = list(filter(lambda x: self.sign[x] == "o", self.var_list))
    l2 = list(filter(lambda x: self.sign[x] == "+", self.var_list))
    l3 = list(filter(lambda x: self.sign[x] == "-", self.var_list))
    try:
        if self.settings["complex"] == "on":
            real = ", real=True"
        elif self.settings["complex"] == "off":
            real = ""
    except Exception:
        real = ", real=True"

    if l1:
        a = ", ".join(l1) + " = " + "_sm.symbols(" + "'" +\
            " ".join(l1) + "'" + real + ")\n"
        self.write(a)
    if l2:
        a = ", ".join(l2) + " = " + "_sm.symbols(" + "'" +\
            " ".join(l2) + "'" + real + ", nonnegative=True)\n"
        self.write(a)
    if l3:
        a = ", ".join(l3) + " = " + "_sm.symbols(" + "'" + \
            " ".join(l3) + "'" + real + ", nonpositive=True)\n"
        self.write(a)
    self.var_list = []


def processVariables(self, ctx):
    # Specified F = x*N1> + y*N2>
    name = ctx.ID().getText().lower()
    if "=" in ctx.getText():
        text = name + "'"*(ctx.getChildCount()-3)
        self.write(text + " = " + self.getValue(ctx.expr()) + "\n")
        return

    # Process variables of the type: Variables qA, qB
    if ctx.getChildCount() == 1:
        self.symbol_table[name] = name
        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
            self.type.update({name: self.getValue(ctx.parentCtx.getChild(0))})

        self.var_list.append(name)
        self.sign[name] = 0

    # Process variables of the type: Variables x', y''
    elif "'" in ctx.getText() and "{" not in ctx.getText():
        if ctx.getText().count("'") > self.maxDegree:
            self.maxDegree = ctx.getText().count("'")
        for i in range(ctx.getChildCount()):
            self.sign[name + strfunc(i)] = i
            self.symbol_table[name + "'"*i] = name + strfunc(i)
            if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                self.type.update({name + "'"*i: self.getValue(ctx.parentCtx.getChild(0))})
            self.var_list.append(name + strfunc(i))

    elif "{" in ctx.getText():
        # Process variables of the type: Variables x{3}, y{2}

        if "'" in ctx.getText():
            dash_count = ctx.getText().count("'")
            if dash_count > self.maxDegree:
                self.maxDegree = dash_count

        if ":" in ctx.getText():
            # Variables C{1:2, 1:2}
            if "," in ctx.getText():
                num1 = int(ctx.INT(0).getText())
                num2 = int(ctx.INT(1).getText()) + 1
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
            # Variables C{1:2}
            else:
                num1 = int(ctx.INT(0).getText())
                num2 = int(ctx.INT(1).getText()) + 1

        # Variables C{1,3}
        elif "," in ctx.getText():
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1
            num3 = 1
            num4 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1

        for i in range(num1, num2):
            try:
                for j in range(num3, num4):
                    try:
                        for z in range(dash_count+1):
                            self.symbol_table.update({name + str(i) + str(j) + "'"*z: name + str(i) + str(j) + strfunc(z)})
                            if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                                self.type.update({name + str(i) + str(j) +  "'"*z: self.getValue(ctx.parentCtx.getChild(0))})
                            self.var_list.append(name + str(i) + str(j) + strfunc(z))
                            self.sign.update({name + str(i) + str(j) + strfunc(z): z})
                            if dash_count > self.maxDegree:
                                self.maxDegree = dash_count
                    except Exception:
                        self.symbol_table.update({name + str(i) + str(j): name + str(i) + str(j)})
                        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                            self.type.update({name + str(i) + str(j): self.getValue(ctx.parentCtx.getChild(0))})
                        self.var_list.append(name + str(i) + str(j))
                        self.sign.update({name + str(i) + str(j): 0})
            except Exception:
                try:
                    for z in range(dash_count+1):
                        self.symbol_table.update({name + str(i) + "'"*z: name + str(i) + strfunc(z)})
                        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                            self.type.update({name + str(i) +  "'"*z: self.getValue(ctx.parentCtx.getChild(0))})
                        self.var_list.append(name + str(i) + strfunc(z))
                        self.sign.update({name + str(i) + strfunc(z): z})
                        if dash_count > self.maxDegree:
                            self.maxDegree = dash_count
                except Exception:
                    self.symbol_table.update({name + str(i): name + str(i)})
                    if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                        self.type.update({name + str(i): self.getValue(ctx.parentCtx.getChild(0))})
                    self.var_list.append(name + str(i))
                    self.sign.update({name + str(i): 0})

def writeVariables(self, ctx):
    #print(self.sign)
    #print(self.symbol_table)
    if self.var_list:
        for i in range(self.maxDegree+1):
            if i == 0:
                j = ""
                t = ""
            else:
                j = str(i)
                t = ", "
            l = []
            for k in list(filter(lambda x: self.sign[x] == i, self.var_list)):
                if i == 0:
                    l.append(k)
                if i == 1:
                    l.append(k[:-1])
                if i > 1:
                    l.append(k[:-2])
            a = ", ".join(list(filter(lambda x: self.sign[x] == i, self.var_list))) + " = " +\
                "_me.dynamicsymbols(" + "'" + " ".join(l) + "'" + t + j + ")\n"
            l = []
            self.write(a)
        self.maxDegree = 0
    self.var_list = []

def processImaginary(self, ctx):
    name = ctx.ID().getText().lower()
    self.symbol_table[name] = name
    self.type[name] = "imaginary"
    self.var_list.append(name)


def writeImaginary(self, ctx):
    a = ", ".join(self.var_list) + " = " + "_sm.symbols(" + "'" + \
        " ".join(self.var_list) + "')\n"
    b = ", ".join(self.var_list) + " = " + "_sm.I\n"
    self.write(a)
    self.write(b)
    self.var_list = []

if AutolevListener:
    class MyListener(AutolevListener):  # type: ignore
        def __init__(self, include_numeric=False):
            # Stores data in tree nodes(tree annotation). Especially useful for expr reconstruction.
            self.tree_property = {}

            # Stores the declared variables, constants etc as they are declared in Autolev and SymPy
            # {"<Autolev symbol>": "<SymPy symbol>"}.
            self.symbol_table = collections.OrderedDict()

            # Similar to symbol_table. Used for storing Physical entities like Frames, Points,
            # Particles, Bodies etc
            self.symbol_table2 = collections.OrderedDict()

            # Used to store nonpositive, nonnegative etc for constants and number of "'"s (order of diff)
            # in variables.
            self.sign = {}

            # Simple list used as a store to pass around variables between the 'process' and 'write'
            # methods.
            self.var_list = []

            # Stores the type of a declared variable (constants, variables, specifieds etc)
            self.type = collections.OrderedDict()

            # Similar to self.type. Used for storing the type of Physical entities like Frames, Points,
            # Particles, Bodies etc
            self.type2 = collections.OrderedDict()

            # These lists are used to distinguish matrix, numeric and vector expressions.
            self.matrix_expr = []
            self.numeric_expr = []
            self.vector_expr = []
            self.fr_expr = []

            self.output_code = []

            # Stores the variables and their rhs for substituting upon the Autolev command EXPLICIT.
            self.explicit = collections.OrderedDict()

            # Write code to import common dependencies.
            self.output_code.append("import sympy.physics.mechanics as _me\n")
            self.output_code.append("import sympy as _sm\n")
            self.output_code.append("import math as m\n")
            self.output_code.append("import numpy as _np\n")
            self.output_code.append("\n")

            # Just a store for the max degree variable in a line.
            self.maxDegree = 0

            # Stores the input parameters which are then used for codegen and numerical analysis.
            self.inputs = collections.OrderedDict()
            # Stores the variables which appear in Output Autolev commands.
            self.outputs = []
            # Stores the settings specified by the user. Ex: Complex on/off, Degrees on/off
            self.settings = {}
            # Boolean which changes the behaviour of some expression reconstruction
            # when parsing Input Autolev commands.
            self.in_inputs = False
            self.in_outputs = False

            # Stores for the physical entities.
            self.newtonian = None
            self.bodies = collections.OrderedDict()
            self.constants = []
            self.forces = collections.OrderedDict()
            self.q_ind = []
            self.q_dep = []
            self.u_ind = []
            self.u_dep = []
            self.kd_eqs = []
            self.dependent_variables = []
            self.kd_equivalents = collections.OrderedDict()
            self.kd_equivalents2 = collections.OrderedDict()
            self.kd_eqs_supplied = None
            self.kane_type = "no_args"
            self.inertia_point = collections.OrderedDict()
            self.kane_parsed = False
            self.t = False

            # PyDy ode code will be included only if this flag is set to True.
            self.include_numeric = include_numeric

        def write(self, string):
            self.output_code.append(string)

        def getValue(self, node):
            return self.tree_property[node]

        def setValue(self, node, value):
            self.tree_property[node] = value

        def getSymbolTable(self):
            return self.symbol_table

        def getType(self):
            return self.type

        def exitVarDecl(self, ctx):
            # This event method handles variable declarations. The parse tree node varDecl contains
            # one or more varDecl2 nodes. Eg varDecl for 'Constants a{1:2, 1:2}, b{1:2}' has two varDecl2
            # nodes(one for a{1:2, 1:2} and one for b{1:2}).

            # Variable declarations are processed and stored in the event method exitVarDecl2.
            # This stored information is used to write the final SymPy output code in the exitVarDecl event method.

            # determine the type of declaration
            if self.getValue(ctx.varType()) == "constant":
                writeConstants(self, ctx)
            elif self.getValue(ctx.varType()) in\
            ("variable", "motionvariable", "motionvariable'", "specified"):
                writeVariables(self, ctx)
            elif self.getValue(ctx.varType()) == "imaginary":
                writeImaginary(self, ctx)

        def exitVarType(self, ctx):
            # Annotate the varType tree node with the type of the variable declaration.
            name = ctx.getChild(0).getText().lower()
            if name[-1] == "s" and name != "bodies":
                self.setValue(ctx, name[:-1])
            else:
                self.setValue(ctx, name)

        def exitVarDecl2(self, ctx):
            # Variable declarations are processed and stored in the event method exitVarDecl2.
            # This stored information is used to write the final SymPy output code in the exitVarDecl event method.
            # This is the case for constants, variables, specifieds etc.

            # This isn't the case for all types of declarations though. For instance
            # Frames A, B, C, N cannot be defined on one line in SymPy. So we do not append A, B, C, N
            # to a var_list or use exitVarDecl. exitVarDecl2 directly writes out to the file.

            # determine the type of declaration
            if self.getValue(ctx.parentCtx.varType()) == "constant":
                processConstants(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) in \
            ("variable", "motionvariable", "motionvariable'", "specified"):
                processVariables(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) == "imaginary":
                processImaginary(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) in ("frame", "newtonian", "point", "particle", "bodies"):
                if "{" in ctx.getText():
                    if ":" in ctx.getText() and "," not in ctx.getText():
                        num1 = int(ctx.INT(0).getText())
                        num2 = int(ctx.INT(1).getText()) + 1
                    elif ":" not in ctx.getText() and "," in ctx.getText():
                        num1 = 1
                        num2 = int(ctx.INT(0).getText()) + 1
                        num3 = 1
                        num4 = int(ctx.INT(1).getText()) + 1
                    elif ":" in ctx.getText() and "," in ctx.getText():
                        num1 = int(ctx.INT(0).getText())
                        num2 = int(ctx.INT(1).getText()) + 1
                        num3 = int(ctx.INT(2).getText())
                        num4 = int(ctx.INT(3).getText()) + 1
                    else:
                        num1 = 1
                        num2 = int(ctx.INT(0).getText()) + 1
                else:
                    num1 = 1
                    num2 = 2
                for i in range(num1, num2):
                    try:
                        for j in range(num3, num4):
                            declare_phy_entities(self, ctx, self.getValue(ctx.parentCtx.varType()), i, j)
                    except Exception:
                        declare_phy_entities(self, ctx, self.getValue(ctx.parentCtx.varType()), i)
        # ================== Subrules of parser rule expr (Start) ====================== #

        def exitId(self, ctx):
            # Tree annotation for ID which is a labeled subrule of the parser rule expr.
            # A_C
            python_keywords = ["and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except",\
            "exec", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "not", "or", "pass", "print",\
            "raise", "return", "try", "while", "with", "yield"]

            if ctx.ID().getText().lower() in python_keywords:
                warnings.warn("Python keywords must not be used as identifiers. Please refer to the list of keywords at https://docs.python.org/2.5/ref/keywords.html",
                SyntaxWarning)

            if "_" in ctx.ID().getText() and ctx.ID().getText().count('_') == 1:
                e1, e2 = ctx.ID().getText().lower().split('_')
                try:
                    if self.type2[e1] == "frame":
                        e1 = self.symbol_table2[e1]
                    elif self.type2[e1] == "bodies":
                        e1 = self.symbol_table2[e1] + "_f"
                    if self.type2[e2] == "frame":
                        e2 = self.symbol_table2[e2]
                    elif self.type2[e2] == "bodies":
                        e2 = self.symbol_table2[e2] + "_f"

                    self.setValue(ctx, e1 + ".dcm(" + e2 + ")")
                except Exception:
                    self.setValue(ctx, ctx.ID().getText().lower())
            else:
                # Reserved constant Pi
                if ctx.ID().getText().lower() == "pi":
                    self.setValue(ctx, "_sm.pi")
                    self.numeric_expr.append(ctx)

                # Reserved variable T (for time)
                elif ctx.ID().getText().lower() == "t":
                    self.setValue(ctx, "_me.dynamicsymbols._t")
                    if not self.in_inputs and not self.in_outputs:
                        self.t = True

                else:
                    idText = ctx.ID().getText().lower() + "'"*(ctx.getChildCount() - 1)
                    if idText in self.type.keys() and self.type[idText] == "matrix":
                        self.matrix_expr.append(ctx)
                    if self.in_inputs:
                        try:
                            self.setValue(ctx, self.symbol_table[idText])
                        except Exception:
                            self.setValue(ctx, idText.lower())
                    else:
                        try:
                            self.setValue(ctx, self.symbol_table[idText])
                        except Exception:
                            pass

        def exitInt(self, ctx):
            # Tree annotation for int which is a labeled subrule of the parser rule expr.
            int_text = ctx.INT().getText()
            self.setValue(ctx, int_text)
            self.numeric_expr.append(ctx)

        def exitFloat(self, ctx):
            # Tree annotation for float which is a labeled subrule of the parser rule expr.
            floatText = ctx.FLOAT().getText()
            self.setValue(ctx, floatText)
            self.numeric_expr.append(ctx)

        def exitAddSub(self, ctx):
            # Tree annotation for AddSub which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr (+|-) expr
            if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, self.getValue(ctx.expr(0)) + ctx.getChild(1).getText() +
                          self.getValue(ctx.expr(1)))

        def exitMulDiv(self, ctx):
            # Tree annotation for MulDiv which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr (*|/) expr
            try:
                if ctx.expr(0) in self.vector_expr and ctx.expr(1) in self.vector_expr:
                    self.setValue(ctx, "_me.outer(" + self.getValue(ctx.expr(0)) + ", " +
                                  self.getValue(ctx.expr(1)) + ")")
                else:
                    if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                        self.matrix_expr.append(ctx)
                    if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                        self.vector_expr.append(ctx)
                    if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                        self.numeric_expr.append(ctx)
                    self.setValue(ctx, self.getValue(ctx.expr(0)) + ctx.getChild(1).getText() +
                                  self.getValue(ctx.expr(1)))
            except Exception:
                pass

        def exitNegativeOne(self, ctx):
            # Tree annotation for negativeOne which is a labeled subrule of the parser rule expr.
            self.setValue(ctx, "-1*" + self.getValue(ctx.getChild(1)))
            if ctx.getChild(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.getChild(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)

        def exitParens(self, ctx):
            # Tree annotation for parens which is a labeled subrule of the parser rule expr.
            # The subrule is expr = '(' expr ')'
            if ctx.expr() in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr() in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr() in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, "(" + self.getValue(ctx.expr()) + ")")

        def exitExponent(self, ctx):
            # Tree annotation for Exponent which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr ^ expr
            if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, self.getValue(ctx.expr(0)) + "**" + self.getValue(ctx.expr(1)))

        def exitExp(self, ctx):
            s = ctx.EXP().getText()[ctx.EXP().getText().index('E')+1:]
            if "-" in s:
                s = s[0] + s[1:].lstrip("0")
            else:
                s = s.lstrip("0")
            self.setValue(ctx, ctx.EXP().getText()[:ctx.EXP().getText().index('E')] +
                          "*10**(" + s + ")")

        def exitFunction(self, ctx):
            # Tree annotation for function which is a labeled subrule of the parser rule expr.

            # The difference between this and FunctionCall is that this is used for non standalone functions
            # appearing in expressions and assignments.
            # Eg:
            # When we come across a standalone function say Expand(E, n:m) then it is categorized as FunctionCall
            # which is a parser rule in itself under rule stat. exitFunctionCall() takes care of it and writes to the file.
            #
            # On the other hand, while we come across E_diff = D(E, y), we annotate the tree node
            # of the function D(E, y) with the SymPy equivalent in exitFunction().
            # In this case it is the method exitAssignment() that writes the code to the file and not exitFunction().

            ch = ctx.getChild(0)
            func_name = ch.getChild(0).getText().lower()

            # Expand(y, n:m) *
            if func_name == "expand":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    # _sm.Matrix([i.expand() for i in z]).reshape(z.shape[0], z.shape[1])
                    self.setValue(ctx, "_sm.Matrix([i.expand() for i in " + expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr + ")" + "." + "expand()")

            # Factor(y, x) *
            elif func_name == "factor":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([_sm.factor(i, " + self.getValue(ch.expr(1)) + ") for i in " +
                                  expr + "])" + ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "_sm.factor(" + "(" + expr + ")" +
                                  ", " + self.getValue(ch.expr(1)) + ")")

            # D(y, x)
            elif func_name == "d":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.diff(" + self.getValue(ch.expr(1)) + ") for i in " +
                                  expr + "])" + ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if ch.getChildCount() == 8:
                        frame = self.symbol_table2[ch.expr(2).getText().lower()]
                        self.setValue(ctx, "(" + expr + ")" + "." + "diff(" + self.getValue(ch.expr(1)) +
                                      ", " + frame + ")")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + "." + "diff(" +
                                      self.getValue(ch.expr(1)) + ")")

            # Dt(y)
            elif func_name == "dt":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.vector_expr:
                    text = "dt("
                else:
                    text = "diff(_sm.Symbol('t')"
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i." + text +
                                  ") for i in " + expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if ch.getChildCount() == 6:
                        frame = self.symbol_table2[ch.expr(1).getText().lower()]
                        self.setValue(ctx, "(" + expr + ")" + "." + "dt(" +
                                      frame + ")")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + "." + text + ")")

            # Explicit(EXPRESS(IMPLICIT>,C))
            elif func_name == "explicit":
                if ch.expr(0) in self.vector_expr:
                    self.vector_expr.append(ctx)
                expr = self.getValue(ch.expr(0))
                if self.explicit.keys():
                    explicit_list = []
                    for i in self.explicit.keys():
                        explicit_list.append(i + ":" + self.explicit[i])
                    self.setValue(ctx, "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})")
                else:
                    self.setValue(ctx, expr)

            # Taylor(y, 0:2, w=a, x=0)
            # TODO: Currently only works with symbols. Make it work for dynamicsymbols.
            elif func_name == "taylor":
                exp = self.getValue(ch.expr(0))
                order = self.getValue(ch.expr(1).expr(1))
                x = (ch.getChildCount()-6)//2
                l = []
                for i in range(x):
                    index = 2 + i
                    child = ch.expr(index)
                    l.append(".series(" + self.getValue(child.getChild(0)) +
                             ", " + self.getValue(child.getChild(2)) +
                             ", " + order + ").removeO()")
                self.setValue(ctx, "(" + exp + ")" + "".join(l))

            # Evaluate(y, a=x, b=2)
            elif func_name == "evaluate":
                expr = self.getValue(ch.expr(0))
                l = []
                x = (ch.getChildCount()-4)//2
                for i in range(x):
                    index = 1 + i
                    child = ch.expr(index)
                    l.append(self.getValue(child.getChild(0)) + ":" +
                             self.getValue(child.getChild(2)))

                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.subs({" + ",".join(l) + "}) for i in " +
                                  expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if self.explicit:
                        explicit_list = []
                        for i in self.explicit.keys():
                            explicit_list.append(i + ":" + self.explicit[i])
                        self.setValue(ctx, "(" + expr + ")" + ".subs({" + ",".join(explicit_list) +
                                      "}).subs({" + ",".join(l) + "})")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + ".subs({" + ",".join(l) + "})")

            # Polynomial([a, b, c], x)
            elif func_name == "polynomial":
                self.setValue(ctx, "_sm.Poly(" + self.getValue(ch.expr(0)) + ", " +
                              self.getValue(ch.expr(1)) + ")")

            # Roots(Poly, x, 2)
            # Roots([1; 2; 3; 4])
            elif func_name == "roots":
                self.matrix_expr.append(ctx)
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.setValue(ctx, "[i.evalf() for i in " + "_sm.solve(" +
                                  "_sm.Poly(" + expr + ", " + "x),x)]")
                else:
                    self.setValue(ctx, "[i.evalf() for i in " + "_sm.solve(" +
                                  expr + ", " + self.getValue(ch.expr(1)) + ")]")

            # Transpose(A), Inv(A)
            elif func_name in ("transpose", "inv", "inverse"):
                self.matrix_expr.append(ctx)
                if func_name == "transpose":
                    e = ".T"
                elif func_name in ("inv", "inverse"):
                    e = "**(-1)"
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + e)

            # Eig(A)
            elif func_name == "eig":
                # "_sm.Matrix([i.evalf() for i in " +
                self.setValue(ctx, "_sm.Matrix([i.evalf() for i in (" +
                              self.getValue(ch.expr(0)) + ").eigenvals().keys()])")

            # Diagmat(n, m, x)
            # Diagmat(3, 1)
            elif func_name == "diagmat":
                self.matrix_expr.append(ctx)
                if ch.getChildCount() == 6:
                    l = []
                    for i in range(int(self.getValue(ch.expr(0)))):
                        l.append(self.getValue(ch.expr(1)) + ",")

                    self.setValue(ctx, "_sm.diag(" + ("".join(l))[:-1] + ")")

                elif ch.getChildCount() == 8:
                    # _sm.Matrix([x if i==j else 0 for i in range(n) for j in range(m)]).reshape(n, m)
                    n = self.getValue(ch.expr(0))
                    m = self.getValue(ch.expr(1))
                    x = self.getValue(ch.expr(2))
                    self.setValue(ctx, "_sm.Matrix([" + x + " if i==j else 0 for i in range(" +
                                  n + ") for j in range(" + m + ")]).reshape(" + n + ", " + m + ")")

            # Cols(A)
            # Cols(A, 1)
            # Cols(A, 1, 2:4, 3)
            elif func_name in ("cols", "rows"):
                self.matrix_expr.append(ctx)
                if func_name == "cols":
                    e1 = ".cols"
                    e2 = ".T."
                else:
                    e1 = ".rows"
                    e2 = "."
                if ch.getChildCount() == 4:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + e1)
                elif ch.getChildCount() == 6:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" +
                                  e1[:-1] + "(" + str(int(self.getValue(ch.expr(1))) - 1) + ")")
                else:
                    l = []
                    for i in range(4, ch.getChildCount()):
                        try:
                            if ch.getChild(i).getChildCount() > 1 and ch.getChild(i).getChild(1).getText() == ":":
                                for j in range(int(ch.getChild(i).getChild(0).getText()),
                                int(ch.getChild(i).getChild(2).getText())+1):
                                    l.append("(" + self.getValue(ch.getChild(2)) + ")" + e2 +
                                             "row(" + str(j-1) + ")")
                            else:
                                l.append("(" + self.getValue(ch.getChild(2)) + ")" + e2 +
                                         "row(" + str(int(ch.getChild(i).getText())-1) + ")")
                        except Exception:
                            pass
                    self.setValue(ctx, "_sm.Matrix([" + ",".join(l) + "])")

            # Det(A) Trace(A)
            elif func_name in ["det", "trace"]:
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + "." +
                              func_name + "()")

            # Element(A, 2, 3)
            elif func_name == "element":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + "[" +
                              str(int(self.getValue(ch.expr(1)))-1) + "," +
                              str(int(self.getValue(ch.expr(2)))-1) + "]")

            elif func_name in \
            ["cos", "sin", "tan", "cosh", "sinh", "tanh", "acos", "asin", "atan",
            "log", "exp", "sqrt", "factorial", "floor", "sign"]:
                self.setValue(ctx, "_sm." + func_name + "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "ceil":
                self.setValue(ctx, "_sm.ceiling" + "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "sqr":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) +
                              ")" + "**2")

            elif func_name == "log10":
                self.setValue(ctx, "_sm.log" +
                              "(" + self.getValue(ch.expr(0)) + ", 10)")

            elif func_name == "atan2":
                self.setValue(ctx, "_sm.atan2" + "(" + self.getValue(ch.expr(0)) + ", " +
                              self.getValue(ch.expr(1)) + ")")

            elif func_name in ["int", "round"]:
                self.setValue(ctx, func_name +
                              "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "abs":
                self.setValue(ctx, "_sm.Abs(" + self.getValue(ch.expr(0)) + ")")

            elif func_name in ["max", "min"]:
                # max(x, y, z)
                l = []
                for i in range(1, ch.getChildCount()):
                    if ch.getChild(i) in self.tree_property.keys():
                        l.append(self.getValue(ch.getChild(i)))
                    elif ch.getChild(i).getText() in [",", "(", ")"]:
                        l.append(ch.getChild(i).getText())
                self.setValue(ctx, "_sm." + ch.getChild(0).getText().capitalize() + "".join(l))

            # Coef(y, x)
            elif func_name == "coef":
                #A41_A53=COEF([RHS(U4);RHS(U5)],[U1,U2,U3])
                if ch.expr(0) in self.matrix_expr and ch.expr(1) in self.matrix_expr:
                    icount = jcount = 0
                    for i in range(ch.expr(0).getChild(0).getChildCount()):
                        try:
                            ch.expr(0).getChild(0).getChild(i).getRuleIndex()
                            icount+=1
                        except Exception:
                            pass
                    for j in range(ch.expr(1).getChild(0).getChildCount()):
                        try:
                            ch.expr(1).getChild(0).getChild(j).getRuleIndex()
                            jcount+=1
                        except Exception:
                            pass
                    l = []
                    for i in range(icount):
                        for j in range(jcount):
                            # a41_a53[i,j] = u4.expand().coeff(u1)
                            l.append(self.getValue(ch.expr(0).getChild(0).expr(i)) + ".expand().coeff("
                                     + self.getValue(ch.expr(1).getChild(0).expr(j)) + ")")
                    self.setValue(ctx, "_sm.Matrix([" + ", ".join(l) + "]).reshape(" + str(icount) + ", " + str(jcount) + ")")
                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) +
                                  ")" + ".expand().coeff(" + self.getValue(ch.expr(1)) + ")")

            # Exclude(y, x) Include(y, x)
            elif func_name in ("exclude", "include"):
                if func_name == "exclude":
                    e = "0"
                else:
                    e = "1"
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.collect(" + self.getValue(ch.expr(1)) + "])" +
                                  ".coeff(" + self.getValue(ch.expr(1)) + "," + e + ")" + "for i in " + expr + ")" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr +
                                  ")" + ".collect(" + self.getValue(ch.expr(1)) + ")" +
                                  ".coeff(" + self.getValue(ch.expr(1)) + "," + e + ")")

            # RHS(y)
            elif func_name == "rhs":
                self.setValue(ctx, self.explicit[self.getValue(ch.expr(0))])

            # Arrange(y, n, x) *
            elif func_name == "arrange":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.collect(" + self.getValue(ch.expr(2)) +
                                  ")" + "for i in " + expr + "])"+
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr +
                                  ")" + ".collect(" + self.getValue(ch.expr(2)) + ")")

            # Replace(y, sin(x)=3)
            elif func_name == "replace":
                l = []
                for i in range(1, ch.getChildCount()):
                    try:
                        if ch.getChild(i).getChild(1).getText() == "=":
                            l.append(self.getValue(ch.getChild(i).getChild(0)) +
                                     ":" + self.getValue(ch.getChild(i).getChild(2)))
                    except Exception:
                        pass
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.subs({" + ",".join(l) + "}) for i in " +
                                  expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" +
                                  ".subs({" + ",".join(l) + "})")

            # Dot(Loop>, N1>)
            elif func_name == "dot":
                l = []
                num = (ch.expr(1).getChild(0).getChildCount()-1)//2
                if ch.expr(1) in self.matrix_expr:
                    for i in range(num):
                        l.append("_me.dot(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1).getChild(0).expr(i)) + ")")
                    self.setValue(ctx, "_sm.Matrix([" + ",".join(l) + "]).reshape(" + str(num) + ", " + "1)")
                else:
                    self.setValue(ctx, "_me.dot(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1)) + ")")
            # Cross(w_A_N>, P_NA_AB>)
            elif func_name == "cross":
                self.vector_expr.append(ctx)
                self.setValue(ctx, "_me.cross(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1)) + ")")

            # Mag(P_O_Q>)
            elif func_name == "mag":
                self.setValue(ctx, self.getValue(ch.expr(0)) + "." + "magnitude()")

            # MATRIX(A, I_R>>)
            elif func_name == "matrix":
                if self.type2[ch.expr(0).getText().lower()] == "frame":
                    text = ""
                elif self.type2[ch.expr(0).getText().lower()] == "bodies":
                    text = "_f"
                self.setValue(ctx, "(" + self.getValue(ch.expr(1)) + ")" + ".to_matrix(" +
                              self.symbol_table2[ch.expr(0).getText().lower()] + text + ")")

            # VECTOR(A, ROWS(EIGVECS,1))
            elif func_name == "vector":
                if self.type2[ch.expr(0).getText().lower()] == "frame":
                    text = ""
                elif self.type2[ch.expr(0).getText().lower()] == "bodies":
                    text = "_f"
                v = self.getValue(ch.expr(1))
                f = self.symbol_table2[ch.expr(0).getText().lower()] + text
                self.setValue(ctx, v + "[0]*" + f + ".x +" + v + "[1]*" + f + ".y +" +
                              v + "[2]*" + f + ".z")

            # Express(A2>, B)
            # Here I am dealing with all the Inertia commands as I expect the users to use Inertia
            # commands only with Express because SymPy needs the Reference frame to be specified unlike Autolev.
            elif func_name == "express":
                self.vector_expr.append(ctx)
                if self.type2[ch.expr(1).getText().lower()] == "frame":
                    frame = self.symbol_table2[ch.expr(1).getText().lower()]
                else:
                    frame = self.symbol_table2[ch.expr(1).getText().lower()] + "_f"
                if ch.expr(0).getText().lower() == "1>>":
                    self.setValue(ctx, "_me.inertia(" + frame + ", 1, 1, 1)")

                elif '_' in ch.expr(0).getText().lower() and ch.expr(0).getText().lower().count('_') == 2\
                and ch.expr(0).getText().lower()[0] == "i" and ch.expr(0).getText().lower()[-2:] == ">>":
                    v1 = ch.expr(0).getText().lower()[:-2].split('_')[1]
                    v2 = ch.expr(0).getText().lower()[:-2].split('_')[2]
                    l = []
                    inertia_func(self, v1, v2, l, frame)
                    self.setValue(ctx, " + ".join(l))

                elif ch.expr(0).getChild(0).getChild(0).getText().lower() == "inertia":
                    if ch.expr(0).getChild(0).getChildCount() == 4:
                        l = []
                        v2 = ch.expr(0).getChild(0).ID(0).getText().lower()
                        for v1 in self.bodies:
                            inertia_func(self, v1, v2, l, frame)
                        self.setValue(ctx, " + ".join(l))

                    else:
                        l = []
                        l2 = []
                        v2 = ch.expr(0).getChild(0).ID(0).getText().lower()
                        for i in range(1, (ch.expr(0).getChild(0).getChildCount()-2)//2):
                            l2.append(ch.expr(0).getChild(0).ID(i).getText().lower())
                        for v1 in l2:
                            inertia_func(self, v1, v2, l, frame)
                        self.setValue(ctx, " + ".join(l))

                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + ".express(" +
                                  self.symbol_table2[ch.expr(1).getText().lower()] + ")")
            # CM(P)
            elif func_name == "cm":
                if self.type2[ch.expr(0).getText().lower()] == "point":
                    text = ""
                else:
                    text = ".point"
                if ch.getChildCount() == 4:
                    self.setValue(ctx, "_me.functions.center_of_mass(" + self.symbol_table2[ch.expr(0).getText().lower()] +
                                  text + "," + ", ".join(self.bodies.values()) + ")")
                else:
                    bodies = []
                    for i in range(1, (ch.getChildCount()-1)//2):
                        bodies.append(self.symbol_table2[ch.expr(i).getText().lower()])
                    self.setValue(ctx, "_me.functions.center_of_mass(" + self.symbol_table2[ch.expr(0).getText().lower()] +
                                  text + "," + ", ".join(bodies) + ")")

            # PARTIALS(V_P1_E>,U1)
            elif func_name == "partials":
                speeds = []
                for i in range(1, (ch.getChildCount()-1)//2):
                    if self.kd_equivalents2:
                        speeds.append(self.kd_equivalents2[self.symbol_table[ch.expr(i).getText().lower()]])
                    else:
                        speeds.append(self.symbol_table[ch.expr(i).getText().lower()])
                v1, v2, v3 = ch.expr(0).getText().lower().replace(">","").split('_')
                if self.type2[v2] == "point":
                    point = self.symbol_table2[v2]
                elif self.type2[v2] == "particle":
                    point = self.symbol_table2[v2] + ".point"
                frame = self.symbol_table2[v3]
                self.setValue(ctx, point + ".partial_velocity(" + frame + ", " + ",".join(speeds) + ")")

            # UnitVec(A1>+A2>+A3>)
            elif func_name == "unitvec":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + ".normalize()")

            # Units(deg, rad)
            elif func_name == "units":
                if ch.expr(0).getText().lower() == "deg" and ch.expr(1).getText().lower() == "rad":
                    factor = 0.0174533
                elif ch.expr(0).getText().lower() == "rad" and ch.expr(1).getText().lower() == "deg":
                    factor = 57.2958
                self.setValue(ctx, str(factor))
            # Mass(A)
            elif func_name == "mass":
                l = []
                try:
                    ch.ID(0).getText().lower()
                    for i in range((ch.getChildCount()-1)//2):
                        l.append(self.symbol_table2[ch.ID(i).getText().lower()] + ".mass")
                    self.setValue(ctx, "+".join(l))
                except Exception:
                    for i in self.bodies.keys():
                        l.append(self.bodies[i] + ".mass")
                    self.setValue(ctx, "+".join(l))

            # Fr() FrStar()
            # _me.KanesMethod(n, q_ind, u_ind, kd, velocity_constraints).kanes_equations(pl, fl)[0]
            elif func_name in ["fr", "frstar"]:
                if not self.kane_parsed:
                    if self.kd_eqs:
                        for i in self.kd_eqs:
                            self.q_ind.append(self.symbol_table[i.strip().split('-')[0].replace("'","")])
                            self.u_ind.append(self.symbol_table[i.strip().split('-')[1].replace("'","")])

                    for i in range(len(self.kd_eqs)):
                        self.kd_eqs[i] = self.symbol_table[self.kd_eqs[i].strip().split('-')[0]] + " - " +\
                                         self.symbol_table[self.kd_eqs[i].strip().split('-')[1]]

                    # Do all of this if kd_eqs are not specified
                    if not self.kd_eqs:
                        self.kd_eqs_supplied = False
                        self.matrix_expr.append(ctx)
                        for i in self.type.keys():
                            if self.type[i] == "motionvariable":
                                if self.sign[self.symbol_table[i.lower()]] == 0:
                                    self.q_ind.append(self.symbol_table[i.lower()])
                                elif self.sign[self.symbol_table[i.lower()]] == 1:
                                    name = "u_" + self.symbol_table[i.lower()]
                                    self.symbol_table.update({name: name})
                                    self.write(name + " = " + "_me.dynamicsymbols('" + name + "')\n")
                                    if self.symbol_table[i.lower()] not in self.dependent_variables:
                                        self.u_ind.append(name)
                                        self.kd_equivalents.update({name: self.symbol_table[i.lower()]})
                                    else:
                                        self.u_dep.append(name)
                                        self.kd_equivalents.update({name: self.symbol_table[i.lower()]})

                        for i in self.kd_equivalents.keys():
                            self.kd_eqs.append(self.kd_equivalents[i] + "-" + i)

                    if not self.u_ind and not self.kd_eqs:
                        self.u_ind = self.q_ind.copy()
                        self.q_ind = []

                # deal with velocity constraints
                if self.dependent_variables:
                    for i in self.dependent_variables:
                        self.u_dep.append(i)
                        if i in self.u_ind:
                            self.u_ind.remove(i)


                self.u_dep[:] = [i for i in self.u_dep if i not in self.kd_equivalents.values()]

                force_list = []
                for i in self.forces.keys():
                    force_list.append("(" + i + "," + self.forces[i] + ")")
                if self.u_dep:
                    u_dep_text = ", u_dependent=[" + ", ".join(self.u_dep) + "]"
                else:
                    u_dep_text = ""
                if self.dependent_variables:
                    velocity_constraints_text = ", velocity_constraints = velocity_constraints"
                else:
                    velocity_constraints_text = ""
                if ctx.parentCtx not in self.fr_expr:
                    self.write("kd_eqs = [" + ", ".join(self.kd_eqs) + "]\n")
                    self.write("forceList = " + "[" + ", ".join(force_list) + "]\n")
                    self.write("kane = _me.KanesMethod(" + self.newtonian + ", " + "q_ind=[" +
                            ",".join(self.q_ind) + "], " + "u_ind=[" +
                            ", ".join(self.u_ind) + "]" + u_dep_text + ", " +
                            "kd_eqs = kd_eqs" + velocity_constraints_text + ")\n")
                    self.write("fr, frstar = kane." + "kanes_equations([" +
                                ", ".join(self.bodies.values()) + "], forceList)\n")
                    self.fr_expr.append(ctx.parentCtx)
                self.kane_parsed = True
                self.setValue(ctx, func_name)

        def exitMatrices(self, ctx):
            # Tree annotation for Matrices which is a labeled subrule of the parser rule expr.

            # MO = [a, b; c, d]
            # we generate _sm.Matrix([a, b, c, d]).reshape(2, 2)
            # The reshape values are determined by counting the "," and ";" in the Autolev matrix

            # Eg:
            # [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]
            # semicolon_count = 3 and rows = 3+1 = 4
            # comma_count = 8 and cols = 8/rows + 1 = 8/4 + 1 = 3

            # TODO** Parse block matrices
            self.matrix_expr.append(ctx)
            l = []
            semicolon_count = 0
            comma_count = 0
            for i in range(ctx.matrix().getChildCount()):
                child = ctx.matrix().getChild(i)
                if child == AutolevParser.ExprContext:
                    l.append(self.getValue(child))
                elif child.getText() == ";":
                    semicolon_count += 1
                    l.append(",")
                elif child.getText() == ",":
                    comma_count += 1
                    l.append(",")
                else:
                    try:
                        try:
                            l.append(self.getValue(child))
                        except Exception:
                            l.append(self.symbol_table[child.getText().lower()])
                    except Exception:
                        l.append(child.getText().lower())
            num_of_rows = semicolon_count + 1
            num_of_cols = (comma_count//num_of_rows) + 1

            self.setValue(ctx, "_sm.Matrix(" + "".join(l) + ")" + ".reshape(" +
                          str(num_of_rows) + ", " + str(num_of_cols) + ")")

        def exitVectorOrDyadic(self, ctx):
            self.vector_expr.append(ctx)
            ch = ctx.vec()

            if ch.getChild(0).getText() == "0>":
                self.setValue(ctx, "0")

            elif ch.getChild(0).getText() == "1>>":
                self.setValue(ctx, "1>>")

            elif "_" in ch.ID().getText() and ch.ID().getText().count('_') == 2:
                vec_text = ch.getText().lower()
                v1, v2, v3 = ch.ID().getText().lower().split('_')

                if v1 == "p":
                    if self.type2[v2] == "point":
                        e2 = self.symbol_table2[v2]
                    elif self.type2[v2] == "particle":
                        e2 = self.symbol_table2[v2] + ".point"
                    if self.type2[v3] == "point":
                        e3 = self.symbol_table2[v3]
                    elif self.type2[v3] == "particle":
                        e3 = self.symbol_table2[v3] + ".point"
                    get_vec = e3 + ".pos_from(" + e2 + ")"
                    self.setValue(ctx, get_vec)

                elif v1 in ("w", "alf"):
                    if v1 == "w":
                        text = ".ang_vel_in("
                    elif v1 == "alf":
                        text = ".ang_acc_in("
                    if self.type2[v2] == "bodies":
                        e2 = self.symbol_table2[v2] + "_f"
                    elif self.type2[v2] == "frame":
                        e2 = self.symbol_table2[v2]
                    if self.type2[v3] == "bodies":
                        e3 = self.symbol_table2[v3] + "_f"
                    elif self.type2[v3] == "frame":
                        e3 = self.symbol_table2[v3]
                    get_vec = e2 + text + e3 + ")"
                    self.setValue(ctx, get_vec)

                elif v1 in ("v", "a"):
                    if v1 == "v":
                        text = ".vel("
                    elif v1 == "a":
                        text = ".acc("
                    if self.type2[v2] == "point":
                        e2 = self.symbol_table2[v2]
                    elif self.type2[v2] == "particle":
                        e2 = self.symbol_table2[v2] + ".point"
                    get_vec = e2 + text + self.symbol_table2[v3] + ")"
                    self.setValue(ctx, get_vec)

                else:
                    self.setValue(ctx, vec_text.replace(">", ""))

            else:
                vec_text = ch.getText().lower()
                name = self.symbol_table[vec_text]
                self.setValue(ctx, name)

        def exitIndexing(self, ctx):
            if ctx.getChildCount() == 4:
                try:
                    int_text = str(int(self.getValue(ctx.getChild(2))) - 1)
                except Exception:
                    int_text = self.getValue(ctx.getChild(2)) + " - 1"
                self.setValue(ctx, ctx.ID().getText().lower() + "[" + int_text + "]")
            elif ctx.getChildCount() == 6:
                try:
                    int_text1 = str(int(self.getValue(ctx.getChild(2))) - 1)
                except Exception:
                    int_text1 = self.getValue(ctx.getChild(2)) + " - 1"
                try:
                    int_text2 = str(int(self.getValue(ctx.getChild(4))) - 1)
                except Exception:
                    int_text2 = self.getValue(ctx.getChild(2)) + " - 1"
                self.setValue(ctx, ctx.ID().getText().lower() + "[" + int_text1 + ", " + int_text2 + "]")


        # ================== Subrules of parser rule expr (End) ====================== #

        def exitRegularAssign(self, ctx):
            # Handle assignments of type ID = expr
            if ctx.equals().getText() in ["=", "+=", "-=", "*=", "/="]:
                equals = ctx.equals().getText()
            elif ctx.equals().getText() == ":=":
                equals = " = "
            elif ctx.equals().getText() == "^=":
                equals = "**="

            try:
                a = ctx.ID().getText().lower() + "'"*ctx.diff().getText().count("'")
            except Exception:
                a = ctx.ID().getText().lower()

            if a in self.type.keys() and self.type[a] in ("motionvariable", "motionvariable'") and\
            self.type[ctx.expr().getText().lower()] in ("motionvariable", "motionvariable'"):
                b = ctx.expr().getText().lower()
                if "'" in b and "'" not in a:
                    a, b = b, a
                if not self.kane_parsed:
                    self.kd_eqs.append(a + "-" + b)
                    self.kd_equivalents.update({self.symbol_table[a]:
                                                self.symbol_table[b]})
                    self.kd_equivalents2.update({self.symbol_table[b]:
                                                    self.symbol_table[a]})

            if a in self.symbol_table.keys() and a in self.type.keys() and self.type[a] in ("variable", "motionvariable"):
                self.explicit.update({self.symbol_table[a]: self.getValue(ctx.expr())})

            else:
                if ctx.expr() in self.matrix_expr:
                    self.type.update({a: "matrix"})

                try:
                    b = self.symbol_table[a]
                except KeyError:
                    self.symbol_table[a] = a

                if "_" in a and a.count("_") == 1:
                    e1, e2 = a.split('_')
                    if e1 in self.type2.keys() and self.type2[e1] in ("frame", "bodies")\
                    and e2 in self.type2.keys() and self.type2[e2] in ("frame", "bodies"):
                        if self.type2[e1] == "bodies":
                            t1 = "_f"
                        else:
                            t1 = ""
                        if self.type2[e2] == "bodies":
                            t2 = "_f"
                        else:
                            t2 = ""

                        self.write(self.symbol_table2[e2] + t2 + ".orient(" + self.symbol_table2[e1] +
                                   t1 + ", 'DCM', " + self.getValue(ctx.expr()) + ")\n")
                    else:
                        self.write(self.symbol_table[a] + " " + equals + " " +
                                    self.getValue(ctx.expr()) + "\n")
                else:
                    self.write(self.symbol_table[a] + " " + equals + " " +
                                self.getValue(ctx.expr()) + "\n")

        def exitIndexAssign(self, ctx):
            # Handle assignments of type ID[index] = expr
                if ctx.equals().getText() in ["=", "+=", "-=", "*=", "/="]:
                    equals = ctx.equals().getText()
                elif ctx.equals().getText() == ":=":
                    equals = " = "
                elif ctx.equals().getText() == "^=":
                    equals = "**="

                text = ctx.ID().getText().lower()
                self.type.update({text: "matrix"})
                # Handle assignments of type ID[2] = expr
                if ctx.index().getChildCount() == 1:
                    if ctx.index().getChild(0).getText() == "1":
                        self.type.update({text: "matrix"})
                        self.symbol_table.update({text: text})
                        self.write(text + " = " + "_sm.Matrix([[0]])\n")
                        self.write(text + "[0] = " + self.getValue(ctx.expr()) + "\n")
                    else:
                        # m = m.row_insert(m.shape[0], _sm.Matrix([[0]]))
                        self.write(text + " = " + text +
                                   ".row_insert(" + text + ".shape[0]" + ", " + "_sm.Matrix([[0]])" + ")\n")
                        self.write(text + "[" + text + ".shape[0]-1" + "] = " + self.getValue(ctx.expr()) + "\n")

                # Handle assignments of type ID[2, 2] = expr
                elif ctx.index().getChildCount() == 3:
                    l = []
                    try:
                        l.append(str(int(self.getValue(ctx.index().getChild(0)))-1))
                    except Exception:
                        l.append(self.getValue(ctx.index().getChild(0)) + "-1")
                    l.append(",")
                    try:
                        l.append(str(int(self.getValue(ctx.index().getChild(2)))-1))
                    except Exception:
                        l.append(self.getValue(ctx.index().getChild(2)) + "-1")
                    self.write(self.symbol_table[ctx.ID().getText().lower()] +
                               "[" + "".join(l) + "]" + " " + equals + " " + self.getValue(ctx.expr()) + "\n")

        def exitVecAssign(self, ctx):
            # Handle assignments of the type vec = expr
            ch = ctx.vec()
            vec_text = ch.getText().lower()

            if "_" in ch.ID().getText():
                num = ch.ID().getText().count('_')

                if num == 2:
                    v1, v2, v3 = ch.ID().getText().lower().split('_')

                    if v1 == "p":
                        if self.type2[v2] == "point":
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        if self.type2[v3] == "point":
                            e3 = self.symbol_table2[v3]
                        elif self.type2[v3] == "particle":
                            e3 = self.symbol_table2[v3] + ".point"
                        # ab.set_pos(na, la*a.x)
                        self.write(e3 + ".set_pos(" + e2 + ", " + self.getValue(ctx.expr()) + ")\n")

                    elif v1 in ("w", "alf"):
                        if v1 == "w":
                            text = ".set_ang_vel("
                        elif v1 == "alf":
                            text = ".set_ang_acc("
                        # a.set_ang_vel(n, qad*a.z)
                        if self.type2[v2] == "bodies":
                            e2 = self.symbol_table2[v2] + "_f"
                        else:
                            e2 = self.symbol_table2[v2]
                        if self.type2[v3] == "bodies":
                            e3 = self.symbol_table2[v3] + "_f"
                        else:
                            e3 = self.symbol_table2[v3]
                        self.write(e2 + text + e3 + ", " + self.getValue(ctx.expr()) + ")\n")

                    elif v1 in ("v", "a"):
                        if v1 == "v":
                            text = ".set_vel("
                        elif v1 == "a":
                            text = ".set_acc("
                        if self.type2[v2] == "point":
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        self.write(e2 + text + self.symbol_table2[v3] +
                                   ", " + self.getValue(ctx.expr()) + ")\n")
                    elif v1 == "i":
                        if v2 in self.type2.keys() and self.type2[v2] == "bodies":
                            self.write(self.symbol_table2[v2] + ".inertia = (" + self.getValue(ctx.expr()) +
                            ", " + self.symbol_table2[v3] + ")\n")
                            self.inertia_point.update({v2: v3})
                        elif v2 in self.type2.keys() and self.type2[v2] == "particle":
                            self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                        else:
                            self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                    else:
                        self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")

                elif num == 1:
                    v1, v2 = ch.ID().getText().lower().split('_')

                    if v1 in ("force", "torque"):
                        if self.type2[v2] in ("point", "frame"):
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        self.symbol_table.update({vec_text: ch.ID().getText().lower()})

                        if e2 in self.forces.keys():
                            self.forces[e2] = self.forces[e2] + " + " + self.getValue(ctx.expr())
                        else:
                            self.forces.update({e2: self.getValue(ctx.expr())})
                        self.write(ch.ID().getText().lower() + " = " + self.forces[e2] + "\n")

                    else:
                        name = ch.ID().getText().lower()
                        self.symbol_table.update({vec_text: name})
                        self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                else:
                    name = ch.ID().getText().lower()
                    self.symbol_table.update({vec_text: name})
                    self.write(name + " " + ctx.getChild(1).getText() + " " + self.getValue(ctx.expr()) + "\n")
            else:
                name = ch.ID().getText().lower()
                self.symbol_table.update({vec_text: name})
                self.write(name + " " + ctx.getChild(1).getText() + " " + self.getValue(ctx.expr()) + "\n")

        def enterInputs2(self, ctx):
            self.in_inputs = True

        # Inputs
        def exitInputs2(self, ctx):
            # Stores numerical values given by the input command which
            # are used for codegen and numerical analysis.
            if ctx.getChildCount() == 3:
                try:
                    self.inputs.update({self.symbol_table[ctx.id_diff().getText().lower()]: self.getValue(ctx.expr(0))})
                except Exception:
                    self.inputs.update({ctx.id_diff().getText().lower(): self.getValue(ctx.expr(0))})
            elif ctx.getChildCount() == 4:
                try:
                    self.inputs.update({self.symbol_table[ctx.id_diff().getText().lower()]:
                    (self.getValue(ctx.expr(0)), self.getValue(ctx.expr(1)))})
                except Exception:
                    self.inputs.update({ctx.id_diff().getText().lower():
                    (self.getValue(ctx.expr(0)), self.getValue(ctx.expr(1)))})

            self.in_inputs = False

        def enterOutputs(self, ctx):
            self.in_outputs = True
        def exitOutputs(self, ctx):
            self.in_outputs = False

        def exitOutputs2(self, ctx):
            try:
                if "[" in ctx.expr(1).getText():
                    self.outputs.append(self.symbol_table[ctx.expr(0).getText().lower()] +
                                        ctx.expr(1).getText().lower())
                else:
                    self.outputs.append(self.symbol_table[ctx.expr(0).getText().lower()])

            except Exception:
                pass

        # Code commands
        def exitCodegen(self, ctx):
            # Handles the CODE() command ie the solvers and the codgen part.
            # Uses linsolve for the algebraic solvers and nsolve for non linear solvers.

            if ctx.functionCall().getChild(0).getText().lower() == "algebraic":
                matrix_name = self.getValue(ctx.functionCall().expr(0))
                e = []
                d = []
                for i in range(1, (ctx.functionCall().getChildCount()-2)//2):
                    a = self.getValue(ctx.functionCall().expr(i))
                    e.append(a)

                for i in self.inputs.keys():
                    d.append(i + ":" + self.inputs[i])
                self.write(matrix_name + "_list" + " = " + "[]\n")
                self.write("for i in " + matrix_name + ":  " + matrix_name +
                           "_list" + ".append(i.subs({" + ", ".join(d) + "}))\n")
                self.write("print(_sm.linsolve(" + matrix_name + "_list" + ", " + ",".join(e) + "))\n")

            elif ctx.functionCall().getChild(0).getText().lower() == "nonlinear":
                e = []
                d = []
                guess = []
                for i in range(1, (ctx.functionCall().getChildCount()-2)//2):
                    a = self.getValue(ctx.functionCall().expr(i))
                    e.append(a)
                #print(self.inputs)
                for i in self.inputs.keys():
                    if i in self.symbol_table.keys():
                        if type(self.inputs[i]) is tuple:
                            j, z = self.inputs[i]
                        else:
                            j = self.inputs[i]
                            z = ""
                        if i not in e:
                            if z == "deg":
                                d.append(i + ":" + "_np.deg2rad(" + j + ")")
                            else:
                                d.append(i + ":" + j)
                        else:
                            if z == "deg":
                                guess.append("_np.deg2rad(" + j + ")")
                            else:
                                guess.append(j)

                self.write("matrix_list" + " = " + "[]\n")
                self.write("for i in " + self.getValue(ctx.functionCall().expr(0)) + ":")
                self.write("matrix_list" + ".append(i.subs({" + ", ".join(d) + "}))\n")
                self.write("print(_sm.nsolve(matrix_list," + "(" + ",".join(e) + ")" +
                           ",(" + ",".join(guess) + ")" + "))\n")

            elif ctx.functionCall().getChild(0).getText().lower() in ["ode", "dynamics"] and self.include_numeric:
                if self.kane_type == "no_args":
                    for i in self.symbol_table.keys():
                        try:
                            if self.type[i] == "constants" or self.type[self.symbol_table[i]] == "constants":
                                self.constants.append(self.symbol_table[i])
                        except Exception:
                            pass
                    q_add_u = self.q_ind + self.q_dep + self.u_ind + self.u_dep
                    x0 = []
                    for i in q_add_u:
                        try:
                            if i in self.inputs.keys():
                                if type(self.inputs[i]) is tuple:
                                    if self.inputs[i][1] == "deg":
                                        x0.append(i + ":" + "_np.deg2rad(" + self.inputs[i][0] + ")")
                                    else:
                                        x0.append(i + ":" + self.inputs[i][0])
                                else:
                                    x0.append(i + ":" + self.inputs[i])
                            elif self.kd_equivalents[i] in self.inputs.keys():
                                if type(self.inputs[self.kd_equivalents[i]]) is tuple:
                                    x0.append(i + ":" + self.inputs[self.kd_equivalents[i]][0])
                                else:
                                    x0.append(i + ":" + self.inputs[self.kd_equivalents[i]])
                        except Exception:
                            pass

                    # numerical constants
                    numerical_constants = []
                    for i in self.constants:
                        if i in self.inputs.keys():
                            if type(self.inputs[i]) is tuple:
                                numerical_constants.append(self.inputs[i][0])
                            else:
                                numerical_constants.append(self.inputs[i])

                    # t = linspace
                    t_final = self.inputs["tfinal"]
                    integ_stp = self.inputs["integstp"]

                    self.write("from pydy.system import System\n")
                    const_list = []
                    if numerical_constants:
                        for i in range(len(self.constants)):
                            const_list.append(self.constants[i] + ":" + numerical_constants[i])
                    specifieds = []
                    if self.t:
                        specifieds.append("_me.dynamicsymbols('t')" + ":" + "lambda x, t: t")

                    for i in self.inputs:
                        if i in self.symbol_table.keys() and self.symbol_table[i] not in\
                        self.constants + self.q_ind + self.q_dep + self.u_ind + self.u_dep:
                            specifieds.append(self.symbol_table[i] + ":" + self.inputs[i])

                    self.write("sys = System(kane, constants = {" + ", ".join(const_list) + "},\n" +
                               "specifieds={" + ", ".join(specifieds) + "},\n" +
                               "initial_conditions={" + ", ".join(x0) + "},\n" +
                               "times = _np.linspace(0.0, " + str(t_final) + ", " + str(t_final) +
                               "/" + str(integ_stp) + "))\n\ny=sys.integrate()\n")

                    # For outputs other than qs and us.
                    other_outputs = []
                    for i in self.outputs:
                        if i not in q_add_u:
                            if "[" in i:
                                other_outputs.append((i[:-3] + i[-2], i[:-3] + "[" + str(int(i[-2])-1) + "]"))
                            else:
                                other_outputs.append((i, i))

                    for i in other_outputs:
                        self.write(i[0] + "_out" + " = " + "[]\n")
                    if other_outputs:
                        self.write("for i in y:\n")
                        self.write("    q_u_dict = dict(zip(sys.coordinates+sys.speeds, i))\n")
                        for i in other_outputs:
                            self.write(" "*4 + i[0] + "_out" + ".append(" + i[1] + ".subs(q_u_dict)" +
                                    ".subs(sys.constants).evalf())\n")

        # Standalone function calls (used for dual functions)
        def exitFunctionCall(self, ctx):
            # Basically deals with standalone function calls ie functions which are not a part of
            # expressions and assignments. Autolev Dual functions can both appear in standalone
            # function calls and also on the right hand side as part of expr or assignment.

            # Dual functions are indicated by a * in the comments below

            # Checks if the function is a statement on its own
            if ctx.parentCtx.getRuleIndex() == AutolevParser.RULE_stat:
                func_name = ctx.getChild(0).getText().lower()
                # Expand(E, n:m) *
                if func_name == "expand":
                    # If the first argument is a pre declared variable.
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]
                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([i.expand() for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(symbol + " = " + symbol + "." + "expand()\n")

                # Factor(E, x) *
                elif func_name == "factor":
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]
                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([_sm.factor(i," + self.getValue(ctx.expr(1)) +
                                   ") for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(expr + " = " + "_sm.factor(" + expr + ", " +
                                   self.getValue(ctx.expr(1)) + ")\n")

                # Solve(Zero, x, y)
                elif func_name == "solve":
                    l = []
                    l2 = []
                    num = 0
                    for i in range(1, ctx.getChildCount()):
                        if ctx.getChild(i).getText() == ",":
                            num+=1
                        try:
                            l.append(self.getValue(ctx.getChild(i)))
                        except Exception:
                            l.append(ctx.getChild(i).getText())

                        if i != 2:
                            try:
                                l2.append(self.getValue(ctx.getChild(i)))
                            except Exception:
                                pass

                    for i in l2:
                        self.explicit.update({i: "_sm.solve" + "".join(l) + "[" + i + "]"})

                    self.write("print(_sm.solve" + "".join(l) + ")\n")

                # Arrange(y, n, x) *
                elif func_name == "arrange":
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]

                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([i.collect(" + self.getValue(ctx.expr(2)) +
                                   ")" + "for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(self.getValue(ctx.expr(0)) + ".collect(" +
                                   self.getValue(ctx.expr(2)) + ")\n")

                # Eig(M, EigenValue, EigenVec)
                elif func_name == "eig":
                    self.symbol_table.update({ctx.expr(1).getText().lower(): ctx.expr(1).getText().lower()})
                    self.symbol_table.update({ctx.expr(2).getText().lower(): ctx.expr(2).getText().lower()})
                    # _sm.Matrix([i.evalf() for i in (i_s_so).eigenvals().keys()])
                    self.write(ctx.expr(1).getText().lower() + " = " +
                               "_sm.Matrix([i.evalf() for i in " +
                               "(" + self.getValue(ctx.expr(0)) + ")" + ".eigenvals().keys()])\n")
                    # _sm.Matrix([i[2][0].evalf() for i in (i_s_o).eigenvects()]).reshape(i_s_o.shape[0], i_s_o.shape[1])
                    self.write(ctx.expr(2).getText().lower() + " = " +
                               "_sm.Matrix([i[2][0].evalf() for i in " + "(" + self.getValue(ctx.expr(0)) + ")" +
                               ".eigenvects()]).reshape(" + self.getValue(ctx.expr(0)) + ".shape[0], " +
                               self.getValue(ctx.expr(0)) + ".shape[1])\n")

                # Simprot(N, A, 3, qA)
                elif func_name == "simprot":
                    # A.orient(N, 'Axis', qA, N.z)
                    if self.type2[ctx.expr(0).getText().lower()] == "frame":
                        frame1 = self.symbol_table2[ctx.expr(0).getText().lower()]
                    elif self.type2[ctx.expr(0).getText().lower()] == "bodies":
                        frame1 = self.symbol_table2[ctx.expr(0).getText().lower()] + "_f"
                    if self.type2[ctx.expr(1).getText().lower()] == "frame":
                        frame2 = self.symbol_table2[ctx.expr(1).getText().lower()]
                    elif self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        frame2 = self.symbol_table2[ctx.expr(1).getText().lower()] + "_f"
                    e2 = ""
                    if ctx.expr(2).getText()[0] == "-":
                        e2 = "-1*"
                    if ctx.expr(2).getText() in ("1", "-1"):
                        e = frame1 + ".x"
                    elif ctx.expr(2).getText() in ("2", "-2"):
                        e = frame1 + ".y"
                    elif ctx.expr(2).getText() in ("3", "-3"):
                        e = frame1 + ".z"
                    else:
                        e = self.getValue(ctx.expr(2))
                        e2 = ""

                    if "degrees" in self.settings.keys() and self.settings["degrees"] == "off":
                        value = self.getValue(ctx.expr(3))
                    else:
                        if ctx.expr(3) in self.numeric_expr:
                            value = "_np.deg2rad(" + self.getValue(ctx.expr(3)) + ")"
                        else:
                            value = self.getValue(ctx.expr(3))
                    self.write(frame2 + ".orient(" + frame1 +
                               ", " + "'Axis'" + ", " + "[" + value +
                               ", " + e2 + e + "]" + ")\n")

                # Express(A2>, B) *
                elif func_name == "express":
                    if self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        f = "_f"
                    else:
                        f = ""

                    if '_' in ctx.expr(0).getText().lower() and ctx.expr(0).getText().count('_') == 2:
                        vec = ctx.expr(0).getText().lower().replace(">", "").split('_')
                        v1 = self.symbol_table2[vec[1]]
                        v2 = self.symbol_table2[vec[2]]
                        if vec[0] == "p":
                            self.write(v2 + ".set_pos(" + v1 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        elif vec[0] == "v":
                            self.write(v1 + ".set_vel(" + v2 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        elif vec[0] == "a":
                            self.write(v1 + ".set_acc(" + v2 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        else:
                            self.write(self.getValue(ctx.expr(0)) + " = " + "(" + self.getValue(ctx.expr(0)) + ")" + ".express(" +
                                        self.symbol_table2[ctx.expr(1).getText().lower()] + f + ")\n")
                    else:
                        self.write(self.getValue(ctx.expr(0)) + " = " + "(" + self.getValue(ctx.expr(0)) + ")" + ".express(" +
                                    self.symbol_table2[ctx.expr(1).getText().lower()] + f + ")\n")

                # Angvel(A, B)
                elif func_name == "angvel":
                    self.write("print(" + self.symbol_table2[ctx.expr(1).getText().lower()] +
                               ".ang_vel_in(" + self.symbol_table2[ctx.expr(0).getText().lower()] + "))\n")

                # v2pts(N, A, O, P)
                elif func_name in ("v2pts", "a2pts", "v2pt", "a1pt"):
                    if func_name == "v2pts":
                        text = ".v2pt_theory("
                    elif func_name == "a2pts":
                        text = ".a2pt_theory("
                    elif func_name == "v1pt":
                        text = ".v1pt_theory("
                    elif func_name == "a1pt":
                        text = ".a1pt_theory("
                    if self.type2[ctx.expr(1).getText().lower()] == "frame":
                        frame = self.symbol_table2[ctx.expr(1).getText().lower()]
                    elif self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        frame = self.symbol_table2[ctx.expr(1).getText().lower()] + "_f"
                    expr_list = []
                    for i in range(2, 4):
                        if self.type2[ctx.expr(i).getText().lower()] == "point":
                            expr_list.append(self.symbol_table2[ctx.expr(i).getText().lower()])
                        elif self.type2[ctx.expr(i).getText().lower()] == "particle":
                            expr_list.append(self.symbol_table2[ctx.expr(i).getText().lower()] + ".point")

                    self.write(expr_list[1] + text + expr_list[0] +
                               "," + self.symbol_table2[ctx.expr(0).getText().lower()] + "," +
                               frame + ")\n")

                # Gravity(g*N1>)
                elif func_name == "gravity":
                    for i in self.bodies.keys():
                        if self.type2[i] == "bodies":
                            e = self.symbol_table2[i] + ".masscenter"
                        elif self.type2[i] == "particle":
                            e = self.symbol_table2[i] + ".point"
                        if e in self.forces.keys():
                            self.forces[e] = self.forces[e] + self.symbol_table2[i] +\
                                             ".mass*(" + self.getValue(ctx.expr(0)) + ")"
                        else:
                            self.forces.update({e: self.symbol_table2[i] +
                                               ".mass*(" + self.getValue(ctx.expr(0)) + ")"})
                        self.write("force_" + i + " = " + self.forces[e] + "\n")

                # Explicit(EXPRESS(IMPLICIT>,C))
                elif func_name == "explicit":
                    if ctx.expr(0) in self.vector_expr:
                        self.vector_expr.append(ctx)
                    expr = self.getValue(ctx.expr(0))
                    if self.explicit.keys():
                        explicit_list = []
                        for i in self.explicit.keys():
                            explicit_list.append(i + ":" + self.explicit[i])
                        if '_' in ctx.expr(0).getText().lower() and ctx.expr(0).getText().count('_') == 2:
                            vec = ctx.expr(0).getText().lower().replace(">", "").split('_')
                            v1 = self.symbol_table2[vec[1]]
                            v2 = self.symbol_table2[vec[2]]
                            if vec[0] == "p":
                                self.write(v2 + ".set_pos(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            elif vec[0] == "v":
                                self.write(v2 + ".set_vel(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            elif vec[0] == "a":
                                self.write(v2 + ".set_acc(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            else:
                                self.write(expr + " = " + "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})\n")
                        else:
                            self.write(expr + " = " + "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})\n")

                # Force(O/Q, -k*Stretch*Uvec>)
                elif func_name in ("force", "torque"):

                    if "/" in ctx.expr(0).getText().lower():
                        p1 = ctx.expr(0).getText().lower().split('/')[0]
                        p2 = ctx.expr(0).getText().lower().split('/')[1]
                        if self.type2[p1] in ("point", "frame"):
                            pt1 = self.symbol_table2[p1]
                        elif self.type2[p1] == "particle":
                            pt1 = self.symbol_table2[p1] + ".point"
                        if self.type2[p2] in ("point", "frame"):
                            pt2 = self.symbol_table2[p2]
                        elif self.type2[p2] == "particle":
                            pt2 = self.symbol_table2[p2] + ".point"
                        if pt1 in self.forces.keys():
                            self.forces[pt1] = self.forces[pt1] + " + -1*("+self.getValue(ctx.expr(1)) + ")"
                            self.write("force_" + p1 + " = " + self.forces[pt1] + "\n")
                        else:
                            self.forces.update({pt1: "-1*("+self.getValue(ctx.expr(1)) + ")"})
                            self.write("force_" + p1 + " = " + self.forces[pt1] + "\n")
                        if pt2 in self.forces.keys():
                            self.forces[pt2] = self.forces[pt2] + "+ " + self.getValue(ctx.expr(1))
                            self.write("force_" + p2 + " = " + self.forces[pt2] + "\n")
                        else:
                            self.forces.update({pt2: self.getValue(ctx.expr(1))})
                            self.write("force_" + p2 + " = " + self.forces[pt2] + "\n")

                    elif ctx.expr(0).getChildCount() == 1:
                        p1 = ctx.expr(0).getText().lower()
                        if self.type2[p1] in ("point", "frame"):
                            pt1 = self.symbol_table2[p1]
                        elif self.type2[p1] == "particle":
                            pt1 = self.symbol_table2[p1] + ".point"
                        if pt1 in self.forces.keys():
                            self.forces[pt1] = self.forces[pt1] + "+ -1*(" + self.getValue(ctx.expr(1)) + ")"
                        else:
                            self.forces.update({pt1: "-1*(" + self.getValue(ctx.expr(1)) + ")"})

                # Constrain(Dependent[qB])
                elif func_name == "constrain":
                    if ctx.getChild(2).getChild(0).getText().lower() == "dependent":
                        self.write("velocity_constraints = [i for i in dependent]\n")
                    x = (ctx.expr(0).getChildCount()-2)//2
                    for i in range(x):
                        self.dependent_variables.append(self.getValue(ctx.expr(0).expr(i)))

                # Kane()
                elif func_name == "kane":
                    if ctx.getChildCount() == 3:
                        self.kane_type = "no_args"

        # Settings
        def exitSettings(self, ctx):
            # Stores settings like Complex on/off, Degrees on/off etc in self.settings.
            try:
                self.settings.update({ctx.getChild(0).getText().lower():
                                     ctx.getChild(1).getText().lower()})
            except Exception:
                pass

        def exitMassDecl2(self, ctx):
            # Used for declaring the masses of particles and rigidbodies.
            particle = self.symbol_table2[ctx.getChild(0).getText().lower()]
            if ctx.getText().count("=") == 2:
                if ctx.expr().expr(1) in self.numeric_expr:
                    e = "_sm.S(" + self.getValue(ctx.expr().expr(1)) + ")"
                else:
                    e = self.getValue(ctx.expr().expr(1))
                self.symbol_table.update({ctx.expr().expr(0).getText().lower(): ctx.expr().expr(0).getText().lower()})
                self.write(ctx.expr().expr(0).getText().lower() + " = " + e + "\n")
                mass = ctx.expr().expr(0).getText().lower()
            else:
                try:
                    if ctx.expr() in self.numeric_expr:
                        mass = "_sm.S(" + self.getValue(ctx.expr()) + ")"
                    else:
                        mass = self.getValue(ctx.expr())
                except Exception:
                    a_text = ctx.expr().getText().lower()
                    self.symbol_table.update({a_text: a_text})
                    self.type.update({a_text: "constants"})
                    self.write(a_text + " = " + "_sm.symbols('" + a_text + "')\n")
                    mass = a_text

            self.write(particle + ".mass = " + mass + "\n")

        def exitInertiaDecl(self, ctx):
            inertia_list = []
            try:
                ctx.ID(1).getText()
                num = 5
            except Exception:
                num = 2
            for i in range((ctx.getChildCount()-num)//2):
                try:
                    if ctx.expr(i) in self.numeric_expr:
                        inertia_list.append("_sm.S(" + self.getValue(ctx.expr(i)) + ")")
                    else:
                        inertia_list.append(self.getValue(ctx.expr(i)))
                except Exception:
                    a_text = ctx.expr(i).getText().lower()
                    self.symbol_table.update({a_text: a_text})
                    self.type.update({a_text: "constants"})
                    self.write(a_text + " = " + "_sm.symbols('" + a_text + "')\n")
                    inertia_list.append(a_text)

            if len(inertia_list) < 6:
                for i in range(6-len(inertia_list)):
                    inertia_list.append("0")
            # body_a.inertia = (_me.inertia(body_a, I1, I2, I3, 0, 0, 0), body_a_cm)
            try:
                frame = self.symbol_table2[ctx.ID(1).getText().lower()]
                point = self.symbol_table2[ctx.ID(0).getText().lower().split('_')[1]]
                body = self.symbol_table2[ctx.ID(0).getText().lower().split('_')[0]]
                self.inertia_point.update({ctx.ID(0).getText().lower().split('_')[0]
                                          : ctx.ID(0).getText().lower().split('_')[1]})
                self.write(body + ".inertia" + " = " + "(_me.inertia(" + frame + ", " +
                           ", ".join(inertia_list) + "), " + point + ")\n")

            except Exception:
                body_name = self.symbol_table2[ctx.ID(0).getText().lower()]
                body_name_cm = body_name + "_cm"
                self.inertia_point.update({ctx.ID(0).getText().lower(): ctx.ID(0).getText().lower() + "o"})
                self.write(body_name + ".inertia" + " = " + "(_me.inertia(" + body_name + "_f" + ", " +
                           ", ".join(inertia_list) + "), " + body_name_cm + ")\n")
<|endoftext|>import os
import subprocess
import glob

from sympy.utilities.misc import debug

here = os.path.dirname(__file__)
grammar_file = os.path.abspath(os.path.join(here, "Autolev.g4"))
dir_autolev_antlr = os.path.join(here, "_antlr")

header = '''\
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt
'''


def check_antlr_version():
    debug("Checking antlr4 version...")

    try:
        debug(subprocess.check_output(["antlr4"])
              .decode('utf-8').split("\n")[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        debug("The 'antlr4' command line tool is not installed, "
              "or not on your PATH.\n"
              "> Please refer to the README.md file for more information.")
        return False


def build_parser(output_dir=dir_autolev_antlr):
    check_antlr_version()

    debug("Updating ANTLR-generated code in {}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "__init__.py"), "w+") as fp:
        fp.write(header)

    args = [
        "antlr4",
        grammar_file,
        "-o", output_dir,
        "-no-visitor",
    ]

    debug("Running code generation...\n\t$ {}".format(" ".join(args)))
    subprocess.check_output(args, cwd=output_dir)

    debug("Applying headers, removing unnecessary files and renaming...")
    # Handle case insensitive file systems. If the files are already
    # generated, they will be written to autolev* but Autolev*.* won't match them.
    for path in (glob.glob(os.path.join(output_dir, "Autolev*.*")) or
        glob.glob(os.path.join(output_dir, "autolev*.*"))):

        # Remove files ending in .interp or .tokens as they are not needed.
        if not path.endswith(".py"):
            os.unlink(path)
            continue

        new_path = os.path.join(output_dir, os.path.basename(path).lower())
        with open(path, 'r') as f:
            lines = [line.rstrip().replace('AutolevParser import', 'autolevparser import') +'\n'
                     for line in f]

        os.unlink(path)

        with open(new_path, "w") as out_file:
            offset = 0
            while lines[offset].startswith('#'):
                offset += 1
            out_file.write(header)
            out_file.writelines(lines[offset:])

        debug("\t{}".format(new_path))

    return True


if __name__ == "__main__":
    build_parser()
<|endoftext|>from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on

@doctest_depends_on(modules=('antlr4',))
def parse_autolev(autolev_code, include_numeric=False):
    """Parses Autolev code (version 4.1) to SymPy code.

    Parameters
    =========
    autolev_code : Can be an str or any object with a readlines() method (such as a file handle or StringIO).
    include_numeric : boolean, optional
        If True NumPy, PyDy, or other numeric code is included for numeric evaluation lines in the Autolev code.

    Returns
    =======
    sympy_code : str
        Equivalent SymPy and/or numpy/pydy code as the input code.


    Example (Double Pendulum)
    =========================
    >>> my_al_text = ("MOTIONVARIABLES' Q{2}', U{2}'",
    ... "CONSTANTS L,M,G",
    ... "NEWTONIAN N",
    ... "FRAMES A,B",
    ... "SIMPROT(N, A, 3, Q1)",
    ... "SIMPROT(N, B, 3, Q2)",
    ... "W_A_N>=U1*N3>",
    ... "W_B_N>=U2*N3>",
    ... "POINT O",
    ... "PARTICLES P,R",
    ... "P_O_P> = L*A1>",
    ... "P_P_R> = L*B1>",
    ... "V_O_N> = 0>",
    ... "V2PTS(N, A, O, P)",
    ... "V2PTS(N, B, P, R)",
    ... "MASS P=M, R=M",
    ... "Q1' = U1",
    ... "Q2' = U2",
    ... "GRAVITY(G*N1>)",
    ... "ZERO = FR() + FRSTAR()",
    ... "KANE()",
    ... "INPUT M=1,G=9.81,L=1",
    ... "INPUT Q1=.1,Q2=.2,U1=0,U2=0",
    ... "INPUT TFINAL=10, INTEGSTP=.01",
    ... "CODE DYNAMICS() some_filename.c")
    >>> my_al_text = '\\n'.join(my_al_text)
    >>> from sympy.parsing.autolev import parse_autolev
    >>> print(parse_autolev(my_al_text, include_numeric=True))
    import sympy.physics.mechanics as _me
    import sympy as _sm
    import math as m
    import numpy as _np
    <BLANKLINE>
    q1, q2, u1, u2 = _me.dynamicsymbols('q1 q2 u1 u2')
    q1_d, q2_d, u1_d, u2_d = _me.dynamicsymbols('q1_ q2_ u1_ u2_', 1)
    l, m, g = _sm.symbols('l m g', real=True)
    frame_n = _me.ReferenceFrame('n')
    frame_a = _me.ReferenceFrame('a')
    frame_b = _me.ReferenceFrame('b')
    frame_a.orient(frame_n, 'Axis', [q1, frame_n.z])
    frame_b.orient(frame_n, 'Axis', [q2, frame_n.z])
    frame_a.set_ang_vel(frame_n, u1*frame_n.z)
    frame_b.set_ang_vel(frame_n, u2*frame_n.z)
    point_o = _me.Point('o')
    particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
    particle_r = _me.Particle('r', _me.Point('r_pt'), _sm.Symbol('m'))
    particle_p.point.set_pos(point_o, l*frame_a.x)
    particle_r.point.set_pos(particle_p.point, l*frame_b.x)
    point_o.set_vel(frame_n, 0)
    particle_p.point.v2pt_theory(point_o,frame_n,frame_a)
    particle_r.point.v2pt_theory(particle_p.point,frame_n,frame_b)
    particle_p.mass = m
    particle_r.mass = m
    force_p = particle_p.mass*(g*frame_n.x)
    force_r = particle_r.mass*(g*frame_n.x)
    kd_eqs = [q1_d - u1, q2_d - u2]
    forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x)), (particle_r.point,particle_r.mass*(g*frame_n.x))]
    kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u1, u2], kd_eqs = kd_eqs)
    fr, frstar = kane.kanes_equations([particle_p, particle_r], forceList)
    zero = fr+frstar
    from pydy.system import System
    sys = System(kane, constants = {l:1, m:1, g:9.81},
    specifieds={},
    initial_conditions={q1:.1, q2:.2, u1:0, u2:0},
    times = _np.linspace(0.0, 10, 10/.01))
    <BLANKLINE>
    y=sys.integrate()
    <BLANKLINE>
    """

    _autolev = import_module(
        'sympy.parsing.autolev._parse_autolev_antlr',
        import_kwargs={'fromlist': ['X']})

    if _autolev is not None:
        return _autolev.parse_autolev(autolev_code, include_numeric)
<|endoftext|>from importlib.metadata import version
from sympy.external import import_module


autolevparser = import_module('sympy.parsing.autolev._antlr.autolevparser',
                              import_kwargs={'fromlist': ['AutolevParser']})
autolevlexer = import_module('sympy.parsing.autolev._antlr.autolevlexer',
                             import_kwargs={'fromlist': ['AutolevLexer']})
autolevlistener = import_module('sympy.parsing.autolev._antlr.autolevlistener',
                                import_kwargs={'fromlist': ['AutolevListener']})

AutolevParser = getattr(autolevparser, 'AutolevParser', None)
AutolevLexer = getattr(autolevlexer, 'AutolevLexer', None)
AutolevListener = getattr(autolevlistener, 'AutolevListener', None)


def parse_autolev(autolev_code, include_numeric):
    antlr4 = import_module('antlr4')
    if not antlr4 or not version('antlr4-python3-runtime').startswith('4.11'):
        raise ImportError("Autolev parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime)"
                          " conda (antlr-python-runtime), version 4.11")
    try:
        l = autolev_code.readlines()
        input_stream = antlr4.InputStream("".join(l))
    except Exception:
        input_stream = antlr4.InputStream(autolev_code)

    if AutolevListener:
        from ._listener_autolev_antlr import MyListener
        lexer = AutolevLexer(input_stream)
        token_stream = antlr4.CommonTokenStream(lexer)
        parser = AutolevParser(token_stream)
        tree = parser.prog()
        my_listener = MyListener(include_numeric)
        walker = antlr4.ParseTreeWalker()
        walker.walk(my_listener, tree)
        return "".join(my_listener.output_code)
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
e1 = (x+y)**2+(x-y)**3
e2 = (x-y)**2
e3 = x**2+y**2+2*x*y
m1 = _sm.Matrix([e1,e2]).reshape(2, 1)
m2 = _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)
m3 = m1+_sm.Matrix([x,y]).reshape(2, 1)
am = _sm.Matrix([i.expand() for i in m1]).reshape((m1).shape[0], (m1).shape[1])
cm = _sm.Matrix([i.expand() for i in _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)]).reshape((_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[0], (_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[1])
em = _sm.Matrix([i.expand() for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
f = (e1).expand()
g = (e2).expand()
a = _sm.factor((e3), x)
bm = _sm.Matrix([_sm.factor(i, x) for i in m1]).reshape((m1).shape[0], (m1).shape[1])
cm = _sm.Matrix([_sm.factor(i, x) for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
a = (e3).diff(x)
b = (e3).diff(y)
cm = _sm.Matrix([i.diff(x) for i in m2]).reshape((m2).shape[0], (m2).shape[1])
dm = _sm.Matrix([i.diff(x) for i in m1+_sm.Matrix([x,y]).reshape(2, 1)]).reshape((m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[0], (m1+_sm.Matrix([x,y]).reshape(2, 1)).shape[1])
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_b.orient(frame_a, 'DCM', _sm.Matrix([1,0,0,1,0,0,1,0,0]).reshape(3, 3))
v1 = x*frame_a.x+y*frame_a.y+x*y*frame_a.z
e = (v1).diff(x, frame_b)
fm = _sm.Matrix([i.diff(_sm.Symbol('t')) for i in m1]).reshape((m1).shape[0], (m1).shape[1])
gm = _sm.Matrix([i.diff(_sm.Symbol('t')) for i in _sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)]).reshape((_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[0], (_sm.Matrix([(x+y)**2,(x-y)**2]).reshape(1, 2)).shape[1])
h = (v1).dt(frame_b)
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

f = _sm.S(3)
g = _sm.S(9.81)
a, b = _sm.symbols('a b', real=True)
s, s1 = _sm.symbols('s s1', real=True)
s2, s3 = _sm.symbols('s2 s3', real=True, nonnegative=True)
s4 = _sm.symbols('s4', real=True, nonpositive=True)
k1, k2, k3, k4, l1, l2, l3, p11, p12, p13, p21, p22, p23 = _sm.symbols('k1 k2 k3 k4 l1 l2 l3 p11 p12 p13 p21 p22 p23', real=True)
c11, c12, c13, c21, c22, c23 = _sm.symbols('c11 c12 c13 c21 c22 c23', real=True)
e1 = a*f+s2-g
e2 = f**2+k3*k2*g
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_n = _me.ReferenceFrame('n')
x1, x2, x3 = _me.dynamicsymbols('x1 x2 x3')
l = _sm.symbols('l', real=True)
v1 = x1*frame_a.x+x2*frame_a.y+x3*frame_a.z
v2 = x1*frame_b.x+x2*frame_b.y+x3*frame_b.z
v3 = x1*frame_n.x+x2*frame_n.y+x3*frame_n.z
v = v1+v2+v3
point_c = _me.Point('c')
point_d = _me.Point('d')
point_po1 = _me.Point('po1')
point_po2 = _me.Point('po2')
point_po3 = _me.Point('po3')
particle_l = _me.Particle('l', _me.Point('l_pt'), _sm.Symbol('m'))
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
particle_p3 = _me.Particle('p3', _me.Point('p3_pt'), _sm.Symbol('m'))
body_s_cm = _me.Point('s_cm')
body_s_cm.set_vel(frame_n, 0)
body_s_f = _me.ReferenceFrame('s_f')
body_s = _me.RigidBody('s', body_s_cm, body_s_f, _sm.symbols('m'), (_me.outer(body_s_f.x,body_s_f.x),body_s_cm))
body_r1_cm = _me.Point('r1_cm')
body_r1_cm.set_vel(frame_n, 0)
body_r1_f = _me.ReferenceFrame('r1_f')
body_r1 = _me.RigidBody('r1', body_r1_cm, body_r1_f, _sm.symbols('m'), (_me.outer(body_r1_f.x,body_r1_f.x),body_r1_cm))
body_r2_cm = _me.Point('r2_cm')
body_r2_cm.set_vel(frame_n, 0)
body_r2_f = _me.ReferenceFrame('r2_f')
body_r2 = _me.RigidBody('r2', body_r2_cm, body_r2_f, _sm.symbols('m'), (_me.outer(body_r2_f.x,body_r2_f.x),body_r2_cm))
v4 = x1*body_s_f.x+x2*body_s_f.y+x3*body_s_f.z
body_s_cm.set_pos(point_c, l*frame_n.x)
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
a, b = _sm.symbols('a b', real=True)
e = a*(b*x+y)**2
m = _sm.Matrix([e,e]).reshape(2, 1)
e = e.expand()
m = _sm.Matrix([i.expand() for i in m]).reshape((m).shape[0], (m).shape[1])
e = _sm.factor(e, x)
m = _sm.Matrix([_sm.factor(i,x) for i in m]).reshape((m).shape[0], (m).shape[1])
eqn = _sm.Matrix([[0]])
eqn[0] = a*x+b*y
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
eqn[eqn.shape[0]-1] = 2*a*x-3*b*y
print(_sm.solve(eqn,x,y))
rhs_y = _sm.solve(eqn,x,y)[y]
e = (x+y)**2+2*x**2
e.collect(x)
a, b, c = _sm.symbols('a b c', real=True)
m = _sm.Matrix([a,b,c,0]).reshape(2, 2)
m2 = _sm.Matrix([i.subs({a:1,b:2,c:3}) for i in m]).reshape((m).shape[0], (m).shape[1])
eigvalue = _sm.Matrix([i.evalf() for i in (m2).eigenvals().keys()])
eigvec = _sm.Matrix([i[2][0].evalf() for i in (m2).eigenvects()]).reshape(m2.shape[0], m2.shape[1])
frame_n = _me.ReferenceFrame('n')
frame_a = _me.ReferenceFrame('a')
frame_a.orient(frame_n, 'Axis', [x, frame_n.x])
frame_a.orient(frame_n, 'Axis', [_sm.pi/2, frame_n.x])
c1, c2, c3 = _sm.symbols('c1 c2 c3', real=True)
v = c1*frame_a.x+c2*frame_a.y+c3*frame_a.z
point_o = _me.Point('o')
point_p = _me.Point('p')
point_o.set_pos(point_p, c1*frame_a.x)
v = (v).express(frame_n)
point_o.set_pos(point_p, (point_o.pos_from(point_p)).express(frame_n))
frame_a.set_ang_vel(frame_n, c3*frame_a.z)
print(frame_n.ang_vel_in(frame_a))
point_p.v2pt_theory(point_o,frame_n,frame_a)
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
particle_p2.point.v2pt_theory(particle_p1.point,frame_n,frame_a)
point_p.a2pt_theory(particle_p1.point,frame_n,frame_a)
body_b1_cm = _me.Point('b1_cm')
body_b1_cm.set_vel(frame_n, 0)
body_b1_f = _me.ReferenceFrame('b1_f')
body_b1 = _me.RigidBody('b1', body_b1_cm, body_b1_f, _sm.symbols('m'), (_me.outer(body_b1_f.x,body_b1_f.x),body_b1_cm))
body_b2_cm = _me.Point('b2_cm')
body_b2_cm.set_vel(frame_n, 0)
body_b2_f = _me.ReferenceFrame('b2_f')
body_b2 = _me.RigidBody('b2', body_b2_cm, body_b2_f, _sm.symbols('m'), (_me.outer(body_b2_f.x,body_b2_f.x),body_b2_cm))
g = _sm.symbols('g', real=True)
force_p1 = particle_p1.mass*(g*frame_n.x)
force_p2 = particle_p2.mass*(g*frame_n.x)
force_b1 = body_b1.mass*(g*frame_n.x)
force_b2 = body_b2.mass*(g*frame_n.x)
z = _me.dynamicsymbols('z')
v = x*frame_a.x+y*frame_a.z
point_o.set_pos(point_p, x*frame_a.x+y*frame_a.y)
v = (v).subs({x:2*z, y:z})
point_o.set_pos(point_p, (point_o.pos_from(point_p)).subs({x:2*z, y:z}))
force_o = -1*(x*y*frame_a.x)
force_p1 = particle_p1.mass*(g*frame_n.x)+ x*y*frame_a.x
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
a11, a12, a21, a22, b1, b2 = _sm.symbols('a11 a12 a21 a22 b1 b2', real=True)
eqn = _sm.Matrix([[0]])
eqn[0] = a11*x+a12*y-b1
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
eqn[eqn.shape[0]-1] = a21*x+a22*y-b2
eqn_list = []
for i in eqn:  eqn_list.append(i.subs({a11:2, a12:5, a21:3, a22:4, b1:7, b2:6}))
print(_sm.linsolve(eqn_list, x,y))
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
q1, q2, q3 = _me.dynamicsymbols('q1 q2 q3')
frame_b.orient(frame_a, 'Axis', [q3, frame_a.x])
dcm = frame_a.dcm(frame_b)
m = dcm*3-frame_a.dcm(frame_b)
r = _me.dynamicsymbols('r')
circle_area = _sm.pi*r**2
u, a = _me.dynamicsymbols('u a')
x, y = _me.dynamicsymbols('x y')
s = u*_me.dynamicsymbols._t-1/2*a*_me.dynamicsymbols._t**2
expr1 = 2*a*0.5-1.25+0.25
expr2 = -1*x**2+y**2+0.25*(x+y)**2
expr3 = 0.5*10**(-10)
dyadic = _me.outer(frame_a.x, frame_a.x)+_me.outer(frame_a.y, frame_a.y)+_me.outer(frame_a.z, frame_a.z)
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
a, b, r = _sm.symbols('a b r', real=True)
eqn = _sm.Matrix([[0]])
eqn[0] = a*x**3+b*y**2-r
eqn = eqn.row_insert(eqn.shape[0], _sm.Matrix([[0]]))
eqn[eqn.shape[0]-1] = a*_sm.sin(x)**2+b*_sm.cos(2*y)-r**2
matrix_list = []
for i in eqn:matrix_list.append(i.subs({a:2.0, b:3.0, r:1.0}))
print(_sm.nsolve(matrix_list,(x,y),(_np.deg2rad(30),3.14)))
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
e = _sm.cos(x)+_sm.sin(x)+_sm.tan(x)+_sm.cosh(x)+_sm.sinh(x)+_sm.tanh(x)+_sm.acos(x)+_sm.asin(x)+_sm.atan(x)+_sm.log(x)+_sm.exp(x)+_sm.sqrt(x)+_sm.factorial(x)+_sm.ceiling(x)+_sm.floor(x)+_sm.sign(x)
e = (x)**2+_sm.log(x, 10)
a = _sm.Abs(-1*1)+int(1.5)+round(1.9)
e1 = 2*x+3*y
e2 = x+y
am = _sm.Matrix([e1.expand().coeff(x), e1.expand().coeff(y), e2.expand().coeff(x), e2.expand().coeff(y)]).reshape(2, 2)
b = (e1).expand().coeff(x)
c = (e2).expand().coeff(y)
d1 = (e1).collect(x).coeff(x,0)
d2 = (e1).collect(x).coeff(x,1)
fm = _sm.Matrix([i.collect(x)for i in _sm.Matrix([e1,e2]).reshape(1, 2)]).reshape((_sm.Matrix([e1,e2]).reshape(1, 2)).shape[0], (_sm.Matrix([e1,e2]).reshape(1, 2)).shape[1])
f = (e1).collect(y)
g = (e1).subs({x:2*x})
gm = _sm.Matrix([i.subs({x:3}) for i in _sm.Matrix([e1,e2]).reshape(2, 1)]).reshape((_sm.Matrix([e1,e2]).reshape(2, 1)).shape[0], (_sm.Matrix([e1,e2]).reshape(2, 1)).shape[1])
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
theta = _me.dynamicsymbols('theta')
frame_b.orient(frame_a, 'Axis', [theta, frame_a.z])
v1 = 2*frame_a.x-3*frame_a.y+frame_a.z
v2 = frame_b.x+frame_b.y+frame_b.z
a = _me.dot(v1, v2)
bm = _sm.Matrix([_me.dot(v1, v2),_me.dot(v1, 2*v2)]).reshape(2, 1)
c = _me.cross(v1, v2)
d = 2*v1.magnitude()+3*v1.magnitude()
dyadic = _me.outer(3*frame_a.x, frame_a.x)+_me.outer(frame_a.y, frame_a.y)+_me.outer(2*frame_a.z, frame_a.z)
am = (dyadic).to_matrix(frame_b)
m = _sm.Matrix([1,2,3]).reshape(3, 1)
v = m[0]*frame_a.x +m[1]*frame_a.y +m[2]*frame_a.z
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_n = _me.ReferenceFrame('n')
frame_a = _me.ReferenceFrame('a')
a = 0
d = _me.inertia(frame_a, 1, 1, 1)
point_po1 = _me.Point('po1')
point_po2 = _me.Point('po2')
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
c1, c2, c3 = _me.dynamicsymbols('c1 c2 c3')
c1_d, c2_d, c3_d = _me.dynamicsymbols('c1_ c2_ c3_', 1)
body_r_cm = _me.Point('r_cm')
body_r_cm.set_vel(frame_n, 0)
body_r_f = _me.ReferenceFrame('r_f')
body_r = _me.RigidBody('r', body_r_cm, body_r_f, _sm.symbols('m'), (_me.outer(body_r_f.x,body_r_f.x),body_r_cm))
point_po2.set_pos(particle_p1.point, c1*frame_a.x)
v = 2*point_po2.pos_from(particle_p1.point)+c2*frame_a.y
frame_a.set_ang_vel(frame_n, c3*frame_a.z)
v = 2*frame_a.ang_vel_in(frame_n)+c2*frame_a.y
body_r_f.set_ang_vel(frame_n, c3*frame_a.z)
v = 2*body_r_f.ang_vel_in(frame_n)+c2*frame_a.y
frame_a.set_ang_acc(frame_n, (frame_a.ang_vel_in(frame_n)).dt(frame_a))
v = 2*frame_a.ang_acc_in(frame_n)+c2*frame_a.y
particle_p1.point.set_vel(frame_a, c1*frame_a.x+c3*frame_a.y)
body_r_cm.set_acc(frame_n, c2*frame_a.y)
v_a = _me.cross(body_r_cm.acc(frame_n), particle_p1.point.vel(frame_a))
x_b_c = v_a
x_b_d = 2*x_b_c
a_b_c_d_e = x_b_d*2
a_b_c = 2*c1*c2*c3
a_b_c += 2*c1
a_b_c  =  3*c1
q1, q2, u1, u2 = _me.dynamicsymbols('q1 q2 u1 u2')
q1_d, q2_d, u1_d, u2_d = _me.dynamicsymbols('q1_ q2_ u1_ u2_', 1)
x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
x_dd, y_dd = _me.dynamicsymbols('x_ y_', 2)
yy = _me.dynamicsymbols('yy')
yy = x*x_d**2+1
m = _sm.Matrix([[0]])
m[0] = 2*x
m = m.row_insert(m.shape[0], _sm.Matrix([[0]]))
m[m.shape[0]-1] = 2*y
a = 2*m[0]
m = _sm.Matrix([1,2,3,4,5,6,7,8,9]).reshape(3, 3)
m[0,1] = 5
a = m[0, 1]*2
force_ro = q1*frame_n.x
torque_a = q2*frame_n.z
force_ro = q1*frame_n.x + q2*frame_n.y
f = force_ro*2
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2 = _me.dynamicsymbols('q1 q2')
x, y, z = _me.dynamicsymbols('x y z')
e = q1+q2
a = (e).subs({q1:x**2+y**2, q2:x-y})
e2 = _sm.cos(x)
e3 = _sm.cos(x*y)
a = (e2).series(x, 0, 2).removeO()
b = (e3).series(x, 0, 2).removeO().series(y, 0, 2).removeO()
e = ((x+y)**2).expand()
a = (e).subs({q1:x**2+y**2,q2:x-y}).subs({x:1,y:z})
bm = _sm.Matrix([i.subs({x:1,y:z}) for i in _sm.Matrix([e,2*e]).reshape(2, 1)]).reshape((_sm.Matrix([e,2*e]).reshape(2, 1)).shape[0], (_sm.Matrix([e,2*e]).reshape(2, 1)).shape[1])
e = q1+q2
a = (e).subs({q1:x**2+y**2,q2:x-y}).subs({x:2,y:z**2})
j, k, l = _sm.symbols('j k l', real=True)
p1 = _sm.Poly(_sm.Matrix([j,k,l]).reshape(1, 3), x)
p2 = _sm.Poly(j*x+k, x)
root1 = [i.evalf() for i in _sm.solve(p1, x)]
root2 = [i.evalf() for i in _sm.solve(_sm.Poly(_sm.Matrix([1,2,3]).reshape(3, 1), x),x)]
m = _sm.Matrix([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(4, 4)
am = (m).T+m
bm = _sm.Matrix([i.evalf() for i in (m).eigenvals().keys()])
c1 = _sm.diag(1,1,1,1)
c2 = _sm.Matrix([2 if i==j else 0 for i in range(3) for j in range(4)]).reshape(3, 4)
dm = (m+c1)**(-1)
e = (m+c1).det()+(_sm.Matrix([1,0,0,1]).reshape(2, 2)).trace()
f = (m)[1,2]
a = (m).cols
bm = (m).col(0)
cm = _sm.Matrix([(m).T.row(0),(m).T.row(1),(m).T.row(2),(m).T.row(3),(m).T.row(2)])
dm = (m).row(0)
em = _sm.Matrix([(m).row(0),(m).row(1),(m).row(2),(m).row(3),(m).row(2)])
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

frame_a = _me.ReferenceFrame('a')
c1, c2, c3 = _sm.symbols('c1 c2 c3', real=True)
a = _me.inertia(frame_a, 1, 1, 1)
particle_p1 = _me.Particle('p1', _me.Point('p1_pt'), _sm.Symbol('m'))
particle_p2 = _me.Particle('p2', _me.Point('p2_pt'), _sm.Symbol('m'))
body_r_cm = _me.Point('r_cm')
body_r_f = _me.ReferenceFrame('r_f')
body_r = _me.RigidBody('r', body_r_cm, body_r_f, _sm.symbols('m'), (_me.outer(body_r_f.x,body_r_f.x),body_r_cm))
frame_a.orient(body_r_f, 'DCM', _sm.Matrix([1,1,1,1,1,0,0,0,1]).reshape(3, 3))
point_o = _me.Point('o')
m1 = _sm.symbols('m1')
particle_p1.mass = m1
m2 = _sm.symbols('m2')
particle_p2.mass = m2
mr = _sm.symbols('mr')
body_r.mass = mr
i1 = _sm.symbols('i1')
i2 = _sm.symbols('i2')
i3 = _sm.symbols('i3')
body_r.inertia = (_me.inertia(body_r_f, i1, i2, i3, 0, 0, 0), body_r_cm)
point_o.set_pos(particle_p1.point, c1*frame_a.x)
point_o.set_pos(particle_p2.point, c2*frame_a.y)
point_o.set_pos(body_r_cm, c3*frame_a.z)
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a)
a = _me.inertia_of_point_mass(particle_p2.mass, particle_p2.point.pos_from(point_o), frame_a)
a = body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a) + _me.inertia_of_point_mass(particle_p2.mass, particle_p2.point.pos_from(point_o), frame_a) + body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)
a = _me.inertia_of_point_mass(particle_p1.mass, particle_p1.point.pos_from(point_o), frame_a) + body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)
a = body_r.inertia[0] + _me.inertia_of_point_mass(body_r.mass, body_r.masscenter.pos_from(point_o), frame_a)
a = body_r.inertia[0]
particle_p2.point.set_pos(particle_p1.point, c1*frame_a.x+c2*frame_a.y)
body_r_cm.set_pos(particle_p1.point, c3*frame_a.x)
body_r_cm.set_pos(particle_p2.point, c3*frame_a.y)
b = _me.functions.center_of_mass(point_o,particle_p1, particle_p2, body_r)
b = _me.functions.center_of_mass(point_o,particle_p1, body_r)
b = _me.functions.center_of_mass(particle_p1.point,particle_p1, particle_p2, body_r)
u1, u2, u3 = _me.dynamicsymbols('u1 u2 u3')
v = u1*frame_a.x+u2*frame_a.y+u3*frame_a.z
u = (v+c1*frame_a.x).normalize()
particle_p1.point.set_vel(frame_a, u1*frame_a.x)
a = particle_p1.point.partial_velocity(frame_a, u1)
m = particle_p1.mass+body_r.mass
m = particle_p2.mass
m = particle_p1.mass+particle_p2.mass+body_r.mass
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

x1, x2 = _me.dynamicsymbols('x1 x2')
f1 = x1*x2+3*x1**2
f2 = x1*_me.dynamicsymbols._t+x2*_me.dynamicsymbols._t**2
x, y = _me.dynamicsymbols('x y')
x_d, y_d = _me.dynamicsymbols('x_ y_', 1)
y_dd = _me.dynamicsymbols('y_', 2)
q1, q2, q3, u1, u2 = _me.dynamicsymbols('q1 q2 q3 u1 u2')
p1, p2 = _me.dynamicsymbols('p1 p2')
p1_d, p2_d = _me.dynamicsymbols('p1_ p2_', 1)
w1, w2, w3, r1, r2 = _me.dynamicsymbols('w1 w2 w3 r1 r2')
w1_d, w2_d, w3_d, r1_d, r2_d = _me.dynamicsymbols('w1_ w2_ w3_ r1_ r2_', 1)
r1_dd, r2_dd = _me.dynamicsymbols('r1_ r2_', 2)
c11, c12, c21, c22 = _me.dynamicsymbols('c11 c12 c21 c22')
d11, d12, d13 = _me.dynamicsymbols('d11 d12 d13')
j1, j2 = _me.dynamicsymbols('j1 j2')
n = _sm.symbols('n')
n = _sm.I
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2 = _me.dynamicsymbols('q1 q2')
q1_d, q2_d = _me.dynamicsymbols('q1_ q2_', 1)
q1_dd, q2_dd = _me.dynamicsymbols('q1_ q2_', 2)
l, m, g = _sm.symbols('l m g', real=True)
frame_n = _me.ReferenceFrame('n')
point_pn = _me.Point('pn')
point_pn.set_vel(frame_n, 0)
theta1 = _sm.atan(q2/q1)
frame_a = _me.ReferenceFrame('a')
frame_a.orient(frame_n, 'Axis', [theta1, frame_n.z])
particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
particle_p.point.set_pos(point_pn, q1*frame_n.x+q2*frame_n.y)
particle_p.mass = m
particle_p.point.set_vel(frame_n, (point_pn.pos_from(particle_p.point)).dt(frame_n))
f_v = _me.dot((particle_p.point.vel(frame_n)).express(frame_a), frame_a.x)
force_p = particle_p.mass*(g*frame_n.x)
dependent = _sm.Matrix([[0]])
dependent[0] = f_v
velocity_constraints = [i for i in dependent]
u_q1_d = _me.dynamicsymbols('u_q1_d')
u_q2_d = _me.dynamicsymbols('u_q2_d')
kd_eqs = [q1_d-u_q1_d, q2_d-u_q2_d]
forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x))]
kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u_q2_d], u_dependent=[u_q1_d], kd_eqs = kd_eqs, velocity_constraints = velocity_constraints)
fr, frstar = kane.kanes_equations([particle_p], forceList)
zero = fr+frstar
f_c = point_pn.pos_from(particle_p.point).magnitude()-l
config = _sm.Matrix([[0]])
config[0] = f_c
zero = zero.row_insert(zero.shape[0], _sm.Matrix([[0]]))
zero[zero.shape[0]-1] = config[0]
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

g, lb, w, h = _sm.symbols('g lb w h', real=True)
theta, phi, omega, alpha = _me.dynamicsymbols('theta phi omega alpha')
theta_d, phi_d, omega_d, alpha_d = _me.dynamicsymbols('theta_ phi_ omega_ alpha_', 1)
theta_dd, phi_dd = _me.dynamicsymbols('theta_ phi_', 2)
frame_n = _me.ReferenceFrame('n')
body_a_cm = _me.Point('a_cm')
body_a_cm.set_vel(frame_n, 0)
body_a_f = _me.ReferenceFrame('a_f')
body_a = _me.RigidBody('a', body_a_cm, body_a_f, _sm.symbols('m'), (_me.outer(body_a_f.x,body_a_f.x),body_a_cm))
body_b_cm = _me.Point('b_cm')
body_b_cm.set_vel(frame_n, 0)
body_b_f = _me.ReferenceFrame('b_f')
body_b = _me.RigidBody('b', body_b_cm, body_b_f, _sm.symbols('m'), (_me.outer(body_b_f.x,body_b_f.x),body_b_cm))
body_a_f.orient(frame_n, 'Axis', [theta, frame_n.y])
body_b_f.orient(body_a_f, 'Axis', [phi, body_a_f.z])
point_o = _me.Point('o')
la = (lb-h/2)/2
body_a_cm.set_pos(point_o, la*body_a_f.z)
body_b_cm.set_pos(point_o, lb*body_a_f.z)
body_a_f.set_ang_vel(frame_n, omega*frame_n.y)
body_b_f.set_ang_vel(body_a_f, alpha*body_a_f.z)
point_o.set_vel(frame_n, 0)
body_a_cm.v2pt_theory(point_o,frame_n,body_a_f)
body_b_cm.v2pt_theory(point_o,frame_n,body_a_f)
ma = _sm.symbols('ma')
body_a.mass = ma
mb = _sm.symbols('mb')
body_b.mass = mb
iaxx = 1/12*ma*(2*la)**2
iayy = iaxx
iazz = 0
ibxx = 1/12*mb*h**2
ibyy = 1/12*mb*(w**2+h**2)
ibzz = 1/12*mb*w**2
body_a.inertia = (_me.inertia(body_a_f, iaxx, iayy, iazz, 0, 0, 0), body_a_cm)
body_b.inertia = (_me.inertia(body_b_f, ibxx, ibyy, ibzz, 0, 0, 0), body_b_cm)
force_a = body_a.mass*(g*frame_n.z)
force_b = body_b.mass*(g*frame_n.z)
kd_eqs = [theta_d - omega, phi_d - alpha]
forceList = [(body_a.masscenter,body_a.mass*(g*frame_n.z)), (body_b.masscenter,body_b.mass*(g*frame_n.z))]
kane = _me.KanesMethod(frame_n, q_ind=[theta,phi], u_ind=[omega, alpha], kd_eqs = kd_eqs)
fr, frstar = kane.kanes_equations([body_a, body_b], forceList)
zero = fr+frstar
from pydy.system import System
sys = System(kane, constants = {g:9.81, lb:0.2, w:0.2, h:0.1, ma:0.01, mb:0.1},
specifieds={},
initial_conditions={theta:_np.deg2rad(90), phi:_np.deg2rad(0.5), omega:0, alpha:0},
times = _np.linspace(0.0, 10, 10/0.02))

y=sys.integrate()
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

q1, q2, u1, u2 = _me.dynamicsymbols('q1 q2 u1 u2')
q1_d, q2_d, u1_d, u2_d = _me.dynamicsymbols('q1_ q2_ u1_ u2_', 1)
l, m, g = _sm.symbols('l m g', real=True)
frame_n = _me.ReferenceFrame('n')
frame_a = _me.ReferenceFrame('a')
frame_b = _me.ReferenceFrame('b')
frame_a.orient(frame_n, 'Axis', [q1, frame_n.z])
frame_b.orient(frame_n, 'Axis', [q2, frame_n.z])
frame_a.set_ang_vel(frame_n, u1*frame_n.z)
frame_b.set_ang_vel(frame_n, u2*frame_n.z)
point_o = _me.Point('o')
particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
particle_r = _me.Particle('r', _me.Point('r_pt'), _sm.Symbol('m'))
particle_p.point.set_pos(point_o, l*frame_a.x)
particle_r.point.set_pos(particle_p.point, l*frame_b.x)
point_o.set_vel(frame_n, 0)
particle_p.point.v2pt_theory(point_o,frame_n,frame_a)
particle_r.point.v2pt_theory(particle_p.point,frame_n,frame_b)
particle_p.mass = m
particle_r.mass = m
force_p = particle_p.mass*(g*frame_n.x)
force_r = particle_r.mass*(g*frame_n.x)
kd_eqs = [q1_d - u1, q2_d - u2]
forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x)), (particle_r.point,particle_r.mass*(g*frame_n.x))]
kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u1, u2], kd_eqs = kd_eqs)
fr, frstar = kane.kanes_equations([particle_p, particle_r], forceList)
zero = fr+frstar
from pydy.system import System
sys = System(kane, constants = {l:1, m:1, g:9.81},
specifieds={},
initial_conditions={q1:.1, q2:.2, u1:0, u2:0},
times = _np.linspace(0.0, 10, 10/.01))

y=sys.integrate()
<|endoftext|>import sympy.physics.mechanics as _me
import sympy as _sm
import math as m
import numpy as _np

m, k, b, g = _sm.symbols('m k b g', real=True)
position, speed = _me.dynamicsymbols('position speed')
position_d, speed_d = _me.dynamicsymbols('position_ speed_', 1)
o = _me.dynamicsymbols('o')
force = o*_sm.sin(_me.dynamicsymbols._t)
frame_ceiling = _me.ReferenceFrame('ceiling')
point_origin = _me.Point('origin')
point_origin.set_vel(frame_ceiling, 0)
particle_block = _me.Particle('block', _me.Point('block_pt'), _sm.Symbol('m'))
particle_block.point.set_pos(point_origin, position*frame_ceiling.x)
particle_block.mass = m
particle_block.point.set_vel(frame_ceiling, speed*frame_ceiling.x)
force_magnitude = m*g-k*position-b*speed+force
force_block = (force_magnitude*frame_ceiling.x).subs({position_d:speed})
kd_eqs = [position_d - speed]
forceList = [(particle_block.point,(force_magnitude*frame_ceiling.x).subs({position_d:speed}))]
kane = _me.KanesMethod(frame_ceiling, q_ind=[position], u_ind=[speed], kd_eqs = kd_eqs)
fr, frstar = kane.kanes_equations([particle_block], forceList)
zero = fr+frstar
from pydy.system import System
sys = System(kane, constants = {m:1.0, k:1.0, b:0.2, g:9.8},
specifieds={_me.dynamicsymbols('t'):lambda x, t: t, o:2},
initial_conditions={position:0.1, speed:-1*1.0},
times = _np.linspace(0.0, 10.0, 10.0/0.01))

y=sys.integrate()
<|endoftext|>from sympy.external import import_module

lfortran = import_module('lfortran')

if lfortran:
    from sympy.codegen.ast import (Variable, IntBaseType, FloatBaseType, String,
                                   Return, FunctionDefinition, Assignment)
    from sympy.core import Add, Mul, Integer, Float
    from sympy.core.symbol import Symbol

    asr_mod = lfortran.asr
    asr = lfortran.asr.asr
    src_to_ast = lfortran.ast.src_to_ast
    ast_to_asr = lfortran.semantic.ast_to_asr.ast_to_asr

    """
    This module contains all the necessary Classes and Function used to Parse
    Fortran code into SymPy expression

    The module and its API are currently under development and experimental.
    It is also dependent on LFortran for the ASR that is converted to SymPy syntax
    which is also under development.
    The module only supports the features currently supported by the LFortran ASR
    which will be updated as the development of LFortran and this module progresses

    You might find unexpected bugs and exceptions while using the module, feel free
    to report them to the SymPy Issue Tracker

    The API for the module might also change while in development if better and
    more effective ways are discovered for the process

    Features Supported
    ==================

    - Variable Declarations (integers and reals)
    - Function Definitions
    - Assignments and Basic Binary Operations


    Notes
    =====

    The module depends on an external dependency

    LFortran : Required to parse Fortran source code into ASR


    References
    ==========

    .. [1] https://github.com/sympy/sympy/issues
    .. [2] https://gitlab.com/lfortran/lfortran
    .. [3] https://docs.lfortran.org/

    """


    class ASR2PyVisitor(asr.ASTVisitor):  # type: ignore
        """
        Visitor Class for LFortran ASR

        It is a Visitor class derived from asr.ASRVisitor which visits all the
        nodes of the LFortran ASR and creates corresponding AST node for each
        ASR node

        """

        def __init__(self):
            """Initialize the Parser"""
            self._py_ast = []

        def visit_TranslationUnit(self, node):
            """
            Function to visit all the elements of the Translation Unit
            created by LFortran ASR
            """
            for s in node.global_scope.symbols:
                sym = node.global_scope.symbols[s]
                self.visit(sym)
            for item in node.items:
                self.visit(item)

        def visit_Assignment(self, node):
            """Visitor Function for Assignment

            Visits each Assignment is the LFortran ASR and creates corresponding
            assignment for SymPy.

            Notes
            =====

            The function currently only supports variable assignment and binary
            operation assignments of varying multitudes. Any type of numberS or
            array is not supported.

            Raises
            ======

            NotImplementedError() when called for Numeric assignments or Arrays

            """
            # TODO: Arithmetic Assignment
            if isinstance(node.target, asr.Variable):
                target = node.target
                value = node.value
                if isinstance(value, asr.Variable):
                    new_node = Assignment(
                            Variable(
                                    target.name
                                ),
                            Variable(
                                    value.name
                                )
                        )
                elif (type(value) == asr.BinOp):
                    exp_ast = call_visitor(value)
                    for expr in exp_ast:
                        new_node = Assignment(
                                Variable(target.name),
                                expr
                            )
                else:
                    raise NotImplementedError("Numeric assignments not supported")
            else:
                raise NotImplementedError("Arrays not supported")
            self._py_ast.append(new_node)

        def visit_BinOp(self, node):
            """Visitor Function for Binary Operations

            Visits each binary operation present in the LFortran ASR like addition,
            subtraction, multiplication, division and creates the corresponding
            operation node in SymPy's AST

            In case of more than one binary operations, the function calls the
            call_visitor() function on the child nodes of the binary operations
            recursively until all the operations have been processed.

            Notes
            =====

            The function currently only supports binary operations with Variables
            or other binary operations. Numerics are not supported as of yet.

            Raises
            ======

            NotImplementedError() when called for Numeric assignments

            """
            # TODO: Integer Binary Operations
            op = node.op
            lhs = node.left
            rhs = node.right

            if (type(lhs) == asr.Variable):
                left_value = Symbol(lhs.name)
            elif(type(lhs) == asr.BinOp):
                l_exp_ast = call_visitor(lhs)
                for exp in l_exp_ast:
                    left_value = exp
            else:
                raise NotImplementedError("Numbers Currently not supported")

            if (type(rhs) == asr.Variable):
                right_value = Symbol(rhs.name)
            elif(type(rhs) == asr.BinOp):
                r_exp_ast = call_visitor(rhs)
                for exp in r_exp_ast:
                    right_value = exp
            else:
                raise NotImplementedError("Numbers Currently not supported")

            if isinstance(op, asr.Add):
                new_node = Add(left_value, right_value)
            elif isinstance(op, asr.Sub):
                new_node = Add(left_value, -right_value)
            elif isinstance(op, asr.Div):
                new_node = Mul(left_value, 1/right_value)
            elif isinstance(op, asr.Mul):
                new_node = Mul(left_value, right_value)

            self._py_ast.append(new_node)

        def visit_Variable(self, node):
            """Visitor Function for Variable Declaration

            Visits each variable declaration present in the ASR and creates a
            Symbol declaration for each variable

            Notes
            =====

            The functions currently only support declaration of integer and
            real variables. Other data types are still under development.

            Raises
            ======

            NotImplementedError() when called for unsupported data types

            """
            if isinstance(node.type, asr.Integer):
                var_type = IntBaseType(String('integer'))
                value = Integer(0)
            elif isinstance(node.type, asr.Real):
                var_type = FloatBaseType(String('real'))
                value = Float(0.0)
            else:
                raise NotImplementedError("Data type not supported")

            if not (node.intent == 'in'):
                new_node = Variable(
                    node.name
                ).as_Declaration(
                    type = var_type,
                    value = value
                )
                self._py_ast.append(new_node)

        def visit_Sequence(self, seq):
            """Visitor Function for code sequence

            Visits a code sequence/ block and calls the visitor function on all the
            children of the code block to create corresponding code in python

            """
            if seq is not None:
                for node in seq:
                    self._py_ast.append(call_visitor(node))

        def visit_Num(self, node):
            """Visitor Function for Numbers in ASR

            This function is currently under development and will be updated
            with improvements in the LFortran ASR

            """
            # TODO:Numbers when the LFortran ASR is updated
            # self._py_ast.append(Integer(node.n))
            pass

        def visit_Function(self, node):
            """Visitor Function for function Definitions

            Visits each function definition present in the ASR and creates a
            function definition node in the Python AST with all the elements of the
            given function

            The functions declare all the variables required as SymPy symbols in
            the function before the function definition

            This function also the call_visior_function to parse the contents of
            the function body

            """
            # TODO: Return statement, variable declaration
            fn_args = [Variable(arg_iter.name) for arg_iter in node.args]
            fn_body = []
            fn_name = node.name
            for i in node.body:
                fn_ast = call_visitor(i)
            try:
                fn_body_expr = fn_ast
            except UnboundLocalError:
                fn_body_expr = []
            for sym in node.symtab.symbols:
                decl = call_visitor(node.symtab.symbols[sym])
                for symbols in decl:
                    fn_body.append(symbols)
            for elem in fn_body_expr:
                fn_body.append(elem)
            fn_body.append(
                Return(
                    Variable(
                        node.return_var.name
                    )
                )
            )
            if isinstance(node.return_var.type, asr.Integer):
                ret_type = IntBaseType(String('integer'))
            elif isinstance(node.return_var.type, asr.Real):
                ret_type = FloatBaseType(String('real'))
            else:
                raise NotImplementedError("Data type not supported")
            new_node = FunctionDefinition(
                        return_type = ret_type,
                        name = fn_name,
                        parameters = fn_args,
                        body = fn_body
                    )
            self._py_ast.append(new_node)

        def ret_ast(self):
            """Returns the AST nodes"""
            return self._py_ast
else:
    class ASR2PyVisitor():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError('lfortran not available')

def call_visitor(fort_node):
    """Calls the AST Visitor on the Module

    This function is used to call the AST visitor for a program or module
    It imports all the required modules and calls the visit() function
    on the given node

    Parameters
    ==========

    fort_node : LFortran ASR object
        Node for the operation for which the NodeVisitor is called

    Returns
    =======

    res_ast : list
        list of SymPy AST Nodes

    """
    v = ASR2PyVisitor()
    v.visit(fort_node)
    res_ast = v.ret_ast()
    return res_ast


def src_to_sympy(src):
    """Wrapper function to convert the given Fortran source code to SymPy Expressions

    Parameters
    ==========

    src : string
        A string with the Fortran source code

    Returns
    =======

    py_src : string
        A string with the Python source code compatible with SymPy

    """
    a_ast = src_to_ast(src, translation_unit=False)
    a = ast_to_asr(a_ast)
    py_src = call_visitor(a)
    return py_src
<|endoftext|>from __future__ import annotations
import random
import itertools
from typing import Sequence as tSequence
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.intfunc import igcd
from sympy.core.numbers import (Integer, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.logic.boolalg import (And, Not, Or)
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.solvers.solveset import linsolve
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import strongly_connected_components
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import (RandomIndexedSymbol, random_symbols, RandomSymbol,
                            _symbol_converter, _value_check, pspace, given,
                           dependent, is_random, sample_iter, Distribution,
                           Density)
from sympy.stats.stochastic_process import StochasticPSpace
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.stats.frv_types import Bernoulli, BernoulliDistribution, FiniteRV
from sympy.stats.drv_types import Poisson, PoissonDistribution
from sympy.stats.crv_types import Normal, NormalDistribution, Gamma, GammaDistribution
from sympy.core.sympify import _sympify, sympify

EmptySet = S.EmptySet

__all__ = [
    'StochasticProcess',
    'DiscreteTimeStochasticProcess',
    'DiscreteMarkovChain',
    'TransitionMatrixOf',
    'StochasticStateSpaceOf',
    'GeneratorMatrixOf',
    'ContinuousMarkovChain',
    'BernoulliProcess',
    'PoissonProcess',
    'WienerProcess',
    'GammaProcess'
]


@is_random.register(Indexed)
def _(x):
    return is_random(x.base)

@is_random.register(RandomIndexedSymbol)  # type: ignore
def _(x):
    return True

def _set_converter(itr):
    """
    Helper function for converting list/tuple/set to Set.
    If parameter is not an instance of list/tuple/set then
    no operation is performed.

    Returns
    =======

    Set
        The argument converted to Set.


    Raises
    ======

    TypeError
        If the argument is not an instance of list/tuple/set.
    """
    if isinstance(itr, (list, tuple, set)):
        itr = FiniteSet(*itr)
    if not isinstance(itr, Set):
        raise TypeError("%s is not an instance of list/tuple/set."%(itr))
    return itr

def _state_converter(itr: tSequence) -> Tuple | Range:
    """
    Helper function for converting list/tuple/set/Range/Tuple/FiniteSet
    to tuple/Range.
    """
    itr_ret: Tuple | Range

    if isinstance(itr, (Tuple, set, FiniteSet)):
        itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))

    elif isinstance(itr, (list, tuple)):
        # check if states are unique
        if len(set(itr)) != len(itr):
            raise ValueError('The state space must have unique elements.')
        itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))

    elif isinstance(itr, Range):
        # the only ordered set in SymPy I know of
        # try to convert to tuple
        try:
            itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))
        except (TypeError, ValueError):
            itr_ret = itr

    else:
        raise TypeError("%s is not an instance of list/tuple/set/Range/Tuple/FiniteSet." % (itr))
    return itr_ret

def _sym_sympify(arg):
    """
    Converts an arbitrary expression to a type that can be used inside SymPy.
    As generally strings are unwise to use in the expressions,
    it returns the Symbol of argument if the string type argument is passed.

    Parameters
    =========

    arg: The parameter to be converted to be used in SymPy.

    Returns
    =======

    The converted parameter.

    """
    if isinstance(arg, str):
        return Symbol(arg)
    else:
        return _sympify(arg)

def _matrix_checks(matrix):
    if not isinstance(matrix, (Matrix, MatrixSymbol, ImmutableMatrix)):
        raise TypeError("Transition probabilities either should "
                            "be a Matrix or a MatrixSymbol.")
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError("%s is not a square matrix"%(matrix))
    if isinstance(matrix, Matrix):
        matrix = ImmutableMatrix(matrix.tolist())
    return matrix

class StochasticProcess(Basic):
    """
    Base class for all the stochastic processes whether
    discrete or continuous.

    Parameters
    ==========

    sym: Symbol or str
    state_space: Set
        The state space of the stochastic process, by default S.Reals.
        For discrete sets it is zero indexed.

    See Also
    ========

    DiscreteTimeStochasticProcess
    """

    index_set = S.Reals

    def __new__(cls, sym, state_space=S.Reals, **kwargs):
        sym = _symbol_converter(sym)
        state_space = _set_converter(state_space)
        return Basic.__new__(cls, sym, state_space)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def state_space(self) -> FiniteSet | Range:
        if not isinstance(self.args[1], (FiniteSet, Range)):
            assert isinstance(self.args[1], Tuple)
            return FiniteSet(*self.args[1])
        return self.args[1]

    def _deprecation_warn_distribution(self):
        sympy_deprecation_warning(
            """
            Calling the distribution method with a RandomIndexedSymbol
            argument, like X.distribution(X(t)) is deprecated. Instead, call
            distribution() with the given timestamp, like

            X.distribution(t)
            """,
            deprecated_since_version="1.7.1",
            active_deprecations_target="deprecated-distribution-randomindexedsymbol",
            stacklevel=4,
        )

    def distribution(self, key=None):
        if key is None:
            self._deprecation_warn_distribution()
        return Distribution()

    def density(self, x):
        return Density()

    def __call__(self, time):
        """
        Overridden in ContinuousTimeStochasticProcess.
        """
        raise NotImplementedError("Use [] for indexing discrete time stochastic process.")

    def __getitem__(self, time):
        """
        Overridden in DiscreteTimeStochasticProcess.
        """
        raise NotImplementedError("Use () for indexing continuous time stochastic process.")

    def probability(self, condition):
        raise NotImplementedError()

    def joint_distribution(self, *args):
        """
        Computes the joint distribution of the random indexed variables.

        Parameters
        ==========

        args: iterable
            The finite list of random indexed variables/the key of a stochastic
            process whose joint distribution has to be computed.

        Returns
        =======

        JointDistribution
            The joint distribution of the list of random indexed variables.
            An unevaluated object is returned if it is not possible to
            compute the joint distribution.

        Raises
        ======

        ValueError: When the arguments passed are not of type RandomIndexSymbol
        or Number.
        """
        args = list(args)
        for i, arg in enumerate(args):
            if S(arg).is_Number:
                if self.index_set.is_subset(S.Integers):
                    args[i] = self.__getitem__(arg)
                else:
                    args[i] = self.__call__(arg)
            elif not isinstance(arg, RandomIndexedSymbol):
                raise ValueError("Expected a RandomIndexedSymbol or "
                                "key not  %s"%(type(arg)))

        if args[0].pspace.distribution == Distribution():
            return JointDistribution(*args)
        density = Lambda(tuple(args),
                         expr=Mul.fromiter(arg.pspace.process.density(arg) for arg in args))
        return JointDistributionHandmade(density)

    def expectation(self, condition, given_condition):
        raise NotImplementedError("Abstract method for expectation queries.")

    def sample(self):
        raise NotImplementedError("Abstract method for sampling queries.")

class DiscreteTimeStochasticProcess(StochasticProcess):
    """
    Base class for all discrete stochastic processes.
    """
    def __getitem__(self, time):
        """
        For indexing discrete time stochastic processes.

        Returns
        =======

        RandomIndexedSymbol
        """
        time = sympify(time)
        if not time.is_symbol and time not in self.index_set:
            raise IndexError("%s is not in the index set of %s"%(time, self.symbol))
        idx_obj = Indexed(self.symbol, time)
        pspace_obj = StochasticPSpace(self.symbol, self, self.distribution(time))
        return RandomIndexedSymbol(idx_obj, pspace_obj)

class ContinuousTimeStochasticProcess(StochasticProcess):
    """
    Base class for all continuous time stochastic process.
    """
    def __call__(self, time):
        """
        For indexing continuous time stochastic processes.

        Returns
        =======

        RandomIndexedSymbol
        """
        time = sympify(time)
        if not time.is_symbol and time not in self.index_set:
            raise IndexError("%s is not in the index set of %s"%(time, self.symbol))
        func_obj = Function(self.symbol)(time)
        pspace_obj = StochasticPSpace(self.symbol, self, self.distribution(time))
        return RandomIndexedSymbol(func_obj, pspace_obj)

class TransitionMatrixOf(Boolean):
    """
    Assumes that the matrix is the transition matrix
    of the process.
    """

    def __new__(cls, process, matrix):
        if not isinstance(process, DiscreteMarkovChain):
            raise ValueError("Currently only DiscreteMarkovChain "
                                "support TransitionMatrixOf.")
        matrix = _matrix_checks(matrix)
        return Basic.__new__(cls, process, matrix)

    process = property(lambda self: self.args[0])
    matrix = property(lambda self: self.args[1])

class GeneratorMatrixOf(TransitionMatrixOf):
    """
    Assumes that the matrix is the generator matrix
    of the process.
    """

    def __new__(cls, process, matrix):
        if not isinstance(process, ContinuousMarkovChain):
            raise ValueError("Currently only ContinuousMarkovChain "
                                "support GeneratorMatrixOf.")
        matrix = _matrix_checks(matrix)
        return Basic.__new__(cls, process, matrix)

class StochasticStateSpaceOf(Boolean):

    def __new__(cls, process, state_space):
        if not isinstance(process, (DiscreteMarkovChain, ContinuousMarkovChain)):
            raise ValueError("Currently only DiscreteMarkovChain and ContinuousMarkovChain "
                                "support StochasticStateSpaceOf.")
        state_space = _state_converter(state_space)
        if isinstance(state_space, Range):
            ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
        else:
            ss_size = len(state_space)
        state_index = Range(ss_size)
        return Basic.__new__(cls, process, state_index)

    process = property(lambda self: self.args[0])
    state_index = property(lambda self: self.args[1])

class MarkovProcess(StochasticProcess):
    """
    Contains methods that handle queries
    common to Markov processes.
    """

    @property
    def number_of_states(self) -> Integer | Symbol:
        """
        The number of states in the Markov Chain.
        """
        return _sympify(self.args[2].shape[0]) # type: ignore

    @property
    def _state_index(self):
        """
        Returns state index as Range.
        """
        return self.args[1]

    @classmethod
    def _sanity_checks(cls, state_space, trans_probs):
        # Try to never have None as state_space or trans_probs.
        # This helps a lot if we get it done at the start.
        if (state_space is None) and (trans_probs is None):
            _n = Dummy('n', integer=True, nonnegative=True)
            state_space = _state_converter(Range(_n))
            trans_probs = _matrix_checks(MatrixSymbol('_T', _n, _n))

        elif state_space is None:
            trans_probs = _matrix_checks(trans_probs)
            state_space = _state_converter(Range(trans_probs.shape[0]))

        elif trans_probs is None:
            state_space = _state_converter(state_space)
            if isinstance(state_space, Range):
                _n = ceiling((state_space.stop - state_space.start) / state_space.step)
            else:
                _n = len(state_space)
            trans_probs = MatrixSymbol('_T', _n, _n)

        else:
            state_space = _state_converter(state_space)
            trans_probs = _matrix_checks(trans_probs)
            # Range object doesn't want to give a symbolic size
            # so we do it ourselves.
            if isinstance(state_space, Range):
                ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
            else:
                ss_size = len(state_space)
            if ss_size != trans_probs.shape[0]:
                raise ValueError('The size of the state space and the number of '
                                 'rows of the transition matrix must be the same.')

        return state_space, trans_probs

    def _extract_information(self, given_condition):
        """
        Helper function to extract information, like,
        transition matrix/generator matrix, state space, etc.
        """
        if isinstance(self, DiscreteMarkovChain):
            trans_probs = self.transition_probabilities
            state_index = self._state_index
        elif isinstance(self, ContinuousMarkovChain):
            trans_probs = self.generator_matrix
            state_index = self._state_index
        if isinstance(given_condition, And):
            gcs = given_condition.args
            given_condition = S.true
            for gc in gcs:
                if isinstance(gc, TransitionMatrixOf):
                    trans_probs = gc.matrix
                if isinstance(gc, StochasticStateSpaceOf):
                    state_index = gc.state_index
                if isinstance(gc, Relational):
                    given_condition = given_condition & gc
        if isinstance(given_condition, TransitionMatrixOf):
            trans_probs = given_condition.matrix
            given_condition = S.true
        if isinstance(given_condition, StochasticStateSpaceOf):
            state_index = given_condition.state_index
            given_condition = S.true
        return trans_probs, state_index, given_condition

    def _check_trans_probs(self, trans_probs, row_sum=1):
        """
        Helper function for checking the validity of transition
        probabilities.
        """
        if not isinstance(trans_probs, MatrixSymbol):
            rows = trans_probs.tolist()
            for row in rows:
                if (sum(row) - row_sum) != 0:
                    raise ValueError("Values in a row must sum to %s. "
                    "If you are using Float or floats then please use Rational."%(row_sum))

    def _work_out_state_index(self, state_index, given_condition, trans_probs):
        """
        Helper function to extract state space if there
        is a random symbol in the given condition.
        """
        # if given condition is None, then there is no need to work out
        # state_space from random variables
        if given_condition != None:
            rand_var = list(given_condition.atoms(RandomSymbol) -
                        given_condition.atoms(RandomIndexedSymbol))
            if len(rand_var) == 1:
                state_index = rand_var[0].pspace.set

        # `not None` is `True`. So the old test fails for symbolic sizes.
        # Need to build the statement differently.
        sym_cond = not self.number_of_states.is_Integer
        cond1 = not sym_cond and len(state_index) != trans_probs.shape[0]
        if cond1:
            raise ValueError("state space is not compatible with the transition probabilities.")
        if not isinstance(trans_probs.shape[0], Symbol):
            state_index = FiniteSet(*range(trans_probs.shape[0]))
        return state_index

    @cacheit
    def _preprocess(self, given_condition, evaluate):
        """
