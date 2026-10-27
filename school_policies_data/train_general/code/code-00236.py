            raise AssertionError("Padding + d1 length must equal d2 length")

        # for every padding size, we also consider broadcasting
        broadcast_padding = [
            broadcast_dim(simulate_padding, d2, d11, d12, j, True)
            for j in range(len(d2) - i)
        ]

        # we consider the possibilities for broadcasting for every dimension. Since we already
        # padded d1, we do not consider it while broadcasting
        all_broadcasting_possibilities = (
            generate_all_broadcasting_possibilities_no_padding(
                d1, d2[(len(d2) - i) :], d11[(len(d2) - i) :], d12[(len(d2) - i) :]
            )
        )
        # combine all constraints into a conjunction
        c = Conj(
            [
                e1,
                e11,
                e2,
                e12,
                *broadcast_padding,
                all_broadcasting_possibilities,
                *nat_constraints,
            ]
        )
        res.append(c)

    return Disj(res), counter


def no_broadcast_dim_with_index(
    d1: list[DVar], d2: list[DVar], d3: list[DVar], d4: list[DVar], i: int
) -> Constraint:
    """
    Args:
        d1: input 1
        d2: input 2
        d3: simulated broadcasting for input 1
        d4: simulated broadcasting for input 2
        i: the rank of the resulting tensor addition

    Returns: Constraints for when no broadcasting occurs
    """
    return Conj(
        [
            Disj(
                [
                    Conj(
                        [
                            BinConstraintD(d1[i], 1, op_eq),
                            BinConstraintD(d2[i], 1, op_eq),
                        ]
                    ),
                    Conj(
                        [
                            BinConstraintD(d1[i], 1, op_neq),
                            BinConstraintD(d2[i], 1, op_neq),
                        ]
                    ),
                ]
            ),
            BinConstraintD(d1[i], d3[i], op_eq),
            BinConstraintD(d2[i], d4[i], op_eq),
        ]
    )


def gen_lists_of_dims(
    num_tensors: int, dim_size: int, counter: int
) -> tuple[list[list[DVar]], int]:
    """
    Generate lists of DVar to represent tensor dimensions
    Args:
        num_tensors: the required number of tensors
        dim_size: the number of dimensions for each tensor
        counter: variable tracking

    Returns: A list of a list of tensor dimensions

    """
    res = []

    for _ in range(num_tensors):
        dims, counter = gen_tensor_dims(dim_size, counter)
        res.append(dims)

    return res, counter


def create_equality_constraints_for_broadcasting(
    e1: TVar,
    e2: TVar,
    e11: TVar,
    e12: TVar,
    d1: list[DVar],
    d2: list[DVar],
    d11: list[DVar],
    d12: list[DVar],
) -> list[BinConstraintT]:
    """
    Create equality constraints for when no broadcasting occurs
    Args:
        e1: Input 1
        e2: Input 2
        e11: Broadcasted input 1
        e12: Broadcasted input 2
        d1: Variables that store dimensions for e1
        d2: Variables that store dimensions for e2
        d11: Variables that store dimensions for e11
        d12: Variables that store dimensions for e22

    Returns: Four equality constraints

    """

    e1_tensor = BinConstraintT(e1, TensorType(d1), op_eq)
    e11_tensor = BinConstraintT(e11, TensorType(d11), op_eq)
    e2_tensor = BinConstraintT(e2, TensorType(d2), op_eq)
    e12_tensor = BinConstraintT(e12, TensorType(d12), op_eq)
    return [e1_tensor, e11_tensor, e2_tensor, e12_tensor]


def gen_consistency_constraints(
    constraint: BinConstraintT, counter: int
) -> tuple[list[Constraint], int]:
    """
    Args:
        constraint: Consistency constraint on tensors
        counter: for variable tracking

    Returns: Equality and consistency constraints on dimensions

    """

    all_constraints: list[Constraint] = []

    for i in range(1, MAX_TENSOR_RANK + 1):
        new_dims_rhs_1, counter = gen_tensor_dims(i, counter)
        new_dims_rhs_2, counter = gen_tensor_dims(i, counter)

        nat_constraints = gen_nat_constraints(new_dims_rhs_1 + new_dims_rhs_2)

        c_tensor_i = Conj(
            [
                BinConstraintT(constraint.lhs, TensorType(new_dims_rhs_1), op_eq),
                BinConstraintT(constraint.rhs, TensorType(new_dims_rhs_2), op_eq),
            ]
            + [
                BinConstraintD(d1, d2, op_consistency)
                for d1, d2 in zip(new_dims_rhs_1, new_dims_rhs_2)
            ]
            + nat_constraints
        )

        all_constraints.append(c_tensor_i)

    return all_constraints, counter


def gen_greatest_upper_bound(
    constraint: TGreatestUpperBound, counter: int
) -> tuple[list[Constraint], int]:
    """
    Args:
        constraint: Greatest upper bound on tensors
        counter: variable tracking

    Returns: A set of equality constraints and DGreatestUpperBound constraints

    """

    all_constraints: list[Constraint] = []

    for i in range(1, MAX_TENSOR_RANK + 1):
        c: list[Constraint] = []
        dims1, counter = gen_tensor_dims(i, counter)
        c1tensor = TensorType(dims1)

        dims2, counter = gen_tensor_dims(i, counter)
        c2tensor = TensorType(dims2)

        dims3, counter = gen_tensor_dims(i, counter)
        c3tensor = TensorType(dims3)

        c += [
            BinConstraintT(constraint.rhs1, c1tensor, op_eq),
            BinConstraintT(constraint.rhs2, c2tensor, op_eq),
            BinConstraintT(constraint.res, c3tensor, op_eq),
        ] + gen_nat_constraints(dims1 + dims2 + dims3)

        if not (len(c3tensor.dims) == len(c1tensor.dims) == len(c2tensor.dims)):
            raise AssertionError("Tensor args lengths must be equal")
        for i in range(len(c3tensor.dims)):
            c.append(
                DGreatestUpperBound(
                    c3tensor.dims[i], c1tensor.dims[i], c2tensor.dims[i]
                )
            )

        all_constraints.append(Conj(c))
    return all_constraints, counter


def generate_all_broadcasting_possibilities_no_padding(
    d1: list[DVar], d2: list[DVar], d11: list[DVar], d12: list[DVar]
) -> Constraint:
    """
    Generate broadcasting constraints assuming no padding. Broadcasting can happen at any dimension.
    We look at all combinations for all dimensions in d1 and d2
    Args:
        d1: input1 dimensions
        d2: input2 dimensions
        d11: broadcasted input1 dimensions
        d12: broadcasted input2 dimensions

    Returns: broadcasting constraints relating the input dimensions to the broadcasted dimensions

    """

    size = len(d1)

    res2 = []

    for i in range(size):
        t1 = broadcast_dim(d1, d2, d11, d12, i)
        t2 = broadcast_dim(d2, d1, d12, d11, i)
        t3 = no_broadcast_dim_with_index(d1, d2, d11, d12, i)

        res2.append(Disj([t1, t2, t3]))

    return Conj(res2)


def gen_broadcasting_constraints(
    e1: TVar, e2: TVar, e11: TVar, e12: TVar, i: int, counter: int
) -> tuple[Constraint, Constraint, Constraint, list[BinConstraintD], int]:
    """
    Simulates broadcasting on e1 and e2 and returns the results
    respectively in e11 and e12. Because of gradual types,
    e1 and e2 may not be equal. Similarly, e11 and e12 may not
    be equal. e11 and e12 should be guaranteed to be consistent
    as they represent the shapes of the tensors to be added after
    broadcasting.
    Args:
        e1: TVar representing the type of input 1
        e2: TVar representing the type of input 2
        e11: TVar representing the broadcasted input 1
        e12: TVar representing the broadcasted input 2
        i: The rank of the resulting type of addition
        counter: for variable tracking

    Returns: Simplified broadcasting constraints

    """
    dims, counter = gen_lists_of_dims(4, i, counter)
    [d1, d2, d3, d4] = dims
    nat_dims_i = gen_nat_constraints(list(itertools.chain.from_iterable(dims)))

    initialize_tensors_constraints = create_equality_constraints_for_broadcasting(
        e1, e2, e11, e12, d1, d2, d3, d4
    )

    [e1_tensor, e11_tensor, e2_tensor, e12_tensor] = initialize_tensors_constraints

    # without padding, broadcast all possibilities for tensors of size i
    final_tensor_constraint_no_padding = Conj(
        [
            *initialize_tensors_constraints,
            generate_all_broadcasting_possibilities_no_padding(d1, d2, d3, d4),
        ]
    )

    # with padding, broadcast all possibilities for tensors of size i
    final_tensor_constraint_padding_arg1, counter = apply_padding(
        e1, e11_tensor, e2_tensor, e12_tensor, d2, d3, d4, counter
    )

    final_tensor_constraint_padding_arg2, counter = apply_padding(
        e2, e12_tensor, e1_tensor, e11_tensor, d1, d4, d3, counter
    )

    return (
        final_tensor_constraint_no_padding,
        final_tensor_constraint_padding_arg1,
        final_tensor_constraint_padding_arg2,
        nat_dims_i,
        counter,
    )
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ApplyBroadcasting",
    "BinConstraintD",
    "BinConstraintT",
    "BinaryConstraint",
    "BVar",
    "CalcConv",
    "CalcMaxPool",
    "CalcProduct",
    "CanReshape",
    "Conj",
    "Constraint",
    "DGreatestUpperBound",
    "Disj",
    "DVar",
    "F",
    "GetItem",
    "GetItemTensor",
    "IndexSelect",
    "Prod",
    "T",
    "TGreatestUpperBound",
    "Transpose",
    "TVar",
    "is_algebraic_expression",
    "is_bool_expr",
    "is_dim",
]

from torch.fx.experimental.migrate_gradual_types.operation import (
    op_add,
    op_div,
    op_eq,
    op_gt,
    op_lt,
    op_mod,
    op_mul,
    op_neq,
    op_sub,
)
from torch.fx.tensor_type import _DynType, Dyn, TensorType


class Constraint:
    pass


class Conj(Constraint):
    def __init__(self, conjuncts: Sequence[Constraint]) -> None:
        """
        :param conjuncts: Conjunction of constraints
        """
        self.conjucts = list(conjuncts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Conj):
            return self.conjucts == other.conjucts
        else:
            return False

    def __repr__(self) -> str:
        return f"And({self.conjucts})"


class Disj(Constraint):
    def __init__(self, disjuncts: Sequence[Constraint]) -> None:
        """
        :param disjuncts: Disjunction of constraints
        """
        self.disjuncts = list(disjuncts)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Disj):
            return self.disjuncts == other.disjuncts
        else:
            return False

    def __repr__(self) -> str:
        return f"Or({self.disjuncts})"


class Prod(Constraint):
    def __init__(self, products: Sequence[DVar | int | _DynType]) -> None:
        """
        :param products: lists of dimensions to multiply
        """
        self.products = list(products)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Prod):
            return self.products == other.products
        else:
            return False

    def __repr__(self) -> str:
        return f"Product({self.products})"


class T(Constraint):
    """
    True
    """

    def __init__(self) -> None:
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, T)

    def __repr__(self) -> str:
        return "True"


class F(Constraint):
    """
    False
    """

    def __init__(self) -> None:
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, F)

    def __repr__(self) -> str:
        return "False"


class BinaryConstraint(Constraint):
    """
    Represents all binary operations
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        """
        :param lhs: lhs of the constraint
        :param rhs: rhs of the constraint
        :param op: string representing the operation
        """
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BinaryConstraint):
            return (
                self.lhs == other.lhs and self.rhs == other.rhs and self.op == other.op
            )
        else:
            return False

    def __repr__(self) -> str:
        return f"({self.lhs} {self.op} {self.rhs})"


class BinConstraintT(BinaryConstraint):
    """
    Binary constraints about tensors
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        if not (
            (isinstance(lhs, (TVar, TensorType, int)) or lhs == Dyn)
            and (isinstance(rhs, (TVar, TensorType, int)) or rhs == Dyn)
        ):
            raise AssertionError(f"Invalid types: lhs={type(lhs)}, rhs={type(rhs)}")
        super().__init__(lhs, rhs, op)


class BinConstraintD(BinaryConstraint):
    """
    Binary constraints about dimensions
    """

    def __init__(self, lhs: _Operand, rhs: _Operand, op: str | None) -> None:
        if not (is_algebraic_expression(lhs) or is_dim(lhs) or is_bool_expr(lhs)):
            raise AssertionError(f"Invalid lhs type: {type(lhs)}")
        if not (is_algebraic_expression(rhs) or is_dim(rhs) or is_bool_expr(rhs)):
            raise AssertionError(f"Invalid rhs type: {type(rhs)}")

        super().__init__(lhs, rhs, op)


class TGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for tensors with dynamic type
    """

    def __init__(self, res: TVar, rhs1: TVar, rhs2: TVar) -> None:
        """
        :param res: tensor variable that stores the result of the output
        :param rhs1: tensor or tensor variable
        :param rhs2: tensor or tensor variable
        """
        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self) -> str:
        return f"{self.res} = {self.rhs1}\u2294*{self.rhs2}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TGreatestUpperBound):
            return (
                self.res == other.res
                and self.rhs1 == other.rhs1
                and self.rhs2 == other.rhs2
            )
        else:
            return False


class DGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for dimensions
    """

    def __init__(
        self,
        res: DVar | int | _DynType,
        rhs1: DVar | int | _DynType,
        rhs2: DVar | int | _DynType,
    ) -> None:
        """
        :param res: Dimension variable to store the result
        :param rhs1: dimension variable 1
        :param rhs2: dimension variable 2
        """
        if not is_dim(res):
            raise AssertionError(f"Expected dimension for res, got {type(res)}")
        if not is_dim(rhs1):
            raise AssertionError(f"Expected dimension for rhs1, got {type(rhs1)}")
        if not is_dim(rhs2):
            raise AssertionError(f"Expected dimension for rhs2, got {type(rhs2)}")

        self.res = res
        self.rhs1 = rhs1
        self.rhs2 = rhs2

    def __repr__(self) -> str:
        return f"{self.res} = {self.rhs1}\u2294{self.rhs2}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DGreatestUpperBound):
            return (
                self.res == other.res
                and self.rhs1 == other.rhs1
                and self.rhs2 == other.rhs2
            )
        else:
            return False


class CanReshape(Constraint):
    """
    can_reshape constraint
    """

    def __init__(self, src: TVar, target: TensorType) -> None:
        """
        :param src: tensor variable
        :param target: tensor
        """
        self.src = src
        self.target = target

    def __repr__(self) -> str:
        return f"can-reshape({self.src}, {self.target})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CanReshape):
            return self.src == other.src and self.target == other.target
        else:
            return False


class IndexSelect(Constraint):
    def __init__(
        self,
        tensor_size: int,
        input_var: TVar,
        dim_replace: DVar | _DynType,
        index: int,
        output: TVar,
    ) -> None:
        """
        Args:
            input_var: input to index_select
            tensor_size: tensor size we are considering
            dim_replace: the dimension of the output at "index"
            index: location of the dimensions to replace in the input
            output: variable to store the result
        """
        if not isinstance(input_var, TVar):
            raise AssertionError(f"Expected TVar, got {type(input_var)}")
        if not isinstance(output, TVar):
            raise AssertionError(f"Expected TVar, got {type(output)}")
        if not (isinstance(dim_replace, DVar) or dim_replace == Dyn):
            raise AssertionError(f"Expected DVar or Dyn, got {type(dim_replace)}")
        if not isinstance(index, int):
            raise AssertionError(f"Expected int, got {type(index)}")

        self.input_var = input_var
        self.tensor_size = tensor_size
        self.dim_replace = dim_replace
        self.index = index
        self.output = output

    def __repr__(self) -> str:
        return (
            f" {self.output} = "
            f"IndexSelect({self.input_var}, "
            f"tensor_size: {self.tensor_size}, "
            f"{self.dim_replace}, "
            f"{self.index})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IndexSelect):
            return (
                self.tensor_size == other.tensor_size
                and self.dim_replace == other.dim_replace
                and self.index == other.index
                and self.output == other.output
                and self.input_var == other.input_var
            )
        else:
            return False


class Transpose(Constraint):
    def __init__(
        self, tensor_size: int, input_var: TVar, index1: int, index2: int, output: TVar
    ) -> None:
        """
        Args:
            tensor_size: current tensor size
            input_var: variable to hold input
            index1: dimension 1
            index2: dimension 2
            output: output that stores result
        """
        if not isinstance(input_var, TVar):
            raise AssertionError(f"Expected TVar, got {type(input_var)}")
        if not isinstance(output, TVar):
            raise AssertionError(f"Expected TVar, got {type(output)}")
        if not isinstance(index1, int):
            raise AssertionError(f"Expected int, got {type(index1)}")
        if not isinstance(index2, int):
            raise AssertionError(f"Expected int, got {type(index2)}")

        self.input_var = input_var
        self.tensor_size = tensor_size
        self.index1 = index1
        self.index2 = index2
        self.output = output

    def __repr__(self) -> str:
        return (
            f" {self.output} = "
            f"Transpose({self.input_var}, "
            f"tensor_size: {self.tensor_size}, "
            f"{self.index1}, "
            f"{self.index2})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Transpose):
            return (
                self.tensor_size == other.tensor_size
                and self.index1 == other.index1
                and self.index2 == other.index2
                and self.output == other.output
                and self.input_var == other.input_var
            )
        else:
            return False


class GetItem(Constraint):
    def __init__(
        self, tensor_size: int, index: int, res: DVar, input_var: TVar
    ) -> None:
        """
        Constraint for getting item given a tensor size
        :param tensor_size: actual number
        :param index: actual number representing the index
        :param res: dimension variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        if not isinstance(res, DVar):
            raise AssertionError(f"Expected DVar, got {type(res)}")

        self.res = res
        self.tensor_size = tensor_size
        self.index = index
        self.input_var = input_var

    def __repr__(self) -> str:
        return f" {self.res} = GetItem({self.input_var}, tensor_size: {self.tensor_size}, {self.index})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GetItem):
            return (
                self.res == other.res
                and self.tensor_size == other.tensor_size
                and self.index == other.index
                and self.input_var == other.input_var
            )
        else:
            return False


class GetItemTensor(Constraint):
    def __init__(
        self,
        tensor_size: int,
        index_tuple: tuple[None | slice, ...],
        res: TVar,
        input_var: TVar,
    ) -> None:
        """
        Constraint for getting item given a tensor size
        However, when the argument is a tuple, we will
        expect a tensor
        :param tensor_size: actual number representing the rank
        :param index_tuple: tuple for indexing
        :param res: tensor variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
        if not isinstance(res, TVar):
            raise AssertionError(f"Expected TVar, got {type(res)}")

        self.res = res
        self.tensor_size = tensor_size
        self.index_tuple = index_tuple
        self.input_var = input_var

    def __repr__(self) -> str:
        return f" {self.res} = GetItemT({self.input_var}, tensor_size: {self.tensor_size}, {self.index_tuple})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GetItemTensor):
            return (
                self.res == other.res
                and self.tensor_size == other.tensor_size
                and self.index_tuple == other.index_tuple
                and self.input_var == other.input_var
            )
        else:
            return False


class CalcConv(Constraint):
    def __init__(
        self,
        conv_result: TVar,
        input_var: TVar,
        c_out: int,
        kernel: int | tuple[int, int],
        padding: int | tuple[int, int],
        stride: int | tuple[int, int],
        dilation: int | tuple[int, int],
        matching_constraint_vars: list[DVar],
    ) -> None:
        """
        :param conv_result: the convolution result
        :param input_var: input to convolution
        :param c_out: output channel type
        :param kernel: kernel tuple
        """
        self.conv_result = conv_result
        self.input_var = input_var
        self.c_out = c_out
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self) -> str:
        return (
            f"{self.conv_result} ="
            f" calc-conv({self.input_var},"
            f" {self.c_out}, {self.kernel}, "
            f"{self.padding}, {self.stride},"
            f" {self.dilation})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CalcConv):
            return (
                self.conv_result == other.conv_result
                and self.input_var == other.input_var
                and self.c_out == other.c_out
                and self.kernel == other.kernel
                and self.padding == other.padding
                and self.stride == other.stride
                and self.dilation == other.dilation
                and self.matching_constraint == other.matching_constraint
            )
        else:
            return False


class CalcMaxPool(Constraint):
    def __init__(
        self,
        maxpool_result: TVar,
        input_var: TVar,
        kernel: int | tuple[int, int],
        padding: int | tuple[int, int],
        stride: int | tuple[int, int],
        dilation: int | tuple[int, int],
        matching_constraint_vars: list[DVar],
    ) -> None:
        """
        :param maxpool_result: the result of maxpool
        :param input_var: input to convolution
        :param kernel: kernel tuple
        """
        self.maxpool_result = maxpool_result
        self.input_var = input_var
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.matching_constraint = matching_constraint_vars

    def __repr__(self) -> str:
        return (
            f"{self.maxpool_result} ="
            f" calc-maxpool({self.input_var},"
            f"  {self.kernel}, "
            f"{self.padding}, {self.stride},"
            f" {self.dilation})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CalcMaxPool):
            return (
                self.maxpool_result == other.maxpool_result
                and self.input_var == other.input_var
                and self.kernel == other.kernel
                and self.padding == other.padding
                and self.stride == other.stride
                and self.dilation == other.dilation
                and self.matching_constraint == other.matching_constraint
            )
        else:
            return False


class ApplyBroadcasting(Constraint):
    def __init__(self, res1: TVar, res2: TVar, input1: TVar, input2: TVar) -> None:
        """
        :param res1: resulting tensor 1
        :param res2: resulting tensor 2
        :param input1: tensor variable 1
        :param input2: tensor variable 2
        """
        self.res1 = res1
        self.res2 = res2
        self.input1 = input1
        self.input2 = input2

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ApplyBroadcasting):
            return (
                self.res1 == other.res1
                and self.res2 == other.res2
                and self.input1 == other.input1
                and self.input2 == other.input2
            )
        else:
            return False

    def __repr__(self) -> str:
        return (
            f"{self.res1}, {self.res2} ="
            f" apply-broadcasting({self.input1},"
            f" {self.input2})"
        )


class CalcProduct(Constraint):
    """
    Given correct dimensions, calculate the product for flatten accounting for Dyn
    """

    def __init__(
        self, start: int, end: int, flattened: TVar, dims_to_flatten: list[DVar]
    ) -> None:
        """
        :param start: start index
        :param end: end index
        :param flattened: variable to store the product
        :param dims_to_flatten: the type which we will flatten
        """
        if not isinstance(dims_to_flatten, list):
            raise AssertionError(f"Expected list, got {type(dims_to_flatten)}")
        if not isinstance(flattened, TVar):
            raise AssertionError(f"Expected TVar, got {type(flattened)}")
        if not isinstance(start, int):
            raise AssertionError(f"Expected int, got {type(start)}")
        if not isinstance(end, int):
            raise AssertionError(f"Expected int, got {type(end)}")

        self.start = start
        self.end = end
        self.dims_to_flatten = dims_to_flatten
        self.flattened = flattened

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CalcProduct):
            return (
                self.start == other.start
                and self.end == other.end
                and self.dims_to_flatten == other.dims_to_flatten
                and self.flattened == other.flattened
            )

        else:
            return False

    def __repr__(self) -> str:
        return f"{self.flattened} = CalcProduct({self.start}, {self.end}, {self.dims_to_flatten})"


class TVar:
    """
    Tensor variable with no tensor constructor
    """

    def __init__(self, tvar: int) -> None:
        """
        :param tvar: tensor variable
        """
        self.tvar = tvar

    def __repr__(self) -> str:
        return f"TV({self.tvar})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TVar):
            return self.tvar == other.tvar
        else:
            return False


class DVar:
    """
    Dimension variable
    """

    def __init__(self, c: int) -> None:
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self) -> str:
        return f"DV({self.c})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DVar):
            return self.c == other.c
        else:
            return False


class BVar:
    """
    Boolean variable
    """

    def __init__(self, c: int) -> None:
        """
        :param c: character or number
        """
        self.c = c

    def __repr__(self) -> str:
        return f"BV({self.c})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BVar):
            return self.c == other.c
        else:
            return False


_Operand: TypeAlias = (
    TVar
    | TensorType
    | DVar
    | int
    | float
    | bool
    | _DynType
    | BinConstraintD
    | Prod
    | BVar
    | Conj
    | Disj
    | None
)


def is_algebraic_expression(constraint: object) -> bool:
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_add, op_sub, op_div, op_mul, op_mod]
    else:
        return isinstance(constraint, Prod)


def is_bool_expr(constraint: object) -> bool:
    if isinstance(constraint, BinConstraintD):
        return constraint.op in [op_gt, op_lt, op_neq, op_eq]
    else:
        return isinstance(constraint, (BVar, Conj, Disj))


def is_dim(d: object) -> bool:
    return isinstance(d, (DVar, int)) or d == Dyn
<|endoftext|>from __future__ import annotations

import collections
import operator
from collections.abc import Mapping
from functools import reduce
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_K = TypeVar("_K")
_V = TypeVar("_V")
_K2 = TypeVar("_K2")
_V2 = TypeVar("_V2")
_T = TypeVar("_T")


__all__ = [
    "merge",
    "merge_with",
    "valmap",
    "keymap",
    "itemmap",
    "valfilter",
    "keyfilter",
    "itemfilter",
    "assoc",
    "dissoc",
    "assoc_in",
    "update_in",
    "get_in",
]


def _get_factory(f: Callable[..., object], kwargs: dict[str, object]) -> type:
    factory: type = kwargs.pop("factory", dict)  # type: ignore[assignment]
    if kwargs:
        raise TypeError(
            f"{f.__name__}() got an unexpected keyword argument '{kwargs.popitem()[0]}'"
        )
    return factory


def merge(*dicts: Mapping[object, object], **kwargs: object) -> object:
    """Merge a collection of dictionaries

    >>> merge({1: "one"}, {2: "two"})
    {1: 'one', 2: 'two'}

    Later dictionaries have precedence

    >>> merge({1: 2, 3: 4}, {3: 3, 4: 4})
    {1: 2, 3: 3, 4: 4}

    See Also:
        merge_with
    """
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]
    factory = _get_factory(merge, kwargs)

    rv = factory()
    for d in dicts:
        rv.update(d)
    return rv


def merge_with(
    func: Callable[..., object], *dicts: Mapping[object, object], **kwargs: object
) -> object:
    """Merge dictionaries and apply function to combined values

    A key may occur in more than one dict, and all values mapped from the key
    will be passed to the function as a list, such as func([val1, val2, ...]).

    >>> merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
    {1: 11, 2: 22}

    >>> merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})  # doctest: +SKIP
    {1: 1, 2: 2, 3: 30}

    See Also:
        merge
    """
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]
    factory = _get_factory(merge_with, kwargs)

    result = factory()
    for d in dicts:
        for k, v in d.items():
            if k not in result:
                result[k] = [v]
            else:
                result[k].append(v)
    return valmap(func, result, factory)


def valmap(
    func: Callable[[_V], _V2], d: Mapping[_K, _V], factory: type = dict
) -> dict[_K, _V2]:
    """Apply function to values of dictionary

    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    >>> valmap(sum, bills)  # doctest: +SKIP
    {'Alice': 65, 'Bob': 45}

    See Also:
        keymap
        itemmap
    """
    rv = factory()
    rv.update(zip(d.keys(), map(func, d.values())))
    return rv


def keymap(
    func: Callable[[_K], _K2], d: Mapping[_K, _V], factory: type = dict
) -> dict[_K2, _V]:
    """Apply function to keys of dictionary

    >>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
    >>> keymap(str.lower, bills)  # doctest: +SKIP
    {'alice': [20, 15, 30], 'bob': [10, 35]}

    See Also:
        valmap
        itemmap
    """
    rv = factory()
    rv.update(zip(map(func, d.keys()), d.values()))
    return rv


def itemmap(
    func: Callable[[tuple[_K, _V]], object], d: Mapping[_K, _V], factory: type = dict
) -> dict[object, object]:
    """Apply function to items of dictionary

    >>> accountids = {"Alice": 10, "Bob": 20}
    >>> itemmap(reversed, accountids)  # doctest: +SKIP
    {10: "Alice", 20: "Bob"}

    See Also:
        keymap
        valmap
    """
    rv = factory()
    rv.update(map(func, d.items()))
    return rv


def valfilter(
    predicate: Callable[[_V], bool], d: Mapping[_K, _V], factory: type = dict
) -> dict[_K, _V]:
    """Filter items in dictionary by value

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilter(iseven, d)
    {1: 2, 3: 4}

    See Also:
        keyfilter
        itemfilter
        valmap
    """
    rv = factory()
    for k, v in d.items():
        if predicate(v):
            rv[k] = v
    return rv


def keyfilter(
    predicate: Callable[[_K], bool], d: Mapping[_K, _V], factory: type = dict
) -> dict[_K, _V]:
    """Filter items in dictionary by key

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> keyfilter(iseven, d)
    {2: 3, 4: 5}

    See Also:
        valfilter
        itemfilter
        keymap
    """
    rv = factory()
    for k, v in d.items():
        if predicate(k):
            rv[k] = v
    return rv


def itemfilter(
    predicate: Callable[[tuple[_K, _V]], bool], d: Mapping[_K, _V], factory: type = dict
) -> dict[_K, _V]:
    """Filter items in dictionary by item

    >>> def isvalid(item):
    ...     k, v = item
    ...     return k % 2 == 0 and v < 4

    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> itemfilter(isvalid, d)
    {2: 3}

    See Also:
        keyfilter
        valfilter
        itemmap
    """
    rv = factory()
    for item in d.items():
        if predicate(item):
            k, v = item
            rv[k] = v
    return rv


def assoc(
    d: Mapping[_K, _V], key: object, value: object, factory: type = dict
) -> dict[_K, _V]:
    """Return a new dict with new key value pair

    New dict has d[key] set to value. Does not modify the initial dictionary.

    >>> assoc({"x": 1}, "x", 2)
    {'x': 2}
    >>> assoc({"x": 1}, "y", 3)  # doctest: +SKIP
    {'x': 1, 'y': 3}
    """
    d2 = factory()
    d2.update(d)
    d2[key] = value
    return d2


def dissoc(d: Mapping[object, object], *keys: object, **kwargs: object) -> object:
    """Return a new dict with the given key(s) removed.

    New dict has d[key] deleted for each supplied key.
    Does not modify the initial dictionary.

    >>> dissoc({"x": 1, "y": 2}, "y")
    {'x': 1}
    >>> dissoc({"x": 1, "y": 2}, "y", "x")
    {}
    >>> dissoc({"x": 1}, "y")  # Ignores missing keys
    {'x': 1}
    """
    factory = _get_factory(dissoc, kwargs)
    d2 = factory()

    if len(keys) < len(d) * 0.6:
        d2.update(d)
        for key in keys:
            if key in d2:
                del d2[key]
    else:
        remaining = set(d)
        remaining.difference_update(keys)
        for k in remaining:
            d2[k] = d[k]
    return d2


def assoc_in(
    d: Mapping[object, object],
    keys: Iterable[object],
    value: object,
    factory: type = dict,
) -> object:
    """Return a new dict with new, potentially nested, key value pair

    >>> purchase = {
    ...     "name": "Alice",
    ...     "order": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
    ...     "credit card": "5555-1234-1234-1234",
    ... }
    >>> assoc_in(purchase, ["order", "costs"], [0.25, 1.00])  # doctest: +SKIP
    {'credit card': '5555-1234-1234-1234',
     'name': 'Alice',
     'order': {'costs': [0.25, 1.00], 'items': ['Apple', 'Orange']}}
    """
    return update_in(d, keys, lambda x: value, value, factory)


def update_in(
    d: Mapping[object, object],
    keys: Iterable[object],
    func: Callable[..., object],
    default: object = None,
    factory: type = dict,
) -> object:
    """Update value in a (potentially) nested dictionary

    inputs:
    d - dictionary on which to operate
    keys - list or tuple giving the location of the value to be changed in d
    func - function to operate on that value

    If keys == [k0,..,kX] and d[k0]..[kX] == v, update_in returns a copy of the
    original dictionary with v replaced by func(v), but does not mutate the
    original dictionary.

    If k0 is not a key in d, update_in creates nested dictionaries to the depth
    specified by the keys, with the innermost value set to func(default).

    >>> inc = lambda x: x + 1
    >>> update_in({"a": 0}, ["a"], inc)
    {'a': 1}

    >>> transaction = {
    ...     "name": "Alice",
    ...     "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
    ...     "credit card": "5555-1234-1234-1234",
    ... }
    >>> update_in(transaction, ["purchase", "costs"], sum)  # doctest: +SKIP
    {'credit card': '5555-1234-1234-1234',
     'name': 'Alice',
     'purchase': {'costs': 1.75, 'items': ['Apple', 'Orange']}}

    >>> # updating a value when k0 is not in d
    >>> update_in({}, [1, 2, 3], str, default="bar")
    {1: {2: {3: 'bar'}}}
    >>> update_in({1: "foo"}, [2, 3, 4], inc, 0)
    {1: 'foo', 2: {3: {4: 1}}}
    """
    ks = iter(keys)
    k = next(ks)

    rv = inner = factory()
    rv.update(d)

    for key in ks:
        if k in d:
            d = d[k]  # pyrefly: ignore[bad-assignment]
            dtemp = factory()
            dtemp.update(d)
        else:
            d = dtemp = factory()

        inner[k] = inner = dtemp
        k = key

    if k in d:
        inner[k] = func(d[k])
    else:
        inner[k] = func(default)
    return rv


def get_in(
    keys: Iterable[object],
    coll: object,
    default: object = None,
    no_default: bool = False,
) -> object:
    """Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.

    If coll[i0][i1]...[iX] cannot be found, returns ``default``, unless
    ``no_default`` is specified, then it raises KeyError or IndexError.

    ``get_in`` is a generalization of ``operator.getitem`` for nested data
    structures such as dictionaries and lists.

    >>> transaction = {
    ...     "name": "Alice",
    ...     "purchase": {"items": ["Apple", "Orange"], "costs": [0.50, 1.25]},
    ...     "credit card": "5555-1234-1234-1234",
    ... }
    >>> get_in(["purchase", "items", 0], transaction)
    'Apple'
    >>> get_in(["name"], transaction)
    'Alice'
    >>> get_in(["purchase", "total"], transaction)
    >>> get_in(["purchase", "items", "apple"], transaction)
    >>> get_in(["purchase", "items", 10], transaction)
    >>> get_in(["purchase", "total"], transaction, 0)
    0
    >>> get_in(["y"], {}, no_default=True)
    Traceback (most recent call last):
        ...
    KeyError: 'y'

    See Also:
        itertoolz.get
        operator.getitem
    """
    try:
        return reduce(
            operator.getitem,
            keys,  # pyrefly: ignore[bad-argument-type]
            coll,  # pyrefly: ignore[bad-argument-type]
        )
    except (KeyError, IndexError, TypeError):
        if no_default:
            raise
        return default


def getter(index: object) -> Callable[..., object]:
    if isinstance(index, list):
        if len(index) == 1:
            index = index[0]
            return lambda x: (x[index],)
        elif index:
            return operator.itemgetter(*index)
        else:
            return lambda x: ()
    else:
        return operator.itemgetter(index)


def groupby(key: object, seq: Iterable[object]) -> dict[object, list[object]]:
    """Group a collection by a key function

    >>> names = ["Alice", "Bob", "Charlie", "Dan", "Edith", "Frank"]
    >>> groupby(len, names)  # doctest: +SKIP
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}

    >>> iseven = lambda x: x % 2 == 0
    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}

    Non-callable keys imply grouping on a member.

    >>> groupby(
    ...     "gender",
    ...     [
    ...         {"name": "Alice", "gender": "F"},
    ...         {"name": "Bob", "gender": "M"},
    ...         {"name": "Charlie", "gender": "M"},
    ...     ],
    ... )  # doctest:+SKIP
    {'F': [{'gender': 'F', 'name': 'Alice'}],
     'M': [{'gender': 'M', 'name': 'Bob'},
           {'gender': 'M', 'name': 'Charlie'}]}

    Not to be confused with ``itertools.groupby``

    See Also:
        countby
    """
    if not callable(key):
        key = getter(key)
    d = collections.defaultdict(lambda: [].append)  # type: ignore[var-annotated]
    for item in seq:
        d[key(item)](item)
    rv: dict[object, list[object]] = {}
    for k, v in d.items():
        rv[k] = v.__self__  # type: ignore[attr-defined]
    return rv


def first(seq: Iterable[_T]) -> _T:
    """The first element in a sequence

    >>> first("ABC")
    'A'
    """
    return next(iter(seq))
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .variable import Var

_T = TypeVar("_T")


__all__ = ["hashable", "transitive_get", "raises", "reverse_dict", "xfail", "freeze"]


def hashable(x: object) -> bool:
    try:
        hash(x)
        return True
    except TypeError:
        return False


def transitive_get(key: object, d: dict[Var, object]) -> object:
    """Transitive dict.get
    >>> d = {1: 2, 2: 3, 3: 4}
    >>> d.get(1)
    2
    >>> transitive_get(1, d)
    4
    """
    while hashable(key) and key in d:
        key = d[key]
    return key


def raises(
    err: type[BaseException],
    lamda: Callable[[], object],  # codespell:ignore lamda
) -> bool:
    try:
        lamda()  # codespell:ignore lamda
        return False
    except err:
        return True


# Taken from theano/theano/gof/sched.py
# Avoids licensing issues because this was written by Matthew Rocklin
def _toposort(edges: dict[_T, Iterable[_T]]) -> list[_T]:
    """Topological sort algorithm by Kahn [1] - O(nodes + vertices)
    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges
    >>> # xdoctest: +SKIP
    >>> _toposort({1: (2, 3), 2: (3,)})
    [1, 2, 3]
    Closely follows the wikipedia page [2]
    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = {k: set(val) for k, val in incoming_edges.items()}
    S = {v for v in edges if v not in incoming_edges}
    L = []

    while S:
        n = S.pop()
        L.append(n)
        for m in edges.get(n, ()):
            if n not in incoming_edges[m]:
                raise AssertionError(f"Expected {n} in incoming_edges[{m}]")
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S.add(m)
    if any(incoming_edges.get(v) for v in edges):
        raise ValueError("Input has cycles")
    return L


def reverse_dict(d: dict[_T, Iterable[_T]]) -> dict[_T, tuple[_T, ...]]:
    """Reverses direction of dependence dict.

    >>> d = {"a": (1, 2), "b": (2, 3), "c": ()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}

    .. note::
        dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.
    """
    result = {}  # type: ignore[var-annotated]
    for key in d:
        for val in d[key]:
            # pyrefly: ignore [unsupported-operation]
            result[val] = result.get(val, ()) + (key,)
    return result  # pyrefly: ignore[bad-return]


def xfail(func: Callable[[], object]) -> None:
    try:
        func()
        raise Exception("XFailed test passed")  # pragma:nocover  # noqa: TRY002
    except Exception:
        pass


def freeze(d: object) -> object:
    """Freeze container to hashable form
    >>> freeze(1)
    1
    >>> freeze([1, 2])
    (1, 2)
    >>> freeze({1: 2})  # doctest: +SKIP
    frozenset([(1, 2)])
    """
    if isinstance(d, dict):
        return frozenset(map(freeze, d.items()))
    if isinstance(d, set):
        return frozenset(map(freeze, d))
    if isinstance(d, (tuple, list)):
        return tuple(map(freeze, d))
    return d
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

from .core import reify, unify  # type: ignore[attr-defined]


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .variable import Var

from .unification_tools import first, groupby  # type: ignore[import]
from .utils import _toposort, freeze
from .variable import isvar


class Dispatcher:
    def __init__(self, name: str) -> None:
        self.name = name
        self.funcs: dict[object, Callable[..., object]] = {}
        self.ordering: list[object] = []

    def add(self, signature: tuple[object, ...], func: Callable[..., object]) -> None:
        self.funcs[freeze(signature)] = func
        self.ordering = ordering(self.funcs)

    def __call__(self, *args: object, **kwargs: object) -> object:
        func, _ = self.resolve(args)
        return func(*args, **kwargs)

    def resolve(
        self, args: tuple[object, ...]
    ) -> tuple[Callable[..., object], dict[Var, object]]:
        n = len(args)
        for signature in self.ordering:
            if len(signature) != n:  # pyrefly: ignore[bad-argument-type]
                continue
            s = unify(freeze(args), signature)
            if s is not False:
                result = self.funcs[signature]
                return result, s  # pyrefly: ignore[bad-return]
        raise NotImplementedError(
            "No match found. \nKnown matches: "
            + str(self.ordering)
            + "\nInput: "
            + str(args)
        )

    def register(self, *signature: object) -> Callable[..., object]:
        def _(func: Callable[..., object]) -> Dispatcher:
            self.add(signature, func)
            return self

        return _


class VarDispatcher(Dispatcher):
    """A dispatcher that calls functions with variable names
    >>> # xdoctest: +SKIP
    >>> d = VarDispatcher("d")
    >>> x = var("x")
    >>> @d.register("inc", x)
    ... def f(x):
    ...     return x + 1
    >>> @d.register("double", x)
    ... def f(x):
    ...     return x * 2
    >>> d("inc", 10)
    11
    >>> d("double", 10)
    20
    """

    def __call__(self, *args: object, **kwargs: object) -> object:
        func, s = self.resolve(args)
        d = {k.token: v for k, v in s.items()}  # pyrefly: ignore[missing-attribute]
        return func(**d)


global_namespace: dict[str, Dispatcher] = {}


def match(*signature: object, **kwargs: object) -> Callable[..., object]:
    namespace: dict[str, Dispatcher] = kwargs.get(  # type: ignore[assignment]
        "namespace", global_namespace
    )
    dispatcher_cls: type[Dispatcher] = kwargs.get(  # type: ignore[assignment]
        "Dispatcher", Dispatcher
    )

    def _(func: Callable[..., object]) -> Dispatcher:
        name = func.__name__

        if name not in namespace:
            namespace[name] = dispatcher_cls(name)
        d = namespace[name]

        d.add(signature, func)

        return d

    return _


def supercedes(a: object, b: object) -> bool:
    """``a`` is a more specific match than ``b``"""
    if isvar(b) and not isvar(a):
        return True
    s = unify(a, b)
    if s is False:
        return False
    s = {
        k: v
        for k, v in s.items()  # pyrefly: ignore[missing-attribute]
        if not isvar(k) or not isvar(v)
    }
    if reify(a, s) == a:
        return True
    if reify(b, s) == b:
        return False
    return False


# Taken from multipledispatch
def edge(a: object, b: object, tie_breaker: Callable[[object], int] = hash) -> bool:
    """A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
    if supercedes(a, b):
        if supercedes(b, a):
            return tie_breaker(a) > tie_breaker(b)
        else:
            return True
    return False


# Taken from multipledispatch
def ordering(signatures: Iterable[object]) -> list[object]:
    """A sane ordering of signatures to check, first to last
    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))  # pyrefly: ignore[bad-argument-type]
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(first, edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    edges = {k: [b for a, b in v] for k, v in edges.items()}  # type: ignore[attr-defined, assignment]
    return _toposort(edges)  # pyrefly: ignore[bad-argument-type]
<|endoftext|>from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING
from typing_extensions import TypeVarTuple, Unpack

from .dispatch import dispatch
from .unification_tools import assoc  # type: ignore[import]
from .utils import transitive_get as walk
from .variable import isvar


if TYPE_CHECKING:
    from .variable import Var


_Ts = TypeVarTuple("_Ts")

__all__ = ["reify", "unify"]

###############
# Reification #
###############


@dispatch(Iterator, dict)
def _reify(t: Iterator[object], s: dict[Var, object]) -> Iterator[object]:
    return map(partial(reify, s=s), t)
    # return (reify(arg, s) for arg in t)


@dispatch(tuple, dict)  # type: ignore[no-redef]
def _reify(t: tuple[Unpack[_Ts]], s: dict[Var, object]) -> tuple[Unpack[_Ts]]:
    return tuple(reify(iter(t), s))  # pyrefly: ignore[bad-argument-type, bad-return]


@dispatch(list, dict)  # type: ignore[no-redef]
def _reify(t: list[object], s: dict[Var, object]) -> list[object]:
    return list(reify(iter(t), s))  # pyrefly: ignore[bad-argument-type]


@dispatch(dict, dict)  # type: ignore[no-redef]
def _reify(d: dict[object, object], s: dict[Var, object]) -> dict[object, object]:
    return {k: reify(v, s) for k, v in d.items()}


@dispatch(object, dict)  # type: ignore[no-redef]
def _reify(o: object, s: dict[Var, object]) -> object:
    return o  # catch all, just return the object


def reify(e: object, s: dict[Var, object]) -> object:
    """Replace variables of expression with substitution
    >>> # xdoctest: +SKIP
    >>> x, y = var(), var()
    >>> e = (1, x, (3, y))
    >>> s = {x: 2, y: 4}
    >>> reify(e, s)
    (1, 2, (3, 4))
    >>> e = {1: x, 3: (y, 5)}
    >>> reify(e, s)
    {1: 2, 3: (4, 5)}
    """
    if isvar(e):
        return reify(s[e], s) if e in s else e
    return _reify(e, s)


###############
# Unification #
###############

seq = tuple, list, Iterator


@dispatch(seq, seq, dict)  # type: ignore[arg-type]
def _unify(
    u: Sequence[object], v: Sequence[object], s: dict[Var, object]
) -> dict[Var, object] | bool:
    if len(u) != len(v):
        return False
    for uu, vv in zip(u, v):  # avoiding recursion
        s = unify(uu, vv, s)  # pyrefly: ignore[bad-assignment]
        if s is False:
            return False
    return s


#
# @dispatch((set, frozenset), (set, frozenset), dict)
# def _unify(u, v, s):
#     i = u & v
#     u = u - i
#     v = v - i
#     return _unify(sorted(u), sorted(v), s)
#
#
# @dispatch(dict, dict, dict)
# def _unify(u, v, s):
#     if len(u) != len(v):
#         return False
#     for key, uval in iteritems(u):
#         if key not in v:
#             return False
#         s = unify(uval, v[key], s)
#         if s is False:
#             return False
#     return s
#
#
# @dispatch(object, object, dict)
# def _unify(u, v, s):
#     return False  # catch all


@dispatch(object, object, dict)
def unify(
    u: object, v: object, s: dict[Var, object]
) -> dict[Var, object] | bool:  # no check at the moment
    """Find substitution so that u == v while satisfying s
    >>> x = var("x")
    >>> unify((1, x), (1, 2), {})
    {~x: 2}
    """
    u = walk(u, s)
    v = walk(v, s)
    if u == v:
        return s
    if isvar(u):
        return assoc(s, u, v)
    if isvar(v):
        return assoc(s, v, u)
    return _unify(u, v, s)


@dispatch(object, object)  # type: ignore[no-redef]
# The noqa suppresses F811 for this multipledispatch overload redefinition
# (it replaces the bare-name-statement hack that previously silenced it).
def unify(u: object, v: object) -> dict[Var, object] | bool:  # noqa: F811
    return unify(u, v, {})
<|endoftext|>from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Hashable
    from typing import Literal

from .dispatch import dispatch
from .utils import hashable


_global_logic_variables: set[Hashable] = set()
_glv = _global_logic_variables


class Var:
    """Logic Variable"""

    _id = 1

    def __new__(cls, *token: Hashable) -> Var:  # noqa: PYI034
        if len(token) == 0:
            token = f"_{Var._id}"  # type: ignore[assignment]
            Var._id += 1
        elif len(token) == 1:
            token = token[0]  # pyrefly: ignore[bad-assignment]

        obj = object.__new__(cls)
        obj.token = token  # type: ignore[attr-defined]
        return obj

    def __str__(self) -> str:
        return "~" + str(self.token)  # type: ignore[attr-defined]

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.token == other.token  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self), self.token))  # type: ignore[attr-defined]


def var() -> Callable[..., Var]:
    return lambda *args: Var(*args)


def vars() -> Callable[[int], list[Callable[..., Var]]]:
    return lambda n: [var() for i in range(n)]


@dispatch(Var)
def isvar(v: Var) -> Literal[True]:
    return True


@dispatch(object)  # type: ignore[no-redef]
# The noqa suppresses F811 for this multipledispatch overload redefinition
# (it replaces the bare-name-statement hack that previously silenced it).
def isvar(o: object) -> bool:  # noqa: F811
    return bool(_glv and hashable(o) and o in _glv)


@contextmanager
def variables(*variables: Hashable) -> Generator[None, None, None]:
    """
    Context manager for logic variables

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> from __future__ import with_statement
        >>> with variables(1):
        ...     print(isvar(1))
        True
        >>> print(isvar(1))
        False
        >>> # Normal approach
        >>> from unification import unify
        >>> x = var("x")
        >>> unify(x, 1)
        {~x: 1}
        >>> # Context Manager approach
        >>> with variables("x"):
        ...     print(unify("x", 1))
        {'x': 1}
    """
    old_global_logic_variables = _global_logic_variables.copy()
    _global_logic_variables.update(set(variables))
    try:
        yield
    finally:
        _global_logic_variables.clear()
        _global_logic_variables.update(old_global_logic_variables)
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

from .core import (  # type: ignore[attr-defined]
    _reify as core_reify,
    _unify as core_unify,
    reify,
    unify,
)
from .dispatch import dispatch


if TYPE_CHECKING:
    from .variable import Var


__all__ = ["unifiable", "reify_object", "unify_object"]


def unifiable(cls: type) -> type:
    """Register standard unify and reify operations on class
    This uses the type and __dict__ or __slots__ attributes to define the
    nature of the term
    See Also:
    >>> # xdoctest: +SKIP
    >>> class A(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> unifiable(A)
    <class 'unification.more.A'>
    >>> x = var("x")
    >>> a = A(1, 2)
    >>> b = A(1, x)
    >>> unify(a, b, {})
    {~x: 2}
    """
    core_unify.add((cls, cls, dict), unify_object)  # type: ignore[attr-defined]
    core_reify.add((cls, dict), reify_object)  # type: ignore[attr-defined]

    return cls


#########
# Reify #
#########


def reify_object(o: object, s: dict[Var, object]) -> object:
    """Reify a Python object with a substitution
    >>> # xdoctest: +SKIP
    >>> class Foo(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...
    ...     def __str__(self):
    ...         return "Foo(%s, %s)" % (str(self.a), str(self.b))
    >>> x = var("x")
    >>> f = Foo(1, x)
    >>> print(f)
    Foo(1, ~x)
    >>> print(reify_object(f, {x: 2}))
    Foo(1, 2)
    """
    if hasattr(o, "__slots__"):
        return _reify_object_slots(o, s)
    else:
        return _reify_object_dict(o, s)


def _reify_object_dict(o: object, s: dict[Var, object]) -> object:
    obj = object.__new__(type(o))
    d = reify(o.__dict__, s)  # pyrefly: ignore[missing-attribute]
    if d == o.__dict__:  # pyrefly: ignore[missing-attribute]
        return o
    obj.__dict__.update(d)  # pyrefly: ignore[missing-attribute, no-matching-overload]
    return obj


def _reify_object_slots(o: object, s: dict[Var, object]) -> object:
    attrs = [
        getattr(o, attr)
        for attr in o.__slots__  # pyrefly: ignore[missing-attribute]
    ]
    new_attrs = reify(attrs, s)
    if attrs == new_attrs:
        return o
    else:
        newobj = object.__new__(type(o))
        for slot, attr in zip(
            o.__slots__,  # pyrefly: ignore[missing-attribute]
            new_attrs,  # pyrefly: ignore[bad-argument-type]
        ):
            setattr(newobj, slot, attr)
        return newobj


@dispatch(slice, dict)
def _reify(o: slice, s: dict[Var, object]) -> slice:
    """Reify a Python ``slice`` object"""

    return slice(*reify((o.start, o.stop, o.step), s))  # pyrefly: ignore[not-iterable]


#########
# Unify #
#########


def unify_object(
    u: object, v: object, s: dict[Var, object]
) -> dict[Var, object] | bool:
    """Unify two Python objects
    Unifies their type and ``__dict__`` attributes
    >>> # xdoctest: +SKIP
    >>> class Foo(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...
    ...     def __str__(self):
    ...         return "Foo(%s, %s)" % (str(self.a), str(self.b))
    >>> x = var("x")
    >>> f = Foo(1, x)
    >>> g = Foo(1, 2)
    >>> unify_object(f, g, {})
    {~x: 2}
    """
    if type(u) is not type(v):
        return False
    if hasattr(u, "__slots__"):
        return unify(
            [
                getattr(u, slot)
                for slot in u.__slots__  # pyrefly: ignore[missing-attribute]
            ],
            [
                getattr(v, slot)
                for slot in v.__slots__  # pyrefly: ignore[missing-attribute]
            ],
            s,
        )
    else:
        return unify(
            u.__dict__,  # pyrefly: ignore[missing-attribute]
            v.__dict__,  # pyrefly: ignore[missing-attribute]
            s,
        )


@dispatch(slice, slice, dict)
def _unify(u: slice, v: slice, s: dict[Var, object]) -> dict[Var, object] | bool:
    """Unify a Python ``slice`` object"""
    return unify((u.start, u.stop, u.step), (v.start, v.stop, v.step), s)
<|endoftext|>from __future__ import annotations

import operator
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

from .utils import _toposort, groupby
from .variadic import isvariadic


__all__ = [
    "AmbiguityWarning",
    "supercedes",
    "consistent",
    "ambiguous",
    "ambiguities",
    "super_signature",
    "edge",
    "ordering",
]


class AmbiguityWarning(Warning):
    pass


def supercedes(a: tuple[type, ...], b: tuple[type, ...]) -> bool:
    """A is consistent and strictly more specific than B"""
    if len(a) < len(b):
        # only case is if a is empty and b is variadic
        return not a and len(b) == 1 and isvariadic(b[-1])
    elif len(a) == len(b):
        return all(map(issubclass, a, b))
    else:
        # len(a) > len(b)
        p1 = 0
        p2 = 0
        while p1 < len(a) and p2 < len(b):
            cur_a = a[p1]
            cur_b = b[p2]
            if not (isvariadic(cur_a) or isvariadic(cur_b)):
                if not issubclass(cur_a, cur_b):
                    return False
                p1 += 1
                p2 += 1
            elif isvariadic(cur_a):
                if p1 != len(a) - 1:
                    raise AssertionError(
                        f"Expected p1={p1} to equal len(a)-1={len(a) - 1}"
                    )
                return p2 == len(b) - 1 and issubclass(cur_a, cur_b)
            elif isvariadic(cur_b):
                if p2 != len(b) - 1:
                    raise AssertionError(
                        f"Expected p2={p2} to equal len(b)-1={len(b) - 1}"
                    )
                if not issubclass(cur_a, cur_b):
                    return False
                p1 += 1
        return p2 == len(b) - 1 and p1 == len(a)


def consistent(a: tuple[type, ...], b: tuple[type, ...]) -> bool:
    """It is possible for an argument list to satisfy both A and B"""

    # Need to check for empty args
    if not a:
        return not b or isvariadic(b[0])
    if not b:
        return not a or isvariadic(a[0])

    # Non-empty args check for mutual subclasses
    if len(a) == len(b):
        return all(issubclass(aa, bb) or issubclass(bb, aa) for aa, bb in zip(a, b))
    else:
        p1 = 0
        p2 = 0
        while p1 < len(a) and p2 < len(b):
            cur_a = a[p1]
            cur_b = b[p2]
            if not issubclass(cur_b, cur_a) and not issubclass(cur_a, cur_b):
                return False
            if not (isvariadic(cur_a) or isvariadic(cur_b)):
                p1 += 1
                p2 += 1
            elif isvariadic(cur_a):
                p2 += 1
            elif isvariadic(cur_b):
                p1 += 1
        # We only need to check for variadic ends
        # Variadic types are guaranteed to be the last element
        return (
            isvariadic(cur_a)  # type: ignore[possibly-undefined]
            and p2 == len(b)
            or isvariadic(cur_b)  # type: ignore[possibly-undefined]
            and p1 == len(a)
        )


def ambiguous(a: tuple[type, ...], b: tuple[type, ...]) -> bool:
    """A is consistent with B but neither is strictly more specific"""
    return consistent(a, b) and not (supercedes(a, b) or supercedes(b, a))


def ambiguities(
    signatures: Iterable[tuple[type, ...]],
) -> set[tuple[tuple[type, ...], tuple[type, ...]]]:
    """All signature pairs such that A is ambiguous with B"""
    signatures = list(map(tuple, signatures))
    return {
        (a, b)
        for a in signatures
        for b in signatures
        if hash(a) < hash(b)
        and ambiguous(a, b)
        and not any(supercedes(c, a) and supercedes(c, b) for c in signatures)
    }


def super_signature(signatures: Sequence[tuple[type, ...]]) -> list[type]:
    """A signature that would break ambiguities"""
    n = len(signatures[0])
    if not all(len(s) == n for s in signatures):
        raise AssertionError("All signatures must have the same length")

    return [max((type.mro(sig[i]) for sig in signatures), key=len)[0] for i in range(n)]


def edge(
    a: tuple[type, ...],
    b: tuple[type, ...],
    tie_breaker: Callable[[tuple[type, ...]], int] = hash,
) -> bool:
    """A should be checked before B
    Tie broken by tie_breaker, defaults to ``hash``
    """
    # A either supersedes B and B does not supersede A or if B does then call
    # tie_breaker
    return supercedes(a, b) and (
        not supercedes(b, a) or tie_breaker(a) > tie_breaker(b)
    )


def ordering(signatures: Iterable[tuple[type, ...]]) -> list[tuple[type, ...]]:
    """A sane ordering of signatures to check, first to last
    Topological sort of edges as given by ``edge`` and ``supercedes``
    """
    signatures = list(map(tuple, signatures))
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    edges = groupby(operator.itemgetter(0), edges)
    for s in signatures:
        if s not in edges:
            edges[s] = []
    topo_edges: dict[
        tuple[type, ...], list[tuple[type, ...]]
    ] = {  # pyrefly: ignore[bad-assignment]
        k: [b for a, b in v]
        for k, v in edges.items()  # type: ignore[attr-defined]
    }
    return _toposort(topo_edges)
<|endoftext|>from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

_T = TypeVar("_T")


__all__ = ["raises", "expand_tuples", "reverse_dict", "groupby", "typename"]


def raises(
    err: type[BaseException],
    lamda: Callable[[], object],  # codespell:ignore lamda
) -> bool:
    try:
        lamda()  # codespell:ignore lamda
        return False
    except err:
        return True


def expand_tuples(L: Sequence[type | tuple[type, ...]]) -> list[tuple[type, ...]]:
    """
    >>> expand_tuples([1, (2, 3)])
    [(1, 2), (1, 3)]
    >>> expand_tuples([1, 2])
    [(1, 2)]
    """
    if not L:
        return [()]
    elif not isinstance(L[0], tuple):
        rest = expand_tuples(L[1:])
        return [(L[0],) + t for t in rest]
    else:
        rest = expand_tuples(L[1:])
        return [(item,) + t for t in rest for item in L[0]]


# Taken from theano/theano/gof/sched.py
# Avoids licensing issues because this was written by Matthew Rocklin
def _toposort(edges: Mapping[_T, Iterable[_T]]) -> list[_T]:
    """Topological sort algorithm by Kahn [1] - O(nodes + vertices)
    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges
    >>> _toposort({1: (2, 3), 2: (3,)})
    [1, 2, 3]
    >>> # Closely follows the wikipedia page [2]
    >>> # [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    >>> # Communications of the ACM
    >>> # [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = OrderedDict((k, set(val)) for k, val in incoming_edges.items())
    S = OrderedDict.fromkeys(v for v in edges if v not in incoming_edges)
    L = []

    while S:
        n, _ = S.popitem()
        L.append(n)
        for m in edges.get(n, ()):
            if n not in incoming_edges[m]:
                raise AssertionError(f"Expected {n} in incoming_edges[{m}]")
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S[m] = None
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L


def reverse_dict(
    d: Mapping[_T, Iterable[_T]],
) -> OrderedDict[_T, tuple[_T, ...]]:
    """Reverses direction of dependence dict.

    >>> d = {"a": (1, 2), "b": (2, 3), "c": ()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}

    .. note::
        dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.
    """
    result = OrderedDict()  # type: ignore[var-annotated]
    for key in d:
        for val in d[key]:
            # pyrefly: ignore [unsupported-operation]
            result[val] = result.get(val, ()) + (key,)
    return result  # pyrefly: ignore[bad-return]


# Taken from toolz
# Avoids licensing issues because this version was authored by Matthew Rocklin
def groupby(
    func: Callable[[_T], object], seq: Iterable[_T]
) -> OrderedDict[object, list[_T]]:
    """Group a collection by a key function
    >>> names = ["Alice", "Bob", "Charlie", "Dan", "Edith", "Frank"]
    >>> groupby(len, names)  # doctest: +SKIP
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}
    >>> iseven = lambda x: x % 2 == 0
    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}
    See Also:
        ``countby``
    """

    d = OrderedDict()  # type: ignore[var-annotated]
    for item in seq:
        key = func(item)
        if key not in d:
            d[key] = []
        d[key].append(item)
    return d


def typename(type: type | tuple[type, ...]) -> str:
    """Get the name of `type`.
    Parameters
    ----------
    type : Union[Type, Tuple[Type]]
    Returns
    -------
    str
        The name of `type` or a tuple of the names of the types in `type`.
    Examples
    --------
    >>> typename(int)
    'int'
    >>> typename((int, float))
    '(int, float)'
    """
    try:
        return type.__name__  # pyrefly: ignore[missing-attribute]
    except AttributeError:
        if len(type) == 1:
            return typename(*type)
        return f"({', '.join(map(typename, type))})"
<|endoftext|>from __future__ import annotations

from .utils import typename


__all__ = ["VariadicSignatureType", "isvariadic", "VariadicSignatureMeta", "Variadic"]


class VariadicSignatureType(type):
    # checking if subclass is a subclass of self
    def __subclasscheck__(cls, subclass: type) -> bool:
        other_type = subclass.variadic_type if isvariadic(subclass) else (subclass,)
        return subclass is cls or all(
            issubclass(other, cls.variadic_type)  # type: ignore[attr-defined]
            for other in other_type
        )

    def __eq__(cls, other: object) -> bool:
        """
        Return True if other has the same variadic type
        Parameters
        ----------
        other : object (type)
            The object (type) to check
        Returns
        -------
        bool
            Whether or not `other` is equal to `self`
        """
        return isvariadic(other) and set(cls.variadic_type) == set(other.variadic_type)  # type: ignore[attr-defined]

    def __hash__(cls) -> int:
        return hash((type(cls), frozenset(cls.variadic_type)))  # type: ignore[attr-defined]


def isvariadic(obj: type) -> bool:
    """Check whether the type `obj` is variadic.
    Parameters
    ----------
    obj : type
        The type to check
    Returns
    -------
    bool
        Whether or not `obj` is variadic
    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> isvariadic(int)
    False
    >>> isvariadic(Variadic[int])
    True
    """
    return isinstance(obj, VariadicSignatureType)


class VariadicSignatureMeta(type):
    """A metaclass that overrides ``__getitem__`` on the class. This is used to
    generate a new type for Variadic signatures. See the Variadic class for
    examples of how this behaves.
    """

    def __getitem__(
        cls, variadic_type: type | tuple[type, ...]
    ) -> VariadicSignatureType:
        if not (isinstance(variadic_type, (type, tuple)) or type(variadic_type)):
            raise ValueError(
                "Variadic types must be type or tuple of types"
                " (Variadic[int] or Variadic[(int, float)]"
            )

        if not isinstance(variadic_type, tuple):
            variadic_type = (variadic_type,)
        return VariadicSignatureType(
            f"Variadic[{typename(variadic_type)}]",
            (),
            dict(variadic_type=variadic_type, __slots__=()),
        )


class Variadic(metaclass=VariadicSignatureMeta):
    """A class whose getitem method can be used to generate a new type
    representing a specific variadic signature.
    Examples
    --------
    >>> # xdoctest: +SKIP
    >>> Variadic[int]  # any number of int arguments
    <class 'multipledispatch.variadic.Variadic[int]'>
    >>> Variadic[(int, str)]  # any number of one of int or str arguments
    <class 'multipledispatch.variadic.Variadic[(int, str)]'>
    >>> issubclass(int, Variadic[int])
    True
    >>> issubclass(int, Variadic[(int, str)])
    True
    >>> issubclass(str, Variadic[(int, str)])
    True
    >>> issubclass(float, Variadic[(int, str)])
    False
    """
<|endoftext|>from __future__ import annotations

import inspect
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import TypeVarTuple, Unpack


if TYPE_CHECKING:
    from collections.abc import Callable

from .dispatcher import Dispatcher, MethodDispatcher


global_namespace: dict[str, Dispatcher] = {}

__all__ = ["dispatch", "ismethod"]

_T = TypeVar("_T")
_Ts = TypeVarTuple("_Ts")


def dispatch(
    *types: Unpack[_Ts], **kwargs: Any
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Dispatch function on the types of the inputs
    Supports dispatch on all non-keyword arguments.
    Collects implementations based on the function name.  Ignores namespaces.
    If ambiguous type signatures occur a warning is raised when the function is
    defined suggesting the additional method to break the ambiguity.

    Example:
        >>> # xdoctest: +SKIP
        >>> @dispatch(int)
        ... def f(x):
        ...     return x + 1
        >>> @dispatch(float)
        ... def f(x):
        ...     return x - 1
        >>> # xdoctest: +SKIP
        >>> f(3)
        4
        >>> f(3.0)
        2.0
        >>> # Specify an isolated namespace with the namespace keyword argument
        >>> my_namespace = {}
        >>> @dispatch(int, namespace=my_namespace)
        ... def foo(x):
        ...     return x + 1
        >>> # Dispatch on instance methods within classes
        >>> class MyClass(object):
        ...     @dispatch(list)
        ...     def __init__(self, data):
        ...         self.data = data
        ...
        ...     @dispatch(int)
        ...     def __init__(self, datum):
        ...         self.data = [datum]
        >>> MyClass([1, 2, 3]).data
        [1, 2, 3]
        >>> MyClass(3).data
        [3]
    """
    namespace = kwargs.get("namespace", global_namespace)

    types_tuple: tuple[type, ...] = tuple(types)  # type: ignore[arg-type]

    def _df(func: Callable[..., _T]) -> Callable[..., _T]:
        name = func.__name__

        if ismethod(func):
            dispatcher = inspect.currentframe().f_back.f_locals.get(  # type: ignore[union-attr]
                name,  # type: ignore[union-attr]
                MethodDispatcher(name),
            )
        else:
            if name not in namespace:
                namespace[name] = Dispatcher(name)
            dispatcher = namespace[name]

        dispatcher.add(types_tuple, func)
        return dispatcher

    return _df  # type: ignore[return-value]


def ismethod(func: Callable[..., object]) -> bool:
    """Is func a method?
    Note that this has to work as the method is defined but before the class is
    defined.  At this stage methods look like functions.
    """
    if hasattr(inspect, "signature"):
        signature = inspect.signature(func)
        return signature.parameters.get("self", None) is not None
    else:
        spec = inspect.getfullargspec(func)  # type: ignore[union-attr, assignment]
        return bool(spec and spec.args and spec.args[0] == "self")
<|endoftext|>from __future__ import annotations

import inspect
import itertools as itl
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import deprecated
from warnings import warn


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

from .conflict import ambiguities, AmbiguityWarning, ordering, super_signature
from .utils import expand_tuples
from .variadic import isvariadic, Variadic


_T = TypeVar("_T")


__all__ = [
    "MDNotImplementedError",
    "ambiguity_warn",
    "halt_ordering",
    "restart_ordering",
    "variadic_signature_matches_iter",
    "variadic_signature_matches",
    "Dispatcher",
    "source",
    "MethodDispatcher",
    "str_signature",
    "warning_text",
]


class MDNotImplementedError(NotImplementedError):
    """A NotImplementedError for multiple dispatch"""


def ambiguity_warn(
    dispatcher: Dispatcher, ambiguities: set[tuple[tuple[type, ...], tuple[type, ...]]]
) -> None:
    """Raise warning when ambiguity is detected.

    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher

    See Also
    --------
    Dispatcher.add
    warning_text
    """
    warn(warning_text(dispatcher.name, ambiguities), AmbiguityWarning)


@deprecated(
    "`halt_ordering` is deprecated, you can safely remove this call.",
    category=FutureWarning,
)
def halt_ordering() -> None:
    """Deprecated interface to temporarily disable ordering."""


@deprecated(
    "`restart_ordering` is deprecated, if you would like to eagerly order the dispatchers, "
    "you should call the `reorder()` method on each dispatcher.",
    category=FutureWarning,
)
def restart_ordering(on_ambiguity: Callable[..., None] = ambiguity_warn) -> None:
    """Deprecated interface to temporarily resume ordering."""


def variadic_signature_matches_iter(
    types: tuple[type, ...], full_signature: tuple[type, ...]
) -> Generator[bool, None, None]:
    """Check if a set of input types matches a variadic signature.

    Notes
    -----
    The algorithm is as follows:

    Initialize the current signature to the first in the sequence.
    For each type in ``types``:

    - If the current signature is variadic

      - If the type matches the signature, yield True
      - Else, try to get the next signature.
        If no signatures are left we can't possibly have a match,
        so yield False.

    - Else, yield True if the type matches the current signature.
      Get the next signature.
    """
    sigiter = iter(full_signature)
    sig = next(sigiter)
    for typ in types:
        matches = issubclass(typ, sig)
        yield matches
        if not isvariadic(sig):
            # we're not matching a variadic argument, so move to the next
            # element in the signature
            sig = next(sigiter)
    else:
        try:
            sig = next(sigiter)
        except StopIteration:
            if not isvariadic(sig):
                raise AssertionError("Expected variadic signature") from None
            yield True
        else:
            # We have signature items left over, so all of our arguments
            # haven't matched
            yield False


def variadic_signature_matches(
    types: tuple[type, ...], full_signature: tuple[type, ...]
) -> bool:
    # No arguments always matches a variadic signature
    if not full_signature:
        raise AssertionError("full_signature is empty")
    return all(variadic_signature_matches_iter(types, full_signature))


class Dispatcher:
    """Dispatch methods based on type signature
    Use ``dispatch`` to add implementations
    Examples
    --------
    >>> # xdoctest: +SKIP("bad import name")
    >>> from multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1
    >>> @dispatch(float)
    ... def f(x):
    ...     return x - 1
    >>> f(3)
    4
    >>> f(3.0)
    2.0
    """

    __slots__ = "__name__", "name", "funcs", "_ordering", "_cache", "doc"

    def __init__(self, name: str, doc: str | None = None) -> None:
        self.name = self.__name__ = name
        self.funcs: dict[tuple[type, ...], Callable[..., object]] = {}
        self.doc = doc

        self._cache: dict[tuple[type, ...], Callable[..., object]] = {}

    def register(
        self, *types: type, **kwargs: object
    ) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
        """register dispatcher with new implementation
        >>> # xdoctest: +SKIP
        >>> f = Dispatcher("f")
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1
        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1
        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]
        >>> f(1)
        2
        >>> f(1.0)
        0.0
        >>> f([1, 2, 3])
        [3, 2, 1]
        """

        def _df(func: Callable[..., _T]) -> Callable[..., _T]:
            self.add(types, func, **kwargs)  # type: ignore[call-arg]
            return func

        return _df

    @classmethod
    def get_func_params(
        cls, func: Callable[..., object]
    ) -> Iterable[inspect.Parameter] | None:
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return sig.parameters.values()

    @classmethod
    def get_func_annotations(
        cls, func: Callable[..., object]
    ) -> tuple[type, ...] | None:
        """get annotations of function positional parameters"""
        params = cls.get_func_params(func)
        if params:
            Parameter = inspect.Parameter

            params = (
                param
                for param in params
                if param.kind
                in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            )

            annotations = tuple(param.annotation for param in params)

            if all(ann is not Parameter.empty for ann in annotations):
                return annotations

    def add(self, signature: tuple[type, ...], func: Callable[..., object]) -> None:
        """Add new types/method pair to dispatcher
        >>> # xdoctest: +SKIP
        >>> D = Dispatcher("add")
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)
        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>
        >>> # When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        >>> # with a dispatcher/itself, and a set of ambiguous type signature pairs
        >>> # as inputs.  See ``ambiguity_warn`` for an example.
        """
        # Handle annotations
        if not signature:
            annotations = self.get_func_annotations(func)
            if annotations:
                signature = annotations

        # Handle union types
        if any(isinstance(typ, tuple) for typ in signature):
            for typs in expand_tuples(signature):
                self.add(typs, func)
            return

        new_signature = []

        for index, typ in enumerate(signature, start=1):
            if not isinstance(typ, (type, list)):
                str_sig = ", ".join(
                    c.__name__ if isinstance(c, type) else str(c) for c in signature
                )
                raise TypeError(
                    f"Tried to dispatch on non-type: {typ}\n"
                    f"In signature: <{str_sig}>\n"
                    f"In function: {self.name}"
                )

            # handle variadic signatures
            if isinstance(typ, list):
                if index != len(signature):
                    raise TypeError("Variadic signature must be the last element")

                if len(typ) != 1:
                    raise TypeError(
                        "Variadic signature must contain exactly one element. "
                        "To use a variadic union type place the desired types "
                        "inside of a tuple, e.g., [(int, str)]"
                    )
                # pyrefly: ignore [bad-specialization]
                new_signature.append(Variadic[typ[0]])
            else:
                new_signature.append(typ)  # pyrefly: ignore[bad-argument-type]

        self.funcs[tuple(new_signature)] = func
        self._cache.clear()

        try:
            del self._ordering
        except AttributeError:
            pass

    @property
    def ordering(self) -> list[tuple[type, ...]]:
        try:
            return self._ordering
        except AttributeError:
            return self.reorder()

    def reorder(
        self, on_ambiguity: Callable[..., None] = ambiguity_warn
    ) -> list[tuple[type, ...]]:
        self._ordering = od = ordering(self.funcs)
        amb = ambiguities(self.funcs)
        if amb:
            on_ambiguity(self, amb)
        return od

    def __call__(self, *args: object, **kwargs: object) -> object:
        types = tuple(type(arg) for arg in args)
        try:
            func = self._cache[types]
        except KeyError as e:
            func = self.dispatch(*types)
            if not func:
                raise NotImplementedError(
                    f"Could not find signature for {self.name}: <{str_signature(types)}>"
                ) from e
            self._cache[types] = func
        try:
            return func(*args, **kwargs)

        except MDNotImplementedError as e:
            funcs = self.dispatch_iter(*types)
            next(funcs)  # burn first
            for func in funcs:
                try:
                    return func(*args, **kwargs)
                except MDNotImplementedError:
                    pass

            raise NotImplementedError(
                "Matching functions for "
                f"{self.name}: <{str_signature(types)}> found, but none completed successfully",
            ) from e

    def __str__(self) -> str:
        return f"<dispatched {self.name}>"

    __repr__ = __str__

    def dispatch(self, *types: type) -> Callable[..., object] | None:
        """Determine appropriate implementation for this type signature
        This method is internal.  Users should call this object as a function.
        Implementation resolution occurs within the ``__call__`` method.
        >>> # xdoctest: +SKIP
        >>> from multipledispatch import dispatch
        >>> @dispatch(int)
        ... def inc(x):
        ...     return x + 1
        >>> implementation = inc.dispatch(int)
        >>> implementation(3)
        4
        >>> print(inc.dispatch(float))
        None
        See Also:
          ``multipledispatch.conflict`` - module to determine resolution order
        """

        if types in self.funcs:
            return self.funcs[types]

        try:
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    def dispatch_iter(
        self, *types: type
    ) -> Generator[Callable[..., object], None, None]:
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):
                result = self.funcs[signature]
                yield result
            elif len(signature) and isvariadic(signature[-1]):
                if variadic_signature_matches(types, signature):
                    result = self.funcs[signature]
                    yield result

    @deprecated(
        "`resolve()` is deprecated, use `dispatch(*types)`", category=FutureWarning
    )
    def resolve(self, types: tuple[type, ...]) -> Callable[..., object] | None:
        """Determine appropriate implementation for this type signature
        .. deprecated:: 0.4.4
            Use ``dispatch(*types)`` instead
        """
        return self.dispatch(*types)

    def __getstate__(self) -> dict[str, Any]:
        return {"name": self.name, "funcs": self.funcs}

    def __setstate__(self, d: dict[str, Any]) -> None:
        self.name = d["name"]
        self.funcs = d["funcs"]
        self._ordering = ordering(self.funcs)
        self._cache = {}

    @property
    def __doc__(self):  # type: ignore[override]
        docs = [f"Multiply dispatched method: {self.name}"]

        if self.doc:
            docs.append(self.doc)

        other: list[str] = []
        for sig in self.ordering[::-1]:
            func = self.funcs[sig]
            if func.__doc__:
                s = f"Inputs: <{str_signature(sig)}>\n"
                s += "-" * len(s) + "\n"
                s += func.__doc__.strip()
                docs.append(s)
            else:
                other.append(str_signature(sig))

        if other:
            docs.append("Other signatures:\n    " + "\n    ".join(other))

        return "\n\n".join(docs)

    def _help(self, *args: object) -> str | None:
        return self.dispatch(*map(type, args)).__doc__

    def help(self, *args: object, **kwargs: object) -> None:
        """Print docstring for the function corresponding to inputs"""
        print(self._help(*args))

    def _source(self, *args: object) -> str:
        func = self.dispatch(*map(type, args))
        if not func:
            raise TypeError("No function found")
        return source(func)

    def source(self, *args: object, **kwargs: object) -> None:
        """Print source code for the function corresponding to inputs"""
        print(self._source(*args))


def source(func: Callable[..., object]) -> str:
    s = f"File: {inspect.getsourcefile(func)}\n\n"
    s = s + inspect.getsource(func)
    return s


class MethodDispatcher(Dispatcher):
    """Dispatch methods based on type signature
    See Also:
        Dispatcher
    """

    __slots__ = ("obj", "cls")

    @classmethod
    def get_func_params(
        cls, func: Callable[..., object]
    ) -> Iterator[inspect.Parameter] | None:
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return itl.islice(sig.parameters.values(), 1, None)

    def __get__(self, instance: object | None, owner: type) -> MethodDispatcher:
        self.obj = instance
        self.cls = owner
        return self

    def __call__(self, *args: object, **kwargs: object) -> object:
        types = tuple(type(arg) for arg in args)
        func = self.dispatch(*types)
        if not func:
            raise NotImplementedError(
                f"Could not find signature for {self.name}: <{str_signature(types)}>"
            )
        return func(self.obj, *args, **kwargs)


def str_signature(sig: Iterable[type]) -> str:
    """String representation of type signature
    >>> str_signature((int, float))
    'int, float'
    """
    return ", ".join(cls.__name__ for cls in sig)


def warning_text(name: str, amb: set[tuple[tuple[type, ...], tuple[type, ...]]]) -> str:
    """The text for ambiguity warnings"""
    text = f"\nAmbiguities exist in dispatched function {name}\n\n"
    text += "The following signatures may result in ambiguous behavior:\n"
    for pair in amb:
        text += "\t" + ", ".join("[" + str_signature(s) + "]" for s in pair) + "\n"
    text += "\n\nConsider making the following additions:\n\n"
    text += "\n\n".join(
        [
            "@dispatch(" + str_signature(super_signature(s)) + f")\ndef {name}(...)"
            for s in amb
        ]
    )
    return text
<|endoftext|>import torch
from torch.export import ExportedProgram


class LoweredBackendModule(torch.nn.Module):
    def __init__(
        self,
        original_exported_program: ExportedProgram,
        backend_id: str,
        *,
        module_name: str | None = None,
    ) -> None:
        super().__init__()
        self._backend_id = backend_id
        self._module_name = module_name
        self._original_exported_program = original_exported_program

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def module_name(self) -> str | None:
        return self._module_name

    @property
    def original_module(self) -> ExportedProgram:
        return self._original_exported_program

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return torch._higher_order_ops.executorch_call_delegate(self, *args, **kwargs)
<|endoftext|>import types

import torch
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.export.pt2_archive._package import AOTI_FILES, package_pt2
from torch.types import FileLike

from ._lowered_aoti_module import LoweredBackendModule


def get_new_ep_with_flat_inputs_outputs(ep: ExportedProgram) -> ExportedProgram:
    class FlattenedModule(torch.nn.Module):
        def __init__(
            self,
            original_module: torch.fx.GraphModule,
            in_spec: pytree.TreeSpec,
            out_spec: pytree.TreeSpec,
        ) -> None:
            super().__init__()
            self.original_module = original_module
            self.in_spec = in_spec
            self.out_spec = out_spec

        def forward(self, *flat_inputs):  # type: ignore[no-untyped-def]
            # Unflatten inputs to original structure
            inputs = pytree.tree_unflatten(flat_inputs, self.in_spec)
            args, kwargs = inputs
            outputs = self.original_module(*args, **kwargs)
            # Flatten outputs
            flat_outputs, _ = pytree.tree_flatten(outputs)
            return tuple(flat_outputs)

    flattened_module = FlattenedModule(
        ep.module(), ep.call_spec.in_spec, ep.call_spec.out_spec
    )
    args, kwargs = ep.example_inputs
    flat_inputs, _ = pytree.tree_flatten((args, kwargs))
    flat_ep = torch.export.export(flattened_module, tuple(flat_inputs))

    return flat_ep


def lower_exported_program(
    exported_program: ExportedProgram, model_name: str, backend_id: str
) -> tuple[ExportedProgram, AOTI_FILES]:
    """
    Lower an exported program to AOTInductor and return a delegate ExportedProgram
    with the `executorch_call_delegate` HOP
    """
    args, kwargs = exported_program.example_inputs
    out_spec = exported_program.call_spec.out_spec
    flat_ep = get_new_ep_with_flat_inputs_outputs(exported_program)
    flat_inputs, _ = pytree.tree_flatten((args, kwargs))

    aoti_files = torch._inductor.aot_compile(
        flat_ep.module(), tuple(flat_inputs), options={"aot_inductor.package": True}
    )
    if not isinstance(aoti_files, list):
        raise AssertionError(
            f"aoti_files must be a list, got {type(aoti_files).__name__}"
        )

    lowered_aoti_module = LoweredBackendModule(
        flat_ep, backend_id, module_name=model_name
    )

    def patched_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        flat_inputs, _ = pytree.tree_flatten((args, kwargs))
        flat_outputs = torch._higher_order_ops.executorch_call_delegate(
            self, *flat_inputs
        )
        if out_spec is not None and flat_outputs is not None:
            return pytree.tree_unflatten(flat_outputs, out_spec)
        else:
            return flat_outputs

    lowered_aoti_module.forward = types.MethodType(patched_forward, lowered_aoti_module)  # type: ignore[method-assign]

    aoti_delegate_ep = torch.export.export(lowered_aoti_module, args, kwargs)

    return aoti_delegate_ep, aoti_files


def package_nativert_with_aoti_delegate(
    f: FileLike,
    model_name: str,
    backend_id: str,
    original_ep: ExportedProgram,
    delegate_ep: ExportedProgram,
    delegate_files: AOTI_FILES,
) -> None:
    """
    Package a pt2 archive file that can be consumed by NativeRT with AOTI Delegate
    """
    package_pt2(
        f,
        exported_programs={
            model_name: original_ep,
            f"{model_name}-{backend_id}": delegate_ep,
        },
        aoti_files={f"{model_name}-{backend_id}": delegate_files},  # type: ignore[dict-item]
    )
    return
<|endoftext|>import itertools
import unittest.mock
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import TypeVar
from typing_extensions import ParamSpec

import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
from torch._C import DispatchKey


__all__ = ["enable_python_dispatcher", "no_python_dispatcher", "enable_pre_dispatch"]

no_python_dispatcher = torch._C._DisablePythonDispatcher
enable_python_dispatcher = torch._C._EnablePythonDispatcher
enable_pre_dispatch = torch._C._EnablePreDispatch

CROSSREF_FUNCTIONALIZE = False

_P = ParamSpec("_P")
_T = TypeVar("_T")
_R = TypeVar("_R")


def all_py_loaded_overloads() -> Iterator[torch._ops.OpOverload]:
    """
    Warning: the set of overloads this will report is very subtle.  It is precisely
    the set of torch.ops functions that have actually been accessed from Python
    (e.g., we actually called torch.ops.aten.blah at some point.  This is DIFFERENT
    from the set of registered operators, which will in general be a larger set,
    as this would include all operators which we ran C++ static initializers or
    Python operator registration on.  This does not eagerly populate the list on
    torch.ops.aten; this list is lazy!

    In other words, this is good for traversing over everything that has an
    OpOverload object allocated in Python.  We use it for cache invalidation, but
    don't rely on this list being complete.

    Note that even if we did report all C++ registered overloads, this isn't guaranteed
    to be complete either, as a subsequent lazy load of a library which triggers more
    registrations could add more things to the set.
    """
    for ns in torch.ops:
        packets = getattr(torch.ops, ns)
        for op_name in packets:
            packet = getattr(packets, op_name)
            for overload in packet:
                yield getattr(packet, overload)


@contextmanager
def suspend_functionalization() -> Generator[None, None, None]:
    f_tls = torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.Functionalize
    )
    f_rv = torch._C._functionalization_reapply_views_tls()
    if f_tls:
        torch._disable_functionalization()
    try:
        yield
    finally:
        if f_tls:
            torch._enable_functionalization(reapply_views=f_rv)


def check_tensor_metadata_matches(
    nv: torch.Tensor, rv: torch.Tensor, desc: Callable[[], str]
) -> None:
    if not callable(desc):
        raise AssertionError(f"desc must be callable, got {type(desc)}")
    if nv.size() != rv.size():
        raise AssertionError(f"{desc()}: sizes {nv.size()} != {rv.size()}")
    if nv.dtype != rv.dtype:
        raise AssertionError(f"{desc()}: dtype {nv.dtype} != {rv.dtype}")
    same_strides, idx = torch._prims_common.check_significant_strides(
        nv, rv, only_cuda=False
    )
    if not same_strides:
        raise AssertionError(
            f"{desc()}: strides {nv.stride()} != {rv.stride()} (mismatch at index {idx})"
        )


def check_metadata_matches(n: object, r: object, desc: Callable[[], str]) -> None:
    if not callable(desc):
        raise AssertionError(f"desc must be callable, got {type(desc)}")
    n_vals, _n_spec = pytree.tree_flatten(n)
    r_vals, _r_spec = pytree.tree_flatten(r)
    # TODO: test the specs match; empirically  sometimes we have a tuple
    # on one side and a list on the other
    if len(n_vals) != len(r_vals):
        raise AssertionError(f"{len(n_vals)} != {len(r_vals)}")
    for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
        if not isinstance(rv, torch.Tensor):
            continue
        check_tensor_metadata_matches(nv, rv, lambda: f"{desc()} output {i}")


class Lit:
    def __init__(self, s: str) -> None:
        self.s = s

    def __repr__(self) -> str:
        return self.s


def _fmt(a: object) -> object:
    if isinstance(a, torch.Tensor):
        return Lit(
            f"torch.empty_strided({tuple(a.size())}, {a.stride()}, dtype={a.dtype})"
        )
    else:
        return a


def make_crossref_functionalize(
    op: torch._ops.OpOverload[_P, _T], final_key: DispatchKey
) -> Callable[_P, _T] | DispatchKey:
    from torch._subclasses.fake_tensor import FakeTensorMode

    # This case is pretty weird, suppress it for now
    if op is torch.ops.aten.lift_fresh.default:
        return final_key

    def handler(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        fake_mode = FakeTensorMode()

        def fakeify_defun(t: _R) -> _R | torch._subclasses.fake_tensor.FakeTensor:
            if isinstance(t, torch.Tensor):
                if torch._is_functional_tensor(t):
                    r = torch._from_functional_tensor(t)
                    # NB: This assumes that the inner tensor sizes/strides match
                    # the outer tensor sizes/strides.  This doesn't necessarily have to
                    # be the case, see discussion at
                    # https://github.com/pytorch/pytorch/pull/87610/files/401ddeda1d769bedc88a12de332c7357b60e51a4#r1007264456
                    if t.size() != r.size():
                        raise AssertionError(f"size mismatch: {t.size()} != {r.size()}")
                    if t.stride() != r.stride():
                        raise AssertionError(
                            f"stride mismatch: {t.stride()} != {r.stride()}"
                        )
                else:
                    r = t
                # TODO: suppress guards
                return fake_mode.from_tensor(r)
            return t

        def maybe_detach(t: _R) -> _R | torch.Tensor:
            if isinstance(t, torch.Tensor):
                return t.detach()
            else:
                return t

        # TODO: This probably does the wrong thing if you're running other
        # substantive modes with the normal op outside here
        with (
            torch.utils._python_dispatch._disable_current_modes(),
            suspend_functionalization(),
        ):
            f_args, f_kwargs = pytree.tree_map(fakeify_defun, (args, kwargs))
            orig_f_args, orig_f_kwargs = pytree.tree_map(
                maybe_detach, (f_args, f_kwargs)
            )
            with fake_mode:
                f_r = op(*f_args, **f_kwargs)  # pyrefly: ignore [invalid-param-spec]
        r = op._op_dk(final_key, *args, **kwargs)

        def desc() -> str:
            fmt_args = ", ".join(
                itertools.chain(
                    (repr(pytree.tree_map(_fmt, a)) for a in orig_f_args),
                    (
                        f"{k}={pytree.tree_map(_fmt, v)}"
                        for k, v in orig_f_kwargs.items()
                    ),
                )
            )
            return f"{op}({fmt_args})"

        check_metadata_matches(f_r, r, desc)
        return r

    return handler


# NB: enabling this is slow, don't do it in a hot loop.  This is purely
# for debugging purposes.
@contextmanager
def enable_crossref_functionalize() -> Generator[None, None, None]:
    for op in all_py_loaded_overloads():
        op._uncache_dispatch(torch._C.DispatchKey.Functionalize)
    try:
        with (
            enable_python_dispatcher(),
            unittest.mock.patch("torch._dispatch.python.CROSSREF_FUNCTIONALIZE", True),
        ):
            yield
    finally:
        for op in all_py_loaded_overloads():
            op._uncache_dispatch(torch._C.DispatchKey.Functionalize)
<|endoftext|># mypy: allow-untyped-defs
import math
import warnings
from collections.abc import Callable
from functools import total_ordering

import torch
from torch import inf, Tensor

from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .half_normal import HalfNormal
from .independent import Independent
from .laplace import Laplace
from .lowrank_multivariate_normal import (
    _batch_lowrank_logdet,
    _batch_lowrank_mahalanobis,
    LowRankMultivariateNormal,
)
from .multivariate_normal import _batch_mahalanobis, MultivariateNormal
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .pareto import Pareto
from .poisson import Poisson
from .transformed_distribution import TransformedDistribution
from .uniform import Uniform
from .utils import _sum_rightmost, euler_constant as _euler_gamma


_KL_REGISTRY: dict[
    tuple[type, type], Callable
] = {}  # Source of truth mapping a few general (type, type) pairs to functions.
_KL_MEMOIZE: dict[
    tuple[type, type], Callable
] = {}  # Memoized version mapping many specific (type, type) pairs to functions.

__all__ = ["register_kl", "kl_divergence"]


def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError(
            f"Expected type_p to be a Distribution subclass but got {type_p}"
        )
    if not isinstance(type_q, type) and issubclass(type_q, Distribution):
        raise TypeError(
            f"Expected type_q to be a Distribution subclass but got {type_q}"
        )

    def decorator(fun):
        _KL_REGISTRY[type_p, type_q] = fun
        _KL_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun

    return decorator


@total_ordering
class _Match:
    __slots__ = ["types"]

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def _dispatch_kl(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [
        (super_p, super_q)
        for super_p, super_q in _KL_REGISTRY
        if issubclass(type_p, super_p) and issubclass(type_q, super_q)
    ]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    # mypy isn't smart enough to know that _Match implements __lt__
    # see: https://github.com/python/typing/issues/760#issuecomment-710670503
    left_p, left_q = min(_Match(*m) for m in matches).types  # type: ignore[type-var]
    right_q, right_p = min(_Match(*reversed(m)) for m in matches).types  # type: ignore[type-var]
    left_fun = _KL_REGISTRY[left_p, left_q]
    right_fun = _KL_REGISTRY[right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn(
            f"Ambiguous kl_divergence({type_p.__name__}, {type_q.__name__}). "
            f"Please register_kl({left_p.__name__}, {right_q.__name__})",
            RuntimeWarning,
            stacklevel=2,
        )
    return left_fun


def _infinite_like(tensor):
    """
    Helper function for obtaining infinite KL Divergence throughout
    """
    return torch.full_like(tensor, inf)


def _x_log_x(tensor):
    """
    Utility function for calculating x log x
    """
    return torch.special.xlogy(tensor, tensor)  # produces correct result for x=0


def _batch_trace_XXT(bmat):
    """
    Utility function for calculating the trace of XX^{T} with X having arbitrary trailing batch dimensions
    """
    n = bmat.size(-1)
    m = bmat.size(-2)
    flat_trace = bmat.reshape(-1, m * n).pow(2).sum(-1)
    return flat_trace.reshape(bmat.shape[:-2])


def kl_divergence(p: Distribution, q: Distribution) -> Tensor:
    r"""
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \log\frac {p(x)} {q(x)} \,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    try:
        fun = _KL_MEMOIZE[type(p), type(q)]
    except KeyError:
        fun = _dispatch_kl(type(p), type(q))
        _KL_MEMOIZE[type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError(
            f"No KL(p || q) is implemented for p type {p.__class__.__name__} and q type {q.__class__.__name__}"
        )
    return fun(p, q)


################################################################################
# KL Divergence Implementations
################################################################################

# Same distributions


@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p, q):
    t1 = p.probs * (
        torch.nn.functional.softplus(-q.logits)
        - torch.nn.functional.softplus(-p.logits)
    )
    t1[q.probs == 0] = inf
    t1[p.probs == 0] = 0
    t2 = (1 - p.probs) * (
        torch.nn.functional.softplus(q.logits) - torch.nn.functional.softplus(p.logits)
    )
    t2[q.probs == 1] = inf
    t2[p.probs == 1] = 0
    return t1 + t2


@register_kl(Beta, Beta)
def _kl_beta_beta(p, q):
    sum_params_p = p.concentration1 + p.concentration0
    sum_params_q = q.concentration1 + q.concentration0
    t1 = q.concentration1.lgamma() + q.concentration0.lgamma() + (sum_params_p).lgamma()
    t2 = p.concentration1.lgamma() + p.concentration0.lgamma() + (sum_params_q).lgamma()
    t3 = (p.concentration1 - q.concentration1) * torch.digamma(p.concentration1)
    t4 = (p.concentration0 - q.concentration0) * torch.digamma(p.concentration0)
    t5 = (sum_params_q - sum_params_p) * torch.digamma(sum_params_p)
    return t1 - t2 + t3 + t4 + t5


@register_kl(Binomial, Binomial)
def _kl_binomial_binomial(p, q):
    # from https://math.stackexchange.com/questions/2214993/
    # kullback-leibler-divergence-for-binomial-distributions-p-and-q
    if (p.total_count < q.total_count).any():
        raise NotImplementedError(
            "KL between Binomials where q.total_count > p.total_count is not implemented"
        )
    kl = p.total_count * (
        p.probs * (p.logits - q.logits) + (-p.probs).log1p() - (-q.probs).log1p()
    )
    inf_idxs = p.total_count > q.total_count
    kl[inf_idxs] = _infinite_like(kl[inf_idxs])
    return kl


@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


@register_kl(ContinuousBernoulli, ContinuousBernoulli)
def _kl_continuous_bernoulli_continuous_bernoulli(p, q):
    t1 = p.mean * (p.logits - q.logits)
    t2 = p._cont_bern_log_norm() + torch.log1p(-p.probs)
    t3 = -q._cont_bern_log_norm() - torch.log1p(-q.probs)
    return t1 + t2 + t3


@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p, q):
    # From https://statproofbook.github.io/P/dir-kl.html
    sum_p_concentration = p.concentration.sum(-1)
    sum_q_concentration = q.concentration.sum(-1)
    t1 = sum_p_concentration.lgamma() - sum_q_concentration.lgamma()
    t2 = (p.concentration.lgamma() - q.concentration.lgamma()).sum(-1)
    t3 = p.concentration - q.concentration
    t4 = p.concentration.digamma() - sum_p_concentration.digamma().unsqueeze(-1)
    return t1 - t2 + (t3 * t4).sum(-1)


@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p, q):
    rate_ratio = q.rate / p.rate
    t1 = -rate_ratio.log()
    return t1 + rate_ratio - 1


@register_kl(ExponentialFamily, ExponentialFamily)
def _kl_expfamily_expfamily(p, q):
    if type(p) is not type(q):
        raise NotImplementedError(
            "The cross KL-divergence between different exponential families cannot \
                            be computed using Bregman divergences"
        )
    p_nparams = [np.detach().requires_grad_() for np in p._natural_params]
    q_nparams = q._natural_params
    lg_normal = p._log_normalizer(*p_nparams)
    gradients = torch.autograd.grad(lg_normal.sum(), p_nparams, create_graph=True)
    result = q._log_normalizer(*q_nparams) - lg_normal
    for pnp, qnp, g in zip(p_nparams, q_nparams, gradients):
        term = (qnp - pnp) * g
        result -= _sum_rightmost(term, len(q.event_shape))
    return result


@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p, q):
    t1 = q.concentration * (p.rate / q.rate).log()
    t2 = torch.lgamma(q.concentration) - torch.lgamma(p.concentration)
    t3 = (p.concentration - q.concentration) * torch.digamma(p.concentration)
    t4 = (q.rate - p.rate) * (p.concentration / p.rate)
    return t1 + t2 + t3 + t4


@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p, q):
    ct1 = p.scale / q.scale
    ct2 = q.loc / q.scale
    ct3 = p.loc / q.scale
    t1 = -ct1.log() - ct2 + ct3
    t2 = ct1 * _euler_gamma
    t3 = torch.exp(ct2 + (1 + ct1).lgamma() - ct3)
    return t1 + t2 + t3 - (1 + _euler_gamma)


@register_kl(Geometric, Geometric)
def _kl_geometric_geometric(p, q):
    return -p.entropy() - torch.log1p(-q.probs) / p.probs - q.logits


@register_kl(HalfNormal, HalfNormal)
def _kl_halfnormal_halfnormal(p, q):
    return _kl_normal_normal(p.base_dist, q.base_dist)


@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p, q):
    # From http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    scale_ratio = p.scale / q.scale
    loc_abs_diff = (p.loc - q.loc).abs()
    t1 = -scale_ratio.log()
    t2 = loc_abs_diff / q.scale
    t3 = scale_ratio * torch.exp(-loc_abs_diff / p.scale)
    return t1 + t2 + t3 - 1


@register_kl(LowRankMultivariateNormal, LowRankMultivariateNormal)
def _kl_lowrankmultivariatenormal_lowrankmultivariatenormal(p, q):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two Low Rank Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    term1 = _batch_lowrank_logdet(
        q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q._capacitance_tril
    ) - _batch_lowrank_logdet(
        p._unbroadcasted_cov_factor, p._unbroadcasted_cov_diag, p._capacitance_tril
    )
    term3 = _batch_lowrank_mahalanobis(
        q._unbroadcasted_cov_factor,
        q._unbroadcasted_cov_diag,
        q.loc - p.loc,
        q._capacitance_tril,
    )
    # Expands term2 according to
    # inv(qcov) @ pcov = [inv(qD) - inv(qD) @ qW @ inv(qC) @ qW.T @ inv(qD)] @ (pW @ pW.T + pD)
    #                  = [inv(qD) - A.T @ A] @ (pD + pW @ pW.T)
    qWt_qDinv = q._unbroadcasted_cov_factor.mT / q._unbroadcasted_cov_diag.unsqueeze(-2)
    A = torch.linalg.solve_triangular(q._capacitance_tril, qWt_qDinv, upper=False)
    term21 = (p._unbroadcasted_cov_diag / q._unbroadcasted_cov_diag).sum(-1)
    term22 = _batch_trace_XXT(
        p._unbroadcasted_cov_factor * q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1)
    )
    term23 = _batch_trace_XXT(A * p._unbroadcasted_cov_diag.sqrt().unsqueeze(-2))
    term24 = _batch_trace_XXT(A.matmul(p._unbroadcasted_cov_factor))
    term2 = term21 + term22 - term23 - term24
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(MultivariateNormal, LowRankMultivariateNormal)
def _kl_multivariatenormal_lowrankmultivariatenormal(p, q):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two (Low Rank) Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    term1 = _batch_lowrank_logdet(
        q._unbroadcasted_cov_factor, q._unbroadcasted_cov_diag, q._capacitance_tril
    ) - 2 * p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    term3 = _batch_lowrank_mahalanobis(
        q._unbroadcasted_cov_factor,
        q._unbroadcasted_cov_diag,
        q.loc - p.loc,
        q._capacitance_tril,
    )
    # Expands term2 according to
    # inv(qcov) @ pcov = [inv(qD) - inv(qD) @ qW @ inv(qC) @ qW.T @ inv(qD)] @ p_tril @ p_tril.T
    #                  = [inv(qD) - A.T @ A] @ p_tril @ p_tril.T
    qWt_qDinv = q._unbroadcasted_cov_factor.mT / q._unbroadcasted_cov_diag.unsqueeze(-2)
    A = torch.linalg.solve_triangular(q._capacitance_tril, qWt_qDinv, upper=False)
    term21 = _batch_trace_XXT(
        p._unbroadcasted_scale_tril * q._unbroadcasted_cov_diag.rsqrt().unsqueeze(-1)
    )
    term22 = _batch_trace_XXT(A.matmul(p._unbroadcasted_scale_tril))
    term2 = term21 - term22
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(LowRankMultivariateNormal, MultivariateNormal)
def _kl_lowrankmultivariatenormal_multivariatenormal(p, q):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two (Low Rank) Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    term1 = 2 * q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(
        -1
    ) - _batch_lowrank_logdet(
        p._unbroadcasted_cov_factor, p._unbroadcasted_cov_diag, p._capacitance_tril
    )
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))
    # Expands term2 according to
    # inv(qcov) @ pcov = inv(q_tril @ q_tril.T) @ (pW @ pW.T + pD)
    combined_batch_shape = torch._C._infer_size(
        q._unbroadcasted_scale_tril.shape[:-2], p._unbroadcasted_cov_factor.shape[:-2]
    )
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_cov_factor = p._unbroadcasted_cov_factor.expand(
        combined_batch_shape + (n, p.cov_factor.size(-1))
    )
    p_cov_diag = torch.diag_embed(p._unbroadcasted_cov_diag.sqrt()).expand(
        combined_batch_shape + (n, n)
    )
    term21 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_cov_factor, upper=False)
    )
    term22 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_cov_diag, upper=False)
    )
    term2 = term21 + term22
    return 0.5 * (term1 + term2 + term3 - p.event_shape[0])


@register_kl(MultivariateNormal, MultivariateNormal)
def _kl_multivariatenormal_multivariatenormal(p, q):
    # From https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL-divergence between two Multivariate Normals with\
                          different event shapes cannot be computed"
        )

    half_term1 = q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(
        -1
    ) - p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    combined_batch_shape = torch._C._infer_size(
        q._unbroadcasted_scale_tril.shape[:-2], p._unbroadcasted_scale_tril.shape[:-2]
    )
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_scale_tril = p._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    term2 = _batch_trace_XXT(
        torch.linalg.solve_triangular(q_scale_tril, p_scale_tril, upper=False)
    )
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))
    return half_term1 + 0.5 * (term2 + term3 - n)


@register_kl(Normal, Normal)
def _kl_normal_normal(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


@register_kl(OneHotCategorical, OneHotCategorical)
def _kl_onehotcategorical_onehotcategorical(p, q):
    return _kl_categorical_categorical(p._categorical, q._categorical)


@register_kl(Pareto, Pareto)
def _kl_pareto_pareto(p, q):
    # From http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf
    scale_ratio = p.scale / q.scale
    alpha_ratio = q.alpha / p.alpha
    t1 = q.alpha * scale_ratio.log()
    t2 = -alpha_ratio.log()
    result = t1 + t2 + alpha_ratio - 1
    result[p.support.lower_bound < q.support.lower_bound] = inf
    return result


@register_kl(Poisson, Poisson)
def _kl_poisson_poisson(p, q):
    return p.rate * (p.rate.log() - q.rate.log()) - (p.rate - q.rate)


@register_kl(TransformedDistribution, TransformedDistribution)
def _kl_transformed_transformed(p, q):
    if p.transforms != q.transforms:
        raise NotImplementedError
    if p.event_shape != q.event_shape:
        raise NotImplementedError
    return kl_divergence(p.base_dist, q.base_dist)


@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    result = ((q.high - q.low) / (p.high - p.low)).log()
    result[(q.low > p.low) | (q.high < p.high)] = inf
    return result


# Different distributions
@register_kl(Bernoulli, Poisson)
def _kl_bernoulli_poisson(p, q):
    return -p.entropy() - (p.probs * q.rate.log() - q.rate)


@register_kl(Beta, ContinuousBernoulli)
def _kl_beta_continuous_bernoulli(p, q):
    return (
        -p.entropy()
        - p.mean * q.logits
        - torch.log1p(-q.probs)
        - q._cont_bern_log_norm()
    )


@register_kl(Beta, Pareto)
def _kl_beta_infinity(p, q):
    return _infinite_like(p.concentration1)


@register_kl(Beta, Exponential)
def _kl_beta_exponential(p, q):
    return (
        -p.entropy()
        - q.rate.log()
        + q.rate * (p.concentration1 / (p.concentration1 + p.concentration0))
    )


@register_kl(Beta, Gamma)
def _kl_beta_gamma(p, q):
    t1 = -p.entropy()
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (q.concentration - 1) * (
        p.concentration1.digamma() - (p.concentration1 + p.concentration0).digamma()
    )
    t4 = q.rate * p.concentration1 / (p.concentration1 + p.concentration0)
    return t1 + t2 - t3 + t4


# TODO: Add Beta-Laplace KL Divergence


@register_kl(Beta, Normal)
def _kl_beta_normal(p, q):
    E_beta = p.concentration1 / (p.concentration1 + p.concentration0)
    var_normal = q.scale.pow(2)
    t1 = -p.entropy()
    t2 = 0.5 * (var_normal * 2 * math.pi).log()
    t3 = (
        E_beta * (1 - E_beta) / (p.concentration1 + p.concentration0 + 1)
        + E_beta.pow(2)
    ) * 0.5
    t4 = q.loc * E_beta
    t5 = q.loc.pow(2) * 0.5
    return t1 + t2 + (t3 - t4 + t5) / var_normal


@register_kl(Beta, Uniform)
def _kl_beta_uniform(p, q):
    result = -p.entropy() + (q.high - q.low).log()
    result[(q.low > p.support.lower_bound) | (q.high < p.support.upper_bound)] = inf
    return result


# Note that the KL between a ContinuousBernoulli and Beta has no closed form


@register_kl(ContinuousBernoulli, Pareto)
def _kl_continuous_bernoulli_infinity(p, q):
    return _infinite_like(p.probs)


@register_kl(ContinuousBernoulli, Exponential)
def _kl_continuous_bernoulli_exponential(p, q):
    return -p.entropy() - torch.log(q.rate) + q.rate * p.mean


# Note that the KL between a ContinuousBernoulli and Gamma has no closed form
# TODO: Add ContinuousBernoulli-Laplace KL Divergence


@register_kl(ContinuousBernoulli, Normal)
def _kl_continuous_bernoulli_normal(p, q):
    t1 = -p.entropy()
    t2 = 0.5 * (math.log(2.0 * math.pi) + torch.square(q.loc / q.scale)) + torch.log(
        q.scale
    )
    t3 = (p.variance + torch.square(p.mean) - 2.0 * q.loc * p.mean) / (
        2.0 * torch.square(q.scale)
    )
    return t1 + t2 + t3


@register_kl(ContinuousBernoulli, Uniform)
def _kl_continuous_bernoulli_uniform(p, q):
    result = -p.entropy() + (q.high - q.low).log()
    return torch.where(
        torch.max(
            torch.ge(q.low, p.support.lower_bound),
            torch.le(q.high, p.support.upper_bound),
        ),
        torch.ones_like(result) * inf,
        result,
    )


@register_kl(Exponential, Beta)
@register_kl(Exponential, ContinuousBernoulli)
@register_kl(Exponential, Pareto)
@register_kl(Exponential, Uniform)
def _kl_exponential_infinity(p, q):
    return _infinite_like(p.rate)


@register_kl(Exponential, Gamma)
def _kl_exponential_gamma(p, q):
    ratio = q.rate / p.rate
    t1 = -q.concentration * torch.log(ratio)
    return (
        t1
        + ratio
        + q.concentration.lgamma()
        + q.concentration * _euler_gamma
        - (1 + _euler_gamma)
    )


@register_kl(Exponential, Gumbel)
def _kl_exponential_gumbel(p, q):
    scale_rate_prod = p.rate * q.scale
    loc_scale_ratio = q.loc / q.scale
    t1 = scale_rate_prod.log() - 1
    t2 = torch.exp(loc_scale_ratio) * scale_rate_prod / (scale_rate_prod + 1)
    t3 = scale_rate_prod.reciprocal()
    return t1 - loc_scale_ratio + t2 + t3


# TODO: Add Exponential-Laplace KL Divergence


@register_kl(Exponential, Normal)
def _kl_exponential_normal(p, q):
    var_normal = q.scale.pow(2)
    rate_sqr = p.rate.pow(2)
    t1 = 0.5 * torch.log(rate_sqr * var_normal * 2 * math.pi)
    t2 = rate_sqr.reciprocal()
    t3 = q.loc / p.rate
    t4 = q.loc.pow(2) * 0.5
    return t1 - 1 + (t2 - t3 + t4) / var_normal


@register_kl(Gamma, Beta)
@register_kl(Gamma, ContinuousBernoulli)
@register_kl(Gamma, Pareto)
@register_kl(Gamma, Uniform)
def _kl_gamma_infinity(p, q):
    return _infinite_like(p.concentration)


@register_kl(Gamma, Exponential)
def _kl_gamma_exponential(p, q):
    return -p.entropy() - q.rate.log() + q.rate * p.concentration / p.rate


@register_kl(Gamma, Gumbel)
def _kl_gamma_gumbel(p, q):
    beta_scale_prod = p.rate * q.scale
    loc_scale_ratio = q.loc / q.scale
    t1 = (
        (p.concentration - 1) * p.concentration.digamma()
        - p.concentration.lgamma()
        - p.concentration
    )
    t2 = beta_scale_prod.log() + p.concentration / beta_scale_prod
    t3 = (
        torch.exp(loc_scale_ratio)
        * (1 + beta_scale_prod.reciprocal()).pow(-p.concentration)
        - loc_scale_ratio
    )
    return t1 + t2 + t3


# TODO: Add Gamma-Laplace KL Divergence


@register_kl(Gamma, Normal)
def _kl_gamma_normal(p, q):
    var_normal = q.scale.pow(2)
    beta_sqr = p.rate.pow(2)
    t1 = (
        0.5 * torch.log(beta_sqr * var_normal * 2 * math.pi)
        - p.concentration
        - p.concentration.lgamma()
    )
    t2 = 0.5 * (p.concentration.pow(2) + p.concentration) / beta_sqr
    t3 = q.loc * p.concentration / p.rate
    t4 = 0.5 * q.loc.pow(2)
    return (
        t1
        + (p.concentration - 1) * p.concentration.digamma()
        + (t2 - t3 + t4) / var_normal
    )


@register_kl(Gumbel, Beta)
@register_kl(Gumbel, ContinuousBernoulli)
@register_kl(Gumbel, Exponential)
@register_kl(Gumbel, Gamma)
@register_kl(Gumbel, Pareto)
@register_kl(Gumbel, Uniform)
def _kl_gumbel_infinity(p, q):
    return _infinite_like(p.loc)


# TODO: Add Gumbel-Laplace KL Divergence


@register_kl(Gumbel, Normal)
def _kl_gumbel_normal(p, q):
    param_ratio = p.scale / q.scale
    t1 = (param_ratio / math.sqrt(2 * math.pi)).log()
    t2 = (math.pi * param_ratio * 0.5).pow(2) / 3
    t3 = ((p.loc + p.scale * _euler_gamma - q.loc) / q.scale).pow(2) * 0.5
    return -t1 + t2 + t3 - (_euler_gamma + 1)


@register_kl(Laplace, Beta)
@register_kl(Laplace, ContinuousBernoulli)
@register_kl(Laplace, Exponential)
@register_kl(Laplace, Gamma)
@register_kl(Laplace, Pareto)
@register_kl(Laplace, Uniform)
def _kl_laplace_infinity(p, q):
    return _infinite_like(p.loc)


@register_kl(Laplace, Normal)
def _kl_laplace_normal(p, q):
    var_normal = q.scale.pow(2)
    scale_sqr_var_ratio = p.scale.pow(2) / var_normal
    t1 = 0.5 * torch.log(2 * scale_sqr_var_ratio / math.pi)
    t2 = 0.5 * p.loc.pow(2)
    t3 = p.loc * q.loc
    t4 = 0.5 * q.loc.pow(2)
    return -t1 + scale_sqr_var_ratio + (t2 - t3 + t4) / var_normal - 1


@register_kl(Normal, Beta)
@register_kl(Normal, ContinuousBernoulli)
@register_kl(Normal, Exponential)
@register_kl(Normal, Gamma)
@register_kl(Normal, Pareto)
@register_kl(Normal, Uniform)
def _kl_normal_infinity(p, q):
    return _infinite_like(p.loc)


@register_kl(Normal, Gumbel)
def _kl_normal_gumbel(p, q):
    mean_scale_ratio = p.loc / q.scale
    var_scale_sqr_ratio = (p.scale / q.scale).pow(2)
    loc_scale_ratio = q.loc / q.scale
    t1 = var_scale_sqr_ratio.log() * 0.5
    t2 = mean_scale_ratio - loc_scale_ratio
    t3 = torch.exp(-mean_scale_ratio + 0.5 * var_scale_sqr_ratio + loc_scale_ratio)
    return -t1 + t2 + t3 - (0.5 * (1 + math.log(2 * math.pi)))


@register_kl(Normal, Laplace)
def _kl_normal_laplace(p, q):
    loc_diff = p.loc - q.loc
    scale_ratio = p.scale / q.scale
    loc_diff_scale_ratio = loc_diff / p.scale
    t1 = torch.log(scale_ratio)
    t2 = (
        math.sqrt(2 / math.pi) * p.scale * torch.exp(-0.5 * loc_diff_scale_ratio.pow(2))
    )
    t3 = loc_diff * torch.erf(math.sqrt(0.5) * loc_diff_scale_ratio)
    return -t1 + (t2 + t3) / q.scale - (0.5 * (1 + math.log(0.5 * math.pi)))


@register_kl(Pareto, Beta)
@register_kl(Pareto, ContinuousBernoulli)
@register_kl(Pareto, Uniform)
def _kl_pareto_infinity(p, q):
    return _infinite_like(p.scale)


@register_kl(Pareto, Exponential)
def _kl_pareto_exponential(p, q):
    scale_rate_prod = p.scale * q.rate
    t1 = (p.alpha / scale_rate_prod).log()
    t2 = p.alpha.reciprocal()
    t3 = p.alpha * scale_rate_prod / (p.alpha - 1)
    result = t1 - t2 + t3 - 1
    result[p.alpha <= 1] = inf
    return result


@register_kl(Pareto, Gamma)
def _kl_pareto_gamma(p, q):
    common_term = p.scale.log() + p.alpha.reciprocal()
    t1 = p.alpha.log() - common_term
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (1 - q.concentration) * common_term
    t4 = q.rate * p.alpha * p.scale / (p.alpha - 1)
    result = t1 + t2 + t3 + t4 - 1
    result[p.alpha <= 1] = inf
    return result


# TODO: Add Pareto-Laplace KL Divergence


@register_kl(Pareto, Normal)
def _kl_pareto_normal(p, q):
    var_normal = 2 * q.scale.pow(2)
    common_term = p.scale / (p.alpha - 1)
    t1 = (math.sqrt(2 * math.pi) * q.scale * p.alpha / p.scale).log()
    t2 = p.alpha.reciprocal()
    t3 = p.alpha * common_term.pow(2) / (p.alpha - 2)
    t4 = (p.alpha * common_term - q.loc).pow(2)
    result = t1 - t2 + (t3 + t4) / var_normal - 1
    result[p.alpha <= 2] = inf
    return result


@register_kl(Poisson, Bernoulli)
@register_kl(Poisson, Binomial)
def _kl_poisson_infinity(p, q):
    return _infinite_like(p.rate)


@register_kl(Uniform, Beta)
def _kl_uniform_beta(p, q):
    common_term = p.high - p.low
    t1 = torch.log(common_term)
    t2 = (
        (q.concentration1 - 1)
        * (_x_log_x(p.high) - _x_log_x(p.low) - common_term)
        / common_term
    )
    t3 = (
        (q.concentration0 - 1)
        * (_x_log_x(1 - p.high) - _x_log_x(1 - p.low) + common_term)
        / common_term
    )
    t4 = (
        q.concentration1.lgamma()
        + q.concentration0.lgamma()
        - (q.concentration1 + q.concentration0).lgamma()
    )
    result = t3 + t4 - t1 - t2
    result[(p.high > q.support.upper_bound) | (p.low < q.support.lower_bound)] = inf
    return result


@register_kl(Uniform, ContinuousBernoulli)
def _kl_uniform_continuous_bernoulli(p, q):
    result = (
        -p.entropy()
        - p.mean * q.logits
        - torch.log1p(-q.probs)
        - q._cont_bern_log_norm()
    )
    return torch.where(
        torch.max(
            torch.ge(p.high, q.support.upper_bound),
            torch.le(p.low, q.support.lower_bound),
        ),
        torch.ones_like(result) * inf,
        result,
    )


@register_kl(Uniform, Exponential)
def _kl_uniform_exponetial(p, q):
    result = q.rate * (p.high + p.low) / 2 - ((p.high - p.low) * q.rate).log()
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Uniform, Gamma)
def _kl_uniform_gamma(p, q):
    common_term = p.high - p.low
    t1 = common_term.log()
    t2 = q.concentration.lgamma() - q.concentration * q.rate.log()
    t3 = (
        (1 - q.concentration)
        * (_x_log_x(p.high) - _x_log_x(p.low) - common_term)
        / common_term
    )
    t4 = q.rate * (p.high + p.low) / 2
    result = -t1 + t2 + t3 + t4
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Uniform, Gumbel)
def _kl_uniform_gumbel(p, q):
    common_term = q.scale / (p.high - p.low)
    high_loc_diff = (p.high - q.loc) / q.scale
    low_loc_diff = (p.low - q.loc) / q.scale
    t1 = common_term.log() + 0.5 * (high_loc_diff + low_loc_diff)
    t2 = common_term * (torch.exp(-high_loc_diff) - torch.exp(-low_loc_diff))
    return t1 - t2


# TODO: Uniform-Laplace KL Divergence


@register_kl(Uniform, Normal)
def _kl_uniform_normal(p, q):
    common_term = p.high - p.low
    t1 = (math.sqrt(math.pi * 2) * q.scale / common_term).log()
    t2 = (common_term).pow(2) / 12
    t3 = ((p.high + p.low - 2 * q.loc) / 2).pow(2)
    return t1 + 0.5 * (t2 + t3) / q.scale.pow(2)


@register_kl(Uniform, Pareto)
def _kl_uniform_pareto(p, q):
    support_uniform = p.high - p.low
    t1 = (q.alpha * q.scale.pow(q.alpha) * (support_uniform)).log()
    t2 = (_x_log_x(p.high) - _x_log_x(p.low) - support_uniform) / support_uniform
    result = t2 * (q.alpha + 1) - t1
    result[p.low < q.support.lower_bound] = inf
    return result


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)


@register_kl(Cauchy, Cauchy)
def _kl_cauchy_cauchy(p, q):
    # From https://arxiv.org/abs/1905.10965
    t1 = ((p.scale + q.scale).pow(2) + (p.loc - q.loc).pow(2)).log()
    t2 = (4 * p.scale * q.scale).log()
    return t1 - t2


def _add_kl_info():
    """Appends a list of implemented KL functions to the doc for kl_divergence."""
    rows = [
        "KL divergence is currently implemented for the following distribution pairs:"
    ]
    for p, q in sorted(
        _KL_REGISTRY, key=lambda p_q: (p_q[0].__name__, p_q[1].__name__)
    ):
        rows.append(
            f"* :class:`~torch.distributions.{p.__name__}` and :class:`~torch.distributions.{q.__name__}`"
        )
    kl_info = "\n\t".join(rows)
    if kl_divergence.__doc__:
        # pyrefly: ignore [missing-attribute]
        kl_divergence.__doc__ += kl_info
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _Number, Number


__all__ = ["Geometric"]


class Geometric(Distribution):
    r"""
    Creates a Geometric distribution parameterized by :attr:`probs`,
    where :attr:`probs` is the probability of success of Bernoulli trials.

    .. math::

        P(X=k) = (1-p)^{k} p, k = 0, 1, ...

    .. note::
        :func:`torch.distributions.geometric.Geometric` :math:`(k+1)`-th trial is the first success
        hence draws samples in :math:`\{0, 1, \ldots\}`, whereas
        :func:`torch.Tensor.geometric_` `k`-th trial is the first success hence draws samples in :math:`\{1, 2, \ldots\}`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Geometric(torch.tensor([0.3]))
        >>> m.sample()  # underlying Bernoulli has 30% chance 1; 70% chance 0
        tensor([ 2.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`. Must be in range (0, 1]
        logits (Number, Tensor): the log-odds of sampling `1`.
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.nonnegative_integer

    def __init__(
        self,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            # pyrefly: ignore [read-only]
            (self.probs,) = broadcast_all(probs)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            # pyrefly: ignore [read-only]
            (self.logits,) = broadcast_all(logits)
        probs_or_logits = probs if probs is not None else logits
        if isinstance(probs_or_logits, _Number):
            batch_shape = torch.Size()
        else:
            if probs_or_logits is None:
                raise AssertionError("probs_or_logits is unexpectedly None")
            batch_shape = probs_or_logits.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self._validate_args and probs is not None:
            # Add an extra check beyond unit_interval
            value = self.probs
            valid = value > 0
            if not valid.all():
                invalid_value = value.data[~valid]
                raise ValueError(
                    "Expected parameter probs "
                    f"({type(value).__name__} of shape {tuple(value.shape)}) "
                    f"of distribution {repr(self)} "
                    f"to be positive but found invalid values:\n{invalid_value}"
                )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Geometric, _instance)
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
        super(Geometric, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return 1.0 / self.probs - 1.0

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.probs)

    @property
    def variance(self) -> Tensor:
        return (1.0 / self.probs - 1.0) / self.probs

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits, is_binary=True)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        tiny = torch.finfo(self.probs.dtype).tiny
        with torch.no_grad():
            if torch._C._get_tracing_state():
                # [JIT WORKAROUND] lack of support for .uniform_()
                u = torch.rand(shape, dtype=self.probs.dtype, device=self.probs.device)
                u = u.clamp(min=tiny)
            else:
                u = self.probs.new(shape).uniform_(tiny, 1)
            return (u.log() / (-self.probs).log1p()).floor()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value, probs = broadcast_all(value, self.probs)
        probs = probs.clone(memory_format=torch.contiguous_format)
        probs[(probs == 1) & (value == 0)] = 0
        return value * (-probs).log1p() + self.probs.log()

    def entropy(self):
        return (
            binary_cross_entropy_with_logits(self.logits, self.probs, reduction="none")
            / self.probs
        )
<|endoftext|># mypy: allow-untyped-defs
import functools
import math
import operator
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import (
    _sum_rightmost,
    broadcast_all,
    lazy_property,
    tril_matrix_to_vec,
    vec_to_tril_matrix,
)
from torch.nn.functional import pad, softplus
from torch.types import _Number


__all__ = [
    "AbsTransform",
    "AffineTransform",
    "CatTransform",
    "ComposeTransform",
    "CorrCholeskyTransform",
    "CumulativeDistributionTransform",
    "ExpTransform",
    "IndependentTransform",
    "LowerCholeskyTransform",
    "PositiveDefiniteTransform",
    "PowerTransform",
    "ReshapeTransform",
    "SigmoidTransform",
    "SoftplusTransform",
    "TanhTransform",
    "SoftmaxTransform",
    "StackTransform",
    "StickBreakingTransform",
    "Transform",
    "identity_transform",
]


class Transform:
    """
    Abstract class for invertible transformations with computable log
    det jacobians. They are primarily used in
    :class:`torch.distributions.TransformedDistribution`.

    Caching is useful for transforms whose inverses are either expensive or
    numerically unstable. Note that care must be taken with memoized values
    since the autograd graph may be reversed. For example while the following
    works with or without caching::

        y = t(x)
        t.log_abs_det_jacobian(x, y).backward()  # x will receive gradients.

    However the following will error when caching due to dependency reversal::

        y = t(x)
        z = t.inv(y)
        grad(z.sum(), [y])  # error because z is x

    Derived classes should implement one or both of :meth:`_call` or
    :meth:`_inverse`. Derived classes that set `bijective=True` should also
    implement :meth:`log_abs_det_jacobian`.

    Args:
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.

    Attributes:
        domain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid inputs to this transform.
        codomain (:class:`~torch.distributions.constraints.Constraint`):
            The constraint representing valid outputs to this transform
            which are inputs to the inverse transform.
        bijective (bool): Whether this transform is bijective. A transform
            ``t`` is bijective iff ``t.inv(t(x)) == x`` and
            ``t(t.inv(y)) == y`` for every ``x`` in the domain and ``y`` in
            the codomain. Transforms that are not bijective should at least
            maintain the weaker pseudoinverse properties
            ``t(t.inv(t(x)) == t(x)`` and ``t.inv(t(t.inv(y))) == t.inv(y)``.
        sign (int or Tensor): For bijective univariate transforms, this
            should be +1 or -1 depending on whether transform is monotone
            increasing or decreasing.
    """

    bijective = False
    domain: constraints.Constraint
    codomain: constraints.Constraint

    def __init__(self, cache_size: int = 0) -> None:
        self._cache_size = cache_size
        self._inv: Transform | None = None
        if cache_size == 0:
            pass  # default behavior
        elif cache_size == 1:
            self._cached_x_y = None, None
        else:
            raise ValueError("cache_size must be 0 or 1")
        super().__init__()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_inv"] = None
        return state

    @property
    def event_dim(self) -> int:
        if self.domain.event_dim == self.codomain.event_dim:
            return self.domain.event_dim
        raise ValueError("Please use either .domain.event_dim or .codomain.event_dim")

    @property
    def inv(self) -> "Transform":
        """
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        """
        inv = self._inv
        if inv is None:
            inv = _InverseTransform(self)
            self._inv = inv
        return inv

    @property
    def sign(self) -> int:
        """
        Returns the sign of the determinant of the Jacobian, if applicable.
        In general this only makes sense for bijective transforms.
        """
        raise NotImplementedError

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        if type(self).__init__ is Transform.__init__:
            return type(self)(cache_size=cache_size)
        raise NotImplementedError(f"{type(self)}.with_cache is not implemented")

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        # Necessary for Python2
        return not self.__eq__(other)

    def __call__(self, x):
        """
        Computes the transform `x => y`.
        """
        if self._cache_size == 0:
            return self._call(x)
        x_old, y_old = self._cached_x_y
        if x is x_old:
            return y_old
        y = self._call(x)
        self._cached_x_y = x, y
        return y

    def _inv_call(self, y):
        """
        Inverts the transform `y => x`.
        """
        if self._cache_size == 0:
            return self._inverse(y)
        x_old, y_old = self._cached_x_y
        if y is y_old:
            return x_old
        x = self._inverse(y)
        self._cached_x_y = x, y
        return x

    def _call(self, x):
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError

    def _inverse(self, y):
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def forward_shape(self, shape):
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
        return shape

    def inverse_shape(self, shape):
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """
        return shape


class _InverseTransform(Transform):
    """
    Inverts a single :class:`Transform`.
    This class is private; please instead use the ``Transform.inv`` property.
    """

    def __init__(self, transform: Transform) -> None:
        super().__init__(cache_size=transform._cache_size)
        self._inv: Transform | None = transform

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def domain(self):
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv.codomain

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def codomain(self):
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv.domain

    @property
    def bijective(self) -> bool:  # type: ignore[override]
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv.bijective

    @property
    def sign(self) -> int:
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv.sign

    @property
    def inv(self) -> Transform:
        return self._inv  # pyrefly: ignore[bad-return]

    def with_cache(self, cache_size=1):
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self.inv.with_cache(cache_size).inv

    def __eq__(self, other):
        if not isinstance(other, _InverseTransform):
            return False
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv == other._inv

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self._inv)})"

    def __call__(self, x):
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return self._inv._inv_call(x)

    def log_abs_det_jacobian(self, x, y):
        if self._inv is None:
            raise AssertionError("_inv must not be None")
        return -self._inv.log_abs_det_jacobian(y, x)

    def forward_shape(self, shape):
        return self._inv.inverse_shape(shape)  # pyrefly: ignore[missing-attribute]

    def inverse_shape(self, shape):
        return self._inv.forward_shape(shape)  # pyrefly: ignore[missing-attribute]


class ComposeTransform(Transform):
    """
    Composes multiple transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform`): A list of transforms to compose.
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported.
    """

    def __init__(self, parts: list[Transform], cache_size: int = 0) -> None:
        if cache_size:
            parts = [part.with_cache(cache_size) for part in parts]
        super().__init__(cache_size=cache_size)
        self.parts = parts

    def __eq__(self, other):
        if not isinstance(other, ComposeTransform):
            return False
        return self.parts == other.parts

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def domain(self):
        if not self.parts:
            return constraints.real
        domain = self.parts[0].domain
        # Adjust event_dim to be maximum among all parts.
        event_dim = self.parts[-1].codomain.event_dim
        for part in reversed(self.parts):
            event_dim += part.domain.event_dim - part.codomain.event_dim
            event_dim = max(event_dim, part.domain.event_dim)
        if event_dim < domain.event_dim:
            raise AssertionError(
                f"event_dim {event_dim} must be >= domain.event_dim {domain.event_dim}"
            )
        if event_dim > domain.event_dim:
            domain = constraints.independent(domain, event_dim - domain.event_dim)
        return domain

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def codomain(self):
        if not self.parts:
            return constraints.real
        codomain = self.parts[-1].codomain
        # Adjust event_dim to be maximum among all parts.
        event_dim = self.parts[0].domain.event_dim
        for part in self.parts:
            event_dim += part.codomain.event_dim - part.domain.event_dim
            event_dim = max(event_dim, part.codomain.event_dim)
        if event_dim < codomain.event_dim:
            raise AssertionError(
                f"event_dim {event_dim} must be >= codomain.event_dim {codomain.event_dim}"
            )
        if event_dim > codomain.event_dim:
            codomain = constraints.independent(codomain, event_dim - codomain.event_dim)
        return codomain

    @lazy_property
    def bijective(self) -> bool:  # type: ignore[override]
        return all(p.bijective for p in self.parts)

    @lazy_property
    def sign(self) -> int:  # type: ignore[override]
        sign = 1
        for p in self.parts:
            sign = sign * p.sign
        return sign

    @property
    def inv(self) -> Transform:
        inv = self._inv
        if inv is None:
            inv = ComposeTransform([p.inv for p in reversed(self.parts)])
            self._inv = inv
            inv._inv = self
        return inv

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return ComposeTransform(self.parts, cache_size=cache_size)

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def log_abs_det_jacobian(self, x, y):
        if not self.parts:
            return torch.zeros_like(x)

        # Compute intermediates. This will be free if parts[:-1] are all cached.
        xs = [x]
        for part in self.parts[:-1]:
            xs.append(part(xs[-1]))
        xs.append(y)

        terms = []
        event_dim = self.domain.event_dim
        for part, x, y in zip(self.parts, xs[:-1], xs[1:]):
            terms.append(
                _sum_rightmost(
                    part.log_abs_det_jacobian(x, y), event_dim - part.domain.event_dim
                )
            )
            event_dim += part.codomain.event_dim - part.domain.event_dim
        return functools.reduce(operator.add, terms)

    def forward_shape(self, shape):
        for part in self.parts:
            shape = part.forward_shape(shape)
        return shape

    def inverse_shape(self, shape):
        for part in reversed(self.parts):
            shape = part.inverse_shape(shape)
        return shape

    def __repr__(self):
        fmt_string = self.__class__.__name__ + "(\n    "
        fmt_string += ",\n    ".join([p.__repr__() for p in self.parts])
        fmt_string += "\n)"
        return fmt_string


identity_transform = ComposeTransform([])


class IndependentTransform(Transform):
    """
    Wrapper around another transform to treat
    ``reinterpreted_batch_ndims``-many extra of the right most dimensions as
    dependent. This has no effect on the forward or backward transforms, but
    does sum out ``reinterpreted_batch_ndims``-many of the rightmost dimensions
    in :meth:`log_abs_det_jacobian`.

    Args:
        base_transform (:class:`Transform`): A base transform.
        reinterpreted_batch_ndims (int): The number of extra rightmost
            dimensions to treat as dependent.
    """

    def __init__(
        self,
        base_transform: Transform,
        reinterpreted_batch_ndims: int,
        cache_size: int = 0,
    ) -> None:
        super().__init__(cache_size=cache_size)
        self.base_transform = base_transform.with_cache(cache_size)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return IndependentTransform(
            self.base_transform, self.reinterpreted_batch_ndims, cache_size=cache_size
        )

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def domain(self):
        return constraints.independent(
            self.base_transform.domain, self.reinterpreted_batch_ndims
        )

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def codomain(self):
        return constraints.independent(
            self.base_transform.codomain, self.reinterpreted_batch_ndims
        )

    @property
    def bijective(self) -> bool:  # type: ignore[override]
        return self.base_transform.bijective

    @property
    def sign(self) -> int:
        return self.base_transform.sign

    def _call(self, x):
        if x.dim() < self.domain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform(x)

    def _inverse(self, y):
        if y.dim() < self.codomain.event_dim:
            raise ValueError("Too few dimensions on input")
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x, y):
        result = self.base_transform.log_abs_det_jacobian(x, y)
        result = _sum_rightmost(result, self.reinterpreted_batch_ndims)
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.base_transform)}, {self.reinterpreted_batch_ndims})"

    def forward_shape(self, shape):
        return self.base_transform.forward_shape(shape)

    def inverse_shape(self, shape):
        return self.base_transform.inverse_shape(shape)


class ReshapeTransform(Transform):
    """
    Unit Jacobian transform to reshape the rightmost part of a tensor.

    Note that ``in_shape`` and ``out_shape`` must have the same number of
    elements, just as for :meth:`torch.Tensor.reshape`.

    Arguments:
        in_shape (torch.Size): The input event shape.
        out_shape (torch.Size): The output event shape.
        cache_size (int): Size of cache. If zero, no caching is done. If one,
            the latest single value is cached. Only 0 and 1 are supported. (Default 0.)
    """

    bijective = True

    def __init__(
        self,
        in_shape: torch.Size,
        out_shape: torch.Size,
        cache_size: int = 0,
    ) -> None:
        self.in_shape = torch.Size(in_shape)
        self.out_shape = torch.Size(out_shape)
        if self.in_shape.numel() != self.out_shape.numel():
            raise ValueError("in_shape, out_shape have different numbers of elements")
        super().__init__(cache_size=cache_size)

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def domain(self):
        return constraints.independent(constraints.real, len(self.in_shape))

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def codomain(self):
        return constraints.independent(constraints.real, len(self.out_shape))

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return ReshapeTransform(self.in_shape, self.out_shape, cache_size=cache_size)

    def _call(self, x):
        batch_shape = x.shape[: x.dim() - len(self.in_shape)]
        return x.reshape(batch_shape + self.out_shape)

    def _inverse(self, y):
        batch_shape = y.shape[: y.dim() - len(self.out_shape)]
        return y.reshape(batch_shape + self.in_shape)

    def log_abs_det_jacobian(self, x, y):
        batch_shape = x.shape[: x.dim() - len(self.in_shape)]
        return x.new_zeros(batch_shape)

    def forward_shape(self, shape):
        if len(shape) < len(self.in_shape):
            raise ValueError("Too few dimensions on input")
        cut = len(shape) - len(self.in_shape)
        if shape[cut:] != self.in_shape:
            raise ValueError(
                f"Shape mismatch: expected {shape[cut:]} but got {self.in_shape}"
            )
        return shape[:cut] + self.out_shape

    def inverse_shape(self, shape):
        if len(shape) < len(self.out_shape):
            raise ValueError("Too few dimensions on input")
        cut = len(shape) - len(self.out_shape)
        if shape[cut:] != self.out_shape:
            raise ValueError(
                f"Shape mismatch: expected {shape[cut:]} but got {self.out_shape}"
            )
        return shape[:cut] + self.in_shape


class ExpTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \exp(x)`.
    """

    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, ExpTransform)

    def _call(self, x):
        return x.exp()

    def _inverse(self, y):
        return y.log()

    def log_abs_det_jacobian(self, x, y):
        return x


class PowerTransform(Transform):
    r"""
    Transform via the mapping :math:`y = x^{\text{exponent}}`.
    """

    domain = constraints.positive
    codomain = constraints.positive
    bijective = True

    def __init__(self, exponent: Tensor, cache_size: int = 0) -> None:
        super().__init__(cache_size=cache_size)
        (self.exponent,) = broadcast_all(exponent)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return PowerTransform(self.exponent, cache_size=cache_size)

    @lazy_property
    def sign(self) -> int:  # type: ignore[override]
        return self.exponent.sign()  # type: ignore[return-value]

    def __eq__(self, other):
        if not isinstance(other, PowerTransform):
            return False
        return self.exponent.eq(other.exponent).all().item()

    def _call(self, x):
        return x.pow(self.exponent)

    def _inverse(self, y):
        return y.pow(1 / self.exponent)

    def log_abs_det_jacobian(self, x, y):
        return (self.exponent * y / x).abs().log()

    def forward_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))

    def inverse_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, "shape", ()))


def _clipped_sigmoid(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1.0 - finfo.eps)


class SigmoidTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
    """

    domain = constraints.real
    codomain = constraints.unit_interval
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SigmoidTransform)

    def _call(self, x):
        return _clipped_sigmoid(x)

    def _inverse(self, y):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -F.softplus(-x) - F.softplus(x)


class SoftplusTransform(Transform):
    r"""
    Transform via the mapping :math:`\text{Softplus}(x) = \log(1 + \exp(x))`.
    The implementation reverts to the linear function when :math:`x > 20`.
    """

    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return softplus(x)

    def _inverse(self, y):
        return (-y).expm1().neg().log() + y

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x)


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.

    It is equivalent to

    .. code-block:: python

        ComposeTransform(
            [
                AffineTransform(0.0, 2.0),
                SigmoidTransform(),
                AffineTransform(-1.0, 2.0),
            ]
        )

    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.

    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    """

    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2.0 * (math.log(2.0) - x - softplus(-2.0 * x))


class AbsTransform(Transform):
    r"""Transform via the mapping :math:`y = |x|`."""

    domain = constraints.real
    codomain = constraints.positive

    def __eq__(self, other):
        return isinstance(other, AbsTransform)

    def _call(self, x):
        return x.abs()

    def _inverse(self, y):
        return y


class AffineTransform(Transform):
    r"""
    Transform via the pointwise affine mapping :math:`y = \text{loc} + \text{scale} \times x`.

    Args:
        loc (Tensor or float): Location parameter.
        scale (Tensor or float): Scale parameter.
        event_dim (int): Optional size of `event_shape`. This should be zero
            for univariate random variables, 1 for distributions over vectors,
            2 for distributions over matrices, etc.
    """

    bijective = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        event_dim: int = 0,
        cache_size: int = 0,
    ) -> None:
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale = scale
        self._event_dim = event_dim

    @property
    def event_dim(self) -> int:
        return self._event_dim

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def domain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    @constraints.dependent_property(is_discrete=False)
    # pyrefly: ignore [bad-override]
    def codomain(self):
        if self.event_dim == 0:
            return constraints.real
        return constraints.independent(constraints.real, self.event_dim)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return AffineTransform(
            self.loc, self.scale, self.event_dim, cache_size=cache_size
        )

    def __eq__(self, other):
        if not isinstance(other, AffineTransform):
            return False

        if isinstance(self.loc, _Number) and isinstance(other.loc, _Number):
            if self.loc != other.loc:
                return False
        else:
            if not (self.loc == other.loc).all().item():  # type: ignore[union-attr]
                return False

        if isinstance(self.scale, _Number) and isinstance(other.scale, _Number):
            if self.scale != other.scale:
                return False
        else:
            if not (self.scale == other.scale).all().item():  # type: ignore[union-attr]
                return False

        return True

    @property
    def sign(self) -> Tensor | int:  # type: ignore[override]
        if isinstance(self.scale, _Number):
            return 1 if float(self.scale) > 0 else -1 if float(self.scale) < 0 else 0
        return self.scale.sign()

    def _call(self, x):
        return self.loc + self.scale * x

    def _inverse(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x, y):
        shape = x.shape
        scale = self.scale
        if isinstance(scale, _Number):
            result = torch.full_like(x, math.log(abs(scale)))
        else:
            result = torch.abs(scale).log()
        if self.event_dim:
            result_size = result.size()[: -self.event_dim] + (-1,)
            result = result.view(result_size).sum(-1)
            shape = shape[: -self.event_dim]
        return result.expand(shape)

    def forward_shape(self, shape):
        return torch.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )

    def inverse_shape(self, shape):
        return torch.broadcast_shapes(
            shape, getattr(self.loc, "shape", ()), getattr(self.scale, "shape", ())
        )


class CorrCholeskyTransform(Transform):
    r"""
    Transforms an unconstrained real vector :math:`x` with length :math:`D*(D-1)/2` into the
    Cholesky factor of a D-dimension correlation matrix. This Cholesky factor is a lower
    triangular matrix with positive diagonals and unit Euclidean norm for each row.
    The transform is processed as follows:

        1. First we convert x into a lower triangular matrix in row order.
        2. For each row :math:`X_i` of the lower triangular part, we apply a *signed* version of
           class :class:`StickBreakingTransform` to transform :math:`X_i` into a
           unit Euclidean length vector using the following steps:
           - Scales into the interval :math:`(-1, 1)` domain: :math:`r_i = \tanh(X_i)`.
           - Transforms into an unsigned domain: :math:`z_i = r_i^2`.
           - Applies :math:`s_i = StickBreakingTransform(z_i)`.
           - Transforms back into signed domain: :math:`y_i = sign(r_i) * \sqrt{s_i}`.
    """

    domain = constraints.real_vector
    codomain = constraints.corr_cholesky
    bijective = True

    def _call(self, x):
        x = torch.tanh(x)
        eps = torch.finfo(x.dtype).eps
        x = x.clamp(min=-1 + eps, max=1 - eps)
        r = vec_to_tril_matrix(x, diag=-1)
        # apply stick-breaking on the squared values
        # Note that y = sign(r) * sqrt(z * z1m_cumprod)
        #             = (sign(r) * sqrt(z)) * sqrt(z1m_cumprod) = r * sqrt(z1m_cumprod)
        z = r**2
        z1m_cumprod_sqrt = (1 - z).sqrt().cumprod(-1)
        # Diagonal elements must be 1.
        r = r + torch.eye(r.shape[-1], dtype=r.dtype, device=r.device)
        y = r * pad(z1m_cumprod_sqrt[..., :-1], [1, 0], value=1)
        return y

    def _inverse(self, y):
        # inverse stick-breaking
        # See: https://mc-stan.org/docs/2_18/reference-manual/cholesky-factors-of-correlation-matrices-1.html
        y_cumsum = 1 - torch.cumsum(y * y, dim=-1)
        y_cumsum_shifted = pad(y_cumsum[..., :-1], [1, 0], value=1)
        y_vec = tril_matrix_to_vec(y, diag=-1)
        y_cumsum_vec = tril_matrix_to_vec(y_cumsum_shifted, diag=-1)
        t = y_vec / (y_cumsum_vec).sqrt()
        # inverse of tanh
        x = (t.log1p() - t.neg().log1p()) / 2
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # Because domain and codomain are two spaces with different dimensions, determinant of
        # Jacobian is not well-defined. We return `log_abs_det_jacobian` of `x` and the
        # flattened lower triangular part of `y`.

        # See: https://mc-stan.org/docs/2_18/reference-manual/cholesky-factors-of-correlation-matrices-1.html
        y1m_cumsum = 1 - (y * y).cumsum(dim=-1)
        # by taking diagonal=-2, we don't need to shift z_cumprod to the right
        # also works for 2 x 2 matrix
        y1m_cumsum_tril = tril_matrix_to_vec(y1m_cumsum, diag=-2)
        stick_breaking_logdet = 0.5 * (y1m_cumsum_tril).log().sum(-1)
        tanh_logdet = -2 * (x + softplus(-2 * x) - math.log(2.0)).sum(dim=-1)
        return stick_breaking_logdet + tanh_logdet

    def forward_shape(self, shape):
        # Reshape from (..., N) to (..., D, D).
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        N = shape[-1]
        D = round((0.25 + 2 * N) ** 0.5 + 0.5)
        if D * (D - 1) // 2 != N:
            raise ValueError("Input is not a flattened lower-diagonal number")
        return shape[:-1] + (D, D)

    def inverse_shape(self, shape):
        # Reshape from (..., D, D) to (..., N).
        if len(shape) < 2:
            raise ValueError("Too few dimensions on input")
        if shape[-2] != shape[-1]:
            raise ValueError("Input is not square")
        D = shape[-1]
        N = D * (D - 1) // 2
        return shape[:-2] + (N,)


class SoftmaxTransform(Transform):
    r"""
    Transform from unconstrained space to the simplex via :math:`y = \exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """

    domain = constraints.real_vector
    codomain = constraints.simplex

    def __eq__(self, other):
        return isinstance(other, SoftmaxTransform)

    def _call(self, x):
        logprobs = x
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        return probs / probs.sum(-1, True)

    def _inverse(self, y):
        probs = y
        return probs.log()

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape


class StickBreakingTransform(Transform):
    """
    Transform from unconstrained space to the simplex of one additional
    dimension via a stick-breaking process.

    This transform arises as an iterated sigmoid transform in a stick-breaking
    construction of the `Dirichlet` distribution: the first logit is
    transformed via sigmoid to the first probability and the probability of
    everything else, and then the process recurses.

    This is bijective and appropriate for use in HMC; however it mixes
    coordinates together and is less appropriate for optimization.
    """

    domain = constraints.real_vector
    codomain = constraints.simplex
    bijective = True

    def __eq__(self, other):
        return isinstance(other, StickBreakingTransform)

    def _call(self, x):
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        z = _clipped_sigmoid(x - offset.log())
        z_cumprod = (1 - z).cumprod(-1)
        y = pad(z, [0, 1], value=1) * pad(z_cumprod, [1, 0], value=1)
        return y

    def _inverse(self, y):
        y_crop = y[..., :-1]
        offset = y.shape[-1] - y.new_ones(y_crop.shape[-1]).cumsum(-1)
        sf = 1 - y_crop.cumsum(-1)
        # we clamp to make sure that sf is positive which sometimes does not
        # happen when y[-1] ~ 0 or y[:-1].sum() ~ 1
        sf = torch.clamp(sf, min=torch.finfo(y.dtype).tiny)
        x = y_crop.log() - sf.log() + offset.log()
        return x

    def log_abs_det_jacobian(self, x, y):
        offset = x.shape[-1] + 1 - x.new_ones(x.shape[-1]).cumsum(-1)
        x = x - offset.log()
        # use the identity 1 - sigmoid(x) = exp(-x) * sigmoid(x)
        detJ = (-x + F.logsigmoid(x) + y[..., :-1].log()).sum(-1)
        return detJ

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] + 1,)

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError("Too few dimensions on input")
        return shape[:-1] + (shape[-1] - 1,)


class LowerCholeskyTransform(Transform):
    """
    Transform from unconstrained matrices to lower-triangular matrices with
    nonnegative diagonal entries.

    This is useful for parameterizing positive definite matrices in terms of
    their Cholesky factorization.
    """

    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.lower_cholesky

    def __eq__(self, other):
        return isinstance(other, LowerCholeskyTransform)

    def _call(self, x):
        return x.tril(-1) + x.diagonal(dim1=-2, dim2=-1).exp().diag_embed()

    def _inverse(self, y):
        return y.tril(-1) + y.diagonal(dim1=-2, dim2=-1).log().diag_embed()


class PositiveDefiniteTransform(Transform):
    """
    Transform from unconstrained matrices to positive-definite matrices.
    """

    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.positive_definite

    def __eq__(self, other):
        return isinstance(other, PositiveDefiniteTransform)

    def _call(self, x):
        x = LowerCholeskyTransform()(x)
        return x @ x.mT

    def _inverse(self, y):
        y = torch.linalg.cholesky(y)
        return LowerCholeskyTransform().inv(y)


class CatTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`, of length `lengths[dim]`,
    in a way compatible with :func:`torch.cat`.

    Example::

       x0 = torch.cat([torch.range(1, 10), torch.range(1, 10)], dim=0)
       x = torch.cat([x0, x0], dim=0)
       t0 = CatTransform([ExpTransform(), identity_transform], dim=0, lengths=[10, 10])
       t = CatTransform([t0, t0], dim=0, lengths=[20, 20])
       y = t(x)
    """

    transforms: list[Transform]

    def __init__(
        self,
        tseq: Sequence[Transform],
        dim: int = 0,
        lengths: Sequence[int] | None = None,
        cache_size: int = 0,
    ) -> None:
        if not all(isinstance(t, Transform) for t in tseq):
            raise AssertionError("All elements of tseq must be Transform instances")
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
        self.transforms = list(tseq)
        if lengths is None:
            lengths = [1] * len(self.transforms)
        self.lengths = list(lengths)
        if len(self.lengths) != len(self.transforms):
            raise AssertionError(
                f"lengths ({len(self.lengths)}) must match transforms ({len(self.transforms)})"
            )
        self.dim = dim

    @lazy_property
    def event_dim(self) -> int:  # type: ignore[override]
        return max(t.event_dim for t in self.transforms)

    @lazy_property
    def length(self) -> int:
        return sum(self.lengths)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return CatTransform(self.transforms, self.dim, self.lengths, cache_size)

    def _call(self, x):
        if not (-x.dim() <= self.dim < x.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for tensor with {x.dim()} dimensions"
            )
        if x.size(self.dim) != self.length:
            raise AssertionError(
                f"x.size({self.dim}) = {x.size(self.dim)} must equal length {self.length}"
            )
        yslices = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslices.append(trans(xslice))
            start = start + length  # avoid += for jit compat
        return torch.cat(yslices, dim=self.dim)

    def _inverse(self, y):
        if not (-y.dim() <= self.dim < y.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for tensor with {y.dim()} dimensions"
            )
        if y.size(self.dim) != self.length:
            raise AssertionError(
                f"y.size({self.dim}) = {y.size(self.dim)} must equal length {self.length}"
            )
        xslices = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            yslice = y.narrow(self.dim, start, length)
            xslices.append(trans.inv(yslice))
            start = start + length  # avoid += for jit compat
        return torch.cat(xslices, dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        if not (-x.dim() <= self.dim < x.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for x with {x.dim()} dimensions"
            )
        if x.size(self.dim) != self.length:
            raise AssertionError(
                f"x.size({self.dim}) = {x.size(self.dim)} must equal length {self.length}"
            )
        if not (-y.dim() <= self.dim < y.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for y with {y.dim()} dimensions"
            )
        if y.size(self.dim) != self.length:
            raise AssertionError(
                f"y.size({self.dim}) = {y.size(self.dim)} must equal length {self.length}"
            )
        logdetjacs = []
        start = 0
        for trans, length in zip(self.transforms, self.lengths):
            xslice = x.narrow(self.dim, start, length)
            yslice = y.narrow(self.dim, start, length)
            logdetjac = trans.log_abs_det_jacobian(xslice, yslice)
            if trans.event_dim < self.event_dim:
                logdetjac = _sum_rightmost(logdetjac, self.event_dim - trans.event_dim)
            logdetjacs.append(logdetjac)
            start = start + length  # avoid += for jit compat
        # Decide whether to concatenate or sum.
        dim = self.dim
        if dim >= 0:
            dim = dim - x.dim()
        dim = dim + self.event_dim
        if dim < 0:
            return torch.cat(logdetjacs, dim=dim)
        else:
            return sum(logdetjacs)

    @property
    def bijective(self) -> bool:  # type: ignore[override]
        return all(t.bijective for t in self.transforms)

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def domain(self):
        return constraints.cat(
            [t.domain for t in self.transforms], self.dim, self.lengths
        )

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def codomain(self):
        return constraints.cat(
            [t.codomain for t in self.transforms], self.dim, self.lengths
        )


class StackTransform(Transform):
    """
    Transform functor that applies a sequence of transforms `tseq`
    component-wise to each submatrix at `dim`
    in a way compatible with :func:`torch.stack`.

    Example::

       x = torch.stack([torch.range(1, 10), torch.range(1, 10)], dim=1)
       t = StackTransform([ExpTransform(), identity_transform], dim=1)
       y = t(x)
    """

    transforms: list[Transform]

    def __init__(
        self, tseq: Sequence[Transform], dim: int = 0, cache_size: int = 0
    ) -> None:
        if not all(isinstance(t, Transform) for t in tseq):
            raise AssertionError("All elements of tseq must be Transform instances")
        if cache_size:
            tseq = [t.with_cache(cache_size) for t in tseq]
        super().__init__(cache_size=cache_size)
        self.transforms = list(tseq)
        self.dim = dim

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return StackTransform(self.transforms, self.dim, cache_size)

    def _slice(self, z):
        return [z.select(self.dim, i) for i in range(z.size(self.dim))]

    def _call(self, x):
        if not (-x.dim() <= self.dim < x.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for tensor with {x.dim()} dimensions"
            )
        if x.size(self.dim) != len(self.transforms):
            raise AssertionError(
                f"x.size({self.dim}) = {x.size(self.dim)} must equal len(transforms) {len(self.transforms)}"
            )
        yslices = []
        for xslice, trans in zip(self._slice(x), self.transforms):
            yslices.append(trans(xslice))
        return torch.stack(yslices, dim=self.dim)

    def _inverse(self, y):
        if not (-y.dim() <= self.dim < y.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for tensor with {y.dim()} dimensions"
            )
        if y.size(self.dim) != len(self.transforms):
            raise AssertionError(
                f"y.size({self.dim}) = {y.size(self.dim)} must equal len(transforms) {len(self.transforms)}"
            )
        xslices = []
        for yslice, trans in zip(self._slice(y), self.transforms):
            xslices.append(trans.inv(yslice))
        return torch.stack(xslices, dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        if not (-x.dim() <= self.dim < x.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for x with {x.dim()} dimensions"
            )
        if x.size(self.dim) != len(self.transforms):
            raise AssertionError(
                f"x.size({self.dim}) = {x.size(self.dim)} must equal len(transforms) {len(self.transforms)}"
            )
        if not (-y.dim() <= self.dim < y.dim()):
            raise AssertionError(
                f"dim {self.dim} out of range for y with {y.dim()} dimensions"
            )
        if y.size(self.dim) != len(self.transforms):
            raise AssertionError(
                f"y.size({self.dim}) = {y.size(self.dim)} must equal len(transforms) {len(self.transforms)}"
            )
        logdetjacs = []
        yslices = self._slice(y)
        xslices = self._slice(x)
        for xslice, yslice, trans in zip(xslices, yslices, self.transforms):
            logdetjacs.append(trans.log_abs_det_jacobian(xslice, yslice))
        return torch.stack(logdetjacs, dim=self.dim)

    @property
    def bijective(self) -> bool:  # type: ignore[override]
        return all(t.bijective for t in self.transforms)

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def domain(self):
        return constraints.stack([t.domain for t in self.transforms], self.dim)

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def codomain(self):
        return constraints.stack([t.codomain for t in self.transforms], self.dim)


class CumulativeDistributionTransform(Transform):
    """
    Transform via the cumulative distribution function of a probability distribution.

    Args:
        distribution (Distribution): Distribution whose cumulative distribution function to use for
            the transformation.

    Example::

        # Construct a Gaussian copula from a multivariate normal.
        base_dist = MultivariateNormal(
            loc=torch.zeros(2),
            scale_tril=LKJCholesky(2).sample(),
        )
        transform = CumulativeDistributionTransform(Normal(0, 1))
        copula = TransformedDistribution(base_dist, [transform])
    """

    bijective = True
    codomain = constraints.unit_interval
    sign = +1

    def __init__(self, distribution: Distribution, cache_size: int = 0) -> None:
        super().__init__(cache_size=cache_size)
        self.distribution = distribution

    @property
    def domain(self) -> constraints.Constraint | None:  # type: ignore[override]
        return self.distribution.support

    def _call(self, x):
        return self.distribution.cdf(x)

    def _inverse(self, y):
        return self.distribution.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        return self.distribution.log_prob(x)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return CumulativeDistributionTransform(self.distribution, cache_size=cache_size)
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform


__all__ = ["InverseGamma"]


class InverseGamma(TransformedDistribution):
    r"""
    Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate`
    where::

        X ~ Gamma(concentration, rate)
        Y = 1 / X ~ InverseGamma(concentration, rate)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = InverseGamma(torch.tensor([2.0]), torch.tensor([3.0]))
        >>> m.sample()
        tensor([ 1.2953])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.positive
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Gamma

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        neg_one = -base_dist.rate.new_ones(())
        super().__init__(
            base_dist, PowerTransform(neg_one), validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def concentration(self) -> Tensor:
        return self.base_dist.concentration

    @property
    def rate(self) -> Tensor:
        return self.base_dist.rate

    @property
    def mean(self) -> Tensor:
        result = self.rate / (self.concentration - 1)
        return torch.where(self.concentration > 1, result, torch.inf)

    @property
    def mode(self) -> Tensor:
        return self.rate / (self.concentration + 1)

    @property
    def variance(self) -> Tensor:
        result = self.rate.square() / (
            (self.concentration - 1).square() * (self.concentration - 2)
        )
        return torch.where(self.concentration > 2, result, torch.inf)

    def entropy(self):
        return (
            self.concentration
            + self.rate.log()
            + self.concentration.lgamma()
            - (1 + self.concentration) * self.concentration.digamma()
        )
<|endoftext|># mypy: allow-untyped-defs
"""
This closely follows the implementation in NumPyro (https://github.com/pyro-ppl/numpyro).

Original copyright notice:

# Copyright: Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
"""

import math

import torch
from torch import Tensor
from torch.distributions import Beta, constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["LKJCholesky"]


class LKJCholesky(Distribution):
    r"""
    LKJ distribution for lower Cholesky factor of correlation matrices.
    The distribution is controlled by ``concentration`` parameter :math:`\eta`
    to make the probability of the correlation matrix :math:`M` generated from
    a Cholesky factor proportional to :math:`\det(M)^{\eta - 1}`. Because of that,
    when ``concentration == 1``, we have a uniform distribution over Cholesky
    factors of correlation matrices::

        L ~ LKJCholesky(dim, concentration)
        X = L @ L' ~ LKJCorr(dim, concentration)

    Note that this distribution samples the
    Cholesky factor of correlation matrices and not the correlation matrices
    themselves and thereby differs slightly from the derivations in [1] for
    the `LKJCorr` distribution. For sampling, this uses the Onion method from
    [1] Section 3.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> l = LKJCholesky(3, 0.5)
        >>> l.sample()  # l @ l.T is a sample of a correlation 3x3 matrix
        tensor([[ 1.0000,  0.0000,  0.0000],
                [ 0.3516,  0.9361,  0.0000],
                [-0.1899,  0.4748,  0.8593]])

    Args:
        dimension (dim): dimension of the matrices
        concentration (float or Tensor): concentration/shape parameter of the
            distribution (often referred to as eta)

    **References**

    [1] `Generating random correlation matrices based on vines and extended onion method` (2009),
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
    Journal of Multivariate Analysis. 100. 10.1016/j.jmva.2009.04.008
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"concentration": constraints.positive}
    support = constraints.corr_cholesky

    def __init__(
        self,
        dim: int,
        concentration: Tensor | float = 1.0,
        validate_args: bool | None = None,
    ) -> None:
        if dim < 2:
            raise ValueError(
                f"Expected dim to be an integer greater than or equal to 2. Found dim={dim}."
            )
        self.dim = dim
        (self.concentration,) = broadcast_all(concentration)
        batch_shape = self.concentration.size()
        event_shape = torch.Size((dim, dim))
        # This is used to draw vectorized samples from the beta distribution in Sec. 3.2 of [1].
        marginal_conc = self.concentration + 0.5 * (self.dim - 2)
        offset = torch.arange(
            self.dim - 1,
            dtype=self.concentration.dtype,
            device=self.concentration.device,
        )
        offset = torch.cat([offset.new_zeros((1,)), offset])
        beta_conc1 = offset + 0.5
        beta_conc0 = marginal_conc.unsqueeze(-1) - 0.5 * offset
        self._beta = Beta(beta_conc1, beta_conc0)
        super().__init__(batch_shape, event_shape, validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LKJCholesky, _instance)
        batch_shape = torch.Size(batch_shape)
        new.dim = self.dim
        new.concentration = self.concentration.expand(batch_shape)
        new._beta = self._beta.expand(batch_shape + (self.dim,))
        super(LKJCholesky, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # This uses the Onion method, but there are a few differences from [1] Sec. 3.2:
        # - This vectorizes the for loop and also works for heterogeneous eta.
        # - Same algorithm generalizes to n=1.
        # - The procedure is simplified since we are sampling the cholesky factor of
        #   the correlation matrix instead of the correlation matrix itself. As such,
        #   we only need to generate `w`.
        y = self._beta.sample(sample_shape).unsqueeze(-1)
        u_normal = torch.randn(
            self._extended_shape(sample_shape), dtype=y.dtype, device=y.device
        ).tril(-1)
        u_hypersphere = u_normal / u_normal.norm(dim=-1, keepdim=True)
        # Replace NaNs in first row
        u_hypersphere[..., 0, :].fill_(0.0)
        w = torch.sqrt(y) * u_hypersphere
        # Fill diagonal elements; clamp for numerical stability
        eps = torch.finfo(w.dtype).tiny
        diag_elems = torch.clamp(1 - torch.sum(w**2, dim=-1), min=eps).sqrt()
        w += torch.diag_embed(diag_elems)
        return w

    def log_prob(self, value):
        # See: https://mc-stan.org/docs/2_25/functions-reference/cholesky-lkj-correlation-distribution.html
        # The probability of a correlation matrix is proportional to
        #   determinant ** (concentration - 1) = prod(L_ii ^ 2(concentration - 1))
        # Additionally, the Jacobian of the transformation from Cholesky factor to
        # correlation matrix is:
        #   prod(L_ii ^ (D - i))
        # So the probability of a Cholesky factor is proportional to
        #   prod(L_ii ^ (2 * concentration - 2 + D - i)) = prod(L_ii ^ order_i)
        # with order_i = 2 * concentration - 2 + D - i
        if self._validate_args:
            self._validate_sample(value)
        diag_elems = value.diagonal(dim1=-1, dim2=-2)[..., 1:]
        order = torch.arange(2, self.dim + 1, device=self.concentration.device)
        order = 2 * (self.concentration - 1).unsqueeze(-1) + self.dim - order
        unnormalized_log_pdf = torch.sum(order * diag_elems.log(), dim=-1)
        # Compute normalization constant (page 1999 of [1])
        dm1 = self.dim - 1
        alpha = self.concentration + 0.5 * dm1
        denominator = torch.lgamma(alpha) * dm1
        numerator = torch.mvlgamma(alpha - 0.5, dm1)
        # pi_constant in [1] is D * (D - 1) / 4 * log(pi)
        # pi_constant in multigammaln is (D - 1) * (D - 2) / 4 * log(pi)
        # hence, we need to add a pi_constant = (D - 1) * log(pi) / 2
        pi_constant = 0.5 * dm1 * math.log(math.pi)
        normalize_term = pi_constant + numerator - denominator
        return unnormalized_log_pdf - normalize_term
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, Number


__all__ = ["Poisson"]


class Poisson(ExponentialFamily):
    r"""
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"rate": constraints.nonnegative}
    support = constraints.nonnegative_integer

    @property
    def mean(self) -> Tensor:
        return self.rate

    @property
    def mode(self) -> Tensor:
        return self.rate.floor()

    @property
    def variance(self) -> Tensor:
        return self.rate

    def __init__(
        self,
        rate: Tensor | Number,
        validate_args: bool | None = None,
    ) -> None:
        (self.rate,) = broadcast_all(rate)
        if isinstance(rate, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Poisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Poisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        rate, value = broadcast_all(self.rate, value)
        return value.xlogy(rate) - rate - (value + 1).lgamma()

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (torch.log(self.rate),)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        return torch.exp(x)
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch import inf, nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Cauchy"]


class Cauchy(Distribution):
    r"""
    Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
    independent normally distributed random variables with means `0` follows a
    Cauchy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
        tensor([ 2.3214])

    Args:
        loc (float or Tensor): mode or median of the distribution.
        scale (float or Tensor): half width at half maximum.
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
