    :exc:`~requests.exceptions.ReadTimeout` errors.
    """


class ConnectTimeout(ConnectionError, Timeout):
    """The request timed out while trying to connect to the remote server.

    Requests that produced this error are safe to retry.
    """


class ReadTimeout(Timeout):
    """The server did not send any data in the allotted amount of time."""


class URLRequired(RequestException):
    """A valid URL is required to make a request."""


class TooManyRedirects(RequestException):
    """Too many redirects."""


class MissingSchema(RequestException, ValueError):
    """The URL scheme (e.g. http or https) is missing."""


class InvalidSchema(RequestException, ValueError):
    """The URL scheme provided is either invalid or unsupported."""


class InvalidURL(RequestException, ValueError):
    """The URL provided was somehow invalid."""


class InvalidHeader(RequestException, ValueError):
    """The header value provided was somehow invalid."""


class InvalidProxyURL(InvalidURL):
    """The proxy URL provided is invalid."""


class ChunkedEncodingError(RequestException):
    """The server declared chunked encoding but sent an invalid chunk."""


class ContentDecodingError(RequestException, BaseHTTPError):
    """Failed to decode response content."""


class StreamConsumedError(RequestException, TypeError):
    """The content for this response was already consumed."""


class RetryError(RequestException):
    """Custom retries logic failed"""


class UnrewindableBodyError(RequestException):
    """Requests encountered an error when trying to rewind a body."""


# Warnings


class RequestsWarning(Warning):
    """Base warning for Requests."""


class FileModeWarning(RequestsWarning, DeprecationWarning):
    """A file was opened in text mode, but Requests determined its binary length."""


class RequestsDependencyWarning(RequestsWarning):
    """An imported dependency doesn't match the expected version range."""
<|endoftext|>from __future__ import annotations

import abc
import dataclasses
import enum
import itertools
import logging
import re
import typing

import chess
import chess.engine
import chess.svg

from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Set, TextIO, Tuple, Type, TypeVar, Optional, Union
from chess import Color, Square

if typing.TYPE_CHECKING:
    from typing_extensions import Self


LOGGER = logging.getLogger(__name__)


# Reference of Numeric Annotation Glyphs (NAGs):
# https://en.wikipedia.org/wiki/Numeric_Annotation_Glyphs

NAG_NULL = 0

NAG_GOOD_MOVE = 1
"""A good move. Can also be indicated by ``!`` in PGN notation."""

NAG_MISTAKE = 2
"""A mistake. Can also be indicated by ``?`` in PGN notation."""

NAG_BRILLIANT_MOVE = 3
"""A brilliant move. Can also be indicated by ``!!`` in PGN notation."""

NAG_BLUNDER = 4
"""A blunder. Can also be indicated by ``??`` in PGN notation."""

NAG_SPECULATIVE_MOVE = 5
"""A speculative move. Can also be indicated by ``!?`` in PGN notation."""

NAG_DUBIOUS_MOVE = 6
"""A dubious move. Can also be indicated by ``?!`` in PGN notation."""

NAG_FORCED_MOVE = 7
NAG_SINGULAR_MOVE = 8
NAG_WORST_MOVE = 9
NAG_DRAWISH_POSITION = 10
NAG_QUIET_POSITION = 11
NAG_ACTIVE_POSITION = 12
NAG_UNCLEAR_POSITION = 13
NAG_WHITE_SLIGHT_ADVANTAGE = 14
NAG_BLACK_SLIGHT_ADVANTAGE = 15
NAG_WHITE_MODERATE_ADVANTAGE = 16
NAG_BLACK_MODERATE_ADVANTAGE = 17
NAG_WHITE_DECISIVE_ADVANTAGE = 18
NAG_BLACK_DECISIVE_ADVANTAGE = 19

NAG_WHITE_ZUGZWANG = 22
NAG_BLACK_ZUGZWANG = 23

NAG_WHITE_MODERATE_COUNTERPLAY = 132
NAG_BLACK_MODERATE_COUNTERPLAY = 133
NAG_WHITE_DECISIVE_COUNTERPLAY = 134
NAG_BLACK_DECISIVE_COUNTERPLAY = 135
NAG_WHITE_MODERATE_TIME_PRESSURE = 136
NAG_BLACK_MODERATE_TIME_PRESSURE = 137
NAG_WHITE_SEVERE_TIME_PRESSURE = 138
NAG_BLACK_SEVERE_TIME_PRESSURE = 139

NAG_NOVELTY = 146


TAG_REGEX = re.compile(r"^\[([A-Za-z0-9][A-Za-z0-9_+#=:-]*)\s+\"([^\r]*)\"\]\s*$")

TAG_NAME_REGEX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+#=:-]*\Z")

MOVETEXT_REGEX = re.compile(r"""
    (
        [NBKRQ]?[a-h]?[1-8]?[\-x]?[a-h][1-8](?:=?[nbrqkNBRQK])?
        |[PNBRQK]?@[a-h][1-8]
        |--
        |Z0
        |0000
        |@@@@
        |O-O(?:-O)?
        |0-0(?:-0)?
    )
    |(\{.*)
    |(;.*)
    |(\$[0-9]+)
    |(\()
    |(\))
    |(\*|1-0|0-1|1/2-1/2)
    |([\?!]{1,2})
    """, re.DOTALL | re.VERBOSE)

SKIP_MOVETEXT_REGEX = re.compile(r""";|\{|\}""")


CLOCK_REGEX = re.compile(r"""(?P<prefix>\s?)\[%clk\s(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d*)?)\](?P<suffix>\s?)""")
EMT_REGEX = re.compile(r"""(?P<prefix>\s?)\[%emt\s(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d*)?)\](?P<suffix>\s?)""")

EVAL_REGEX = re.compile(r"""
    (?P<prefix>\s?)
    \[%eval\s(?:
        \#(?P<mate>[+-]?\d+)
        |(?P<cp>[+-]?(?:\d{0,10}\.\d{1,2}|\d{1,10}\.?))
    )(?:
        ,(?P<depth>\d+)
    )?\]
    (?P<suffix>\s?)
    """, re.VERBOSE)

ARROWS_REGEX = re.compile(r"""
    (?P<prefix>\s?)
    \[%(?:csl|cal)\s(?P<arrows>
        [RGYB][a-h][1-8](?:[a-h][1-8])?
        (?:,[RGYB][a-h][1-8](?:[a-h][1-8])?)*
    )\]
    (?P<suffix>\s?)
    """, re.VERBOSE)

def _condense_affix(infix: str) -> Callable[[typing.Match[str]], str]:
    def repl(match: typing.Match[str]) -> str:
        if infix:
            return match.group("prefix") + infix + match.group("suffix")
        else:
            return match.group("prefix") and match.group("suffix")
    return repl


TAG_ROSTER = ["Event", "Site", "Date", "Round", "White", "Black", "Result"]


class SkipType(enum.Enum):
    SKIP = None

SKIP = SkipType.SKIP


ResultT = TypeVar("ResultT", covariant=True)


class TimeControlType(enum.Enum):
    UNKNOWN = 0
    UNLIMITED = 1
    STANDARD = 2
    RAPID = 3
    BLITZ = 4
    BULLET = 5


@dataclasses.dataclass
class TimeControlPart:
    moves: int = 0
    time: int = 0
    increment: float = 0
    delay: float = 0


@dataclasses.dataclass
class TimeControl:
    """
    PGN TimeControl Parser
    Spec: http://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm#c9.6

    Not Yet Implemented:
    - Hourglass/Sandclock ('*' prefix)
    - Differentiating between Bronstein and Simple Delay (Not part of the PGN Spec)
      - More Info: https://en.wikipedia.org/wiki/Chess_clock#Timing_methods
    """

    parts: list[TimeControlPart] = dataclasses.field(default_factory=list)
    type: TimeControlType = TimeControlType.UNKNOWN


class _AcceptFrame:
    def __init__(self, node: ChildNode, *, is_variation: bool = False, sidelines: bool = True):
        self.state = "pre"
        self.node = node
        self.is_variation = is_variation
        self.variations = iter(itertools.islice(node.parent.variations, 1, None) if sidelines else [])
        self.in_variation = False


class GameNode(abc.ABC):
    parent: Optional[GameNode]
    """The parent node or ``None`` if this is the root node of the game."""

    move: Optional[chess.Move]
    """
    The move leading to this node or ``None`` if this is the root node of the
    game.
    """

    variations: List[ChildNode]
    """A list of child nodes."""

    comment: str
    """
    A comment that goes behind the move leading to this node. Comments
    that occur before any moves are assigned to the root node.
    """

    starting_comment: str
    nags: Set[int]

    def __init__(self, *, comment: str = "") -> None:
        self.parent = None
        self.move = None
        self.variations = []
        self.comment = comment

        # Deprecated: These should be properties of ChildNode, but need to
        # remain here for backwards compatibility.
        self.starting_comment = ""
        self.nags = set()

    @abc.abstractmethod
    def board(self) -> chess.Board:
        """
        Gets a board with the position of the node.

        For the root node, this is the default starting position (for the
        ``Variant``) unless the ``FEN`` header tag is set.

        It's a copy, so modifying the board will not alter the game.

        Complexity is `O(n)`.
        """

    @abc.abstractmethod
    def ply(self) -> int:
        """
        Returns the number of half-moves up to this node, as indicated by
        fullmove number and turn of the position.
        See :func:`chess.Board.ply()`.

        Usually this is equal to the number of parent nodes, but it may be
        more if the game was started from a custom position.

        Complexity is `O(n)`.
        """

    def turn(self) -> Color:
        """
        Gets the color to move at this node. See :data:`chess.Board.turn`.

        Complexity is `O(n)`.
        """
        return self.ply() % 2 == 0

    def root(self) -> GameNode:
        node = self
        while node.parent:
            node = node.parent
        return node

    def game(self) -> Game:
        """
        Gets the root node, i.e., the game.

        Complexity is `O(n)`.
        """
        root = self.root()
        assert isinstance(root, Game), "GameNode not rooted in Game"
        return root

    def end(self) -> GameNode:
        """
        Follows the main variation to the end and returns the last node.

        Complexity is `O(n)`.
        """
        node = self

        while node.variations:
            node = node.variations[0]

        return node

    def is_end(self) -> bool:
        """
        Checks if this node is the last node in the current variation.

        Complexity is `O(1)`.
        """
        return not self.variations

    def starts_variation(self) -> bool:
        """
        Checks if this node starts a variation (and can thus have a starting
        comment). The root node does not start a variation and can have no
        starting comment.

        For example, in ``1. e4 e5 (1... c5 2. Nf3) 2. Nf3``, the node holding
        1... c5 starts a variation.

        Complexity is `O(1)`.
        """
        if not self.parent or not self.parent.variations:
            return False

        return self.parent.variations[0] != self

    def is_mainline(self) -> bool:
        """
        Checks if the node is in the mainline of the game.

        Complexity is `O(n)`.
        """
        node = self

        while node.parent:
            parent = node.parent

            if not parent.variations or parent.variations[0] != node:
                return False

            node = parent

        return True

    def is_main_variation(self) -> bool:
        """
        Checks if this node is the first variation from the point of view of its
        parent. The root node is also in the main variation.

        Complexity is `O(1)`.
        """
        if not self.parent:
            return True

        return not self.parent.variations or self.parent.variations[0] == self

    def __getitem__(self, move: Union[int, chess.Move, GameNode]) -> ChildNode:
        try:
            return self.variations[move]  # type: ignore
        except TypeError:
            for variation in self.variations:
                if variation.move == move or variation == move:
                    return variation

        raise KeyError(move)

    def __contains__(self, move: Union[int, chess.Move, GameNode]) -> bool:
        try:
            self[move]
        except KeyError:
            return False
        else:
            return True

    def variation(self, move: Union[int, chess.Move, GameNode]) -> ChildNode:
        """
        Gets a child node by either the move or the variation index.
        """
        return self[move]

    def has_variation(self, move: Union[int, chess.Move, GameNode]) -> bool:
        """Checks if this node has the given variation."""
        return move in self

    def promote_to_main(self, move: Union[int, chess.Move, GameNode]) -> None:
        """Promotes the given *move* to the main variation."""
        variation = self[move]
        self.variations.remove(variation)
        self.variations.insert(0, variation)

    def promote(self, move: Union[int, chess.Move, GameNode]) -> None:
        """Moves a variation one up in the list of variations."""
        variation = self[move]
        i = self.variations.index(variation)
        if i > 0:
            self.variations[i - 1], self.variations[i] = self.variations[i], self.variations[i - 1]

    def demote(self, move: Union[int, chess.Move, GameNode]) -> None:
        """Moves a variation one down in the list of variations."""
        variation = self[move]
        i = self.variations.index(variation)
        if i < len(self.variations) - 1:
            self.variations[i + 1], self.variations[i] = self.variations[i], self.variations[i + 1]

    def remove_variation(self, move: Union[int, chess.Move, GameNode]) -> None:
        """Removes a variation."""
        self.variations.remove(self.variation(move))

    def add_variation(self, move: chess.Move, *, comment: str = "", starting_comment: str = "", nags: Iterable[int] = []) -> ChildNode:
        """Creates a child node with the given attributes."""
        # Instanciate ChildNode only in this method.
        return ChildNode(self, move, comment=comment, starting_comment=starting_comment, nags=nags)

    def add_main_variation(self, move: chess.Move, *, comment: str = "", nags: Iterable[int] = []) -> ChildNode:
        """
        Creates a child node with the given attributes and promotes it to the
        main variation.
        """
        node = self.add_variation(move, comment=comment, nags=nags)
        self.variations.insert(0, self.variations.pop())
        return node

    def next(self) -> Optional[ChildNode]:
        """
        Returns the first node of the mainline after this node, or ``None`` if
        this node does not have any children.

        Complexity is `O(1)`.
        """
        return self.variations[0] if self.variations else None

    def mainline(self) -> Mainline[ChildNode]:
        """Returns an iterable over the mainline starting after this node."""
        return Mainline(self, lambda node: node)

    def mainline_moves(self) -> Mainline[chess.Move]:
        """Returns an iterable over the main moves after this node."""
        return Mainline(self, lambda node: node.move)

    def add_line(self, moves: Iterable[chess.Move], *, comment: str = "", starting_comment: str = "", nags: Iterable[int] = []) -> GameNode:
        """
        Creates a sequence of child nodes for the given list of moves.
        Adds *comment* and *nags* to the last node of the line and returns it.
        """
        node = self

        # Add line.
        for move in moves:
            node = node.add_variation(move, starting_comment=starting_comment)
            starting_comment = ""

        # Merge comment and NAGs.
        if node.comment:
            node.comment += " " + comment
        else:
            node.comment = comment

        node.nags.update(nags)

        return node

    def eval(self) -> Optional[chess.engine.PovScore]:
        """
        Parses the first valid ``[%eval ...]`` annotation in the comment of
        this node, if any.

        Complexity is `O(n)`.
        """
        match = EVAL_REGEX.search(self.comment)
        if not match:
            return None

        turn = self.turn()

        if match.group("mate"):
            mate = int(match.group("mate"))
            score: chess.engine.Score = chess.engine.Mate(mate)
            if mate == 0:
                # Resolve this ambiguity in the specification in favor of
                # standard chess: The player to move after mate is the player
                # who has been mated.
                return chess.engine.PovScore(score, turn)
        else:
            score = chess.engine.Cp(round(float(match.group("cp")) * 100))

        return chess.engine.PovScore(score if turn else -score, turn)

    def eval_depth(self) -> Optional[int]:
        """
        Parses the first valid ``[%eval ...]`` annotation in the comment of
        this node and returns the corresponding depth, if any.

        Complexity is `O(1)`.
        """
        match = EVAL_REGEX.search(self.comment)
        return int(match.group("depth")) if match and match.group("depth") else None

    def set_eval(self, score: Optional[chess.engine.PovScore], depth: Optional[int] = None) -> None:
        """
        Replaces the first valid ``[%eval ...]`` annotation in the comment of
        this node or adds a new one.
        """
        eval = ""
        if score is not None:
            depth_suffix = "" if depth is None else f",{max(depth, 0):d}"
            cp = score.white().score()
            if cp is not None:
                eval = f"[%eval {float(cp) / 100:.2f}{depth_suffix}]"
            elif score.white().mate():
                eval = f"[%eval #{score.white().mate()}{depth_suffix}]"

        self.comment, found = EVAL_REGEX.subn(_condense_affix(eval), self.comment, count=1)

        if not found and eval:
            if self.comment and not self.comment.endswith(" "):
                self.comment += " "
            self.comment += eval

    def arrows(self) -> List[chess.svg.Arrow]:
        """
        Parses all ``[%csl ...]`` and ``[%cal ...]`` annotations in the comment
        of this node.

        Returns a list of :class:`arrows <chess.svg.Arrow>`.
        """
        arrows = []
        for match in ARROWS_REGEX.finditer(self.comment):
            for group in match.group("arrows").split(","):
                arrows.append(chess.svg.Arrow.from_pgn(group))

        return arrows

    def set_arrows(self, arrows: Iterable[Union[chess.svg.Arrow, Tuple[Square, Square]]]) -> None:
        """
        Replaces all valid ``[%csl ...]`` and ``[%cal ...]`` annotations in
        the comment of this node or adds new ones.
        """
        csl: List[str] = []
        cal: List[str] = []

        for arrow in arrows:
            try:
                tail, head = arrow  # type: ignore
                arrow = chess.svg.Arrow(tail, head)
            except TypeError:
                pass
            (csl if arrow.tail == arrow.head else cal).append(arrow.pgn())  # type: ignore

        self.comment = ARROWS_REGEX.sub(_condense_affix(""), self.comment)

        prefix = ""
        if csl:
            prefix += f"[%csl {','.join(csl)}]"
        if cal:
            prefix += f"[%cal {','.join(cal)}]"

        if prefix and self.comment and not self.comment.startswith(" ") and not self.comment.startswith("\n"):
            self.comment = prefix + " " + self.comment
        else:
            self.comment = prefix + self.comment

    def clock(self) -> Optional[float]:
        """
        Parses the first valid ``[%clk ...]`` annotation in the comment of
        this node, if any.

        Returns the player's remaining time to the next time control after this
        move, in seconds.
        """
        match = CLOCK_REGEX.search(self.comment)
        if match is None:
            return None
        return int(match.group("hours")) * 3600 + int(match.group("minutes")) * 60 + float(match.group("seconds"))

    def set_clock(self, seconds: Optional[float]) -> None:
        """
        Replaces the first valid ``[%clk ...]`` annotation in the comment of
        this node or adds a new one.
        """
        clk = ""
        if seconds is not None:
            seconds = max(0, seconds)
            hours = int(seconds // 3600)
            minutes = int(seconds % 3600 // 60)
            seconds = seconds % 3600 % 60
            seconds_part = f"{seconds:06.3f}".rstrip("0").rstrip(".")
            clk = f"[%clk {hours:d}:{minutes:02d}:{seconds_part}]"

        self.comment, found = CLOCK_REGEX.subn(_condense_affix(clk), self.comment, count=1)

        if not found and clk:
            if self.comment and not self.comment.endswith(" ") and not self.comment.endswith("\n"):
                self.comment += " "
            self.comment += clk

    def emt(self) -> Optional[float]:
        """
        Parses the first valid ``[%emt ...]`` annotation in the comment of
        this node, if any.

        Returns the player's elapsed move time use for the comment of this
        move, in seconds.
        """
        match = EMT_REGEX.search(self.comment)
        if match is None:
            return None
        return int(match.group("hours")) * 3600 + int(match.group("minutes")) * 60 + float(match.group("seconds"))

    def set_emt(self, seconds: Optional[float]) -> None:
        """
        Replaces the first valid ``[%emt ...]`` annotation in the comment of
        this node or adds a new one.
        """
        emt = ""
        if seconds is not None:
            seconds = max(0, seconds)
            hours = int(seconds // 3600)
            minutes = int(seconds % 3600 // 60)
            seconds = seconds % 3600 % 60
            seconds_part = f"{seconds:06.3f}".rstrip("0").rstrip(".")
            emt = f"[%emt {hours:d}:{minutes:02d}:{seconds_part}]"

        self.comment, found = EMT_REGEX.subn(_condense_affix(emt), self.comment, count=1)

        if not found and emt:
            if self.comment and not self.comment.endswith(" ") and not self.comment.endswith("\n"):
                self.comment += " "
            self.comment += emt

    @abc.abstractmethod
    def accept(self, visitor: BaseVisitor[ResultT]) -> ResultT:
        """
        Traverses game nodes in PGN order using the given *visitor*. Starts with
        the move leading to this node. Returns the *visitor* result.
        """

    def accept_subgame(self, visitor: BaseVisitor[ResultT]) -> ResultT:
        """
        Traverses headers and game nodes in PGN order, as if the game was
        starting after this node. Returns the *visitor* result.
        """
        if visitor.begin_game() is not SKIP:
            game = self.game()
            board = self.board()

            dummy_game = Game.without_tag_roster()
            dummy_game.setup(board)

            visitor.begin_headers()

            for tagname, tagvalue in game.headers.items():
                if tagname not in dummy_game.headers:
                    visitor.visit_header(tagname, tagvalue)
            for tagname, tagvalue in dummy_game.headers.items():
                visitor.visit_header(tagname, tagvalue)

            if visitor.end_headers() is not SKIP:
                visitor.visit_board(board)

                if self.variations:
                    self.variations[0]._accept(board, visitor)

                visitor.visit_result(game.headers.get("Result", "*"))

        visitor.end_game()
        return visitor.result()

    def __str__(self) -> str:
        return self.accept(StringExporter(columns=None))


class ChildNode(GameNode):
    """
    A child node of a game, with the move leading to it.
    Extends :class:`~chess.pgn.GameNode`.
    """

    parent: GameNode
    """The parent node."""

    move: chess.Move
    """The move leading to this node."""

    starting_comment: str
    """
    A comment for the start of a variation. Only nodes that
    actually start a variation (:func:`~chess.pgn.GameNode.starts_variation()`
    checks this) can have a starting comment. The root node can not have
    a starting comment.
    """

    nags: Set[int]
    """
    A set of NAGs as integers. NAGs always go behind a move, so the root
    node of the game will never have NAGs.
    """

    def __init__(self, parent: GameNode, move: chess.Move, *, comment: str = "", starting_comment: str = "", nags: Iterable[int] = []) -> None:
        super().__init__(comment=comment)
        self.parent = parent
        self.move = move
        self.parent.variations.append(self)

        self.nags.update(nags)
        self.starting_comment = starting_comment

    def board(self) -> chess.Board:
        stack: List[chess.Move] = []
        node: GameNode = self

        while node.move is not None and node.parent is not None:
            stack.append(node.move)
            node = node.parent

        board = node.game().board()

        while stack:
            board.push(stack.pop())

        return board

    def ply(self) -> int:
        ply = 0
        node: GameNode = self
        while node.parent is not None:
            ply += 1
            node = node.parent
        return node.game().ply() + ply

    def san(self) -> str:
        """
        Gets the standard algebraic notation of the move leading to this node.
        See :func:`chess.Board.san()`.

        Do not call this on the root node.

        Complexity is `O(n)`.
        """
        return self.parent.board().san(self.move)

    def uci(self, *, chess960: Optional[bool] = None) -> str:
        """
        Gets the UCI notation of the move leading to this node.
        See :func:`chess.Board.uci()`.

        Do not call this on the root node.

        Complexity is `O(n)`.
        """
        return self.parent.board().uci(self.move, chess960=chess960)

    def end(self) -> ChildNode:
        """
        Follows the main variation to the end and returns the last node.

        Complexity is `O(n)`.
        """
        return typing.cast(ChildNode, super().end())

    def _accept_node(self, parent_board: chess.Board, visitor: BaseVisitor[ResultT]) -> None:
        if self.starting_comment:
            visitor.visit_comment(self.starting_comment)

        visitor.visit_move(parent_board, self.move)

        parent_board.push(self.move)
        visitor.visit_board(parent_board)
        parent_board.pop()

        for nag in sorted(self.nags):
            visitor.visit_nag(nag)

        if self.comment:
            visitor.visit_comment(self.comment)

    def _accept(self, parent_board: chess.Board, visitor: BaseVisitor[ResultT], *, sidelines: bool = True) -> None:
        stack = [_AcceptFrame(self, sidelines=sidelines)]

        while stack:
            top = stack[-1]

            if top.in_variation:
                top.in_variation = False
                visitor.end_variation()

            if top.state == "pre":
                top.node._accept_node(parent_board, visitor)
                top.state = "variations"
            elif top.state == "variations":
                try:
                    variation = next(top.variations)
                except StopIteration:
                    if top.node.variations:
                        parent_board.push(top.node.move)
                        stack.append(_AcceptFrame(top.node.variations[0], sidelines=True))
                        top.state = "post"
                    else:
                        top.state = "end"
                else:
                    if visitor.begin_variation() is not SKIP:
                        stack.append(_AcceptFrame(variation, sidelines=False, is_variation=True))
                    top.in_variation = True
            elif top.state == "post":
                parent_board.pop()
                top.state = "end"
            else:
                stack.pop()

    def accept(self, visitor: BaseVisitor[ResultT]) -> ResultT:
        self._accept(self.parent.board(), visitor, sidelines=False)
        return visitor.result()

    def __repr__(self) -> str:
        try:
            parent_board = self.parent.board()
        except ValueError:
            return f"<{type(self).__name__} at {id(self):#x} (dangling: {self.move})>"
        else:
            return "<{} at {:#x} ({}{} {} ...)>".format(
                type(self).__name__,
                id(self),
                parent_board.fullmove_number,
                "." if parent_board.turn == chess.WHITE else "...",
                parent_board.san(self.move))


GameT = TypeVar("GameT", bound="Game")

class Game(GameNode):
    """
    The root node of a game with extra information such as headers and the
    starting position. Extends :class:`~chess.pgn.GameNode`.
    """

    headers: Headers
    """
    A mapping of headers. By default, the following 7 headers are provided
    (Seven Tag Roster):

    >>> import chess.pgn
    >>>
    >>> game = chess.pgn.Game()
    >>> game.headers
    Headers(Event='?', Site='?', Date='????.??.??', Round='?', White='?', Black='?', Result='*')
    """

    errors: List[Exception]
    """
    A list of errors (such as illegal or ambiguous moves) encountered while
    parsing the game.
    """

    def __init__(self, headers: Optional[Union[Mapping[str, str], Iterable[Tuple[str, str]]]] = None) -> None:
        super().__init__()
        self.headers = Headers(headers)
        self.errors = []

    def board(self) -> chess.Board:
        return self.headers.board()

    def ply(self) -> int:
        # Optimization: Parse FEN only for custom starting positions.
        return self.board().ply() if "FEN" in self.headers else 0

    def setup(self, board: Union[chess.Board, str]) -> None:
        """
        Sets up a specific starting position. This sets (or resets) the
        ``FEN``, ``SetUp``, and ``Variant`` header tags.
        """
        try:
            fen = board.fen()  # type: ignore
            setup = typing.cast(chess.Board, board)
        except AttributeError:
            setup = chess.Board(board)  # type: ignore
            setup.chess960 = setup.has_chess960_castling_rights()
            fen = setup.fen()

        if fen == type(setup).starting_fen:
            self.headers.pop("FEN", None)
            self.headers.pop("SetUp", None)
        else:
            self.headers["FEN"] = fen
            self.headers["SetUp"] = "1"

        if type(setup).aliases[0] == "Standard" and setup.chess960:
            self.headers["Variant"] = "Chess960"
        elif type(setup).aliases[0] != "Standard":
            self.headers["Variant"] = type(setup).aliases[0]
            self.headers["FEN"] = fen
        else:
            self.headers.pop("Variant", None)

    def accept(self, visitor: BaseVisitor[ResultT]) -> ResultT:
        """
        Traverses the game in PGN order using the given *visitor*. Returns
        the *visitor* result.
        """
        if visitor.begin_game() is not SKIP:
            for tagname, tagvalue in self.headers.items():
                visitor.visit_header(tagname, tagvalue)
            if visitor.end_headers() is not SKIP:
                board = self.board()
                visitor.visit_board(board)

                if self.comment:
                    visitor.visit_comment(self.comment)

                if self.variations:
                    self.variations[0]._accept(board, visitor)

                visitor.visit_result(self.headers.get("Result", "*"))

        visitor.end_game()
        return visitor.result()

    def time_control(self) -> TimeControl:
        """
        Returns the time control of the game. If the game has no time control
        information, the default time control ('UNKNOWN') is returned.
        """
        time_control_header = self.headers.get("TimeControl", "")
        return parse_time_control(time_control_header)

    @classmethod
    def from_board(cls: Type[GameT], board: chess.Board) -> GameT:
        """Creates a game from the move stack of a :class:`~chess.Board()`."""
        # Setup the initial position.
        game = cls()
        game.setup(board.root())
        node: GameNode = game

        # Replay all moves.
        for move in board.move_stack:
            node = node.add_variation(move)

        game.headers["Result"] = board.result()
        return game

    @classmethod
    def without_tag_roster(cls: Type[GameT]) -> GameT:
        """Creates an empty game without the default Seven Tag Roster."""
        return cls(headers={})

    @classmethod
    def builder(cls: Type[GameT]) -> GameBuilder[GameT]:
        return GameBuilder(Game=cls)

    def __repr__(self) -> str:
        return "<{} at {:#x} ({!r} vs. {!r}, {!r} at {!r}{})>".format(
            type(self).__name__,
            id(self),
            self.headers.get("White", "?"),
            self.headers.get("Black", "?"),
            self.headers.get("Date", "????.??.??"),
            self.headers.get("Site", "?"),
            f", {len(self.errors)} errors" if self.errors else "")


HeadersT = TypeVar("HeadersT", bound="Headers")

class Headers(MutableMapping[str, str]):
    def __init__(self, data: Optional[Union[Mapping[str, str], Iterable[Tuple[str, str]]]] = None, **kwargs: str) -> None:
        self._tag_roster: Dict[str, str] = {}
        self._others: Dict[str, str] = {}

        if data is None:
            data = {
                "Event": "?",
                "Site": "?",
                "Date": "????.??.??",
                "Round": "?",
                "White": "?",
                "Black": "?",
                "Result": "*"
            }

        self.update(data, **kwargs)

    def is_chess960(self) -> bool:
        return self.get("Variant", "").lower() in [
            "chess960",
            "chess 960",
            "fischerandom",  # Cute Chess
            "fischerrandom",
            "fischer random",
        ]

    def is_wild(self) -> bool:
        # http://www.freechess.org/Help/HelpFiles/wild.html
        return self.get("Variant", "").lower() in [
            "wild/0", "wild/1", "wild/2", "wild/3", "wild/4", "wild/5",
            "wild/6", "wild/7", "wild/8", "wild/8a"]

    def variant(self) -> Type[chess.Board]:
        if "Variant" not in self or self.is_chess960() or self.is_wild():
            return chess.Board
        else:
            from chess.variant import find_variant
            return find_variant(self["Variant"])

    def board(self) -> chess.Board:
        VariantBoard = self.variant()
        fen = self.get("FEN", VariantBoard.starting_fen)
        board = VariantBoard(fen, chess960=self.is_chess960())
        board.chess960 = board.chess960 or board.has_chess960_castling_rights()
        return board

    def __setitem__(self, key: str, value: str) -> None:
        if key in TAG_ROSTER:
            self._tag_roster[key] = value
        elif not TAG_NAME_REGEX.match(key):
            raise ValueError(f"invalid pgn header tag: {key!r}")
        elif "\n" in value or "\r" in value:
            raise ValueError(f"line break in pgn header {key}: {value!r}")
        else:
            self._others[key] = value

    def __getitem__(self, key: str) -> str:
        return self._tag_roster[key] if key in TAG_ROSTER else self._others[key]

    def __delitem__(self, key: str) -> None:
        if key in TAG_ROSTER:
            del self._tag_roster[key]
        else:
            del self._others[key]

    def __iter__(self) -> Iterator[str]:
        for key in TAG_ROSTER:
            if key in self._tag_roster:
                yield key

        yield from self._others

    def __len__(self) -> int:
        return len(self._tag_roster) + len(self._others)

    def copy(self) -> Self:
        return type(self)(self)

    def __copy__(self) -> Self:
        return self.copy()

    def __repr__(self) -> str:
        return "{}({})".format(
            type(self).__name__,
            ", ".join("{}={!r}".format(key, value) for key, value in self.items()))

    @classmethod
    def builder(cls: Type[HeadersT]) -> HeadersBuilder[HeadersT]:
        return HeadersBuilder(Headers=cls)


MainlineMapT = TypeVar("MainlineMapT")

class Mainline(Generic[MainlineMapT]):
    def __init__(self, start: GameNode, f: Callable[[ChildNode], MainlineMapT]) -> None:
        self.start = start
        self.f = f

    def __bool__(self) -> bool:
        return bool(self.start.variations)

    def __iter__(self) -> Iterator[MainlineMapT]:
        node = self.start
        while node.variations:
            node = node.variations[0]
            yield self.f(node)

    def __reversed__(self) -> Iterator[MainlineMapT]:
        node = self.start.end()
        while node.parent and node != self.start:
            yield self.f(typing.cast(ChildNode, node))
            node = node.parent

    def accept(self, visitor: BaseVisitor[ResultT]) -> ResultT:
        node = self.start
        board = self.start.board()
        while node.variations:
            node = node.variations[0]
            node._accept_node(board, visitor)
            board.push(node.move)
        return visitor.result()

    def __str__(self) -> str:
        return self.accept(StringExporter(columns=None))

    def __repr__(self) -> str:
        return f"<Mainline at {id(self):#x} ({self.accept(StringExporter(columns=None, comments=False))})>"


class BaseVisitor(abc.ABC, Generic[ResultT]):
    """
    Base class for visitors.

    Use with :func:`chess.pgn.Game.accept()` or
    :func:`chess.pgn.GameNode.accept()` or :func:`chess.pgn.read_game()`.

    The methods are called in PGN order.
    """

    def begin_game(self) -> Optional[SkipType]:
        """Called at the start of a game."""
        pass

    def begin_headers(self) -> Optional[Headers]:
        """Called before visiting game headers."""
        pass

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        """Called for each game header."""
        pass

    def end_headers(self) -> Optional[SkipType]:
        """Called after visiting game headers."""
        pass

    def begin_parse_san(self, board: chess.Board, san: str) -> Optional[SkipType]:
        """
        When the visitor is used by a parser, this is called at the start of
        each standard algebraic notation detailing a move.
        """
        pass

    def parse_san(self, board: chess.Board, san: str) -> chess.Move:
        """
        When the visitor is used by a parser, this is called to parse a move
        in standard algebraic notation.

        You can override the default implementation to work around specific
        quirks of your input format.

        .. deprecated:: 1.1
            This method is very limited, because it is only called on moves
            that the parser recognizes in the first place. Instead of adding
            workarounds here, please report common quirks so that
            they can be handled for everyone.
        """
        return board.parse_san(san)

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        """
        Called for each move.

        *board* is the board state before the move. The board state must be
        restored before the traversal continues.
        """
        pass

    def visit_board(self, board: chess.Board) -> None:
        """
        Called for the starting position of the game and after each move.

        The board state must be restored before the traversal continues.
        """
        pass

    def visit_comment(self, comment: str) -> None:
        """Called for each comment."""
        pass

    def visit_nag(self, nag: int) -> None:
        """Called for each NAG."""
        pass

    def begin_variation(self) -> Optional[SkipType]:
        """
        Called at the start of a new variation. It is not called for the
        mainline of the game.
        """
        pass

    def end_variation(self) -> None:
        """Concludes a variation."""
        pass

    def visit_result(self, result: str) -> None:
        """
        Called at the end of a game with the value from the ``Result`` header.
        """
        pass

    def end_game(self) -> None:
        """Called at the end of a game."""
        pass

    @abc.abstractmethod
    def result(self) -> ResultT:
        """Called to get the result of the visitor."""

    def handle_error(self, error: Exception) -> None:
        """Called for encountered errors. Defaults to raising an exception."""
        raise error


class GameBuilder(BaseVisitor[GameT]):
    """
    Creates a game model. Default visitor for :func:`~chess.pgn.read_game()`.
    """

    @typing.overload
    def __init__(self: GameBuilder[Game]) -> None: ...
    @typing.overload
    def __init__(self: GameBuilder[GameT], *, Game: Type[GameT]) -> None: ...
    def __init__(self, *, Game: Any = Game) -> None:
        self.Game = Game

    def begin_game(self) -> None:
        self.game: GameT = self.Game()

        self.variation_stack: List[GameNode] = [self.game]
        self.starting_comment = ""
        self.in_variation = False

    def begin_headers(self) -> Headers:
        return self.game.headers

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.game.headers[tagname] = tagvalue

    def visit_nag(self, nag: int) -> None:
        self.variation_stack[-1].nags.add(nag)

    def begin_variation(self) -> None:
        parent = self.variation_stack[-1].parent
        assert parent is not None, "begin_variation called, but root node on top of stack"
        self.variation_stack.append(parent)
        self.in_variation = False

    def end_variation(self) -> None:
        self.variation_stack.pop()

    def visit_result(self, result: str) -> None:
        if self.game.headers.get("Result", "*") == "*":
            self.game.headers["Result"] = result

    def visit_comment(self, comment: str) -> None:
        if self.in_variation or (self.variation_stack[-1].parent is None and self.variation_stack[-1].is_end()):
            # Add as a comment for the current node if in the middle of
            # a variation. Add as a comment for the game if the comment
            # starts before any move.
            new_comment = [self.variation_stack[-1].comment, comment]
            self.variation_stack[-1].comment = " ".join(filter(None, new_comment))
        else:
            # Otherwise, it is a starting comment.
            new_comment = [self.starting_comment, comment]
            self.starting_comment = " ".join(filter(None, new_comment))

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self.variation_stack[-1] = self.variation_stack[-1].add_variation(move)
        self.variation_stack[-1].starting_comment = self.starting_comment
        self.starting_comment = ""
        self.in_variation = True

    def handle_error(self, error: Exception) -> None:
        """
        Populates :data:`chess.pgn.Game.errors` with encountered errors and
        logs them.

        You can silence the log and handle errors yourself after parsing:

        >>> import chess.pgn
        >>> import logging
        >>>
        >>> logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)
        >>>
        >>> pgn = open("data/pgn/kasparov-deep-blue-1997.pgn")
        >>>
        >>> game = chess.pgn.read_game(pgn)
        >>> game.errors  # List of exceptions
        []

        You can also override this method to hook into error handling:

        >>> import chess.pgn
        >>>
        >>> class MyGameBuilder(chess.pgn.GameBuilder):
        >>>     def handle_error(self, error: Exception) -> None:
        >>>         pass  # Ignore error
        >>>
        >>> pgn = open("data/pgn/kasparov-deep-blue-1997.pgn")
        >>>
        >>> game = chess.pgn.read_game(pgn, Visitor=MyGameBuilder)
        """
        LOGGER.error("%s while parsing %r", error, self.game)
        self.game.errors.append(error)

    def result(self) -> GameT:
        """
        Returns the visited :class:`~chess.pgn.Game()`.
        """
        return self.game


class HeadersBuilder(BaseVisitor[HeadersT]):
    """Collects headers into a dictionary."""

    @typing.overload
    def __init__(self: HeadersBuilder[Headers]) -> None: ...
    @typing.overload
    def __init__(self: HeadersBuilder[HeadersT], *, Headers: Type[Headers]) -> None: ...
    def __init__(self, *, Headers: Any = Headers) -> None:
        self.Headers = Headers

    def begin_headers(self) -> HeadersT:
        self.headers: HeadersT = self.Headers({})
        return self.headers

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.headers[tagname] = tagvalue

    def end_headers(self) -> SkipType:
        return SKIP

    def result(self) -> HeadersT:
        return self.headers


class BoardBuilder(BaseVisitor[chess.Board]):
    """
    Returns the final position of the game. The mainline of the game is
    on the move stack.
    """

    def begin_game(self) -> None:
        self.skip_variation_depth = 0

    def begin_variation(self) -> SkipType:
        self.skip_variation_depth += 1
        return SKIP

    def end_variation(self) -> None:
        self.skip_variation_depth = max(self.skip_variation_depth - 1, 0)

    def visit_board(self, board: chess.Board) -> None:
        if not self.skip_variation_depth:
            self.board = board

    def result(self) -> chess.Board:
        return self.board


class SkipVisitor(BaseVisitor[Literal[True]]):
    """Skips a game."""

    def begin_game(self) -> SkipType:
        return SKIP

    def end_headers(self) -> SkipType:
        return SKIP

    def begin_variation(self) -> SkipType:
        return SKIP

    def result(self) -> Literal[True]:
        return True


class StringExporterMixin:
    def __init__(self, *, columns: Optional[int] = 80, headers: bool = True, comments: bool = True, variations: bool = True):
        self.columns = columns
        self.headers = headers
        self.comments = comments
        self.variations = variations

        self.found_headers = False

        self.force_movenumber = True

        self.lines: List[str] = []
        self.current_line = ""
        self.variation_depth = 0

    def flush_current_line(self) -> None:
        if self.current_line:
            self.lines.append(self.current_line.rstrip())
        self.current_line = ""

    def write_token(self, token: str) -> None:
        if self.columns is not None and self.columns - len(self.current_line) < len(token):
            self.flush_current_line()
        self.current_line += token

    def write_line(self, line: str = "") -> None:
        self.flush_current_line()
        self.lines.append(line.rstrip())

    def end_game(self) -> None:
        self.write_line()

    def begin_headers(self) -> None:
        self.found_headers = False

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        if self.headers:
            self.found_headers = True
            self.write_line(f"[{tagname} \"{tagvalue}\"]")

    def end_headers(self) -> None:
        if self.found_headers:
            self.write_line()

    def begin_variation(self) -> Optional[SkipType]:
        self.variation_depth += 1

        if self.variations:
            self.write_token("( ")
            self.force_movenumber = True
            return None
        else:
            return SKIP

    def end_variation(self) -> None:
        self.variation_depth -= 1

        if self.variations:
            self.write_token(") ")
            self.force_movenumber = True

    def visit_comment(self, comment: str) -> None:
        if self.comments and (self.variations or not self.variation_depth):
            self.write_token("{ " + comment.replace("}", "").strip() + " } ")
            self.force_movenumber = True

    def visit_nag(self, nag: int) -> None:
        if self.comments and (self.variations or not self.variation_depth):
            self.write_token("$" + str(nag) + " ")

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        if self.variations or not self.variation_depth:
            # Write the move number.
            if board.turn == chess.WHITE:
                self.write_token(str(board.fullmove_number) + ". ")
            elif self.force_movenumber:
                self.write_token(str(board.fullmove_number) + "... ")

            # Write the SAN.
            self.write_token(board.san(move) + " ")

            self.force_movenumber = False

    def visit_result(self, result: str) -> None:
        self.write_token(result + " ")


class StringExporter(StringExporterMixin, BaseVisitor[str]):
    """
    Allows exporting a game as a string.

    >>> import chess.pgn
    >>>
    >>> game = chess.pgn.Game()
    >>>
    >>> exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    >>> pgn_string = game.accept(exporter)

    Only *columns* characters are written per line. If *columns* is ``None``,
    then the entire movetext will be on a single line. This does not affect
    header tags and comments.

    There will be no newline characters at the end of the string.
    """

    def result(self) -> str:
        if self.current_line:
            return "\n".join(itertools.chain(self.lines, [self.current_line.rstrip()])).rstrip()
        else:
            return "\n".join(self.lines).rstrip()

    def __str__(self) -> str:
        return self.result()


class FileExporter(StringExporterMixin, BaseVisitor[int]):
    """
    Acts like a :class:`~chess.pgn.StringExporter`, but games are written
    directly into a text file.

    There will always be a blank line after each game. Handling encodings is up
    to the caller.

    >>> import chess.pgn
    >>>
    >>> game = chess.pgn.Game()
    >>>
    >>> new_pgn = open("/dev/null", "w", encoding="utf-8")
    >>> exporter = chess.pgn.FileExporter(new_pgn)
    >>> game.accept(exporter)
    """

    def __init__(self, handle: TextIO, *, columns: Optional[int] = 80, headers: bool = True, comments: bool = True, variations: bool = True):
        super().__init__(columns=columns, headers=headers, comments=comments, variations=variations)
        self.handle = handle

    def begin_game(self) -> None:
        self.written: int = 0
        super().begin_game()

    def flush_current_line(self) -> None:
        if self.current_line:
            self.written += self.handle.write(self.current_line.rstrip())
            self.written += self.handle.write("\n")
        self.current_line = ""

    def write_line(self, line: str = "") -> None:
        self.flush_current_line()
        self.written += self.handle.write(line.rstrip())
        self.written += self.handle.write("\n")

    def result(self) -> int:
        return self.written

    def __repr__(self) -> str:
        return f"<FileExporter at {id(self):#x}>"

    def __str__(self) -> str:
        return self.__repr__()


@typing.overload
def read_game(handle: TextIO) -> Optional[Game]: ...
@typing.overload
def read_game(handle: TextIO, *, Visitor: Callable[[], BaseVisitor[ResultT]]) -> Optional[ResultT]: ...
def read_game(handle: TextIO, *, Visitor: Any = GameBuilder) -> Any:
    """
    Reads a game from a file opened in text mode.

    >>> import chess.pgn
    >>>
    >>> pgn = open("data/pgn/kasparov-deep-blue-1997.pgn")
    >>>
    >>> first_game = chess.pgn.read_game(pgn)
    >>> second_game = chess.pgn.read_game(pgn)
    >>>
    >>> first_game.headers["Event"]
    'IBM Man-Machine, New York USA'
    >>>
    >>> # Iterate through all moves and play them on a board.
    >>> board = first_game.board()
    >>> for move in first_game.mainline_moves():
    ...     board.push(move)
    ...
    >>> board
    Board('4r3/6P1/2p2P1k/1p6/pP2p1R1/P1B5/2P2K2/3r4 b - - 0 45')

    By using text mode, the parser does not need to handle encodings. It is the
    caller's responsibility to open the file with the correct encoding.
    PGN files are usually ASCII or UTF-8 encoded, sometimes with BOM (which
    this parser automatically ignores). See :func:`open` for options to
    deal with encoding errors.

    >>> pgn = open("data/pgn/kasparov-deep-blue-1997.pgn", encoding="utf-8")

    Use :class:`~io.StringIO` to parse games from a string.

    >>> import io
    >>>
    >>> pgn = io.StringIO("1. e4 e5 2. Nf3 *")
    >>> game = chess.pgn.read_game(pgn)

    The end of a game is determined by a completely blank line or the end of
    the file. (Of course, blank lines in comments are possible).

    According to the PGN standard, at least the usual seven header tags are
    required for a valid game. This parser also handles games without any
    headers just fine.

    The parser is relatively forgiving when it comes to errors. It skips over
    tokens it can not parse. By default, any exceptions are logged and
    collected in :data:`Game.errors <chess.pgn.Game.errors>`. This behavior can
    be :func:`overridden <chess.pgn.GameBuilder.handle_error>`.

    Returns the parsed game or ``None`` if the end of file is reached.
    """
    visitor = Visitor()

    found_game = False
    skipping_game = False
    managed_headers: Optional[Headers] = None
    unmanaged_headers: Optional[Headers] = None

    # Ignore leading empty lines and comments.
    line = handle.readline().lstrip("\ufeff")
    while line.isspace() or line.startswith("%") or line.startswith(";"):
        line = handle.readline()

    # Parse game headers.
    consecutive_empty_lines = 0
    while line:
        # Ignore comments.
        if line.startswith("%") or line.startswith(";"):
            line = handle.readline()
            continue

        # Ignore up to one consecutive empty line between headers.
        if consecutive_empty_lines < 1 and line.isspace():
            consecutive_empty_lines += 1
            line = handle.readline()
            continue

        # First token of the game.
        if not found_game:
            found_game = True
            skipping_game = visitor.begin_game() is SKIP
            if not skipping_game:
                managed_headers = visitor.begin_headers()
                if not isinstance(managed_headers, Headers):
                    unmanaged_headers = Headers({})

        if not line.startswith("["):
            break

        consecutive_empty_lines = 0

        if not skipping_game:
            tag_match = TAG_REGEX.match(line)
            if tag_match:
                visitor.visit_header(tag_match.group(1), tag_match.group(2))
                if unmanaged_headers is not None:
                    unmanaged_headers[tag_match.group(1)] = tag_match.group(2)
            else:
                # Ignore invalid or malformed headers.
                line = handle.readline()
                continue

        line = handle.readline()

    if not found_game:
        return None

    if not skipping_game:
        skipping_game = visitor.end_headers() is SKIP

    if not skipping_game:
        # Chess variant.
        headers = managed_headers if unmanaged_headers is None else unmanaged_headers
        assert headers is not None, "got neither managed nor unmanaged headers"
        try:
            VariantBoard = headers.variant()
        except ValueError as error:
            visitor.handle_error(error)
            VariantBoard = chess.Board

        # Initial position.
        fen = headers.get("FEN", VariantBoard.starting_fen)
        try:
            board = VariantBoard(fen, chess960=headers.is_chess960())
        except ValueError as error:
            visitor.handle_error(error)
            skipping_game = True
        else:
            board.chess960 = board.chess960 or board.has_chess960_castling_rights()
            board_stack = [board]
            visitor.visit_board(board)

    # Fast path: Skip entire game.
    if skipping_game:
        in_comment = False

        while line:
            if not in_comment:
                if line.isspace():
                    break
                elif line.startswith("%"):
                    line = handle.readline()
                    continue

            for match in SKIP_MOVETEXT_REGEX.finditer(line):
                token = match.group(0)
                if token == "{":
                    in_comment = True
                elif not in_comment and token == ";":
                    break
                elif token == "}":
                    in_comment = False

            line = handle.readline()

        visitor.end_game()
        return visitor.result()

    # Parse movetext.
    skip_variation_depth = 0
    fresh_line = True
    while line:
        if fresh_line:
            # Ignore comments.
            if line.startswith("%") or line.startswith(";"):
                line = handle.readline()
                continue
            # An empty line means the end of a game.
            if line.isspace():
                visitor.end_game()
                return visitor.result()
        fresh_line = True

        for match in MOVETEXT_REGEX.finditer(line):
            token = match.group(0)

            if token.startswith("{"):
                # Consume until the end of the comment.
                start_index = 2 if token.startswith("{ ") else 1
                line = token[start_index:]

                comment_lines = []
                while line and "}" not in line:
                    comment_lines.append(line)
                    line = handle.readline()

                if line:
                    close_index = line.find("}")
                    end_index = close_index - 1 if close_index > 0 and line[close_index - 1] == " " else close_index
                    comment_lines.append(line[:end_index])
                    line = line[close_index + 1:]

                if not skip_variation_depth:
                    visitor.visit_comment("".join(comment_lines))

                # Continue with the current line.
                fresh_line = False
                break
            elif token == "(":
                if skip_variation_depth:
                    skip_variation_depth += 1
                elif board_stack[-1].move_stack:
                    if visitor.begin_variation() is SKIP:
                        skip_variation_depth = 1
                    else:
                        board = board_stack[-1].copy()
                        board.pop()
                        board_stack.append(board)
            elif token == ")":
                if skip_variation_depth == 1:
                    skip_variation_depth = 0
                    visitor.end_variation()
                elif skip_variation_depth:
                    skip_variation_depth -= 1
                elif len(board_stack) > 1:
                    visitor.end_variation()
                    board_stack.pop()
            elif skip_variation_depth:
                continue
            elif token.startswith(";"):
                break
            elif token.startswith("$"):
                # Found a NAG.
                visitor.visit_nag(int(token[1:]))
            elif token == "?":
                visitor.visit_nag(NAG_MISTAKE)
            elif token == "??":
                visitor.visit_nag(NAG_BLUNDER)
            elif token == "!":
                visitor.visit_nag(NAG_GOOD_MOVE)
            elif token == "!!":
                visitor.visit_nag(NAG_BRILLIANT_MOVE)
            elif token == "!?":
                visitor.visit_nag(NAG_SPECULATIVE_MOVE)
            elif token == "?!":
                visitor.visit_nag(NAG_DUBIOUS_MOVE)
            elif token in ["1-0", "0-1", "1/2-1/2", "*"] and len(board_stack) == 1:
                visitor.visit_result(token)
            else:
                # Parse SAN tokens.
                if visitor.begin_parse_san(board_stack[-1], token) is not SKIP:
                    try:
                        move = visitor.parse_san(board_stack[-1], token)
                    except ValueError as error:
                        visitor.handle_error(error)
                        skip_variation_depth = 1
                    else:
                        visitor.visit_move(board_stack[-1], move)
                        board_stack[-1].push(move)
                visitor.visit_board(board_stack[-1])

        if fresh_line:
            line = handle.readline()

    visitor.end_game()
    return visitor.result()


def read_headers(handle: TextIO) -> Optional[Headers]:
    """
    Reads game headers from a PGN file opened in text mode. Skips the rest of
    the game.

    Since actually parsing many games from a big file is relatively expensive,
    this is a better way to look only for specific games and then seek and
    parse them later.

    This example scans for the first game with Kasparov as the white player.

    >>> import chess.pgn
    >>>
    >>> pgn = open("data/pgn/kasparov-deep-blue-1997.pgn")
    >>>
    >>> kasparov_offsets = []
    >>>
    >>> while True:
    ...     offset = pgn.tell()
    ...
    ...     headers = chess.pgn.read_headers(pgn)
    ...     if headers is None:
    ...         break
    ...
    ...     if "Kasparov" in headers.get("White", "?"):
    ...         kasparov_offsets.append(offset)

    Then it can later be seeked and parsed.

    >>> for offset in kasparov_offsets:
    ...     pgn.seek(offset)
    ...     chess.pgn.read_game(pgn)  # doctest: +ELLIPSIS
    0
    <Game at ... ('Garry Kasparov' vs. 'Deep Blue (Computer)', 1997.??.??)>
    1436
    <Game at ... ('Garry Kasparov' vs. 'Deep Blue (Computer)', 1997.??.??)>
    3067
    <Game at ... ('Garry Kasparov' vs. 'Deep Blue (Computer)', 1997.??.??)>
    """
    return read_game(handle, Visitor=HeadersBuilder)


def skip_game(handle: TextIO) -> bool:
    """
    Skips a game. Returns ``True`` if a game was found and skipped.
    """
    return bool(read_game(handle, Visitor=SkipVisitor))


def parse_time_control(time_control: str) -> TimeControl:
    tc = TimeControl()

    if not time_control:
        return tc

    if time_control.startswith("?"):
        return tc

    if time_control.startswith("-"):
        tc.type = TimeControlType.UNLIMITED
        return tc

    def _parse_part(part: str) -> TimeControlPart:
        tcp = TimeControlPart()

        moves_time, *bonus = part.split("+")

        if bonus:
            _bonus = bonus[0]
            if _bonus.lower().endswith("d"):
                tcp.delay = float(_bonus[:-1])
            else:
                tcp.increment = float(_bonus)

        moves, *time = moves_time.split("/")
        if time:
            tcp.moves = int(moves)
            tcp.time = int(time[0])
        else:
            tcp.moves = 0
            tcp.time = int(moves)

        return tcp

    tc.parts = [_parse_part(part) for part in time_control.split(":")]

    if len(tc.parts) > 1:
        for part in tc.parts[:-1]:
            if part.moves == 0:
                raise ValueError("Only last part can be 'sudden death'.")

    # Classification according to https://www.fide.com/FIDE/handbook/LawsOfChess.pdf
    # (Bullet added)
    base_time = tc.parts[0].time
    increment = tc.parts[0].increment
    if (base_time + 60 * increment) < 3 * 60:
        tc.type = TimeControlType.BULLET
    elif (base_time + 60 * increment) < 15 * 60:
        tc.type = TimeControlType.BLITZ
    elif (base_time + 60 * increment) < 60 * 60:
        tc.type = TimeControlType.RAPID
    else:
        tc.type = TimeControlType.STANDARD

    return tc
<|endoftext|>from __future__ import annotations

import collections
import math
import mmap
import os
import re
import struct
import threading
import typing

import chess

from types import TracebackType
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type, TypeVar, Union

if typing.TYPE_CHECKING:
    from typing_extensions import Self


UINT64_BE = struct.Struct(">Q")
UINT32 = struct.Struct("<I")
UINT32_BE = struct.Struct(">I")
UINT16 = struct.Struct("<H")

TBPIECES = 7

TRIANGLE = [
    6, 0, 1, 2, 2, 1, 0, 6,
    0, 7, 3, 4, 4, 3, 7, 0,
    1, 3, 8, 5, 5, 8, 3, 1,
    2, 4, 5, 9, 9, 5, 4, 2,
    2, 4, 5, 9, 9, 5, 4, 2,
    1, 3, 8, 5, 5, 8, 3, 1,
    0, 7, 3, 4, 4, 3, 7, 0,
    6, 0, 1, 2, 2, 1, 0, 6,
]

INVTRIANGLE = [1, 2, 3, 10, 11, 19, 0, 9, 18, 27]

def offdiag(square: chess.Square) -> int:
    return chess.square_rank(square) - chess.square_file(square)

def flipdiag(square: chess.Square) -> chess.Square:
    return ((square >> 3) | (square << 3)) & 63

LOWER = [
    28,  0,  1,  2,  3,  4,  5,  6,
     0, 29,  7,  8,  9, 10, 11, 12,
     1,  7, 30, 13, 14, 15, 16, 17,
     2,  8, 13, 31, 18, 19, 20, 21,
     3,  9, 14, 18, 32, 22, 23, 24,
     4, 10, 15, 19, 22, 33, 25, 26,
     5, 11, 16, 20, 23, 25, 34, 27,
     6, 12, 17, 21, 24, 26, 27, 35,
]

DIAG = [
     0,  0,  0,  0,  0,  0,  0,  8,
     0,  1,  0,  0,  0,  0,  9,  0,
     0,  0,  2,  0,  0, 10,  0,  0,
     0,  0,  0,  3, 11,  0,  0,  0,
     0,  0,  0, 12,  4,  0,  0,  0,
     0,  0, 13,  0,  0,  5,  0,  0,
     0, 14,  0,  0,  0,  0,  6,  0,
    15,  0,  0,  0,  0,  0,  0,  7,
]

FLAP = [
    0,  0,  0,  0,  0,  0,  0, 0,
    0,  6, 12, 18, 18, 12,  6, 0,
    1,  7, 13, 19, 19, 13,  7, 1,
    2,  8, 14, 20, 20, 14,  8, 2,
    3,  9, 15, 21, 21, 15,  9, 3,
    4, 10, 16, 22, 22, 16, 10, 4,
    5, 11, 17, 23, 23, 17, 11, 5,
    0,  0,  0,  0,  0,  0,  0, 0,
]

PTWIST = [
     0,  0,  0,  0,  0,  0,  0,  0,
    47, 35, 23, 11, 10, 22, 34, 46,
    45, 33, 21,  9,  8, 20, 32, 44,
    43, 31, 19,  7,  6, 18, 30, 42,
    41, 29, 17,  5,  4, 16, 28, 40,
    39, 27, 15,  3,  2, 14, 26, 38,
    37, 25, 13,  1,  0, 12, 24, 36,
     0,  0,  0,  0,  0,  0,  0,  0,
]

INVFLAP = [
     8, 16, 24, 32, 40, 48,
     9, 17, 25, 33, 41, 49,
    10, 18, 26, 34, 42, 50,
    11, 19, 27, 35, 43, 51,
]

FILE_TO_FILE = [0, 1, 2, 3, 3, 2, 1, 0]

KK_IDX = [[
     -1,  -1,  -1,   0,   1,   2,   3,   4,
     -1,  -1,  -1,   5,   6,   7,   8,   9,
     10,  11,  12,  13,  14,  15,  16,  17,
     18,  19,  20,  21,  22,  23,  24,  25,
     26,  27,  28,  29,  30,  31,  32,  33,
     34,  35,  36,  37,  38,  39,  40,  41,
     42,  43,  44,  45,  46,  47,  48,  49,
     50,  51,  52,  53,  54,  55,  56,  57,
], [
     58,  -1,  -1,  -1,  59,  60,  61,  62,
     63,  -1,  -1,  -1,  64,  65,  66,  67,
     68,  69,  70,  71,  72,  73,  74,  75,
     76,  77,  78,  79,  80,  81,  82,  83,
     84,  85,  86,  87,  88,  89,  90,  91,
     92,  93,  94,  95,  96,  97,  98,  99,
    100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
], [
    116, 117,  -1,  -1,  -1, 118, 119, 120,
    121, 122,  -1,  -1,  -1, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165,
    166, 167, 168, 169, 170, 171, 172, 173,
], [
    174,  -1,  -1,  -1, 175, 176, 177, 178,
    179,  -1,  -1,  -1, 180, 181, 182, 183,
    184,  -1,  -1,  -1, 185, 186, 187, 188,
    189, 190, 191, 192, 193, 194, 195, 196,
    197, 198, 199, 200, 201, 202, 203, 204,
    205, 206, 207, 208, 209, 210, 211, 212,
    213, 214, 215, 216, 217, 218, 219, 220,
    221, 222, 223, 224, 225, 226, 227, 228,
], [
    229, 230,  -1,  -1,  -1, 231, 232, 233,
    234, 235,  -1,  -1,  -1, 236, 237, 238,
    239, 240,  -1,  -1,  -1, 241, 242, 243,
    244, 245, 246, 247, 248, 249, 250, 251,
    252, 253, 254, 255, 256, 257, 258, 259,
    260, 261, 262, 263, 264, 265, 266, 267,
    268, 269, 270, 271, 272, 273, 274, 275,
    276, 277, 278, 279, 280, 281, 282, 283,
], [
    284, 285, 286, 287, 288, 289, 290, 291,
    292, 293,  -1,  -1,  -1, 294, 295, 296,
    297, 298,  -1,  -1,  -1, 299, 300, 301,
    302, 303,  -1,  -1,  -1, 304, 305, 306,
    307, 308, 309, 310, 311, 312, 313, 314,
    315, 316, 317, 318, 319, 320, 321, 322,
    323, 324, 325, 326, 327, 328, 329, 330,
    331, 332, 333, 334, 335, 336, 337, 338,
], [
     -1,  -1, 339, 340, 341, 342, 343, 344,
     -1,  -1, 345, 346, 347, 348, 349, 350,
     -1,  -1, 441, 351, 352, 353, 354, 355,
     -1,  -1,  -1, 442, 356, 357, 358, 359,
     -1,  -1,  -1,  -1, 443, 360, 361, 362,
     -1,  -1,  -1,  -1,  -1, 444, 363, 364,
     -1,  -1,  -1,  -1,  -1,  -1, 445, 365,
     -1,  -1,  -1,  -1,  -1,  -1,  -1, 446,
], [
     -1,  -1,  -1, 366, 367, 368, 369, 370,
     -1,  -1,  -1, 371, 372, 373, 374, 375,
     -1,  -1,  -1, 376, 377, 378, 379, 380,
     -1,  -1,  -1, 447, 381, 382, 383, 384,
     -1,  -1,  -1,  -1, 448, 385, 386, 387,
     -1,  -1,  -1,  -1,  -1, 449, 388, 389,
     -1,  -1,  -1,  -1,  -1,  -1, 450, 390,
     -1,  -1,  -1,  -1,  -1,  -1,  -1, 451,
], [
    452, 391, 392, 393, 394, 395, 396, 397,
     -1,  -1,  -1,  -1, 398, 399, 400, 401,
     -1,  -1,  -1,  -1, 402, 403, 404, 405,
     -1,  -1,  -1,  -1, 406, 407, 408, 409,
     -1,  -1,  -1,  -1, 453, 410, 411, 412,
     -1,  -1,  -1,  -1,  -1, 454, 413, 414,
     -1,  -1,  -1,  -1,  -1,  -1, 455, 415,
     -1,  -1,  -1,  -1,  -1,  -1,  -1, 456,
], [
    457, 416, 417, 418, 419, 420, 421, 422,
     -1, 458, 423, 424, 425, 426, 427, 428,
     -1,  -1,  -1,  -1,  -1, 429, 430, 431,
     -1,  -1,  -1,  -1,  -1, 432, 433, 434,
     -1,  -1,  -1,  -1,  -1, 435, 436, 437,
     -1,  -1,  -1,  -1,  -1, 459, 438, 439,
     -1,  -1,  -1,  -1,  -1,  -1, 460, 440,
     -1,  -1,  -1,  -1,  -1,  -1,  -1, 461,
]]

PP_IDX = [[
      0,  -1,   1,   2,   3,   4,   5,   6,
      7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,
     23,  24,  25,  26,  27,  28,  29,  30,
     31,  32,  33,  34,  35,  36,  37,  38,
     39,  40,  41,  42,  43,  44,  45,  46,
     -1,  47,  48,  49,  50,  51,  52,  53,
     54,  55,  56,  57,  58,  59,  60,  61,
], [
     62,  -1,  -1,  63,  64,  65,  -1,  66,
     -1,  67,  68,  69,  70,  71,  72,  -1,
     73,  74,  75,  76,  77,  78,  79,  80,
     81,  82,  83,  84,  85,  86,  87,  88,
     89,  90,  91,  92,  93,  94,  95,  96,
     -1,  97,  98,  99, 100, 101, 102, 103,
     -1, 104, 105, 106, 107, 108, 109,  -1,
    110,  -1, 111, 112, 113, 114,  -1, 115,
], [
    116,  -1,  -1,  -1, 117,  -1,  -1, 118,
     -1, 119, 120, 121, 122, 123, 124,  -1,
     -1, 125, 126, 127, 128, 129, 130,  -1,
    131, 132, 133, 134, 135, 136, 137, 138,
     -1, 139, 140, 141, 142, 143, 144, 145,
     -1, 146, 147, 148, 149, 150, 151,  -1,
     -1, 152, 153, 154, 155, 156, 157,  -1,
    158,  -1,  -1, 159, 160,  -1,  -1, 161,
], [
    162,  -1,  -1,  -1,  -1,  -1,  -1, 163,
     -1, 164,  -1, 165, 166, 167, 168,  -1,
     -1, 169, 170, 171, 172, 173, 174,  -1,
     -1, 175, 176, 177, 178, 179, 180,  -1,
     -1, 181, 182, 183, 184, 185, 186,  -1,
     -1,  -1, 187, 188, 189, 190, 191,  -1,
     -1, 192, 193, 194, 195, 196, 197,  -1,
    198,  -1,  -1,  -1,  -1,  -1,  -1, 199,
], [
    200,  -1,  -1,  -1,  -1,  -1,  -1, 201,
     -1, 202,  -1,  -1, 203,  -1, 204,  -1,
     -1,  -1, 205, 206, 207, 208,  -1,  -1,
     -1, 209, 210, 211, 212, 213, 214,  -1,
     -1,  -1, 215, 216, 217, 218, 219,  -1,
     -1,  -1, 220, 221, 222, 223,  -1,  -1,
     -1, 224,  -1, 225, 226,  -1, 227,  -1,
    228,  -1,  -1,  -1,  -1,  -1,  -1, 229,
], [
    230,  -1,  -1,  -1,  -1,  -1,  -1, 231,
     -1, 232,  -1,  -1,  -1,  -1, 233,  -1,
     -1,  -1, 234,  -1, 235, 236,  -1,  -1,
     -1,  -1, 237, 238, 239, 240,  -1,  -1,
     -1,  -1,  -1, 241, 242, 243,  -1,  -1,
     -1,  -1, 244, 245, 246, 247,  -1,  -1,
     -1, 248,  -1,  -1,  -1,  -1, 249,  -1,
    250,  -1,  -1,  -1,  -1,  -1,  -1, 251,
], [
     -1,  -1,  -1,  -1,  -1,  -1,  -1, 259,
     -1, 252,  -1,  -1,  -1,  -1, 260,  -1,
     -1,  -1, 253,  -1,  -1, 261,  -1,  -1,
     -1,  -1,  -1, 254, 262,  -1,  -1,  -1,
     -1,  -1,  -1,  -1, 255,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1, 256,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1, 257,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1, 258,
], [
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1, 268,  -1,
     -1,  -1, 263,  -1,  -1, 269,  -1,  -1,
     -1,  -1,  -1, 264, 270,  -1,  -1,  -1,
     -1,  -1,  -1,  -1, 265,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1, 266,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1, 267,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
], [
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1, 274,  -1,  -1,
     -1,  -1,  -1, 271, 275,  -1,  -1,  -1,
     -1,  -1,  -1,  -1, 272,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1, 273,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
], [
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1, 277,  -1,  -1,  -1,
     -1,  -1,  -1,  -1, 276,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
]]

def test45(sq: chess.Square) -> bool:
    return bool(chess.BB_SQUARES[sq] & (chess.BB_A5 | chess.BB_A6 | chess.BB_A7 |
                                        chess.BB_B5 | chess.BB_B6 |
                                        chess.BB_C5))

MTWIST = [
    15, 63, 55, 47, 40, 48, 56, 12,
    62, 11, 39, 31, 24, 32,  8, 57,
    54, 38,  7, 23, 16,  4, 33, 49,
    46, 30, 22,  3,  0, 17, 25, 41,
    45, 29, 21,  2,  1, 18, 26, 42,
    53, 37,  6, 20, 19,  5, 34, 50,
    61, 10, 36, 28, 27, 35,  9, 58,
    14, 60, 52, 44, 43, 51, 59, 13,
]

def binom(x: int, y: int) -> int:
    try:
        return math.factorial(x) // math.factorial(y) // math.factorial(x - y)
    except ValueError:
        return 0

PAWNIDX = [[0 for _ in range(24)] for _ in range(5)]

PFACTOR = [[0 for _ in range(4)] for _ in range(5)]

for i in range(5):
    j = 0

    s = 0
    while j < 6:
        PAWNIDX[i][j] = s
        s += 1 if i == 0 else binom(PTWIST[INVFLAP[j]], i)
        j += 1
    PFACTOR[i][0] = s

    s = 0
    while j < 12:
        PAWNIDX[i][j] = s
        s += 1 if i == 0 else binom(PTWIST[INVFLAP[j]], i)
        j += 1
    PFACTOR[i][1] = s

    s = 0
    while j < 18:
        PAWNIDX[i][j] = s
        s += 1 if i == 0 else binom(PTWIST[INVFLAP[j]], i)
        j += 1
    PFACTOR[i][2] = s

    s = 0
    while j < 24:
        PAWNIDX[i][j] = s
        s += 1 if i == 0 else binom(PTWIST[INVFLAP[j]], i)
        j += 1
    PFACTOR[i][3] = s

MULTIDX = [[0 for _ in range(10)] for _ in range(5)]

MFACTOR = [0 for _ in range(5)]

for i in range(5):
    s = 0
    for j in range(10):
        MULTIDX[i][j] = s
        s += 1 if i == 0 else binom(MTWIST[INVTRIANGLE[j]], i)
    MFACTOR[i] = s

WDL_TO_MAP = [1, 3, 0, 2, 0]

PA_FLAGS = [8, 0, 0, 0, 4]

WDL_TO_DTZ = [-1, -101, 0, 101, 1]

PCHR = ["K", "Q", "R", "B", "N", "P"]

TABLENAME_REGEX = re.compile(r"^[KQRBNP]+v[KQRBNP]+\Z")


def is_tablename(name: str, *, one_king: bool = True, piece_count: Optional[int] = TBPIECES, normalized: bool = True) -> bool:
    return (
        (piece_count is None or len(name) <= piece_count + 1) and
        bool(TABLENAME_REGEX.match(name)) and
        (not normalized or normalize_tablename(name) == name) and
        (not one_king or (name != "KvK" and name.startswith("K") and "vK" in name)))


def tablenames(*, one_king: bool = True, piece_count: int = 6) -> Iterator[str]:
    first = "K" if one_king else "P"

    targets: List[str] = []

    white_pieces = piece_count - 2
    black_pieces = 0
    while white_pieces >= black_pieces:
        targets.append(first + "P" * white_pieces + "v" + first + "P" * black_pieces)
        white_pieces -= 1
        black_pieces += 1

    return all_dependencies(targets, one_king=one_king)


def normalize_tablename(name: str, *, mirror: bool = False) -> str:
    w, b = name.split("v", 1)
    w = "".join(sorted(w, key=PCHR.index))
    b = "".join(sorted(b, key=PCHR.index))
    if mirror ^ ((len(w), [PCHR.index(c) for c in b]) < (len(b), [PCHR.index(c) for c in w])):
        return b + "v" + w
    else:
        return w + "v" + b


def _dependencies(target: str, *, one_king: bool = True) -> Iterator[str]:
    w, b = target.split("v", 1)

    for p in PCHR:
        if p == "K" and one_king:
            continue

        # Promotions.
        if p != "P" and "P" in w:
            yield normalize_tablename(w.replace("P", p, 1) + "v" + b)
        if p != "P" and "P" in b:
            yield normalize_tablename(w + "v" + b.replace("P", p, 1))

        # Captures.
        if p in w and len(w) > 1:
            yield normalize_tablename(w.replace(p, "", 1) + "v" + b)
        if p in b and len(b) > 1:
            yield normalize_tablename(w + "v" + b.replace(p, "", 1))


def dependencies(target: str, *, one_king: bool = True) -> Iterator[str]:
    closed: Set[str] = set()
    if one_king:
        closed.add("KvK")

    for dependency in _dependencies(target, one_king=one_king):
        if dependency not in closed and len(dependency) > 2:
            yield dependency
            closed.add(dependency)


def all_dependencies(targets: Iterable[str], *, one_king: bool = True) -> Iterator[str]:
    closed: Set[str] = set()
    if one_king:
        closed.add("KvK")

    open_list = [normalize_tablename(target) for target in targets]

    while open_list:
        name = open_list.pop()
        if name in closed:
            continue

        yield name
        closed.add(name)

        open_list.extend(_dependencies(name, one_king=one_king))


def calc_key(board: chess.BaseBoard, *, mirror: bool = False) -> str:
    w = board.occupied_co[chess.WHITE ^ mirror]
    b = board.occupied_co[chess.BLACK ^ mirror]

    return "".join([
        "K" * chess.popcount(board.kings & w),
        "Q" * chess.popcount(board.queens & w),
        "R" * chess.popcount(board.rooks & w),
        "B" * chess.popcount(board.bishops & w),
        "N" * chess.popcount(board.knights & w),
        "P" * chess.popcount(board.pawns & w),
        "v",
        "K" * chess.popcount(board.kings & b),
        "Q" * chess.popcount(board.queens & b),
        "R" * chess.popcount(board.rooks & b),
        "B" * chess.popcount(board.bishops & b),
        "N" * chess.popcount(board.knights & b),
        "P" * chess.popcount(board.pawns & b),
    ])


def recalc_key(pieces: List[chess.PieceType], *, mirror: bool = False) -> str:
    # Some endgames are stored with a different key than their filename
    # indicates: http://talkchess.com/forum/viewtopic.php?p=695509#695509

    w = 8 if mirror else 0
    b = 0 if mirror else 8

    return "".join([
        "K" * pieces.count(6 ^ w),
        "Q" * pieces.count(5 ^ w),
        "R" * pieces.count(4 ^ w),
        "B" * pieces.count(3 ^ w),
        "N" * pieces.count(2 ^ w),
        "P" * pieces.count(1 ^ w),
        "v",
        "K" * pieces.count(6 ^ b),
        "Q" * pieces.count(5 ^ b),
        "R" * pieces.count(4 ^ b),
        "B" * pieces.count(3 ^ b),
        "N" * pieces.count(2 ^ b),
        "P" * pieces.count(1 ^ b),
    ])


def subfactor(k: int, n: int) -> int:
    f = n
    l = 1

    for i in range(1, k):
        f *= n - i
        l *= i + 1

    return f // l


def dtz_before_zeroing(wdl: int) -> int:
    return ((wdl > 0) - (wdl < 0)) * (1 if abs(wdl) == 2 else 101)


class MissingTableError(KeyError):
    """Can not probe position because a required table is missing."""
    pass


class PairsData:
    indextable: int
    sizetable: int
    data: int
    offset: int
    symlen: List[int]
    sympat: int
    blocksize: int
    idxbits: int
    min_len: int
    base: List[int]


class PawnFileData:
    def __init__(self) -> None:
        self.precomp: Dict[int, PairsData] = {}
        self.factor: Dict[int, List[int]] = {}
        self.pieces: Dict[int, List[int]] = {}
        self.norm: Dict[int, List[int]] = {}


class PawnFileDataDtz:
    precomp: PairsData
    factor: List[int]
    pieces: List[int]
    norm: List[int]


class Table:
    size: List[int]

    def __init__(self, path: str, *, variant: Type[chess.Board] = chess.Board) -> None:
        self.path = path
        self.variant = variant

        self.write_lock = threading.RLock()
        self.initialized = False
        self.data: Optional[mmap.mmap] = None

        self.read_condition = threading.Condition()
        self.read_count = 0

        tablename, _ = os.path.splitext(os.path.basename(path))
        self.key = normalize_tablename(tablename)
        self.mirrored_key = normalize_tablename(tablename, mirror=True)
        self.symmetric = self.key == self.mirrored_key

        # Leave the v out of the tablename to get the number of pieces.
        self.num = len(tablename) - 1

        self.has_pawns = "P" in tablename

        black_part, white_part = tablename.split("v")
        if self.has_pawns:
            self.pawns = [white_part.count("P"), black_part.count("P")]
            if self.pawns[1] > 0 and (self.pawns[0] == 0 or self.pawns[1] < self.pawns[0]):
                self.pawns[0], self.pawns[1] = self.pawns[1], self.pawns[0]
        else:
            j = 0
            for piece_type in PCHR:
                if black_part.count(piece_type) == 1:
                    j += 1
                if white_part.count(piece_type) == 1:
                    j += 1
            if j >= 3:
                self.enc_type = 0
            elif j == 2:
                self.enc_type = 2
            else:  # only for suicide
                j = 16
                for _ in range(16):
                    for piece_type in PCHR:
                        if 1 < black_part.count(piece_type) < j:
                            j = black_part.count(piece_type)
                        if 1 < white_part.count(piece_type) < j:
                            j = white_part.count(piece_type)
                        self.enc_type = 1 + j

    def init_mmap(self) -> None:
        if self.data is None:
            fd = os.open(self.path, os.O_RDONLY | os.O_BINARY if hasattr(os, "O_BINARY") else os.O_RDONLY)
            try:
                data = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)

            if data.size() % 64 != 16:
                raise IOError(f"invalid file size: ensure {self.path!r} is a valid syzygy tablebase file")

            try:
                # Unix
                data.madvise(mmap.MADV_RANDOM)
            except AttributeError:
                pass

            self.data = data

    def check_magic(self, magic: Optional[bytes], pawnless_magic: Optional[bytes]) -> None:
        assert self.data

        valid_magics = [magic, self.has_pawns and pawnless_magic]
        if self.data[:min(4, len(self.data))] not in valid_magics:
            raise IOError(f"invalid magic header: ensure {self.path!r} is a valid syzygy tablebase file")

    def setup_pairs(self, data_ptr: int, tb_size: int, size_idx: int, wdl: int) -> PairsData:
        assert self.data

        d = PairsData()

        self._flags = self.data[data_ptr]
        if self.data[data_ptr] & 0x80:
            d.idxbits = 0
            if wdl:
                d.min_len = self.data[data_ptr + 1]
            else:
                # http://www.talkchess.com/forum/viewtopic.php?p=698093#698093
                d.min_len = 1 if self.variant.captures_compulsory else 0
            self._next = data_ptr + 2
            self.size[size_idx + 0] = 0
            self.size[size_idx + 1] = 0
            self.size[size_idx + 2] = 0
            return d

        d.blocksize = self.data[data_ptr + 1]
        d.idxbits = self.data[data_ptr + 2]

        real_num_blocks = self.read_uint32(data_ptr + 4)
        num_blocks = real_num_blocks + self.data[data_ptr + 3]
        max_len = self.data[data_ptr + 8]
        min_len = self.data[data_ptr + 9]
        h = max_len - min_len + 1
        num_syms = self.read_uint16(data_ptr + 10 + 2 * h)

        d.offset = data_ptr + 10
        d.symlen = [0 for _ in range(h * 8 + num_syms)]
        d.sympat = data_ptr + 12 + 2 * h
        d.min_len = min_len

        self._next = data_ptr + 12 + 2 * h + 3 * num_syms + (num_syms & 1)

        num_indices = (tb_size + (1 << d.idxbits) - 1) >> d.idxbits
        self.size[size_idx + 0] = 6 * num_indices
        self.size[size_idx + 1] = 2 * num_blocks
        self.size[size_idx + 2] = (1 << d.blocksize) * real_num_blocks

        tmp = [0 for _ in range(num_syms)]
        for i in range(num_syms):
            if not tmp[i]:
                self.calc_symlen(d, i, tmp)

        d.base = [0 for _ in range(h)]
        d.base[h - 1] = 0
        for i in range(h - 2, -1, -1):
            d.base[i] = (d.base[i + 1] + self.read_uint16(d.offset + i * 2) - self.read_uint16(d.offset + i * 2 + 2)) // 2
        for i in range(h):
            d.base[i] <<= 64 - (min_len + i)

        d.offset -= 2 * d.min_len

        return d

    def set_norm_piece(self, norm: List[int], pieces: List[int]) -> None:
        if self.enc_type == 0:
            norm[0] = 3
        elif self.enc_type == 2:
            norm[0] = 2
        else:
            norm[0] = self.enc_type - 1

        i = norm[0]
        while i < self.num:
            j = i
            while j < self.num and pieces[j] == pieces[i]:
                norm[i] += 1
                j += 1
            i += norm[i]

    def calc_factors_piece(self, factor: List[int], order: int, norm: List[int]) -> int:
        PIVFAC = [31332, 0, 518, 278] if self.variant.connected_kings else [31332, 28056, 462]

        n = 64 - norm[0]

        f = 1
        i = norm[0]
        k = 0
        while i < self.num or k == order:
            if k == order:
                factor[0] = f
                if self.enc_type < 4:
                    f *= PIVFAC[self.enc_type]
                else:
                    f *= MFACTOR[self.enc_type - 2]
            else:
                factor[i] = f
                f *= subfactor(norm[i], n)
                n -= norm[i]
                i += norm[i]
            k += 1

        return f

    def calc_factors_pawn(self, factor: List[int], order: int, order2: int, norm: List[int], f: int) -> int:
        i = norm[0]
        if order2 < 0x0f:
            i += norm[i]
        n = 64 - i

        fac = 1
        k = 0
        while i < self.num or k in [order, order2]:
            if k == order:
                factor[0] = fac
                fac *= PFACTOR[norm[0] - 1][f]
            elif k == order2:
                factor[norm[0]] = fac
                fac *= subfactor(norm[norm[0]], 48 - norm[0])
            else:
                factor[i] = fac
                fac *= subfactor(norm[i], n)
                n -= norm[i]
                i += norm[i]
            k += 1

        return fac

    def set_norm_pawn(self, norm: List[int], pieces: List[int]) -> None:
        norm[0] = self.pawns[0]
        if self.pawns[1]:
            norm[self.pawns[0]] = self.pawns[1]

        i = self.pawns[0] + self.pawns[1]
        while i < self.num:
            j = i
            while j < self.num and pieces[j] == pieces[i]:
                norm[i] += 1
                j += 1
            i += norm[i]

    def calc_symlen(self, d: PairsData, s: int, tmp: List[int]) -> None:
        assert self.data

        w = d.sympat + 3 * s
        s2 = (self.data[w + 2] << 4) | (self.data[w + 1] >> 4)
        if s2 == 0x0fff:
            d.symlen[s] = 0
        else:
            s1 = ((self.data[w + 1] & 0xf) << 8) | self.data[w]
            if not tmp[s1]:
                self.calc_symlen(d, s1, tmp)
            if not tmp[s2]:
                self.calc_symlen(d, s2, tmp)
            d.symlen[s] = d.symlen[s1] + d.symlen[s2] + 1
        tmp[s] = 1

    def pawn_file(self, pos: List[chess.Square]) -> int:
        for i in range(1, self.pawns[0]):
            if FLAP[pos[0]] > FLAP[pos[i]]:
                pos[0], pos[i] = pos[i], pos[0]

        return FILE_TO_FILE[pos[0] & 0x07]

    def encode_piece(self, norm: List[int], pos: List[chess.Square], factor: List[int]) -> int:
        n = self.num

        if self.enc_type < 3:
            if pos[0] & 0x04:
                for i in range(n):
                    pos[i] ^= 0x07

            if pos[0] & 0x20:
                for i in range(n):
                    pos[i] ^= 0x38

            for i in range(n):
                if offdiag(pos[i]):
                    break
            if i < (3 if self.enc_type == 0 else 2) and offdiag(pos[i]) > 0:
                for i in range(n):
                    pos[i] = flipdiag(pos[i])

        if self.enc_type == 0:  # 111
            i = int(pos[1] > pos[0])
            j = int(pos[2] > pos[0]) + int(pos[2] > pos[1])

            if offdiag(pos[0]):
                idx = TRIANGLE[pos[0]] * 63 * 62 + (pos[1] - i) * 62 + (pos[2] - j)
            elif offdiag(pos[1]):
                idx = 6 * 63 * 62 + DIAG[pos[0]] * 28 * 62 + LOWER[pos[1]] * 62 + pos[2] - j
            elif offdiag(pos[2]):
                idx = 6 * 63 * 62 + 4 * 28 * 62 + (DIAG[pos[0]]) * 7 * 28 + (DIAG[pos[1]] - i) * 28 + LOWER[pos[2]]
            else:
                idx = 6 * 63 * 62 + 4 * 28 * 62 + 4 * 7 * 28 + (DIAG[pos[0]] * 7 * 6) + (DIAG[pos[1]] - i) * 6 + (DIAG[pos[2]] - j)
            i = 3
        elif self.enc_type == 2:  # K2
            if not self.variant.connected_kings:
                idx = KK_IDX[TRIANGLE[pos[0]]][pos[1]]
            else:
                i = pos[1] > pos[0]

                if offdiag(pos[0]):
                    idx = TRIANGLE[pos[0]] * 63 + (pos[1] - i)
                elif offdiag(pos[1]):
                    idx = 6 * 63 + DIAG[pos[0]] * 28 + LOWER[pos[1]]
                else:
                    idx = 6 * 63 + 4 * 28 + (DIAG[pos[0]]) * 7 + (DIAG[pos[1]] - i)

            i = 2
        elif self.enc_type == 3:  # 2, e.g. KKvK
            if TRIANGLE[pos[0]] > TRIANGLE[pos[1]]:
                pos[0], pos[1] = pos[1], pos[0]
            if pos[0] & 0x04:
                for i in range(n):
                    pos[i] ^= 0x07
            if pos[0] & 0x20:
                for i in range(n):
                    pos[i] ^= 0x38
            if offdiag(pos[0]) > 0 or (offdiag(pos[0]) == 0 and offdiag(pos[1]) > 0):
                for i in range(n):
                    pos[i] = flipdiag(pos[i])
            if test45(pos[1]) and TRIANGLE[pos[0]] == TRIANGLE[pos[1]]:
                pos[0], pos[1] = pos[1], pos[0]
                for i in range(n):
                    pos[i] = flipdiag(pos[i] ^ 0x38)
            idx = PP_IDX[TRIANGLE[pos[0]]][pos[1]]
            i = 2
        else:  # 3 and higher, e.g. KKKvK and KKKKvK
            for i in range(1, norm[0]):
                if TRIANGLE[pos[0]] > TRIANGLE[pos[i]]:
                    pos[0], pos[i] = pos[i], pos[0]
            if pos[0] & 0x04:
                for i in range(n):
                    pos[i] ^= 0x07
            if pos[0] & 0x20:
                for i in range(n):
                    pos[i] ^= 0x38
            if offdiag(pos[0]) > 0:
                for i in range(n):
                    pos[i] = flipdiag(pos[i])
            for i in range(1, norm[0]):
                for j in range(i + 1, norm[0]):
                    if MTWIST[pos[i]] > MTWIST[pos[j]]:
                        pos[i], pos[j] = pos[j], pos[i]

            idx = MULTIDX[norm[0] - 1][TRIANGLE[pos[0]]]
            i = 1
            while i < norm[0]:
                idx += binom(MTWIST[pos[i]], i)
                i += 1

        idx *= factor[0]

        while i < n:
            t = norm[i]

            for j in range(i, i + t):
                for k in range(j + 1, i + t):
                    # Swap.
                    if pos[j] > pos[k]:
                        pos[j], pos[k] = pos[k], pos[j]

            s = 0

            for m in range(i, i + t):
                p = pos[m]
                j = 0
                for l in range(i):
                    j += int(p > pos[l])
                s += binom(p - j, m - i + 1)

            idx += s * factor[i]
            i += t

        return idx

    def encode_pawn(self, norm: List[int], pos: List[chess.Square], factor: List[int]) -> int:
        n = self.num

        if pos[0] & 0x04:
            for i in range(n):
                pos[i] ^= 0x07

        for i in range(1, self.pawns[0]):
            for j in range(i + 1, self.pawns[0]):
                if PTWIST[pos[i]] < PTWIST[pos[j]]:
                    pos[i], pos[j] = pos[j], pos[i]

        t = self.pawns[0] - 1
        idx = PAWNIDX[t][FLAP[pos[0]]]
        for i in range(t, 0, -1):
            idx += binom(PTWIST[pos[i]], t - i + 1)
        idx *= factor[0]

        # Remaining pawns.
        i = self.pawns[0]
        t = i + self.pawns[1]
        if t > i:
            for j in range(i, t):
                for k in range(j + 1, t):
                    if pos[j] > pos[k]:
                        pos[j], pos[k] = pos[k], pos[j]
            s = 0
            for m in range(i, t):
                p = pos[m]
                j = 0
                for k in range(i):
                    j += int(p > pos[k])
                s += binom(p - j - 8, m - i + 1)
            idx += s * factor[i]
            i = t

        while i < n:
            t = norm[i]
            for j in range(i, i + t):
                for k in range(j + 1, i + t):
                    if pos[j] > pos[k]:
                        pos[j], pos[k] = pos[k], pos[j]

            s = 0
            for m in range(i, i + t):
                p = pos[m]
                j = 0
                for k in range(i):
                    j += int(p > pos[k])
                s += binom(p - j, m - i + 1)

            idx += s * factor[i]
            i += t

        return idx

    def decompress_pairs(self, d: PairsData, idx: int) -> int:
        assert self.data

        if not d.idxbits:
            return d.min_len

        mainidx = idx >> d.idxbits
        litidx = (idx & (1 << d.idxbits) - 1) - (1 << (d.idxbits - 1))
        block = self.read_uint32(d.indextable + 6 * mainidx)

        idx_offset = self.read_uint16(d.indextable + 6 * mainidx + 4)
        litidx += idx_offset

        if litidx < 0:
            while litidx < 0:
                block -= 1
                litidx += self.read_uint16(d.sizetable + 2 * block) + 1
        else:
            while litidx > self.read_uint16(d.sizetable + 2 * block):
                litidx -= self.read_uint16(d.sizetable + 2 * block) + 1
                block += 1

        ptr = d.data + (block << d.blocksize)

        m = d.min_len
        base_idx = -m
        symlen_idx = 0

        code = self.read_uint64_be(ptr)

        ptr += 2 * 4
        bitcnt = 0  # Number of empty bits in code
        while True:
            l = m
            while code < d.base[base_idx + l]:
                l += 1
            sym = self.read_uint16(d.offset + l * 2)
            sym += (code - d.base[base_idx + l]) >> (64 - l)
            if litidx < d.symlen[symlen_idx + sym] + 1:
                break
            litidx -= d.symlen[symlen_idx + sym] + 1
            code <<= l
            bitcnt += l
            if bitcnt >= 32:
                bitcnt -= 32
                code |= self.read_uint32_be(ptr) << bitcnt
                ptr += 4

            # Cut off at 64bit.
            code &= 0xffffffffffffffff

        sympat = d.sympat
        while d.symlen[symlen_idx + sym]:
            w = sympat + 3 * sym
            s1 = ((self.data[w + 1] & 0xf) << 8) | self.data[w]
            if litidx < d.symlen[symlen_idx + s1] + 1:
                sym = s1
            else:
                litidx -= d.symlen[symlen_idx + s1] + 1
                sym = (self.data[w + 2] << 4) | (self.data[w + 1] >> 4)

        w = sympat + 3 * sym
        if isinstance(self, DtzTable):
            return ((self.data[w + 1] & 0x0f) << 8) | self.data[w]
        else:
            return self.data[w]

    def read_uint64_be(self, data_ptr: int) -> int:
        return UINT64_BE.unpack_from(self.data, data_ptr)[0]  # type: ignore

    def read_uint32(self, data_ptr: int) -> int:
        return UINT32.unpack_from(self.data, data_ptr)[0]  # type: ignore

    def read_uint32_be(self, data_ptr: int) -> int:
        return UINT32_BE.unpack_from(self.data, data_ptr)[0]  # type: ignore

    def read_uint16(self, data_ptr: int) -> int:
        return UINT16.unpack_from(self.data, data_ptr)[0]  # type: ignore

    def close(self) -> None:
        with self.write_lock:
            with self.read_condition:
                while self.read_count > 0:
                    self.read_condition.wait()

                if self.data is not None:
                    self.data.close()
                    self.data = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()


class WdlTable(Table):
    _next: int
    _flags: int

    def init_table_wdl(self) -> None:
        with self.write_lock:
            self.init_mmap()
            assert self.data

            if self.initialized:
                return

            self.check_magic(self.variant.tbw_magic, self.variant.pawnless_tbw_magic)

            self.tb_size = [0 for _ in range(8)]
            self.size = [0 for _ in range(8 * 3)]

            # Used if there are only pieces.
            self.precomp: Dict[int, PairsData] = {}
            self.pieces: Dict[int, List[int]] = {}
            self.factor = [[0 for _ in range(TBPIECES)] for _ in range(2)]
            self.norm = [[0 for _ in range(self.num)] for _ in range(2)]

            # Used if there are pawns.
            self.files = [PawnFileData() for _ in range(4)]

            split = self.data[4] & 0x01
            files = 4 if self.data[4] & 0x02 else 1

            data_ptr = 5

            if not self.has_pawns:
                self.setup_pieces_piece(data_ptr)
                data_ptr += self.num + 1
                data_ptr += data_ptr & 0x01

                self.precomp[0] = self.setup_pairs(data_ptr, self.tb_size[0], 0, True)
                data_ptr = self._next
                if split:
                    self.precomp[1] = self.setup_pairs(data_ptr, self.tb_size[1], 3, True)
                    data_ptr = self._next

                self.precomp[0].indextable = data_ptr
                data_ptr += self.size[0]
                if split:
                    self.precomp[1].indextable = data_ptr
                    data_ptr += self.size[3]

                self.precomp[0].sizetable = data_ptr
                data_ptr += self.size[1]
                if split:
                    self.precomp[1].sizetable = data_ptr
                    data_ptr += self.size[4]

                data_ptr = (data_ptr + 0x3f) & ~0x3f
                self.precomp[0].data = data_ptr
                data_ptr += self.size[2]
                if split:
                    data_ptr = (data_ptr + 0x3f) & ~0x3f
                    self.precomp[1].data = data_ptr

                self.key = recalc_key(self.pieces[0])
                self.mirrored_key = recalc_key(self.pieces[0], mirror=True)
            else:
                s = 1 + int(self.pawns[1] > 0)
                for f in range(4):
                    self.setup_pieces_pawn(data_ptr, 2 * f, f)
                    data_ptr += self.num + s
                data_ptr += data_ptr & 0x01

                for f in range(files):
                    self.files[f].precomp[0] = self.setup_pairs(data_ptr, self.tb_size[2 * f], 6 * f, True)
                    data_ptr = self._next
                    if split:
                        self.files[f].precomp[1] = self.setup_pairs(data_ptr, self.tb_size[2 * f + 1], 6 * f + 3, True)
                        data_ptr = self._next

                for f in range(files):
                    self.files[f].precomp[0].indextable = data_ptr
                    data_ptr += self.size[6 * f]
                    if split:
                        self.files[f].precomp[1].indextable = data_ptr
                        data_ptr += self.size[6 * f + 3]

                for f in range(files):
                    self.files[f].precomp[0].sizetable = data_ptr
                    data_ptr += self.size[6 * f + 1]
                    if split:
                        self.files[f].precomp[1].sizetable = data_ptr
                        data_ptr += self.size[6 * f + 4]

                for f in range(files):
                    data_ptr = (data_ptr + 0x3f) & ~0x3f
                    self.files[f].precomp[0].data = data_ptr
                    data_ptr += self.size[6 * f + 2]
                    if split:
                        data_ptr = (data_ptr + 0x3f) & ~0x3f
                        self.files[f].precomp[1].data = data_ptr
                        data_ptr += self.size[6 * f + 5]

            self.initialized = True

    def setup_pieces_pawn(self, p_data: int, p_tb_size: int, f: int) -> None:
        assert self.data

        j = 1 + int(self.pawns[1] > 0)
        order = self.data[p_data] & 0x0f
        order2 = self.data[p_data + 1] & 0x0f if self.pawns[1] else 0x0f
        self.files[f].pieces[0] = [self.data[p_data + i + j] & 0x0f for i in range(self.num)]
        self.files[f].norm[0] = [0 for _ in range(self.num)]
        self.set_norm_pawn(self.files[f].norm[0], self.files[f].pieces[0])
        self.files[f].factor[0] = [0 for _ in range(TBPIECES)]
        self.tb_size[p_tb_size] = self.calc_factors_pawn(self.files[f].factor[0], order, order2, self.files[f].norm[0], f)

        order = self.data[p_data] >> 4
        order2 = self.data[p_data + 1] >> 4 if self.pawns[1] else 0x0f
        self.files[f].pieces[1] = [self.data[p_data + i + j] >> 4 for i in range(self.num)]
        self.files[f].norm[1] = [0 for _ in range(self.num)]
        self.set_norm_pawn(self.files[f].norm[1], self.files[f].pieces[1])
        self.files[f].factor[1] = [0 for _ in range(TBPIECES)]
        self.tb_size[p_tb_size + 1] = self.calc_factors_pawn(self.files[f].factor[1], order, order2, self.files[f].norm[1], f)

    def setup_pieces_piece(self, p_data: int) -> None:
        assert self.data

        self.pieces[0] = [self.data[p_data + i + 1] & 0x0f for i in range(self.num)]
        order = self.data[p_data] & 0x0f
        self.set_norm_piece(self.norm[0], self.pieces[0])
        self.tb_size[0] = self.calc_factors_piece(self.factor[0], order, self.norm[0])

        self.pieces[1] = [self.data[p_data + i + 1] >> 4 for i in range(self.num)]
        order = self.data[p_data] >> 4
        self.set_norm_piece(self.norm[1], self.pieces[1])
        self.tb_size[1] = self.calc_factors_piece(self.factor[1], order, self.norm[1])

    def probe_wdl_table(self, board: chess.Board) -> int:
        try:
            with self.read_condition:
                self.read_count += 1
            return self._probe_wdl_table(board)
        finally:
            with self.read_condition:
                self.read_count -= 1
                self.read_condition.notify()

    def _probe_wdl_table(self, board: chess.Board) -> int:
        self.init_table_wdl()

        key = calc_key(board)

        if not self.symmetric:
            if key != self.key:
                cmirror = 8
                mirror = 0x38
                bside = int(board.turn == chess.WHITE)
            else:
                cmirror = mirror = 0
                bside = int(board.turn != chess.WHITE)
        else:
            cmirror = 0 if board.turn == chess.WHITE else 8
            mirror = 0 if board.turn == chess.WHITE else 0x38
            bside = 0

        if not self.has_pawns:
            p = [0 for _ in range(TBPIECES)]
            i = 0
            while i < self.num:
                piece_type = self.pieces[bside][i] & 0x07
                color = (self.pieces[bside][i] ^ cmirror) >> 3
                bb = board.pieces_mask(piece_type, chess.WHITE if color == 0 else chess.BLACK)

                for square in chess.scan_forward(bb):
                    p[i] = square
                    i += 1

            idx = self.encode_piece(self.norm[bside], p, self.factor[bside])
            res = self.decompress_pairs(self.precomp[bside], idx)
        else:
            p = [0 for _ in range(TBPIECES)]
            i = 0
            k = self.files[0].pieces[0][0] ^ cmirror
            color = k >> 3
            piece_type = k & 0x07
            bb = board.pieces_mask(piece_type, chess.WHITE if color == 0 else chess.BLACK)

            for square in chess.scan_forward(bb):
                p[i] = square ^ mirror
                i += 1

            f = self.pawn_file(p)
            pc = self.files[f].pieces[bside]
            while i < self.num:
                color = (pc[i] ^ cmirror) >> 3
                piece_type = pc[i] & 0x07
                bb = board.pieces_mask(piece_type, chess.WHITE if color == 0 else chess.BLACK)

                for square in chess.scan_forward(bb):
                    p[i] = square ^ mirror
                    i += 1

            idx = self.encode_pawn(self.files[f].norm[bside], p, self.files[f].factor[bside])
            res = self.decompress_pairs(self.files[f].precomp[bside], idx)

        return res - 2


class DtzTable(Table):

    def init_table_dtz(self) -> None:
        with self.write_lock:
            self.init_mmap()
            assert self.data

            if self.initialized:
                return

            self.check_magic(self.variant.tbz_magic, self.variant.pawnless_tbz_magic)

            self.factor = [0 for _ in range(TBPIECES)]
            self.norm = [0 for _ in range(self.num)]
            self.tb_size = [0, 0, 0, 0]
            self.size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.files = [PawnFileDataDtz() for _ in range(4)]

            files = 4 if self.data[4] & 0x02 else 1

            p_data = 5

            if not self.has_pawns:
                self.map_idx: List[List[int]] = [[0, 0, 0, 0]]

                self.setup_pieces_piece_dtz(p_data, 0)
                p_data += self.num + 1
                p_data += p_data & 0x01

                self.precomp = self.setup_pairs(p_data, self.tb_size[0], 0, False)
                self.flags: Union[int, List[int]] = self._flags
                p_data = self._next
                self.p_map = p_data
                if self.flags & 2:
                    if not self.flags & 16:
                        for i in range(4):
                            self.map_idx[0][i] = p_data + 1 - self.p_map
                            p_data += 1 + self.data[p_data]
                    else:
                        for i in range(4):
                            self.map_idx[0][i] = (p_data + 2 - self.p_map) // 2
                            p_data += 2 + 2 * self.read_uint16(p_data)
                p_data += p_data & 0x01

                self.precomp.indextable = p_data
                p_data += self.size[0]

                self.precomp.sizetable = p_data
                p_data += self.size[1]

                p_data = (p_data + 0x3f) & ~0x3f
                self.precomp.data = p_data
                p_data += self.size[2]

                self.key = recalc_key(self.pieces)
                self.mirrored_key = recalc_key(self.pieces, mirror=True)
            else:
                s = 1 + int(self.pawns[1] > 0)
                for f in range(4):
                    self.setup_pieces_pawn_dtz(p_data, f, f)
                    p_data += self.num + s
                p_data += p_data & 0x01

                self.flags = []
                for f in range(files):
                    self.files[f].precomp = self.setup_pairs(p_data, self.tb_size[f], 3 * f, False)
                    p_data = self._next
                    self.flags.append(self._flags)

                self.map_idx = []
                self.p_map = p_data
                for f in range(files):
                    self.map_idx.append([])
                    if self.flags[f] & 2:
                        if not self.flags[f] & 16:
                            for _ in range(4):
                                self.map_idx[-1].append(p_data + 1 - self.p_map)
                                p_data += 1 + self.data[p_data]
                        else:
                            p_data += p_data & 0x01
                            for _ in range(4):
                                self.map_idx[-1].append((p_data + 2 - self.p_map) // 2)
                                p_data += 2 + 2 * self.read_uint16(p_data)
                p_data += p_data & 0x01

                for f in range(files):
                    self.files[f].precomp.indextable = p_data
                    p_data += self.size[3 * f]

                for f in range(files):
                    self.files[f].precomp.sizetable = p_data
                    p_data += self.size[3 * f + 1]

                for f in range(files):
                    p_data = (p_data + 0x3f) & ~0x3f
                    self.files[f].precomp.data = p_data
                    p_data += self.size[3 * f + 2]

            self.initialized = True

    def probe_dtz_table(self, board: chess.Board, wdl: int) -> Tuple[int, int]:
        try:
            with self.read_condition:
                self.read_count += 1
            return self._probe_dtz_table(board, wdl)
        finally:
            with self.read_condition:
                self.read_count -= 1
                self.read_condition.notify()

    def _probe_dtz_table(self, board: chess.Board, wdl: int) -> Tuple[int, int]:
        self.init_table_dtz()
        assert self.data

        key = calc_key(board)

        if not self.symmetric:
            if key != self.key:
                cmirror = 8
                mirror = 0x38
                bside = int(board.turn == chess.WHITE)
            else:
                cmirror = mirror = 0
                bside = int(board.turn != chess.WHITE)
        else:
            cmirror = 0 if board.turn == chess.WHITE else 8
            mirror = 0 if board.turn == chess.WHITE else 0x38
            bside = 0

        if not self.has_pawns:
            assert isinstance(self.flags, int)

            if (self.flags & 1) != bside and not self.symmetric:
                return 0, -1

            pc = self.pieces
            p = [0 for _ in range(TBPIECES)]
            i = 0
            while i < self.num:
                piece_type = pc[i] & 0x07
                color = (pc[i] ^ cmirror) >> 3
                bb = board.pieces_mask(piece_type, chess.WHITE if color == 0 else chess.BLACK)

                for square in chess.scan_forward(bb):
                    p[i] = square
                    i += 1

            idx = self.encode_piece(self.norm, p, self.factor)
            res = self.decompress_pairs(self.precomp, idx)

            if self.flags & 2:
                if not self.flags & 16:
                    res = self.data[self.p_map + self.map_idx[0][WDL_TO_MAP[wdl + 2]] + res]
                else:
                    res = self.read_uint16(self.p_map + 2 * (self.map_idx[0][WDL_TO_MAP[wdl + 2]] + res))

            if (not (self.flags & PA_FLAGS[wdl + 2])) or (wdl & 1):
                res *= 2
        else:
            assert isinstance(self.flags, list)

            k = self.files[0].pieces[0] ^ cmirror
            piece_type = k & 0x07
            color = k >> 3
            bb = board.pieces_mask(piece_type, chess.WHITE if color == 0 else chess.BLACK)

            i = 0
            p = [0 for _ in range(TBPIECES)]
            for square in chess.scan_forward(bb):
                p[i] = square ^ mirror
                i += 1
            f = self.pawn_file(p)
            if self.flags[f] & 1 != bside:
                return 0, -1

            pc = self.files[f].pieces
            while i < self.num:
                piece_type = pc[i] & 0x07
                color = (pc[i] ^ cmirror) >> 3
                bb = board.pieces_mask(piece_type, chess.WHITE if color == 0 else chess.BLACK)

                for square in chess.scan_forward(bb):
                    p[i] = square ^ mirror
                    i += 1

            idx = self.encode_pawn(self.files[f].norm, p, self.files[f].factor)
            res = self.decompress_pairs(self.files[f].precomp, idx)

            if self.flags[f] & 2:
                if not self.flags[f] & 16:
                    res = self.data[self.p_map + self.map_idx[f][WDL_TO_MAP[wdl + 2]] + res]
                else:
                    res = self.read_uint16(self.p_map + 2 * (self.map_idx[f][WDL_TO_MAP[wdl + 2]] + res))

            if (not (self.flags[f] & PA_FLAGS[wdl + 2])) or (wdl & 1):
                res *= 2

        return res, 1

    def setup_pieces_piece_dtz(self, p_data: int, p_tb_size: int) -> None:
        assert self.data

        self.pieces = [self.data[p_data + i + 1] & 0x0f for i in range(self.num)]
        order = self.data[p_data] & 0x0f
        self.set_norm_piece(self.norm, self.pieces)
        self.tb_size[p_tb_size] = self.calc_factors_piece(self.factor, order, self.norm)

    def setup_pieces_pawn_dtz(self, p_data: int, p_tb_size: int, f: int) -> None:
        assert self.data

        j = 1 + int(self.pawns[1] > 0)
        order = self.data[p_data] & 0x0f
        order2 = self.data[p_data + 1] & 0x0f if self.pawns[1] else 0x0f
        self.files[f].pieces = [self.data[p_data + i + j] & 0x0f for i in range(self.num)]

        self.files[f].norm = [0 for _ in range(self.num)]
        self.set_norm_pawn(self.files[f].norm, self.files[f].pieces)

        self.files[f].factor = [0 for _ in range(TBPIECES)]
        self.tb_size[p_tb_size] = self.calc_factors_pawn(self.files[f].factor, order, order2, self.files[f].norm, f)


class Tablebase:
    """
    Manages a collection of tablebase files for probing.
    """
    def __init__(self, *, max_fds: Optional[int] = 128, VariantBoard: Type[chess.Board] = chess.Board) -> None:
        self.variant = VariantBoard

        self.max_fds = max_fds
        self.lru: Deque[Table] = collections.deque()
        self.lru_lock = threading.Lock()

        self.wdl: Dict[str, Table] = {}
        self.dtz: Dict[str, Table] = {}

    def _bump_lru(self, table: Table) -> None:
        if self.max_fds is None:
            return

        with self.lru_lock:
            try:
                self.lru.remove(table)
                self.lru.appendleft(table)
            except ValueError:
                self.lru.appendleft(table)

                if len(self.lru) > self.max_fds:
                    self.lru.pop().close()

    def _open_table(self, hashtable: Dict[str, Table], Table: Type[Table], path: str) -> int:
        table = Table(path, variant=self.variant)

        if table.key in hashtable:
            hashtable[table.key].close()

        hashtable[table.key] = table
        hashtable[table.mirrored_key] = table
        return 1

    def add_directory(self, directory: str, *, load_wdl: bool = True, load_dtz: bool = True) -> int:
        """
        Adds tables from a directory.

        By default, all available tables with the correct file names
        (e.g., WDL files like ``KQvKN.rtbw`` and DTZ files like ``KRBvK.rtbz``)
        are added.

        The relevant files are lazily opened when the tablebase is actually
        probed.

        Returns the number of table files that were found.
        """
        directory = os.path.abspath(directory)
        return sum(self.add_file(os.path.join(directory, filename), load_wdl=load_wdl, load_dtz=load_dtz) for filename in os.listdir(directory))

    def add_file(self, path: str, *, load_wdl: bool = True, load_dtz: bool = True) -> int:
        tablename, ext = os.path.splitext(os.path.basename(path))
        if is_tablename(tablename, one_king=self.variant.one_king) and os.path.isfile(path):
            if load_wdl:
                if ext == self.variant.tbw_suffix:
                    return self._open_table(self.wdl, WdlTable, path)
                elif "P" not in tablename and ext == self.variant.pawnless_tbw_suffix:
                    return self._open_table(self.wdl, WdlTable, path)
            if load_dtz:
                if ext == self.variant.tbz_suffix:
                    return self._open_table(self.dtz, DtzTable, path)
                elif "P" not in tablename and ext == self.variant.pawnless_tbz_suffix:
                    return self._open_table(self.dtz, DtzTable, path)
        return 0

    def probe_wdl_table(self, board: chess.Board) -> int:
        # Test for variant end.
        if board.is_variant_win():
            return 2
        elif board.is_variant_draw():
            return 0
        elif board.is_variant_loss():
            return -2

        # Test for KvK.
        if self.variant.one_king and board.kings == board.occupied:
            return 0

        key = calc_key(board)
        try:
            table = typing.cast(WdlTable, self.wdl[key])
        except KeyError:
            if chess.popcount(board.occupied) > TBPIECES:
                raise KeyError(f"syzygy tables support up to {TBPIECES} pieces, not {chess.popcount(board.occupied)}: {board.fen()}")
            raise MissingTableError(f"did not find wdl table {key}")

        self._bump_lru(table)

        return table.probe_wdl_table(board)

    def probe_ab(self, board: chess.Board, alpha: int, beta: int, threats: bool = False) -> Tuple[int, int]:
        # Check preconditions.
        if board.uci_variant != self.variant.uci_variant:
            raise KeyError(f"tablebase has been opened for {self.variant.uci_variant}, probed with: {board.uci_variant}")
        if board.castling_rights:
            raise KeyError(f"syzygy tables do not contain positions with castling rights: {board.fen()}")

        # Probing resolves captures, so sometimes we can obtain results for
        # positions that have more pieces than the maximum number of supported
        # pieces. We artificially limit this to one additional level, to
        # make sure search remains somewhat bounded.
        if chess.popcount(board.occupied) > TBPIECES + 1:
            raise KeyError(f"syzygy tables support up to {TBPIECES} pieces, not {chess.popcount(board.occupied)}: {board.fen()}")

        # Special case: Variant with compulsory captures.
        if self.variant.captures_compulsory:
            if board.is_variant_win():
                return 2, 2
            elif board.is_variant_loss():
                return -2, 2
            elif board.is_variant_draw():
                return 0, 2

            return self.sprobe_ab(board, alpha, beta, threats)

        # Generate non-ep captures.
        for move in board.generate_legal_moves(to_mask=board.occupied_co[not board.turn]):
            board.push(move)
            try:
                v_plus, _ = self.probe_ab(board, -beta, -alpha)
                v = -v_plus
            finally:
                board.pop()

            if v > alpha:
                if v >= beta:
                    return v, 2
                alpha = v

        v = self.probe_wdl_table(board)

        if alpha >= v:
            return alpha, 1 + int(alpha > 0)
        else:
            return v, 1

    def sprobe_ab(self, board: chess.Board, alpha: int, beta: int, threats: bool = False) -> Tuple[int, int]:
        if chess.popcount(board.occupied_co[not board.turn]) > 1:
            v, captures_found = self.sprobe_capts(board, alpha, beta)
            if captures_found:
                return v, 2
        else:
            if any(board.generate_legal_captures()):
                return -2, 2

        threats_found = False

        if threats or chess.popcount(board.occupied) >= 6:
            for threat in board.generate_legal_moves(~board.pawns):
                board.push(threat)
                try:
                    v_plus, captures_found = self.sprobe_capts(board, -beta, -alpha)
                    v = -v_plus
                finally:
                    board.pop()

                if captures_found and v > alpha:
                    threats_found = True
                    alpha = v
                    if alpha >= beta:
                        return v, 3

        v = self.probe_wdl_table(board)
        if v > alpha:
            return v, 1
        else:
            return alpha, 3 if threats_found else 1

    def sprobe_capts(self, board: chess.Board, alpha: int, beta: int) -> Tuple[int, int]:
        captures_found = False

        for move in board.generate_legal_captures():
            captures_found = True

            board.push(move)
            try:
                v_plus, _ = self.sprobe_ab(board, -beta, -alpha)
                v = -v_plus
            finally:
                board.pop()

            alpha = max(v, alpha)

            if alpha >= beta:
                break

        return alpha, captures_found

    def probe_wdl(self, board: chess.Board) -> int:
        """
        Probes WDL tables for win/draw/loss information under the 50-move rule,
        assuming the position has been reached directly after a capture or
        pawn move.

        Probing is thread-safe when done with different *board* objects and
        if *board* objects are not modified during probing.

        Returns ``2`` if the side to move is winning, ``0`` if the position is
        a draw and ``-2`` if the side to move is losing.

        Returns ``1`` in case of a cursed win and ``-1`` in case of a blessed
        loss. Mate can be forced but the position can be drawn due to the
        fifty-move rule.

        >>> import chess
        >>> import chess.syzygy
        >>>
        >>> with chess.syzygy.open_tablebase("data/syzygy/regular") as tablebase:
        ...     board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")
        ...     print(tablebase.probe_wdl(board))
        ...
        -2

        :raises: :exc:`KeyError` (or specifically
            :exc:`chess.syzygy.MissingTableError`) if the position could not
            be found in the tablebase. Use
            :func:`~chess.syzygy.Tablebase.get_wdl()` if you prefer to get
            ``None`` instead of an exception.

            Note that probing corrupted table files is undefined behavior.
        """
        # Probe.
        v, _ = self.probe_ab(board, -2, 2)

        # If en passant is not possible, we are done.
        if not board.ep_square or self.variant.captures_compulsory:
            return v

        # Now handle en passant.
        v1 = -3

        # Look at all legal en passant captures.
        for move in board.generate_legal_ep():
            board.push(move)
            try:
                v0_plus, _ = self.probe_ab(board, -2, 2)
                v0 = -v0_plus
            finally:
                board.pop()

            if v0 > v1:
                v1 = v0

        if v1 > -3:
            if v1 >= v:
                v = v1
            elif v == 0:
                # If there is not at least one legal non-en-passant move we are
                # forced to play the losing en passant cature.
                if all(board.is_en_passant(move) for move in board.generate_legal_moves()):
                    v = v1

        return v

    def get_wdl(self, board: chess.Board, default: Optional[int] = None) -> Optional[int]:
        try:
            return self.probe_wdl(board)
        except KeyError:
            return default

    def probe_dtz_table(self, board: chess.Board, wdl: int) -> Tuple[int, int]:
        key = calc_key(board)
        try:
            table = typing.cast(DtzTable, self.dtz[key])
        except KeyError:
            raise MissingTableError(f"did not find dtz table {key}")

        self._bump_lru(table)

        return table.probe_dtz_table(board, wdl)

    def probe_dtz_no_ep(self, board: chess.Board) -> int:
        wdl, success = self.probe_ab(board, -2, 2, threats=True)

        if wdl == 0:
            return 0

        if success == 2 or not board.occupied_co[board.turn] & ~board.pawns:
            return dtz_before_zeroing(wdl)

        if wdl > 0:
            # The position is a win or a cursed win by a threat move.
            if success == 3:
                return 2 if wdl == 2 else 102

            # Generate all legal non-capturing pawn moves.
            for move in board.generate_legal_moves(board.pawns, ~board.occupied):
                if board.is_capture(move):
                    # En passant.
                    continue

                board.push(move)
                try:
                    v = -self.probe_wdl(board)
                finally:
                    board.pop()

                if v == wdl:
                    return 1 if v == 2 else 101

        dtz, success = self.probe_dtz_table(board, wdl)
        if success >= 0:
            return dtz_before_zeroing(wdl) + (dtz if wdl > 0 else -dtz)

        if wdl > 0:
            best = 0xffff

            for move in board.generate_legal_moves(~board.pawns, ~board.occupied):
                board.push(move)
                try:
                    v = -self.probe_dtz(board)

                    if v == 1 and board.is_checkmate():
                        best = 1
                    elif v > 0 and v + 1 < best:
                        best = v + 1
                finally:
                    board.pop()

            return best
        else:
            best = -1

            for move in board.generate_legal_moves():
                board.push(move)

                try:
                    if board.halfmove_clock == 0:
                        if wdl == -2:
                            v = -1
                        else:
                            v, success = self.probe_ab(board, 1, 2, threats=True)
                            v = 0 if v == 2 else -101
                    else:
                        v = -self.probe_dtz(board) - 1
                finally:
                    board.pop()

                if v < best:
                    best = v

            return best

    def probe_dtz(self, board: chess.Board) -> int:
        """
        Probes DTZ tables for
        `DTZ50'' information with rounding <https://syzygy-tables.info/metrics#dtz>`_.

        Minmaxing the DTZ50'' values guarantees winning a won position
        (and drawing a drawn position), because it makes progress keeping the
        win in hand.
        However, the lines are not always the most straightforward ways to win.
        Engines like Stockfish calculate themselves, checking with DTZ, but
        only play according to DTZ if they can not manage on their own.

        Returns a positive value if the side to move is winning, ``0`` if the
        position is a draw, and a negative value if the side to move is losing.
        More precisely:

        +-----+------------------+--------------------------------------------+
        | WDL | DTZ              |                                            |
        +=====+==================+============================================+
        |  -2 | -100 <= n <= -1  | Unconditional loss (assuming 50-move       |
        |     |                  | counter is zero), where a zeroing move can |
        |     |                  | be forced in -n plies.                     |
        +-----+------------------+--------------------------------------------+
        |  -1 |         n < -100 | Loss, but draw under the 50-move rule.     |
        |     |                  | A zeroing move can be forced in -n plies   |
        |     |                  | or -n - 100 plies (if a later phase is     |
        |     |                  | responsible for the blessed loss).         |
        +-----+------------------+--------------------------------------------+
        |   0 |         0        | Draw.                                      |
        +-----+------------------+--------------------------------------------+
        |   1 |   100 < n        | Win, but draw under the 50-move rule.      |
        |     |                  | A zeroing move can be forced in n plies or |
        |     |                  | n - 100 plies (if a later phase is         |
        |     |                  | responsible for the cursed win).           |
        +-----+------------------+--------------------------------------------+
        |   2 |    1 <= n <= 100 | Unconditional win (assuming 50-move        |
        |     |                  | counter is zero), where a zeroing move can |
        |     |                  | be forced in n plies.                      |
        +-----+------------------+--------------------------------------------+

        The return value can be off by one: a return value -n can mean a
        losing zeroing move in in n + 1 plies and a return value +n can mean a
        winning zeroing move in n + 1 plies.
        This implies some primary tablebase lines may waste up to 1 ply.
        Rounding is never used for endgame phases where it would change the
        game theoretical outcome.

        This means users need to be careful in positions that are nearly drawn
        under the 50-move rule! Carelessly wasting 1 more ply by not following
        the tablebase recommendation, for a total of 2 wasted plies, may
        change the outcome of the game.

        >>> import chess
        >>> import chess.syzygy
        >>>
        >>> with chess.syzygy.open_tablebase("data/syzygy/regular") as tablebase:
        ...     board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")
        ...     print(tablebase.probe_dtz(board))
        ...
        -53

        Probing is thread-safe when done with different *board* objects and
        if *board* objects are not modified during probing.

        Both DTZ and WDL tables are required in order to probe for DTZ.

        :raises: :exc:`KeyError` (or specifically
            :exc:`chess.syzygy.MissingTableError`) if the position could not
            be found in the tablebase. Use
            :func:`~chess.syzygy.Tablebase.get_dtz()` if you prefer to get
            ``None`` instead of an exception.

            Note that probing corrupted table files is undefined behavior.
        """
        v = self.probe_dtz_no_ep(board)

        if not board.ep_square or self.variant.captures_compulsory:
            return v

        v1 = -3

        # Generate all en passant moves.
        for move in board.generate_legal_ep():
            board.push(move)
            try:
                v0_plus, _ = self.probe_ab(board, -2, 2)
                v0 = -v0_plus
            finally:
                board.pop()

            if v0 > v1:
                v1 = v0

        if v1 > -3:
            v1 = WDL_TO_DTZ[v1 + 2]
            if v < -100:
                if v1 >= 0:
                    v = v1
            elif v < 0:
                if v1 >= 0 or v1 < -100:
                    v = v1
            elif v > 100:
                if v1 > 0:
                    v = v1
            elif v > 0:
                if v1 == 1:
                    v = v1
            elif v1 >= 0:
                v = v1
            else:
                if all(board.is_en_passant(move) for move in board.generate_legal_moves()):
                    v = v1

        return v

    def get_dtz(self, board: chess.Board, default: Optional[int] = None) -> Optional[int]:
        try:
            return self.probe_dtz(board)
        except KeyError:
            return default

    def close(self) -> None:
        """Closes all loaded tables."""
        while self.wdl:
            _, wdl = self.wdl.popitem()
            wdl.close()

        while self.dtz:
            _, dtz = self.dtz.popitem()
            dtz.close()

        self.lru.clear()

    def __enter__(self) -> Tablebase:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()


def open_tablebase(directory: str, *, load_wdl: bool = True, load_dtz: bool = True, max_fds: Optional[int] = 128, VariantBoard: Type[chess.Board] = chess.Board) -> Tablebase:
    """
    Opens a collection of tables for probing. See
    :class:`~chess.syzygy.Tablebase`.

    .. note::

        Generally probing requires tablebase files for the specific
        material composition, **as well as** material compositions transitively
        reachable by captures and promotions.
        This is important because 6-piece and 5-piece (let alone 7-piece) files
        are often distributed separately, but are both required for 6-piece
        positions. Use :func:`~chess.syzygy.Tablebase.add_directory()` to load
        tables from additional directories.

    :param max_fds: If *max_fds* is not ``None``, will at most use *max_fds*
        open file descriptors at any given time. The least recently used tables
        are closed, if necessary.
    """
    tables = Tablebase(max_fds=max_fds, VariantBoard=VariantBoard)
    tables.add_directory(directory, load_wdl=load_wdl, load_dtz=load_dtz)
    return tables
<|endoftext|>from __future__ import annotations

import chess
import struct
import os
import mmap
import random
import typing

from types import TracebackType
from typing import Callable, Container, Iterator, List, NamedTuple, Optional, Type, Union


StrOrBytesPath = Union[str, bytes, "os.PathLike[str]", "os.PathLike[bytes]"]


ENTRY_STRUCT = struct.Struct(">QHHI")


POLYGLOT_RANDOM_ARRAY = [
    0x9D39247E33776D41, 0x2AF7398005AAA5C7, 0x44DB015024623547, 0x9C15F73E62A76AE2,
    0x75834465489C0C89, 0x3290AC3A203001BF, 0x0FBBAD1F61042279, 0xE83A908FF2FB60CA,
    0x0D7E765D58755C10, 0x1A083822CEAFE02D, 0x9605D5F0E25EC3B0, 0xD021FF5CD13A2ED5,
    0x40BDF15D4A672E32, 0x011355146FD56395, 0x5DB4832046F3D9E5, 0x239F8B2D7FF719CC,
    0x05D1A1AE85B49AA1, 0x679F848F6E8FC971, 0x7449BBFF801FED0B, 0x7D11CDB1C3B7ADF0,
    0x82C7709E781EB7CC, 0xF3218F1C9510786C, 0x331478F3AF51BBE6, 0x4BB38DE5E7219443,
    0xAA649C6EBCFD50FC, 0x8DBD98A352AFD40B, 0x87D2074B81D79217, 0x19F3C751D3E92AE1,
    0xB4AB30F062B19ABF, 0x7B0500AC42047AC4, 0xC9452CA81A09D85D, 0x24AA6C514DA27500,
    0x4C9F34427501B447, 0x14A68FD73C910841, 0xA71B9B83461CBD93, 0x03488B95B0F1850F,
    0x637B2B34FF93C040, 0x09D1BC9A3DD90A94, 0x3575668334A1DD3B, 0x735E2B97A4C45A23,
    0x18727070F1BD400B, 0x1FCBACD259BF02E7, 0xD310A7C2CE9B6555, 0xBF983FE0FE5D8244,
    0x9F74D14F7454A824, 0x51EBDC4AB9BA3035, 0x5C82C505DB9AB0FA, 0xFCF7FE8A3430B241,
    0x3253A729B9BA3DDE, 0x8C74C368081B3075, 0xB9BC6C87167C33E7, 0x7EF48F2B83024E20,
    0x11D505D4C351BD7F, 0x6568FCA92C76A243, 0x4DE0B0F40F32A7B8, 0x96D693460CC37E5D,
    0x42E240CB63689F2F, 0x6D2BDCDAE2919661, 0x42880B0236E4D951, 0x5F0F4A5898171BB6,
    0x39F890F579F92F88, 0x93C5B5F47356388B, 0x63DC359D8D231B78, 0xEC16CA8AEA98AD76,
    0x5355F900C2A82DC7, 0x07FB9F855A997142, 0x5093417AA8A7ED5E, 0x7BCBC38DA25A7F3C,
    0x19FC8A768CF4B6D4, 0x637A7780DECFC0D9, 0x8249A47AEE0E41F7, 0x79AD695501E7D1E8,
    0x14ACBAF4777D5776, 0xF145B6BECCDEA195, 0xDABF2AC8201752FC, 0x24C3C94DF9C8D3F6,
    0xBB6E2924F03912EA, 0x0CE26C0B95C980D9, 0xA49CD132BFBF7CC4, 0xE99D662AF4243939,
    0x27E6AD7891165C3F, 0x8535F040B9744FF1, 0x54B3F4FA5F40D873, 0x72B12C32127FED2B,
    0xEE954D3C7B411F47, 0x9A85AC909A24EAA1, 0x70AC4CD9F04F21F5, 0xF9B89D3E99A075C2,
    0x87B3E2B2B5C907B1, 0xA366E5B8C54F48B8, 0xAE4A9346CC3F7CF2, 0x1920C04D47267BBD,
    0x87BF02C6B49E2AE9, 0x092237AC237F3859, 0xFF07F64EF8ED14D0, 0x8DE8DCA9F03CC54E,
    0x9C1633264DB49C89, 0xB3F22C3D0B0B38ED, 0x390E5FB44D01144B, 0x5BFEA5B4712768E9,
    0x1E1032911FA78984, 0x9A74ACB964E78CB3, 0x4F80F7A035DAFB04, 0x6304D09A0B3738C4,
    0x2171E64683023A08, 0x5B9B63EB9CEFF80C, 0x506AACF489889342, 0x1881AFC9A3A701D6,
    0x6503080440750644, 0xDFD395339CDBF4A7, 0xEF927DBCF00C20F2, 0x7B32F7D1E03680EC,
    0xB9FD7620E7316243, 0x05A7E8A57DB91B77, 0xB5889C6E15630A75, 0x4A750A09CE9573F7,
    0xCF464CEC899A2F8A, 0xF538639CE705B824, 0x3C79A0FF5580EF7F, 0xEDE6C87F8477609D,
    0x799E81F05BC93F31, 0x86536B8CF3428A8C, 0x97D7374C60087B73, 0xA246637CFF328532,
    0x043FCAE60CC0EBA0, 0x920E449535DD359E, 0x70EB093B15B290CC, 0x73A1921916591CBD,
    0x56436C9FE1A1AA8D, 0xEFAC4B70633B8F81, 0xBB215798D45DF7AF, 0x45F20042F24F1768,
    0x930F80F4E8EB7462, 0xFF6712FFCFD75EA1, 0xAE623FD67468AA70, 0xDD2C5BC84BC8D8FC,
    0x7EED120D54CF2DD9, 0x22FE545401165F1C, 0xC91800E98FB99929, 0x808BD68E6AC10365,
    0xDEC468145B7605F6, 0x1BEDE3A3AEF53302, 0x43539603D6C55602, 0xAA969B5C691CCB7A,
    0xA87832D392EFEE56, 0x65942C7B3C7E11AE, 0xDED2D633CAD004F6, 0x21F08570F420E565,
    0xB415938D7DA94E3C, 0x91B859E59ECB6350, 0x10CFF333E0ED804A, 0x28AED140BE0BB7DD,
    0xC5CC1D89724FA456, 0x5648F680F11A2741, 0x2D255069F0B7DAB3, 0x9BC5A38EF729ABD4,
    0xEF2F054308F6A2BC, 0xAF2042F5CC5C2858, 0x480412BAB7F5BE2A, 0xAEF3AF4A563DFE43,
    0x19AFE59AE451497F, 0x52593803DFF1E840, 0xF4F076E65F2CE6F0, 0x11379625747D5AF3,
    0xBCE5D2248682C115, 0x9DA4243DE836994F, 0x066F70B33FE09017, 0x4DC4DE189B671A1C,
    0x51039AB7712457C3, 0xC07A3F80C31FB4B4, 0xB46EE9C5E64A6E7C, 0xB3819A42ABE61C87,
    0x21A007933A522A20, 0x2DF16F761598AA4F, 0x763C4A1371B368FD, 0xF793C46702E086A0,
    0xD7288E012AEB8D31, 0xDE336A2A4BC1C44B, 0x0BF692B38D079F23, 0x2C604A7A177326B3,
    0x4850E73E03EB6064, 0xCFC447F1E53C8E1B, 0xB05CA3F564268D99, 0x9AE182C8BC9474E8,
    0xA4FC4BD4FC5558CA, 0xE755178D58FC4E76, 0x69B97DB1A4C03DFE, 0xF9B5B7C4ACC67C96,
    0xFC6A82D64B8655FB, 0x9C684CB6C4D24417, 0x8EC97D2917456ED0, 0x6703DF9D2924E97E,
    0xC547F57E42A7444E, 0x78E37644E7CAD29E, 0xFE9A44E9362F05FA, 0x08BD35CC38336615,
    0x9315E5EB3A129ACE, 0x94061B871E04DF75, 0xDF1D9F9D784BA010, 0x3BBA57B68871B59D,
    0xD2B7ADEEDED1F73F, 0xF7A255D83BC373F8, 0xD7F4F2448C0CEB81, 0xD95BE88CD210FFA7,
    0x336F52F8FF4728E7, 0xA74049DAC312AC71, 0xA2F61BB6E437FDB5, 0x4F2A5CB07F6A35B3,
    0x87D380BDA5BF7859, 0x16B9F7E06C453A21, 0x7BA2484C8A0FD54E, 0xF3A678CAD9A2E38C,
    0x39B0BF7DDE437BA2, 0xFCAF55C1BF8A4424, 0x18FCF680573FA594, 0x4C0563B89F495AC3,
    0x40E087931A00930D, 0x8CFFA9412EB642C1, 0x68CA39053261169F, 0x7A1EE967D27579E2,
    0x9D1D60E5076F5B6F, 0x3810E399B6F65BA2, 0x32095B6D4AB5F9B1, 0x35CAB62109DD038A,
    0xA90B24499FCFAFB1, 0x77A225A07CC2C6BD, 0x513E5E634C70E331, 0x4361C0CA3F692F12,
    0xD941ACA44B20A45B, 0x528F7C8602C5807B, 0x52AB92BEB9613989, 0x9D1DFA2EFC557F73,
    0x722FF175F572C348, 0x1D1260A51107FE97, 0x7A249A57EC0C9BA2, 0x04208FE9E8F7F2D6,
    0x5A110C6058B920A0, 0x0CD9A497658A5698, 0x56FD23C8F9715A4C, 0x284C847B9D887AAE,
    0x04FEABFBBDB619CB, 0x742E1E651C60BA83, 0x9A9632E65904AD3C, 0x881B82A13B51B9E2,
    0x506E6744CD974924, 0xB0183DB56FFC6A79, 0x0ED9B915C66ED37E, 0x5E11E86D5873D484,
    0xF678647E3519AC6E, 0x1B85D488D0F20CC5, 0xDAB9FE6525D89021, 0x0D151D86ADB73615,
    0xA865A54EDCC0F019, 0x93C42566AEF98FFB, 0x99E7AFEABE000731, 0x48CBFF086DDF285A,
    0x7F9B6AF1EBF78BAF, 0x58627E1A149BBA21, 0x2CD16E2ABD791E33, 0xD363EFF5F0977996,
    0x0CE2A38C344A6EED, 0x1A804AADB9CFA741, 0x907F30421D78C5DE, 0x501F65EDB3034D07,
    0x37624AE5A48FA6E9, 0x957BAF61700CFF4E, 0x3A6C27934E31188A, 0xD49503536ABCA345,
    0x088E049589C432E0, 0xF943AEE7FEBF21B8, 0x6C3B8E3E336139D3, 0x364F6FFA464EE52E,
    0xD60F6DCEDC314222, 0x56963B0DCA418FC0, 0x16F50EDF91E513AF, 0xEF1955914B609F93,
    0x565601C0364E3228, 0xECB53939887E8175, 0xBAC7A9A18531294B, 0xB344C470397BBA52,
    0x65D34954DAF3CEBD, 0xB4B81B3FA97511E2, 0xB422061193D6F6A7, 0x071582401C38434D,
    0x7A13F18BBEDC4FF5, 0xBC4097B116C524D2, 0x59B97885E2F2EA28, 0x99170A5DC3115544,
    0x6F423357E7C6A9F9, 0x325928EE6E6F8794, 0xD0E4366228B03343, 0x565C31F7DE89EA27,
    0x30F5611484119414, 0xD873DB391292ED4F, 0x7BD94E1D8E17DEBC, 0xC7D9F16864A76E94,
    0x947AE053EE56E63C, 0xC8C93882F9475F5F, 0x3A9BF55BA91F81CA, 0xD9A11FBB3D9808E4,
    0x0FD22063EDC29FCA, 0xB3F256D8ACA0B0B9, 0xB03031A8B4516E84, 0x35DD37D5871448AF,
    0xE9F6082B05542E4E, 0xEBFAFA33D7254B59, 0x9255ABB50D532280, 0xB9AB4CE57F2D34F3,
    0x693501D628297551, 0xC62C58F97DD949BF, 0xCD454F8F19C5126A, 0xBBE83F4ECC2BDECB,
    0xDC842B7E2819E230, 0xBA89142E007503B8, 0xA3BC941D0A5061CB, 0xE9F6760E32CD8021,
    0x09C7E552BC76492F, 0x852F54934DA55CC9, 0x8107FCCF064FCF56, 0x098954D51FFF6580,
    0x23B70EDB1955C4BF, 0xC330DE426430F69D, 0x4715ED43E8A45C0A, 0xA8D7E4DAB780A08D,
    0x0572B974F03CE0BB, 0xB57D2E985E1419C7, 0xE8D9ECBE2CF3D73F, 0x2FE4B17170E59750,
    0x11317BA87905E790, 0x7FBF21EC8A1F45EC, 0x1725CABFCB045B00, 0x964E915CD5E2B207,
    0x3E2B8BCBF016D66D, 0xBE7444E39328A0AC, 0xF85B2B4FBCDE44B7, 0x49353FEA39BA63B1,
    0x1DD01AAFCD53486A, 0x1FCA8A92FD719F85, 0xFC7C95D827357AFA, 0x18A6A990C8B35EBD,
    0xCCCB7005C6B9C28D, 0x3BDBB92C43B17F26, 0xAA70B5B4F89695A2, 0xE94C39A54A98307F,
    0xB7A0B174CFF6F36E, 0xD4DBA84729AF48AD, 0x2E18BC1AD9704A68, 0x2DE0966DAF2F8B1C,
    0xB9C11D5B1E43A07E, 0x64972D68DEE33360, 0x94628D38D0C20584, 0xDBC0D2B6AB90A559,
    0xD2733C4335C6A72F, 0x7E75D99D94A70F4D, 0x6CED1983376FA72B, 0x97FCAACBF030BC24,
    0x7B77497B32503B12, 0x8547EDDFB81CCB94, 0x79999CDFF70902CB, 0xCFFE1939438E9B24,
    0x829626E3892D95D7, 0x92FAE24291F2B3F1, 0x63E22C147B9C3403, 0xC678B6D860284A1C,
    0x5873888850659AE7, 0x0981DCD296A8736D, 0x9F65789A6509A440, 0x9FF38FED72E9052F,
    0xE479EE5B9930578C, 0xE7F28ECD2D49EECD, 0x56C074A581EA17FE, 0x5544F7D774B14AEF,
    0x7B3F0195FC6F290F, 0x12153635B2C0CF57, 0x7F5126DBBA5E0CA7, 0x7A76956C3EAFB413,
    0x3D5774A11D31AB39, 0x8A1B083821F40CB4, 0x7B4A38E32537DF62, 0x950113646D1D6E03,
    0x4DA8979A0041E8A9, 0x3BC36E078F7515D7, 0x5D0A12F27AD310D1, 0x7F9D1A2E1EBE1327,
    0xDA3A361B1C5157B1, 0xDCDD7D20903D0C25, 0x36833336D068F707, 0xCE68341F79893389,
    0xAB9090168DD05F34, 0x43954B3252DC25E5, 0xB438C2B67F98E5E9, 0x10DCD78E3851A492,
    0xDBC27AB5447822BF, 0x9B3CDB65F82CA382, 0xB67B7896167B4C84, 0xBFCED1B0048EAC50,
    0xA9119B60369FFEBD, 0x1FFF7AC80904BF45, 0xAC12FB171817EEE7, 0xAF08DA9177DDA93D,
    0x1B0CAB936E65C744, 0xB559EB1D04E5E932, 0xC37B45B3F8D6F2BA, 0xC3A9DC228CAAC9E9,
    0xF3B8B6675A6507FF, 0x9FC477DE4ED681DA, 0x67378D8ECCEF96CB, 0x6DD856D94D259236,
    0xA319CE15B0B4DB31, 0x073973751F12DD5E, 0x8A8E849EB32781A5, 0xE1925C71285279F5,
    0x74C04BF1790C0EFE, 0x4DDA48153C94938A, 0x9D266D6A1CC0542C, 0x7440FB816508C4FE,
    0x13328503DF48229F, 0xD6BF7BAEE43CAC40, 0x4838D65F6EF6748F, 0x1E152328F3318DEA,
    0x8F8419A348F296BF, 0x72C8834A5957B511, 0xD7A023A73260B45C, 0x94EBC8ABCFB56DAE,
    0x9FC10D0F989993E0, 0xDE68A2355B93CAE6, 0xA44CFE79AE538BBE, 0x9D1D84FCCE371425,
    0x51D2B1AB2DDFB636, 0x2FD7E4B9E72CD38C, 0x65CA5B96B7552210, 0xDD69A0D8AB3B546D,
    0x604D51B25FBF70E2, 0x73AA8A564FB7AC9E, 0x1A8C1E992B941148, 0xAAC40A2703D9BEA0,
    0x764DBEAE7FA4F3A6, 0x1E99B96E70A9BE8B, 0x2C5E9DEB57EF4743, 0x3A938FEE32D29981,
    0x26E6DB8FFDF5ADFE, 0x469356C504EC9F9D, 0xC8763C5B08D1908C, 0x3F6C6AF859D80055,
    0x7F7CC39420A3A545, 0x9BFB227EBDF4C5CE, 0x89039D79D6FC5C5C, 0x8FE88B57305E2AB6,
    0xA09E8C8C35AB96DE, 0xFA7E393983325753, 0xD6B6D0ECC617C699, 0xDFEA21EA9E7557E3,
    0xB67C1FA481680AF8, 0xCA1E3785A9E724E5, 0x1CFC8BED0D681639, 0xD18D8549D140CAEA,
    0x4ED0FE7E9DC91335, 0xE4DBF0634473F5D2, 0x1761F93A44D5AEFE, 0x53898E4C3910DA55,
    0x734DE8181F6EC39A, 0x2680B122BAA28D97, 0x298AF231C85BAFAB, 0x7983EED3740847D5,
    0x66C1A2A1A60CD889, 0x9E17E49642A3E4C1, 0xEDB454E7BADC0805, 0x50B704CAB602C329,
    0x4CC317FB9CDDD023, 0x66B4835D9EAFEA22, 0x219B97E26FFC81BD, 0x261E4E4C0A333A9D,
    0x1FE2CCA76517DB90, 0xD7504DFA8816EDBB, 0xB9571FA04DC089C8, 0x1DDC0325259B27DE,
    0xCF3F4688801EB9AA, 0xF4F5D05C10CAB243, 0x38B6525C21A42B0E, 0x36F60E2BA4FA6800,
    0xEB3593803173E0CE, 0x9C4CD6257C5A3603, 0xAF0C317D32ADAA8A, 0x258E5A80C7204C4B,
    0x8B889D624D44885D, 0xF4D14597E660F855, 0xD4347F66EC8941C3, 0xE699ED85B0DFB40D,
    0x2472F6207C2D0484, 0xC2A1E7B5B459AEB5, 0xAB4F6451CC1D45EC, 0x63767572AE3D6174,
    0xA59E0BD101731A28, 0x116D0016CB948F09, 0x2CF9C8CA052F6E9F, 0x0B090A7560A968E3,
    0xABEEDDB2DDE06FF1, 0x58EFC10B06A2068D, 0xC6E57A78FBD986E0, 0x2EAB8CA63CE802D7,
    0x14A195640116F336, 0x7C0828DD624EC390, 0xD74BBE77E6116AC7, 0x804456AF10F5FB53,
    0xEBE9EA2ADF4321C7, 0x03219A39EE587A30, 0x49787FEF17AF9924, 0xA1E9300CD8520548,
    0x5B45E522E4B1B4EF, 0xB49C3B3995091A36, 0xD4490AD526F14431, 0x12A8F216AF9418C2,
    0x001F837CC7350524, 0x1877B51E57A764D5, 0xA2853B80F17F58EE, 0x993E1DE72D36D310,
    0xB3598080CE64A656, 0x252F59CF0D9F04BB, 0xD23C8E176D113600, 0x1BDA0492E7E4586E,
    0x21E0BD5026C619BF, 0x3B097ADAF088F94E, 0x8D14DEDB30BE846E, 0xF95CFFA23AF5F6F4,
    0x3871700761B3F743, 0xCA672B91E9E4FA16, 0x64C8E531BFF53B55, 0x241260ED4AD1E87D,
    0x106C09B972D2E822, 0x7FBA195410E5CA30, 0x7884D9BC6CB569D8, 0x0647DFEDCD894A29,
    0x63573FF03E224774, 0x4FC8E9560F91B123, 0x1DB956E450275779, 0xB8D91274B9E9D4FB,
    0xA2EBEE47E2FBFCE1, 0xD9F1F30CCD97FB09, 0xEFED53D75FD64E6B, 0x2E6D02C36017F67F,
    0xA9AA4D20DB084E9B, 0xB64BE8D8B25396C1, 0x70CB6AF7C2D5BCF0, 0x98F076A4F7A2322E,
    0xBF84470805E69B5F, 0x94C3251F06F90CF3, 0x3E003E616A6591E9, 0xB925A6CD0421AFF3,
    0x61BDD1307C66E300, 0xBF8D5108E27E0D48, 0x240AB57A8B888B20, 0xFC87614BAF287E07,
    0xEF02CDD06FFDB432, 0xA1082C0466DF6C0A, 0x8215E577001332C8, 0xD39BB9C3A48DB6CF,
    0x2738259634305C14, 0x61CF4F94C97DF93D, 0x1B6BACA2AE4E125B, 0x758F450C88572E0B,
    0x959F587D507A8359, 0xB063E962E045F54D, 0x60E8ED72C0DFF5D1, 0x7B64978555326F9F,
    0xFD080D236DA814BA, 0x8C90FD9B083F4558, 0x106F72FE81E2C590, 0x7976033A39F7D952,
    0xA4EC0132764CA04B, 0x733EA705FAE4FA77, 0xB4D8F77BC3E56167, 0x9E21F4F903B33FD9,
    0x9D765E419FB69F6D, 0xD30C088BA61EA5EF, 0x5D94337FBFAF7F5B, 0x1A4E4822EB4D7A59,
    0x6FFE73E81B637FB3, 0xDDF957BC36D8B9CA, 0x64D0E29EEA8838B3, 0x08DD9BDFD96B9F63,
    0x087E79E5A57D1D13, 0xE328E230E3E2B3FB, 0x1C2559E30F0946BE, 0x720BF5F26F4D2EAA,
    0xB0774D261CC609DB, 0x443F64EC5A371195, 0x4112CF68649A260E, 0xD813F2FAB7F5C5CA,
    0x660D3257380841EE, 0x59AC2C7873F910A3, 0xE846963877671A17, 0x93B633ABFA3469F8,
    0xC0C0F5A60EF4CDCF, 0xCAF21ECD4377B28C, 0x57277707199B8175, 0x506C11B9D90E8B1D,
    0xD83CC2687A19255F, 0x4A29C6465A314CD1, 0xED2DF21216235097, 0xB5635C95FF7296E2,
    0x22AF003AB672E811, 0x52E762596BF68235, 0x9AEBA33AC6ECC6B0, 0x944F6DE09134DFB6,
    0x6C47BEC883A7DE39, 0x6AD047C430A12104, 0xA5B1CFDBA0AB4067, 0x7C45D833AFF07862,
    0x5092EF950A16DA0B, 0x9338E69C052B8E7B, 0x455A4B4CFE30E3F5, 0x6B02E63195AD0CF8,
    0x6B17B224BAD6BF27, 0xD1E0CCD25BB9C169, 0xDE0C89A556B9AE70, 0x50065E535A213CF6,
    0x9C1169FA2777B874, 0x78EDEFD694AF1EED, 0x6DC93D9526A50E68, 0xEE97F453F06791ED,
    0x32AB0EDB696703D3, 0x3A6853C7E70757A7, 0x31865CED6120F37D, 0x67FEF95D92607890,
    0x1F2B1D1F15F6DC9C, 0xB69E38A8965C6B65, 0xAA9119FF184CCCF4, 0xF43C732873F24C13,
    0xFB4A3D794A9A80D2, 0x3550C2321FD6109C, 0x371F77E76BB8417E, 0x6BFA9AAE5EC05779,
    0xCD04F3FF001A4778, 0xE3273522064480CA, 0x9F91508BFFCFC14A, 0x049A7F41061A9E60,
    0xFCB6BE43A9F2FE9B, 0x08DE8A1C7797DA9B, 0x8F9887E6078735A1, 0xB5B4071DBFC73A66,
    0x230E343DFBA08D33, 0x43ED7F5A0FAE657D, 0x3A88A0FBBCB05C63, 0x21874B8B4D2DBC4F,
    0x1BDEA12E35F6A8C9, 0x53C065C6C8E63528, 0xE34A1D250E7A8D6B, 0xD6B04D3B7651DD7E,
    0x5E90277E7CB39E2D, 0x2C046F22062DC67D, 0xB10BB459132D0A26, 0x3FA9DDFB67E2F199,
    0x0E09B88E1914F7AF, 0x10E8B35AF3EEAB37, 0x9EEDECA8E272B933, 0xD4C718BC4AE8AE5F,
    0x81536D601170FC20, 0x91B534F885818A06, 0xEC8177F83F900978, 0x190E714FADA5156E,
    0xB592BF39B0364963, 0x89C350C893AE7DC1, 0xAC042E70F8B383F2, 0xB49B52E587A1EE60,
    0xFB152FE3FF26DA89, 0x3E666E6F69AE2C15, 0x3B544EBE544C19F9, 0xE805A1E290CF2456,
    0x24B33C9D7ED25117, 0xE74733427B72F0C1, 0x0A804D18B7097475, 0x57E3306D881EDB4F,
    0x4AE7D6A36EB5DBCB, 0x2D8D5432157064C8, 0xD1E649DE1E7F268B, 0x8A328A1CEDFE552C,
    0x07A3AEC79624C7DA, 0x84547DDC3E203C94, 0x990A98FD5071D263, 0x1A4FF12616EEFC89,
    0xF6F7FD1431714200, 0x30C05B1BA332F41C, 0x8D2636B81555A786, 0x46C9FEB55D120902,
    0xCCEC0A73B49C9921, 0x4E9D2827355FC492, 0x19EBB029435DCB0F, 0x4659D2B743848A2C,
    0x963EF2C96B33BE31, 0x74F85198B05A2E7D, 0x5A0F544DD2B1FB18, 0x03727073C2E134B1,
    0xC7F6AA2DE59AEA61, 0x352787BAA0D7C22F, 0x9853EAB63B5E0B35, 0xABBDCDD7ED5C0860,
    0xCF05DAF5AC8D77B0, 0x49CAD48CEBF4A71E, 0x7A4C10EC2158C4A6, 0xD9E92AA246BF719E,
    0x13AE978D09FE5557, 0x730499AF921549FF, 0x4E4B705B92903BA4, 0xFF577222C14F0A3A,
    0x55B6344CF97AAFAE, 0xB862225B055B6960, 0xCAC09AFBDDD2CDB4, 0xDAF8E9829FE96B5F,
    0xB5FDFC5D3132C498, 0x310CB380DB6F7503, 0xE87FBB46217A360E, 0x2102AE466EBB1148,
    0xF8549E1A3AA5E00D, 0x07A69AFDCC42261A, 0xC4C118BFE78FEAAE, 0xF9F4892ED96BD438,
    0x1AF3DBE25D8F45DA, 0xF5B4B0B0D2DEEEB4, 0x962ACEEFA82E1C84, 0x046E3ECAAF453CE9,
    0xF05D129681949A4C, 0x964781CE734B3C84, 0x9C2ED44081CE5FBD, 0x522E23F3925E319E,
    0x177E00F9FC32F791, 0x2BC60A63A6F3B3F2, 0x222BBFAE61725606, 0x486289DDCC3D6780,
    0x7DC7785B8EFDFC80, 0x8AF38731C02BA980, 0x1FAB64EA29A2DDF7, 0xE4D9429322CD065A,
    0x9DA058C67844F20C, 0x24C0E332B70019B0, 0x233003B5A6CFE6AD, 0xD586BD01C5C217F6,
    0x5E5637885F29BC2B, 0x7EBA726D8C94094B, 0x0A56A5F0BFE39272, 0xD79476A84EE20D06,
    0x9E4C1269BAA4BF37, 0x17EFEE45B0DEE640, 0x1D95B0A5FCF90BC6, 0x93CBE0B699C2585D,
    0x65FA4F227A2B6D79, 0xD5F9E858292504D5, 0xC2B5A03F71471A6F, 0x59300222B4561E00,
    0xCE2F8642CA0712DC, 0x7CA9723FBB2E8988, 0x2785338347F2BA08, 0xC61BB3A141E50E8C,
    0x150F361DAB9DEC26, 0x9F6A419D382595F4, 0x64A53DC924FE7AC9, 0x142DE49FFF7A7C3D,
    0x0C335248857FA9E7, 0x0A9C32D5EAE45305, 0xE6C42178C4BBB92E, 0x71F1CE2490D20B07,
    0xF1BCC3D275AFE51A, 0xE728E8C83C334074, 0x96FBF83A12884624, 0x81A1549FD6573DA5,
    0x5FA7867CAF35E149, 0x56986E2EF3ED091B, 0x917F1DD5F8886C61, 0xD20D8C88C8FFE65F,
    0x31D71DCE64B2C310, 0xF165B587DF898190, 0xA57E6339DD2CF3A0, 0x1EF6E6DBB1961EC9,
    0x70CC73D90BC26E24, 0xE21A6B35DF0C3AD7, 0x003A93D8B2806962, 0x1C99DED33CB890A1,
    0xCF3145DE0ADD4289, 0xD0E4427A5514FB72, 0x77C621CC9FB3A483, 0x67A34DAC4356550B,
    0xF8D626AAAF278509
]


class ZobristHasher:
    def __init__(self, array: List[int]) -> None:
        assert len(array) >= 781
        self.array = array

    def hash_board(self, board: chess.BaseBoard) -> int:
        zobrist_hash = 0

        for pivot, squares in enumerate(board.occupied_co):
            for square in chess.scan_reversed(squares):
                piece_index = (typing.cast(chess.PieceType, board.piece_type_at(square)) - 1) * 2 + pivot
                zobrist_hash ^= self.array[64 * piece_index + square]

        return zobrist_hash

    def hash_castling(self, board: chess.Board) -> int:
        zobrist_hash = 0

        # Hash in the castling flags.
        if board.has_kingside_castling_rights(chess.WHITE):
            zobrist_hash ^= self.array[768]
        if board.has_queenside_castling_rights(chess.WHITE):
            zobrist_hash ^= self.array[768 + 1]
        if board.has_kingside_castling_rights(chess.BLACK):
            zobrist_hash ^= self.array[768 + 2]
        if board.has_queenside_castling_rights(chess.BLACK):
            zobrist_hash ^= self.array[768 + 3]

        return zobrist_hash

    def hash_ep_square(self, board: chess.Board) -> int:
        # Hash in the en passant file.
        if board.ep_square:
            # But only if there's actually a pawn ready to capture it. Legality
            # of the potential capture is irrelevant.
            if board.turn == chess.WHITE:
                ep_mask = chess.shift_down(chess.BB_SQUARES[board.ep_square])
            else:
                ep_mask = chess.shift_up(chess.BB_SQUARES[board.ep_square])
            ep_mask = chess.shift_left(ep_mask) | chess.shift_right(ep_mask)

            if ep_mask & board.pawns & board.occupied_co[board.turn]:
                return self.array[772 + chess.square_file(board.ep_square)]
        return 0

    def hash_turn(self, board: chess.Board) -> int:
        # Hash in the turn.
        return self.array[780] if board.turn == chess.WHITE else 0

    def __call__(self, board: chess.Board) -> int:
        return (self.hash_board(board) ^ self.hash_castling(board) ^
                self.hash_ep_square(board) ^ self.hash_turn(board))


def zobrist_hash(board: chess.Board, *, _hasher: Callable[[chess.Board], int] = ZobristHasher(POLYGLOT_RANDOM_ARRAY)) -> int:
    """
    Calculates the Polyglot Zobrist hash of the position.

    A Zobrist hash is an XOR of pseudo-random values picked from
    an array. Which values are picked is decided by features of the
    position, such as piece positions, castling rights and en passant
    squares.
    """
    return _hasher(board)


class Entry(NamedTuple):
    """An entry from a Polyglot opening book."""

    key: int
    """The Zobrist hash of the position."""

    raw_move: int
    """
    The raw binary representation of the move. Use
    :data:`~chess.polyglot.Entry.move` instead.
    """

    weight: int
    """An integer value that can be used as the weight for this entry."""

    learn: int
    """Another integer value that can be used for extra information."""

    move: chess.Move
    """The :class:`~chess.Move`."""


class _EmptyMmap(bytearray):
    def size(self) -> int:
        return 0

    def close(self) -> None:
        pass

    def madvise(self, option: int) -> None:
        pass


def _randint(rng: Optional[random.Random], a: int, b: int) -> int:
    return random.randint(a, b) if rng is None else rng.randint(a, b)


class MemoryMappedReader:
    """Maps a Polyglot opening book to memory."""

    def __init__(self, filename: StrOrBytesPath) -> None:
        fd = os.open(filename, os.O_RDONLY | os.O_BINARY if hasattr(os, "O_BINARY") else os.O_RDONLY)
        try:
            self.mmap: Union[mmap.mmap, _EmptyMmap] = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            self.mmap = _EmptyMmap()  # Workaround for empty opening books.
        finally:
            os.close(fd)

        if self.mmap.size() % ENTRY_STRUCT.size != 0:
            raise IOError(f"invalid file size: ensure {filename!r} is a valid polyglot opening book")

        try:
            # Unix
            self.mmap.madvise(mmap.MADV_RANDOM)
        except AttributeError:
            pass

    def __enter__(self) -> MemoryMappedReader:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        return self.close()

    def __len__(self) -> int:
        return self.mmap.size() // ENTRY_STRUCT.size

    def __getitem__(self, index: int) -> Entry:
        if index < 0:
            index = len(self) + index

        try:
            key, raw_move, weight, learn = ENTRY_STRUCT.unpack_from(self.mmap, index * ENTRY_STRUCT.size)
        except struct.error:
            raise IndexError()

        # Extract source and target square.
        to_square = raw_move & 0x3f
        from_square = (raw_move >> 6) & 0x3f

        # Extract the promotion type.
        promotion_part = (raw_move >> 12) & 0x7
        promotion = promotion_part + 1 if promotion_part else None

        # Piece drop.
        if from_square == to_square:
            promotion, drop = None, promotion
        else:
            drop = None

        # Entry with move (not normalized).
        move = chess.Move(from_square, to_square, promotion, drop)
        return Entry(key, raw_move, weight, learn, move)

    def __iter__(self) -> Iterator[Entry]:
        for i in range(len(self)):
            yield self[i]

    def bisect_key_left(self, key: int) -> int:
        lo = 0
        hi = len(self)

        while lo < hi:
            mid = (lo + hi) // 2
            mid_key, _, _, _ = ENTRY_STRUCT.unpack_from(self.mmap, mid * ENTRY_STRUCT.size)
            if mid_key < key:
                lo = mid + 1
            else:
                hi = mid

        return lo

    def __contains__(self, entry: Entry) -> bool:
        return any(current == entry for current in self.find_all(entry.key, minimum_weight=entry.weight))

    def find_all(self, board: Union[chess.Board, int], *, minimum_weight: int = 1, exclude_moves: Container[chess.Move] = []) -> Iterator[Entry]:
        """Seeks a specific position and yields corresponding entries."""
        try:
            key = int(board)  # type: ignore
            context: Optional[chess.Board] = None
        except (TypeError, ValueError):
            context = typing.cast(chess.Board, board)
            key = zobrist_hash(context)

        i = self.bisect_key_left(key)
        size = len(self)

        while i < size:
            entry = self[i]
            i += 1

            if entry.key != key:
                break

            if entry.weight < minimum_weight:
                continue

            if context:
                move = context._from_chess960(context.chess960, entry.move.from_square, entry.move.to_square, entry.move.promotion, entry.move.drop)
                entry = Entry(entry.key, entry.raw_move, entry.weight, entry.learn, move)

            if exclude_moves and entry.move in exclude_moves:
                continue

            if context and not context.is_legal(entry.move):
                continue

            yield entry

    def find(self, board: Union[chess.Board, int], *, minimum_weight: int = 1, exclude_moves: Container[chess.Move] = []) -> Entry:
        """
        Finds the main entry for the given position or Zobrist hash.

        The main entry is the (first) entry with the highest weight.

        By default, entries with weight ``0`` are excluded. This is a common
        way to delete entries from an opening book without compacting it. Pass
        *minimum_weight* ``0`` to select all entries.

        :raises: :exc:`IndexError` if no entries are found. Use
            :func:`~chess.polyglot.MemoryMappedReader.get()` if you prefer to
            get ``None`` instead of an exception.
        """
        try:
            return max(self.find_all(board, minimum_weight=minimum_weight, exclude_moves=exclude_moves), key=lambda entry: entry.weight)
        except ValueError:
            raise IndexError()

    def get(self, board: Union[chess.Board, int], default: Optional[Entry] = None, *, minimum_weight: int = 1, exclude_moves: Container[chess.Move] = []) -> Optional[Entry]:
        try:
            return self.find(board, minimum_weight=minimum_weight, exclude_moves=exclude_moves)
        except IndexError:
            return default

    def choice(self, board: Union[chess.Board, int], *, minimum_weight: int = 1, exclude_moves: Container[chess.Move] = [], random: Optional[random.Random] = None) -> Entry:
        """
        Uniformly selects a random entry for the given position.

        :raises: :exc:`IndexError` if no entries are found.
        """
        chosen_entry = None

        for i, entry in enumerate(self.find_all(board, minimum_weight=minimum_weight, exclude_moves=exclude_moves)):
            if chosen_entry is None or _randint(random, 0, i) == i:
                chosen_entry = entry

        if chosen_entry is None:
            raise IndexError()

        return chosen_entry

    def weighted_choice(self, board: Union[chess.Board, int], *, exclude_moves: Container[chess.Move] = [], random: Optional[random.Random] = None) -> Entry:
        """
        Selects a random entry for the given position, distributed by the
        weights of the entries.

        :raises: :exc:`IndexError` if no entries are found.
        """
        total_weights = sum(entry.weight for entry in self.find_all(board, exclude_moves=exclude_moves))
        if not total_weights:
            raise IndexError()

        choice = _randint(random, 0, total_weights - 1)

        current_sum = 0
        for entry in self.find_all(board, exclude_moves=exclude_moves):
            current_sum += entry.weight
            if current_sum > choice:
                return entry

        assert False

    def close(self) -> None:
        """Closes the reader."""
        self.mmap.close()


def open_reader(path: StrOrBytesPath) -> MemoryMappedReader:
    """
    Creates a reader for the file at the given path.

    The following example opens a book to find all entries for the start
    position:

    >>> import chess
    >>> import chess.polyglot
    >>>
    >>> board = chess.Board()
    >>>
    >>> with chess.polyglot.open_reader("data/polyglot/performance.bin") as reader:
    ...    for entry in reader.find_all(board):
    ...        print(entry.move, entry.weight, entry.learn)
    e2e4 1 0
    d2d4 1 0
    c2c4 1 0
    """
    return MemoryMappedReader(path)
<|endoftext|>from __future__ import annotations

import chess
import itertools
import typing

from typing import Dict, Generic, Hashable, Iterable, Iterator, List, Optional, Type, TypeVar, Union

if typing.TYPE_CHECKING:
    from typing_extensions import Self


class SuicideBoard(chess.Board):

    aliases = ["Suicide", "Suicide chess"]
    uci_variant = "suicide"
    xboard_variant = "suicide"

    tbw_suffix = ".stbw"
    tbz_suffix = ".stbz"
    tbw_magic = b"\x7b\xf6\x93\x15"
    tbz_magic = b"\xe4\xcf\xe7\x23"
    pawnless_tbw_suffix = ".gtbw"
    pawnless_tbz_suffix = ".gtbz"
    pawnless_tbw_magic = b"\xbc\x55\xbc\x21"
    pawnless_tbz_magic = b"\xd6\xf5\x1b\x50"
    connected_kings = True
    one_king = False
    captures_compulsory = True

    def pin_mask(self, color: chess.Color, square: chess.Square) -> chess.Bitboard:
        return chess.BB_ALL

    def _attacked_for_king(self, path: chess.Bitboard, occupied: chess.Bitboard) -> bool:
        return False

    def checkers_mask(self) -> chess.Bitboard:
        return chess.BB_EMPTY

    def gives_check(self, move: chess.Move) -> bool:
        return False

    def is_into_check(self, move: chess.Move) -> bool:
        return False

    def was_into_check(self) -> bool:
        return False

    def _material_balance(self) -> int:
        return (chess.popcount(self.occupied_co[self.turn]) -
                chess.popcount(self.occupied_co[not self.turn]))

    def is_variant_end(self) -> bool:
        return not all(has_pieces for has_pieces in self.occupied_co)

    def is_variant_win(self) -> bool:
        if not self.occupied_co[self.turn]:
            return True
        else:
            return self.is_stalemate() and self._material_balance() < 0

    def is_variant_loss(self) -> bool:
        if not self.occupied_co[self.turn]:
            return False
        else:
            return self.is_stalemate() and self._material_balance() > 0

    def is_variant_draw(self) -> bool:
        if not self.occupied_co[self.turn]:
            return False
        else:
            return self.is_stalemate() and self._material_balance() == 0

    def has_insufficient_material(self, color: chess.Color) -> bool:
        if not self.occupied_co[color]:
            return False
        elif not self.occupied_co[not color]:
            return True
        elif self.occupied == self.bishops:
            # In a position with only bishops, check if all our bishops can be
            # captured.
            we_some_on_light = bool(self.occupied_co[color] & chess.BB_LIGHT_SQUARES)
            we_some_on_dark = bool(self.occupied_co[color] & chess.BB_DARK_SQUARES)
            they_all_on_dark = not (self.occupied_co[not color] & chess.BB_LIGHT_SQUARES)
            they_all_on_light = not (self.occupied_co[not color] & chess.BB_DARK_SQUARES)
            return (we_some_on_light and they_all_on_dark) or (we_some_on_dark and they_all_on_light)
        elif self.occupied == self.knights and chess.popcount(self.knights) == 2:
            return (
                self.turn == color ^
                bool(self.occupied_co[chess.WHITE] & chess.BB_LIGHT_SQUARES) ^
                bool(self.occupied_co[chess.BLACK] & chess.BB_DARK_SQUARES))
        else:
            return False

    def generate_pseudo_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        for move in super().generate_pseudo_legal_moves(from_mask, to_mask):
            # Add king promotions.
            if move.promotion == chess.QUEEN:
                yield chess.Move(move.from_square, move.to_square, chess.KING)

            yield move

    def generate_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        if self.is_variant_end():
            return

        # Generate captures first.
        found_capture = False
        for move in self.generate_pseudo_legal_captures():
            if chess.BB_SQUARES[move.from_square] & from_mask and chess.BB_SQUARES[move.to_square] & to_mask:
                yield move
            found_capture = True

        # Captures are mandatory. Stop here if any were found.
        if not found_capture:
            not_them = to_mask & ~self.occupied_co[not self.turn]
            for move in self.generate_pseudo_legal_moves(from_mask, not_them):
                if not self.is_en_passant(move):
                    yield move

    def is_legal(self, move: chess.Move) -> bool:
        if not super().is_legal(move):
            return False

        if self.is_capture(move):
            return True
        else:
            return not any(self.generate_pseudo_legal_captures())

    def _transposition_key(self) -> Hashable:
        if self.has_chess960_castling_rights():
            return (super()._transposition_key(), self.kings & self.promoted)
        else:
            return super()._transposition_key()

    def board_fen(self, *, promoted: Optional[bool] = None) -> str:
        if promoted is None:
            promoted = self.has_chess960_castling_rights()
        return super().board_fen(promoted=promoted)

    def status(self) -> chess.Status:
        status = super().status()
        status &= ~chess.STATUS_NO_WHITE_KING
        status &= ~chess.STATUS_NO_BLACK_KING
        status &= ~chess.STATUS_TOO_MANY_KINGS
        status &= ~chess.STATUS_OPPOSITE_CHECK
        return status


class GiveawayBoard(SuicideBoard):

    aliases = ["Giveaway", "Giveaway chess", "Give away", "Give away chess"]
    uci_variant = "giveaway"
    xboard_variant = "giveaway"

    tbw_suffix = ".gtbw"
    tbz_suffix = ".gtbz"
    tbw_magic = b"\xbc\x55\xbc\x21"
    tbz_magic = b"\xd6\xf5\x1b\x50"
    pawnless_tbw_suffix = ".stbw"
    pawnless_tbz_suffix = ".stbz"
    pawnless_tbw_magic = b"\x7b\xf6\x93\x15"
    pawnless_tbz_magic = b"\xe4\xcf\xe7\x23"

    def is_variant_win(self) -> bool:
        return not self.occupied_co[self.turn] or self.is_stalemate()

    def is_variant_loss(self) -> bool:
        return False

    def is_variant_draw(self) -> bool:
        return False


class AntichessBoard(GiveawayBoard):

    aliases = ["Antichess", "Anti chess", "Anti"]
    uci_variant = "antichess"  # Unofficial
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)

    def reset(self) -> None:
        super().reset()
        self.castling_rights = chess.BB_EMPTY


class AtomicBoard(chess.Board):

    aliases = ["Atomic", "Atom", "Atomic chess"]
    uci_variant = "atomic"
    xboard_variant = "atomic"

    tbw_suffix = ".atbw"
    tbz_suffix = ".atbz"
    tbw_magic = b"\x55\x8d\xa4\x49"
    tbz_magic = b"\x91\xa9\x5e\xeb"
    connected_kings = True

    def is_variant_end(self) -> bool:
        return not all(self.kings & side for side in self.occupied_co)

    def is_variant_win(self) -> bool:
        return bool(self.kings and not self.kings & self.occupied_co[not self.turn])

    def is_variant_loss(self) -> bool:
        return bool(self.kings and not self.kings & self.occupied_co[self.turn])

    def has_insufficient_material(self, color: chess.Color) -> bool:
        # Remaining material does not matter if opponent's king is already
        # exploded.
        if not (self.occupied_co[not color] & self.kings):
            return False

        # Bare king can not mate.
        if not (self.occupied_co[color] & ~self.kings):
            return True

        # As long as the opponent's king is not alone, there is always a chance
        # their own pieces explode next to it.
        if self.occupied_co[not color] & ~self.kings:
            # Unless there are only bishops that cannot explode each other.
            if self.occupied == self.bishops | self.kings:
                if not (self.bishops & self.occupied_co[chess.WHITE] & chess.BB_DARK_SQUARES):
                    return not (self.bishops & self.occupied_co[chess.BLACK] & chess.BB_LIGHT_SQUARES)
                if not (self.bishops & self.occupied_co[chess.WHITE] & chess.BB_LIGHT_SQUARES):
                    return not (self.bishops & self.occupied_co[chess.BLACK] & chess.BB_DARK_SQUARES)
            return False

        # Queen or pawn (future queen) can give mate against bare king.
        if self.queens or self.pawns:
            return False

        # Single knight, bishop or rook cannot mate against bare king.
        if chess.popcount(self.knights | self.bishops | self.rooks) == 1:
            return True

        # Two knights cannot mate against bare king.
        if self.occupied == self.knights | self.kings:
            return chess.popcount(self.knights) <= 2

        return False

    def _attacked_for_king(self, path: chess.Bitboard, occupied: chess.Bitboard) -> bool:
        # Can castle onto attacked squares if they are connected to the
        # enemy king.
        enemy_kings = self.kings & self.occupied_co[not self.turn]
        for enemy_king in chess.scan_forward(enemy_kings):
            path &= ~chess.BB_KING_ATTACKS[enemy_king]

        return super()._attacked_for_king(path, occupied)

    def _kings_connected(self) -> bool:
        white_kings = self.kings & self.occupied_co[chess.WHITE]
        black_kings = self.kings & self.occupied_co[chess.BLACK]
        return any(chess.BB_KING_ATTACKS[sq] & black_kings for sq in chess.scan_forward(white_kings))

    def _push_capture(self, move: chess.Move, capture_square: chess.Square, piece_type: chess.PieceType, was_promoted: bool) -> None:
        explosion_radius = chess.BB_KING_ATTACKS[move.to_square] & ~self.pawns

        # Destroy castling rights.
        self.castling_rights &= ~explosion_radius
        if explosion_radius & self.kings & self.occupied_co[chess.WHITE] & ~self.promoted:
            self.castling_rights &= ~chess.BB_RANK_1
        if explosion_radius & self.kings & self.occupied_co[chess.BLACK] & ~self.promoted:
            self.castling_rights &= ~chess.BB_RANK_8

        # Explode the capturing piece.
        self._remove_piece_at(move.to_square)

        # Explode all non pawns around.
        for explosion in chess.scan_forward(explosion_radius):
            self._remove_piece_at(explosion)

    def checkers_mask(self) -> chess.Bitboard:
        return chess.BB_EMPTY if self._kings_connected() else super().checkers_mask()

    def was_into_check(self) -> bool:
        return not self._kings_connected() and super().was_into_check()

    def is_into_check(self, move: chess.Move) -> bool:
        self.push(move)
        was_into_check = self.was_into_check()
        self.pop()
        return was_into_check

    def is_legal(self, move: chess.Move) -> bool:
        if self.is_variant_end():
            return False

        if not self.is_pseudo_legal(move):
            return False

        self.push(move)
        legal = bool(self.kings) and not self.is_variant_win() and (self.is_variant_loss() or not self.was_into_check())
        self.pop()

        return legal

    def is_stalemate(self) -> bool:
        return not self.is_variant_loss() and super().is_stalemate()

    def generate_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        for move in self.generate_pseudo_legal_moves(from_mask, to_mask):
            if self.is_legal(move):
                yield move

    def status(self) -> chess.Status:
        status = super().status()
        status &= ~chess.STATUS_OPPOSITE_CHECK
        if self.turn == chess.WHITE:
            status &= ~chess.STATUS_NO_WHITE_KING
        else:
            status &= ~chess.STATUS_NO_BLACK_KING
        if chess.popcount(self.checkers_mask()) <= 14:
            status &= ~chess.STATUS_TOO_MANY_CHECKERS
        if self._valid_ep_square() is None:
            status &= ~chess.STATUS_IMPOSSIBLE_CHECK
        return status


class KingOfTheHillBoard(chess.Board):

    aliases = ["King of the Hill", "KOTH", "kingOfTheHill"]
    uci_variant = "kingofthehill"
    xboard_variant = "kingofthehill"  # Unofficial

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def is_variant_end(self) -> bool:
        return bool(self.kings & chess.BB_CENTER)

    def is_variant_win(self) -> bool:
        return bool(self.kings & self.occupied_co[self.turn] & chess.BB_CENTER)

    def is_variant_loss(self) -> bool:
        return bool(self.kings & self.occupied_co[not self.turn] & chess.BB_CENTER)

    def has_insufficient_material(self, color: chess.Color) -> bool:
        return False


class RacingKingsBoard(chess.Board):

    aliases = ["Racing Kings", "Racing", "Race", "racingkings"]
    uci_variant = "racingkings"
    xboard_variant = "racingkings"  # Unofficial
    starting_fen = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)

    def reset(self) -> None:
        self.set_fen(type(self).starting_fen)

    def is_legal(self, move: chess.Move) -> bool:
        return super().is_legal(move) and not self.gives_check(move)

    def generate_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        for move in super().generate_legal_moves(from_mask, to_mask):
            if not self.gives_check(move):
                yield move

    def is_variant_end(self) -> bool:
        if not self.kings & chess.BB_RANK_8:
            return False

        black_kings = self.kings & self.occupied_co[chess.BLACK]
        if self.turn == chess.WHITE or black_kings & chess.BB_RANK_8 or not black_kings:
            return True

        # White has reached the backrank. The game is over if black can not
        # also reach the backrank on the next move. Check if there are any
        # safe squares for the king.
        black_king = chess.msb(black_kings)
        targets = chess.BB_KING_ATTACKS[black_king] & chess.BB_RANK_8 & ~self.occupied_co[chess.BLACK]
        return all(self.attackers_mask(chess.WHITE, target) for target in chess.scan_forward(targets))

    def is_variant_draw(self) -> bool:
        in_goal = self.kings & chess.BB_RANK_8
        return all(in_goal & side for side in self.occupied_co)

    def is_variant_loss(self) -> bool:
        return self.is_variant_end() and not self.kings & self.occupied_co[self.turn] & chess.BB_RANK_8

    def is_variant_win(self) -> bool:
        in_goal = self.kings & chess.BB_RANK_8
        return (
            self.is_variant_end() and
            bool(in_goal & self.occupied_co[self.turn]) and
            not in_goal & self.occupied_co[not self.turn])

    def has_insufficient_material(self, color: chess.Color) -> bool:
        return False

    def status(self) -> chess.Status:
        status = super().status()
        if self.is_check():
            status |= chess.STATUS_RACE_CHECK | chess.STATUS_TOO_MANY_CHECKERS | chess.STATUS_IMPOSSIBLE_CHECK
        if self.turn == chess.BLACK and all(self.occupied_co[co] & self.kings & chess.BB_RANK_8 for co in chess.COLORS):
            status |= chess.STATUS_RACE_OVER
        if self.pawns:
            status |= chess.STATUS_RACE_MATERIAL
        for color in chess.COLORS:
            if chess.popcount(self.occupied_co[color] & self.knights) > 2:
                status |= chess.STATUS_RACE_MATERIAL
            if chess.popcount(self.occupied_co[color] & self.bishops) > 2:
                status |= chess.STATUS_RACE_MATERIAL
            if chess.popcount(self.occupied_co[color] & self.rooks) > 2:
                status |= chess.STATUS_RACE_MATERIAL
            if chess.popcount(self.occupied_co[color] & self.queens) > 1:
                status |= chess.STATUS_RACE_MATERIAL
        return status


class HordeBoard(chess.Board):

    aliases = ["Horde", "Horde chess"]
    uci_variant = "horde"
    xboard_variant = "horde"  # Unofficial
    starting_fen = "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1"

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)

    def reset(self) -> None:
        self.set_fen(type(self).starting_fen)

    def is_variant_end(self) -> bool:
        return not all(has_pieces for has_pieces in self.occupied_co)

    def is_variant_draw(self) -> bool:
        return not self.occupied

    def is_variant_loss(self) -> bool:
        return bool(self.occupied) and not self.occupied_co[self.turn]

    def is_variant_win(self) -> bool:
        return bool(self.occupied) and not self.occupied_co[not self.turn]

    def has_insufficient_material(self, color: chess.Color) -> bool:
        # The side with the king can always win by capturing the Horde.
        if color == chess.BLACK:
            return False

        # See https://github.com/stevepapazis/horde-insufficient-material-tests
        # for how the following has been derived.

        white = self.occupied_co[chess.WHITE]
        queens = chess.popcount(white & self.queens)
        pawns = chess.popcount(white & self.pawns)
        rooks = chess.popcount(white & self.rooks)
        bishops = chess.popcount(white & self.bishops)
        knights = chess.popcount(white & self.knights)

        # Two same color bishops suffice to cover all the light and dark
        # squares around the enemy king.
        horde_darkb = chess.popcount(chess.BB_DARK_SQUARES & white & self.bishops)
        horde_lightb = chess.popcount(chess.BB_LIGHT_SQUARES & white & self.bishops)
        horde_bishop_co = chess.WHITE if horde_lightb >= 1 else chess.BLACK
        horde_num = (
            pawns + knights + rooks + queens +
            (horde_darkb if horde_darkb <= 2 else 2) +
            (horde_lightb if horde_lightb <= 2 else 2)
        )

        pieces = self.occupied_co[chess.BLACK]
        pieces_pawns = chess.popcount(pieces & self.pawns)
        pieces_bishops = chess.popcount(pieces & self.bishops)
        pieces_knights = chess.popcount(pieces & self.knights)
        pieces_rooks = chess.popcount(pieces & self.rooks)
        pieces_queens = chess.popcount(pieces & self.queens)
        pieces_darkb = chess.popcount(chess.BB_DARK_SQUARES & pieces & self.bishops)
        pieces_lightb = chess.popcount(chess.BB_LIGHT_SQUARES & pieces & self.bishops)
        pieces_num = chess.popcount(pieces)

        def pieces_oppositeb_of(square_color: chess.Color) -> int:
            return pieces_darkb if square_color == chess.WHITE else pieces_lightb

        def pieces_sameb_as(square_color: chess.Color) -> int:
            return pieces_lightb if square_color == chess.WHITE else pieces_darkb

        def pieces_of_type_not(piece: int) -> int:
            return pieces_num - piece

        def has_bishop_pair(side: chess.Color) -> bool:
            return (horde_lightb >= 1 and horde_darkb >= 1) if side == chess.WHITE else (pieces_lightb >= 1 and pieces_darkb >= 1)

        if horde_num == 0:
            return True
        if horde_num >= 4:
            # Four or more white pieces can always deliver mate.
            return False
        if (pawns >= 1 or queens >= 1) and horde_num >= 2:
            # Pawns/queens are never insufficient material when paired with any other
            # piece (a pawn promotes to a queen and delivers mate).
            return False
        if rooks >= 1 and horde_num >= 2:
            # A rook is insufficient material only when it is paired with a bishop
            # against a lone king. The horde can mate in any other case.
            # A rook on A1 and a bishop on C3 mate a king on B1 when there is a
            # friendly pawn/opposite-color-bishop/rook/queen on C2.
            # A rook on B8 and a bishop C3 mate a king on A1 when there is a friendly
            # knight on A2.
            if not (horde_num == 2 and rooks == 1 and bishops == 1 and pieces_of_type_not(pieces_sameb_as(horde_bishop_co)) == 1):
                return False

        if horde_num == 1:
            if pieces_num == 1:
                # A lone piece cannot mate a lone king.
                return True
            elif queens == 1:
                # The horde has a lone queen.
                # A lone queen mates a king on A1 bounded by:
                # - a pawn/rook on A2
                # - two same color bishops on A2, B1
                # We ignore every other mating case, since it can be reduced to
                # the two previous cases (e.g. a black pawn on A2 and a black
                # bishop on B1).
                return not (
                    pieces_pawns >= 1 or
                    pieces_rooks >= 1 or
                    pieces_lightb >= 2 or
                    pieces_darkb >= 2
                )
            elif pawns == 1:
                # Promote the pawn to a queen or a knight and check whether
                # white can mate.
                pawn_square = chess.SquareSet(self.pawns & white).pop()
                promote_to_queen = self.copy(stack=False)
                promote_to_queen.set_piece_at(pawn_square, chess.Piece(chess.QUEEN, chess.WHITE))
                promote_to_knight = self.copy(stack=False)
                promote_to_knight.set_piece_at(pawn_square, chess.Piece(chess.KNIGHT, chess.WHITE))
                return promote_to_queen.has_insufficient_material(chess.WHITE) and promote_to_knight.has_insufficient_material(chess.WHITE)
            elif rooks == 1:
                # A lone rook mates a king on A8 bounded by a pawn/rook on A7 and a
                # pawn/knight on B7. We ignore every other case, since it can be
                # reduced to the two previous cases.
                # (e.g. three pawns on A7, B7, C7)
                return not (
                    pieces_pawns >= 2 or
                    (pieces_rooks >= 1 and pieces_pawns >= 1) or
                    (pieces_rooks >= 1 and pieces_knights >= 1) or
                    (pieces_pawns >= 1 and pieces_knights >= 1)
                )
            elif bishops == 1:
                # The horde has a lone bishop.
                return not (
                    # The king can be mated on A1 if there is a pawn/opposite-color-bishop
                    # on A2 and an opposite-color-bishop on B1.
                    # If black has two or more pawns, white gets the benefit of the doubt;
                    # there is an outside chance that white promotes its pawns to
                    # opposite-color-bishops and selfmates theirself.
                    # Every other case that the king is mated by the bishop requires that
                    # black has two pawns or two opposite-color-bishop or a pawn and an
                    # opposite-color-bishop.
                    # For example a king on A3 can be mated if there is
                    # a pawn/opposite-color-bishop on A4, a pawn/opposite-color-bishop on
                    # B3, a pawn/bishop/rook/queen on A2 and any other piece on B2.
                    pieces_oppositeb_of(horde_bishop_co) >= 2 or
                    (pieces_oppositeb_of(horde_bishop_co) >= 1 and pieces_pawns >= 1) or
                    pieces_pawns >= 2
                )
            elif knights == 1:
                # The horde has a lone knight.
                return not (
                    # The king on A1 can be smother mated by a knight on C2 if there is
                    # a pawn/knight/bishop on B2, a knight/rook on B1 and any other piece
                    # on A2.
                    # Moreover, when black has four or more pieces and two of them are
                    # pawns, black can promote their pawns and selfmate theirself.
                    pieces_num >= 4 and (
                        pieces_knights >= 2 or pieces_pawns >= 2 or
                        (pieces_rooks >= 1 and pieces_knights >= 1) or
                        (pieces_rooks >= 1 and pieces_bishops >= 1) or
                        (pieces_knights >= 1 and pieces_bishops >= 1) or
                        (pieces_rooks >= 1 and pieces_pawns >= 1) or
                        (pieces_knights >= 1 and pieces_pawns >= 1) or
                        (pieces_bishops >= 1 and pieces_pawns >= 1) or
                        (has_bishop_pair(chess.BLACK) and pieces_pawns >= 1)
                    ) and
                    (pieces_of_type_not(pieces_darkb) >= 3 if pieces_darkb >= 2 else True) and
                    (pieces_of_type_not(pieces_lightb) >= 3 if pieces_lightb >= 2 else True)
                )
        elif horde_num == 2:  # By this point, we only need to deal with white's minor pieces.
            if pieces_num == 1:
                # Two minor pieces cannot mate a lone king.
                return True
            elif knights == 2:
                # A king on A1 is mated by two knights, if it is obstructed by a
                # pawn/bishop/knight on B2. On the other hand, if black only has
                # major pieces it is a draw.
                return not (pieces_pawns + pieces_bishops + pieces_knights >= 1)
            elif has_bishop_pair(chess.WHITE):
                return not (
                    # A king on A1 obstructed by a pawn/bishop on A2 is mated
                    # by the bishop pair.
                    pieces_pawns >= 1 or pieces_bishops >= 1 or
                    # A pawn/bishop/knight on B4, a pawn/bishop/rook/queen on
                    # A4 and the king on A3 enable Boden's mate by the bishop
                    # pair. In every other case white cannot win.
                    (pieces_knights >= 1 and pieces_rooks + pieces_queens >= 1)
                )
            elif bishops >= 1 and knights >= 1:
                # The horde has a bishop and a knight.
                return not (
                    # A king on A1 obstructed by a pawn/opposite-color-bishop on
                    # A2 is mated by a knight on D2 and a bishop on C3.
                    pieces_pawns >= 1 or pieces_oppositeb_of(horde_bishop_co) >= 1 or
                    # A king on A1 bounded by two friendly pieces on A2 and B1 is
                    # mated when the knight moves from D4 to C2 so that both the
                    # knight and the bishop deliver check.
                    pieces_of_type_not(pieces_sameb_as(horde_bishop_co)) >= 3
                )
            else:
                # The horde has two or more bishops on the same color.
                # White can only win if black has enough material to obstruct
                # the squares of the opposite color around the king.
                return not (
                    # A king on A1 obstructed by a pawn/opposite-bishop/knight
                    # on A2 and a opposite-bishop/knight on B1 is mated by two
                    # bishops on B2 and C3. This position is theoretically
                    # achievable even when black has two pawns or when they
                    # have a pawn and an opposite color bishop.
                    (pieces_pawns >= 1 and pieces_oppositeb_of(horde_bishop_co) >= 1) or
                    (pieces_pawns >= 1 and pieces_knights >= 1) or
                    (pieces_oppositeb_of(horde_bishop_co) >= 1 and pieces_knights >= 1) or
                    (pieces_oppositeb_of(horde_bishop_co) >= 2) or
                    pieces_knights >= 2 or
                    pieces_pawns >= 2
                    # In every other case, white can only draw.
                )
        elif horde_num == 3:
            # A king in the corner is mated by two knights and a bishop or three
            # knights or the bishop pair and a knight/bishop.
            if (knights == 2 and bishops == 1) or knights == 3 or has_bishop_pair(chess.WHITE):
                return False
            else:
                # White has two same color bishops and a knight.
                # A king on A1 is mated by a bishop on B2, a bishop on C1 and a
                # knight on C3, as long as there is another black piece to waste
                # a tempo.
                return pieces_num == 1

        return True

    def status(self) -> chess.Status:
        status = super().status()
        status &= ~chess.STATUS_NO_WHITE_KING

        if chess.popcount(self.occupied_co[chess.WHITE]) <= 36:
            status &= ~chess.STATUS_TOO_MANY_WHITE_PIECES
            status &= ~chess.STATUS_TOO_MANY_WHITE_PAWNS

        if not self.pawns & chess.BB_RANK_8 and not self.occupied_co[chess.BLACK] & self.pawns & chess.BB_RANK_1:
            status &= ~chess.STATUS_PAWNS_ON_BACKRANK

        if self.occupied_co[chess.WHITE] & self.kings:
            status |= chess.STATUS_TOO_MANY_KINGS

        return status


ThreeCheckBoardT = TypeVar("ThreeCheckBoardT", bound="ThreeCheckBoard")

class _ThreeCheckBoardState:
    def __init__(self, board: ThreeCheckBoard) -> None:
        self.remaining_checks_w = board.remaining_checks[chess.WHITE]
        self.remaining_checks_b = board.remaining_checks[chess.BLACK]

    def restore(self, board: ThreeCheckBoard) -> None:
        board.remaining_checks[chess.WHITE] = self.remaining_checks_w
        board.remaining_checks[chess.BLACK] = self.remaining_checks_b

class ThreeCheckBoard(chess.Board):

    aliases = ["Three-check", "Three check", "Threecheck", "Three check chess", "3-check", "3 check", "3check"]
    uci_variant = "3check"
    xboard_variant = "3check"
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1"

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        self.remaining_checks = [3, 3]
        self._three_check_stack: List[_ThreeCheckBoardState] = []
        super().__init__(fen, chess960=chess960)

    def clear_stack(self) -> None:
        super().clear_stack()
        self._three_check_stack.clear()

    def reset_board(self) -> None:
        super().reset_board()
        self.remaining_checks[chess.WHITE] = 3
        self.remaining_checks[chess.BLACK] = 3

    def clear_board(self) -> None:
        super().clear_board()
        self.remaining_checks[chess.WHITE] = 3
        self.remaining_checks[chess.BLACK] = 3

    def push(self, move: chess.Move) -> None:
        self._three_check_stack.append(_ThreeCheckBoardState(self))
        super().push(move)
        if self.is_check():
            self.remaining_checks[not self.turn] -= 1

    def pop(self) -> chess.Move:
        move = super().pop()
        self._three_check_stack.pop().restore(self)
        return move

    def has_insufficient_material(self, color: chess.Color) -> bool:
        # Any remaining piece can give check.
        return not (self.occupied_co[color] & ~self.kings)

    def set_epd(self, epd: str) -> Dict[str, Union[None, str, int, float, chess.Move, List[chess.Move]]]:
        parts = epd.strip().rstrip(";").split(None, 5)

        # Parse ops.
        if len(parts) > 5:
            operations = self._parse_epd_ops(parts.pop(), lambda: type(self)(" ".join(parts) + " 0 1"))
            parts.append(str(operations["hmvc"]) if "hmvc" in operations else "0")
            parts.append(str(operations["fmvn"]) if "fmvn" in operations else "1")
            self.set_fen(" ".join(parts))
            return operations
        else:
            self.set_fen(epd)
            return {}

    def set_fen(self, fen: str) -> None:
        parts = fen.split()

        # Extract check part.
        if len(parts) >= 7 and parts[6][0] == "+":
            check_part = parts.pop(6)
            try:
                w, b = check_part[1:].split("+", 1)
                wc, bc = 3 - int(w), 3 - int(b)
            except ValueError:
                raise ValueError(f"invalid check part in lichess three-check fen: {check_part!r}")
        elif len(parts) >= 5 and "+" in parts[4]:
            check_part = parts.pop(4)
            try:
                w, b = check_part.split("+", 1)
                wc, bc = int(w), int(b)
            except ValueError:
                raise ValueError(f"invalid check part in three-check fen: {check_part!r}")
        else:
            wc, bc = 3, 3

        # Set fen.
        super().set_fen(" ".join(parts))
        self.remaining_checks[chess.WHITE] = wc
        self.remaining_checks[chess.BLACK] = bc

    def epd(self, shredder: bool = False, en_passant: chess.EnPassantSpec = "legal", promoted: Optional[bool] = None, **operations: Union[None, str, int, float, chess.Move, Iterable[chess.Move]]) -> str:
        epd = [super().epd(shredder=shredder, en_passant=en_passant, promoted=promoted),
               "{:d}+{:d}".format(max(self.remaining_checks[chess.WHITE], 0),
                                  max(self.remaining_checks[chess.BLACK], 0))]
        if operations:
            epd.append(self._epd_operations(operations))
        return " ".join(epd)

    def is_variant_end(self) -> bool:
        return any(remaining_checks <= 0 for remaining_checks in self.remaining_checks)

    def is_variant_draw(self) -> bool:
        return self.remaining_checks[chess.WHITE] <= 0 and self.remaining_checks[chess.BLACK] <= 0

    def is_variant_loss(self) -> bool:
        return self.remaining_checks[not self.turn] <= 0 < self.remaining_checks[self.turn]

    def is_variant_win(self) -> bool:
        return self.remaining_checks[self.turn] <= 0 < self.remaining_checks[not self.turn]

    def is_irreversible(self, move: chess.Move) -> bool:
        return super().is_irreversible(move) or self.gives_check(move)

    def _transposition_key(self) -> Hashable:
        return (super()._transposition_key(),
                self.remaining_checks[chess.WHITE], self.remaining_checks[chess.BLACK])

    def copy(self, *, stack: Union[bool, int] = True) -> Self:
        board = super().copy(stack=stack)
        board.remaining_checks = self.remaining_checks.copy()
        if stack:
            stack = len(self.move_stack) if stack is True else stack
            board._three_check_stack = self._three_check_stack[-stack:]
        return board

    def root(self) -> Self:
        if self._three_check_stack:
            board = super().root()
            self._three_check_stack[0].restore(board)
            return board
        else:
            return self.copy(stack=False)

    def mirror(self) -> Self:
        board = super().mirror()
        board.remaining_checks[chess.WHITE] = self.remaining_checks[chess.BLACK]
        board.remaining_checks[chess.BLACK] = self.remaining_checks[chess.WHITE]
        return board


CrazyhouseBoardT = TypeVar("CrazyhouseBoardT", bound="CrazyhouseBoard")

class _CrazyhouseBoardState:
    def __init__(self, board: CrazyhouseBoard) -> None:
        self.pockets_w = board.pockets[chess.WHITE].copy()
        self.pockets_b = board.pockets[chess.BLACK].copy()

    def restore(self, board: CrazyhouseBoard) -> None:
        board.pockets[chess.WHITE] = self.pockets_w
        board.pockets[chess.BLACK] = self.pockets_b

CrazyhousePocketT = TypeVar("CrazyhousePocketT", bound="CrazyhousePocket")

class CrazyhousePocket:
    """A Crazyhouse pocket with a counter for each piece type."""

    def __init__(self, symbols: Iterable[str] = "") -> None:
        self.reset()
        for symbol in symbols:
            self.add(chess.PIECE_SYMBOLS.index(symbol))

    def reset(self) -> None:
        """Clears the pocket."""
        self._pieces = [-1, 0, 0, 0, 0, 0, 0]

    def add(self, piece_type: chess.PieceType) -> None:
        """Adds a piece of the given type to this pocket."""
        self._pieces[piece_type] += 1

    def remove(self, piece_type: chess.PieceType) -> None:
        """Removes a piece of the given type from this pocket."""
        assert self._pieces[piece_type], f"cannot remove {chess.piece_symbol(piece_type)} from {self!r}"
        self._pieces[piece_type] -= 1

    def count(self, piece_type: chess.PieceType) -> int:
        """Returns the number of pieces of the given type in the pocket."""
        return self._pieces[piece_type]

    def __str__(self) -> str:
        return "".join(chess.piece_symbol(pt) * self.count(pt) for pt in reversed(chess.PIECE_TYPES))

    def __len__(self) -> int:
        return sum(self._pieces[1:])

    def __repr__(self) -> str:
        return f"CrazyhousePocket('{self}')"

    def copy(self) -> Self:
        """Returns a copy of this pocket."""
        pocket = type(self)()
        pocket._pieces = self._pieces[:]
        return pocket

class CrazyhouseBoard(chess.Board):

    aliases = ["Crazyhouse", "Crazy House", "House", "ZH"]
    uci_variant = "crazyhouse"
    xboard_variant = "crazyhouse"
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        self.pockets = [CrazyhousePocket(), CrazyhousePocket()]
        self._crazyhouse_stack: List[_CrazyhouseBoardState] = []
        super().__init__(fen, chess960=chess960)

    def clear_stack(self) -> None:
        super().clear_stack()
        self._crazyhouse_stack.clear()

    def reset_board(self) -> None:
        super().reset_board()
        self.pockets[chess.WHITE].reset()
        self.pockets[chess.BLACK].reset()

    def clear_board(self) -> None:
        super().clear_board()
        self.pockets[chess.WHITE].reset()
        self.pockets[chess.BLACK].reset()

    def push(self, move: chess.Move) -> None:
        self._crazyhouse_stack.append(_CrazyhouseBoardState(self))
        super().push(move)
        if move.drop:
            self.pockets[not self.turn].remove(move.drop)

    def _push_capture(self, move: chess.Move, capture_square: chess.Square, piece_type: chess.PieceType, was_promoted: bool) -> None:
        if was_promoted:
            self.pockets[self.turn].add(chess.PAWN)
        else:
            self.pockets[self.turn].add(piece_type)

    def pop(self) -> chess.Move:
        move = super().pop()
        self._crazyhouse_stack.pop().restore(self)
        return move

    def _is_halfmoves(self, n: int) -> bool:
        # No draw by 50-move rule or 75-move rule.
        return False

    def is_irreversible(self, move: chess.Move) -> bool:
        return self._reduces_castling_rights(move)

    def _transposition_key(self) -> Hashable:
        return (super()._transposition_key(),
                self.promoted,
                str(self.pockets[chess.WHITE]), str(self.pockets[chess.BLACK]))

    def legal_drop_squares_mask(self) -> chess.Bitboard:
        king = self.king(self.turn)
        if king is None:
            return ~self.occupied

        king_attackers = self.attackers_mask(not self.turn, king)

        if not king_attackers:
            return ~self.occupied
        elif chess.popcount(king_attackers) == 1:
            return chess.between(king, chess.msb(king_attackers)) & ~self.occupied
        else:
            return chess.BB_EMPTY

    def legal_drop_squares(self) -> chess.SquareSet:
        """
        Gets the squares where the side to move could legally drop a piece.
        Does *not* check whether they actually have a suitable piece in their
        pocket.

        It is legal to drop a checkmate.

        Returns a :class:`set of squares <chess.SquareSet>`.
        """
        return chess.SquareSet(self.legal_drop_squares_mask())

    def is_pseudo_legal(self, move: chess.Move) -> bool:
        if move.drop and move.from_square == move.to_square:
            return (
                move.drop != chess.KING and
                not chess.BB_SQUARES[move.to_square] & self.occupied and
                not (move.drop == chess.PAWN and chess.BB_SQUARES[move.to_square] & chess.BB_BACKRANKS) and
                self.pockets[self.turn].count(move.drop) > 0)
        else:
            return super().is_pseudo_legal(move)

    def is_legal(self, move: chess.Move) -> bool:
        if move.drop:
            return self.is_pseudo_legal(move) and bool(self.legal_drop_squares_mask() & chess.BB_SQUARES[move.to_square])
        else:
            return super().is_legal(move)

    def generate_pseudo_legal_drops(self, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        for pt in chess.PIECE_TYPES:
