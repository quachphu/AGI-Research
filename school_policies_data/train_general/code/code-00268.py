    def add_value(self, metric_name: str, metric_value: int):
        self.handler.emit(
            MetricData(time.time(), self.group_name, metric_name, metric_value)
        )


_metrics_map: dict[str, MetricHandler] = {}
_default_metrics_handler: MetricHandler = NullMetricHandler()


# pyre-fixme[9]: group has type `str`; used as `None`.
def configure(handler: MetricHandler, group: str | None = None):
    if group is None:
        global _default_metrics_handler
        # pyre-fixme[9]: _default_metrics_handler has type `NullMetricHandler`; used
        #  as `MetricHandler`.
        _default_metrics_handler = handler
    else:
        _metrics_map[group] = handler


def getStream(group: str):
    handler = _metrics_map.get(group, _default_metrics_handler)
    return MetricStream(group, handler)


def _get_metric_name(fn):
    qualname = fn.__qualname__
    split = qualname.split(".")
    if len(split) == 1:
        module = fn.__module__
        if module:
            return module.split(".")[-1] + "." + split[0]
        else:
            return split[0]
    else:
        return qualname


def prof(fn=None, group: str = "torchelastic"):
    r"""
    @profile decorator publishes duration.ms, count, success, failure metrics for the function that it decorates.

    The metric name defaults to the qualified name (``class_name.def_name``) of the function.
    If the function does not belong to a class, it uses the leaf module name instead.

    Usage

    ::

     @metrics.prof
     def x():
         pass


     @metrics.prof(group="agent")
     def y():
         pass
    """

    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            key = _get_metric_name(f)
            try:
                start = time.time()
                result = f(*args, **kwargs)
                put_metric(f"{key}.success", 1, group)
            except Exception:
                put_metric(f"{key}.failure", 1, group)
                raise
            finally:
                put_metric(f"{key}.duration.ms", get_elapsed_time_ms(start), group)  # type: ignore[possibly-undefined]
            return result

        return wrapper

    if fn:
        return wrap(fn)
    else:
        return wrap


@deprecated("Deprecated, use `@prof` instead", category=FutureWarning)
def profile(group=None):
    """
    @profile decorator adds latency and success/failure metrics to any given function.

    Usage

    ::

     @metrics.profile("my_metric_group")
     def some_function(<arguments>):
    """

    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                # pyrefly: ignore [bad-argument-type]
                publish_metric(group, f"{func.__name__}.success", 1)
            except Exception:
                # pyrefly: ignore [bad-argument-type]
                publish_metric(group, f"{func.__name__}.failure", 1)
                raise
            finally:
                publish_metric(
                    # pyrefly: ignore [bad-argument-type]
                    group,
                    f"{func.__name__}.duration.ms",
                    get_elapsed_time_ms(start_time),  # type: ignore[possibly-undefined]
                )
            return result

        return wrapper

    return wrap


def put_metric(metric_name: str, metric_value: int, metric_group: str = "torchelastic"):
    """
    Publish a metric data point.

    Usage

    ::

     put_metric("metric_name", 1)
     put_metric("metric_name", 1, "metric_group_name")
    """
    getStream(metric_group).add_value(metric_name, metric_value)


@deprecated(
    "Deprecated, use `put_metric(metric_group)(metric_name, metric_value)` instead",
    category=FutureWarning,
)
def publish_metric(metric_group: str, metric_name: str, metric_value: int):
    metric_stream = getStream(metric_group)
    metric_stream.add_value(metric_name, metric_value)


def get_elapsed_time_ms(start_time_in_seconds: float):
    """Return the elapsed time in millis from the given start time."""
    end_time = time.time()
    return int((end_time - start_time_in_seconds) * 1000)
<|endoftext|>#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Module contains events processing mechanisms that are integrated with the standard python logging.

Example of usage:

::

  from torch.distributed.elastic import events

  event = events.Event(
      name="test_event", source=events.EventSource.WORKER, metadata={...}
  )
  events.get_logging_handler(destination="console").info(event)

"""

import inspect
import logging
import os
import socket
import traceback
from typing import Optional

from torch.distributed.elastic.events.handlers import get_logging_handler

from .api import Event, EventMetadataValue, EventSource, NodeState, RdzvEvent


_events_loggers: dict[str, logging.Logger] = {}


def _get_or_create_logger(destination: str = "null") -> logging.Logger:
    """
    Construct python logger based on the destination type or extends if provided.

    Available destination could be found in ``handlers.py`` file.
    The constructed logger does not propagate messages to the upper level loggers,
    e.g. root logger. This makes sure that a single event can be processed once.

    Args:
        destination: The string representation of the event handler.
            Available handlers found in ``handlers`` module
    """
    global _events_loggers

    if destination not in _events_loggers:
        _events_logger = logging.getLogger(f"torchelastic-events-{destination}")
        _events_logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
        # Do not propagate message to the root logger
        _events_logger.propagate = False

        logging_handler = get_logging_handler(destination)
        _events_logger.addHandler(logging_handler)

        # Add the logger to the global dictionary
        _events_loggers[destination] = _events_logger

    return _events_loggers[destination]


def record(event: Event, destination: str = "null") -> None:
    _get_or_create_logger(destination).info(event.serialize())


def record_rdzv_event(event: RdzvEvent) -> None:
    _get_or_create_logger("dynamic_rendezvous").info(event.serialize())


def construct_and_record_rdzv_event(
    run_id: str,
    message: str,
    node_state: NodeState,
    name: str = "",
    hostname: str = "",
    pid: int | None = None,
    master_endpoint: str = "",
    local_id: int | None = None,
    rank: int | None = None,
) -> None:
    """
    Initialize rendezvous event object and record its operations.

    Args:
        run_id (str): The run id of the rendezvous.
        message (str): The message describing the event.
        node_state (NodeState): The state of the node (INIT, RUNNING, SUCCEEDED, FAILED).
        name (str): Event name. (E.g. Current action being performed).
        hostname (str): Hostname of the node.
        pid (Optional[int]): The process id of the node.
        master_endpoint (str): The master endpoint for the rendezvous store, if known.
        local_id (Optional[int]):  The local_id of the node, if defined in dynamic_rendezvous.py
        rank (Optional[int]): The rank of the node, if known.
    Returns:
        None
    Example:
        >>> # See DynamicRendezvousHandler class
        >>> def _record(
        ...     self,
        ...     message: str,
        ...     node_state: NodeState = NodeState.RUNNING,
        ...     rank: Optional[int] = None,
        ... ) -> None:
        ...     construct_and_record_rdzv_event(
        ...         name=f"{self.__class__.__name__}.{get_method_name()}",
        ...         run_id=self._settings.run_id,
        ...         message=message,
        ...         node_state=node_state,
        ...         hostname=self._this_node.addr,
        ...         pid=self._this_node.pid,
        ...         local_id=self._this_node.local_id,
        ...         rank=rank,
        ...     )
    """
    # We don't want to perform an extra computation if not needed.
    if isinstance(get_logging_handler("dynamic_rendezvous"), logging.NullHandler):
        return

    # Set up parameters.
    if not hostname:
        hostname = socket.getfqdn()
    if not pid:
        pid = os.getpid()

    # Determines which file called this function.
    callstack = inspect.stack()
    filename = "no_file"
    if len(callstack) > 1:
        stack_depth_1 = callstack[1]
        filename = os.path.basename(stack_depth_1.filename)
        if not name:
            name = stack_depth_1.function

    # Delete the callstack variable. If kept, this can mess with python's
    # garbage collector as we are holding on to stack frame information in
    # the inspect module.
    del callstack

    # Set up error trace if this is an exception
    if node_state == NodeState.FAILED:
        error_trace = traceback.format_exc()
    else:
        error_trace = ""

    # Initialize event object
    event = RdzvEvent(
        name=f"{filename}:{name}",
        run_id=run_id,
        message=message,
        hostname=hostname,
        pid=pid,
        node_state=node_state,
        master_endpoint=master_endpoint,
        rank=rank,
        local_id=local_id,
        error_trace=error_trace,
    )

    # Finally, record the event.
    record_rdzv_event(event)
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Union


__all__ = ["EventSource", "Event", "NodeState", "RdzvEvent"]

EventMetadataValue = str | int | float | bool | None


class EventSource(str, Enum):
    """Known identifiers of the event producers."""

    AGENT = "AGENT"
    WORKER = "WORKER"


@dataclass
class Event:
    """
    The class represents the generic event that occurs during the torchelastic job execution.

    The event can be any kind of meaningful action.

    Args:
        name: event name.
        source: the event producer, e.g. agent or worker
        timestamp: timestamp in milliseconds when event occurred.
        metadata: additional data that is associated with the event.
    """

    name: str
    source: EventSource
    timestamp: int = 0
    metadata: dict[str, EventMetadataValue] = field(default_factory=dict)

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "Event"]) -> "Event":
        if isinstance(data, Event):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict["source"] = EventSource[data_dict["source"]]  # type: ignore[possibly-undefined]
        # pyrefly: ignore [unbound-name]
        return Event(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))


class NodeState(str, Enum):
    """The states that a node can be in rendezvous."""

    INIT = "INIT"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class RdzvEvent:
    """
    Dataclass to represent any rendezvous event.

    Args:
        name: Event name. (E.g. Current action being performed)
        run_id: The run id of the rendezvous
        message: The message describing the event
        hostname: Hostname of the node
        pid: The process id of the node
        node_state: The state of the node (INIT, RUNNING, SUCCEEDED, FAILED)
        master_endpoint: The master endpoint for the rendezvous store, if known
        rank: The rank of the node, if known
        local_id: The local_id of the node, if defined in dynamic_rendezvous.py
        error_trace: Error stack trace, if this is an error event.
    """

    name: str
    run_id: str
    message: str
    hostname: str
    pid: int
    node_state: NodeState
    master_endpoint: str = ""
    rank: int | None = None
    local_id: int | None = None
    error_trace: str = ""

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "RdzvEvent"]) -> "RdzvEvent":
        if isinstance(data, RdzvEvent):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict["node_state"] = NodeState[data_dict["node_state"]]  # type: ignore[possibly-undefined]
        # pyrefly: ignore [unbound-name]
        return RdzvEvent(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging


_log_handlers: dict[str, logging.Handler] = {
    "console": logging.StreamHandler(),
    "dynamic_rendezvous": logging.NullHandler(),
    "null": logging.NullHandler(),
}


def get_logging_handler(destination: str = "null") -> logging.Handler:
    global _log_handlers
    return _log_handlers[destination]
<|endoftext|>import os

import torch.distributed as dist


def get_rank() -> int:
    return int(os.environ["RANK"])


def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def tcpstore_client(prefix: str = "debug_server") -> dist.Store:
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = int(os.environ["MASTER_PORT"])

    store = dist.TCPStore(
        host_name=MASTER_ADDR,
        port=MASTER_PORT,
        is_master=False,
    )
    if prefix:
        store = dist.PrefixStore(prefix, store)
    return store
<|endoftext|>import asyncio
import json
import logging
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from jinja2 import DictLoader, Environment

from torch.distributed.debug._store import get_world_size, tcpstore_client


logger: logging.Logger = logging.getLogger(__name__)

_DEFAULT_FETCH_TIMEOUT: float = 60.0


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Response:
    status_code: int
    text: str

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self):
        return json.loads(self.text)


@dataclass(slots=True)
class NavLink:
    path: str
    label: str


@dataclass(slots=True)
class Route:
    path: str
    handler: Callable[["HTTPRequestHandler"], bytes]


class DebugHandler(ABC):
    fetch_timeout: float = _DEFAULT_FETCH_TIMEOUT

    @abstractmethod
    def routes(self) -> list[Route]: ...

    @abstractmethod
    def nav_links(self) -> list[NavLink]: ...

    def templates(self) -> dict[str, str]:
        return {}

    def dump(self) -> str | None:
        return None

    def dump_filename(self) -> str:
        return type(self).__name__.lower()


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def fetch_thread_pool(urls: list[str], timeout: float) -> list[Response]:
    # late import for optional dependency
    import requests

    max_workers = 20

    def get(url: str) -> Response:
        try:
            resp = requests.post(url, timeout=timeout)
            return Response(resp.status_code, resp.text)
        except requests.exceptions.Timeout as e:
            return Response(408, f"Timeout: {e}")
        except requests.exceptions.ConnectionError as e:
            return Response(503, f"ConnectionError: {e}")
        except Exception as e:
            return Response(502, f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resps = list(executor.map(get, urls))

    return resps


def fetch_aiohttp(urls: list[str], timeout: float) -> list[Response]:
    # late import for optional dependency
    # pyrefly: ignore [missing-import]
    import aiohttp

    async def fetch(session: aiohttp.ClientSession, url: str) -> Response:
        try:
            async with session.post(url) as resp:
                text = await resp.text()
                return Response(resp.status, text)
        except asyncio.TimeoutError as e:
            return Response(408, f"TimeoutError: {e}")
        except aiohttp.ClientError as e:
            return Response(503, f"{type(e).__name__}: {e}")
        except Exception as e:
            return Response(502, f"{type(e).__name__}: {e}")

    async def gather(urls: list[str]) -> list[Response]:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            return list(await asyncio.gather(*[fetch(session, url) for url in urls]))

    return asyncio.run(gather(urls))


def fetch_all(
    endpoint: str, args: str = "", *, timeout: float
) -> tuple[list[str], list[Response]]:
    store = tcpstore_client()
    keys = [f"rank{r}" for r in range(get_world_size())]
    addrs = store.multi_get(keys)
    addrs = [f"{addr.decode()}/handler/{endpoint}?{args}" for addr in addrs]

    try:
        resps = fetch_aiohttp(addrs, timeout=timeout)
    except ImportError:
        resps = fetch_thread_pool(addrs, timeout=timeout)

    return addrs, resps


def format_json(blob: str):
    parsed = json.loads(blob)
    return json.dumps(parsed, indent=2)


def format_fetch_summary(addrs: list[str], resps: list[Response]) -> str | None:
    """Return a summary string if any workers failed, or None if all succeeded."""
    failed = [(i, r) for i, r in enumerate(resps) if r.status_code != 200]
    if not failed:
        return None
    total = len(addrs)
    ok = total - len(failed)
    lines = [f"PARTIAL DATA: {ok}/{total} workers responded"]
    for rank, resp in failed:
        lines.append(f"  Rank {rank}: {resp.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template constants
# ---------------------------------------------------------------------------


BASE_TEMPLATE = """
<!doctype html>
<head>
    <title>{% block title %}{% endblock %} - PyTorch Distributed</title>
    <link rel="shortcut icon" type="image/x-icon" href="https://pytorch.org/favicon.ico?">

    <style>
        body {
            margin: 0;
            font-family:
                -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
                "Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji",
                "Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";
            font-size: 1rem;
            font-weight: 400;
            line-height: 1.5;
            color: #212529;
            text-align: left;
            background-color: #fff;
        }
        h1, h2, h2, h4, h5, h6, .h1, .h2, .h2, .h4, .h5, .h6 {
            margin-bottom: .5rem;
            font-weight: 500;
            line-height: 1.2;
        }
        nav {
            background-color: rgba(0, 0, 0, 0.17);
            padding: 10px;
            display: flex;
            align-items: center;
            padding: 16px;
            justify-content: flex-start;
        }
        nav h1 {
            display: inline-block;
            margin: 0;
        }
        nav a {
           margin: 0 8px;
        }
        section {
            max-width: 1280px;
            padding: 16px;
            margin: 0 auto;
        }
        pre {
            white-space: pre-wrap;
            max-width: 100%;
        }
    </style>
</head>

<nav>
    <h1>Torch Distributed Debug Server</h1>

    {{ nav_links | safe }}
</nav>

<section class="content">
  {% block header %}{% endblock %}
  {% block content %}{% endblock %}
</section>
    """

RAW_RESP_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{title}}{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ resp.text }}</pre>
        {% endif %}
    {% endfor %}
{% endblock %}
    """

JSON_RESP_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{ title }}{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ format_json(resp.text) }}</pre>
        {% endif %}
    {% endfor %}
{% endblock %}
    """


# ---------------------------------------------------------------------------
# PeriodicDumper
# ---------------------------------------------------------------------------


class PeriodicDumper:
    def __init__(
        self,
        handlers: list[DebugHandler],
        output_dir: str,
        interval_seconds: float = 60.0,
    ) -> None:
        self._handlers = handlers
        self._output_dir = output_dir
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        os.makedirs(self._output_dir, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="distributed.debug.PeriodicDumper",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for handler in self._handlers:
                try:
                    content = handler.dump()
                except Exception:
                    logger.exception("Failed to dump %s", handler.dump_filename())
                    continue
                if content is None:
                    continue
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{handler.dump_filename()}_{timestamp}.txt"
                path = os.path.join(self._output_dir, filename)
                try:
                    with open(path, "w") as f:
                        f.write(content)
                except Exception:
                    logger.exception("Failed to write dump to %s", path)
            self._stop_event.wait(self._interval_seconds)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _IPv6HTTPServer(ThreadingHTTPServer):
    address_family: socket.AddressFamily = socket.AF_INET6  # pyre-ignore
    request_queue_size: int = 1024


class HTTPRequestHandler(BaseHTTPRequestHandler):
    frontend: "FrontendServer"

    def log_message(self, format, *args):
        logger.info(
            "%s %s",
            self.client_address[0],
            format % args,
        )

    def do_GET(self):
        self.frontend._handle_request(self)

    def get_path(self) -> str:
        return urlparse(self.path).path

    def get_query(self) -> dict[str, list[str]]:
        return parse_qs(self.get_raw_query())

    def get_raw_query(self) -> str:
        return urlparse(self.path).query

    def get_query_arg(
        self, name: str, default: object = None, type: type = str
    ) -> object:
        query = self.get_query()
        if name not in query:
            return default
        return type(query[name][0])


class FrontendServer:
    def __init__(
        self,
        port: int,
        handlers: list[DebugHandler] | None = None,
    ):
        if handlers is None:
            from torch.distributed.debug._debug_handlers import default_handlers

            handlers = default_handlers()

        # Build nav HTML from handlers
        nav_html = "\n".join(
            f'    <a href="{link.path}">{link.label}</a> <!--@lint-ignore-->'
            for handler in handlers
            for link in handler.nav_links()
        )

        # Merge all handler templates + shared templates
        all_templates: dict[str, str] = {
            "base.html": BASE_TEMPLATE,
            "raw_resp.html": RAW_RESP_TEMPLATE,
            "json_resp.html": JSON_RESP_TEMPLATE,
        }
        for handler in handlers:
            all_templates.update(handler.templates())

        loader = DictLoader(all_templates)
        self._jinja_env = Environment(loader=loader, enable_async=True)
        self._jinja_env.globals.update(
            zip=zip,
            format_json=format_json,
            enumerate=enumerate,
            nav_links=nav_html,
        )

        # Build route table from handlers
        self._routes: dict[str, Callable[[HTTPRequestHandler], bytes]] = {}
        for handler in handlers:
            for route in handler.routes():
                self._routes[route.path] = route.handler

        self._handlers = handlers

        # Create HTTP server
        RequestHandlerClass = type(
            "HTTPRequestHandler",
            (HTTPRequestHandler,),
            {"frontend": self},
        )

        server_address = ("", port)
        self._server = _IPv6HTTPServer(server_address, RequestHandlerClass)

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
            name="distributed.debug.FrontendServer",
        )
        self._thread.start()

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception:
            logger.exception("got exception in frontend server")

    def join(self) -> None:
        self._thread.join()

    def _handle_request(self, req: HTTPRequestHandler) -> None:
        path = req.get_path()
        if path not in self._routes:
            req.send_error(404, f"Handler not found: {path}")
            return

        handler = self._routes[path]
        try:
            resp = handler(req)
        # Catch SystemExit to not crash when FlightRecorder errors.
        except (Exception, SystemExit) as e:
            logger.exception(
                "Exception in frontend server when handling %s",
                path,
            )
            req.send_error(500, f"Exception: {repr(e)}")
            return

        req.send_response(200)
        req.send_header("Content-type", "text/html")
        req.end_headers()
        req.wfile.write(resp)

    def render_template(self, template: str, **kwargs: object) -> bytes:
        return self._jinja_env.get_template(template).render(**kwargs).encode()


def main(
    port: int,
    dump_dir: str | None,
    dump_interval: float,
    handlers: list[DebugHandler],
    enabled_dumps: set[str],
    fetch_timeout: float = 60.0,
) -> None:
    for handler in handlers:
        handler.fetch_timeout = fetch_timeout

    logger.setLevel(logging.INFO)

    server = FrontendServer(port=port, handlers=handlers)
    logger.info("Frontend server started on port %d", server._server.server_port)

    dumper: PeriodicDumper | None = None
    if dump_dir is not None:
        dumper = PeriodicDumper(
            [
                handler
                for handler in handlers
                if handler.dump_filename() in enabled_dumps
            ],
            dump_dir,
            dump_interval,
        )
        dumper.start()
        logger.info(
            "Periodic dumper started, writing to %s every %.0fs",
            dump_dir,
            dump_interval,
        )

    try:
        server.join()
    finally:
        if dumper is not None:
            dumper.stop()
<|endoftext|>from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabulate import tabulate

from torch.distributed.debug._frontend import (
    DebugHandler,
    fetch_all,
    format_fetch_summary,
    format_json,
    NavLink,
    Response,
    Route,
)
from torch.distributed.debug._store import tcpstore_client
from torch.distributed.flight_recorder.components.builder import build_db
from torch.distributed.flight_recorder.components.config_manager import JobConfig
from torch.distributed.flight_recorder.components.types import (
    Collective,
    Database,
    Group,
    Membership,
    NCCLCall,
)


if TYPE_CHECKING:
    from torch.distributed.debug._frontend import FrontendServer, HTTPRequestHandler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handler-specific templates
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
  <h1>{% block title %}Index{% endblock %}</h1>
{% endblock %}
{% block content %}
Hi
{% endblock %}
    """

PYSPY_DUMP_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}py-spy Stack Traces{% endblock %}</h1>
{% endblock %}
{% block content %}
    <form action="" method="get">
        <input type="checkbox" id="native" name="native" value="1"/>
        <label for="native">Native</label>
        <input type="checkbox" id="subprocesses" name="subprocesses" value="1"/>
        <label for="subprocesses">Subprocesses</label>
        <input type="submit" value="Submit">
    </form>

    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ resp.text }}</pre>
        {% endif %}
    {% endfor %}
{% endblock %}
    """

FR_TRACE_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{ title }}{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% if fetch_summary %}<pre>{{ fetch_summary }}</pre>{% endif %}
    <h2>Groups</h2>
    {{ groups | safe }}
    <h2>Memberships</h2>
    {{ memberships | safe }}
    <h2>Collectives</h2>
    {{ collectives | safe }}
    <h2>NCCL Calls</h2>
    {{ ncclcalls | safe }}
{% endblock %}
    """

PROFILE_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}torch.profiler{% endblock %}</h1>
{% endblock %}

{% block content %}
    <form action="" method="get">
        <label for="duration">Duration (seconds):</label>
        <input type="number" id="duration" name="duration" value="{{ duration }}" min="1" max="60">
        <input type="submit" value="Submit">
    </form>

    <script>
    function stringToArrayBuffer(str) {
        const encoder = new TextEncoder();
        return encoder.encode(str).buffer;
    }
    async function openPerfetto(data) {
        const ui = window.open('https://ui.perfetto.dev/#!/');
        if (!ui) { alert('Popup blocked. Allow popups for this page and click again.'); return; }

        // Perfetto readiness handshake: PING until we receive PONG
        await new Promise((resolve, reject) => {
        const onMsg = (e) => {
            if (e.source === ui && e.data === 'PONG') {
            window.removeEventListener('message', onMsg);
            clearInterval(pinger);
            resolve();
            }
        };
        window.addEventListener('message', onMsg);
        const pinger = setInterval(() => { try { ui.postMessage('PING', '*'); } catch (_e) {} }, 250);
        setTimeout(() => { clearInterval(pinger); window.removeEventListener('message', onMsg); reject(); }, 20000);
        }).catch(() => { alert('Perfetto UI did not respond. Try again.'); return; });

        ui.postMessage({
        perfetto: {
            buffer: stringToArrayBuffer(JSON.stringify(data)),
            title: "torch profiler",
            fileName: "trace.json",
        }
        }, '*');
    }
    </script>

    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <script>
            function run{{ i }}() {
                var data = {{ resp.text | safe }};
                openPerfetto(data);
            }
            </script>

            <button onclick="run{{ i }}()">View {{ i }}</button>
        {% endif %}
    {% endfor %}
{% endblock %}
    """

TCPSTORE_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}TCPStore Keys{% endblock %}</h1>
{% endblock %}
{% block content %}
    <pre>
    {% for k, v in zip(keys, values) -%}
{{ k }}: {{ v | truncate(100) }}
    {% endfor %}
    </pre>
{% endblock %}
    """


# ---------------------------------------------------------------------------
# Handler classes
# ---------------------------------------------------------------------------


class IndexHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/", "Home")]

    def templates(self) -> dict[str, str]:
        return {"index.html": INDEX_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        return req.frontend.render_template("index.html")


class StacksHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/stacks", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/stacks", "Python Stack Traces")]

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("dump_traceback", timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "raw_resp.html", title="Stacks", addrs=addrs, resps=resps
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all("dump_traceback", timeout=self.fetch_timeout)
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(
                resp.text if resp.status_code == 200 else f"Error: {resp.status_code}"
            )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "stacks"


class PySpyHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/pyspy_dump", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/pyspy_dump", "py-spy Stacks")]

    def templates(self) -> dict[str, str]:
        return {"pyspy_dump.html": PYSPY_DUMP_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        query = req.get_raw_query()
        if "nonblocking" not in query:
            query = f"nonblocking=1&{query}" if query else "nonblocking=1"
        addrs, resps = fetch_all("pyspy_dump", query, timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "pyspy_dump.html",
            addrs=addrs,
            resps=resps,
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all(
            "pyspy_dump", "nonblocking=1", timeout=self.fetch_timeout
        )
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(
                resp.text if resp.status_code == 200 else f"Error: {resp.status_code}"
            )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "pyspy_dump"


class FlightRecorderHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [
            Route("/fr_trace", self._handle_fr_trace),
            Route("/fr_trace_json", self._handle_fr_trace_json),
            Route("/fr_trace_nccl", self._handle_fr_trace_nccl),
            Route("/fr_trace_nccl_json", self._handle_fr_trace_nccl_json),
        ]

    def nav_links(self) -> list[NavLink]:
        return [
            NavLink("/fr_trace", "FlightRecorder CPU"),
            NavLink("/fr_trace_json", "(JSON)"),
            NavLink("/fr_trace_nccl", "FlightRecorder NCCL"),
            NavLink("/fr_trace_nccl_json", "(JSON)"),
        ]

    def templates(self) -> dict[str, str]:
        return {"fr_trace.html": FR_TRACE_TEMPLATE}

    @staticmethod
    def _build_db(addrs: list[str], resps: list[Response]) -> Database:
        config = JobConfig()
        args = config.parse_args(args=[])
        args.allow_incomplete_ranks = True
        args.verbose = True

        details = {}
        for rank, resp in enumerate(resps):
            if resp.status_code != 200:
                continue
            dump = {
                "rank": rank,
                "host_name": addrs[rank],
                **resp.json(),
            }
            if "entries" not in dump:
                dump["entries"] = []
            details[f"rank{rank}.json"] = dump

        if not details:
            raise RuntimeError("All workers failed to respond")

        version = next(iter(details.values()))["version"]
        # pyrefly: ignore [bad-argument-type]
        return build_db(details, args, version)

    def _render_tables(
        self, server: FrontendServer, addrs: list[str], resps: list[Response]
    ) -> bytes:
        db = self._build_db(addrs, resps)
        return server.render_template(
            "fr_trace.html",
            title="FlightRecorder",
            fetch_summary=format_fetch_summary(addrs, resps),
            groups=tabulate(db.groups, headers=Group._fields, tablefmt="html"),
            memberships=tabulate(
                db.memberships, headers=Membership._fields, tablefmt="html"
            ),
            collectives=tabulate(
                db.collectives, headers=Collective._fields, tablefmt="html"
            ),
            ncclcalls=tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="html"),
        )

    def _handle_fr_trace(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("fr_trace_json", timeout=self.fetch_timeout)
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_fr_trace_json(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("fr_trace_json", timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "json_resp.html",
            title="FlightRecorder",
            addrs=addrs,
            resps=resps,
        )

    def _handle_fr_trace_nccl(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all(
            "dump_nccl_trace_json", "onlyactive=true", timeout=self.fetch_timeout
        )
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_fr_trace_nccl_json(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all(
            "dump_nccl_trace_json", "onlyactive=true", timeout=self.fetch_timeout
        )
        return req.frontend.render_template(
            "json_resp.html",
            title="FlightRecorder NCCL",
            addrs=addrs,
            resps=resps,
        )

    def dump(self) -> str | None:
        parts = []

        addrs, resps = fetch_all("fr_trace_json", timeout=self.fetch_timeout)
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        db = self._build_db(addrs, resps)
        parts.extend(
            [
                "=== FR Trace ===",
                "--- Groups ---",
                tabulate(db.groups, headers=Group._fields, tablefmt="plain"),
                "--- Memberships ---",
                tabulate(db.memberships, headers=Membership._fields, tablefmt="plain"),
                "--- Collectives ---",
                tabulate(db.collectives, headers=Collective._fields, tablefmt="plain"),
                "--- NCCL Calls ---",
                tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="plain"),
            ]
        )

        try:
            nccl_addrs, nccl_resps = fetch_all(
                "dump_nccl_trace_json", "onlyactive=true", timeout=self.fetch_timeout
            )
            nccl_db = self._build_db(nccl_addrs, nccl_resps)
            parts.extend(
                [
                    "",
                    "=== FR Trace NCCL ===",
                    "--- Groups ---",
                    tabulate(nccl_db.groups, headers=Group._fields, tablefmt="plain"),
                    "--- Memberships ---",
                    tabulate(
                        nccl_db.memberships,
                        headers=Membership._fields,
                        tablefmt="plain",
                    ),
                    "--- Collectives ---",
                    tabulate(
                        nccl_db.collectives,
                        headers=Collective._fields,
                        tablefmt="plain",
                    ),
                    "--- NCCL Calls ---",
                    tabulate(
                        nccl_db.ncclcalls,
                        headers=NCCLCall._fields,
                        tablefmt="plain",
                    ),
                ]
            )
        except Exception:
            parts.append("\n=== FR Trace NCCL ===\nFailed to fetch NCCL trace")

        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "fr_trace"


class ProfilerHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/profile", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/profile", "torch profiler")]

    def templates(self) -> dict[str, str]:
        return {"profile.html": PROFILE_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        duration = req.get_query_arg("duration", default=1.0, type=float)
        addrs, resps = fetch_all(
            "torch_profile", f"duration={duration}", timeout=self.fetch_timeout
        )
        return req.frontend.render_template("profile.html", addrs=addrs, resps=resps)


class WaitCountersHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/wait_counters", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/wait_counters", "Wait Counters")]

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("wait_counter_values", timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "json_resp.html", title="Wait Counters", addrs=addrs, resps=resps
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all("wait_counter_values", timeout=self.fetch_timeout)
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            if resp.status_code == 200:
                parts.append(format_json(resp.text))
            else:
                parts.append(f"Error: {resp.status_code}")
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "wait_counters"


class TCPStoreHandler(DebugHandler):
    def routes(self) -> list[Route]:
        return [Route("/tcpstore", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/tcpstore", "TCPStore")]

    def templates(self) -> dict[str, str]:
        return {"tcpstore.html": TCPSTORE_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        store = tcpstore_client(prefix="")
        keys = store.list_keys()
        keys.sort()
        values = [repr(v) for v in store.multi_get(keys)]
        return req.frontend.render_template("tcpstore.html", keys=keys, values=values)

    def dump(self) -> str | None:
        store = tcpstore_client(prefix="")
        keys = store.list_keys()
        keys.sort()
        values = [repr(v) for v in store.multi_get(keys)]
        parts = [f"{k}: {v}" for k, v in zip(keys, values)]
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "tcpstore"


class TorchCommsFlightRecorderHandler(DebugHandler):
    """Handler for TorchComms FlightRecorder trace data."""

    def routes(self) -> list[Route]:
        return [
            Route("/torchcomms_fr_trace", self._handle_torchcomms_fr_trace),
            Route("/torchcomms_fr_trace_json", self._handle_torchcomms_fr_trace_json),
        ]

    def nav_links(self) -> list[NavLink]:
        return [
            NavLink("/torchcomms_fr_trace", "TorchComms FR"),
            NavLink("/torchcomms_fr_trace_json", "(JSON)"),
        ]

    def templates(self) -> dict[str, str]:
        return {"fr_trace.html": FR_TRACE_TEMPLATE}

    def _handle_torchcomms_fr_trace(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all(
            "torchcomms_fr_trace_json", "onlyactive=true", timeout=self.fetch_timeout
        )
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_torchcomms_fr_trace_json(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all(
            "torchcomms_fr_trace_json", "onlyactive=true", timeout=self.fetch_timeout
        )
        return req.frontend.render_template(
            "json_resp.html",
            title="TorchComms FlightRecorder",
            addrs=addrs,
            resps=resps,
        )

    def _render_tables(
        self, server: FrontendServer, addrs: list[str], resps: list[Response]
    ) -> bytes:
        db = FlightRecorderHandler._build_db(addrs, resps)
        return server.render_template(
            "fr_trace.html",
            title="TorchComms FlightRecorder",
            fetch_summary=format_fetch_summary(addrs, resps),
            groups=tabulate(db.groups, headers=Group._fields, tablefmt="html"),
            memberships=tabulate(
                db.memberships, headers=Membership._fields, tablefmt="html"
            ),
            collectives=tabulate(
                db.collectives, headers=Collective._fields, tablefmt="html"
            ),
            ncclcalls=tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="html"),
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all("torchcomms_fr_trace_json", timeout=self.fetch_timeout)
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        db = FlightRecorderHandler._build_db(addrs, resps)
        parts.extend(
            [
                "=== TorchComms FR Trace ===",
                "--- Groups ---",
                tabulate(db.groups, headers=Group._fields, tablefmt="plain"),
                "--- Memberships ---",
                tabulate(db.memberships, headers=Membership._fields, tablefmt="plain"),
                "--- Collectives ---",
                tabulate(db.collectives, headers=Collective._fields, tablefmt="plain"),
                "--- NCCL Calls ---",
                tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="plain"),
            ]
        )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "torchcomms_fr_trace"


HEALTH_CHECK_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}TorchComms Health Check{% endblock %}</h1>
{% endblock %}
{% block content %}
    <h2>Health Status</h2>
    {% if fetch_summary %}<pre>{{ fetch_summary }}</pre>{% endif %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h3>Rank {{ i }}: {{ addr }}</h3>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ resp.text }}</pre>
        {% endif %}
    {% endfor %}
    {% if dump_triggered %}
        <h2>Flight Recorder Dump (triggered by unhealthy rank)</h2>
        {% for i, (addr, resp) in enumerate(zip(dump_addrs, dump_resps)) %}
            <h3>Rank {{ i }}: {{ addr }}</h3>
            {% if resp.status_code != 200 %}
                <p>Failed to dump: status={{ resp.status_code }}</p>
                <pre>{{ resp.text }}</pre>
            {% else %}
                <pre>{{ resp.text }}</pre>
            {% endif %}
        {% endfor %}
    {% endif %}
{% endblock %}
    """


class TorchCommsHealthCheckHandler(DebugHandler):
    """Health check that detects watchdog timeouts and triggers FR dumps."""

    def routes(self) -> list[Route]:
        return [Route("/torchcomms_health_check", self._handle)]

    def nav_links(self) -> list[NavLink]:
        return [NavLink("/torchcomms_health_check", "TorchComms Health")]

    def templates(self) -> dict[str, str]:
        return {"torchcomms_health_check.html": HEALTH_CHECK_TEMPLATE}

    @staticmethod
    def _any_unhealthy(resps: list[Response]) -> bool:
        """Return True if any rank explicitly reports itself as unhealthy.

        Only responses with a 200 status code are inspected.  Non-200
        responses (connection errors, timeouts, etc.) are intentionally
        ignored so that a single unreachable rank does not trigger an
        expensive Flight Recorder dump across the entire job.  A dump is
        only triggered when a rank is reachable *and* positively reports
        ``{"healthy": false}``.
        """
        for resp in resps:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    logger.exception(
                        "failed to parse health check response as JSON; "
                        "treating rank as healthy"
                    )
                    continue
                if not data.get("healthy", True):
                    return True
        return False

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        addrs, resps = fetch_all("torchcomms_health_check", timeout=self.fetch_timeout)
        dump_triggered = False
        dump_addrs: list[str] = []
        dump_resps: list[Response] = []
        if self._any_unhealthy(resps):
            dump_addrs, dump_resps = fetch_all(
                "torchcomms_fr_dump_file", timeout=self.fetch_timeout
            )
            dump_triggered = True
        return req.frontend.render_template(
            "torchcomms_health_check.html",
            fetch_summary=format_fetch_summary(addrs, resps),
            addrs=addrs,
            resps=resps,
            dump_triggered=dump_triggered,
            dump_addrs=dump_addrs,
            dump_resps=dump_resps,
        )

    def dump(self) -> str | None:
        addrs, resps = fetch_all("torchcomms_health_check", timeout=self.fetch_timeout)
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(
                resp.text if resp.status_code == 200 else f"Error: {resp.status_code}"
            )
        if self._any_unhealthy(resps):
            parts.append("")
            parts.append("=== Unhealthy rank detected, triggering FR dump ===")
            dump_addrs, dump_resps = fetch_all(
                "torchcomms_fr_dump_file", timeout=self.fetch_timeout
            )
            for i, (addr, resp) in enumerate(zip(dump_addrs, dump_resps)):
                parts.append(f"--- Rank {i}: {addr} ---")
                parts.append(
                    resp.text
                    if resp.status_code == 200
                    else f"Error: {resp.status_code}"
                )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        return "torchcomms_health_check"


def default_handlers() -> list[DebugHandler]:
    return [
        IndexHandler(),
        StacksHandler(),
        PySpyHandler(),
        FlightRecorderHandler(),
        TorchCommsFlightRecorderHandler(),
        TorchCommsHealthCheckHandler(),
        ProfilerHandler(),
        WaitCountersHandler(),
        TCPStoreHandler(),
    ]
<|endoftext|>import logging
import multiprocessing
import socket
from typing import Literal, TYPE_CHECKING

# import for registration side effect
import torch.distributed.debug._handlers
from torch._C._distributed_c10d import _WorkerServer
from torch.distributed.debug._store import get_rank, tcpstore_client


if TYPE_CHECKING:
    from torch.distributed.debug._frontend import DebugHandler


__all__ = [
    "start_debug_server",
    "stop_debug_server",
]

logger: logging.Logger = logging.getLogger(__name__)

_WORKER_SERVER: _WorkerServer | None = None
_DEBUG_SERVER_PROC: multiprocessing.Process | None = None


def start_debug_server(
    port: int = 25999,
    worker_port: int = 0,
    start_method: Literal["fork", "spawn", "forkserver"] | None = None,
    dump_dir: str | None = None,
    dump_interval: float = 60.0,
    enabled_dumps: set[str] | None = None,
    handlers: list["DebugHandler"] | None = None,
    fetch_timeout: float = 60.0,
) -> None:
    """
    Start the debug server stack on all workers. The frontend debug server is
    only started on rank0 while the per rank worker servers are started on all
    ranks.

    This server provides an HTTP frontend that allows for debugging slow and
    deadlocked distributed jobs across all ranks simultaneously. This collects
    data such as stack traces, FlightRecorder events, and performance profiles.

    This depends on dependencies which are not installed by default.

    Dependencies:
    - Jinja2
    - aiohttp

    WARNING: This is intended to only be used in trusted network environments.
    The debug server is not designed to be secure and should not be exposed to
    the public internet. See SECURITY.md for more details.

    WARNING: This is an experimental feature and may change at any time.

    Args:
        port (int): The port to start the frontend debug server on.
        worker_port (int): The port to start the worker server on. Defaults to 0, which
            will cause the worker server to bind to an ephemeral port.
        start_method (str | None): The multiprocessing start method to use for the
            frontend server process. One of "fork", "spawn", or "forkserver".
            If None, uses the default start method. Using "spawn" is recommended
            when using CUDA or when fork safety is a concern.
        dump_dir (str | None): Directory to write periodic debug dumps to. If None,
            periodic dumping is disabled.
        dump_interval (float): Seconds between periodic dumps. Defaults to 60.
        enabled_dumps (set[str] | None): Set of handler dump filenames to enable
            (e.g. {"stacks", "fr_trace", "tcpstore"}). If None, all handlers that
            implement dump() are enabled.
        handlers (list[DebugHandler] | None): List of debug handlers to use. If None,
            uses the default handlers. See torch.distributed.debug._handlers for
            the default handlers.
        fetch_timeout (float): Timeout in seconds for fetching data from individual
            workers. Defaults to 60. Workers that don't respond within this time
            will be reported as unavailable.
    """
    global _WORKER_SERVER, _DEBUG_SERVER_PROC

    if _WORKER_SERVER is not None:
        raise AssertionError("debug server already started")
    if _DEBUG_SERVER_PROC is not None:
        raise AssertionError("debug server already started")

    logger.info("Starting debug server on port %d", port)

    store = tcpstore_client()

    _WORKER_SERVER = _WorkerServer("::", worker_port)

    RANK = get_rank()
    store.set(f"rank{RANK}", f"http://{socket.gethostname()}:{_WORKER_SERVER.port}")

    if RANK == 0:
        from torch.distributed.debug._debug_handlers import default_handlers
        from torch.distributed.debug._frontend import main

        if handlers is None:
            handlers = default_handlers()
        if enabled_dumps is None:
            enabled_dumps = {
                "stacks",
                "fr_trace",
                "torchcomms_health_check",
            }

        main_kwargs = {
            "port": port,
            "dump_dir": dump_dir,
            "dump_interval": dump_interval,
            "enabled_dumps": enabled_dumps,
            "handlers": handlers,
            "fetch_timeout": fetch_timeout,
        }

        if start_method is not None:
            ctx = multiprocessing.get_context(start_method)
            # pyre-ignore[16]: BaseContext has Process attribute at runtime
            _DEBUG_SERVER_PROC = ctx.Process(
                target=main, kwargs=main_kwargs, daemon=True
            )
        else:
            _DEBUG_SERVER_PROC = multiprocessing.Process(
                target=main, kwargs=main_kwargs, daemon=True
            )
        _DEBUG_SERVER_PROC.start()


def stop_debug_server() -> None:
    """
    Shutdown the debug server and stop the frontend debug server process.
    """
    global _WORKER_SERVER, _DEBUG_SERVER_PROC

    if _DEBUG_SERVER_PROC is None:
        raise AssertionError
    if _WORKER_SERVER is None:
        raise AssertionError

    logger.info("Stopping debug server")

    _DEBUG_SERVER_PROC.terminate()
    _WORKER_SERVER.shutdown()
    _DEBUG_SERVER_PROC.join()

    _WORKER_SERVER = None
    _DEBUG_SERVER_PROC = None
<|endoftext|>import pathlib
import tempfile
import time

from torch._C._distributed_c10d import _register_handler, _Request, _Response
from torch.profiler import _ExperimentalConfig, profile


def _torch_profile(req: _Request, resp: _Response) -> None:
    experimental_config = _ExperimentalConfig(
        profile_all_threads=True,
    )
    duration = float(req.get_param("duration"))
    with profile(record_shapes=True, experimental_config=experimental_config) as prof:
        time.sleep(duration)

    with tempfile.NamedTemporaryFile(prefix="torch_debug", suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        resp.set_content(pathlib.Path(f.name).read_bytes(), "application/json")
        resp.set_status(200)


_register_handler("torch_profile", _torch_profile)
<|endoftext|># Keep old package for BC purposes, this file should be removed once
# everything moves to the `torch.distributed._shard` package.
import sys
import warnings

import torch
from torch.distributed._shard.sharding_spec import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`torch.distributed._sharding_spec` will be deprecated, "
        "use `torch.distributed._shard.sharding_spec` instead",
        DeprecationWarning,
        stacklevel=2,
    )

import torch.distributed._shard.sharding_spec as _sharding_spec


sys.modules["torch.distributed._sharding_spec"] = _sharding_spec
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from types import TracebackType


def is_available() -> bool:
    return hasattr(torch._C, "_dist_autograd_init")


if is_available() and not torch._C._dist_autograd_init():
    raise RuntimeError("Failed to initialize torch.distributed.autograd")

if is_available():
    from torch._C._distributed_autograd import (
        _current_context,
        _get_debug_info,
        _get_max_id,
        _init,
        _is_valid_context,
        _new_context,
        _release_context,
        _retrieve_context,
        backward,
        DistAutogradContext,
        get_gradients,
    )

__all__ = ["context", "is_available"]


class context:
    """
    Context object to wrap forward and backward passes when using
    distributed autograd. The ``context_id`` generated in the ``with``
    statement  is required to uniquely identify a distributed backward pass
    on all workers. Each worker stores metadata associated with this
    ``context_id``, which is required to correctly execute a distributed
    autograd pass.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>     t1 = torch.rand((3, 3), requires_grad=True)
        >>>     t2 = torch.rand((3, 3), requires_grad=True)
        >>>     loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()
        >>>     dist_autograd.backward(context_id, [loss])
    """

    def __enter__(self) -> int:
        self.autograd_context = _new_context()
        return self.autograd_context._context_id()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _release_context(self.autograd_context._context_id())
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from torch.fx.node import Argument


def friendly_debug_info(v: object) -> Argument:
    """
    Helper function to print out debug info in a friendly way.
    """
    if isinstance(v, torch.Tensor):
        return f"Tensor({v.shape}, grad={v.requires_grad}, dtype={v.dtype})"
    else:
        return str(v)


def map_debug_info(a: Argument) -> Argument:
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
    return torch.fx.node.map_aggregate(a, friendly_debug_info)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import collections
import logging
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter

from ._debug import map_debug_info


logger = logging.getLogger(__name__)


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Node | None:
    """
    Get the grad function or grad accumulator for a tensor.

    Accumulate grad nodes are lazily created, so we need to a
    dummy view in order to trigger its creation.
    """
    if t.requires_grad and t.grad_fn is None:
        # if no grad function (leaf tensors) we use view
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        if grad_fn is not None:
            return grad_fn.next_functions[0][0]
        else:
            raise RuntimeError(
                "Attempted to get grad_fn, but got None."
                "Is this being created in a no-grad context?"
            )
    else:
        return t.grad_fn


def reverse_closure(
    roots: list[Node], target_nodes: set[Node], reverse_edges_dict
) -> tuple[set[Node], set[Node]]:
    """
    This function returns the reverse closure of the given roots,
    i.e. the set of nodes that can be reached from the roots by following the
    reverse edges of the graph. The target_nodes are the nodes that we want to
    include in the closure.
    """
    # Recurse until we reach a target node
    closure: set[Node] = set()
    visited_target_nodes = set()
    q: collections.deque[Node] = collections.deque()
    for node in roots:
        if node is not None and node not in closure:
            closure.add(node)
            q.append(node)
    while q:
        node = q.popleft()
        reverse_edges = reverse_edges_dict[node]
        for fn in reverse_edges:
            if fn in closure or fn is None:
                continue
            if fn in target_nodes:
                visited_target_nodes.add(fn)
                continue
            closure.add(fn)
            q.append(fn)
    return closure, visited_target_nodes


def construct_reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]:
    q: collections.deque[Node] = collections.deque()
    root_seen: set[Node] = set()
    reverse_edges_dict: dict[Node, list[Node]] = collections.defaultdict(list)
    for node in roots:
        if node is not None and node not in root_seen:
            q.append(node)
            root_seen.add(node)
    while q:
        node = q.popleft()
        for fn, _ in node.next_functions:
            if fn is not None:
                if len(reverse_edges_dict[fn]) == 0:
                    q.append(fn)
                reverse_edges_dict[fn].append(node)
    return reverse_edges_dict


def get_param_groups(
    inputs: list[Node], params: list[Node], reverse_edges_dict
) -> list[dict[str, Any]]:
    """
    Given a list of inputs and a list of parameters, return a list of parameter
    groups, where each group contains the parameters and the intermediates that
    are connected to the parameters.

    The returned list of parameter groups is a list of dictionaries, where each
    dictionary contains the following keys:
    - "params": a set of parameters
    - "intermediates": a set of intermediates

    The returned list of parameter groups is a list of dictionaries,
    """
    # reverse graph that starts with inputs, and goes up to the dOutput or the loss,
    # but omits weights and any subgraphs connecting weights to this closure
    inputs_closure, _ = reverse_closure(inputs, set(), reverse_edges_dict)
    param_groups: dict[Node, dict[str, set]] = dict()  # keyed on intermediates
    for param in params:
        closure, intersected = reverse_closure(
            [param], inputs_closure, reverse_edges_dict
        )
        param_group: dict[str, set] = {
            "params": {param},
            "intermediates": intersected,
        }
        for input_node in intersected:
            existing = param_groups.get(input_node)
            if existing is not None:
                existing["params"] = existing["params"].union(param_group["params"])
                existing["intermediates"] = existing["intermediates"].union(
                    param_group["intermediates"]
                )
                param_group = existing
            else:
                param_groups[input_node] = param_group

    # Sanity check: union of all param_groups params should be equal to all params
    union_params: set[Node] = set()
    seen_ids: set[int] = set()
    unique_param_groups = []
    for param_group in param_groups.values():
        if id(param_group) not in seen_ids:
            seen_ids.add(id(param_group))
            unique_param_groups.append(param_group)
            union_params = union_params.union(param_group["params"])

    # The assert will only be true if the input tensor requires gradients,
    # otherwise the autograd graph will miss the first layer of inputs
    # assert union_params == set(params)
    return unique_param_groups


def _autograd_grad_for_inputs(
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[Any],
    grad_outputs: Sequence[torch.Tensor | None] | None = None,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Compute input gradients, returning ``None`` for non-grad inputs."""
    # Some inputs may not be used or may not require gradients, so we filter them out
    # before calling autograd.grad and place None for those positions in the result.
    grad_indices: list[int] = []
    inputs_requiring_grad: list[torch.Tensor] = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            grad_indices.append(i)
            inputs_requiring_grad.append(inp)

    if not inputs_requiring_grad:
        return tuple(None for _ in inputs)

    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs_requiring_grad,
        grad_outputs=grad_outputs,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )

    result: list[torch.Tensor | None] = [None] * len(inputs)
    for idx, g in zip(grad_indices, grads, strict=True):
        result[idx] = g
    return tuple(result)


def stage_backward_input(
    stage_outputs_or_loss: list[torch.Tensor],
    output_grads: list[torch.Tensor] | None,
    input_values: list[Any],
    weights: Iterator[Parameter],
) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]]]:
    """
    Compute the gradients for only the stage inputs with
    respect to the stage outputs (if non-last stage) or loss (if last stage)

    After computing input gradients, we save the intermediate nodes in `param_groups`
    for later use in stage_backward_weight. We don't need to save any other intermediate nodes
    that aren't needed for dW because when we do dW calculation, we start from saved intermediates.
    Detaching the stage_outputs_or_loss at the end of this function is important as
    it frees up the memory that the autograd graph is anticipating to be used later (but doesn't actually need).
    """
    valid_outputs: list[torch.Tensor] = []
    valid_output_grads: list[torch.Tensor | None] = []
    for i, stage_output in enumerate(stage_outputs_or_loss):
        if not stage_output.requires_grad and stage_output.grad_fn is None:
            continue
        valid_outputs.append(stage_output)
        valid_output_grads.append(
            torch.ones_like(stage_output) if output_grads is None else output_grads[i]
        )

    stage_output_grad_fns: list[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, valid_outputs))
    )
    stage_input_grad_fns: list[Node] = list(
        filter(
            None,
            (
                _get_grad_fn_or_grad_acc(inp)
                for inp in input_values
                if isinstance(inp, torch.Tensor)
            ),
        )
    )
    weight_grad_fns: list[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, weights))
    )

    reverse_edges_dict = construct_reverse_graph(stage_output_grad_fns)
    param_groups = get_param_groups(
        stage_input_grad_fns, weight_grad_fns, reverse_edges_dict
    )

    handles = []
    try:
        for param_group in param_groups:
            intermediates = list(param_group["intermediates"])
            param_group["intermediates"] = intermediates
            for i, intermediate in enumerate(intermediates):

                def get_hook(param_group, i):
                    def hook(grad_inputs):
                        if param_group.get("grads", None) is None:
                            param_group["grads"] = [None] * len(
                                param_group["intermediates"]
                            )
                        param_group["grads"][i] = grad_inputs

                    return hook

                handle = intermediate.register_prehook(get_hook(param_group, i))
                handles.append(handle)

        if valid_outputs:
            dinputs = _autograd_grad_for_inputs(
                valid_outputs,
                input_values,
                valid_output_grads,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            dinputs = tuple(None for _ in input_values)

        for inp, dinput in zip(input_values, dinputs):
            if isinstance(inp, torch.Tensor) and dinput is not None:
                if inp.grad is None:
                    inp.grad = dinput
                else:
                    inp.grad += dinput

        # drop output side graph state we no longer need
        for t in stage_outputs_or_loss:
            t.detach_()

        return dinputs, param_groups
    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward input:
        Stage output: {map_debug_info(stage_outputs_or_loss)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e
    finally:
        for handle in handles:
            handle.remove()


def stage_backward_weight(
    weights: Iterator[Parameter], param_groups: list[dict[str, Any]], retain_graph=False
) -> tuple[torch.Tensor | None, ...]:
    # map weights to param_group_weights
    grad_acc_to_weight = {}
    weight_grads: list[torch.Tensor | None] = []
    for index, weight in enumerate(weights):
        grad_acc = _get_grad_fn_or_grad_acc(weight)
        grad_acc_to_weight[grad_acc] = weight, index
        weight_grads.append(weight.grad)

    for param_group in param_groups:
        valid_edges = []
        valid_grad_outputs: list[torch.Tensor] = []

        for grads_tuple, intermediate in zip(
            param_group["grads"], param_group["intermediates"]
        ):
            for i, grad in enumerate(grads_tuple):
                if grad is not None:
                    valid_edges.append(GradientEdge(intermediate, i))
                    # pyrefly: ignore [bad-argument-type]
                    valid_grad_outputs.append(grad)

        # clamp saved grads to avoid double counting through descendant intermediates.
        handles = []
        try:
            if len(param_group["intermediates"]) > 1:
                for grads_tuple, intermediate in zip(
                    param_group["grads"],
                    param_group["intermediates"],
                    strict=True,
                ):
                    handles.append(
                        intermediate.register_prehook(
                            lambda grad_outputs, sg=grads_tuple: sg
                        )
                    )

            # break the hook -> param_group -> intermediates cycle
            del param_group["intermediates"]

            if valid_edges:
                weights_edges = tuple(GradientEdge(w, 0) for w in param_group["params"])
                dweights = torch.autograd.grad(
                    valid_edges,
                    weights_edges,
                    grad_outputs=valid_grad_outputs,
                    retain_graph=retain_graph,
                )

                del param_group["grads"]

                for grad_acc, dw in zip(param_group["params"], dweights):
                    weight, index = grad_acc_to_weight[grad_acc]
                    if weight.grad is None:
                        weight.grad = dw
                    else:
                        weight.grad += dw
        finally:
            for handle in handles:
                handle.remove()
    return tuple(weight_grads)


def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: list[int] | None = None,  # deprecated, not used
) -> tuple[torch.Tensor | None, ...]:
    """
    This is a helper function to:
    1. compute the gradients for the stage inputs, and
    2. accumulate gradients for the stage module's parameters.

    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """
    if outputs_with_grads_idxs is not None:
        # Deprecated, not used in runtime calls, only exists in compiler
        stage_output = [stage_output[i] for i in outputs_with_grads_idxs]
        output_grads = [output_grads[i] for i in outputs_with_grads_idxs]

    try:
        # stage_output may be a composite datatype like dict. Extract all individual
        # tensor values here
        stage_output_tensors: list[torch.Tensor] = []
        output_grad_tensors: list[torch.Tensor | None] = []

        def extract_tensors_with_grads(
            output_val,
            grad_val,
            # Don't delete me- see [Note: ref cycle]
            extract_tensors_with_grads,
        ):
            if isinstance(output_val, torch.Tensor):
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                if not isinstance(grad_val, (torch.Tensor, type(None))):
                    raise AssertionError(
                        f"Expected Tensor or None gradient but got {type(grad_val)}"
                    )
                stage_output_tensors.append(output_val)
                output_grad_tensors.append(grad_val)
            elif isinstance(output_val, (tuple, list)):
                if grad_val is None:
                    return
                if not isinstance(grad_val, (tuple, list)):
                    raise AssertionError(
                        f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
                    )
                if not len(output_val) == len(grad_val):
                    raise AssertionError(
                        f"Expected len(output_val) == len(grad_val), got {len(output_val)} != {len(grad_val)}"
                    )
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(
                        ov,
                        gv,
                        extract_tensors_with_grads,
                    )
            elif isinstance(output_val, dict):
                if grad_val is None:
                    return
                if not isinstance(grad_val, dict):
                    raise AssertionError(f"Expected dict, got {type(grad_val)}")
                if not set(output_val.keys()) == set(grad_val.keys()):
                    raise AssertionError(
                        f"Expected keys {set(output_val.keys())}, got {set(grad_val.keys())}"
                    )
                for k in output_val:
                    extract_tensors_with_grads(
                        output_val[k], grad_val[k], extract_tensors_with_grads
                    )
            else:
                # Output is a non-tensor type; just ignore it
                pass

        # Note: ref cycle
        # break a ref cycle that would keep tensors alive until GC runs
        # 1. extract_tensors_with_grads refers to a cell that holds refs to any vars defined in stage_backward
        #    and used in extract_tensors_with_grads
        # 2. extract_tensors_with_grads referred to both stage_output_tensors, output_grad_tensors,
        #    and to itself (extract_tensors_with_grads) since it makes a recursive call
        # 3. stage_output_tensors was kept alive by the above refcycle, and it holds activation tensors, which is bad
        # fix -> explicitly pass in the ref to the fn, so there is no gc cycle anymore
        extract_tensors_with_grads(
            stage_output, output_grads, extract_tensors_with_grads
        )

        torch.autograd.backward(
            stage_output_tensors,
            grad_tensors=output_grad_tensors,  # type: ignore[arg-type]
        )

        # Extract gradients wrt the input values
        grad_inputs: list[torch.Tensor | None] = []
        for val in input_values:
            if isinstance(val, torch.Tensor):
                grad_inputs.append(val.grad)
                # Since gradients that will pass back to previous stages do not require gradient accumulation,
                # by decrementing the gradients' reference count at this point, the memory of gradients will be
                # returned to the allocator as soon as the next micro batch's get_bwd_send_ops comes and current
                # asynchronous send completes.
                # This prevents the gradients from persisting in GPU memory for the entire duration of step_microbatches
                # until clear_runtime_states() is called.
                val.grad = None
            else:
                grad_inputs.append(None)

        # Alternative impl: `torch.autograd.grad`.
        # Note that `torch.autograd.grad` will not accumulate gradients into the
        # model's parameters.
        """
        inputs_with_grad = []
        for val in input_values:
            if isinstance(val, torch.Tensor) and val.requires_grad:
                inputs_with_grad.append(val)

        grad_inputs = torch.autograd.grad(
            stage_output_tensors, inputs_with_grad, output_grad_tensors,  # type: ignore[arg-type]
        )
        """

    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward:
        Stage output: {map_debug_info(stage_output)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e

    return tuple(grad_inputs)


# TODO: handling requires_grad=False dynamically. Can we analyze this during initial
# IR emission?
def _null_coalesce_accumulate(lhs, rhs):
    """
    Coalesce two values, even if one of them is null, returning the non-null
    value.
    """
    if lhs is None:
        return rhs
    elif rhs is None:
        return lhs
    else:
        return torch.add(lhs, rhs)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import copy
import csv
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Literal, NamedTuple, Protocol

import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import FSDPModule, UnshardHandle
from torch.nn.modules.loss import _Loss
from torch.profiler import record_function

from ._utils import (
    generate_rank_to_stage_mapping,
    generate_stage_to_rank_mapping,
    InferenceMode,
)
from .microbatch import (
    _split_tensor,
    merge_chunks,
    split_args_kwargs_into_chunks,
    TensorChunkSpec,
)
from .stage import _PipelineStageBase, _RecvInfo, PipelineStage


__all__ = [
    "get_schedule_class",
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
    "ScheduleInterleavedZeroBubble",
    "ScheduleZBVZeroBubble",
    "ScheduleDualPipeV",
]

logger = logging.getLogger(__name__)


class _ComputationType(str, Enum):
    # TODO(whc) rename to _ActType?
    FORWARD = "F"
    BACKWARD_INPUT = "I"
    BACKWARD_WEIGHT = "W"
    UNSHARD = "UNSHARD"
    RESHARD = "RESHARD"
    SEND_F = "SEND_F"
    RECV_F = "RECV_F"
    SEND_B = "SEND_B"
    RECV_B = "RECV_B"
    FULL_BACKWARD = "B"
    OVERLAP_F_B = "OVERLAP_F_B"
    REDUCE_GRAD = "REDUCE_GRAD"

    @staticmethod
    def from_str(action: str) -> "_ComputationType":
        try:
            return _ComputationType(action)
        except ValueError as exc:
            raise RuntimeError(f"Invalid computation type {action}") from exc


FORWARD = _ComputationType.FORWARD
BACKWARD_INPUT = _ComputationType.BACKWARD_INPUT
BACKWARD_WEIGHT = _ComputationType.BACKWARD_WEIGHT
UNSHARD = _ComputationType.UNSHARD
RESHARD = _ComputationType.RESHARD
SEND_F = _ComputationType.SEND_F
RECV_F = _ComputationType.RECV_F
SEND_B = _ComputationType.SEND_B
RECV_B = _ComputationType.RECV_B
FULL_BACKWARD = _ComputationType.FULL_BACKWARD
OVERLAP_F_B = _ComputationType.OVERLAP_F_B
REDUCE_GRAD = _ComputationType.REDUCE_GRAD


# Targets (e.g. labels) are always split along the batch dim (0). Use
# _split_tensor so DTensor targets preserve their Shard placements instead of
# being silently all-gathered to Replicate by the default dispatch path.
_TARGET_CHUNK_SPEC = TensorChunkSpec(0)


# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

# Helper to parse an action string like 1F0 into a tuple of (stage_index, computation_type, microbatch_index)
_action_regex = re.compile(
    r"(\d+)(F|I|B|W|UNSHARD|RESHARD|REDUCE_GRAD|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)"
)


class _Action(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: int | None = None
    sub_actions: tuple["_Action", ...] | None = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.sub_actions is not None:
            # Use recursive repr for sub_actions
            sub_action_reprs = [repr(sub_action) for sub_action in self.sub_actions]
            return f"({';'.join(sub_action_reprs)}){self.computation_type.value}"
        else:
            repr_str = str(self.stage_index)
            # Use .value to get the short string (e.g., "F", "B") instead of the full enum name
            repr_str += self.computation_type.value
            if self.microbatch_index is not None:
                repr_str += str(self.microbatch_index)
            return repr_str

    @property
    def is_compute_op(self) -> bool:
        return self.computation_type in (
            FORWARD,
            FULL_BACKWARD,
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            OVERLAP_F_B,
        )

    @staticmethod
    def from_str(action_string: str):
        """
        Reverse of __repr__

        String should be formatted as [stage][action type][(microbatch)]
            e.g. `2F0`, `1UNSHARD`, `3SEND_F1`
        """
        action_string = action_string.strip()
        if action_string == "":
            return None

        # Check for sub_actions format: [sub_action1;sub_action2;...]ComputationType
        if action_string.startswith("(") and ")" in action_string:
            # Find the closing bracket to separate sub_actions from computation type
            bracket_end = action_string.find(")")
            sub_part = action_string[
                1:bracket_end
            ]  # Remove '[' and get content before ']'
            computation_type_part = action_string[
                bracket_end + 1 :
            ]  # Get part after ']'

            # Parse sub_actions
            sub_actions = []
            if sub_part.strip():
                for sub_str in sub_part.split(";"):
                    sub_action = _Action.from_str(sub_str.strip())
                    if sub_action is not None:
                        sub_actions.append(sub_action)

            # For sub_actions format, we create an action with just the computation type
            # The stage_index and microbatch_index are not meaningful for the container action
            return _Action(
                stage_index=-1,  # Placeholder, not meaningful for sub_actions container
                computation_type=_ComputationType.from_str(computation_type_part),
                microbatch_index=None,
                sub_actions=tuple(sub_actions) if sub_actions else None,
            )

        # Handle regular single action format
        if match := _action_regex.match(action_string):
            stage_index, computation_type, microbatch_index = match.groups()
            return _Action(
                int(stage_index),
                _ComputationType.from_str(computation_type),
                int(microbatch_index) if len(microbatch_index) else None,
            )
        elif action_string == "":
            return None
        raise RuntimeError(
            f"Invalid action string: {action_string}, should be formatted as [stage][action type][(microbatch)] e.g. 2F0"
        )


@lru_cache
def _get_profiler_function_name(action: _Action) -> str:
    return f"PP:{str(action)}"


def _format_pipeline_order(
    pipeline_order: dict[int, list[_Action | None]],
    error_step_number: int | None = None,
) -> str:
    """
    Formats the pipeline order in a timestep (row) x rank (column) grid of actions
    and returns the formatted string.

    If `error_step_number` is passed in, an additional label will be added to signify which step
    that it is erroring on.
    """

    # don't mutate the original
    pipeline_order = copy.deepcopy(pipeline_order)

    # Replace None with ""
    for rank in pipeline_order:
        for i in range(len(pipeline_order[rank])):
            if pipeline_order[rank][i] is None:
                # TODO make a real 'None action' that prints as empty string and make mypy happy
                pipeline_order[rank][i] = ""  # type: ignore[call-overload]

    # Calculate the maximum number of steps across all ranks
    num_steps = max(len(actions) for actions in pipeline_order.values())
    step_labels = [
        "Step " + str(i).zfill(len(str(num_steps - 1))) for i in range(num_steps)
    ]
    # Sorting the dictionary by keys and retrieving values in that order
    rank_actions = [
        pipeline_order.get(key, [""] * num_steps) for key in sorted(pipeline_order)
    ]
    # Transpose the list of lists (rows to columns)
    # pyrefly: ignore [no-matching-overload]
    transposed_actions = list(itertools.zip_longest(*rank_actions, fillvalue=""))
    # Generate column labels for ranks
    num_ranks = len(pipeline_order)
    rank_labels = ["Rank " + str(i) for i in range(num_ranks)]
    # Calculate the maximum length of each column, considering labels
    max_lengths = [
        max(len(str(item)) if item is not None else 0 for item in col)
        for col in zip(step_labels, *transposed_actions)
    ]
    # Format the header row with rank labels
    header_row = " " * (len(step_labels[0]) + 2) + " ".join(
        f"{label:<{max_lengths[i]}}" for i, label in enumerate(rank_labels)
    )
    # Format each row with its corresponding label
    formatted_rows = [
        f"{label}: "
        + " ".join(f"{str(item):<{max_lengths[i]}}" for i, item in enumerate(row))
        + (
            " <-- ERROR HERE"
            if error_step_number is not None
            and int(label.split()[1]) == error_step_number
            else ""
        )
        for label, row in zip(step_labels, transposed_actions)
    ]
    # Join the rows into a single string
    formatted_table = header_row + "\n" + "\n".join(formatted_rows) + "\n"
    return formatted_table


class _PipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn

        # See documentation in `PipelineScheduleSingle` / `PipelineScheduleMulti`
        self.scale_grads = scale_grads

        # Chunking specification for positional inputs. (default: `None`)
        self._args_chunk_spec = args_chunk_spec
        # Chunking specification for keyword inputs. (default: `None`)
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """

        # Derived
        self._has_backward = self._loss_fn is not None

        # Holds the losses for each microbatch.
        self._internal_losses: list[torch.Tensor] = []
        logger.info("Using %s", self.__class__.__name__)

    def _maybe_compute_loss(
        self, stage, output, target_mbs, mb_index, loss_kwargs=None
    ):
        if stage.is_last and self._loss_fn is not None:
            loss = self._compute_loss(output, target_mbs[mb_index], loss_kwargs)  # type: ignore[index]
            self._internal_losses.append(loss)

    def _maybe_get_loss(self, stage, mb_index):
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._loss_fn is not None and valid_index:
            return self._internal_losses[mb_index]
        elif len(self._internal_losses) != 0 and not valid_index:
            raise RuntimeError(
                f"Loss for microbatch {mb_index} is not available. "
                f"Available losses for microbatches: {self._internal_losses}"
            )
        else:
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # if stages not a list turn into a list
        if not isinstance(stages, list):
            stages = [stages]
        contains_last_stage = any(stage.is_last for stage in stages)

        # Return losses if there is a container passed in
        if contains_last_stage and losses is not None:
            if len(self._internal_losses) != self._n_microbatches:
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}"
                )

            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(self._internal_losses)

        self._internal_losses.clear()

    def _warmup_p2p(
        self,
        stages: list[_PipelineStageBase],
        has_backward: bool,
        p2p_done: bool,
    ) -> None:
        """Run the P2P warm-up protocol for the given stages.

        For ``PipelineStage`` instances this executes the forward/backward vote
        protocol (which warms up 2-rank sub-communicators) and sets each
        stage's ``_inference_mode``.  For other stage types it falls back to
        the legacy ``_get_init_p2p_neighbors_ops`` + ``_batch_p2p`` path.

        Args:
            stages: The pipeline stages owned by this rank.
            has_backward: Whether the schedule includes a backward pass.
            p2p_done: ``True`` if P2P neighbours have already been initialised
                (avoids redundant init on eval↔train mode switches).
        """
        if all(isinstance(stage, PipelineStage) for stage in stages):
            # A fake process group cannot exchange real data: a cross-rank
            # vote recv reads zeros and selects DYNAMIC, which then fails in
            # `_recv_meta` (nothing was actually sent). Since dynamic
            # inference can never work across ranks of a fake group, decide
            # locally in that case: STATIC if every local stage has complete
            # metadata, else error. Same-rank-only pipelines (e.g. a
            # single-rank fake world) don't communicate, so the normal vote
            # still works and DYNAMIC remains usable there.
            pp_stages = cast(list[PipelineStage], stages)
            has_cross_rank = any(
                (not st.is_first and not st._is_same_rank(st.stage_index - 1))
                or (not st.is_last and not st._is_same_rank(st.stage_index + 1))
                for st in pp_stages
            )
            if has_cross_rank and any(
                dist.get_backend(st.group) == "fake" for st in pp_stages
            ):
                for st in pp_stages:
                    if InferenceMode.needs_dynamic(st._user_meta, has_backward):
                        raise RuntimeError(
                            f"Stage {st.stage_index} requires dynamic shape "
                            "inference, which is not supported with a fake "
                            "process group. Provide complete static metadata "
                            "(inputs/outputs, plus input_grads/output_grads "
                            "for DTensors with backward) to the PipelineStage "
                            "constructor."
                        )
                    st._inference_mode = InferenceMode.STATIC
                logger.debug(
                    "Fake process group detected; set inference_mode=static "
                    "for %d stage(s) without voting",
                    len(stages),
                )
                return
            acc: torch.Tensor | None = None
            for stage in cast(list[PipelineStage], stages):
                acc = stage._warmup_forward_vote(has_backward, received_acc=acc)
            result: torch.Tensor | None = acc
            determined_mode: InferenceMode | None = None
            for stage in reversed(cast(list[PipelineStage], stages)):
                result = stage._warmup_backward_result(received_result=result)
                if result is None:
                    raise RuntimeError("P2P warm-up voting failed")
                determined_mode = (
                    InferenceMode.STATIC
                    if result.item() == 1
                    else InferenceMode.DYNAMIC
                )
                stage._inference_mode = determined_mode
            logger.debug(
                "Rank determined inference_mode=%s for %d stage(s)",
                determined_mode.value if determined_mode else "None",
                len(stages),
            )
        elif not p2p_done:
            all_ops: list[dist.P2POp] = []
            for stage in stages:
                all_ops.extend(stage._get_init_p2p_neighbors_ops())
            _wait_batch_p2p(_batch_p2p(all_ops))

        # TODO: STATIC mode group communicator warm-up gap
        # The vote protocol above warms up 2-rank sub-communicators
        # (used by `_batch_p2p` homogeneous fast-path).  In DYNAMIC mode,
        # `_send_meta`/`_recv_meta` (called during `_prepare_forward_infra` →
        # `_forward_metadata_inference`) also warm up the *group* communicator
        # (used by `_batch_p2p` mixed-op path).  In STATIC mode, metadata
        # inference is skipped, so the group communicator is NOT warmed up —
        # it will be lazily created on the first mixed `_batch_p2p` call
        # (e.g., 1F1B steady-state with both sends and recvs).
        # Fix: run `_get_init_p2p_neighbors_ops` + `_batch_p2p` after the
        # vote, gated by `not p2p_done`.

    def _initialize_pp_stages(
        self,
        stages: list[_PipelineStageBase],
        args: tuple[Any, ...] | Any,
        kwargs: dict[str, Any] | None,
        target: Any,
        fwd_initialized: bool,
        bwd_initialized: bool,
        loss_kwargs: dict[str, Any] | None = None,
    ) -> tuple[bool, bool]:
        """Common stage initialization shared by Single and Multi schedules.

        Handles mode-change detection (eval↔train), P2P warm-up, RNG forking,
        forward / backward metadata inference, and FSDP cleanup.

        Returns the updated ``(fwd_initialized, bwd_initialized)`` flags.
        """
        # Detect eval↔train mode switch: if has_backward changed since last
        # init, re-initialize both fwd (recv buffers need different
        # requires_grad) and bwd.  p2p_done avoids redundant P2P warm-up.
        p2p_done = fwd_initialized
        if fwd_initialized and (self._has_backward != bwd_initialized):
            fwd_initialized = False
            bwd_initialized = False

        needs_fwd = not fwd_initialized
        needs_bwd = self._has_backward and not bwd_initialized

        if not needs_fwd and not needs_bwd:
            return fwd_initialized, bwd_initialized

        if needs_fwd:
            self._warmup_p2p(stages, self._has_backward, p2p_done)

        # Fork RNG so metadata inference doesn't perturb training RNG.
        devices = list(
            {
                torch.device(stage.device)
                for stage in stages
                if torch.device(stage.device).type != "cpu"
            }
        )
        device_type = devices[0].type if devices else None
        # Assert all device types are the same
        if device_type is not None and not all(
            device.type == device_type for device in devices
        ):
            device_types = {device.type for device in devices}
            raise AssertionError(
                "All stages must have the same device type for RNG forking. "
                f"Found device types: {device_types}"
            )
        pipeline_stages = [
            stage for stage in stages if isinstance(stage, PipelineStage)
        ]
        for stage in pipeline_stages:
            stage._pre_metadata_inference_backup()

        try:
            with torch.random.fork_rng(devices=devices, device_type=device_type):
                if needs_fwd:
                    next_stage_args: Any = None
                    for stage in stages:
                        stage_args = args if stage.is_first else next_stage_args
                        next_stage_args = stage._prepare_forward_infra(
                            self._n_microbatches,
                            stage_args,
                            kwargs,
                            has_backward=self._has_backward,
                        )
                    fwd_initialized = True

                if needs_bwd:
                    prev_stage_grad_meta: Any = None
                    for stage in reversed(stages):
                        prev_stage_grad_meta = stage._prepare_backward_infra(
                            self._n_microbatches,
                            loss_fn=self._loss_fn,
                            target=target,
                            received_grad_meta=prev_stage_grad_meta,
                            loss_kwargs=loss_kwargs,
                        )
                    bwd_initialized = True
        finally:
            for stage in pipeline_stages:
                stage._post_metadata_inference_cleanup()

        return fwd_initialized, bwd_initialized

    @abstractmethod
    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
            loss_kwargs: extra keyword arguments forwarded to the loss function.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        *args,
        target=None,
        losses: list | None = None,
        return_outputs=True,
        loss_kwargs: dict[str, Any] | None = None,
        arg_mbs: Any = None,
        kwarg_mbs: Any = None,
        target_mbs: Any = None,
        **kwargs,
    ):
        r"""Run one iteration of the pipeline schedule."""
        raise NotImplementedError

    def eval(
        self,
        *args,
        target=None,
        losses: list | None = None,
        arg_mbs: Any = None,
        kwarg_mbs: Any = None,
        target_mbs: Any = None,
        **kwargs,
    ):
        r"""Run one forward-only iteration of the pipeline schedule.

        ``eval`` uses the same input contract as ``step``, but temporarily
        disables backward execution. By default, ``args``, ``kwargs``, and
        ``target`` are full-batch values that the schedule splits into
        microbatches. When the caller has already split inputs into
        microbatches, pass them through ``arg_mbs``, ``kwarg_mbs``, and
        ``target_mbs`` instead.

        Args:
            \*args (Any): Whole-batch positional root inputs when this rank owns
                the first pipeline stage. Do not pass positional inputs with
                pre-split inputs.
            target (Any, optional): Whole-batch target for loss computation.
                When passing pre-split inputs, pass targets through
                ``target_mbs`` instead. Default: ``None``.
            losses (list, optional): Mutable list populated with one loss per
                microbatch when this schedule owns the last stage and a
                ``loss_fn`` was configured. Default: ``None``.
            arg_mbs (list[tuple], optional): Pre-split positional inputs, one
                tuple per microbatch. Default: ``None``.
            kwarg_mbs (list[dict], optional): Pre-split keyword inputs, one
                dict per microbatch. Default: ``None``.
            target_mbs (list, optional): Pre-split targets, one entry per
                microbatch. Default: ``None``.
            \*\*kwargs (Any): Whole-batch keyword root inputs when this rank owns
                the first pipeline stage. Do not pass keyword inputs with
                pre-split inputs.

        Returns:
            Any or None: The merged output from the last stage when this rank
                owns the last stage; otherwise ``None``.

        Raises:
            TypeError: If a pre-split microbatch container has the wrong type.
            ValueError: If whole-batch and pre-split inputs are mixed, or if a
                pre-split container does not have one entry per microbatch.

        Examples::

            >>> # xdoctest: +SKIP("requires a constructed distributed pipeline schedule")
            >>> output = schedule.eval(x, mask=mask)
            >>> arg_mbs = [(x0,), (x1,)]
            >>> kwarg_mbs = [{"mask": mask0}, {"mask": mask1}]
            >>> output = schedule.eval(
            ...     arg_mbs=arg_mbs,
            ...     kwarg_mbs=kwarg_mbs,
            ... )
        """
        # Save the original has_backward state
        original_has_backward = self._has_backward
        try:
            self._has_backward = False
            return self.step(
                *args,
                target=target,
                losses=losses,
                arg_mbs=arg_mbs,
                kwarg_mbs=kwarg_mbs,
                target_mbs=target_mbs,
                **kwargs,
            )
        finally:
            # Restore the original state
            self._has_backward = original_has_backward

    def _check_inputs(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
    ) -> tuple[list, list]:
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            if len(mbs) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(mbs)}"
                )

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            if not isinstance(losses, list):
                raise TypeError(f"losses must be a list but got a {type(losses)}")

        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output, target, loss_kwargs=None):
        return self._loss_fn(output, target, **(loss_kwargs or {}))  # type: ignore[misc]

    def _split_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                self._args_chunk_spec,
                self._kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _get_microbatch_inputs(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        target: Any,
        arg_mbs: Any,
        kwarg_mbs: Any,
        target_mbs: Any,
    ) -> tuple[list | None, list | None, list | None]:
        pre_split = any(mbs is not None for mbs in (arg_mbs, kwarg_mbs, target_mbs))
        if not pre_split:
            args_split, kwargs_split = self._split_inputs(args, kwargs)
            targets_split = (
                list(_split_tensor(target, _TARGET_CHUNK_SPEC, self._n_microbatches))
                if target is not None
                else None
            )
            return args_split, kwargs_split, targets_split

        if args:
            raise ValueError(
                "When using pre-split inputs, pass pre-split positional inputs "
                "through arg_mbs=... instead of positional args."
            )

        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(
                f"Unexpected keyword arguments with pre-split inputs: {names}. "
                "Pass pre-split keyword inputs through kwarg_mbs=..."
            )

        if target is not None:
            raise ValueError(
                "When using pre-split inputs, pass pre-split targets through "
                "target_mbs=... instead of target=..."
            )

        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs)

        for mb_index, (arg_mb, kwarg_mb) in enumerate(
            zip(arg_mbs, kwarg_mbs, strict=True)
        ):
            if not isinstance(arg_mb, tuple):
                raise TypeError(
                    "arg_mbs must be a list of tuples, but "
                    f"arg_mbs[{mb_index}] is a {type(arg_mb)}"
                )

            if not isinstance(kwarg_mb, dict):
                raise TypeError(
                    "kwarg_mbs must be a list of dicts, but "
                    f"kwarg_mbs[{mb_index}] is a {type(kwarg_mb)}"
                )

        return arg_mbs, kwarg_mbs, target_mbs

    def _merge_outputs(self, output_chunks: list[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


def _batch_p2p(p2p_ops: list[dist.P2POp], desc: str | None = None) -> list[dist.Work]:
    """
    Wrapper over batch_isend_irecv that avoids coalescing for homogeneous
    batches (all-send or all-recv).  Coalescing serializes ops on a single
    CUDA stream, which causes head-of-line blocking when independent P2P ops
    could otherwise overlap.  Mixed batches still use batch_isend_irecv for
    deadlock avoidance.
    """
    if len(p2p_ops) == 0:
        return []
    desc_str = f"{desc}, " if desc else ""
    logger.debug("batch_p2p %s%s", desc_str, p2p_ops)

    # Per-direction P2P (config.pipeline_per_direction_p2p) tags forward and
    # backward ops with different communicators. A fused batch (e.g. 1F1B's
    # fwd_sends + bwd_recvs) then spans >1 group; issue each group's ops as their
    # own batch so they run on separate comms/streams instead of one FIFO. When
    # all ops share a group (the default), this is a no-op fast path.
    ops_by_group: dict[str, list[dist.P2POp]] = {}
    for p in p2p_ops:
        ops_by_group.setdefault(p.group.group_name, []).append(p)
    if len(ops_by_group) > 1:
        works: list[dist.Work] = []
        # Issue the groups in a deterministic, rank-independent order (by group
        # name -- identical on every rank for a given group). The same fused
        # batch is built in opposite list order on neighboring ranks (e.g.
        # ``fwd_sends + bwd_recvs`` vs ``bwd_sends + fwd_recvs``); sorting keeps
        # every rank issuing the two communicators in the same order.
        for _, group_ops in sorted(ops_by_group.items()):
            works += _batch_p2p(group_ops, desc=desc)
        return works

    op_types = {p.op for p in p2p_ops}
    if op_types == {dist.isend}:
        send_works = [
            p.op(p.tensor, group=p.group, tag=p.tag, group_dst=p.group_peer)
            for p in p2p_ops
        ]
        return [work for work in send_works if work is not None]
    if op_types == {dist.irecv}:
        recv_works = [
            p.op(p.tensor, group=p.group, tag=p.tag, group_src=p.group_peer)
            for p in p2p_ops
        ]
        return [work for work in recv_works if work is not None]

    return dist.batch_isend_irecv(p2p_ops)


def _sorted_batch_p2p(
    p2p_ops: list[dist.P2POp], desc: str | None = None
) -> dict[int, list[dist.Work]]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: dict[int, list[dist.P2POp]] = defaultdict(list)
    work_by_peer: dict[int, list[dist.Work]] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


def _wait_batch_p2p(work: list[dist.Work]):
    """
    Waits for a list of dist.Work (typically from _batch_p2p / _sorted_batch_p2p).
    """
    for w in work:
        w.wait()


class PipelineScheduleSingle(_PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.

    Gradients are scaled by num_microbatches depending on the `scale_grads` argument, defaulting to True.  This setting
    should match the configuration of your loss_fn, which may either average losses (scale_grads=True)
    or sum losses (scale_grads=False).
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        self._stage_forward_initialized = False
        self._stage_backward_initialized = False

        self.pipeline_order: dict[int, list[_Action | None]] | None = (
            self._get_pipeline_order()
        )

    def _initialize_stage(self, args, kwargs, target=None, loss_kwargs=None):
        (
            self._stage_forward_initialized,
            self._stage_backward_initialized,
        ) = self._initialize_pp_stages(
            [self._stage],
            args,
            kwargs,
            target,
            self._stage_forward_initialized,
            self._stage_backward_initialized,
            loss_kwargs=loss_kwargs,
        )

    def step(
        self,
        *args,
        target=None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
        arg_mbs: Any = None,
        kwarg_mbs: Any = None,
        target_mbs: Any = None,
        **kwargs,
    ):
        r"""Run one training iteration of a single-stage pipeline schedule.

        By default, ``args``, ``kwargs``, and ``target`` are full-batch values
        that the schedule splits into microbatches. When the caller has already
        split inputs into microbatches, pass them through ``arg_mbs``,
        ``kwarg_mbs``, and ``target_mbs`` instead.

        Args:
            \*args (Any): Whole-batch positional inputs for the first pipeline
                stage. Do not pass positional inputs with pre-split inputs.
            target (Any, optional): Whole-batch target for loss computation.
                When passing pre-split inputs, pass targets through
                ``target_mbs`` instead. Default: ``None``.
            losses (list, optional): Mutable list populated with one loss per
                microbatch when this schedule owns the last stage and a
                ``loss_fn`` was configured. Default: ``None``.
            return_outputs (bool, optional): Whether to merge and return output
                chunks on the last stage. Default: ``True``.
            loss_kwargs (dict, optional): Extra keyword arguments forwarded to
                the configured ``loss_fn``. Default: ``None``.
            arg_mbs (list[tuple], optional): Pre-split positional inputs, one
                tuple per microbatch. Default: ``None``.
            kwarg_mbs (list[dict], optional): Pre-split keyword inputs, one
                dict per microbatch. Default: ``None``.
            target_mbs (list, optional): Pre-split targets, one entry per
                microbatch. Default: ``None``.
            \*\*kwargs (Any): Whole-batch keyword inputs for the first pipeline
                stage. Do not pass keyword inputs with pre-split inputs.

        Returns:
            Any or None: The merged output from the last stage when this rank
                owns the last stage and ``return_outputs=True``; otherwise
                ``None``.

        Raises:
            RuntimeError: If backward computation is enabled and ``step`` is
                called under ``torch.no_grad()``. Use :meth:`eval` for
                forward-only execution.
            TypeError: If a pre-split microbatch container has the wrong type.
            ValueError: If whole-batch and pre-split inputs are mixed, or if a
                pre-split container does not have one entry per microbatch.

        Examples::

            >>> # xdoctest: +SKIP("requires a constructed distributed pipeline schedule")
            >>> output = schedule.step(x, mask=mask, target=target)
            >>> arg_mbs = [(x0,), (x1,)]
            >>> kwarg_mbs = [{"mask": mask0}, {"mask": mask1}]
            >>> target_mbs = [target0, target1]
            >>> output = schedule.step(
            ...     arg_mbs=arg_mbs,
            ...     kwarg_mbs=kwarg_mbs,
            ...     target_mbs=target_mbs,
            ... )
        """
        if self._has_backward and not torch.is_grad_enabled():
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation; "
                "it should not be used under torch.no_grad() context. "
                "Please call eval() instead."
            )

        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward

        # Clean per iteration
        self._stage.clear_runtime_states()

        args_split, kwargs_split, targets_split = self._get_microbatch_inputs(
            args,
            kwargs,
            target,
            arg_mbs,
            kwarg_mbs,
            target_mbs,
        )

        # Run microbatches
        self._step_microbatches(
            args_split,
            kwargs_split,
            targets_split,
            losses,
            return_outputs,
            loss_kwargs=loss_kwargs,
        )

        # Return merged results per original format
        if self._stage.is_last and return_outputs:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline execution order as a schedule IR.

        The returned IR is a dictionary mapping rank IDs to lists of actions.
        Each action is either an _Action object representing computation to perform,
        or None representing a deliberate idle step.

        The None values are used to represent pipeline bubbles where a rank
        must wait for dependencies from other ranks before proceeding. However
        during execution, with  the _PipelineScheduleRuntime, these Nones are
        skipped since the relevant communication (send/recv) will be scheduled and waited on.

        Returns:
            A dictionary mapping rank -> list of actions
        """
        return None


class _ScheduleForwardOnly(PipelineScheduleSingle):
    """
    The forward-only schedule.
    Will go through all the microbatches and perform only the forward pass
    """

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ):
        """
        Run one iteration of the pipeline schedule
        """
        if target_mbs is not None or losses is not None:
            raise RuntimeError(
                "Forward-only schedule does not support loss computation"
            )

        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        maybe_first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stage(arg_mbs[0], kwarg_mbs[0], maybe_first_target)

        # Delay send waits
        fwd_sends_to_wait: list[list[dist.Work]] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            _wait_batch_p2p(work)


class ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        maybe_first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stage(
            arg_mbs[0], kwarg_mbs[0], maybe_first_target, loss_kwargs
        )

        # Delay send waits
        fwd_sends_to_wait: list[list[dist.Work]] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                output = self._stage.forward_one_chunk(
                    i, arg_mbs[i], kwarg_mbs[i], save_forward_output=return_outputs
                )  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)

            self._maybe_compute_loss(self._stage, output, target_mbs, i, loss_kwargs)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            _wait_batch_p2p(work)

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: list[list[dist.Work]] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    _wait_batch_p2p(work)

                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(
                    i,
                    loss=loss,
                    last_backward=i == self._n_microbatches - 1,
                )

                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Backwarded microbatch %s", self._stage.stage_index, i)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            _wait_batch_p2p(work)

        # Update losses if there is a container passed in
        self._update_losses(self._stage, losses)

        self._stage.perform_reduce_grad(self._n_microbatches if self.scale_grads else 1)

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline order for GPipe schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """
        pipeline_order = {}
        pp_group_size = self._num_stages

        for rank in range(pp_group_size):
            actions: list[_Action | None] = []

            # 1. Initial delay based on rank position
            warmup_delay = rank
            actions.extend([None] * warmup_delay)

            # 2. Forward passes for all microbatches
            for mb_idx in range(self._n_microbatches):
                actions.append(_Action(rank, _ComputationType.FORWARD, mb_idx))

            # 3. Wait period before backward passes can begin
            backward_delay = 3 * (pp_group_size - 1 - rank)
            actions.extend([None] * backward_delay)

            # 4. Backward passes for all microbatches
            for mb_idx in range(self._n_microbatches):
                actions.append(_Action(rank, _ComputationType.FULL_BACKWARD, mb_idx))

            pipeline_order[rank] = _add_reduce_grad(actions, self._n_microbatches)

        return pipeline_order  # type: ignore[return-value]


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        scale_grads: bool = True,
    ):
        super().__init__(
            stage=stage,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        if n_microbatches < self._num_stages:
            raise ValueError(
                f"Number of microbatches ({n_microbatches}) must be greater than \
or equal to the number of stages ({self._num_stages})."
            )

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
            return_outputs: whether to return the outputs from the last stage.
            loss_kwargs: extra keyword arguments forwarded to the loss function.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        maybe_first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stage(
            arg_mbs[0], kwarg_mbs[0], maybe_first_target, loss_kwargs
        )

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_work: list[dist.Work] = []
        fwd_sends = []
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            _wait_batch_p2p(_batch_p2p(fwd_recvs, desc="fwd_recv"))

            # Compute
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            _wait_batch_p2p(send_work)

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last forward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(
                self._stage, output, target_mbs, fwd_mb_index, loss_kwargs
            )
            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)

            # Now, we need to fire the fwd_sends and bwd_recvs together
            _wait_batch_p2p(_batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv"))

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)

            # Fuse it with bwd_sends above
            _wait_batch_p2p(_batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv"))

            # Now do the fwd
            output = self._stage.forward_one_chunk(
                fwd_mb_index,
                arg_mbs[fwd_mb_index],
                kwarg_mbs[fwd_mb_index],
                save_forward_output=return_outputs,
            )  # type: ignore[index]

            # Compute loss
            self._maybe_compute_loss(
                self._stage, output, target_mbs, fwd_mb_index, loss_kwargs
            )

            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = _batch_p2p(bwd_sends, desc="bwd_send")

        # Cooldown
        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            _wait_batch_p2p(_batch_p2p(bwd_recvs, desc="bwd_recv"))

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Clear previous chunk's backward sends (hopefully they have well finished)
            _wait_batch_p2p(send_work)

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1

        # Wait for the last backward send to finish
        _wait_batch_p2p(send_work)

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        self._stage.perform_reduce_grad(self._n_microbatches if self.scale_grads else 1)

    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline order for 1F1B schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """
        pipeline_order = {}
        pp_group_size = self._num_stages

        for rank in range(pp_group_size):
            actions: list[_Action | None] = []

            # 1. Warmup phase: initial delay based on rank
            actions.extend([None] * rank)

            # 2. Initial forward passes before 1F1B phase
            num_forward = (pp_group_size - 1) - rank
            forward_mb = 0
            for i in range(num_forward):
                actions.append(_Action(rank, _ComputationType.FORWARD, i))
                forward_mb = i

            # 3. Wait for backward to be ready
            wait_for_1f1b = max(0, 2 * (pp_group_size - 1 - rank))
            actions.extend([None] * wait_for_1f1b)

            # 4. 1F1B steady state phase
            backward_mb = 0
            remaining_forward = self._n_microbatches - num_forward

            while remaining_forward > 0:
                # One forward
                forward_mb += 1
                actions.append(_Action(rank, _ComputationType.FORWARD, forward_mb))
                remaining_forward -= 1

                # One backward
                actions.append(
                    _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                )
                backward_mb += 1

            # 5. Cooldown phase: remaining backward passes
            remaining_backward = self._n_microbatches - backward_mb

            while remaining_backward > 0:
                # Add None and backward actions in alternating pattern
                # based on distance from the last stage
                if (pp_group_size - rank) > 0:
                    actions.append(None)
                    # Decrement the wait counter only if we still have backward passes to do
                    if remaining_backward > 0:
                        actions.append(
                            _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                        )
                        backward_mb += 1
                        remaining_backward -= 1
                else:
                    # If we're at the last stage, just add backward actions without None
                    actions.append(
                        _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                    )
                    backward_mb += 1
                    remaining_backward -= 1

            pipeline_order[rank] = _add_reduce_grad(actions, self._n_microbatches)
        return pipeline_order


def _requires_reduce_grad(action_type: _ComputationType) -> bool:
    return action_type in (W, B)


def _add_reduce_grad(
    actions: list[_Action | None], n_microbatches: int
) -> list[_Action | None]:
    """
    REDUCE_GRAD refers to joint across minibatches grad reduction.
    reduce_grad frees memory and we want to schedule it just after the last "backward"-like stage.
    """
    actions_with_reduce_grad: list[_Action | None] = []
    cnt: dict[int, int] = defaultdict(int)

    def _leaf_action(a, to_schedule):
        if _requires_reduce_grad(a.computation_type):
            stage_index = a.stage_index
            cnt[stage_index] += 1
            if cnt[stage_index] == n_microbatches:
                to_schedule.append(stage_index)

    for a in actions:
        if a is None:
            continue
        actions_with_reduce_grad.append(a)
        schedule_reduce_grad_stage_idxs: list[int] = []
        if a.computation_type == OVERLAP_F_B and a.sub_actions is not None:
            for sub_action in a.sub_actions:
                _leaf_action(sub_action, schedule_reduce_grad_stage_idxs)
        else:
            _leaf_action(a, schedule_reduce_grad_stage_idxs)

        for stage_idx in schedule_reduce_grad_stage_idxs:
            actions_with_reduce_grad.append(_Action(stage_idx, REDUCE_GRAD, None))
    return actions_with_reduce_grad


def _add_unshard_reshard(
    compute_actions: list[_Action | None],
    max_active_stages: int = 3,
) -> list[_Action]:
    """Given a basic schedule involving only compute actions (F,B,W,OVERLAP_F_B), add UNSHARD/RESHARD actions for FSDP.

    UNSHARD refers to fetching the full contents of an FSDP-sharded layer, requiring an all-gather operation.
    RESHARD does the opposite, releasing memory (but doing no communication)

    We abandon the "timestep lock"  during lowering

    max_active_stages controls how many prefetches we allow. It should be measured in mb and tuneable but in practice
    3 stages is probably the thing we want?
    (to account for having one f and one b active, and something else prefetching?)
    """

    def next_stage_indices(count: int, next_actions: list[_Action | None]) -> list[int]:
        """Remove duplicates (same stage, different microbatch), find next 'count' stages that will do compute."""
        seen: set[int] = set()
        ret: list[int] = []

        for a in next_actions:
            if a is not None:
                # Handle OVERLAP_F_B actions by checking their sub_actions
                if a.computation_type == OVERLAP_F_B and a.sub_actions is not None:
                    for sub_action in a.sub_actions:
                        if sub_action.stage_index not in seen:
                            seen.add(sub_action.stage_index)
                            ret.append(sub_action.stage_index)
                    if len(ret) >= count:
                        break
                else:
                    # Regular action
                    if a.stage_index not in seen:
                        seen.add(a.stage_index)
                        ret.append(a.stage_index)
                        if len(ret) == count:
                            break
        return ret

    active_stages: set[int] = set()
    fsdp_aware_actions: list[_Action] = []

    def _unshard(stage_index: int):
        active_stages.add(stage_index)
        fsdp_aware_actions.append(_Action(stage_index, UNSHARD, None))

    def _reshard(stage_index: int):
        active_stages.remove(stage_index)
        fsdp_aware_actions.append(_Action(stage_index, RESHARD, None))

    for i, action in enumerate(compute_actions):
        if action is None:
            continue

        # We prefetch the next N stages we'll see, dropping existing stages to make room
        next_n = next_stage_indices(max_active_stages, compute_actions[i:])
        # Fetch needs to be ordered correctly, so don't use a set
        fetch = list(filter(lambda s: s not in active_stages, next_n))
        # Unclear what the best policy is for eviction, but we can maintain order so we do
        evict = list(filter(lambda s: s not in next_n, active_stages))

        # logger.debug(
        #     "_add_unshard_reshard Step %d active: %s fetch %s, evict %s",
        #     i,
        #     active_stages,
        #     fetch,
        #     evict,
        # )

        for stage in evict:
            _reshard(stage)
        for stage in fetch:
            _unshard(stage)
        fsdp_aware_actions.append(action)

    # Reshard all remaining active stages after processing all operations
    for stage in list(active_stages):
        _reshard(stage)

    return fsdp_aware_actions


def _merge_bw(
    compute_actions: list[_Action | None],
) -> list[_Action]:
    """Given a basic schedule involving only compute actions (F,I,W), merge adjacent I and W ops into B ops.
    (note: I = BACKWARD_INPUT, W = BACKWARD_WEIGHT, B = FULL_BACKWARD)

    B refers to running the whole backward (not separating grad_input and grad_weight), which can be more efficient
    in some cases.
    """
    merged_actions = []
    while compute_actions:
        action = compute_actions.pop(0)
        if action is None:
            continue

        # Remove any None actions and find the next non-None action
        while len(compute_actions) and compute_actions[0] is None:
            compute_actions.pop(0)

        # Get the next action if it exists
        next_action = compute_actions[0] if len(compute_actions) > 0 else None

        if (
            action.computation_type == BACKWARD_INPUT
            and next_action is not None
            and next_action.computation_type == BACKWARD_WEIGHT
            and action.stage_index == next_action.stage_index
            and action.microbatch_index == next_action.microbatch_index
        ):
            merged_actions.append(
                _Action(action.stage_index, FULL_BACKWARD, action.microbatch_index)
            )
            compute_actions.pop(0)
        else:
            merged_actions.append(action)
    return merged_actions


def _add_send_recv(
    compute_actions: dict[int, list[_Action]],
    stage_to_rank: Callable[[int], int],
    num_stages: int,
) -> dict[int, list[_Action]]:
    """
    Transforms a compute-only schedule into a complete schedule with communication actions.

    For actions with sub-actions (OVERLAP_F_B) we ensure that all the subactions have been
    computed and the communication is ready
    """
    comm_actions: dict[int, list[_Action]] = {rank: [] for rank in compute_actions}
    prev_actions: dict[int, set[_Action]] = {rank: set() for rank in compute_actions}

    def _has_comms(action: _Action) -> bool:
        if action.computation_type == F:
            return action.stage_index != num_stages - 1 and stage_to_rank(
                action.stage_index + 1
            ) != stage_to_rank(action.stage_index)
        elif action.computation_type in (BACKWARD_INPUT, FULL_BACKWARD):
            return action.stage_index != 0 and stage_to_rank(
                action.stage_index - 1
            ) != stage_to_rank(action.stage_index)
        return False

    def _get_comms(action: _Action) -> tuple[_Action, _Action]:
        if not _has_comms(action):
            raise AssertionError(f"{action} is not a valid comm action")
        stage_idx = action.stage_index
        ctype = action.computation_type
        mb_idx = action.microbatch_index
        send = _Action(stage_idx, SEND_F if ctype == F else SEND_B, mb_idx)
        recv_stage_idx = stage_idx + 1 if ctype == F else stage_idx - 1
        recv = _Action(recv_stage_idx, RECV_F if ctype == F else RECV_B, mb_idx)
        return send, recv

    def _ready_to_schedule(action: _Action | None, prev_actions: set[_Action]) -> bool:
        """We don't put our own recv ops in the schedule, we let a sender on another rank put our recv ops in place.
        This helps ensure a sane (non-hanging) ordering of sends and recvs.
        But it also means we might not be able to schedule our next compute action yet.
        """
        if action is None:
            return True
        elif action.computation_type == F and action.stage_index != 0:
            if (
                _Action(action.stage_index, RECV_F, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index - 1, F, action.microbatch_index)
                in prev_actions
            ):
                return True
            return False
        elif (
            action.computation_type in (BACKWARD_INPUT, FULL_BACKWARD)
            and action.stage_index != num_stages - 1
        ):
            if (
                _Action(action.stage_index, RECV_B, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index + 1, BACKWARD_INPUT, action.microbatch_index)
                in prev_actions
            ):
                return True
            elif (
                _Action(action.stage_index + 1, FULL_BACKWARD, action.microbatch_index)
                in prev_actions
            ):
                return True
            return False
        else:
            return True

    while compute_actions:
        progress = False
        # go in order of ranks even if dict keys aren't ordered
        for rank in sorted(compute_actions):
            if not (len(compute_actions[rank]) > 0):
                raise AssertionError(f"{rank=}, {len(compute_actions[rank])=}")
            action = compute_actions[rank][0]
            # handle case where parent action (e.g. OVERLAP_F_B) can be comprised of subactions
            if action is not None and action.sub_actions is not None:
                all_actions = action.sub_actions
            else:
                all_actions = (action,)

            if not all(_ready_to_schedule(a, prev_actions[rank]) for a in all_actions):
                continue

            # The action's dependencies are satisfied, so add to schedule
            if action is not None:
                comm_actions[rank].append(action)
                for a in all_actions:
                    prev_actions[rank].add(a)
                    if _has_comms(a):
                        send, recv = _get_comms(a)
                        # TODO we can avoid send/recv if the 2 stages are on the same rank.
                        # should we avoid that in the runtime or here?
                        comm_actions[rank].append(send)
                        prev_actions[rank].add(send)
                        comm_actions[stage_to_rank(recv.stage_index)].append(recv)
                        prev_actions[stage_to_rank(recv.stage_index)].add(recv)

            compute_actions[rank].pop(0)
            if len(compute_actions[rank]) == 0:
                del compute_actions[rank]
            progress = True
        if not progress:
            raise AssertionError(
                "Malformed compute schedule, can't schedule sends/recvs"
            )
    return comm_actions


def _defer_recv_ops(
    actions: dict[int, list[_Action]],
    stage_to_rank: Callable[[int], int],
) -> dict[int, list[_Action]]:
    """
    Defers RECV operations to reduce interference with unrelated compute ops,
    while maintaining deadlock-safe ordering via rank-parity P2P ordering.

    By default, the schedule places RECV ops as early as possible (ASAP) to overlap
    P2P communication with computation. However, on some platforms (e.g., AMD ROCm),
    a pending RECV can block unrelated compute ops that also use the communication
    fabric (e.g., FSDP allgather inside a forward pass), creating pipeline bubbles.

    This function defers each RECV to as late as possible, subject to:
      1. A RECV must appear before the compute op that consumes its data.
      2. Deadlock avoidance via rank-parity ordering (see pytorch/pytorch#172668):
         - When current rank > peer rank: SEND before RECV is safe, so deferred
           RECVs from that peer are NOT flushed before SENDs to that peer.
         - When current rank < peer rank: RECV before SEND is required, so
           deferred RECVs from that peer ARE flushed before SENDs to that peer.

    This breaks circular waits: the lower-ranked side always posts RECV first,
    providing the matching target for the higher-ranked side's SEND.
    """
    RECV_F = _ComputationType.RECV_F
    RECV_B = _ComputationType.RECV_B
    SEND_F = _ComputationType.SEND_F
    SEND_B = _ComputationType.SEND_B

    def _recv_peer_rank(action: _Action) -> int:
        if action.computation_type == RECV_F:
            return stage_to_rank(action.stage_index - 1)
        else:
            return stage_to_rank(action.stage_index + 1)

    def _send_peer_rank(action: _Action) -> int:
        if action.computation_type == SEND_F:
            return stage_to_rank(action.stage_index + 1)
        else:
            return stage_to_rank(action.stage_index - 1)

    result: dict[int, list[_Action]] = {}
    for rank, action_list in actions.items():
        new_actions: list[_Action] = []
        deferred: dict[tuple[int, _ComputationType, int | None], _Action] = {}

        for action in action_list:
            if action.computation_type in (RECV_F, RECV_B):
                key = (
                    action.stage_index,
                    action.computation_type,
                    action.microbatch_index,
                )
                deferred[key] = action
                continue

            if action.computation_type in (SEND_F, SEND_B):
                peer = _send_peer_rank(action)
                # Rank-parity rule: only flush RECVs when rank < peer
                # (lower rank does RECV before SEND)
                if rank < peer:
                    to_flush = [
                        k for k, v in deferred.items() if _recv_peer_rank(v) == peer
                    ]
                    for key in to_flush:
                        new_actions.append(deferred.pop(key))

            # Constraint 1: before a compute op, flush the RECV it consumes
            consumers = (
                action.sub_actions if action.sub_actions is not None else (action,)
            )
            for sub in consumers:
                if sub.computation_type == FORWARD:
                    key = (sub.stage_index, RECV_F, sub.microbatch_index)
                    if key in deferred:
                        new_actions.append(deferred.pop(key))
                elif sub.computation_type in (FULL_BACKWARD, BACKWARD_INPUT):
                    key = (sub.stage_index, RECV_B, sub.microbatch_index)
                    if key in deferred:
                        new_actions.append(deferred.pop(key))

            new_actions.append(action)

        if deferred:
            raise AssertionError(
                f"Malformed input schedule on rank {rank}: leftover RECV ops "
                f"with no consumer found: {list(deferred.values())}. "
                "Every RECV must be consumed by a downstream compute op."
            )

        result[rank] = new_actions
    return result


def _validate_schedule(
    actions: dict[int, list[_Action | None]],
    pp_group_size: int,
    num_stages: int,
    num_microbatches: int,
) -> dict[int, int]:
    if not (len(actions) == pp_group_size):
        raise AssertionError(
            f"Schedule has incorrect number of ranks - expected {pp_group_size}, actual {len(actions)}"
        )
    for rank in range(pp_group_size):
        if rank not in actions:
            raise AssertionError(f"Schedule is missing actions for rank {rank}")

    # We will count all the actions per stage and ensure they happen in a valid order
    # (e.g. F before (B, I) before W for a given microbatch)
    stage_actions: dict[int, dict[_ComputationType, set]] = {
        stage_id: {
            F: set(),
            B: set(),
            I: set(),
            W: set(),
        }
        for stage_id in range(num_stages)
    }
    stage_index_to_rank_mapping = {}

    def _process_action(action: _Action, rank: int, step: int):
        """Process a single action and update stage_actions and stage_index_to_rank_mapping"""
        s_id = action.stage_index
        ctype = action.computation_type
        mb_id = action.microbatch_index

        if ctype == F:
            stage_actions[s_id][F].add(mb_id)
        elif ctype == B:
            if mb_id not in stage_actions[s_id][F]:
                error_msg = (
                    f"Rank {rank}, step {step}: Running Full Backward for stage {s_id}, "
                    f"microbatch {mb_id} without first running Forward"
                )
                formatted_schedule = _format_pipeline_order(
                    actions, error_step_number=step
                )
                full_error_msg = (
                    f"{error_msg}\n\nFull pipeline schedule:\n{formatted_schedule}"
                )
                raise AssertionError(full_error_msg)
            stage_actions[s_id][B].add(mb_id)
        elif ctype == I:
            if mb_id not in stage_actions[s_id][F]:
                error_msg = (
                    f"Rank {rank}, step {step}: Running Backward Input for stage {s_id}, "
                    f"microbatch {mb_id} without first running Forward"
                )
                formatted_schedule = _format_pipeline_order(
                    actions, error_step_number=step
                )
                full_error_msg = (
                    f"{error_msg}\n\nFull pipeline schedule:\n{formatted_schedule}"
                )
                raise AssertionError(full_error_msg)
            stage_actions[s_id][I].add(mb_id)
        elif ctype == W:
            if mb_id not in stage_actions[s_id][I]:
                error_msg = (
                    f"Rank {rank}, step {step}: Running Backward Weight for stage {s_id}, "
                    f"microbatch {mb_id} without first running Backward Input"
                )
                formatted_schedule = _format_pipeline_order(
                    actions, error_step_number=step
                )
                full_error_msg = (
                    f"{error_msg}\n\nFull pipeline schedule:\n{formatted_schedule}"
                )
                raise AssertionError(full_error_msg)
            stage_actions[s_id][W].add(mb_id)

        if s_id not in stage_index_to_rank_mapping:
            stage_index_to_rank_mapping[s_id] = rank
        else:
            existing_rank = stage_index_to_rank_mapping[s_id]
            if not (rank == existing_rank):
                raise AssertionError(
                    f"Rank {rank}, step {step}: Stage {s_id} is assigned to both rank {rank} and rank {existing_rank}"
                )

    for rank in actions:
        for step, action in enumerate(actions[rank]):
            if action is None:
                continue
            if not isinstance(action, _Action):
                raise AssertionError(
                    f"Rank {rank}, step {step}: Got an invalid action: {action}, expected instance of _Action"
                )

            # Check if action has sub_actions
            if action.sub_actions is not None:
                # Process each sub_action instead of the main action
                for sub_action in action.sub_actions:
                    _process_action(sub_action, rank, step)
            else:
                # Process the main action normally
                _process_action(action, rank, step)

    for s_id in stage_actions:
        f_mb = len(stage_actions[s_id][F])
        b_mb = len(stage_actions[s_id][B])
        i_mb = len(stage_actions[s_id][I])
        w_mb = len(stage_actions[s_id][W])

        if not (f_mb == num_microbatches):
            raise AssertionError(
                f"Got {f_mb} {F} microbatches for stage {s_id}, expected {num_microbatches}"
            )

        if not (i_mb == w_mb):
            raise AssertionError(
                f"Invalid backward microbatches for stage {s_id}: I and W must have equal counts, \
            but got I={i_mb}, W={w_mb}"
            )

        if not (b_mb + (i_mb + w_mb) // 2 == num_microbatches):
            raise AssertionError(
                f"Invalid backward microbatches for stage {s_id}: expected {num_microbatches} total backwards, \
            but got B={b_mb}, I={i_mb}, W={w_mb}"
            )
    return stage_index_to_rank_mapping


class PipelineScheduleMulti(_PipelineSchedule):
    """
    Base class for multi-stage schedules.
    Implements the `step` method.

    Gradients are scaled by num_microbatches depending on the `scale_grads` argument, defaulting to True.  This setting
    should match the configuration of your loss_fn, which may either average losses (scale_grads=True)
    or sum losses (scale_grads=False).
    """

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
        kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
        output_merge_spec: dict[str, Any] | tuple[Any] | None = None,
        use_full_backward: bool | None = None,
        scale_grads: bool = True,
        backward_requires_autograd: bool = True,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        # Self attributes
        self._stages = stages
        self._num_stages = stages[0].num_stages
        self.pp_group_size = stages[0].group_size
        self.rank = stages[0].group_rank
        # Set the pipeline stage states
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        self._stages_forward_initialized = False
        self._stages_backward_initialized = False

        # avoid putting a reference to 'self' inside the lambda, it creates a ref cycle
        has_loss: bool = self._loss_fn is not None
        self._should_compute_loss = lambda stage: stage.is_last and has_loss

        # This will be set during init of derived schedules
        self.pipeline_order: dict[int, list[_Action | None]] = {}

        # When using a custom backward function, we may or may not need autograd to be used
        # for the backward pass. This flag is used to determine whether or torch.is_grad_enabled()
        # check should be performed before the step function.
        self._backward_requires_autograd = backward_requires_autograd

        if use_full_backward is not None:
            logger.warning(
                "Deprecation warning: 'use_full_backward' is no longer supported. "
                "Simply stop passing it, and everything should still work fine."
            )

    def _initialize_stages(
        self, args: tuple[Any, ...], kwargs, target=None, loss_kwargs=None
    ):
        reinit_for_mode_switch = self._stages_forward_initialized and (
            self._has_backward != self._stages_backward_initialized
        )
        forward_initialized_before = self._stages_forward_initialized
        (
            self._stages_forward_initialized,
            self._stages_backward_initialized,
        ) = self._initialize_pp_stages(
            self._stages,
            args,
            kwargs,
            target,
            self._stages_forward_initialized,
            self._stages_backward_initialized,
            loss_kwargs=loss_kwargs,
        )

        if self._stages_forward_initialized and (
            not forward_initialized_before or reinit_for_mode_switch
        ):
            self._validate_adjacent_stage_communication()

    def _validate_adjacent_stage_communication(self) -> None:
        """Validate that stage communication follows adjacent-stage topology only."""

        def _check_stage_indices(
            stage_idx: int,
            direction: str,
            actual_stage_indices: set[int],
            expected_stage_indices: set[int],
        ) -> None:
            non_adjacent_stage_indices = actual_stage_indices - expected_stage_indices
            if non_adjacent_stage_indices:
                raise RuntimeError(
                    "PipelineScheduleMulti only supports adjacent-stage "
                    f"communication, but stage {stage_idx} has {direction} "
                    f"stages {sorted(actual_stage_indices)} with "
                    f"non-adjacent stages {sorted(non_adjacent_stage_indices)} "
                    f"(allowed adjacent stages: "
                    f"{sorted(expected_stage_indices)}). This commonly "
                    "indicates skip connections, which are unsupported in "
                    "this schedule runtime."
                )

        for stage in self._stages:
            stage_idx = stage.stage_index
            actual_fwd_recv_sources: set[int] = {
                info.source
                for info in stage.args_recv_info[0]
                if isinstance(info, _RecvInfo) and info.source is not None
            }
            expected_fwd_recv_sources = set() if stage.is_first else {stage_idx - 1}
            _check_stage_indices(
                stage_idx,
                "forward recv",
                actual_fwd_recv_sources,
                expected_fwd_recv_sources,
            )

            # act_send_info is keyed by output index (not microbatch index),
            # so .values() yields per-output destination lists.
            actual_fwd_send_dests: set[int] = {
                dst
                for dsts in stage.act_send_info.values()
                for dst in dsts
                if dst is not None
            }
            expected_fwd_send_dests = set() if stage.is_last else {stage_idx + 1}
            _check_stage_indices(
                stage_idx,
                "forward send",
                actual_fwd_send_dests,
                expected_fwd_send_dests,
            )

    def _validate_and_set_stage_mapping(
        self, actions: dict[int, list[_Action | None]]
    ) -> None:
        """
        Allocates the stage index to rank mapping which is needed for communication
        """
        self.stage_index_to_group_rank = _validate_schedule(
            actions,
            self.pp_group_size,
            self._num_stages,
            self._n_microbatches,
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

    def _dump_csv(self, filename):
        """Dump a CSV representation of the schedule into a file with the provided filename."""
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for rank in self.pipeline_order:
                writer.writerow(self.pipeline_order[rank])

    def _load_csv(
        self,
        filename: str,
        format: Literal["compute_only", "compute_comms"] = "compute_only",
    ):
        """Load a CSV representation of the schedule from a file with the provided filename.
        This API will most likely get renamed/refactored so is marked as internal for now.

        format must be "compute_only" for PipelineScheduleMulti.
        """
        if format != "compute_only":
            raise AssertionError(f'format must be "compute_only", got {format}')
        with open(filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for rank, row in enumerate(reader):
                self.pipeline_order[rank] = [_Action.from_str(s) for s in row]

        # Validates the order of the pipeline actions and infers the stage_to_rank_mapping.
        # This will overwrite the default stage_to_rank_mapping created in the constructor
        self._validate_and_set_stage_mapping(self.pipeline_order)

    def step(
        self,
        *args,
        target=None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
        arg_mbs: Any = None,
        kwarg_mbs: Any = None,
        target_mbs: Any = None,
        **kwargs,
    ):
        r"""Run one training iteration of a multi-stage pipeline schedule.

        By default, ``args``, ``kwargs``, and ``target`` are full-batch values
        that the schedule splits into microbatches. When the caller has already
        split inputs into microbatches, pass them through ``arg_mbs``,
        ``kwarg_mbs``, and ``target_mbs`` instead.

        Args:
            \*args (Any): Whole-batch positional root inputs when this rank owns
                the first pipeline stage. Do not pass positional inputs with
                pre-split inputs.
            target (Any, optional): Whole-batch target for loss computation.
                When passing pre-split inputs, pass targets through
                ``target_mbs`` instead. Default: ``None``.
            losses (list, optional): Mutable list populated with one loss per
                microbatch when this schedule owns the last stage and a
                ``loss_fn`` was configured. Default: ``None``.
            return_outputs (bool, optional): Whether to merge and return output
                chunks on the last stage. Default: ``True``.
            loss_kwargs (dict, optional): Extra keyword arguments forwarded to
                the configured ``loss_fn``. Default: ``None``.
            arg_mbs (list[tuple], optional): Pre-split positional inputs, one
                tuple per microbatch. Default: ``None``.
            kwarg_mbs (list[dict], optional): Pre-split keyword inputs, one
                dict per microbatch. Default: ``None``.
            target_mbs (list, optional): Pre-split targets, one entry per
                microbatch. Default: ``None``.
            \*\*kwargs (Any): Whole-batch keyword root inputs when this rank owns
                the first pipeline stage. Do not pass keyword inputs with
                pre-split inputs.

        Returns:
            Any or None: The merged output from the last stage when this rank
                owns the last stage and ``return_outputs=True``; otherwise
                ``None``.

        Raises:
            RuntimeError: If backward computation is enabled and ``step`` is
                called under ``torch.no_grad()``. Use :meth:`eval` for
                forward-only execution.
            TypeError: If a pre-split microbatch container has the wrong type.
            ValueError: If whole-batch and pre-split inputs are mixed, or if a
                pre-split container does not have one entry per microbatch.

        Examples::

            >>> # xdoctest: +SKIP("requires a constructed distributed pipeline schedule")
            >>> output = schedule.step(x, mask=mask, target=target)
            >>> arg_mbs = [(x0,), (x1,)]
            >>> kwarg_mbs = [{"mask": mask0}, {"mask": mask1}]
            >>> target_mbs = [target0, target1]
            >>> output = schedule.step(
            ...     arg_mbs=arg_mbs,
            ...     kwarg_mbs=kwarg_mbs,
            ...     target_mbs=target_mbs,
            ... )
        """
        if (
            self._has_backward
            and self._backward_requires_autograd
            and not torch.is_grad_enabled()
        ):
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation; "
                "it should not be used under torch.no_grad() context. "
                "Please call eval() instead."
            )

        # Set the same has_backward flag for stage object
        for stage in self._stages:
            stage.has_backward = self._has_backward

        # Clean per iteration
        for stage in self._stages:
            stage.clear_runtime_states()

        args_split, kwargs_split, targets_split = self._get_microbatch_inputs(
            args,
            kwargs,
            target,
            arg_mbs,
            kwarg_mbs,
            target_mbs,
        )

        # Run microbatches
        self._step_microbatches(
            args_split,
            kwargs_split,
            targets_split,
            losses,
            return_outputs,
            loss_kwargs=loss_kwargs,
        )

        # Return merged results per original format
        for stage in self._stages:
            if stage.is_last and return_outputs:
                return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage or we do not return output chunks
        return None

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ):
        """
        Operate on the microbatches for looped schedules (multiple stages on each rank).

        TODO: Does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        maybe_first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stages(
            arg_mbs[0], kwarg_mbs[0], maybe_first_target, loss_kwargs
        )

        # Based on the plan in Step 1 created in __init__:
        # 2. Perform communication based on the pipeline_order
        stage_index_to_stage: dict[int, _PipelineStageBase] = {
            stage.stage_index: stage for stage in self._stages
        }

        # determine prev_rank and next_rank based on which ranks are next to
        # the stages in the pipeline_order
        all_prev_ranks: set[int] = set()
        all_next_ranks: set[int] = set()
        for stage_index in stage_index_to_stage:
            # TODO: assumption that stages only communicate from distances of +1/-1 (no skip connections)
            if stage_index > 0:
                all_prev_ranks.add(self.stage_index_to_group_rank[stage_index - 1])
            if stage_index < self._num_stages - 1:
                all_next_ranks.add(self.stage_index_to_group_rank[stage_index + 1])
        # count either full_backward or backward_weight together, to determine when to sync DP grads
        backward_counter: Counter[int] = Counter()
        for time_step, action in enumerate(self.pipeline_order[self.rank]):
            try:
                ops: list[dist.P2POp] = []
                if action is not None:
                    computation_type = action.computation_type
                    mb_index = action.microbatch_index
                    stage_index = action.stage_index
                    if mb_index is None:
                        raise AssertionError(
                            "All currently supported action types require valid microbatch_index"
                        )
                    if computation_type == _ComputationType.FORWARD:
                        # perform forward computation
                        stage = stage_index_to_stage[stage_index]
                        output = stage.forward_one_chunk(
                            mb_index,
                            arg_mbs[mb_index],
                            kwarg_mbs[mb_index],
                            save_forward_output=return_outputs,
                        )
                        self._maybe_compute_loss(
                            stage, output, target_mbs, mb_index, loss_kwargs
                        )
                        ops.extend(stage.get_fwd_send_ops(mb_index))
                    elif computation_type == _ComputationType.FULL_BACKWARD:
                        # perform backward computation
                        stage = stage_index_to_stage[stage_index]
                        loss = self._maybe_get_loss(stage, mb_index)
                        backward_counter[stage_index] += 1
                        last_backward = (
                            backward_counter[stage_index] == self._n_microbatches
                        )
                        grad_scale_factor = (
                            self._n_microbatches if self.scale_grads else 1
                        )
                        stage.backward_one_chunk(
                            mb_index,
                            loss=loss,
                            full_backward=True,
                            last_backward=last_backward,
                        )
                        if last_backward:
                            stage.scale_grads(grad_scale_factor)

                        ops.extend(stage.get_bwd_send_ops(mb_index))
                    elif computation_type == _ComputationType.BACKWARD_INPUT:
                        # perform backward computation
                        stage = stage_index_to_stage[stage_index]
                        loss = self._maybe_get_loss(stage, mb_index)
                        stage.backward_one_chunk(
                            mb_index,
                            loss=loss,
                            full_backward=False,
                            last_backward=False,
                        )
                        ops.extend(stage.get_bwd_send_ops(mb_index))
                    elif computation_type == _ComputationType.BACKWARD_WEIGHT:
                        # perform weight update
                        stage = stage_index_to_stage[stage_index]
                        backward_counter[stage_index] += 1
                        last_backward = (
                            backward_counter[stage_index] == self._n_microbatches
                        )
                        grad_scale_factor = (
                            self._n_microbatches if self.scale_grads else 1
                        )
                        stage.backward_weight_one_chunk(
                            mb_index,
                            last_backward=last_backward,
                        )
                        if last_backward:
                            stage.scale_grads(grad_scale_factor)
                    else:
                        raise ValueError(f"Unknown computation type {computation_type}")

                # Look at the neighboring ranks for this current timestep and determine whether
                # this current rank needs to do any recv communication
                for prev_rank in all_prev_ranks:
                    prev_rank_ops = self.pipeline_order[prev_rank]
                    prev_rank_action = None
                    if time_step < len(prev_rank_ops):
                        prev_rank_action = prev_rank_ops[time_step]
                    if prev_rank_action is not None:
                        computation_type = prev_rank_action.computation_type
                        mb_index = prev_rank_action.microbatch_index
                        stage_index = prev_rank_action.stage_index
                        if mb_index is None:
                            raise AssertionError(
                                "All currently supported action types require valid microbatch_index"
                            )
                        # Only handle sends for the forward from a previous rank
                        if computation_type == _ComputationType.FORWARD:
                            # If not the last stage, then receive fwd activations
                            if stage_index + 1 in stage_index_to_stage:
                                # TODO: We are assuming that stage will always receive from stage-1
                                # however that is not necessarily true of get_fwd_recv_ops
                                stage = stage_index_to_stage[stage_index + 1]
                                ops.extend(stage.get_fwd_recv_ops(mb_index))
                        elif computation_type in (
                            FULL_BACKWARD,
                            BACKWARD_INPUT,
                            BACKWARD_WEIGHT,
                        ):
                            # Previous rank doing backward has no influence for the current rank forward recv
                            pass
                        else:
                            raise ValueError(
                                f"Unknown computation type {computation_type}"
                            )
                for next_rank in all_next_ranks:
                    next_rank_ops = self.pipeline_order[next_rank]
                    next_rank_action = None
                    if time_step < len(next_rank_ops):
                        next_rank_action = next_rank_ops[time_step]
                    if next_rank_action is not None:
                        computation_type = next_rank_action.computation_type
                        mb_index = next_rank_action.microbatch_index
                        stage_index = next_rank_action.stage_index
                        if not (mb_index is not None):
                            raise AssertionError(
                                "All currently supported action types require valid microbatch_index"
                            )
                        # Only handle receives for the backwards from a next rank
                        if computation_type in (FORWARD, BACKWARD_WEIGHT):
                            # Next rank doing forward or weight update has no influence for the current rank backward recv
                            pass
                        elif computation_type in (BACKWARD_INPUT, FULL_BACKWARD):
                            # If not the first stage, then receive bwd gradients
                            if stage_index - 1 in stage_index_to_stage:
                                # TODO: We are assuming that stage will always receive from stage+1
                                # however that is not necessarily true of get_bwd_recv_ops
                                stage = stage_index_to_stage[stage_index - 1]
                                ops.extend(stage.get_bwd_recv_ops(mb_index))
                        else:
                            raise ValueError(
                                f"Unknown computation type {computation_type}"
                            )

                # do the communication
                _wait_batch_p2p(_batch_p2p(ops))
            except Exception as e:
                logger.error(
                    "[Rank %s] pipeline schedule %s caught the following exception '%s' \
at time_step %s when running action %s",
                    self.rank,
                    self.__class__.__name__,
                    e,
                    time_step,
                    action,
                )
                logger.error(
                    "%s",
                    _format_pipeline_order(
                        self.pipeline_order, error_step_number=time_step
                    ),
                )
                raise e
        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)


@dataclass
class _PipelineContext:
    """Context passed to custom functions during pipeline execution."""

    schedule_ref: _PipelineSchedule
    arg_mbs: list[tuple] | None = None
    kwarg_mbs: list[dict] | None = None
    target_mbs: list | None = None
    losses: list | None = None


class _CustomFunctionProtocol(Protocol):
    def __call__(self, action: _Action, ctx: _PipelineContext) -> None: ...


class _PipelineScheduleRuntime(PipelineScheduleMulti):
    """
    Provides a simple runtime that requires a 'schedule IR' including specified communication operations.

    Can be instantiated directly by creating _PipelineScheduleRuntime and calling load_csv, or can be
    subclassed and the subclass can be responsible for creating a schedule IR.
    """

    def __init__(self, *args, **kwargs):
        self._defer_pp_recv: bool = kwargs.pop("defer_pp_recv", False)
        super().__init__(*args, **kwargs)
        # Action to custom function mapping
        self._comp_type_to_function_map: dict[_ComputationType, Callable] = {}
        # count either full_backward or backward_weight together, to determine when to sync DP grads
        self.backward_counter: Counter[int] = Counter()

        # recv ops indexed by (stage_idx, mb_idx) need to be waited on before use
        self.bwd_recv_ops: dict[tuple[int, int], list[dist.Work]] = {}
        self.fwd_recv_ops: dict[tuple[int, int], list[dist.Work]] = {}

        # we track which stages are 'active' when used with FSDP, and wait on unshard ops before computing on stages
        self.unshard_ops: dict[int, list[UnshardHandle]] = defaultdict(list)
        self.unsharded_stages = set()

    def register_custom_function(
        self,
        computation_type: _ComputationType,
        custom_function: _CustomFunctionProtocol,
    ) -> None:
        """
        Register a custom function to be executed for a specific computation type.

        Args:
            computation_type: The computation type for which to register the custom function
            custom_function: The function to execute when this computation type is encountered.
                Must have signature: (action: _Action, ctx: _PipelineContext) -> None
        """
        # Ensure that the computation type is valid
        if computation_type not in (
            FORWARD,
            FULL_BACKWARD,
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            OVERLAP_F_B,
            UNSHARD,
            RESHARD,
            REDUCE_GRAD,
        ):
            raise ValueError(
                f"Invalid computation type {computation_type}. Only FORWARD, FULL_BACKWARD, \
                BACKWARD_INPUT, BACKWARD_WEIGHT, OVERLAP_F_B, UNSHARD, RESHARD and REDUCE_GRAD are supported."
            )

        # Check if computation_type is already registered
        if computation_type in self._comp_type_to_function_map:
            logger.warning(
                "Computation type %s is already registered. "
                "Overwriting the existing custom function.",
                computation_type,
            )

        self._comp_type_to_function_map[computation_type] = custom_function

    def _prepare_schedule_with_comms(
        self,
        actions: dict[int, list[_Action | None]],
        format: Literal["compute_only", "compute_comms"] = "compute_only",
    ):
        """
        Given an in-memory representation for a simple compute-only schedule, lower it to a complex schedule including
        communication actions.  Stores the schedule in self, and must be called before running step_mo()
        """
        # validate the provided actions are valid and overrides the default stage_index_to_group_rank
        super()._validate_and_set_stage_mapping(actions)

        self.pipeline_order_with_comms: dict[int, list[_Action]] = {}
        if format == "compute_comms":
            for rank in actions:
                self.pipeline_order_with_comms[rank] = []
                for action in actions[rank]:
                    if action is None:
                        raise AssertionError(
                            f"Expected action to be not None, got {type(action)}"
                        )
                    self.pipeline_order_with_comms[rank].append(action)
            # TODO what level of validation should we offer for compute+comms schedule?
        elif format == "compute_only":
            # Validate that the schedule does not have comms already added to it
            for rank, action_list in actions.items():
                for i, action in enumerate(action_list):
                    if action is not None:
                        if not action.is_compute_op:
                            raise ValueError(
                                f"Expected compute-only schedule but found communication action "
                                f"'{action}' at rank {rank}, position {i}. "
                                f"Communication actions (e.g. SEND_F, RECV_F, etc.) "
                                f"should not be present when format='compute_only'."
                            )

            # Perform schedule lowering
            for rank in actions:
                self.pipeline_order_with_comms[rank] = _add_unshard_reshard(
                    actions[rank]
                )
                self.pipeline_order_with_comms[rank] = _add_reduce_grad(  # type: ignore[assignment]
                    self.pipeline_order_with_comms[rank],  # type: ignore[arg-type]
                    self._n_microbatches,
                )

            self.pipeline_order_with_comms = _add_send_recv(
                self.pipeline_order_with_comms,
                stage_to_rank=lambda s: self.stage_index_to_group_rank[s],
                num_stages=self._num_stages,
            )

            if self._defer_pp_recv:
                self.pipeline_order_with_comms = _defer_recv_ops(
                    self.pipeline_order_with_comms,
                    stage_to_rank=lambda s: self.stage_index_to_group_rank[s],
                )
        else:
            raise NotImplementedError(f"{format=} is not implemented")

    def _load_csv(
        self,
        filename: str,
        format: Literal["compute_only", "compute_comms"] = "compute_only",
    ):
        """Loads a csv in simple format and then lowers it to include communication actions

        format must be either "compute_only" or "compute_comms".  If compute_only, the lowering passes
        will automatically be run to generate a compute_comms schedule.
        """
        if format == "compute_only":
            # this will populate self.pipeline_order
            super()._load_csv(filename)
            # this will populate self.pipeline_order_with_comms
            self._prepare_schedule_with_comms(self.pipeline_order)
        elif format == "compute_comms":
            actions = {}
            with open(filename, newline="") as csvfile:
                reader = csv.reader(csvfile)
                for rank, row in enumerate(reader):
                    actions[rank] = [_Action.from_str(s) for s in row]
                self._prepare_schedule_with_comms(actions, format=format)
        else:
            raise NotImplementedError(f"{format=} is not implemented")

    def _dump_csv(
        self,
        filename: str,
        format: Literal["compute_only", "compute_comms"] = "compute_comms",
    ):
        """Dump a CSV representation of the schedule into a file with the provided filename."""
        if format == "compute_only":
            if self.pipeline_order is None:
                raise AssertionError("Compute only schedule must be available")
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for rank in self.pipeline_order:
                    writer.writerow(self.pipeline_order[rank])
        elif format == "compute_comms":
            if self.pipeline_order_with_comms is None:
                raise AssertionError(
                    "Must initialize compute_comms schedule before dump_csv"
                )
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for rank in self.pipeline_order_with_comms:
                    writer.writerow(self.pipeline_order_with_comms[rank])

    def _simulate(self):
        return _simulate_comms_compute(
            self.pipeline_order_with_comms,
            lambda s: self.stage_index_to_group_rank[s],
            self._num_stages,
        )

    def _assert_unsharded(self, stage: _PipelineStageBase):
        """If an unshard is active for `stage_idx`, wait() it and mark `stage_idx` unshared."""
        stage_uses_fsdp = isinstance(stage.submod, FSDPModule)
        if stage_uses_fsdp:
            stage_idx = stage.stage_index
            if stage_idx in self.unshard_ops:
                for op in self.unshard_ops[stage_idx]:
                    op.wait()
                del self.unshard_ops[stage_idx]
                self.unsharded_stages.add(stage_idx)
            if stage_idx not in self.unsharded_stages:
                raise AssertionError(f"Attempted to compute on sharded {stage_idx=}")

    def _step_microbatches(
        self,
        arg_mbs: list | None = None,
        kwarg_mbs: list | None = None,
        target_mbs: list | None = None,
        losses: list | None = None,
        return_outputs: bool = True,
        loss_kwargs: dict[str, Any] | None = None,
    ):
        """
        Operate on the microbatches for looped schedules (multiple stages on each rank).

        TODO: Does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)
        maybe_first_target = target_mbs[0] if target_mbs is not None else None
        self._initialize_stages(
            arg_mbs[0], kwarg_mbs[0], maybe_first_target, loss_kwargs
        )

        # Based on the plan in Step 1 created in __init__:
        # 2. Perform communication based on the pipeline_order
        stage_index_to_stage: dict[int, _PipelineStageBase] = {
            stage.stage_index: stage for stage in self._stages
        }

        if self.pipeline_order_with_comms is None:
            raise AssertionError(
                "Must call _prepare_schedule_with_comms() before calling _step_microbatches()"
            )

        # send ops should be waited on before step() exists, mainly for hygiene
        send_ops: list[list[dist.Work]] = []

        def _perform_action(action: _Action) -> None:
            comp_type = action.computation_type
            mb_index: int = (
                action.microbatch_index if action.microbatch_index is not None else -1
            )
            if not (
                mb_index >= 0
                or comp_type
                in (
                    UNSHARD,
                    RESHARD,
                    REDUCE_GRAD,
                )
            ):
                raise AssertionError(f"{action=} missing mb_index")
            stage_idx = action.stage_index
            stage = stage_index_to_stage[stage_idx]
            stage_uses_fsdp = isinstance(stage.submod, FSDPModule)
            # see [Note: V-schedule special case]
            is_next_stage_on_this_rank = stage_idx + 1 in stage_index_to_stage
            is_prev_stage_on_this_rank = stage_idx - 1 in stage_index_to_stage

            # TODO(whc) it's not actually safe to use _batch_p2p here in the uncommon case the model has skip-connections,
            # since we do not want to batch up ops between more than a pair of ranks.  _sorted_batch_p2p would be
            # safe to use instead.
            # However, I was wondering if I should avoid calling batched operators at all in the case that there is
            # only one operator per batch.  I could iterate through the 'fwd_send_ops' one by one and run them.
            if comp_type == SEND_F:
                send_ops.append(_batch_p2p(stage.get_fwd_send_ops(mb_index)))
            elif comp_type == SEND_B:
                send_ops.append(_batch_p2p(stage.get_bwd_send_ops(mb_index)))
            elif comp_type == RECV_F:
                if (stage_idx, mb_index) in self.fwd_recv_ops:
                    raise AssertionError(
                        f"Recv twice for {stage_idx=} {mb_index=} without executing forward"
                    )
                self.fwd_recv_ops[(stage_idx, mb_index)] = _batch_p2p(
                    stage.get_fwd_recv_ops(mb_index)
                )
            elif comp_type == RECV_B:
                if (stage_idx, mb_index) in self.bwd_recv_ops:
                    raise AssertionError(
                        f"Recv twice for {stage_idx=} {mb_index=} without executing backward"
                    )
                self.bwd_recv_ops[(stage_idx, mb_index)] = _batch_p2p(
                    stage.get_bwd_recv_ops(mb_index)
                )
            elif comp_type == UNSHARD:
                if stage_uses_fsdp:
                    if not (
                        stage_idx not in self.unsharded_stages
                        and stage_idx not in self.unshard_ops
                    ):
                        raise AssertionError(f"Unsharding the same {stage_idx=} twice")
                    for submodule in stage.submod.modules():
                        if not isinstance(submodule, FSDPModule):
                            continue
                        handle = cast(UnshardHandle, submodule.unshard(async_op=True))
                        self.unshard_ops[stage_idx].append(handle)
            elif comp_type == RESHARD:
                if stage_uses_fsdp:
                    if stage_idx not in self.unsharded_stages:
                        raise AssertionError(
                            f"Resharding {stage_idx=} without unsharding"
                        )
                    if stage_idx in self.unshard_ops:
                        raise AssertionError(
                            f"Resharding {stage_idx=} before finishing unshard"
                        )
                    for submodule in stage.submod.modules():
                        if not isinstance(submodule, FSDPModule):
                            continue
                        submodule.reshard()
                    self.unsharded_stages.remove(stage_idx)
            elif comp_type == FORWARD:
                self._assert_unsharded(stage)

                if (
                    not stage.is_first
                    # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                    and not is_prev_stage_on_this_rank
                ):
                    if (stage_idx, mb_index) not in self.fwd_recv_ops:
                        raise AssertionError(
                            f"Computing {action=} before receiving input"
                        )
                    _wait_batch_p2p(self.fwd_recv_ops.pop((stage_idx, mb_index)))

                output = stage.forward_one_chunk(
                    mb_index,
                    arg_mbs[mb_index],  # type: ignore[index]
                    kwarg_mbs[mb_index],  # type: ignore[index]
                    save_forward_output=return_outputs,
                )
                self._maybe_compute_loss(
                    stage, output, target_mbs, mb_index, loss_kwargs
                )

                # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
                # see [Note: V-schedule special case]
                if is_next_stage_on_this_rank:
                    stage_index_to_stage[stage_idx + 1].set_local_fwd_input(
                        output, mb_index
                    )

            elif comp_type == FULL_BACKWARD:
                self._assert_unsharded(stage)

                if (
                    not stage.is_last
                    # no recv op expected for V-schedule special case (see [Note: V-schedule special case])
                    and not is_next_stage_on_this_rank
                ):
                    if (stage_idx, mb_index) not in self.bwd_recv_ops:
                        raise AssertionError(
                            f"Attempted to run compute {action=} before receiving input"
                        )
                    _wait_batch_p2p(self.bwd_recv_ops.pop((stage_idx, mb_index)))
                loss = self._maybe_get_loss(stage, mb_index)
                self.backward_counter[stage_idx] += 1
                last_backward = self.backward_counter[stage_idx] == self._n_microbatches
                stage.backward_one_chunk(
                    mb_index,
                    loss=loss,
                    full_backward=True,
                    last_backward=last_backward,
                )
                # SEND/RECV op are avoided for special case with 2 adjacent stages on same rank
                # see [Note: V-schedule special case]
                if is_prev_stage_on_this_rank:
                    stage_index_to_stage[stage_idx - 1].set_local_bwd_input(
                        stage.get_local_bwd_output(mb_index), mb_index
                    )
            elif comp_type == BACKWARD_INPUT:
                self._assert_unsharded(stage)

                if not stage.is_last and not is_next_stage_on_this_rank:
                    if (stage_idx, mb_index) not in self.bwd_recv_ops:
                        raise AssertionError(
                            f"Attempted to run compute {action=} before receiving input"
                        )
                    _wait_batch_p2p(self.bwd_recv_ops.pop((stage_idx, mb_index)))
                loss = self._maybe_get_loss(stage, mb_index)
                stage.backward_one_chunk(
                    mb_index,
                    loss=loss,
                    full_backward=False,
                    last_backward=False,
                )
