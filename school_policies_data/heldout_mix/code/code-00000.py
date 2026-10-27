from .functions import defun_wrapped

@defun_wrapped
def squarew(ctx, t, amplitude=1, period=1):
    P = period
    A = amplitude
    return A*((-1)**ctx.floor(2*t/P))

@defun_wrapped
def trianglew(ctx, t, amplitude=1, period=1):
    A = amplitude
    P = period

    return 2*A*(0.5 - ctx.fabs(1 - 2*ctx.frac(t/P + 0.25)))

@defun_wrapped
def sawtoothw(ctx, t, amplitude=1, period=1):
    A = amplitude
    P = period
    return A*ctx.frac(t/P)

@defun_wrapped
def unit_triangle(ctx, t, amplitude=1):
    A = amplitude
    if t <= -1 or t >= 1:
        return ctx.zero
    return A*(-ctx.fabs(t) + 1)

@defun_wrapped
def sigmoid(ctx, t, amplitude=1):
    A = amplitude
    return A / (1 + ctx.exp(-t))
<|endoftext|>from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape, canonicalize_dtype


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    padding: str = "zero"
    round_f32_to_tf32: bool = False

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.strides) == rank, f"rank mismatch: {self}"
        assert len(self.block_shape) == rank, f"rank mismatch: {self}"
        assert rank > 0, "rank must not be zero"
        assert rank <= 5, "rank cannot be more than 5"
        ty = type(self.base)
        if ty.__name__ not in ("FakeTensor", "FunctionalTensor"):
            assert self.base.data_ptr() % 16 == 0, "base must be 16-byte aligned"
        validate_block_shape(self.block_shape)
        elem_bytes = self.base.dtype.itemsize
        for stride in self.strides[:-1]:
            assert (stride * elem_bytes) % 16 == 0, "strides must be 16-byte aligned"
        for shape_dim in self.shape:
            assert shape_dim > 0, "shape must be positive"
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert self.padding == "zero" or self.padding == "nan", "Illegal value for padding"
        if self.padding == "nan":
            assert self.base.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"
        if self.round_f32_to_tf32:
            dtype_name = canonicalize_dtype(self.base.dtype)
            assert dtype_name == "fp32", "round_f32_to_tf32 is only supported for float32 tensors"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], padding="zero", round_f32_to_tf32=False):
        return TensorDescriptor(tensor, tensor.shape, tensor.stride(), block_shape, padding, round_f32_to_tf32)
<|endoftext|>import datetime
import hashlib
import logging
import os
import time
import urllib.parse
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from . import constants
from .hf_api import whoami
from .utils import experimental, get_token


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import fastapi


@dataclass
class OAuthOrgInfo:
    """
    Information about an organization linked to a user logged in with OAuth.

    Attributes:
        sub (`str`):
            Unique identifier for the org. OpenID Connect field.
        name (`str`):
            The org's full name. OpenID Connect field.
        preferred_username (`str`):
            The org's username. OpenID Connect field.
        picture (`str`):
            The org's profile picture URL. OpenID Connect field.
        plan (`str`, *optional*):
            The org's plan (e.g., "enterprise", "team"). Hugging Face field.
        can_pay (`Optional[bool]`, *optional*):
            Whether the org has a payment method set up. Hugging Face field.
        role_in_org (`Optional[str]`, *optional*):
            The user's role in the org. Hugging Face field.
        security_restrictions (`Optional[list[Literal["ip", "token-policy", "mfa", "sso"]]]`, *optional*):
            Array of security restrictions that the user hasn't completed for this org. Possible values: "ip", "token-policy", "mfa", "sso". Hugging Face field.
    """

    sub: str
    name: str
    preferred_username: str
    picture: str
    plan: str | None = None
    can_pay: bool | None = None
    role_in_org: str | None = None
    security_restrictions: list[Literal["ip", "token-policy", "mfa", "sso"]] | None = None


@dataclass
class OAuthUserInfo:
    """
    Information about a user logged in with OAuth.

    Attributes:
        sub (`str`):
            Unique identifier for the user, even in case of rename. OpenID Connect field.
        name (`str`):
            The user's full name. OpenID Connect field.
        preferred_username (`str`):
            The user's username. OpenID Connect field.
        email_verified (`Optional[bool]`, *optional*):
            Indicates if the user's email is verified. OpenID Connect field.
        email (`Optional[str]`, *optional*):
            The user's email address. OpenID Connect field.
        picture (`str`):
            The user's profile picture URL. OpenID Connect field.
        profile (`str`):
            The user's profile URL. OpenID Connect field.
        website (`Optional[str]`, *optional*):
            The user's website URL. OpenID Connect field.
        is_pro (`bool`):
            Whether the user is a pro user. Hugging Face field.
        can_pay (`Optional[bool]`, *optional*):
            Whether the user has a payment method set up. Hugging Face field.
        orgs (`Optional[list[OrgInfo]]`, *optional*):
            List of organizations the user is part of. Hugging Face field.
    """

    sub: str
    name: str
    preferred_username: str
    email_verified: bool | None
    email: str | None
    picture: str
    profile: str
    website: str | None
    is_pro: bool
    can_pay: bool | None
    orgs: list[OAuthOrgInfo] | None


@dataclass
class OAuthInfo:
    """
    Information about the OAuth login.

    Attributes:
        access_token (`str`):
            The access token.
        access_token_expires_at (`datetime.datetime`):
            The expiration date of the access token.
        user_info ([`OAuthUserInfo`]):
            The user information.
        state (`str`, *optional*):
            State passed to the OAuth provider in the original request to the OAuth provider.
        scope (`str`):
            Granted scope.
    """

    access_token: str
    access_token_expires_at: datetime.datetime
    user_info: OAuthUserInfo
    state: str | None
    scope: str


@experimental
def attach_huggingface_oauth(app: "fastapi.FastAPI", route_prefix: str = "/"):
    """
    Add OAuth endpoints to a FastAPI app to enable OAuth login with Hugging Face.

    How to use:
    - Call this method on your FastAPI app to add the OAuth endpoints.
    - Inside your route handlers, call `parse_huggingface_oauth(request)` to retrieve the OAuth info.
    - If user is logged in, an [`OAuthInfo`] object is returned with the user's info. If not, `None` is returned.
    - In your app, make sure to add links to `/oauth/huggingface/login` and `/oauth/huggingface/logout` for the user to log in and out.

    Example:
    ```py
    from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth

    # Create a FastAPI app
    app = FastAPI()

    # Add OAuth endpoints to the FastAPI app
    attach_huggingface_oauth(app)

    # Add a route that greets the user if they are logged in
    @app.get("/")
    def greet_json(request: Request):
        # Retrieve the OAuth info from the request
        oauth_info = parse_huggingface_oauth(request)  # e.g. OAuthInfo dataclass
        if oauth_info is None:
            return {"msg": "Not logged in!"}
        return {"msg": f"Hello, {oauth_info.user_info.preferred_username}!"}
    ```
    """
    # TODO: handle generic case (handling OAuth in a non-Space environment with custom dev values) (low priority)

    # Add SessionMiddleware to the FastAPI app to store the OAuth info in the session.
    # Session Middleware requires a secret key to sign the cookies. Let's use a hash
    # of the OAuth secret key to make it unique to the Space + updated in case OAuth
    # config gets updated. When ran locally, we use an empty string as a secret key.
    try:
        from starlette.middleware.sessions import SessionMiddleware
    except ImportError as e:
        raise ImportError(
            "Cannot initialize OAuth to due a missing library. Please run `pip install huggingface_hub[oauth]` or add "
            "`huggingface_hub[oauth]` to your requirements.txt file in order to install the required dependencies."
        ) from e
    session_secret = (constants.OAUTH_CLIENT_SECRET or "") + "-v1"
    app.add_middleware(
        SessionMiddleware,  # type: ignore
        secret_key=hashlib.sha256(session_secret.encode()).hexdigest(),
        same_site="none",
        https_only=True,
    )  # type: ignore

    # Add OAuth endpoints to the FastAPI app:
    #   - {route_prefix}/oauth/huggingface/login
    #   - {route_prefix}/oauth/huggingface/callback
    #   - {route_prefix}/oauth/huggingface/logout
    # If the app is running in a Space, OAuth is enabled normally.
    # Otherwise, we mock the endpoints to make the user log in with a fake user profile - without any calls to hf.co.
    route_prefix = route_prefix.strip("/")
    if os.getenv("SPACE_ID") is not None:
        logger.info("OAuth is enabled in the Space. Adding OAuth routes.")
        _add_oauth_routes(app, route_prefix=route_prefix)
    else:
        logger.info("App is not running in a Space. Adding mocked OAuth routes.")
        _add_mocked_oauth_routes(app, route_prefix=route_prefix)


def parse_huggingface_oauth(request: "fastapi.Request") -> OAuthInfo | None:
    """
    Returns the information from a logged-in user as a [`OAuthInfo`] object.

    For flexibility and future-proofing, this method is very lax in its parsing and does not raise errors.
    Missing fields are set to `None` without a warning.

    Return `None`, if the user is not logged in (no info in session cookie).

    See [`attach_huggingface_oauth`] for an example on how to use this method.
    """
    if "oauth_info" not in request.session:
        logger.debug("No OAuth info in session.")
        return None

    logger.debug("Parsing OAuth info from session.")
    oauth_data = request.session["oauth_info"]
    user_data = oauth_data.get("userinfo", {})
    orgs_data = user_data.get("orgs", [])

    orgs = (
        [
            OAuthOrgInfo(
                sub=org.get("sub"),
                name=org.get("name"),
                preferred_username=org.get("preferred_username"),
                picture=org.get("picture"),
                plan=org.get("plan"),
                can_pay=org.get("canPay"),
                role_in_org=org.get("roleInOrg"),
                security_restrictions=org.get("securityRestrictions"),
            )
            for org in orgs_data
        ]
        if orgs_data
        else None
    )

    user_info = OAuthUserInfo(
        sub=user_data.get("sub"),
        name=user_data.get("name"),
        preferred_username=user_data.get("preferred_username"),
        email_verified=user_data.get("email_verified"),
        email=user_data.get("email"),
        picture=user_data.get("picture"),
        profile=user_data.get("profile"),
        website=user_data.get("website"),
        is_pro=user_data.get("isPro"),
        can_pay=user_data.get("canPay"),
        orgs=orgs,
    )

    return OAuthInfo(
        access_token=oauth_data.get("access_token"),
        access_token_expires_at=datetime.datetime.fromtimestamp(oauth_data.get("expires_at")),
        user_info=user_info,
        state=oauth_data.get("state"),
        scope=oauth_data.get("scope"),
    )


def _add_oauth_routes(app: "fastapi.FastAPI", route_prefix: str) -> None:
    """Add OAuth routes to the FastAPI app (login, callback handler and logout)."""
    try:
        import fastapi
        from authlib.integrations.base_client.errors import MismatchingStateError
        from authlib.integrations.starlette_client import OAuth
        from fastapi.responses import RedirectResponse
    except ImportError as e:
        raise ImportError(
            "Cannot initialize OAuth to due a missing library. Please run `pip install huggingface_hub[oauth]` or add "
            "`huggingface_hub[oauth]` to your requirements.txt file."
        ) from e

    # Check environment variables
    msg = (
        "OAuth is required but '{}' environment variable is not set. Make sure you've enabled OAuth in your Space by"
        " setting `hf_oauth: true` in the Space metadata."
    )
    if constants.OAUTH_CLIENT_ID is None:
        raise ValueError(msg.format("OAUTH_CLIENT_ID"))
    if constants.OAUTH_CLIENT_SECRET is None:
        raise ValueError(msg.format("OAUTH_CLIENT_SECRET"))
    if constants.OAUTH_SCOPES is None:
        raise ValueError(msg.format("OAUTH_SCOPES"))
    if constants.OPENID_PROVIDER_URL is None:
        raise ValueError(msg.format("OPENID_PROVIDER_URL"))

    # Register OAuth server
    oauth = OAuth()
    oauth.register(
        name="huggingface",
        client_id=constants.OAUTH_CLIENT_ID,
        client_secret=constants.OAUTH_CLIENT_SECRET,
        client_kwargs={"scope": constants.OAUTH_SCOPES},
        server_metadata_url=constants.OPENID_PROVIDER_URL + "/.well-known/openid-configuration",
    )

    login_uri, callback_uri, logout_uri = _get_oauth_uris(route_prefix)

    # Register OAuth endpoints
    @app.get(login_uri)
    async def oauth_login(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that redirects to HF OAuth page."""
        redirect_uri = _generate_redirect_uri(request)
        return await oauth.huggingface.authorize_redirect(request, redirect_uri)  # type: ignore

    @app.get(callback_uri)
    async def oauth_redirect_callback(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that handles the OAuth callback."""
        try:
            oauth_info = await oauth.huggingface.authorize_access_token(request)  # type: ignore
        except MismatchingStateError:
            # Parse query params
            nb_redirects = int(request.query_params.get("_nb_redirects", 0))
            target_url = request.query_params.get("_target_url")

            # Build redirect URI with the same query params as before and bump nb_redirects count
            query_params: dict[str, int | str] = {"_nb_redirects": nb_redirects + 1}
            if target_url:
                query_params["_target_url"] = target_url

            redirect_uri = f"{login_uri}?{urllib.parse.urlencode(query_params)}"

            # If the user is redirected more than 3 times, it is very likely that the cookie is not working properly.
            # (e.g. browser is blocking third-party cookies in iframe). In this case, redirect the user in the
            # non-iframe view.
            if nb_redirects > constants.OAUTH_MAX_REDIRECTS:
                host = os.environ.get("SPACE_HOST")
                if host is None:  # cannot happen in a Space
                    raise RuntimeError(
                        "App is not running in a Space (SPACE_HOST environment variable is not set). Cannot redirect to non-iframe view."
                    ) from None
                host_url = "https://" + host.rstrip("/")
                return RedirectResponse(host_url + redirect_uri)

            # Redirect the user to the login page again
            return RedirectResponse(redirect_uri)

        # OAuth login worked => store the user info in the session and redirect
        logger.debug("Successfully logged in with OAuth. Storing user info in session.")
        request.session["oauth_info"] = oauth_info
        return RedirectResponse(_get_redirect_target(request))

    @app.get(logout_uri)
    async def oauth_logout(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that logs out the user (e.g. delete info from cookie session)."""
        logger.debug("Logged out with OAuth. Removing user info from session.")
        request.session.pop("oauth_info", None)
        return RedirectResponse(_get_redirect_target(request))


def _add_mocked_oauth_routes(app: "fastapi.FastAPI", route_prefix: str = "/") -> None:
    """Add fake oauth routes if app is run locally and OAuth is enabled.

    Using OAuth will have the same behavior as in a Space but instead of authenticating with HF, a mocked user profile
    is added to the session.
    """
    try:
        import fastapi
        from fastapi.responses import RedirectResponse
        from starlette.datastructures import URL
    except ImportError as e:
        raise ImportError(
            "Cannot initialize OAuth to due a missing library. Please run `pip install huggingface_hub[oauth]` or add "
            "`huggingface_hub[oauth]` to your requirements.txt file."
        ) from e

    warnings.warn(
        "OAuth is not supported outside of a Space environment. To help you debug your app locally, the oauth endpoints"
        " are mocked to return your profile and token. To make it work, your machine must be logged in to Huggingface."
    )
    mocked_oauth_info = _get_mocked_oauth_info()

    login_uri, callback_uri, logout_uri = _get_oauth_uris(route_prefix)

    # Define OAuth routes
    @app.get(login_uri)
    async def oauth_login(request: fastapi.Request) -> RedirectResponse:
        """Fake endpoint that redirects to HF OAuth page."""
        # Define target (where to redirect after login)
        redirect_uri = _generate_redirect_uri(request)
        return RedirectResponse(callback_uri + "?" + urllib.parse.urlencode({"_target_url": redirect_uri}))

    @app.get(callback_uri)
    async def oauth_redirect_callback(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that handles the OAuth callback."""
        request.session["oauth_info"] = mocked_oauth_info
        return RedirectResponse(_get_redirect_target(request))

    @app.get(logout_uri)
    async def oauth_logout(request: fastapi.Request) -> RedirectResponse:
        """Endpoint that logs out the user (e.g. delete cookie session)."""
        request.session.pop("oauth_info", None)
        logout_url = URL("/").include_query_params(**request.query_params)
        return RedirectResponse(url=logout_url, status_code=302)  # see https://github.com/gradio-app/gradio/pull/9659


def _generate_redirect_uri(request: "fastapi.Request") -> str:
    if "_target_url" in request.query_params:
        # if `_target_url` already in query params => respect it
        target = request.query_params["_target_url"]
    else:
        # otherwise => keep query params
        target = "/?" + urllib.parse.urlencode(request.query_params)

    redirect_uri = request.url_for("oauth_redirect_callback").include_query_params(_target_url=target)
    redirect_uri_as_str = str(redirect_uri)
    if redirect_uri.netloc.endswith(".hf.space"):
        # In Space, FastAPI redirect as http but we want https
        redirect_uri_as_str = redirect_uri_as_str.replace("http://", "https://")
    return redirect_uri_as_str


def _get_redirect_target(request: "fastapi.Request", default_target: str = "/") -> str:
    return request.query_params.get("_target_url", default_target)


def _get_mocked_oauth_info() -> dict:
    token = get_token()
    if token is None:
        raise ValueError(
            "Your machine must be logged in to HF to debug an OAuth app locally. Please"
            " run `hf auth login` or set `HF_TOKEN` as environment variable "
            "with one of your access token. You can generate a new token in your "
            "settings page (https://huggingface.co/settings/tokens)."
        )

    user = whoami()
    if user["type"] != "user":
        raise ValueError(
            "Your machine is not logged in with a personal account. Please use a "
            "personal access token. You can generate a new token in your settings page"
            " (https://huggingface.co/settings/tokens)."
        )

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 8 * 60 * 60,  # 8 hours
        "id_token": "FOOBAR",
        "scope": "openid profile",
        "refresh_token": "hf_oauth__refresh_token",
        "expires_at": int(time.time()) + 8 * 60 * 60,  # 8 hours
        "userinfo": {
            "sub": "0123456789",
            "name": user["fullname"],
            "preferred_username": user["name"],
            "profile": f"https://huggingface.co/{user['name']}",
            "picture": user["avatarUrl"],
            "website": "",
            "aud": "00000000-0000-0000-0000-000000000000",
            "auth_time": 1691672844,
            "nonce": "aaaaaaaaaaaaaaaaaaa",
            "iat": 1691672844,
            "exp": 1691676444,
            "iss": "https://huggingface.co",
        },
    }


def _get_oauth_uris(route_prefix: str = "/") -> tuple[str, str, str]:
    route_prefix = route_prefix.strip("/")
    if route_prefix:
        route_prefix = f"/{route_prefix}"
    return (
        f"{route_prefix}/oauth/huggingface/login",
        f"{route_prefix}/oauth/huggingface/callback",
        f"{route_prefix}/oauth/huggingface/logout",
    )
<|endoftext|>from __future__ import annotations

import collections
import os
import sys
import warnings
from typing import IO

import PIL

from . import Image

modules = {
    "pil": ("PIL._imaging", "PILLOW_VERSION"),
    "tkinter": ("PIL._tkinter_finder", "tk_version"),
    "freetype2": ("PIL._imagingft", "freetype2_version"),
    "littlecms2": ("PIL._imagingcms", "littlecms_version"),
    "webp": ("PIL._webp", "webpdecoder_version"),
    "avif": ("PIL._avif", "libavif_version"),
}


def check_module(feature: str) -> bool:
    """
    Checks if a module is available.

    :param feature: The module to check for.
    :returns: ``True`` if available, ``False`` otherwise.
    :raises ValueError: If the module is not defined in this version of Pillow.
    """
    if feature not in modules:
        msg = f"Unknown module {feature}"
        raise ValueError(msg)

    module, ver = modules[feature]

    try:
        __import__(module)
        return True
    except ModuleNotFoundError:
        return False
    except ImportError as ex:
        warnings.warn(str(ex))
        return False


def version_module(feature: str) -> str | None:
    """
    :param feature: The module to check for.
    :returns:
        The loaded version number as a string, or ``None`` if unknown or not available.
    :raises ValueError: If the module is not defined in this version of Pillow.
    """
    if not check_module(feature):
        return None

    module, ver = modules[feature]

    return getattr(__import__(module, fromlist=[ver]), ver)


def get_supported_modules() -> list[str]:
    """
    :returns: A list of all supported modules.
    """
    return [f for f in modules if check_module(f)]


codecs = {
    "jpg": ("jpeg", "jpeglib"),
    "jpg_2000": ("jpeg2k", "jp2klib"),
    "zlib": ("zip", "zlib"),
    "libtiff": ("libtiff", "libtiff"),
}


def check_codec(feature: str) -> bool:
    """
    Checks if a codec is available.

    :param feature: The codec to check for.
    :returns: ``True`` if available, ``False`` otherwise.
    :raises ValueError: If the codec is not defined in this version of Pillow.
    """
    if feature not in codecs:
        msg = f"Unknown codec {feature}"
        raise ValueError(msg)

    codec, lib = codecs[feature]

    return f"{codec}_encoder" in dir(Image.core)


def version_codec(feature: str) -> str | None:
    """
    :param feature: The codec to check for.
    :returns:
        The version number as a string, or ``None`` if not available.
        Checked at compile time for ``jpg``, run-time otherwise.
    :raises ValueError: If the codec is not defined in this version of Pillow.
    """
    if not check_codec(feature):
        return None

    codec, lib = codecs[feature]

    version = getattr(Image.core, f"{lib}_version")

    if feature == "libtiff":
        return version.split("\n")[0].split("Version ")[1]

    return version


def get_supported_codecs() -> list[str]:
    """
    :returns: A list of all supported codecs.
    """
    return [f for f in codecs if check_codec(f)]


features: dict[str, tuple[str, str, str | None]] = {
    "raqm": ("PIL._imagingft", "HAVE_RAQM", "raqm_version"),
    "fribidi": ("PIL._imagingft", "HAVE_FRIBIDI", "fribidi_version"),
    "harfbuzz": ("PIL._imagingft", "HAVE_HARFBUZZ", "harfbuzz_version"),
    "libjpeg_turbo": ("PIL._imaging", "HAVE_LIBJPEGTURBO", "libjpeg_turbo_version"),
    "mozjpeg": ("PIL._imaging", "HAVE_MOZJPEG", "libjpeg_turbo_version"),
    "zlib_ng": ("PIL._imaging", "HAVE_ZLIBNG", "zlib_ng_version"),
    "libimagequant": ("PIL._imaging", "HAVE_LIBIMAGEQUANT", "imagequant_version"),
    "xcb": ("PIL._imaging", "HAVE_XCB", None),
}


def check_feature(feature: str) -> bool | None:
    """
    Checks if a feature is available.

    :param feature: The feature to check for.
    :returns: ``True`` if available, ``False`` if unavailable, ``None`` if unknown.
    :raises ValueError: If the feature is not defined in this version of Pillow.
    """
    if feature not in features:
        msg = f"Unknown feature {feature}"
        raise ValueError(msg)

    module, flag, ver = features[feature]

    try:
        imported_module = __import__(module, fromlist=["PIL"])
        return getattr(imported_module, flag)
    except ModuleNotFoundError:
        return None
    except ImportError as ex:
        warnings.warn(str(ex))
        return None


def version_feature(feature: str) -> str | None:
    """
    :param feature: The feature to check for.
    :returns: The version number as a string, or ``None`` if not available.
    :raises ValueError: If the feature is not defined in this version of Pillow.
    """
    if not check_feature(feature):
        return None

    module, flag, ver = features[feature]

    if ver is None:
        return None

    return getattr(__import__(module, fromlist=[ver]), ver)


def get_supported_features() -> list[str]:
    """
    :returns: A list of all supported features.
    """
    return [f for f in features if check_feature(f)]


def check(feature: str) -> bool | None:
    """
    :param feature: A module, codec, or feature name.
    :returns:
        ``True`` if the module, codec, or feature is available,
        ``False`` or ``None`` otherwise.
    """

    if feature in modules:
        return check_module(feature)
    if feature in codecs:
        return check_codec(feature)
    if feature in features:
        return check_feature(feature)
    warnings.warn(f"Unknown feature '{feature}'.", stacklevel=2)
    return False


def version(feature: str) -> str | None:
    """
    :param feature:
        The module, codec, or feature to check for.
    :returns:
        The version number as a string, or ``None`` if unknown or not available.
    """
    if feature in modules:
        return version_module(feature)
    if feature in codecs:
        return version_codec(feature)
    if feature in features:
        return version_feature(feature)
    return None


def get_supported() -> list[str]:
    """
    :returns: A list of all supported modules, features, and codecs.
    """

    ret = get_supported_modules()
    ret.extend(get_supported_features())
    ret.extend(get_supported_codecs())
    return ret


def pilinfo(out: IO[str] | None = None, supported_formats: bool = True) -> None:
    """
    Prints information about this installation of Pillow.
    This function can be called with ``python3 -m PIL``.
    It can also be called with ``python3 -m PIL.report`` or ``python3 -m PIL --report``
    to have "supported_formats" set to ``False``, omitting the list of all supported
    image file formats.

    :param out:
        The output stream to print to. Defaults to ``sys.stdout`` if ``None``.
    :param supported_formats:
        If ``True``, a list of all supported image file formats will be printed.
    """

    if out is None:
        out = sys.stdout

    Image.init()

    print("-" * 68, file=out)
    print(f"Pillow {PIL.__version__}", file=out)
    py_version_lines = sys.version.splitlines()
    print(f"Python {py_version_lines[0].strip()}", file=out)
    for py_version in py_version_lines[1:]:
        print(f"       {py_version.strip()}", file=out)
    print("-" * 68, file=out)
    print(f"Python executable is {sys.executable or 'unknown'}", file=out)
    if sys.prefix != sys.base_prefix:
        print(f"Environment Python files loaded from {sys.prefix}", file=out)
    print(f"System Python files loaded from {sys.base_prefix}", file=out)
    print("-" * 68, file=out)
    print(
        f"Python Pillow modules loaded from {os.path.dirname(Image.__file__)}",
        file=out,
    )
    print(
        f"Binary Pillow modules loaded from {os.path.dirname(Image.core.__file__)}",
        file=out,
    )
    print("-" * 68, file=out)

    for name, feature in [
        ("pil", "PIL CORE"),
        ("tkinter", "TKINTER"),
        ("freetype2", "FREETYPE2"),
        ("littlecms2", "LITTLECMS2"),
        ("webp", "WEBP"),
        ("avif", "AVIF"),
        ("jpg", "JPEG"),
        ("jpg_2000", "OPENJPEG (JPEG2000)"),
        ("zlib", "ZLIB (PNG/ZIP)"),
        ("libtiff", "LIBTIFF"),
        ("raqm", "RAQM (Bidirectional Text)"),
        ("libimagequant", "LIBIMAGEQUANT (Quantization method)"),
        ("xcb", "XCB (X protocol)"),
    ]:
        if check(name):
            v: str | None = None
            if name == "jpg":
                libjpeg_turbo_version = version_feature("libjpeg_turbo")
                if libjpeg_turbo_version is not None:
                    v = "mozjpeg" if check_feature("mozjpeg") else "libjpeg-turbo"
                    v += " " + libjpeg_turbo_version
            if v is None:
                v = version(name)
            if v is not None:
                version_static = name in ("pil", "jpg")
                if name == "littlecms2":
                    # this check is also in src/_imagingcms.c:setup_module()
                    version_static = tuple(int(x) for x in v.split(".")) < (2, 7)
                t = "compiled for" if version_static else "loaded"
                if name == "zlib":
                    zlib_ng_version = version_feature("zlib_ng")
                    if zlib_ng_version is not None:
                        v += ", compiled for zlib-ng " + zlib_ng_version
                elif name == "raqm":
                    for f in ("fribidi", "harfbuzz"):
                        v2 = version_feature(f)
                        if v2 is not None:
                            v += f", {f} {v2}"
                print("---", feature, "support ok,", t, v, file=out)
            else:
                print("---", feature, "support ok", file=out)
        else:
            print("***", feature, "support not installed", file=out)
    print("-" * 68, file=out)

    if supported_formats:
        extensions = collections.defaultdict(list)
        for ext, i in Image.EXTENSION.items():
            extensions[i].append(ext)

        for i in sorted(Image.ID):
            line = f"{i}"
            if i in Image.MIME:
                line = f"{line} {Image.MIME[i]}"
            print(line, file=out)

            if i in extensions:
                print(
                    "Extensions: {}".format(", ".join(sorted(extensions[i]))), file=out
                )

            features = []
            if i in Image.OPEN:
                features.append("open")
            if i in Image.SAVE:
                features.append("save")
            if i in Image.SAVE_ALL:
                features.append("save_all")
            if i in Image.DECODERS:
                features.append("decode")
            if i in Image.ENCODERS:
                features.append("encode")

            print("Features: {}".format(", ".join(features)), file=out)
            print("-" * 68, file=out)
<|endoftext|>from __future__ import annotations

import abc
import asyncio
import collections
import concurrent.futures
import contextlib
import copy
import dataclasses
import enum
import logging
import math
import shlex
import subprocess
import sys
import threading
import time
import typing
import re

import chess

from chess import Color
from types import TracebackType
from typing import Any, Callable, Coroutine, Deque, Dict, Generator, Generic, Iterable, Iterator, List, Literal, Mapping, MutableMapping, Optional, Tuple, Type, TypedDict, TypeVar, Union

if typing.TYPE_CHECKING:
    from typing_extensions import override
else:
    F = typing.TypeVar("F", bound=Callable[..., Any])
    def override(fn: F, /) -> F:
        return fn

if typing.TYPE_CHECKING:
    from typing_extensions import Self

WdlModel = Literal["sf", "sf16.1", "sf16", "sf15.1", "sf15", "sf14", "sf12", "lichess"]


T = TypeVar("T")
ProtocolT = TypeVar("ProtocolT", bound="Protocol")

ConfigValue = Union[str, int, bool, None]
ConfigMapping = Mapping[str, ConfigValue]


LOGGER = logging.getLogger(__name__)


MANAGED_OPTIONS = ["uci_chess960", "uci_variant", "multipv", "ponder"]


# No longer needed, but alias kept around for compatibility.
EventLoopPolicy = asyncio.DefaultEventLoopPolicy


def run_in_background(coroutine: Callable[[concurrent.futures.Future[T]], Coroutine[Any, Any, None]], *, name: Optional[str] = None, debug: Optional[bool] = None) -> T:
    """
    Runs ``coroutine(future)`` in a new event loop on a background thread.

    Blocks on *future* and returns the result as soon as it is resolved.
    The coroutine and all remaining tasks continue running in the background
    until complete.
    """
    assert asyncio.iscoroutinefunction(coroutine)

    future: concurrent.futures.Future[T] = concurrent.futures.Future()

    def background() -> None:
        try:
            asyncio.run(coroutine(future), debug=debug)
            future.cancel()
        except Exception as exc:
            future.set_exception(exc)

    threading.Thread(target=background, name=name).start()
    return future.result()


class EngineError(RuntimeError):
    """Runtime error caused by a misbehaving engine or incorrect usage."""


class EngineTerminatedError(EngineError):
    """The engine process exited unexpectedly."""


class AnalysisComplete(Exception):
    """
    Raised when analysis is complete, all information has been consumed, but
    further information was requested.
    """


@dataclasses.dataclass(frozen=True)
class Option:
    """Information about an available engine option."""

    name: str
    """The name of the option."""

    type: str
    """
    The type of the option.

    +--------+-----+------+------------------------------------------------+
    | type   | UCI | CECP | value                                          |
    +========+=====+======+================================================+
    | check  | X   | X    | ``True`` or ``False``                          |
    +--------+-----+------+------------------------------------------------+
    | spin   | X   | X    | integer, between *min* and *max*               |
    +--------+-----+------+------------------------------------------------+
    | combo  | X   | X    | string, one of *var*                           |
    +--------+-----+------+------------------------------------------------+
    | button | X   | X    | ``None``                                       |
    +--------+-----+------+------------------------------------------------+
    | reset  |     | X    | ``None``                                       |
    +--------+-----+------+------------------------------------------------+
    | save   |     | X    | ``None``                                       |
    +--------+-----+------+------------------------------------------------+
    | string | X   | X    | string without line breaks                     |
    +--------+-----+------+------------------------------------------------+
    | file   |     | X    | string, interpreted as the path to a file      |
    +--------+-----+------+------------------------------------------------+
    | path   |     | X    | string, interpreted as the path to a directory |
    +--------+-----+------+------------------------------------------------+
    """

    default: ConfigValue
    """The default value of the option."""

    min: Optional[int]
    """The minimum integer value of a *spin* option."""

    max: Optional[int]
    """The maximum integer value of a *spin* option."""

    var: Optional[List[str]]
    """A list of allowed string values for a *combo* option."""

    def parse(self, value: ConfigValue) -> ConfigValue:
        if self.type == "check":
            return value and value != "false"
        elif self.type == "spin":
            try:
                value = int(value)  # type: ignore
            except ValueError:
                raise EngineError(f"expected integer for spin option {self.name!r}, got: {value!r}")
            if self.min is not None and value < self.min:
                raise EngineError(f"expected value for option {self.name!r} to be at least {self.min}, got: {value}")
            if self.max is not None and self.max < value:
                raise EngineError(f"expected value for option {self.name!r} to be at most {self.max}, got: {value}")
            return value
        elif self.type == "combo":
            value = str(value)
            if value not in (self.var or []):
                raise EngineError("invalid value for combo option {!r}, got: {} (available: {})".format(self.name, value, ", ".join(self.var) if self.var else "-"))
            return value
        elif self.type in ["button", "reset", "save"]:
            return None
        elif self.type in ["string", "file", "path"]:
            value = str(value)
            if "\n" in value or "\r" in value:
                raise EngineError(f"invalid line-break in string option {self.name!r}: {value!r}")
            return value
        else:
            raise EngineError(f"unknown option type: {self.type!r}")

    def is_managed(self) -> bool:
        """
        Some options are managed automatically: ``UCI_Chess960``,
        ``UCI_Variant``, ``MultiPV``, ``Ponder``.
        """
        return self.name.lower() in MANAGED_OPTIONS


@dataclasses.dataclass
class Limit:
    """Search-termination condition."""

    time: Optional[float] = None
    """Search exactly *time* seconds."""

    depth: Optional[int] = None
    """Search *depth* ply only."""

    nodes: Optional[int] = None
    """Search only a limited number of *nodes*."""

    mate: Optional[int] = None
    """Search for a mate in *mate* moves."""

    white_clock: Optional[float] = None
    """Time in seconds remaining for White."""

    black_clock: Optional[float] = None
    """Time in seconds remaining for Black."""

    white_inc: Optional[float] = None
    """Fisher increment for White, in seconds."""

    black_inc: Optional[float] = None
    """Fisher increment for Black, in seconds."""

    remaining_moves: Optional[int] = None
    """
    Number of moves to the next time control. If this is not set, but
    *white_clock* and *black_clock* are, then it is sudden death.
    """

    clock_id: object = None
    """
    An identifier to use with XBoard engines to signal that the time
    control has changed. When this field changes, Xboard engines are
    sent level or st commands as appropriate. Otherwise, only time
    and otim commands are sent to update the engine's clock.
    """

    def __repr__(self) -> str:
        # Like default __repr__, but without None values.
        return "{}({})".format(
            type(self).__name__,
            ", ".join("{}={!r}".format(attr, getattr(self, attr))
                      for attr in ["time", "depth", "nodes", "mate", "white_clock", "black_clock", "white_inc", "black_inc", "remaining_moves"]
                      if getattr(self, attr) is not None))


class InfoDict(TypedDict, total=False):
    """
    Dictionary of aggregated information sent by the engine.

    Commonly used keys are: ``score`` (a :class:`~chess.engine.PovScore`),
    ``pv`` (a list of :class:`~chess.Move` objects), ``depth``,
    ``seldepth``, ``time`` (in seconds), ``nodes``, ``nps``, ``multipv``
    (``1`` for the mainline).

    Others: ``tbhits``, ``currmove``, ``currmovenumber``, ``hashfull``,
    ``cpuload``, ``refutation``, ``currline``, ``ebf`` (effective branching factor),
    ``wdl`` (a :class:`~chess.engine.PovWdl`), and ``string``.
    """
    score: PovScore
    pv: List[chess.Move]
    depth: int
    seldepth: int
    time: float
    nodes: int
    nps: int
    tbhits: int
    multipv: int
    currmove: chess.Move
    currmovenumber: int
    hashfull: int
    cpuload: int
    refutation: Dict[chess.Move, List[chess.Move]]
    currline: Dict[int, List[chess.Move]]
    ebf: float
    wdl: PovWdl
    string: str


class PlayResult:
    """Returned by :func:`chess.engine.Protocol.play()`."""

    move: Optional[chess.Move]
    """The best move according to the engine, or ``None``."""

    ponder: Optional[chess.Move]
    """The response that the engine expects after *move*, or ``None``."""

    info: InfoDict
    """
    A dictionary of extra :class:`information <chess.engine.InfoDict>`
    sent by the engine, if selected with the *info* argument of
    :func:`~chess.engine.Protocol.play()`.
    """

    draw_offered: bool
    """Whether the engine offered a draw before moving."""

    resigned: bool
    """Whether the engine resigned."""

    def __init__(self,
                 move: Optional[chess.Move],
                 ponder: Optional[chess.Move],
                 info: Optional[InfoDict] = None,
                 *,
                 draw_offered: bool = False,
                 resigned: bool = False) -> None:
        self.move = move
        self.ponder = ponder
        self.info = info or {}
        self.draw_offered = draw_offered
        self.resigned = resigned

    def __repr__(self) -> str:
        return "<{} at {:#x} (move={}, ponder={}, info={}, draw_offered={}, resigned={})>".format(
            type(self).__name__, id(self), self.move, self.ponder, self.info,
            self.draw_offered, self.resigned)


class Info(enum.IntFlag):
    """Used to filter information sent by the chess engine."""
    NONE = 0
    BASIC = 1
    SCORE = 2
    PV = 4
    REFUTATION = 8
    CURRLINE = 16
    ALL = BASIC | SCORE | PV | REFUTATION | CURRLINE

INFO_NONE = Info.NONE
INFO_BASIC = Info.BASIC
INFO_SCORE = Info.SCORE
INFO_PV = Info.PV
INFO_REFUTATION = Info.REFUTATION
INFO_CURRLINE = Info.CURRLINE
INFO_ALL = Info.ALL


@dataclasses.dataclass
class Opponent:
    """Used to store information about an engine's opponent."""

    name: Optional[str]
    """The name of the opponent."""

    title: Optional[str]
    """The opponent's title--for example, GM, IM, or BOT."""

    rating: Optional[int]
    """The opponent's ELO rating."""

    is_engine: Optional[bool]
    """Whether the opponent is a chess engine/computer program."""


class PovScore:
    """A relative :class:`~chess.engine.Score` and the point of view."""

    relative: Score
    """The relative :class:`~chess.engine.Score`."""

    turn: Color
    """The point of view (``chess.WHITE`` or ``chess.BLACK``)."""

    def __init__(self, relative: Score, turn: Color) -> None:
        self.relative = relative
        self.turn = turn

    def white(self) -> Score:
        """Gets the score from White's point of view."""
        return self.pov(chess.WHITE)

    def black(self) -> Score:
        """Gets the score from Black's point of view."""
        return self.pov(chess.BLACK)

    def pov(self, color: Color) -> Score:
        """Gets the score from the point of view of the given *color*."""
        return self.relative if self.turn == color else -self.relative

    def is_mate(self) -> bool:
        """Tests if this is a mate score."""
        return self.relative.is_mate()

    def wdl(self, *, model: WdlModel = "sf", ply: int = 30) -> PovWdl:
        """See :func:`~chess.engine.Score.wdl()`."""
        return PovWdl(self.relative.wdl(model=model, ply=ply), self.turn)

    def __repr__(self) -> str:
        return "PovScore({!r}, {})".format(self.relative, "WHITE" if self.turn else "BLACK")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PovScore):
            return self.white() == other.white()
        else:
            return NotImplemented


class Score(abc.ABC):
    """
    Evaluation of a position.

    The score can be :class:`~chess.engine.Cp` (centi-pawns),
    :class:`~chess.engine.Mate` or :py:data:`~chess.engine.MateGiven`.
    A positive value indicates an advantage.

    There is a total order defined on centi-pawn and mate scores.

    >>> from chess.engine import Cp, Mate, MateGiven
    >>>
    >>> Mate(-0) < Mate(-1) < Cp(-50) < Cp(200) < Mate(4) < Mate(1) < MateGiven
    True

    Scores can be negated to change the point of view:

    >>> -Cp(20)
    Cp(-20)

    >>> -Mate(-4)
    Mate(+4)

    >>> -Mate(0)
    MateGiven
    """

    @typing.overload
    @abc.abstractmethod
    def score(self, *, mate_score: int) -> int: ...
    @typing.overload
    @abc.abstractmethod
    def score(self, *, mate_score: Optional[int] = None) -> Optional[int]: ...
    @abc.abstractmethod
    def score(self, *, mate_score: Optional[int] = None) -> Optional[int]:
        """
        Returns the centi-pawn score as an integer or ``None``.

        You can optionally pass a large value to convert mate scores to
        centi-pawn scores.

        >>> Cp(-300).score()
        -300
        >>> Mate(5).score() is None
        True
        >>> Mate(5).score(mate_score=100000)
        99995
        """

    @abc.abstractmethod
    def mate(self) -> Optional[int]:
        """
        Returns the number of plies to mate, negative if we are getting
        mated, or ``None``.

        .. warning::
            This conflates ``Mate(0)`` (we lost) and ``MateGiven``
            (we won) to ``0``.
        """

    def is_mate(self) -> bool:
        """Tests if this is a mate score."""
        return self.mate() is not None

    @abc.abstractmethod
    def wdl(self, *, model: WdlModel = "sf", ply: int = 30) -> Wdl:
        """
        Returns statistics for the expected outcome of this game, based on
        a *model*, given that this score is reached at *ply*.

        Scores have a total order, but it makes little sense to compute
        the difference between two scores. For example, going from
        ``Cp(-100)`` to ``Cp(+100)`` is much more significant than going
        from ``Cp(+300)`` to ``Cp(+500)``. It is better to compute differences
        of the expectation values for the outcome of the game (based on winning
        chances and drawing chances).

        >>> Cp(100).wdl().expectation() - Cp(-100).wdl().expectation()  # doctest: +ELLIPSIS
        0.379...

        >>> Cp(500).wdl().expectation() - Cp(300).wdl().expectation()  # doctest: +ELLIPSIS
        0.015...

        :param model:
            * ``sf``, the WDL model used by the latest Stockfish
              (currently ``sf16``).
            * ``sf16``, the WDL model used by Stockfish 16.
            * ``sf15.1``, the WDL model used by Stockfish 15.1.
            * ``sf15``, the WDL model used by Stockfish 15.
            * ``sf14``, the WDL model used by Stockfish 14.
            * ``sf12``, the WDL model used by Stockfish 12.
            * ``lichess``, the win rate model used by Lichess.
              Does not use *ply*, and does not consider drawing chances.
        :param ply: The number of half-moves played since the starting
            position. Models may scale scores slightly differently based on
            this. Defaults to middle game.
        """

    @abc.abstractmethod
    def __neg__(self) -> Score:
        ...

    @abc.abstractmethod
    def __pos__(self) -> Score:
        ...

    @abc.abstractmethod
    def __abs__(self) -> Score:
        ...

    def _score_tuple(self) -> Tuple[bool, bool, bool, int, Optional[int]]:
        mate = self.mate()
        return (
            isinstance(self, MateGivenType),
            mate is not None and mate > 0,
            mate is None,
            -(mate or 0),
            self.score(),
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self._score_tuple() == other._score_tuple()
        else:
            return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self._score_tuple() < other._score_tuple()
        else:
            return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self._score_tuple() <= other._score_tuple()
        else:
            return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self._score_tuple() > other._score_tuple()
        else:
            return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, Score):
            return self._score_tuple() >= other._score_tuple()
        else:
            return NotImplemented

def _sf16_1_wins(cp: int, *, ply: int) -> int:
    # https://github.com/official-stockfish/Stockfish/blob/sf_16.1/src/uci.cpp#L48
    NormalizeToPawnValue = 356
    # https://github.com/official-stockfish/Stockfish/blob/sf_16.1/src/uci.cpp#L383-L384
    m = min(120, max(8, ply / 2 + 1)) / 32
    a = (((-1.06249702 * m + 7.42016937) * m + 0.89425629) * m) + 348.60356174
    b = (((-5.33122190 * m + 39.57831533) * m + -90.84473771) * m) + 123.40620748
    x = min(4000, max(cp * NormalizeToPawnValue / 100, -4000))
    return int(0.5 + 1000 / (1 + math.exp((a - x) / b)))

def _sf16_wins(cp: int, *, ply: int) -> int:
    # https://github.com/official-stockfish/Stockfish/blob/sf_16/src/uci.h#L38
    NormalizeToPawnValue = 328
    # https://github.com/official-stockfish/Stockfish/blob/sf_16/src/uci.cpp#L200-L224
    m = min(240, max(ply, 0)) / 64
    a = (((0.38036525 * m + -2.82015070) * m + 23.17882135) * m) + 307.36768407
    b = (((-2.29434733 * m + 13.27689788) * m + -14.26828904) * m) + 63.45318330
    x = min(4000, max(cp * NormalizeToPawnValue / 100, -4000))
    return int(0.5 + 1000 / (1 + math.exp((a - x) / b)))

def _sf15_1_wins(cp: int, *, ply: int) -> int:
    # https://github.com/official-stockfish/Stockfish/blob/sf_15.1/src/uci.h#L38
    NormalizeToPawnValue = 361
    # https://github.com/official-stockfish/Stockfish/blob/sf_15.1/src/uci.cpp#L200-L224
    m = min(240, max(ply, 0)) / 64
    a = (((-0.58270499 * m + 2.68512549) * m + 15.24638015) * m) + 344.49745382
    b = (((-2.65734562 * m + 15.96509799) * m + -20.69040836) * m) + 73.61029937
    x = min(4000, max(cp * NormalizeToPawnValue / 100, -4000))
    return int(0.5 + 1000 / (1 + math.exp((a - x) / b)))

def _sf15_wins(cp: int, *, ply: int) -> int:
    # https://github.com/official-stockfish/Stockfish/blob/sf_15/src/uci.cpp#L200-L220
    m = min(240, max(ply, 0)) / 64
    a = (((-1.17202460e-1 * m + 5.94729104e-1) * m + 1.12065546e+1) * m) + 1.22606222e+2
    b = (((-1.79066759 * m + 11.30759193) * m + -17.43677612) * m) + 36.47147479
    x = min(2000, max(cp, -2000))
    return int(0.5 + 1000 / (1 + math.exp((a - x) / b)))

def _sf14_wins(cp: int, *, ply: int) -> int:
    # https://github.com/official-stockfish/Stockfish/blob/sf_14/src/uci.cpp#L200-L220
    m = min(240, max(ply, 0)) / 64
    a = (((-3.68389304 * m + 30.07065921) * m + -60.52878723) * m) + 149.53378557
    b = (((-2.01818570 * m + 15.85685038) * m + -29.83452023) * m) + 47.59078827
    x = min(2000, max(cp, -2000))
    return int(0.5 + 1000 / (1 + math.exp((a - x) / b)))

def _sf12_wins(cp: int, *, ply: int) -> int:
    # https://github.com/official-stockfish/Stockfish/blob/sf_12/src/uci.cpp#L198-L218
    m = min(240, max(ply, 0)) / 64
    a = (((-8.24404295 * m + 64.23892342) * m + -95.73056462) * m) + 153.86478679
    b = (((-3.37154371 * m + 28.44489198) * m + -56.67657741) * m) + 72.05858751
    x = min(1000, max(cp, -1000))
    return int(0.5 + 1000 / (1 + math.exp((a - x) / b)))

def _lichess_raw_wins(cp: int) -> int:
    # https://github.com/lichess-org/lila/pull/11148
    # https://github.com/lichess-org/lila/blob/2242b0a08faa06e7be5508d338ede7bb09049777/modules/analyse/src/main/WinPercent.scala#L26-L30
    return round(1000 / (1 + math.exp(-0.00368208 * cp)))


class Cp(Score):
    """Centi-pawn score."""

    def __init__(self, cp: int) -> None:
        self.cp = cp

    def mate(self) -> None:
        return None

    def score(self, *, mate_score: Optional[int] = None) -> int:
        return self.cp

    def wdl(self, *, model: WdlModel = "sf", ply: int = 30) -> Wdl:
        if model == "lichess":
            wins = _lichess_raw_wins(max(-1000, min(self.cp, 1000)))
            losses = 1000 - wins
        elif model == "sf12":
            wins = _sf12_wins(self.cp, ply=ply)
            losses = _sf12_wins(-self.cp, ply=ply)
        elif model == "sf14":
            wins = _sf14_wins(self.cp, ply=ply)
            losses = _sf14_wins(-self.cp, ply=ply)
        elif model == "sf15":
            wins = _sf15_wins(self.cp, ply=ply)
            losses = _sf15_wins(-self.cp, ply=ply)
        elif model == "sf15.1":
            wins = _sf15_1_wins(self.cp, ply=ply)
            losses = _sf15_1_wins(-self.cp, ply=ply)
        elif model == "sf16":
            wins = _sf16_wins(self.cp, ply=ply)
            losses = _sf16_wins(-self.cp, ply=ply)
        else:
            wins = _sf16_1_wins(self.cp, ply=ply)
            losses = _sf16_1_wins(-self.cp, ply=ply)
        draws = 1000 - wins - losses
        return Wdl(wins, draws, losses)

    def __str__(self) -> str:
        return f"+{self.cp:d}" if self.cp > 0 else str(self.cp)

    def __repr__(self) -> str:
        return f"Cp({self})"

    def __neg__(self) -> Cp:
        return Cp(-self.cp)

    def __pos__(self) -> Cp:
        return Cp(self.cp)

    def __abs__(self) -> Cp:
        return Cp(abs(self.cp))


class Mate(Score):
    """Mate score."""

    def __init__(self, moves: int) -> None:
        self.moves = moves

    def mate(self) -> int:
        return self.moves

    @typing.overload
    def score(self, *, mate_score: int) -> int: ...
    @typing.overload
    def score(self, *, mate_score: Optional[int] = None) -> Optional[int]: ...
    def score(self, *, mate_score: Optional[int] = None) -> Optional[int]:
        if mate_score is None:
            return None
        elif self.moves > 0:
            return mate_score - self.moves
        else:
            return -mate_score - self.moves

    def wdl(self, *, model: WdlModel = "sf", ply: int = 30) -> Wdl:
        if model == "lichess":
            cp = (21 - min(10, abs(self.moves))) * 100
            wins = _lichess_raw_wins(cp)
            return Wdl(wins, 0, 1000 - wins) if self.moves > 0 else Wdl(1000 - wins, 0, wins)
        else:
            return Wdl(1000, 0, 0) if self.moves > 0 else Wdl(0, 0, 1000)

    def __str__(self) -> str:
        return f"#+{self.moves}" if self.moves > 0 else f"#-{abs(self.moves)}"

    def __repr__(self) -> str:
        return "Mate({})".format(str(self).lstrip("#"))

    def __neg__(self) -> Union[MateGivenType, Mate]:
        return MateGiven if not self.moves else Mate(-self.moves)

    def __pos__(self) -> Mate:
        return Mate(self.moves)

    def __abs__(self) -> Union[MateGivenType, Mate]:
        return MateGiven if not self.moves else Mate(abs(self.moves))


class MateGivenType(Score):
    """Winning mate score, equivalent to ``-Mate(0)``."""

    def mate(self) -> int:
        return 0

    @typing.overload
    def score(self, *, mate_score: int) -> int: ...
    @typing.overload
    def score(self, *, mate_score: Optional[int] = None) -> Optional[int]: ...
    def score(self, *, mate_score: Optional[int] = None) -> Optional[int]:
        return mate_score

    def wdl(self, *, model: WdlModel = "sf", ply: int = 30) -> Wdl:
        return Wdl(1000, 0, 0)

    def __neg__(self) -> Mate:
        return Mate(0)

    def __pos__(self) -> MateGivenType:
        return self

    def __abs__(self) -> MateGivenType:
        return self

    def __repr__(self) -> str:
        return "MateGiven"

    def __str__(self) -> str:
        return "#+0"

MateGiven = MateGivenType()


class PovWdl:
    """
    Relative :class:`win/draw/loss statistics <chess.engine.Wdl>` and the point
    of view.

    .. deprecated:: 1.2
        Behaves like a tuple
        ``(wdl.relative.wins, wdl.relative.draws, wdl.relative.losses)``
        for backwards compatibility. But it is recommended to use the provided
        fields and methods instead.
    """

    relative: Wdl
    """The relative :class:`~chess.engine.Wdl`."""

    turn: Color
    """The point of view (``chess.WHITE`` or ``chess.BLACK``)."""

    def __init__(self, relative: Wdl, turn: Color) -> None:
        self.relative = relative
        self.turn = turn

    def white(self) -> Wdl:
        """Gets the :class:`~chess.engine.Wdl` from White's point of view."""
        return self.pov(chess.WHITE)

    def black(self) -> Wdl:
        """Gets the :class:`~chess.engine.Wdl` from Black's point of view."""
        return self.pov(chess.BLACK)

    def pov(self, color: Color) -> Wdl:
        """
        Gets the :class:`~chess.engine.Wdl` from the point of view of the given
        *color*.
        """
        return self.relative if self.turn == color else -self.relative

    def __bool__(self) -> bool:
        return bool(self.relative)

    def __repr__(self) -> str:
        return "PovWdl({!r}, {})".format(self.relative, "WHITE" if self.turn else "BLACK")

    # Unfortunately in python-chess v1.1.0, info["wdl"] was a simple tuple
    # of the relative permille values, so we have to support __iter__,
    # __len__, __getitem__, and equality comparisons with other tuples.
    # Never mind the ordering, because that's not a sensible operation, anyway.

    def __iter__(self) -> Iterator[int]:
        yield self.relative.wins
        yield self.relative.draws
        yield self.relative.losses

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> int:
        return (self.relative.wins, self.relative.draws, self.relative.losses)[idx]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PovWdl):
            return self.white() == other.white()
        elif isinstance(other, tuple):
            return (self.relative.wins, self.relative.draws, self.relative.losses) == other
        else:
            return NotImplemented


@dataclasses.dataclass
class Wdl:
    """Win/draw/loss statistics."""

    wins: int
    """The number of wins."""

    draws: int
    """The number of draws."""

    losses: int
    """The number of losses."""

    def total(self) -> int:
        """
        Returns the total number of games. Usually, ``wdl`` reported by engines
        is scaled to 1000 games.
        """
        return self.wins + self.draws + self.losses

    def winning_chance(self) -> float:
        """Returns the relative frequency of wins."""
        return self.wins / self.total()

    def drawing_chance(self) -> float:
        """Returns the relative frequency of draws."""
        return self.draws / self.total()

    def losing_chance(self) -> float:
        """Returns the relative frequency of losses."""
        return self.losses / self.total()

    def expectation(self) -> float:
        """
        Returns the expectation value, where a win is valued 1, a draw is
        valued 0.5, and a loss is valued 0.
        """
        return (self.wins + 0.5 * self.draws) / self.total()

    def __bool__(self) -> bool:
        return bool(self.total())

    def __iter__(self) -> Iterator[int]:
        yield self.wins
        yield self.draws
        yield self.losses

    def __reversed__(self) -> Iterator[int]:
        yield self.losses
        yield self.draws
        yield self.wins

    def __pos__(self) -> Wdl:
        return self

    def __neg__(self) -> Wdl:
        return Wdl(self.losses, self.draws, self.wins)


class MockTransport(asyncio.SubprocessTransport, asyncio.WriteTransport):
    def __init__(self, protocol: Protocol) -> None:
        super().__init__()
        self.protocol = protocol
        self.expectations: Deque[Tuple[str, List[str]]] = collections.deque()
        self.expected_pings = 0
        self.stdin_buffer = bytearray()
        self.protocol.connection_made(self)

    def expect(self, expectation: str, responses: List[str] = []) -> None:
        self.expectations.append((expectation, responses))

    def expect_ping(self) -> None:
        self.expected_pings += 1

    def assert_done(self) -> None:
        assert not self.expectations, f"pending expectations: {self.expectations}"

    def get_pipe_transport(self, fd: int) -> Optional[asyncio.BaseTransport]:
        assert fd == 0, f"expected 0 for stdin, got {fd}"
        return self

    def write(self, data: bytes) -> None:
        self.stdin_buffer.extend(data)
        while b"\n" in self.stdin_buffer:
            line_bytes, self.stdin_buffer = self.stdin_buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8")

            if line.startswith("ping ") and self.expected_pings:
                self.expected_pings -= 1
                self.protocol.pipe_data_received(1, (line.replace("ping ", "pong ") + "\n").encode("utf-8"))
            else:
                assert self.expectations, f"unexpected: {line!r}"
                expectation, responses = self.expectations.popleft()
                assert expectation == line, f"expected {expectation}, got: {line}"
                if responses:
                    self.protocol.loop.call_soon(self.protocol.pipe_data_received, 1, "\n".join(responses + [""]).encode("utf-8"))

    def get_pid(self) -> int:
        return id(self)

    def get_returncode(self) -> Optional[int]:
        return None if self.expectations else 0


class Protocol(asyncio.SubprocessProtocol, metaclass=abc.ABCMeta):
    """Protocol for communicating with a chess engine process."""

    options: MutableMapping[str, Option]
    """Dictionary of available options."""

    id: Dict[str, str]
    """
    Dictionary of information about the engine. Common keys are ``name``
    and ``author``.
    """

    returncode: asyncio.Future[int]
    """Future: Exit code of the process."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.transport: Optional[asyncio.SubprocessTransport] = None

        self.buffer = {
            1: bytearray(),  # stdout
            2: bytearray(),  # stderr
        }

        self.command: Optional[BaseCommand[Any]] = None
        self.next_command: Optional[BaseCommand[Any]] = None

        self.initialized = False
        self.returncode: asyncio.Future[int] = asyncio.Future()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        # SubprocessTransport expected, but not checked to allow duck typing.
        self.transport = transport  # type: ignore
        LOGGER.debug("%s: Connection made", self)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        assert self.transport is not None
        code = self.transport.get_returncode()
        assert code is not None, "connect lost, but got no returncode"
        LOGGER.debug("%s: Connection lost (exit code: %d, error: %s)", self, code, exc)

        # Terminate commands.
        command, self.command = self.command, None
        next_command, self.next_command = self.next_command, None
        if command:
            command._engine_terminated(code)
        if next_command:
            next_command._engine_terminated(code)

        self.returncode.set_result(code)

    def process_exited(self) -> None:
        LOGGER.debug("%s: Process exited", self)

    def send_line(self, line: str) -> None:
        LOGGER.debug("%s: << %s", self, line)
        assert self.transport is not None, "cannot send line before connection is made"
        stdin = self.transport.get_pipe_transport(0)
        # WriteTransport expected, but not checked to allow duck typing.
        stdin.write((line + "\n").encode("utf-8"))  # type: ignore

    def pipe_data_received(self, fd: int, data: Union[bytes, str]) -> None:
        self.buffer[fd].extend(data)  # type: ignore
        while b"\n" in self.buffer[fd]:
            line_bytes, self.buffer[fd] = self.buffer[fd].split(b"\n", 1)
            if line_bytes.endswith(b"\r"):
                line_bytes = line_bytes[:-1]
            try:
                line = line_bytes.decode("utf-8")
            except UnicodeDecodeError as err:
                LOGGER.warning("%s: >> %r (%s)", self, bytes(line_bytes), err)
            else:
                if fd == 1:
                    self._line_received(line)
                else:
                    self.error_line_received(line)

    def error_line_received(self, line: str) -> None:
        LOGGER.warning("%s: stderr >> %s", self, line)

    def _line_received(self, line: str) -> None:
        LOGGER.debug("%s: >> %s", self, line)

        self.line_received(line)

        if self.command:
            self.command._line_received(line)

    def line_received(self, line: str) -> None:
        pass

    async def communicate(self, command_factory: Callable[[Self], BaseCommand[T]]) -> T:
        command = command_factory(self)

        if self.returncode.done():
            raise EngineTerminatedError(f"engine process dead (exit code: {self.returncode.result()})")

        assert command.state == CommandState.NEW, command.state

        if self.next_command is not None:
            self.next_command.result.cancel()
            self.next_command.finished.cancel()
            self.next_command.set_finished()

        self.next_command = command

        def previous_command_finished() -> None:
            self.command, self.next_command = self.next_command, None
            if self.command is not None:
                cmd = self.command

                def cancel_if_cancelled(result: asyncio.Future[T]) -> None:
                    if result.cancelled():
                        cmd._cancel()

                cmd.result.add_done_callback(cancel_if_cancelled)
                cmd._start()
                cmd.add_finished_callback(previous_command_finished)

        if self.command is None:
            previous_command_finished()
        elif not self.command.result.done():
            self.command.result.cancel()
        elif not self.command.result.cancelled():
            self.command._cancel()

        return await command.result

    def __repr__(self) -> str:
        pid = self.transport.get_pid() if self.transport is not None else "?"
        return f"<{type(self).__name__} (pid={pid})>"

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initializes the engine."""

    @abc.abstractmethod
    async def ping(self) -> None:
        """
        Pings the engine and waits for a response. Used to ensure the engine
        is still alive and idle.
        """

    @abc.abstractmethod
    async def configure(self, options: ConfigMapping) -> None:
        """
        Configures global engine options.

        :param options: A dictionary of engine options where the keys are
            names of :data:`~chess.engine.Protocol.options`. Do not set options
            that are managed automatically
            (:func:`chess.engine.Option.is_managed()`).
        """

    @abc.abstractmethod
    async def send_opponent_information(self, *, opponent: Optional[Opponent] = None, engine_rating: Optional[int] = None) -> None:
        """
        Sends the engine information about its opponent. The information will
        be sent after a new game is announced and before the first move. This
        method should be called before the first move of a game--i.e., the
        first call to :func:`chess.engine.Protocol.play()`.

        :param opponent: Optional. An instance of :class:`chess.engine.Opponent` that has the opponent's information.
        :param engine_rating: Optional. This engine's own rating. Only used by XBoard engines.
        """

    @abc.abstractmethod
    async def play(self, board: chess.Board, limit: Limit, *, game: object = None, info: Info = INFO_NONE, ponder: bool = False, draw_offered: bool = False, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}, opponent: Optional[Opponent] = None) -> PlayResult:
        """
        Plays a position.

        :param board: The position. The entire move stack will be sent to the
            engine.
        :param limit: An instance of :class:`chess.engine.Limit` that
            determines when to stop thinking.
        :param game: Optional. An arbitrary object that identifies the game.
            Will automatically inform the engine if the object is not equal
            to the previous game (e.g., ``ucinewgame``, ``new``).
        :param info: Selects which additional information to retrieve from the
            engine. ``INFO_NONE``, ``INFO_BASIC`` (basic information that is
            trivial to obtain), ``INFO_SCORE``, ``INFO_PV``,
            ``INFO_REFUTATION``, ``INFO_CURRLINE``, ``INFO_ALL`` or any
            bitwise combination. Some overhead is associated with parsing
            extra information.
        :param ponder: Whether the engine should keep analysing in the
            background even after the result has been returned.
        :param draw_offered: Whether the engine's opponent has offered a draw.
            Ignored by UCI engines.
        :param root_moves: Optional. Consider only root moves from this list.
        :param options: Optional. A dictionary of engine options for the
            analysis. The previous configuration will be restored after the
            analysis is complete. You can permanently apply a configuration
            with :func:`~chess.engine.Protocol.configure()`.
        :param opponent: Optional. Information about a new opponent. Information
            about the original opponent will be restored once the move is
            complete. New opponent information can be made permanent with
            :func:`~chess.engine.Protocol.send_opponent_information()`.
        """

    @typing.overload
    async def analyse(self, board: chess.Board, limit: Limit, *, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> InfoDict: ...
    @typing.overload
    async def analyse(self, board: chess.Board, limit: Limit, *, multipv: int, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> List[InfoDict]: ...
    @typing.overload
    async def analyse(self, board: chess.Board, limit: Limit, *, multipv: Optional[int] = None, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> Union[List[InfoDict], InfoDict]: ...
    async def analyse(self, board: chess.Board, limit: Limit, *, multipv: Optional[int] = None, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> Union[List[InfoDict], InfoDict]:
        """
        Analyses a position and returns a dictionary of
        :class:`information <chess.engine.InfoDict>`.

        :param board: The position to analyse. The entire move stack will be
            sent to the engine.
        :param limit: An instance of :class:`chess.engine.Limit` that
            determines when to stop the analysis.
        :param multipv: Optional. Analyse multiple root moves. Will return
            a list of at most *multipv* dictionaries rather than just a single
            info dictionary.
        :param game: Optional. An arbitrary object that identifies the game.
            Will automatically inform the engine if the object is not equal
            to the previous game (e.g., ``ucinewgame``, ``new``).
        :param info: Selects which information to retrieve from the
            engine. ``INFO_NONE``, ``INFO_BASIC`` (basic information that is
            trivial to obtain), ``INFO_SCORE``, ``INFO_PV``,
            ``INFO_REFUTATION``, ``INFO_CURRLINE``, ``INFO_ALL`` or any
            bitwise combination. Some overhead is associated with parsing
            extra information.
        :param root_moves: Optional. Limit analysis to a list of root moves.
        :param options: Optional. A dictionary of engine options for the
            analysis. The previous configuration will be restored after the
            analysis is complete. You can permanently apply a configuration
            with :func:`~chess.engine.Protocol.configure()`.
        """
        analysis = await self.analysis(board, limit, multipv=multipv, game=game, info=info, root_moves=root_moves, options=options)

        with analysis:
            await analysis.wait()

        return analysis.info if multipv is None else analysis.multipv

    @abc.abstractmethod
    async def analysis(self, board: chess.Board, limit: Optional[Limit] = None, *, multipv: Optional[int] = None, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> AnalysisResult:
        """
        Starts analysing a position.

        :param board: The position to analyse. The entire move stack will be
            sent to the engine.
        :param limit: Optional. An instance of :class:`chess.engine.Limit`
            that determines when to stop the analysis. Analysis is infinite
            by default.
        :param multipv: Optional. Analyse multiple root moves.
        :param game: Optional. An arbitrary object that identifies the game.
            Will automatically inform the engine if the object is not equal
            to the previous game (e.g., ``ucinewgame``, ``new``).
        :param info: Selects which information to retrieve from the
            engine. ``INFO_NONE``, ``INFO_BASIC`` (basic information that is
            trivial to obtain), ``INFO_SCORE``, ``INFO_PV``,
            ``INFO_REFUTATION``, ``INFO_CURRLINE``, ``INFO_ALL`` or any
            bitwise combination. Some overhead is associated with parsing
            extra information.
        :param root_moves: Optional. Limit analysis to a list of root moves.
        :param options: Optional. A dictionary of engine options for the
            analysis. The previous configuration will be restored after the
            analysis is complete. You can permanently apply a configuration
            with :func:`~chess.engine.Protocol.configure()`.

        Returns :class:`~chess.engine.AnalysisResult`, a handle that allows
        asynchronously iterating over the information sent by the engine
        and stopping the analysis at any time.
        """

    @abc.abstractmethod
    async def send_game_result(self, board: chess.Board, winner: Optional[Color] = None, game_ending: Optional[str] = None, game_complete: bool = True) -> None:
        """
        Sends the engine the result of the game.

        XBoard engines receive the final moves and a line of the form
        ``result <winner> {<ending>}``. The ``<winner>`` field is one of ``1-0``,
        ``0-1``, ``1/2-1/2``, or ``*`` to indicate white won, black won, draw,
        or adjournment, respectively. The ``<ending>`` field is a description
        of the specific reason for the end of the game: "White mates",
        "Time forfeiture", "Stalemate", etc.

        UCI engines do not expect end-of-game information and so are not
        sent anything.

        :param board: The final state of the board.
        :param winner: Optional. Specify the winner of the game. This is useful
            if the result of the game is not evident from the board--e.g., time
            forfeiture or draw by agreement. If not ``None``, this parameter
            overrides any winner derivable from the board.
        :param game_ending: Optional. Text describing the reason for the game
            ending. Similarly to the winner parameter, this overrides any game
            result derivable from the board.
        :param game_complete: Optional. Whether the game reached completion.
        """

    @abc.abstractmethod
    async def quit(self) -> None:
        """Asks the engine to shut down."""

    @classmethod
    async def popen(cls: Type[ProtocolT], command: Union[str, List[str]], *, setpgrp: bool = False, **popen_args: Any) -> Tuple[asyncio.SubprocessTransport, ProtocolT]:
        if not isinstance(command, list):
            command = [command]

        if setpgrp:
            try:
                # Windows.
                popen_args["creationflags"] = popen_args.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            except AttributeError:
                # Unix.
                if sys.version_info >= (3, 11):
                    popen_args["process_group"] = 0
                else:
                    # Before Python 3.11
                    popen_args["start_new_session"] = True

        return await asyncio.get_running_loop().subprocess_exec(cls, *command, **popen_args)


class CommandState(enum.Enum):
    NEW = enum.auto()
    ACTIVE = enum.auto()
    CANCELLING = enum.auto()
    DONE = enum.auto()


class BaseCommand(Generic[T]):
    def __init__(self, engine: Protocol) -> None:
        self._engine = engine

        self.state = CommandState.NEW

        self.result: asyncio.Future[T] = asyncio.Future()
        self.finished: asyncio.Future[None] = asyncio.Future()

        self._finished_callbacks: List[Callable[[], None]] = []

    def add_finished_callback(self, callback: Callable[[], None]) -> None:
        self._finished_callbacks.append(callback)
        self._dispatch_finished()

    def _dispatch_finished(self) -> None:
        if self.finished.done():
            while self._finished_callbacks:
                self._finished_callbacks.pop()()

    def _engine_terminated(self, code: int) -> None:
        hint = ", binary not compatible with cpu?" if code in [-4, 0xc000001d] else ""
        exc = EngineTerminatedError(f"engine process died unexpectedly (exit code: {code}{hint})")
        if self.state == CommandState.ACTIVE:
            self.engine_terminated(exc)
        elif self.state == CommandState.CANCELLING:
            self.finished.set_result(None)
            self._dispatch_finished()
        elif self.state == CommandState.NEW:
            self._handle_exception(exc)

    def _handle_exception(self, exc: Exception) -> None:
        if not self.result.done():
            self.result.set_exception(exc)
        else:
            self._engine.loop.call_exception_handler({ # XXX
                "message": f"{type(self).__name__} failed after returning preliminary result ({self.result!r})",
                "exception": exc,
                "protocol": self._engine,
                "transport": self._engine.transport,
            })

        if not self.finished.done():
            self.finished.set_result(None)
            self._dispatch_finished()

    def set_finished(self) -> None:
        assert self.state in [CommandState.ACTIVE, CommandState.CANCELLING], self.state
        if not self.result.done():
            self.result.set_exception(EngineError(f"engine command finished before returning result: {self!r}"))
        self.state = CommandState.DONE
        self.finished.set_result(None)
        self._dispatch_finished()

    def _cancel(self) -> None:
        if self.state != CommandState.CANCELLING and self.state != CommandState.DONE:
            assert self.state == CommandState.ACTIVE, self.state
            self.state = CommandState.CANCELLING
            self.cancel()

    def _start(self) -> None:
        assert self.state == CommandState.NEW, self.state
        self.state = CommandState.ACTIVE
        try:
            self.check_initialized()
            self.start()
        except EngineError as err:
            self._handle_exception(err)

    def _line_received(self, line: str) -> None:
        assert self.state in [CommandState.ACTIVE, CommandState.CANCELLING], self.state
        try:
            self.line_received(line)
        except EngineError as err:
            self._handle_exception(err)

    def cancel(self) -> None:
        pass

    def check_initialized(self) -> None:
        if not self._engine.initialized:
            raise EngineError("tried to run command, but engine is not initialized")

    def start(self) -> None:
        raise NotImplementedError

    def line_received(self, line: str) -> None:
        pass

    def engine_terminated(self, exc: Exception) -> None:
        self._handle_exception(exc)

    def __repr__(self) -> str:
        return "<{} at {:#x} (state={}, result={}, finished={}>".format(type(self).__name__, id(self), self.state, self.result, self.finished)


class UciProtocol(Protocol):
    """
    An implementation of the
    `Universal Chess Interface <https://www.chessprogramming.org/UCI>`_
    protocol.
    """

    def __init__(self) -> None:
        super().__init__()
        self.options: UciOptionMap[Option] = UciOptionMap()
        self.config: UciOptionMap[ConfigValue] = UciOptionMap()
        self.target_config: UciOptionMap[ConfigValue] = UciOptionMap()
        self.id = {}
        self.board = chess.Board()
        self.game: object = None
        self.first_game = True
        self.may_ponderhit: Optional[chess.Board] = None
        self.ponderhit = False

    async def initialize(self) -> None:
        class UciInitializeCommand(BaseCommand[None]):
            def __init__(self, engine: UciProtocol):
                super().__init__(engine)
                self.engine = engine

            @override
            def check_initialized(self) -> None:
                if self.engine.initialized:
                    raise EngineError("engine already initialized")

            @override
            def start(self) -> None:
                self.engine.send_line("uci")

            @override
            def line_received(self, line: str) -> None:
                token, remaining = _next_token(line)
                if line.strip() == "uciok" and not self.result.done():
                    self.engine.initialized = True
                    self.result.set_result(None)
                    self.set_finished()
                elif token == "option":
                    self._option(remaining)
                elif token == "id":
                    self._id(remaining)

            def _option(self, arg: str) -> None:
                current_parameter = None
                option_parts: dict[str, str] = {k: "" for k in ["name", "type", "default", "min", "max"]}
                var = []

                parameters = list(option_parts.keys()) + ['var']
                inner_regex = '|'.join([fr"\b{parameter}\b" for parameter in parameters])
                option_regex = fr"\s*({inner_regex})\s*"
                for token in re.split(option_regex, arg.strip()):
                    if token == "var" or (token in option_parts and not option_parts[token]):
                        current_parameter = token
                    elif current_parameter == "var":
                        var.append(token)
                    elif current_parameter:
                        option_parts[current_parameter] = token

                def parse_min_max_value(option_parts: dict[str, str], which: Literal["min", "max"]) -> Optional[int]:
                    try:
                        number = option_parts[which]
                        return int(number) if number else None
                    except ValueError:
                        LOGGER.exception(f"Exception parsing option {which}")
                        return None

                name = option_parts["name"]
                type = option_parts["type"]
                default = option_parts["default"]
                min = parse_min_max_value(option_parts, "min")
                max = parse_min_max_value(option_parts, "max")

                without_default = Option(name, type, None, min, max, var)
                option = Option(without_default.name, without_default.type, without_default.parse(default), min, max, var)
                self.engine.options[option.name] = option

                if option.default is not None:
                    self.engine.config[option.name] = option.default
                if option.default is not None and not option.is_managed() and option.name.lower() != "uci_analysemode":
                    self.engine.target_config[option.name] = option.default

            def _id(self, arg: str) -> None:
                key, value = _next_token(arg)
                self.engine.id[key] = value.strip()

        return await self.communicate(UciInitializeCommand)

    def _isready(self) -> None:
        self.send_line("isready")

    def _opponent_info(self) -> None:
        opponent_info = self.config.get("UCI_Opponent") or self.target_config.get("UCI_Opponent")
        if opponent_info:
            self.send_line(f"setoption name UCI_Opponent value {opponent_info}")

    def _ucinewgame(self) -> None:
        self.send_line("ucinewgame")
        self._opponent_info()
        self.first_game = False
        self.ponderhit = False

    def debug(self, on: bool = True) -> None:
        """
        Switches debug mode of the engine on or off. This does not interrupt
        other ongoing operations.
        """
        if on:
            self.send_line("debug on")
        else:
            self.send_line("debug off")

    async def ping(self) -> None:
        class UciPingCommand(BaseCommand[None]):
            def __init__(self, engine: UciProtocol) -> None:
                super().__init__(engine)
                self.engine =  engine

            def start(self) -> None:
                self.engine._isready()

            @override
            def line_received(self, line: str) -> None:
                if line.strip() == "readyok":
                    self.result.set_result(None)
                    self.set_finished()
                else:
                    LOGGER.warning("%s: Unexpected engine output: %r", self.engine, line)

        return await self.communicate(UciPingCommand)

    def _changed_options(self, options: ConfigMapping) -> bool:
        return any(value is None or value != self.config.get(name) for name, value in _chain_config(options, self.target_config))

    def _setoption(self, name: str, value: ConfigValue) -> None:
        try:
            value = self.options[name].parse(value)
        except KeyError:
            raise EngineError("engine does not support option {} (available options: {})".format(name, ", ".join(self.options)))

        if value is None or value != self.config.get(name):
            builder = ["setoption name", name]
            if value is False:
                builder.append("value false")
            elif value is True:
                builder.append("value true")
            elif value is not None:
                builder.append("value")
                builder.append(str(value))

            if name != "UCI_Opponent":  # sent after ucinewgame
                self.send_line(" ".join(builder))
            self.config[name] = value

    def _configure(self, options: ConfigMapping) -> None:
        for name, value in _chain_config(options, self.target_config):
            if name.lower() in MANAGED_OPTIONS:
                raise EngineError("cannot set {} which is automatically managed".format(name))
            self._setoption(name, value)

    async def configure(self, options: ConfigMapping) -> None:
        class UciConfigureCommand(BaseCommand[None]):
            def __init__(self, engine: UciProtocol):
                super().__init__(engine)
                self.engine = engine

            def start(self) -> None:
                self.engine._configure(options)
                self.engine.target_config.update({name: value for name, value in options.items() if value is not None})
                self.result.set_result(None)
                self.set_finished()

        return await self.communicate(UciConfigureCommand)

    def _opponent_configuration(self, *, opponent: Optional[Opponent] = None) -> ConfigMapping:
        if opponent and opponent.name and "UCI_Opponent" in self.options:
            rating = opponent.rating or "none"
            title = opponent.title or "none"
            player_type = "computer" if opponent.is_engine else "human"
            return {"UCI_Opponent": f"{title} {rating} {player_type} {opponent.name}"}
        else:
            return {}

    async def send_opponent_information(self, *, opponent: Optional[Opponent] = None, engine_rating: Optional[int] = None) -> None:
        return await self.configure(self._opponent_configuration(opponent=opponent))

    def _position(self, board: chess.Board) -> None:
        # Select UCI_Variant and UCI_Chess960.
        uci_variant = type(board).uci_variant
        if "UCI_Variant" in self.options:
            self._setoption("UCI_Variant", uci_variant)
        elif uci_variant != "chess":
            raise EngineError("engine does not support UCI_Variant")

        if "UCI_Chess960" in self.options:
            self._setoption("UCI_Chess960", board.chess960)
        elif board.chess960:
            raise EngineError("engine does not support UCI_Chess960")

        # Send starting position.
        builder = ["position"]
        safe_history = all(board.move_stack)
        root = board.root() if safe_history else board
        fen = root.fen(shredder=board.chess960, en_passant="fen")
        if uci_variant == "chess" and fen == chess.STARTING_FEN:
            builder.append("startpos")
        else:
            builder.append("fen")
            builder.append(fen)

        # Send moves.
        if not safe_history:
            LOGGER.warning("Not transmitting history with null moves to UCI engine")
        elif board.move_stack:
            builder.append("moves")
            builder.extend(move.uci() for move in board.move_stack)

        self.send_line(" ".join(builder))
        self.board = board.copy(stack=False)

    def _go(self, limit: Limit, *, root_moves: Optional[Iterable[chess.Move]] = None, ponder: bool = False, infinite: bool = False) -> None:
        builder = ["go"]
        if ponder:
            builder.append("ponder")
        if limit.white_clock is not None:
            builder.append("wtime")
            builder.append(str(max(1, round(limit.white_clock * 1000))))
        if limit.black_clock is not None:
            builder.append("btime")
            builder.append(str(max(1, round(limit.black_clock * 1000))))
        if limit.white_inc is not None:
            builder.append("winc")
            builder.append(str(round(limit.white_inc * 1000)))
        if limit.black_inc is not None:
            builder.append("binc")
            builder.append(str(round(limit.black_inc * 1000)))
        if limit.remaining_moves is not None and int(limit.remaining_moves) > 0:
            builder.append("movestogo")
            builder.append(str(int(limit.remaining_moves)))
        if limit.depth is not None:
            builder.append("depth")
            builder.append(str(max(1, int(limit.depth))))
        if limit.nodes is not None:
            builder.append("nodes")
            builder.append(str(max(1, int(limit.nodes))))
        if limit.mate is not None:
            builder.append("mate")
            builder.append(str(max(1, int(limit.mate))))
        if limit.time is not None:
            builder.append("movetime")
            builder.append(str(max(1, round(limit.time * 1000))))
        if infinite:
            builder.append("infinite")
        if root_moves is not None:
            builder.append("searchmoves")
            if root_moves:
                builder.extend(move.uci() for move in root_moves)
            else:
                # Work around searchmoves followed by nothing.
                builder.append("0000")
        self.send_line(" ".join(builder))

    async def play(self, board: chess.Board, limit: Limit, *, game: object = None, info: Info = INFO_NONE, ponder: bool = False, draw_offered: bool = False, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}, opponent: Optional[Opponent] = None) -> PlayResult:
        new_options: Dict[str, ConfigValue] = {}
        for name, value in options.items():
            new_options[name] = value
        new_options.update(self._opponent_configuration(opponent=opponent))

        engine = self

        class UciPlayCommand(BaseCommand[PlayResult]):
            def __init__(self, engine: UciProtocol):
                super().__init__(engine)
                self.engine = engine

                # May ponderhit only in the same game and with unchanged target
                # options. The managed options UCI_AnalyseMode, Ponder, and
                # MultiPV never change between pondering play commands.
                engine.may_ponderhit = board if ponder and not engine.first_game and game == engine.game and not engine._changed_options(new_options) else None

            @override
            def start(self) -> None:
                self.info: InfoDict = {}
                self.pondering: Optional[chess.Board] = None
                self.sent_isready = False
                self.start_time = time.perf_counter()

                if self.engine.ponderhit:
                    self.engine.ponderhit = False
                    self.engine.send_line("ponderhit")
                    return

                if "UCI_AnalyseMode" in self.engine.options and "UCI_AnalyseMode" not in self.engine.target_config and all(name.lower() != "uci_analysemode" for name in new_options):
                    self.engine._setoption("UCI_AnalyseMode", False)
                if "Ponder" in self.engine.options:
                    self.engine._setoption("Ponder", ponder)
                if "MultiPV" in self.engine.options:
                    self.engine._setoption("MultiPV", self.engine.options["MultiPV"].default)

                new_opponent = new_options.get("UCI_Opponent") or self.engine.target_config.get("UCI_Opponent")
                opponent_changed = new_opponent != self.engine.config.get("UCI_Opponent")
                self.engine._configure(new_options)

                if self.engine.first_game or self.engine.game != game or opponent_changed:
                    self.engine.game = game
                    self.engine._ucinewgame()
                    self.sent_isready = True
                    self.engine._isready()
                else:
                    self._readyok()

            @override
            def line_received(self, line: str) -> None:
                token, remaining = _next_token(line)
                if token == "info":
                    self._info(remaining)
                elif token == "bestmove":
                    self._bestmove(remaining)
                elif line.strip() == "readyok" and self.sent_isready:
                    self._readyok()
                else:
                    LOGGER.warning("%s: Unexpected engine output: %r", self.engine, line)

            def _readyok(self) -> None:
                self.sent_isready = False
                engine._position(board)
                engine._go(limit, root_moves=root_moves)

            def _info(self, arg: str) -> None:
                if not self.pondering:
                    self.info.update(_parse_uci_info(arg, self.engine.board, info))

            def _bestmove(self, arg: str) -> None:
                if self.pondering:
                    self.pondering = None
                elif not self.result.cancelled():
                    best = _parse_uci_bestmove(self.engine.board, arg)
                    self.result.set_result(PlayResult(best.move, best.ponder, self.info))

                    if ponder and best.move and best.ponder:
                        self.pondering = board.copy()
                        self.pondering.push(best.move)
                        self.pondering.push(best.ponder)
                        self.engine._position(self.pondering)

                        # Adjust clocks for pondering.
                        time_used = time.perf_counter() - self.start_time
                        ponder_limit = copy.copy(limit)
                        if ponder_limit.white_clock is not None:
                            ponder_limit.white_clock += (ponder_limit.white_inc or 0.0)
                            if self.pondering.turn == chess.WHITE:
                                ponder_limit.white_clock -= time_used
                        if ponder_limit.black_clock is not None:
                            ponder_limit.black_clock += (ponder_limit.black_inc or 0.0)
                            if self.pondering.turn == chess.BLACK:
                                ponder_limit.black_clock -= time_used
                        if ponder_limit.remaining_moves:
                            ponder_limit.remaining_moves -= 1

                        self.engine._go(ponder_limit, ponder=True)

                if not self.pondering:
                    self.end()

            def end(self) -> None:
                engine.may_ponderhit = None
                self.set_finished()

            @override
            def cancel(self) -> None:
                if self.engine.may_ponderhit and self.pondering and self.engine.may_ponderhit.move_stack == self.pondering.move_stack and self.engine.may_ponderhit == self.pondering:
                    self.engine.ponderhit = True
                    self.end()
                else:
                    self.engine.send_line("stop")

            @override
            def engine_terminated(self, exc: Exception) -> None:
                # Allow terminating engine while pondering.
                if not self.result.done():
                    super().engine_terminated(exc)

        return await self.communicate(UciPlayCommand)

    async def analysis(self, board: chess.Board, limit: Optional[Limit] = None, *, multipv: Optional[int] = None, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> AnalysisResult:
        class UciAnalysisCommand(BaseCommand[AnalysisResult]):
            def __init__(self, engine: UciProtocol):
                super().__init__(engine)
                self.engine = engine

            def start(self) -> None:
                self.analysis = AnalysisResult(stop=lambda: self.cancel())
                self.sent_isready = False

                if "Ponder" in self.engine.options:
                    self.engine._setoption("Ponder", False)
                if "UCI_AnalyseMode" in self.engine.options and "UCI_AnalyseMode" not in self.engine.target_config and all(name.lower() != "uci_analysemode" for name in options):
                    self.engine._setoption("UCI_AnalyseMode", True)
                if "MultiPV" in self.engine.options or (multipv and multipv > 1):
                    self.engine._setoption("MultiPV", 1 if multipv is None else multipv)

                self.engine._configure(options)

                if self.engine.first_game or self.engine.game != game:
                    self.engine.game = game
                    self.engine._ucinewgame()
                    self.sent_isready = True
                    self.engine._isready()
                else:
                    self._readyok()

            @override
            def line_received(self, line: str) -> None:
                token, remaining = _next_token(line)
                if token == "info":
                    self._info(remaining)
                elif token == "bestmove":
                    self._bestmove(remaining)
                elif line.strip() == "readyok" and self.sent_isready:
                    self._readyok()
                else:
                    LOGGER.warning("%s: Unexpected engine output: %r", self.engine, line)

            def _readyok(self) -> None:
                self.sent_isready = False
                self.engine._position(board)

                if limit:
                    self.engine._go(limit, root_moves=root_moves)
                else:
                    self.engine._go(Limit(), root_moves=root_moves, infinite=True)

                self.result.set_result(self.analysis)

            def _info(self, arg: str) -> None:
                self.analysis.post(_parse_uci_info(arg, self.engine.board, info))

            def _bestmove(self, arg: str) -> None:
                if not self.result.done():
                    raise EngineError("was not searching, but engine sent bestmove")
                best = _parse_uci_bestmove(self.engine.board, arg)
                self.set_finished()
                self.analysis.set_finished(best)

            @override
            def cancel(self) -> None:
                self.engine.send_line("stop")

            @override
            def engine_terminated(self, exc: Exception) -> None:
                LOGGER.debug("%s: Closing analysis because engine has been terminated (error: %s)", self.engine, exc)
                self.analysis.set_exception(exc)

        return await self.communicate(UciAnalysisCommand)

    async def send_game_result(self, board: chess.Board, winner: Optional[Color] = None, game_ending: Optional[str] = None, game_complete: bool = True) -> None:
        pass

    async def quit(self) -> None:
        self.send_line("quit")
        await asyncio.shield(self.returncode)


UCI_REGEX = re.compile(r"^[a-h][1-8][a-h][1-8][pnbrqk]?|[PNBRQK]@[a-h][1-8]|0000\Z")

def _create_variation_line(root_board: chess.Board, line: str) -> tuple[list[chess.Move], str]:
    board = root_board.copy(stack=False)
    currline: list[chess.Move] = []
    while True:
        next_move, remaining_line_after_move = _next_token(line)
        if UCI_REGEX.match(next_move):
            currline.append(board.push_uci(next_move))
            line = remaining_line_after_move
        else:
            return currline, line


def _parse_uci_info(arg: str, root_board: chess.Board, selector: Info = INFO_ALL) -> InfoDict:
    info: InfoDict = {}
    if not selector:
        return info

    remaining_line = arg
    while remaining_line:
        parameter, remaining_line = _next_token(remaining_line)

        if parameter == "string":
            info["string"] = remaining_line
            break
        elif parameter in ["depth", "seldepth", "nodes", "multipv", "currmovenumber", "hashfull", "nps", "tbhits", "cpuload"]:
            try:
                number, remaining_line = _next_token(remaining_line)
                info[parameter] = int(number)  # type: ignore
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing %s from info: %r", parameter, arg)
        elif parameter == "time":
            try:
                time_ms, remaining_line = _next_token(remaining_line)
                info["time"] = int(time_ms) / 1000.0
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing %s from info: %r", parameter, arg)
        elif parameter == "ebf":
            try:
                number, remaining_line = _next_token(remaining_line)
                info["ebf"] = float(number)
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing %s from info: %r", parameter, arg)
        elif parameter == "score" and selector & INFO_SCORE:
            try:
                kind, remaining_line = _next_token(remaining_line)
                value, remaining_line = _next_token(remaining_line)
                token, remaining_after_token = _next_token(remaining_line)
                if token in ["lowerbound", "upperbound"]:
                    info[token] = True  # type: ignore
                    remaining_line = remaining_after_token
                if kind == "cp":
                    info["score"] = PovScore(Cp(int(value)), root_board.turn)
                elif kind == "mate":
                    info["score"] = PovScore(Mate(int(value)), root_board.turn)
                else:
                    LOGGER.error("Unknown score kind %r in info (expected cp or mate): %r", kind, arg)
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing score from info: %r", arg)
        elif parameter == "currmove":
            try:
                current_move, remaining_line = _next_token(remaining_line)
                info["currmove"] = chess.Move.from_uci(current_move)
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing currmove from info: %r", arg)
        elif parameter == "currline" and selector & INFO_CURRLINE:
            try:
                if "currline" not in info:
                    info["currline"] = {}

                cpunr_text, remaining_line = _next_token(remaining_line)
                cpunr = int(cpunr_text)
                currline, remaining_line = _create_variation_line(root_board, remaining_line)
                info["currline"][cpunr] = currline
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing currline from info: %r, position at root: %s", arg, root_board.fen())
        elif parameter == "refutation" and selector & INFO_REFUTATION:
            try:
                if "refutation" not in info:
                    info["refutation"] = {}

                board = root_board.copy(stack=False)
                refuted_text, remaining_line = _next_token(remaining_line)
                refuted = board.push_uci(refuted_text)

                refuted_by, remaining_line = _create_variation_line(board, remaining_line)
                info["refutation"][refuted] = refuted_by
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing refutation from info: %r, position at root: %s", arg, root_board.fen())
        elif parameter == "pv" and selector & INFO_PV:
            try:
                pv, remaining_line = _create_variation_line(root_board, remaining_line)
                info["pv"] = pv
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing pv from info: %r, position at root: %s", arg, root_board.fen())
        elif parameter == "wdl":
            try:
                wins, remaining_line = _next_token(remaining_line)
                draws, remaining_line = _next_token(remaining_line)
                losses, remaining_line = _next_token(remaining_line)
                info["wdl"] = PovWdl(Wdl(int(wins), int(draws), int(losses)), root_board.turn)
            except (ValueError, IndexError):
                LOGGER.error("Exception parsing wdl from info: %r", arg)

    return info

def _parse_uci_bestmove(board: chess.Board, args: str) -> BestMove:
    tokens = args.split()

    move = None
    ponder = None

    if tokens and tokens[0] not in ["(none)", "NULL"]:
        try:
            # AnMon 5.75 uses uppercase letters to denote promotion types.
            move = board.push_uci(tokens[0].lower())
        except ValueError as err:
            raise EngineError(err)

        try:
            # Houdini 1.5 sends NULL instead of skipping the token.
            if len(tokens) >= 3 and tokens[1] == "ponder" and tokens[2] not in ["(none)", "NULL"]:
                ponder = board.parse_uci(tokens[2].lower())
        except ValueError:
            LOGGER.exception("Engine sent invalid ponder move")
        finally:
            board.pop()

    return BestMove(move, ponder)


def _chain_config(a: ConfigMapping, b: ConfigMapping) -> Iterator[Tuple[str, ConfigValue]]:
    for name, value in a.items():
        yield name, value
    for name, value in b.items():
        if name not in a:
            yield name, value


class UciOptionMap(MutableMapping[str, T]):
    """Dictionary with case-insensitive keys."""

    def __init__(self, data: Optional[Iterable[Tuple[str, T]]] = None, **kwargs: T) -> None:
        self._store: Dict[str, Tuple[str, T]] = {}
        if data is None:
            data = {}
        self.update(data, **kwargs)

    def __setitem__(self, key: str, value: T) -> None:
        self._store[key.lower()] = (key, value)

    def __getitem__(self, key: str) -> T:
        return self._store[key.lower()][1]

    def __delitem__(self, key: str) -> None:
        del self._store[key.lower()]

    def __iter__(self) -> Iterator[str]:
        return (casedkey for casedkey, _ in self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __eq__(self, other: object) -> bool:
        try:
            for key, value in self.items():
                if key not in other or other[key] != value:  # type: ignore
                    return False

            for key, value in other.items():  # type: ignore
                if key not in self or self[key] != value:
                    return False

            return True
        except (TypeError, AttributeError):
            return NotImplemented

    def copy(self) -> UciOptionMap[T]:
        return type(self)(self._store.values())

    def __copy__(self) -> UciOptionMap[T]:
        return self.copy()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self.items())!r})"


XBOARD_ERROR_REGEX = re.compile(r"^\s*(Error|Illegal move)(\s*\([^()]+\))?\s*:")


class XBoardProtocol(Protocol):
    """
    An implementation of the
    `XBoard protocol <http://hgm.nubati.net/CECP.html>`__ (CECP).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features: Dict[str, Union[int, str]] = {}
        self.id = {}
        self.options = {
            "random": Option("random", "check", False, None, None, None),
            "computer": Option("computer", "check", False, None, None, None),
            "name": Option("name", "string", "", None, None, None),
            "engine_rating": Option("engine_rating", "spin", 0, None, None, None),
            "opponent_rating": Option("opponent_rating", "spin", 0, None, None, None)
        }
        self.config: Dict[str, ConfigValue] = {}
        self.target_config: Dict[str, ConfigValue] = {}
        self.board = chess.Board()
        self.game: object = None
        self.clock_id: object = None
        self.first_game = True

    async def initialize(self) -> None:
        class XBoardInitializeCommand(BaseCommand[None]):
            def __init__(self, engine: XBoardProtocol):
                super().__init__(engine)
                self.engine = engine

            @override
            def check_initialized(self) -> None:
                if self.engine.initialized:
                    raise EngineError("engine already initialized")

            @override
            def start(self) -> None:
                self.engine.send_line("xboard")
                self.engine.send_line("protover 2")
                self.timeout_handle = self.engine.loop.call_later(2.0, lambda: self.timeout())

            def timeout(self) -> None:
                LOGGER.error("%s: Timeout during initialization", self.engine)
                self.end()

            @override
            def line_received(self, line: str) -> None:
                token, remaining = _next_token(line)
                if token.startswith("#"):
                    pass
                elif token == "feature":
                    self._feature(remaining)
                elif XBOARD_ERROR_REGEX.match(line):
                    raise EngineError(line)

            def _feature(self, arg: str) -> None:
                for feature in shlex.split(arg):
                    key, value = feature.split("=", 1)
                    if key == "option":
                        option = _parse_xboard_option(value)
                        if option.name not in ["random", "computer", "cores", "memory"]:
                            self.engine.options[option.name] = option
                    else:
                        try:
                            self.engine.features[key] = int(value)
                        except ValueError:
                            self.engine.features[key] = value

                if "done" in self.engine.features:
                    self.timeout_handle.cancel()
                if self.engine.features.get("done"):
                    self.end()

            def end(self) -> None:
                if not self.engine.features.get("ping", 0):
                    self.result.set_exception(EngineError("xboard engine did not declare required feature: ping"))
                    self.set_finished()
                    return
                if not self.engine.features.get("setboard", 0):
                    self.result.set_exception(EngineError("xboard engine did not declare required feature: setboard"))
                    self.set_finished()
                    return

                if not self.engine.features.get("reuse", 1):
                    LOGGER.warning("%s: Rejecting feature reuse=0", self.engine)
                    self.engine.send_line("rejected reuse")
                if not self.engine.features.get("sigterm", 1):
                    LOGGER.warning("%s: Rejecting feature sigterm=0", self.engine)
                    self.engine.send_line("rejected sigterm")
                if self.engine.features.get("san", 0):
                    LOGGER.warning("%s: Rejecting feature san=1", self.engine)
                    self.engine.send_line("rejected san")

                if "myname" in self.engine.features:
                    self.engine.id["name"] = str(self.engine.features["myname"])

                if self.engine.features.get("memory", 0):
                    self.engine.options["memory"] = Option("memory", "spin", 16, 1, None, None)
                    self.engine.send_line("accepted memory")
                if self.engine.features.get("smp", 0):
                    self.engine.options["cores"] = Option("cores", "spin", 1, 1, None, None)
                    self.engine.send_line("accepted smp")
                if self.engine.features.get("egt"):
                    for egt in str(self.engine.features["egt"]).split(","):
                        name = f"egtpath {egt}"
                        self.engine.options[name] = Option(name, "path", None, None, None, None)
                    self.engine.send_line("accepted egt")

                for option in self.engine.options.values():
                    if option.default is not None:
                        self.engine.config[option.name] = option.default
                    if option.default is not None and not option.is_managed():
                        self.engine.target_config[option.name] = option.default

                self.engine.initialized = True
                self.result.set_result(None)
                self.set_finished()

        return await self.communicate(XBoardInitializeCommand)

    def _ping(self, n: int) -> None:
        self.send_line(f"ping {n}")

    def _variant(self, variant: Optional[str]) -> None:
        variants = str(self.features.get("variants", "")).split(",")
        if not variant or variant not in variants:
            raise EngineError("unsupported xboard variant: {} (available: {})".format(variant, ", ".join(variants)))

        self.send_line(f"variant {variant}")

    def _new(self, board: chess.Board, game: object, options: ConfigMapping, opponent: Optional[Opponent] = None) -> None:
        self._configure(options)
        self._configure(self._opponent_configuration(opponent=opponent))

        # Set up starting position.
        root = board.root()
        new_options = any(param in options for param in ("random", "computer"))
        new_game = self.first_game or self.game != game or new_options or opponent or root != self.board.root()
        self.game = game
        self.first_game = False
        if new_game:
            self.board = root
            self.send_line("new")

            variant = type(board).xboard_variant
            if variant == "normal" and board.chess960:
                self._variant("fischerandom")
            elif variant != "normal":
                self._variant(variant)

            if self.config.get("random"):
                self.send_line("random")

            opponent_name = self.config.get("name")
            if opponent_name and self.features.get("name", True):
                self.send_line(f"name {opponent_name}")

            opponent_rating = self.config.get("opponent_rating")
            engine_rating = self.config.get("engine_rating")
            if engine_rating or opponent_rating:
                self.send_line(f"rating {engine_rating or 0} {opponent_rating or 0}")

            if self.config.get("computer"):
                self.send_line("computer")

            self.send_line("force")

            fen = root.fen(shredder=board.chess960, en_passant="fen")
            if variant != "normal" or fen != chess.STARTING_FEN or board.chess960:
                self.send_line(f"setboard {fen}")
        else:
            self.send_line("force")

        # Undo moves until common position.
        common_stack_len = 0
        if not new_game:
            for left, right in zip(self.board.move_stack, board.move_stack):
                if left == right:
                    common_stack_len += 1
                else:
                    break

            while len(self.board.move_stack) > common_stack_len + 1:
                self.send_line("remove")
                self.board.pop()
                self.board.pop()

            while len(self.board.move_stack) > common_stack_len:
                self.send_line("undo")
                self.board.pop()

        # Play moves from board stack.
        for move in board.move_stack[common_stack_len:]:
            if not move:
                LOGGER.warning("Null move (in %s) may not be supported by all XBoard engines", self.board.fen())
            prefix = "usermove " if self.features.get("usermove", 0) else ""
            self.send_line(prefix + self.board.xboard(move))
            self.board.push(move)

    async def ping(self) -> None:
        class XBoardPingCommand(BaseCommand[None]):
            def __init__(self, engine: XBoardProtocol):
                super().__init__(engine)
                self.engine = engine

            @override
            def start(self) -> None:
                n = id(self) & 0xffff
                self.pong = f"pong {n}"
                self.engine._ping(n)

            @override
            def line_received(self, line: str) -> None:
                if line == self.pong:
                    self.result.set_result(None)
                    self.set_finished()
                elif not line.startswith("#"):
                    LOGGER.warning("%s: Unexpected engine output: %r", self.engine, line)
                elif XBOARD_ERROR_REGEX.match(line):
                    raise EngineError(line)

        return await self.communicate(XBoardPingCommand)

    async def play(self, board: chess.Board, limit: Limit, *, game: object = None, info: Info = INFO_NONE, ponder: bool = False, draw_offered: bool = False, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}, opponent: Optional[Opponent] = None) -> PlayResult:
        if root_moves is not None:
            raise EngineError("play with root_moves, but xboard supports 'include' only in analysis mode")

        class XBoardPlayCommand(BaseCommand[PlayResult]):
            def __init__(self, engine: XBoardProtocol):
                super().__init__(engine)
                self.engine = engine

            @override
            def start(self) -> None:
                self.play_result = PlayResult(None, None)
                self.stopped = False
                self.pong_after_move: Optional[str] = None
                self.pong_after_ponder: Optional[str] = None

                # Set game, position and configure.
                self.engine._new(board, game, options, opponent)

                # Limit or time control.
                clock = limit.white_clock if board.turn else limit.black_clock
                increment = limit.white_inc if board.turn else limit.black_inc
                if limit.clock_id is None or limit.clock_id != self.engine.clock_id:
                    self._send_time_control(clock, increment)
                self.engine.clock_id = limit.clock_id
                if limit.nodes is not None:
                    if limit.time is not None or limit.white_clock is not None or limit.black_clock is not None or increment is not None:
                        raise EngineError("xboard does not support mixing node limits with time limits")

                    if "nps" not in self.engine.features:
                        LOGGER.warning("%s: Engine did not explicitly declare support for node limits (feature nps=?)")
                    elif not self.engine.features["nps"]:
                        raise EngineError("xboard engine does not support node limits (feature nps=0)")

                    self.engine.send_line("nps 1")
                    self.engine.send_line(f"st {max(1, int(limit.nodes))}")
                if limit.depth is not None:
                    self.engine.send_line(f"sd {max(1, int(limit.depth))}")
                if limit.white_clock is not None:
                    self.engine.send_line("{} {}".format("time" if board.turn else "otim", max(1, round(limit.white_clock * 100))))
                if limit.black_clock is not None:
                    self.engine.send_line("{} {}".format("otim" if board.turn else "time", max(1, round(limit.black_clock * 100))))

                if draw_offered and self.engine.features.get("draw", 1):
                    self.engine.send_line("draw")

                # Start thinking.
                self.engine.send_line("post" if info else "nopost")
                self.engine.send_line("hard" if ponder else "easy")
                self.engine.send_line("go")

            @override
            def line_received(self, line: str) -> None:
                token, remaining = _next_token(line)
                if token == "move":
                    self._move(remaining.strip())
                elif token == "Hint:":
                    self._hint(remaining.strip())
                elif token == "pong":
                    pong_line = f"{token} {remaining.strip()}"
                    if pong_line == self.pong_after_move:
                        if not self.result.done():
                            self.result.set_result(self.play_result)
                        if not ponder:
                            self.set_finished()
                    elif pong_line == self.pong_after_ponder:
                        if not self.result.done():
                            self.result.set_result(self.play_result)
                        self.set_finished()
                elif f"{token} {remaining.strip()}" == "offer draw":
                    if not self.result.done():
                        self.play_result.draw_offered = True
                    self._ping_after_move()
                elif line.strip() == "resign":
                    if not self.result.done():
                        self.play_result.resigned = True
                    self._ping_after_move()
                elif token in ["1-0", "0-1", "1/2-1/2"]:
                    if "resign" in line and not self.result.done():
                        self.play_result.resigned = True
                    self._ping_after_move()
                elif token.startswith("#"):
                    pass
                elif XBOARD_ERROR_REGEX.match(line):
                    self.engine.first_game = True  # Board state might no longer be in sync
                    raise EngineError(line)
                elif len(line.split()) >= 4 and line.lstrip()[0].isdigit():
                    self._post(line)
                else:
                    LOGGER.warning("%s: Unexpected engine output: %r", self.engine, line)

            def _send_time_control(self, clock: Optional[float], increment: Optional[float]) -> None:
                if limit.remaining_moves or clock is not None or increment is not None:
                    base_mins, base_secs = divmod(int(clock or 0), 60)
                    self.engine.send_line(f"level {limit.remaining_moves or 0} {base_mins}:{base_secs:02d} {increment or 0}")
                if limit.time is not None:
                    self.engine.send_line(f"st {max(0.01, limit.time)}")

            def _post(self, line: str) -> None:
                if not self.result.done():
                    self.play_result.info = _parse_xboard_post(line, self.engine.board, info)

            def _move(self, arg: str) -> None:
                if not self.result.done() and self.play_result.move is None:
                    try:
                        self.play_result.move = self.engine.board.push_xboard(arg)
                    except ValueError as err:
                        self.result.set_exception(EngineError(err))
                    else:
                        self._ping_after_move()
                else:
                    try:
                        self.engine.board.push_xboard(arg)
                    except ValueError:
                        LOGGER.exception("Exception playing unexpected move")

            def _hint(self, arg: str) -> None:
                if not self.result.done() and self.play_result.move is not None and self.play_result.ponder is None:
                    try:
                        self.play_result.ponder = self.engine.board.parse_xboard(arg)
                    except ValueError:
                        LOGGER.exception("Exception parsing hint")
                else:
                    LOGGER.warning("Unexpected hint: %r", arg)

            def _ping_after_move(self) -> None:
                if self.pong_after_move is None:
                    n = id(self) & 0xffff
                    self.pong_after_move = f"pong {n}"
                    self.engine._ping(n)

            @override
            def cancel(self) -> None:
                if self.stopped:
                    return
                self.stopped = True

                if self.result.cancelled():
                    self.engine.send_line("?")

                if ponder:
                    self.engine.send_line("easy")

                    n = (id(self) + 1) & 0xffff
                    self.pong_after_ponder = f"pong {n}"
                    self.engine._ping(n)

            @override
            def engine_terminated(self, exc: Exception) -> None:
                # Allow terminating engine while pondering.
                if not self.result.done():
                    super().engine_terminated(exc)

        return await self.communicate(XBoardPlayCommand)

    async def analysis(self, board: chess.Board, limit: Optional[Limit] = None, *, multipv: Optional[int] = None, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> AnalysisResult:
        if multipv is not None:
            raise EngineError("xboard engine does not support multipv")

        if limit is not None and (limit.white_clock is not None or limit.black_clock is not None):
            raise EngineError("xboard analysis does not support clock limits")

        class XBoardAnalysisCommand(BaseCommand[AnalysisResult]):
            def __init__(self, engine: XBoardProtocol):
                super().__init__(engine)
                self.engine = engine

            @override
            def start(self) -> None:
                self.stopped = False
                self.best_move: Optional[chess.Move] = None
                self.analysis = AnalysisResult(stop=lambda: self.cancel())
                self.final_pong: Optional[str] = None

                self.engine._new(board, game, options)

                if root_moves is not None:
                    if not self.engine.features.get("exclude", 0):
                        raise EngineError("xboard engine does not support root_moves (feature exclude=0)")

                    self.engine.send_line("exclude all")
                    for move in root_moves:
                        self.engine.send_line(f"include {self.engine.board.xboard(move)}")

                self.engine.send_line("post")
                self.engine.send_line("analyze")

                self.result.set_result(self.analysis)

                if limit is not None and limit.time is not None:
                    self.time_limit_handle: Optional[asyncio.Handle] = self.engine.loop.call_later(limit.time, lambda: self.cancel())
                else:
                    self.time_limit_handle = None

            @override
            def line_received(self, line: str) -> None:
                token, remaining = _next_token(line)
                if token.startswith("#"):
                    pass
                elif len(line.split()) >= 4 and line.lstrip()[0].isdigit():
                    self._post(line)
                elif f"{token} {remaining.strip()}" == self.final_pong:
                    self.end()
                elif XBOARD_ERROR_REGEX.match(line):
                    self.engine.first_game = True  # Board state might no longer be in sync
                    raise EngineError(line)
                else:
                    LOGGER.warning("%s: Unexpected engine output: %r", self.engine, line)

            def _post(self, line: str) -> None:
                post_info = _parse_xboard_post(line, self.engine.board, info)
                self.analysis.post(post_info)

                pv = post_info.get("pv")
                if pv:
                    self.best_move = pv[0]

                if limit is not None:
                    if limit.time is not None and post_info.get("time", 0) >= limit.time:
                        self.cancel()
                    elif limit.nodes is not None and post_info.get("nodes", 0) >= limit.nodes:
                        self.cancel()
                    elif limit.depth is not None and post_info.get("depth", 0) >= limit.depth:
                        self.cancel()
                    elif limit.mate is not None and "score" in post_info:
                        if post_info["score"].relative >= Mate(limit.mate):
                            self.cancel()

            def end(self) -> None:
                if self.time_limit_handle:
                    self.time_limit_handle.cancel()

                self.set_finished()
                self.analysis.set_finished(BestMove(self.best_move, None))

            @override
            def cancel(self) -> None:
                if self.stopped:
                    return
                self.stopped = True

                self.engine.send_line(".")
                self.engine.send_line("exit")

                n = id(self) & 0xffff
                self.final_pong = f"pong {n}"
                self.engine._ping(n)

            @override
            def engine_terminated(self, exc: Exception) -> None:
                LOGGER.debug("%s: Closing analysis because engine has been terminated (error: %s)", self.engine, exc)

                if self.time_limit_handle:
                    self.time_limit_handle.cancel()

                self.analysis.set_exception(exc)

        return await self.communicate(XBoardAnalysisCommand)

    def _setoption(self, name: str, value: ConfigValue) -> None:
        if value is not None and value == self.config.get(name):
            return

        try:
            option = self.options[name]
        except KeyError:
            raise EngineError(f"unsupported xboard option or command: {name}")

        self.config[name] = value = option.parse(value)

        if name in ["random", "computer", "name", "engine_rating", "opponent_rating"]:
            # Applied in _new.
            pass
        elif name in ["memory", "cores"] or name.startswith("egtpath "):
            self.send_line(f"{name} {value}")
        elif value is None:
            self.send_line(f"option {name}")
        elif value is True:
            self.send_line(f"option {name}=1")
        elif value is False:
            self.send_line(f"option {name}=0")
        else:
            self.send_line(f"option {name}={value}")

    def _configure(self, options: ConfigMapping) -> None:
        for name, value in _chain_config(options, self.target_config):
            if name.lower() in MANAGED_OPTIONS:
                raise EngineError(f"cannot set {name} which is automatically managed")
            self._setoption(name, value)

    async def configure(self, options: ConfigMapping) -> None:
        class XBoardConfigureCommand(BaseCommand[None]):
            def __init__(self, engine: XBoardProtocol):
                super().__init__(engine)
                self.engine = engine

            @override
            def start(self) -> None:
                self.engine._configure(options)
                self.engine.target_config.update({name: value for name, value in options.items() if value is not None})
                self.result.set_result(None)
                self.set_finished()

        return await self.communicate(XBoardConfigureCommand)

    def _opponent_configuration(self, *, opponent: Optional[Opponent] = None, engine_rating: Optional[int] = None) -> ConfigMapping:
        if opponent is None:
            return {}

        opponent_info: Dict[str, Union[int, bool, str]] = {"engine_rating": engine_rating or self.target_config.get("engine_rating") or 0,
                                                           "opponent_rating": opponent.rating or 0,
                                                           "computer": opponent.is_engine or False}

        if opponent.name and self.features.get("name", True):
            opponent_info["name"] = f"{opponent.title or ''} {opponent.name}".strip()

        return opponent_info

    async def send_opponent_information(self, *, opponent: Optional[Opponent] = None, engine_rating: Optional[int] = None) -> None:
        return await self.configure(self._opponent_configuration(opponent=opponent, engine_rating=engine_rating))

    async def send_game_result(self, board: chess.Board, winner: Optional[Color] = None, game_ending: Optional[str] = None, game_complete: bool = True) -> None:
        class XBoardGameResultCommand(BaseCommand[None]):
            def __init__(self, engine: XBoardProtocol):
                super().__init__(engine)
                self.engine = engine

            @override
            def start(self) -> None:
                if game_ending and any(c in game_ending for c in "{}\n\r"):
                    raise EngineError(f"invalid line break or curly braces in game ending message: {game_ending!r}")

                self.engine._new(board, self.engine.game, {})  # Send final moves to engine.

                outcome = board.outcome(claim_draw=True)

                if not game_complete:
                    result = "*"
                    ending = game_ending or ""
                elif winner is not None or game_ending:
                    result = "1-0" if winner == chess.WHITE else "0-1" if winner == chess.BLACK else "1/2-1/2"
                    ending = game_ending or ""
                elif outcome is not None and outcome.winner is not None:
                    result = outcome.result()
                    winning_color = "White" if outcome.winner == chess.WHITE else "Black"
                    is_checkmate = outcome.termination == chess.Termination.CHECKMATE
                    ending = f"{winning_color} {'mates' if is_checkmate else 'variant win'}"
                elif outcome is not None:
                    result = outcome.result()
                    ending = outcome.termination.name.capitalize().replace("_", " ")
                else:
                    result = "*"
                    ending = ""

                ending_text = f"{{{ending}}}" if ending else ""
                self.engine.send_line(f"result {result} {ending_text}".strip())
                self.result.set_result(None)
                self.set_finished()

        return await self.communicate(XBoardGameResultCommand)

    async def quit(self) -> None:
        self.send_line("quit")
        await asyncio.shield(self.returncode)


def _parse_xboard_option(feature: str) -> Option:
    params = feature.split()

    name = params[0]
    type = params[1][1:]
    default: Optional[ConfigValue] = None
    min = None
    max = None
    var = None

    if type == "combo":
        var = []
        choices = params[2:]
        for choice in choices:
            if choice == "///":
                continue
            elif choice[0] == "*":
                default = choice[1:]
                var.append(choice[1:])
            else:
                var.append(choice)
    elif type == "check":
        default = int(params[2])
    elif type in ["string", "file", "path"]:
        if len(params) > 2:
            default = params[2]
        else:
            default = ""
    elif type == "spin":
        default = int(params[2])
        min = int(params[3])
        max = int(params[4])

    return Option(name, type, default, min, max, var)


def _parse_xboard_post(line: str, root_board: chess.Board, selector: Info = INFO_ALL) -> InfoDict:
    # Format: depth score time nodes [seldepth [nps [tbhits]]] pv
    info: InfoDict = {}

    # Split leading integer tokens from pv.
    pv_tokens = line.split()
    integer_tokens = []
    while pv_tokens:
        token = pv_tokens.pop(0)
        try:
            integer_tokens.append(int(token))
        except ValueError:
            pv_tokens.insert(0, token)
            break

    if len(integer_tokens) < 4:
        return info

    # Required integer tokens.
    info["depth"] = integer_tokens.pop(0)
    cp = integer_tokens.pop(0)
    info["time"] = int(integer_tokens.pop(0)) / 100
    info["nodes"] = int(integer_tokens.pop(0))

    # Score.
    if cp <= -100000:
        score: Score = Mate(cp + 100000)
    elif cp == 100000:
        score = MateGiven
    elif cp >= 100000:
        score = Mate(cp - 100000)
    else:
        score = Cp(cp)
    info["score"] = PovScore(score, root_board.turn)

    # Optional integer tokens.
    if integer_tokens:
        info["seldepth"] = integer_tokens.pop(0)
    if integer_tokens:
        info["nps"] = integer_tokens.pop(0)

    while len(integer_tokens) > 1:
        # Reserved for future extensions.
        integer_tokens.pop(0)

    if integer_tokens:
        info["tbhits"] = integer_tokens.pop(0)

    # Principal variation.
    pv = []
    board = root_board.copy(stack=False)
    for token in pv_tokens:
        if token.rstrip(".").isdigit():
            continue

        try:
            pv.append(board.push_xboard(token))
        except ValueError:
            break

        if not (selector & INFO_PV):
            break
    info["pv"] = pv

    return info


def _next_token(line: str) -> tuple[str, str]:
    """
    Get the next token in a whitespace-delimited line of text.

    The result is returned as a 2-part tuple of strings.

    If the input line is empty or all whitespace, then the result is two
    empty strings.

    If the input line is not empty and not completely whitespace, then
    the first element of the returned tuple is a single word with
    leading and trailing whitespace removed. The second element is the
    unchanged rest of the line.
    """
    parts = line.split(maxsplit=1)
    return parts[0] if parts else "", parts[1] if len(parts) == 2 else ""


class BestMove:
    """Returned by :func:`chess.engine.AnalysisResult.wait()`."""

    move: Optional[chess.Move]
    """The best move according to the engine, or ``None``."""

    ponder: Optional[chess.Move]
    """The response that the engine expects after *move*, or ``None``."""

    def __init__(self, move: Optional[chess.Move], ponder: Optional[chess.Move]):
        self.move = move
        self.ponder = ponder

    def __repr__(self) -> str:
        return "<{} at {:#x} (move={}, ponder={}>".format(
            type(self).__name__, id(self), self.move, self.ponder)


class AnalysisResult:
    """
    Handle to ongoing engine analysis.
    Returned by :func:`chess.engine.Protocol.analysis()`.

    Can be used to asynchronously iterate over information sent by the engine.

    Automatically stops the analysis when used as a context manager.
    """

    multipv: List[InfoDict]
    """
    A list of dictionaries with aggregated information sent by the engine.
    One item for each root move.
    """

    def __init__(self, stop: Optional[Callable[[], None]] = None):
        self._stop = stop
        self._queue: asyncio.Queue[InfoDict] = asyncio.Queue()
        self._posted_kork = False
        self._seen_kork = False
        self._finished: asyncio.Future[BestMove] = asyncio.Future()
        self.multipv = [{}]

    def post(self, info: InfoDict) -> None:
        # Empty dictionary reserved for kork.
        if not info:
            return

        multipv = info.get("multipv", 1)
        while len(self.multipv) < multipv:
            self.multipv.append({})
        self.multipv[multipv - 1].update(info)

        self._queue.put_nowait(info)

    def _kork(self) -> None:
        if not self._posted_kork:
            self._posted_kork = True
            self._queue.put_nowait({})

    def set_finished(self, best: BestMove) -> None:
        if not self._finished.done():
            self._finished.set_result(best)
        self._kork()

    def set_exception(self, exc: Exception) -> None:
        self._finished.set_exception(exc)
        self._kork()

    @property
    def info(self) -> InfoDict:
        """
        A dictionary of aggregated information sent by the engine. This is
        actually an alias for ``multipv[0]``.
        """
        return self.multipv[0]

    def stop(self) -> None:
        """Stops the analysis as soon as possible."""
        if self._stop and not self._posted_kork:
            self._stop()
            self._stop = None

    async def wait(self) -> BestMove:
        """Waits until the analysis is finished."""
        return await self._finished

    async def get(self) -> InfoDict:
        """
        Waits for the next dictionary of information from the engine and
        returns it.

        It might be more convenient to use ``async for info in analysis: ...``.

        :raises: :exc:`chess.engine.AnalysisComplete` if the analysis is
            complete (or has been stopped) and all information has been
            consumed. Use :func:`~chess.engine.AnalysisResult.next()` if you
            prefer to get ``None`` instead of an exception.
        """
        if self._seen_kork:
            raise AnalysisComplete()

        info = await self._queue.get()
        if not info:
            # Empty dictionary marks end.
            self._seen_kork = True
            await self._finished
            raise AnalysisComplete()

        return info

    def would_block(self) -> bool:
        """
        Checks if calling :func:`~chess.engine.AnalysisResult.get()`,
        calling :func:`~chess.engine.AnalysisResult.next()`,
        or advancing the iterator one step would require waiting for the
        engine.

        These functions would return immediately if information is
        pending (queue is not
        :func:`empty <chess.engine.AnalysisResult.empty()>`) or if the search
        is finished.
        """
        return not self._seen_kork and self._queue.empty()

    def empty(self) -> bool:
        """
        Checks if all current information has been consumed.

        If the queue is empty, but the analysis is still ongoing, then further
        information can become available in the future.
        """
        return self._seen_kork or self._queue.qsize() <= self._posted_kork

    async def next(self) -> Optional[InfoDict]:
        try:
            return await self.get()
        except AnalysisComplete:
            return None

    def __aiter__(self) -> AnalysisResult:
        return self

    async def __anext__(self) -> InfoDict:
        try:
            return await self.get()
        except AnalysisComplete:
            raise StopAsyncIteration

    def __enter__(self) -> AnalysisResult:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.stop()


async def popen_uci(command: Union[str, List[str]], *, setpgrp: bool = False, **popen_args: Any) -> Tuple[asyncio.SubprocessTransport, UciProtocol]:
    """
    Spawns and initializes a UCI engine.

    :param command: Path of the engine executable, or a list including the
        path and arguments.
    :param setpgrp: Open the engine process in a new process group. This will
        stop signals (such as keyboard interrupts) from propagating from the
        parent process. Defaults to ``False``.
    :param popen_args: Additional arguments for
        `popen <https://docs.python.org/3/library/subprocess.html#popen-constructor>`_.
        Do not set ``stdin``, ``stdout``, ``bufsize`` or
        ``universal_newlines``.

    Returns a subprocess transport and engine protocol pair.
    """
    transport, protocol = await UciProtocol.popen(command, setpgrp=setpgrp, **popen_args)
    try:
        await protocol.initialize()
    except:
        transport.close()
        raise
    return transport, protocol


async def popen_xboard(command: Union[str, List[str]], *, setpgrp: bool = False, **popen_args: Any) -> Tuple[asyncio.SubprocessTransport, XBoardProtocol]:
    """
    Spawns and initializes an XBoard engine.

    :param command: Path of the engine executable, or a list including the
        path and arguments.
    :param setpgrp: Open the engine process in a new process group. This will
        stop signals (such as keyboard interrupts) from propagating from the
        parent process. Defaults to ``False``.
    :param popen_args: Additional arguments for
        `popen <https://docs.python.org/3/library/subprocess.html#popen-constructor>`_.
        Do not set ``stdin``, ``stdout``, ``bufsize`` or
        ``universal_newlines``.

    Returns a subprocess transport and engine protocol pair.
    """
    transport, protocol = await XBoardProtocol.popen(command, setpgrp=setpgrp, **popen_args)
    try:
        await protocol.initialize()
    except:
        transport.close()
        raise
    return transport, protocol


async def _async(sync: Callable[[], T]) -> T:
    return sync()


class SimpleEngine:
    """
    Synchronous wrapper around a transport and engine protocol pair. Provides
    the same methods and attributes as :class:`chess.engine.Protocol`
    with blocking functions instead of coroutines.

    You may not concurrently modify objects passed to any of the methods. Other
    than that, :class:`~chess.engine.SimpleEngine` is thread-safe. When sending
    a new command to the engine, any previous running command will be cancelled
    as soon as possible.

    Methods will raise :class:`asyncio.TimeoutError` if an operation takes
    *timeout* seconds longer than expected (unless *timeout* is ``None``).

    Automatically closes the transport when used as a context manager.
    """

    def __init__(self, transport: asyncio.SubprocessTransport, protocol: Protocol, *, timeout: Optional[float] = 10.0) -> None:
        self.transport = transport
        self.protocol = protocol
        self.timeout = timeout

        self._shutdown_lock = threading.Lock()
        self._shutdown = False
        self.shutdown_event = asyncio.Event()

        self.returncode: concurrent.futures.Future[int] = concurrent.futures.Future()

    def _timeout_for(self, limit: Optional[Limit]) -> Optional[float]:
        if self.timeout is None or limit is None or limit.time is None:
            return None
        return self.timeout + limit.time

    @contextlib.contextmanager
    def _not_shut_down(self) -> Generator[None, None, None]:
        with self._shutdown_lock:
            if self._shutdown:
                raise EngineTerminatedError("engine event loop dead")
            yield

    @property
    def options(self) -> MutableMapping[str, Option]:
        with self._not_shut_down():
            coro = _async(lambda: copy.copy(self.protocol.options))
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    @property
    def id(self) -> Mapping[str, str]:
        with self._not_shut_down():
            coro = _async(lambda: self.protocol.id.copy())
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    def communicate(self, command_factory: Callable[[Protocol], BaseCommand[T]]) -> T:
        with self._not_shut_down():
            coro = self.protocol.communicate(command_factory)
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    def configure(self, options: ConfigMapping) -> None:
        with self._not_shut_down():
            coro = asyncio.wait_for(self.protocol.configure(options), self.timeout)
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    def send_opponent_information(self, *, opponent: Optional[Opponent] = None, engine_rating: Optional[int] = None) -> None:
        with self._not_shut_down():
            coro = asyncio.wait_for(
                self.protocol.send_opponent_information(opponent=opponent, engine_rating=engine_rating),
                self.timeout)
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    def ping(self) -> None:
        with self._not_shut_down():
            coro = asyncio.wait_for(self.protocol.ping(), self.timeout)
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    def play(self, board: chess.Board, limit: Limit, *, game: object = None, info: Info = INFO_NONE, ponder: bool = False, draw_offered: bool = False, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}, opponent: Optional[Opponent] = None) -> PlayResult:
        with self._not_shut_down():
            coro = asyncio.wait_for(
                self.protocol.play(board, limit, game=game, info=info, ponder=ponder, draw_offered=draw_offered, root_moves=root_moves, options=options, opponent=opponent),
                self._timeout_for(limit))
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    @typing.overload
    def analyse(self, board: chess.Board, limit: Limit, *, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> InfoDict: ...
    @typing.overload
    def analyse(self, board: chess.Board, limit: Limit, *, multipv: int, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> List[InfoDict]: ...
    @typing.overload
    def analyse(self, board: chess.Board, limit: Limit, *, multipv: Optional[int] = None, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> Union[InfoDict, List[InfoDict]]: ...
    def analyse(self, board: chess.Board, limit: Limit, *, multipv: Optional[int] = None, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> Union[InfoDict, List[InfoDict]]:
        with self._not_shut_down():
            coro = asyncio.wait_for(
                self.protocol.analyse(board, limit, multipv=multipv, game=game, info=info, root_moves=root_moves, options=options),
                self._timeout_for(limit))
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    def analysis(self, board: chess.Board, limit: Optional[Limit] = None, *, multipv: Optional[int] = None, game: object = None, info: Info = INFO_ALL, root_moves: Optional[Iterable[chess.Move]] = None, options: ConfigMapping = {}) -> SimpleAnalysisResult:
        with self._not_shut_down():
            coro = asyncio.wait_for(
                self.protocol.analysis(board, limit, multipv=multipv, game=game, info=info, root_moves=root_moves, options=options),
                self.timeout)  # Timeout until analysis is *started*
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return SimpleAnalysisResult(self, future.result())

    def send_game_result(self, board: chess.Board, winner: Optional[Color] = None, game_ending: Optional[str] = None, game_complete: bool = True) -> None:
        with self._not_shut_down():
            coro = asyncio.wait_for(self.protocol.send_game_result(board, winner, game_ending, game_complete), self.timeout)
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    def quit(self) -> None:
        with self._not_shut_down():
            coro = asyncio.wait_for(self.protocol.quit(), self.timeout)
            future = asyncio.run_coroutine_threadsafe(coro, self.protocol.loop)
        return future.result()

    def close(self) -> None:
        """
        Closes the transport and the background event loop as soon as possible.
        """
        def _shutdown() -> None:
            self.transport.close()
            self.shutdown_event.set()

        with self._shutdown_lock:
            if not self._shutdown:
                self._shutdown = True
                self.protocol.loop.call_soon_threadsafe(_shutdown)

    @classmethod
    def popen(cls, Protocol: Type[Protocol], command: Union[str, List[str]], *, timeout: Optional[float] = 10.0, debug: Optional[bool] = None, setpgrp: bool = False, **popen_args: Any) -> SimpleEngine:
        async def background(future: concurrent.futures.Future[SimpleEngine]) -> None:
            transport, protocol = await Protocol.popen(command, setpgrp=setpgrp, **popen_args)
            threading.current_thread().name = f"{cls.__name__} (pid={transport.get_pid()})"
            simple_engine = cls(transport, protocol, timeout=timeout)
            try:
                await asyncio.wait_for(protocol.initialize(), timeout)
                future.set_result(simple_engine)
                returncode = await protocol.returncode
                simple_engine.returncode.set_result(returncode)
            finally:
                simple_engine.close()
            await simple_engine.shutdown_event.wait()

        return run_in_background(background, name=f"{cls.__name__} (command={command!r})", debug=debug)

    @classmethod
    def popen_uci(cls, command: Union[str, List[str]], *, timeout: Optional[float] = 10.0, debug: Optional[bool] = None, setpgrp: bool = False, **popen_args: Any) -> SimpleEngine:
        """
        Spawns and initializes a UCI engine.
        Returns a :class:`~chess.engine.SimpleEngine` instance.
        """
        return cls.popen(UciProtocol, command, timeout=timeout, debug=debug, setpgrp=setpgrp, **popen_args)

    @classmethod
    def popen_xboard(cls, command: Union[str, List[str]], *, timeout: Optional[float] = 10.0, debug: Optional[bool] = None, setpgrp: bool = False, **popen_args: Any) -> SimpleEngine:
        """
        Spawns and initializes an XBoard engine.
        Returns a :class:`~chess.engine.SimpleEngine` instance.
        """
        return cls.popen(XBoardProtocol, command, timeout=timeout, debug=debug, setpgrp=setpgrp, **popen_args)

    def __enter__(self) -> SimpleEngine:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()

    def __repr__(self) -> str:
        pid = self.transport.get_pid()  # This happens to be thread-safe
        return f"<{type(self).__name__} (pid={pid})>"


class SimpleAnalysisResult:
    """
    Synchronous wrapper around :class:`~chess.engine.AnalysisResult`. Returned
    by :func:`chess.engine.SimpleEngine.analysis()`.
    """

    def __init__(self, simple_engine: SimpleEngine, inner: AnalysisResult) -> None:
        self.simple_engine = simple_engine
        self.inner = inner

    @property
    def info(self) -> InfoDict:
        with self.simple_engine._not_shut_down():
            coro = _async(lambda: self.inner.info.copy())
            future = asyncio.run_coroutine_threadsafe(coro, self.simple_engine.protocol.loop)
        return future.result()

    @property
    def multipv(self) -> List[InfoDict]:
        with self.simple_engine._not_shut_down():
            coro = _async(lambda: [info.copy() for info in self.inner.multipv])
            future = asyncio.run_coroutine_threadsafe(coro, self.simple_engine.protocol.loop)
        return future.result()

    def stop(self) -> None:
        with self.simple_engine._not_shut_down():
            self.simple_engine.protocol.loop.call_soon_threadsafe(self.inner.stop)

    def wait(self) -> BestMove:
        with self.simple_engine._not_shut_down():
            future = asyncio.run_coroutine_threadsafe(self.inner.wait(), self.simple_engine.protocol.loop)
        return future.result()

    def would_block(self) -> bool:
        with self.simple_engine._not_shut_down():
            future = asyncio.run_coroutine_threadsafe(_async(self.inner.would_block), self.simple_engine.protocol.loop)
        return future.result()

    def empty(self) -> bool:
        with self.simple_engine._not_shut_down():
            future = asyncio.run_coroutine_threadsafe(_async(self.inner.empty), self.simple_engine.protocol.loop)
        return future.result()

    def get(self) -> InfoDict:
        with self.simple_engine._not_shut_down():
            future = asyncio.run_coroutine_threadsafe(self.inner.get(), self.simple_engine.protocol.loop)
        return future.result()

    def next(self) -> Optional[InfoDict]:
        with self.simple_engine._not_shut_down():
            future = asyncio.run_coroutine_threadsafe(self.inner.next(), self.simple_engine.protocol.loop)
        return future.result()

    def __iter__(self) -> Iterator[InfoDict]:
        with self.simple_engine._not_shut_down():
            self.simple_engine.protocol.loop.call_soon_threadsafe(self.inner.__aiter__)
        return self

    def __next__(self) -> InfoDict:
        try:
            with self.simple_engine._not_shut_down():
                future = asyncio.run_coroutine_threadsafe(self.inner.__anext__(), self.simple_engine.protocol.loop)
            return future.result()
        except StopAsyncIteration:
            raise StopIteration

    def __enter__(self) -> SimpleAnalysisResult:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.stop()
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, strjoin
from . import DefaultTable
import array
from collections.abc import Mapping

hdmxHeaderFormat = """
	>   # big endian!
	version:	H
	numRecords:	H
	recordSize:	l
"""


class _GlyphnamedList(Mapping):
    def __init__(self, reverseGlyphOrder, data):
        self._array = data
        self._map = dict(reverseGlyphOrder)

    def __getitem__(self, k):
        return self._array[self._map[k]]

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def keys(self):
        return self._map.keys()


class table__h_d_m_x(DefaultTable.DefaultTable):
    """Horizontal Device Metrics table

    The ``hdmx`` table is an optional table that stores advance widths for
    glyph outlines at specified pixel sizes.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/hdmx
    """

    def decompile(self, data, ttFont):
        numGlyphs = ttFont["maxp"].numGlyphs
        glyphOrder = ttFont.getGlyphOrder()
        dummy, data = sstruct.unpack2(hdmxHeaderFormat, data, self)
        self.hdmx = {}
        for i in range(self.numRecords):
            ppem = byteord(data[0])
            maxSize = byteord(data[1])
            widths = _GlyphnamedList(
                ttFont.getReverseGlyphMap(), array.array("B", data[2 : 2 + numGlyphs])
            )
            self.hdmx[ppem] = widths
            data = data[self.recordSize :]
        assert len(data) == 0, "too much hdmx data"

    def compile(self, ttFont):
        self.version = 0
        numGlyphs = ttFont["maxp"].numGlyphs
        glyphOrder = ttFont.getGlyphOrder()
        self.recordSize = 4 * ((2 + numGlyphs + 3) // 4)
        pad = (self.recordSize - 2 - numGlyphs) * b"\0"
        self.numRecords = len(self.hdmx)
        data = sstruct.pack(hdmxHeaderFormat, self)
        items = sorted(self.hdmx.items())
        for ppem, widths in items:
            data = data + bytechr(ppem) + bytechr(max(widths.values()))
            for glyphName in glyphOrder:
                width = widths[glyphName]
                data = data + bytechr(width)
            data = data + pad
        return data

    def toXML(self, writer, ttFont):
        writer.begintag("hdmxData")
        writer.newline()
        ppems = sorted(self.hdmx.keys())
        records = []
        format = ""
        for ppem in ppems:
            widths = self.hdmx[ppem]
            records.append(widths)
            format = format + "%4d"
        glyphNames = ttFont.getGlyphOrder()[:]
        glyphNames.sort()
        maxNameLen = max(map(len, glyphNames))
        format = "%" + repr(maxNameLen) + "s:" + format + " ;"
        writer.write(format % (("ppem",) + tuple(ppems)))
        writer.newline()
        writer.newline()
        for glyphName in glyphNames:
            row = []
            for ppem in ppems:
                widths = self.hdmx[ppem]
                row.append(widths[glyphName])
            if ";" in glyphName:
                glyphName = "\\x3b".join(glyphName.split(";"))
            writer.write(format % ((glyphName,) + tuple(row)))
            writer.newline()
        writer.endtag("hdmxData")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name != "hdmxData":
            return
        content = strjoin(content)
        lines = content.split(";")
        topRow = lines[0].split()
        assert topRow[0] == "ppem:", "illegal hdmx format"
        ppems = list(map(int, topRow[1:]))
        self.hdmx = hdmx = {}
        for ppem in ppems:
            hdmx[ppem] = {}
        lines = (line.split() for line in lines[1:])
        for line in lines:
            if not line:
                continue
            assert line[0][-1] == ":", "illegal hdmx format"
            glyphName = line[0][:-1]
            if "\\" in glyphName:
                from fontTools.misc.textTools import safeEval

                glyphName = safeEval('"""' + glyphName + '"""')
            line = list(map(int, line[1:]))
            assert len(line) == len(ppems), "illegal hdmx format"
            for i, ppem in enumerate(ppems):
                hdmx[ppem][glyphName] = line[i]
<|endoftext|>def __getattr__(attr_name):
    from numpy._core import arrayprint
    from ._utils import _raise_warning
    ret = getattr(arrayprint, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.core.arrayprint' has no attribute {attr_name}")
    _raise_warning(attr_name, "arrayprint")
    return ret
<|endoftext|># Added Fortran compiler support to config. Currently useful only for
# try_compile call. try_run works but is untested for most of Fortran
# compilers (they must define linker_exe first).
# Pearu Peterson
import os
import signal
import subprocess
import sys
import textwrap
import warnings

from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from distutils.ccompiler import CompileError, LinkError
import distutils
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.mingw32ccompiler import generate_manifest
from numpy.distutils.command.autodist import (check_gcc_function_attribute,
                                              check_gcc_function_attribute_with_intrinsics,
                                              check_gcc_variable_attribute,
                                              check_gcc_version_at_least,
                                              check_inline,
                                              check_restrict,
                                              check_compiler_gcc)

LANG_EXT['f77'] = '.f'
LANG_EXT['f90'] = '.f90'

class config(old_config):
    old_config.user_options += [
        ('fcompiler=', None, "specify the Fortran compiler type"),
        ]

    def initialize_options(self):
        self.fcompiler = None
        old_config.initialize_options(self)

    def _check_compiler (self):
        old_config._check_compiler(self)
        from numpy.distutils.fcompiler import FCompiler, new_fcompiler

        if sys.platform == 'win32' and (self.compiler.compiler_type in
                                        ('msvc', 'intelw', 'intelemw')):
            # XXX: hack to circumvent a python 2.6 bug with msvc9compiler:
            # initialize call query_vcvarsall, which throws an OSError, and
            # causes an error along the way without much information. We try to
            # catch it here, hoping it is early enough, and print a helpful
            # message instead of Error: None.
            if not self.compiler.initialized:
                try:
                    self.compiler.initialize()
                except OSError as e:
                    msg = textwrap.dedent("""\
                        Could not initialize compiler instance: do you have Visual Studio
                        installed?  If you are trying to build with MinGW, please use "python setup.py
                        build -c mingw32" instead.  If you have Visual Studio installed, check it is
                        correctly installed, and the right version (VS 2015 as of this writing).

                        Original exception was: %s, and the Compiler class was %s
                        ============================================================================""") \
                        % (e, self.compiler.__class__.__name__)
                    print(textwrap.dedent("""\
                        ============================================================================"""))
                    raise distutils.errors.DistutilsPlatformError(msg) from e

            # After MSVC is initialized, add an explicit /MANIFEST to linker
            # flags.  See issues gh-4245 and gh-4101 for details.  Also
            # relevant are issues 4431 and 16296 on the Python bug tracker.
            from distutils import msvc9compiler
            if msvc9compiler.get_build_version() >= 10:
                for ldflags in [self.compiler.ldflags_shared,
                                self.compiler.ldflags_shared_debug]:
                    if '/MANIFEST' not in ldflags:
                        ldflags.append('/MANIFEST')

        if not isinstance(self.fcompiler, FCompiler):
            self.fcompiler = new_fcompiler(compiler=self.fcompiler,
                                           dry_run=self.dry_run, force=1,
                                           c_compiler=self.compiler)
            if self.fcompiler is not None:
                self.fcompiler.customize(self.distribution)
                if self.fcompiler.get_version():
                    self.fcompiler.customize_cmd(self)
                    self.fcompiler.show_customization()

    def _wrap_method(self, mth, lang, args):
        from distutils.ccompiler import CompileError
        from distutils.errors import DistutilsExecError
        save_compiler = self.compiler
        if lang in ['f77', 'f90']:
            self.compiler = self.fcompiler
        if self.compiler is None:
            raise CompileError('%s compiler is not set' % (lang,))
        try:
            ret = mth(*((self,)+args))
        except (DistutilsExecError, CompileError) as e:
            self.compiler = save_compiler
            raise CompileError from e
        self.compiler = save_compiler
        return ret

    def _compile (self, body, headers, include_dirs, lang):
        src, obj = self._wrap_method(old_config._compile, lang,
                                     (body, headers, include_dirs, lang))
        # _compile in unixcompiler.py sometimes creates .d dependency files.
        # Clean them up.
        self.temp_files.append(obj + '.d')
        return src, obj

    def _link (self, body,
               headers, include_dirs,
               libraries, library_dirs, lang):
        if self.compiler.compiler_type=='msvc':
            libraries = (libraries or [])[:]
            library_dirs = (library_dirs or [])[:]
            if lang in ['f77', 'f90']:
                lang = 'c' # always use system linker when using MSVC compiler
                if self.fcompiler:
                    for d in self.fcompiler.library_dirs or []:
                        # correct path when compiling in Cygwin but with
                        # normal Win Python
                        if d.startswith('/usr/lib'):
                            try:
                                d = subprocess.check_output(['cygpath',
                                                             '-w', d])
                            except (OSError, subprocess.CalledProcessError):
                                pass
                            else:
                                d = filepath_from_subprocess_output(d)
                        library_dirs.append(d)
                    for libname in self.fcompiler.libraries or []:
                        if libname not in libraries:
                            libraries.append(libname)
            for libname in libraries:
                if libname.startswith('msvc'): continue
                fileexists = False
                for libdir in library_dirs or []:
                    libfile = os.path.join(libdir, '%s.lib' % (libname))
                    if os.path.isfile(libfile):
                        fileexists = True
                        break
                if fileexists: continue
                # make g77-compiled static libs available to MSVC
                fileexists = False
                for libdir in library_dirs:
                    libfile = os.path.join(libdir, 'lib%s.a' % (libname))
                    if os.path.isfile(libfile):
                        # copy libname.a file to name.lib so that MSVC linker
                        # can find it
                        libfile2 = os.path.join(libdir, '%s.lib' % (libname))
                        copy_file(libfile, libfile2)
                        self.temp_files.append(libfile2)
                        fileexists = True
                        break
                if fileexists: continue
                log.warn('could not find library %r in directories %s' \
                         % (libname, library_dirs))
        elif self.compiler.compiler_type == 'mingw32':
            generate_manifest(self)
        return self._wrap_method(old_config._link, lang,
                                 (body, headers, include_dirs,
                                  libraries, library_dirs, lang))

    def check_header(self, header, include_dirs=None, library_dirs=None, lang='c'):
        self._check_compiler()
        return self.try_compile(
                "/* we need a dummy line to make distutils happy */",
                [header], include_dirs)

    def check_decl(self, symbol,
                   headers=None, include_dirs=None):
        self._check_compiler()
        body = textwrap.dedent("""
            int main(void)
            {
            #ifndef %s
                (void) %s;
            #endif
                ;
                return 0;
            }""") % (symbol, symbol)

        return self.try_compile(body, headers, include_dirs)

    def check_macro_true(self, symbol,
                         headers=None, include_dirs=None):
        self._check_compiler()
        body = textwrap.dedent("""
            int main(void)
            {
            #if %s
            #else
            #error false or undefined macro
            #endif
                ;
                return 0;
            }""") % (symbol,)

        return self.try_compile(body, headers, include_dirs)

    def check_type(self, type_name, headers=None, include_dirs=None,
            library_dirs=None):
        """Check type availability. Return True if the type can be compiled,
        False otherwise"""
        self._check_compiler()

        # First check the type can be compiled
        body = textwrap.dedent(r"""
            int main(void) {
              if ((%(name)s *) 0)
                return 0;
              if (sizeof (%(name)s))
                return 0;
            }
            """) % {'name': type_name}

        st = False
        try:
            try:
                self._compile(body % {'type': type_name},
                        headers, include_dirs, 'c')
                st = True
            except distutils.errors.CompileError:
                st = False
        finally:
            self._clean()

        return st

    def check_type_size(self, type_name, headers=None, include_dirs=None, library_dirs=None, expected=None):
        """Check size of a given type."""
        self._check_compiler()

        # First check the type can be compiled
        body = textwrap.dedent(r"""
            typedef %(type)s npy_check_sizeof_type;
            int main (void)
            {
                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) >= 0)];
                test_array [0] = 0

                ;
                return 0;
            }
            """)
        self._compile(body % {'type': type_name},
                headers, include_dirs, 'c')
        self._clean()

        if expected:
            body = textwrap.dedent(r"""
                typedef %(type)s npy_check_sizeof_type;
                int main (void)
                {
                    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) == %(size)s)];
                    test_array [0] = 0

                    ;
                    return 0;
                }
                """)
            for size in expected:
                try:
                    self._compile(body % {'type': type_name, 'size': size},
                            headers, include_dirs, 'c')
                    self._clean()
                    return size
                except CompileError:
                    pass

        # this fails to *compile* if size > sizeof(type)
        body = textwrap.dedent(r"""
            typedef %(type)s npy_check_sizeof_type;
            int main (void)
            {
                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) <= %(size)s)];
                test_array [0] = 0

                ;
                return 0;
            }
            """)

        # The principle is simple: we first find low and high bounds of size
        # for the type, where low/high are looked up on a log scale. Then, we
        # do a binary search to find the exact size between low and high
        low = 0
        mid = 0
        while True:
            try:
                self._compile(body % {'type': type_name, 'size': mid},
                        headers, include_dirs, 'c')
                self._clean()
                break
            except CompileError:
                #log.info("failure to test for bound %d" % mid)
                low = mid + 1
                mid = 2 * mid + 1

        high = mid
        # Binary search:
        while low != high:
            mid = (high - low) // 2 + low
            try:
                self._compile(body % {'type': type_name, 'size': mid},
                        headers, include_dirs, 'c')
                self._clean()
                high = mid
            except CompileError:
                low = mid + 1
        return low

    def check_func(self, func,
                   headers=None, include_dirs=None,
                   libraries=None, library_dirs=None,
                   decl=False, call=False, call_args=None):
        # clean up distutils's config a bit: add void to main(), and
        # return a value.
        self._check_compiler()
        body = []
        if decl:
            if type(decl) == str:
                body.append(decl)
            else:
                body.append("int %s (void);" % func)
        # Handle MSVC intrinsics: force MS compiler to make a function call.
        # Useful to test for some functions when built with optimization on, to
        # avoid build error because the intrinsic and our 'fake' test
        # declaration do not match.
        body.append("#ifdef _MSC_VER")
        body.append("#pragma function(%s)" % func)
        body.append("#endif")
        body.append("int main (void) {")
        if call:
            if call_args is None:
                call_args = ''
            body.append("  %s(%s);" % (func, call_args))
        else:
            body.append("  %s;" % func)
        body.append("  return 0;")
        body.append("}")
        body = '\n'.join(body) + "\n"

        return self.try_link(body, headers, include_dirs,
                             libraries, library_dirs)

    def check_funcs_once(self, funcs,
                   headers=None, include_dirs=None,
                   libraries=None, library_dirs=None,
                   decl=False, call=False, call_args=None):
        """Check a list of functions at once.

        This is useful to speed up things, since all the functions in the funcs
        list will be put in one compilation unit.

        Arguments
        ---------
        funcs : seq
            list of functions to test
        include_dirs : seq
            list of header paths
        libraries : seq
            list of libraries to link the code snippet to
        library_dirs : seq
            list of library paths
        decl : dict
            for every (key, value), the declaration in the value will be
            used for function in key. If a function is not in the
            dictionary, no declaration will be used.
        call : dict
            for every item (f, value), if the value is True, a call will be
            done to the function f.
        """
        self._check_compiler()
        body = []
        if decl:
            for f, v in decl.items():
                if v:
                    body.append("int %s (void);" % f)

        # Handle MS intrinsics. See check_func for more info.
        body.append("#ifdef _MSC_VER")
        for func in funcs:
            body.append("#pragma function(%s)" % func)
        body.append("#endif")

        body.append("int main (void) {")
        if call:
            for f in funcs:
                if f in call and call[f]:
                    if not (call_args and f in call_args and call_args[f]):
                        args = ''
                    else:
                        args = call_args[f]
                    body.append("  %s(%s);" % (f, args))
                else:
                    body.append("  %s;" % f)
        else:
            for f in funcs:
                body.append("  %s;" % f)
        body.append("  return 0;")
        body.append("}")
        body = '\n'.join(body) + "\n"

        return self.try_link(body, headers, include_dirs,
                             libraries, library_dirs)

    def check_inline(self):
        """Return the inline keyword recognized by the compiler, empty string
        otherwise."""
        return check_inline(self)

    def check_restrict(self):
        """Return the restrict keyword recognized by the compiler, empty string
        otherwise."""
        return check_restrict(self)

    def check_compiler_gcc(self):
        """Return True if the C compiler is gcc"""
        return check_compiler_gcc(self)

    def check_gcc_function_attribute(self, attribute, name):
        return check_gcc_function_attribute(self, attribute, name)

    def check_gcc_function_attribute_with_intrinsics(self, attribute, name,
                                                     code, include):
        return check_gcc_function_attribute_with_intrinsics(self, attribute,
                                                            name, code, include)

    def check_gcc_variable_attribute(self, attribute):
        return check_gcc_variable_attribute(self, attribute)

    def check_gcc_version_at_least(self, major, minor=0, patchlevel=0):
        """Return True if the GCC version is greater than or equal to the
        specified version."""
        return check_gcc_version_at_least(self, major, minor, patchlevel)

    def get_output(self, body, headers=None, include_dirs=None,
                   libraries=None, library_dirs=None,
                   lang="c", use_tee=None):
        """Try to compile, link to an executable, and run a program
        built from 'body' and 'headers'. Returns the exit status code
        of the program and its output.
        """
        # 2008-11-16, RemoveMe
        warnings.warn("\n+++++++++++++++++++++++++++++++++++++++++++++++++\n"
                      "Usage of get_output is deprecated: please do not \n"
                      "use it anymore, and avoid configuration checks \n"
                      "involving running executable on the target machine.\n"
                      "+++++++++++++++++++++++++++++++++++++++++++++++++\n",
                      DeprecationWarning, stacklevel=2)
        self._check_compiler()
        exitcode, output = 255, ''
        try:
            grabber = GrabStdout()
            try:
                src, obj, exe = self._link(body, headers, include_dirs,
                                           libraries, library_dirs, lang)
                grabber.restore()
            except Exception:
                output = grabber.data
                grabber.restore()
                raise
            exe = os.path.join('.', exe)
            try:
                # specify cwd arg for consistency with
                # historic usage pattern of exec_command()
                # also, note that exe appears to be a string,
                # which exec_command() handled, but we now
                # use a list for check_output() -- this assumes
                # that exe is always a single command
                output = subprocess.check_output([exe], cwd='.')
            except subprocess.CalledProcessError as exc:
                exitstatus = exc.returncode
                output = ''
            except OSError:
                # preserve the EnvironmentError exit status
                # used historically in exec_command()
                exitstatus = 127
                output = ''
            else:
                output = filepath_from_subprocess_output(output)
            if hasattr(os, 'WEXITSTATUS'):
                exitcode = os.WEXITSTATUS(exitstatus)
                if os.WIFSIGNALED(exitstatus):
                    sig = os.WTERMSIG(exitstatus)
                    log.error('subprocess exited with signal %d' % (sig,))
                    if sig == signal.SIGINT:
                        # control-C
                        raise KeyboardInterrupt
            else:
                exitcode = exitstatus
            log.info("success!")
        except (CompileError, LinkError):
            log.info("failure.")
        self._clean()
        return exitcode, output

class GrabStdout:

    def __init__(self):
        self.sys_stdout = sys.stdout
        self.data = ''
        sys.stdout = self

    def write (self, data):
        self.sys_stdout.write(data)
        self.data += data

    def flush (self):
        self.sys_stdout.flush()

    def restore(self):
        sys.stdout = self.sys_stdout
<|endoftext|>"""Common IO api utilities"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import codecs
from collections import defaultdict
from collections.abc import (
    Hashable,
    Mapping,
    Sequence,
)
import dataclasses
import functools
import gzip
from io import (
    BufferedIOBase,
    BytesIO,
    RawIOBase,
    StringIO,
    TextIOBase,
    TextIOWrapper,
)
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    DefaultDict,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)
from urllib.parse import (
    urljoin,
    urlparse as parse_url,
    uses_netloc,
    uses_params,
    uses_relative,
)
import warnings
import zipfile

from pandas._typing import (
    BaseBuffer,
    ReadCsvBuffer,
)
from pandas.compat import (
    get_bz2_file,
    get_lzma_file,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool,
    is_file_like,
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import ABCMultiIndex

from pandas.core.shared_docs import _shared_docs

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")
_RFC_3986_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+\-+.]*://")

BaseBufferT = TypeVar("BaseBufferT", bound=BaseBuffer)


if TYPE_CHECKING:
    from types import TracebackType

    from pandas._typing import (
        CompressionDict,
        CompressionOptions,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import MultiIndex


@dataclasses.dataclass
class IOArgs:
    """
    Return value of io/common.py:_get_filepath_or_buffer.
    """

    filepath_or_buffer: str | BaseBuffer
    encoding: str
    mode: str
    compression: CompressionDict
    should_close: bool = False


@dataclasses.dataclass
class IOHandles(Generic[AnyStr]):
    """
    Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    handle: The file handle to be used.
    created_handles: All file handles that are created by get_handle
    is_wrapped: Whether a TextIOWrapper needs to be detached.
    """

    # handle might not implement the IO-interface
    handle: IO[AnyStr]
    compression: CompressionDict
    created_handles: list[IO[bytes] | IO[str]] = dataclasses.field(default_factory=list)
    is_wrapped: bool = False

    def close(self) -> None:
        """
        Close all created buffers.

        Note: If a TextIOWrapper was inserted, it is flushed and detached to
        avoid closing the potentially user-created buffer.
        """
        if self.is_wrapped:
            assert isinstance(self.handle, TextIOWrapper)
            self.handle.flush()
            self.handle.detach()
            self.created_handles.remove(self.handle)
        for handle in self.created_handles:
            handle.close()
        self.created_handles = []
        self.is_wrapped = False

    def __enter__(self) -> IOHandles[AnyStr]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def is_url(url: object) -> bool:
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
    if not isinstance(url, str):
        return False
    return parse_url(url).scheme in _VALID_URLS


@overload
def _expand_user(filepath_or_buffer: str) -> str:
    ...


@overload
def _expand_user(filepath_or_buffer: BaseBufferT) -> BaseBufferT:
    ...


def _expand_user(filepath_or_buffer: str | BaseBufferT) -> str | BaseBufferT:
    """
    Return the argument with an initial component of ~ or ~user
    replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : object to be converted if possible

    Returns
    -------
    expanded_filepath_or_buffer : an expanded filepath or the
                                  input if not expandable
    """
    if isinstance(filepath_or_buffer, str):
        return os.path.expanduser(filepath_or_buffer)
    return filepath_or_buffer


def validate_header_arg(header: object) -> None:
    if header is None:
        return
    if is_integer(header):
        header = cast(int, header)
        if header < 0:
            # GH 27779
            raise ValueError(
                "Passing negative integer to header is invalid. "
                "For no header, use header=None instead"
            )
        return
    if is_list_like(header, allow_sets=False):
        header = cast(Sequence, header)
        if not all(map(is_integer, header)):
            raise ValueError("header must be integer or list of integers")
        if any(i < 0 for i in header):
            raise ValueError("cannot specify multi-index header with negative integers")
        return
    if is_bool(header):
        raise TypeError(
            "Passing a bool to header is invalid. Use header=None for no header or "
            "header=int or list-like of ints to specify "
            "the row(s) making up the column names"
        )
    # GH 16338
    raise ValueError("header must be integer or list of integers")


@overload
def stringify_path(filepath_or_buffer: FilePath, convert_file_like: bool = ...) -> str:
    ...


@overload
def stringify_path(
    filepath_or_buffer: BaseBufferT, convert_file_like: bool = ...
) -> BaseBufferT:
    ...


def stringify_path(
    filepath_or_buffer: FilePath | BaseBufferT,
    convert_file_like: bool = False,
) -> str | BaseBufferT:
    """
    Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object to be converted

    Returns
    -------
    str_filepath_or_buffer : maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    if not convert_file_like and is_file_like(filepath_or_buffer):
        # GH 38125: some fsspec objects implement os.PathLike but have already opened a
        # file. This prevents opening the file a second time. infer_compression calls
        # this function with convert_file_like=True to infer the compression.
        return cast(BaseBufferT, filepath_or_buffer)

    if isinstance(filepath_or_buffer, os.PathLike):
        filepath_or_buffer = filepath_or_buffer.__fspath__()
    return _expand_user(filepath_or_buffer)


def urlopen(*args, **kwargs):
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
    import urllib.request

    return urllib.request.urlopen(*args, **kwargs)


def is_fsspec_url(url: FilePath | BaseBuffer) -> bool:
    """
    Returns true if the given URL looks like
    something fsspec can handle
    """
    return (
        isinstance(url, str)
        and bool(_RFC_3986_PATTERN.match(url))
        and not url.startswith(("http://", "https://"))
    )


@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "filepath_or_buffer",
)
def _get_filepath_or_buffer(
    filepath_or_buffer: FilePath | BaseBuffer,
    encoding: str = "utf-8",
    compression: CompressionOptions | None = None,
    mode: str = "r",
    storage_options: StorageOptions | None = None,
) -> IOArgs:
    """
    If the filepath_or_buffer is a url, translate and return the buffer.
    Otherwise passthrough.

    Parameters
    ----------
    filepath_or_buffer : a url, filepath (str, py.path.local or pathlib.Path),
                         or buffer
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    encoding : the encoding to use to decode bytes, default is 'utf-8'
    mode : str, optional

    {storage_options}


    Returns the dataclass IOArgs.
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)

    # handle compression dict
    compression_method, compression = get_compression_method(compression)
    compression_method = infer_compression(filepath_or_buffer, compression_method)

    # GH21227 internal compression is not used for non-binary handles.
    if compression_method and hasattr(filepath_or_buffer, "write") and "b" not in mode:
        warnings.warn(
            "compression has no effect when passing a non-binary object as input.",
            RuntimeWarning,
            stacklevel=find_stack_level(),
        )
        compression_method = None

    compression = dict(compression, method=compression_method)

    # bz2 and xz do not write the byte order mark for utf-16 and utf-32
    # print a warning when writing such files
    if (
        "w" in mode
        and compression_method in ["bz2", "xz"]
        and encoding in ["utf-16", "utf-32"]
    ):
        warnings.warn(
            f"{compression} will not write the byte order mark for {encoding}",
            UnicodeWarning,
            stacklevel=find_stack_level(),
        )

    # Use binary mode when converting path-like objects to file-like objects (fsspec)
    # except when text mode is explicitly requested. The original mode is returned if
    # fsspec is not used.
    fsspec_mode = mode
    if "t" not in fsspec_mode and "b" not in fsspec_mode:
        fsspec_mode += "b"

    if isinstance(filepath_or_buffer, str) and is_url(filepath_or_buffer):
        # TODO: fsspec can also handle HTTP via requests, but leaving this
        # unchanged. using fsspec appears to break the ability to infer if the
        # server responded with gzipped data
        storage_options = storage_options or {}

        # waiting until now for importing to match intended lazy logic of
        # urlopen function defined elsewhere in this module
        import urllib.request

        # assuming storage_options is to be interpreted as headers
        req_info = urllib.request.Request(filepath_or_buffer, headers=storage_options)
        with urlopen(req_info) as req:
            content_encoding = req.headers.get("Content-Encoding", None)
            if content_encoding == "gzip":
                # Override compression based on Content-Encoding header
                compression = {"method": "gzip"}
            reader = BytesIO(req.read())
        return IOArgs(
            filepath_or_buffer=reader,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )

    if is_fsspec_url(filepath_or_buffer):
        assert isinstance(
            filepath_or_buffer, str
        )  # just to appease mypy for this branch
        # two special-case s3-like protocols; these have special meaning in Hadoop,
        # but are equivalent to just "s3" from fsspec's point of view
        # cc #11071
        if filepath_or_buffer.startswith("s3a://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3a://", "s3://")
        if filepath_or_buffer.startswith("s3n://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3n://", "s3://")
        fsspec = import_optional_dependency("fsspec")

        # If botocore is installed we fallback to reading with anon=True
        # to allow reads from public buckets
        err_types_to_retry_with_anon: list[Any] = []
        try:
            import_optional_dependency("botocore")
            from botocore.exceptions import (
                ClientError,
                NoCredentialsError,
            )

            err_types_to_retry_with_anon = [
                ClientError,
                NoCredentialsError,
                PermissionError,
            ]
        except ImportError:
            pass

        try:
            file_obj = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            ).open()
        # GH 34626 Reads from Public Buckets without Credentials needs anon=True
        except tuple(err_types_to_retry_with_anon):
            if storage_options is None:
                storage_options = {"anon": True}
            else:
                # don't mutate user input.
                storage_options = dict(storage_options)
                storage_options["anon"] = True
            file_obj = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            ).open()

        return IOArgs(
            filepath_or_buffer=file_obj,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )
    elif storage_options:
        raise ValueError(
            "storage_options passed with file object or non-fsspec file path"
        )

    if isinstance(filepath_or_buffer, (str, bytes, mmap.mmap)):
        return IOArgs(
            filepath_or_buffer=_expand_user(filepath_or_buffer),
            encoding=encoding,
            compression=compression,
            should_close=False,
            mode=mode,
        )

    # is_file_like requires (read | write) & __iter__ but __iter__ is only
    # needed for read_csv(engine=python)
    if not (
        hasattr(filepath_or_buffer, "read") or hasattr(filepath_or_buffer, "write")
    ):
        msg = f"Invalid file path or buffer object type: {type(filepath_or_buffer)}"
        raise ValueError(msg)

    return IOArgs(
        filepath_or_buffer=filepath_or_buffer,
        encoding=encoding,
        compression=compression,
        should_close=False,
        mode=mode,
    )


def file_path_to_url(path: str) -> str:
    """
    converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """
    # lazify expensive import (~30ms)
    from urllib.request import pathname2url

    return urljoin("file:", pathname2url(path))


extension_to_compression = {
    ".tar": "tar",
    ".tar.gz": "tar",
    ".tar.bz2": "tar",
    ".tar.xz": "tar",
    ".gz": "gzip",
    ".bz2": "bz2",
    ".zip": "zip",
    ".xz": "xz",
    ".zst": "zstd",
}
_supported_compressions = set(extension_to_compression.values())


def get_compression_method(
    compression: CompressionOptions,
) -> tuple[str | None, CompressionDict]:
    """
    Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
