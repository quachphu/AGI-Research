OSStatus = c_int32

CFErrorRef = POINTER(CFError)
CFDataRef = POINTER(CFData)
CFStringRef = POINTER(CFString)
CFArrayRef = POINTER(CFArray)
CFMutableArrayRef = POINTER(CFMutableArray)
CFArrayCallBacks = c_void_p
CFOptionFlags = c_uint32

SecCertificateRef = POINTER(c_void_p)
SecPolicyRef = POINTER(c_void_p)
SecTrustRef = POINTER(c_void_p)
SecTrustResultType = c_uint32
SecTrustOptionFlags = c_uint32

try:
    Security.SecCertificateCreateWithData.argtypes = [CFAllocatorRef, CFDataRef]
    Security.SecCertificateCreateWithData.restype = SecCertificateRef

    Security.SecCertificateCopyData.argtypes = [SecCertificateRef]
    Security.SecCertificateCopyData.restype = CFDataRef

    Security.SecCopyErrorMessageString.argtypes = [OSStatus, c_void_p]
    Security.SecCopyErrorMessageString.restype = CFStringRef

    Security.SecTrustSetAnchorCertificates.argtypes = [SecTrustRef, CFArrayRef]
    Security.SecTrustSetAnchorCertificates.restype = OSStatus

    Security.SecTrustSetAnchorCertificatesOnly.argtypes = [SecTrustRef, Boolean]
    Security.SecTrustSetAnchorCertificatesOnly.restype = OSStatus

    Security.SecPolicyCreateRevocation.argtypes = [CFOptionFlags]
    Security.SecPolicyCreateRevocation.restype = SecPolicyRef

    Security.SecPolicyCreateSSL.argtypes = [Boolean, CFStringRef]
    Security.SecPolicyCreateSSL.restype = SecPolicyRef

    Security.SecTrustCreateWithCertificates.argtypes = [
        CFTypeRef,
        CFTypeRef,
        POINTER(SecTrustRef),
    ]
    Security.SecTrustCreateWithCertificates.restype = OSStatus

    Security.SecTrustGetTrustResult.argtypes = [
        SecTrustRef,
        POINTER(SecTrustResultType),
    ]
    Security.SecTrustGetTrustResult.restype = OSStatus

    Security.SecTrustEvaluate.argtypes = [
        SecTrustRef,
        POINTER(SecTrustResultType),
    ]
    Security.SecTrustEvaluate.restype = OSStatus

    Security.SecTrustRef = SecTrustRef  # type: ignore[attr-defined]
    Security.SecTrustResultType = SecTrustResultType  # type: ignore[attr-defined]
    Security.OSStatus = OSStatus  # type: ignore[attr-defined]

    kSecRevocationUseAnyAvailableMethod = 3
    kSecRevocationRequirePositiveResponse = 8

    CoreFoundation.CFRelease.argtypes = [CFTypeRef]
    CoreFoundation.CFRelease.restype = None

    CoreFoundation.CFGetTypeID.argtypes = [CFTypeRef]
    CoreFoundation.CFGetTypeID.restype = CFTypeID

    CoreFoundation.CFStringCreateWithCString.argtypes = [
        CFAllocatorRef,
        c_char_p,
        CFStringEncoding,
    ]
    CoreFoundation.CFStringCreateWithCString.restype = CFStringRef

    CoreFoundation.CFStringGetCStringPtr.argtypes = [CFStringRef, CFStringEncoding]
    CoreFoundation.CFStringGetCStringPtr.restype = c_char_p

    CoreFoundation.CFStringGetCString.argtypes = [
        CFStringRef,
        c_char_p,
        CFIndex,
        CFStringEncoding,
    ]
    CoreFoundation.CFStringGetCString.restype = c_bool

    CoreFoundation.CFDataCreate.argtypes = [CFAllocatorRef, c_char_p, CFIndex]
    CoreFoundation.CFDataCreate.restype = CFDataRef

    CoreFoundation.CFDataGetLength.argtypes = [CFDataRef]
    CoreFoundation.CFDataGetLength.restype = CFIndex

    CoreFoundation.CFDataGetBytePtr.argtypes = [CFDataRef]
    CoreFoundation.CFDataGetBytePtr.restype = c_void_p

    CoreFoundation.CFArrayCreate.argtypes = [
        CFAllocatorRef,
        POINTER(CFTypeRef),
        CFIndex,
        CFArrayCallBacks,
    ]
    CoreFoundation.CFArrayCreate.restype = CFArrayRef

    CoreFoundation.CFArrayCreateMutable.argtypes = [
        CFAllocatorRef,
        CFIndex,
        CFArrayCallBacks,
    ]
    CoreFoundation.CFArrayCreateMutable.restype = CFMutableArrayRef

    CoreFoundation.CFArrayAppendValue.argtypes = [CFMutableArrayRef, c_void_p]
    CoreFoundation.CFArrayAppendValue.restype = None

    CoreFoundation.CFArrayGetCount.argtypes = [CFArrayRef]
    CoreFoundation.CFArrayGetCount.restype = CFIndex

    CoreFoundation.CFArrayGetValueAtIndex.argtypes = [CFArrayRef, CFIndex]
    CoreFoundation.CFArrayGetValueAtIndex.restype = c_void_p

    CoreFoundation.CFErrorGetCode.argtypes = [CFErrorRef]
    CoreFoundation.CFErrorGetCode.restype = CFIndex

    CoreFoundation.CFErrorCopyDescription.argtypes = [CFErrorRef]
    CoreFoundation.CFErrorCopyDescription.restype = CFStringRef

    CoreFoundation.kCFAllocatorDefault = CFAllocatorRef.in_dll(  # type: ignore[attr-defined]
        CoreFoundation, "kCFAllocatorDefault"
    )
    CoreFoundation.kCFTypeArrayCallBacks = c_void_p.in_dll(  # type: ignore[attr-defined]
        CoreFoundation, "kCFTypeArrayCallBacks"
    )

    CoreFoundation.CFTypeRef = CFTypeRef  # type: ignore[attr-defined]
    CoreFoundation.CFArrayRef = CFArrayRef  # type: ignore[attr-defined]
    CoreFoundation.CFStringRef = CFStringRef  # type: ignore[attr-defined]
    CoreFoundation.CFErrorRef = CFErrorRef  # type: ignore[attr-defined]

except AttributeError as e:
    raise ImportError(f"Error initializing ctypes: {e}") from None

# SecTrustEvaluateWithError is macOS 10.14+
if _is_macos_version_10_14_or_later:
    try:
        Security.SecTrustEvaluateWithError.argtypes = [
            SecTrustRef,
            POINTER(CFErrorRef),
        ]
        Security.SecTrustEvaluateWithError.restype = c_bool
    except AttributeError as e:
        raise ImportError(f"Error initializing ctypes: {e}") from None


def _handle_osstatus(result: OSStatus, _: typing.Any, args: typing.Any) -> typing.Any:
    """
    Raises an error if the OSStatus value is non-zero.
    """
    if int(result) == 0:
        return args

    # Returns a CFString which we need to transform
    # into a UTF-8 Python string.
    error_message_cfstring = None
    try:
        error_message_cfstring = Security.SecCopyErrorMessageString(result, None)

        # First step is convert the CFString into a C string pointer.
        # We try the fast no-copy way first.
        error_message_cfstring_c_void_p = ctypes.cast(
            error_message_cfstring, ctypes.POINTER(ctypes.c_void_p)
        )
        message = CoreFoundation.CFStringGetCStringPtr(
            error_message_cfstring_c_void_p, CFConst.kCFStringEncodingUTF8
        )

        # Quoting the Apple dev docs:
        #
        # "A pointer to a C string or NULL if the internal
        # storage of theString does not allow this to be
        # returned efficiently."
        #
        # So we need to get our hands dirty.
        if message is None:
            buffer = ctypes.create_string_buffer(1024)
            result = CoreFoundation.CFStringGetCString(
                error_message_cfstring_c_void_p,
                buffer,
                1024,
                CFConst.kCFStringEncodingUTF8,
            )
            if not result:
                raise OSError("Error copying C string from CFStringRef")
            message = buffer.value

    finally:
        if error_message_cfstring is not None:
            CoreFoundation.CFRelease(error_message_cfstring)

    # If no message can be found for this status we come
    # up with a generic one that forwards the status code.
    if message is None or message == "":
        message = f"SecureTransport operation returned a non-zero OSStatus: {result}"

    raise ssl.SSLError(message)


Security.SecTrustCreateWithCertificates.errcheck = _handle_osstatus  # type: ignore[assignment]
Security.SecTrustSetAnchorCertificates.errcheck = _handle_osstatus  # type: ignore[assignment]
Security.SecTrustSetAnchorCertificatesOnly.errcheck = _handle_osstatus  # type: ignore[assignment]
Security.SecTrustGetTrustResult.errcheck = _handle_osstatus  # type: ignore[assignment]
Security.SecTrustEvaluate.errcheck = _handle_osstatus  # type: ignore[assignment]


class CFConst:
    """CoreFoundation constants"""

    kCFStringEncodingUTF8 = CFStringEncoding(0x08000100)

    errSecIncompleteCertRevocationCheck = -67635
    errSecHostNameMismatch = -67602
    errSecCertificateExpired = -67818
    errSecNotTrusted = -67843


def _bytes_to_cf_data_ref(value: bytes) -> CFDataRef:  # type: ignore[valid-type]
    return CoreFoundation.CFDataCreate(  # type: ignore[no-any-return]
        CoreFoundation.kCFAllocatorDefault, value, len(value)
    )


def _bytes_to_cf_string(value: bytes) -> CFString:
    """
    Given a Python binary data, create a CFString.
    The string must be CFReleased by the caller.
    """
    c_str = ctypes.c_char_p(value)
    cf_str = CoreFoundation.CFStringCreateWithCString(
        CoreFoundation.kCFAllocatorDefault,
        c_str,
        CFConst.kCFStringEncodingUTF8,
    )
    return cf_str  # type: ignore[no-any-return]


def _cf_string_ref_to_str(cf_string_ref: CFStringRef) -> str | None:  # type: ignore[valid-type]
    """
    Creates a Unicode string from a CFString object. Used entirely for error
    reporting.
    Yes, it annoys me quite a lot that this function is this complex.
    """

    string = CoreFoundation.CFStringGetCStringPtr(
        cf_string_ref, CFConst.kCFStringEncodingUTF8
    )
    if string is None:
        buffer = ctypes.create_string_buffer(1024)
        result = CoreFoundation.CFStringGetCString(
            cf_string_ref, buffer, 1024, CFConst.kCFStringEncodingUTF8
        )
        if not result:
            raise OSError("Error copying C string from CFStringRef")
        string = buffer.value
    if string is not None:
        string = string.decode("utf-8")
    return string  # type: ignore[no-any-return]


def _der_certs_to_cf_cert_array(certs: list[bytes]) -> CFMutableArrayRef:  # type: ignore[valid-type]
    """Builds a CFArray of SecCertificateRefs from a list of DER-encoded certificates.
    Responsibility of the caller to call CoreFoundation.CFRelease on the CFArray.
    """
    cf_array = CoreFoundation.CFArrayCreateMutable(
        CoreFoundation.kCFAllocatorDefault,
        0,
        ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
    )
    if not cf_array:
        raise MemoryError("Unable to allocate memory!")

    for cert_data in certs:
        cf_data = None
        sec_cert_ref = None
        try:
            cf_data = _bytes_to_cf_data_ref(cert_data)
            sec_cert_ref = Security.SecCertificateCreateWithData(
                CoreFoundation.kCFAllocatorDefault, cf_data
            )
            CoreFoundation.CFArrayAppendValue(cf_array, sec_cert_ref)
        finally:
            if cf_data:
                CoreFoundation.CFRelease(cf_data)
            if sec_cert_ref:
                CoreFoundation.CFRelease(sec_cert_ref)

    return cf_array  # type: ignore[no-any-return]


@contextlib.contextmanager
def _configure_context(ctx: ssl.SSLContext) -> typing.Iterator[None]:
    check_hostname = ctx.check_hostname
    verify_mode = ctx.verify_mode
    ctx.check_hostname = False
    _set_ssl_context_verify_mode(ctx, ssl.CERT_NONE)
    try:
        yield
    finally:
        ctx.check_hostname = check_hostname
        _set_ssl_context_verify_mode(ctx, verify_mode)


def _verify_peercerts_impl(
    ssl_context: ssl.SSLContext,
    cert_chain: list[bytes],
    server_hostname: str | None = None,
) -> None:
    certs = None
    policies = None
    trust = None
    try:
        # Only set a hostname on the policy if we're verifying the hostname
        # on the leaf certificate.
        if server_hostname is not None and ssl_context.check_hostname:
            cf_str_hostname = None
            try:
                cf_str_hostname = _bytes_to_cf_string(server_hostname.encode("ascii"))
                ssl_policy = Security.SecPolicyCreateSSL(True, cf_str_hostname)
            finally:
                if cf_str_hostname:
                    CoreFoundation.CFRelease(cf_str_hostname)
        else:
            ssl_policy = Security.SecPolicyCreateSSL(True, None)

        policies = ssl_policy
        if ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_CHAIN:
            # Add explicit policy requiring positive revocation checks
            policies = CoreFoundation.CFArrayCreateMutable(
                CoreFoundation.kCFAllocatorDefault,
                0,
                ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks),
            )
            CoreFoundation.CFArrayAppendValue(policies, ssl_policy)
            CoreFoundation.CFRelease(ssl_policy)
            revocation_policy = Security.SecPolicyCreateRevocation(
                kSecRevocationUseAnyAvailableMethod
                | kSecRevocationRequirePositiveResponse
            )
            CoreFoundation.CFArrayAppendValue(policies, revocation_policy)
            CoreFoundation.CFRelease(revocation_policy)
        elif ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_LEAF:
            raise NotImplementedError("VERIFY_CRL_CHECK_LEAF not implemented for macOS")

        certs = None
        try:
            certs = _der_certs_to_cf_cert_array(cert_chain)

            # Now that we have certificates loaded and a SecPolicy
            # we can finally create a SecTrust object!
            trust = Security.SecTrustRef()
            Security.SecTrustCreateWithCertificates(
                certs, policies, ctypes.byref(trust)
            )

        finally:
            # The certs are now being held by SecTrust so we can
            # release our handles for the array.
            if certs:
                CoreFoundation.CFRelease(certs)

        # If there are additional trust anchors to load we need to transform
        # the list of DER-encoded certificates into a CFArray.
        ctx_ca_certs_der: list[bytes] | None = ssl_context.get_ca_certs(
            binary_form=True
        )
        if ctx_ca_certs_der:
            ctx_ca_certs = None
            try:
                ctx_ca_certs = _der_certs_to_cf_cert_array(ctx_ca_certs_der)
                Security.SecTrustSetAnchorCertificates(trust, ctx_ca_certs)
            finally:
                if ctx_ca_certs:
                    CoreFoundation.CFRelease(ctx_ca_certs)

        # We always want system certificates.
        Security.SecTrustSetAnchorCertificatesOnly(trust, False)

        # macOS 10.13 and earlier don't support SecTrustEvaluateWithError()
        # so we use SecTrustEvaluate() which means we need to construct error
        # messages ourselves.
        if _is_macos_version_10_14_or_later:
            _verify_peercerts_impl_macos_10_14(ssl_context, trust)
        else:
            _verify_peercerts_impl_macos_10_13(ssl_context, trust)
    finally:
        if policies:
            CoreFoundation.CFRelease(policies)
        if trust:
            CoreFoundation.CFRelease(trust)


def _verify_peercerts_impl_macos_10_13(
    ssl_context: ssl.SSLContext, sec_trust_ref: typing.Any
) -> None:
    """Verify using 'SecTrustEvaluate' API for macOS 10.13 and earlier.
    macOS 10.14 added the 'SecTrustEvaluateWithError' API.
    """
    sec_trust_result_type = Security.SecTrustResultType()
    Security.SecTrustEvaluate(sec_trust_ref, ctypes.byref(sec_trust_result_type))

    try:
        sec_trust_result_type_as_int = int(sec_trust_result_type.value)
    except (ValueError, TypeError):
        sec_trust_result_type_as_int = -1

    # Apple doesn't document these values in their own API docs.
    # See: https://github.com/xybp888/iOS-SDKs/blob/master/iPhoneOS13.0.sdk/System/Library/Frameworks/Security.framework/Headers/SecTrust.h#L84
    if (
        ssl_context.verify_mode == ssl.CERT_REQUIRED
        and sec_trust_result_type_as_int not in (1, 4)
    ):
        # Note that we're not able to ignore only hostname errors
        # for macOS 10.13 and earlier, so check_hostname=False will
        # still return an error.
        sec_trust_result_type_to_message = {
            0: "Invalid trust result type",
            # 1: "Trust evaluation succeeded",
            2: "User confirmation required",
            3: "User specified that certificate is not trusted",
            # 4: "Trust result is unspecified",
            5: "Recoverable trust failure occurred",
            6: "Fatal trust failure occurred",
            7: "Other error occurred, certificate may be revoked",
        }
        error_message = sec_trust_result_type_to_message.get(
            sec_trust_result_type_as_int,
            f"Unknown trust result: {sec_trust_result_type_as_int}",
        )

        err = ssl.SSLCertVerificationError(error_message)
        err.verify_message = error_message
        err.verify_code = sec_trust_result_type_as_int
        raise err


def _verify_peercerts_impl_macos_10_14(
    ssl_context: ssl.SSLContext, sec_trust_ref: typing.Any
) -> None:
    """Verify using 'SecTrustEvaluateWithError' API for macOS 10.14+."""
    cf_error = CoreFoundation.CFErrorRef()
    sec_trust_eval_result = Security.SecTrustEvaluateWithError(
        sec_trust_ref, ctypes.byref(cf_error)
    )
    # sec_trust_eval_result is a bool (0 or 1)
    # where 1 means that the certs are trusted.
    if sec_trust_eval_result == 1:
        is_trusted = True
    elif sec_trust_eval_result == 0:
        is_trusted = False
    else:
        raise ssl.SSLError(
            f"Unknown result from Security.SecTrustEvaluateWithError: {sec_trust_eval_result!r}"
        )

    cf_error_code = 0
    if not is_trusted:
        cf_error_code = CoreFoundation.CFErrorGetCode(cf_error)

        # If the error is a known failure that we're
        # explicitly okay with from SSLContext configuration
        # we can set is_trusted accordingly.
        if ssl_context.verify_mode != ssl.CERT_REQUIRED and (
            cf_error_code == CFConst.errSecNotTrusted
            or cf_error_code == CFConst.errSecCertificateExpired
        ):
            is_trusted = True

    # If we're still not trusted then we start to
    # construct and raise the SSLCertVerificationError.
    if not is_trusted:
        cf_error_string_ref = None
        try:
            cf_error_string_ref = CoreFoundation.CFErrorCopyDescription(cf_error)

            # Can this ever return 'None' if there's a CFError?
            cf_error_message = (
                _cf_string_ref_to_str(cf_error_string_ref)
                or "Certificate verification failed"
            )

            # TODO: Not sure if we need the SecTrustResultType for anything?
            # We only care whether or not it's a success or failure for now.
            sec_trust_result_type = Security.SecTrustResultType()
            Security.SecTrustGetTrustResult(
                sec_trust_ref, ctypes.byref(sec_trust_result_type)
            )

            err = ssl.SSLCertVerificationError(cf_error_message)
            err.verify_message = cf_error_message
            err.verify_code = cf_error_code
            raise err
        finally:
            if cf_error_string_ref:
                CoreFoundation.CFRelease(cf_error_string_ref)
<|endoftext|>import contextlib
import os
import re
import ssl
import typing

# candidates based on https://github.com/tiran/certifi-system-store by Christian Heimes
_CA_FILE_CANDIDATES = [
    # Alpine, Arch, Fedora 34-43, OpenWRT, RHEL 9-10, BSD
    "/etc/ssl/cert.pem",
    # Fedora 44+, RHEL 11+
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
    # Fedora <= 34, RHEL <= 9, CentOS <= 9
    "/etc/pki/tls/cert.pem",
    # Debian, Ubuntu (requires ca-certificates)
    "/etc/ssl/certs/ca-certificates.crt",
    # SUSE
    "/etc/ssl/ca-bundle.pem",
]

_HASHED_CERT_FILENAME_RE = re.compile(r"^[0-9a-fA-F]{8}\.[0-9]$")


@contextlib.contextmanager
def _configure_context(ctx: ssl.SSLContext) -> typing.Iterator[None]:
    # First, check whether the default locations from OpenSSL
    # seem like they will give us a usable set of CA certs.
    # ssl.get_default_verify_paths already takes care of:
    # - getting cafile from either the SSL_CERT_FILE env var
    #   or the path configured when OpenSSL was compiled,
    #   and verifying that that path exists
    # - getting capath from either the SSL_CERT_DIR env var
    #   or the path configured when OpenSSL was compiled,
    #   and verifying that that path exists
    # In addition we'll check whether capath appears to contain certs.
    defaults = ssl.get_default_verify_paths()
    if defaults.cafile or (defaults.capath and _capath_contains_certs(defaults.capath)):
        ctx.set_default_verify_paths()
    else:
        # cafile from OpenSSL doesn't exist
        # and capath from OpenSSL doesn't contain certs.
        # Let's search other common locations instead.
        for cafile in _CA_FILE_CANDIDATES:
            if os.path.isfile(cafile):
                ctx.load_verify_locations(cafile=cafile)
                break

    yield


def _capath_contains_certs(capath: str) -> bool:
    """Check whether capath exists and contains certs in the expected format."""
    if not os.path.isdir(capath):
        return False
    return any(_HASHED_CERT_FILENAME_RE.match(name) for name in os.listdir(capath))


def _verify_peercerts_impl(
    ssl_context: ssl.SSLContext,
    cert_chain: list[bytes],
    server_hostname: str | None = None,
) -> None:
    # This is a no-op because we've enabled SSLContext's built-in
    # verification via verify_mode=CERT_REQUIRED, and don't need to repeat it.
    pass
<|endoftext|>import contextlib  # noqa: I001
import ssl
import typing
from ctypes import WinDLL  # type: ignore[attr-defined]
from ctypes import WinError  # type: ignore[attr-defined]
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_ulong,
    c_void_p,
    c_wchar_p,
    cast,
    create_unicode_buffer,
    pointer,
    sizeof,
)
from ctypes.wintypes import (
    BOOL,
    DWORD,
    HANDLE,
    LONG,
    LPCSTR,
    LPCVOID,
    LPCWSTR,
    LPFILETIME,
    LPSTR,
    LPWSTR,
)
from typing import TYPE_CHECKING, Any

from ._ssl_constants import _set_ssl_context_verify_mode

HCERTCHAINENGINE = HANDLE
HCERTSTORE = HANDLE
HCRYPTPROV_LEGACY = HANDLE


class CERT_CONTEXT(Structure):
    _fields_ = (
        ("dwCertEncodingType", DWORD),
        ("pbCertEncoded", c_void_p),
        ("cbCertEncoded", DWORD),
        ("pCertInfo", c_void_p),
        ("hCertStore", HCERTSTORE),
    )


PCERT_CONTEXT = POINTER(CERT_CONTEXT)
PCCERT_CONTEXT = POINTER(PCERT_CONTEXT)


class CERT_ENHKEY_USAGE(Structure):
    _fields_ = (
        ("cUsageIdentifier", DWORD),
        ("rgpszUsageIdentifier", POINTER(LPSTR)),
    )


PCERT_ENHKEY_USAGE = POINTER(CERT_ENHKEY_USAGE)


class CERT_USAGE_MATCH(Structure):
    _fields_ = (
        ("dwType", DWORD),
        ("Usage", CERT_ENHKEY_USAGE),
    )


class CERT_CHAIN_PARA(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("RequestedUsage", CERT_USAGE_MATCH),
        ("RequestedIssuancePolicy", CERT_USAGE_MATCH),
        ("dwUrlRetrievalTimeout", DWORD),
        ("fCheckRevocationFreshnessTime", BOOL),
        ("dwRevocationFreshnessTime", DWORD),
        ("pftCacheResync", LPFILETIME),
        ("pStrongSignPara", c_void_p),
        ("dwStrongSignFlags", DWORD),
    )


if TYPE_CHECKING:
    PCERT_CHAIN_PARA = pointer[CERT_CHAIN_PARA]  # type: ignore[misc]
else:
    PCERT_CHAIN_PARA = POINTER(CERT_CHAIN_PARA)


class CERT_TRUST_STATUS(Structure):
    _fields_ = (
        ("dwErrorStatus", DWORD),
        ("dwInfoStatus", DWORD),
    )


class CERT_CHAIN_ELEMENT(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("pCertContext", PCERT_CONTEXT),
        ("TrustStatus", CERT_TRUST_STATUS),
        ("pRevocationInfo", c_void_p),
        ("pIssuanceUsage", PCERT_ENHKEY_USAGE),
        ("pApplicationUsage", PCERT_ENHKEY_USAGE),
        ("pwszExtendedErrorInfo", LPCWSTR),
    )


PCERT_CHAIN_ELEMENT = POINTER(CERT_CHAIN_ELEMENT)


class CERT_SIMPLE_CHAIN(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("TrustStatus", CERT_TRUST_STATUS),
        ("cElement", DWORD),
        ("rgpElement", POINTER(PCERT_CHAIN_ELEMENT)),
        ("pTrustListInfo", c_void_p),
        ("fHasRevocationFreshnessTime", BOOL),
        ("dwRevocationFreshnessTime", DWORD),
    )


PCERT_SIMPLE_CHAIN = POINTER(CERT_SIMPLE_CHAIN)


class CERT_CHAIN_CONTEXT(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("TrustStatus", CERT_TRUST_STATUS),
        ("cChain", DWORD),
        ("rgpChain", POINTER(PCERT_SIMPLE_CHAIN)),
        ("cLowerQualityChainContext", DWORD),
        ("rgpLowerQualityChainContext", c_void_p),
        ("fHasRevocationFreshnessTime", BOOL),
        ("dwRevocationFreshnessTime", DWORD),
    )


PCERT_CHAIN_CONTEXT = POINTER(CERT_CHAIN_CONTEXT)
PCCERT_CHAIN_CONTEXT = POINTER(PCERT_CHAIN_CONTEXT)


class SSL_EXTRA_CERT_CHAIN_POLICY_PARA(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("dwAuthType", DWORD),
        ("fdwChecks", DWORD),
        ("pwszServerName", LPCWSTR),
    )


class CERT_CHAIN_POLICY_PARA(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("dwFlags", DWORD),
        ("pvExtraPolicyPara", c_void_p),
    )


PCERT_CHAIN_POLICY_PARA = POINTER(CERT_CHAIN_POLICY_PARA)


class CERT_CHAIN_POLICY_STATUS(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("dwError", DWORD),
        ("lChainIndex", LONG),
        ("lElementIndex", LONG),
        ("pvExtraPolicyStatus", c_void_p),
    )


PCERT_CHAIN_POLICY_STATUS = POINTER(CERT_CHAIN_POLICY_STATUS)


class CERT_CHAIN_ENGINE_CONFIG(Structure):
    _fields_ = (
        ("cbSize", DWORD),
        ("hRestrictedRoot", HCERTSTORE),
        ("hRestrictedTrust", HCERTSTORE),
        ("hRestrictedOther", HCERTSTORE),
        ("cAdditionalStore", DWORD),
        ("rghAdditionalStore", c_void_p),
        ("dwFlags", DWORD),
        ("dwUrlRetrievalTimeout", DWORD),
        ("MaximumCachedCertificates", DWORD),
        ("CycleDetectionModulus", DWORD),
        ("hExclusiveRoot", HCERTSTORE),
        ("hExclusiveTrustedPeople", HCERTSTORE),
        ("dwExclusiveFlags", DWORD),
    )


PCERT_CHAIN_ENGINE_CONFIG = POINTER(CERT_CHAIN_ENGINE_CONFIG)
PHCERTCHAINENGINE = POINTER(HCERTCHAINENGINE)

X509_ASN_ENCODING = 0x00000001
PKCS_7_ASN_ENCODING = 0x00010000
CERT_STORE_PROV_MEMORY = b"Memory"
CERT_STORE_ADD_USE_EXISTING = 2
USAGE_MATCH_TYPE_OR = 1
OID_PKIX_KP_SERVER_AUTH = c_char_p(b"1.3.6.1.5.5.7.3.1")
CERT_CHAIN_REVOCATION_CHECK_END_CERT = 0x10000000
CERT_CHAIN_REVOCATION_CHECK_CHAIN = 0x20000000
CERT_CHAIN_POLICY_IGNORE_ALL_NOT_TIME_VALID_FLAGS = 0x00000007
CERT_CHAIN_POLICY_IGNORE_INVALID_BASIC_CONSTRAINTS_FLAG = 0x00000008
CERT_CHAIN_POLICY_ALLOW_UNKNOWN_CA_FLAG = 0x00000010
CERT_CHAIN_POLICY_IGNORE_INVALID_NAME_FLAG = 0x00000040
CERT_CHAIN_POLICY_IGNORE_WRONG_USAGE_FLAG = 0x00000020
CERT_CHAIN_POLICY_IGNORE_INVALID_POLICY_FLAG = 0x00000080
CERT_CHAIN_POLICY_IGNORE_ALL_REV_UNKNOWN_FLAGS = 0x00000F00
CERT_CHAIN_POLICY_ALLOW_TESTROOT_FLAG = 0x00008000
CERT_CHAIN_POLICY_TRUST_TESTROOT_FLAG = 0x00004000
SECURITY_FLAG_IGNORE_CERT_CN_INVALID = 0x00001000
AUTHTYPE_SERVER = 2
CERT_CHAIN_POLICY_SSL = 4
FORMAT_MESSAGE_FROM_SYSTEM = 0x00001000
FORMAT_MESSAGE_IGNORE_INSERTS = 0x00000200

# Flags to set for SSLContext.verify_mode=CERT_NONE
CERT_CHAIN_POLICY_VERIFY_MODE_NONE_FLAGS = (
    CERT_CHAIN_POLICY_IGNORE_ALL_NOT_TIME_VALID_FLAGS
    | CERT_CHAIN_POLICY_IGNORE_INVALID_BASIC_CONSTRAINTS_FLAG
    | CERT_CHAIN_POLICY_ALLOW_UNKNOWN_CA_FLAG
    | CERT_CHAIN_POLICY_IGNORE_INVALID_NAME_FLAG
    | CERT_CHAIN_POLICY_IGNORE_WRONG_USAGE_FLAG
    | CERT_CHAIN_POLICY_IGNORE_INVALID_POLICY_FLAG
    | CERT_CHAIN_POLICY_IGNORE_ALL_REV_UNKNOWN_FLAGS
    | CERT_CHAIN_POLICY_ALLOW_TESTROOT_FLAG
    | CERT_CHAIN_POLICY_TRUST_TESTROOT_FLAG
)

wincrypt = WinDLL("crypt32.dll")
kernel32 = WinDLL("kernel32.dll")


def _handle_win_error(result: bool, _: Any, args: Any) -> Any:
    if not result:
        # Note, actually raises OSError after calling GetLastError and FormatMessage
        raise WinError()
    return args


CertCreateCertificateChainEngine = wincrypt.CertCreateCertificateChainEngine
CertCreateCertificateChainEngine.argtypes = (
    PCERT_CHAIN_ENGINE_CONFIG,
    PHCERTCHAINENGINE,
)
CertCreateCertificateChainEngine.errcheck = _handle_win_error

CertOpenStore = wincrypt.CertOpenStore
CertOpenStore.argtypes = (LPCSTR, DWORD, HCRYPTPROV_LEGACY, DWORD, c_void_p)
CertOpenStore.restype = HCERTSTORE
CertOpenStore.errcheck = _handle_win_error

CertAddEncodedCertificateToStore = wincrypt.CertAddEncodedCertificateToStore
CertAddEncodedCertificateToStore.argtypes = (
    HCERTSTORE,
    DWORD,
    c_char_p,
    DWORD,
    DWORD,
    PCCERT_CONTEXT,
)
CertAddEncodedCertificateToStore.restype = BOOL

CertCreateCertificateContext = wincrypt.CertCreateCertificateContext
CertCreateCertificateContext.argtypes = (DWORD, c_char_p, DWORD)
CertCreateCertificateContext.restype = PCERT_CONTEXT
CertCreateCertificateContext.errcheck = _handle_win_error

CertGetCertificateChain = wincrypt.CertGetCertificateChain
CertGetCertificateChain.argtypes = (
    HCERTCHAINENGINE,
    PCERT_CONTEXT,
    LPFILETIME,
    HCERTSTORE,
    PCERT_CHAIN_PARA,
    DWORD,
    c_void_p,
    PCCERT_CHAIN_CONTEXT,
)
CertGetCertificateChain.restype = BOOL
CertGetCertificateChain.errcheck = _handle_win_error

CertVerifyCertificateChainPolicy = wincrypt.CertVerifyCertificateChainPolicy
CertVerifyCertificateChainPolicy.argtypes = (
    c_ulong,
    PCERT_CHAIN_CONTEXT,
    PCERT_CHAIN_POLICY_PARA,
    PCERT_CHAIN_POLICY_STATUS,
)
CertVerifyCertificateChainPolicy.restype = BOOL

CertCloseStore = wincrypt.CertCloseStore
CertCloseStore.argtypes = (HCERTSTORE, DWORD)
CertCloseStore.restype = BOOL
CertCloseStore.errcheck = _handle_win_error

CertFreeCertificateChain = wincrypt.CertFreeCertificateChain
CertFreeCertificateChain.argtypes = (PCERT_CHAIN_CONTEXT,)

CertFreeCertificateContext = wincrypt.CertFreeCertificateContext
CertFreeCertificateContext.argtypes = (PCERT_CONTEXT,)

CertFreeCertificateChainEngine = wincrypt.CertFreeCertificateChainEngine
CertFreeCertificateChainEngine.argtypes = (HCERTCHAINENGINE,)

FormatMessageW = kernel32.FormatMessageW
FormatMessageW.argtypes = (
    DWORD,
    LPCVOID,
    DWORD,
    DWORD,
    LPWSTR,
    DWORD,
    c_void_p,
)
FormatMessageW.restype = DWORD


def _verify_peercerts_impl(
    ssl_context: ssl.SSLContext,
    cert_chain: list[bytes],
    server_hostname: str | None = None,
) -> None:
    """Verify the cert_chain from the server using Windows APIs."""

    # If the peer didn't send any certificates then
    # we can't do verification. Raise an error.
    if not cert_chain:
        raise ssl.SSLCertVerificationError("Peer sent no certificates to verify")

    pCertContext = None
    hIntermediateCertStore = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, None, 0, None)
    try:
        # Add intermediate certs to an in-memory cert store
        for cert_bytes in cert_chain[1:]:
            CertAddEncodedCertificateToStore(
                hIntermediateCertStore,
                X509_ASN_ENCODING | PKCS_7_ASN_ENCODING,
                cert_bytes,
                len(cert_bytes),
                CERT_STORE_ADD_USE_EXISTING,
                None,
            )

        # Cert context for leaf cert
        leaf_cert = cert_chain[0]
        pCertContext = CertCreateCertificateContext(
            X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, leaf_cert, len(leaf_cert)
        )

        # Chain params to match certs for serverAuth extended usage
        cert_enhkey_usage = CERT_ENHKEY_USAGE()
        cert_enhkey_usage.cUsageIdentifier = 1
        cert_enhkey_usage.rgpszUsageIdentifier = (c_char_p * 1)(OID_PKIX_KP_SERVER_AUTH)
        cert_usage_match = CERT_USAGE_MATCH()
        cert_usage_match.Usage = cert_enhkey_usage
        chain_params = CERT_CHAIN_PARA()
        chain_params.RequestedUsage = cert_usage_match
        chain_params.cbSize = sizeof(chain_params)
        pChainPara = pointer(chain_params)

        if ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_CHAIN:
            chain_flags = CERT_CHAIN_REVOCATION_CHECK_CHAIN
        elif ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_LEAF:
            chain_flags = CERT_CHAIN_REVOCATION_CHECK_END_CERT
        else:
            chain_flags = 0

        try:
            # First attempt to verify using the default Windows system trust roots
            # (default chain engine).
            _get_and_verify_cert_chain(
                ssl_context,
                None,
                hIntermediateCertStore,
                pCertContext,
                pChainPara,
                server_hostname,
                chain_flags=chain_flags,
            )
        except ssl.SSLCertVerificationError as e:
            # If that fails but custom CA certs have been added
            # to the SSLContext using load_verify_locations,
            # try verifying using a custom chain engine
            # that trusts the custom CA certs.
            custom_ca_certs: list[bytes] | None = ssl_context.get_ca_certs(
                binary_form=True
            )
            if custom_ca_certs:
                try:
                    _verify_using_custom_ca_certs(
                        ssl_context,
                        custom_ca_certs,
                        hIntermediateCertStore,
                        pCertContext,
                        pChainPara,
                        server_hostname,
                        chain_flags=chain_flags,
                    )
                # Raise the original error, not the new error.
                except ssl.SSLCertVerificationError:
                    raise e from None
            else:
                raise
    finally:
        CertCloseStore(hIntermediateCertStore, 0)
        if pCertContext:
            CertFreeCertificateContext(pCertContext)


def _get_and_verify_cert_chain(
    ssl_context: ssl.SSLContext,
    hChainEngine: HCERTCHAINENGINE | None,
    hIntermediateCertStore: HCERTSTORE,
    pPeerCertContext: c_void_p,
    pChainPara: PCERT_CHAIN_PARA,  # type: ignore[valid-type]
    server_hostname: str | None,
    chain_flags: int,
) -> None:
    ppChainContext = None
    try:
        # Get cert chain
        ppChainContext = pointer(PCERT_CHAIN_CONTEXT())
        CertGetCertificateChain(
            hChainEngine,  # chain engine
            pPeerCertContext,  # leaf cert context
            None,  # current system time
            hIntermediateCertStore,  # additional in-memory cert store
            pChainPara,  # chain-building parameters
            chain_flags,
            None,  # reserved
            ppChainContext,  # the resulting chain context
        )
        pChainContext = ppChainContext.contents

        # Verify cert chain
        ssl_extra_cert_chain_policy_para = SSL_EXTRA_CERT_CHAIN_POLICY_PARA()
        ssl_extra_cert_chain_policy_para.cbSize = sizeof(
            ssl_extra_cert_chain_policy_para
        )
        ssl_extra_cert_chain_policy_para.dwAuthType = AUTHTYPE_SERVER
        ssl_extra_cert_chain_policy_para.fdwChecks = 0
        if ssl_context.check_hostname is False:
            ssl_extra_cert_chain_policy_para.fdwChecks = (
                SECURITY_FLAG_IGNORE_CERT_CN_INVALID
            )
        if server_hostname:
            ssl_extra_cert_chain_policy_para.pwszServerName = c_wchar_p(server_hostname)

        chain_policy = CERT_CHAIN_POLICY_PARA()
        chain_policy.pvExtraPolicyPara = cast(
            pointer(ssl_extra_cert_chain_policy_para), c_void_p
        )
        if ssl_context.verify_mode == ssl.CERT_NONE:
            chain_policy.dwFlags |= CERT_CHAIN_POLICY_VERIFY_MODE_NONE_FLAGS
        chain_policy.cbSize = sizeof(chain_policy)

        pPolicyPara = pointer(chain_policy)
        policy_status = CERT_CHAIN_POLICY_STATUS()
        policy_status.cbSize = sizeof(policy_status)
        pPolicyStatus = pointer(policy_status)
        CertVerifyCertificateChainPolicy(
            CERT_CHAIN_POLICY_SSL,
            pChainContext,
            pPolicyPara,
            pPolicyStatus,
        )

        # Check status
        error_code = policy_status.dwError
        if error_code:
            # Try getting a human readable message for an error code.
            error_message_buf = create_unicode_buffer(1024)
            error_message_chars = FormatMessageW(
                FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                None,
                error_code,
                0,
                error_message_buf,
                sizeof(error_message_buf),
                None,
            )

            # See if we received a message for the error,
            # otherwise we use a generic error with the
            # error code and hope that it's search-able.
            if error_message_chars <= 0:
                error_message = f"Certificate chain policy error {error_code:#x} [{policy_status.lElementIndex}]"
            else:
                error_message = error_message_buf.value.strip()

            err = ssl.SSLCertVerificationError(error_message)
            err.verify_message = error_message
            err.verify_code = error_code
            raise err from None
    finally:
        if ppChainContext:
            CertFreeCertificateChain(ppChainContext.contents)


def _verify_using_custom_ca_certs(
    ssl_context: ssl.SSLContext,
    custom_ca_certs: list[bytes],
    hIntermediateCertStore: HCERTSTORE,
    pPeerCertContext: c_void_p,
    pChainPara: PCERT_CHAIN_PARA,  # type: ignore[valid-type]
    server_hostname: str | None,
    chain_flags: int,
) -> None:
    hChainEngine = None
    hRootCertStore = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, None, 0, None)
    try:
        # Add custom CA certs to an in-memory cert store
        for cert_bytes in custom_ca_certs:
            CertAddEncodedCertificateToStore(
                hRootCertStore,
                X509_ASN_ENCODING | PKCS_7_ASN_ENCODING,
                cert_bytes,
                len(cert_bytes),
                CERT_STORE_ADD_USE_EXISTING,
                None,
            )

        # Create a custom cert chain engine which exclusively trusts
        # certs from our hRootCertStore
        cert_chain_engine_config = CERT_CHAIN_ENGINE_CONFIG()
        cert_chain_engine_config.cbSize = sizeof(cert_chain_engine_config)
        cert_chain_engine_config.hExclusiveRoot = hRootCertStore
        pConfig = pointer(cert_chain_engine_config)
        phChainEngine = pointer(HCERTCHAINENGINE())
        CertCreateCertificateChainEngine(
            pConfig,
            phChainEngine,
        )
        hChainEngine = phChainEngine.contents

        # Get and verify a cert chain using the custom chain engine
        _get_and_verify_cert_chain(
            ssl_context,
            hChainEngine,
            hIntermediateCertStore,
            pPeerCertContext,
            pChainPara,
            server_hostname,
            chain_flags,
        )
    finally:
        if hChainEngine:
            CertFreeCertificateChainEngine(hChainEngine)
        CertCloseStore(hRootCertStore, 0)


@contextlib.contextmanager
def _configure_context(ctx: ssl.SSLContext) -> typing.Iterator[None]:
    check_hostname = ctx.check_hostname
    verify_mode = ctx.verify_mode
    ctx.check_hostname = False
    _set_ssl_context_verify_mode(ctx, ssl.CERT_NONE)
    try:
        yield
    finally:
        ctx.check_hostname = check_hostname
        _set_ssl_context_verify_mode(ctx, verify_mode)
<|endoftext|>import ssl
import sys

# Hold on to the original class so we can create it consistently
# even if we inject our own SSLContext into the ssl module.
_original_SSLContext = ssl.SSLContext
_original_super_SSLContext = super(_original_SSLContext, _original_SSLContext)

# CPython is known to be good, but non-CPython implementations
# may implement SSLContext differently so to be safe we don't
# subclass the SSLContext.

# This is returned by truststore.SSLContext.__class__()
_truststore_SSLContext_dunder_class: type | None

# This value is the superclass of truststore.SSLContext.
_truststore_SSLContext_super_class: type

if sys.implementation.name == "cpython":
    _truststore_SSLContext_super_class = _original_SSLContext
    _truststore_SSLContext_dunder_class = None
else:
    _truststore_SSLContext_super_class = object
    _truststore_SSLContext_dunder_class = _original_SSLContext


def _set_ssl_context_verify_mode(
    ssl_context: ssl.SSLContext, verify_mode: ssl.VerifyMode
) -> None:
    _original_super_SSLContext.verify_mode.__set__(ssl_context, verify_mode)  # type: ignore[attr-defined]
<|endoftext|>"""Verify certificates using native system trust stores"""

import sys as _sys

if _sys.version_info < (3, 10):  # noqa: UP036
    raise ImportError("truststore requires Python 3.10 or later")

# Detect Python runtimes which don't implement SSLObject.get_unverified_chain() API
# This API only became public in Python 3.13 but was available in CPython and PyPy since 3.10.
if _sys.version_info < (3, 13) and _sys.implementation.name not in ("cpython", "pypy"):
    try:
        import ssl as _ssl
    except ImportError:
        raise ImportError("truststore requires the 'ssl' module")
    else:
        _sslmem = _ssl.MemoryBIO()
        _sslobj = _ssl.create_default_context().wrap_bio(
            _sslmem,
            _sslmem,
        )
        try:
            while not hasattr(_sslobj, "get_unverified_chain"):
                _sslobj = _sslobj._sslobj  # type: ignore[attr-defined]
        except AttributeError:
            raise ImportError(
                "truststore requires peer certificate chain APIs to be available"
            ) from None

        del _ssl, _sslobj, _sslmem

from ._api import SSLContext, extract_from_ssl, inject_into_ssl

del _api, _sys  # type: ignore[name-defined] # noqa: F821

__all__ = ["SSLContext", "inject_into_ssl", "extract_from_ssl"]
__version__ = "0.10.5"
<|endoftext|>import _ssl
import os
import platform
import socket
import ssl
import sys
import threading
import typing

from ._ssl_constants import (
    _original_SSLContext,
    _original_super_SSLContext,
    _truststore_SSLContext_dunder_class,
    _truststore_SSLContext_super_class,
)

if platform.system() == "Windows":
    from ._windows import _configure_context, _verify_peercerts_impl
elif platform.system() == "Darwin":
    from ._macos import _configure_context, _verify_peercerts_impl
else:
    from ._openssl import _configure_context, _verify_peercerts_impl

if typing.TYPE_CHECKING:
    from typing_extensions import Buffer

# From typeshed/stdlib/ssl.pyi
_StrOrBytesPath: typing.TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]
_PasswordType: typing.TypeAlias = str | bytes | typing.Callable[[], str | bytes]


def inject_into_ssl() -> None:
    """Injects the :class:`truststore.SSLContext` into the ``ssl``
    module by replacing :class:`ssl.SSLContext`.
    """
    setattr(ssl, "SSLContext", SSLContext)
    # urllib3 holds on to its own reference of ssl.SSLContext
    # so we need to replace that reference too.
    try:
        import urllib3.util.ssl_ as urllib3_ssl

        setattr(urllib3_ssl, "SSLContext", SSLContext)
    except ImportError:
        pass

    # requests starting with 2.32.0 added a preloaded SSL context to improve concurrent performance;
    # this unfortunately leads to a RecursionError, which can be avoided by patching the preloaded SSL context with
    # the truststore patched instance
    # also see https://github.com/psf/requests/pull/6667
    try:
        import requests.adapters

        preloaded_context = getattr(requests.adapters, "_preloaded_ssl_context", None)
        if preloaded_context is not None:
            setattr(
                requests.adapters,
                "_preloaded_ssl_context",
                SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            )
    except ImportError:
        pass


def extract_from_ssl() -> None:
    """Restores the :class:`ssl.SSLContext` class to its original state"""
    setattr(ssl, "SSLContext", _original_SSLContext)
    try:
        import urllib3.util.ssl_ as urllib3_ssl

        urllib3_ssl.SSLContext = _original_SSLContext
    except ImportError:
        pass


class SSLContext(_truststore_SSLContext_super_class):  # type: ignore[misc]
    """SSLContext API that uses system certificates on all platforms"""

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        # Dirty hack to get around isinstance() checks
        # for ssl.SSLContext instances in aiohttp/trustme
        # when using non-CPython implementations.
        return _truststore_SSLContext_dunder_class or SSLContext

    def __init__(self, protocol: int = None) -> None:  # type: ignore[assignment]
        self._ctx = _original_SSLContext(protocol)
        self._ctx_lock = threading.RLock()

        ctx_lock = self._ctx_lock  # Reference used by sslobject/sslsocket classes.

        class TruststoreSSLObject(ssl.SSLObject):
            # This object exists because wrap_bio() doesn't
            # immediately do the handshake so we need to do
            # certificate verifications after SSLObject.do_handshake()

            def do_handshake(self) -> None:
                with ctx_lock:
                    ret = super().do_handshake()
                    _verify_peercerts(self, server_hostname=self.server_hostname)
                    return ret

        class TruststoreSSLSocket(ssl.SSLSocket):
            # Handles do_handshake_on_connect=False, explicit do_handshake()
            # calls, and the handshake that SSLSocket.connect() triggers.
            # Must never be called inside _configure_context(). wrap_socket()
            # guarantees this by always wrapping with do_handshake_on_connect=False.
            def do_handshake(self, block: bool = False) -> None:
                with ctx_lock:
                    ret = super().do_handshake(block)
                    try:
                        _verify_peercerts(self, server_hostname=self.server_hostname)
                    except Exception:
                        self.close()
                        raise
                return ret

        self._ctx.sslsocket_class = TruststoreSSLSocket
        self._ctx.sslobject_class = TruststoreSSLObject

    def wrap_socket(
        self,
        sock: socket.socket,
        server_side: bool = False,
        do_handshake_on_connect: bool = True,
        suppress_ragged_eofs: bool = True,
        server_hostname: str | None = None,
        session: ssl.SSLSession | None = None,
    ) -> ssl.SSLSocket:
        if self.check_hostname and not server_hostname:
            raise ValueError("check_hostname requires server_hostname")

        with self._ctx_lock:
            with _configure_context(self._ctx):
                ssl_sock = self._ctx.wrap_socket(
                    sock,
                    server_side=server_side,
                    server_hostname=server_hostname,
                    do_handshake_on_connect=do_handshake_on_connect,
                    suppress_ragged_eofs=suppress_ragged_eofs,
                    session=session,
                )

            # NOTE: Below is still inside ctx_lock but not
            # inside the _configure_context().

            # Indicates that a handshake completed and
            # we can verify now, otherwise verification
            # will happen in TruststoreSSLSocket.do_handshake()
            if ssl_sock.version() is not None:
                try:
                    _verify_peercerts(ssl_sock, server_hostname=server_hostname)
                except Exception:
                    ssl_sock.close()
                    raise

        return ssl_sock

    def wrap_bio(
        self,
        incoming: ssl.MemoryBIO,
        outgoing: ssl.MemoryBIO,
        server_side: bool = False,
        server_hostname: str | None = None,
        session: ssl.SSLSession | None = None,
    ) -> ssl.SSLObject:
        if self.check_hostname and not server_hostname:
            raise ValueError("check_hostname requires server_hostname")

        # We need to lock around the .__enter__()
        # but we don't need to lock within the
        # context manager, so we need to expand the
        # syntactic sugar of the `with` statement.
        with self._ctx_lock, _configure_context(self._ctx):
            ssl_obj = self._ctx.wrap_bio(
                incoming,
                outgoing,
                server_hostname=server_hostname,
                server_side=server_side,
                session=session,
            )
        return ssl_obj

    def load_verify_locations(
        self,
        cafile: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None = None,
        capath: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None = None,
        cadata: typing.Union[str, "Buffer", None] = None,
    ) -> None:
        return self._ctx.load_verify_locations(
            cafile=cafile, capath=capath, cadata=cadata
        )

    def load_cert_chain(
        self,
        certfile: _StrOrBytesPath,
        keyfile: _StrOrBytesPath | None = None,
        password: _PasswordType | None = None,
    ) -> None:
        return self._ctx.load_cert_chain(
            certfile=certfile, keyfile=keyfile, password=password
        )

    def load_default_certs(
        self, purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH
    ) -> None:
        return self._ctx.load_default_certs(purpose)

    def set_alpn_protocols(self, alpn_protocols: typing.Iterable[str]) -> None:
        return self._ctx.set_alpn_protocols(alpn_protocols)

    def set_npn_protocols(self, npn_protocols: typing.Iterable[str]) -> None:
        return self._ctx.set_npn_protocols(npn_protocols)

    def set_ciphers(self, __cipherlist: str) -> None:
        return self._ctx.set_ciphers(__cipherlist)

    def get_ciphers(self) -> typing.Any:
        return self._ctx.get_ciphers()

    def session_stats(self) -> dict[str, int]:
        return self._ctx.session_stats()

    def cert_store_stats(self) -> dict[str, int]:
        raise NotImplementedError()

    def set_default_verify_paths(self) -> None:
        self._ctx.set_default_verify_paths()

    @typing.overload
    def get_ca_certs(
        self, binary_form: typing.Literal[False] = ...
    ) -> list[typing.Any]: ...

    @typing.overload
    def get_ca_certs(self, binary_form: typing.Literal[True] = ...) -> list[bytes]: ...

    @typing.overload
    def get_ca_certs(self, binary_form: bool = ...) -> typing.Any: ...

    def get_ca_certs(self, binary_form: bool = False) -> list[typing.Any] | list[bytes]:
        raise NotImplementedError()

    @property
    def check_hostname(self) -> bool:
        with self._ctx_lock:
            return self._ctx.check_hostname

    @check_hostname.setter
    def check_hostname(self, value: bool) -> None:
        with self._ctx_lock:
            self._ctx.check_hostname = value

    @property
    def hostname_checks_common_name(self) -> bool:
        return self._ctx.hostname_checks_common_name

    @hostname_checks_common_name.setter
    def hostname_checks_common_name(self, value: bool) -> None:
        self._ctx.hostname_checks_common_name = value

    @property
    def keylog_filename(self) -> str:
        return self._ctx.keylog_filename

    @keylog_filename.setter
    def keylog_filename(self, value: str) -> None:
        self._ctx.keylog_filename = value

    @property
    def maximum_version(self) -> ssl.TLSVersion:
        return self._ctx.maximum_version

    @maximum_version.setter
    def maximum_version(self, value: ssl.TLSVersion) -> None:
        _original_super_SSLContext.maximum_version.__set__(  # type: ignore[attr-defined]
            self._ctx, value
        )

    @property
    def minimum_version(self) -> ssl.TLSVersion:
        return self._ctx.minimum_version

    @minimum_version.setter
    def minimum_version(self, value: ssl.TLSVersion) -> None:
        _original_super_SSLContext.minimum_version.__set__(  # type: ignore[attr-defined]
            self._ctx, value
        )

    @property
    def options(self) -> ssl.Options:
        return self._ctx.options

    @options.setter
    def options(self, value: ssl.Options) -> None:
        _original_super_SSLContext.options.__set__(  # type: ignore[attr-defined]
            self._ctx, value
        )

    @property
    def post_handshake_auth(self) -> bool:
        return self._ctx.post_handshake_auth

    @post_handshake_auth.setter
    def post_handshake_auth(self, value: bool) -> None:
        self._ctx.post_handshake_auth = value

    @property
    def protocol(self) -> ssl._SSLMethod:
        return self._ctx.protocol

    @property
    def security_level(self) -> int:
        return self._ctx.security_level

    @property
    def verify_flags(self) -> ssl.VerifyFlags:
        return self._ctx.verify_flags

    @verify_flags.setter
    def verify_flags(self, value: ssl.VerifyFlags) -> None:
        _original_super_SSLContext.verify_flags.__set__(  # type: ignore[attr-defined]
            self._ctx, value
        )

    @property
    def verify_mode(self) -> ssl.VerifyMode:
        with self._ctx_lock:
            return self._ctx.verify_mode

    @verify_mode.setter
    def verify_mode(self, value: ssl.VerifyMode) -> None:
        with self._ctx_lock:
            _original_super_SSLContext.verify_mode.__set__(  # type: ignore[attr-defined]
                self._ctx, value
            )


# Python 3.13+ makes get_unverified_chain() a public API that only returns DER
# encoded certificates. We detect whether we need to call public_bytes() for 3.10->3.12
# Pre-3.13 returned None instead of an empty list from get_unverified_chain()
if sys.version_info >= (3, 13):

    def _get_unverified_chain_bytes(sslobj: ssl.SSLObject) -> list[bytes]:
        unverified_chain = sslobj.get_unverified_chain() or ()
        return [
            cert if isinstance(cert, bytes) else cert.public_bytes(_ssl.ENCODING_DER)
            for cert in unverified_chain
        ]

else:

    def _get_unverified_chain_bytes(sslobj: ssl.SSLObject) -> list[bytes]:
        unverified_chain = sslobj.get_unverified_chain() or ()  # type: ignore[attr-defined]
        return [cert.public_bytes(_ssl.ENCODING_DER) for cert in unverified_chain]


def _verify_peercerts(
    sock_or_sslobj: ssl.SSLSocket | ssl.SSLObject, server_hostname: str | None
) -> None:
    """
    Verifies the peer certificates from an SSLSocket or SSLObject
    against the certificates in the OS trust store.
    """
    if sock_or_sslobj.context.check_hostname and not server_hostname:
        raise ValueError("check_hostname requires server_hostname")

    sslobj: ssl.SSLObject = sock_or_sslobj  # type: ignore[assignment]
    try:
        while not hasattr(sslobj, "get_unverified_chain"):
            sslobj = sslobj._sslobj  # type: ignore[attr-defined]
    except AttributeError:
        pass

    cert_bytes = _get_unverified_chain_bytes(sslobj)
    _verify_peercerts_impl(
        sock_or_sslobj.context, cert_bytes, server_hostname=server_hostname
    )
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from contourpy import FillType, LineType
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.types import MOVETO, code_dtype, offset_dtype, point_dtype

if TYPE_CHECKING:
    import contourpy._contourpy as cpy


# Minimalist array-checking functions that check dtype, ndims and shape only.
# They do not walk the arrays to check the contents for performance reasons.
def check_code_array(codes: Any) -> None:
    if not isinstance(codes, np.ndarray):
        raise TypeError(f"Expected numpy array not {type(codes)}")
    if codes.dtype != code_dtype:
        raise ValueError(f"Expected numpy array of dtype {code_dtype} not {codes.dtype}")
    if not (codes.ndim == 1 and len(codes) > 1):
        raise ValueError(f"Expected numpy array of shape (?,) not {codes.shape}")
    if codes[0] != MOVETO:
        raise ValueError(f"First element of code array must be {MOVETO}, not {codes[0]}")


def check_offset_array(offsets: Any) -> None:
    if not isinstance(offsets, np.ndarray):
        raise TypeError(f"Expected numpy array not {type(offsets)}")
    if offsets.dtype != offset_dtype:
        raise ValueError(f"Expected numpy array of dtype {offset_dtype} not {offsets.dtype}")
    if not (offsets.ndim == 1 and len(offsets) > 1):
        raise ValueError(f"Expected numpy array of shape (?,) not {offsets.shape}")
    if offsets[0] != 0:
        raise ValueError(f"First element of offset array must be 0, not {offsets[0]}")


def check_point_array(points: Any) -> None:
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Expected numpy array not {type(points)}")
    if points.dtype != point_dtype:
        raise ValueError(f"Expected numpy array of dtype {point_dtype} not {points.dtype}")
    if not (points.ndim == 2 and points.shape[1] ==2 and points.shape[0] > 1):
        raise ValueError(f"Expected numpy array of shape (?, 2) not {points.shape}")


def _check_tuple_of_lists_with_same_length(
    maybe_tuple: Any,
    tuple_length: int,
    allow_empty_lists: bool = True,
) -> None:
    if not isinstance(maybe_tuple, tuple):
        raise TypeError(f"Expected tuple not {type(maybe_tuple)}")
    if len(maybe_tuple) != tuple_length:
        raise ValueError(f"Expected tuple of length {tuple_length} not {len(maybe_tuple)}")
    for maybe_list in maybe_tuple:
        if not isinstance(maybe_list, list):
            msg = f"Expected tuple to contain {tuple_length} lists but found a {type(maybe_list)}"
            raise TypeError(msg)
    lengths = [len(item) for item in maybe_tuple]
    if len(set(lengths)) != 1:
        msg = f"Expected {tuple_length} lists with same length but lengths are {lengths}"
        raise ValueError(msg)
    if not allow_empty_lists and lengths[0] == 0:
        raise ValueError(f"Expected {tuple_length} non-empty lists")


def check_filled(filled: cpy.FillReturn, fill_type: FillType | str) -> None:
    fill_type = as_fill_type(fill_type)

    if fill_type == FillType.OuterCode:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_OuterCode, filled)
        _check_tuple_of_lists_with_same_length(filled, 2)
        for i, (points, codes) in enumerate(zip(*filled)):
            check_point_array(points)
            check_code_array(codes)
            if len(points) != len(codes):
                raise ValueError(f"Points and codes have different lengths in polygon {i}")
    elif fill_type == FillType.OuterOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_OuterOffset, filled)
        _check_tuple_of_lists_with_same_length(filled, 2)
        for i, (points, offsets) in enumerate(zip(*filled)):
            check_point_array(points)
            check_offset_array(offsets)
            if offsets[-1] != len(points):
                raise ValueError(f"Inconsistent points and offsets in polygon {i}")
    elif fill_type == FillType.ChunkCombinedCode:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCode, filled)
        _check_tuple_of_lists_with_same_length(filled, 2, allow_empty_lists=False)
        for chunk, (points_or_none, codes_or_none) in enumerate(zip(*filled)):
            if points_or_none is not None and codes_or_none is not None:
                check_point_array(points_or_none)
                check_code_array(codes_or_none)
                if len(points_or_none) != len(codes_or_none):
                    raise ValueError(f"Points and codes have different lengths in chunk {chunk}")
            elif not (points_or_none is None and codes_or_none is None):
                raise ValueError(f"Inconsistent Nones in chunk {chunk}")
    elif fill_type == FillType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffset, filled)
        _check_tuple_of_lists_with_same_length(filled, 2, allow_empty_lists=False)
        for chunk, (points_or_none, offsets_or_none) in enumerate(zip(*filled)):
            if points_or_none is not None and offsets_or_none is not None:
                check_point_array(points_or_none)
                check_offset_array(offsets_or_none)
                if offsets_or_none[-1] != len(points_or_none):
                    raise ValueError(f"Inconsistent points and offsets in chunk {chunk}")
            elif not (points_or_none is None and offsets_or_none is None):
                raise ValueError(f"Inconsistent Nones in chunk {chunk}")
    elif fill_type == FillType.ChunkCombinedCodeOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCodeOffset, filled)
        _check_tuple_of_lists_with_same_length(filled, 3, allow_empty_lists=False)
        for i, (points_or_none, codes_or_none, outer_offsets_or_none) in enumerate(zip(*filled)):
            if (points_or_none is not None and codes_or_none is not None and
                    outer_offsets_or_none is not None):
                check_point_array(points_or_none)
                check_code_array(codes_or_none)
                check_offset_array(outer_offsets_or_none)
                if len(codes_or_none) != len(points_or_none):
                    raise ValueError(f"Points and codes have different lengths in chunk {i}")
                if outer_offsets_or_none[-1] != len(codes_or_none):
                    raise ValueError(f"Inconsistent codes and outer_offsets in chunk {i}")
            elif not (points_or_none is None and codes_or_none is None and
                      outer_offsets_or_none is None):
                raise ValueError(f"Inconsistent Nones in chunk {i}")
    elif fill_type == FillType.ChunkCombinedOffsetOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffsetOffset, filled)
        _check_tuple_of_lists_with_same_length(filled, 3, allow_empty_lists=False)
        for i, (points_or_none, offsets_or_none, outer_offsets_or_none) in enumerate(zip(*filled)):
            if (points_or_none is not None and offsets_or_none is not None and
                    outer_offsets_or_none is not None):
                check_point_array(points_or_none)
                check_offset_array(offsets_or_none)
                check_offset_array(outer_offsets_or_none)
                if offsets_or_none[-1] != len(points_or_none):
                    raise ValueError(f"Inconsistent points and offsets in chunk {i}")
                if outer_offsets_or_none[-1] != len(offsets_or_none) - 1:
                    raise ValueError(f"Inconsistent offsets and outer_offsets in chunk {i}")
            elif not (points_or_none is None and offsets_or_none is None and
                      outer_offsets_or_none is None):
                raise ValueError(f"Inconsistent Nones in chunk {i}")
    else:
        raise ValueError(f"Invalid FillType {fill_type}")


def check_lines(lines: cpy.LineReturn, line_type: LineType | str) -> None:
    line_type = as_line_type(line_type)

    if line_type == LineType.Separate:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_Separate, lines)
        if not isinstance(lines, list):
            raise TypeError(f"Expected list not {type(lines)}")
        for points in lines:
            check_point_array(points)
    elif line_type == LineType.SeparateCode:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_SeparateCode, lines)
        _check_tuple_of_lists_with_same_length(lines, 2)
        for i, (points, codes) in enumerate(zip(*lines)):
            check_point_array(points)
            check_code_array(codes)
            if len(points) != len(codes):
                raise ValueError(f"Points and codes have different lengths in line {i}")
    elif line_type == LineType.ChunkCombinedCode:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedCode, lines)
        _check_tuple_of_lists_with_same_length(lines, 2, allow_empty_lists=False)
        for chunk, (points_or_none, codes_or_none) in enumerate(zip(*lines)):
            if points_or_none is not None and codes_or_none is not None:
                check_point_array(points_or_none)
                check_code_array(codes_or_none)
                if len(points_or_none) != len(codes_or_none):
                    raise ValueError(f"Points and codes have different lengths in chunk {chunk}")
            elif not (points_or_none is None and codes_or_none is None):
                raise ValueError(f"Inconsistent Nones in chunk {chunk}")
    elif line_type == LineType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedOffset, lines)
        _check_tuple_of_lists_with_same_length(lines, 2, allow_empty_lists=False)
        for chunk, (points_or_none, offsets_or_none) in enumerate(zip(*lines)):
            if points_or_none is not None and offsets_or_none is not None:
                check_point_array(points_or_none)
                check_offset_array(offsets_or_none)
                if offsets_or_none[-1] != len(points_or_none):
                    raise ValueError(f"Inconsistent points and offsets in chunk {chunk}")
            elif not (points_or_none is None and offsets_or_none is None):
                raise ValueError(f"Inconsistent Nones in chunk {chunk}")
    elif line_type == LineType.ChunkCombinedNan:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedNan, lines)
        _check_tuple_of_lists_with_same_length(lines, 1, allow_empty_lists=False)
        for _chunk, points_or_none in enumerate(lines[0]):
            if points_or_none is not None:
                check_point_array(points_or_none)
    else:
        raise ValueError(f"Invalid LineType {line_type}")
<|endoftext|>from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, cast

import numpy as np

from contourpy._contourpy import FillType, LineType
import contourpy.array as arr
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines
from contourpy.types import MOVETO, offset_dtype

if TYPE_CHECKING:
    import contourpy._contourpy as cpy


def _convert_filled_from_OuterCode(
    filled: cpy.FillReturn_OuterCode,
    fill_type_to: FillType,
) -> cpy.FillReturn:
    if fill_type_to == FillType.OuterCode:
        return filled
    elif fill_type_to == FillType.OuterOffset:
        return (filled[0], [arr.offsets_from_codes(codes) for codes in filled[1]])

    if len(filled[0]) > 0:
        points = arr.concat_points(filled[0])
        codes = arr.concat_codes(filled[1])
    else:
        points = None
        codes = None

    if fill_type_to == FillType.ChunkCombinedCode:
        return ([points], [codes])
    elif fill_type_to == FillType.ChunkCombinedOffset:
        return ([points], [None if codes is None else arr.offsets_from_codes(codes)])
    elif fill_type_to == FillType.ChunkCombinedCodeOffset:
        outer_offsets = None if points is None else arr.offsets_from_lengths(filled[0])
        ret1: cpy.FillReturn_ChunkCombinedCodeOffset = ([points], [codes], [outer_offsets])
        return ret1
    elif fill_type_to == FillType.ChunkCombinedOffsetOffset:
        if codes is None:
            ret2: cpy.FillReturn_ChunkCombinedOffsetOffset = ([None], [None], [None])
        else:
            offsets = arr.offsets_from_codes(codes)
            outer_offsets = arr.outer_offsets_from_list_of_codes(filled[1])
            ret2 = ([points], [offsets], [outer_offsets])
        return ret2
    else:
        raise ValueError(f"Invalid FillType {fill_type_to}")


def _convert_filled_from_OuterOffset(
    filled: cpy.FillReturn_OuterOffset,
    fill_type_to: FillType,
) -> cpy.FillReturn:
    if fill_type_to == FillType.OuterCode:
        separate_codes = [arr.codes_from_offsets(offsets) for offsets in filled[1]]
        return (filled[0], separate_codes)
    elif fill_type_to == FillType.OuterOffset:
        return filled

    if len(filled[0]) > 0:
        points = arr.concat_points(filled[0])
        offsets = arr.concat_offsets(filled[1])
    else:
        points = None
        offsets = None

    if fill_type_to == FillType.ChunkCombinedCode:
        return ([points], [None if offsets is None else arr.codes_from_offsets(offsets)])
    elif fill_type_to == FillType.ChunkCombinedOffset:
        return ([points], [offsets])
    elif fill_type_to == FillType.ChunkCombinedCodeOffset:
        if offsets is None:
            ret1: cpy.FillReturn_ChunkCombinedCodeOffset = ([None], [None], [None])
        else:
            codes = arr.codes_from_offsets(offsets)
            outer_offsets = arr.offsets_from_lengths(filled[0])
            ret1 = ([points], [codes], [outer_offsets])
        return ret1
    elif fill_type_to == FillType.ChunkCombinedOffsetOffset:
        if points is None:
            ret2: cpy.FillReturn_ChunkCombinedOffsetOffset = ([None], [None], [None])
        else:
            outer_offsets = arr.outer_offsets_from_list_of_offsets(filled[1])
            ret2 = ([points], [offsets], [outer_offsets])
        return ret2
    else:
        raise ValueError(f"Invalid FillType {fill_type_to}")


def _convert_filled_from_ChunkCombinedCode(
    filled: cpy.FillReturn_ChunkCombinedCode,
    fill_type_to: FillType,
) -> cpy.FillReturn:
    if fill_type_to == FillType.ChunkCombinedCode:
        return filled
    elif fill_type_to == FillType.ChunkCombinedOffset:
        codes = [None if codes is None else arr.offsets_from_codes(codes) for codes in filled[1]]
        return (filled[0], codes)
    else:
        raise ValueError(
            f"Conversion from {FillType.ChunkCombinedCode} to {fill_type_to} not supported")


def _convert_filled_from_ChunkCombinedOffset(
    filled: cpy.FillReturn_ChunkCombinedOffset,
    fill_type_to: FillType,
) -> cpy.FillReturn:
    if fill_type_to == FillType.ChunkCombinedCode:
        chunk_codes: list[cpy.CodeArray | None] = []
        for points, offsets in zip(*filled):
            if points is None:
                chunk_codes.append(None)
            else:
                if TYPE_CHECKING:
                    assert offsets is not None
                chunk_codes.append(arr.codes_from_offsets_and_points(offsets, points))
        return (filled[0], chunk_codes)
    elif fill_type_to == FillType.ChunkCombinedOffset:
        return filled
    else:
        raise ValueError(
            f"Conversion from {FillType.ChunkCombinedOffset} to {fill_type_to} not supported")


def _convert_filled_from_ChunkCombinedCodeOffset(
    filled: cpy.FillReturn_ChunkCombinedCodeOffset,
    fill_type_to: FillType,
) -> cpy.FillReturn:
    if fill_type_to == FillType.OuterCode:
        separate_points = []
        separate_codes = []
        for points, codes, outer_offsets in zip(*filled):
            if points is not None:
                if TYPE_CHECKING:
                    assert codes is not None
                    assert outer_offsets is not None
                separate_points += arr.split_points_by_offsets(points, outer_offsets)
                separate_codes += arr.split_codes_by_offsets(codes, outer_offsets)
        return (separate_points, separate_codes)
    elif fill_type_to == FillType.OuterOffset:
        separate_points = []
        separate_offsets = []
        for points, codes, outer_offsets in zip(*filled):
            if points is not None:
                if TYPE_CHECKING:
                    assert codes is not None
                    assert outer_offsets is not None
                separate_points += arr.split_points_by_offsets(points, outer_offsets)
                separate_codes = arr.split_codes_by_offsets(codes, outer_offsets)
                separate_offsets += [arr.offsets_from_codes(codes) for codes in separate_codes]
        return (separate_points, separate_offsets)
    elif fill_type_to == FillType.ChunkCombinedCode:
        ret1: cpy.FillReturn_ChunkCombinedCode = (filled[0], filled[1])
        return ret1
    elif fill_type_to == FillType.ChunkCombinedOffset:
        all_offsets = [None if codes is None else arr.offsets_from_codes(codes)
                       for codes in filled[1]]
        ret2: cpy.FillReturn_ChunkCombinedOffset = (filled[0], all_offsets)
        return ret2
    elif fill_type_to == FillType.ChunkCombinedCodeOffset:
        return filled
    elif fill_type_to == FillType.ChunkCombinedOffsetOffset:
        chunk_offsets: list[cpy.OffsetArray | None] = []
        chunk_outer_offsets: list[cpy.OffsetArray | None] = []
        for codes, outer_offsets in zip(*filled[1:]):
            if codes is None:
                chunk_offsets.append(None)
                chunk_outer_offsets.append(None)
            else:
                if TYPE_CHECKING:
                    assert outer_offsets is not None
                offsets = arr.offsets_from_codes(codes)
                outer_offsets = np.array([np.nonzero(offsets == oo)[0][0] for oo in outer_offsets],
                                         dtype=offset_dtype)
                chunk_offsets.append(offsets)
                chunk_outer_offsets.append(outer_offsets)
        ret3: cpy.FillReturn_ChunkCombinedOffsetOffset = (
            filled[0], chunk_offsets, chunk_outer_offsets,
        )
        return ret3
    else:
        raise ValueError(f"Invalid FillType {fill_type_to}")


def _convert_filled_from_ChunkCombinedOffsetOffset(
    filled: cpy.FillReturn_ChunkCombinedOffsetOffset,
    fill_type_to: FillType,
) -> cpy.FillReturn:
    if fill_type_to == FillType.OuterCode:
        separate_points = []
        separate_codes = []
        for points, offsets, outer_offsets in zip(*filled):
            if points is not None:
                if TYPE_CHECKING:
                    assert offsets is not None
                    assert outer_offsets is not None
                codes = arr.codes_from_offsets_and_points(offsets, points)
                outer_offsets = offsets[outer_offsets]
                separate_points += arr.split_points_by_offsets(points, outer_offsets)
                separate_codes += arr.split_codes_by_offsets(codes, outer_offsets)
        return (separate_points, separate_codes)
    elif fill_type_to == FillType.OuterOffset:
        separate_points = []
        separate_offsets = []
        for points, offsets, outer_offsets in zip(*filled):
            if points is not None:
                if TYPE_CHECKING:
                    assert offsets is not None
                    assert outer_offsets is not None
                if len(outer_offsets) > 2:
                    separate_offsets += [offsets[s:e+1] - offsets[s] for s, e in
                                         pairwise(outer_offsets)]
                else:
                    separate_offsets.append(offsets)
                separate_points += arr.split_points_by_offsets(points, offsets[outer_offsets])
        return (separate_points, separate_offsets)
    elif fill_type_to == FillType.ChunkCombinedCode:
        chunk_codes: list[cpy.CodeArray | None] = []
        for points, offsets, outer_offsets in zip(*filled):
            if points is None:
                chunk_codes.append(None)
            else:
                if TYPE_CHECKING:
                    assert offsets is not None
                    assert outer_offsets is not None
                chunk_codes.append(arr.codes_from_offsets_and_points(offsets, points))
        ret1: cpy.FillReturn_ChunkCombinedCode = (filled[0], chunk_codes)
        return ret1
    elif fill_type_to == FillType.ChunkCombinedOffset:
        return (filled[0], filled[1])
    elif fill_type_to == FillType.ChunkCombinedCodeOffset:
        chunk_codes = []
        chunk_outer_offsets: list[cpy.OffsetArray | None] = []
        for points, offsets, outer_offsets in zip(*filled):
            if points is None:
                chunk_codes.append(None)
                chunk_outer_offsets.append(None)
            else:
                if TYPE_CHECKING:
                    assert offsets is not None
                    assert outer_offsets is not None
                chunk_codes.append(arr.codes_from_offsets_and_points(offsets, points))
                chunk_outer_offsets.append(offsets[outer_offsets])
        ret2: cpy.FillReturn_ChunkCombinedCodeOffset = (filled[0], chunk_codes, chunk_outer_offsets)
        return ret2
    elif fill_type_to == FillType.ChunkCombinedOffsetOffset:
        return filled
    else:
        raise ValueError(f"Invalid FillType {fill_type_to}")


def convert_filled(
    filled: cpy.FillReturn,
    fill_type_from: FillType | str,
    fill_type_to:  FillType | str,
) -> cpy.FillReturn:
    """Convert filled contours from one :class:`~.FillType` to another.

    Args:
        filled (sequence of arrays): Filled contour polygons to convert, such as those returned by
            :meth:`.ContourGenerator.filled`.
        fill_type_from (FillType or str): :class:`~.FillType` to convert from as enum or
            string equivalent.
        fill_type_to (FillType or str): :class:`~.FillType` to convert to as enum or string
            equivalent.

    Return:
        Converted filled contour polygons.

    When converting non-chunked fill types (``FillType.OuterCode`` or ``FillType.OuterOffset``) to
    chunked ones, all polygons are placed in the first chunk. When converting in the other
    direction, all chunk information is discarded. Converting a fill type that is not aware of the
    relationship between outer boundaries and contained holes (``FillType.ChunkCombinedCode`` or
    ``FillType.ChunkCombinedOffset``) to one that is will raise a ``ValueError``.

    .. versionadded:: 1.2.0
    """
    fill_type_from = as_fill_type(fill_type_from)
    fill_type_to = as_fill_type(fill_type_to)

    check_filled(filled, fill_type_from)

    if fill_type_from == FillType.OuterCode:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_OuterCode, filled)
        return _convert_filled_from_OuterCode(filled, fill_type_to)
    elif fill_type_from == FillType.OuterOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_OuterOffset, filled)
        return _convert_filled_from_OuterOffset(filled, fill_type_to)
    elif fill_type_from == FillType.ChunkCombinedCode:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCode, filled)
        return _convert_filled_from_ChunkCombinedCode(filled, fill_type_to)
    elif fill_type_from == FillType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffset, filled)
        return _convert_filled_from_ChunkCombinedOffset(filled, fill_type_to)
    elif fill_type_from == FillType.ChunkCombinedCodeOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCodeOffset, filled)
        return _convert_filled_from_ChunkCombinedCodeOffset(filled, fill_type_to)
    elif fill_type_from == FillType.ChunkCombinedOffsetOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffsetOffset, filled)
        return _convert_filled_from_ChunkCombinedOffsetOffset(filled, fill_type_to)
    else:
        raise ValueError(f"Invalid FillType {fill_type_from}")


def _convert_lines_from_Separate(
    lines: cpy.LineReturn_Separate,
    line_type_to: LineType,
) -> cpy.LineReturn:
    if line_type_to == LineType.Separate:
        return lines
    elif line_type_to == LineType.SeparateCode:
        separate_codes = [arr.codes_from_points(line) for line in lines]
        return (lines, separate_codes)
    elif line_type_to == LineType.ChunkCombinedCode:
        if not lines:
            ret1: cpy.LineReturn_ChunkCombinedCode = ([None], [None])
        else:
            points = arr.concat_points(lines)
            offsets = arr.offsets_from_lengths(lines)
            codes = arr.codes_from_offsets_and_points(offsets, points)
            ret1 = ([points], [codes])
        return ret1
    elif line_type_to == LineType.ChunkCombinedOffset:
        if not lines:
            ret2: cpy.LineReturn_ChunkCombinedOffset = ([None], [None])
        else:
            ret2 = ([arr.concat_points(lines)], [arr.offsets_from_lengths(lines)])
        return ret2
    elif line_type_to == LineType.ChunkCombinedNan:
        if not lines:
            ret3: cpy.LineReturn_ChunkCombinedNan = ([None],)
        else:
            ret3 = ([arr.concat_points_with_nan(lines)],)
        return ret3
    else:
        raise ValueError(f"Invalid LineType {line_type_to}")


def _convert_lines_from_SeparateCode(
    lines: cpy.LineReturn_SeparateCode,
    line_type_to: LineType,
) -> cpy.LineReturn:
    if line_type_to == LineType.Separate:
        # Drop codes.
        return lines[0]
    elif line_type_to == LineType.SeparateCode:
        return lines
    elif line_type_to == LineType.ChunkCombinedCode:
        if not lines[0]:
            ret1: cpy.LineReturn_ChunkCombinedCode = ([None], [None])
        else:
            ret1 = ([arr.concat_points(lines[0])], [arr.concat_codes(lines[1])])
        return ret1
    elif line_type_to == LineType.ChunkCombinedOffset:
        if not lines[0]:
            ret2: cpy.LineReturn_ChunkCombinedOffset = ([None], [None])
        else:
            ret2 = ([arr.concat_points(lines[0])], [arr.offsets_from_lengths(lines[0])])
        return ret2
    elif line_type_to == LineType.ChunkCombinedNan:
        if not lines[0]:
            ret3: cpy.LineReturn_ChunkCombinedNan = ([None],)
        else:
            ret3 = ([arr.concat_points_with_nan(lines[0])],)
        return ret3
    else:
        raise ValueError(f"Invalid LineType {line_type_to}")


def _convert_lines_from_ChunkCombinedCode(
    lines: cpy.LineReturn_ChunkCombinedCode,
    line_type_to: LineType,
) -> cpy.LineReturn:
    if line_type_to in (LineType.Separate, LineType.SeparateCode):
        separate_lines = []
        for points, codes in zip(*lines):
            if points is not None:
                if TYPE_CHECKING:
                    assert codes is not None
                split_at = np.nonzero(codes == MOVETO)[0]
                if len(split_at) > 1:
                    separate_lines += np.split(points, split_at[1:])
                else:
                    separate_lines.append(points)
        if line_type_to == LineType.Separate:
            return separate_lines
        else:
            separate_codes = [arr.codes_from_points(line) for line in separate_lines]
            return (separate_lines, separate_codes)
    elif line_type_to == LineType.ChunkCombinedCode:
        return lines
    elif line_type_to == LineType.ChunkCombinedOffset:
        chunk_offsets = [None if codes is None else arr.offsets_from_codes(codes)
                         for codes in lines[1]]
        return (lines[0], chunk_offsets)
    elif line_type_to == LineType.ChunkCombinedNan:
        points_nan: list[cpy.PointArray | None] = []
        for points, codes in zip(*lines):
            if points is None:
                points_nan.append(None)
            else:
                if TYPE_CHECKING:
                    assert codes is not None
                offsets = arr.offsets_from_codes(codes)
                points_nan.append(arr.insert_nan_at_offsets(points, offsets))
        return (points_nan,)
    else:
        raise ValueError(f"Invalid LineType {line_type_to}")


def _convert_lines_from_ChunkCombinedOffset(
    lines: cpy.LineReturn_ChunkCombinedOffset,
    line_type_to: LineType,
) -> cpy.LineReturn:
    if line_type_to in (LineType.Separate, LineType.SeparateCode):
        separate_lines = []
        for points, offsets in zip(*lines):
            if points is not None:
                if TYPE_CHECKING:
                    assert offsets is not None
                separate_lines += arr.split_points_by_offsets(points, offsets)
        if line_type_to == LineType.Separate:
            return separate_lines
        else:
            separate_codes = [arr.codes_from_points(line) for line in separate_lines]
            return (separate_lines, separate_codes)
    elif line_type_to == LineType.ChunkCombinedCode:
        chunk_codes: list[cpy.CodeArray | None] = []
        for points, offsets in zip(*lines):
            if points is None:
                chunk_codes.append(None)
            else:
                if TYPE_CHECKING:
                    assert offsets is not None
                chunk_codes.append(arr.codes_from_offsets_and_points(offsets, points))
        return (lines[0], chunk_codes)
    elif line_type_to == LineType.ChunkCombinedOffset:
        return lines
    elif line_type_to == LineType.ChunkCombinedNan:
        points_nan: list[cpy.PointArray | None] = []
        for points, offsets in zip(*lines):
            if points is None:
                points_nan.append(None)
            else:
                if TYPE_CHECKING:
                    assert offsets is not None
                points_nan.append(arr.insert_nan_at_offsets(points, offsets))
        return (points_nan,)
    else:
        raise ValueError(f"Invalid LineType {line_type_to}")


def _convert_lines_from_ChunkCombinedNan(
    lines: cpy.LineReturn_ChunkCombinedNan,
    line_type_to: LineType,
) -> cpy.LineReturn:
    if line_type_to in (LineType.Separate, LineType.SeparateCode):
        separate_lines = []
        for points in lines[0]:
            if points is not None:
                separate_lines += arr.split_points_at_nan(points)
        if line_type_to == LineType.Separate:
            return separate_lines
        else:
            separate_codes = [arr.codes_from_points(points) for points in separate_lines]
            return (separate_lines, separate_codes)
    elif line_type_to == LineType.ChunkCombinedCode:
        chunk_points: list[cpy.PointArray | None] = []
        chunk_codes: list[cpy.CodeArray | None] = []
        for points in lines[0]:
            if points is None:
                chunk_points.append(None)
                chunk_codes.append(None)
            else:
                points, offsets = arr.remove_nan(points)
                chunk_points.append(points)
                chunk_codes.append(arr.codes_from_offsets_and_points(offsets, points))
        return (chunk_points, chunk_codes)
    elif line_type_to == LineType.ChunkCombinedOffset:
        chunk_points = []
        chunk_offsets: list[cpy.OffsetArray | None] = []
        for points in lines[0]:
            if points is None:
                chunk_points.append(None)
                chunk_offsets.append(None)
            else:
                points, offsets = arr.remove_nan(points)
                chunk_points.append(points)
                chunk_offsets.append(offsets)
        return (chunk_points, chunk_offsets)
    elif line_type_to == LineType.ChunkCombinedNan:
        return lines
    else:
        raise ValueError(f"Invalid LineType {line_type_to}")


def convert_lines(
    lines: cpy.LineReturn,
    line_type_from: LineType | str,
    line_type_to:  LineType | str,
) -> cpy.LineReturn:
    """Convert contour lines from one :class:`~.LineType` to another.

    Args:
        lines (sequence of arrays): Contour lines to convert, such as those returned by
            :meth:`.ContourGenerator.lines`.
        line_type_from (LineType or str): :class:`~.LineType` to convert from as enum or
            string equivalent.
        line_type_to (LineType or str): :class:`~.LineType` to convert to as enum or string
            equivalent.

    Return:
        Converted contour lines.

    When converting non-chunked line types (``LineType.Separate`` or ``LineType.SeparateCode``) to
    chunked ones (``LineType.ChunkCombinedCode``, ``LineType.ChunkCombinedOffset`` or
    ``LineType.ChunkCombinedNan``), all lines are placed in the first chunk. When converting in the
    other direction, all chunk information is discarded.

    .. versionadded:: 1.2.0
    """
    line_type_from = as_line_type(line_type_from)
    line_type_to = as_line_type(line_type_to)

    check_lines(lines, line_type_from)

    if line_type_from == LineType.Separate:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_Separate, lines)
        return _convert_lines_from_Separate(lines, line_type_to)
    elif line_type_from == LineType.SeparateCode:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_SeparateCode, lines)
        return _convert_lines_from_SeparateCode(lines, line_type_to)
    elif line_type_from == LineType.ChunkCombinedCode:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedCode, lines)
        return _convert_lines_from_ChunkCombinedCode(lines, line_type_to)
    elif line_type_from == LineType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedOffset, lines)
        return _convert_lines_from_ChunkCombinedOffset(lines, line_type_to)
    elif line_type_from == LineType.ChunkCombinedNan:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedNan, lines)
        return _convert_lines_from_ChunkCombinedNan(lines, line_type_to)
    else:
        raise ValueError(f"Invalid LineType {line_type_from}")


def convert_multi_filled(
    multi_filled: list[cpy.FillReturn],
    fill_type_from: FillType | str,
    fill_type_to:  FillType | str,
) -> list[cpy.FillReturn]:
    """Convert multiple sets of filled contours from one :class:`~.FillType` to another.

    Args:
        multi_filled (nested sequence of arrays): Filled contour polygons to convert, such as those
            returned by :meth:`.ContourGenerator.multi_filled`.
        fill_type_from (FillType or str): :class:`~.FillType` to convert from as enum or
            string equivalent.
        fill_type_to (FillType or str): :class:`~.FillType` to convert to as enum or string
            equivalent.

    Return:
        Converted sets filled contour polygons.

    When converting non-chunked fill types (``FillType.OuterCode`` or ``FillType.OuterOffset``) to
    chunked ones, all polygons are placed in the first chunk. When converting in the other
    direction, all chunk information is discarded. Converting a fill type that is not aware of the
    relationship between outer boundaries and contained holes (``FillType.ChunkCombinedCode`` or
    ``FillType.ChunkCombinedOffset``) to one that is will raise a ``ValueError``.

    .. versionadded:: 1.3.0
    """
    fill_type_from = as_fill_type(fill_type_from)
    fill_type_to = as_fill_type(fill_type_to)

    return [convert_filled(filled, fill_type_from, fill_type_to) for filled in multi_filled]


def convert_multi_lines(
    multi_lines: list[cpy.LineReturn],
    line_type_from: LineType | str,
    line_type_to:  LineType | str,
) -> list[cpy.LineReturn]:
    """Convert multiple sets of contour lines from one :class:`~.LineType` to another.

    Args:
        multi_lines (nested sequence of arrays): Contour lines to convert, such as those returned by
            :meth:`.ContourGenerator.multi_lines`.
        line_type_from (LineType or str): :class:`~.LineType` to convert from as enum or
            string equivalent.
        line_type_to (LineType or str): :class:`~.LineType` to convert to as enum or string
            equivalent.

    Return:
        Converted set of contour lines.

    When converting non-chunked line types (``LineType.Separate`` or ``LineType.SeparateCode``) to
    chunked ones (``LineType.ChunkCombinedCode``, ``LineType.ChunkCombinedOffset`` or
    ``LineType.ChunkCombinedNan``), all lines are placed in the first chunk. When converting in the
    other direction, all chunk information is discarded.

    .. versionadded:: 1.3.0
    """
    line_type_from = as_line_type(line_type_from)
    line_type_to = as_line_type(line_type_to)

    return [convert_lines(lines, line_type_from, line_type_to) for lines in multi_lines]
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from contourpy._contourpy import (
    ContourGenerator,
    FillType,
    LineType,
    Mpl2005ContourGenerator,
    Mpl2014ContourGenerator,
    SerialContourGenerator,
    ThreadedContourGenerator,
    ZInterp,
    max_threads,
)
from contourpy._version import __version__
from contourpy.chunk import calc_chunk_sizes
from contourpy.convert import (
    convert_filled,
    convert_lines,
    convert_multi_filled,
    convert_multi_lines,
)
from contourpy.dechunk import (
    dechunk_filled,
    dechunk_lines,
    dechunk_multi_filled,
    dechunk_multi_lines,
)
from contourpy.enum_util import as_fill_type, as_line_type, as_z_interp

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike

    from ._contourpy import CoordinateArray, MaskArray

__all__ = [
    "__version__",
    "contour_generator",
    "convert_filled",
    "convert_lines",
    "convert_multi_filled",
    "convert_multi_lines",
    "dechunk_filled",
    "dechunk_lines",
    "dechunk_multi_filled",
    "dechunk_multi_lines",
    "max_threads",
    "FillType",
    "LineType",
    "ContourGenerator",
    "Mpl2005ContourGenerator",
    "Mpl2014ContourGenerator",
    "SerialContourGenerator",
    "ThreadedContourGenerator",
    "ZInterp",
]


# Simple mapping of algorithm name to class name.
_class_lookup: dict[str, type[ContourGenerator]] = {
    "mpl2005": Mpl2005ContourGenerator,
    "mpl2014": Mpl2014ContourGenerator,
    "serial": SerialContourGenerator,
    "threaded": ThreadedContourGenerator,
}


def _remove_z_mask(
    z: ArrayLike | np.ma.MaskedArray[Any, Any] | None,
) -> tuple[CoordinateArray, MaskArray | None]:
    # Preserve mask if present.
    z_array = np.ma.asarray(z, dtype=np.float64)  # type: ignore[no-untyped-call]
    z_masked = np.ma.masked_invalid(z_array, copy=False)  # type: ignore[no-untyped-call]

    if np.ma.is_masked(z_masked):  # type: ignore[no-untyped-call]
        mask = np.ma.getmask(z_masked)  # type: ignore[no-untyped-call]
    else:
        mask = None

    return np.ma.getdata(z_masked), mask  # type: ignore[no-untyped-call]


def contour_generator(
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    z: ArrayLike | np.ma.MaskedArray[Any, Any] | None = None,
    *,
    name: str = "serial",
    corner_mask: bool | None = None,
    line_type: LineType | str | None = None,
    fill_type: FillType | str | None = None,
    chunk_size: int | tuple[int, int] | None = None,
    chunk_count: int | tuple[int, int] | None = None,
    total_chunk_count: int | None = None,
    quad_as_tri: bool = False,
    z_interp: ZInterp | str | None = ZInterp.Linear,
    thread_count: int = 0,
) -> ContourGenerator:
    """Create and return a :class:`~.ContourGenerator` object.

    The class and properties of the returned :class:`~.ContourGenerator` are determined by the
    function arguments, with sensible defaults.

    Args:
        x (array-like of shape (ny, nx) or (nx,), optional): The x-coordinates of the ``z`` values.
            May be 2D with the same shape as ``z.shape``, or 1D with length ``nx = z.shape[1]``.
            If not specified are assumed to be ``np.arange(nx)``. Must be ordered monotonically.
        y (array-like of shape (ny, nx) or (ny,), optional): The y-coordinates of the ``z`` values.
            May be 2D with the same shape as ``z.shape``, or 1D with length ``ny = z.shape[0]``.
            If not specified are assumed to be ``np.arange(ny)``. Must be ordered monotonically.
        z (array-like of shape (ny, nx), may be a masked array): The 2D gridded values to calculate
            the contours of.  May be a masked array, and any invalid values (``np.inf`` or
            ``np.nan``) will also be masked out.
        name (str): Algorithm name, one of ``"serial"``, ``"threaded"``, ``"mpl2005"`` or
            ``"mpl2014"``, default ``"serial"``.
        corner_mask (bool, optional): Enable/disable corner masking, which only has an effect if
            ``z`` is a masked array. If ``False``, any quad touching a masked point is masked out.
            If ``True``, only the triangular corners of quads nearest these points are always masked
            out, other triangular corners comprising three unmasked points are contoured as usual.
            If not specified, uses the default provided by the algorithm ``name``.
        line_type (LineType or str, optional): The format of contour line data returned from calls
            to :meth:`~.ContourGenerator.lines`, specified either as a :class:`~.LineType` or its
            string equivalent such as ``"SeparateCode"``.
            If not specified, uses the default provided by the algorithm ``name``.
            The relationship between the :class:`~.LineType` enum and the data format returned from
            :meth:`~.ContourGenerator.lines` is explained at :ref:`line_type`.
        fill_type (FillType or str, optional): The format of filled contour data returned from calls
            to :meth:`~.ContourGenerator.filled`, specified either as a :class:`~.FillType` or its
            string equivalent such as ``"OuterOffset"``.
            If not specified, uses the default provided by the algorithm ``name``.
            The relationship between the :class:`~.FillType` enum and the data format returned from
            :meth:`~.ContourGenerator.filled` is explained at :ref:`fill_type`.
        chunk_size (int or tuple(int, int), optional): Chunk size in (y, x) directions, or the same
            size in both directions if only one value is specified.
        chunk_count (int or tuple(int, int), optional): Chunk count in (y, x) directions, or the
            same count in both directions if only one value is specified.
        total_chunk_count (int, optional): Total number of chunks.
        quad_as_tri (bool): Enable/disable treating quads as 4 triangles, default ``False``.
            If ``False``, a contour line within a quad is a straight line between points on two of
            its edges. If ``True``, each full quad is divided into 4 triangles using a virtual point
            at the centre (mean x, y of the corner points) and a contour line is piecewise linear
            within those triangles. Corner-masked triangles are not affected by this setting, only
            full unmasked quads.
        z_interp (ZInterp or str, optional): How to interpolate ``z`` values when determining where
            contour lines intersect the edges of quads and the ``z`` values of the central points of
            quads, specified either as a :class:`~contourpy.ZInterp` or its string equivalent such
            as ``"Log"``. Default is ``ZInterp.Linear``.
        thread_count (int): Number of threads to use for contour calculation, default 0. Threads can
            only be used with an algorithm ``name`` that supports threads (currently only
            ``name="threaded"``) and there must be at least the same number of chunks as threads.
            If ``thread_count=0`` and ``name="threaded"`` then it uses the maximum number of threads
            as determined by the C++11 call ``std::thread::hardware_concurrency()``. If ``name`` is
            something other than ``"threaded"`` then the ``thread_count`` will be set to ``1``.

    Return:
        :class:`~.ContourGenerator`.

    Note:
        A maximum of one of ``chunk_size``, ``chunk_count`` and ``total_chunk_count`` may be
        specified.

    Warning:
        The ``name="mpl2005"`` algorithm does not implement chunking for contour lines.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z, mask = _remove_z_mask(z)

    # Check arguments: z.
    if z.ndim != 2:
        raise TypeError(f"Input z must be 2D, not {z.ndim}D")

    if z.shape[0] < 2 or z.shape[1] < 2:
        raise TypeError(f"Input z must be at least a (2, 2) shaped array, but has shape {z.shape}")

    ny, nx = z.shape

    # Check arguments: x and y.
    if x.ndim != y.ndim:
        raise TypeError(f"Number of dimensions of x ({x.ndim}) and y ({y.ndim}) do not match")

    if x.ndim == 0:
        x = np.arange(nx, dtype=np.float64)
        y = np.arange(ny, dtype=np.float64)
        x, y = np.meshgrid(x, y)
    elif x.ndim == 1:
        if len(x) != nx:
            raise TypeError(f"Length of x ({len(x)}) must match number of columns in z ({nx})")
        if len(y) != ny:
            raise TypeError(f"Length of y ({len(y)}) must match number of rows in z ({ny})")
        x, y = np.meshgrid(x, y)
    elif x.ndim == 2:
        if x.shape != z.shape:
            raise TypeError(f"Shapes of x {x.shape} and z {z.shape} do not match")
        if y.shape != z.shape:
            raise TypeError(f"Shapes of y {y.shape} and z {z.shape} do not match")
    else:
        raise TypeError(f"Inputs x and y must be None, 1D or 2D, not {x.ndim}D")

    # Check mask shape just in case.
    if mask is not None and mask.shape != z.shape:
        raise ValueError("If mask is set it must be a 2D array with the same shape as z")

    # Check arguments: name.
    if name not in _class_lookup:
        raise ValueError(f"Unrecognised contour generator name: {name}")

    # Check arguments: chunk_size, chunk_count and total_chunk_count.
    y_chunk_size, x_chunk_size = calc_chunk_sizes(
        chunk_size, chunk_count, total_chunk_count, ny, nx)

    cls = _class_lookup[name]

    # Check arguments: corner_mask.
    if corner_mask is None:
        # Set it to default, which is True if the algorithm supports it.
        corner_mask = cls.supports_corner_mask()
    elif corner_mask and not cls.supports_corner_mask():
        raise ValueError(f"{name} contour generator does not support corner_mask=True")

    # Check arguments: line_type.
    if line_type is None:
        line_type = cls.default_line_type
    else:
        line_type = as_line_type(line_type)

    if not cls.supports_line_type(line_type):
        raise ValueError(f"{name} contour generator does not support line_type {line_type}")

    # Check arguments: fill_type.
    if fill_type is None:
        fill_type = cls.default_fill_type
    else:
        fill_type = as_fill_type(fill_type)

    if not cls.supports_fill_type(fill_type):
        raise ValueError(f"{name} contour generator does not support fill_type {fill_type}")

    # Check arguments: quad_as_tri.
    if quad_as_tri and not cls.supports_quad_as_tri():
        raise ValueError(f"{name} contour generator does not support quad_as_tri=True")

    # Check arguments: z_interp.
    if z_interp is None:
        z_interp = ZInterp.Linear
    else:
        z_interp = as_z_interp(z_interp)

    if z_interp != ZInterp.Linear and not cls.supports_z_interp():
        raise ValueError(f"{name} contour generator does not support z_interp {z_interp}")

    # Check arguments: thread_count.
    if thread_count not in (0, 1) and not cls.supports_threads():
        raise ValueError(f"{name} contour generator does not support thread_count {thread_count}")

    # Prepare args and kwargs for contour generator constructor.
    args = [x, y, z, mask]
    kwargs: dict[str, int | bool | LineType | FillType | ZInterp] = {
        "x_chunk_size": x_chunk_size,
        "y_chunk_size": y_chunk_size,
    }

    if name not in ("mpl2005", "mpl2014"):
        kwargs["line_type"] = line_type
        kwargs["fill_type"] = fill_type

    if cls.supports_corner_mask():
        kwargs["corner_mask"] = corner_mask

    if cls.supports_quad_as_tri():
        kwargs["quad_as_tri"] = quad_as_tri

    if cls.supports_z_interp():
        kwargs["z_interp"] = z_interp

    if cls.supports_threads():
        kwargs["thread_count"] = thread_count

    # Create contour generator.
    return cls(*args, **kwargs)
<|endoftext|>from __future__ import annotations

from contourpy._contourpy import FillType, LineType, ZInterp


def as_fill_type(fill_type: FillType | str) -> FillType:
    """Coerce a FillType or string value to a FillType.

    Args:
        fill_type (FillType or str): Value to convert.

    Return:
        FillType: Converted value.
    """
    if isinstance(fill_type, str):
        try:
            return FillType.__members__[fill_type]
        except KeyError as e:
            raise ValueError(f"'{fill_type}' is not a valid FillType") from e
    else:
        return fill_type


def as_line_type(line_type: LineType | str) -> LineType:
    """Coerce a LineType or string value to a LineType.

    Args:
        line_type (LineType or str): Value to convert.

    Return:
        LineType: Converted value.
    """
    if isinstance(line_type, str):
        try:
            return LineType.__members__[line_type]
        except KeyError as e:
            raise ValueError(f"'{line_type}' is not a valid LineType") from e
    else:
        return line_type


def as_z_interp(z_interp: ZInterp | str) -> ZInterp:
    """Coerce a ZInterp or string value to a ZInterp.

    Args:
        z_interp (ZInterp or str): Value to convert.

    Return:
        ZInterp: Converted value.
    """
    if isinstance(z_interp, str):
        try:
            return ZInterp.__members__[z_interp]
        except KeyError as e:
            raise ValueError(f"'{z_interp}' is not a valid ZInterp") from e
    else:
        return z_interp
<|endoftext|>from __future__ import annotations

import math


def calc_chunk_sizes(
    chunk_size: int | tuple[int, int] | None,
    chunk_count: int | tuple[int, int] | None,
    total_chunk_count: int | None,
    ny: int,
    nx: int,
) -> tuple[int, int]:
    """Calculate chunk sizes.

    Args:
        chunk_size (int or tuple(int, int), optional): Chunk size in (y, x) directions, or the same
            size in both directions if only one is specified. Cannot be negative.
        chunk_count (int or tuple(int, int), optional): Chunk count in (y, x) directions, or the
            same count in both directions if only one is specified. If less than 1, set to 1.
        total_chunk_count (int, optional): Total number of chunks. If less than 1, set to 1.
        ny (int): Number of grid points in y-direction.
        nx (int): Number of grid points in x-direction.

    Return:
        tuple(int, int): Chunk sizes (y_chunk_size, x_chunk_size).

    Note:
        Zero or one of ``chunk_size``, ``chunk_count`` and ``total_chunk_count`` should be
        specified.
    """
    if sum([chunk_size is not None, chunk_count is not None, total_chunk_count is not None]) > 1:
        raise ValueError("Only one of chunk_size, chunk_count and total_chunk_count should be set")

    if nx < 2 or ny < 2:
        raise ValueError(f"(ny, nx) must be at least (2, 2), not ({ny}, {nx})")

    if total_chunk_count is not None:
        max_chunk_count = (nx-1)*(ny-1)
        total_chunk_count = min(max(total_chunk_count, 1), max_chunk_count)
        if total_chunk_count == 1:
            chunk_size = 0
        elif total_chunk_count == max_chunk_count:
            chunk_size = (1, 1)
        else:
            factors = two_factors(total_chunk_count)
            if ny > nx:
                chunk_count = factors
            else:
                chunk_count = (factors[1], factors[0])

    if chunk_count is not None:
        if isinstance(chunk_count, tuple):
            y_chunk_count, x_chunk_count = chunk_count
        else:
            y_chunk_count = x_chunk_count = chunk_count
        x_chunk_count = min(max(x_chunk_count, 1), nx-1)
        y_chunk_count = min(max(y_chunk_count, 1), ny-1)
        chunk_size = (math.ceil((ny-1) / y_chunk_count), math.ceil((nx-1) / x_chunk_count))

    if chunk_size is None:
        y_chunk_size = x_chunk_size = 0
    elif isinstance(chunk_size, tuple):
        y_chunk_size, x_chunk_size = chunk_size
    else:
        y_chunk_size = x_chunk_size = chunk_size

    if x_chunk_size < 0 or y_chunk_size < 0:
        raise ValueError("chunk_size cannot be negative")

    return y_chunk_size, x_chunk_size


def two_factors(n: int) -> tuple[int, int]:
    """Split an integer into two integer factors.

    The two factors will be as close as possible to the sqrt of n, and are returned in decreasing
    order.  Worst case returns (n, 1).

    Args:
        n (int): The integer to factorize, must be positive.

    Return:
        tuple(int, int): The two factors of n, in decreasing order.
    """
    if n < 0:
        raise ValueError(f"two_factors expects positive integer not {n}")

    i = math.ceil(math.sqrt(n))
    while n % i != 0:
        i -= 1
    j = n // i
    if i > j:
        return i, j
    else:
        return j, i
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, cast

from contourpy._contourpy import FillType, LineType
from contourpy.array import (
    concat_codes_or_none,
    concat_offsets_or_none,
    concat_points_or_none,
    concat_points_or_none_with_nan,
)
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines

if TYPE_CHECKING:
    import contourpy._contourpy as cpy


def dechunk_filled(filled: cpy.FillReturn, fill_type: FillType | str) -> cpy.FillReturn:
    """Return the specified filled contours with chunked data moved into the first chunk.

    Filled contours that are not chunked (``FillType.OuterCode`` and ``FillType.OuterOffset``) and
    those that are but only contain a single chunk are returned unmodified. Individual polygons are
    unchanged, they are not geometrically combined.

    Args:
        filled (sequence of arrays): Filled contour data, such as returned by
            :meth:`.ContourGenerator.filled`.
        fill_type (FillType or str): Type of :meth:`~.ContourGenerator.filled` as enum or string
            equivalent.

    Return:
        Filled contours in a single chunk.

    .. versionadded:: 1.2.0
    """
    fill_type = as_fill_type(fill_type)

    if fill_type in (FillType.OuterCode, FillType.OuterOffset):
        # No-op if fill_type is not chunked.
        return filled

    check_filled(filled, fill_type)
    if len(filled[0]) < 2:
        # No-op if just one chunk.
        return filled

    if TYPE_CHECKING:
        filled = cast(cpy.FillReturn_Chunk, filled)
    points = concat_points_or_none(filled[0])

    if fill_type == FillType.ChunkCombinedCode:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCode, filled)
        if points is None:
            ret1: cpy.FillReturn_ChunkCombinedCode = ([None], [None])
        else:
            ret1 = ([points], [concat_codes_or_none(filled[1])])
        return ret1
    elif fill_type == FillType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffset, filled)
        if points is None:
            ret2: cpy.FillReturn_ChunkCombinedOffset = ([None], [None])
        else:
            ret2 = ([points], [concat_offsets_or_none(filled[1])])
        return ret2
    elif fill_type == FillType.ChunkCombinedCodeOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedCodeOffset, filled)
        if points is None:
            ret3: cpy.FillReturn_ChunkCombinedCodeOffset = ([None], [None], [None])
        else:
            outer_offsets = concat_offsets_or_none(filled[2])
            ret3 = ([points], [concat_codes_or_none(filled[1])], [outer_offsets])
        return ret3
    elif fill_type == FillType.ChunkCombinedOffsetOffset:
        if TYPE_CHECKING:
            filled = cast(cpy.FillReturn_ChunkCombinedOffsetOffset, filled)
        if points is None:
            ret4: cpy.FillReturn_ChunkCombinedOffsetOffset = ([None], [None], [None])
        else:
            outer_offsets = concat_offsets_or_none(filled[2])
            ret4 = ([points], [concat_offsets_or_none(filled[1])], [outer_offsets])
        return ret4
    else:
        raise ValueError(f"Invalid FillType {fill_type}")


def dechunk_lines(lines: cpy.LineReturn, line_type: LineType | str) -> cpy.LineReturn:
    """Return the specified contour lines with chunked data moved into the first chunk.

    Contour lines that are not chunked (``LineType.Separate`` and ``LineType.SeparateCode``) and
    those that are but only contain a single chunk are returned unmodified. Individual lines are
    unchanged, they are not geometrically combined.

    Args:
        lines (sequence of arrays): Contour line data, such as returned by
            :meth:`.ContourGenerator.lines`.
        line_type (LineType or str): Type of :meth:`~.ContourGenerator.lines` as enum or string
            equivalent.

    Return:
        Contour lines in a single chunk.

    .. versionadded:: 1.2.0
    """
    line_type = as_line_type(line_type)

    if line_type in (LineType.Separate, LineType.SeparateCode):
        # No-op if line_type is not chunked.
        return lines

    check_lines(lines, line_type)
    if len(lines[0]) < 2:
        # No-op if just one chunk.
        return lines

    if TYPE_CHECKING:
        lines = cast(cpy.LineReturn_Chunk, lines)

    if line_type == LineType.ChunkCombinedCode:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedCode, lines)
        points = concat_points_or_none(lines[0])
        if points is None:
            ret1: cpy.LineReturn_ChunkCombinedCode = ([None], [None])
        else:
            ret1 = ([points], [concat_codes_or_none(lines[1])])
        return ret1
    elif line_type == LineType.ChunkCombinedOffset:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedOffset, lines)
        points = concat_points_or_none(lines[0])
        if points is None:
            ret2: cpy.LineReturn_ChunkCombinedOffset = ([None], [None])
        else:
            ret2 = ([points], [concat_offsets_or_none(lines[1])])
        return ret2
    elif line_type == LineType.ChunkCombinedNan:
        if TYPE_CHECKING:
            lines = cast(cpy.LineReturn_ChunkCombinedNan, lines)
        points = concat_points_or_none_with_nan(lines[0])
        ret3: cpy.LineReturn_ChunkCombinedNan = ([points],)
        return ret3
    else:
        raise ValueError(f"Invalid LineType {line_type}")


def dechunk_multi_filled(
    multi_filled: list[cpy.FillReturn],
    fill_type: FillType | str,
) -> list[cpy.FillReturn]:
    """Return multiple sets of filled contours with chunked data moved into the first chunks.

    Filled contours that are not chunked (``FillType.OuterCode`` and ``FillType.OuterOffset``) and
    those that are but only contain a single chunk are returned unmodified. Individual polygons are
    unchanged, they are not geometrically combined.

    Args:
        multi_filled (nested sequence of arrays): Filled contour data, such as returned by
            :meth:`.ContourGenerator.multi_filled`.
        fill_type (FillType or str): Type of :meth:`~.ContourGenerator.filled` as enum or string
            equivalent.

    Return:
        Multiple sets of filled contours in a single chunk.

    .. versionadded:: 1.3.0
    """
    fill_type = as_fill_type(fill_type)

    if fill_type in (FillType.OuterCode, FillType.OuterOffset):
        # No-op if fill_type is not chunked.
        return multi_filled

    return [dechunk_filled(filled, fill_type) for filled in multi_filled]


def dechunk_multi_lines(
    multi_lines: list[cpy.LineReturn],
    line_type: LineType | str,
) -> list[cpy.LineReturn]:
    """Return multiple sets of contour lines with all chunked data moved into the first chunks.

    Contour lines that are not chunked (``LineType.Separate`` and ``LineType.SeparateCode``) and
    those that are but only contain a single chunk are returned unmodified. Individual lines are
    unchanged, they are not geometrically combined.

    Args:
        multi_lines (nested sequence of arrays): Contour line data, such as returned by
            :meth:`.ContourGenerator.multi_lines`.
        line_type (LineType or str): Type of :meth:`~.ContourGenerator.lines` as enum or string
            equivalent.

    Return:
        Multiple sets of contour lines in a single chunk.

    .. versionadded:: 1.3.0
    """
    line_type = as_line_type(line_type)

    if line_type in (LineType.Separate, LineType.SeparateCode):
        # No-op if line_type is not chunked.
        return multi_lines

    return [dechunk_lines(lines, line_type) for lines in multi_lines]
<|endoftext|>from __future__ import annotations

from itertools import chain, pairwise
from typing import TYPE_CHECKING

import numpy as np

from contourpy.typecheck import check_code_array, check_offset_array, check_point_array
from contourpy.types import CLOSEPOLY, LINETO, MOVETO, code_dtype, offset_dtype, point_dtype

if TYPE_CHECKING:
    import contourpy._contourpy as cpy


def codes_from_offsets(offsets: cpy.OffsetArray) -> cpy.CodeArray:
    """Determine codes from offsets, assuming they all correspond to closed polygons.
    """
    check_offset_array(offsets)

    n = offsets[-1]
    codes = np.full(n, LINETO, dtype=code_dtype)
    codes[offsets[:-1]] = MOVETO
    codes[offsets[1:] - 1] = CLOSEPOLY
    return codes


def codes_from_offsets_and_points(
    offsets: cpy.OffsetArray,
    points: cpy.PointArray,
) -> cpy.CodeArray:
    """Determine codes from offsets and points, using the equality of the start and end points of
    each line to determine if lines are closed or not.
    """
    check_offset_array(offsets)
    check_point_array(points)

    codes = np.full(len(points), LINETO, dtype=code_dtype)
    codes[offsets[:-1]] = MOVETO

    end_offsets = offsets[1:] - 1
    closed = np.all(points[offsets[:-1]] == points[end_offsets], axis=1)
    codes[end_offsets[closed]] = CLOSEPOLY

    return codes


def codes_from_points(points: cpy.PointArray) -> cpy.CodeArray:
    """Determine codes for a single line, using the equality of the start and end points to
    determine if the line is closed or not.
    """
    check_point_array(points)

    n = len(points)
    codes = np.full(n, LINETO, dtype=code_dtype)
    codes[0] = MOVETO
    if np.all(points[0] == points[-1]):
        codes[-1] = CLOSEPOLY
    return codes


def concat_codes(list_of_codes: list[cpy.CodeArray]) -> cpy.CodeArray:
    """Concatenate a list of codes arrays into a single code array.
    """
    if not list_of_codes:
        raise ValueError("Empty list passed to concat_codes")

    return np.concatenate(list_of_codes, dtype=code_dtype)


def concat_codes_or_none(list_of_codes_or_none: list[cpy.CodeArray | None]) -> cpy.CodeArray | None:
    """Concatenate a list of codes arrays or None into a single code array or None.
    """
    list_of_codes = [codes for codes in list_of_codes_or_none if codes is not None]
    if list_of_codes:
        return concat_codes(list_of_codes)
    else:
        return None


def concat_offsets(list_of_offsets: list[cpy.OffsetArray]) -> cpy.OffsetArray:
    """Concatenate a list of offsets arrays into a single offset array.
    """
    if not list_of_offsets:
        raise ValueError("Empty list passed to concat_offsets")

    n = len(list_of_offsets)
    cumulative = np.cumsum([offsets[-1] for offsets in list_of_offsets], dtype=offset_dtype)
    ret: cpy.OffsetArray = np.concatenate(
        (list_of_offsets[0], *(list_of_offsets[i+1][1:] + cumulative[i] for i in range(n-1))),
        dtype=offset_dtype,
    )
    return ret


def concat_offsets_or_none(
    list_of_offsets_or_none: list[cpy.OffsetArray | None],
) -> cpy.OffsetArray | None:
    """Concatenate a list of offsets arrays or None into a single offset array or None.
    """
    list_of_offsets = [offsets for offsets in list_of_offsets_or_none if offsets is not None]
    if list_of_offsets:
        return concat_offsets(list_of_offsets)
    else:
        return None


def concat_points(list_of_points: list[cpy.PointArray]) -> cpy.PointArray:
    """Concatenate a list of point arrays into a single point array.
    """
    if not list_of_points:
        raise ValueError("Empty list passed to concat_points")

    return np.concatenate(list_of_points, dtype=point_dtype)


def concat_points_or_none(
    list_of_points_or_none: list[cpy.PointArray | None],
) -> cpy.PointArray | None:
    """Concatenate a list of point arrays or None into a single point array or None.
    """
    list_of_points = [points for points in list_of_points_or_none if points is not None]
    if list_of_points:
        return concat_points(list_of_points)
    else:
        return None


def concat_points_or_none_with_nan(
    list_of_points_or_none: list[cpy.PointArray | None],
) -> cpy.PointArray | None:
    """Concatenate a list of points or None into a single point array or None, with NaNs used to
    separate each line.
    """
    list_of_points = [points for points in list_of_points_or_none if points is not None]
    if list_of_points:
        return concat_points_with_nan(list_of_points)
    else:
        return None


def concat_points_with_nan(list_of_points: list[cpy.PointArray]) -> cpy.PointArray:
    """Concatenate a list of points into a single point array with NaNs used to separate each line.
    """
    if not list_of_points:
        raise ValueError("Empty list passed to concat_points_with_nan")

    if len(list_of_points) == 1:
        return list_of_points[0]
    else:
        nan_spacer = np.full((1, 2), np.nan, dtype=point_dtype)
        list_of_points = [list_of_points[0],
                          *list(chain(*((nan_spacer, x) for x in list_of_points[1:])))]
        return concat_points(list_of_points)


def insert_nan_at_offsets(points: cpy.PointArray, offsets: cpy.OffsetArray) -> cpy.PointArray:
    """Insert NaNs into a point array at locations specified by an offset array.
    """
    check_point_array(points)
    check_offset_array(offsets)

    if len(offsets) <= 2:
        return points
    else:
        nan_spacer = np.array([np.nan, np.nan], dtype=point_dtype)
        # Convert offsets to int64 to avoid numpy error when mixing signed and unsigned ints.
        return np.insert(points, offsets[1:-1].astype(np.int64), nan_spacer, axis=0)


def offsets_from_codes(codes: cpy.CodeArray) -> cpy.OffsetArray:
    """Determine offsets from codes using locations of MOVETO codes.
    """
    check_code_array(codes)

    return np.append(np.nonzero(codes == MOVETO)[0], len(codes)).astype(offset_dtype)


def offsets_from_lengths(list_of_points: list[cpy.PointArray]) -> cpy.OffsetArray:
    """Determine offsets from lengths of point arrays.
    """
    if not list_of_points:
        raise ValueError("Empty list passed to offsets_from_lengths")

    return np.cumsum([0] + [len(line) for line in list_of_points], dtype=offset_dtype)


def outer_offsets_from_list_of_codes(list_of_codes: list[cpy.CodeArray]) -> cpy.OffsetArray:
    """Determine outer offsets from codes using locations of MOVETO codes.
    """
    if not list_of_codes:
        raise ValueError("Empty list passed to outer_offsets_from_list_of_codes")

    return np.cumsum([0] + [np.count_nonzero(codes == MOVETO) for codes in list_of_codes],
                     dtype=offset_dtype)


def outer_offsets_from_list_of_offsets(list_of_offsets: list[cpy.OffsetArray]) -> cpy.OffsetArray:
    """Determine outer offsets from a list of offsets.
    """
    if not list_of_offsets:
        raise ValueError("Empty list passed to outer_offsets_from_list_of_offsets")

    return np.cumsum([0] + [len(offsets)-1 for offsets in list_of_offsets], dtype=offset_dtype)


def remove_nan(points: cpy.PointArray) -> tuple[cpy.PointArray, cpy.OffsetArray]:
    """Remove NaN from a points array, also return the offsets corresponding to the NaN removed.
    """
    check_point_array(points)

    nan_offsets = np.nonzero(np.isnan(points[:, 0]))[0]
    if len(nan_offsets) == 0:
        return points, np.array([0, len(points)], dtype=offset_dtype)
    else:
        points = np.delete(points, nan_offsets, axis=0)
        nan_offsets -= np.arange(len(nan_offsets))
        offsets: cpy.OffsetArray = np.empty(len(nan_offsets)+2, dtype=offset_dtype)
        offsets[0] = 0
        offsets[1:-1] = nan_offsets
        offsets[-1] = len(points)
        return points, offsets


def split_codes_by_offsets(codes: cpy.CodeArray, offsets: cpy.OffsetArray) -> list[cpy.CodeArray]:
    """Split a code array at locations specified by an offset array into a list of code arrays.
    """
    check_code_array(codes)
    check_offset_array(offsets)

    if len(offsets) > 2:
        return np.split(codes, offsets[1:-1])
    else:
        return [codes]


def split_points_by_offsets(
    points: cpy.PointArray,
    offsets: cpy.OffsetArray,
) -> list[cpy.PointArray]:
    """Split a point array at locations specified by an offset array into a list of point arrays.
    """
    check_point_array(points)
    check_offset_array(offsets)

    if len(offsets) > 2:
        return np.split(points, offsets[1:-1])
    else:
        return [points]


def split_points_at_nan(points: cpy.PointArray) -> list[cpy.PointArray]:
    """Split a points array at NaNs into a list of point arrays.
    """
    check_point_array(points)

    nan_offsets = np.nonzero(np.isnan(points[:, 0]))[0]
    if len(nan_offsets) == 0:
        return [points]
    else:
        nan_offsets = np.concatenate(([-1], nan_offsets, [len(points)]))
        return [points[s+1:e] for s, e in pairwise(nan_offsets)]
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, cast

from contourpy import FillType, LineType
from contourpy.array import offsets_from_codes
from contourpy.convert import convert_lines
from contourpy.dechunk import dechunk_lines

if TYPE_CHECKING:
    from contourpy._contourpy import (
        CoordinateArray,
        FillReturn,
        LineReturn,
        LineReturn_ChunkCombinedNan,
    )


def filled_to_bokeh(
    filled: FillReturn,
    fill_type: FillType,
) -> tuple[list[list[CoordinateArray]], list[list[CoordinateArray]]]:
    xs: list[list[CoordinateArray]] = []
    ys: list[list[CoordinateArray]] = []
    if fill_type in (FillType.OuterOffset, FillType.ChunkCombinedOffset,
                     FillType.OuterCode, FillType.ChunkCombinedCode):
        have_codes = fill_type in (FillType.OuterCode, FillType.ChunkCombinedCode)

        for points, offsets in zip(*filled):
            if points is None:
                continue
            if have_codes:
                offsets = offsets_from_codes(offsets)
            xs.append([])  # New outer with zero or more holes.
            ys.append([])
            for i in range(len(offsets)-1):
                xys = points[offsets[i]:offsets[i+1]]
                xs[-1].append(xys[:, 0])
                ys[-1].append(xys[:, 1])
    elif fill_type in (FillType.ChunkCombinedCodeOffset, FillType.ChunkCombinedOffsetOffset):
        for points, codes_or_offsets, outer_offsets in zip(*filled):
            if points is None:
                continue
            for j in range(len(outer_offsets)-1):
                if fill_type == FillType.ChunkCombinedCodeOffset:
                    codes = codes_or_offsets[outer_offsets[j]:outer_offsets[j+1]]
                    offsets = offsets_from_codes(codes) + outer_offsets[j]
                else:
                    offsets = codes_or_offsets[outer_offsets[j]:outer_offsets[j+1]+1]
                xs.append([])  # New outer with zero or more holes.
                ys.append([])
                for k in range(len(offsets)-1):
                    xys = points[offsets[k]:offsets[k+1]]
                    xs[-1].append(xys[:, 0])
                    ys[-1].append(xys[:, 1])
    else:
        raise RuntimeError(f"Conversion of FillType {fill_type} to Bokeh is not implemented")

    return xs, ys


def lines_to_bokeh(
    lines: LineReturn,
    line_type: LineType,
) -> tuple[CoordinateArray | None, CoordinateArray | None]:
    lines = convert_lines(lines, line_type, LineType.ChunkCombinedNan)
    lines = dechunk_lines(lines, LineType.ChunkCombinedNan)
    if TYPE_CHECKING:
        lines = cast(LineReturn_ChunkCombinedNan, lines)
    points = lines[0][0]
    if points is None:
        return None, None
    else:
        return points[:, 0], points[:, 1]
<|endoftext|># _build_config.py.in is converted into _build_config.py during the meson build process.

from __future__ import annotations


def build_config() -> dict[str, str]:
    """
    Return a dictionary containing build configuration settings.

    All dictionary keys and values are strings, for example ``False`` is
    returned as ``"False"``.

        .. versionadded:: 1.1.0
    """
    return dict(
        # Python settings
        python_version="3.10",
        python_install_dir=r"/usr/local/lib/python3.10/site-packages/",
        python_path=r"/tmp/build-env-dfa9bu3d/bin/python",

        # Package versions
        contourpy_version="1.3.2",
        meson_version="1.7.2",
        mesonpy_version="0.17.1",
        pybind11_version="2.13.6",

        # Misc meson settings
        meson_backend="ninja",
        build_dir=r"/project/.mesonpy-0roiogjz/lib/contourpy/util",
        source_dir=r"/project/lib/contourpy/util",
        cross_build="False",

        # Build options
        build_options=r"-Dbuildtype=release -Db_ndebug=if-release -Db_vscrt=md -Dvsenv=True --native-file=/project/.mesonpy-0roiogjz/meson-python-native-file.ini",
        buildtype="release",
        cpp_std="c++17",
        debug="False",
        optimization="3",
        vsenv="True",
        b_ndebug="if-release",
        b_vscrt="from_buildtype",

        # C++ compiler
        compiler_name="gcc",
        compiler_version="10.2.1",
        linker_id="ld.bfd",
        compile_command="c++",

        # Host machine
        host_cpu="x86_64",
        host_cpu_family="x86_64",
        host_cpu_endian="little",
        host_cpu_system="linux",

        # Build machine, same as host machine if not a cross_build
        build_cpu="x86_64",
        build_cpu_family="x86_64",
        build_cpu_endian="little",
        build_cpu_system="linux",
    )
<|endoftext|>from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, cast

import matplotlib.path as mpath
import numpy as np

from contourpy import FillType, LineType
from contourpy.array import codes_from_offsets

if TYPE_CHECKING:
    from contourpy._contourpy import FillReturn, LineReturn, LineReturn_Separate


def filled_to_mpl_paths(filled: FillReturn, fill_type: FillType) -> list[mpath.Path]:
    if fill_type in (FillType.OuterCode, FillType.ChunkCombinedCode):
        paths = [mpath.Path(points, codes) for points, codes in zip(*filled) if points is not None]
    elif fill_type in (FillType.OuterOffset, FillType.ChunkCombinedOffset):
        paths = [mpath.Path(points, codes_from_offsets(offsets))
                 for points, offsets in zip(*filled) if points is not None]
    elif fill_type == FillType.ChunkCombinedCodeOffset:
        paths = []
        for points, codes, outer_offsets in zip(*filled):
            if points is None:
                continue
            points = np.split(points, outer_offsets[1:-1])
            codes = np.split(codes, outer_offsets[1:-1])
            paths += [mpath.Path(p, c) for p, c in zip(points, codes)]
    elif fill_type == FillType.ChunkCombinedOffsetOffset:
        paths = []
        for points, offsets, outer_offsets in zip(*filled):
            if points is None:
                continue
            for i in range(len(outer_offsets)-1):
                offs = offsets[outer_offsets[i]:outer_offsets[i+1]+1]
                pts = points[offs[0]:offs[-1]]
                paths += [mpath.Path(pts, codes_from_offsets(offs - offs[0]))]
    else:
        raise RuntimeError(f"Conversion of FillType {fill_type} to MPL Paths is not implemented")
    return paths


def lines_to_mpl_paths(lines: LineReturn, line_type: LineType) -> list[mpath.Path]:
    if line_type == LineType.Separate:
        if TYPE_CHECKING:
            lines = cast(LineReturn_Separate, lines)
        paths = []
        for line in lines:
            # Drawing as Paths so that they can be closed correctly.
            closed = line[0, 0] == line[-1, 0] and line[0, 1] == line[-1, 1]
            paths.append(mpath.Path(line, closed=closed))
    elif line_type in (LineType.SeparateCode, LineType.ChunkCombinedCode):
        paths = [mpath.Path(points, codes) for points, codes in zip(*lines) if points is not None]
    elif line_type == LineType.ChunkCombinedOffset:
        paths = []
        for points, offsets in zip(*lines):
            if points is None:
                continue
            for i in range(len(offsets)-1):
                line = points[offsets[i]:offsets[i+1]]
                closed = line[0, 0] == line[-1, 0] and line[0, 1] == line[-1, 1]
                paths.append(mpath.Path(line, closed=closed))
    elif line_type == LineType.ChunkCombinedNan:
        paths = []
        for points in lines[0]:
            if points is None:
                continue
            nan_offsets = np.nonzero(np.isnan(points[:, 0]))[0]
            nan_offsets = np.concatenate([[-1], nan_offsets, [len(points)]])
            for s, e in pairwise(nan_offsets):
                line = points[s+1:e]
                closed = line[0, 0] == line[-1, 0] and line[0, 1] == line[-1, 1]
                paths.append(mpath.Path(line, closed=closed))
    else:
        raise RuntimeError(f"Conversion of LineType {line_type} to MPL Paths is not implemented")
    return paths
<|endoftext|>from __future__ import annotations

import io
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np

from contourpy import FillType, LineType
from contourpy.convert import convert_filled, convert_lines
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.mpl_util import filled_to_mpl_paths, lines_to_mpl_paths
from contourpy.util.renderer import Renderer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    import contourpy._contourpy as cpy


class MplRenderer(Renderer):
    """Utility renderer using Matplotlib to render a grid of plots over the same (x, y) range.

    Args:
        nrows (int, optional): Number of rows of plots, default ``1``.
        ncols (int, optional): Number of columns of plots, default ``1``.
        figsize (tuple(float, float), optional): Figure size in inches, default ``(9, 9)``.
        show_frame (bool, optional): Whether to show frame and axes ticks, default ``True``.
        backend (str, optional): Matplotlib backend to use or ``None`` for default backend.
            Default ``None``.
        gridspec_kw (dict, optional): Gridspec keyword arguments to pass to ``plt.subplots``,
            default None.
    """
    _axes: Sequence[Axes]
    _fig: Figure
    _want_tight: bool

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] = (9, 9),
        show_frame: bool = True,
        backend: str | None = None,
        gridspec_kw: dict[str, Any] | None = None,
    ) -> None:
        if backend is not None:
            import matplotlib as mpl
            mpl.use(backend)

        kwargs: dict[str, Any] = {"figsize": figsize, "squeeze": False,
                                  "sharex": True, "sharey": True}
        if gridspec_kw is not None:
            kwargs["gridspec_kw"] = gridspec_kw
        else:
            kwargs["subplot_kw"] = {"aspect": "equal"}

        self._fig, axes = plt.subplots(nrows, ncols, **kwargs)
        self._axes = axes.flatten()
        if not show_frame:
            for ax in self._axes:
                ax.axis("off")

        self._want_tight = True

    def __del__(self) -> None:
        if hasattr(self, "_fig"):
            plt.close(self._fig)

    def _autoscale(self) -> None:
        # Using axes._need_autoscale attribute if need to autoscale before rendering after adding
        # lines/filled.  Only want to autoscale once per axes regardless of how many lines/filled
        # added.
        for ax in self._axes:
            if getattr(ax, "_need_autoscale", False):
                ax.autoscale_view(tight=True)
                ax._need_autoscale = False  # type: ignore[attr-defined]
        if self._want_tight and len(self._axes) > 1:
            self._fig.tight_layout()

    def _get_ax(self, ax: Axes | int) -> Axes:
        if isinstance(ax, int):
            ax = self._axes[ax]
        return ax

    def filled(
        self,
        filled: cpy.FillReturn,
        fill_type: FillType | str,
        ax: Axes | int = 0,
        color: str = "C0",
        alpha: float = 0.7,
    ) -> None:
        """Plot filled contours on a single Axes.

        Args:
            filled (sequence of arrays): Filled contour data as returned by
                :meth:`~.ContourGenerator.filled`.
            fill_type (FillType or str): Type of :meth:`~.ContourGenerator.filled` data as returned
                by :attr:`~.ContourGenerator.fill_type`, or string equivalent
            ax (int or Maplotlib Axes, optional): Which axes to plot on, default ``0``.
            color (str, optional): Color to plot with. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``tab10`` colormap. Default ``"C0"``.
            alpha (float, optional): Opacity to plot with, default ``0.7``.
        """
        fill_type = as_fill_type(fill_type)
        ax = self._get_ax(ax)
        paths = filled_to_mpl_paths(filled, fill_type)
        collection = mcollections.PathCollection(
            paths, facecolors=color, edgecolors="none", lw=0, alpha=alpha)
        ax.add_collection(collection)
        ax._need_autoscale = True  # type: ignore[attr-defined]

    def grid(
        self,
        x: ArrayLike,
        y: ArrayLike,
        ax: Axes | int = 0,
        color: str = "black",
        alpha: float = 0.1,
        point_color: str | None = None,
        quad_as_tri_alpha: float = 0,
    ) -> None:
        """Plot quad grid lines on a single Axes.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
            color (str, optional): Color to plot grid lines, default ``"black"``.
            alpha (float, optional): Opacity to plot lines with, default ``0.1``.
            point_color (str, optional): Color to plot grid points or ``None`` if grid points
                should not be plotted, default ``None``.
            quad_as_tri_alpha (float, optional): Opacity to plot ``quad_as_tri`` grid, default 0.

        Colors may be a string color or the letter ``"C"`` followed by an integer in the range
        ``"C0"`` to ``"C9"`` to use a color from the ``tab10`` colormap.

        Warning:
            ``quad_as_tri_alpha > 0`` plots all quads as though they are unmasked.
        """
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        kwargs: dict[str, Any] = {"color": color, "alpha": alpha}
        ax.plot(x, y, x.T, y.T, **kwargs)
        if quad_as_tri_alpha > 0:
            # Assumes no quad mask.
            xmid = 0.25*(x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
            ymid = 0.25*(y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])
            kwargs["alpha"] = quad_as_tri_alpha
            ax.plot(
                np.stack((x[:-1, :-1], xmid, x[1:, 1:])).reshape((3, -1)),
                np.stack((y[:-1, :-1], ymid, y[1:, 1:])).reshape((3, -1)),
                np.stack((x[1:, :-1], xmid, x[:-1, 1:])).reshape((3, -1)),
                np.stack((y[1:, :-1], ymid, y[:-1, 1:])).reshape((3, -1)),
                **kwargs)
        if point_color is not None:
            ax.plot(x, y, color=point_color, alpha=alpha, marker="o", lw=0)
        ax._need_autoscale = True  # type: ignore[attr-defined]

    def lines(
        self,
        lines: cpy.LineReturn,
        line_type: LineType | str,
        ax: Axes | int = 0,
        color: str = "C0",
        alpha: float = 1.0,
        linewidth: float = 1,
    ) -> None:
        """Plot contour lines on a single Axes.

        Args:
            lines (sequence of arrays): Contour line data as returned by
                :meth:`~.ContourGenerator.lines`.
            line_type (LineType or str): Type of :meth:`~.ContourGenerator.lines` data as returned
                by :attr:`~.ContourGenerator.line_type`, or string equivalent.
            ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
            color (str, optional): Color to plot lines. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``tab10`` colormap. Default ``"C0"``.
            alpha (float, optional): Opacity to plot lines with, default ``1.0``.
            linewidth (float, optional): Width of lines, default ``1``.
        """
        line_type = as_line_type(line_type)
        ax = self._get_ax(ax)
        paths = lines_to_mpl_paths(lines, line_type)
        collection = mcollections.PathCollection(
            paths, facecolors="none", edgecolors=color, lw=linewidth, alpha=alpha)
        ax.add_collection(collection)
        ax._need_autoscale = True  # type: ignore[attr-defined]

    def mask(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike | np.ma.MaskedArray[Any, Any],
        ax: Axes | int = 0,
        color: str = "black",
    ) -> None:
        """Plot masked out grid points as circles on a single Axes.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            z (masked array of shape (ny, nx): z-values.
            ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
            color (str, optional): Circle color, default ``"black"``.
        """
        mask = np.ma.getmask(z)  # type: ignore[no-untyped-call]
        if mask is np.ma.nomask:
            return
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        ax.plot(x[mask], y[mask], "o", c=color)

    def save(self, filename: str, transparent: bool = False) -> None:
        """Save plots to SVG or PNG file.

        Args:
            filename (str): Filename to save to.
            transparent (bool, optional): Whether background should be transparent, default
                ``False``.
        """
        self._autoscale()
        self._fig.savefig(filename, transparent=transparent)

    def save_to_buffer(self) -> io.BytesIO:
        """Save plots to an ``io.BytesIO`` buffer.

        Return:
            BytesIO: PNG image buffer.
        """
        self._autoscale()
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png")
        buf.seek(0)
        return buf

    def show(self) -> None:
        """Show plots in an interactive window, in the usual Matplotlib manner.
        """
        self._autoscale()
        plt.show()

    def title(self, title: str, ax: Axes | int = 0, color: str | None = None) -> None:
        """Set the title of a single Axes.

        Args:
            title (str): Title text.
            ax (int or Matplotlib Axes, optional): Which Axes to set the title of, default ``0``.
            color (str, optional): Color to set title. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``tab10`` colormap. Default is ``None`` which uses Matplotlib's default title color
                that depends on the stylesheet in use.
        """
        if color:
            self._get_ax(ax).set_title(title, color=color)
        else:
            self._get_ax(ax).set_title(title)

    def z_values(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        ax: Axes | int = 0,
        color: str = "green",
        fmt: str = ".1f",
        quad_as_tri: bool = False,
    ) -> None:
        """Show ``z`` values on a single Axes.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            z (array-like of shape (ny, nx): z-values.
            ax (int or Matplotlib Axes, optional): Which Axes to plot on, default ``0``.
            color (str, optional): Color of added text. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``tab10`` colormap. Default ``"green"``.
            fmt (str, optional): Format to display z-values, default ``".1f"``.
            quad_as_tri (bool, optional): Whether to show z-values at the ``quad_as_tri`` centers
                of quads.

        Warning:
            ``quad_as_tri=True`` shows z-values for all quads, even if masked.
        """
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        for j in range(ny):
            for i in range(nx):
                ax.text(x[j, i], y[j, i], f"{z[j, i]:{fmt}}", ha="center", va="center",
                        color=color, clip_on=True)
        if quad_as_tri:
            for j in range(ny-1):
                for i in range(nx-1):
                    xx = np.mean(x[j:j+2, i:i+2], dtype=np.float64)
                    yy = np.mean(y[j:j+2, i:i+2], dtype=np.float64)
                    zz = np.mean(z[j:j+2, i:i+2])
                    ax.text(xx, yy, f"{zz:{fmt}}", ha="center", va="center", color=color,
                            clip_on=True)


class MplTestRenderer(MplRenderer):
    """Test renderer implemented using Matplotlib.

    No whitespace around plots and no spines/ticks displayed.
    Uses Agg backend, so can only save to file/buffer, cannot call ``show()``.
    """
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] = (9, 9),
    ) -> None:
        gridspec = {
            "left": 0.01,
            "right": 0.99,
            "top": 0.99,
            "bottom": 0.01,
            "wspace": 0.01,
            "hspace": 0.01,
        }
        super().__init__(
            nrows, ncols, figsize, show_frame=True, backend="Agg", gridspec_kw=gridspec,
        )

        for ax in self._axes:
            ax.set_xmargin(0.0)
            ax.set_ymargin(0.0)
            ax.set_xticks([])
            ax.set_yticks([])

        self._want_tight = False


class MplDebugRenderer(MplRenderer):
    """Debug renderer implemented using Matplotlib.

    Extends ``MplRenderer`` to add extra information to help in debugging such as markers, arrows,
    text, etc.
    """
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] = (9, 9),
        show_frame: bool = True,
    ) -> None:
        super().__init__(nrows, ncols, figsize, show_frame)

    def _arrow(
        self,
        ax: Axes,
        line_start: cpy.CoordinateArray,
        line_end: cpy.CoordinateArray,
        color: str,
        alpha: float,
        arrow_size: float,
    ) -> None:
        mid = 0.5*(line_start + line_end)
        along = line_end - line_start
        along /= np.sqrt(np.dot(along, along))  # Unit vector.
        right = np.asarray((along[1], -along[0]))
        arrow = np.stack((
            mid - (along*0.5 - right)*arrow_size,
            mid + along*0.5*arrow_size,
            mid - (along*0.5 + right)*arrow_size,
        ))
        ax.plot(arrow[:, 0], arrow[:, 1], "-", c=color, alpha=alpha)

    def filled(
        self,
        filled: cpy.FillReturn,
        fill_type: FillType | str,
        ax: Axes | int = 0,
        color: str = "C1",
        alpha: float = 0.7,
        line_color: str = "C0",
        line_alpha: float = 0.7,
        point_color: str = "C0",
        start_point_color: str = "red",
        arrow_size: float = 0.1,
    ) -> None:
        fill_type = as_fill_type(fill_type)
        super().filled(filled, fill_type, ax, color, alpha)

        if line_color is None and point_color is None:
            return

        ax = self._get_ax(ax)
        filled = convert_filled(filled, fill_type, FillType.ChunkCombinedOffset)

        # Lines.
        if line_color is not None:
            for points, offsets in zip(*filled):
                if points is None:
                    continue
                for start, end in pairwise(offsets):
                    xys = points[start:end]
                    ax.plot(xys[:, 0], xys[:, 1], c=line_color, alpha=line_alpha)

                    if arrow_size > 0.0:
                        n = len(xys)
                        for i in range(n-1):
                            self._arrow(ax, xys[i], xys[i+1], line_color, line_alpha, arrow_size)

        # Points.
        if point_color is not None:
            for points, offsets in zip(*filled):
                if points is None:
                    continue
                mask = np.ones(offsets[-1], dtype=bool)
                mask[offsets[1:]-1] = False  # Exclude end points.
                if start_point_color is not None:
                    start_indices = offsets[:-1]
                    mask[start_indices] = False  # Exclude start points.
                ax.plot(
                    points[:, 0][mask], points[:, 1][mask], "o", c=point_color, alpha=line_alpha)

                if start_point_color is not None:
                    ax.plot(points[:, 0][start_indices], points[:, 1][start_indices], "o",
                            c=start_point_color, alpha=line_alpha)

    def lines(
        self,
        lines: cpy.LineReturn,
        line_type: LineType | str,
        ax: Axes | int = 0,
        color: str = "C0",
        alpha: float = 1.0,
        linewidth: float = 1,
        point_color: str = "C0",
        start_point_color: str = "red",
        arrow_size: float = 0.1,
    ) -> None:
        line_type = as_line_type(line_type)
        super().lines(lines, line_type, ax, color, alpha, linewidth)

        if arrow_size == 0.0 and point_color is None:
            return

        ax = self._get_ax(ax)
        separate_lines = convert_lines(lines, line_type, LineType.Separate)
        if TYPE_CHECKING:
            separate_lines = cast(cpy.LineReturn_Separate, separate_lines)

        if arrow_size > 0.0:
            for line in separate_lines:
                for i in range(len(line)-1):
                    self._arrow(ax, line[i], line[i+1], color, alpha, arrow_size)

        if point_color is not None:
            for line in separate_lines:
                start_index = 0
                end_index = len(line)
                if start_point_color is not None:
                    ax.plot(line[0, 0], line[0, 1], "o", c=start_point_color, alpha=alpha)
                    start_index = 1
                    if line[0][0] == line[-1][0] and line[0][1] == line[-1][1]:
                        end_index -= 1
                ax.plot(line[start_index:end_index, 0], line[start_index:end_index, 1], "o",
                        c=color, alpha=alpha)

    def point_numbers(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        ax: Axes | int = 0,
        color: str = "red",
    ) -> None:
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        for j in range(ny):
            for i in range(nx):
                quad = i + j*nx
                ax.text(x[j, i], y[j, i], str(quad), ha="right", va="top", color=color,
                        clip_on=True)

    def quad_numbers(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        ax: Axes | int = 0,
        color: str = "blue",
    ) -> None:
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        for j in range(1, ny):
            for i in range(1, nx):
                quad = i + j*nx
                xmid = x[j-1:j+1, i-1:i+1].mean()
                ymid = y[j-1:j+1, i-1:i+1].mean()
                ax.text(xmid, ymid, str(quad), ha="center", va="center", color=color, clip_on=True)

    def z_levels(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        lower_level: float,
        upper_level: float | None = None,
        ax: Axes | int = 0,
        color: str = "green",
    ) -> None:
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        for j in range(ny):
            for i in range(nx):
                zz = z[j, i]
                if upper_level is not None and zz > upper_level:
                    z_level = 2
                elif zz > lower_level:
                    z_level = 1
                else:
                    z_level = 0
                ax.text(x[j, i], y[j, i], str(z_level), ha="left", va="bottom", color=color,
                        clip_on=True)
<|endoftext|>from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from bokeh.io import export_png, export_svg, show
from bokeh.io.export import get_screenshot_as_png
from bokeh.layouts import gridplot
from bokeh.models.annotations.labels import Label
from bokeh.palettes import Category10
from bokeh.plotting import figure
import numpy as np

from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.bokeh_util import filled_to_bokeh, lines_to_bokeh
from contourpy.util.renderer import Renderer

if TYPE_CHECKING:
    from bokeh.core.enums import OutputBackendType
    from bokeh.models import GridPlot
    from bokeh.palettes import Palette
    from numpy.typing import ArrayLike
    from selenium.webdriver.remote.webdriver import WebDriver

    from contourpy import FillType, LineType
    from contourpy._contourpy import FillReturn, LineReturn


class BokehRenderer(Renderer):
    """Utility renderer using Bokeh to render a grid of plots over the same (x, y) range.

    Args:
        nrows (int, optional): Number of rows of plots, default ``1``.
        ncols (int, optional): Number of columns of plots, default ``1``.
        figsize (tuple(float, float), optional): Figure size in inches (assuming 100 dpi), default
            ``(9, 9)``.
        show_frame (bool, optional): Whether to show frame and axes ticks, default ``True``.
        want_svg (bool, optional): Whether output is required in SVG format or not, default
            ``False``.

    Warning:
        :class:`~.BokehRenderer`, unlike :class:`~.MplRenderer`, needs to be told in advance if
        output to SVG format will be required later, otherwise it will assume PNG output.
    """
    _figures: list[figure]
    _layout: GridPlot
    _palette: Palette
    _want_svg: bool

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] = (9, 9),
        show_frame: bool = True,
        want_svg: bool = False,
    ) -> None:
        self._want_svg = want_svg
        self._palette = Category10[10]

        total_size = 100*np.asarray(figsize, dtype=int)  # Assuming 100 dpi.

        nfigures = nrows*ncols
        self._figures = []
        backend: OutputBackendType = "svg" if self._want_svg else "canvas"
        for _ in range(nfigures):
            fig = figure(output_backend=backend)
            fig.xgrid.visible = False  # type: ignore[attr-defined]
            fig.ygrid.visible = False  # type: ignore[attr-defined]
            self._figures.append(fig)
            if not show_frame:
                fig.outline_line_color = None
                fig.axis.visible = False  # type: ignore[attr-defined]

        self._layout = gridplot(
            self._figures, ncols=ncols, toolbar_location=None,  # type: ignore[arg-type]
            width=total_size[0] // ncols, height=total_size[1] // nrows)

    def _convert_color(self, color: str) -> str:
        if isinstance(color, str) and color[0] == "C":
            index = int(color[1:])
            color = self._palette[index]
        return color

    def _get_figure(self, ax: figure | int) -> figure:
        if isinstance(ax, int):
            ax = self._figures[ax]
        return ax

    def filled(
        self,
        filled: FillReturn,
        fill_type: FillType | str,
        ax: figure | int = 0,
        color: str = "C0",
        alpha: float = 0.7,
    ) -> None:
        """Plot filled contours on a single plot.

        Args:
            filled (sequence of arrays): Filled contour data as returned by
                :meth:`~.ContourGenerator.filled`.
            fill_type (FillType or str): Type of :meth:`~.ContourGenerator.filled` data as returned
                by :attr:`~.ContourGenerator.fill_type`, or a string equivalent.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Color to plot with. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``Category10`` palette. Default ``"C0"``.
            alpha (float, optional): Opacity to plot with, default ``0.7``.
        """
        fill_type = as_fill_type(fill_type)
        fig = self._get_figure(ax)
        color = self._convert_color(color)
        xs, ys = filled_to_bokeh(filled, fill_type)
        if len(xs) > 0:
            fig.multi_polygons(xs=[xs], ys=[ys], color=color, fill_alpha=alpha, line_width=0)  # type: ignore[arg-type]

    def grid(
        self,
        x: ArrayLike,
        y: ArrayLike,
        ax: figure | int = 0,
        color: str = "black",
        alpha: float = 0.1,
        point_color: str | None = None,
        quad_as_tri_alpha: float = 0,
    ) -> None:
        """Plot quad grid lines on a single plot.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Color to plot grid lines, default ``"black"``.
            alpha (float, optional): Opacity to plot lines with, default ``0.1``.
            point_color (str, optional): Color to plot grid points or ``None`` if grid points
                should not be plotted, default ``None``.
            quad_as_tri_alpha (float, optional): Opacity to plot ``quad_as_tri`` grid, default
                ``0``.

        Colors may be a string color or the letter ``"C"`` followed by an integer in the range
        ``"C0"`` to ``"C9"`` to use a color from the ``Category10`` palette.

        Warning:
            ``quad_as_tri_alpha > 0`` plots all quads as though they are unmasked.
        """
        fig = self._get_figure(ax)
        x, y = self._grid_as_2d(x, y)
        xs = list(x) + list(x.T)
        ys = list(y) + list(y.T)
        kwargs = {"line_color": color, "alpha": alpha}
        fig.multi_line(xs, ys, **kwargs)
        if quad_as_tri_alpha > 0:
            # Assumes no quad mask.
            xmid = (0.25*(x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])).ravel()
            ymid = (0.25*(y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])).ravel()
            fig.multi_line(
                list(np.stack((x[:-1, :-1].ravel(), xmid, x[1:, 1:].ravel()), axis=1)),
                list(np.stack((y[:-1, :-1].ravel(), ymid, y[1:, 1:].ravel()), axis=1)),
                **kwargs)
            fig.multi_line(
                list(np.stack((x[:-1, 1:].ravel(), xmid, x[1:, :-1].ravel()), axis=1)),
                list(np.stack((y[:-1, 1:].ravel(), ymid, y[1:, :-1].ravel()), axis=1)),
                **kwargs)
        if point_color is not None:
            fig.scatter(
                x=x.ravel(), y=y.ravel(), fill_color=color, line_color=None, alpha=alpha,
                marker="circle", size=8)

    def lines(
        self,
        lines: LineReturn,
        line_type: LineType | str,
        ax: figure | int = 0,
        color: str = "C0",
        alpha: float = 1.0,
        linewidth: float = 1,
    ) -> None:
        """Plot contour lines on a single plot.

        Args:
            lines (sequence of arrays): Contour line data as returned by
                :meth:`~.ContourGenerator.lines`.
            line_type (LineType or str): Type of :meth:`~.ContourGenerator.lines` data as returned
                by :attr:`~.ContourGenerator.line_type`, or a string equivalent.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Color to plot lines. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``Category10`` palette. Default ``"C0"``.
            alpha (float, optional): Opacity to plot lines with, default ``1.0``.
            linewidth (float, optional): Width of lines, default ``1``.

        Note:
            Assumes all lines are open line strips not closed line loops.
        """
        line_type = as_line_type(line_type)
        fig = self._get_figure(ax)
        color = self._convert_color(color)
        xs, ys = lines_to_bokeh(lines, line_type)
        if xs is not None:
            assert ys is not None
            fig.line(xs, ys, line_color=color, line_alpha=alpha, line_width=linewidth)

    def mask(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike | np.ma.MaskedArray[Any, Any],
        ax: figure | int = 0,
        color: str = "black",
    ) -> None:
        """Plot masked out grid points as circles on a single plot.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            z (masked array of shape (ny, nx): z-values.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Circle color, default ``"black"``.
        """
        mask = np.ma.getmask(z)  # type: ignore[no-untyped-call]
        if mask is np.ma.nomask:
            return
        fig = self._get_figure(ax)
        color = self._convert_color(color)
        x, y = self._grid_as_2d(x, y)
        fig.scatter(x[mask], y[mask], fill_color=color, marker="circle", size=10)

    def save(
        self,
        filename: str,
        transparent: bool = False,
        *,
        webdriver: WebDriver | None = None,
    ) -> None:
        """Save plots to SVG or PNG file.

        Args:
            filename (str): Filename to save to.
            transparent (bool, optional): Whether background should be transparent, default
                ``False``.
            webdriver (WebDriver, optional): Selenium WebDriver instance to use to create the image.

                .. versionadded:: 1.1.1

        Warning:
            To output to SVG file, ``want_svg=True`` must have been passed to the constructor.
        """
        if transparent:
            for fig in self._figures:
                fig.background_fill_color = None
                fig.border_fill_color = None

        if self._want_svg:
            export_svg(self._layout, filename=filename, webdriver=webdriver)
        else:
            export_png(self._layout, filename=filename, webdriver=webdriver)

    def save_to_buffer(self, *, webdriver: WebDriver | None = None) -> io.BytesIO:
        """Save plots to an ``io.BytesIO`` buffer.

        Args:
            webdriver (WebDriver, optional): Selenium WebDriver instance to use to create the image.

                .. versionadded:: 1.1.1

        Return:
            BytesIO: PNG image buffer.
        """
        image = get_screenshot_as_png(self._layout, driver=webdriver)
        buffer = io.BytesIO()
        image.save(buffer, "png")
        return buffer

    def show(self) -> None:
        """Show plots in web browser, in usual Bokeh manner.
        """
        show(self._layout)

    def title(self, title: str, ax: figure | int = 0, color: str | None = None) -> None:
        """Set the title of a single plot.

        Args:
            title (str): Title text.
            ax (int or Bokeh Figure, optional): Which plot to set the title of, default ``0``.
            color (str, optional): Color to set title. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``Category10`` palette. Default ``None`` which is ``black``.
        """
        fig = self._get_figure(ax)
        fig.title = title
        fig.title.align = "center"  # type: ignore[attr-defined]
        if color is not None:
            fig.title.text_color = self._convert_color(color)  # type: ignore[attr-defined]

    def z_values(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        ax: figure | int = 0,
        color: str = "green",
        fmt: str = ".1f",
        quad_as_tri: bool = False,
    ) -> None:
        """Show ``z`` values on a single plot.

        Args:
            x (array-like of shape (ny, nx) or (nx,)): The x-coordinates of the grid points.
            y (array-like of shape (ny, nx) or (ny,)): The y-coordinates of the grid points.
            z (array-like of shape (ny, nx): z-values.
            ax (int or Bokeh Figure, optional): Which plot to use, default ``0``.
            color (str, optional): Color of added text. May be a string color or the letter ``"C"``
                followed by an integer in the range ``"C0"`` to ``"C9"`` to use a color from the
                ``Category10`` palette. Default ``"green"``.
            fmt (str, optional): Format to display z-values, default ``".1f"``.
            quad_as_tri (bool, optional): Whether to show z-values at the ``quad_as_tri`` centres
                of quads.

        Warning:
            ``quad_as_tri=True`` shows z-values for all quads, even if masked.
        """
        fig = self._get_figure(ax)
        color = self._convert_color(color)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        kwargs = {"text_color": color, "text_align": "center", "text_baseline": "middle"}
        for j in range(ny):
            for i in range(nx):
                label = Label(x=x[j, i], y=y[j, i], text=f"{z[j, i]:{fmt}}", **kwargs)  # type: ignore[arg-type]
                fig.add_layout(label)
        if quad_as_tri:
            for j in range(ny-1):
                for i in range(nx-1):
                    xx = np.mean(x[j:j+2, i:i+2])
                    yy = np.mean(y[j:j+2, i:i+2])
                    zz = np.mean(z[j:j+2, i:i+2])
                    fig.add_layout(Label(x=xx, y=yy, text=f"{zz:{fmt}}", **kwargs))  # type: ignore[arg-type]
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from contourpy._contourpy import CoordinateArray


def simple(
    shape: tuple[int, int], want_mask: bool = False,
) -> tuple[CoordinateArray, CoordinateArray, CoordinateArray | np.ma.MaskedArray[Any, Any]]:
    """Return simple test data consisting of the sum of two gaussians.

    Args:
        shape (tuple(int, int)): 2D shape of data to return.
        want_mask (bool, optional): Whether test data should be masked or not, default ``False``.

    Return:
        Tuple of 3 arrays: ``x``, ``y``, ``z`` test data, ``z`` will be masked if
        ``want_mask=True``.
    """
    ny, nx = shape
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    x, y = np.meshgrid(x, y)

    xscale = nx - 1.0
    yscale = ny - 1.0

    # z is sum of 2D gaussians.
    amp = np.asarray([1.0, -1.0, 0.8, -0.9, 0.7])
    mid = np.asarray([[0.4, 0.2], [0.3, 0.8], [0.9, 0.75], [0.7, 0.3], [0.05, 0.7]])
    width = np.asarray([0.4, 0.2, 0.2, 0.2, 0.1])

    z = np.zeros_like(x)
    for i in range(len(amp)):
        z += amp[i]*np.exp(-((x/xscale - mid[i, 0])**2 + (y/yscale - mid[i, 1])**2) / width[i]**2)

    if want_mask:
        mask = np.logical_or(
            ((x/xscale - 1.0)**2 / 0.2 + (y/yscale - 0.0)**2 / 0.1) < 1.0,
            ((x/xscale - 0.2)**2 / 0.02 + (y/yscale - 0.45)**2 / 0.08) < 1.0,
        )
        z = np.ma.array(z, mask=mask)  # type: ignore[no-untyped-call]

    return x, y, z


def random(
    shape: tuple[int, int], seed: int = 2187, mask_fraction: float = 0.0,
) -> tuple[CoordinateArray, CoordinateArray, CoordinateArray | np.ma.MaskedArray[Any, Any]]:
    """Return random test data in the range 0 to 1.

    Args:
        shape (tuple(int, int)): 2D shape of data to return.
        seed (int, optional): Seed for random number generator, default 2187.
        mask_fraction (float, optional): Fraction of elements to mask, default 0.

    Return:
        Tuple of 3 arrays: ``x``, ``y``, ``z`` test data, ``z`` will be masked if
        ``mask_fraction`` is greater than zero.
    """
    ny, nx = shape
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    x, y = np.meshgrid(x, y)

    rng = np.random.default_rng(seed)
    z = rng.uniform(size=shape)

    if mask_fraction > 0.0:
        mask_fraction = min(mask_fraction, 0.99)
        mask = rng.uniform(size=shape) < mask_fraction
        z = np.ma.array(z, mask=mask)  # type: ignore[no-untyped-call]

    return x, y, z
<|endoftext|>from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import io

    from numpy.typing import ArrayLike

    from contourpy._contourpy import CoordinateArray, FillReturn, FillType, LineReturn, LineType


class Renderer(ABC):
    """Abstract base class for renderers."""

    def _grid_as_2d(self, x: ArrayLike, y: ArrayLike) -> tuple[CoordinateArray, CoordinateArray]:
        x = np.asarray(x)
        y = np.asarray(y)
        if x.ndim == 1:
            x, y = np.meshgrid(x, y)
        return x, y

    @abstractmethod
    def filled(
        self,
        filled: FillReturn,
        fill_type: FillType | str,
        ax: Any = 0,
        color: str = "C0",
        alpha: float = 0.7,
    ) -> None:
        pass

    @abstractmethod
    def grid(
        self,
        x: ArrayLike,
        y: ArrayLike,
        ax: Any = 0,
        color: str = "black",
        alpha: float = 0.1,
        point_color: str | None = None,
        quad_as_tri_alpha: float = 0,
    ) -> None:
        pass

    @abstractmethod
    def lines(
        self,
        lines: LineReturn,
        line_type: LineType | str,
        ax: Any = 0,
        color: str = "C0",
        alpha: float = 1.0,
        linewidth: float = 1,
    ) -> None:
        pass

    @abstractmethod
    def mask(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike | np.ma.MaskedArray[Any, Any],
        ax: Any = 0,
        color: str = "black",
    ) -> None:
        pass

    def multi_filled(
        self,
        multi_filled: list[FillReturn],
        fill_type: FillType | str,
        ax: Any = 0,
        color: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot multiple sets of filled contours on a single axes.

        Args:
            multi_filled (list of filled contour arrays): Multiple filled contour sets as returned
                by :meth:`.ContourGenerator.multi_filled`.
            fill_type (FillType or str): Type of filled data as returned by
                :attr:`~.ContourGenerator.fill_type`, or string equivalent.
            ax (int or Renderer-specific axes or figure object, optional): Which axes to plot on,
                default ``0``.
            color (str or None, optional): If a string color then this same color is used for all
                filled contours. If ``None``, the default, then the filled contour sets use colors
                from the ``tab10`` colormap in order, wrapping around to the beginning if more than
                10 sets of filled contours are rendered.
            kwargs: All other keyword argument are passed on to
                :meth:`.Renderer.filled` unchanged.

        .. versionadded:: 1.3.0
        """
        if color is not None:
            kwargs["color"] = color
        for i, filled in enumerate(multi_filled):
            if color is None:
                kwargs["color"] = f"C{i % 10}"
            self.filled(filled, fill_type, ax, **kwargs)

    def multi_lines(
        self,
        multi_lines: list[LineReturn],
        line_type: LineType | str,
        ax: Any = 0,
        color: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Plot multiple sets of contour lines on a single axes.

        Args:
            multi_lines (list of contour line arrays): Multiple contour line sets as returned by
                :meth:`.ContourGenerator.multi_lines`.
            line_type (LineType or str): Type of line data as returned by
                :attr:`~.ContourGenerator.line_type`, or string equivalent.
            ax (int or Renderer-specific axes or figure object, optional): Which axes to plot on,
                default ``0``.
            color (str or None, optional): If a string color then this same color is used for all
                lines. If ``None``, the default, then the line sets use colors from the ``tab10``
                colormap in order, wrapping around to the beginning if more than 10 sets of lines
                are rendered.
            kwargs: All other keyword argument are passed on to
                :meth:`Renderer.lines` unchanged.

        .. versionadded:: 1.3.0
        """
        if color is not None:
            kwargs["color"] = color
        for i, lines in enumerate(multi_lines):
            if color is None:
                kwargs["color"] = f"C{i % 10}"
            self.lines(lines, line_type, ax, **kwargs)

    @abstractmethod
    def save(self, filename: str, transparent: bool = False) -> None:
        pass

    @abstractmethod
    def save_to_buffer(self) -> io.BytesIO:
        pass

    @abstractmethod
    def show(self) -> None:
        pass

    @abstractmethod
    def title(self, title: str, ax: Any = 0, color: str | None = None) -> None:
        pass

    @abstractmethod
    def z_values(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        ax: Any = 0,
        color: str = "green",
        fmt: str = ".1f",
        quad_as_tri: bool = False,
    ) -> None:
        pass
<|endoftext|>from __future__ import annotations

import base64
import ssl
import typing
import urllib.parse
from collections.abc import AsyncGenerator

from ._utils import safe_async_iterate

# Functions for typechecking...


ByteOrStr = bytes | str
HeadersAsSequence = typing.Sequence[tuple[ByteOrStr, ByteOrStr]]
HeadersAsMapping = typing.Mapping[ByteOrStr, ByteOrStr]
HeaderTypes = HeadersAsSequence | HeadersAsMapping | None

Extensions = typing.MutableMapping[str, typing.Any]


def enforce_bytes(value: bytes | str, *, name: str) -> bytes:
    """
    Any arguments that are ultimately represented as bytes can be specified
    either as bytes or as strings.

    However we enforce that any string arguments must only contain characters in
    the plain ASCII range. chr(0)...chr(127). If you need to use characters
    outside that range then be precise, and use a byte-wise argument.
    """
    if isinstance(value, str):
        try:
            return value.encode("ascii")
        except UnicodeEncodeError:
            raise TypeError(f"{name} strings may not include unicode characters.")
    elif isinstance(value, bytes):
        return value

    seen_type = type(value).__name__
    raise TypeError(f"{name} must be bytes or str, but got {seen_type}.")


def enforce_url(value: URL | bytes | str, *, name: str) -> URL:
    """
    Type check for URL parameters.
    """
    if isinstance(value, (bytes, str)):
        return URL(value)
    elif isinstance(value, URL):
        return value

    seen_type = type(value).__name__
    raise TypeError(f"{name} must be a URL, bytes, or str, but got {seen_type}.")


def enforce_headers(
    value: HeadersAsMapping | HeadersAsSequence | None = None, *, name: str
) -> list[tuple[bytes, bytes]]:
    """
    Convenience function that ensure all items in request or response headers
    are either bytes or strings in the plain ASCII range.
    """
    if value is None:
        return []
    elif isinstance(value, typing.Mapping):
        return [
            (
                enforce_bytes(k, name="header name"),
                enforce_bytes(v, name="header value"),
            )
            for k, v in value.items()
        ]
    elif isinstance(value, typing.Sequence):
        return [
            (
                enforce_bytes(k, name="header name"),
                enforce_bytes(v, name="header value"),
            )
            for k, v in value
        ]

    seen_type = type(value).__name__
    raise TypeError(f"{name} must be a mapping or sequence of two-tuples, but got {seen_type}.")


def enforce_stream(
    value: bytes | typing.Iterable[bytes] | typing.AsyncIterable[bytes] | None,
    *,
    name: str,
) -> typing.Iterable[bytes] | typing.AsyncIterable[bytes]:
    if value is None:
        return ByteStream(b"")
    elif isinstance(value, bytes):
        return ByteStream(value)
    return value


# * https://tools.ietf.org/html/rfc3986#section-3.2.3
# * https://url.spec.whatwg.org/#url-miscellaneous
# * https://url.spec.whatwg.org/#scheme-state
DEFAULT_PORTS = {
    b"ftp": 21,
    b"http": 80,
    b"https": 443,
    b"ws": 80,
    b"wss": 443,
}


def include_request_headers(
    headers: list[tuple[bytes, bytes]],
    *,
    url: URL,
    content: None | bytes | typing.Iterable[bytes] | typing.AsyncIterable[bytes],
) -> list[tuple[bytes, bytes]]:
    headers_set = {k.lower() for k, _v in headers}

    if b"host" not in headers_set:
        default_port = DEFAULT_PORTS.get(url.scheme)
        if url.port is None or url.port == default_port:
            header_value = url.host
        else:
            header_value = b"%b:%d" % (url.host, url.port)
        headers = [(b"Host", header_value)] + headers

    if content is not None and b"content-length" not in headers_set and b"transfer-encoding" not in headers_set:
        if isinstance(content, bytes):
            content_length = str(len(content)).encode("ascii")
            headers += [(b"Content-Length", content_length)]
        else:
            headers += [(b"Transfer-Encoding", b"chunked")]  # pragma: no cover

    return headers


# Interfaces for byte streams...


class ByteStream:
    """
    A container for non-streaming content, and that supports both sync and async
    stream iteration.
    """

    def __init__(self, content: bytes) -> None:
        self._content = content

    def __iter__(self) -> typing.Iterator[bytes]:
        yield self._content

    async def __aiter__(self) -> AsyncGenerator[bytes]:
        yield self._content

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{len(self._content)} bytes]>"


class Origin:
    def __init__(self, scheme: bytes, host: bytes, port: int) -> None:
        self.scheme = scheme
        self.host = host
        self.port = port

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, Origin)
            and self.scheme == other.scheme
            and self.host == other.host
            and self.port == other.port
        )

    def __str__(self) -> str:
        scheme = self.scheme.decode("ascii")
        host = self.host.decode("ascii")
        port = str(self.port)
        return f"{scheme}://{host}:{port}"


class URL:
    """
    Represents the URL against which an HTTP request may be made.

    The URL may either be specified as a plain string, for convenience:

    ```python
    url = httpcore2.URL("https://www.example.com/")
    ```

    Or be constructed with explicitly pre-parsed components:

    ```python
    url = httpcore2.URL(scheme=b'https', host=b'www.example.com', port=None, target=b'/')
    ```

    Using this second more explicit style allows integrations that are using
    `httpcore` to pass through URLs that have already been parsed in order to use
    libraries such as `rfc-3986` rather than relying on the stdlib. It also ensures
    that URL parsing is treated identically at both the networking level and at any
    higher layers of abstraction.

    The four components are important here, as they allow the URL to be precisely
    specified in a pre-parsed format. They also allow certain types of request to
    be created that could not otherwise be expressed.

    For example, an HTTP request to `http://www.example.com/` forwarded via a proxy
    at `http://localhost:8080`...

    ```python
    # Constructs an HTTP request with a complete URL as the target:
    # GET https://www.example.com/ HTTP/1.1
    url = httpcore2.URL(
        scheme=b'http',
        host=b'localhost',
        port=8080,
        target=b'https://www.example.com/'
    )
    request = httpcore2.Request(
        method="GET",
        url=url
    )
    ```

    Another example is constructing an `OPTIONS *` request...

    ```python
    # Constructs an 'OPTIONS *' HTTP request:
    # OPTIONS * HTTP/1.1
    url = httpcore2.URL(scheme=b'https', host=b'www.example.com', target=b'*')
    request = httpcore2.Request(method="OPTIONS", url=url)
    ```

    This kind of request is not possible to formulate with a URL string,
    because the `/` delimiter is always used to demark the target from the
    host/port portion of the URL.

    For convenience, string-like arguments may be specified either as strings or
    as bytes. However, once a request is being issue over-the-wire, the URL
    components are always ultimately required to be a bytewise representation.

    In order to avoid any ambiguity over character encodings, when strings are used
    as arguments, they must be strictly limited to the ASCII range `chr(0)`-`chr(127)`.
    If you require a bytewise representation that is outside this range you must
    handle the character encoding directly, and pass a bytes instance.
    """

    def __init__(
        self,
        url: bytes | str = "",
        *,
        scheme: bytes | str = b"",
        host: bytes | str = b"",
        port: int | None = None,
        target: bytes | str = b"",
    ) -> None:
        """
        Parameters:
            url: The complete URL as a string or bytes.
            scheme: The URL scheme as a string or bytes.
                Typically either `"http"` or `"https"`.
            host: The URL host as a string or bytes. Such as `"www.example.com"`.
            port: The port to connect to. Either an integer or `None`.
            target: The target of the HTTP request. Such as `"/items?search=red"`.
        """
        if url:
            parsed = urllib.parse.urlparse(enforce_bytes(url, name="url"))
            self.scheme = parsed.scheme
            self.host = parsed.hostname or b""
            self.port = parsed.port
            self.target = (parsed.path or b"/") + (b"?" + parsed.query if parsed.query else b"")
        else:
            self.scheme = enforce_bytes(scheme, name="scheme")
            self.host = enforce_bytes(host, name="host")
            self.port = port
            self.target = enforce_bytes(target, name="target")

    @property
    def origin(self) -> Origin:
        default_port = {
            b"http": 80,
            b"https": 443,
            b"ws": 80,
            b"wss": 443,
            b"socks5": 1080,
            b"socks5h": 1080,
        }[self.scheme]
        return Origin(scheme=self.scheme, host=self.host, port=self.port or default_port)

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, URL)
            and other.scheme == self.scheme
            and other.host == self.host
            and other.port == self.port
            and other.target == self.target
        )

    def __bytes__(self) -> bytes:
        if self.port is None:
            return b"%b://%b%b" % (self.scheme, self.host, self.target)
        return b"%b://%b:%d%b" % (self.scheme, self.host, self.port, self.target)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(scheme={self.scheme!r}, "
            f"host={self.host!r}, port={self.port!r}, target={self.target!r})"
        )


class Request:
    """
    An HTTP request.
    """

    def __init__(
        self,
        method: bytes | str,
        url: URL | bytes | str,
        *,
        headers: HeaderTypes = None,
        content: bytes | typing.Iterable[bytes] | typing.AsyncIterable[bytes] | None = None,
        extensions: Extensions | None = None,
    ) -> None:
        """
        Parameters:
            method: The HTTP request method, either as a string or bytes.
                For example: `GET`.
            url: The request URL, either as a `URL` instance, or as a string or bytes.
                For example: `"https://www.example.com".`
            headers: The HTTP request headers.
            content: The content of the request body.
            extensions: A dictionary of optional extra information included on
                the request. Possible keys include `"timeout"`, and `"trace"`.
        """
        self.method: bytes = enforce_bytes(method, name="method")
        self.url: URL = enforce_url(url, name="url")
        self.headers: list[tuple[bytes, bytes]] = enforce_headers(headers, name="headers")
        self.stream: typing.Iterable[bytes] | typing.AsyncIterable[bytes] = enforce_stream(content, name="content")
        self.extensions = {} if extensions is None else extensions

        if "target" in self.extensions:
            self.url = URL(
                scheme=self.url.scheme,
                host=self.url.host,
                port=self.url.port,
                target=self.extensions["target"],
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.method!r}]>"


class Response:
    """
    An HTTP response.
    """

    def __init__(
        self,
        status: int,
        *,
        headers: HeaderTypes = None,
        content: bytes | typing.Iterable[bytes] | typing.AsyncIterable[bytes] | None = None,
        extensions: Extensions | None = None,
    ) -> None:
        """
        Parameters:
            status: The HTTP status code of the response. For example `200`.
            headers: The HTTP response headers.
            content: The content of the response body.
            extensions: A dictionary of optional extra information included on
                the responseself.Possible keys include `"http_version"`,
                `"reason_phrase"`, and `"network_stream"`.
        """
        self.status: int = status
        self.headers: list[tuple[bytes, bytes]] = enforce_headers(headers, name="headers")
        self.stream: typing.Iterable[bytes] | typing.AsyncIterable[bytes] = enforce_stream(content, name="content")
        self.extensions = {} if extensions is None else extensions

        self._stream_consumed = False

    @property
    def content(self) -> bytes:
        if not hasattr(self, "_content"):
            if isinstance(self.stream, typing.Iterable):
                raise RuntimeError(
                    "Attempted to access 'response.content' on a streaming response. Call 'response.read()' first."
                )
            else:
                raise RuntimeError(
                    "Attempted to access 'response.content' on a streaming response. "
                    "Call 'await response.aread()' first."
                )
        return self._content

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [{self.status}]>"

    # Sync interface...

    def read(self) -> bytes:
        if not isinstance(self.stream, typing.Iterable):  # pragma: no cover
            raise RuntimeError(
                "Attempted to read an asynchronous response using 'response.read()'. "
                "You should use 'await response.aread()' instead."
            )
        if not hasattr(self, "_content"):
            self._content = b"".join(list(self.iter_stream()))
        return self._content

    def iter_stream(self) -> typing.Iterator[bytes]:
        if not isinstance(self.stream, typing.Iterable):  # pragma: no cover
            raise RuntimeError(
                "Attempted to stream an asynchronous response using 'for ... in "
                "response.iter_stream()'. "
                "You should use 'async for ... in response.aiter_stream()' instead."
            )
        if self._stream_consumed:
            raise RuntimeError("Attempted to call 'for ... in response.iter_stream()' more than once.")
        self._stream_consumed = True
        yield from self.stream

    def close(self) -> None:
        if not isinstance(self.stream, typing.Iterable):  # pragma: no cover
            raise RuntimeError(
                "Attempted to close an asynchronous response using 'response.close()'. "
                "You should use 'await response.aclose()' instead."
            )
        if hasattr(self.stream, "close"):
            self.stream.close()

    # Async interface...

    async def aread(self) -> bytes:
        if not isinstance(self.stream, typing.AsyncIterable):  # pragma: no cover
            raise RuntimeError(
                "Attempted to read an synchronous response using "
                "'await response.aread()'. "
                "You should use 'response.read()' instead."
            )
        if not hasattr(self, "_content"):
            async with safe_async_iterate(self.aiter_stream()) as parts:
                self._content = b"".join([part async for part in parts])
        return self._content

    async def aiter_stream(self) -> AsyncGenerator[bytes]:
        if not isinstance(self.stream, typing.AsyncIterable):  # pragma: no cover
            raise RuntimeError(
                "Attempted to stream an synchronous response using 'async for ... in "
                "response.aiter_stream()'. "
                "You should use 'for ... in response.iter_stream()' instead."
            )
        if self._stream_consumed:
            raise RuntimeError("Attempted to call 'async for ... in response.aiter_stream()' more than once.")
        self._stream_consumed = True
        async with safe_async_iterate(self.stream) as iterator:
            async for chunk in iterator:
                yield chunk

    async def aclose(self) -> None:
        if not isinstance(self.stream, typing.AsyncIterable):  # pragma: no cover
            raise RuntimeError(
                "Attempted to close a synchronous response using "
                "'await response.aclose()'. "
                "You should use 'response.close()' instead."
            )
        if hasattr(self.stream, "aclose"):
            await self.stream.aclose()


class Proxy:
    def __init__(
        self,
        url: URL | bytes | str,
        auth: tuple[bytes | str, bytes | str] | None = None,
        headers: HeadersAsMapping | HeadersAsSequence | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ):
        self.url = enforce_url(url, name="url")
        self.headers = enforce_headers(headers, name="headers")
        self.ssl_context = ssl_context

        if auth is not None:
            username = enforce_bytes(auth[0], name="auth")
            password = enforce_bytes(auth[1], name="auth")
            userpass = username + b":" + password
            authorization = b"Basic " + base64.b64encode(userpass)
            self.auth: tuple[bytes, bytes] | None = (username, password)
            self.headers = [(b"Proxy-Authorization", authorization)] + self.headers
        else:
            self.auth = None
<|endoftext|>from __future__ import annotations

import threading
import types

from ._exceptions import ExceptionMapping, PoolTimeout, map_exceptions

# Our async synchronization primitives use either 'anyio' or 'trio' depending
# on if they're running under asyncio or trio.

try:
    import trio
except (ImportError, NotImplementedError):  # pragma: no cover
    trio = None  # type: ignore

try:
    import anyio
except ImportError:  # pragma: no cover
    anyio = None  # type: ignore

try:
    import sniffio
except ImportError:  # pragma: no cover
    sniffio = None  # type: ignore


def current_async_library() -> str:
    # Determine if we're running under trio or asyncio.
    # See https://sniffio.readthedocs.io/en/latest/
    if sniffio is None:  # pragma: no cover
        environment = "asyncio"
    else:
        environment = sniffio.current_async_library()

    if environment not in ("asyncio", "trio"):  # pragma: no cover
        raise RuntimeError("Running under an unsupported async environment.")

    if environment == "asyncio" and anyio is None:  # pragma: no cover
        raise RuntimeError("Running with asyncio requires installation of 'httpcore[asyncio]'.")

    if environment == "trio" and trio is None:  # pragma: no cover
        raise RuntimeError("Running with trio requires installation of 'httpcore[trio]'.")

    return environment


class AsyncLock:
    """
    This is a standard lock.

    In the sync case `Lock` provides thread locking.
    In the async case `AsyncLock` provides async locking.
    """

    def __init__(self) -> None:
        self._backend = ""

    def setup(self) -> None:
        """
        Detect if we're running under 'asyncio' or 'trio' and create
        a lock with the correct implementation.
        """
        self._backend = current_async_library()
        if self._backend == "trio":
            self._trio_lock = trio.Lock()
        elif self._backend == "asyncio":
            self._anyio_lock = anyio.Lock(fast_acquire=True)

    async def __aenter__(self) -> AsyncLock:
        if not self._backend:
            self.setup()

        if self._backend == "trio":
            await self._trio_lock.acquire()
        elif self._backend == "asyncio":
            await self._anyio_lock.acquire()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        if self._backend == "trio":
            self._trio_lock.release()
        elif self._backend == "asyncio":
            self._anyio_lock.release()


class AsyncThreadLock:
    """
    This is a threading-only lock for no-I/O contexts.

    In the sync case `ThreadLock` provides thread locking.
    In the async case `AsyncThreadLock` is a no-op.
    """

    def __enter__(self) -> AsyncThreadLock:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        pass


class AsyncEvent:
    def __init__(self) -> None:
        self._backend = ""

    def setup(self) -> None:
        """
        Detect if we're running under 'asyncio' or 'trio' and create
        a lock with the correct implementation.
        """
        self._backend = current_async_library()
        if self._backend == "trio":
            self._trio_event = trio.Event()
        elif self._backend == "asyncio":
            self._anyio_event = anyio.Event()

    def set(self) -> None:
        if not self._backend:
            self.setup()

        if self._backend == "trio":
            self._trio_event.set()
        elif self._backend == "asyncio":
            self._anyio_event.set()

    async def wait(self, timeout: float | None = None) -> None:
        if not self._backend:
            self.setup()

        if self._backend == "trio":
            trio_exc_map: ExceptionMapping = {trio.TooSlowError: PoolTimeout}
            timeout_or_inf = float("inf") if timeout is None else timeout
            with map_exceptions(trio_exc_map):
                with trio.fail_after(timeout_or_inf):
                    await self._trio_event.wait()
        elif self._backend == "asyncio":
            anyio_exc_map: ExceptionMapping = {TimeoutError: PoolTimeout}
            with map_exceptions(anyio_exc_map):
                with anyio.fail_after(timeout):
                    await self._anyio_event.wait()


class AsyncSemaphore:
    def __init__(self, bound: int) -> None:
        self._bound = bound
        self._backend = ""

    def setup(self) -> None:
        """
        Detect if we're running under 'asyncio' or 'trio' and create
        a semaphore with the correct implementation.
        """
        self._backend = current_async_library()
        if self._backend == "trio":
            self._trio_semaphore = trio.Semaphore(initial_value=self._bound, max_value=self._bound)
        elif self._backend == "asyncio":
            self._anyio_semaphore = anyio.Semaphore(initial_value=self._bound, max_value=self._bound, fast_acquire=True)

    async def acquire(self) -> None:
        if not self._backend:
            self.setup()

        if self._backend == "trio":
            await self._trio_semaphore.acquire()
        elif self._backend == "asyncio":
            await self._anyio_semaphore.acquire()

    async def release(self) -> None:
        if self._backend == "trio":
            self._trio_semaphore.release()
        elif self._backend == "asyncio":
            self._anyio_semaphore.release()


class AsyncShieldCancellation:
    # For certain portions of our codebase where we're dealing with
    # closing connections during exception handling we want to shield
    # the operation from being cancelled.
    #
    # with AsyncShieldCancellation():
    #     ... # clean-up operations, shielded from cancellation.

    def __init__(self) -> None:
        """
        Detect if we're running under 'asyncio' or 'trio' and create
        a shielded scope with the correct implementation.
        """
        self._backend = current_async_library()

        if self._backend == "trio":
            self._trio_shield = trio.CancelScope(shield=True)
        elif self._backend == "asyncio":
            self._anyio_shield = anyio.CancelScope(shield=True)

    def __enter__(self) -> AsyncShieldCancellation:
        if self._backend == "trio":
            self._trio_shield.__enter__()
        elif self._backend == "asyncio":
            self._anyio_shield.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: types.TracebackType | None = None,
    ) -> None:
        if self._backend == "trio":
            self._trio_shield.__exit__(exc_type, exc_value, traceback)
        elif self._backend == "asyncio":
            self._anyio_shield.__exit__(exc_type, exc_value, traceback)


# Our thread-based synchronization primitives...


class Lock:
    """
    This is a standard lock.

    In the sync case `Lock` provides thread locking.
    In the async case `AsyncLock` provides async locking.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

