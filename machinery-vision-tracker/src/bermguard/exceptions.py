"""Expected errors exposed by the CLI without a traceback."""


class BermGuardError(Exception):
    """Base class for controlled pipeline failures."""


class ConfigurationError(BermGuardError):
    """Configuration is absent, malformed or violates the contract."""


class InputDiscoveryError(BermGuardError):
    """The input directory does not contain a usable video corpus."""


class OutputExistsError(BermGuardError):
    """The requested destination is not an empty, isolated workspace."""


class VideoProcessingError(BermGuardError):
    """A video cannot be decoded, encoded or validated."""
