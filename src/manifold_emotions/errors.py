"""Domain-specific exception hierarchy."""


class ManifoldEmotionsError(Exception):
    """Root of the project's exception tree."""


class ConfigError(ManifoldEmotionsError):
    """Raised when config.yaml or environment variables are missing or invalid."""


class CorpusError(ManifoldEmotionsError):
    """Raised when story generation fails or produces invalid output."""


class CaptureError(ManifoldEmotionsError):
    """Raised when the vLLM capture pipeline returns an error or invalid payload."""


class JudgeError(ManifoldEmotionsError):
    """Raised when the LLM judge call fails or returns unparseable output."""
