"""AI-likeness analysis: windows, providers, stylometric indicators, fusion and bands.

Scores are indicators of AI-like style, never evidence of authorship.
"""

from detection.analyzer import DocumentAnalysis, FusionConfig, SentenceAnalysis, analyze
from detection.classifier import WindowClassifierProvider, ml_inference_classifier, pool_windows
from detection.sapling import QuotaExceededError, SaplingProvider
from detection.stylometric import StylometricProvider
from detection.types import Band, DetectorProvider, SentenceRef, SentenceScore, Window
from detection.windows import build_windows, eligible_words

__all__ = [
    "Band",
    "DetectorProvider",
    "DocumentAnalysis",
    "FusionConfig",
    "QuotaExceededError",
    "SaplingProvider",
    "SentenceAnalysis",
    "SentenceRef",
    "SentenceScore",
    "StylometricProvider",
    "Window",
    "WindowClassifierProvider",
    "analyze",
    "build_windows",
    "eligible_words",
    "ml_inference_classifier",
    "pool_windows",
]
