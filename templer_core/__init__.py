"""
Templer Core - Template Intelligence Engine v2.0
A comprehensive library for intelligent template processing without manual preparation.
"""

from .patterns import PATTERNS, PatternMatcher
from .auto_detector import AutoDetector
from .learning_engine import LearningEngine
from .document_parser import DocumentParser, DocumentStructure
from .table_processor import TableProcessor, TableStructure
from .format_preserver import FormatPreserver
from .conditionals import ConditionalHandler
from .validation import ValidationEngine
from .template_analyzer import TemplateAnalyzer, ContentType, AnalysisResult, DynamicValue

__version__ = "2.0.0"
__all__ = [
    "PATTERNS",
    "PatternMatcher",
    "AutoDetector",
    "LearningEngine",
    "DocumentParser",
    "DocumentStructure",
    "TableProcessor",
    "TableStructure",
    "FormatPreserver",
    "ConditionalHandler",
    "ValidationEngine",
    "TemplateAnalyzer",
    "ContentType",
    "AnalysisResult",
]
