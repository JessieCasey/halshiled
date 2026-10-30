"""halshield: composable hallucination-prevention layers for LangGraph nodes."""

from .contract import ABSTAIN_MSG, Node, State
from .errors import HalshieldError, InvalidStateError, LLMRequestError
from .interfaces import Layer, LLMClient, Retriever, RevisionLayer
from .layers import Abstain, CoVe, Critic, RAGGrounding, SelfConsistency
from .llm import OpenAILLM, get_llm, set_llm
from .presets import Protection
from .retrieval import HybridRetriever, LangChainRetriever, QueryExpansionRetriever, QueryRewriter
from .run_context import RunContext, UsageTotals
from .shield import Shield
from .utils import cited

__version__ = "0.1.0"

__all__ = [
    "ABSTAIN_MSG",
    "Abstain",
    "CoVe",
    "Critic",
    "HalshieldError",
    "HybridRetriever",
    "InvalidStateError",
    "LLMClient",
    "LLMRequestError",
    "LangChainRetriever",
    "Layer",
    "Node",
    "OpenAILLM",
    "Protection",
    "QueryExpansionRetriever",
    "QueryRewriter",
    "RAGGrounding",
    "Retriever",
    "RevisionLayer",
    "RunContext",
    "SelfConsistency",
    "Shield",
    "State",
    "UsageTotals",
    "cited",
    "get_llm",
    "set_llm",
]
