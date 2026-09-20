"""The Prompt Library: Amoeba's versioned cognitive family tree.

`model` is the vocabulary, `store` the immutable persistence and governance,
`resolver` the inheritance and composition, `bootstrap` the shipped baseline,
and `cascade` the propagation of a parent change. The RPC surface lives one
level up in `prompt_api`, because it belongs to the Harness rather than to the
library.

The `prompts/` directory ships beside this module on purpose: it is the only
origin of a root, and shipping it inside the package means the baseline travels
with the code and is reviewed in the repository rather than typed into a
running system.
"""

from .model import (MODEL_VARS, PROMPT_MODES, ROOTS, NamespaceError,
                    ProfileRef, compose, parse_ref, validate_namespace)
from .resolver import ResolvedProfile, Resolver, resolve_chain
from .store import APPROVED, STATES, PromptStore, canonical_local

__all__ = [
    "MODEL_VARS", "PROMPT_MODES", "ROOTS", "NamespaceError", "ProfileRef",
    "compose", "parse_ref", "validate_namespace",
    "ResolvedProfile", "Resolver", "resolve_chain",
    "APPROVED", "STATES", "PromptStore", "canonical_local",
]
