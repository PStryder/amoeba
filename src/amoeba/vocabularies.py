"""Which argument values each capability accepts, declared where it is enforced.

A capability that takes `post_type` or `kind` and does not say what the valid
values are is present and undiscoverable. In the first live pressure test Ego
tried `post_type="incident_synthesis"` and `kind="incident_synthesis"`, was
refused both times, and was never told why beyond "unknown post type" -- the
allowed values were attached to the refusal and dropped on the way back.

Every entry here names the very constant its check uses, so the declaration
and the enforcement cannot drift: there is one tuple, and both read it.
`test_every_declared_vocabulary_is_the_one_enforced` holds that at the source.
"""

from __future__ import annotations

from .ego_api import MESSAGE_KINDS, REVIEW_SUBJECTS
from .homeostasis import RECONSTITUTION_MODES
from .id_api import FINDING_KINDS, SEVERITIES, TARGET_ROLES
from .prompt_api import INCARNATION_DETAILS, VERDICTS as PROMPT_VERDICTS
from .promptlib.model import PROMPT_MODES
from .store.board_repo import AUTHOR_KINDS, POST_TYPES
from .store.memory_repo import MEMORY_KINDS
from .store.work_repo import BOARD_ACCESS, WORK_CLASSES

# (verb, argument) -> (the accepted values, where the check lives, its name)
ARGUMENT_VOCABULARIES: dict[tuple[str, str], tuple[tuple[str, ...], str, str]] = {
    ("board_post", "post_type"): (POST_TYPES, "store/board_repo.py", "POST_TYPES"),
    ("board_post", "author_kind"): (AUTHOR_KINDS, "store/board_repo.py", "AUTHOR_KINDS"),
    ("board_read", "post_types"): (POST_TYPES, "store/board_repo.py", "POST_TYPES"),
    ("ego_propose_memory", "kind"): (MEMORY_KINDS, "store/memory_repo.py", "MEMORY_KINDS"),
    ("recall", "kinds"): (MEMORY_KINDS, "store/memory_repo.py", "MEMORY_KINDS"),
    ("ego_request_work", "work_class"): (WORK_CLASSES, "ego_api.py", "WORK_CLASSES"),
    ("ego_request_work", "board_access"): (BOARD_ACCESS, "ego_api.py", "BOARD_ACCESS"),
    ("ego_work_message", "kind"): (MESSAGE_KINDS, "ego_api.py", "MESSAGE_KINDS"),
    ("ego_request_id_review", "subject"): (REVIEW_SUBJECTS, "ego_api.py", "REVIEW_SUBJECTS"),
    ("id_raise_finding", "kind"): (FINDING_KINDS, "id_api.py", "FINDING_KINDS"),
    ("id_escalate_to_operator", "severity"): (SEVERITIES, "id_api.py", "SEVERITIES"),
    ("id_request_rejuvenation", "target_role"): (TARGET_ROLES, "id_api.py", "TARGET_ROLES"),
    ("id_request_rejuvenation", "mode"): (RECONSTITUTION_MODES, "homeostasis.py",
                                          "RECONSTITUTION_MODES"),
    ("id_propose_prompt", "target_role"): (TARGET_ROLES, "id_api.py", "TARGET_ROLES"),
    ("id_evaluate_prompt", "verdict"): (PROMPT_VERDICTS, "prompt_api.py", "VERDICTS"),
    ("id_propose_profile", "prompt_mode"): (PROMPT_MODES, "promptlib/model.py",
                                            "PROMPT_MODES"),
    ("prompt_incarnations", "detail"): (INCARNATION_DETAILS, "prompt_api.py",
                                        "INCARNATION_DETAILS"),
}


def allowed(verb: str, argument: str) -> tuple[str, ...] | None:
    entry = ARGUMENT_VOCABULARIES.get((verb, argument))
    return entry[0] if entry else None
