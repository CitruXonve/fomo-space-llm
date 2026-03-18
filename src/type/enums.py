from enum import Enum


class ContentFormat(Enum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    TXT = "txt"


class CodingMode(str, Enum):
    AUTO = "auto"           # detect from message keywords / KB content
    CODING = "coding"       # emphasize code blocks, syntax, imports
    NON_CODING = "non_coding"  # conversational, plain-text focus


class ContextPersistence(str, Enum):
    EPHEMERAL = "ephemeral"    # use only current call's KB context
    PERSISTENT = "persistent"  # accumulate KB contexts across conversation turns


class ContextScope(str, Enum):
    LOCAL = "local"        # isolated to the current session
    CATEGORY = "category"  # shared across sessions in the same category / user group
    GLOBAL = "global"      # shared across all sessions
