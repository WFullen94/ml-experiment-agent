class LoopDetectedError(Exception):
    """Same tool called with identical arguments twice in one trajectory."""


class MaxRevisionsExceededError(Exception):
    """Gate blocked the plan more than MAX_REVISIONS times without approval."""
