"""Failure boundaries for the chat workflow; failures never supply substitute results."""


class WorkflowExecutionError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)
