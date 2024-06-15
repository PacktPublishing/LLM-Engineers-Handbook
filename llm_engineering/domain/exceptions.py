class LLMTwinException(Exception):
    pass


class ImproperlyConfigured(LLMTwinException):
    pass


class JSONDecodeError(LLMTwinException):
    pass


# TODO: Add custom exceptions in the code where needed.
