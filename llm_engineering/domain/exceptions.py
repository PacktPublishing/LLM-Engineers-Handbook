class LLMTwinException(Exception):
    pass


class ImproperlyConfigured(LLMTwinException):
    pass


class JSONDecodeError(LLMTwinException):
    pass
