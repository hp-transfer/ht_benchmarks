import contextlib as cl


class Task:
    def __init__(self, representation, identifier, evaluate_fn):
        self.representation = representation
        self.identifier = identifier
        self.evaluate = evaluate_fn

    def __repr__(self):
        return f"Task(representation={self.representation})"

    def __hash__(self):
        return hash(str(self.identifier))

    # https://stackoverflow.com/questions/2345944/exclude-objects-field-from-pickling-in-python
    def __getstate__(self):
        state = self.__dict__.copy()
        with cl.suppress(AttributeError):
            del self.evaluate
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.evaluate = None
