"""
exceptions.py
----------------

Wrap exceptions.
"""


class ExceptionWrapper(object):
    """
    Create a dummy object which will raise an exception when attributes
    are accessed (i.e. when used as a module) or when called (i.e.
    when used like a function)

    For soft dependencies we want to survive failing to import but
    we would like to raise an appropriate error when the functionality is
    actually requested so the user gets an easily debuggable message.
    """

    def __init__(self, exception):
        self.exception = exception

    def __getattribute__(self, *args, **kwargs):
        # will raise when this object is accessed like an object
        # if it's asking for our class type return None
        # this allows isinstance() checks to not re-raise
        if args[0] == "__class__":
            return None.__class__
        # otherwise raise our original exception
        raise super(ExceptionWrapper, self).__getattribute__("exception")

    def __call__(self, *args, **kwargs):
        # will raise when this object is called like a function
        raise super(ExceptionWrapper, self).__getattribute__("exception")
