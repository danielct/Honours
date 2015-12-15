from copy import copy


class dec(object):
    def __init__(self, *args):
        self.args = copy(args)

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            if len(self.args) == 0:
                newArgs = [arg.to_base_units().magnitude for arg in args]
                newkwargs = [arg.to_base_units().magnitude
                             for arg in kwargs.values()]
            else:
                if len(kwargs) != 0:
                    raise ValueError("Can't specify which arguments to modify\
                                     and also use kwargs")
                newkwargs = kwargs
                newArgs = [arg.to_base_units().magnitude if i in self.args
                           else arg for (i, arg) in enumerate(args)]
            return f(*newArgs) if len(kwargs) == 0 else f(*newArgs, **newkwargs)
        return wrapped_f
