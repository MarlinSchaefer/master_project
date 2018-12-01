def g(arg, **kwargs):
    val = kwargs[arg]
    del kwargs[arg]
    return(val)

def f(**kwargs):
    print("Kwargs before:")
    print(kwargs)
    print
    for key, val in kwargs.items():
        print(g(key, **kwargs))
    print
    print("Kwargs after:")
    print(kwargs)

f(test1=1, test2=2, test3=3)
