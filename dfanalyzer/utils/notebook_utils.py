try:
    from IPython import get_ipython

    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        IN_JUPYTER = True
    else:
        IN_JUPYTER = False
except (NameError, ImportError):
    IN_JUPYTER = False
