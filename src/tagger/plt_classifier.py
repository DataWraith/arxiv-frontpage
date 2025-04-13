from napkinxc.models import PLT

VECTOR_SIZE = 2**22


def make_classifier():
    plt = PLT("classifier", seed=314159, threads=1)
    return plt
