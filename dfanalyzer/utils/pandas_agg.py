def unique_set_flatten_pd(s):
    return set().union(*s.dropna())


def unique_set_pd(s):
    return set(s)


unique_set_flatten_pd.__name__ = "unique"
unique_set_pd.__name__ = "unique"
