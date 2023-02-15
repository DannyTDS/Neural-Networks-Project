def lerp(a, b, t):
    return (1 - t) * a + t * b


def infiniteloop(dataloader):
    while True:
        for *x, y in iter(dataloader):
            yield *x, y