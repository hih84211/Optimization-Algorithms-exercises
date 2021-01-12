from math import sqrt

rho = (3 - sqrt(5)) / 2


def _gss(f, a, c, b, range):
    print('Interval:' + str([min([a, b]), max([a, b])]))
    if abs(b - a) < range:
        return [min([a, b]), max([a, b])]

    d = c + rho * (b - c)
    if f(d) < f(c):
        return _gss(f, c, d, b, range)
    else:
        return _gss(f, d, c, a, range)


def golden_section_search(f, a0, b0, range):
    return _gss(f, a0, (rho * (b0 - a0)), b0, range)


f = lambda x: (x ** 4) - 10 * (x ** 3) + 40 * (x ** 2) - 50 * x
l1 = golden_section_search(f, 0, 2, .005)
print('\n' + 'Final interval:' + str(l1[0]-l1[1]))

