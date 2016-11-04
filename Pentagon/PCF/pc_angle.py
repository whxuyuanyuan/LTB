def absolute(a):
    if (a>=0 and a<=72):
        return a
    if (a<0):
        return absolute(a + 72)
    if (a>72):
        return absolute(a - 72)


def pairCorrelationFunction_2D(x, y, a, row, col, rMax, dr, da):
    """Compute the two-dimensional pair correlation function, also known
    as the radial distribution function, for a set of circular particles
    contained in a square region of a plane.  This simple function finds
    reference particles such that a circle of radius rMax drawn around the
    particle will fit entirely within the square, eliminating the need to
    compensate for edge effects.  If no such particles exist, an error is
    returned. Try a smaller rMax...or write some code to handle edge effects! ;)
    Arguments:
        x               an array of x positions of centers of particles
        y               an array of y positions of centers of particles
        S               length of each side of the square region of the plane
        rMax            outer diameter of largest annulus
        dr              increment for increasing radius of annulus
    Returns a tuple: (g, radii, interior_indices)
        g(r)            a numpy array containing the correlation function g(r)
        radii           a numpy array containing the radii of the
                        annuli used to compute g(r)
        reference_indices   indices of reference particles
    """
    from numpy import zeros, sqrt, where, pi, mean, arange, histogram
    # Number of particles in ring/area of ring/number of reference particles/number density
    # area of ring = pi*(r_outer**2 - r_inner**2)

    # Find particles which are close enough to the box center that a circle of radius
    # rMax will not cross any edge of the box
    bools1 = x > rMax
    bools2 = x < (col - rMax)
    bools3 = y > rMax
    bools4 = y < (row - rMax)
    interior_indices, = where(bools1 * bools2 * bools3 * bools4)
    num_interior_particles = len(interior_indices)

    if num_interior_particles < 1:
        raise  RuntimeError ("No particles found for which a circle of radius rMax\
                will lie entirely within a square of side length S.  Decrease rMax\
                or increase the size of the square.")

    edges = arange(0., rMax + 1.1 * dr, dr)
    num_increments = len(edges) - 1
    num_angle = 72 / da
    for i in range(len(a)):
        a[i] = absolute(a[i])

    g = zeros([num_angle, num_interior_particles, num_increments])
    radii = zeros(num_increments)
    numberDensity = float(len(x)) / float(row * col)

    # Compute pairwise correlation for each interior particle
    for angle in range(0, 72-da, da):
        bools5 = a >= angle
        bools6 = a <= angle + da
        angle_indices, = where(bools5 * bools6)
        x_angle = x[angle_indices]
        y_angle = y[angle_indices]
        for p in range(num_interior_particles):
            index = interior_indices[p]
            d = sqrt((x[index] - x_angle)**2 + (y[index] - y_angle)**2)
            if (index < len(x_angle)):
                d[index] = 2 * rMax
            (result, bins) = histogram(d, bins=edges, normed=False)
            g[angle, p, :] = result/numberDensity

    # Average g(r) for all interior particles and compute radii
    g_average = zeros([num_angle, num_increments])
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        for angle in range(0, 72-da, da):
            g_average[angle, i] = mean(g[angle, :, i]) / (pi * (rOuter**2 - rInner**2))

    return (g_average, radii, interior_indices)
####
