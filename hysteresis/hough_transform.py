import numpy as np

def _prob_hough_line(img, threshold, line_length, line_gap, theta, seed=None):
    """Return lines from a progressive probabilistic line Hough transform.
    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int
        Threshold
    line_length : int
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggressively.
    theta : 1D ndarray, dtype=double
        Angles at which to compute the transform, in radians.
    seed : {None, int, `numpy.random.Generator`, optional}
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
        Seed to initialize the random number generator.
    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y0)),
      indicating line start and end.
    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.
    """
    height = img.shape[0]
    width = img.shape[1]

    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[ndim=2, dtype=cnp.uint8_t] mask = np.zeros((height, width), dtype=np.uint8)
    cdef Py_ssize_t *line_end = \
        <Py_ssize_t *>PyMem_Malloc(4 * sizeof(Py_ssize_t))
    if not line_end:
        raise MemoryError('could not allocate line_end')
    cdef Py_ssize_t max_distance, offset, num_indexes, index
    cdef double a, b
    cdef Py_ssize_t nidxs, i, j, k, x, y, px, py, accum_idx, max_theta
    cdef Py_ssize_t xflag, x0, y0, dx0, dy0, dx, dy, gap, x1, y1, count
    cdef cnp.int64_t value, max_value,
    cdef int shift = 16
    cdef int good_line
    cdef Py_ssize_t nlines = 0
    cdef Py_ssize_t lines_max = 2 ** 15  # maximum line number cutoff
    cdef cnp.intp_t[:, :, ::1] lines = np.zeros((lines_max, 2, 2),
                                                dtype=np.intp)
    max_distance = 2 * <Py_ssize_t>ceil((sqrt(img.shape[0] * img.shape[0] +
                                              img.shape[1] * img.shape[1])))
    cdef cnp.int64_t[:, ::1] accum = np.zeros((max_distance, theta.shape[0]),
                                              dtype=np.int64)
    offset = max_distance / 2
    cdef Py_ssize_t nthetas = theta.shape[0]

    # compute sine and cosine of angles
    cdef cnp.double_t[::1] ctheta = np.cos(theta)
    cdef cnp.double_t[::1] stheta = np.sin(theta)

    # find the nonzero indexes
    cdef cnp.intp_t[:] y_idxs, x_idxs
    y_idxs, x_idxs = np.nonzero(img)

    # mask all non-zero indexes
    mask[y_idxs, x_idxs] = 1

    count = len(x_idxs)
    random_state = np.random.default_rng(seed)
    random_ = np.arange(count, dtype=np.intp)
    random_state.shuffle(random_)
    cdef cnp.intp_t[::1] random = random_

    with nogil:
        while count > 0:
            count -= 1
            # select random non-zero point
            index = random[count]
            x = x_idxs[index]
            y = y_idxs[index]

            # if previously eliminated, skip
            if not mask[y, x]:
                continue

            value = 0
            max_value = threshold - 1
            max_theta = -1

            # apply hough transform on point
            for j in range(nthetas):
                accum_idx = round((ctheta[j] * x + stheta[j] * y)) + offset
                accum[accum_idx, j] += 1
                value = accum[accum_idx, j]
                if value > max_value:
                    max_value = value
                    max_theta = j
            if max_value < threshold:
                continue

            # from the random point walk in opposite directions and find line
            # beginning and end
            a = -stheta[max_theta]
            b = ctheta[max_theta]
            x0 = x
            y0 = y
            # calculate gradient of walks using fixed point math
            xflag = fabs(a) > fabs(b)
            if xflag:
                if a > 0:
                    dx0 = 1
                else:
                    dx0 = -1
                dy0 = round(b * (1 << shift) / fabs(a))
                y0 = (y0 << shift) + (1 << (shift - 1))
            else:
                if b > 0:
                    dy0 = 1
                else:
                    dy0 = -1
                dx0 = round(a * (1 << shift) / fabs(b))
                x0 = (x0 << shift) + (1 << (shift - 1))

            # pass 1: walk the line, merging lines less than specified gap
            # length
            for k in range(2):
                gap = 0
                px = x0
                py = y0
                dx = dx0
                dy = dy0
                if k > 0:
                    dx = -dx
                    dy = -dy
                while 1:
                    if xflag:
                        x1 = px
                        y1 = py >> shift
                    else:
                        x1 = px >> shift
                        y1 = py
                    # check when line exits image boundary
                    if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
                        break
                    gap += 1
                    # if non-zero point found, continue the line
                    if mask[y1, x1]:
                        gap = 0
                        line_end[2*k] = x1
                        line_end[2*k + 1] = y1
                    # if gap to this point was too large, end the line
                    elif gap > line_gap:
                        break
                    px += dx
                    py += dy

            # confirm line length is sufficient
            good_line = (abs(line_end[3] - line_end[1]) >= line_length or
                         abs(line_end[2] - line_end[0]) >= line_length)

            # pass 2: walk the line again and reset accumulator and mask
            for k in range(2):
                px = x0
                py = y0
                dx = dx0
                dy = dy0
                if k > 0:
                    dx = -dx
                    dy = -dy
                while 1:
                    if xflag:
                        x1 = px
                        y1 = py >> shift
                    else:
                        x1 = px >> shift
                        y1 = py
                    # if non-zero point found, continue the line
                    if mask[y1, x1]:
                        if good_line:
                            accum_idx = round(
                                (ctheta[j] * x1 + stheta[j] * y1)) + offset
                            accum[accum_idx, max_theta] -= 1
                            mask[y1, x1] = 0
                    # exit when the point is the line end
                    if x1 == line_end[2*k] and y1 == line_end[2*k + 1]:
                        break
                    px += dx
                    py += dy

            # add line to the result
            if good_line:
                lines[nlines, 0, 0] = line_end[0]
                lines[nlines, 0, 1] = line_end[1]
                lines[nlines, 1, 0] = line_end[2]
                lines[nlines, 1, 1] = line_end[3]
                nlines += 1
                if nlines >= lines_max:
                    break

    PyMem_Free(line_end)
    return [((line[0, 0], line[0, 1]), (line[1, 0], line[1, 1]))
            for line in lines[:nlines]]                               

def probabilistic_hough_line(image, threshold=10, line_length=50, line_gap=10,
                             theta=None, seed=None):
    """Return lines from a progressive probabilistic line Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    threshold : int, optional
        Threshold
    line_length : int, optional
        Minimum accepted length of detected lines.
        Increase the parameter to extract longer lines.
    line_gap : int, optional
        Maximum gap between pixels to still form a line.
        Increase the parameter to merge broken lines more aggressively.
    theta : 1D ndarray, dtype=double, optional
        Angles at which to compute the transform, in radians.
        Defaults to a vector of 180 angles evenly spaced in the
        range [-pi/2, pi/2).
    seed : int, optional
        Seed to initialize the random number generator.

    Returns
    -------
    lines : list
      List of lines identified, lines in format ((x0, y0), (x1, y1)),
      indicating line start and end.

    References
    ----------
    .. [1] C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
           Hough transform for line detection", in IEEE Computer Society
           Conference on Computer Vision and Pattern Recognition, 1999.
    """

    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')

    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)

    return _prob_hough_line(image, threshold=threshold, line_length=line_length,
                            line_gap=line_gap, theta=theta, seed=seed)