# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
# Copyright (c) 2015, Gamelan Labs, Inc.
# Copyright (c) 2016, Google, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from itertools import izip
import numpy
import scipy.stats
from numpy import pi
from numpy.testing import rand
from nose import SkipTest
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_greater
from nose.tools import assert_less
from goftests import seed_all
from goftests import get_dim
from goftests import multinomial_goodness_of_fit
from goftests import discrete_goodness_of_fit
from goftests import auto_density_goodness_of_fit
from goftests import mixed_density_goodness_of_fit
from goftests import split_discrete_continuous
from goftests import volume_of_sphere

TEST_FAILURE_RATE = 5e-4


def test_multinomial_goodness_of_fit():
    for dim in range(2, 20):
        yield _test_multinomial_goodness_of_fit, dim


def _test_multinomial_goodness_of_fit(dim):
    seed_all(0)
    sample_count = int(1e5)
    probs = numpy.random.dirichlet([1] * dim)

    counts = numpy.random.multinomial(sample_count, probs)
    p_good = multinomial_goodness_of_fit(probs, counts, sample_count)
    assert_greater(p_good, TEST_FAILURE_RATE)

    unif_counts = numpy.random.multinomial(sample_count, [1. / dim] * dim)
    p_bad = multinomial_goodness_of_fit(probs, unif_counts, sample_count)
    assert_less(p_bad, TEST_FAILURE_RATE)


def test_volume_of_sphere():
    for r in [0.1, 1.0, 10.0]:
        assert_almost_equal(volume_of_sphere(1, r), 2.0 * r)
        assert_almost_equal(volume_of_sphere(2, r), pi * r ** 2)
        assert_almost_equal(volume_of_sphere(3, r), 4 / 3.0 * pi * r ** 3)


split_examples = [
    {'mixed': False, 'discrete': False, 'continuous': []},
    {'mixed': 0, 'discrete': 0, 'continuous': []},
    {'mixed': 'abc', 'discrete': 'abc', 'continuous': []},
    {'mixed': 0.0, 'discrete': None, 'continuous': [0.0]},
    {'mixed': (), 'discrete': (), 'continuous': []},
    {'mixed': [], 'discrete': (), 'continuous': []},
    {'mixed': (0,), 'discrete': (0, ), 'continuous': []},
    {'mixed': [0, ], 'discrete': (0, ), 'continuous': []},
    {'mixed': (0.0, ), 'discrete': (None, ), 'continuous': [0.0]},
    {'mixed': [0.0, ], 'discrete': (None, ), 'continuous': [0.0]},
    {
        'mixed': [True, 1, 'xyz', 3.14, [None, (), ([2.71],)]],
        'discrete': (True, 1, 'xyz', None, (None, (), ((None,),))),
        'continuous': [3.14, 2.71],
    },
    {
        'mixed': numpy.zeros(3),
        'discrete': (None, None, None),
        'continuous': [0.0, 0.0, 0.0],
    },
]


def split_example(i):
    example = split_examples[i]
    discrete, continuous = split_discrete_continuous(example['mixed'])
    assert_equal(discrete, example['discrete'])
    assert_almost_equal(continuous, example['continuous'])


def test_split_continuous_discrete():
    for i in xrange(len(split_examples)):
        yield split_example, i


seed_all(0)
default_params = {
    'bernoulli': [(0.2,)],
    'beta': [
        (0.5, 0.5),
        (0.5, 1.5),
        (0.5, 2.5),
    ],
    'binom': [(40, 0.4)],
    'dirichlet': [
        ([2.0, 2.5],),
        ([2.0, 2.5, 3.0],),
        ([2.0, 2.5, 3.0, 3.5],),
    ],
    'erlang': [(7,)],
    'dlaplace': [(0.8,)],
    'frechet': [tuple(2 * rand(1)) + (0,) + tuple(2 * rand(2))],
    'geom': [(0.1,)],
    'hypergeom': [(40, 14, 24)],
    'logser': [(0.9,)],
    'multivariate_normal': [
        (numpy.ones(1), numpy.eye(1)),
        (numpy.ones(2), numpy.eye(2)),
        (numpy.ones(3), numpy.eye(3)),
    ],
    'nbinom': [(40, 0.4)],
    'ncf': [(27, 27, 0.415784417992)],
    'planck': [(0.51,)],
    'poisson': [(20,)],
    'reciprocal': [tuple(numpy.array([0, 1]) + rand(1)[0])],
    'triang': [tuple(rand(1))],
    'truncnorm': [(0.1, 2.0)],
    'vonmises': [tuple(1.0 + rand(1))],
    'wrapcauchy': [(0.5,)],
    'zipf': [(1.2,)],
}

known_failures = set([
    'alpha',
    'boltzmann',
    'gausshyper',  # very slow
    'ksone',  # ???
    'levy_stable',  # ???
    'randint',  # too sparse
    'rv_continuous',  # abstract
    'rv_discrete',  # abstract
    'zipf',  # bug?
    'invwishart',  # matrix
    'wishart',  # matrix
    'matrix_normal',  # matrix
])


def transform_dirichlet(ps):
    dim = len(ps)
    assert dim > 1
    # return ps[:-1] - ps[-1] * (dim ** 0.5 - 1.0) / (dim - 1.0)
    return ps[:-1]


transforms = {
    'dirichlet': transform_dirichlet,
}


def _test_scipy_stats(name):
    if name in known_failures:
        raise SkipTest('known failure')
    dist = getattr(scipy.stats, name)
    try:
        params = default_params[name]
    except KeyError:
        params = [tuple(1.0 + rand(dist.numargs))]
    for param in params:
        print 'param = {}'.format(param)
        dim = get_dim(dist.rvs(*param, size=2)[0])
        sample_count = 100 + 1000 * dim
        samples = list(dist.rvs(*param, size=sample_count))
        if name in transforms:
            transformed = map(transforms[name], samples)
        else:
            transformed = samples

        if hasattr(dist, 'pmf'):
            probs = [dist.pmf(sample, *param) for sample in samples]
            probs_dict = dict(izip(samples, probs))
            gof = discrete_goodness_of_fit(transformed, probs_dict, plot=True)
        else:
            probs = [dist.pdf(sample, *param) for sample in samples]
            gof = auto_density_goodness_of_fit(transformed, probs, plot=True)
        assert_greater(gof, TEST_FAILURE_RATE)

        gof = mixed_density_goodness_of_fit(transformed, probs, plot=True)
        assert_greater(gof, TEST_FAILURE_RATE)


def test_scipy_stats():
    seed_all(0)
    for name in dir(scipy.stats):
        if hasattr(getattr(scipy.stats, name), 'rvs'):
            yield _test_scipy_stats, name
