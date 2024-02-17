# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
# Copyright (c) 2015, Gamelan Labs, Inc.
# Copyright (c) 2016, Google, Inc.
# Copyright (c) 2016, Gamelan Labs, Inc.
# Copyright (c) 2019, Gamalon, Inc.
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
from __future__ import division
try:
    from itertools import izip as zip
except ImportError:
    pass
from itertools import product
import random
from unittest import skip
from unittest import TestCase

import numpy
import scipy.stats
from numpy import pi
from numpy.random import rand

from goftests import get_dim
from goftests import multinomial_goodness_of_fit
from goftests import discrete_goodness_of_fit
from goftests import auto_density_goodness_of_fit
from goftests import mixed_density_goodness_of_fit
from goftests import split_discrete_continuous
from goftests import volume_of_sphere
from goftests import chi2sf

NUM_BASE_SAMPLES = 250

NUM_SAMPLES_SCALE = 1000

TEST_FAILURE_RATE = 5e-4


class TestMultinomialGoodnessOfFit(TestCase):

    def test_multinomial_goodness_of_fit(self):
        random.seed(0)
        numpy.random.seed(0)
        for dim in range(2, 20):
            sample_count = int(1e5)
            probs = numpy.random.dirichlet([1] * dim)

            counts = numpy.random.multinomial(sample_count, probs)
            p_good = multinomial_goodness_of_fit(probs, counts, sample_count)
            self.assertGreater(p_good, TEST_FAILURE_RATE)

            unif = [1 / dim] * dim
            unif_counts = numpy.random.multinomial(sample_count, unif)
            p_bad = multinomial_goodness_of_fit(probs, unif_counts,
                                                sample_count)
            self.assertLess(p_bad, TEST_FAILURE_RATE)


class TestVolumeOfSphere(TestCase):

    def test_volume_of_sphere(self):
        for r in [0.1, 1.0, 10.0]:
            self.assertAlmostEqual(volume_of_sphere(1, r), 2 * r)
            self.assertAlmostEqual(volume_of_sphere(2, r), pi * r ** 2)
            self.assertAlmostEqual(volume_of_sphere(3, r), 4 / 3 * pi * r ** 3)


SPLIT_EXAMPLES = [
    (False, False, []),
    (0, 0, []),
    ('abc', 'abc', []),
    (0.0, None, [0.0]),
    ((), (), []),
    ([], (), []),
    ((0, ), (0, ), []),
    ([0], (0, ), []),
    ((0.0, ), (None, ), [0.0]),
    ([0.0], (None, ), [0.0]),
    ([True, 1, 'xyz', 3.14, [None, (), ([2.71],)]],
     (True, 1, 'xyz', None, (None, (), ((None,),))),
     [3.14, 2.71]),
    (numpy.zeros(3), (None, None, None), [0.0, 0.0, 0.0]),
]


class TestSplitDiscreteContinuous(TestCase):

    def test_split_continuous_discrete(self):
        for mixed, discrete, continuous in SPLIT_EXAMPLES:
            d, c = split_discrete_continuous(mixed)
            self.assertEqual(d, discrete)
            self.assertAlmostEqual(c, continuous)


class TestChi2CDF(TestCase):

    def test_chi2cdf(self):
        xlist = numpy.linspace(0, 100, 500)
        slist = numpy.arange(1, 41, 1.5)
        for s, x in product(slist, xlist):
            self.assertAlmostEqual(scipy.stats.chi2.sf(x, s), chi2sf(x, s))


class DistributionTestBase(object):
    """Abstract base class for probability distribution unit tests.

    This class supplies two test methods, :meth:`.test_goodness_of_fit`
    and :meth:`.test_mixed_density_goodness_of_fit` for testing the
    goodness of fit functions.

    Subclasses must override and implement one class attribute and two
    instance methods. The :attr:`.dist` class attribute must be set to
    one of SciPy probability distribution constructors in
    :mod:`scipy.stats`. The :meth:`.goodness_of_fit` method must return
    the result of calling one of the goodness of fit functions being
    tested. The :meth:`.probabilites` method must return an object
    representing the probabilities for each sample; the output depends
    on the format of the inputs to the :meth:`.goodness_of_fit` method.

    Subclasses may also set the :attr:`.params` attribute, which is a
    list of tuples that will be provided as arguments to the underlying
    SciPy distribution constructor as specified in :attr:`.dist`. If not
    specified, random arguments will be provided.

    If samples drawn from :attr:`.dist` must be modified in some way
    before the PDF or PMF can be computed, then subclasses may override
    the :meth:`._sample_postprocessing` method.

    """

    #: The SciPy distribution constructor to test.
    dist = None

    #: An optional list of arguments to the distribution constructor.
    #:
    #: Each tuple in this list will be provided as the positional
    #: arguments to the distribution constructor specified in
    #: :attr:`.dist`. If not specified, random arguments will be
    #: provided.
    params = None

    def setUp(self):
        random.seed(0)
        numpy.random.seed(0)

    def _sample_postprocessing(self, sample):
        """Modify a sample drawn from the distribution.

        This method returns a modified version of `sample`, but that
        modification may be arbitrary. This modified sample is the one
        for which the PDF and the goodness-of-fit are computed.

        By default, this is a no-op, but subclasses may wish to override
        this method to modify sample in some way.

        """
        return sample

    def dist_params(self):
        # If there are no parameters, then we provide a random one.
        if self.params is None:
            params = [tuple(1 + rand(self.dist.numargs))]
        else:
            params = self.params
        return params

    def test_mixed_density_goodness_of_fit(self):
        for param in self.dist_params():
            dim = get_dim(self.dist.rvs(*param, size=2)[0])
            sample_count = NUM_BASE_SAMPLES + NUM_SAMPLES_SCALE * dim
            samples = self.dist.rvs(*param, size=sample_count)
            samples = list(map(self._sample_postprocessing, samples))
            probabilities = [self.pdf(sample, *param) for sample in samples]
            gof = mixed_density_goodness_of_fit(samples, probabilities)
            self.assertGreater(gof, TEST_FAILURE_RATE)

    def test_good_fit(self):
        for param in self.dist_params():
            dim = get_dim(self.dist.rvs(*param, size=2)[0])
            sample_count = NUM_BASE_SAMPLES + NUM_SAMPLES_SCALE * dim
            samples = self.dist.rvs(*param, size=sample_count)
            samples = list(map(self._sample_postprocessing, samples))
            probabilities = [self.pdf(sample, *param) for sample in samples]
            gof = self.goodness_of_fit(samples, probabilities)
            self.assertGreater(gof, TEST_FAILURE_RATE)

    def goodness_of_fit(self, samples, probabilities):
        raise NotImplementedError


class ContinuousTestBase(DistributionTestBase):
    """Abstract base class for testing continuous probability distributions.

    Concrete subclasses must set the :attr:`.dist` attribute to be the
    constructor for a continuous probability distribution.

    """

    def goodness_of_fit(self, samples, probabilities):
        gof = auto_density_goodness_of_fit(samples, probabilities)
        return gof

    def pdf(self, *args, **kw):
        return self.dist.pdf(*args, **kw)


class DiscreteTestBase(DistributionTestBase):
    """Abstract base class for testing discrete probability distributions.

    Concrete subclasses must set the :attr:`.dist` attribute to be the
    constructor for a discrete probability distribution.

    """

    def goodness_of_fit(self, samples, probabilities):
        probs_dict = dict(zip(samples, probabilities))
        gof = discrete_goodness_of_fit(samples, probs_dict)
        return gof

    def pdf(self, *args, **kw):
        return self.dist.pmf(*args, **kw)


#
# Multivariate probability distributions.
#

class TestMultivariateNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.multivariate_normal

    params = [
        (numpy.ones(1), numpy.eye(1)),
        (numpy.ones(2), numpy.eye(2)),
        (numpy.ones(3), numpy.eye(3)),
    ]


class TestDirichlet(ContinuousTestBase, TestCase):

    dist = scipy.stats.dirichlet

    params = [
        ([2.0, 2.5],),
        ([2.0, 2.5, 3.0],),
        ([2.0, 2.5, 3.0, 3.5],),
    ]

    def _sample_postprocessing(self, value):
        """Project onto all but the last dimension."""
        return value[:-1]


#
# Discrete probability distributions.
#

class TestBernoulli(DiscreteTestBase, TestCase):

    dist = scipy.stats.bernoulli

    params = [(0.2, )]


class TestBinomial(DiscreteTestBase, TestCase):

    dist = scipy.stats.binom

    params = [(40, 0.4)]


@skip('')
class TestBoltzmann(DiscreteTestBase, TestCase):

    dist = scipy.stats.boltzmann


class TestDiscreteLaplacian(DiscreteTestBase, TestCase):

    dist = scipy.stats.dlaplace

    params = [(0.8, )]


class TestGeometric(DiscreteTestBase, TestCase):

    dist = scipy.stats.geom

    params = [(0.1, )]


class TestHypergeometric(DiscreteTestBase, TestCase):

    dist = scipy.stats.hypergeom

    params = [(40, 14, 24)]


class TestLogSeries(DiscreteTestBase, TestCase):

    dist = scipy.stats.logser

    params = [(0.9, )]


class TestNegativeBinomial(DiscreteTestBase, TestCase):

    dist = scipy.stats.nbinom

    params = [(40, 0.4)]


class TestPlanck(DiscreteTestBase, TestCase):

    dist = scipy.stats.planck

    params = [(0.51, )]


class TestPoisson(DiscreteTestBase, TestCase):

    dist = scipy.stats.poisson

    params = [(20, )]


@skip('too sparse')
class TestRandInt(DiscreteTestBase, TestCase):

    dist = scipy.stats.randint


class TestSkellam(DiscreteTestBase, TestCase):

    dist = scipy.stats.skellam


@skip('bug?')
class TestZipf(DiscreteTestBase, TestCase):

    dist = scipy.stats.zipf

    params = [(1.2, )]

#
# Continuous probability distributions.
#


@skip('')
class TestAlpha(ContinuousTestBase, TestCase):

    dist = scipy.stats.alpha


class TestAnglit(ContinuousTestBase, TestCase):

    dist = scipy.stats.anglit


class TestArcsine(ContinuousTestBase, TestCase):

    dist = scipy.stats.arcsine


class TestBeta(ContinuousTestBase, TestCase):

    dist = scipy.stats.beta

    params = [
        (0.5, 0.5),
        (0.5, 1.5),
        (0.5, 2.5),
    ]


class TestBetaPrime(ContinuousTestBase, TestCase):

    dist = scipy.stats.betaprime


class TestBradford(ContinuousTestBase, TestCase):

    dist = scipy.stats.bradford


class TestBurr(ContinuousTestBase, TestCase):

    dist = scipy.stats.burr


class TestCauchy(ContinuousTestBase, TestCase):

    dist = scipy.stats.cauchy


class TestChi(ContinuousTestBase, TestCase):

    dist = scipy.stats.chi


class TestChiSquared(ContinuousTestBase, TestCase):

    dist = scipy.stats.chi2


class TestCosine(ContinuousTestBase, TestCase):

    dist = scipy.stats.cosine


class TestDoubleGamma(ContinuousTestBase, TestCase):

    dist = scipy.stats.dgamma


class TestDoubleWeibull(ContinuousTestBase, TestCase):

    dist = scipy.stats.dweibull


class TestErlang(ContinuousTestBase, TestCase):

    dist = scipy.stats.erlang

    params = [(7, )]


class TestExponential(ContinuousTestBase, TestCase):

    dist = scipy.stats.expon

    params = [(7, )]


class TestExponentiallyModifiedNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.exponnorm


class TestExponentiatedWeibull(ContinuousTestBase, TestCase):

    dist = scipy.stats.exponweib


class TestExponentialPower(ContinuousTestBase, TestCase):

    dist = scipy.stats.exponpow


class TestF(ContinuousTestBase, TestCase):

    dist = scipy.stats.f


class TestFatigueLife(ContinuousTestBase, TestCase):

    dist = scipy.stats.fatiguelife


class TestFisk(ContinuousTestBase, TestCase):

    dist = scipy.stats.fisk


class TestFoldedCauchy(ContinuousTestBase, TestCase):

    dist = scipy.stats.foldcauchy


class TestFoldedNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.foldnorm


class TestGeneralizedLogistic(ContinuousTestBase, TestCase):

    dist = scipy.stats.genlogistic


class TestGeneralizedNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.gennorm


class TestGeneralizedPareto(ContinuousTestBase, TestCase):

    dist = scipy.stats.genpareto


class TestGeneralizedExponential(ContinuousTestBase, TestCase):

    dist = scipy.stats.genexpon


class TestGeneralizedExtreme(ContinuousTestBase, TestCase):

    dist = scipy.stats.genextreme


@skip('very slow')
class TestGaussHypergeometric(ContinuousTestBase, TestCase):

    dist = scipy.stats.gausshyper


class TestGamma(ContinuousTestBase, TestCase):

    dist = scipy.stats.gamma


class TestGeneralizedGamma(ContinuousTestBase, TestCase):

    dist = scipy.stats.gengamma


class TestGeneralizedHalfLogistic(ContinuousTestBase, TestCase):

    dist = scipy.stats.genhalflogistic


class TestGibrat(ContinuousTestBase, TestCase):

    dist = scipy.stats.gibrat


class TestGompertz(ContinuousTestBase, TestCase):

    dist = scipy.stats.gompertz


class TestGumbelRight(ContinuousTestBase, TestCase):

    dist = scipy.stats.gumbel_r


class TestGumbelLeft(ContinuousTestBase, TestCase):

    dist = scipy.stats.gumbel_l


class TestHalfCauchy(ContinuousTestBase, TestCase):

    dist = scipy.stats.halfcauchy


class TestHalfLogistic(ContinuousTestBase, TestCase):

    dist = scipy.stats.halflogistic


class TestHalfNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.halfnorm


class TestHalfGeneralizedNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.halfgennorm


class TestHyperbolicSecant(ContinuousTestBase, TestCase):

    dist = scipy.stats.hypsecant


class TestInverseGamma(ContinuousTestBase, TestCase):

    dist = scipy.stats.invgamma


class TestInverseGauss(ContinuousTestBase, TestCase):

    dist = scipy.stats.invgauss


class TestInverseWeibull(ContinuousTestBase, TestCase):

    dist = scipy.stats.invweibull


class TestJohnsonSB(ContinuousTestBase, TestCase):

    dist = scipy.stats.johnsonsb


class TestJohnsonSU(ContinuousTestBase, TestCase):

    dist = scipy.stats.johnsonsu


@skip('???')
class TestKolmogorovSmirnovOneSided(ContinuousTestBase, TestCase):

    dist = scipy.stats.ksone


class TestKolmogorovSmirnovTwoSided(ContinuousTestBase, TestCase):

    dist = scipy.stats.kstwobign


class TestLaplace(ContinuousTestBase, TestCase):

    dist = scipy.stats.laplace


class TestLevy(ContinuousTestBase, TestCase):

    dist = scipy.stats.levy


class TestLeftSkewedLevy(ContinuousTestBase, TestCase):

    dist = scipy.stats.levy_l


@skip('???')
class TestLevyStable(ContinuousTestBase, TestCase):

    dist = scipy.stats.levy_stable


class TestLogistic(ContinuousTestBase, TestCase):

    dist = scipy.stats.logistic


class TestLogGamma(ContinuousTestBase, TestCase):

    dist = scipy.stats.loggamma


class TestLogLaplace(ContinuousTestBase, TestCase):

    dist = scipy.stats.loglaplace


class TestLogNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.lognorm


class TestLomax(ContinuousTestBase, TestCase):

    dist = scipy.stats.lomax


class TestMaxwell(ContinuousTestBase, TestCase):

    dist = scipy.stats.maxwell


class TestMielke(ContinuousTestBase, TestCase):

    dist = scipy.stats.mielke


class TestNakagami(ContinuousTestBase, TestCase):

    dist = scipy.stats.nakagami


class TestNonCentralChiSquared(ContinuousTestBase, TestCase):

    dist = scipy.stats.ncx2


class TestNonCentralF(ContinuousTestBase, TestCase):

    dist = scipy.stats.ncf

    params = [(27, 27, 0.415784417992)]


class TestNonCentralT(ContinuousTestBase, TestCase):

    dist = scipy.stats.nct


class TestNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.norm


class TestPareto(ContinuousTestBase, TestCase):

    dist = scipy.stats.pareto


class TestPearson3(ContinuousTestBase, TestCase):

    dist = scipy.stats.pearson3


class TestPowerLaw(ContinuousTestBase, TestCase):

    dist = scipy.stats.powerlaw


class TestPowerNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.powernorm


class TestRDistributed(ContinuousTestBase, TestCase):

    dist = scipy.stats.rdist


class TestReciprocal(ContinuousTestBase, TestCase):

    dist = scipy.stats.reciprocal

    params = [tuple(numpy.array([0, 1]) + rand(1)[0])]


class TestRayleigh(ContinuousTestBase, TestCase):

    dist = scipy.stats.rayleigh


class TestRice(ContinuousTestBase, TestCase):

    dist = scipy.stats.rice


class TestReciprocalInverseGaussian(ContinuousTestBase, TestCase):

    dist = scipy.stats.recipinvgauss


class TestSemicircular(ContinuousTestBase, TestCase):

    dist = scipy.stats.semicircular


class TestT(ContinuousTestBase, TestCase):

    dist = scipy.stats.t


class TestTrapz(ContinuousTestBase, TestCase):

    dist = scipy.stats.trapz

    params = [(1 / 3, 2 / 3)]


class TestTriangular(ContinuousTestBase, TestCase):

    dist = scipy.stats.triang

    params = [tuple(rand(1))]


class TestTruncatedExponential(ContinuousTestBase, TestCase):

    dist = scipy.stats.truncexpon


class TestTruncatedNormal(ContinuousTestBase, TestCase):

    dist = scipy.stats.truncnorm

    params = [(0.1, 2.0)]


class TestTukeyLambda(ContinuousTestBase, TestCase):

    dist = scipy.stats.tukeylambda


class TestUniform(ContinuousTestBase, TestCase):

    dist = scipy.stats.uniform


class TestVonMises(ContinuousTestBase, TestCase):

    dist = scipy.stats.vonmises

    params = [tuple(1.0 + rand(1))]


class TestVonMisesLine(ContinuousTestBase, TestCase):

    dist = scipy.stats.vonmises_line


class TestWald(ContinuousTestBase, TestCase):

    dist = scipy.stats.wald


class TestWeibullMin(ContinuousTestBase, TestCase):

    # This also covers what was previously available as `frechet_r`.
    dist = scipy.stats.weibull_min


class TestWeibullMax(ContinuousTestBase, TestCase):

    # This also covers what was previously available as `frechet_l`.
    dist = scipy.stats.weibull_max


class TestWrappedCauchy(ContinuousTestBase, TestCase):

    dist = scipy.stats.wrapcauchy

    params = [(0.5,)]
