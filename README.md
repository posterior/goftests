[![Build Status](https://travis-ci.org/posterior/goftests.svg?branch=master)](https://travis-ci.org/posterior/goftests)
[![Code Quality](http://img.shields.io/scrutinizer/g/posterior/goftests.svg)](https://scrutinizer-ci.com/g/posterior/goftests/code-structure/master/hot-spots)
[![Latest Version](https://badge.fury.io/py/goftests.svg)](https://pypi.python.org/pypi/goftests)

# Goftests

Goftests implements goodness of fit tests for general datatypes.
Goftests is intended for unit testing random samplers that generate arbitrary
plain-old-data, and focuses on robustness rather than statistical efficiency.

## Installing

    pip install goftests

## Using goodness of fit tests

Goftests implements generic statistical tests for Monte Carlo samplers that
generate (sample, probability) pairs.

## Adding new tests

The goodness of fit tests are mostly implemented by reduction to other tests,
eventually reducing to the multinomial goodness of fit test which uses Pearson's &chi;<sup>2</sup> test on each of the multinomial's bins.

![Reductions](/doc/reductions.png)

To implement a new test, you can implement from scratch,
reduce to another test in goftests,
or reduce to standard tests in another package like
[scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html#statistical-functions)
or 
[statsmodels](http://statsmodels.sourceforge.net/stable/stats.html#goodness-of-fit-tests-and-measures).

## License

Copyright (c) 2014 Salesforce.com, Inc. All rights reserved. <br/>
Copyright (c) 2015 Gamelan Labs, Inc. <br/>
Licensed under the Revised BSD License.
See [LICENSE.txt](LICENSE.txt) for details.
