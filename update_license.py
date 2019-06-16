# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
# Copyright (c) 2015, Gamelan Labs, Inc.
# Copyright (c) 2016, Google, Inc.
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

from __future__ import print_function
import fnmatch
import os
import re
import parsable

dir_blacklist = [
    '.git',
]

FILES = sorted(
    os.path.join(root, filename)
    for root, dirnames, filenames in os.walk('.')
    if not any(d in root.split('/') for d in dir_blacklist)
    for filename in fnmatch.filter(filenames, '*.py')
)

LICENSE = []
with open('LICENSE.txt') as f:
    for line in f:
        LICENSE.append('# {}'.format(line.rstrip()).strip())


@parsable.command
def show():
    '''
    List all files that should have a license.
    '''
    for filename in FILES:
        print(filename)


def read_and_strip_lines(filename):
    lines = []
    with open(filename) as i:
        writing = False
        for line in i:
            line = line.rstrip()
            if not writing and line and not line.startswith('#'):
                writing = True
            if writing:
                lines.append(line)
    return lines


def write_lines(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            print(line, file=f)


@parsable.command
def strip():
    '''
    Strip headers from all files.
    '''
    for filename in FILES:
        lines = read_and_strip_lines(filename)
        write_lines(lines, filename)


@parsable.command
def update():
    '''
    Update headers on all files to match LICNESE.txt.
    '''
    for filename in FILES:
        extension = re.search(r'\.[^.]*$', filename).group()
        lines = read_and_strip_lines(filename)
        if lines and lines[0]:
            if extension == '.py' and lines[0].startswith('class '):
                lines = [''] + lines  # pep8 compliance
            write_lines(LICENSE + [''] + lines, filename)


if __name__ == '__main__':
    parsable.dispatch()
