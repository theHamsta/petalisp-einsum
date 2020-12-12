#! /bin/sh
#
# test.sh
# Copyright (C) 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.
#

PACKAGE_NAME=petalisp-einsum

sbcl --eval '(setf *debugger-hook* (lambda (c h) (declare (ignore c h)) (uiop:quit -1)))' \
     --eval "(progn (ql:quickload :$PACKAGE_NAME)(ql:quickload :$PACKAGE_NAME/tests)(asdf:test-system :$PACKAGE_NAME/tests) (exit))"
