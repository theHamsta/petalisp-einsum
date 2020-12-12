(defpackage petalisp-einsum/tests/main
  (:use :cl
        :petalisp-einsum
        :petalisp
        :rove)
  (:import-from :petalisp.test-suite
                :approximately-equal)
  (:import-from :petalisp.examples.linear-algebra
                :matmul))
(in-package :petalisp-einsum/tests/main)

(defparameter *mat-a* (aops:rand '(4 3)))
(defparameter *mat-b* (aops:rand '(3 5)))
(defparameter *vec-a* (aops:rand '(3)))
(defparameter *vec-b* (aops:rand '(3)))

(deftest test-parse-spec
  (testing "Parsing of the specification"
    (ok (petalisp-einsum::parse-spec "hj, i -> i"))
    (ok (petalisp-einsum::parse-spec "jh  i ->  i jh, h"))))

(deftest test-execute
  (testing "It can execute"
    (ok (compute (einsum "i, i -> i" 1 2)))
    (ok (compute (einsum "ij, ji -> ij" (aops:rand '(3 3)) (aops:rand '(3 3)))))
    (ok (compute (einsum "jik, ijk -> ij" (aops:rand '(3 3)) (aops:rand '(3 3)))))
    (ok (compute (einsum "ij -> i" (aops:rand '(3 3)))))
    (ok (compute (einsum "  i,
                         j     ->           ij " (aops:rand '(3)) (aops:rand '(3)))))
    (ok (compute (einsum "ii, ij -> ij" (aops:rand '(3)) (aops:rand '(3 3)))))
    (ok (compute (einsum "i, j -> ijk" (aops:rand '(3)) (aops:rand '(3)))))
    (ok (numberp (compute (einsum "ijk ijk" (aops:rand '(3 5 4)) 1))))
    (ok (= 1 (petalisp.core:rank (compute (einsum "ijk ik" (aops:rand '(3 5 4)) 1)))))
    (ok (numberp (compute (einsum "ijk" (aops:rand '(3 5 4))))))
    ;(ok (compute (einsum "i, j -> iji" (aops:rand '(3)) (aops:rand '(3)))))
    (ok (compute (einsum "i, i -> ij" (aops:rand '(3)) (aops:rand '(3)))))
    (ok (compute (einsum "i -> ikkk" (aops:rand '(3)))))
   ; (ok (compute (einsum "i -> iiii" (aops:rand '(3)))))
    (ok (numberp (compute (einsum "i i" (aops:rand '(3)) (aops:rand '(3))))))
    (ok (= 1 (rank (compute (einsum "i ij" (aops:rand '(3)) (aops:rand '(3 3)))))))
    (ok (array-shape (compute (einsum "ij -> ijkl" (aops:rand '(3 4))))))
    ;; not possible with current Petalisp master. Cannot have same input axis in output mask.
    ;(ok (array-shape (compute (einsum "ij -> ijji" (aops:rand '(3 4))))))
    (ok (array-shape (compute (einsum "ij kl -> ijkl" (aops:rand '(3 4)) (aops:rand '(3 4))))))
    (ok (multiple-value-call #'compute (einsum "ij, ji -> ij ji" (aops:rand '(3 3)) (aops:rand '(3 3)))))))

(deftest test-calculation
  (testing "It does the intended thing"
    (ok (approximately-equal (compute (einsum "jik, ikj -> jk" *mat-a* *mat-b*))
                             (compute (matmul *mat-a* *mat-b*))))
    (ok (approximately-equal (compute (einsum "ij, jk -> ik" *mat-a* *mat-b*))
                             (compute (matmul *mat-a* *mat-b*))))
    (ok (approximately-equal (compute (einsum "ij jk" *mat-a* *mat-b*))
                             (compute (matmul *mat-a* *mat-b*))))
    (ok (approximately-equal (compute (einsum "ijk, jki -> ik" *mat-a* *mat-b*))
                             (compute (matmul *mat-a* *mat-b*))))))

(deftest test-start-variant
  (testing "You can use alternative functions"
    (ok (approximately-equal (compute (einsum* "i, i" (list *vec-a* *vec-b*) #'max #'min))
                             (compute (β #'min (α #'max *vec-a* *vec-b*)))))))

;(deftest test-diagonal-entries
  ;(testing "It can extract diagonal entries"
    ;(ok (approximately-equal (compute (einsum "ii" #2A((1 5 5)
                                                       ;(5 2 5)
                                                       ;(5 5 3))))
                             ;#(1 2 3)))))
