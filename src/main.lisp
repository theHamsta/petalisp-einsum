(defpackage petalisp-einsum
  (:use :cl
        :petalisp)
  (:import-from :petalisp.core
                :make-transformation
                :rank)
  (:import-from :alexandria
                :iota)
  (:export :einsum
           :*foreach-op*
           :*reduce-op*
           :*reduce-initial-value*))
(in-package :petalisp-einsum)

(defparameter *word-scanner* (ppcre:create-scanner "\\w+"))
(defparameter *axis-scanner* (ppcre:create-scanner "(\\w)"))
(defparameter *split-scanner* (ppcre:create-scanner "->"))
(defparameter *foreach-op* #'*)
(defparameter *reduce-op* #'+)
(defparameter *reduce-initial-value* 0)

(defstruct (einsum-spec)
  input-specs
  output-specs
  input-axes
  reduce-axes)

(declaim (optimize (debug 3)))

(defun determine-output-specs (input-specs)
  (let ((non-unique-axes nil))
    (list (sort
            (reduce (lambda (a b) (let ((common (intersection a b)))
                                    (setf non-unique-axes (concatenate 'list common non-unique-axes))
                                    (set-difference (union a b) non-unique-axes)))
                    (mapcar (lambda (x) (coerce x 'list)) input-specs))
            #'char-lessp))))

(defun parse-spec (spec)
  (let ((splits (ppcre:split *split-scanner* spec)))
    (let ((input-specs (ppcre:all-matches-as-strings *word-scanner* (first splits)))
          (given-output-specs (when (< 0 (length splits)) (ppcre:all-matches-as-strings *word-scanner* (second splits)))))
      (let* ((input-axes (sort (reduce #'union (mapcar (lambda (x) (coerce x 'list)) input-specs)) #'char-lessp))
             (output-specs (or given-output-specs (determine-output-specs input-specs)))
             (reduce-axes (mapcar (lambda (x) (set-difference input-axes (coerce x 'list))) output-specs)))
        (make-einsum-spec
          :input-specs input-specs
          :output-specs output-specs
          :input-axes input-axes
          :reduce-axes reduce-axes)))))

(defun prepare-arrays (inputs spec &optional all-axes)
  (mapcar (lambda (i in)
            (if (= 0 (rank i))
                  i
                  (reshape i (make-transformation :input-mask (make-list (rank i))
                                                  :output-mask (map 'list
                                                                    (lambda (out-axis) (position out-axis (subseq in 0 (rank i))))
                                                                    (or all-axes (sort (copy-seq in) #'char-lessp)))))))
          inputs
          spec))

(defun einsum (spec &rest arrays)
  (let* ((spec (parse-spec spec))
         (prepared-inputs (prepare-arrays arrays (einsum-spec-input-specs spec) (einsum-spec-input-axes spec)))
         (common-result (apply #'α `(,*foreach-op* ,@prepared-inputs)))
         (reduced-results (mapcar (lambda (s) 
                                    (loop for axis in s
                                          with tmp = common-result
                                          do (progn
                                               (setf tmp (β* *reduce-op*
                                                           *reduce-initial-value*
                                                           tmp
                                                           (position axis (einsum-spec-input-axes spec)))))
                                          finally (return tmp)))
                          (einsum-spec-reduce-axes spec))))
      (values-list (prepare-arrays reduced-results (einsum-spec-output-specs spec)))))
