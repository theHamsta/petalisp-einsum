(defpackage petalisp-einsum
  (:use :cl
        :petalisp)
  (:import-from :petalisp.core
                :make-transformation
                :rank)
  (:import-from :alexandria
                :iota)
  (:export :einsum
           :einsum*))
(in-package :petalisp-einsum)

(defparameter *word-scanner* (ppcre:create-scanner "\\w+"))
(defparameter *axis-scanner* (ppcre:create-scanner "(\\w)"))
(defparameter *split-scanner* (ppcre:create-scanner "->"))
(defparameter *foreach-op* #'*)
(defparameter *reduce-op* #'+)
(defparameter *reduce-initial-value* nil)

(defstruct (einsum-spec)
  input-specs
  output-specs
  input-axes
  reduce-axes)

(defun determine-output-specs (input-specs)
  (list
    (unless (= 1 (length input-specs)) ; One entry only? Reduce over all output axis
      (let ((non-unique-axes nil))
        (sort
          (reduce (lambda (a b) (let ((common (intersection a b)))
                                  (setf non-unique-axes (append common non-unique-axes))
                                  (set-difference (union a b) non-unique-axes)))
                  (mapcar (lambda (x) (coerce x 'list)) input-specs))
          #'char-lessp)))))

(defun parse-spec (spec)
  (let ((splits (ppcre:split *split-scanner* spec)))
    (let ((input-specs (ppcre:all-matches-as-strings *word-scanner* (first splits)))
          (given-output-specs (when (< 0 (length splits))
                                (ppcre:all-matches-as-strings *word-scanner* (second splits)))))
      (let* ((input-axes (sort (reduce #'union (mapcar (lambda (x) (coerce x 'list)) input-specs)) #'char-lessp))
             (output-specs (or given-output-specs
                               (determine-output-specs input-specs)))
             (reduce-axes (mapcar (lambda (x) (sort (set-difference input-axes (coerce x 'list)) #'char-lessp))
                                  output-specs)))
        (make-einsum-spec
          :input-specs input-specs
          :output-specs output-specs
          :input-axes input-axes
          :reduce-axes reduce-axes)))))

(defun prepare-arrays (inputs specs &optional all-axes)
  (mapcar (lambda (i in)
            (if (= 0 (rank i))
                  i
                  (reshape i (make-transformation :input-mask (make-list (rank i))
                                                  :output-mask (map 'list
                                                                    (lambda (out-axis)
                                                                      (position out-axis (subseq in 0 (rank i))))
                                                                    (or all-axes (sort (copy-seq in) #'char-lessp)))))))
          inputs
          specs))

(defun einsum (spec &rest arrays)
  (let* ((spec (parse-spec spec))
         (prepared-inputs (prepare-arrays arrays (einsum-spec-input-specs spec) (einsum-spec-input-axes spec)))
         (common-result (apply #'α `(,*foreach-op* ,@prepared-inputs)))
         (reduced-results (mapcar (lambda (s) 
                                    (loop for axis in (reverse s)
                                          with tmp = common-result
                                          do (setf tmp (β* *reduce-op*
                                                           *reduce-initial-value*
                                                           tmp
                                                           (position axis (einsum-spec-input-axes spec))))
                                          finally (return tmp)))
                          (einsum-spec-reduce-axes spec))))
      (values-list (prepare-arrays reduced-results (einsum-spec-output-specs spec)))))

(defun einsum* (spec arrays &optional (foreach-op #'*) (reduce-op #'+) reduce-initial-value)
  (let ((*foreach-op* foreach-op)
        (*reduce-op* reduce-op)
        (*reduce-initial-value* reduce-initial-value))
    (apply #'einsum `(,spec ,@arrays))))
