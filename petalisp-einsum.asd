(defsystem "petalisp-einsum"
  :version "0.1.0"
  :author "Stephan Seitz"
  :license "GPL"
  :depends-on ("petalisp"
               "cl-ppcre"
               "alexandria")
  :components ((:module "src"
                :components
                ((:file "main"))))
  :description ""
  :in-order-to ((test-op (test-op "petalisp-einsum/tests"))))

(defsystem "petalisp-einsum/tests"
  :author "Stephan Seitz"
  :license "GPL"
  :depends-on ("petalisp-einsum"
               "rove"
               "array-operations"
               "petalisp.test-suite")
  :components ((:module "tests"
                :components
                ((:file "main"))))
  :description "Test system for petalisp-einsum"
  :perform (test-op (op c) (symbol-call :rove :run c)))
