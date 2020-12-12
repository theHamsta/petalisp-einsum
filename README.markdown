# petalisp-einsum

## Usage

The only exported function of this package is `einsum` and its variant `einsum*`.
It tries to emulate the behavior of Numpy's/PyTorch's einsum function.

E. g. the following expression will calculate the matrix product of two arrays.
We calculate the product over the different axis indices and sum over non-unique indices.

```lisp
(ql:quickload :array-operations)
(defparameter *mat-a* (aops:rand '(4 3)))
(defparameter *mat-b* (aops:rand '(3 5)))

(petalisp:compute (einsum "ij jk" *mat-a* *mat-b*))
``` 

When you specify one or multiple results it will sum over axis specifiers not present in the result.
So the following will calculate the matrix product and its transpose (commas are not obligatory).

```lisp
(multiple-value-call #'compute (einsum "ij, jk -> ik ki" *mat-a* *mat-b*))
```

The stared version allows to set alternatives for the elementwise operation and reduction.

```lisp
(petalisp:compute (einsum "ij jk" (list *mat-a* *mat-b*) #'max #'min))
``` 

## Installation

With [[https://www.quicklisp.org/beta/][quicklisp]] installed, clone this
repository to your local projects folder and then load it via quicklisp.

```lisp
(ql:quickload :petalisp-einsum)
```

## Author

* Stephan Seitz (stephan.seitz@fau.de)

## Copyright

Copyright (c) 2020 Stephan Seitz (stephan.seitz@fau.de)

## License

Licensed under the GPL License.
