package main

import (
	"fmt"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/cblas128"
	"math"
)

type QuantumCircuit struct {
	numQubits int
	gates     []Gate
}

func main() {

}

// Add a singular or multi-qubit Hadamard gate to this circuit.
// Takes in a list of qubit indices that the Hadamard gate should apply to.
//func (qc *QuantumCircuit) H(hQubits []int) {
//	if len(hQubits) > qc.numQubits {
//		panic("Too many qubits provided for H gate")
//	}
//
//	// hLocations contains the locations, marked with 1, where the H gate will apply.
//	var hLocations = make([]int, qc.numQubits)
//	for _, q := range hQubits {
//		if q < 0 || q >= qc.numQubits {
//			panic("Qubit out of range")
//		}
//		hLocations[q] = 1
//	}
//	qc.gates = append(qc.gates, createH(hLocations))
//}

// Determines equality between two cblas128.General matrices, using epsilon a complex number.
// Two matrices are equal if their dimensions are equal and their values are equal,
// given a complex epsilon. Two complex numbers a+bi and c+di are considered to be equal
// when abs(a-c) < real(epsilon) and abs(b-d) < imag(epsilon).
func equal(a, b cblas128.General, epsilon complex128) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Stride != b.Stride {
		return false
	}
	if len(a.Data) != len(b.Data) {
		return false
	}

	for i := range a.Data {
		if math.Abs(real(a.Data[i])-real(b.Data[i])) > real(epsilon) ||
			math.Abs(imag(a.Data[i])-imag(b.Data[i])) > imag(epsilon) {
			return false
		}
	}

	return true
}

// Creates the Kronecker product of matrices A and B
// Returns a reference to the matrix containing the result
func kronecker(a, b cblas128.General) *cblas128.General {
	ar, ac := a.Rows, a.Cols
	br, bc := b.Rows, b.Cols
	out := cblas128.General{
		Rows:   ar * br,
		Cols:   ac * bc,
		Stride: ac * bc,
		Data:   make([]complex128, (ar*br)*(ac*bc)),
	}

	for rowA := 0; rowA < ar; rowA++ {
		for colA := 0; colA < ac; colA++ {
			indexA := rowA*a.Stride + colA

			for rowB := 0; rowB < br; rowB++ {
				for colB := 0; colB < bc; colB++ {
					indexB := rowB*b.Stride + colB

					outRow := rowA*br + rowB
					outCol := colA*bc + colB
					indexOut := outRow*out.Stride + outCol
					out.Data[indexOut] = a.Data[indexA] * b.Data[indexB]
				}
			}
		}
	}

	return &out
}

// Does standard matrix multiplication of A and B
// Returns a reference to the matrix containing the result
func mul(a, b *cblas128.General) *cblas128.General {
	var c = cblas128.General{
		Rows:   a.Rows,
		Cols:   b.Cols,
		Stride: b.Cols,
		Data:   make([]complex128, a.Rows*b.Cols),
	}
	cblas128.Gemm(blas.NoTrans, blas.NoTrans, 1, *a, *b, 0, c)
	return &c
}

func FormatMat(X cblas128.General) string {
	formatNum := func(c complex128) string {
		formatImag := func(f float64) string {
			var imagComp string
			if f == 1 {
				imagComp += "i"
			} else if f == -1 {
				imagComp += "-i"
			} else {
				imagComp += fmt.Sprintf("%vi", f)
			}
			return imagComp
		}

		var s string
		if imag(c) != 0 && real(c) != 0 {
			s += fmt.Sprintf("(%v+%v)", real(c), formatImag(imag(c)))
		} else if imag(c) == 0 {
			s += fmt.Sprintf("%v", real(c))
		} else if real(c) == 0 {
			s += formatImag(imag(c))
		}
		return s
	}

	var str string

	str += "["
	for r := 0; r < X.Rows; r++ {
		for c := 0; c < X.Cols; c++ {
			num := X.Data[(X.Cols*r)+c]
			str += formatNum(num)

			// Put comma afterwards except last item
			if c != X.Cols-1 || r != X.Rows-1 {
				str += ", "
			}
		}
		// Put newline except at last item
		if r != X.Rows-1 {
			str += "\n "
		}
	}
	str += "]"
	return str
}
