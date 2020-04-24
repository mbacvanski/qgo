package main

import (
	"gonum.org/v1/gonum/blas/cblas128"
	"math"
)

type GateName int

const (
	HADAMARD = iota
	IDENTITY = iota
	PAULI_X  = iota
)

func (g *Gate) Name() string {
	return [...]string{"Hadamard", "Identity", "Pauli-X"}[g.name]
}

type Gate struct {
	cblas128.General
	name GateName
}

var (
	H = Gate{
		General: cblas128.General{
			Rows:   2,
			Cols:   2,
			Stride: 2,
			Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2, 1 / math.Sqrt2, -1 / math.Sqrt2},
		},
		name: HADAMARD,
	}

	I = Gate{
		General: cblas128.General{
			Rows:   2,
			Cols:   2,
			Stride: 2,
			Data:   []complex128{1, 0, 0, 1},
		},
		name: IDENTITY,
	}

	X = Gate{
		General: cblas128.General{
			Rows:   2,
			Cols:   2,
			Stride: 2,
			Data:   []complex128{0, 1, 1, 0},
		},
		name: PAULI_X,
	}
)

// Creates a multi-qubit Hadamard gate across the qubits specified here
func createH(qubits []int) *Gate {
	var mat cblas128.General
	if qubits[0] == 0 {
		mat = I.General
	} else if qubits[0] == 1 {
		mat = H.General
	}

	for i := 1; i < len(qubits); i++ {
		if qubits[i] == 0 {
			mat = *kronecker(mat, I.General)
		} else if qubits[i] == 1 {
			mat = *kronecker(mat, H.General)
		}
	}

	return &Gate{
		General: mat,
		name:    HADAMARD,
	}
}

// Creates a single Pauli-X gate operating on the given qubit.
// Will not fail if
func createX(qubits []int) *Gate {
	var mat cblas128.General
	if qubits[0] == 0 {
		mat = I.General
	} else if qubits[0] == 1 {
		mat = X.General
	}

	for i := 1; i < len(qubits); i++ {
		if qubits[i] == 0 {
			mat = *kronecker(mat, X.General)
		} else if qubits[i] == 1 {
			mat = *kronecker(mat, H.General)
		}
	}

	return &Gate{
		General: mat,
		name:    PAULI_X,
	}
}

// Determines equality between two gates, using epsilon as a complex number.
// Two gates are equal if their names are equal and their matrix representations
// are also equal, given a complex epsilon. Two complex numbers a+bi and c+di
// are considered to be equal if abs(a-c) < real(epsilon) and abs(b-d) < imag(epsilon).
func (g *Gate) Equals(b *Gate, epsilon complex128) bool {
	if g.name != b.name {
		return false
	}
	return equal(g.General, b.General, epsilon)
}
