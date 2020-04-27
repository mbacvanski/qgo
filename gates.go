package main

import (
	"math"
	"sort"
)

type GateName int

const (
	HADAMARD = iota
	WIRE     = iota
	PAULIX   = iota
	CX       = iota
)

func (g *Gate) Name() string {
	return [...]string{"Hadamard", "Identity", "Pauli-X", "C-X"}[g.name]
}

type Gate struct {
	Matrix
	name GateName
}

var (
	// UTILITY MATRICES

	ZeroKet = NewKet(Matrix{ // Ket of 0 = |0>
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1, 0},
	})

	ZeroBra = NewBra(Matrix{ // Bra of 0 = <0|
		Rows:   1,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{1, 0},
	})

	OneKet = NewKet(Matrix{ // Ket of 1 = |1>
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{0, 1},
	})

	OneBra = NewBra(Matrix{ // Bra of 1 = <1|
		Rows:   1,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{0, 1},
	})

	HPlusKet = NewKet(Matrix{ // Ket of hadamard plus state = |+>
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2},
	})

	HMinusKet = NewKet(Matrix{ // Ket of hadamard minus state = |->
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1 / math.Sqrt2, -1 / math.Sqrt2},
	})

	I = Matrix(Matrix{ // Identity Matrix
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{1, 0, 0, 1},
	})

	H = Matrix(Matrix{ // Hadamard Matrix
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2, 1 / math.Sqrt2, -1 / math.Sqrt2},
	})

	X = Matrix(Matrix{ // Pauli-X Matrix
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{0, 1, 1, 0},
	})
)

// Creates a multi-qubit Hadamard gate across the qubits specified here
func createH(qubits []int, numQubits int) *Gate {
	sort.Ints(qubits)

	mat := Matrix{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}

	qubitCount := 0
	for i := 0; i < numQubits; i++ {
		if i == qubits[qubitCount] {
			mat = *mat.Kronecker(H)

			if qubitCount < len(qubits)-1 {
				qubitCount++
			}
		} else {
			mat = *mat.Kronecker(I)
		}
	}

	return &Gate{
		Matrix: mat,
		name:   HADAMARD,
	}
}

// Creates a single Pauli-X gate operating on the given qubit.
// Will not fail if given multiple qubits to apply X gate on.
func createX(qubit, numQubits int) *Gate {
	mat := Matrix{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}

	for i := 0; i < numQubits; i++ {
		if i == qubit {
			mat = *mat.Kronecker(X)
		} else {
			mat = *mat.Kronecker(I)
		}
	}

	return &Gate{
		Matrix: mat,
		name:   PAULIX,
	}
}

// Creates a controlled-X gate, given the number of the qubit that acts as the control,
// the qubit that acts as the target, and the total number of qubits.
// Reference: http://www.sakkaris.com/tutorials/quantum_control_gates.html
func createCX(control, target, numQubits int) *Gate {
	controlMatrix := Matrix{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}
	targetMatrix := Matrix{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}

	// If the control qubit is |0> leave the target qubit alone
	// If the control qubit is |1>, apply X to the target qubit
	for i := 0; i < numQubits; i++ {
		if i == control {
			controlMatrix = *controlMatrix.Kronecker(*(Matrix)(ZeroKet).Mul((Matrix)(ZeroBra)))
			targetMatrix = *targetMatrix.Kronecker(*(Matrix)(OneKet).Mul((Matrix)(OneBra)))
		} else if i == target {
			controlMatrix = *controlMatrix.Kronecker(I)
			targetMatrix = *targetMatrix.Kronecker(X)
		} else {
			controlMatrix = *controlMatrix.Kronecker(I)
			targetMatrix = *targetMatrix.Kronecker(I)
		}
	}

	combinedMatrix := controlMatrix.Add(targetMatrix)
	return &Gate{
		Matrix: *combinedMatrix,
		name:   CX,
	}
}

// Combines a set of gates into one using matrix multiplication
func Combine(name GateName, matrices ...Gate) *Gate {
	outMatrix := matrices[len(matrices)-1].Matrix
	for i := len(matrices) - 2; i >= 0; i-- {
		outMatrix = *outMatrix.Mul(matrices[i].Matrix)
	}
	return &Gate{
		Matrix: outMatrix,
		name:   name,
	}
}

// Determines equality between two gates, using epsilon as a complex number.
// Two gates are Equals if their names are Equals and their matrix representations
// are also Equals, given a complex epsilon. Two complex numbers a+bi and c+di
// are considered to be Equals if abs(a-c) < real(epsilon) and abs(b-d) < imag(epsilon).
func (g *Gate) Equals(b *Gate, epsilon complex128) bool {
	if g.name != b.name {
		return false
	}
	return g.Matrix.Equals(b.Matrix, epsilon)
}
