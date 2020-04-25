package main

import (
	"gonum.org/v1/gonum/blas/cblas128"
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
	cblas128.General
	name GateName
}

var (
	// UTILITY MATRICES

	ZeroKet = cblas128.General{ // Ket of 0 = |0>
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1, 0},
	}

	ZeroBra = cblas128.General{ // Bra of 0 = <0|
		Rows:   1,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{1, 0},
	}

	OneKet = cblas128.General{ // Ket of 1 = |1>
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{0, 1},
	}

	OneBra = cblas128.General{ // Bra of 1 = <1|
		Rows:   1,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{0, 1},
	}

	HPlusKet = cblas128.General{ // Ket of hadamard plus state = |+>
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2},
	}

	HMinusKet = cblas128.General{ // Ket of hadamard minus state = |->
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1 / math.Sqrt2, -1 / math.Sqrt2},
	}

	I = cblas128.General{ // Identity Matrix
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{1, 0, 0, 1},
	}

	H = cblas128.General{ // Hadamard Matrix
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2, 1 / math.Sqrt2, -1 / math.Sqrt2},
	}

	X = cblas128.General{ // Pauli-X Matrix
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{0, 1, 1, 0},
	}
)

// Creates a multi-qubit Hadamard gate across the qubits specified here
func createH(qubits []int, numQubits int) *Gate {
	sort.Ints(qubits)

	mat := cblas128.General{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}

	qubitCount := 0
	for i := 0; i < numQubits; i++ {
		if i == qubits[qubitCount] {
			mat = *kronecker(mat, H)

			if qubitCount < len(qubits)-1 {
				qubitCount++
			}
		} else {
			mat = *kronecker(mat, I)
		}
	}

	return &Gate{
		General: mat,
		name:    HADAMARD,
	}
}

// Creates a single Pauli-X gate operating on the given qubit.
// Will not fail if given multiple qubits to apply X gate on.
func createX(qubit, numQubits int) *Gate {
	var mat cblas128.General
	if qubit == 0 {
		mat = X
	} else {
		mat = I
	}

	for i := 1; i < numQubits; i++ {
		if i == qubit {
			mat = *kronecker(mat, X)
		} else {
			mat = *kronecker(mat, I)
		}
	}

	return &Gate{
		General: mat,
		name:    PAULIX,
	}
}

// Creates a controlled-X gate, given the number of the qubit that acts as the control,
// the qubit that acts as the target, and the total number of qubits.
// Reference: http://www.sakkaris.com/tutorials/quantum_control_gates.html
func createCX(control, target, numQubits int) *Gate {
	controlMatrix := cblas128.General{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}
	targetMatrix := cblas128.General{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}

	// If the control qubit is |0> leave the target qubit alone
	// If the control qubit is |1>, apply X to the target qubit
	for i := 0; i < numQubits; i++ {
		if i == control {
			controlMatrix = *kronecker(controlMatrix, *mul(&ZeroKet, &ZeroBra))
			targetMatrix = *kronecker(targetMatrix, *mul(&OneKet, &OneBra))
		} else if i == target {
			controlMatrix = *kronecker(controlMatrix, I)
			targetMatrix = *kronecker(targetMatrix, X)
		} else {
			controlMatrix = *kronecker(controlMatrix, I)
			targetMatrix = *kronecker(targetMatrix, I)
		}
	}

	combinedMatrix := add(&controlMatrix, &targetMatrix)
	return &Gate{
		General: *combinedMatrix,
		name:    CX,
	}
}

// Combines a set of gates into one using matrix multiplication
func Combine(name GateName, matrices ...Gate) *Gate {
	outMatrix := matrices[len(matrices)-1].General
	for i := len(matrices) - 2; i >= 0; i-- {
		outMatrix = *mul(&outMatrix, &matrices[i].General)
	}
	return &Gate{
		General: outMatrix,
		name:    name,
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
