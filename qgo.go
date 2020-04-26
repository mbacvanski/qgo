package main

import (
	"fmt"
	"math"
)

type QuantumCircuit struct {
	numQubits    int
	gates        []Gate
	compileValid bool
	compiled     Gate
}

type QuantumCircuitExecution struct {
	in       []Ket
	register ColVec
	out      ColVec
}

type QuantumMeasurement struct {
}

//func (qce *QuantumCircuitExecution) MeasureProbabilities() *QuantumMeasurement {
//
//}

//func (qce *QuantumCircuitExecution) MeasureProbability(basis []cblas128.Vector) float64 {
//	// Check dimensions of the measurement basis
//	if len(basis) != qce.out {
//		panic("Not enough basis vectors for measurement basis")
//	}
//
//
//}

func (qc *QuantumCircuit) addGate(g Gate) {
	qc.compileValid = false
	qc.gates = append(qc.gates, g)
}

// Add a singular or multi-qubit Hadamard gate to this circuit.
// Takes in a list of qubit indices that the Hadamard gate should apply to.
func (qc *QuantumCircuit) H(hQubits []int) {
	if len(hQubits) > qc.numQubits {
		panic("Too many qubits provided for H gate")
	}
	qc.addGate(*createH(hQubits, qc.numQubits))
}

// Adds a CNOT (C-X) gate to this circuit. Takes in the control and target
// qubit indices that the gate should operate on.
func (qc *QuantumCircuit) CX(control, target int) {
	qc.addGate(*createCX(control, target, qc.numQubits))
}

// Compiles all gates in the circuit into one compiled operation
func (qc *QuantumCircuit) Compile() {
	qc.compiled = *Combine(42, qc.gates...)
	qc.compileValid = true
}

// Executes the circuit with the given register as input.
// Register is given as an array of 2x1 matrices, given each state.
func (qc *QuantumCircuit) Exec(qubitStates []Ket) *QuantumCircuitExecution {
	if len(qubitStates) != qc.numQubits {
		panic(fmt.Sprintf("Cannot execute qubitStates of size %v on circuit with %v qubits", len(qubitStates), qc.numQubits))
	}

	if !qc.compileValid {
		qc.Compile()
	}

	input := Matrix{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}
	for i := 0; i < len(qubitStates); i++ {
		input = *kronecker(input, Matrix(qubitStates[i]))
	}

	return &QuantumCircuitExecution{
		in:       qubitStates,
		register: NewColumnVec(input),
		out:      NewColumnVec(*mul(&qc.compiled.Matrix, &input)),
	}
}

// Determines equality between two cblas128.Matrix matrices, using epsilon a complex number.
// Two matrices are equal if their dimensions are equal and their values are equal,
// given a complex epsilon. Two complex numbers a+bi and c+di are considered to be equal
// when abs(a-c) < real(epsilon) and abs(b-d) < imag(epsilon).
func equal(a, b Matrix, epsilon complex128) bool {
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
