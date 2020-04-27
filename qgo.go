package main

import (
	"fmt"
	"math"
	"strconv"
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

// Computes the probabilites of measuring all outcomes in the standard Z bases
func (qce *QuantumCircuitExecution) MeasureProbabilities() []float64 {
	probabilities := make([]float64, qce.out.Size())

	numQubits := int(math.Log2(float64(qce.out.Size())))

	for i := 0; i < qce.out.Size(); i++ {
		byteStr := fmt.Sprintf("%0"+strconv.Itoa(numQubits)+"v", strconv.FormatInt(int64(i), 2))
		var basis []Ket
		for _, c := range byteStr {
			if c == '0' {
				basis = append(basis, ZeroKet)
			} else if c == '1' {
				basis = append(basis, OneKet)
			}
		}

		probabilities[i] = qce.MeasureProbability(basis)
	}

	return probabilities
}

// Measures the probability of reading out a certain vector
func (qce *QuantumCircuitExecution) MeasureProbability(basis []Ket) float64 {
	// Check dimensions of the measurement basis
	// We need n basis vectors to measure a space of 2^n
	if float64(len(basis)) != math.Log2(float64(qce.out.Size())) {
		panic("Not enough basis vectors for measurement basis")
	}

	// Assemble basis vector by Kronecker products
	basisVec := KronKets(basis)

	// Magnitude of projection of output onto basis
	magnitude := qce.out.Dotp(basisVec) / basisVec.Dotp(basisVec)

	return real(magnitude * magnitude)
}

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

	input := KronKets(qubitStates)

	return &QuantumCircuitExecution{
		in:       qubitStates,
		register: input,
		out:      NewColVec(*qc.compiled.Matrix.Mul((Matrix)(input))),
	}
}
