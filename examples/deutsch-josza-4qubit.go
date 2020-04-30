package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"qgo/sim"
	"strconv"
)

// Deutsch-Josza Algorithm with three input register qubits and one output qubit
func main() {
	qc := sim.NewQuantumCircuit(4)
	qc.H([]int{0, 1, 2, 3})

	// To make a constant oracle, simply NOT the output qubit
	constantOracle := sim.NewQuantumCircuit(4)
	constantOracle.X(3)

	// To make a balanced oracle, we can perform a CNOT against the output qubit
	// using each of the input register qubits as controls.
	balancedOracle := sim.NewQuantumCircuit(4)
	balancedOracle.CX(0, 3)
	balancedOracle.CX(1, 3)
	balancedOracle.CX(2, 3)

	fmt.Println("Enter\n[1] For a constant oracle\n[2] For a balanced oracle")
	reader := bufio.NewReader(os.Stdin)
	r, _, _ := reader.ReadRune()
	choice, _ := strconv.Atoi(string(r))

	if choice == 1 {
		qc.AddCircuit(constantOracle)
	} else if choice == 2 {
		qc.AddCircuit(balancedOracle)
	} else {
		fmt.Println("Bad choice. Exiting.")
		os.Exit(1)
	}

	qc.H([]int{0, 1, 2})

	// Run the circuit providing the |1111> state as the input
	output := qc.Exec([]sim.Ket{sim.OneKet, sim.OneKet, sim.OneKet, sim.OneKet})

	// Measure the probability that the first qubit is 0
	prob := output.MeasureProbabilityOn(0)

	if math.Abs(prob-1) < 0.000001 {
		fmt.Println("The oracle was constant")
	} else {
		fmt.Println("The oracle was balanced")
	}
}
