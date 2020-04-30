package main

import (
	"fmt"
	"math"
	"math/rand"
	"qgo/sim"
	"time"
)

// Deutsch-Josza algorithm with 1 qubit input and 1 qubit output
func main() {
	// Make a new QuantumCircuit
	qce := sim.NewQuantumCircuit(2)

	// Apply two-qubit hadamard gate to put the register in the equal superposition
	qce.H([]int{0, 1})

	// What function is our oracle?
	rand.Seed(time.Now().Unix())
	r := rand.Intn(3)
	if r == 0 {
		// Balanced, f(0) != f(1)
		qce.CX(0, 1)
		fmt.Println("Making a balanced function")
	} else if r == 1 {
		// Constant, f(x) always 1
		qce.X(1)
		fmt.Println("Making a constant function")
	} else if r == 2 {
		// Constant, f(x) always 0
		// Do nothing, that is take every
		fmt.Println("Making a constant function")
	}

	qce.H([]int{0})

	fmt.Println("The circuit will now determine through one execution whether the oracle was balanced or constant:")
	probs := qce.Exec([]sim.Ket{sim.OneKet, sim.OneKet}).MeasureProbabilities()

	// [0    <- P(|00>) at index 0
	//  0    <- P(|01>) at index 1
	//  0.5  <- P(|10>) at index 2
	//  0.5] <- P(|11>) at index 3

	// If the output has a 1 in the first qubit, then the oracle was constant
	if floatEqual(probs[2]+probs[3], 1, 0.000001) {
		fmt.Println("The oracle was constant")
	}

	// If the output has a 0 in the first qubit, then the oracle was balanced
	if floatEqual(probs[0]+probs[1], 1, 0.000001) {
		fmt.Println("the oracle was balanced")
	}
}

func floatEqual(a, b, epsilon float64) bool {
	return math.Abs(b-a) < epsilon
}
