package main

import (
	"fmt"
	"gonum.org/v1/gonum/blas/cblas128"
	"math"
	"reflect"
	"testing"
)

func Test_FormatMat(t *testing.T) {
	tests := []struct {
		name     string
		mat      cblas128.General
		expected string
	}{
		{"Pauli X matrix",
			cblas128.General{
				Rows:   2,
				Cols:   2,
				Stride: 2,
				Data:   []complex128{0, 1, 1, 0},
			},
			"[0, 1, \n 1, 0]",
		},

		{"Pauli Y matrix",
			cblas128.General{
				Rows:   2,
				Cols:   2,
				Stride: 2,
				Data:   []complex128{0, -1i, 1i, 0},
			},
			"[0, -i, \n i, 0]",
		},

		{"C-Z (CNOT) matrix",
			cblas128.General{
				Rows:   4,
				Cols:   4,
				Stride: 4,
				Data: []complex128{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 0, 1,
					0, 0, 1, 0,
				}},
			"[1, 0, 0, 0, \n" +
				" 0, 1, 0, 0, \n" +
				" 0, 0, 0, 1, \n" +
				" 0, 0, 1, 0]",
		},
		{
			name: "Matrix with real and imaginary components",
			mat: cblas128.General{
				Rows:   2,
				Cols:   2,
				Stride: 2,
				Data: []complex128{
					1.9 + 3.2i, 3.1 + 1.01i,
					4 + 5i, 2.3 - 6i,
				},
			},
			expected: "[(1.9+3.2i), (3.1+1.01i), \n (4+5i), (2.3+-6i)]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := FormatMat(tt.mat)
			if tt.expected != out {
				t.Errorf("Output %v did not match expected: %v", out, tt.expected)
			}
		})
	}
}

func Test_kronecker(t *testing.T) {
	type args struct {
		a cblas128.General
		b cblas128.General
	}
	tests := []struct {
		name string
		args args
		want *cblas128.General
	}{
		{
			name: "1x1 Kronecker",
			args: args{
				a: cblas128.General{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{4},
				},
				b: cblas128.General{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{3},
				},
			},
			want: &cblas128.General{
				Rows:   1,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{12},
			},
		},
		{
			name: "2x2 Kronecker with imaginary numbers",
			args: args{
				a: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 2i, 5, 3i},
				},
				b: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{3 + 2i, 1 + 5i, 2 + 1i, 7},
				},
			},
			want: &cblas128.General{
				Rows:   4,
				Cols:   4,
				Stride: 4,
				Data: []complex128{3 + 2i, 1 + 5i, -4 + 6i, -10 + 2i,
					2 + 1i, 7, -2 + 4i, 14i,
					15 + 10i, 5 + 25i, -6 + 9i, -15 + 3i,
					10 + 5i, 35, -3 + 6i, 21i},
			},
		},
		{
			name: "2x1 Matrix",
			args: args{
				a: cblas128.General{
					Rows:   2,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1 / math.Sqrt2, 2 / math.Sqrt2},
				},
				b: cblas128.General{
					Rows:   2,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1i, 0},
				},
			},
			want: &cblas128.General{
				Rows:   4,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1i / math.Sqrt2, 0, 2i / math.Sqrt2, 0},
			},
		},
		{
			name: "X I",
			args: args{
				a: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{0, 1, 1, 0},
				},
				b: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 0, 0, 1},
				},
			},
			want: &cblas128.General{
				Rows:   4,
				Cols:   4,
				Stride: 4,
				Data:   []complex128{0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0},
			},
		},
		{
			name: "H I",
			args: args{
				a: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2, 1 / math.Sqrt2, -1 / math.Sqrt2},
				},
				b: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 0, 0, 1},
				},
			},
			want: &cblas128.General{
				Rows:   4,
				Cols:   4,
				Stride: 4,
				Data: []complex128{
					0.70710678, 0, 0.70710678, 0,
					0, 0.70710678, 0, 0.70710678,
					0.70710678, 0, -0.70710678, 0,
					0, 0.70710678, 0, -0.70710678,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := kronecker(tt.args.a, tt.args.b); !equal(*got, *tt.want, 0.00000001+0.00000001i) {
				t.Errorf("kronecker() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_mul(t *testing.T) {
	type args struct {
		a *cblas128.General
		b *cblas128.General
	}
	tests := []struct {
		name string
		args args
		want *cblas128.General
	}{
		{
			name: "1x1 Matrix",
			args: args{
				a: &cblas128.General{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{3 + 1i},
				},
				b: &cblas128.General{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{4 + 5i},
				},
			},
			want: &cblas128.General{
				Rows:   1,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{7 + 19i},
			},
		},
		{
			name: "4x4 x 4x1 Multiplication",
			args: args{
				a: &cblas128.General{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data:   []complex128{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0},
				},
				b: &cblas128.General{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0, 0, 1, 0},
				},
			},
			want: &cblas128.General{
				Rows:   4,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{0, 0, 0, 1},
			},
		},
		{
			name: "Ket * Bra",
			args: args{
				a: &cblas128.General{ // Ket of 0
					Rows:   2,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1, 0},
				},
				b: &cblas128.General{ // Bra of 0
					Rows:   1,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 0},
				},
			},
			want: &cblas128.General{
				Rows:   2,
				Cols:   2,
				Stride: 2,
				Data:   []complex128{1, 0, 0, 0},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mul(tt.args.a, tt.args.b); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("mul() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_add(t *testing.T) {
	type args struct {
		a *cblas128.General
		b *cblas128.General
	}
	tests := []struct {
		name string
		args args
		want *cblas128.General
	}{
		{
			name: "Add 2x2",
			args: args{
				a: &cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 2, 3, 4},
				},
				b: &cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{5, 6, 7, 8},
				},
			},
			want: &cblas128.General{
				Rows:   2,
				Cols:   2,
				Stride: 2,
				Data:   []complex128{6, 8, 10, 12},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := add(tt.args.a, tt.args.b); !equal(*got, *tt.want, StdEpsilon) {
				t.Errorf("add() = %v, want %v", got, tt.want)
			}
		})
	}
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Does not panic on mismatched dimensions")
		}
	}()

	add(&cblas128.General{
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{1, 2, 3, 4},
	}, &cblas128.General{
		Rows:   1,
		Cols:   5,
		Stride: 3,
		Data:   []complex128{1, 2, 4},
	})
}

func TestQuantumCircuit_H(t *testing.T) {
	t.Run("Add H to circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 1,
			gates:     []Gate{},
		}
		expected := Gate{
			General: H,
			name:    HADAMARD,
		}
		qc.H([]int{0})
		if len(qc.gates) != 1 || !qc.gates[0].Equals(&expected, StdEpsilon) {
			fmt.Printf("length %v", len(qc.gates))
			t.Errorf("Add H to circuit makes circuit %v, expected %v", qc.gates[0], expected)
		}
	})

	t.Run("Add three qubit H to circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 3,
			gates:     []Gate{},
		}
		expected3 := Gate{
			General: cblas128.General{
				Rows:   8,
				Cols:   8,
				Stride: 8,
				Data: []complex128{
					0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339,
					0.35355339, -0.35355339, 0.35355339, -0.35355339, 0.35355339, -0.35355339, 0.35355339, -0.35355339,
					0.35355339, 0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339, -0.35355339, -0.35355339,
					0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339, -0.35355339, -0.35355339, 0.35355339,
					0.35355339, 0.35355339, 0.35355339, 0.35355339, -0.35355339, -0.35355339, -0.35355339, -0.35355339,
					0.35355339, -0.35355339, 0.35355339, -0.35355339, -0.35355339, 0.35355339, -0.35355339, 0.35355339,
					0.35355339, 0.35355339, -0.35355339, -0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339,
					0.35355339, -0.35355339, -0.35355339, 0.35355339, -0.35355339, 0.35355339, 0.35355339, -0.35355339,
				},
			},
			name: HADAMARD,
		}
		qc.H([]int{0, 1, 2})
		if len(qc.gates) != 1 || !qc.gates[0].Equals(&expected3, StdEpsilon) {
			fmt.Printf("length %v", len(qc.gates))
			t.Errorf("Add H to circuit makes circuit %v, expected %v", qc.gates[0], expected3)
		}
	})

	t.Run("Panic on too many qubits", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 2,
			gates:     []Gate{},
		}
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Does not panic on too many qubits")
			}
		}()
		qc.H([]int{0, 1, 2})
	})
}

func TestQuantumCircuit_CX(t *testing.T) {
	t.Run("Add CX to circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 2,
			gates:     []Gate{},
		}
		expected := Gate{
			General: cblas128.General{
				Rows:   4,
				Cols:   4,
				Stride: 4,
				Data: []complex128{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 0, 1,
					0, 0, 1, 0,
				},
			},
			name: CX,
		}
		qc.CX(0, 1)
		if len(qc.gates) != 1 || !qc.gates[0].Equals(&expected, StdEpsilon) {
			fmt.Printf("length %v", len(qc.gates))
			t.Errorf("Add CX to circuit makes circuit %v, expected %v", qc.gates[0], expected)
		}
	})

	t.Run("Add CX skipping over qubits to circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 4,
			gates:     []Gate{},
		}
		expected := Gate{
			General: cblas128.General{
				Rows:   16,
				Cols:   16,
				Stride: 16,
				Data: []complex128{
					1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
					0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
				},
			},
			name: CX,
		}
		qc.CX(2, 3)
		if len(qc.gates) != 1 || !qc.gates[0].Equals(&expected, StdEpsilon) {
			fmt.Printf("length %v", len(qc.gates))
			t.Errorf("Add CX to circuit makes circuit %v, expected %v", qc.gates[0], expected)
		}
	})
}

func TestQuantumCircuit_Compile(t *testing.T) {
	t.Run("Compile CX on two qubits", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 2,
			gates:     []Gate{*createCX(0, 1, 2)},
		}
		expected := *createCX(0, 1, 2)
		qc.Compile()
		if !equal(qc.compiled.General, expected.General, StdEpsilon) {
			t.Errorf("Compiling X gate makes circuit %v, expected %v", qc.compiled, expected)
		}
		if !qc.compileValid {
			t.Errorf("Compiling circuit does not make compile valid")
		}
	})

	t.Run("Compile H(0) CX", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 2,
			gates:     []Gate{*createH([]int{0}, 2), *createCX(0, 1, 2)},
		}
		expectedMatrix := cblas128.General{
			Rows:   4,
			Cols:   4,
			Stride: 4,
			Data: []complex128{
				0.70710678, 0, 0.70710678, 0,
				0, 0.70710678, 0, 0.70710678,
				0, 0.70710678, 0, -0.70710678,
				0.70710678, 0, -0.70710678, 0,
			},
		}
		qc.Compile()
		if !equal(qc.compiled.General, expectedMatrix, StdEpsilon) {
			t.Errorf("Compiling H(0) CX gate makes circuit %v, expected %v", qc.compiled.General, expectedMatrix)
		}
		if !qc.compileValid {
			t.Errorf("Compiling circuit does not make compile valid")
		}
	})
}

func TestQuantumCircuit_Exec(t *testing.T) {
	type fields struct {
		numQubits int
		gates     []Gate
		compiled  Gate
	}
	type args struct {
		register []cblas128.General
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *QuantumCircuitExecution
	}{
		{
			name: "Single qubit wire",
			fields: fields{
				numQubits: 1,
				gates: []Gate{
					{
						General: I,
						name:    WIRE,
					},
				},
				compiled: Gate{
					General: I,
					name:    WIRE,
				},
			},
			args: args{
				register: []cblas128.General{ZeroKet},
			},
			want: &QuantumCircuitExecution{
				in:  []cblas128.General{ZeroKet},
				out: ZeroKet,
			},
		},
		{
			name: "Two-qubit Hadamard on |01>",
			fields: fields{
				numQubits: 2,
				gates: []Gate{
					*createH([]int{0, 1}, 2),
				},
				compiled: *createH([]int{0, 1}, 2),
			},
			args: args{
				register: []cblas128.General{ZeroKet, OneKet},
			},
			want: &QuantumCircuitExecution{
				in: []cblas128.General{ZeroKet, OneKet},
				out: cblas128.General{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0.5, -0.5, 0.5, -0.5},
				},
			},
		},
		{
			name: "Three qubit Hadamard on |1+->",
			fields: fields{
				numQubits: 3,
				gates: []Gate{
					*createH([]int{0, 1, 2}, 3),
				},
				compiled: *createH([]int{0, 1, 2}, 3),
			},
			args: args{
				register: []cblas128.General{OneKet, HPlusKet, HMinusKet},
			},
			want: &QuantumCircuitExecution{
				in: []cblas128.General{OneKet, HPlusKet, HMinusKet},
				out: cblas128.General{
					Rows:   8,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0, 0.70710678, 0, 0, 0, -0.70710678, 0, 0},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qc := &QuantumCircuit{
				numQubits: tt.fields.numQubits,
				gates:     tt.fields.gates,
				compiled:  tt.fields.compiled,
			}
			if got := qc.Exec(tt.args.register); !equal(got.out, tt.want.out, StdEpsilon) {
				t.Errorf("Exec() = %v, want %v", got, tt.want)
			}
		})
	}
	t.Run("Panic on not enough qubits", func(t *testing.T) {
		qc := QuantumCircuit{
			numQubits:    2,
			gates:        []Gate{*createH([]int{0}, 2)},
			compileValid: false,
			compiled:     Gate{},
		}

		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Does not panic on not enough qubit inputs")
			}
		}()

		qc.Exec([]cblas128.General{ZeroKet})
	})
}

func TestQuantumCircuit_addGate(t *testing.T) {
	h := createH([]int{0, 1}, 2)
	t.Run("Add gate to empty circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits:    2,
			gates:        []Gate{},
			compileValid: false,
			compiled:     Gate{},
		}
		g := h
		qc.addGate(*g)

		if !(len(qc.gates) == 1) || !qc.gates[0].Equals(g, StdEpsilon) || qc.compileValid {
			t.Errorf("addGate() = %v, want %v", qc.gates, []Gate{*g})
		}
	})

	t.Run("Add gate to existing compiled circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits:    2,
			gates:        []Gate{*h},
			compileValid: true,
			compiled:     *h,
		}

		cx := createCX(0, 1, 2)
		qc.addGate(*cx)
		if !(len(qc.gates) == 2) || !qc.gates[0].Equals(h, StdEpsilon) || !qc.gates[1].Equals(cx, StdEpsilon) || qc.compileValid {
			t.Errorf("addGate() = %v, want %v", qc.gates, []Gate{*h, *cx})
		}
	})
}
