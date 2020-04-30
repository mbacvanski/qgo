package sim

import (
	"fmt"
	"math"
	"testing"
)

const FloatEpsilon = 0.00000001

func TestQuantumCircuit_H(t *testing.T) {
	t.Run("Add H to circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 1,
			gates:     []Gate{},
		}
		expected := Gate{
			Matrix: H,
			name:   HADAMARD,
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
			Matrix: Matrix{
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

func TestQuantumCircuit_X(t *testing.T) {
	t.Run("Add X to single qubit circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits:    1,
			gates:        []Gate{},
			compileValid: false,
			compiled:     Gate{},
		}
		qc.X(0)
		expected := createX(0, 1)
		if len(qc.gates) != 1 || !qc.gates[0].Equals(expected, StdEpsilon) {
			t.Errorf("qc.gates = %v, expected %v", qc.gates, []Gate{*expected})
		}
	})

	t.Run("Add X to one out of three qubits", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits:    3,
			gates:        []Gate{},
			compileValid: false,
			compiled:     Gate{},
		}
		qc.X(1)
		expected := createX(1, 3)
		if len(qc.gates) != 1 || !qc.gates[0].Equals(expected, StdEpsilon) {
			t.Errorf("qc.gates = %v, expected %v", qc.gates, []Gate{*expected})
		}
	})
}

func TestQuantumCircuit_CX(t *testing.T) {
	t.Run("Add CX to circuit", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 2,
			gates:     []Gate{},
		}
		expected := Gate{
			Matrix: Matrix{
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
			Matrix: Matrix{
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
	t.Run("Compilation on no gates", func(t *testing.T) {
		qc := QuantumCircuit{
			numQubits:    2,
			gates:        []Gate{},
			compileValid: false,
			compiled:     Gate{},
		}
		expected := QuantumCircuit{
			numQubits:    2,
			gates:        []Gate{},
			compileValid: true,
			compiled: Gate{
				Matrix: Matrix{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data: []complex128{
						1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						0, 0, 0, 1,
					},
				},
				name: 1,
			},
		}

		qc.Compile()

		if !qc.compiled.Equals(&expected.compiled, StdEpsilon) {
			t.Errorf("Compile() got %v, expected %v", qc.compiled, expected.compiled)
		}
	})

	t.Run("Compile CX on two qubits", func(t *testing.T) {
		qc := &QuantumCircuit{
			numQubits: 2,
			gates:     []Gate{*createCX(0, 1, 2)},
		}
		expected := *createCX(0, 1, 2)
		qc.Compile()
		if !qc.compiled.Matrix.Equals(expected.Matrix, StdEpsilon) {
			t.Errorf("Compiling CX gate makes circuit %v, expected %v", qc.compiled, expected)
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
		expectedMatrix := Matrix{
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
		if !qc.compiled.Matrix.Equals(expectedMatrix, StdEpsilon) {
			t.Errorf("Compiling H(0) CX gate makes circuit %v, expected %v", qc.compiled.Matrix, expectedMatrix)
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
		register []Ket
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
						Matrix: I,
						name:   WIRE,
					},
				},
				compiled: Gate{
					Matrix: I,
					name:   WIRE,
				},
			},
			args: args{
				register: []Ket{ZeroKet},
			},
			want: &QuantumCircuitExecution{
				in:  []Ket{ZeroKet},
				out: ColVec(ZeroKet),
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
				register: []Ket{ZeroKet, OneKet},
			},
			want: &QuantumCircuitExecution{
				in: []Ket{ZeroKet, OneKet},
				out: ColVec{
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
				register: []Ket{OneKet, HPlusKet, HMinusKet},
			},
			want: &QuantumCircuitExecution{
				in: []Ket{OneKet, HPlusKet, HMinusKet},
				out: ColVec{
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
			if got := qc.Exec(tt.args.register); !Matrix(got.out).Equals(Matrix(tt.want.out), StdEpsilon) {
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

		qc.Exec([]Ket{ZeroKet})
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

func TestQuantumCircuitExecution_MeasureProbability(t *testing.T) {
	type fields struct {
		in       []Ket
		register ColVec
		out      ColVec
	}
	type args struct {
		basis []Ket
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   float64
	}{
		{
			name: "Measure same ket as output",
			fields: fields{
				in:       []Ket{OneKet},
				register: ColVec(OneKet),
				out:      ColVec(OneKet),
			},
			args: args{
				basis: []Ket{OneKet},
			},
			want: 1,
		},
		{
			name: "Measure orthogonal ket as output",
			fields: fields{
				in:       []Ket{OneKet},
				register: ColVec(OneKet),
				out:      ColVec(OneKet),
			},
			args: args{
				basis: []Ket{ZeroKet},
			},
			want: 0,
		},
		{
			name: "Measure H+ state in standard basis",
			fields: fields{
				in:       []Ket{HPlusKet},
				register: ColVec(HPlusKet),
				out:      ColVec(HPlusKet),
			},
			args: args{
				basis: []Ket{OneKet},
			},
			want: 0.5,
		},
		{
			name: "Measure H+ state across two qubits",
			fields: fields{
				in:       []Ket{ZeroKet, ZeroKet},
				register: KronKets([]Ket{ZeroKet, ZeroKet}),
				out: ColVec{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0.5, 0.5, 0.5, 0.5},
				},
			},
			args: args{
				basis: []Ket{OneKet, ZeroKet},
			},
			want: 0.25,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qce := &QuantumCircuitExecution{
				in:       tt.fields.in,
				register: tt.fields.register,
				out:      tt.fields.out,
			}
			if got := qce.MeasureProbability(tt.args.basis); !floatEqual(got, tt.want, FloatEpsilon) {
				t.Errorf("MeasureProbability() = %v, want %v", got, tt.want)
			}
		})
	}

	t.Run("Panic on bad basis size", func(t *testing.T) {
		qce := &QuantumCircuitExecution{
			in: []Ket{OneKet, ZeroKet},
			register: ColVec{
				Rows:   4,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{0, 0, 1, 0},
			},
			out: ColVec{
				Rows:   4,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{0, 0, 1, 0},
			},
		}
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("Does not panic on not enough qubits")
			}
		}()
		qce.MeasureProbability([]Ket{OneKet})
	})
}

func floatEqual(a, b, epsilon float64) bool {
	return math.Abs(b-a) < epsilon
}

func floatsEqual(a, b []float64, epsilon float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !floatEqual(a[i], b[i], epsilon) {
			return false
		}
	}
	return true
}

func TestQuantumCircuitExecution_MeasureProbabilities(t *testing.T) {
	type fields struct {
		in       []Ket
		register ColVec
		out      ColVec
	}
	tests := []struct {
		name   string
		fields fields
		want   []float64
	}{
		{
			name: "Measure zero ket outcomes",
			fields: fields{
				in:       []Ket{ZeroKet},
				register: ColVec(ZeroKet),
				out:      ColVec(ZeroKet),
			},
			want: []float64{1, 0},
		},
		{
			name: "Measure [0 0 1 0] as |10>",
			fields: fields{
				in: []Ket{ZeroKet, OneKet},
				register: ColVec{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0, 1, 0, 0},
				},
				out: ColVec{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0, 0, 1, 0},
				},
			},
			want: []float64{0, 0, 1, 0},
		},
		{
			name: "Measure two qubit hadamard state",
			fields: fields{
				in: []Ket{ZeroKet, ZeroKet},
				register: ColVec{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1, 0, 0, 0},
				},
				out: ColVec{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0.5, 0.5, 0.5, 0.5},
				},
			},
			want: []float64{0.25, 0.25, 0.25, 0.25},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qce := &QuantumCircuitExecution{
				in:       tt.fields.in,
				register: tt.fields.register,
				out:      tt.fields.out,
			}
			if got := qce.MeasureProbabilities(); !floatsEqual(got, tt.want, FloatEpsilon) {
				t.Errorf("MeasureProbabilities() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestQuantumCircuit_AddCircuit(t *testing.T) {
	t.Run("Add empty circuit", func(t *testing.T) {
		qc := QuantumCircuit{
			numQubits:    2,
			gates:        []Gate{*createH([]int{0, 1}, 2)},
			compileValid: true,
			compiled:     *createH([]int{0, 1}, 2),
		}
		toAdd := QuantumCircuit{
			numQubits:    2,
			gates:        []Gate{},
			compileValid: false,
			compiled:     Gate{},
		}
		expected := QuantumCircuit{
			numQubits:    2,
			gates:        []Gate{*createH([]int{0, 1}, 2), *createWire(2)},
			compileValid: false,
			compiled:     *createH([]int{0, 1}, 2),
		}

		qc.AddCircuit(toAdd)

		if qc.numQubits != expected.numQubits || len(qc.gates) != len(expected.gates) || qc.compileValid != expected.compileValid {
			t.Errorf("AddCircuit got %v, wanted %v", qc, expected)
		}
	})
	/*
	     {2 [{{4 4 4 [(0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001-0i)]} 0}
	         {{4 4 4 [(1+0i) (0+0i) (0+0i) (0+0i) (0+0i) (1+0i) (0+0i) (0+0i) (0+0i) (0+0i) (1+0i) (0+0i) (0+0i) (0+0i) (0+0i) (1+0i)]} 1}] false {{4 4 4 [(0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001-0i)]} 0}},

	     {2 [{{4 4 4 [(0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001-0i)]} 0}]
	   false {{4 4 4 [(0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001+0i) (-0.5000000000000001+0i) (-0.5000000000000001+0i) (0.5000000000000001-0i)]} 0}}
	*/

}

func TestQuantumCircuitExecution_MeasureProbabilityOn(t *testing.T) {
	type fields struct {
		in       []Ket
		register ColVec
		out      ColVec
	}
	type args struct {
		qubit int
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   float64
	}{
		{
			name: "Probability of measuring a single off qubit",
			fields: fields{
				in:       []Ket{ZeroKet},
				register: ColVec(ZeroKet),
				out:      ColVec(ZeroKet),
			},
			args: args{qubit: 0},
			want: 0,
		},
		{
			name: "Probability of measuring a single on qubit",
			fields: fields{
				in:       []Ket{OneKet},
				register: ColVec(OneKet),
				out:      ColVec(OneKet),
			},
			args: args{qubit: 0},
			want: 1,
		},
		{
			name: "Probability of measuring the first qubit ON in the equal superposition state in 3 qubits",
			fields: fields{
				in:       []Ket{ZeroKet, ZeroKet, ZeroKet},
				register: KronKets([]Ket{ZeroKet, ZeroKet, ZeroKet}),
				out: ColVec{
					Rows:   8,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339},
				},
			},
			args: args{qubit: 1},
			want: 0.5,
		},
		{
			name: "Probability of measuring 1 in first qubit of four qubit state",
			fields: fields{
				in:       []Ket{OneKet, OneKet, OneKet, OneKet},
				register: KronKets([]Ket{OneKet, OneKet, OneKet, OneKet}),
				out: ColVec{
					Rows:   16,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.707106781, 0.707106781},
				},
			},
			args: args{qubit: 0},
			want: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qce := &QuantumCircuitExecution{
				in:       tt.fields.in,
				register: tt.fields.register,
				out:      tt.fields.out,
			}
			if got := qce.MeasureProbabilityOn(tt.args.qubit); !floatEqual(got, tt.want, FloatEpsilon) {
				t.Errorf("MeasureProbabilityOn() = %v, want %v", got, tt.want)
			}
		})
	}
}
