package main

import (
	"gonum.org/v1/gonum/blas/cblas128"
	"math"
	"testing"
)

const StdEpsilon = 0.00000001 + 0.00000001i

func TestGate_Name(t *testing.T) {
	type fields struct {
		General cblas128.General
		name    GateName
	}
	tests := []struct {
		name   string
		fields fields
		want   string
	}{
		{
			name: "Single Hadamard Gate",
			fields: fields{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2, 1 / math.Sqrt2, -1 / math.Sqrt2},
				},
				name: HADAMARD,
			},
			want: "Hadamard",
		},
		{
			name: "Single Wire (Identity Gate)",
			fields: fields{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 0, 0, 1},
				},
				name: IDENTITY,
			},
			want: "Identity",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Gate{
				General: tt.fields.General,
				name:    tt.fields.name,
			}
			if got := g.Name(); got != tt.want {
				t.Errorf("Name() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCreateH(t *testing.T) {
	type args struct {
		qubits    []int
		numQubits int
	}
	tests := []struct {
		name string
		args args
		want *Gate
	}{
		{
			name: "Single qubit Hadamard",
			args: args{
				qubits:    []int{0},
				numQubits: 1,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2, 1 / math.Sqrt2, -1 / math.Sqrt2},
				},
				name: HADAMARD,
			},
		},
		{
			name: "Two qubit Hadamard, next to each other",
			args: args{
				qubits:    []int{0, 1},
				numQubits: 2,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data: []complex128{
						0.5, 0.5, 0.5, 0.5,
						0.5, -0.5, 0.5, -0.5,
						0.5, 0.5, -0.5, -0.5,
						0.5, -0.5, -0.5, 0.5,
					},
				},
				name: HADAMARD,
			},
		},
		{
			name: "Three qubit Hadamard, next to each other",
			args: args{
				qubits:    []int{0, 1, 2},
				numQubits: 3,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   8,
					Cols:   8,
					Stride: 8,
					Data: []complex128{0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339, 0.35355339,
						0.35355339, -0.35355339, 0.35355339, -0.35355339, 0.35355339, -0.35355339, 0.35355339, -0.35355339,
						0.35355339, 0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339, -0.35355339, -0.35355339,
						0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339, -0.35355339, -0.35355339, 0.35355339,
						0.35355339, 0.35355339, 0.35355339, 0.35355339, -0.35355339, -0.35355339, -0.35355339, -0.35355339,
						0.35355339, -0.35355339, 0.35355339, -0.35355339, -0.35355339, 0.35355339, -0.35355339, 0.35355339,
						0.35355339, 0.35355339, -0.35355339, -0.35355339, -0.35355339, -0.35355339, 0.35355339, 0.35355339,
						0.35355339, -0.35355339, -0.35355339, 0.35355339, -0.35355339, 0.35355339, 0.35355339, -0.35355339},
				},
				name: HADAMARD,
			},
		},
		{
			name: "H I",
			args: args{
				qubits:    []int{0},
				numQubits: 2,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data: []complex128{
						0.70710678, 0, 0.70710678, 0,
						0, 0.70710678, 0, 0.70710678,
						0.70710678, 0, -0.70710678, -0,
						0, 0.70710678, -0, -0.70710678,
					},
				},
				name: HADAMARD,
			},
		},
		{
			name: "I H I",
			args: args{
				qubits:    []int{1},
				numQubits: 3,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   8,
					Cols:   8,
					Stride: 8,
					Data: []complex128{
						0.70710678, 0, 0.70710678, 0, 0, 0, 0, 0,
						0, 0.70710678, 0, 0.70710678, 0, 0, 0, 0,
						0.70710678, 0, -0.70710678, -0, 0, 0, -0, -0,
						0, 0.70710678, -0, -0.70710678, 0, 0, -0, -0,
						0, 0, 0, 0, 0.70710678, 0, 0.70710678, 0,
						0, 0, 0, 0, 0, 0.70710678, 0, 0.70710678,
						0, 0, -0, -0, 0.70710678, 0, -0.70710678, -0,
						0, 0, -0, -0, 0, 0.70710678, -0, -0.70710678},
				},
				name: HADAMARD,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := createH(tt.args.qubits, tt.args.numQubits); !got.Equals(tt.want, StdEpsilon) {
				t.Errorf("createH() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGate_Equals(t *testing.T) {
	type args struct {
		b       *Gate
		epsilon complex128
	}
	sameGate := Gate{
		General: cblas128.General{
			Rows:   2,
			Cols:   2,
			Stride: 2,
			Data:   []complex128{1, 1, 1, 1},
		},
		name: 15,
	}

	tests := []struct {
		name   string
		fields Gate
		args   args
		want   bool
	}{
		{
			name:   "Same Gate",
			fields: sameGate,
			args: args{
				b:       &sameGate,
				epsilon: 0,
			},
			want: true,
		},
		{
			name: "Empty Gates are Equal",
			fields: Gate{
				General: cblas128.General{
					Rows:   0,
					Cols:   0,
					Stride: 0,
					Data:   nil,
				},
				name: HADAMARD,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   0,
						Cols:   0,
						Stride: 0,
						Data:   nil,
					},
					name: HADAMARD,
				},
				epsilon: 0,
			},
			want: true,
		},

		{
			name: "Name Difference",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 2, 3, 4},
				},
				name: HADAMARD,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   2,
						Cols:   2,
						Stride: 2,
						Data:   []complex128{1, 2, 3, 4},
					},
					name: IDENTITY,
				},
				epsilon: 0.000001 + 0.000001i,
			},
			want: false,
		},
		{
			name: "Different row count",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1, 2},
				},
				name: 0,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   1,
						Cols:   1,
						Stride: 1,
						Data:   []complex128{1, 2},
					},
					name: 0,
				},
				epsilon: 0,
			},
			want: false,
		},
		{
			name: "Different col count",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   4,
					Stride: 4,
					Data:   []complex128{1, 2, 3, 4, 5, 6, 7, 8},
				},
				name: 0,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   2,
						Cols:   3,
						Stride: 4,
						Data:   []complex128{1, 2, 3, 4, 5, 6, 7, 8},
					},
					name: 0,
				},
				epsilon: 0,
			},
			want: false,
		},
		{
			name: "Different stride",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 2, 3, 4},
				},
				name: 0,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   2,
						Cols:   2,
						Stride: 3,
						Data:   []complex128{1, 2, 3, 4},
					},
					name: 0,
				},
				epsilon: 0,
			},
			want: false,
		},
		{
			name: "Different data length",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 2, 3, 4},
				},
				name: 0,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   2,
						Cols:   2,
						Stride: 2,
						Data:   []complex128{1, 2, 3, 4, 5},
					},
					name: 0,
				},
				epsilon: 0,
			},
			want: false,
		},
		{
			name: "Difference in Real > Epsilon -> Not Equal",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1.01 + 1i, 1 + 2i, 3 - 2i, 2 + 0.2i},
				},
				name: 0,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   2,
						Cols:   2,
						Stride: 2,
						Data:   []complex128{1.025 + 1i, 1 + 2i, 3 - 2i, 2 + 0.2i},
					},
					name: 0,
				},
				epsilon: 0.01,
			},
			want: false,
		},
		{
			name: "Difference in Imaginary > Epsilon -> Not Equal",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1.01 + 1i, 1 + 2i, 3 - 2i, 2 + 0.2i},
				},
				name: 0,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   2,
						Cols:   2,
						Stride: 2,
						Data:   []complex128{1.01 + 1i, 1 + 2.01i, 3 - 2i, 2 + 0.2i},
					},
					name: 0,
				},
				epsilon: 0.009i,
			},
			want: false,
		},
		{
			name: "Difference in Real < Epsilon -> Equal",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1.01 + 1i, 1 + 2i, 3 - 2i, 2 + 0.2i},
				},
				name: 0,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   2,
						Cols:   2,
						Stride: 2,
						Data:   []complex128{1.025 + 1i, 1 + 2i, 3 - 2i, 2 + 0.2i},
					},
					name: 0,
				},
				epsilon: 0.02,
			},
			want: true,
		},
		{
			name: "Difference in Imaginary < Epsilon -> Equal",
			fields: Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1.01 + 1i, 1 + 2i, 3 - 2i, 2 + 0.2i},
				},
				name: 0,
			},
			args: args{
				b: &Gate{
					General: cblas128.General{
						Rows:   2,
						Cols:   2,
						Stride: 2,
						Data:   []complex128{1.01 + 1i, 1 + 2.01i, 3 - 2i, 2 + 0.2i},
					},
					name: 0,
				},
				epsilon: 0.015i,
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.fields.Equals(tt.args.b, tt.args.epsilon); got != tt.want {
				t.Errorf("Equals() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_createX(t *testing.T) {
	type args struct {
		qubit     int
		numQubits int
	}
	tests := []struct {
		name string
		args args
		want *Gate
	}{
		{
			name: "Single Qubit X",
			args: args{
				qubit:     0,
				numQubits: 1,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{0, 1, 1, 0},
				},
				name: PAULIX,
			},
		},
		{
			name: "X I",
			args: args{
				qubit:     0,
				numQubits: 2,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data: []complex128{
						0, 0, 1, 0,
						0, 0, 0, 1,
						1, 0, 0, 0,
						0, 1, 0, 0,
					},
				},
				name: PAULIX,
			},
		},
		{
			name: "I X",
			args: args{
				qubit:     1,
				numQubits: 2,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data: []complex128{
						0, 1, 0, 0,
						1, 0, 0, 0,
						0, 0, 0, 1,
						0, 0, 1, 0,
					},
				},
				name: PAULIX,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := createX(tt.args.qubit, tt.args.numQubits); !got.Equals(tt.want, StdEpsilon) {
				t.Errorf("createX() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_createCX(t *testing.T) {
	type args struct {
		control   int
		target    int
		numQubits int
	}
	tests := []struct {
		name string
		args args
		want *Gate
	}{
		{
			name: "Standard CX",
			args: args{
				control:   0,
				target:    1,
				numQubits: 2,
			},
			want: &Gate{
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
			},
		},
		{
			name: "Upside down CX",
			args: args{
				control:   1,
				target:    0,
				numQubits: 2,
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data: []complex128{
						1, 0, 0, 0,
						0, 0, 0, 1,
						0, 0, 1, 0,
						0, 1, 0, 0,
					},
				},
				name: CX,
			},
		},
		{
			name: "4 Qubits CX",
			args: args{
				control:   2,
				target:    3,
				numQubits: 4,
			},
			want: &Gate{
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
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := createCX(tt.args.control, tt.args.target, tt.args.numQubits); !got.Equals(tt.want, StdEpsilon) {
				t.Errorf("createCX() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCombine(t *testing.T) {
	type args struct {
		name     GateName
		matrices []Gate
	}
	tests := []struct {
		name string
		args args
		want *Gate
	}{
		{
			name: "H H",
			args: args{
				name:     IDENTITY,
				matrices: []Gate{*createH([]int{0}, 1), *createH([]int{0}, 1)},
			},
			want: &Gate{
				General: I,
				name:    IDENTITY,
			},
		},
		{
			name: "H(0) CX",
			args: args{
				name:     55,
				matrices: []Gate{*createH([]int{0}, 2), *createCX(0, 1, 2)},
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data: []complex128{
						0.70710678, 0, 0.70710678, 0,
						0, 0.70710678, 0, 0.70710678,
						0, 0.70710678, 0, -0.70710678,
						0.70710678, 0, -0.70710678, 0,
					},
				},
				name: 55,
			},
		},

		{
			name: "CX H",
			args: args{
				name:     55,
				matrices: []Gate{*createCX(0, 1, 2), *createH([]int{0}, 2)},
			},
			want: &Gate{
				General: cblas128.General{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data: []complex128{
						0.70710678, 0, 0, 0.70710678,
						0, 0.70710678, 0.70710678, 0,
						0.70710678, 0, 0, -0.70710678,
						0, 0.70710678, -0.70710678, 0,
					},
				},
				name: 55,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Combine(tt.args.name, tt.args.matrices...); !got.Equals(tt.want, StdEpsilon) {
				t.Errorf("Combine() = %v, want %v", got, tt.want)
			}
		})
	}
}
