package main

import (
	"math"
	"reflect"
	"testing"
)

func Test_FormatMat(t *testing.T) {
	tests := []struct {
		name     string
		mat      Matrix
		expected string
	}{
		{"Pauli X matrix",
			Matrix{
				Rows:   2,
				Cols:   2,
				Stride: 2,
				Data:   []complex128{0, 1, 1, 0},
			},
			"[0, 1, \n 1, 0]",
		},

		{"Pauli Y matrix",
			Matrix{
				Rows:   2,
				Cols:   2,
				Stride: 2,
				Data:   []complex128{0, -1i, 1i, 0},
			},
			"[0, -i, \n i, 0]",
		},

		{"C-Z (CNOT) matrix",
			Matrix{
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
			mat: Matrix{
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
		a Matrix
		b Matrix
	}
	tests := []struct {
		name string
		args args
		want *Matrix
	}{
		{
			name: "1x1 Kronecker",
			args: args{
				a: Matrix{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{4},
				},
				b: Matrix{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{3},
				},
			},
			want: &Matrix{
				Rows:   1,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{12},
			},
		},
		{
			name: "2x2 Kronecker with imaginary numbers",
			args: args{
				a: Matrix{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 2i, 5, 3i},
				},
				b: Matrix{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{3 + 2i, 1 + 5i, 2 + 1i, 7},
				},
			},
			want: &Matrix{
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
				a: Matrix{
					Rows:   2,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1 / math.Sqrt2, 2 / math.Sqrt2},
				},
				b: Matrix{
					Rows:   2,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1i, 0},
				},
			},
			want: &Matrix{
				Rows:   4,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1i / math.Sqrt2, 0, 2i / math.Sqrt2, 0},
			},
		},
		{
			name: "X I",
			args: args{
				a: Matrix{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{0, 1, 1, 0},
				},
				b: Matrix{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 0, 0, 1},
				},
			},
			want: &Matrix{
				Rows:   4,
				Cols:   4,
				Stride: 4,
				Data:   []complex128{0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0},
			},
		},
		{
			name: "H I",
			args: args{
				a: Matrix{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2, 1 / math.Sqrt2, -1 / math.Sqrt2},
				},
				b: Matrix{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 0, 0, 1},
				},
			},
			want: &Matrix{
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
		a *Matrix
		b *Matrix
	}
	tests := []struct {
		name string
		args args
		want *Matrix
	}{
		{
			name: "1x1 Matrix",
			args: args{
				a: &Matrix{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{3 + 1i},
				},
				b: &Matrix{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{4 + 5i},
				},
			},
			want: &Matrix{
				Rows:   1,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{7 + 19i},
			},
		},
		{
			name: "4x4 x 4x1 Multiplication",
			args: args{
				a: &Matrix{
					Rows:   4,
					Cols:   4,
					Stride: 4,
					Data:   []complex128{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0},
				},
				b: &Matrix{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{0, 0, 1, 0},
				},
			},
			want: &Matrix{
				Rows:   4,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{0, 0, 0, 1},
			},
		},
		{
			name: "Ket * Bra",
			args: args{
				a: &Matrix{ // Ket of 0
					Rows:   2,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1, 0},
				},
				b: &Matrix{ // Bra of 0
					Rows:   1,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 0},
				},
			},
			want: &Matrix{
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
		a *Matrix
		b *Matrix
	}
	tests := []struct {
		name string
		args args
		want *Matrix
	}{
		{
			name: "Add 2x2",
			args: args{
				a: &Matrix{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{1, 2, 3, 4},
				},
				b: &Matrix{
					Rows:   2,
					Cols:   2,
					Stride: 2,
					Data:   []complex128{5, 6, 7, 8},
				},
			},
			want: &Matrix{
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

	add(&Matrix{
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{1, 2, 3, 4},
	}, &Matrix{
		Rows:   1,
		Cols:   5,
		Stride: 3,
		Data:   []complex128{1, 2, 4},
	})
}

func TestNewColumnVec(t *testing.T) {
	type args struct {
		mat Matrix
	}
	tests := []struct {
		name string
		args args
		want ColVec
	}{
		{
			name: "No change to existing column vec",
			args: args{
				mat: Matrix{
					Rows:   4,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1, 2, 3, 4},
				},
			},
			want: ColVec{
				Rows:   4,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1, 2, 3, 4},
			},
		},
		{
			name: "Move 2x3 matrix to column",
			args: args{
				mat: Matrix{
					Rows:   2,
					Cols:   3,
					Stride: 3,
					Data:   []complex128{1, 2, 3, 4, 5, 6},
				},
			},
			want: ColVec{
				Rows:   6,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1, 2, 3, 4, 5, 6},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewColumnVec(tt.args.mat); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewColumnVec() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_IsColumnVec(t *testing.T) {
	tests := []struct {
		name string
		m    Matrix
		want bool
	}{
		{
			name: "Rectangle matrix not a column vec",
			m: Matrix{
				Rows:   2,
				Cols:   3,
				Stride: 3,
				Data:   []complex128{1, 2, 3, 4, 5, 6},
			},
			want: false,
		},
		{
			name: "Column matrix is a column vector",
			m: Matrix{
				Rows:   5,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1, 2, 3, 4, 5},
			},
			want: true,
		},
		{
			name: "Row matrix not a column vector",
			m: Matrix{
				Rows:   1,
				Cols:   5,
				Stride: 5,
				Data:   []complex128{1, 2, 3, 4, 5},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.IsColumnVec(); got != tt.want {
				t.Errorf("IsColumnVec() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_IsRowVec(t *testing.T) {
	tests := []struct {
		name string
		m    Matrix
		want bool
	}{
		{
			name: "Rectangle matrix not a column vec",
			m: Matrix{
				Rows:   2,
				Cols:   3,
				Stride: 3,
				Data:   []complex128{1, 2, 3, 4, 5, 6},
			},
			want: false,
		},
		{
			name: "Row matrix is a row vector",
			m: Matrix{
				Rows:   1,
				Cols:   5,
				Stride: 5,
				Data:   []complex128{1, 2, 3, 4, 5},
			},
			want: true,
		},
		{
			name: "Column matrix is not a row vector",
			m: Matrix{
				Rows:   5,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1, 2, 3, 4, 5},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.IsRowVec(); got != tt.want {
				t.Errorf("IsKet() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewKet(t *testing.T) {
	type args struct {
		mat Matrix
	}
	tests := []struct {
		name string
		args args
		want Ket
	}{
		{
			name: "Length 2 Column Vec to Ket",
			args: args{
				mat: Matrix{
					Rows:   2,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{1, 2},
				},
			},
			want: Ket{
				Rows:   2,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1, 2},
			},
		},
		{
			name: "Length 1 Column Vec to Ket",
			args: args{
				mat: Matrix{
					Rows:   1,
					Cols:   1,
					Stride: 1,
					Data:   []complex128{5},
				},
			},
			want: Ket{
				Rows:   2,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{5, 0},
			},
		},
		{
			name: "Length 0 Column Vec to Ket",
			args: args{
				mat: Matrix{
					Rows:   0,
					Cols:   0,
					Stride: 0,
					Data:   []complex128{},
				},
			},
			want: Ket{
				Rows:   2,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{0, 0},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewKet(tt.args.mat); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewKet() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewRowVec(t *testing.T) {
	type args struct {
		mat Matrix
	}
	tests := []struct {
		name string
		args args
		want RowVec
	}{

		{
			name: "No change to existing row vec",
			args: args{
				mat: Matrix{
					Rows:   1,
					Cols:   4,
					Stride: 4,
					Data:   []complex128{1, 2, 3, 4},
				},
			},
			want: RowVec{
				Rows:   1,
				Cols:   4,
				Stride: 4,
				Data:   []complex128{1, 2, 3, 4},
			},
		},
		{
			name: "Move 2x3 matrix to row",
			args: args{
				mat: Matrix{
					Rows:   2,
					Cols:   3,
					Stride: 3,
					Data:   []complex128{1, 2, 3, 4, 5, 6},
				},
			},
			want: RowVec{
				Rows:   1,
				Cols:   6,
				Stride: 6,
				Data:   []complex128{1, 2, 3, 4, 5, 6},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewRowVec(tt.args.mat); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewRowVec() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestColVec_IsBra(t *testing.T) {
	tests := []struct {
		name string
		c    ColVec
		want bool
	}{
		{
			name: "Col Vec with 2 Rows is Bra",
			c: ColVec{
				Rows:   2,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1, 2},
			},
			want: true,
		},
		{
			name: "Column Vec with 3 Rows is not Bra",
			c: ColVec{
				Rows:   3,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1, 2, 3},
			},
			want: false,
		},
		{
			name: "Column Vec with 1 Row is not Bra",
			c: ColVec{
				Rows:   1,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.c.IsBra(); got != tt.want {
				t.Errorf("IsBra() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRowVec_IsKet(t *testing.T) {
	tests := []struct {
		name string
		v    RowVec
		want bool
	}{
		{
			name: "Row Vec with 2 Cols is Ket",
			v: RowVec{
				Rows:   1,
				Cols:   2,
				Stride: 2,
				Data:   []complex128{1, 2},
			},
			want: true,
		},
		{
			name: "Row Vec with 3 Cols is not Bra",
			v: RowVec{
				Rows:   1,
				Cols:   3,
				Stride: 3,
				Data:   []complex128{1, 2, 3},
			},
			want: false,
		},
		{
			name: "Row Vec with 1 Row is not Bra",
			v: RowVec{
				Rows:   1,
				Cols:   1,
				Stride: 1,
				Data:   []complex128{1},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.v.IsKet(); got != tt.want {
				t.Errorf("IsKet() = %v, want %v", got, tt.want)
			}
		})
	}
}
