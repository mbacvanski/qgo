package main

import (
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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mul(tt.args.a, tt.args.b); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("mul() = %v, want %v", got, tt.want)
			}
		})
	}
}
