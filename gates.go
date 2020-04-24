package main

import (
	"gonum.org/v1/gonum/blas/cblas128"
	"math"
)

type GateName int

const (
	HADAMARD = iota
	IDENTITY = iota
)

func (g *Gate) Name() string {
	return [...]string{"Hadamard", "Identity"}[g.name]
}

type Gate struct {
	cblas128.General
	name GateName
}

func (g *Gate) Equals(b *Gate, epsilon complex128) bool {
	if g == b {
		return true
	}
	if g.name != b.name {
		return false
	}
	if g.General.Rows != b.General.Rows || g.General.Cols != b.General.Cols || g.General.Stride != b.General.Stride {
		return false
	}
	if len(g.General.Data) != len(b.General.Data) {
		return false
	}
	for i := range g.General.Data {
		adata := g.General.Data[i]
		bdata := b.General.Data[i]
		if math.Abs(real(adata)-real(bdata)) > real(epsilon) ||
			math.Abs(imag(adata)-imag(bdata)) > imag(epsilon) {
			return false
		}
	}
	return true
}

var (
	H = Gate{
		General: cblas128.General{
			Rows:   2,
			Cols:   2,
			Stride: 2,
			Data:   []complex128{1 / math.Sqrt2, 1 / math.Sqrt2, 1 / math.Sqrt2, -1 / math.Sqrt2},
		},
		name: HADAMARD,
	}

	I = Gate{
		General: cblas128.General{
			Rows:   2,
			Cols:   2,
			Stride: 2,
			Data:   []complex128{1, 1, 1, 1},
		},
		name: IDENTITY,
	}
)

func CreateH(qubits []int) *Gate {
	var gate cblas128.General
	if qubits[0] == 0 {
		gate = I.General
	} else if qubits[0] == 1 {
		gate = H.General
	}

	for i := 1; i < len(qubits); i++ {
		if qubits[i] == 0 {
			gate = *kronecker(gate, I.General)
		} else if qubits[i] == 1 {
			gate = *kronecker(gate, H.General)
		}
	}

	return &Gate{
		General: gate,
		name:    HADAMARD,
	}
}
