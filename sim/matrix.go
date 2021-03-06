package sim

import (
	"fmt"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/cblas128"
	"math"
)

type Matrix cblas128.General
type ColVec Matrix
type Ket ColVec
type RowVec Matrix
type Bra RowVec

// Coerces a matrix into column vector format
func NewColVec(mat Matrix) ColVec {
	return ColVec{
		Rows:   mat.Rows * mat.Cols,
		Cols:   1,
		Stride: 1,
		Data:   mat.Data,
	}
}

func (c *ColVec) Size() int {
	return c.Rows
}

// Is the given matrix a column vector?
func (a *Matrix) IsColumnVec() bool {
	return a.Cols == 1
}

// Coerces a matrix into Ket format
// Truncates all data past the first two elements
func NewKet(mat Matrix) Ket {
	ket := Ket{
		Rows:   2,
		Cols:   1,
		Stride: 1,
		Data:   make([]complex128, 2),
	}
	if len(mat.Data) > 0 {
		ket.Data[0] = mat.Data[0]
	}
	if len(mat.Data) > 1 {
		ket.Data[1] = mat.Data[1]
	}
	return ket
}

// Is the given column vector a ket?
func (v *RowVec) IsKet() bool {
	return v.Cols == 2
}

// Coerces a matrix into row vector format
func NewRowVec(mat Matrix) RowVec {
	return RowVec{
		Rows:   1,
		Cols:   mat.Rows * mat.Cols,
		Stride: mat.Rows * mat.Cols,
		Data:   mat.Data,
	}
}

// Is the given matrix a row vector?
func (a *Matrix) IsRowVec() bool {
	return a.Rows == 1
}

// Coerces a matrix into Bra format
// Truncates all data past the first two elements
func NewBra(mat Matrix) Bra {
	return Bra{
		Rows:   1,
		Cols:   2,
		Stride: 2,
		Data:   []complex128{mat.Data[0], mat.Data[1]},
	}
}

// Is the given row vector a bra?
func (c *ColVec) IsBra() bool {
	return c.Rows == 2
}

// Computes the dot product of a matrix with another
func (c *ColVec) Dotp(b ColVec) complex128 {
	matA := cblas128.Vector{
		N:    c.Rows * c.Cols,
		Inc:  c.Stride / c.Cols,
		Data: c.Data,
	}

	matB := cblas128.Vector{
		N:    b.Rows * b.Cols,
		Inc:  c.Stride / c.Cols,
		Data: b.Data,
	}

	return cblas128.Dotc(matA, matB)
}

// Creates the Kronecker product of matrices M and B
// Returns a reference to the matrix containing the result
func (a Matrix) Kronecker(b Matrix) *Matrix {
	ar, ac := a.Rows, a.Cols
	br, bc := b.Rows, b.Cols
	out := Matrix{
		Rows:   ar * br,
		Cols:   ac * bc,
		Stride: ac * bc,
		Data:   make([]complex128, (ar*br)*(ac*bc)),
	}

	for rowA := 0; rowA < ar; rowA++ {
		for colA := 0; colA < ac; colA++ {
			indexA := rowA*a.Stride + colA

			for rowB := 0; rowB < br; rowB++ {
				for colB := 0; colB < bc; colB++ {
					indexB := rowB*b.Stride + colB

					outRow := rowA*br + rowB
					outCol := colA*bc + colB
					indexOut := outRow*out.Stride + outCol
					out.Data[indexOut] = a.Data[indexA] * b.Data[indexB]
				}
			}
		}
	}

	return &out
}

// Does standard matrix multiplication of A and B.
// Returns a reference to the matrix containing the result
func (a Matrix) Mul(b Matrix) *Matrix {
	var c = Matrix{
		Rows:   a.Rows,
		Cols:   b.Cols,
		Stride: b.Cols,
		Data:   make([]complex128, a.Rows*b.Cols),
	}
	cblas128.Gemm(blas.NoTrans, blas.NoTrans, 1, cblas128.General(a), cblas128.General(b), 0, cblas128.General(c))
	return &c
}

func (a Matrix) Add(b Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Stride != b.Stride {
		panic("Cannot Add matrices of differing dimensions")
	}

	out := Matrix{
		Rows:   b.Rows,
		Cols:   b.Cols,
		Stride: b.Stride,
		Data:   make([]complex128, b.Rows*b.Cols),
	}

	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}

	return &out
}

func FormatMat(X Matrix) string {
	formatNum := func(c complex128) string {
		formatImag := func(f float64) string {
			var imagComp string
			if f == 1 {
				imagComp += "i"
			} else if f == -1 {
				imagComp += "-i"
			} else {
				imagComp += fmt.Sprintf("%vi", f)
			}
			return imagComp
		}

		var s string
		if imag(c) != 0 && real(c) != 0 {
			s += fmt.Sprintf("(%v+%v)", real(c), formatImag(imag(c)))
		} else if imag(c) == 0 {
			s += fmt.Sprintf("%v", real(c))
		} else if real(c) == 0 {
			s += formatImag(imag(c))
		}
		return s
	}

	var str string

	str += "["
	for r := 0; r < X.Rows; r++ {
		for c := 0; c < X.Cols; c++ {
			num := X.Data[(X.Cols*r)+c]
			str += formatNum(num)

			// Put comma afterwards except last item
			if c != X.Cols-1 || r != X.Rows-1 {
				str += ", "
			}
		}
		// Put newline except at last item
		if r != X.Rows-1 {
			str += "\n "
		}
	}
	str += "]"
	return str
}

// Apply Kronecker multiplication to the list of kets, in the order
func KronKets(kets []Ket) ColVec {
	vec := Matrix{
		Rows:   1,
		Cols:   1,
		Stride: 1,
		Data:   []complex128{1},
	}
	for i := 0; i < len(kets); i++ {
		vec = *vec.Kronecker(Matrix(kets[i]))
	}
	return NewColVec(vec)
}

// Determines equality between two Matrix matrices, using epsilon a complex number.
// Two matrices are Equals if their dimensions are Equals and their values are Equals,
// given a complex epsilon. Two complex numbers a+bi and c+di are considered to be Equals
// when abs(a-c) < real(epsilon) and abs(b-d) < imag(epsilon).
func (a Matrix) Equals(b Matrix, epsilon complex128) bool {
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

// Creates the identity matrix of given size
func Identity(size int) Matrix {
	m := Matrix{
		Rows:   size,
		Cols:   size,
		Stride: size,
		Data:   make([]complex128, size*size),
	}

	for i := 0; i < size; i++ {
		m.Data[i*size+i] = 1
	}

	return m
}
