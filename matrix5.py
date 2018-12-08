#copyright: Xiang 2018
import fractions
from fractions import Fraction as F
from fractions import Fraction as Fraction
import random

class InvalidMatrix(Exception):
    def __init__(self, num_row, item_num, normal_item_num):
        err = 'You got {} elements in your row {} (humain 1-base),\
which should be {} elements'.format(item_num, num_row, normal_item_num)
        Exception.__init__(self,err)
        self.num_row = num_row
        self.item_num = item_num
        self.normal_item_num = normal_item_num


class InvalidMultiplication(Exception):
    def __init__(self, row1, elem2):
        err = 'got {} row(s) in first matrix, while {} element(s)\
in each row of the 2nd matrix'.format(row1, elem2)
        Exception.__init__(self, err)
        self.row = row1
        self.elem = elem2

class InvalidInput(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

class InvalidAddition(InvalidInput):
    pass

class Matrix:
    def __init__(self, *lines, identity_matrix = False, israndom = False, \
                 vertical = False):
        '''
        0.create a matrix, if you want a identity matrix, please give
          the number of rows as parameter and then switch 'identity_matrix' to True
          eg: m = Matrix(4, identity_matrix=True)
        (1.Lorsque l'on veut donner une fraction comme un element de la matrix,
          il suiffit de entrer F(nominateur, dénominateur) pour repésenter une fraction)
        1.if you want to enter a fraction as the element of the matrix,
        use F(numarator, denominator) to represent a fraction.
        2.if you want a random matrix, please give 4 integers as paramter,
          the frist and the second are two boundary of the random number (included),
          the third and the forth dight represente the #row and #column
          eg: m = Matrix(0, 10, 4, 5, rondom = True)
          which will populate a 4*5 matrix with integer from 0 to 10 (included)
        3.if you want to enter the matrix vertically, switch 'vertical' parameter to True
        4.if the input matrix is a augmented matrix which present the coefficient of a
          linear system, switch 'aug'parameter to True.'''
        if identity_matrix == True:
            self.matrix = []
            for i in range(lines[0]):
                tmp = []
                for j in range(i):
                    tmp.append(0)
                tmp.append(1)
                for j in range(i+1, lines[0]):
                    tmp.append(0)
                self.matrix.append(tmp)

        elif israndom == True:
            self.matrix = []
            for i in range(lines[2]):
                tmp = []
                for j in range(lines[3]):
                    tmp.append(random.randint(lines[0], lines[1]))
                self.matrix.append(tmp)

        elif vertical == True:
            self.matrix = []
            for i in range(len(lines[0])):
                tmp = []
                for j in range(len(lines)):
                    tmp.append(lines[j][i])
                self.matrix.append(tmp)

        else:
            self.matrix = [line for line in lines]

        def isvalid(m):
            '''
            return True if a matrix is valid, False otherwise'''
            rang = len(m[0])
            for i in range(len(m)):
                if len(m[i]) != rang:
                    raise InvalidMatrix(len(m[i]), i+1, rang)
            return True
        self.validity = isvalid(self.matrix)
        self.isidentity = identity_matrix
        self.israndom = israndom
        self.num_row = len(self.matrix) #number of row
        self.num_col = len(self.matrix[0]) #number of column
        self.aug = aug

    def copy(self):  # provided by Vida
        '''matrix -> 2Dlist
            return a 2D list copy of the current matrix,
            which will not influence the original matrix'''
        copy = []
        for row in self.matrix:
            l = []
            for item in row:
                l.append(item)
            copy.append(l)
        return copy

    def is_zero(self):
        '''return True if the only element of a matrix is 0'''
        if self.matrix == [[0]]:
            return True

    def __getitem__(self, i):
        '''return one specified list-type column'''
        if i > self.num_row:
            raise IndexError('Out of index')
        return self.matrix[i]

    def __setitem__(self, i, value):
        '''int,list -> None
           replace a sepcified column by a given list.'''
        if i > self.num_row:
            raise IndexError('Out of index')
        if len(value) != len(self.matrix[i]):
            raise InvalidMatrix(i+1, len(value),self.num_col)
        self.matrix[i] = value 

    def __repr__(self):
        '''print the matrix in a particuler way'''
        local_max = 0
        for i in range(self.num_row):
            for item in self.matrix[i]:
                if len(str(item)) > local_max:
                    local_max = len(str(item))

        result = ''
        for j in range(self.num_row):
            for k in range(len(self.matrix[j])):
                current_length = len(str(self.matrix[j][k]))
                if k == 0:
                    result += '| '
                result += str(self.matrix[j][k]) + ' '*(local_max-current_length+1)
                if k == len(self.matrix[j])-1:
                    result += '| \n'
        return result.strip()

    def __neg__(self):
        '''return a negative version of the given list'''
        tmp = self.copy()
        for i in range(len(tmp)):
            for j in range(len(tmp[i])):
                tmp[i][j] = -1*self.matrix[i][j]
        return Matrix(*tmp)

    def __add__(self, other):
        '''
        (Matrix, Matrix/num) -> Matrix
        add self and other, no matter other's type (constant, matrix)'''
        if type(other) == Matrix:
            m_add = []
            for i in range(self.num_row):
                tmp = []
                for j in range(len(self.matrix[i])):
                    tmp.append(self.matrix[i][j] + other.matrix[i][j])
                m_add.append(tmp)
            return Matrix(*m_add)

        elif type(other) == int:
            tmp = self.matrix[:]
            for i in range(len(tmp)):
                for j in range(len(tmp[i])):
                    tmp[i][j] += other
            return Matrix(*tmp)

        else:
            raise TypeError

    def __sub__(self, other):
        '''substance'''
        return self.__add__(-other)

    def __mul__(self, other):
        '''multiplication of two matrix or a real and a matrix'''
        if type(other) == int:
            m_mul = self.copy()
            for i in range(len(m_mul)):
                for j in range(len(m_mul[0])):
                    m_mul[i][j] *= other

        elif type(other) == Matrix:
            if len(self.matrix) != len(other.matrix[0]):
                raise InvalidMultiplication(len(self.matrix), len(other.matrix[0]))
            m_mul = []
            tmp = 0
            for i in range(self.num_row):
                m_mul.append([])
                for j in range(len(other.matrix[0])):
                    for k in range(len(self.matrix[i])):
                        tmp += self.matrix[i][k]*other.matrix[k][j]
                    m_mul[i].append(tmp)
                    tmp = 0
        return Matrix(*m_mul)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        '''return True of the two matrix are equal, False otherwise.'''
        return self.copy() == other.copy()

    def get_col(self, i):
        '''return one column(machine, 0-base) of a matrix as a list'''
        if i > self.num_col:
            raise IndexError('Out of index')
        tmp = []
        for j in range(self.num_row):
            tmp.append(self.matrix[j][i])
        return tmp

    def aug(self, other=None, left_hand=False):
        '''
        (matrix, matrix) -> matrix
        return a augmented matrix of the given matrix,the default value is a zero column.
        if you want to augment the 'other' at left-hand of the self matrix,
        switch left_hand to True.
        Precondition: the two matrix should have same number of row.'''
        def creat_default_zero_column():
            zero_column = []
            for i in range(self.num_row):
                zero_column.append([0])
            return Matrix(*zero_column)

        if type(other) != Matrix and other == None:
            other = creat_default_zero_column()
        if left_hand == True:
            return other.aug(self, left_hand = False)
        else:
            tmp = self.copy()
            for i in range(len(other.matrix)):
                tmp[i] = self.matrix[i] + other.matrix[i]
            return Matrix(*tmp)

    def tsp(self):
        '''return a transposed matrix of the original matrix'''
        return Matrix(*self.matrix, vertical = True)

    def me(self, reverse = False):
        '''
        return the R.E.F(row echelon form, in french m.e.) of the matrix'''
        i, j = 0, 0
        me = self.copy()

        while i < len(me[0]) and j < len(me)-1:
            found_non_zero = False
            for o in range(j, len(me)): #find a non-zero pivot, replace the current line.
                if me[o][i] != 0 and found_non_zero == False:
                    me[j], me[o] = me[o], me[j]
                    found_non_zero = True

            if found_non_zero:
                me[j] = [F(elem, me[j][i]) for elem in me[j]] #current pivot line: me[j]
                for k in range(j+1, len(me)):
                    #modifier tous les entrees au-desous du pivot a zero.
                    tmp_line = []
                    for q in range(len(me[k])):
                        tmp_line.append(me[k][q]-me[k][i]*me[j][q])
                    me[k] = tmp_line
                i += 1
                j += 1
            else:
                i += 1

        for line in me:
            i = 0
            while i < len(line) and line[i] == 0:
                i += 1
            if i < len(line) and line[i] != 1:
                coe = line[i]
                for p in range(len(line)):
                    line[p] = F(line[p], coe)

        if reverse == True:
            me = me[::-1]

        return Matrix(*me)

    def mer(self):
        '''
        return the RREF form of a matrix(reduced row echelon fomr)
        return la mer forme d'une matrice'''
        mer = self.me(reverse = True).matrix

        h = 0
        while h < len(mer) and mer[h].count(0) == len(mer[h]):  #marquer la ligne non-tout nul.
            h += 1
        while h < len(mer):
            j = 0
            while j < len(mer[h]) and mer[h][j] == 0 :  #marquer l'index a partir duquelle on va commencer.
                j += 1

            for v in range(h+1 ,len(mer)):
                coefficient = mer[v][j]
                for l in range(len(mer[v])):
                    mer[v][l] -= mer[h][l]*coefficient
            h += 1

        mer = mer[::-1]
        return Matrix(*mer)

    def ref(self):
        return self.me()

    def rref(self):
        return self.mer()

    def lig_b(self):
        '''
        Matrix -> tuple(str, set)
        return a base of the matrix row space '''
        lig = self.me().matrix
        base = []
        for row in lig:
           tmp = []
           if row.count(0) != len(row):
              for elem in row:
                  tmp.append(elem)
              base.append(Vector(*tmp).__str__())
        return ('Dim(lig_espace/row_space) = {}'.format(len(base)), \
                set(base))

    def basic_v(self):
        '''Matrix -> list
        return the index(0-base) of basic variables of the matrix'''
        i = 0   #indicator of row
        j = 0   #indicator of column
        mer = self.mer().matrix
        b_v = []  #index of basic variable
        while mer[i][j] == 0:
            j += 1
        while i < len(mer) and j < len(mer[0]):
            if mer[i][j] != 0:
                b_v.append(j)
                i += 1
                j += 1
            elif mer[i][j] == 0 and j < len(mer[0]):
                j += 1
        return b_v

    def col_b(self):
        '''Matrix -> set
        return a base of the matrix column space '''
        if self.is_zero():
            return {0}
        col_list = self.basic_v()
        tmp = [Vector(*self.get_col(i)).__str__() for i in col_list]
        col_base = tuple(tmp)
        print('Dim(col_espace/col_space) = {}'.format(len(col_base)))
        return set(col_base)

    def nul_b(self):
        '''Matrix -> set
        return a base of the matrix nul sapce'''
        if self.is_zero():
            return {0}
        def set_row_to_index_position(self):
            copy = self.mer().copy()
            while len(copy) < self.num_col:
                copy.append([0 for i in range(self.num_col)])
            i = 0
            j = 0
            basic_v = self.basic_v()
            while j < len(copy) and i < self.num_col:
                if i in basic_v:
                    copy[j], copy[i] = copy[i], copy[j]
                    i += 1
                    j += 1
                else:
                    i += 1
            return Matrix(*copy)

        s_copy = set_row_to_index_position(self)
        basic_v_index = self.basic_v()
        free_v_index = [i for i in range(self.num_col) if i not in basic_v_index]
        nul_base = []

        for index in free_v_index:
            tmp = []
            i = 0
            while i < self.num_row:
                base = -s_copy.get_col(index)[i]
                if type(base) == fractions.Fraction:
                    base = str(base)
                tmp.append(base)
                i += 1
            while i < self.num_col:
                tmp.append(0)
                i += 1
            tmp[index] = 1
            if self.aug == True:  #####12.8
                tmp.pop()
            nul_base.append(tuple(tmp))
        nul = False
        if free_v_index == []:
            nul_base = (0,)
            nul = True
        if not nul:
            print('Dim(Nul_space) = {}'.format(len(nul_base)))
        else:
            print('Dim(Nul_space) = 0')
        return set(nul_base)

    def inverse(self):
        '''return the inverse matrix of the given matrix
        Precondition: the given matrix should be a n*n square matrix'''
        if self.num_row != self.num_col:
            raise InvalidInput('the matrix is non-invertible')
        aug = self.aug(Matrix(len(self.matrix), identity_matrix = True))
        aug = aug.mer()
        for i in range(len(aug.matrix)):
            aug.matrix[i] = aug.matrix[i][len(aug.matrix[i])//2 :]
        return Matrix(*aug.matrix)

    def eliminate(self, i= None, j=None):
        '''(matrix, int, int) -> matrix
        return a matrix which the #i row and #j column (0-base) has been eliminated'''
        copy = self.copy()
        if (i != None and i > len(copy)) or (j != None and j > len(copy[0])):
            raise IndexError('Out of range')

        if j != None:
            for line in copy:
                del line[j]
                
        if i != None:
            copy.remove(copy[i])
        return Matrix(*copy)

    def det(self):
        '''matrix -> num
        return the determinant of a matrix
        Precondition: the matrix shold be a squre matrix.'''
        if self.num_row != self.num_col:
            raise InvalidInput('the input should be a square matrix.')
        copy = self.copy()
        if self.num_col == 2 and self.num_col == 2:
            return self[0][0]*self[1][1] - self[0][1]*self[1][0]
        result = 0
        for i in range(self.num_col):
            result += (-1)**(0+i) * self[0][i] * self.eliminate(0, i).det()
        return result

    @classmethod
    def li(cls, *vectors):
        '''tuple -> bool
           return True if the given vectors are linéairement indépendant,
           False otherwise.'''
        tmp_m = Matrix(*vectors)
        if len(tmp_m.lig_b()[1]) < len(tmp_m.matrix):
            return False
        return True


class Vector:
    def __init__(self, *num):
        self.vector = list(num)

    def __repr__(self):
        result = '('
        for i in range(len(self.vector) - 1):
            result = result + self.vector[i].__str__() + ', '
        result = result + self.vector[-1].__str__() + ')'
        return result

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError('Out of range.')
        return self.vector[i]

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        i = 0
        while i < len(self) :
            if self[i] != other[i] :
                return False
            i += 1
        return True

    def __add__(self, other):
        if len(self) != len(other):
            raise InvalidAddition('{} elem in first, {} in second'.format(len(self),len(other)))
        tmp = []
        for i in range(len(self)):
            tmp.append(self[i] + other[i])
        return Vector(*tmp)

    def __sub__(self, other):
        return self.__add__(-other)
    
    def __neg__(self):
        return Vector(*[-i for i in self.vector])

    def __mul__(self, other):
        if type(other) == Vector:
            if len(self) != len(other):
                raise InvalidAddition('{} elem in first, {} in second'.format(len(self),len(other)))
            tmp = 0
            for i in range(len(self)):
                tmp += self[i] * other[i]
            return tmp
        elif type(other) == int:
            return Vector(*[i*other for i in self.vector])

    def __rmul__(self, other):
        return self.__mul__(other)


class Span:
    def __init__(self, *vectors):
        '''vector/list -> None
        initialize a vector space by the given vectors(or lists)'''
        self.vectors = [vector for vector in vectors]
        for i in range(len(self.vectors)):
            if type(self.vectors[i]) != Vector:
                self.vectors[i] = Vector(*self.vectors[i])

    def __repr__(self):
        result = '{'
        for vector in self.vectors[:-1]:
            result += vector.__repr__() + ','
        result += self.vectors[-1].__repr__() + '}'
        return result

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, i):
        if i > len(self.vectors):
            raise IndexError('Out of index')
        return self.vectors[i]

    def li(self):
        '''Span -> bool
        return True if all the vectors in the span is linear independant,
        False otherwise'''
        return Matrix.li(*self)
