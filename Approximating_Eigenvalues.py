import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Approximating Eigenvalues
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this section we look at some methods that can be used to approximate the eigenvalues of a matrix $A$.  Although it is possible to find the exact eigenvalues for small matrices, the approach is impractical for larger matrices.

        Most introductory textbooks demonstrate a direct way to compute eigenvalues of an $n\times n$ matrix $A$ by computing roots of an associated $n$th degree polynomial, known as the *characteristic polynomial*.  For example, suppose $A$ is a $2\times 2$ matrix.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rr} a & b  \\ c & d \end{array}\right]
        \end{equation}
        $$

        The eigenvalues of $A$ are solutions to the quadratic equation $\lambda^2 - (a+d)\lambda + ad-bc = 0$, which can be written explicitly in terms of $a$, $b$, $c$, and $d$ using the quadratic formula.  The challenges with larger matrices are that the polynomial is more difficult to construct, and the roots cannot be easily found with a formula.

        The algorithms we describe in this section are iterative methods.  They generate a sequence of vectors $\{X^{(1)}, X^{(2)}, X^{(3)}, ... \}$ that approach a true eigenvector of the matrix under consideration.  An approximation of the corresponding eigenvalue can then be computed by multiplying the approximate eigenvector by $A$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Power method

        The first algorithm we introduce for approximating eigenvalues is known as the **Power Method**.  This method generates a sequence of vectors by repeated matrix multiplication.  Under suitable conditions, the sequence of vectors approaches the eigenvector associated with the eigenvalue that is largest in absolute value.    

        For the simplest explanation, suppose that $A$ is an $n\times n$ diagonalizable matrix with eigenvectors $\{V_1, V_2, ... V_n\}$, and that $\lambda_1$ is the eigenvalue of $A$ that is largest in absolute value.  To begin the Power Method, we choose any nonzero vector and label it $X^{(0)}$.  We can express $X^{(0)}$  as a linear combination of the eigenvectors since they form a basis for $\mathbb{R}^n$.

        $$
        \begin{equation}
        X^{(0)} = c_1V_1 + c_2V_2 + ... c_nV_n
        \end{equation}
        $$

        We now form a sequence of vectors $X^{(1)}$, $X^{(2)}$, $X^{(3)}$, ..., by setting $X^{(m)}= AX^{(m-1)}$.  Each of these vectors is also easly expressed in terms of the eigenvectors.

        $$
        \begin{eqnarray*}
        X^{(1)} = AX^{(0)} & = & c_1AV_1 + c_2AV_2 + ... c_nAV_n \\
                           & = & c_1\lambda_1V_1 + c_2\lambda_2V_2 + ... c_n\lambda_nV_n \\
        X^{(2)} = AX^{(1)} & = & c_1\lambda_1AV_1 + c_2\lambda_2AV_2 + ... c_n\lambda_nAV_n \\
                           & = & c_1\lambda_1^2V_1 + c_2\lambda_2^2V_2 + ... c_n\lambda_n^2V_n \\
                           & \vdots & \\
        X^{(m)} = AX^{(m-1)} & = & c_1\lambda_1^{m-1}AV_1 + c_2\lambda_2^{m-1}AV_2 + ... c_n\lambda_n^{m-1}AV_n \\
                           & = & c_1\lambda_1^mV_1 + c_2\lambda_2^mV_2 + ... c_n\lambda_n^mV_n 
        \end{eqnarray*}
        $$

        In the expression for $X^{(m)}$, we can then factor out $\lambda_1^m$ to understand what happens as $m$ gets large.

        $$
        \begin{equation}
        X^{(m)} =  \lambda_1^m\left(c_1V_1 + c_2\left(\frac{\lambda_2}{\lambda_1}\right)^mV_2 + ... c_n\left(\frac{\lambda_n}{\lambda_1}\right)^mV_n\right) 
        \end{equation}
        $$

        If $|\lambda_1| > |\lambda_i|$ for all $i\neq 1$, then $|\lambda_i/\lambda_1|< 1$ and $(\lambda_i/\lambda_1)^m$ will approach zero as $m$ gets large.  This means that if we repeatedly multiply a vector by the matrix $A$, eventually we will get a vector that is very nearly in the direction of the eigenvector that corresponds to the $\lambda_1$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's demonstrate the calculation on the matrix shown here before we discuss the method further.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrrr} -2 & 6 & 2 & -8 \\ -6 & 0 & 12 & 12 \\ -6 & 0 & 12 & 12 \\ -10 & 3 & 7 & 14 \end{array}\right]
        \end{equation}
        $$

        As a matter of practicality, it is common to scale the vectors in the sequence to unit length as the Power Method is applied.  If the vectors in the sequence are not scaled, their magnitudes will grow if $\lambda_1>1$ or decay if $\lambda_1<1$.    Since all components of the vectors get divided by the same factor when the vector is scaled, this step doesn't change the ultimate behavior of the sequence.  The scaled sequence of vectors still approaches the direction of the eigenvector. 

        We choose an arbitrary $X^{(0)}$ and calculate $X^{(20)}$ using the following rule.

        $$
        \begin{equation}
        X^{(m)}=\frac{AX^{(m-1)}}{||AX^{(m-1)}||}
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import laguide as lag
    A = np.array([[-2, 6, 2, -8], [-6, 0, 12, 12], [-6, 0, 12, 12], [-10, 3, 7, 14]])
    X = np.array([[1], [0], [0], [0]])
    _m = 0
    while _m < 20:
        X = A @ X
        X = X / lag.Magnitude(X)
        _m = _m + 1
    print(X)
    return A, X, lag, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now if $X$ is the eigenvector of $A$ with unit magnitude, then $|AX| = |\lambda_1X| = |\lambda_1|$.  We can therefore approximate $|\lambda_1|$ with $|AX|$.
        """
    )
    return


@app.cell
def _(A, X, lag):
    print(lag.Magnitude(A@X))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It appears that 24 is an estimate for $\lambda_1$.   To determine if our calculation is correct, we can compare $AX$ with $\lambda_1X$.  
        """
    )
    return


@app.cell
def _(A, X):
    print(A@X - 24*X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Indeed the difference $AX-24X$ is small.  Note that in this case, we can even do the calculation with integer multiplication.  Notice that $X$ has 0 in the first entry and the other entries are equal.  If we set these entries to 1, the result is easy to calculate even without the aid of the computer.  (*Remember that we can change the magnitude of an eigenvector and it is still an eigenvector.*) 

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrrr} -2 & 6 & 2 & -8 \\ -6 & 0 & 12 & 12 \\ -6 & 0 & 12 & 12 \\ -10 & 3 & 7 & 14 \end{array}\right]
        \left[ \begin{array}{r} 0 \\ 1\\ 1 \\ 1 \end{array}\right] =
        \left[ \begin{array}{r} 0 \\ 24\\ 24 \\ 24 \end{array}\right] = 24X
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In practice, we do not know how many iterations we need to perform in order to get a good approximation of the eigenvector.  Instead we should specify a condition upon which we will be satisfied with the approximation and terminate the iteration.  For example, since $||AX^{(m)}||\approx \lambda_1$ and $AX^{(m)}\approx \lambda_1X^{(m)}$ we might require that $AX^{(m)} - ||AX^{(m)}||X^{(m)} < \epsilon$ for some small number $\epsilon$ known as a tolerance.  This condition ensures that $X^{(m)}$ functions roughly like an eigenvector.  It is also best to include in the code a limit on the number of iterations that will be carried out.  This ensures that the computation will eventually end, even if a satisfactory result has not yet been achieved.
        """
    )
    return


@app.cell
def _(A, lag, np):
    X_1 = np.array([[1], [0], [0], [0]])
    _m = 0
    _tolerance = 0.0001
    _MAX_ITERATIONS = 100
    Y = A @ X_1
    _difference = Y - lag.Magnitude(Y) * X_1
    while _m < _MAX_ITERATIONS and lag.Magnitude(_difference) > _tolerance:
        X_1 = Y
        X_1 = X_1 / lag.Magnitude(X_1)
        Y = A @ X_1
        _difference = Y - lag.Magnitude(Y) * X_1
        _m = _m + 1
    print('Eigenvector is approximately:')
    print(X_1, '\n')
    print('Magnitude of the eigenvalue is approximately:')
    print(lag.Magnitude(Y), '\n')
    print('Magnitude of the difference is:')
    print(lag.Magnitude(_difference))
    return X_1, Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A more common condition to require is that $||X^{(m)} - X^{(m-1})|| < \epsilon$ for a given tolerance $\epsilon$.  This condition merely requires that the vectors in the sequence get close to one another, not that they are actually approximate an eigenvector.  
        """
    )
    return


@app.cell
def _(A, Y, lag, np):
    X_2 = np.array([[1], [0], [0], [0]])
    _m = 0
    _tolerance = 0.0001
    _MAX_ITERATIONS = 100
    _difference = X_2
    while _m < _MAX_ITERATIONS and lag.Magnitude(_difference) > _tolerance:
        _X_previous = X_2
        X_2 = A @ X_2
        X_2 = X_2 / lag.Magnitude(X_2)
        _difference = X_2 - _X_previous
        _m = _m + 1
    print('Eigenvector is approximately:')
    print(X_2, '\n')
    print('Magnitude of the eigenvalue is approximately:')
    print(lag.Magnitude(Y), '\n')
    print('Magnitude of the difference is:')
    print(lag.Magnitude(_difference))
    return (X_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        While the Power Method is easy to understand and apply, it does have disadvantages.  The most apparent disadvantage is that the method only applies to the largest eigenvalue.  This is not a huge detriment since applications often only require an approximation of the largest eigenvalue.  Also, as we will demonstrate below, it is possible to easily modify the method to approximate the other eigenvalues.  A more significant disadvantage is that the rate at which the sequence converges can be slow in some circumstances.  For example, we can see that if $|\lambda_1|$ is close to $|\lambda_2|$, then $|\lambda_1/\lambda_2|^m$ approaches zero more slowly as $m$ gets large.  The Power Method may fail to converge at all if $|\lambda_1| = |\lambda_2|$, which occurs if $\lambda_1 = -\lambda_2$, or if $\lambda_1$ and $\lambda_2$ are a complex conjugate pair.  Additionally, the method may perform poorly if the $V_1$ component of $X^{(0)}$ is too small.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Inverse power method

        The **Inverse Power Method** is a modified version of the Power Method that allows us to approximate eigenvalues that are *not the largest*.  All that is needed to make the modification is two simple facts that relate changes in a matrix to changes in the eigenvalues of that matrix.  Let's suppose that $A$ is an invertible $n\times n$ matrix with eigenvalue $\lambda$ and corresponding eigenvector $V$, so that $AV=\lambda V$.  If we multiply this equation by $A^{-1}$, we get $V=\lambda A^{-1}V$, which can then be divided by $\lambda$ to illustrate the useful fact.

        $$
        \begin{equation}
        A^{-1}V = \frac{1}{\lambda}V
        \end{equation}
        $$

        If $\lambda$ is an eigenvalue of $A$, then $\lambda^{-1}$ is an eigenvalue of $A^{-1}$.  Furthermore the eigenvector of $A$ is also an eigenvector of $A^{-1}$.  The important point here is that if $\lambda_n$ is the smallest eigenvalue of $A$, then $\lambda_n^{-1}$ is the *largest* eigenvector of $A^{-1}$.  If we want to approximate the smallest eigenvalue of $A$, we can just apply the Power Method to $A^{-1}$.

        We demonstrate the calculation for the following $3\times 3$ matrix.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrrr} 9 & -1 & -3 \\ 0 & 6 & 0 \\ -6 & 3 & 6 \end{array}\right]
        \end{equation}
        $$

        Again we choose an arbitrary $X^{(0)}$, and generate a sequence of vectors by multiplying by $A^{-1}$ and scaling the result to unit length.

        $$
        \begin{equation}
        X^{(m)}=\frac{A^{-1}X^{(m-1)}}{||A^{-1}X^{(m-1)}||}
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    X_3 = np.array([[0], [1], [0]])
    _m = 0
    _tolerance = 0.0001
    _MAX_ITERATIONS = 100
    _difference = X_3
    A_1 = np.array([[9, -1, -3], [0, 6, 0], [-6, 3, 6]])
    A_inv = lag.Inverse(A_1)
    while _m < _MAX_ITERATIONS and lag.Magnitude(_difference) > _tolerance:
        _X_previous = X_3
        X_3 = A_inv @ X_3
        X_3 = X_3 / lag.Magnitude(X_3)
        _difference = X_3 - _X_previous
        _m = _m + 1
    print('Eigenvector is approximately:')
    print(X_3, '\n')
    print('Magnitude of the eigenvalue of A inverse is approximately:')
    print(lag.Magnitude(A_inv @ X_3), '\n')
    print('Magnitude of the eigenvalue of A is approximately:')
    print(lag.Magnitude(A_1 @ X_3), '\n')
    return A_1, A_inv, X_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The exact value of the smallest eigenvalue of $A$ is 3, which again can be verified by calculation.

        $$
        \begin{equation}
        AV = \left[ \begin{array}{rrrr} 9 & -1 & -3 \\ 0 & 6 & 0 \\ -6 & 3 & 6 \end{array}\right]
        \left[ \begin{array}{r} 1 \\ 0\\ 2 \end{array}\right] =
        \left[ \begin{array}{r} 3 \\ 0 \\ 6 \end{array}\right] = 3V
        \end{equation}
        $$

        In our discussion of [Inverse Matrices](Inverse_Matrices.ipynb) we noted that the construction of an inverse matrix is quite expensive since it requires the solution of $n$ systems of size $n\times n$.  An alternative to constructing $A^{-1}$ and computing the  $X^{(m)}=A^{-1}X^{(m-1)}$ is to solve the system $AX^{(m)}=X^{(m-1)}$ to obtain $X^{(m)}$.  This means that we solve one $n\times n$ system for every iteration.  This appears to require more work than the construction of $A^{-1}$, but in fact it is less since every system involves the same coefficient matrix.  We can therefore save much work by performing elimination only once and storing the result in an $LU$ factorization.  With the the matrix $A$ factored, each system $AX^{(m)}=X^{(m-1)}$ only requires one forward substitution and one backward substitution.  
        """
    )
    return


@app.cell
def _(lag, np):
    import scipy.linalg as sla
    X_4 = np.array([[0], [1], [0]])
    _m = 0
    _tolerance = 0.0001
    _MAX_ITERATIONS = 100
    _difference = X_4
    A_2 = np.array([[9, -1, -3], [0, 6, 0], [-6, 3, 6]])
    _LU_factorization = sla.lu_factor(A_2)
    while _m < _MAX_ITERATIONS and lag.Magnitude(_difference) > _tolerance:
        _X_previous = X_4
        X_4 = sla.lu_solve(_LU_factorization, X_4)
        X_4 = X_4 / lag.Magnitude(X_4)
        _difference = X_4 - _X_previous
        _m = _m + 1
    print('Eigenvector is approximately:')
    print(X_4, '\n')
    print('Magnitude of the eigenvalue of A inverse is approximately:')
    print(lag.Magnitude(sla.lu_solve(_LU_factorization, X_4)), '\n')
    print('Magnitude of the eigenvalue of A is approximately:')
    print(lag.Magnitude(A_2 @ X_4), '\n')
    return A_2, X_4, sla


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Shifted inverse power method

        Using a small modification to the Inverse Power Method, we can also approximate eigenvalues that are not the smallest.  For this variation of the method, we need to observe that if we "shift" the diagonal entries of a matrix by a scalar $\mu$, all of the eigenvalues of the matrix are also shifted by $\mu$.  Let $A$ be an $n\times n$ matrix with eigenvalue $\lambda$ and corresponding eigenvector $V$, so that $AV=\lambda V$.  Then $(A-\mu I)V = AV - \mu V = \lambda V - \mu V = (\lambda-\mu)V$, which means that $V$ is also an eigenvector of the matrix $(A-\mu I)$ corresponding to the eigenvalue $\lambda -\mu$.  

        $$
        \begin{equation}
        \frac{1}{\lambda_1-\mu}, \frac{1}{\lambda_2-\mu}, \frac{1}{\lambda_3-\mu}, ....,\frac{1}{\lambda_n-\mu} 
        \end{equation}
        $$

        This is useful because it allows us to now use the Inverse Power Method to approximate the eigenvalue of $A$ that lies closest to $\mu$.  For example, if $\mu$ is closest to $\lambda_2$, then $|\lambda_2-\mu| < |\lambda_i -\mu|$ for all other $i\neq 2$, which means that $(\lambda_2-\mu)$ can be approximated by applying the Inverse Power Method to $(A-\mu I)$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We demonstrate the computation of the middle eigenvalue of the matrix from the previous example.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrrr} 9 & -1 & -3 \\ 0 & 6 & 0 \\ -6 & 3 & 6 \end{array}\right]
        \end{equation}
        $$

        By using the Inverse Power Method we determined that the smallest eigenvalue of $A$ is 3.  Applying the Power Method directly will show that the largest eigenvalue of $A$ is 12.  Since the third eigenvalue must lie somewhere in between these extremes, we choose $\mu$ to be exactly in the middle at $7.5$.  Note that once we have a good approximation to the eigenvector with $X^{(m)}$, we can approximate the eigenvalue of $A$ with $||AX^{(m)}||$.
        """
    )
    return


@app.cell
def _(lag, np, sla):
    X_5 = np.array([[0], [1], [0]])
    _m = 0
    _tolerance = 0.0001
    _MAX_ITERATIONS = 100
    _difference = X_5
    A_3 = np.array([[9, -1, -3], [0, 6, 0], [-6, 3, 6]])
    I = np.eye(3)
    mu = 7.5
    Shifted_A = A_3 - mu * I
    _LU_factorization = sla.lu_factor(Shifted_A)
    while _m < _MAX_ITERATIONS and lag.Magnitude(_difference) > _tolerance:
        _X_previous = X_5
        X_5 = sla.lu_solve(_LU_factorization, X_5)
        X_5 = X_5 / lag.Magnitude(X_5)
        _difference = X_5 - _X_previous
        _m = _m + 1
    print('Eigenvector is approximately:')
    print(X_5, '\n')
    print('Eigenvalue of A is approximately:')
    print(lag.Magnitude(A_3 @ X_5))
    return A_3, I, Shifted_A, X_5, mu


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Let $A$ be the matrix from the Inverse Power Method example.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr} 9 & -1 & -3 \\ 0 & 6 & 0 \\ -6 & 3 & 6 \end{array}\right]
        \end{equation}
        $$

        ($a$) Use the Power Method to approximate the largest eigenvalue $\lambda_1$.  Verify that the exact value of $\lambda_1$ is 12.
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ($b$) Apply the Inverse Power Method with a shift of $\mu = 10$.  Explain why the results differ from those in the example.
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ($c$) Apply the Inverse Power Method with a shift of $\mu = 7.5$ and the initial vector given below.  Explain why the sequence of vectors approach the eigenvector corresponding to $\lambda_1$

        $$
        \begin{equation}
        X^{(0)} = \left[ \begin{array}{r} 1 \\ 0  \\ 0 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:*** Let $B$ be the following matrix.

        $$
        \begin{equation}
        B = \left[ \begin{array}{rrrr} -2 & -18 & 6 \\ -11 & 3 & 11 \\ -27 & 15 & 31 \end{array}\right]
        \end{equation}
        $$

        ($a$) Apply the Power Method and Inverse Power Method with shifts to approximate all eigenvalues of the matrix $B$. (*Note that one of the eigenvalues of this matrix is negative.*)
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ($b$) Check your results using the $\texttt{eig}$ function in SciPy.
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### References

        - Burden, Richard L. et al. *Numerical Analysis*. 10th ed., Cengage Learning, 2014
        - Golub, Gene H. and Charles F. Van Loan. *Matrix Computations*., The Johns Hopkins University Press, 1989

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
