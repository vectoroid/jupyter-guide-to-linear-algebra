import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Applications of Linear Systems and Matrix Algebra

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The need to solve linear systems arises in a wide variety of fields.  We show in this section an example of the linear system that arises in fitting a curve to a data set.  We also discuss the application of matrix algebra to cryptography and graph theory.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Interpolation

        Interpolation is the process of estimating unknown data values that fall between known values.  This process commonly involves fitting a curve through the known set of data points in order to make predictions about the unknown values.  The curve is described by a set of parameters, and "fitting the curve" means choosing the parameters such that the curve best represents the data.  A simple way to fit a curve is to require that it passes through all the data provided.  

        Let's look at the the data points $(2,8)$, $(5,12)$, $(6,14)$, and $(15,15)$ as an example.
        """
    )
    return


@app.cell
def _():
    # "%matplotlib inline\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport laguide as lag\n\nx = np.array([2,5,6,15])\ny = np.array([8,12,14,15])\n\nfig,ax = plt.subplots()\nax.scatter(x,y,color='red');\n\nax.set_xlim(0,20);\nax.set_ylim(0,20);\nax.set_xlabel('x');\nax.set_ylabel('y');\nax.grid(True)" command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Polynomials are a common curve used for interpolation.  In this case, since we have four data points there will be four equations that must be satisfied for the graph to pass through each point.  We will choose a third degree polynomial, $P_3$, since that will give us four parameters with which to satisfy the equations.

        $$
        \begin{equation}
        P_3(x) = a_0 + a_1x + a_2x^2 + a_3x^3
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The data gives us the four equations $P_3(2) = 8$, $P_3(5) = 12$, $P_3(6) = 14$, and $P_3(15) = 15$.  This set of equations is a linear system for the unknown coefficients.


        $$
        \begin{eqnarray*}
        a_0 + 2a_1 + 2^2a_2 + 2^3a_3 & = & 8\\
        a_0 + 5a_1 + 5^2a_2 + 5^3a_3 & = & 12\\
        a_0 + 6a_1 + 6^2a_2 + 6^3a_3 & = & 14\\
        a_0 + 15a_1 + 15^2a_2 + 15^3a_3 & = & 15
        \end{eqnarray*}
        $$

        We assemble the matrix $A$ and right-hand side vector $B$ as NumPy arrays.
        """
    )
    return


@app.cell
def _(np, x, y):
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    for _i in range(4):
        B[_i, 0] = y[_i]
        for _j in range(4):
            A[_i, _j] = x[_i] ** _j
    print(A, '\n')
    print(B)
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can solve the system $AX=B$ with elimination.  To avoid confusion with the $x$ variable, we will label our solution $\texttt{coeffs}$ since they represent the coefficients of the polynomial.
        """
    )
    return


@app.cell
def _(A, B, lag):
    coeffs = lag.SolveSystem(A,B)
    print(coeffs)
    return (coeffs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, we plot the graph of the polynomial over the data to see the fit. 
        """
    )
    return


@app.cell
def _(coeffs, np, plt, x, y):
    x_fit = np.linspace(x[0],x[3],50)
    y_fit = coeffs[0] + coeffs[1]*x_fit + coeffs[2]*x_fit**2 + coeffs[3]*x_fit**3

    fig,ax = plt.subplots()

    ax.scatter(x,y,color='red');
    ax.plot(x_fit,y_fit,'b');
    ax.set_xlim(0,20);
    ax.set_ylim(0,30);
    ax.set_xlabel('x');
    ax.set_ylabel('y');
    ax.grid(True);
    return ax, fig, x_fit, y_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Although the curve that we produce does indeed pass through each of the data points, this polynomial may not be the best model of the underlying process.  One potential concern is that the curve does not seem to connect the third and fourth data point in a direct way, but rather exhibits an oscillation.  When constructing a curve to fit a set of data points, there are other factors that may be more important than simply requiring that the curve passes through each point.  In a later chapter we will revisit the problem and consider the idea of finding a curve that "fits" the data without actually passing through each point. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Exercises
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** In 2017, researchers from the Universities of British Columbia, Alberta, and Toronto published their findings regarding the population of snowshoe hares around Kluane Lake, Yukon. They measured the density of hares per hectare, taking a reading once every two years. Here are some of their measurements:

        | Measurement #    | Density per ha. |
        | ---------------- | --------------- |
        | 1                | 0.26            |
        | 2                | 0.20            |
        | 3                | 1.17            |
        | 4                | 2.65            |
        | 5                | 0.14            |
        | 6                | 0.42            |
        | 7                | 1.65            |
        | 8                | 2.73            |
        | 9                | 0.09            |
        | 10               | 0.21            |

        $(a)$ Find the unique ninth degree polynomial whose graph passes through each of these points.  Plot the data points together with the graph of the polynomial to observe the fit.
           
        $(b)$ Using the polynomial that we found, what should we expect the density of hares to be if we measured their population in the year between the third and fourth measurement? What about between the fourth and the fifth?
         
        $(c)$ Why might this method of interpolation not be an appropriate model of our data over time?
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
        **Exercise 2:** The further you travel out into the solar system and away from the Sun, the slower an object must be travelling to remain in its orbit. Here are the average radii of the orbits of the planets in our solar system, and their average orbital velocity around the Sun.

        |Planet                           | Distance from Sun (million km)  | Orbital Velocity (km/s)         |
        | ------------------------------- | ------------------------------- | ------------------------------- |
        |Mercury                          | 57.9                            | 47.4                            |
        |Venus                            | 108.2                           | 35.0                            | 
        |Earth                            | 149.6                           | 29.8                            |
        |Mars                             | 228.0                           | 24.1                            |
        |Jupiter                          | 778.5                           | 13.1                            |
        |Saturn                           | 1432.0                          | 9.7                             |
        |Uranus                           | 2867.0                          | 6.8                             |
        |Neptune                          | 4515.0                          | 5.4                             |

        $(a)$ Find the unique first degree polynomial whose graph passes through points defined by Mercury and Jupiter.  Plot the data points together with the graph of the polynomial to observe the fit. Amend your polynomial and graph by adding Saturn, and then Earth. What do you notice as you add more points? What if you had started with different planets?
           
        $(b)$ Expand your work in part $(a)$ to a seventh degree poynomial that passes through all eight planets. The first object in the Kuiper Belt, Ceres, was discovered by Giuseppe Piazzi in 1801. Ceres has an average distance from the sun of 413.5 million km. Based on the points on the graph, estimate the orbital velocity of Ceres. What does the polynomial suggest the value would be?
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
        ### Cryptography

        One of the primary goals of cryptography is to secure communications by encoding messages.  The encoded message is difficult to read by anyone except for the intended recipient, who possesses some secret knowledge that allows them to reverse the process and decode the message.  The original message is known as _plaintext_, and the encoded message is known as _ciphertext_.  We demonstrate here the well-known Hill Cipher, which is a method of transforming plaintext to ciphertext using familiar matrix multiplication.

        #### Encryption

        Let's assume that the plaintext and ciphertext will be made up of characters from the same predefined _alphabet_.  This alphabet can contain letters, numbers, punctuation, and any number of other symbols that might be appropriate.  We also need a square invertible matrix $B$, called the encryption matrix.  Given these parameters, we take the following steps to get from plaintext to ciphertext.

        1. Translate the plaintext from alphabet characters into a list of numbers.
        2. Divide the list of numbers into a collection of $N\times 1$ vectors.  Use these numbers as columns in a plaintext array $P$.
        3. Apply the cipher by multiplying each vector by the matrix $B$.  This produces a ciphertext array $C=BP$.
        4. Translate the columns of $C$ back into a string of alphabet characters.

        In order to get started, we first need a way to transform a message that includes letters, numbers, and possibly other characters, into to a message that consists of only numbers.  The easiest thing to do is to substitute each possible character in the message for a number.  For example we might let A=1, B=2, C=3, and so on.  In order to make this process less obvious, we might scramble the order of the numbers (A=23, B=5, C=12, ...), or substitute single numbers for common groups of letters (TH=32, EE=20, ING=17, ...).  However we choose to convert our message from text to numbers, there will still be patterns that remain among the numbers due to the natural patterns of the underlying language.  

        For the purpose of this demonstration, we will make use of a list named $\texttt{alphabet}$ to convert between letters and numbers.  This list is included in the $\texttt{hillcipher}$ module, which contains some other functions that we will need.  If we print the list, we can see that it contains a space, a period, a question mark, and all uppercase letters.   
        """
    )
    return


@app.cell
def _():
    import hillcipher as hc
    print(hc.alphabet)
    return (hc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To find which character goes with which number, we can use the number as an index to the list as usual.
        """
    )
    return


@app.cell
def _(hc):
    _num = 2
    _char = hc.alphabet[_num]
    print('The character', _char, 'is associated with the number', _num, '.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To find which number is associated with each letter, we need to look up the index for that letter. 
        """
    )
    return


@app.cell
def _(hc):
    _char = 'R'
    _num = hc.alphabet.index(_char)
    print('The number', _num, 'is associated with the character', _char, '.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that the code in the first cell will produce an error if we choose an index that is larger than 28, and the code in the second cell will produce an error if we choose a letter that is not in $\texttt{alphabet}$. 

        To complete this step of the encryption, we write a loop that builds a list of numbers corresponding to each letter in a message and define a NumPy array based on that list.
        """
    )
    return


@app.cell
def _(hc, np):
    plaintext = 'WE DONT HAVE NUMBERS IN OUR ALPHABET.  WE HAVE TO SPELL TWO.'
    number_message = []
    for _char in plaintext:
        number_message.append(hc.alphabet.index(_char))
    array_message = np.array(number_message)
    print(array_message)
    return array_message, number_message, plaintext


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The next step is to arrange our numerical message into the plaintext array $P$ which must have dimensions compatible with our encryption matrix $B$.  We will choose $B$ to be a $4\times 4$ matrix for this example.

        $$
        B = \left[ \begin{array}{rrrr} 1 & 0 & -2 & -1 \\ 3 & -1 & -3 & 2 \\ 2 & 0 & -4 & 4 \\ 2 & 1 & -1 & -1 \end{array}\right]
        $$

        We now take the plaintext message and break it into chunks which contain 4 numbers each.  (*Note that if the number of characters in our message is not divisible by 4, we can add extra characters at the end.*)  Each chunk will form a column of $P$, and we will have as many columns as needed to accommodate the entire message.  We make use of the $\texttt{reshape}$ and $\texttt{transpose}$ methods to manipulate the entries in the array.
        """
    )
    return


@app.cell
def _(array_message, number_message):
    # Find the number of chucks needed.
    chunks = int(len(number_message)/4)
    P = array_message.reshape((chunks,4))
    P = P.transpose()
    print(P)
    return P, chunks


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The next step is one multiplication multiplication. 
        """
    )
    return


@app.cell
def _(P, np):
    B_1 = np.array([[1, 0, -2, -1], [3, -1, -3, 2], [2, 0, -4, 4], [2, 1, -1, -1]])
    C = B_1 @ P
    print(C)
    return B_1, C


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The final task is to translate the numbers in $C$ back to characters of the alphabet.  This step requires a bit more thought since most of the numbers in $C$ are outside the index range of the alphabet (0-28).  We can let these numbers be associated with letters by letting the count wrap around at 29.  That is, 29 is associated with the same character as 0, 30 is associated with the same character as 1, and so forth.  We can also let -1 be associated with the same character as 28, -2 be associated with the same character as 27, and so forth to address negative numbers.  This concept is known as *congruence* in *modular arithmetic* and is performed using the $\texttt{%}$ operator in Python.  Try it out!   
        """
    )
    return


@app.cell
def _():
    print(80%29)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With that detail in place, we can step through the elements of $C$ and generate the final encrypted message from the characters in our alphabet.  
        """
    )
    return


@app.cell
def _(C, hc):
    encrypted_message = ''
    for _j in range(C.shape[1]):
        for _i in range(C.shape[0]):
            encrypted_message = encrypted_message + hc.alphabet[C[_i, _j] % 29]
    print(encrypted_message)
    return (encrypted_message,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that this last step of the process cannot be reversed.  It is not possible to start with the encrypted message and determine the ciphertext array $C$.  Indeed, there are an infinite number of matrices that could have generated this same encrypted message since any of the entries can be have a multiple of 29 added to it.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Decryption
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If the encryption matrix $B$ is known, we might hope to multiply by the inverse matrix to determine the plaintext array $P = B^{-1}C$.  As we just noted however, we cannot reconstruct $C$ from the encrypted message.  To overcome this difficulty, we need to find the *modular inverse* of the matrix $B$.  That is, we need a matrix $B^{-1}$ such that the entries of $B^{-1}B$ are congruent to the entries of $I$ in the modular arithmetic.  (*Note that we use the same symbol for the modular inverse as we do for the usual inverse matrix.*)

        The method for finding the modular inverse is beyond our scope, but we will make use of the $\texttt{ModularInverseMatrix}$ function in the $\texttt{hillcipher}$ module to calculate $B^{-1}$ for the purpose of demonstration.
        """
    )
    return


@app.cell
def _(B_1, hc):
    B_inv = hc.ModularInverseMatrix(B_1)
    print(B_inv, '\n')
    print(B_inv @ B_1, '\n')
    print(B_inv @ B_1 % 29)
    return (B_inv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What is important to observe here is that $B^{-1}B$ is not exactly $I$, but is congruent to $I$ if we work out the equivalent number between 0 and 28 for each entry.  To proceed with the decryption, we assemble the encrypted message into an array (we call it $\texttt{decryptionC}$ to avoid confusion) and multiply by $B^{-1}$. 
        """
    )
    return


@app.cell
def _(B_inv, encrypted_message, hc, np):
    number_of_columns = int(len(encrypted_message) / 4)
    decryptionC = np.zeros((4, number_of_columns), dtype='int')
    k = 0
    for _j in range(number_of_columns):
        for _i in range(4):
            decryptionC[_i, _j] = hc.alphabet.index(encrypted_message[k])
            k = k + 1
    decryptionP = B_inv @ decryptionC
    print(decryptionC)
    print('\n')
    print(decryptionP)
    return decryptionC, decryptionP, k, number_of_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Again, multiplication by $B^{-1}$ produces numbers that are outside of the range 0-28, so again we apply modular congruence before translating the numbers back into characters in the alphabet.
        """
    )
    return


@app.cell
def _(decryptionP, hc, number_of_columns):
    decrypted_message = ''
    for _j in range(number_of_columns):
        for _i in range(4):
            decrypted_message = decrypted_message + hc.alphabet[decryptionP[_i, _j] % 29]
    print(decrypted_message)
    return (decrypted_message,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Exercises

        The $\texttt{hillcipher}$ module contains two functions that carry out the same steps we have demonstrated here to encrypt/decrypt a message using the included alphabet.  The following cell shows how they are called.
        """
    )
    return


@app.cell
def _(B_1, hc):
    msg = 'Water in short supply.  Send help soon!'
    en_msg = hc.HillCipherEncryption(msg, B_1)
    print(en_msg)
    de_msg = hc.HillCipherDecryption(en_msg, B_1)
    print(de_msg)
    return de_msg, en_msg, msg


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that these functions do some error checking of the original message to be sure that all letters are in uppercase, and no characters outside the alphabet are included in the message that gets encrypted.

        **Exercise 1:** Create your own encryption matrix and apply the steps above to encrypt a message of your choosing.  Check the results with those produced by $\texttt{HillCipherEncryption}$.
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
        **Exercise 2:** Create your own alphabet list to translate messages into numbers.
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
        ### Graph Theory
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A **directed graph** is defined as a set of nodes together with ordered pairs of nodes which define directed edges.  For example, a graph might consist of nodes labeled 0, 1, 2, and 3, and edges (0,1), (1,3), (2,0), (2,1), and (3,1).    The graph can be visualized using points for the nodes and arrows for the edges.

        ![title](img/Graph_example.png)

        Graphs may be used to represent many types of information.  The nodes might represent hubs in a transportation network (such as airports), and the edges the possible routes (flights) available in the network.  The nodes could also represent players in a tournament, with the edges representing results of individual competitions between the players.  The nodes might also represent units of interconnected computer code, with the edges indicating how each unit depends on the others.

        A graph can be described using an **adjacency matrix** $A$.  The entries $a_{ij}$ of the matrix are 1 if there is an edge from node $i$ to node $j$ and 0 otherwise.        
        """
    )
    return


@app.cell
def _(np):
    A_1 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    print(A_1)
    return (A_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For the visualizations we use functions included in the $\texttt{laguide}$ module.  These functions make use of $\texttt{matplotlib}$ and a module called $\texttt{networkx}$.  The first function we will use is called $\texttt{DrawGraph}$, which requires an adjacency matrix in the form of a NumPy array.  
        """
    )
    return


@app.cell
def _(A_1, lag):
    node_positions = lag.DrawGraph(A_1)
    return (node_positions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The function produces a visualization with $\texttt{matplotlib}$, and then returns a Python object containing the positions of the nodes used in making the picture.  Notice that the nodes in this picture are in a different arrangement than in the first picture.  Both pictures represent the same graph so long as edges connect the nodes in the same way.  When we call the function $\texttt{DrawGraph}$, it will generate suitable positions for the nodes *unless we supply it with pre-defined positions.*  We only need to use supply node positions if we want to replicate previous node positions, or if we draw additional features on a graph under consideration.

        We demonstrate here with two examples.  If we use $\texttt{DrawGraph}$ this way
        ```
        old_node_positions = lag.DrawGraph(A,node_positions)
        ```
        we will get an exact copy of the previous picture, and $\texttt{old_node_positions}$ will be a copy of $\texttt{node_positions}$.
        """
    )
    return


@app.cell
def _(A_1, lag, node_positions):
    old_node_positions = lag.DrawGraph(A_1, node_positions)
    return (old_node_positions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we use $\texttt{DrawGraph}$ this way
        ```
        new_node_positions = lag.DrawGraph(A)
        ```
        the function generates a new set of positions and draws a different, but equivalent, picture.
        """
    )
    return


@app.cell
def _(A_1, lag):
    new_node_positions = lag.DrawGraph(A_1)
    return (new_node_positions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For the discussion in this section we will place two restrictions on the edges.

        1.  No edges will be allowed from a node to itself.
        2.  No edges will be repeated.

        The first restriction means that the adjacency matrix will have zeros along the main diagonal.  

        Here is another example of a graph, this one with six nodes.
        """
    )
    return


@app.cell
def _(lag, np):
    B_2 = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
    print(B_2)
    node_positions_1 = lag.DrawGraph(B_2)
    return B_2, node_positions_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Counting paths

        One use of the adjacency matrix is in computation of paths from one node to another.  The $(i,j)$ entries in *powers* of the adjacency matrix will tell us the number of paths between nodes $i$ and $j$ that have length equal to the power.  For example, if $B$ is the adjacency matrix, then $B^3$ contains information about paths of length 3.  

        Let's try it out for the previous example.   
        """
    )
    return


@app.cell
def _(B_2):
    print(B_2 @ B_2 @ B_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see for example that $B^3_{50} = 2$.  This means that there are 2 paths of length 3 from node 5 to node 0.  Similarly we see that $B^3_{22} = 1$, which means that there is only 1 path of length 3 from node 2 back to itself.  Check the graph for yourself to verify these counts.   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Finding cliques

        A **clique** is a set of three or more nodes within a graph such that there is an edge between each node in the clique *in both directions*.  In the example shown below, nodes 2, 3, and 4 form a clique.

        ![title](img/Clique_example.png)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that node 1 in this graph is not part of the clique because it is not connected in both directions to nodes 2 and 4.  In order for a node to be included in the clique, it must be connected in both directions to *every other node in the clique.*

        Identifying cliques in large graphs is a difficult problem.  A graph may contain many cliques, and it may be that some nodes belong to multiple different cliques.  If in the previous example there were an edge from node 2 to node 1, then the graph would have a clique made up of nodes 1, 2 and 3, as well as the clique made up of nodes 2, 3 and 4.

        ![title](img/Clique_example2.png)

        Counting paths using adjacency matrices can be used as a way to identify which nodes are in cliques.  Since these nodes must be connected to other clique nodes in both directions, we will only allow steps in the paths between nodes that are connected by edges in both directions.  If a node is in a clique, there is at least one such path of length 3 that starts and ends at that node.

        Starting with an adjacency matrix $A$, we have a two step process:

        1. Build an adjacency matrix $C$ that includes only the edges between nodes that are connected in both directions.  That is, $c_{ij} = 1$ only if $a_{ij} = 1$ and $a_{ji}=1$.
        2. Use $C^3$ to identify which nodes have paths to themselves of length 3.  That is, check which entries on the main diagonal of $C^3$ are nonzero.
        """
    )
    return


@app.cell
def _(lag, np):
    A_2 = np.array([[0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]])
    print(A_2)
    node_positions_2 = lag.DrawGraph(A_2)
    return A_2, node_positions_2


@app.cell
def _(A_2, np):
    C_1 = np.zeros((5, 5))
    for _i in range(5):
        for _j in range(5):
            if A_2[_i, _j] == 1 and A_2[_j, _i] == 1:
                C_1[_i, _j] = 1
    print(C_1)
    print('\n')
    print(C_1 @ C_1 @ C_1)
    return (C_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see from the result that nodes 0, 1, and 3 are clique nodes.  Since there are only three such nodes, they all must be in the same clique.  We can highlight the clique nodes using the function $\texttt{HighlightSubgraph}$, which works similarly to $\texttt{DrawGraph}$ but also accepts a list of nodes to highlight.
        """
    )
    return


@app.cell
def _(A_2, lag, node_positions_2):
    lag.HighlightSubgraph(A_2, node_positions_2, [0, 1, 3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's try one more example with a larger graph that we build at random.
        """
    )
    return


@app.cell
def _(lag, np):
    import random
    N = 10
    R = np.zeros((N, N))
    for _i in range(N):
        for _j in range(N):
            if random.random() > 0.5 and _i != _j:
                R[_i, _j] = 1
    big_graph_positions = lag.DrawGraph(R)
    return N, R, big_graph_positions, random


@app.cell
def _(N, R, big_graph_positions, lag, np):
    C_2 = np.zeros((N, N))
    for _i in range(N):
        for _j in range(N):
            if R[_i, _j] == 1 and R[_j, _i] == 1:
                C_2[_i, _j] = 1
    _C_3 = C_2 @ C_2 @ C_2
    _clique_nodes = []
    for _i in range(N):
        if _C_3[_i, _i] != 0:
            _clique_nodes.append(_i)
    print(_C_3)
    lag.HighlightSubgraph(R, big_graph_positions, _clique_nodes)
    return (C_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Although the exact picture will change with each new random matrix, we will likely see that there are more than three nodes in the list $\texttt{clique_nodes}$, and that this set of nodes defines more than one clique.  For a graph with 10 nodes, we can still see fairly easily how to sort the nodes into different cliques.  For large graphs we would want to make use of a sorting algorithm.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Tournament analysis

        In a **dominance directed graph** every node is connected to each other node, but only in one direction.  This means the in the adjacency matrix $D$, either $d_{ij} = 1$, or $d_{ji}=1$, but not both.  A dominance directed graph could be used to represent the results of a tournament where every player faced each other player exactly once, and either won or lost the competition.  An edge from player 2 to player 4 means that player 2 won.  Let's look at an example.  
        """
    )
    return


@app.cell
def _(lag, np):
    D = np.array([[0,1,1,0,0],[0,0,0,0,1],[0,1,0,1,1],[1,1,0,0,1],[1,0,0,0,0]])
    print(D)
    positions = lag.DrawGraph(D)
    return D, positions


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this scenario the sum of the rows of $D$ indicate how many wins were earned by each player.  In the example, players 2 and 3 both earned 3 wins, player 0 earned 2 wins, and players 1 and 4 earned a single win each.  In order to distinguish the tournament standings between players 2 and 3, we might also look at all the two-step paths between players using the matrix $D^2$.
        """
    )
    return


@app.cell
def _(D):
    print(D@D)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The number of two-step paths from player 2 represents the number of wins among the players that were defeated by player 2.  Thus if we want to determine a player's tournament standing based not only on their number of wins, but also on the relative standings of the players they defeated, we can add the number of one-step paths (direct wins) to the number of two-step paths (indirect wins).   
        """
    )
    return


@app.cell
def _(D):
    print(D + D@D)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Looking at the sum of the rows of $D + D^2$ we see that player 2 has the highest rating in this particular tournament.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Exercises

        **Exercise 1:** Create your own adjacency matrix and use $\texttt{DrawGraph}$ to make a visualization of the associated graph.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Edit your matrix and redraw the graph to see the changes.  Reuse the positions if you want the nodes to remain in the same locations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Rerun the code in the two cells below until it produces a graph that has at least 5 nodes in the list $\texttt{clique_nodes}$.  Sort the nodes if they belong to different cliques.
        """
    )
    return


@app.cell
def _(lag, np, random):
    N_1 = 10
    R_1 = np.zeros((N_1, N_1))
    for _i in range(N_1):
        for _j in range(N_1):
            if random.random() > 0.5 and _i != _j:
                R_1[_i, _j] = 1
    big_graph_positions_1 = lag.DrawGraph(R_1)
    return N_1, R_1, big_graph_positions_1


@app.cell
def _(N_1, R_1, big_graph_positions_1, lag, np):
    C_3 = np.zeros((N_1, N_1))
    for _i in range(N_1):
        for _j in range(N_1):
            if R_1[_i, _j] == 1 and R_1[_j, _i] == 1:
                C_3[_i, _j] = 1
    _C_3 = C_3 @ C_3 @ C_3
    _clique_nodes = []
    for _i in range(N_1):
        if _C_3[_i, _i] != 0:
            _clique_nodes.append(_i)
    print(_C_3)
    lag.HighlightSubgraph(R_1, big_graph_positions_1, _clique_nodes)
    return (C_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Write a code cell to sort the $\texttt{clique_nodes}$ list into different cliques.  Print the result or store it in a list of lists.   
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

        - Anton, Howard and Chris Rorres. *Elementary Linear Algebra Applications Version*. 8th ed., John Wiley & Sons Inc., 2000. 

        - Krebs, Charles J.; Boonstra, Rudy; Boutin, Stan (2017), Using experimentation to understand the 10‐year snowshoe hare cycle in the boreal forest of North America, Journal of Animal Ecology, Article-journal, https://doi.org/10.1111/1365-2656.12720

        - Kwak, Jin Ho and Sungpyo Hong.  *Linear Algebra*. 2nd ed., Birkhauser., 2004.

        - Williams, David, *Planetary Fact Sheet*, https://nssdc.gsfc.nasa.gov/planetary/factsheet/, NASA Goddard Space Flight Center, 2021
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
