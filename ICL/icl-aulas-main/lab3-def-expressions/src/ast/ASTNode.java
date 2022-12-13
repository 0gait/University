package ast;

import compiler.MainCodeBlock;
import util.Coordinates;
import util.Environment;

public interface ASTNode {

    /**
     * Calculates the result of this node using the specified Environment.
     * @param e - A Environment for the interpreter, an id maps to a result (Integer).
     * @return The calculated result.
     */
    int eval(Environment<Integer> e);

    /**
     * Compiles this node using the specified environment and stores the compilation
     * result into the code bloc.
     * 
     * @param c - The code block to store the result of the compilation.
     * @param e - The Environment for the compiler, an id maps to Coordinates.
     */
    void compile(MainCodeBlock c, Environment<Coordinates> e);
	
}

