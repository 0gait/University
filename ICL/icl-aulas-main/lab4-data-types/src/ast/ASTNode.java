package ast;

import compiler.MainCodeBlock;
import types.IType;
import util.Coordinates;
import util.Environment;
import values.IValue;

public interface ASTNode {

    /**
     * Calculates the result of this node using the specified Environment.
     * @param e - A Environment for the interpreter, an id maps to a result.
     * @return The calculated result.
     */
    IValue eval(Environment<IValue> e);

    /**
     * Compiles this node using the specified environment and stores the compilation
     * result into the code bloc.
     * 
     * @param c - The code block to store the result of the compilation.
     * @param e - The Environment for the compiler, an id maps to Coordinates.
     */
    void compile(MainCodeBlock c, Environment<Coordinates> e);

    /**
     * Checks the types.
     * @param e
     * @return
     */
    IType typecheck(Environment<IType> e);


    IType getType();
    
	
}

