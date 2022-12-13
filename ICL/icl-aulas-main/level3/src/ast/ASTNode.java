package ast;

import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import values.IValue;

/**
 * Represents an Abstract Syntax Node.
 */
public interface ASTNode {

    /**
     * Calculates the result of this node using the specified Environment.
     * @param e - An Environment for the interpreter, an id maps to a result.
     * @return The calculated result.
     */
    IValue eval(Environment<IValue> e);

    /**
     * Compiles this node using the specified environment and stores the compilation
     * result into the code block.
     * 
     * @param c - The code block to store the result of the compilation.
     * @param e - The Environment for the compiler, an id maps to Coordinates.
     */
    void compile(MainCodeBlock c, Environment<Coordinates> e);

    /**
     * Checks the types.
     * @param e - The Environment for the typechecker, an id maps to Type.
     * 
     * @return The type of the expression.
     */
    IType typecheck(Environment<IType> e) throws TypeErrorException;

    /**
     * Gets the type of this node.
     * @return The type of this node.
     */
    IType getType();	
}

