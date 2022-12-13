package ast;

import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;

/**
 * A node that has an optimized compile method for boolean expressions.
 */
public interface ASTNodeShortCircuit extends ASTNode
{
    /**
     * Compiles this node using the specified environment and stores the compilation
     * result into the code bloc.
     * This compile is optimized to jump to a specified label without pushing a boolean to the stack.
     * 
     * @param c - The code block to store the result of the compilation.
     * @param e - The Environment for the compiler, an id maps to Coordinates.
     * @param tl - True label.
     * @param fl - False label.
     */
    void compile(MainCodeBlock c, Environment<Coordinates> e, String tl, String fl);
}
