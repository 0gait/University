package compiler;

import java.io.PrintStream;

/**
 * Represents a Code Block, i.e., a class or interface that will be result of compilation.
 */
public abstract class CodeBlock
{
    /**
     * The className of this CodeBlock.
     */
    public final String className;
    
    protected CodeBlock(String className){
        this.className = className;
    }

    /**
     * Dumps the code to a printStream.
     * @param f - The stream to print the generated code.
     */
    public abstract void dump(PrintStream f);
}
