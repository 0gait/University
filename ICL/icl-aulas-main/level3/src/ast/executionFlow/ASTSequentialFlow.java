package ast.executionFlow;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import values.IValue;

public class ASTSequentialFlow extends ASTNodeAbstract
{
    public static final String OPERATOR = ";";
    
    protected ASTNode left, right;

    /**
     * @param left
     * @param right
     */
    public ASTSequentialFlow(ASTNode left, ASTNode right) {
        this.left = left;
        this.right = right;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        
        left.compile(c, e);
        c.emit("pop");
        right.compile(c, e);
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        this.left.eval(e);

        // returns the value of the right node.
        return this.right.eval(e);
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        this.left.typecheck(e);
        type = this.right.typecheck(e);
        return type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        ((ASTNodeAbstract)(this.left)).toString(builder);
        builder.append(";\n");
        ((ASTNodeAbstract)(this.right)).toString(builder);

        return builder;
    }

    
}
