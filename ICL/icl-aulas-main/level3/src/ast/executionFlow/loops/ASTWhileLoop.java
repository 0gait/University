package ast.executionFlow.loops;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import ast.ASTNodeShortCircuit;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.TypeNull;
import types.primitives.TypeBool;
import util.Utils;
import values.IValue;
import values.VNull;
import values.primitive.VBool;

public class ASTWhileLoop extends ASTNodeAbstract
{
    public static final String OPERATOR = "while do";

    protected ASTNode whileConditionNode, bodyNode;

    /**
     * @param whileConditionNode
     * @param bodyNode
     */
    public ASTWhileLoop(ASTNode whileConditionNode, ASTNode bodyNode) {
        this.whileConditionNode = whileConditionNode;
        this.bodyNode = bodyNode;
        this.type = TypeNull.TYPE;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
        if (this.whileConditionNode instanceof ASTNodeShortCircuit)
            compileWithShortCircuit(c, e);
        else
            compileDefault(c, e);
    }

    public void compileDefault(MainCodeBlock c, Environment<Coordinates> e)
    {
        String l1, l2;
        l1 = c.getNewLabelId();
        l2 = c.getNewLabelId();

        c.emit(l1 + ":");
        this.whileConditionNode.compile(c, e);
        c.emit("ifeq " + l2);
        this.bodyNode.compile(c, e);
        c.emit("pop");
        c.emit("goto " + l1);
        c.emit(l2 + ":");

        // Push something to the stack.
        c.emit("sipush 0");
    }

    public void compileWithShortCircuit(MainCodeBlock c, Environment<Coordinates> e)
    {
        String lStart, tl, fl;
        lStart = c.getNewLabelId();
        tl = c.getNewLabelId();
        fl = c.getNewLabelId();

        c.emit(lStart + ":");
        ((ASTNodeShortCircuit)this.whileConditionNode).compile(c, e, tl, fl);
        c.emit(tl + ":");
        this.bodyNode.compile(c, e);
        c.emit("pop");
        c.emit("goto " + lStart);
        c.emit(fl + ":");

        c.emitNull();
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        VBool condition = Utils.checkValueForOperation(this.whileConditionNode.eval(e), VBool.class, OPERATOR);

        while (condition.getValue())
        {
            this.bodyNode.eval(e);
            condition = (VBool)this.whileConditionNode.eval(e);
        }

        return VNull.VALUE;
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        Utils.checkTypeForOperation(this.whileConditionNode.typecheck(e), TypeBool.class,OPERATOR);
        this.bodyNode.typecheck(e);
      
        return this.type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append("while (");
        ((ASTNodeAbstract)this.whileConditionNode).toString(builder);
        builder.append(") do\n\t");
        ((ASTNodeAbstract)this.bodyNode).toString(builder);
        builder.append("\nend\n");

        return builder;
    }
}
