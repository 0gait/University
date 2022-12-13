package ast.binaryOperations;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import ast.ASTNodeShortCircuit;
import ast.binaryOperations.operators.LogicalBinaryOperator;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.primitives.TypeBool;
import util.Utils;
import values.IValue;
import values.primitive.VBool;

public class ASTLogicalBinaryOperation extends ASTNodeAbstract implements ASTNodeShortCircuit
{
    protected final ASTNode left, right;

    protected final LogicalBinaryOperator operator;

    /**
     * @param left
     * @param rigth
     */
    public ASTLogicalBinaryOperation(ASTNode left, ASTNode rigth, LogicalBinaryOperator operator) {
        this.left = left;
        this.right = rigth;
        this.operator = operator;
        type = TypeBool.TYPE;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
        String tl, fl, end;
        tl = c.getNewLabelId();
        fl = c.getNewLabelId();
        end = c.getNewLabelId();

        compile(c, e, tl, fl);
        c.emit(tl + ":");
        c.emit("sipush 1");
        c.emit("goto " + end);
        c.emit(fl + ":");
        c.emit("sipush 0");
        c.emit(end + ":");
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e, String tl, String fl)
    {
        boolean optimizeLeft = this.left instanceof ASTNodeShortCircuit;
        boolean optimizeRight = this.right instanceof ASTNodeShortCircuit;

        switch(this.operator)
        {
            case AND:
                if (optimizeLeft)
                {
                    String auxLabel = c.getNewLabelId();
                    ((ASTNodeShortCircuit)this.left).compile(c, e, auxLabel, fl);
                    c.emit(auxLabel + ":");
                }
                else
                {
                    this.left.compile(c, e);
                    c.emit("ifeq " + fl);
                }
                break;

            case OR:
                if (optimizeLeft)
                {
                    String auxLabel = c.getNewLabelId();
                    ((ASTNodeShortCircuit) this.left).compile(c, e, tl, auxLabel);
                    c.emit(auxLabel + ":");
                }
                else
                {
                    this.left.compile(c, e);
                    c.emit("ifne " + tl);
                }
                break;
        }

        if (optimizeRight)
            ((ASTNodeShortCircuit) this.right).compile(c, e, tl, fl);
        else {
            this.right.compile(c, e);
            c.emit("ifne " + tl);
            c.emit("goto " + fl);
        }
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        VBool val1 = Utils.checkValueForOperation(this.left.eval(e), VBool.class, operator.getOperator());

        IValue result = null;
        switch(this.operator)
        {
            case AND:
                result = new VBool(val1.getValue() && Utils.checkValueForOperation(this.right.eval(e), VBool.class, operator.getOperator()).getValue());
                break;
            case OR:
                result = new VBool(val1.getValue() || Utils.checkValueForOperation(this.right.eval(e), VBool.class, operator.getOperator()).getValue());
                break;
        }

        return result;
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {    
        Utils.checkTypeForOperation(this.left.typecheck(e), TypeBool.class, operator.getOperator());
        return Utils.checkTypeForOperation(this.right.typecheck(e), TypeBool.class, operator.getOperator());
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        ((ASTNodeAbstract)(this.left)).toString(builder);
        builder.append(' ');
        builder.append(this.operator.getOperator());
        builder.append(' ');
        ((ASTNodeAbstract)(this.right)).toString(builder);

        return builder;
    }
}
