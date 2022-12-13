package ast.bool;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.IllegalOperatorException;
import types.IType;
import types.primitives.TypeBool;
import values.IValue;
import values.primitive.VBool;

public class ASTLogicalBinaryOperation extends ASTNodeAbstract
{
    protected final ASTNode left, rigth;

    protected final LogicalBinaryOperator operator;

    /**
     * @param left
     * @param rigth
     */
    public ASTLogicalBinaryOperation(ASTNode left, ASTNode rigth, LogicalBinaryOperator operator) {
        this.left = left;
        this.rigth = rigth;
        this.operator = operator;
        type = TypeBool.TYPE;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
        this.left.compile(c, e);
        this.rigth.compile(c, e);
        switch(this.operator)
        {
            case AND:
              c.emit("iand");
                break;
            case OR:
               c.emit("ior");
                break;
        }   
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        VBool val1 = checkRuntimeType(this.left.eval(e));
        VBool val2 = checkRuntimeType(this.rigth.eval(e));

        IValue result = null;
        switch(this.operator)
        {
            case AND:
                result = new VBool(val1.getValue() && val2.getValue());
                break;
            case OR:
                result = new VBool(val1.getValue() || val2.getValue());
                break;
        }

        return result;
    }

    @Override
    public IType typecheck(Environment<IType> e) {
        
       checkType(this.left.typecheck(e));
        return checkType(this.rigth.typecheck(e));
    }

    protected VBool checkRuntimeType(IValue val)
    {
        boolean checked = val instanceof VBool;
        if (!checked)
            throw new IllegalOperatorException(this.operator.getOperator(), TypeBool.TYPE.show(), val.show());

        return (VBool)val;
    }

    protected IType checkType(IType type)
    {
        boolean checked = type instanceof TypeBool;
        if (!checked)
            throw new IllegalOperatorException(this.operator.getOperator(), TypeBool.TYPE.show(), type.show());

        return type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        this.left.toString(builder);
        builder.append(' ');
        builder.append(this.operator.getOperator());
        builder.append(' ');
        this.rigth.toString(builder);

        return builder;
    }

    



}
