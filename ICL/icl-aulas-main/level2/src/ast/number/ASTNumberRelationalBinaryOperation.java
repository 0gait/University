package ast.number;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.IllegalOperatorException;
import types.IType;
import types.primitives.TypeBool;
import types.primitives.TypeInt;
import values.IValue;
import values.primitive.VBool;
import values.primitive.VInt;

public class ASTNumberRelationalBinaryOperation extends ASTNodeAbstract
{
    protected final ASTNode left, rigth;

    protected final NumberRelationalBinaryOperator operator;

    /**
     * @param left
     * @param rigth
     */
    public ASTNumberRelationalBinaryOperation(ASTNode left, ASTNode rigth, NumberRelationalBinaryOperator operator) {
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
        String l1 = c.getNewLabelId();
        String l2 = c.getNewLabelId();
        switch(this.operator)
        {
            case EQUALS:
            c.emit("if_icmpeq " + l1);
            c.emit("sipush 0");
            c.emit("goto " + l2);
            c.emit(l1 + ":");
            c.emit("sipush 1");
            c.emit(l2 + ":");
                break;
            case GREATER_THAN:
            c.emit("if_icmpgt " + l1);
            c.emit("sipush 0");
            c.emit("goto " + l2);
            c.emit(l1 + ":");
            c.emit("sipush 1");
            c.emit(l2 + ":");
                break;
            case GREATER_THAN_OR_EQUAL_TO:
            c.emit("if_icmpge " + l1);
            c.emit("sipush 0");
            c.emit("goto " + l2);
            c.emit(l1 + ":");
            c.emit("sipush 1");
            c.emit(l2 + ":");
                break;
            case LESS_THAN:
            c.emit("if_icmplt " + l1);
            c.emit("sipush 0");
            c.emit("goto " + l2);
            c.emit(l1 + ":");
            c.emit("sipush 1");
            c.emit(l2 + ":");
                break;
            case LESS_THAN_OR_EQUAL_TO:
            c.emit("if_icmple " + l1);
            c.emit("sipush 0");
            c.emit("goto " + l2);
            c.emit(l1 + ":");
            c.emit("sipush 1");
            c.emit(l2 + ":");
                break;
            case NOT_EQUALS:
            c.emit("if_icmpne " + l1);
            c.emit("sipush 0");
            c.emit("goto " + l2);
            c.emit(l1 + ":");
            c.emit("sipush 1");
            c.emit(l2 + ":");
                break;
        } 
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        VInt val1 = checkRuntimeType(this.left.eval(e));
        VInt val2 = checkRuntimeType(this.rigth.eval(e));

        IValue result = null;
        switch(this.operator)
        {
            case EQUALS:
                result = new VBool(val1.getValue() == val2.getValue());
                break;
            case GREATER_THAN:
                result = new VBool(val1.getValue() > val2.getValue());
                break;
            case GREATER_THAN_OR_EQUAL_TO:
                result = new VBool(val1.getValue() >= val2.getValue());
                break;
            case LESS_THAN:
                result = new VBool(val1.getValue() < val2.getValue());
                break;
            case LESS_THAN_OR_EQUAL_TO:
                result = new VBool(val1.getValue() <= val2.getValue());
                break;
            case NOT_EQUALS:
                result = new VBool(val1.getValue() != val2.getValue());
                break;
        }

        return result;
    }

    @Override
    public IType typecheck(Environment<IType> e) {
        checkType(this.left.typecheck(e));
        checkType(this.rigth.typecheck(e));
        return  type;
    }

    protected VInt checkRuntimeType(IValue val)
    {
        boolean checked = val instanceof VInt;
        if (!checked)
            throw new IllegalOperatorException(this.operator.getOperator(), TypeInt.TYPE.show(), val.show());

        return (VInt)val;
    }
    
    protected IType checkType(IType type)
    {
        boolean checked =  type instanceof TypeInt;
        if (!checked)
            throw new IllegalOperatorException(this.operator.getOperator(), TypeInt.TYPE.show(), type.show());

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
