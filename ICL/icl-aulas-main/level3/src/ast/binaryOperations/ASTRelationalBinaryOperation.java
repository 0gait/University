package ast.binaryOperations;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import ast.ASTNodeShortCircuit;
import ast.binaryOperations.operators.RelationalBinaryOperator;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.IllegalOperatorException;
import typeError.TypeErrorException;
import types.IType;
import types.primitives.TypeBool;
import types.primitives.TypePrimitive;
import types.primitives.TypeString;
import util.Utils;
import values.IValue;
import values.primitive.VBool;
import values.primitive.VPrimitive;

public class ASTRelationalBinaryOperation extends ASTNodeAbstract implements ASTNodeShortCircuit
{
    protected final ASTNode left, right;

    protected final RelationalBinaryOperator operator;

    /**
     * @param left
     * @param right
     */
    public ASTRelationalBinaryOperation(ASTNode left, ASTNode right, RelationalBinaryOperator operator) {
        this.left = left;
        this.right = right;
        this.operator = operator;
        this.type = TypeBool.TYPE;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
        if(left.getType() instanceof TypeString)
            compileString(c, e);
        else   
            compileIntOrBool(c, e);

    }

    private void compileIntOrBool(MainCodeBlock c, Environment<Coordinates> e)
    {
        this.left.compile(c, e);
        this.right.compile(c, e);

        String l1 = c.getNewLabelId();
        String l2 = c.getNewLabelId();

        switch(this.operator)
        {
            case EQUALS:
                c.emit("if_icmpeq " + l1);
                break;
            case GREATER_THAN:
                c.emit("if_icmpgt " + l1);
                break;
            case GREATER_THAN_OR_EQUAL_TO:
                c.emit("if_icmpge " + l1);
                break;
            case LESS_THAN:
                c.emit("if_icmplt " + l1);
                break;
            case LESS_THAN_OR_EQUAL_TO:
                c.emit("if_icmple " + l1);
                break;
            case NOT_EQUALS:
                c.emit("if_icmpne " + l1);
                break;
        }

        c.emit("sipush 0");
        c.emit("goto " + l2);
        c.emit(l1 + ":");
        c.emit("sipush 1");
        c.emit(l2 + ":");
    }

    private void compileString(MainCodeBlock c, Environment<Coordinates> e)
    {
        this.left.compile(c, e);
        this.right.compile(c, e);

        if (this.operator == RelationalBinaryOperator.EQUALS ||
            this.operator == RelationalBinaryOperator.NOT_EQUALS)
        {
            c.emit("invokevirtual java/lang/String/equals(Ljava/lang/Object;)Z");

            if (this.operator == RelationalBinaryOperator.NOT_EQUALS)
            {
                c.emit("sipush 1");
                c.emit("ixor");
            }

            return;
        }

        String l1 = c.getNewLabelId();
        String l2 = c.getNewLabelId();

        c.emit("invokevirtual java/lang/String/compareTo(Ljava/lang/String;)I");

        switch(this.operator)
        {
            case GREATER_THAN:
                c.emit("ifgt " + l1);
                break;
            case GREATER_THAN_OR_EQUAL_TO:
                c.emit("ifge" + l1);
                break;
            case LESS_THAN:
                c.emit("iflt " + l1);
                break;
            case LESS_THAN_OR_EQUAL_TO:
                c.emit("ifle " + l1);
                break;

            default:
                return;
        }

        c.emit("sipush 0");
        c.emit("goto " + l2);
        c.emit(l1 + ":");
        c.emit("sipush 1");
        c.emit(l2 + ":");
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e, String tl, String fl) {
        if(left.getType() instanceof TypeString)
            compileString(c, e, tl, fl);
        else   
            compileIntOrBool(c, e, tl, fl);
    }

    public void compileIntOrBool(MainCodeBlock c, Environment<Coordinates> e, String tl, String fl)
    {
        this.left.compile(c, e);
        this.right.compile(c, e);

        switch(this.operator)
        {
            case EQUALS:
                c.emit("if_icmpeq " + tl);
                break;
            case GREATER_THAN:
                c.emit("if_icmpgt " + tl);
                break;
            case GREATER_THAN_OR_EQUAL_TO:
                c.emit("if_icmpge " + tl);
                break;
            case LESS_THAN:
                c.emit("if_icmplt " + tl);
                break;
            case LESS_THAN_OR_EQUAL_TO:
                c.emit("if_icmple " + tl);
                break;
            case NOT_EQUALS:
                c.emit("if_icmpne " + tl);
                break;
        }

        c.emit("goto " + fl);
    }

    public void compileString(MainCodeBlock c, Environment<Coordinates> e, String tl, String fl)
    {
        this.left.compile(c, e);
        this.right.compile(c, e);

        switch(this.operator)
        {
            case EQUALS:
                c.emit("invokevirtual java/lang/String/equals(Ljava/lang/Object;)Z");
                c.emit("ifne " + tl);
                break;
            case GREATER_THAN:
                c.emit("invokevirtual java/lang/String/compareTo(Ljava/lang/String;)I");
                c.emit("ifgt " + tl);
                break;
            case GREATER_THAN_OR_EQUAL_TO:
                c.emit("invokevirtual java/lang/String/compareTo(Ljava/lang/String;)I");
                c.emit("ifge " + tl);
                break;
            case LESS_THAN:
                c.emit("invokevirtual java/lang/String/compareTo(Ljava/lang/String;)I");
                c.emit("iflt " + tl);
                break;
            case LESS_THAN_OR_EQUAL_TO:
                c.emit("invokevirtual java/lang/String/compareTo(Ljava/lang/String;)I");
                c.emit("ifle " + tl);
                break;
            case NOT_EQUALS:
                c.emit("invokevirtual java/lang/String/equals(Ljava/lang/Object;)Z");
                c.emit("ifeq " + tl);
                break;
        }

        c.emit("goto " + fl);
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        VPrimitive<?> val1 = Utils.checkValueForOperation(this.left.eval(e), VPrimitive.class, operator.getOperator());
        VPrimitive<?> val2 = Utils.checkValueForOperation(this.right.eval(e), VPrimitive.class, operator.getOperator());

        // Check if val1 type equals val2 type
        if (!val1.getClass().equals(val2.getClass()))
            throw new IllegalOperatorException("The type of the 2 expressions must be equal.",
                this.operator.getOperator(), val1.getPrimitiveType().show(), val2.getPrimitiveType().show())
                .toRuntimeException();

        IValue result = null;
        switch(this.operator)
        {
            case EQUALS:
                result = new VBool(val1.equals(val2));
                break;
            case GREATER_THAN:
                result = new VBool(vPrimitiveCompareTo(val1, val2) > 0);
                break;
            case GREATER_THAN_OR_EQUAL_TO:
                result = new VBool(vPrimitiveCompareTo(val1, val2) >= 0);
                break;
            case LESS_THAN:
                result = new VBool(vPrimitiveCompareTo(val1, val2) < 0);
                break;
            case LESS_THAN_OR_EQUAL_TO:
                result = new VBool(vPrimitiveCompareTo(val1, val2) <= 0);
                break;
            case NOT_EQUALS:
                result = new VBool(!val1.equals(val2));
                break;
        }

        return result;
    }

    @SuppressWarnings("unchecked")
    private static int vPrimitiveCompareTo(VPrimitive value1, VPrimitive value2)
    {
        return value1.compareTo(value2);
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        IType type1 = Utils.checkTypeForOperation(this.left.typecheck(e), TypePrimitive.class, operator.getOperator());
        IType type2 = Utils.checkTypeForOperation(this.right.typecheck(e), TypePrimitive.class, operator.getOperator());
        if(!type1.equals(type2))
            throw new IllegalOperatorException("The type of the 2 expressions must be equal.",
            this.operator.getOperator(), type1.show(), type2.show());
            
        return this.type; 
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
