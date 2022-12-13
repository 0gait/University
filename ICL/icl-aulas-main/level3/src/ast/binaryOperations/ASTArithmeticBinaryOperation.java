package ast.binaryOperations;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import ast.binaryOperations.operators.ArithmeticBinaryOperator;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.IllegalOperatorException;
import typeError.TypeErrorException;
import types.IType;
import types.primitives.TypeBool;
import types.primitives.TypeInt;
import types.primitives.TypePrimitive;
import types.primitives.TypeString;
import util.Utils;
import values.IValue;
import values.primitive.VBool;
import values.primitive.VInt;
import values.primitive.VPrimitive;
import values.primitive.VString;

public class ASTArithmeticBinaryOperation extends ASTNodeAbstract
{
    protected final ASTNode left, right;

    protected final ArithmeticBinaryOperator operator;

    /**
     * @param left
     * @param rigth
     */
    public ASTArithmeticBinaryOperation(ASTNode left, ASTNode rigth, ArithmeticBinaryOperator operator) {
        this.left = left;
        this.right = rigth;
        this.operator = operator;
       
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        if( left.getType() instanceof TypeInt)
            compileInt(c, e);
        else 
            compileString(c, e);
    }

    private void compileInt(MainCodeBlock c, Environment<Coordinates> e){
        this.left.compile(c, e);
        this.right.compile(c, e);
        switch(this.operator)
        {
            case ADD:
                c.emit("iadd");
                break;
            case DIV:
                c.emit("idiv");
                break;
            case MUL:
                c.emit("imul");
                break;
            case SUB:
                c.emit("isub");
                break;
        }
    }

    private void compileString(MainCodeBlock c, Environment<Coordinates> e)
    {
        this.left.compile(c, e);
        this.right.compile(c, e);


        if(!(right.getType() instanceof TypeString))
            c.emit(String.format("invokestatic java/lang/String/valueOf(%s)Ljava/lang/String;",
                right.getType().getJvmType()));
        
        c.emit("invokevirtual java/lang/String/concat(Ljava/lang/String;)Ljava/lang/String;");
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        VPrimitive<?> val1 = checkRuntimeTypeLeft(this.left.eval(e));
        IValue val2 = this.right.eval(e);

        if (val1 instanceof VInt)
            return evalInt((VInt)val1, val2);
        
        return evalString((VString)val1, val2);
    }

    private IValue evalInt(VInt value1, IValue value2)
    {
        VInt valueInt2 = Utils.checkValueForOperation(value2, VInt.class, operator.getOperator());
        IValue result = null;

        switch(this.operator)
        {
            case ADD:
                result = new VInt(value1.getValue() + valueInt2.getValue());
                break;
            case DIV:
                result = new VInt(value1.getValue() / valueInt2.getValue());
                break;
            case MUL:
                result = new VInt(value1.getValue() * valueInt2.getValue());
                break;
            case SUB:
                result = new VInt(value1.getValue() - valueInt2.getValue());
                break;
        }

        return result;
    }

    private IValue evalString(VString value1, IValue value2)
    {
        IValue result = null;

        switch(this.operator)
        {
            case ADD:
                VPrimitive<?> value2Primitive =
                    Utils.checkValueForOperation(value2, VPrimitive.class, operator.getOperator());
                
                result = new VString(value1.getValue() + value2Primitive.getValue());
                break;
            default:
                throw new IllegalOperatorException(operator.getOperator(), TypeString.TYPE.show())
                    .toRuntimeException();
        }

        return result;
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        TypePrimitive type1 = checkTypeLeft(this.left.typecheck(e));
        IType type2 = this.right.typecheck(e);

        if (type1 instanceof TypeString)
        {
            if (this.operator == ArithmeticBinaryOperator.ADD)
            {
                Utils.checkTypeForOperation(type2, TypePrimitive.class, operator.getOperator());
                return this.type = TypeString.TYPE;
            }
            else
                throw new IllegalOperatorException(operator.getOperator(), TypeString.TYPE.show());
        } else
        {
            // type 1 is TypeInt
            Utils.checkTypeForOperation(type2, TypeInt.class, operator.getOperator());
            return this.type = TypeInt.TYPE;
        }
    }

    protected VPrimitive<?> checkRuntimeTypeLeft(IValue val)
    {
        boolean checked = (val instanceof VPrimitive) && !(val instanceof VBool);

        if (!checked)
            throw new IllegalOperatorException(operator.getOperator(), val.getType().show())
                .toRuntimeException();

        return (VPrimitive<?>)val;
    }

    protected TypePrimitive checkTypeLeft(IType type) throws TypeErrorException
    {
        boolean checked = (type instanceof TypePrimitive) && !(type instanceof TypeBool);

        if (!checked)
            throw new IllegalOperatorException(operator.getOperator(), type.show());

        return (TypePrimitive)type;
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
