package ast.memory;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import compiler.RefCodeBlock;
import environment.Environment;
import typeError.IllegalOperatorException;
import typeError.TypeErrorException;
import types.IType;
import types.TypeRef;
import values.IValue;
import values.VCell;

public class ASTAssign extends ASTNodeAbstract
{
    public static final String OPERATOR = ":=";

    protected ASTNode left, right;

    /**
     * @param left
     * @param right
     */
    public ASTAssign(ASTNode left, ASTNode right) {
        this.left = left;
        this.right = right;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        this.right.compile(c, e);
        c.emit("dup");
        this.left.compile(c, e);
        c.emit("swap");
        
        RefCodeBlock ref = c.getRefClass(this.right.getType());
        c.emit(String.format("putfield %s/v %s", ref.getClassName(), ref.getValueFieldTypeJVM()));
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        VCell cell = checkRuntimeTypeVCell(this.left.eval(e));
        IValue value = this.right.eval(e);
        cell.setValue(value);
        return value;
    }

    @Override
    public IType typecheck(Environment<IType> e) {
        TypeRef ref = checkTypeRef(this.left.typecheck(e));
        IType valueType = this.right.typecheck(e);

        // check if the type of the value in this reference equals the valueType
        boolean checked = ref.getValueType().equals(valueType);

        if (!checked)
            throw new TypeErrorException(
                String.format("Incompatible type for assignment - %s.\nExpected value type: %s\n",
                    valueType.show(), ref.getValueType().show()));
                    
        type = valueType;
        return valueType;
    }

    protected VCell checkRuntimeTypeVCell(IValue value)
    {
        boolean checked = value instanceof VCell;

        if (!checked)
            throw new IllegalOperatorException(OPERATOR, "Cannot assign a value to a non reference type.",
                "Ref", value.show());

        return (VCell)value;
    }

    protected TypeRef checkTypeRef(IType type)
    {
        boolean checked =  type instanceof TypeRef;

        if (!checked)
            throw new IllegalOperatorException(OPERATOR, "Cannot assign a value to a non reference type.",
                "Ref", type.show());

        return (TypeRef)type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        this.left.toString(builder);
        builder.append(OPERATOR);
        this.right.toString(builder);

        return builder;
    }

    
}
