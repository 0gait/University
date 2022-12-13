package ast.memory;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import compiler.RefCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.TypeRef;
import util.Utils;
import values.IValue;
import values.VCell;

public class ASTAssign extends ASTNodeAbstract
{
    public static final String OPERATOR = ":=";

    private static final String ERROR_MSG = "Cannot assign a value to a non reference type.";

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
        VCell cell = Utils.checkValueForOperation(this.left.eval(e), VCell.class, OPERATOR, ERROR_MSG);
        IValue value = Utils.requireNonNull(this.right.eval(e));
        cell.setValue(value);
        return value;
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        TypeRef ref = Utils.checkTypeForOperation(this.left.typecheck(e), TypeRef.class,OPERATOR, ERROR_MSG);
        IType valueType = this.right.typecheck(e);
        Utils.requireNonNull(valueType);
        // check if the type of the value in this reference equals the valueType
        boolean checked = ref.getValueType().equals(valueType);

        if (!checked)
            throw new TypeErrorException(
                String.format("Incompatible type for assignment - %s.\nExpected value type: %s\n",
                    valueType.show(), ref.getValueType().show()));
                    
        this.type = valueType;
        return valueType;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        ((ASTNodeAbstract)(this.left)).toString(builder);
        builder.append(OPERATOR);
        ((ASTNodeAbstract)(this.right)).toString(builder);

        return builder;
    }
}
