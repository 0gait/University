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

public class ASTDereference extends ASTNodeAbstract
{
    public static final String OPERATOR = "!";

    protected ASTNode reference;

    /**
     * @param reference
     */
    public ASTDereference(ASTNode reference) {
        this.reference = reference;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
       reference.compile(c, e);
       RefCodeBlock ref = c.getRefClass(((TypeRef)reference.getType()).getValueType());
       c.emit(String.format("getfield %s/v %s", ref.getClassName(), ref.getValueFieldTypeJVM()));
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        return Utils.checkValueForOperation(this.reference.eval(e), VCell.class, OPERATOR).getValue();
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        type = Utils.checkTypeForOperation(this.reference.typecheck(e), TypeRef.class, OPERATOR).getValueType();
        return type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append(OPERATOR);
        ((ASTNodeAbstract)this.reference).toString(builder);
        return builder;
    }
}
