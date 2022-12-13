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

public class ASTNew extends ASTNodeAbstract
{
    protected ASTNode val;

    /**
     * @param val
     */
    public ASTNew(ASTNode val) {
        this.val = val;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
        RefCodeBlock ref = c.getRefClass(val.getType());
        String className = ref.getClassName();
        c.emit("new " + className);
        c.emit("dup");
        c.emit(String.format("invokespecial %s/<init>()V", className));
        c.emit("dup");
        val.compile(c, e);
        c.emit(String.format("putfield %s/v %s", className, ref.getValueFieldTypeJVM()));
    }

    @Override
    public IValue eval(Environment<IValue> e) {
        return new VCell(Utils.requireNonNull(this.val.eval(e)));
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        type = new TypeRef(Utils.requireNonNull(this.val.typecheck(e)));
        return type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append("new (");
        ((ASTNodeAbstract)this.val).toString(builder);
        builder.append(')');

        return builder;
    }
}
