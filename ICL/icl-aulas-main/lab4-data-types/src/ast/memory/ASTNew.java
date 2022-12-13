package ast.memory;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.MainCodeBlock;
import compiler.RefCodeBlock;
import types.IType;
import types.TypeRef;
import util.Coordinates;
import util.Environment;
import values.IValue;
import values.VCell;

public class ASTNew extends ASTNodeAbstract {
    protected ASTNode val;

    /**
     * @param val
     */
    public ASTNew(ASTNode val) {
        this.val = val;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
       RefCodeBlock ref = c.createRefClass(val.getType());
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
        return new VCell(this.val.eval(e));
    }

    @Override
    public IType typecheck(Environment<IType> e) {
        type = new TypeRef(this.val.typecheck(e));
        return type;
    }
}
