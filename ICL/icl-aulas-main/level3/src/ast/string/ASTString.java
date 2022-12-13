package ast.string;

import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import types.IType;
import types.primitives.TypeString;
import values.IValue;
import values.primitive.VString;

public class ASTString extends ASTNodeAbstract
{
    private String value;

    /**
     * @param value
     */
    public ASTString(String value) {
        this.value = value;
        this.type = TypeString.TYPE;
    }

    @Override
    public IValue eval(Environment<IValue> e) {
        return new VString(this.value);
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        c.emit(String.format("ldc \"%s\"", value));
    }

    @Override
    public IType typecheck(Environment<IType> e) {
        return this.type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        return builder.append( "\"" + this.value + "\"");
    }
}
