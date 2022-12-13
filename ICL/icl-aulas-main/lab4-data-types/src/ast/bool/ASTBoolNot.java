package ast.bool;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.MainCodeBlock;
import typeError.IllegalOperatorException;
import types.IType;
import types.primitves.TypeBool;
import util.Coordinates;
import util.Environment;
import values.IValue;
import values.primitive.VBool;

public class ASTBoolNot extends ASTNodeAbstract
{
    public static final String OPERATOR = "~";

    protected ASTNode node;

    /**
     * @param node
     */
    public ASTBoolNot(ASTNode node) {
        this.node = node;
        type = TypeBool.TYPE;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        node.compile(c, e);
        c.emit("sipush 1");
        c.emit("ixor");
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        return new VBool( !checkRuntimeType(this.node.eval(e)).getValue() );
    }

    @Override
    public IType typecheck(Environment<IType> e) {
        return checkType(this.node.typecheck(e));
        
    }

    protected VBool checkRuntimeType(IValue val)
    {
        boolean checked = val instanceof VBool;
        if (!checked)
            throw new IllegalOperatorException(OPERATOR);

        return (VBool)val;
    }

    protected IType checkType(IType type)
    {
        boolean checked = type instanceof TypeBool;
        if (!checked)
            throw new IllegalOperatorException(OPERATOR);

        return type;
    }

}
