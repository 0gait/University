package ast.print;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.primitives.TypePrimitive;
import util.Utils;
import values.IValue;

public class ASTPrintln extends ASTNodeAbstract
{
    private static final String OPERATOR = "println - ()";

    protected ASTNode node;

    /**
     * @param node
     */
    public ASTPrintln(ASTNode node) {
        this.node = node;
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
        node.compile(c, e);
        c.emit("dup");
        c.emit("getstatic java/lang/System/out Ljava/io/PrintStream;");
        c.emit("swap");
        c.emit(String.format("invokevirtual java/io/PrintStream/println(%s)V", this.node.getType().getJvmType()));
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        IValue result = Utils.requireNonNull(this.node.eval(e));
        System.out.println(result.show());

        return result;
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        this.type = this.node.typecheck(e);

        // Check if the type can be printed
        Utils.checkTypeForOperation(type, TypePrimitive.class,OPERATOR);

        return this.type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append("println (");
        ((ASTNodeAbstract)(this.node)).toString(builder);
        builder.append(")");

        return builder;
    }
}
