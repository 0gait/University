package ast.print;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.MainCodeBlock;
import types.IType;
import types.TypeRef;
import types.primitves.TypeBool;
import types.primitves.TypeInt;
import util.Coordinates;
import util.Environment;
import values.IValue;

public class ASTPrintln extends ASTNodeAbstract
{
    public static final String OPERATOR = "println";

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

        IType nodeType = node.getType();
        if(nodeType instanceof TypeRef)
            return;

        node.compile(c, e);
        c.emit("dup");
        c.emit("getstatic java/lang/System/out Ljava/io/PrintStream;");
        c.emit("swap");
        
        if(nodeType instanceof TypeInt)
          printInt(c);
        else if (nodeType instanceof TypeBool)
            printBoolean(c);
    }

    private void printBoolean(MainCodeBlock c){
        String l1,l2;
        l1 = c.getNewId();
        l2 = c.getNewId();
        c.emit("ifeq " + l1);
		c.emit("ldc \"true\"");
        c.emit("goto " + l2);
		c.emit(l1 + ": ");
		c.emit("ldc \"false\"");
		c.emit(l2 + ":");
		c.emit("invokevirtual java/io/PrintStream/println(Ljava/lang/String;)V");
    }
    private void printInt(MainCodeBlock c){
       
        c.emit("  ; convert to String;");
        c.emit("invokestatic java/lang/String/valueOf(I)Ljava/lang/String;");
        c.emit("; call println ");
        c.emit("invokevirtual java/io/PrintStream/println(Ljava/lang/String;)V");
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        IValue result = this.node.eval(e);
        System.out.println(result.show());

        return result;
    }

    @Override
    public IType typecheck(Environment<IType> e)
    {
        this.type = this.node.typecheck(e);
        return type;
    }
    
}
