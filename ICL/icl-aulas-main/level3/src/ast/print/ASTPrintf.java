package ast.print;

import java.util.List;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.TypeNull;
import types.primitives.TypePrimitive;
import types.primitives.TypeString;
import util.Utils;
import values.IValue;
import values.VNull;
import values.primitive.VPrimitive;
import values.primitive.VString;

public class ASTPrintf extends ASTNodeAbstract {

    private ASTNode format;
    private List<ASTNode> args;

    
    public ASTPrintf(ASTNode format, List<ASTNode> args) {
        this.format = format;
        this.args = args;
        this.type = TypeNull.TYPE;
    }

    @Override
    public IValue eval(Environment<IValue> e) {
        String format = Utils.checkValueForOperation(this.format.eval(e), VString.class, "printf").getValue();
        Object[] args = this.args.stream().map((arg)-> {
            IValue v = arg.eval(e);
            if(v instanceof VPrimitive)
                return ((VPrimitive)v).getValue();
            return v;
        }).toArray();
      
        System.out.printf(format, args);
        return VNull.VALUE;
    }



    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
        c.emit("getstatic java/lang/System/out Ljava/io/PrintStream;");
        format.compile(c, e);
        c.emit("sipush " + args.size());
        c.emit("anewarray java/lang/Object");
        int i = 0;
        for (ASTNode astNode : args) {
            c.emit("dup");
            c.emit("sipush " + i);
            astNode.compile(c, e);
            if(!(astNode.getType() instanceof TypeString)){
               IType type = astNode.getType();
               c.emit(String.format("invokestatic java/lang/%s/valueOf(%s)Ljava/lang/%s;",type.getSimpleName(),type.getJvmType(), type.getSimpleName()));
            }
               
            c.emit("aastore");
            i++;
        }
        c.emit("invokevirtual java/io/PrintStream/printf(Ljava/lang/String;[Ljava/lang/Object;)Ljava/io/PrintStream;");
        c.emit("pop");
        c.emitNull();
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException {
        Utils.checkTypeForOperation(format.typecheck(e), TypeString.class, "printf");
        for (ASTNode astNode : args) {
            Utils.checkTypeForOperation(astNode.typecheck(e), TypePrimitive.class, "printf", "printf only accepts primitive types");
        }

        return this.type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append("printf (");
        builder.append(((ASTNodeAbstract)(this.format)).toString(builder));
        
        if (!this.args.isEmpty())
        {
            builder.append(", ");

            Utils.toStringList(this.args, (ast) -> {
                ((ASTNodeAbstract)(ast)).toString(builder);
            }, null, null, builder);
        }
        
        builder.append(')');
        return builder;
    }
    
}
