package ast.functions;

import java.util.ArrayList;
import java.util.List;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.ClosureCodeBlock;
import compiler.Coordinates;
import compiler.FrameCodeBlock;
import compiler.MainCodeBlock;
import environment.Environment;
import environment.EnvironmentEntry;
import environment.ITypeEnvEntry;
import types.IType;
import types.TypeFunction;
import util.FunctionArg;
import util.Utils;
import values.IValue;
import values.VFunction;

public class ASTFun extends ASTNodeAbstract
{
    private List<FunctionArg> args;
   
    private ASTNode body;
    
    
    public ASTFun(List<FunctionArg> args, ASTNode body) {
        this.args = args;
        this.body = body;
    }

    @Override
    public IValue eval(Environment<IValue> e) {
      
        return new VFunction(args, body, e);
    }

  

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
       ClosureCodeBlock closure =  c.createClosure((TypeFunction)type);
       Environment<Coordinates> newEnv = e.beginScope();
       int frameId = c.getCurrFrameId();

       int i = 0;
       for (FunctionArg functionArg : args) {
        Coordinates varCoord = new Coordinates(frameId, String.format(FrameCodeBlock.FIELD_NAME_FORMAT, i));
           newEnv.assoc(new EnvironmentEntry<Coordinates>(functionArg.id, varCoord));
           i++;
       }
       body.compile(c, newEnv);
       c.endClosure();
       c.emit(String.format("new closure_%d", closure.id));
       c.emit("dup");
       c.emit(String.format("invokespecial closure_%d/<init>()V", closure.id));
       c.emit("dup");
       c.emitCurrentFrame();
       c.emit(String.format("putfield closure_%d/sl Lf%d;", closure.id, c.getCurrFrameId()));
    }

    @Override
    public IType typecheck(Environment<IType> e) {
      Environment<IType> env = e.beginScope();
      List<IType> types = new ArrayList<>(args.size());
      for (FunctionArg arg : args) {
          env.assoc(new ITypeEnvEntry(arg.id, arg.type));
          types.add(arg.type);
      }
      IType funReturnType = body.typecheck(env);
      this.type = new TypeFunction(types, funReturnType);
    return type;
    }

   
    

    @Override
    public StringBuilder toString(StringBuilder builder) {
        builder.append("Function ");
        Utils.toStringList(this.args, null, null, builder);
        builder.append(" ->\n\t");
        this.body.toString(builder);
        builder.append("\nend\n");

        return builder;

    }

    
    
}
