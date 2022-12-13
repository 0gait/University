package ast.functions;

import java.util.Iterator;
import java.util.List;
import java.util.function.Consumer;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.ClosureInterfaceCodeBlock;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.TypeFunction;
import util.FunctionArg;
import util.Utils;
import values.IValue;
import values.VFunction;

public class ASTApply extends ASTNodeAbstract
{
    private static final String OPERATOR = "Function apply - ()";

    private static final String ERROR_MSG = "The apply operator can only be used for function calls.";

    private ASTNode function;
    private List<ASTNode> args;

    public ASTApply(ASTNode function, List<ASTNode> args) {
        this.function = function;
        this.args = args;
    }
    
    @Override
    public IValue eval(Environment<IValue> e)
    {
        IValue toReturn = null;
        VFunction fun = Utils.checkValueForOperation(function.eval(e), VFunction.class, OPERATOR, ERROR_MSG);

        Environment<IValue> funEnv = fun.getEnv().beginScope();
        Iterator<ASTNode> it = args.iterator();
        Iterator<FunctionArg> names = fun.getArgs().iterator();

        if(args.size() != fun.getArgs().size())
            throw new TypeErrorException("Incorrect number of args for function call. Expected " + 
                fun.getArgs().size() +" and got " + args.size() + " arguments.")
                .toRuntimeException();

        while(it.hasNext())
            funEnv.assoc(names.next().id, it.next().eval(e));
        
        toReturn = fun.getBody().eval(funEnv);
        funEnv.endScope();

        return toReturn;

    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
       function.compile(c, e);
       ClosureInterfaceCodeBlock closureInterface = c.getClosureInterface((TypeFunction)function.getType());
       c.emit("checkcast " + closureInterface.className);

       for (ASTNode astNode : args)
           astNode.compile(c, e);

        c.emit("invokeinterface " + closureInterface.className + "/" +
            closureInterface.getApplySignature() + " " + (args.size() + 1));
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        TypeFunction typeFunction = Utils.checkTypeForOperation(this.function.typecheck(e), TypeFunction.class, OPERATOR,ERROR_MSG);
        Iterator<IType>  itFunTypes = typeFunction.getArgs().iterator();
        Iterator<ASTNode>  it = args.iterator();

        if(args.size() != typeFunction.getArgs().size())
            throw new TypeErrorException("Incorrect number of arguments for function call. Expected " +
                typeFunction.getArgs().size() +" and got " + args.size() + " arguments.");
        
        while(it.hasNext()){
            IType toCheck = it.next().typecheck(e);
            IType expected = itFunTypes.next();

            if(!toCheck.equals(expected))
                throw new TypeErrorException("Argument types don't match. Expected " +
                    expected.show() + " and got " + toCheck.show());
        }

        this.type = typeFunction.getReturnType();
        return this.type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder) {
        ((ASTNodeAbstract)this.function).toString(builder);
        builder.append(" ");
        Utils.toStringList(args, (Consumer<ASTNode>)((arg) -> ((ASTNodeAbstract)arg).toString(builder)), null, Utils.DEFAULT_DELIMITERS, builder);

        return builder;
    }
}
