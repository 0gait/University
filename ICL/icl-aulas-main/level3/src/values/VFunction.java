package values;

import java.util.List;
import java.util.function.Consumer;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import environment.Environment;
import types.IType;
import types.TypeFunction;
import util.FunctionArg;
import util.Utils;

/**
 * Represents a closure value [args, body, env].
 */
public class VFunction implements IValue {

    public final static String TYPE_NAME = TypeFunction.TYPE_NAME;

    private static boolean showActive = false;

    private List<FunctionArg> args;
    private ASTNode body;
    private Environment<IValue> env;

    public VFunction(List<FunctionArg> args, ASTNode body, Environment<IValue> env) {
        this.args = args;
        this.body = body;
        this.env = env;
    }

    @Override
    public String show() {

        if (showActive)
            return "Closure";

        StringBuilder builder = new StringBuilder("Closure [\n\targs = ");
        Utils.toStringList(this.args, (Consumer<FunctionArg>)((arg) -> builder.append(arg.id)), null,
            Utils.DEFAULT_DELIMITERS, builder);

        builder.append("\n\tbody = ");
        ((ASTNodeAbstract)this.body).toString(builder);

        builder.append("\n\tenvironment = ");
        showActive = true;
        this.env.toString(builder);
        showActive = false;

        builder.append("\n]");

        return builder.toString();
    }

    @Override
    public String toString() {
       return show();
    }

    public List<FunctionArg> getArgs() {
        return args;
    }

    public ASTNode getBody() {
        return body;
    }

    public Environment<IValue> getEnv() {
        return env;
    }

    @Override
    public IType getType() {
        return TYPE;
    }

    public static final IType TYPE = new IType() {

        @Override
        public String show() {
            return this.toString();
        }

        @Override
        public String getJvmType() {
            return "";
        }
        
        @Override
        public String toString() {

            return "Function closure";
        }

        @Override
        public String getSimpleName() {
            return null;
        }
    };

}
