package values;

import java.util.List;
import java.util.function.Consumer;

import ast.ASTNode;
import environment.Environment;
import util.FunctionArg;
import util.Utils;

/**
 * Represents a closure value [args, body, env].
 */
public class VFunction implements IValue {

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
        this.body.toString(builder);

        builder.append("\n\tenvironment = ");
        showActive = true;
        this.env.toString(builder);
        showActive = false;

        builder.append("\n]");

        return builder.toString();
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

}
