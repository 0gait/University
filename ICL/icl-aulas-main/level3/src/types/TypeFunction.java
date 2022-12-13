package types;

import java.util.List;

import util.Utils;

/**
 * Represents a type Closure[args, return type].
 */
public class TypeFunction implements IType
{
    public static final String TYPE_NAME = "Function closure";
    public static final String JVM_TYPE = "Ljava/lang/Object;";

    private List<IType> args;
    private IType returnType;

    private boolean recursive;

    public TypeFunction(List<IType> args, IType returnType) {
        this.args = args;
        this.returnType = returnType;
        recursive = false;
    }

    @Override
    public String show() {
        return Utils.toStringList(this.args,
            (arg) ->
            {
                return arg.show();
            },
            null, new StringBuilder("Function "))
            .append(" -> ")
            .append(this.returnType.show()).toString();
    }

    @Override
    public String getJvmType() {
        return JVM_TYPE;
    }

    @Override
    public String toString() {
        return show();
    }

    @Override
    public String getSimpleName() {
        return "closure";
    }

    @Override
    public int hashCode() {
        return args.hashCode() ^ returnType.hashCode();
    }

    public List<IType> getArgs(){
        return args;
    }

    public IType getReturnType(){
        return returnType;
    }

    @Override
    public boolean equals(Object obj) {
        if( this == obj)
            return true;
        if(!(obj instanceof TypeFunction))
            return false;
        TypeFunction other = (TypeFunction)obj;
        return other.getArgs().equals(this.args) && other.getReturnType().equals(returnType);
    }

    /**
     * If this function is recursive.
     * @return True if this function is recursive.
     */
    public boolean isRecursive() {
        return recursive;
    }

    /**
     * Sets this function as recursive.
     * @param recursive true if recursive.
     */
    public void setRecursive() {
        this.recursive = true;
    }

}
