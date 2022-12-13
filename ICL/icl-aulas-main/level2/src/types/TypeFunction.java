package types;

import java.util.List;

import util.Utils;

/**
 * Represents a type Closure[args, return type].
 */
public class TypeFunction implements IType {
    private List<IType> args;
    private IType returnType;

    public TypeFunction(List<IType> args, IType returnType) {
        this.args = args;
        this.returnType = returnType;
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
        return "Ljava/lang/Object;";
    }

    @Override
    public String toString() {

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
        // TODO Auto-generated method stub
        if( this == obj)
            return true;
        if(!(obj instanceof TypeFunction))
            return false;
        TypeFunction other = (TypeFunction)obj;
        return other.getArgs().equals(this.args) && other.getReturnType().equals(returnType);

        
    }

    


}
