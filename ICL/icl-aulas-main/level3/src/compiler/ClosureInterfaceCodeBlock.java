package compiler;

import java.io.PrintStream;
import java.util.function.Function;

import types.IType;
import types.TypeFunction;
import util.Utils;

/**
 * Represents an Interface for all closures with the same signature.
 */
public class ClosureInterfaceCodeBlock extends CodeBlock
{
    private static String CODE =
        ".interface public %s\n" +
        ".super	java/lang/Object\n" +
        ".method public	abstract %s\n" +
        ".end method";

    /**
     * The closure signature type.
     */
    public final TypeFunction typeFunction;
 
    public final String returnTypeString;

    private String applySignature;

    public ClosureInterfaceCodeBlock(TypeFunction typeFunction){
       this(generateInterfaceName(typeFunction),typeFunction);
    }

    public ClosureInterfaceCodeBlock(String interfaceName, TypeFunction typeFunction){
        super(interfaceName);
        this.typeFunction = typeFunction;
        returnTypeString = typeFunction.getReturnType().getJvmType();
    }

    @Override
    public void dump(PrintStream f) { // dumps code to f     
        f.printf(CODE,className, getApplySignature());
        f.flush();
    }

    public static String generateInterfaceName(TypeFunction typeFunction){
        StringBuilder toReturn = new StringBuilder( "closure_interface_");
       return Utils.toStringList(typeFunction.getArgs(), (Function<IType,String>)((arg) -> arg.getSimpleName())
        , "_", null, toReturn).append("_" + typeFunction.getReturnType().getSimpleName()).toString();
    }

    private String generateArgsList(){
        StringBuilder toReturn = new StringBuilder();
        return Utils.toStringList(typeFunction.getArgs(), (Function<IType,String>)((arg) -> arg.getJvmType()),
        "", Utils.DEFAULT_DELIMITERS, toReturn).toString();
    }    

    public int getNumArgs(){
        return typeFunction.getArgs().size();
    }

    public String getApplySignature()
    {
        if (applySignature != null)
            return applySignature;
        
        StringBuilder builder = new StringBuilder("apply");
        builder.append(generateArgsList());
        builder.append(returnTypeString);

        applySignature = builder.toString();

        return applySignature;
    }

}
