package compiler;

import java.io.PrintStream;

import types.IType;

public class RefCodeBlock {
    
    private static final String CLASS_NAME = "ref_of_%s";

    private static final String CODE =

    ".class %s\n" +
    ".super java/lang/Object\n" +
    ".field public v %s\n" +

    ".method public <init>()V\n" +

    "  aload_0\n" +
    "  invokenonvirtual java/lang/Object/<init>()V\n" +
    "  return\n\n" +
    ".end method";


    private String className;

    private String valueFieldType;

    /**
     * Creates a reference code block with the specified type of object
     * this reference points.
     * 
     * @param valueType - The type of value to be referenced.
     */
    public RefCodeBlock(IType valueType)
    {

        className = String.format(CLASS_NAME, valueType);
        valueFieldType = valueType.getJvmType();
    }
 

    /**
     * @return the className
     */
    public String getClassName() {
        return className;
    }

    /**
     * Get the type of value this reference points in the JVM syntax.
     * @return The type of value this reference points.
     */
    public String getValueFieldTypeJVM(){
        return valueFieldType;
    }

    public void dump(PrintStream f) { // dumps code to f
        f.printf(CODE,className,valueFieldType);
        f.flush();
    }
}

