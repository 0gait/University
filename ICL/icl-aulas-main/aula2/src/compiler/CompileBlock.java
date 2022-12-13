package compiler;
import java.io.IOException;
import java.io.PrintStream;

public class CompileBlock {

    public static final String GENERATED_CLASS_DEFAULT_NAME = "MathExpression";

    private static final int MAX = 100;

    private static final String START =

    ".class public %s\n" +
    ".super java/lang/Object\n" +
    ";\n" +
    "; standard initializer\n" +
    ".method public <init>()V\n" +
    "   aload_0\n" +
    "   invokenonvirtual java/lang/Object/<init>()V\n" +
    "   return\n" +
    ".end method\n\n" +
    
    ".method public static main([Ljava/lang/String;)V\n" +
    "      ; set limits used by this method\n" +
    "      .limit locals 10\n" +
    "      .limit stack 256\n\n" +
    
    "      ;    1 - the PrintStream object held in java.lang.System.out\n" +
    "      getstatic java/lang/System/out Ljava/io/PrintStream;\n\n" +
    
    "       ; place your bytecodes here between START and END\n" +
    "       ; START\n\n";

    private static final String END =

    "\n       ; END\n\n" +

    "       ; convert to String;\n" +
    "       invokestatic java/lang/String/valueOf(I)Ljava/lang/String;\n" +
    "       ; call println\n" +
    "       invokevirtual java/io/PrintStream/println(Ljava/lang/String;)V\n" +
    "       return\n" +
    ".end method";

    private String[] code;

    private int pc;

    private String generatedClassName;

    public CompileBlock(String className) throws IOException
    {
        if (className == null || className.equals(""))
            generatedClassName = GENERATED_CLASS_DEFAULT_NAME;
        else
            generatedClassName = className;

        code = new String[MAX];
        pc = 0;
    }

    public void emit(String opcode) {
        code[pc++] = opcode;
    }

    public void dump(PrintStream f) { // dumps code to f

        f.printf(START, generatedClassName);

        for (int i = 0; i < pc; i++)
            f.println(code[i]);

        f.print(END);

        f.flush();
        f.close();
    }

}
