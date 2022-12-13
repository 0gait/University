package compiler;

import java.io.PrintStream;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

import types.IType;
import types.primitives.TypePrimitive;

/**
 * Represents a Closure that implements a specific apply signature.
 * 
 * @see ClosureInterfaceCodeBlock
 */
public class ClosureCodeBlock extends CodeBlock
{
    public static final String START = ".class public %s\n" + 
    ".super	java/lang/Object\n" +	
    ".implements %s\n" + 
    ".field	public sl %s\n" + 
    ".method public	<init>()V\n" +
         "aload_0\n" + 	
         "invokenonvirtual	java/lang/Object/<init>()V\n " + 	
         "return\n" + 	
    ".end method\n";

    public final ClosureCodeBlock previousClosure;

    public final int id;
    public final String staticLink;
    public final ClosureInterfaceCodeBlock closureInterface;
    public final String currFrame, saveFrame;
    private final List<String> code;

    public ClosureCodeBlock(int id, String staticLink, ClosureInterfaceCodeBlock closureInterface,
        ClosureCodeBlock previousClosure) 
    {
        super("closure_" + id);
        this.previousClosure = previousClosure;
        this.id = id;
        this.staticLink = staticLink;
        this.closureInterface = closureInterface;
        code = new LinkedList<>();
        currFrame = "aload " + (closureInterface.getNumArgs() + 1);
        saveFrame = String.format("astore %d", (closureInterface.getNumArgs() + 1));
      
    }

    public void emit(String opcode)
    {
        this.code.add(opcode);
    }

    public void emitCurrentFrame()
    {
        emit(this.currFrame);
    }

    public void saveCurrentFrame()
    {
        emit(this.saveFrame);
    }


    public void emitInitApplyFunction(Function<Integer, FrameCodeBlock> createFrame)
    {
        FrameCodeBlock frame = createFrame.apply(closureInterface.getNumArgs());

        emit(".method public " + closureInterface.getApplySignature());

        // 1 for (this) - closure reference
        // k for num args of apply
        // 1 for SL
        emit(".limit locals " + (1 + closureInterface.getNumArgs() + 1)); 
        emit(".limit stack 256");

        emit(String.format(" new f%d", frame.getFrameId()));
        emit("	dup");
        emit(String.format("  invokespecial f%d/<init>()V", frame.getFrameId()));
        emit("	dup");

        emit("aload 0");
        emit(String.format("getfield closure_%d/sl %s", id, staticLink));
        emit(String.format("putfield f%d/sl %s", frame.getFrameId(), staticLink));
        

        // add local variables to frame
        int fieldIndex = 0;
        for (IType argType : closureInterface.typeFunction.getArgs()) {
            emit("dup");
            if(argType instanceof TypePrimitive)
                emit("iload " + (fieldIndex + 1));
            else
                emit("aload " + (fieldIndex + 1));
            emit(String.format("putfield f%d/x%d %s", frame.getFrameId(), fieldIndex,
                argType.getJvmType()));

            frame.addFieldType(argType.getJvmType());
            
            fieldIndex++;
        }

        // store current sl
        saveCurrentFrame();
    }

    public void endClosure(){
        if(closureInterface.typeFunction.getReturnType() instanceof TypePrimitive)
            emit("ireturn");
        else
            emit("areturn");
        emit(".end method");
    }

    @Override
    public void dump(PrintStream f){
        f.printf(START, className, closureInterface.className,staticLink);
        for (String string : code) {
            f.println(string);
        }
    }

}
