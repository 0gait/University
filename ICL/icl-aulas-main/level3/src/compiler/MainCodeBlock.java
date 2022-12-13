package compiler;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import types.IType;
import types.TypeFunction;
import types.TypeRecord;

public class MainCodeBlock
{

    public static final String GENERATED_CLASS_DEFAULT_NAME = "MathExpression";

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
    
    "       ; place your bytecodes here between START and END\n" +
    "       ; START\n\n" +

    "	    ; initialize frame pointer SL stored in local #4 to null\n" +
    "       aconst_null\n" +
    "       astore 4\n\n";

    private static final String END =

    "\n       ; END\n\n" +
    "   return\n" +
    ".end method";

    private static final String SL_OBJECT_TYPE = "java/lang/Object";

    private static final String SL_PREVIOUS_FRAME_TYPE = "f%d";

    private List<String> code;

    private String generatedClassName;

    private List<FrameCodeBlock> frames;

    private Map<IType,RefCodeBlock> refs;

    private FrameCodeBlock currentFrame;

    private int labelId;

    private Map<String, ClosureInterfaceCodeBlock> closureInterfaces;

    private Map<Integer, ClosureCodeBlock> closures;

    private Map<TypeRecord, RecordCodeBlock> records;

    /**
     * The active closure or null.
     */
    private ClosureCodeBlock activeClosure;

    public MainCodeBlock(String className) throws IOException
    {
        if (className == null || className.equals(""))
            generatedClassName = GENERATED_CLASS_DEFAULT_NAME;
        else
            generatedClassName = className;

        code = new LinkedList<>();
        this.frames = new LinkedList<>();

       currentFrame = null;

        labelId = 0;
        refs = new HashMap<>();

        closureInterfaces = new HashMap<>();
        closures = new HashMap<>();
        records = new HashMap<>();
    }

    public String getNewLabelId(){
        return "L" + labelId++;
    }

    public int getCurrentFrameId(){
        return currentFrame.getFrameId();
    }

    public void emit(String opcode) {
        if (this.activeClosure != null)
            this.activeClosure.emit(opcode);
        else
            code.add(opcode);
    }


    public void emitCurrentFrame()
    {
        if (this.activeClosure != null)
            this.activeClosure.emitCurrentFrame();
        else
            emit("aload 4");
    }

    /**
     * Creates a new Frame.
     * @param numFields - The number of fields in the frame.
     * @return
     */
    private FrameCodeBlock createFrame(int numFields)
    {
        // Create frame
        FrameCodeBlock frame = new FrameCodeBlock(this.frames.size(), numFields, currentFrame == null ? null : currentFrame);
        frames.add(frame);
        currentFrame = frame;
        return frame;
    }

    /**
     * Creates a new Frame and emits the necessary instructions.
     * @param numFields - The number of fields in the frame.
     * @return Frame
     */
    public FrameCodeBlock addFrame(int numFields)
    {
        FrameCodeBlock frame = createFrame(numFields);

        emit(String.format(" new f%d", frame.getFrameId()));
        emit("	dup");
        emit(String.format("  invokespecial f%d/<init>()V", frame.getFrameId()));
        emit("	dup");
        emit("	; store SL in new frame");
        emitCurrentFrame();
        emit(String.format("	putfield f%d/sl %s", frame.getFrameId(), frame.slType));
        emit("	; update SL");
        saveCurrentFrame();

        return frame;
    }

    /**
     * Go back to the previous frame.
     */
    public void endFrame()
    {
        assert this.currentFrame != null;

        FrameCodeBlock previous = this.currentFrame.previous;

        String previousFrameSlType = previous == null
            ? SL_OBJECT_TYPE
            : String.format(SL_PREVIOUS_FRAME_TYPE, previous.getFrameId());

        String getPreviousSL = String.format("getfield f%d/sl L%s;", currentFrame.getFrameId(), previousFrameSlType);

        emitCurrentFrame();
        emit(getPreviousSL);
        saveCurrentFrame();

        currentFrame = currentFrame.previous;
    }
    
    private void saveCurrentFrame(){
        if(this.activeClosure != null)
            this.activeClosure.saveCurrentFrame();
        else
            emit("astore 4");
    }
    public void reachFrameIdFromCurrentFrame(int frameId)
    {
        assert this.currentFrame != null;
        if (currentFrame.getFrameId() == frameId)
            return;

        FrameCodeBlock currentFrame = this.currentFrame;
        FrameCodeBlock previousFrame = currentFrame.previous;

        while (previousFrame != null 
            && currentFrame.getFrameId() != frameId)
        {
            emit(String.format("getfield f%d/sl Lf%d;", currentFrame.getFrameId(), previousFrame.getFrameId()));
            currentFrame = previousFrame;
            previousFrame =currentFrame.previous;
            assert this.currentFrame != null;
        }
    }

    public ClosureCodeBlock createClosure(TypeFunction type)
    {       
        ClosureInterfaceCodeBlock closureInterface = getClosureInterface(type);
     
        // Create Closure and update activeClosure
        String sl = currentFrame == null ? "Ljava/lang/Object;" :
            "L" + currentFrame.className + ";";
        
        ClosureCodeBlock closure = new ClosureCodeBlock(closures.size(), sl, closureInterface, this.activeClosure);
        closures.put(closure.id, closure);
        this.activeClosure = closure;

        // emit init apply function in activeClosure
        this.activeClosure.emitInitApplyFunction((numArgs) -> createFrame(numArgs));

        return this.activeClosure;
    }

    public void endClosure(){
        activeClosure.endClosure();
        activeClosure = activeClosure.previousClosure;
        currentFrame = currentFrame.previous;
    }

    public ClosureInterfaceCodeBlock getClosureInterface(TypeFunction type){
        String closureInterfaceName = ClosureInterfaceCodeBlock.generateInterfaceName(type);  
        return closureInterfaces.computeIfAbsent(closureInterfaceName, (name)-> {return new ClosureInterfaceCodeBlock(name,type);});
    }

    public RecordCodeBlock createRecord(TypeRecord type)
    {
        RecordCodeBlock record = getRecord(type);
        emit("new " + record.className);
        emit("dup");
        emit("invokespecial " + record.className + "/<init>()V");
        emit("dup");

        return record;
    }

    public void emitNull(){
        emit("aconst_null");
    }

    public RecordCodeBlock getRecord(TypeRecord type){
       return records.computeIfAbsent(type, (name)-> {return new RecordCodeBlock(records.size(),type);});
    }

    /**
     * Get a Reference Code Block with the specified value type.
     * 
     * @param valueType - The type of value this reference points.
     * 
     * @return A Reference Code Block with the specified value type.
     */
    public RefCodeBlock getRefClass(IType valueType){
        return refs.computeIfAbsent(valueType, (name)-> {return new RefCodeBlock(valueType);});
    }

    public void dump(File outputFolder) throws IOException, FileNotFoundException
    {
        if (!outputFolder.isDirectory())
            throw new IOException("The output folder does not exist.");

        for (FrameCodeBlock frame : this.frames)
            dumpCodeBlock(frame, outputFolder);

        for (RefCodeBlock ref : this.refs.values())
            dumpCodeBlock(ref, outputFolder);
        
        for( ClosureCodeBlock closure: closures.values())
            dumpCodeBlock(closure, outputFolder);

        for( ClosureInterfaceCodeBlock closureInterface: closureInterfaces.values())
            dumpCodeBlock(closureInterface, outputFolder);

        for( RecordCodeBlock record: records.values())
            dumpCodeBlock(record, outputFolder);
        
        File mainClassFile = createFile(outputFolder, generatedClassName);
        
        try (PrintStream printMainClassFile = new PrintStream(mainClassFile);)
        {
            printMainClassFile.printf(START, generatedClassName);

            this.code.forEach(
                (line) ->
                {
                    printMainClassFile.println(line);
                }
            );

            printMainClassFile.print(END);

            printMainClassFile.flush();
        }
    }

    private File createFile(File outputFolder, String fileName)
    {
        return new File(outputFolder + File.separator + fileName + ".j");
    }

    private void dumpCodeBlock(CodeBlock c, File outputFolder) throws FileNotFoundException{
        String classname = c.className;

        File codeFile = createFile(outputFolder, classname);

        try (PrintStream printCodeFile = new PrintStream(codeFile);) {
            c.dump(printCodeFile);
        }
    }
}
