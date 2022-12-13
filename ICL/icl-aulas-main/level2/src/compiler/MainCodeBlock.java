package compiler;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import environment.Environment;
import types.IType;
import types.TypeFunction;
import util.Pair;

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
    
    "      ;    1 - the PrintStream object held in java.lang.System.out\n" +
   // "      getstatic java/lang/System/out Ljava/io/PrintStream;\n\n" +
    
    "       ; place your bytecodes here between START and END\n" +
    "       ; START\n\n" +

    "	    ; initialize frame pointer SL stored in local #4 to null\n" +
    "       aconst_null\n" +
    "       astore 4\n\n";

    private static final String END =

    "\n       ; END\n\n" +

    // "       ; convert to String;\n" +
    // "       invokestatic java/lang/String/valueOf(I)Ljava/lang/String;\n" +
    // "       ; call println\n" +
    // "       invokevirtual java/io/PrintStream/println(Ljava/lang/String;)V\n" +
    "       return\n" +
    ".end method";



    private static final String SL_OBJECT_TYPE = "java/lang/Object";

    private static final String SL_PREVIOUS_FRAME_TYPE = "f%d";

    private List<String> code;

    private String generatedClassName;

    /**
     * List of pairs: previous frameId, Frame
     */
    private List<FrameCodeBlock> frames;
    private Map<IType,RefCodeBlock> refs;

    private FrameCodeBlock currentFrame;

    private int labelId;


    private Map<String, ClosureInterfaceCodeBlock> closureInterfaces;

    private Map<Integer, ClosureCodeBlock> closures;

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
    }

    public String getNewLabelId(){
        return "L" + labelId++;
    }

    public int getCurrFrameId(){
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
     * 
     * @param env
     * @return The new Environment.
     */
    public Environment<Coordinates> endFrame(Environment<Coordinates> env)
    {
        int previousFrameId;
        String previousFrameSlType;

        // Check Environment depth
        // if depth == 1 then there is no previous frame
        if (env.getDepth() == 1)
        {
            previousFrameSlType = SL_OBJECT_TYPE;
            previousFrameId = FrameCodeBlock.INVALID_FRAME_ID;
        }
        else
        {
            previousFrameId = currentFrame.previous.getFrameId();
            previousFrameSlType = String.format(SL_PREVIOUS_FRAME_TYPE, previousFrameId);
        }

        String getPreviousSL = String.format("getfield f%d/sl L%s;", currentFrame.getFrameId(), previousFrameSlType);

        emitCurrentFrame();
        emit(getPreviousSL);
        emit("astore 4");

       currentFrame = currentFrame.previous;

        return env.endScope();
    }

    private FrameCodeBlock createFrame(int numFields)
    {

        // Create frame
        FrameCodeBlock frame = new FrameCodeBlock(this.frames.size(), numFields, currentFrame == null ? null : currentFrame);

        frames.add(frame);
       
        currentFrame = frame;
        
        return frame;
    }

    /**
     * 
     * @return Pair(New Environment, new Frame).
     */
    public Pair<Environment<Coordinates>, FrameCodeBlock>
        addFrame(Environment<Coordinates> env, int numFields)
    {
        // new scope
        Environment<Coordinates> newEnv = env.beginScope();

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

        return new Pair<>(newEnv, frame);
    }
    
    private void saveCurrentFrame(){
        if(activeClosure != null)
            activeClosure.saveCurrentFrame();
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
        // Create closure interface
        String closureInterfaceName = ClosureInterfaceCodeBlock.generateInterfaceName(type);
       
        ClosureInterfaceCodeBlock closureInterface =  
            closureInterfaces.computeIfAbsent(closureInterfaceName, (name)-> {return new ClosureInterfaceCodeBlock(name,type);});
     
        // Create Closure and update activeClosure
        String sl = currentFrame == null ? "Ljava/lang/Object;" :
            "L" + currentFrame.className + ";";
        
        ClosureCodeBlock closure = new ClosureCodeBlock(closures.size(), sl, closureInterface, activeClosure);
        closures.put(closure.id, closure);
        activeClosure = closure;

        // emit init apply function in activeClosure
        activeClosure.emitInitApplyFunction((numArgs) -> createFrame(numArgs));

        return activeClosure;
    }

    public ClosureCodeBlock getActiveClosure(){
        return activeClosure;
    }

    public void endClosure(){
        activeClosure.endClosure();
        activeClosure = activeClosure.previousClosure;
        currentFrame = currentFrame.previous;

    }

    public void dump(File outputFolder) throws IOException, FileNotFoundException { // dumps code to f

        if (!outputFolder.isDirectory())
            throw new IOException("The output folder does not exist.");

        for (FrameCodeBlock frame : this.frames) {
            dumpCodeBlock(frame, outputFolder);
        }

        for (RefCodeBlock ref : this.refs.values()) {
            dumpCodeBlock(ref, outputFolder);
        }
        for( ClosureCodeBlock closure: closures.values()){
            dumpCodeBlock(closure, outputFolder);
        }

        for( ClosureInterfaceCodeBlock closureInterface: closureInterfaces.values()){
            dumpCodeBlock(closureInterface, outputFolder);
        }

        File mainClassFile = createFile(outputFolder, generatedClassName);
        
        try (PrintStream printMainClassFile = new PrintStream(mainClassFile);) {
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

    /**
     * Creates a new Reference Code Block with the specified value type.
     * 
     * @param valueType - The type of value this reference points.
     * 
     * @return A new Reference Code Block with the specified value type.
     */
    public RefCodeBlock createRefClass(IType valueType){
        RefCodeBlock r = new RefCodeBlock(valueType);
        refs.putIfAbsent(valueType,r );
        return r;
    }

    public ClosureInterfaceCodeBlock getClosureInterface(TypeFunction type){
        String closureInterfaceName = ClosureInterfaceCodeBlock.generateInterfaceName(type);  
        return closureInterfaces.computeIfAbsent(closureInterfaceName, (name)-> {return new ClosureInterfaceCodeBlock(name,type);});

    }

    /**
     * Get a Reference Code Block with the specified value type.
     * 
     * @param valueType - The type of value this reference points.
     * 
     * @return A Reference Code Block with the specified value type.
     */
    public RefCodeBlock getRefClass(IType valueType){
        return refs.get(valueType);
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
