package compiler;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.LinkedList;
import java.util.List;

import util.Coordinates;
import util.Environment;
import util.Pair;

public class MainCodeBlock {

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
    "      getstatic java/lang/System/out Ljava/io/PrintStream;\n\n" +
    
    "       ; place your bytecodes here between START and END\n" +
    "       ; START\n\n" +

    "	    ; initialize frame pointer SL stored in local #4 to null\n" +
    "       aconst_null\n" +
    "       astore 4\n\n";

    private static final String END =

    "\n       ; END\n\n" +

    "       ; convert to String;\n" +
    "       invokestatic java/lang/String/valueOf(I)Ljava/lang/String;\n" +
    "       ; call println\n" +
    "       invokevirtual java/io/PrintStream/println(Ljava/lang/String;)V\n" +
    "       return\n" +
    ".end method";

    private static final String NEW_FRAME =

    "	new f%d\n" +
    "	dup\n" +
    "       invokespecial f%d/<init>()V\n" +
    "	dup\n" +

    "	; store SL in new frame\n" +
    "	aload 4\n" +
    "	putfield f%d/sl L%s;\n" +
    "	; update SL\n" +
    "	astore 4\n\n";


    private static final String SL_OBJECT_TYPE = "java/lang/Object";

    private static final String SL_PREVIOUS_FRAME_TYPE = "f%d";

    private List<String> code;

    private String generatedClassName;

    /**
     * List of pairs: previous frameId, Frame
     */
    private List<Pair<Integer, FrameCodeBlock>> frames;

    private int currentFrameId;

    public MainCodeBlock(String className) throws IOException
    {
        if (className == null || className.equals(""))
            generatedClassName = GENERATED_CLASS_DEFAULT_NAME;
        else
            generatedClassName = className;

        code = new LinkedList<>();
        this.frames = new LinkedList<>();

        this.currentFrameId = FrameCodeBlock.INVALID_PREVIOUS_FRAME_ID;
    }

    public void emit(String opcode) {
        code.add(opcode);
    }


    public void emitCurrentFrame()
    {
        emit("aload 4");
    }

    /**
     * 
     * @param env
     * @return The new Environment.
     */
    public Environment<Coordinates> endFrame(Environment<Coordinates> env)
    {
        emitCurrentFrame();

        int previousFrameId;
        String previousFrameSlType;

        // Check Environment depth
        // if depth == 1 then there is no previous frame
        if (env.getDepth() == 1)
        {
            previousFrameSlType = SL_OBJECT_TYPE;
            previousFrameId = FrameCodeBlock.INVALID_PREVIOUS_FRAME_ID;
        }
        else
        {
            Pair<Integer, FrameCodeBlock> framePair = this.frames.get(this.currentFrameId);
            previousFrameId = framePair.getLeft();
            previousFrameSlType = String.format(SL_PREVIOUS_FRAME_TYPE, previousFrameId);
        }

        emit(String.format("getfield f%d/sl L%s;", this.currentFrameId, previousFrameSlType) );
        emit("astore 4");

        this.currentFrameId = previousFrameId;

        return env.endScope();
    }

    /**
     * 
     * @param numVariables
     * @return Pair(New Environment, new FrameId).
     */
    public Pair<Environment<Coordinates>, Integer>
        addFrame(int numVariables, Environment<Coordinates> env)
    {
        // new scope
        Environment<Coordinates> newEnv = env.beginScope();

        int previousFrameId = this.currentFrameId;

        // Create frame
        FrameCodeBlock frame = new FrameCodeBlock(this.frames.size(), previousFrameId);
        frame.setNumVariables(numVariables);

        Pair<Integer, FrameCodeBlock> framePair = new Pair<>(previousFrameId, frame);
        this.frames.add(framePair);
        this.currentFrameId = frame.getFrameId();
    
        // Check Environment depth
        // if depth == 1 then there is no previous frame
        String frameSlType;
        if (newEnv.getDepth() == 1)
            frameSlType = SL_OBJECT_TYPE;
        else
            frameSlType = String.format(SL_PREVIOUS_FRAME_TYPE, previousFrameId);

        String newFrameInstruction = String.format(NEW_FRAME, this.currentFrameId,
            this.currentFrameId, this.currentFrameId, frameSlType);
        
        emit(newFrameInstruction);

        return new Pair<>(newEnv, this.currentFrameId);
    }

    public void reachFrameIdFromCurrentFrame(int frameId)
    {
        if (this.currentFrameId == frameId)
            return;

        assert this.currentFrameId != FrameCodeBlock.INVALID_PREVIOUS_FRAME_ID;

        int currentFrameId = this.currentFrameId;
        Pair<Integer, FrameCodeBlock> pair = this.frames.get(currentFrameId);
        int previousFrameId = pair.getLeft();

        while (previousFrameId != FrameCodeBlock.INVALID_PREVIOUS_FRAME_ID
            && currentFrameId != frameId)
        {
            emit(String.format("getfield f%d/sl Lf%d;", currentFrameId, previousFrameId));
            currentFrameId = previousFrameId;

            assert this.currentFrameId != FrameCodeBlock.INVALID_PREVIOUS_FRAME_ID;

            if (currentFrameId != frameId)
                pair = this.frames.get(currentFrameId);
        }
    }

    public void dump(File outputFolder) throws IOException, FileNotFoundException { // dumps code to f

        if (!outputFolder.isDirectory())
            throw new IOException("The output folder does not exist.");

        for (Pair<Integer, FrameCodeBlock> pairPreviousFrameIdFrame : this.frames) {
            dumpFrame(pairPreviousFrameIdFrame.getRight(), outputFolder);
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

    private File createFile(File outputFolder, String fileName)
    {
        return new File(outputFolder + File.separator + fileName + ".j");
    }

    private void dumpFrame(FrameCodeBlock frame, File outputFolder) throws FileNotFoundException
    {
        String frameClassName = frame.getClassName();

        File frameFile = createFile(outputFolder, frameClassName);

        try (PrintStream printFrameFile = new PrintStream(frameFile);) {
            frame.dump(printFrameFile);
        }
    }
}
